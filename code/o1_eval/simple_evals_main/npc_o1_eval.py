from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from o1_eval.simple_evals_main.npc_o1_prompts import (
    render_generation_messages,
    render_generation_prompt,
    render_judge_messages,
    render_judge_prompt,
)
from o1_eval.simple_evals_main.npc_o1_schema import (
    O1RubricItem,
    flatten_dataset_rubrics,
    load_o1_items,
    read_jsonl,
    sha256_file,
    summarize_o1_items,
    write_jsonl,
)
from o1_eval.simple_evals_main.npc_o1_scorer import parse_judge_json, score_dataset, validate_judge_output, write_scores


DEFAULT_DATA_PATH = "data/o1_consultrubrics/consultrubrics_v20260504_doctor_revised.jsonl"
DEFAULT_DATASET_NAME = "ConsultRubrics_v20260504"


@dataclass(frozen=True)
class APIModelSpec:
    name: str
    api_model_id: str
    family: str
    concurrency: int = 4
    timeout: int = 180
    max_tokens: int = 2048
    send_top_p: bool = True


API_MODEL_REGISTRY: dict[str, APIModelSpec] = {
    "DeepSeek-V3.1": APIModelSpec(
        "DeepSeek-V3.1",
        "deepseek-v3.1-250821",
        "Open-Source LLMs",
        concurrency=8,
        timeout=120,
    ),
    "DeepSeek-R1": APIModelSpec(
        "DeepSeek-R1",
        "deepseek-r1",
        "Open-Source LLMs",
        concurrency=4,
        timeout=300,
    ),
    "GPT-5.1": APIModelSpec(
        "GPT-5.1",
        "gpt-5.1-2025-11-13",
        "Proprietary (M)LLMs",
        concurrency=4,
        timeout=180,
    ),
    "GPT-4.1": APIModelSpec(
        "GPT-4.1",
        "gpt-4.1-2025-04-14",
        "Proprietary (M)LLMs",
        concurrency=6,
        timeout=120,
    ),
    "Gemini-3.1-Pro-Preview": APIModelSpec(
        "Gemini-3.1-Pro-Preview",
        "gemini-3.1-pro-preview",
        "Proprietary (M)LLMs",
        concurrency=2,
        timeout=360,
    ),
    "Gemini-2.5-Pro": APIModelSpec(
        "Gemini-2.5-Pro",
        "gemini-2.5-pro",
        "Proprietary (M)LLMs",
        concurrency=4,
        timeout=240,
    ),
    "Gemini-2.5-Flash": APIModelSpec(
        "Gemini-2.5-Flash",
        "gemini-2.5-flash",
        "Proprietary (M)LLMs",
        concurrency=8,
        timeout=180,
    ),
    "Claude-Sonnet-4.6": APIModelSpec(
        "Claude-Sonnet-4.6",
        "claude-sonnet-4-6",
        "Proprietary (M)LLMs",
        concurrency=2,
        timeout=300,
        send_top_p=False,
    ),
}


def _json_dump(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def _json_load(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _public_path_ref(value: str | Path | None) -> str | None:
    if value is None:
        return None
    text = str(value)
    if not text:
        return text
    is_windows_abs = len(text) > 2 and text[1] == ":" and text[2] in {"/", "\\"}
    if is_windows_abs:
        return text.replace("\\", "/").rstrip("/").split("/")[-1] or "LOCAL_PATH_REDACTED"
    if Path(text).is_absolute() or text.startswith("~") or is_windows_abs:
        return Path(text).name or "LOCAL_PATH_REDACTED"
    return text


def _public_config_value(key: str, value: Any) -> Any:
    key_lower = key.lower()
    if isinstance(value, list):
        return [_public_config_value(key, item) for item in value]
    if isinstance(value, tuple):
        return [_public_config_value(key, item) for item in value]
    if isinstance(value, str) and (
        "path" in key_lower or key_lower.endswith("_dir") or key_lower.endswith("_root")
    ):
        return _public_path_ref(value)
    return value


def _load_rubric_items(path: str | Path) -> list[O1RubricItem]:
    return [O1RubricItem.from_dict(row) for row in read_jsonl(path)]


def _run_config(args: argparse.Namespace, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    command_args = {
        key: _public_config_value(key, value)
        for key, value in vars(args).items()
        if key != "func" and isinstance(value, (str, int, float, bool, type(None)))
    }
    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_path": _public_path_ref(args.data_path),
        "data_sha256": sha256_file(args.data_path) if Path(args.data_path).exists() else None,
        "judge_prompt_version": "npc_o1_grader_v2_polarity_aware",
        "generation_prompt_version": "npc_o1_generation_v1",
        "scorer_version": "npc_o1_scorer_v1_positive_minus_major2_minor1",
        "scoring_rule": "final_score=max(0, positive_score - 2*major_error_hits - 1*minor_error_hits)",
        "command_args": command_args,
        "runtime_host": "not_recorded",
    }
    if extra:
        config.update({key: _public_config_value(key, value) for key, value in extra.items()})
    return config


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _load_dotenv(path: str | Path) -> dict[str, str]:
    dotenv_path = Path(path)
    if not dotenv_path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in dotenv_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _select_api_models(model_arg: str, min_concurrency: int) -> list[APIModelSpec]:
    if model_arg == "all":
        models = list(API_MODEL_REGISTRY.values())
    else:
        names = _parse_csv(model_arg)
        unknown = [name for name in names if name not in API_MODEL_REGISTRY]
        if unknown:
            raise ValueError(f"Unknown API model(s): {', '.join(unknown)}")
        models = [API_MODEL_REGISTRY[name] for name in names]
    if min_concurrency > 1:
        models = [
            replace(model, concurrency=max(model.concurrency, min_concurrency))
            for model in models
        ]
    return models


def _api_credentials(args: argparse.Namespace) -> tuple[str, str]:
    dotenv_values: dict[str, str] = {}
    for path in [
        Path.cwd() / ".env.local",
        Path.cwd() / ".env",
        Path(__file__).resolve().parent / ".env.local",
        Path(__file__).resolve().parent / ".env",
    ]:
        dotenv_values.update(_load_dotenv(path))

    api_key = (
        os.environ.get(args.api_key_env)
        or os.environ.get("NPCBENCH_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or dotenv_values.get(args.api_key_env)
        or dotenv_values.get("NPCBENCH_API_KEY")
        or dotenv_values.get("OPENAI_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            f"Missing API key. Set {args.api_key_env}, NPCBENCH_API_KEY, or OPENAI_API_KEY."
        )

    base_url = (
        args.base_url
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or dotenv_values.get("OPENAI_BASE_URL")
        or dotenv_values.get("OPENAI_API_BASE")
        or "https://api.openai.com/v1"
    ).rstrip("/")
    if base_url.endswith("/chat/completions"):
        base_url = base_url[: -len("/chat/completions")].rstrip("/")
    return api_key, base_url


def _as_bool(value: str | bool | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _batched(rows: list[Any], batch_size: int) -> list[list[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [rows[idx : idx + batch_size] for idx in range(0, len(rows), batch_size)]


def _set_common_vllm_env(args: argparse.Namespace, gpu_arg: str = "cuda_visible_devices") -> None:
    cuda_visible_devices = getattr(args, gpu_arg, None)
    if cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    os.environ["use_vllm"] = "True"
    os.environ["tensor_parallel_size"] = str(args.tensor_parallel_size)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
    if int(args.tensor_parallel_size) > 1:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if getattr(args, "max_model_len", None):
        os.environ["VLLM_MAX_MODEL_LEN"] = str(args.max_model_len)
        os.environ["max_model_len"] = str(args.max_model_len)
    if getattr(args, "gpu_memory_utilization", None) is not None:
        os.environ["VLLM_GPU_MEMORY_UTILIZATION"] = str(args.gpu_memory_utilization)
    if getattr(args, "vllm_dtype", None):
        os.environ["VLLM_DTYPE"] = str(args.vllm_dtype)
    if getattr(args, "vllm_enforce_eager", None) is not None:
        os.environ["VLLM_ENFORCE_EAGER"] = str(args.vllm_enforce_eager)
    if getattr(args, "vllm_disable_custom_all_reduce", None) is not None:
        os.environ["VLLM_DISABLE_CUSTOM_ALL_REDUCE"] = str(args.vllm_disable_custom_all_reduce)
    if getattr(args, "master_port", None):
        os.environ["master_port"] = str(args.master_port)
    if getattr(args, "top_k", None) is not None:
        os.environ["top_k"] = str(args.top_k)


def _cleanup_vllm_sampler(sampler: Any, label: str) -> list[str]:
    """Release vLLM memory in the order agreed for O1 staged evaluation."""
    warnings: list[str] = []
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel

        destroy_model_parallel()
    except Exception as exc:  # pragma: no cover - depends on vLLM runtime state.
        warnings.append(f"destroy_model_parallel failed for {label}: {exc!r}")

    try:
        llm = getattr(sampler, "llm", None)
        if llm is not None and hasattr(llm, "llm_engine"):
            engine_core = getattr(llm.llm_engine, "engine_core", None)
            if engine_core is not None and hasattr(engine_core, "shutdown"):
                engine_core.shutdown()
    except Exception as exc:  # pragma: no cover - depends on vLLM runtime state.
        warnings.append(f"engine_core.shutdown failed for {label}: {exc!r}")

    try:
        if hasattr(sampler, "llm"):
            del sampler.llm
    except Exception as exc:  # pragma: no cover - depends on wrapper layout.
        warnings.append(f"del sampler.llm failed for {label}: {exc!r}")

    try:
        import ray

        ray.shutdown()
    except Exception as exc:  # pragma: no cover - ray may not be initialized.
        warnings.append(f"ray.shutdown failed for {label}: {exc!r}")

    try:
        del sampler
    except Exception as exc:  # pragma: no cover - defensive only.
        warnings.append(f"del sampler failed for {label}: {exc!r}")

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # pragma: no cover - torch may not be installed locally.
        warnings.append(f"torch.cuda.empty_cache failed for {label}: {exc!r}")
    return warnings


def cmd_prepare(args: argparse.Namespace) -> None:
    items = load_o1_items(args.data_path)
    if args.num_examples is not None:
        items = items[: args.num_examples]
    rubric_items = flatten_dataset_rubrics(items)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(output_dir / "rubric_items.jsonl", [row.to_dict() for row in rubric_items])
    summary = summarize_o1_items(items, rubric_items)
    _json_dump(output_dir / "data_summary.json", summary)
    _json_dump(output_dir / "run_config.json", _run_config(args, {"data_summary": summary}))

    prompt_rows = []
    for item in items:
        prompt_rows.append(
            {
                "item_id": item["item_id"],
                "case_id": item["case_id"],
                "tag": item["tag"],
                "question": item["question"],
                "prompt": render_generation_prompt(item["case_input"], item["question"]),
                "messages": render_generation_messages(item["case_input"], item["question"]),
            }
        )
    write_jsonl(output_dir / "generation_prompts.jsonl", prompt_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def cmd_generate_vllm(args: argparse.Namespace) -> None:
    _set_common_vllm_env(args)
    os.environ["REASONING"] = str(args.reasoning)
    os.environ["max_image_num"] = str(args.max_image_num)
    os.environ["USE_RAG"] = str(args.use_rag)

    medevalkit_path = Path(args.medevalkit_path)
    if str(medevalkit_path) not in sys.path:
        sys.path.insert(0, str(medevalkit_path))

    from LLMs import init_llm

    items = load_o1_items(args.data_path)
    if args.num_examples is not None:
        items = items[: args.num_examples]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args.model_name = args.model_class
    sampler = None
    rows: list[dict[str, Any]] = []
    cleanup_warnings: list[str] = []
    try:
        sampler = init_llm(args)
        for batch in _batched(items, args.batch_size):
            messages_list = [
                {"prompt": render_generation_prompt(item["case_input"], item["question"])}
                for item in batch
            ]
            try:
                responses = sampler.generate_outputs(messages_list)
            except Exception as exc:
                if not args.continue_on_generation_error:
                    raise
                responses = ["" for _ in batch]
                batch_error = repr(exc)
            else:
                batch_error = ""

            if len(responses) != len(batch):
                raise RuntimeError(
                    f"model returned {len(responses)} responses for {len(batch)} O1 prompts"
                )

            for item, prompt_messages, response in zip(batch, messages_list, responses):
                status = "generation_failed" if batch_error else "ok"
                rows.append(
                    {
                        "item_id": item["item_id"],
                        "case_id": item["case_id"],
                        "patient_id": item["patient_id"],
                        "tag": item["tag"],
                        "tag_label": item["tag_label"],
                        "question_slot": item["question_slot"],
                        "model_name": args.model_run_name,
                        "model_class": args.model_class,
                        "model_path": _public_path_ref(args.model_path),
                        "prompt": prompt_messages,
                        "response": response,
                        "generation_status": status,
                        "generation_error": batch_error,
                    }
                )
            write_jsonl(output_dir / "predictions.jsonl", rows)
            _json_dump(output_dir / "results.json", rows)
            print(f"generated {len(rows)}/{len(items)} responses for {args.model_run_name}")
    finally:
        if sampler is not None:
            cleanup_warnings = _cleanup_vllm_sampler(sampler, args.model_run_name)

    generation_config = _run_config(
        args,
        {
            "stage": "generate-vllm",
            "model_run_name": args.model_run_name,
            "model_class": args.model_class,
            "model_path": _public_path_ref(args.model_path),
            "num_predictions": len(rows),
            "cleanup_warnings": cleanup_warnings,
        },
    )
    _json_dump(output_dir / "generation_run_config.json", generation_config)
    _json_dump(output_dir / "run_config.json", {"generation": generation_config})
    if cleanup_warnings:
        print(json.dumps({"cleanup_warnings": cleanup_warnings}, ensure_ascii=False, indent=2))
    print(f"wrote O1 predictions to {output_dir}")


def _api_request_payload(
    model: APIModelSpec,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model.api_model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if model.send_top_p:
        payload["top_p"] = top_p
    return payload


def _is_retryable_http_status(status: int) -> bool:
    return status in {408, 409, 425, 429, 500, 502, 503, 504}


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text") or part.get("content")
                if text:
                    parts.append(str(text))
            elif part:
                parts.append(str(part))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _call_api_o1_completion(
    *,
    api_key: str,
    base_url: str,
    model: APIModelSpec,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_attempts: int,
) -> dict[str, Any]:
    payload = _api_request_payload(
        model,
        messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    request_data = json.dumps(payload).encode("utf-8")
    url = base_url.rstrip("/") + "/chat/completions"
    attempt_latencies: list[float] = []
    last_error = ""

    last_attempt = 0
    for attempt in range(1, max_attempts + 1):
        last_attempt = attempt
        started = time.time()
        request = urllib.request.Request(
            url,
            data=request_data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=model.timeout, context=ssl.create_default_context()
            ) as response:
                response_obj = json.loads(response.read().decode("utf-8"))

            latency = round(time.time() - started, 3)
            attempt_latencies.append(latency)
            choice = (response_obj.get("choices") or [{}])[0]
            message = choice.get("message", {}) if isinstance(choice, dict) else {}
            return {
                "response": _message_content_to_text(message.get("content")),
                "api_attempts": attempt,
                "api_latency_sec": latency,
                "api_attempt_latencies_sec": attempt_latencies,
                "api_error_type": "",
                "finish_reason": choice.get("finish_reason", ""),
                "usage": response_obj.get("usage", {}),
            }
        except urllib.error.HTTPError as exc:
            latency = round(time.time() - started, 3)
            attempt_latencies.append(latency)
            body = exc.read().decode("utf-8", "ignore")[:500]
            last_error = f"HTTP_{exc.code}: {body}"
            if attempt >= max_attempts or not _is_retryable_http_status(exc.code):
                break
        except Exception as exc:  # noqa: BLE001 - errors are persisted for audit.
            latency = round(time.time() - started, 3)
            attempt_latencies.append(latency)
            last_error = f"{type(exc).__name__}: {str(exc)[:300]}"
            if attempt >= max_attempts:
                break
        time.sleep(min(8, 2 * attempt))

    return {
        "response": "",
        "api_attempts": last_attempt,
        "api_latency_sec": None,
        "api_attempt_latencies_sec": attempt_latencies,
        "api_error_type": last_error or "unknown_error",
        "finish_reason": "",
        "usage": {},
    }


def _api_generation_row(
    *,
    item: dict[str, Any],
    item_index: int,
    model: APIModelSpec,
    prompt_messages: list[dict[str, str]],
    api_result: dict[str, Any],
) -> dict[str, Any]:
    response = api_result.get("response", "")
    generation_error = api_result.get("api_error_type", "")
    if generation_error:
        generation_status = "api_failed"
    elif not str(response).strip():
        generation_status = "empty_response"
    else:
        generation_status = "ok"

    return {
        "item_index": item_index,
        "item_id": item["item_id"],
        "case_id": item["case_id"],
        "patient_id": item["patient_id"],
        "tag": item["tag"],
        "tag_label": item["tag_label"],
        "question_slot": item["question_slot"],
        "model_name": model.name,
        "model_class": "api-chat-completions",
        "api_model_id": model.api_model_id,
        "family": model.family,
        "prompt": prompt_messages,
        "response": response,
        "generation_status": generation_status,
        "generation_error": generation_error,
        "api_attempts": api_result.get("api_attempts"),
        "api_latency_sec": api_result.get("api_latency_sec"),
        "api_attempt_latencies_sec": api_result.get("api_attempt_latencies_sec", []),
        "finish_reason": api_result.get("finish_reason", ""),
        "usage": api_result.get("usage", {}),
    }


def _write_api_generation_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    model: APIModelSpec,
    base_url: str,
) -> None:
    ordered_rows = sorted(rows, key=lambda row: int(row.get("item_index", 0)))
    write_jsonl(output_dir / "predictions.jsonl", ordered_rows)
    _json_dump(output_dir / "results.json", ordered_rows)
    generation_config = _run_config(
        args,
        {
            "stage": "generate-api",
            "model_run_name": model.name,
            "model_class": "api-chat-completions",
            "api_model_id": model.api_model_id,
            "family": model.family,
            "base_url": base_url,
            "num_predictions": len(ordered_rows),
            "num_ok_predictions": sum(
                1 for row in ordered_rows if row.get("generation_status") == "ok"
            ),
            "num_failed_predictions": sum(
                1 for row in ordered_rows if row.get("generation_status") != "ok"
            ),
            "api_key_source": "environment_or_dotenv",
            "api_key_recorded": False,
            "send_top_p": model.send_top_p,
        },
    )
    _json_dump(output_dir / "generation_run_config.json", generation_config)
    _json_dump(output_dir / "run_config.json", {"generation": generation_config})


def cmd_generate_api(args: argparse.Namespace) -> None:
    if args.max_attempts < 1:
        raise ValueError("--max_attempts must be >= 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")

    api_key, base_url = _api_credentials(args)
    models = _select_api_models(args.models, args.min_concurrency)
    items = load_o1_items(args.data_path)
    if args.num_examples is not None:
        items = items[: args.num_examples]

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for model in models:
        model = replace(
            model,
            concurrency=max(model.concurrency, args.concurrency),
            max_tokens=args.max_tokens,
        )
        output_dir = output_root / model.name / args.dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, Any]] = []
        existing_by_item: dict[str, dict[str, Any]] = {}
        if (output_dir / "predictions.jsonl").exists() and not args.no_resume:
            existing_rows = read_jsonl(output_dir / "predictions.jsonl")
            existing_by_item = {
                row["item_id"]: row
                for row in _prediction_map(existing_rows).values()
                if row.get("generation_status") == "ok" and row.get("response")
            }
            rows.extend(existing_by_item.values())

        pending_items = [
            (idx, item) for idx, item in enumerate(items) if item["item_id"] not in existing_by_item
        ]
        print(
            f"API O1 generation for {model.name}: "
            f"{len(existing_by_item)} resumed, {len(pending_items)} pending"
        )

        def generate_one(idx_item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
            item_index, item = idx_item
            prompt_messages = render_generation_messages(item["case_input"], item["question"])
            api_result = _call_api_o1_completion(
                api_key=api_key,
                base_url=base_url,
                model=model,
                messages=prompt_messages,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                max_attempts=args.max_attempts,
            )
            return _api_generation_row(
                item=item,
                item_index=item_index,
                model=model,
                prompt_messages=prompt_messages,
                api_result=api_result,
            )

        completed_since_flush = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=model.concurrency) as executor:
            future_to_item = {
                executor.submit(generate_one, idx_item): idx_item for idx_item in pending_items
            }
            for future in concurrent.futures.as_completed(future_to_item):
                row = future.result()
                rows.append(row)
                completed_since_flush += 1
                if completed_since_flush >= args.flush_every:
                    _write_api_generation_outputs(output_dir, rows, args, model, base_url)
                    completed_since_flush = 0
                    print(f"generated {len(rows)}/{len(items)} responses for {model.name}")

        _write_api_generation_outputs(output_dir, rows, args, model, base_url)
        failures = sum(1 for row in rows if row.get("generation_status") != "ok")
        print(
            f"wrote API O1 predictions to {output_dir} "
            f"({len(rows)}/{len(items)} rows, failures={failures})"
        )


def _get_judge_sampler(args: argparse.Namespace):
    if args.judge_provider == "openai":
        from sampler.chat_completion_sampler import OPENAI_SYSTEM_MESSAGE_API, ChatCompletionSampler

        return ChatCompletionSampler(
            model=args.judge_model,
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            temperature=args.judge_temperature,
            max_tokens=args.judge_max_tokens,
        )
    if args.judge_provider == "qwen-local":
        from sampler.qwen_chat_sampler import Qwen3LocalSampler

        return Qwen3LocalSampler(
            model_path=args.judge_model,
            temperature=args.judge_temperature,
            max_new_tokens=args.judge_max_tokens,
            top_p=args.judge_top_p,
            device_map=args.judge_device,
        )
    raise ValueError(f"Unsupported judge_provider: {args.judge_provider}")


def _prediction_map(predictions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapped = {}
    seen = set()
    for index, row in enumerate(predictions, 1):
        item_id = row["item_id"]
        if item_id in seen:
            raise ValueError(f"Duplicate prediction item_id {item_id!r} at record {index}")
        seen.add(item_id)
        if row.get("generation_status", "ok") not in {"ok", "oracle_from_merged_score_answer"}:
            continue
        mapped[item_id] = row
    return mapped


def cmd_write_judge_prompts(args: argparse.Namespace) -> None:
    items = load_o1_items(args.data_path)
    if args.num_examples is not None:
        items = items[: args.num_examples]
    rubric_items = flatten_dataset_rubrics(items)
    predictions = _prediction_map(read_jsonl(args.predictions_path))
    item_by_id = {item["item_id"]: item for item in items}
    rows = []
    for rubric_item in rubric_items:
        pred = predictions.get(rubric_item.item_id)
        if pred is None:
            continue
        item = item_by_id[rubric_item.item_id]
        rows.append(
            {
                "item_id": rubric_item.item_id,
                "rubric_id": rubric_item.rubric_id,
                "prompt": render_judge_prompt(
                    item["case_input"],
                    item["question"],
                    pred.get("response", ""),
                    rubric_item,
                ),
            }
        )
        if args.max_prompts is not None and len(rows) >= args.max_prompts:
            break
    write_jsonl(args.output_path, rows)
    print(f"wrote {len(rows)} judge prompts to {args.output_path}")


def cmd_build_judge_tasks(args: argparse.Namespace) -> None:
    items = load_o1_items(args.data_path)
    if args.num_examples is not None:
        items = items[: args.num_examples]
    if args.rubric_items_path:
        rubric_items = _load_rubric_items(args.rubric_items_path)
        if args.num_examples is not None:
            allowed = {item["item_id"] for item in items}
            rubric_items = [row for row in rubric_items if row.item_id in allowed]
    else:
        rubric_items = flatten_dataset_rubrics(items)

    predictions = _prediction_map(read_jsonl(args.predictions_path))
    item_by_id = {item["item_id"]: item for item in items}
    rows: list[dict[str, Any]] = []
    for rubric_item in rubric_items:
        pred = predictions.get(rubric_item.item_id)
        if pred is None:
            continue
        item = item_by_id[rubric_item.item_id]
        model_name = args.model_name or pred.get("model_name")
        rows.append(
            {
                "task_id": rubric_item.rubric_id,
                "item_id": rubric_item.item_id,
                "rubric_id": rubric_item.rubric_id,
                "case_id": rubric_item.case_id,
                "patient_id": rubric_item.patient_id,
                "tag": rubric_item.tag,
                "tag_label": rubric_item.tag_label,
                "question_slot": rubric_item.question_slot,
                "model_name": model_name,
                "rubric_section": rubric_item.section,
                "messages": render_judge_messages(
                    item["case_input"],
                    item["question"],
                    pred.get("response", ""),
                    rubric_item,
                ),
            }
        )
        if args.max_tasks is not None and len(rows) >= args.max_tasks:
            break
    write_jsonl(args.output_path, rows)
    print(f"wrote {len(rows)} judge tasks to {args.output_path}")


def cmd_judge(args: argparse.Namespace) -> None:
    items = load_o1_items(args.data_path)
    if args.num_examples is not None:
        items = items[: args.num_examples]
    rubric_items = flatten_dataset_rubrics(items)
    predictions = _prediction_map(read_jsonl(args.predictions_path))
    item_by_id = {item["item_id"]: item for item in items}
    sampler = _get_judge_sampler(args)
    if args.judge_provider == "qwen-local" and args.n_threads != 1:
        print("qwen-local judge uses one in-process model; forcing n_threads=1")
        args.n_threads = 1

    tasks = []
    for rubric_item in rubric_items:
        prediction = predictions.get(rubric_item.item_id)
        if prediction is None:
            continue
        item = item_by_id[rubric_item.item_id]
        tasks.append((item, prediction, rubric_item))
        if args.max_judgments is not None and len(tasks) >= args.max_judgments:
            break

    def grade_one(task: tuple[dict[str, Any], dict[str, Any], O1RubricItem]) -> dict[str, Any]:
        item, prediction, rubric_item = task
        messages = render_judge_messages(
            item["case_input"],
            item["question"],
            prediction.get("response", ""),
            rubric_item,
        )
        raw_response = ""
        last_error = ""
        for retry_idx in range(args.max_retries + 1):
            try:
                response = sampler(messages)
                raw_response = response.response_text
                parsed = parse_judge_json(raw_response)
                valid, reason = validate_judge_output(parsed)
                if valid:
                    return {
                        "item_id": rubric_item.item_id,
                        "rubric_id": rubric_item.rubric_id,
                        "case_id": rubric_item.case_id,
                        "tag": rubric_item.tag,
                        "judge_model": args.judge_model,
                        "judge_provider": args.judge_provider,
                        "judge_status": "ok",
                        "retry_count": retry_idx,
                        "criteria_met": parsed["criteria_met"],
                        "explanation": parsed["explanation"],
                        "raw_judge_response": raw_response,
                    }
                last_error = reason
            except Exception as exc:
                last_error = repr(exc)
        return {
            "item_id": rubric_item.item_id,
            "rubric_id": rubric_item.rubric_id,
            "case_id": rubric_item.case_id,
            "tag": rubric_item.tag,
            "judge_model": args.judge_model,
            "judge_provider": args.judge_provider,
            "judge_status": "parse_failed",
            "retry_count": args.max_retries,
            "error": last_error,
            "raw_judge_response": raw_response,
        }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_threads) as executor:
        futures = [executor.submit(grade_one, task) for task in tasks]
        for idx, future in enumerate(concurrent.futures.as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            if idx % args.flush_every == 0:
                write_jsonl(output_path, rows)
                print(f"judged {idx}/{len(tasks)}")
    write_jsonl(output_path, rows)
    print(f"wrote {len(rows)} judgments to {output_path}")


class DirectVLLMJudge:
    def __init__(self, args: argparse.Namespace):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.judge_model_path,
            trust_remote_code=True,
        )
        llm_kwargs: dict[str, Any] = {
            "model": args.judge_model_path,
            "tensor_parallel_size": args.judge_tensor_parallel_size,
            "trust_remote_code": True,
            "gpu_memory_utilization": args.judge_gpu_memory_utilization,
        }
        if args.judge_max_model_len:
            llm_kwargs["max_model_len"] = args.judge_max_model_len
        if args.judge_dtype:
            llm_kwargs["dtype"] = args.judge_dtype
        if args.judge_enforce_eager is not None:
            llm_kwargs["enforce_eager"] = _as_bool(args.judge_enforce_eager)
        if args.judge_disable_custom_all_reduce is not None:
            llm_kwargs["disable_custom_all_reduce"] = _as_bool(
                args.judge_disable_custom_all_reduce
            )
        if args.judge_master_port:
            llm_kwargs["master_port"] = int(args.judge_master_port)
        self.llm = LLM(**llm_kwargs)
        self.sampling_params = SamplingParams(
            temperature=args.judge_temperature,
            top_p=args.judge_top_p,
            max_tokens=args.judge_max_tokens,
        )

    def _to_prompt(self, messages: list[dict[str, str]]) -> str:
        template_kwargs: dict[str, Any] = {}
        if self.args.judge_reasoning_effort:
            template_kwargs["reasoning_effort"] = self.args.judge_reasoning_effort
        if self.args.judge_enable_thinking != "auto":
            template_kwargs["enable_thinking"] = _as_bool(self.args.judge_enable_thinking)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )

    def generate_batch(self, messages_batch: list[list[dict[str, str]]]) -> list[str]:
        prompts = [self._to_prompt(messages) for messages in messages_batch]
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]


def _set_judge_vllm_env(args: argparse.Namespace) -> None:
    runtime = argparse.Namespace(
        cuda_visible_devices=args.judge_cuda_visible_devices,
        tensor_parallel_size=args.judge_tensor_parallel_size,
        max_model_len=args.judge_max_model_len,
        gpu_memory_utilization=args.judge_gpu_memory_utilization,
        vllm_dtype=args.judge_dtype,
        vllm_enforce_eager=args.judge_enforce_eager,
        vllm_disable_custom_all_reduce=args.judge_disable_custom_all_reduce,
        master_port=args.judge_master_port,
        top_k=None,
    )
    _set_common_vllm_env(runtime)


def _judge_task_file(
    judge: DirectVLLMJudge,
    tasks_path: str | Path,
    output_path: str | Path,
    args: argparse.Namespace,
) -> None:
    tasks = read_jsonl(tasks_path)
    if args.max_judgments is not None:
        tasks = tasks[: args.max_judgments]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    done: set[str] = set()
    if output_path.exists() and not args.no_resume:
        existing_rows = read_jsonl(output_path)
        rows = [
            row
            for row in existing_rows
            if row.get("judge_status") == "ok" and row.get("rubric_id")
        ]
        done = {str(row.get("rubric_id")) for row in rows if row.get("rubric_id")}
        tasks = [task for task in tasks if str(task.get("rubric_id")) not in done]
        dropped = len(existing_rows) - len(rows)
        if dropped:
            print(f"resuming {output_path}: retrying {dropped} non-ok judgments")

    pending = tasks
    for retry_idx in range(args.max_retries + 1):
        if not pending:
            break
        retry_pending: list[dict[str, Any]] = []
        for batch in _batched(pending, args.judge_batch_size):
            raw_responses = judge.generate_batch([task["messages"] for task in batch])
            for task, raw_response in zip(batch, raw_responses):
                try:
                    parsed = parse_judge_json(raw_response)
                    valid, reason = validate_judge_output(parsed)
                except Exception as exc:
                    parsed = {}
                    valid = False
                    reason = repr(exc)
                if valid:
                    rows.append(
                        {
                            "item_id": task["item_id"],
                            "rubric_id": task["rubric_id"],
                            "case_id": task["case_id"],
                            "tag": task["tag"],
                            "model_name": task.get("model_name"),
                            "judge_model": args.judge_model_name,
                            "judge_model_path": _public_path_ref(args.judge_model_path),
                            "judge_provider": "direct-vllm",
                            "judge_status": "ok",
                            "retry_count": retry_idx,
                            "criteria_met": parsed["criteria_met"],
                            "explanation": parsed["explanation"],
                            "raw_judge_response": raw_response,
                        }
                    )
                elif retry_idx < args.max_retries:
                    retry_pending.append(task)
                else:
                    rows.append(
                        {
                            "item_id": task["item_id"],
                            "rubric_id": task["rubric_id"],
                            "case_id": task["case_id"],
                            "tag": task["tag"],
                            "model_name": task.get("model_name"),
                            "judge_model": args.judge_model_name,
                            "judge_model_path": _public_path_ref(args.judge_model_path),
                            "judge_provider": "direct-vllm",
                            "judge_status": "parse_failed",
                            "retry_count": retry_idx,
                            "error": reason,
                            "raw_judge_response": raw_response,
                        }
                    )
            if len(rows) % args.flush_every < len(batch):
                write_jsonl(output_path, rows)
                print(f"judged {len(rows)} rows -> {output_path}")
        pending = retry_pending
    write_jsonl(output_path, rows)
    judge_config = {
        "stage": "judge-vllm",
        "tasks_path": _public_path_ref(tasks_path),
        "output_path": _public_path_ref(output_path),
        "judge_model_name": args.judge_model_name,
        "judge_model_path": _public_path_ref(args.judge_model_path),
        "judge_provider": "direct-vllm",
        "judge_tensor_parallel_size": args.judge_tensor_parallel_size,
        "judge_cuda_visible_devices": args.judge_cuda_visible_devices,
        "judge_batch_size": args.judge_batch_size,
        "judge_temperature": args.judge_temperature,
        "judge_top_p": args.judge_top_p,
        "judge_max_tokens": args.judge_max_tokens,
        "judge_max_model_len": args.judge_max_model_len,
        "num_judgments": len(rows),
    }
    _json_dump(output_path.parent / "judge_run_config.json", judge_config)
    print(f"wrote {len(rows)} judgments to {output_path}")


def _api_usage_totals(rows: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for row in rows:
        usage = row.get("usage")
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    return totals


def _judge_api_task_file(
    tasks_path: str | Path,
    output_path: str | Path,
    args: argparse.Namespace,
    *,
    api_key: str,
    base_url: str,
    judge_model: APIModelSpec,
) -> None:
    tasks = read_jsonl(tasks_path)
    if args.max_judgments is not None:
        tasks = tasks[: args.max_judgments]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if output_path.exists() and not args.no_resume:
        existing_rows = read_jsonl(output_path)
        rows = [
            row
            for row in existing_rows
            if row.get("judge_status") == "ok" and row.get("rubric_id")
        ]
        done = {str(row.get("rubric_id")) for row in rows if row.get("rubric_id")}
        tasks = [task for task in tasks if str(task.get("rubric_id")) not in done]
        dropped = len(existing_rows) - len(rows)
        if dropped:
            print(f"resuming {output_path}: retrying {dropped} non-ok judgments")

    def judge_one(task: dict[str, Any]) -> dict[str, Any]:
        raw_response = ""
        last_error = ""
        last_api_result: dict[str, Any] = {}
        for retry_idx in range(args.max_retries + 1):
            api_result = _call_api_o1_completion(
                api_key=api_key,
                base_url=base_url,
                model=judge_model,
                messages=task["messages"],
                temperature=args.judge_temperature,
                top_p=args.judge_top_p,
                max_tokens=args.judge_max_tokens,
                max_attempts=args.api_max_attempts,
            )
            last_api_result = api_result
            raw_response = api_result.get("response", "")
            if api_result.get("api_error_type"):
                last_error = api_result.get("api_error_type", "")
            else:
                try:
                    parsed = parse_judge_json(raw_response)
                    valid, reason = validate_judge_output(parsed)
                except Exception as exc:  # noqa: BLE001 - persisted for audit.
                    parsed = {}
                    valid = False
                    reason = repr(exc)
                if valid:
                    return {
                        "item_id": task["item_id"],
                        "rubric_id": task["rubric_id"],
                        "case_id": task["case_id"],
                        "tag": task["tag"],
                        "model_name": task.get("model_name"),
                        "judge_model": args.judge_model_name,
                        "judge_model_path": args.judge_model_id,
                        "judge_provider": "api-chat-completions",
                        "judge_status": "ok",
                        "retry_count": retry_idx,
                        "criteria_met": parsed["criteria_met"],
                        "explanation": parsed["explanation"],
                        "raw_judge_response": raw_response,
                        "finish_reason": api_result.get("finish_reason", ""),
                        "usage": api_result.get("usage", {}),
                        "api_attempts": api_result.get("api_attempts"),
                        "api_latency_sec": api_result.get("api_latency_sec"),
                        "api_attempt_latencies_sec": api_result.get(
                            "api_attempt_latencies_sec", []
                        ),
                    }
                last_error = reason
        return {
            "item_id": task["item_id"],
            "rubric_id": task["rubric_id"],
            "case_id": task["case_id"],
            "tag": task["tag"],
            "model_name": task.get("model_name"),
            "judge_model": args.judge_model_name,
            "judge_model_path": args.judge_model_id,
            "judge_provider": "api-chat-completions",
            "judge_status": "parse_failed",
            "retry_count": args.max_retries,
            "error": last_error,
            "raw_judge_response": raw_response,
            "finish_reason": last_api_result.get("finish_reason", ""),
            "usage": last_api_result.get("usage", {}),
            "api_attempts": last_api_result.get("api_attempts"),
            "api_latency_sec": last_api_result.get("api_latency_sec"),
            "api_attempt_latencies_sec": last_api_result.get(
                "api_attempt_latencies_sec", []
            ),
        }

    completed_since_flush = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.judge_concurrency) as executor:
        future_to_task = {executor.submit(judge_one, task): task for task in tasks}
        for future in concurrent.futures.as_completed(future_to_task):
            rows.append(future.result())
            completed_since_flush += 1
            if completed_since_flush >= args.flush_every:
                write_jsonl(output_path, rows)
                completed_since_flush = 0
                print(f"judged {len(rows)} rows -> {output_path}")

    write_jsonl(output_path, rows)
    judge_config = _run_config(
        args,
        {
            "stage": "judge-api-tasks",
            "tasks_path": str(tasks_path),
            "output_path": str(output_path),
            "judge_model_name": args.judge_model_name,
            "judge_model_id": args.judge_model_id,
            "judge_provider": "api-chat-completions",
            "base_url": base_url,
            "api_key_source": "environment_or_dotenv",
            "api_key_recorded": False,
            "judge_concurrency": args.judge_concurrency,
            "judge_temperature": args.judge_temperature,
            "judge_top_p": args.judge_top_p,
            "judge_max_tokens": args.judge_max_tokens,
            "judge_timeout": args.judge_timeout,
            "send_top_p": judge_model.send_top_p,
            "num_judgments": len(rows),
            "usage_totals": _api_usage_totals(rows),
        },
    )
    _json_dump(output_path.parent / "judge_run_config.json", judge_config)
    print(f"wrote {len(rows)} API judgments to {output_path}")


def cmd_judge_api_tasks(args: argparse.Namespace) -> None:
    if len(args.tasks_paths) != len(args.output_paths):
        raise ValueError("--tasks_paths and --output_paths must have the same length")
    if args.judge_concurrency < 1:
        raise ValueError("--judge_concurrency must be >= 1")
    if args.api_max_attempts < 1:
        raise ValueError("--api_max_attempts must be >= 1")
    api_key, base_url = _api_credentials(args)
    judge_model = APIModelSpec(
        name=args.judge_model_name,
        api_model_id=args.judge_model_id,
        family="API Judge",
        concurrency=args.judge_concurrency,
        timeout=args.judge_timeout,
        max_tokens=args.judge_max_tokens,
        send_top_p=not args.judge_no_top_p,
    )
    for tasks_path, output_path in zip(args.tasks_paths, args.output_paths):
        _judge_api_task_file(
            tasks_path,
            output_path,
            args,
            api_key=api_key,
            base_url=base_url,
            judge_model=judge_model,
        )


def cmd_judge_vllm(args: argparse.Namespace) -> None:
    if len(args.tasks_paths) != len(args.output_paths):
        raise ValueError("--tasks_paths and --output_paths must have the same length")
    _set_judge_vllm_env(args)
    judge: DirectVLLMJudge | None = None
    cleanup_warnings: list[str] = []
    try:
        judge = DirectVLLMJudge(args)
        for tasks_path, output_path in zip(args.tasks_paths, args.output_paths):
            _judge_task_file(judge, tasks_path, output_path, args)
    finally:
        if judge is not None:
            cleanup_warnings = _cleanup_vllm_sampler(judge, args.judge_model_name)
    if cleanup_warnings:
        print(json.dumps({"cleanup_warnings": cleanup_warnings}, ensure_ascii=False, indent=2))


def cmd_score(args: argparse.Namespace) -> None:
    items = load_o1_items(args.data_path)
    if args.num_examples is not None:
        items = items[: args.num_examples]
    rubric_items = _load_rubric_items(args.rubric_items_path)
    if args.num_examples is not None:
        allowed = {item["item_id"] for item in items}
        rubric_items = [row for row in rubric_items if row.item_id in allowed]
    judgments = read_jsonl(args.judgments_path)
    item_scores, metrics = score_dataset(items, rubric_items, judgments)
    if args.model_name:
        for row in item_scores:
            row["model_name"] = args.model_name
        metrics["model_name"] = args.model_name
    write_scores(args.output_dir, item_scores, metrics)
    output_dir = Path(args.output_dir)
    scoring_config = _run_config(args, {"stage": "score", "metrics": metrics})
    _json_dump(output_dir / "scoring_run_config.json", scoring_config)
    merged_config: dict[str, Any] = {"scoring": scoring_config}
    for name in ["generation", "judge"]:
        stage_path = output_dir / f"{name}_run_config.json"
        if stage_path.exists():
            merged_config[name] = _json_load(stage_path)
    _json_dump(output_dir / "run_config.json", merged_config)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NPCBench O1 ConsultRubrics evaluator")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="Flatten O1 rubrics and write generation prompts")
    p.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_examples", type=int)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("generate-vllm", help="Generate O1 responses with a MedEvalKit vLLM model class")
    p.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_run_name", required=True)
    p.add_argument("--model_class", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--medevalkit_path", default=os.environ.get("MEDEVALKIT_ROOT", "MedEvalKit"))
    p.add_argument("--num_examples", type=int)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--cuda_visible_devices", default=None)
    p.add_argument("--tensor_parallel_size", type=int, default=2)
    p.add_argument("--max_model_len", type=int, default=20000)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--vllm_dtype", default=None)
    p.add_argument("--vllm_enforce_eager", default=None)
    p.add_argument("--vllm_disable_custom_all_reduce", default=None)
    p.add_argument("--master_port", default=None)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--max_image_num", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--reasoning", default="False")
    p.add_argument("--use_rag", default="False")
    p.add_argument("--continue_on_generation_error", action="store_true")
    p.set_defaults(func=cmd_generate_vllm)

    p = sub.add_parser("generate-api", help="Generate O1 responses through an OpenAI-compatible API")
    p.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    p.add_argument("--output_root", default=os.environ.get("NPCBENCH_RESULTS_ROOT", "eval_results"))
    p.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    p.add_argument("--models", default="all", help="Comma-separated API model names or 'all'.")
    p.add_argument("--num_examples", type=int)
    p.add_argument("--base_url", default=None)
    p.add_argument("--api_key_env", default="NPCBENCH_API_KEY")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--min_concurrency", type=int, default=1)
    p.add_argument("--max_attempts", type=int, default=2)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--flush_every", type=int, default=5)
    p.add_argument("--no_resume", action="store_true")
    p.set_defaults(func=cmd_generate_api)

    p = sub.add_parser("write-judge-prompts", help="Render judge prompts without calling a judge model")
    p.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    p.add_argument("--predictions_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--num_examples", type=int)
    p.add_argument("--max_prompts", type=int)
    p.set_defaults(func=cmd_write_judge_prompts)

    p = sub.add_parser("build-judge-tasks", help="Build criterion-level judge tasks from model predictions")
    p.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    p.add_argument("--predictions_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--rubric_items_path")
    p.add_argument("--model_name")
    p.add_argument("--num_examples", type=int)
    p.add_argument("--max_tasks", type=int)
    p.set_defaults(func=cmd_build_judge_tasks)

    p = sub.add_parser("judge", help="Judge existing predictions criterion by criterion")
    p.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    p.add_argument("--predictions_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--num_examples", type=int)
    p.add_argument("--max_judgments", type=int)
    p.add_argument("--judge_provider", choices=["openai", "qwen-local"], default="openai")
    p.add_argument("--judge_model", required=True)
    p.add_argument("--judge_temperature", type=float, default=0.0)
    p.add_argument("--judge_top_p", type=float, default=1.0)
    p.add_argument("--judge_max_tokens", type=int, default=1024)
    p.add_argument("--judge_device", default="cuda:0")
    p.add_argument("--n_threads", type=int, default=8)
    p.add_argument("--max_retries", type=int, default=2)
    p.add_argument("--flush_every", type=int, default=20)
    p.set_defaults(func=cmd_judge)

    p = sub.add_parser(
        "judge-api-tasks",
        help="Judge prebuilt criterion-level O1 tasks through an OpenAI-compatible API",
    )
    p.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    p.add_argument("--tasks_paths", nargs="+", required=True)
    p.add_argument("--output_paths", nargs="+", required=True)
    p.add_argument("--judge_model_name", required=True)
    p.add_argument("--judge_model_id", required=True)
    p.add_argument("--base_url", default=None)
    p.add_argument("--api_key_env", default="NPCBENCH_API_KEY")
    p.add_argument("--judge_concurrency", type=int, default=4)
    p.add_argument("--judge_timeout", type=int, default=180)
    p.add_argument("--judge_temperature", type=float, default=0.0)
    p.add_argument("--judge_top_p", type=float, default=1.0)
    p.add_argument("--judge_no_top_p", action="store_true")
    p.add_argument("--judge_max_tokens", type=int, default=1024)
    p.add_argument("--api_max_attempts", type=int, default=2)
    p.add_argument("--max_judgments", type=int)
    p.add_argument("--max_retries", type=int, default=2)
    p.add_argument("--flush_every", type=int, default=20)
    p.add_argument("--no_resume", action="store_true")
    p.set_defaults(func=cmd_judge_api_tasks)

    p = sub.add_parser("judge-vllm", help="Judge criterion-level O1 tasks with one direct vLLM judge process")
    p.add_argument("--tasks_paths", nargs="+", required=True)
    p.add_argument("--output_paths", nargs="+", required=True)
    p.add_argument("--judge_model_name", required=True)
    p.add_argument("--judge_model_path", required=True)
    p.add_argument("--judge_cuda_visible_devices", default=None)
    p.add_argument("--judge_tensor_parallel_size", type=int, default=2)
    p.add_argument("--judge_batch_size", type=int, default=16)
    p.add_argument("--judge_max_model_len", type=int, default=20000)
    p.add_argument("--judge_gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--judge_dtype", default=None)
    p.add_argument("--judge_enforce_eager", default=None)
    p.add_argument("--judge_disable_custom_all_reduce", default=None)
    p.add_argument("--judge_master_port", default=None)
    p.add_argument("--judge_temperature", type=float, default=0.0)
    p.add_argument("--judge_top_p", type=float, default=1.0)
    p.add_argument("--judge_max_tokens", type=int, default=768)
    p.add_argument("--judge_enable_thinking", choices=["auto", "true", "false"], default="auto")
    p.add_argument("--judge_reasoning_effort", choices=["low", "medium", "high"], default=None)
    p.add_argument("--max_judgments", type=int)
    p.add_argument("--max_retries", type=int, default=2)
    p.add_argument("--flush_every", type=int, default=100)
    p.add_argument("--no_resume", action="store_true")
    p.set_defaults(func=cmd_judge_vllm)

    p = sub.add_parser("score", help="Compute deterministic O1 scores from judgments")
    p.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    p.add_argument("--rubric_items_path", required=True)
    p.add_argument("--judgments_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_examples", type=int)
    p.add_argument("--model_name")
    p.set_defaults(func=cmd_score)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
