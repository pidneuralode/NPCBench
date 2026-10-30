import os
from typing import Any, Dict, List

from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams


class InternVL:
    """vLLM wrapper for InternVL checkpoints.

    InternVL3.5 uses the tokenizer/processor chat template to emit
    ``<IMG_CONTEXT>`` image placeholders. Older hand-written InternVL
    conversation templates emit ``<image>``, which vLLM cannot align with the
    model's multimodal processor and fails prompt replacement. This wrapper
    delegates prompt rendering to AutoProcessor for both text-only and image
    prompts while preserving the MedEvalKit generate_output(s) interface.
    """

    def __init__(self, model_path: str, args: Any):
        super().__init__()
        self.processor = None
        self.tokenizer = None
        self.text_only = os.environ.get("INTERNVL_TEXT_ONLY", "False").lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        if self.text_only:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
        else:
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
        llm_kwargs = {
            "model": model_path,
            "trust_remote_code": True,
            "tensor_parallel_size": int(os.environ.get("tensor_parallel_size", 4)),
            "enforce_eager": True,
            "gpu_memory_utilization": float(
                os.environ.get(
                    "INTERNVL_GPU_MEMORY_UTILIZATION",
                    os.environ.get("gpu_memory_utilization", 0.7),
                )
            ),
            "hf_overrides": {"tie_word_embeddings": False},
            "limit_mm_per_prompt": {"image": int(os.environ.get("max_image_num", 1))},
        }
        max_model_len = os.environ.get("INTERNVL_MAX_MODEL_LEN") or os.environ.get(
            "VLLM_MAX_MODEL_LEN"
        )
        if max_model_len:
            llm_kwargs["max_model_len"] = int(max_model_len)
        if os.environ.get("master_port"):
            llm_kwargs["master_port"] = int(os.environ["master_port"])
        if os.environ.get("VLLM_DISABLE_CUSTOM_ALL_REDUCE"):
            llm_kwargs["disable_custom_all_reduce"] = (
                os.environ["VLLM_DISABLE_CUSTOM_ALL_REDUCE"].lower()
                in {"1", "true", "yes", "y", "on"}
            )

        mm_processor_kwargs = {}
        crop_to_patches = os.environ.get("INTERNVL_CROP_TO_PATCHES")
        if crop_to_patches is not None:
            mm_processor_kwargs["crop_to_patches"] = crop_to_patches.lower() in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }
        max_patches = os.environ.get("INTERNVL_MAX_PATCHES")
        if max_patches:
            mm_processor_kwargs["max_patches"] = int(max_patches)
        min_patches = os.environ.get("INTERNVL_MIN_PATCHES")
        if min_patches:
            mm_processor_kwargs["min_patches"] = int(min_patches)
        if mm_processor_kwargs:
            llm_kwargs["mm_processor_kwargs"] = mm_processor_kwargs

        for key in ("max_num_seqs", "max_num_batched_tokens", "kv_cache_memory_bytes"):
            value = os.environ.get("VLLM_" + key.upper())
            if value is not None:
                llm_kwargs[key] = int(value)
        self.llm = LLM(**llm_kwargs)
        self.sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_tokens=args.max_new_tokens,
            stop_token_ids=[],
        )

    @staticmethod
    def _load_image(image):
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image path not found: {image}")
            return Image.open(image).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image)}")

    def _build_messages(self, messages: Dict[str, Any]):
        prompt = messages["prompt"]
        current_messages: List[Dict[str, Any]] = []
        if "system" in messages:
            current_messages.append({"role": "system", "content": messages["system"]})

        images = []
        if "image" in messages:
            images = [self._load_image(messages["image"])]
        elif "images" in messages:
            images = [self._load_image(image) for image in messages["images"]]

        if images:
            content = [{"type": "image", "image": image} for image in images]
            content.append({"type": "text", "text": prompt})
            current_messages.append({"role": "user", "content": content})
        else:
            current_messages.append({"role": "user", "content": prompt})
        return current_messages, images

    def process_messages(self, messages):
        images = []
        if isinstance(messages, list):
            current_messages = messages
        elif isinstance(messages, dict):
            if "messages" in messages and isinstance(messages["messages"], list):
                current_messages = messages["messages"]
            else:
                current_messages, images = self._build_messages(messages)
        else:
            raise TypeError(f"Unsupported messages type: {type(messages)}")

        if self.processor is not None:
            prompt = self.processor.apply_chat_template(
                current_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            prompt = self.tokenizer.apply_chat_template(
                current_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        llm_inputs = {"prompt": prompt}
        if images:
            llm_inputs["multi_modal_data"] = {"image": images}
        return llm_inputs

    def generate_output(self, messages):
        llm_inputs = self.process_messages(messages)
        outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params)
        return outputs[0].outputs[0].text

    def generate_outputs(self, messages_list):
        llm_inputs_list = [self.process_messages(messages) for messages in messages_list]
        outputs = self.llm.generate(llm_inputs_list, sampling_params=self.sampling_params)
        return [output.outputs[0].text for output in outputs]
