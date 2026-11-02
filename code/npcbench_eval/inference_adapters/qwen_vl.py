import json
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None


class Qwen35_36_MLLM:
    """vLLM wrapper for Qwen3.5/Qwen3.6 image and native-video inputs."""

    def __init__(self, model_path: str, args: Any):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.enable_video = self._parse_bool(
            os.environ.get("NPCFULLCARE_ENABLE_VIDEO", "False")
        )
        self.cine_input = os.environ.get(
            "NPCFULLCARE_CINE_INPUT", "separate"
        ).strip()
        if self.cine_input not in {"separate", "concatenated"}:
            raise ValueError(
                "NPCFULLCARE_CINE_INPUT must be separate or concatenated"
            )
        self.video_max_pixels = int(
            os.environ.get("NPCFULLCARE_VIDEO_MAX_PIXELS", str(256 * 28 * 28))
        )
        self.video_min_pixels = int(
            os.environ.get("NPCFULLCARE_VIDEO_MIN_PIXELS", str(16 * 28 * 28))
        )
        audit_path = os.environ.get("NPCFULLCARE_VIDEO_AUDIT_JSONL", "").strip()
        self.video_audit_path = Path(audit_path) if audit_path else None
        self._audit_lock = threading.Lock()

        limit_mm_per_prompt = {"image": int(args.max_image_num)}
        if self.enable_video:
            limit_mm_per_prompt["video"] = int(
                os.environ.get("NPCFULLCARE_MAX_VIDEO_NUM", "4")
            )
        llm_kwargs = {
            "model": model_path,
            "trust_remote_code": True,
            "tensor_parallel_size": int(os.environ.get("tensor_parallel_size", 1)),
            "dtype": os.environ.get("VLLM_DTYPE", "auto"),
            "max_model_len": int(os.environ.get("VLLM_MAX_MODEL_LEN", 8192)),
            "gpu_memory_utilization": float(
                os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", 0.90)
            ),
            "enforce_eager": self._parse_bool(
                os.environ.get("VLLM_ENFORCE_EAGER", "False")
            ),
            "disable_custom_all_reduce": self._parse_bool(
                os.environ.get("VLLM_DISABLE_CUSTOM_ALL_REDUCE", "False")
            ),
            "limit_mm_per_prompt": limit_mm_per_prompt,
        }
        for key in ("max_num_seqs", "max_num_batched_tokens", "kv_cache_memory_bytes"):
            if os.environ.get("VLLM_" + key.upper()):
                llm_kwargs[key] = int(os.environ["VLLM_" + key.upper()])
        if os.environ.get("master_port"):
            llm_kwargs["master_port"] = int(os.environ["master_port"])
        if os.environ.get("NPCBENCH_MM_PROCESSOR_KWARGS"):
            llm_kwargs["mm_processor_kwargs"] = json.loads(os.environ["NPCBENCH_MM_PROCESSOR_KWARGS"])
        self.llm = LLM(**llm_kwargs)

        self.sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_tokens=args.max_new_tokens,
            top_k=int(os.environ.get("top_k", 20)),
            stop_token_ids=[],
        )

    @staticmethod
    def _parse_bool(value: str) -> bool:
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _load_image(image):
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image path not found: {image}")
            return Image.open(image).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image)}")

    @staticmethod
    def _load_video_manifest(path):
        if not path:
            return {}
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Video manifest not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _probe_video_frame_count(path):
        """Return a physical frame count without requiring a system ffprobe."""
        ffprobe = shutil.which("ffprobe")
        if ffprobe:
            command = [
                ffprobe,
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames,nb_frames",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                str(path),
            ]
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            counts = []
            for line in completed.stdout.splitlines():
                value = line.strip()
                if value.isdigit() and int(value) > 0:
                    counts.append(int(value))
            if counts:
                return max(counts)

        # The formal evaluation environment includes OpenCV even when the
        # ffmpeg command-line utilities are absent.
        try:
            import cv2

            capture = cv2.VideoCapture(str(path))
            try:
                count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            finally:
                capture.release()
            if count > 0:
                return count
        except (ImportError, OSError, ValueError):
            pass

        # PyAV decoding is slower but provides a dependable final fallback.
        try:
            import av

            with av.open(str(path)) as container:
                count = sum(1 for _ in container.decode(video=0))
            if count > 0:
                return count
        except (ImportError, OSError, ValueError):
            pass

        raise ValueError(f"Unable to determine a positive frame count for {path}")

    def _video_inputs(self, messages):
        videos = list(messages.get("videos") or [])
        if self.cine_input == "concatenated":
            concatenated = messages.get("concatenated_video")
            if not concatenated:
                raise ValueError("Cine sample has no concatenated_video")
            videos = [concatenated]
        if not videos:
            return [], []
        if not self.enable_video:
            raise RuntimeError(
                "Video input received while NPCFULLCARE_ENABLE_VIDEO is false"
            )
        if process_vision_info is None:
            raise RuntimeError(
                "qwen_vl_utils is required for native-video evaluation"
            )
        for video in videos:
            if not os.path.exists(video):
                raise FileNotFoundError(f"Video path not found: {video}")

        manifest = self._load_video_manifest(messages.get("video_manifest"))
        requested_counts = []
        if self.cine_input == "separate":
            slot_order = manifest.get("slot_order", [])
            if slot_order:
                for slot in slot_order:
                    sequence = manifest.get("sequence_manifests", {}).get(slot, {})
                    requested_counts.append(int(sequence.get("frame_count", 0)))
            elif len(videos) == 1 and int(manifest.get("frame_count", 0)) > 0:
                requested_counts = [int(manifest["frame_count"])]
        else:
            sequence_count = sum(
                int(sequence.get("frame_count", 0))
                for sequence in manifest.get("sequence_manifests", {}).values()
            )
            title_frames = round(
                float(manifest.get("title_seconds", 0))
                * int(manifest.get("fps", 1))
            ) * len(manifest.get("slot_order", []))
            requested_counts = [sequence_count + title_frames]
        if not requested_counts:
            requested_counts = [
                self._probe_video_frame_count(video) for video in videos
            ]
        if len(requested_counts) != len(videos) or any(
            count <= 0 for count in requested_counts
        ):
            raise ValueError(
                "Video manifest does not provide one positive frame count per video"
            )
        # qwen-vl-utils rounds nframes to FRAME_FACTOR=2. A manifest-derived
        # odd count equal to the physical frame count would therefore round one
        # frame past EOF (for example 139 -> 140). Cap every request to the
        # largest even count that the encoded stream can actually deliver.
        decodable_counts = [
            self._probe_video_frame_count(video) for video in videos
        ]
        requested_counts = [
            min(requested, decodable - decodable % 2)
            for requested, decodable in zip(
                requested_counts,
                decodable_counts,
            )
        ]
        if any(count < 2 for count in requested_counts):
            raise ValueError(
                "Video contains fewer than two decodable frames after capping"
            )
        return videos, requested_counts

    def _build_messages(self, messages: Dict[str, Any]):
        prompt = messages["prompt"]
        current_messages: List[Dict[str, Any]] = []

        if "system" in messages:
            current_messages.append(
                {"role": "system", "content": messages["system"]}
            )

        videos, requested_counts = self._video_inputs(messages)
        if videos:
            content = []
            for video, nframes in zip(videos, requested_counts):
                content.append(
                    {
                        "type": "video",
                        "video": video,
                        "nframes": nframes,
                        "min_pixels": self.video_min_pixels,
                        "max_pixels": self.video_max_pixels,
                    }
                )
            content.append({"type": "text", "text": prompt})
            current_messages.append({"role": "user", "content": content})
            return current_messages, [], videos, requested_counts

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

        return current_messages, images, [], []

    @staticmethod
    def _jsonable(value):
        try:
            json.dumps(value)
            return value
        except TypeError:
            if isinstance(value, dict):
                return {
                    str(key): Qwen35_36_MLLM._jsonable(item)
                    for key, item in value.items()
                }
            if isinstance(value, (list, tuple)):
                return [Qwen35_36_MLLM._jsonable(item) for item in value]
            return str(value)

    def _write_video_audit(self, payload):
        if self.video_audit_path is None:
            return
        self.video_audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self._audit_lock:
            with self.video_audit_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(self._jsonable(payload), ensure_ascii=False) + "\n"
                )

    def process_messages(self, messages):
        if isinstance(messages, list):
            current_messages = messages
            images = []
            videos = []
            requested_counts = []
        elif isinstance(messages, dict):
            if "messages" in messages and isinstance(messages["messages"], list):
                current_messages = messages["messages"]
                images = []
                videos = []
                requested_counts = []
            else:
                (
                    current_messages,
                    images,
                    videos,
                    requested_counts,
                ) = self._build_messages(messages)
        else:
            raise TypeError(f"Unsupported messages type: {type(messages)}")

        prompt = self.processor.apply_chat_template(
            current_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        llm_inputs = {"prompt": prompt}
        if videos:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                current_messages,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            multi_modal_data = {}
            if image_inputs:
                multi_modal_data["image"] = image_inputs
            if video_inputs:
                multi_modal_data["video"] = video_inputs
            if not multi_modal_data.get("video"):
                raise RuntimeError("qwen_vl_utils returned no decoded video input")
            llm_inputs["multi_modal_data"] = multi_modal_data
            if video_kwargs:
                llm_inputs["mm_processor_kwargs"] = video_kwargs
            delivered_videos = []
            for item in video_inputs:
                if (
                    isinstance(item, tuple)
                    and len(item) == 2
                    and isinstance(item[1], dict)
                ):
                    tensor, metadata = item
                else:
                    tensor, metadata = item, {}
                shape = list(getattr(tensor, "shape", []))
                delivered_videos.append(
                    {
                        "tensor_shape": shape,
                        "decoded_frame_count": shape[0] if shape else None,
                        "source_total_frames": metadata.get(
                            "total_num_frames"
                        ),
                        "source_fps": metadata.get("fps"),
                        "frame_indices": metadata.get("frames_indices", []),
                        "video_backend": metadata.get("video_backend", ""),
                    }
                )
            self._write_video_audit(
                {
                    "input_mode": self.cine_input,
                    "video_paths": videos,
                    "requested_frame_counts": requested_counts,
                    "video_count": len(video_inputs),
                    "delivered_videos": delivered_videos,
                    "video_kwargs": self._jsonable(video_kwargs),
                    "prompt_chars": len(prompt),
                }
            )
        elif images:
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
