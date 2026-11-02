import json
import os
import threading
from pathlib import Path

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


class Qwen2_5_VL:
    def __init__(self,model_path,args):
        super().__init__()
        self.enable_video = os.environ.get(
            "NPCFULLCARE_ENABLE_VIDEO", "false"
        ).lower() in {"1", "true", "yes", "y", "on"}
        self.video_min_pixels = int(
            os.environ.get("NPCFULLCARE_VIDEO_MIN_PIXELS", "12544")
        )
        self.video_max_pixels = int(
            os.environ.get("NPCFULLCARE_VIDEO_MAX_PIXELS", "200704")
        )
        audit_path = os.environ.get("NPCFULLCARE_VIDEO_AUDIT_JSONL", "").strip()
        self.video_audit_path = Path(audit_path) if audit_path else None
        self._audit_lock = threading.Lock()
        max_video_num = int(os.environ.get("NPCFULLCARE_MAX_VIDEO_NUM", "4"))
        llm_kwargs = {
            "model": model_path,
            "tensor_parallel_size": int(os.environ.get("tensor_parallel_size", 1)),
            "enforce_eager": True,
            "trust_remote_code": True,
            "limit_mm_per_prompt": {
                "image": args.max_image_num,
                "video": max_video_num,
            },
        }
        if os.environ.get("VLLM_MAX_MODEL_LEN"):
            llm_kwargs["max_model_len"] = int(os.environ["VLLM_MAX_MODEL_LEN"])
        if os.environ.get("master_port"):
            llm_kwargs["master_port"] = int(os.environ["master_port"])
        if os.environ.get("VLLM_GPU_MEMORY_UTILIZATION"):
            llm_kwargs["gpu_memory_utilization"] = float(
                os.environ["VLLM_GPU_MEMORY_UTILIZATION"]
            )
        if os.environ.get("VLLM_DISABLE_CUSTOM_ALL_REDUCE"):
            llm_kwargs["disable_custom_all_reduce"] = (
                os.environ["VLLM_DISABLE_CUSTOM_ALL_REDUCE"].lower()
                in {"1", "true", "yes", "y", "on"}
            )
        for key in ("max_num_seqs", "max_num_batched_tokens", "kv_cache_memory_bytes"):
            value = os.environ.get("VLLM_" + key.upper())
            if value is not None:
                llm_kwargs[key] = int(value)
        if os.environ.get("NPCBENCH_MM_PROCESSOR_KWARGS"):
            llm_kwargs["mm_processor_kwargs"] = json.loads(os.environ["NPCBENCH_MM_PROCESSOR_KWARGS"])
        self.llm = LLM(**llm_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_path)

        self.sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_tokens= args.max_new_tokens,
            stop_token_ids=[],
        )

    @staticmethod
    def _load_video_manifest(manifest_path):
        if not manifest_path:
            return {}
        path = Path(manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"Video manifest not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _probe_video_frame_count(video_path):
        try:
            import cv2

            capture = cv2.VideoCapture(str(video_path))
            try:
                count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            finally:
                capture.release()
            if count > 0:
                return count
        except (ImportError, OSError, ValueError):
            pass

        try:
            import av

            with av.open(str(video_path)) as container:
                count = sum(1 for _ in container.decode(video=0))
            if count > 0:
                return count
        except (ImportError, OSError, ValueError):
            pass

        raise ValueError(f"Unable to determine video frame count: {video_path}")

    def _video_frame_counts(self, videos, manifest_path):
        manifest = self._load_video_manifest(manifest_path)
        requested = []
        slot_order = manifest.get("slot_order", [])
        if slot_order:
            for slot in slot_order:
                sequence = manifest.get("sequence_manifests", {}).get(slot, {})
                requested.append(int(sequence.get("frame_count", 0)))
        elif len(videos) == 1 and int(manifest.get("frame_count", 0)) > 0:
            requested = [int(manifest["frame_count"])]

        decoded = [self._probe_video_frame_count(video) for video in videos]
        if len(requested) != len(videos) or any(count <= 0 for count in requested):
            requested = decoded

        # qwen-vl-utils requires FRAME_FACTOR=2. Never request a rounded frame
        # past EOF for odd-length clips.
        counts = [
            min(wanted, available - available % 2)
            for wanted, available in zip(requested, decoded)
        ]
        if any(count < 2 for count in counts):
            raise ValueError("A cine input has fewer than two decodable frames")
        return counts

    def _write_video_audit(self, payload):
        if self.video_audit_path is None:
            return
        self.video_audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self._audit_lock:
            with self.video_audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def process_messages(self,messages):
        current_messages = []
        source_videos = []

        if isinstance(messages, list):
            current_messages = messages
        elif "messages" in messages and isinstance(messages["messages"], list):
            messages = messages["messages"]
            for message in messages:
                role = message["role"]
                content = message["content"]
                current_messages.append({"role":role,"content":[{"type":"text","text":content}]})

        else:
            prompt = messages["prompt"]
            if "system" in messages:
                system_prompt = messages["system"]
                current_messages.append({"role":"system","content":system_prompt})
            videos = list(messages.get("videos") or [])
            if videos:
                source_videos = videos
                if not self.enable_video:
                    raise RuntimeError(
                        "Video input received while NPCFULLCARE_ENABLE_VIDEO is false"
                    )
                for video in videos:
                    if not os.path.exists(video):
                        raise FileNotFoundError(f"Video path not found: {video}")
                frame_counts = self._video_frame_counts(
                    videos, messages.get("video_manifest")
                )
                content = []
                for video, nframes in zip(videos, frame_counts):
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
            elif "image" in messages:
                image = messages["image"]
                current_messages.append({"role":"user","content":[{"type":"image","image":image},{"type":"text","text":prompt}]})
            elif "images" in messages:
                content = []
                for i,image in enumerate(messages["images"]):
                    content.append({"type":"text","text":f"<image_{i+1}>: "})
                    content.append({"type":"image","image":image})
                content.append({"type":"text","text":messages["prompt"]})
                current_messages.append({"role":"user","content":content})
            else:
                current_messages.append({"role":"user","content":[{"type":"text","text":prompt}]})

        messages = current_messages
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs
            delivered = []
            for item in video_inputs:
                tensor = item[0] if isinstance(item, tuple) else item
                shape = list(getattr(tensor, "shape", []))
                delivered.append(
                    {
                        "tensor_shape": shape,
                        "decoded_frame_count": shape[0] if shape else None,
                    }
                )
            self._write_video_audit(
                {
                    "video_paths": source_videos,
                    "video_count": len(video_inputs),
                    "delivered_videos": delivered,
                    "prompt_chars": len(prompt),
                }
            )

        llm_inputs = {
            "prompt": prompt
        }
        if mm_data:
            llm_inputs["multi_modal_data"] = mm_data
        return llm_inputs


    def generate_output(self,messages):
        llm_inputs = self.process_messages(messages)
        outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params)
        return outputs[0].outputs[0].text

    def generate_outputs(self,messages_list):
        llm_inputs_list = [self.process_messages(messages) for messages in messages_list]
        # from pdb import set_trace;set_trace()
        outputs = self.llm.generate(llm_inputs_list, sampling_params=self.sampling_params)
        res = []
        for output in outputs:
            generated_text = output.outputs[0].text
            res.append(generated_text)
        return res
