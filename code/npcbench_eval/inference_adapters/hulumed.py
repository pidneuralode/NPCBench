import os
from typing import Any, Dict, List

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module


class HuluMed:
    def __init__(self, model_path: str, args: Any):
        super().__init__()
        attn_impl = os.environ.get("HULUMED_ATTENTION_IMPL", "sdpa")
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=os.environ.get("HULUMED_DEVICE_MAP", "auto"),
            attn_implementation=attn_impl,
            trust_remote_code=True,
        ).eval()
        self.processor = self._load_processor(model_path)

        self.temperature = float(args.temperature)
        self.top_p = float(args.top_p)
        self.repetition_penalty = float(args.repetition_penalty)
        self.max_new_tokens = int(args.max_new_tokens)
        self.default_system_prompt = getattr(args, "default_system_prompt", "") or ""
        print("using HuluMed HF wrapper ...")

    @staticmethod
    def _patch_processor_class(processor_cls):
        def _merge_kwargs_compat(
            self,
            ModelProcessorKwargs,
            tokenizer_init_kwargs=None,
            **kwargs,
        ):
            tokenizer_init_kwargs = tokenizer_init_kwargs or {}
            output_kwargs = {
                "text_kwargs": {},
                "images_kwargs": {},
                "audio_kwargs": {},
                "videos_kwargs": {},
                "chat_template_kwargs": {},
                "common_kwargs": {},
            }
            default_kwargs = {k: v.copy() for k, v in output_kwargs.items()}
            annotations = getattr(ModelProcessorKwargs, "__annotations__", {})
            defaults = getattr(ModelProcessorKwargs, "_defaults", {})
            used_keys = set()

            for modality in default_kwargs:
                default_kwargs[modality] = defaults.get(modality, {}).copy()
                modality_ann = annotations.get(modality)
                ann_keys = (
                    getattr(modality_ann, "__annotations__", {})
                    if modality_ann
                    else {}
                )
                for modality_key in ann_keys.keys():
                    if modality_key in tokenizer_init_kwargs:
                        value = (
                            getattr(self.tokenizer, modality_key)
                            if hasattr(self.tokenizer, modality_key)
                            else tokenizer_init_kwargs[modality_key]
                        )
                        default_kwargs[modality][modality_key] = value

            output_kwargs.update(default_kwargs)
            non_modality_kwargs = set(kwargs) - set(output_kwargs)

            for modality in output_kwargs:
                modality_ann = annotations.get(modality)
                ann_keys = (
                    getattr(modality_ann, "__annotations__", {})
                    if modality_ann
                    else {}
                )
                for modality_key in ann_keys.keys():
                    if modality in kwargs and isinstance(kwargs[modality], dict):
                        kwarg_value = kwargs[modality].pop(
                            modality_key, "__empty__"
                        )
                        if (
                            kwarg_value != "__empty__"
                            and modality_key in non_modality_kwargs
                        ):
                            raise ValueError(
                                f"Keyword argument {modality_key} was passed twice."
                            )
                    elif modality_key in kwargs:
                        kwarg_value = kwargs.get(modality_key, "__empty__")
                    else:
                        kwarg_value = "__empty__"
                    if kwarg_value != "__empty__":
                        output_kwargs[modality][modality_key] = kwarg_value
                        used_keys.add(modality_key)

            if any(key in default_kwargs for key in kwargs):
                for modality, subdict in kwargs.items():
                    if modality in default_kwargs and isinstance(subdict, dict):
                        for subkey, subvalue in subdict.items():
                            if subkey not in used_keys:
                                output_kwargs[modality][subkey] = subvalue
                                used_keys.add(subkey)
            else:
                for key in kwargs:
                    if key not in used_keys:
                        output_kwargs["common_kwargs"][key] = kwargs[key]

            for modality in output_kwargs:
                output_kwargs[modality].update(output_kwargs["common_kwargs"])
            return output_kwargs

        processor_cls._merge_kwargs = _merge_kwargs_compat
        return processor_cls

    def _load_processor(self, model_path: str):
        processor_cls = get_class_from_dynamic_module(
            "processing_hulumed.HulumedProcessor",
            model_path,
            trust_remote_code=True,
        )
        image_processor_cls = get_class_from_dynamic_module(
            "image_processing_hulumed.HulumedImageProcessor",
            model_path,
            trust_remote_code=True,
        )
        processor_cls = self._patch_processor_class(processor_cls)
        image_processor = image_processor_cls.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="right",
            fix_mistral_regex=True,
        )
        return processor_cls(image_processor=image_processor, tokenizer=tokenizer)

    @staticmethod
    def _load_image(image):
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image path not found: {image}")
            return Image.open(image).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image)}")

    def _build_messages(self, messages: Dict[str, Any]) -> List[Dict[str, Any]]:
        new_messages: List[Dict[str, Any]] = []
        if "system" in messages:
            new_messages.append({"role": "system", "content": messages["system"]})
        elif self.default_system_prompt:
            new_messages.append(
                {"role": "system", "content": self.default_system_prompt}
            )

        prompt = messages["prompt"]
        images = []
        if "image" in messages:
            images = [self._load_image(messages["image"])]
        elif "images" in messages:
            images = [self._load_image(image) for image in messages["images"]]

        if images:
            content = [{"type": "image", "image": image} for image in images]
            content.append({"type": "text", "text": prompt})
        else:
            content = [{"type": "text", "text": prompt}]
        new_messages.append({"role": "user", "content": content})
        return new_messages

    def process_messages(self, messages):
        if isinstance(messages, dict):
            if "messages" in messages and isinstance(messages["messages"], list):
                conversation = messages["messages"]
            else:
                conversation = self._build_messages(messages)
        elif isinstance(messages, list):
            conversation = messages
        else:
            raise TypeError(f"Unsupported messages type: {type(messages)}")

        inputs = self.processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        inputs = {
            k: v.cuda() if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        return inputs

    def generate_output(self, messages):
        inputs = self.process_messages(messages)
        do_sample = self.temperature > 0
        generate_kwargs = {
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generate_kwargs["temperature"] = self.temperature
            generate_kwargs["top_p"] = self.top_p

        with torch.inference_mode():
            generated_ids = self.llm.generate(**inputs, **generate_kwargs)

        input_len = inputs["input_ids"].shape[-1]
        if generated_ids.shape[-1] > input_len:
            new_tokens = generated_ids[:, input_len:]
        else:
            new_tokens = generated_ids
        output_text = self.processor.batch_decode(
            new_tokens, skip_special_tokens=True, use_think=False
        )
        return output_text[0].strip()

    def generate_outputs(self, messages_list):
        res = []
        for messages in tqdm(messages_list, desc="Generate outputs"):
            res.append(self.generate_output(messages))
        return res

    def async_generate_outputs(self, messages_list, max_concurrency=32):
        raise RuntimeError(
            "HF does not support async generation. Please set USE_ASYNC=False."
        )
