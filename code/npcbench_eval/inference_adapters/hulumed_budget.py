"""Explicit MRI visual budget using the checkpoint's spatial feature merging."""
import os
import numpy as np
import torch
from npcbench_eval.inference_adapters.hulumed import HuluMed as BaseHuluMed


class HuluMed(BaseHuluMed):
    def __init__(self, model_path, args):
        super().__init__(model_path, args)
        self.merge_size = int(os.environ['HULUMED_MERGE_SIZE'])
        self.context_limit = int(os.environ['HULUMED_CONTEXT_LIMIT'])
        self.processor.tokenizer.model_max_length = self.context_limit
        self.visual_budget = int(os.environ['HULUMED_VISUAL_BUDGET'])

    def process_messages(self, messages):
        pathology = getattr(self, 'input_stage', None) == 'pathology'
        merge_size = 1 if pathology else self.merge_size
        self.processor.image_processor.max_tokens = 16384 if pathology else self.visual_budget
        conversation = self._build_messages(messages)
        for message in conversation:
            if isinstance(message['content'], list):
                for item in message['content']:
                    if item['type'] == 'image':
                        item['image'] = np.asarray(item['image'])
        inputs = self.processor(
            conversation=conversation, add_system_prompt=True, add_generation_prompt=True,
            images_kwargs={'merge_size': merge_size},
            common_kwargs={'return_tensors': 'pt'})
        n = int(inputs['input_ids'].shape[-1])
        if n + self.max_new_tokens > self.context_limit:
            raise ValueError(f'Input {n} + output {self.max_new_tokens} exceeds {self.context_limit}; no truncation applied')
        self.last_media_trace = {
            'grid_sizes': inputs.get('grid_sizes', torch.empty((0, 3))).tolist(),
            'merge_sizes': inputs.get('merge_sizes', torch.empty(0)).tolist(),
            'context_limit': self.context_limit,
            'visual_budget': self.processor.image_processor.max_tokens,
            'input_layout': 'HWC arrays',
        }
        inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
        return inputs
