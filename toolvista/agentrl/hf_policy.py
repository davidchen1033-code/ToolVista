"""Local, deterministic checkpoint evaluation without running the GPU trainer."""
from __future__ import annotations

import asyncio
from pathlib import Path

from .prompts import SYSTEM_PROMPT, observation_message
from .rollout import ContextLimit, Generation


class TransformersPolicy:
    def __init__(self, checkpoint: Path, *, max_action_tokens: int = 256, max_context: int = 16384):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(str(checkpoint), trust_remote_code=False)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(checkpoint), dtype="auto", device_map="auto", attn_implementation="sdpa")
        self.model.eval()
        self.max_action_tokens = max_action_tokens
        self.max_context = max_context
        self.messages = []

    def reset(self):
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _generate(self, observation):
        import torch
        from qwen_vl_utils import process_vision_info
        message = observation_message(observation)
        message["content"][0].update(image=observation["image_path"])
        self.messages.append(message)
        prompt = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        images, _ = process_vision_info(self.messages, image_patch_size=16)
        inputs = self.processor(text=[prompt], images=images, return_tensors="pt").to(self.model.device)
        length = inputs["input_ids"].shape[1]
        if length + self.max_action_tokens > self.max_context:
            raise ContextLimit("cold-start context budget exceeded")
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_action_tokens, do_sample=False)
        ids = output[0, length:].tolist()
        text = self.processor.tokenizer.decode(ids, skip_special_tokens=True)
        self.messages.append({"role": "assistant", "content": text})
        return Generation(text, ids)

    async def generate(self, observation, history):
        return await asyncio.to_thread(self._generate, observation)
