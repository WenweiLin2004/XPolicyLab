from __future__ import annotations
from contextlib import nullcontext
from torch import nn
import torch
from PIL import Image

def _resize_images(images, target_size):
    if isinstance(images, Image.Image):
        return images.resize(target_size, resample=Image.Resampling.BICUBIC)
    if isinstance(images, list):
        return [_resize_images(image, target_size) for image in images]
    if isinstance(images, tuple):
        return tuple(_resize_images(image, target_size) for image in images)
    raise TypeError('Unsupported image container for resize')


class QwenVLBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        from transformers import AutoConfig
        hf_config=AutoConfig.from_pretrained(config.framework.qwenvl.base_vlm)
        family=hf_config.model_type
        if family=="qwen2_5_vl":
            from .qwen25_vl import Qwen25VLInterface as Interface
        elif family=="qwen3_vl":
            from .qwen3_vl import Qwen3VLInterface as Interface
        else:
            raise ValueError(f"Unsupported VLM family {family!r}; use Qwen2.5-VL or Qwen3-VL")
        self.interface=Interface(config)
        # FSDP master weights must have one dtype; AMP handles BF16 compute.
        # Never cast the whole framework: the frozen action codec stays FP32.
        self.interface.model.float()
        self.family=family
        self.hf_revision=getattr(hf_config,"_commit_hash",None)
        text=getattr(self.interface.model.config,"text_config",self.interface.model.config)
        self.hidden_size=int(text.hidden_size)
        self.num_layers=int(text.num_hidden_layers)
        self.freeze=bool(config.framework.qwenvl.get("freeze",False))
        self.interface.model.requires_grad_(not self.freeze)
        # We use feature states, not the unused vocabulary output projection.
        if hasattr(self.interface.model,"lm_head"):
            self.interface.model.lm_head.requires_grad_(False)
        if not self.freeze and config.framework.qwenvl.get("gradient_checkpointing",True):
            self.interface.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        self.interface.model.config.use_cache=False
        if config.framework.qwenvl.get("max_pixels") is not None:
            raise ValueError("qwenvl.max_pixels is removed; set data.obs_image_size [width, height]")
        data_config=config.get('data', {})
        raw_image_size=data_config.get('obs_image_size', None)
        if raw_image_size is None:
            self.obs_image_size=None
        else:
            if len(raw_image_size) != 2: raise ValueError('data.obs_image_size must contain [width, height]')
            self.obs_image_size=(int(raw_image_size[0]), int(raw_image_size[1]))
            if min(self.obs_image_size) <= 0: raise ValueError('data.obs_image_size values must be positive')
            # Spark resizes to obs_image_size and has no separate pixel cap.
            # Lock the processor to that area so it does not rescale again.
            pixels=int(self.obs_image_size[0] * self.obs_image_size[1])
            ip=self.interface.processor.image_processor
            if hasattr(ip,"max_pixels"): ip.max_pixels=pixels
            if hasattr(ip,"min_pixels"): ip.min_pixels=pixels
            if hasattr(ip,"size") and isinstance(ip.size,dict):
                ip.size["longest_edge"]=pixels
                if "shortest_edge" in ip.size:
                    ip.size["shortest_edge"]=pixels

    @property
    def model(self):return self.interface.model

    def train(self,mode=True):
        super().train(mode)
        if self.freeze:self.interface.model.eval()
        return self

    def forward(self,examples,num_layers):
        images=[e["image"] for e in examples]
        if self.obs_image_size is not None:
            images=[_resize_images(image, self.obs_image_size) for image in images]
        texts=[e["lang"] for e in examples]
        inputs=self.interface.build_qwenvl_inputs(images=images,instructions=texts)
        # HF's multimodal feature model yields states without allocating LM logits.
        ctx=torch.no_grad() if self.freeze else nullcontext()
        with ctx:
            outputs=self.interface.model.model(**inputs,output_hidden_states=True,
                return_dict=True,use_cache=False)
        states=list(outputs.hidden_states[-num_layers:])
        if len(states)!=num_layers:raise ValueError("Requested more feature layers than VLM provides")
        mask=inputs["attention_mask"].bool()
        return states,mask


__all__ = ["QwenVLBackbone"]
