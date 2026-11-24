#!/usr/bin/env python3
"""用本地缓存权重核对 SmolVLA 的结构事实（只读，不下载）。

在 lerobot-smolvla-repro 环境运行：
  HF_HUB_OFFLINE=1 .../bin/python verify_model_facts.py

输出 JSON：VLM 各组件参数量、文本层数、视觉 token 数、expert 参数量、总参数量。
"""
import glob
import json
import os

import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

ROOT = "/home/lxy/lerobot-smolvla-repro/hf-cache/hub"
VLM_SNAP = glob.glob(os.path.join(ROOT, "models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/*"))[0]
CKPT_SNAP = glob.glob(os.path.join(ROOT, "models--lerobot--smolvla_libero/snapshots/*"))[0]

out = {"vlm_snapshot": VLM_SNAP, "ckpt_snapshot": CKPT_SNAP}

cfg = AutoConfig.from_pretrained(VLM_SNAP)
tc = cfg.text_config
out["text_config"] = {
    k: getattr(tc, k, None)
    for k in ["hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
              "num_key_value_heads", "head_dim", "vocab_size"]
}
vc = cfg.vision_config
out["vision_config"] = {
    k: getattr(vc, k, None) for k in ["hidden_size", "image_size", "patch_size", "num_hidden_layers"]
}

# 1) VLM 参数量拆分（不加权重的 config 实例即可精确计数）
from transformers import SmolVLMForConditionalGeneration  # noqa: E402

vlm = SmolVLMForConditionalGeneration(cfg)
def n(m):
    return sum(p.numel() for p in m.parameters())
out["vlm_params"] = {
    "total": n(vlm),
    "vision_model": n(vlm.model.vision_model),
    "connector": n(vlm.model.connector),
    "text_model": n(vlm.model.text_model),
    "text_embed_tokens": n(vlm.model.text_model.embed_tokens),
    "lm_head": n(vlm.lm_head) if hasattr(vlm, "lm_head") else None,
    "one_text_layer": n(vlm.model.text_model.layers[0]),
}
out["vlm_params"]["first16_text_layers_est"] = out["vlm_params"]["one_text_layer"] * 16

# 2) 视觉 token 数：按 prepare_images 的格式构造 (1,3,512,512) 输入，走真实 vision tower + connector
pixel_values = (torch.rand(1, 3, 512, 512) * 2 - 1).to(torch.bfloat16)
out["pixel_values_shape"] = list(pixel_values.shape)
vlm = vlm.to(torch.bfloat16).eval()
with torch.no_grad():
    hidden = vlm.model.vision_model(pixel_values=pixel_values).last_hidden_state
    out["vision_patch_tokens"] = list(hidden.shape)
    seq = vlm.model.connector(hidden)
    out["visual_tokens_per_image"] = list(seq.shape)

# 3) 完整 SmolVLA 检查点参数量与拆分
from lerobot.policies.smolvla import SmolVLAPolicy  # noqa: E402

out["smolvla"] = {}

pol = SmolVLAPolicy.from_pretrained(CKPT_SNAP, local_files_only=True)
m = pol.model
groups = {"state_proj": m.state_proj, "action_in_proj": m.action_in_proj, "action_out_proj": m.action_out_proj,
          "action_time_mlp_in": m.action_time_mlp_in, "action_time_mlp_out": m.action_time_mlp_out,
          "lm_expert": m.vlm_with_expert.lm_expert, "vlm": m.vlm_with_expert.vlm}
out["smolvla_params"] = {k: sum(p.numel() for p in v.parameters()) for k, v in groups.items()}
out["smolvla_params"]["total"] = sum(p.numel() for p in m.parameters())
out["smolvla_params"]["trainable_requires_grad"] = sum(p.numel() for p in m.parameters() if p.requires_grad)
out["smolvla"]["num_vlm_layers_used"] = m.vlm_with_expert.num_vlm_layers
out["smolvla"]["num_expert_layers"] = m.vlm_with_expert.num_expert_layers
out["smolvla"]["expert_hidden_size"] = m.vlm_with_expert.expert_hidden_size
out["smolvla"]["vlm_hidden_size"] = m.vlm_with_expert.config.text_config.hidden_size

print(json.dumps(out, indent=2, ensure_ascii=False))
