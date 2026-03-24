import os
from types import MethodType
from typing import List, Optional

import torch
import torch.nn as nn
from safetensors import safe_open

from .wan_video_dit import CrossAttention, WanModel, rearrange


class AudioProjModel(nn.Module):
    # 将外部音频特征映射到 Wan DiT 可用的条件维度
    def __init__(self, audio_in_dim: int = 768, cross_attention_dim: int = 2048):
        super().__init__()
        self.proj = nn.Linear(audio_in_dim, cross_attention_dim, bias=False)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, audio_embeds: torch.Tensor) -> torch.Tensor:
        # 输入: [B, L, Cin] -> 输出: [B, L, Cout]
        return self.norm(self.proj(audio_embeds))


class FantasyTalkingAudioConditionModel(nn.Module):
    """
    External audio cross-attention adapter for WanModel.
    This module does NOT modify wan_video_dit.py, and is attached dynamically.

    用法:
    1) 初始化: adapter = FantasyTalkingAudioConditionModel(pipe.dit, 768, 2048)
    2) 推理时给 pipe.dit.forward 透传 audio_embedding / audio_scale
    3) 如需移除挂载: adapter.remove()
    """

    def __init__(
        self,
        wan_dit: WanModel,
        audio_in_dim: int,
        audio_proj_dim: int,
        inject_layers: Optional[List[int]] = None,
        num_heads: Optional[int] = None,
    ):
        super().__init__()
        self.wan_dit = wan_dit
        self.audio_in_dim = audio_in_dim
        self.audio_proj_dim = audio_proj_dim
        self.inject_layers = inject_layers or [0, 4, 8, 12, 16, 20, 24, 27]
        self.num_heads = num_heads or wan_dit.blocks[0].num_heads

        self.proj_model = AudioProjModel(audio_in_dim=audio_in_dim, cross_attention_dim=audio_proj_dim)
        # 每个注入层前都做一次 pre-norm，稳定跨模态残差注入
        self.audio_pre_norm = nn.ModuleList([
            nn.LayerNorm(wan_dit.dim, elementwise_affine=False, eps=1e-6)
            for _ in self.inject_layers
        ])
        # 注入器本质是 CrossAttention(Q=视频token, K/V=音频token)
        self.audio_injector = nn.ModuleList([
            CrossAttention(dim=wan_dit.dim, num_heads=self.num_heads)
            for _ in self.inject_layers
        ])
        # 若音频投影维度与 DiT hidden dim 不同，再做一次线性映射
        self.audio_proj_to_dit = nn.Identity() if audio_proj_dim == wan_dit.dim else nn.Linear(audio_proj_dim, wan_dit.dim, bias=False)

        self._runtime_audio: Optional[torch.Tensor] = None
        self._runtime_audio_scale: float = 1.0
        self._runtime_frames: Optional[int] = None

        self._block_id_map = {layer_id: idx for idx, layer_id in enumerate(self.inject_layers)}
        self._hooks = []
        self._orig_forward = wan_dit.forward
        self._install_hooks()
        self._patch_wan_forward()

    def _install_hooks(self):
        # 从 patch_embedding 输出中获取 latent 帧数 F，用于后续逐帧对齐
        self._hooks.append(self.wan_dit.patch_embedding.register_forward_hook(self._capture_frames_hook))
        for layer_id in self.inject_layers:
            if layer_id < 0 or layer_id >= len(self.wan_dit.blocks):
                continue
            # 在目标 transformer block 后注入音频 cross-attention 残差
            hook = self.wan_dit.blocks[layer_id].register_forward_hook(self._make_block_hook(layer_id))
            self._hooks.append(hook)

    def _patch_wan_forward(self):
        def wrapped_forward(model_self, *args, **kwargs):
            # 额外消费两个外部参数，不影响原有 forward 参数签名
            audio_embedding = kwargs.pop("audio_embedding", None)
            audio_scale = float(kwargs.pop("audio_scale", 1.0))
            # 先投影到统一维度，后续 block hook 直接使用
            self._runtime_audio = self.get_proj_fea(audio_embedding) if audio_embedding is not None else None
            self._runtime_audio_scale = audio_scale
            self._runtime_frames = None
            try:
                return self._orig_forward(*args, **kwargs)
            finally:
                # 清理 runtime 状态，避免跨 batch 污染
                self._runtime_audio = None
                self._runtime_frames = None
                self._runtime_audio_scale = 1.0

        self.wan_dit.forward = MethodType(wrapped_forward, self.wan_dit)

    def _capture_frames_hook(self, module, inputs, output):
        # output: [B, C, F, H, W]
        self._runtime_frames = int(output.shape[2])

    def _make_block_hook(self, block_idx: int):
        def block_hook(module, inputs, output):
            if self._runtime_audio is None:
                return output

            injector_idx = self._block_id_map.get(block_idx)
            if injector_idx is None:
                return output

            hidden_states = output
            audio = self.audio_proj_to_dit(self._runtime_audio).to(dtype=hidden_states.dtype, device=hidden_states.device)

            if audio.dim() == 3:
                audio = audio.unsqueeze(2)
            if audio.dim() != 4:
                raise ValueError(f"audio_embedding must be [B,T,C] or [B,T,N,C], got shape={tuple(audio.shape)}")

            # 严格逐帧对齐: 每个视频 latent 帧只与对应音频帧做 cross-attention
            num_frames = self._runtime_frames or audio.shape[1]
            if audio.shape[1] != num_frames:
                raise ValueError(
                    f"audio/video frame mismatch in cross-attention: video={num_frames}, audio={audio.shape[1]}"
                )

            seq_len = hidden_states.shape[1]
            if seq_len % num_frames != 0:
                raise ValueError(
                    f"hidden token length {seq_len} is not divisible by frame count {num_frames}"
                )

            # [B, T*N, C] -> [B*T, N, C]，把每一帧拆开单独做 cross-attention
            attn_hidden_states = rearrange(hidden_states, "b (t n) c -> (b t) n c", t=num_frames)
            attn_hidden_states = self.audio_pre_norm[injector_idx](attn_hidden_states)
            attn_audio = rearrange(audio, "b t n c -> (b t) n c", t=num_frames)

            residual = self.audio_injector[injector_idx](attn_hidden_states, attn_audio)
            # 回拼到原序列并做残差融合
            residual = rearrange(residual, "(b t) n c -> b (t n) c", t=num_frames)
            return hidden_states + residual * self._runtime_audio_scale

        return block_hook

    def remove(self):
        # 解除所有 hook，并恢复原始 forward
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self.wan_dit.forward = self._orig_forward

    def get_proj_fea(self, audio_fea: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        return self.proj_model(audio_fea) if audio_fea is not None else None

    def load_audio_processor(self, ip_ckpt: str, wan_dit: Optional[WanModel] = None):
        # 支持 safetensors 和普通 torch checkpoint
        if os.path.splitext(ip_ckpt)[-1] == ".safetensors":
            state_dict = {"proj_model": {}, "audio_injector": {}, "audio_pre_norm": {}, "audio_proj_to_dit": {}}
            with safe_open(ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("proj_model."):
                        state_dict["proj_model"][key.replace("proj_model.", "")] = f.get_tensor(key)
                    elif key.startswith("audio_injector."):
                        state_dict["audio_injector"][key.replace("audio_injector.", "")] = f.get_tensor(key)
                    elif key.startswith("audio_pre_norm."):
                        state_dict["audio_pre_norm"][key.replace("audio_pre_norm.", "")] = f.get_tensor(key)
                    elif key.startswith("audio_proj_to_dit."):
                        state_dict["audio_proj_to_dit"][key.replace("audio_proj_to_dit.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(ip_ckpt, map_location="cpu")

        if "proj_model" in state_dict:
            self.proj_model.load_state_dict(state_dict["proj_model"], strict=True)
        if "audio_injector" in state_dict:
            self.audio_injector.load_state_dict(state_dict["audio_injector"], strict=False)
        if "audio_pre_norm" in state_dict:
            self.audio_pre_norm.load_state_dict(state_dict["audio_pre_norm"], strict=False)
        if "audio_proj_to_dit" in state_dict and hasattr(self.audio_proj_to_dit, "load_state_dict"):
            self.audio_proj_to_dit.load_state_dict(state_dict["audio_proj_to_dit"], strict=False)

        if wan_dit is not None and "audio_processor" in state_dict:
            wan_dit.load_state_dict(state_dict["audio_processor"], strict=False)


__all__ = ["FantasyTalkingAudioConditionModel", "AudioProjModel"]
