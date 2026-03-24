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
    2) 推理时给 pipe.dit.forward 透传:
       - audio_embedding (wav2vec, 帧对齐分支)
       - global_audio_embedding 或 emotion_audio_embedding (emotion2vec, 全局分支)
       - audio_scale / audio_frame_scale / audio_global_scale
    3) 如需移除挂载: adapter.remove()
    """

    def __init__(
        self,
        wan_dit: WanModel,
        audio_in_dim: int,
        audio_proj_dim: int,
        global_audio_in_dim: Optional[int] = None,
        inject_layers: Optional[List[int]] = None,
        num_heads: Optional[int] = None,
        enable_frame_aligned_attn: bool = True,
        enable_global_attn: bool = True,
    ):
        super().__init__()
        self.wan_dit = wan_dit
        # audio_in_dim: 帧对齐分支（默认用于 wav2vec）
        # global_audio_in_dim: 全局分支（默认用于 emotion2vec），未提供时回退到 audio_in_dim
        self.audio_in_dim = audio_in_dim
        self.global_audio_in_dim = global_audio_in_dim if global_audio_in_dim is not None else audio_in_dim
        self.audio_proj_dim = audio_proj_dim
        self.inject_layers = inject_layers or [0, 4, 8, 12, 16, 20, 24, 27]
        self.num_heads = num_heads or wan_dit.blocks[0].num_heads
        self.enable_frame_aligned_attn = enable_frame_aligned_attn
        self.enable_global_attn = enable_global_attn

        # 两个分支使用不同音频编码器特征时，各自独立投影:
        # - 帧对齐分支: wav2vec 特征
        # - 全局分支: emotion2vec 特征
        self.proj_model_frame = AudioProjModel(audio_in_dim=self.audio_in_dim, cross_attention_dim=audio_proj_dim)
        self.proj_model_global = AudioProjModel(audio_in_dim=self.global_audio_in_dim, cross_attention_dim=audio_proj_dim)
        # -------- 分支1：帧对齐 cross-attention（逐帧）--------
        # 每个注入层前都做一次 pre-norm，稳定跨模态残差注入
        self.audio_pre_norm_frame = nn.ModuleList([
            nn.LayerNorm(wan_dit.dim, elementwise_affine=False, eps=1e-6)
            for _ in self.inject_layers
        ])
        # 注入器本质是 CrossAttention(Q=视频token, K/V=对应帧音频token)
        self.audio_injector_frame = nn.ModuleList([
            CrossAttention(dim=wan_dit.dim, num_heads=self.num_heads)
            for _ in self.inject_layers
        ])
        # -------- 分支2：全局 cross-attention（整段）--------
        # Q 仍然是全部视频 token，K/V 是整个音频序列（跨所有帧）
        self.audio_pre_norm_global = nn.ModuleList([
            nn.LayerNorm(wan_dit.dim, elementwise_affine=False, eps=1e-6)
            for _ in self.inject_layers
        ])
        self.audio_injector_global = nn.ModuleList([
            CrossAttention(dim=wan_dit.dim, num_heads=self.num_heads)
            for _ in self.inject_layers
        ])
        # 若音频投影维度与 DiT hidden dim 不同，再做一次线性映射
        self.audio_proj_to_dit = nn.Identity() if audio_proj_dim == wan_dit.dim else nn.Linear(audio_proj_dim, wan_dit.dim, bias=False)

        self._runtime_audio_frame: Optional[torch.Tensor] = None
        self._runtime_audio_global: Optional[torch.Tensor] = None
        self._runtime_audio_scale: float = 1.0
        self._runtime_audio_frame_scale: float = 1.0
        self._runtime_audio_global_scale: float = 1.0
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
            # audio_embedding: 帧对齐分支输入（wav2vec）
            # global_audio_embedding / emotion_audio_embedding: 全局分支输入（emotion2vec）
            audio_embedding = kwargs.pop("audio_embedding", None)
            global_audio_embedding = kwargs.pop("global_audio_embedding", None)
            if global_audio_embedding is None:
                global_audio_embedding = kwargs.pop("emotion_audio_embedding", None)
            # audio_scale: 总体缩放；frame/global_scale: 分支缩放
            audio_scale = float(kwargs.pop("audio_scale", 1.0))
            frame_scale = float(kwargs.pop("audio_frame_scale", 1.0))
            global_scale = float(kwargs.pop("audio_global_scale", 1.0))
            # 先分别投影到统一维度，后续 block hook 直接使用
            self._runtime_audio_frame = self.get_proj_fea(audio_embedding, branch="frame") if audio_embedding is not None else None
            self._runtime_audio_global = self.get_proj_fea(global_audio_embedding, branch="global") if global_audio_embedding is not None else None
            self._runtime_audio_scale = audio_scale
            self._runtime_audio_frame_scale = frame_scale
            self._runtime_audio_global_scale = global_scale
            self._runtime_frames = None
            try:
                return self._orig_forward(*args, **kwargs)
            finally:
                # 清理 runtime 状态，避免跨 batch 污染
                self._runtime_audio_frame = None
                self._runtime_audio_global = None
                self._runtime_frames = None
                self._runtime_audio_scale = 1.0
                self._runtime_audio_frame_scale = 1.0
                self._runtime_audio_global_scale = 1.0

        self.wan_dit.forward = MethodType(wrapped_forward, self.wan_dit)

    def _capture_frames_hook(self, module, inputs, output):
        # output: [B, C, F, H, W]
        self._runtime_frames = int(output.shape[2])

    def _make_block_hook(self, block_idx: int):
        def block_hook(module, inputs, output):
            if self._runtime_audio_frame is None and self._runtime_audio_global is None:
                return output

            injector_idx = self._block_id_map.get(block_idx)
            if injector_idx is None:
                return output

            hidden_states = output
            residual_sum = 0

            # ===== 1) 帧对齐分支（wav2vec）=====
            if self.enable_frame_aligned_attn and self._runtime_audio_frame is not None:
                frame_audio = self.audio_proj_to_dit(self._runtime_audio_frame).to(dtype=hidden_states.dtype, device=hidden_states.device)
                if frame_audio.dim() == 3:
                    frame_audio = frame_audio.unsqueeze(2)
                if frame_audio.dim() != 4:
                    raise ValueError(f"frame audio_embedding must be [B,T,C] or [B,T,N,C], got shape={tuple(frame_audio.shape)}")

                # 严格逐帧对齐: 每个视频 latent 帧只与对应音频帧做 cross-attention
                num_frames = self._runtime_frames or frame_audio.shape[1]
                if frame_audio.shape[1] != num_frames:
                    raise ValueError(
                        f"frame audio/video mismatch: video={num_frames}, frame_audio={frame_audio.shape[1]}"
                    )

                seq_len = hidden_states.shape[1]
                if seq_len % num_frames != 0:
                    raise ValueError(
                        f"hidden token length {seq_len} is not divisible by frame count {num_frames}"
                    )

                # [B, T*N, C] -> [B*T, N, C]，把每一帧拆开单独做 cross-attention
                frame_hidden = rearrange(hidden_states, "b (t n) c -> (b t) n c", t=num_frames)
                frame_hidden = self.audio_pre_norm_frame[injector_idx](frame_hidden)
                frame_audio_tokens = rearrange(frame_audio, "b t n c -> (b t) n c", t=num_frames)
                frame_residual = self.audio_injector_frame[injector_idx](frame_hidden, frame_audio_tokens)
                # 回拼到原序列并做残差融合
                frame_residual = rearrange(frame_residual, "(b t) n c -> b (t n) c", t=num_frames)
                residual_sum = residual_sum + frame_residual * self._runtime_audio_frame_scale

            # ===== 2) 全局分支（emotion2vec）=====
            if self.enable_global_attn and self._runtime_audio_global is not None:
                # 全局分支：视频整段 token 与整段音频 token 做 cross-attention
                global_hidden = self.audio_pre_norm_global[injector_idx](hidden_states)
                global_audio = self.audio_proj_to_dit(self._runtime_audio_global).to(dtype=hidden_states.dtype, device=hidden_states.device)
                if global_audio.dim() == 4:
                    global_audio_tokens = rearrange(global_audio, "b t n c -> b (t n) c")
                elif global_audio.dim() == 3:
                    global_audio_tokens = global_audio
                else:
                    raise ValueError(f"global audio_embedding must be [B,L,C] or [B,T,N,C], got shape={tuple(global_audio.shape)}")
                global_residual = self.audio_injector_global[injector_idx](global_hidden, global_audio_tokens)
                residual_sum = residual_sum + global_residual * self._runtime_audio_global_scale

            return hidden_states + residual_sum * self._runtime_audio_scale

        return block_hook

    def remove(self):
        # 解除所有 hook，并恢复原始 forward
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self.wan_dit.forward = self._orig_forward

    def get_proj_fea(self, audio_fea: Optional[torch.Tensor] = None, branch: str = "frame") -> Optional[torch.Tensor]:
        if audio_fea is None:
            return None
        if branch == "frame":
            return self.proj_model_frame(audio_fea)
        if branch == "global":
            return self.proj_model_global(audio_fea)
        raise ValueError(f"Unsupported branch: {branch}")

    def load_audio_processor(self, ip_ckpt: str, wan_dit: Optional[WanModel] = None):
        # 支持 safetensors 和普通 torch checkpoint
        if os.path.splitext(ip_ckpt)[-1] == ".safetensors":
            state_dict = {
                "proj_model": {},
                "proj_model_frame": {},
                "proj_model_global": {},
                "audio_injector_frame": {},
                "audio_pre_norm_frame": {},
                "audio_injector_global": {},
                "audio_pre_norm_global": {},
                "audio_proj_to_dit": {}
            }
            with safe_open(ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("proj_model."):
                        state_dict["proj_model"][key.replace("proj_model.", "")] = f.get_tensor(key)
                    elif key.startswith("proj_model_frame."):
                        state_dict["proj_model_frame"][key.replace("proj_model_frame.", "")] = f.get_tensor(key)
                    elif key.startswith("proj_model_global."):
                        state_dict["proj_model_global"][key.replace("proj_model_global.", "")] = f.get_tensor(key)
                    elif key.startswith("audio_injector_frame."):
                        state_dict["audio_injector_frame"][key.replace("audio_injector_frame.", "")] = f.get_tensor(key)
                    elif key.startswith("audio_pre_norm_frame."):
                        state_dict["audio_pre_norm_frame"][key.replace("audio_pre_norm_frame.", "")] = f.get_tensor(key)
                    elif key.startswith("audio_injector_global."):
                        state_dict["audio_injector_global"][key.replace("audio_injector_global.", "")] = f.get_tensor(key)
                    elif key.startswith("audio_pre_norm_global."):
                        state_dict["audio_pre_norm_global"][key.replace("audio_pre_norm_global.", "")] = f.get_tensor(key)
                    # 兼容旧权重命名（只有单分支时）
                    elif key.startswith("audio_injector."):
                        state_dict["audio_injector_frame"][key.replace("audio_injector.", "")] = f.get_tensor(key)
                    elif key.startswith("audio_pre_norm."):
                        state_dict["audio_pre_norm_frame"][key.replace("audio_pre_norm.", "")] = f.get_tensor(key)
                    elif key.startswith("audio_proj_to_dit."):
                        state_dict["audio_proj_to_dit"][key.replace("audio_proj_to_dit.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(ip_ckpt, map_location="cpu")

        if "proj_model_frame" in state_dict and len(state_dict["proj_model_frame"]) > 0:
            self.proj_model_frame.load_state_dict(state_dict["proj_model_frame"], strict=True)
        if "proj_model_global" in state_dict and len(state_dict["proj_model_global"]) > 0:
            self.proj_model_global.load_state_dict(state_dict["proj_model_global"], strict=True)
        # 兼容老字段：只有一个 proj_model 时，默认加载到 frame 分支
        if "proj_model" in state_dict and len(state_dict["proj_model"]) > 0:
            self.proj_model_frame.load_state_dict(state_dict["proj_model"], strict=True)
        if "audio_injector_frame" in state_dict:
            self.audio_injector_frame.load_state_dict(state_dict["audio_injector_frame"], strict=False)
        if "audio_pre_norm_frame" in state_dict:
            self.audio_pre_norm_frame.load_state_dict(state_dict["audio_pre_norm_frame"], strict=False)
        if "audio_injector_global" in state_dict:
            self.audio_injector_global.load_state_dict(state_dict["audio_injector_global"], strict=False)
        if "audio_pre_norm_global" in state_dict:
            self.audio_pre_norm_global.load_state_dict(state_dict["audio_pre_norm_global"], strict=False)
        # 兼容旧 checkpoint 字段（单分支）
        if "audio_injector" in state_dict:
            self.audio_injector_frame.load_state_dict(state_dict["audio_injector"], strict=False)
        if "audio_pre_norm" in state_dict:
            self.audio_pre_norm_frame.load_state_dict(state_dict["audio_pre_norm"], strict=False)
        if "audio_proj_to_dit" in state_dict and hasattr(self.audio_proj_to_dit, "load_state_dict"):
            self.audio_proj_to_dit.load_state_dict(state_dict["audio_proj_to_dit"], strict=False)

        if wan_dit is not None and "audio_processor" in state_dict:
            wan_dit.load_state_dict(state_dict["audio_processor"], strict=False)


__all__ = ["FantasyTalkingAudioConditionModel", "AudioProjModel"]
