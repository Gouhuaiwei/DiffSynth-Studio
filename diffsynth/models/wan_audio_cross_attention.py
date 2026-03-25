import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from .wan_video_dit import WanModel, flash_attention


class AudioProjModel(nn.Module):
    """音频投影模块: [B, L, Cin] -> [B, L, Cout]."""

    def __init__(self, audio_in_dim=1024, cross_attention_dim=1024):
        super().__init__()
        self.proj = nn.Linear(audio_in_dim, cross_attention_dim, bias=False)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, audio_embeds):
        return self.norm(self.proj(audio_embeds))


class WanCrossAttentionProcessor(nn.Module):
    """
    可插拔 Cross-Attention Processor：
    1) 保留原有 I2V 文本/图像 cross-attn。
    2) 增加帧对齐音频分支（wav2vec 特征）。
    3) 增加全局音频分支（emotion2vec 特征）。
    """

    def __init__(self, context_dim, hidden_dim):
        super().__init__()
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        # 帧对齐分支 K/V
        self.k_proj_frame = nn.Linear(context_dim, hidden_dim, bias=False)
        self.v_proj_frame = nn.Linear(context_dim, hidden_dim, bias=False)
        # 全局分支 K/V
        self.k_proj_global = nn.Linear(context_dim, hidden_dim, bias=False)
        self.v_proj_global = nn.Linear(context_dim, hidden_dim, bias=False)

        # 零初始化，默认不扰动原模型
        nn.init.zeros_(self.k_proj_frame.weight)
        nn.init.zeros_(self.v_proj_frame.weight)
        nn.init.zeros_(self.k_proj_global.weight)
        nn.init.zeros_(self.v_proj_global.weight)

    def __call__(
        self,
        attn: nn.Module,
        x: torch.Tensor,
        context: torch.Tensor,
        audio_proj: Optional[torch.Tensor] = None,
        audio_proj_global: Optional[torch.Tensor] = None,
        latents_num_frames: Optional[int] = None,
        audio_scale: float = 1.0,
        audio_frame_scale: float = 1.0,
        audio_global_scale: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        x:                [B, Lx, C]
        context:          [B, Lc, C]，I2V 时前 257 为图像 token
        audio_proj:       帧对齐音频，[B, T, La, C] 或 [B, T, C]
        audio_proj_global:全局音频，[B, Lg, C] 或 [B, T, Lg, C]
        """
        b, n, d = x.size(0), attn.num_heads, attn.head_dim

        # ---- 原始 I2V cross-attn ----
        if attn.has_image_input:
            context_img = context[:, :257]
            context_txt = context[:, 257:]
        else:
            context_img = None
            context_txt = context

        q = attn.norm_q(attn.q(x)).view(b, -1, n, d)
        k = attn.norm_k(attn.k(context_txt)).view(b, -1, n, d)
        v = attn.v(context_txt).view(b, -1, n, d)

        x_txt = flash_attention(q, k, v, num_heads=attn.num_heads).flatten(2)
        x_base = x_txt

        if context_img is not None:
            k_img = attn.norm_k_img(attn.k_img(context_img)).view(b, -1, n, d)
            v_img = attn.v_img(context_img).view(b, -1, n, d)
            x_img = flash_attention(q, k_img, v_img, num_heads=attn.num_heads).flatten(2)
            x_base = x_base + x_img

        audio_residual = 0

        # ---- 帧对齐分支（wav2vec）----
        if audio_proj is not None:
            if audio_proj.dim() == 3:
                audio_proj = audio_proj.unsqueeze(2)  # [B, T, 1, C]
            if audio_proj.dim() != 4:
                raise ValueError(f"audio_proj must be [B,T,C] or [B,T,N,C], got {tuple(audio_proj.shape)}")

            t = latents_num_frames if latents_num_frames is not None else audio_proj.shape[1]
            if t != audio_proj.shape[1]:
                raise ValueError(f"latents_num_frames({t}) != audio frames({audio_proj.shape[1]})")
            if q.shape[1] % t != 0:
                raise ValueError(f"video tokens {q.shape[1]} cannot be evenly split by frames {t}")

            audio_q = q.view(b * t, -1, n, d)
            audio_k = self.k_proj_frame(audio_proj).view(b * t, -1, n, d)
            audio_v = self.v_proj_frame(audio_proj).view(b * t, -1, n, d)
            audio_x = flash_attention(audio_q, audio_k, audio_v, num_heads=attn.num_heads)
            audio_x = audio_x.view(b, q.size(1), n, d).flatten(2)
            audio_residual = audio_residual + audio_x * audio_frame_scale

        # ---- 全局分支（emotion2vec）----
        if audio_proj_global is not None:
            if audio_proj_global.dim() == 4:
                audio_proj_global = audio_proj_global.flatten(1, 2)  # [B, T*L, C]
            if audio_proj_global.dim() != 3:
                raise ValueError(f"audio_proj_global must be [B,L,C] or [B,T,L,C], got {tuple(audio_proj_global.shape)}")

            global_k = self.k_proj_global(audio_proj_global).view(b, -1, n, d)
            global_v = self.v_proj_global(audio_proj_global).view(b, -1, n, d)
            global_x = flash_attention(q, global_k, global_v, num_heads=attn.num_heads).flatten(2)
            audio_residual = audio_residual + global_x * audio_global_scale

        out = x_base + audio_residual * audio_scale
        return attn.o(out)


class FantasyTalkingAudioConditionModel(nn.Module):
    """
    使用 WanModel 的 attention processor 插拔接口接入 FantasyTalking：
    - 帧对齐分支输入: audio_proj (wav2vec)
    - 全局分支输入:  audio_proj_global (emotion2vec)
    """

    def __init__(
        self,
        wan_dit: WanModel,
        audio_in_dim: int,
        audio_proj_dim: int,
        global_audio_in_dim: Optional[int] = None,
    ):
        super().__init__()
        self.audio_in_dim = audio_in_dim
        self.global_audio_in_dim = global_audio_in_dim if global_audio_in_dim is not None else audio_in_dim
        self.audio_proj_dim = audio_proj_dim

        self.proj_model_frame = AudioProjModel(
            audio_in_dim=self.audio_in_dim,
            cross_attention_dim=audio_proj_dim,
        )
        self.proj_model_global = AudioProjModel(
            audio_in_dim=self.global_audio_in_dim,
            cross_attention_dim=audio_proj_dim,
        )
        self.set_audio_processor(wan_dit)

    def set_audio_processor(self, wan_dit: WanModel):
        attn_procs = {}
        for name in wan_dit.attn_processors.keys():
            attn_procs[name] = WanCrossAttentionProcessor(
                context_dim=self.audio_proj_dim,
                hidden_dim=wan_dit.dim,
            )
        wan_dit.set_attn_processor(attn_procs)

    def load_audio_processor(self, ip_ckpt: str, wan_dit: WanModel):
        if os.path.splitext(ip_ckpt)[-1] == ".safetensors":
            state_dict = {
                "proj_model": {},
                "proj_model_frame": {},
                "proj_model_global": {},
                "audio_processor": {},
            }
            with safe_open(ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("proj_model_frame."):
                        state_dict["proj_model_frame"][key.replace("proj_model_frame.", "")] = f.get_tensor(key)
                    elif key.startswith("proj_model_global."):
                        state_dict["proj_model_global"][key.replace("proj_model_global.", "")] = f.get_tensor(key)
                    elif key.startswith("proj_model."):
                        state_dict["proj_model"][key.replace("proj_model.", "")] = f.get_tensor(key)
                    elif key.startswith("audio_processor."):
                        state_dict["audio_processor"][key.replace("audio_processor.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(ip_ckpt, map_location="cpu")

        if "proj_model_frame" in state_dict and len(state_dict["proj_model_frame"]) > 0:
            self.proj_model_frame.load_state_dict(state_dict["proj_model_frame"], strict=True)
        if "proj_model_global" in state_dict and len(state_dict["proj_model_global"]) > 0:
            self.proj_model_global.load_state_dict(state_dict["proj_model_global"], strict=True)
        if "proj_model" in state_dict and len(state_dict["proj_model"]) > 0:
            self.proj_model_frame.load_state_dict(state_dict["proj_model"], strict=True)

        if "audio_processor" in state_dict:
            wan_dit.load_state_dict(state_dict["audio_processor"], strict=False)

    def get_proj_fea(self, audio_fea=None, branch: str = "frame"):
        if audio_fea is None:
            return None
        if branch == "frame":
            return self.proj_model_frame(audio_fea)
        if branch == "global":
            return self.proj_model_global(audio_fea)
        raise ValueError(f"unsupported branch: {branch}")

    def split_audio_sequence(self, audio_proj_length, num_frames=81):
        """把长音频序列切分成与 latent 帧对应的中心区间。"""
        tokens_per_frame = audio_proj_length / num_frames
        tokens_per_latent_frame = tokens_per_frame * 4
        half_tokens = int(tokens_per_latent_frame / 2)

        pos_indices = []
        for i in range(int((num_frames - 1) / 4) + 1):
            if i == 0:
                pos_indices.append(0)
            else:
                start_token = tokens_per_frame * ((i - 1) * 4 + 1)
                end_token = tokens_per_frame * (i * 4 + 1)
                center_token = int((start_token + end_token) / 2) - 1
                pos_indices.append(center_token)

        pos_idx_ranges = [[idx - half_tokens, idx + half_tokens] for idx in pos_indices]
        pos_idx_ranges[0] = [
            -(half_tokens * 2 - pos_idx_ranges[1][0]),
            pos_idx_ranges[1][0],
        ]
        return pos_idx_ranges

    def split_tensor_with_padding(self, input_tensor, pos_idx_ranges, expand_length=0):
        """根据区间切分音频并补零到等长，返回 [B, F, L, C] 及每段有效长度。"""
        pos_idx_ranges = [[idx[0] - expand_length, idx[1] + expand_length] for idx in pos_idx_ranges]
        sub_sequences = []
        seq_len = input_tensor.size(1)
        max_valid_idx = seq_len - 1
        k_lens_list = []
        for start, end in pos_idx_ranges:
            pad_front = max(-start, 0)
            pad_back = max(end - max_valid_idx, 0)
            valid_start = max(start, 0)
            valid_end = min(end, max_valid_idx)

            if valid_start <= valid_end:
                valid_part = input_tensor[:, valid_start: valid_end + 1, :]
            else:
                valid_part = input_tensor.new_zeros((1, 0, input_tensor.size(2)))

            padded_subseq = F.pad(
                valid_part,
                (0, 0, 0, pad_back + pad_front, 0, 0),
                mode="constant",
                value=0,
            )
            k_lens_list.append(padded_subseq.size(-2) - pad_back - pad_front)
            sub_sequences.append(padded_subseq)

        return torch.stack(sub_sequences, dim=1), torch.tensor(k_lens_list, dtype=torch.long)


__all__ = ["FantasyTalkingAudioConditionModel", "AudioProjModel", "WanCrossAttentionProcessor"]
