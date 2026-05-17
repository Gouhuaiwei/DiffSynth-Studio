import os
from typing import Optional

import torch
import torch.nn as nn
from safetensors import safe_open

from .wan_video_dit import WanModel, AudioCrossAttention


class AudioProjModel(nn.Module):
    """音频投影 MLP: [B, L, Cin] -> [B, L, Cout]."""

    def __init__(self, audio_in_dim=1024, cross_attention_dim=1024):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(audio_in_dim),
            nn.Linear(audio_in_dim, cross_attention_dim),
            nn.GELU(),
            nn.Linear(cross_attention_dim, cross_attention_dim),
            nn.LayerNorm(cross_attention_dim),
        )

    def forward(self, audio_embeds):
        return self.proj(audio_embeds)


class FantasyTalkingAudioConditionModel(nn.Module):
    """FantasyTalking audio-condition wrapper for Wan DiT.

    This module owns the two trainable AudioProjModel MLPs and coordinates
    checkpoint I/O for the in-block AudioCrossAttention modules in WanModel.
    The actual audio attention layers live in wan_video_dit.DiTBlock as
    ``block.audio_cross``.
    """

    def __init__(
        self,
        wan_dit: WanModel,
        audio_in_dim: int,
        audio_proj_dim: Optional[int] = None,
        global_audio_in_dim: Optional[int] = None,
    ):
        super().__init__()
        self.audio_in_dim = audio_in_dim
        self.global_audio_in_dim = global_audio_in_dim if global_audio_in_dim is not None else audio_in_dim
        # 若未显式指定，则默认对齐到 Wan hidden dim
        self.audio_proj_dim = audio_proj_dim if audio_proj_dim is not None else wan_dit.dim

        self.proj_model_frame = AudioProjModel(
            audio_in_dim=self.audio_in_dim,
            cross_attention_dim=self.audio_proj_dim,
        )
        self.proj_model_global = AudioProjModel(
            audio_in_dim=self.global_audio_in_dim,
            cross_attention_dim=self.audio_proj_dim,
        )
        self.validate_audio_cross_attention(wan_dit)

    @property
    def audio_proj_model_frame(self) -> AudioProjModel:
        return self.proj_model_frame

    @property
    def audio_proj_model_global(self) -> AudioProjModel:
        return self.proj_model_global

    def validate_audio_cross_attention(self, wan_dit: WanModel):
        if self.audio_proj_dim != wan_dit.dim:
            raise ValueError(
                f"audio_proj_dim ({self.audio_proj_dim}) must match wan_dit.dim ({wan_dit.dim}) when using in-block audio attention."
            )
        if not any(isinstance(module, AudioCrossAttention) for module in wan_dit.modules()):
            raise ValueError("wan_dit does not contain AudioCrossAttention modules. Please use the updated WanModel with DiTBlock.audio_cross.")

    def audio_cross_modules(self, wan_dit: WanModel):
        return [module for module in wan_dit.modules() if isinstance(module, AudioCrossAttention)]

    def audio_cross_state_dict(self, wan_dit: WanModel):
        return {k: v.detach().cpu() for k, v in wan_dit.state_dict().items() if ".audio_cross." in k}

    def build_state_dict(self, wan_dit: WanModel):
        state_dict = {}
        for k, v in self.audio_proj_model_frame.state_dict().items():
            state_dict[f"proj_model_frame.{k}"] = v.detach().cpu()
        for k, v in self.audio_proj_model_global.state_dict().items():
            state_dict[f"proj_model_global.{k}"] = v.detach().cpu()
        for k, v in self.audio_cross_state_dict(wan_dit).items():
            state_dict[f"audio_blocks.{k}"] = v
        return state_dict


    @staticmethod
    def _normalize_audio_proj_state_dict(state_dict):
        if "proj.weight" in state_dict:
            state_dict = dict(state_dict)
            state_dict["proj.1.weight"] = state_dict.pop("proj.weight")
            if "norm.weight" in state_dict:
                state_dict["proj.4.weight"] = state_dict.pop("norm.weight")
            if "norm.bias" in state_dict:
                state_dict["proj.4.bias"] = state_dict.pop("norm.bias")
        return state_dict

    def set_audio_processor(self, wan_dit: WanModel):
        # Backward-compatible name: audio attention is now implemented by
        # DiTBlock.audio_cross, so setup only validates the in-block modules.
        self.validate_audio_cross_attention(wan_dit)

    def enable_audio_cross_lora(
        self,
        wan_dit: WanModel,
        rank: int = 8,
        alpha: Optional[int] = None,
        freeze_base: bool = False,
    ):
        try:
            from peft import LoraConfig, inject_adapter_in_model
        except ImportError as e:
            raise ImportError("LoRA training requires `peft` to be installed.") from e

        lora_alpha = rank if alpha is None else alpha
        target_modules = ["k_proj_frame", "v_proj_frame", "k_proj_global", "v_proj_global"]
        lora_config = LoraConfig(r=rank, lora_alpha=lora_alpha, target_modules=target_modules)

        if freeze_base:
            for n, p in wan_dit.named_parameters():
                if all(k not in n for k in target_modules):
                    p.requires_grad_(False)

        inject_adapter_in_model(lora_config, wan_dit)
        return {"rank": rank, "alpha": lora_alpha, "freeze_base": freeze_base}

    def enable_audio_processor_lora(self, *args, **kwargs):
        return self.enable_audio_cross_lora(*args, **kwargs)

    def load_audio_condition_model(self, ip_ckpt: str, wan_dit: WanModel):
        if os.path.splitext(ip_ckpt)[-1] == ".safetensors":
            state_dict = {
                "proj_model": {},
                "proj_model_frame": {},
                "proj_model_global": {},
                "audio_processor": {},
                "audio_blocks": {},
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
                    elif key.startswith("audio_blocks."):
                        state_dict["audio_blocks"][key.replace("audio_blocks.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(ip_ckpt, map_location="cpu")

        if "proj_model_frame" in state_dict and len(state_dict["proj_model_frame"]) > 0:
            self.proj_model_frame.load_state_dict(self._normalize_audio_proj_state_dict(state_dict["proj_model_frame"]), strict=False)
        if "proj_model_global" in state_dict and len(state_dict["proj_model_global"]) > 0:
            self.proj_model_global.load_state_dict(self._normalize_audio_proj_state_dict(state_dict["proj_model_global"]), strict=False)
        if "proj_model" in state_dict and len(state_dict["proj_model"]) > 0:
            self.proj_model_frame.load_state_dict(self._normalize_audio_proj_state_dict(state_dict["proj_model"]), strict=False)

        lora_config = state_dict.get("audio_processor_lora_config", None)
        if lora_config is not None:
            self.enable_audio_cross_lora(
                wan_dit,
                rank=int(lora_config.get("rank", 8)),
                alpha=int(lora_config.get("alpha", lora_config.get("rank", 8))),
                freeze_base=bool(lora_config.get("freeze_base", False)),
            )

        if "audio_blocks" in state_dict and len(state_dict["audio_blocks"]) > 0:
            wan_dit.load_state_dict(state_dict["audio_blocks"], strict=False)
        elif "audio_processor" in state_dict:
            # Backward compatibility for older checkpoints.
            wan_dit.load_state_dict(state_dict["audio_processor"], strict=False)

    def load_audio_processor(self, *args, **kwargs):
        return self.load_audio_condition_model(*args, **kwargs)

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


__all__ = ["FantasyTalkingAudioConditionModel", "AudioProjModel"]
