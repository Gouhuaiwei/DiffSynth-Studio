import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange
from .wan_video_camera_controller import SimpleAdapter
from ..core.gradient import gradient_checkpoint_forward

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x,tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    freqs = freqs.to(torch.complex64) if freqs.device.type == "npu" else freqs
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


def set_to_torch_norm(models):
    for model in models:
        for module in model.modules():
            if isinstance(module, RMSNorm):
                module.use_torch_norm = True


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.use_torch_norm = False
        self.normalized_shape = (dim,)

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        if self.use_torch_norm:
            return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        else:
            return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)
        self.set_processor(CrossAttentionProcessor())

    def set_processor(self, processor):
        self.processor = processor

    def get_processor(self):
        return self.processor

    def forward(self, x: torch.Tensor, y: torch.Tensor, **kwargs):
        return self.processor(self, x, y, **kwargs)


class CrossAttentionProcessor:
    def __call__(self, attn: CrossAttention, x: torch.Tensor, y: torch.Tensor, **kwargs):
        if attn.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = attn.norm_q(attn.q(x))
        k = attn.norm_k(attn.k(ctx))
        v = attn.v(ctx)
        x = attn.attn(q, k, v)
        if attn.has_image_input:
            k_img = attn.norm_k_img(attn.k_img(img))
            v_img = attn.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=attn.num_heads)
            x = x + y
        return attn.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual


class AudioCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim, bias=False)
        self.norm_q = RMSNorm(dim, eps=eps)

        self.k_proj_frame = nn.Linear(dim, dim, bias=False)
        self.v_proj_frame = nn.Linear(dim, dim, bias=False)
        self.k_proj_global = nn.Linear(dim, dim, bias=False)
        self.v_proj_global = nn.Linear(dim, dim, bias=False)
        self.reset_parameters()

        self.attn = AttentionModule(self.num_heads)

    def reset_parameters(self):
        self.q.reset_parameters()
        self.o.reset_parameters()
        nn.init.zeros_(self.k_proj_frame.weight)
        nn.init.zeros_(self.v_proj_frame.weight)
        nn.init.zeros_(self.k_proj_global.weight)
        nn.init.zeros_(self.v_proj_global.weight)

    def init_missing_parameters(self, device=None):
        if any(param.is_meta for param in self.parameters(recurse=True)):
            device = torch.device("cpu") if device is None else device
            self.to_empty(device=device)
            self.reset_parameters()

    @staticmethod
    def _align_qkv_dtype_device(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        if k.dtype != q.dtype or k.device != q.device:
            k = k.to(dtype=q.dtype, device=q.device)
        if v.dtype != q.dtype or v.device != q.device:
            v = v.to(dtype=q.dtype, device=q.device)
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        audio_proj: Optional[torch.Tensor] = None,
        audio_proj_global: Optional[torch.Tensor] = None,
        latents_num_frames: Optional[int] = None,
        audio_scale: float = 1.0,
        audio_frame_scale: float = 1.0,
        audio_global_scale: float = 1.0,
    ):
        b = x.size(0)
        q = self.norm_q(self.q(x))

        audio_residual = torch.zeros_like(x)
        if audio_proj is not None:
            t = latents_num_frames if latents_num_frames is not None else audio_proj.shape[1]
            video_t = 4 * (t - 1) + 1
            if audio_proj.dim() == 3:
                if latents_num_frames is not None and audio_proj.shape[1] == video_t:
                    sample_idx = torch.arange(0, video_t, 4, device=audio_proj.device)
                    audio_proj = audio_proj.index_select(1, sample_idx)
                elif latents_num_frames is not None and audio_proj.shape[1] != t:
                    audio_proj = F.interpolate(audio_proj.transpose(1, 2), size=video_t, mode="linear", align_corners=False).transpose(1, 2)
                    sample_idx = torch.arange(0, video_t, 4, device=audio_proj.device)
                    audio_proj = audio_proj.index_select(1, sample_idx)
                audio_proj = audio_proj.unsqueeze(2)
            if audio_proj.dim() != 4:
                raise ValueError(f"audio_proj must be [B,T,C] or [B,T,N,C], got {tuple(audio_proj.shape)}")

            if latents_num_frames is not None and audio_proj.shape[1] == video_t:
                sample_idx = torch.arange(0, video_t, 4, device=audio_proj.device)
                audio_proj = audio_proj.index_select(1, sample_idx)
            elif t != audio_proj.shape[1]:
                src_t = audio_proj.shape[1]
                sample_idx = torch.linspace(0, src_t - 1, t, device=audio_proj.device).round().long()
                audio_proj = audio_proj.index_select(1, sample_idx)
            if q.shape[1] % t != 0:
                raise ValueError(f"video tokens {q.shape[1]} cannot be evenly split by frames {t}")

            audio_proj = audio_proj.to(dtype=self.k_proj_frame.weight.dtype, device=self.k_proj_frame.weight.device)
            tokens_per_frame = q.shape[1] // t
            audio_q = q.view(b, t, tokens_per_frame, -1).reshape(b * t, tokens_per_frame, -1)
            audio_k = self.k_proj_frame(audio_proj).reshape(b * t, -1, q.shape[-1])
            audio_v = self.v_proj_frame(audio_proj).reshape(b * t, -1, q.shape[-1])
            audio_q, audio_k, audio_v = self._align_qkv_dtype_device(audio_q, audio_k, audio_v)
            audio_x = self.attn(audio_q, audio_k, audio_v)
            audio_x = audio_x.view(b, t, tokens_per_frame, -1).reshape(b, q.size(1), -1)
            audio_residual = audio_residual + audio_x * audio_frame_scale

        if audio_proj_global is not None:
            if audio_proj_global.dim() == 4:
                audio_proj_global = audio_proj_global.flatten(1, 2)
            if audio_proj_global.dim() != 3:
                raise ValueError(f"audio_proj_global must be [B,L,C] or [B,T,L,C], got {tuple(audio_proj_global.shape)}")
            audio_proj_global = audio_proj_global.to(dtype=self.k_proj_global.weight.dtype, device=self.k_proj_global.weight.device)
            global_k = self.k_proj_global(audio_proj_global)
            global_v = self.v_proj_global(audio_proj_global)
            q_global, global_k, global_v = self._align_qkv_dtype_device(q, global_k, global_v)
            global_x = self.attn(q_global, global_k, global_v)
            audio_residual = audio_residual + global_x * audio_global_scale

        return self.o(audio_residual * audio_scale)


class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()
        self.audio_cross = AudioCrossAttention(dim, num_heads, eps)

    def forward(self, x, context, t_mod, freqs,
                audio_proj: Optional[torch.Tensor] = None,
                audio_proj_global: Optional[torch.Tensor] = None,
                latents_num_frames: Optional[int] = None,
                audio_scale: float = 1.0,
                audio_frame_scale: float = 1.0,
                audio_global_scale: float = 1.0,
                **kwargs):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
        x_norm = self.norm3(x)
        x = x + self.cross_attn(x_norm, context)
        x = x + self.audio_cross(
            x_norm,
            audio_proj=audio_proj,
            audio_proj_global=audio_proj_global,
            latents_num_frames=latents_num_frames,
            audio_scale=audio_scale,
            audio_frame_scale=audio_frame_scale,
            audio_global_scale=audio_global_scale,
        )
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class WanModel(torch.nn.Module):

    _repeated_blocks = ["DiTBlock"]

    @property
    def attn_processors(self):
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module):
            if hasattr(module, "set_processor") and hasattr(module, "processor"):
                processors[f"{name}.processor"] = module.processor
            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child)

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module)
        return processors

    def init_missing_parameters(self, device=None):
        for module in self.modules():
            if module is not self and hasattr(module, "init_missing_parameters"):
                module.init_missing_parameters(device=device)

    def set_attn_processor(self, processor):
        count = len(self.attn_processors.keys())
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the number of attention layers: {count}."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))
            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module)

    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads

        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.control_adapter = None

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        f, h, w = x.shape[-3:]
        x = rearrange(x, "b c f h w -> b (f h w) c")
        return x, (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2],
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                audio_proj: Optional[torch.Tensor] = None,
                audio_proj_global: Optional[torch.Tensor] = None,
                latents_num_frames: Optional[int] = None,
                audio_scale: float = 1.0,
                audio_frame_scale: float = 1.0,
                audio_global_scale: float = 1.0,
                **kwargs,
                ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)

        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)

        x, (f, h, w) = self.patchify(x)

        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        for block in self.blocks:
            if self.training:
                x = gradient_checkpoint_forward(
                    block,
                    use_gradient_checkpointing,
                    use_gradient_checkpointing_offload,
                    x, context, t_mod, freqs,
                    audio_proj=audio_proj,
                    audio_proj_global=audio_proj_global,
                    latents_num_frames=latents_num_frames if latents_num_frames is not None else f,
                    audio_scale=audio_scale,
                    audio_frame_scale=audio_frame_scale,
                    audio_global_scale=audio_global_scale,
                )
            else:
                x = block(
                    x, context, t_mod, freqs,
                    audio_proj=audio_proj,
                    audio_proj_global=audio_proj_global,
                    latents_num_frames=latents_num_frames if latents_num_frames is not None else f,
                    audio_scale=audio_scale,
                    audio_frame_scale=audio_frame_scale,
                    audio_global_scale=audio_global_scale,
                )

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x
