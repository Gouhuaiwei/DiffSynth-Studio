import argparse
import os

import torch
from PIL import Image
import numpy as np

from diffsynth import ModelManager
from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.utils.data import save_video
from diffsynth.models.wan_audio_cross_attention import FantasyTalkingAudioConditionModel


def parse_args():
    parser = argparse.ArgumentParser(description="FantasyTalking Wan I2V inference (text + ref image + audio)")
    parser.add_argument("--wan_model_dir", type=str, required=True)
    parser.add_argument("--reference_image", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--wav2vec_feature", type=str, default="", help="Path to wav2vec feature tensor (.pt/.pth), shape [B,T,C] or [T,C]")
    parser.add_argument("--emotion2vec_feature", type=str, default="", help="Path to emotion2vec feature tensor (.pt/.pth), shape [B,L,C] or [L,C]")
    parser.add_argument("--input_audio", type=str, default="", help="Raw audio path. If provided and feature files are empty, script extracts wav2vec/emotion2vec features online.")
    parser.add_argument("--wav2vec_model_id", type=str, default="facebook/wav2vec2-base-960h")
    parser.add_argument("--emotion2vec_model_id", type=str, default="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim")
    parser.add_argument("--fantasytalking_model_path", type=str, default="", help="Optional adapter checkpoint")
    parser.add_argument("--output", type=str, default="video_fantasytalking_wan.mp4")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--audio_scale", type=float, default=1.0)
    parser.add_argument("--audio_frame_scale", type=float, default=1.0)
    parser.add_argument("--audio_global_scale", type=float, default=1.0)
    return parser.parse_args()


def load_feature(path: str, device: str, dtype: torch.dtype):
    feat = torch.load(path, map_location="cpu")
    if isinstance(feat, dict):
        # 兼容常见保存格式
        for key in ["features", "embeds", "audio_features", "x"]:
            if key in feat:
                feat = feat[key]
                break
    if feat.dim() == 2:
        feat = feat.unsqueeze(0)
    return feat.to(device=device, dtype=dtype)


def load_waveform(audio_path: str):
    try:
        import soundfile as sf
        wav, sr = sf.read(audio_path)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        return wav.astype(np.float32), sr
    except Exception:
        import torchaudio
        wav, sr = torchaudio.load(audio_path)
        wav = wav.mean(dim=0).numpy().astype(np.float32)
        return wav, sr


def extract_hf_audio_features(waveform, sample_rate, model_id, device, dtype):
    from transformers import AutoFeatureExtractor, AutoModel
    extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device=device, dtype=dtype)
    model.eval()
    inputs = extractor(
        waveform,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device=device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    hidden = outputs.last_hidden_state
    return hidden


def main():
    args = parse_args()
    device = "cuda"
    dtype = torch.bfloat16

    # 1) 加载 Wan I2V 基础模型
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        [
            [
                f"{args.wan_model_dir}/diffusion_pytorch_model-00001-of-00007.safetensors",
                f"{args.wan_model_dir}/diffusion_pytorch_model-00002-of-00007.safetensors",
                f"{args.wan_model_dir}/diffusion_pytorch_model-00003-of-00007.safetensors",
                f"{args.wan_model_dir}/diffusion_pytorch_model-00004-of-00007.safetensors",
                f"{args.wan_model_dir}/diffusion_pytorch_model-00005-of-00007.safetensors",
                f"{args.wan_model_dir}/diffusion_pytorch_model-00006-of-00007.safetensors",
                f"{args.wan_model_dir}/diffusion_pytorch_model-00007-of-00007.safetensors",
            ],
            f"{args.wan_model_dir}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            f"{args.wan_model_dir}/models_t5_umt5-xxl-enc-bf16.pth",
            f"{args.wan_model_dir}/Wan2.1_VAE.pth",
        ],
        torch_dtype=dtype,
    )
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=dtype, device=device)

    # 2) 安装 FantasyTalking processor（processor 插拔方式）
    fantasytalking = FantasyTalkingAudioConditionModel(
        pipe.dit,
        audio_in_dim=768,
        audio_proj_dim=2048,
        global_audio_in_dim=1024,
    ).to(device)

    if args.fantasytalking_model_path and os.path.isfile(args.fantasytalking_model_path):
        fantasytalking.load_audio_processor(args.fantasytalking_model_path, pipe.dit)

    # 3) 读取两路音频特征并投影
    if args.wav2vec_feature and args.emotion2vec_feature:
        wav2vec_fea = load_feature(args.wav2vec_feature, device, dtype)
        emo2vec_fea = load_feature(args.emotion2vec_feature, device, dtype)
    else:
        if not args.input_audio:
            raise ValueError("Please provide either --wav2vec_feature/--emotion2vec_feature or --input_audio.")
        waveform, sample_rate = load_waveform(args.input_audio)
        # 帧对齐分支：wav2vec 特征
        wav2vec_fea = extract_hf_audio_features(
            waveform, sample_rate, args.wav2vec_model_id, device, dtype
        )
        # 全局分支：emotion2vec 特征（默认使用 emotion 领域模型）
        emo2vec_fea = extract_hf_audio_features(
            waveform, sample_rate, args.emotion2vec_model_id, device, dtype
        )

    audio_proj = fantasytalking.get_proj_fea(wav2vec_fea, branch="frame")
    audio_proj_global = fantasytalking.get_proj_fea(emo2vec_fea, branch="global")

    # 4) 用一个轻量 model_fn 覆盖 pipeline 默认 model_fn，显式把音频分支参数送到 dit.forward
    def model_fn_audio(
        dit,
        latents=None,
        timestep=None,
        context=None,
        clip_feature=None,
        y=None,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        **kwargs,
    ):
        latents_num_frames = latents.shape[2]
        return dit(
            x=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            audio_proj=audio_proj,
            audio_proj_global=audio_proj_global,
            latents_num_frames=latents_num_frames,
            audio_scale=args.audio_scale,
            audio_frame_scale=args.audio_frame_scale,
            audio_global_scale=args.audio_global_scale,
        )

    pipe.model_fn = model_fn_audio

    # 5) 推理
    ref_img = Image.open(args.reference_image).convert("RGB").resize((args.width, args.height))

    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        input_image=ref_img,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        seed=args.seed,
        tiled=True,
        switch_DiT_boundary=1.0,
    )

    save_video(video, args.output, fps=15, quality=5)
    print(f"Saved video to: {args.output}")


if __name__ == "__main__":
    main()
