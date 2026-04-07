# Copyright Alibaba Inc. All Rights Reserved.

import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

import librosa
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import fairseq
import soundfile as sf

from diffsynth import ModelManager, WanVideoPipeline
from diffsynth.models.wan_audio_cross_attention import FantasyTalkingAudioConditionModel
from diffsynth.utils.data import save_video


NEGATIVE_PROMPT = (
    "人物静止不动，静止，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
    "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def parse_args():
    parser = argparse.ArgumentParser(description="FantasyTalking Wan I2V inference script.")

    parser.add_argument("--wan_model_dir", type=str, required=True)
    parser.add_argument("--fantasytalking_model_path", type=str, default="")

    parser.add_argument("--wav2vec_model_dir", type=str, default="facebook/wav2vec2-base-960h")
    parser.add_argument("--emotion2vec_user_dir", type=str, required=True, help="Path to emotion2vec fairseq upstream user_dir")
    parser.add_argument("--emotion2vec_ckpt", type=str, required=True, help="Path to emotion2vec fairseq checkpoint, e.g. emotion2vec_base.pt")

    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="A person is speaking to camera.")

    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--image_size", type=int, default=512)

    parser.add_argument("--audio_scale", type=float, default=1.0)
    parser.add_argument("--audio_frame_scale", type=float, default=1.0)
    parser.add_argument("--audio_global_scale", type=float, default=1.0)
    parser.add_argument("--prompt_cfg_scale", type=float, default=5.0)

    parser.add_argument("--max_num_frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=23)
    parser.add_argument("--num_inference_steps", type=int, default=30)

    parser.add_argument("--num_persistent_param_in_dit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1111)
    return parser.parse_args()


def resize_image_by_longest_edge(image_path: str, longest_edge: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    if max(w, h) == longest_edge:
        return image
    scale = float(longest_edge) / float(max(w, h))
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    # Wan I2V 对齐到 16 的倍数
    new_w = max(16, (new_w // 16) * 16)
    new_h = max(16, (new_h // 16) * 16)
    return image.resize((new_w, new_h), Image.LANCZOS)


def get_audio_features_wav2vec(wav2vec, wav2vec_processor, audio_path: str):
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    inputs = wav2vec_processor(audio, sampling_rate=sr, return_tensors="pt", padding=True).input_values
    inputs = inputs.to(device=next(wav2vec.parameters()).device)
    with torch.no_grad():
        out = wav2vec(inputs)
    return out.last_hidden_state


@dataclass
class UserDirModule:
    user_dir: str


def get_audio_features_emotion2vec(emotion_model, normalize: bool, audio_path: str):
    wav, sr = sf.read(audio_path)
    channel = sf.info(audio_path).channels
    assert sr == 16000, f"Sample rate should be 16kHz, got {sr}"
    assert channel == 1, f"Channel should be 1, got {channel}"

    with torch.no_grad():
        source = torch.from_numpy(wav).float().cuda()
        if normalize:
            source = F.layer_norm(source, source.shape)
        source = source.view(1, -1)
        feats = emotion_model.extract_features(source, padding_mask=None)
    return feats["x"]


def load_models(args):
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
        torch_dtype=torch.bfloat16,
    )
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")

    fantasytalking = FantasyTalkingAudioConditionModel(
        pipe.dit,
        audio_in_dim=768,
        audio_proj_dim=2048,
        global_audio_in_dim=768,       # emotion2vec_base 默认 768
    ).to("cuda")
    if args.fantasytalking_model_path and os.path.isfile(args.fantasytalking_model_path):
        fantasytalking.load_audio_processor(args.fantasytalking_model_path, pipe.dit)

    pipe.enable_vram_management(num_persistent_param_in_dit=args.num_persistent_param_in_dit)

    wav2vec_processor = Wav2Vec2Processor.from_pretrained(args.wav2vec_model_dir)
    wav2vec = Wav2Vec2Model.from_pretrained(args.wav2vec_model_dir).to("cuda", dtype=torch.bfloat16)

    fairseq.utils.import_user_module(UserDirModule(args.emotion2vec_user_dir))
    models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([args.emotion2vec_ckpt])
    emotion_model = models[0].eval().cuda()
    emotion_normalize = bool(task.cfg.normalize)

    return pipe, fantasytalking, wav2vec_processor, wav2vec, emotion_model, emotion_normalize


def main(args, pipe, fantasytalking, wav2vec_processor, wav2vec, emotion_model, emotion_normalize):
    os.makedirs(args.output_dir, exist_ok=True)

    duration = librosa.get_duration(path=args.audio_path)
    num_frames = min(int(args.fps * duration // 4) * 4 + 5, args.max_num_frames)

    audio_wav2vec_fea = get_audio_features_wav2vec(wav2vec, wav2vec_processor, args.audio_path)
    audio_emotion2vec_fea = get_audio_features_emotion2vec(emotion_model, emotion_normalize, args.audio_path)

    image = resize_image_by_longest_edge(args.image_path, args.image_size)
    width, height = image.size

    audio_proj_frame = fantasytalking.get_proj_fea(audio_wav2vec_fea.to(dtype=torch.bfloat16), branch="frame")
    audio_proj_global = fantasytalking.get_proj_fea(audio_emotion2vec_fea.to(dtype=torch.bfloat16), branch="global")

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
            audio_proj=audio_proj_frame,
            audio_proj_global=audio_proj_global,
            latents_num_frames=latents_num_frames,
            audio_scale=args.audio_scale,
            audio_frame_scale=args.audio_frame_scale,
            audio_global_scale=args.audio_global_scale,
        )

    pipe.model_fn = model_fn_audio

    video_audio = pipe(
        prompt=args.prompt,
        negative_prompt=NEGATIVE_PROMPT,
        input_image=image,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        tiled=True,
        cfg_scale=args.prompt_cfg_scale,
        switch_DiT_boundary=1.0,
    )

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_path = f"{args.output_dir}/tmp_{Path(args.image_path).stem}_{Path(args.audio_path).stem}_{current_time}.mp4"
    save_video(video_audio, tmp_path, fps=args.fps, quality=5)

    save_path = f"{args.output_dir}/{Path(args.image_path).stem}_{Path(args.audio_path).stem}_{current_time}.mp4"
    final_command = [
        "ffmpeg",
        "-y",
        "-i",
        tmp_path,
        "-i",
        args.audio_path,
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-shortest",
        save_path,
    ]
    subprocess.run(final_command, check=True)
    os.remove(tmp_path)
    return save_path


if __name__ == "__main__":
    args = parse_args()
    pipe, fantasytalking, wav2vec_processor, wav2vec, emotion_model, emotion_normalize = load_models(args)
    out = main(args, pipe, fantasytalking, wav2vec_processor, wav2vec, emotion_model, emotion_normalize)
    print(f"Saved video: {out}")
