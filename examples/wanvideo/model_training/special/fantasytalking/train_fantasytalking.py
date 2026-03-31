import argparse
import json
import os
from dataclasses import dataclass

import librosa
import soundfile as sf
import fairseq
import accelerate
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from diffsynth import ModelManager, WanVideoPipeline
from diffsynth.models.wan_audio_cross_attention import FantasyTalkingAudioConditionModel


class TensorSampleDataset(Dataset):
    """
    JSONL 每行一个样本，至少包含:
    {
      "model_input": "/path/to/sample_inputs.pt",  # 必须包含 latents,timestep,context,target
      "audio_path": "/path/to/sample.wav"
    }
    可选: clip_feature, y
    """

    def __init__(self, metadata_path: str):
        self.items = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.items.append(json.loads(line))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        sample = torch.load(item["model_input"], map_location="cpu")
        sample["audio_path"] = item["audio_path"]
        return sample


def collate_fn(batch):
    # 当前脚本以 batch_size=1 为主，避免不同长度 audio 对齐复杂度
    assert len(batch) == 1, "Please use batch_size=1 for this script."
    return batch[0]


def extract_wav2vec_feature(wav2vec, wav2vec_processor, audio_path: str):
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    inputs = wav2vec_processor(audio, sampling_rate=sr, return_tensors="pt", padding=True).input_values
    inputs = inputs.to(next(wav2vec.parameters()).device)
    with torch.no_grad():
        out = wav2vec(inputs)
    return out.last_hidden_state


def extract_emotion2vec_feature(emotion_model, normalize: bool, audio_path: str):
    wav, sr = sf.read(audio_path)
    channels = sf.info(audio_path).channels
    assert sr == 16000, f"Sample rate should be 16kHz, got {sr}"
    assert channels == 1, f"Channel should be 1, got {channels}"

    with torch.no_grad():
        source = torch.from_numpy(wav).float().cuda()
        if normalize:
            source = F.layer_norm(source, source.shape)
        source = source.view(1, -1)
        feats = emotion_model.extract_features(source, padding_mask=None)
    return feats["x"]


def build_fantasytalking_checkpoint(fantasytalking: FantasyTalkingAudioConditionModel, wan_dit: torch.nn.Module):
    # 仅保存 FantasyTalking 相关参数
    audio_processor_sd = {
        k: v.cpu()
        for k, v in wan_dit.state_dict().items()
        if ".processor." in k
    }
    return {
        "proj_model_frame": fantasytalking.proj_model_frame.state_dict(),
        "proj_model_global": fantasytalking.proj_model_global.state_dict(),
        "audio_processor": audio_processor_sd,
    }


def parse_args():
    parser = argparse.ArgumentParser("Train FantasyTalking adapter (freeze Wan + audio encoders)")
    parser.add_argument("--wan_model_dir", type=str, required=True)
    parser.add_argument("--metadata_path", type=str, required=True, help="JSONL path for training samples")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--wav2vec_model_dir", type=str, default="facebook/wav2vec2-base-960h")
    parser.add_argument("--emotion2vec_user_dir", type=str, required=True)
    parser.add_argument("--emotion2vec_ckpt", type=str, required=True)

    parser.add_argument("--audio_in_dim", type=int, default=768)
    parser.add_argument("--global_audio_in_dim", type=int, default=768)
    parser.add_argument("--audio_proj_dim", type=int, default=2048)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    return parser.parse_args()


def load_models(args):
    # 1) Load Wan pipeline
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

    # 2) Install FantasyTalking adapter
    fantasytalking = FantasyTalkingAudioConditionModel(
        pipe.dit,
        audio_in_dim=args.audio_in_dim,
        audio_proj_dim=args.audio_proj_dim,
        global_audio_in_dim=args.global_audio_in_dim,
    ).to("cuda")

    # 3) Load audio encoders
    wav2vec_processor = Wav2Vec2Processor.from_pretrained(args.wav2vec_model_dir)
    wav2vec = Wav2Vec2Model.from_pretrained(args.wav2vec_model_dir).to("cuda", dtype=torch.bfloat16).eval()

    @dataclass
    class UserDirModule:
        user_dir: str

    fairseq.utils.import_user_module(UserDirModule(args.emotion2vec_user_dir))
    models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([args.emotion2vec_ckpt])
    emotion2vec = models[0].eval().cuda()
    emotion_normalize = bool(task.cfg.normalize)
    return pipe, fantasytalking, wav2vec_processor, wav2vec, emotion2vec, emotion_normalize


def freeze_models(pipe, wav2vec, emotion2vec, fantasytalking):
    # Freeze Wan + wav2vec + emotion2vec
    pipe.dit.requires_grad_(False)
    wav2vec.requires_grad_(False)
    emotion2vec.requires_grad_(False)
    # Unfreeze ONLY FantasyTalking params
    fantasytalking.requires_grad_(True)
    # Unfreeze audio processors injected into Wan DiT
    for module in pipe.dit.modules():
        if hasattr(module, "get_processor"):
            processor = module.get_processor()
            if processor.__class__.__name__ == "WanCrossAttentionProcessor":
                for p in processor.parameters():
                    p.requires_grad_(True)


def main(args, pipe, fantasytalking, wav2vec_processor, wav2vec, emotion2vec, emotion_normalize):
    os.makedirs(args.output_dir, exist_ok=True)
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    freeze_models(pipe, wav2vec, emotion2vec, fantasytalking)

    processor_params = []
    for module in pipe.dit.modules():
        if hasattr(module, "get_processor"):
            processor = module.get_processor()
            if processor.__class__.__name__ == "WanCrossAttentionProcessor":
                processor_params.extend([p for p in processor.parameters() if p.requires_grad])
    trainable_params = [p for p in fantasytalking.parameters() if p.requires_grad] + processor_params
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    dataset = TensorSampleDataset(args.metadata_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    pipe.dit, fantasytalking, optimizer, dataloader = accelerator.prepare(
        pipe.dit, fantasytalking, optimizer, dataloader
    )

    step = 0
    for epoch in range(args.epochs):
        for sample in dataloader:
            # sample_inputs.pt 需包含如下字段:
            # latents:[B,C,F,H,W], timestep:[B], context:[B,L,C], target:[B,C,F,H,W]
            latents = sample["latents"].to("cuda", dtype=torch.bfloat16)
            timestep = sample["timestep"].to("cuda", dtype=torch.bfloat16)
            context = sample["context"].to("cuda", dtype=torch.bfloat16)
            target = sample["target"].to("cuda", dtype=torch.bfloat16)
            clip_feature = sample.get("clip_feature", None)
            y = sample.get("y", None)
            if clip_feature is not None:
                clip_feature = clip_feature.to("cuda", dtype=torch.bfloat16)
            if y is not None:
                y = y.to("cuda", dtype=torch.bfloat16)

            audio_path = sample["audio_path"]
            wav_feat = extract_wav2vec_feature(wav2vec, wav2vec_processor, audio_path)
            emo_feat = extract_emotion2vec_feature(emotion2vec, emotion_normalize, audio_path)

            audio_proj = fantasytalking.get_proj_fea(wav_feat.to(dtype=torch.bfloat16), branch="frame")
            audio_proj_global = fantasytalking.get_proj_fea(emo_feat.to(dtype=torch.bfloat16), branch="global")

            pred = pipe.dit(
                x=latents,
                timestep=timestep,
                context=context,
                clip_feature=clip_feature,
                y=y,
                audio_proj=audio_proj,
                audio_proj_global=audio_proj_global,
                latents_num_frames=latents.shape[2],
                audio_scale=1.0,
                audio_frame_scale=1.0,
                audio_global_scale=1.0,
            )

            loss = F.mse_loss(pred.float(), target.float())
            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            optimizer.step()

            step += 1
            if step % 10 == 0 and accelerator.is_main_process:
                print(f"[epoch {epoch}] step={step}, loss={loss.item():.6f}")

            if step % args.save_every == 0 and accelerator.is_main_process:
                ckpt = build_fantasytalking_checkpoint(
                    accelerator.unwrap_model(fantasytalking),
                    accelerator.unwrap_model(pipe.dit),
                )
                save_path = os.path.join(args.output_dir, f"fantasytalking_step_{step}.pt")
                torch.save(ckpt, save_path)
                print(f"Saved: {save_path}")

    # final save
    if accelerator.is_main_process:
        final_ckpt = build_fantasytalking_checkpoint(
            accelerator.unwrap_model(fantasytalking),
            accelerator.unwrap_model(pipe.dit),
        )
        final_path = os.path.join(args.output_dir, "fantasytalking_final.pt")
        torch.save(final_ckpt, final_path)
        print(f"Training done. Final checkpoint: {final_path}")


if __name__ == "__main__":
    args = parse_args()
    pipe, fantasytalking, wav2vec_processor, wav2vec, emotion2vec, emotion_normalize = load_models(args)
    main(args, pipe, fantasytalking, wav2vec_processor, wav2vec, emotion2vec, emotion_normalize)
