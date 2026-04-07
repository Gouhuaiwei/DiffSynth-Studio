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
from torch.utils.data import Dataset
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from diffsynth import ModelManager, WanVideoPipeline
from diffsynth.core.data.operators import LoadVideo, ImageCropAndResize
from diffsynth.diffusion import DiffusionTrainingModule, FlowMatchSFTLoss, ModelLogger, launch_training_task
from diffsynth.models.wan_audio_cross_attention import FantasyTalkingAudioConditionModel


class TensorSampleDataset(Dataset):
    """
    JSONL 每行一个样本，至少包含:
    {
      "video_path": "/path/to/sample.mp4",
      "audio_path": "/path/to/sample.wav"
      "prompt": "a person talking"
    }
    """

    def __init__(self, metadata_path: str, num_frames: int, height: int, width: int):
        self.items = []
        self.load_from_cache = False
        self.video_loader = LoadVideo(
            num_frames=num_frames,
            time_division_factor=4,
            time_division_remainder=1,
            frame_processor=ImageCropAndResize(height, width, None, 16, 16),
        )
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.items.append(json.loads(line))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        prompt = item.get("prompt", item.get("text", ""))
        video = self.video_loader(item["video_path"])
        return {"prompt": prompt, "video": video, "audio_path": item["audio_path"]}

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
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_frames", type=int, default=81)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    return parser.parse_args()


class FantasyTalkingTrainingModule(DiffusionTrainingModule):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.pipe, self.fantasytalking, self.wav2vec_processor, self.wav2vec, self.emotion2vec, self.emotion_normalize = self.load_models()
        self.freeze_models()

    def load_models(self):
        model_manager = ModelManager(device="cpu")
        model_manager.load_models(
            [
                [
                    f"{self.args.wan_model_dir}/diffusion_pytorch_model-00001-of-00007.safetensors",
                    f"{self.args.wan_model_dir}/diffusion_pytorch_model-00002-of-00007.safetensors",
                    f"{self.args.wan_model_dir}/diffusion_pytorch_model-00003-of-00007.safetensors",
                    f"{self.args.wan_model_dir}/diffusion_pytorch_model-00004-of-00007.safetensors",
                    f"{self.args.wan_model_dir}/diffusion_pytorch_model-00005-of-00007.safetensors",
                    f"{self.args.wan_model_dir}/diffusion_pytorch_model-00006-of-00007.safetensors",
                    f"{self.args.wan_model_dir}/diffusion_pytorch_model-00007-of-00007.safetensors",
                ],
                f"{self.args.wan_model_dir}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                f"{self.args.wan_model_dir}/models_t5_umt5-xxl-enc-bf16.pth",
                f"{self.args.wan_model_dir}/Wan2.1_VAE.pth",
            ],
            torch_dtype=torch.bfloat16,
        )
        pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=self.device)
        fantasytalking = FantasyTalkingAudioConditionModel(
            pipe.dit,
            audio_in_dim=self.args.audio_in_dim,
            audio_proj_dim=self.args.audio_proj_dim,
            global_audio_in_dim=self.args.global_audio_in_dim,
        ).to(self.device)
        wav2vec_processor = Wav2Vec2Processor.from_pretrained(self.args.wav2vec_model_dir)
        wav2vec = Wav2Vec2Model.from_pretrained(self.args.wav2vec_model_dir).to(self.device, dtype=torch.bfloat16).eval()

        @dataclass
        class UserDirModule:
            user_dir: str

        fairseq.utils.import_user_module(UserDirModule(self.args.emotion2vec_user_dir))
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([self.args.emotion2vec_ckpt])
        emotion2vec = models[0].eval().to(self.device)
        emotion_normalize = bool(task.cfg.normalize)
        return pipe, fantasytalking, wav2vec_processor, wav2vec, emotion2vec, emotion_normalize

    def freeze_models(self):
        self.pipe.dit.requires_grad_(False)
        self.wav2vec.requires_grad_(False)
        self.emotion2vec.requires_grad_(False)
        self.fantasytalking.requires_grad_(True)
        for module in self.pipe.dit.modules():
            if hasattr(module, "get_processor"):
                processor = module.get_processor()
                if processor.__class__.__name__ == "WanCrossAttentionProcessor":
                    for p in processor.parameters():
                        p.requires_grad_(True)

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        return build_fantasytalking_checkpoint(self.fantasytalking, self.pipe.dit)

    def forward(self, data, inputs=None):
        sample = data if inputs is None else inputs
        prompt = sample["prompt"]
        video = sample["video"]
        self.pipe.load_models_to_device(["vae", "text_encoder"])
        input_video = self.pipe.preprocess_video(video)
        input_latents = self.pipe.vae.encode(input_video, device=self.pipe.device).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        prompt = [prompt] if isinstance(prompt, str) else prompt
        ids, mask = self.pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.pipe.device)
        mask = mask.to(self.pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            context[:, v:] = 0

        wav_feat = extract_wav2vec_feature(self.wav2vec, self.wav2vec_processor, sample["audio_path"])
        emo_feat = extract_emotion2vec_feature(self.emotion2vec, self.emotion_normalize, sample["audio_path"])
        audio_proj = self.fantasytalking.get_proj_fea(wav_feat.to(dtype=torch.bfloat16), branch="frame")
        audio_proj_global = self.fantasytalking.get_proj_fea(emo_feat.to(dtype=torch.bfloat16), branch="global")

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
                latents_num_frames=input_latents.shape[2],
                audio_scale=1.0,
                audio_frame_scale=1.0,
                audio_global_scale=1.0,
            )

        self.pipe.model_fn = model_fn_audio
        return FlowMatchSFTLoss(
            self.pipe,
            input_latents=input_latents,
            context=context,
            clip_feature=None,
            y=None,
            use_gradient_checkpointing=False,
            use_gradient_checkpointing_offload=False,
        )


if __name__ == "__main__":
    args = parse_args()
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    dataset = TensorSampleDataset(args.metadata_path, num_frames=args.num_frames, height=args.height, width=args.width)
    model = FantasyTalkingTrainingModule(args, device=accelerator.device)
    model_logger = ModelLogger(args.output_dir)
    launch_training_task(
        accelerator,
        dataset,
        model,
        model_logger,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        save_steps=args.save_every,
        num_epochs=args.epochs,
    )
