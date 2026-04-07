import os, argparse, accelerate, warnings
import torch
import torch.nn.functional as F
import fairseq
from dataclasses import dataclass
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *
from diffsynth.models.wan_audio_cross_attention import FantasyTalkingAudioConditionModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def build_fantasytalking_checkpoint(fantasytalking: FantasyTalkingAudioConditionModel, wan_dit: torch.nn.Module):
    audio_processor_sd = {k: v.cpu() for k, v in wan_dit.state_dict().items() if ".processor." in k}
    return {
        "proj_model_frame": fantasytalking.proj_model_frame.state_dict(),
        "proj_model_global": fantasytalking.proj_model_global.state_dict(),
        "audio_processor": audio_processor_sd,
    }


class WanFantasyTalkingTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        wan_model_dir=None,
        wav2vec_model_dir="facebook/wav2vec2-base-960h",
        emotion2vec_user_dir=None,
        emotion2vec_ckpt=None,
        audio_in_dim=768,
        global_audio_in_dim=768,
        audio_proj_dim=2048,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    ):
        super().__init__()
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True

        model_configs = [
            ModelConfig(path=f"{wan_model_dir}/diffusion_pytorch_model-00001-of-00007.safetensors"),
            ModelConfig(path=f"{wan_model_dir}/diffusion_pytorch_model-00002-of-00007.safetensors"),
            ModelConfig(path=f"{wan_model_dir}/diffusion_pytorch_model-00003-of-00007.safetensors"),
            ModelConfig(path=f"{wan_model_dir}/diffusion_pytorch_model-00004-of-00007.safetensors"),
            ModelConfig(path=f"{wan_model_dir}/diffusion_pytorch_model-00005-of-00007.safetensors"),
            ModelConfig(path=f"{wan_model_dir}/diffusion_pytorch_model-00006-of-00007.safetensors"),
            ModelConfig(path=f"{wan_model_dir}/diffusion_pytorch_model-00007-of-00007.safetensors"),
            ModelConfig(path=f"{wan_model_dir}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
            ModelConfig(path=f"{wan_model_dir}/models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(path=f"{wan_model_dir}/Wan2.1_VAE.pth"),
        ]
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )

        self.fantasytalking = FantasyTalkingAudioConditionModel(
            self.pipe.dit,
            audio_in_dim=audio_in_dim,
            audio_proj_dim=audio_proj_dim,
            global_audio_in_dim=global_audio_in_dim,
        ).to(device)
        self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(wav2vec_model_dir)
        self.wav2vec = Wav2Vec2Model.from_pretrained(wav2vec_model_dir).to(device, dtype=torch.bfloat16).eval()

        @dataclass
        class UserDirModule:
            user_dir: str

        fairseq.utils.import_user_module(UserDirModule(emotion2vec_user_dir))
        models, cfg, task_emo = fairseq.checkpoint_utils.load_model_ensemble_and_task([emotion2vec_ckpt])
        self.emotion2vec = models[0].eval().to(device)
        self.emotion_normalize = bool(task_emo.cfg.normalize)

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

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

    def extract_wav2vec_feature(self, input_audio):
        inputs = self.wav2vec_processor(input_audio, sampling_rate=16000, return_tensors="pt", padding=True).input_values
        inputs = inputs.to(next(self.wav2vec.parameters()).device)
        with torch.no_grad():
            out = self.wav2vec(inputs)
        return out.last_hidden_state

    def extract_emotion2vec_feature(self, input_audio):
        with torch.no_grad():
            source = torch.from_numpy(input_audio).float().to(self.pipe.device)
            if self.emotion_normalize:
                source = F.layer_norm(source, source.shape)
            source = source.view(1, -1)
            feats = self.emotion2vec.extract_features(source, padding_mask=None)
        return feats["x"]

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared

    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        return build_fantasytalking_checkpoint(self.fantasytalking, self.pipe.dit)

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)

        inputs_shared, inputs_posi, inputs_nega = inputs
        wav_feat = self.extract_wav2vec_feature(data["input_audio"])
        emo_feat = self.extract_emotion2vec_feature(data["input_audio"])
        audio_proj = self.fantasytalking.get_proj_fea(wav_feat.to(dtype=torch.bfloat16), branch="frame")
        audio_proj_global = self.fantasytalking.get_proj_fea(emo_feat.to(dtype=torch.bfloat16), branch="global")

        base_model_fn = self.pipe.model_fn

        def model_fn_audio(*args, **kwargs):
            kwargs["audio_proj"] = audio_proj
            kwargs["audio_proj_global"] = audio_proj_global
            kwargs["latents_num_frames"] = inputs_shared["input_latents"].shape[2]
            kwargs["audio_scale"] = 1.0
            kwargs["audio_frame_scale"] = 1.0
            kwargs["audio_global_scale"] = 1.0
            return base_model_fn(*args, **kwargs)

        self.pipe.model_fn = model_fn_audio
        loss = self.task_to_loss[self.task](self.pipe, inputs_shared, inputs_posi, inputs_nega)
        if not loss.requires_grad:
            max_timestep_boundary = int(inputs_shared.get("max_timestep_boundary", 1) * len(self.pipe.scheduler.timesteps))
            min_timestep_boundary = int(inputs_shared.get("min_timestep_boundary", 0) * len(self.pipe.scheduler.timesteps))
            timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
            timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)

            noise = torch.randn_like(inputs_shared["input_latents"])
            latents = self.pipe.scheduler.add_noise(inputs_shared["input_latents"], noise, timestep)
            target = self.pipe.scheduler.training_target(inputs_shared["input_latents"], noise, timestep)

            pred = self.pipe.dit(
                x=latents,
                timestep=timestep,
                context=inputs_shared["context"],
                clip_feature=inputs_shared.get("clip_feature", None),
                y=inputs_shared.get("y", None),
                use_gradient_checkpointing=inputs_shared.get("use_gradient_checkpointing", False),
                use_gradient_checkpointing_offload=inputs_shared.get("use_gradient_checkpointing_offload", False),
                audio_proj=audio_proj,
                audio_proj_global=audio_proj_global,
                latents_num_frames=inputs_shared["input_latents"].shape[2],
                audio_scale=1.0,
                audio_frame_scale=1.0,
                audio_global_scale=1.0,
            )
            loss = torch.nn.functional.mse_loss(pred.float(), target.float()) * self.pipe.scheduler.training_weight(timestep)
        self.pipe.model_fn = base_model_fn
        return loss


def wan_parser():
    parser = argparse.ArgumentParser(description="FantasyTalking training with Wan-style runner.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--wan_model_dir", type=str, required=True)
    parser.add_argument("--wav2vec_model_dir", type=str, default="facebook/wav2vec2-base-960h")
    parser.add_argument("--emotion2vec_user_dir", type=str, required=True)
    parser.add_argument("--emotion2vec_ckpt", type=str, required=True)
    parser.add_argument("--audio_in_dim", type=int, default=768)
    parser.add_argument("--global_audio_in_dim", type=int, default=768)
    parser.add_argument("--audio_proj_dim", type=int, default=2048)
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0)
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0)
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true")
    return parser


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )

    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
        special_operator_map={
            "video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(args.height, args.width, None, 16, 16)),
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
        }
    )

    model = WanFantasyTalkingTrainingModule(
        wan_model_dir=args.wan_model_dir,
        wav2vec_model_dir=args.wav2vec_model_dir,
        emotion2vec_user_dir=args.emotion2vec_user_dir,
        emotion2vec_ckpt=args.emotion2vec_ckpt,
        audio_in_dim=args.audio_in_dim,
        global_audio_in_dim=args.global_audio_in_dim,
        audio_proj_dim=args.audio_proj_dim,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )

    model_logger = ModelLogger(args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt)
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
