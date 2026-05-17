#!/usr/bin/env bash
set -euo pipefail

# =========================
# FantasyTalking 训练样例
# =========================
# 运行前请先修改以下路径
WAN_MODEL_DIR="/path/to/Wan2.1-I2V-14B-720P"
DATASET_BASE_PATH="/path/to/fantasytalking_dataset"
DATASET_METADATA_PATH="${DATASET_BASE_PATH}/metadata.jsonl"
OUTPUT_PATH="./models/fantasytalking_exp01"

# emotion2vec（fairseq）配置
EMOTION2VEC_USER_DIR="/path/to/emotion2vec/upstream"
EMOTION2VEC_CKPT="/path/to/emotion2vec_base.pt"
# 可选：从已有 checkpoint 继续训练（safetensors 或 pt）
# RESUME_CKPT="/path/to/step-1000.safetensors"

# 可选：多卡训练
# export CUDA_VISIBLE_DEVICES=0,1,2,3

python examples/wanvideo/model_training/special/fantasytalking/train_fantasytalking.py \
  --wan_model_dir "${WAN_MODEL_DIR}" \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${DATASET_METADATA_PATH}" \
  --data_file_keys "video,input_audio" \
  --output_path "${OUTPUT_PATH}" \
  --emotion2vec_user_dir "${EMOTION2VEC_USER_DIR}" \
  --emotion2vec_ckpt "${EMOTION2VEC_CKPT}" \
  --wav2vec_model_dir "facebook/wav2vec2-base-960h" \
  --audio_in_dim 768 \
  --global_audio_in_dim 768 \
  --task "sft" \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --batch_size 1 \
  --num_epochs 10 \
  --save_steps 200 \
  --gradient_accumulation_steps 1 \
  --dataset_num_workers 4 \
  --height 512 \
  --width 512 \
  --num_frames 81 \
  --max_pixels 1048576 \
  --cfg_scale 1.0 \
  --use_gradient_checkpointing \
  --remove_prefix_in_ckpt ""

# 如果要继续训练，去掉下面这行注释并运行：
#   --resume_from_checkpoint "${RESUME_CKPT}" \

echo "Done. Checkpoints are saved to: ${OUTPUT_PATH}"
