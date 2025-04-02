#!/bin/bash

#SBATCH --partition gpu
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --requeue
#SBATCH --constraint="rtx5000|rtx3090|a5000"
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-04:00:00
#SBATCH --mem-per-cpu 64G
#SBATCH --gres=gpu
#SBATCH --output logs/log_qwen2_finetune_%J.log                # Output log file
#SBATCH --job-name=qwen2_finetune

date;hostname;pwd

module load miniconda
#conda activate qwen2
#conda env remove --prefix /home/sv572/.conda/envs/qwen2
#conda env create -f /home/sv572/.conda/envs/qwen2.yaml --prefix /vast/palmer/home.mccleary/sv572/project/.conda/envs/qwen2
conda activate /vast/palmer/home.mccleary/sv572/project/.conda/envs/qwen2
cd /home/sv572/project/BrainLMv2-dev/Qwen2-VL-Finetune/

# You can use 2B instead of 7B
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

export PYTHONPATH=src:$PYTHONPATH

GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=4
NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

# If you want to tune the `embed_token` with LoRA, You need to tune `lm_head` together
# You should freeze the the merger also, becuase the merger is included in the vision_tower.

deepspeed src/training/train.py \
    --use_liger True \
    --lora_enable True \
    --vision_lora False \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 64 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path /vast/palmer/home.mccleary/sv572/palmer_scratch/qwen_data_10k/formatted_dataset.json \
    --image_folder /vast/palmer/home.mccleary/sv572/palmer_scratch/qwen_data_10k/images \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm True \
    --tune_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output/Qwen7B_UnfreezeVisionTower_20epochs \
    --num_train_epochs 20 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((256 * 28 * 28)) \
    --image_max_pixels $((1280 * 28 * 28)) \
    --learning_rate 2e-4 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 4