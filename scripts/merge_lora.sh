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
#SBATCH --output logs/log_qwen2_merge_weights_%J.log                # Output log file
#SBATCH --job-name=qwen2_merge_weights

date;hostname;pwd

module load miniconda

conda activate /vast/palmer/home.mccleary/sv572/project/.conda/envs/qwen2
cd /home/sv572/project/BrainLMv2-dev/Qwen2-VL-Finetune/

MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"

export PYTHONPATH=src:$PYTHONPATH

python src/merge_lora_weights.py \
    --model-path /vast/palmer/home.mccleary/sv572/project/BrainLMv2-dev/Qwen2-VL-Finetune/output/lora_vision_3B \
    --model-base $MODEL_NAME  \
    --save-model-path /vast/palmer/home.mccleary/sv572/project/BrainLMv2-dev/Qwen2-VL-Finetune/output/lora_vision_3B_merge \
    --safe-serialization