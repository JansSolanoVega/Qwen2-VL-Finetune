#!/bin/bash
#SBATCH --partition bigmem
#SBATCH --nodes 1
#SBATCH --requeue
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu 256G
#SBATCH --job-name format_dataset
#SBATCH --output logs/format_dataset-%J.log

module load miniconda
conda activate unsloth_env
cd /home/sv572/project/BrainLMv2-dev/Qwen2-VL-Finetune
#\--subsample_size 100
DATASET=/gpfs/gibbs/pi/dijk/BrainLM_Datasets/UKB_Large_rsfMRI_and_tffMRI_Arrow_WithRegression_v3_with_metadata/train_ukbiobank

python format_dataset_qwen.py \
    --data_path $DATASET \
    --output_dir /home/sv572/palmer_scratch/qwen_data_10k \
    --images_dir /home/sv572/palmer_scratch/qwen_data_10k/images \
    --prompts_path /home/sv572/project/BrainLMv2-dev/Qwen2-VL-Finetune/prompts_age_only.json \
    --num_samples 10000 \
    --balance_iterations 0 