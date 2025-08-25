#!/bin/bash
#SBATCH -N 1
#SBATCH -A EUHPC_D12_071
#SBATCH -p boost_usr_prod
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=classification
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:2
#SBATCH --output=./slurm/classification_%j.log

export PYTHONPATH=$(pwd)

HF_DATASETS_OFFLINE=1 
HF_HUB_OFFLINE=1

# Call a unified Python script for evaluation
accelerate launch --num_processes=2 --multi_gpu evaluation/extrinsic/scripts/evaluate_classification.py \
    --model_name_or_path $MODEL_PATH \
    --tokenizer_name $WORK/data/HF/clip_tokenizer.hf \
    --image_processor_name $WORK/data/HF/clip_processor.hf \
    --local_files_only