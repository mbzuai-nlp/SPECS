#!/bin/bash
#SBATCH -N 1
#SBATCH -A EUHPC_D12_071
#SBATCH -p boost_usr_prod
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=ARO
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1  # Change to 2 GPUs
#SBATCH --output=./slurm/aro_%j.log

# Load necessary modules or environments if required
# module load python

export PYTHONPATH=$(pwd)

HF_DATASETS_OFFLINE=1 
HF_HUB_OFFLINE=1

# Call the Python script with `accelerate launch` for multi-GPU compute
python /workspace/clip-fine-cap/evaluation/extrinsic/scripts/evaluate_aro.py\
    --model_path $MODEL_PATH