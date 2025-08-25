#!/bin/bash
#SBATCH -N 1
#SBATCH -A EUHPC_D12_071
#SBATCH -p boost_usr_prod
#SBATCH --job-name=LongClipDet
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:4
#SBATCH --output=      # Output and error log file

# Set environment variables
export PYTHONPATH=""
wandb offline

export HF_DATASETS_OFFLINE=1
export PHF_HUB_OFFLINE=1

export OUTPUT_DIR=${SLURM_JOB_ID}'-spec'
mkdir -p $OUTPUT_DIR

# Check available GPUs

# Run distributed training with torchrun
# If multiple GPUs are not available, use a single GPU with multiple processes
torchrun --nproc_per_node=8 train_spec.py \
    --model_path longclip.pt \
    --lambda_contrast 1.0 \
    --lambda_details 8 \
    --lambda_neg 0.8 \
    --epsilon 0\
    --beta 0 \
    --num_train_epochs 6 \
    --per_device_train_batch_size 100 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --weight_decay 1e-2 \
    --warmup_steps 40 \
    --dataset_name  #your dataset\
    --output_dir $OUTPUT_DIR \
    --logging_steps 10 \
    --save_steps 1000 \
    --num_workers 4 
 