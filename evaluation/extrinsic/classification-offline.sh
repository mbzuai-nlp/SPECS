#!/bin/bash
#SBATCH -N 1
#SBATCH -A EUHPC_D12_071
#SBATCH -p boost_usr_prod
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=Extrinsic
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:2

export PYTHONPATH=$(pwd)
export MODEL_PATH="../logs/1.0-2.0-1.0_dn_lr1e-3-ckpt-522"
#export MODEL_PATH="$WORK/dci_pick1/"

HF_DATASETS_OFFLINE=1 
HF_HUB_OFFLINE=1

#python evaluation/extrinsic/scripts/zeroshot.py \
accelerate launch --num_processes=2 --multi_gpu evaluation/extrinsic/scripts/zeroshot.py \
    --model_name_or_path $MODEL_PATH \
    --model_version clipdetails \
    --tokenizer_name $WORK/data/HF/clip_tokenizer.hf \
    --image_processor_name $WORK/data/HF/clip_processor.hf \
    --data_dir $WORK/data/eval/cifar10.hf \
    --data_split test \
    --classes_and_templates cifar10 \
    --load_from_hub False \
    --image_column img \
    --label_column label \
    --local_files_only True

accelerate launch --num_processes=2 --multi_gpu evaluation/extrinsic/scripts/zeroshot.py \
    --model_name_or_path $MODEL_PATH \
    --model_version clipdetails \
    --tokenizer_name $WORK/data/HF/clip_tokenizer.hf \
    --image_processor_name $WORK/data/HF/clip_processor.hf \
    --data_dir $WORK/data/eval/cifar100.hf \
    --data_split test \
    --classes_and_templates cifar100 \
    --load_from_hub False \
    --image_column img \
    --label_column fine_label \
    --local_files_only True

# accelerate launch --num_processes=2 --multi_gpu evaluation/extrinsic/scripts/zeroshot.py \
#     --model_name_or_path $MODEL_PATH \
#     --model_version clipdetails \
#     --tokenizer_name $WORK/data/HF/clip_tokenizer.hf \
#     --image_processor_name $WORK/data/HF/clip_processor.hf \
#     --data_dir $WORK/data/eval/imagenet.hf \
#     --classes_and_templates imagenet \
#     --load_from_hub False \
#     --image_column image \
#     --label_column label \
#     --local_files_only True
