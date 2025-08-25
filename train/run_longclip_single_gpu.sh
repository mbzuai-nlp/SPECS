#!/usr/bin/env bash
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
set -x

# Set CUDA device if you have multiple GPUs
export CUDA_VISIBLE_DEVICES=0

# Set environment variables for distributed training
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# Set Python path to include the Long-CLIP directory
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

# Run with a single GPU
python -u train.py "$@" 