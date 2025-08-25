#!/usr/bin/env bash
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
set -x

# Configuration
NUM_GPUS=4  # Change this to the number of GPUs you want to use
MASTER_PORT=$(( RANDOM % 50000 + 10000 ))

# Set Python path to include the Long-CLIP directory
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

# Run with multiple GPUs using torch.distributed.launch
python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    train.py \
    "$@" 