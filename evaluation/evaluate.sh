#!/bin/bash

# Usage: run_experiments.sh -m <MODEL_PATH>

# Parse the argument for MODEL_PATH
while getopts "m:" opt; do
  case $opt in
    m)
      MODEL_PATH="$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

# Check if MODEL_PATH is provided
if [ -z "$MODEL_PATH" ]; then
  echo "Error: MODEL_PATH is required. Use -m <MODEL_PATH>."
  exit 1
fi

# Export MODEL_PATH so it's available in the scripts
export MODEL_PATH

# 1. ARO Evaluation
aro_job_id=$(sbatch evaluation/extrinsic/evaluate_aro.sh | awk '{print $4}')
echo "ARO job submitted with JobID: aro_$aro_job_id"

# 2. Retrieval Evaluation
retrieval_job_id=$(sbatch evaluation/extrinsic/evaluate_retrieval.sh | awk '{print $4}')
echo "Retrieval job submitted with JobID: retrieval_$retrieval_job_id"

# 3. Classification Evaluation
classification_job_id=$(sbatch evaluation/extrinsic/evaluate_classification.sh | awk '{print $4}')
echo "Classification job submitted with JobID: classification_$classification_job_id"

# 4. Intrinsic Evaluation
intrinsic_job_id=$(sbatch evaluation/intrinsic/run_intrinsic.sh | awk '{print $4}')
echo "Intrinsic job submitted with JobID: intrinsic_$intrinsic_job_id"

echo "All jobs submitted successfully."