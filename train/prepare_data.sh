#!/usr/bin/env bash
set -x

# 配置 - 请根据实际情况修改
INPUT_JSON="/workspace/ShareGPT4V/data/share-captioner_coco_lcs_sam_1246k_1107.json"  # ShareGPT4V JSON 文件路径
IMAGE_BASE_PATH="/workspace/ShareGPT4V/data"  # 图像根目录
OUTPUT_PATH="/workspace/ShareGPT4V/data/sharegpt4v_longclipdetail.hf"  # 输出数据集路径
TOKENIZER_PATH="openai/clip-vit-base-patch32"  # CLIP 分词器路径

# 可选参数
MAX_TOKEN_LENGTH=248  # 最大token长度
SHUFFLE_PORTION=1  # 用于创建负样本的数据集比例
# MAX_SAMPLES=1000  # 用于调试的数据集大小限制，默认不限制

# 设置Python路径包含Long-CLIP目录
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

# 运行数据准备脚本
python prepare_data.py \
    --input_json "$INPUT_JSON" \
    --image_base_path "$IMAGE_BASE_PATH" \
    --output_path "$OUTPUT_PATH" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --max_token_length "$MAX_TOKEN_LENGTH" \
    --shuffle_first_portion "$SHUFFLE_PORTION" \
    --seed 42
    # --max_samples "$MAX_SAMPLES"  # 取消注释以限制处理的样本数量 