#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys
from pathlib import Path

# 将项目根路径添加到 Python 路径
current_file = Path(__file__).resolve()
project_root = current_file.parents[3]  # 向上 3 级到达项目根目录
sys.path.append(str(project_root))

# 添加额外模块路径 - 修改顺序，让 Long-CLIP 在前面
sys.path.append("/workspace/Long-CLIP")
sys.path.append("/workspace/clip-fine-cap")

import pandas as pd
import nltk
from torch.utils.data import DataLoader

# 直接从 Long-CLIP 的绝对路径导入 longclip
sys.path.insert(0, "/workspace/Long-CLIP")
from model import longclip

from aro.dataset_zoo import VG_Relation, VG_Attribution, COCO_Order, Flickr30k_Order
from aro.clip_aro_wrap import AROtoLongCLIPWrap

# 确保 nltk 的 punkt tokenizer 可用
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# 缓存目录配置
ARO_DIR = "/workspace/vision-language-models-are-bows/~/.cache/prerelease_bow"
COCO_DIR = "/workspace/vision-language-models-are-bows/~/.cache/coco/2014"
FLICKR_DIR = "/workspace/vision-language-models-are-bows/~/.cache/flickr30k"


def run_aro_evals(
    model,
    image_processor,
    tokenizer,
    device,
):
    # 加载 ARO 数据集
    vgr_dataset = VG_Relation(image_preprocess=image_processor, root_dir=ARO_DIR)
    vga_dataset = VG_Attribution(image_preprocess=image_processor, root_dir=ARO_DIR)
    coco_order_dataset = COCO_Order(image_preprocess=image_processor, root_dir=COCO_DIR)
    flickr_order_dataset = Flickr30k_Order(
        image_preprocess=image_processor,
        split="test",
        root_dir=FLICKR_DIR,
    )

    # 建立 DataLoader
    vgr_loader = DataLoader(vgr_dataset, batch_size=16, shuffle=False)
    vga_loader = DataLoader(vga_dataset, batch_size=16, shuffle=False)
    coco_loader = DataLoader(coco_order_dataset, batch_size=16, shuffle=False)
    flickr_loader = DataLoader(flickr_order_dataset, batch_size=16, shuffle=False)

    # 将模型包裹成 ARO 评估接口
    aro_wrap = AROtoLongCLIPWrap(model, tokenizer, device)

    # VG-Relation 评估
    vgr_scores = aro_wrap.get_retrieval_scores_batched(vgr_loader)
    vgr_records = vgr_dataset.evaluate_scores(vgr_scores)
    # 排除对称关系
    symmetric = [
        "adjusting", "attached to", "between", "bigger than", "biting", "boarding", "brushing",
        "chewing", "cleaning", "climbing", "close to", "coming from", "coming out of",
        "contain", "crossing", "dragging", "draped over", "drinking", "drinking from",
        "driving", "driving down", "driving on", "eating from", "eating in", "enclosing",
        "exiting", "facing", "filled with", "floating in", "floating on", "flying",
        "flying above", "flying in", "flying over", "flying through", "full of",
        "going down", "going into", "going through", "grazing in", "growing in",
        "growing on", "guiding", "hanging from", "hanging in", "hanging off",
        "hanging over", "higher than", "holding onto", "hugging", "in between",
        "jumping off", "jumping on", "jumping over", "kept in", "larger than",
        "leading", "leaning over", "leaving", "licking", "longer than",
        "looking in", "looking into", "looking out", "looking over", "looking through",
        "lying next to", "lying on top of", "making", "mixed with", "mounted on",
        "moving", "on the back of", "on the edge of", "on the front of",
        "on the other side of", "opening", "painted on", "parked at", "parked beside",
        "parked by", "parked in", "parked in front of", "parked near",
        "parked next to", "perched on", "petting", "piled on", "playing",
        "playing in", "playing on", "playing with", "pouring", "reaching for",
        "reading", "reflected on", "riding on", "running in", "running on",
        "running through", "seen through", "sitting behind", "sitting beside",
        "sitting by", "sitting in front of", "sitting near", "sitting next to",
        "sitting under", "skiing down", "skiing on", "sleeping in", "sleeping on",
        "smiling at", "sniffing", "splashing", "sprinkled on", "stacked on",
        "standing against", "standing around", "standing behind", "standing beside",
        "standing in front of", "standing near", "standing next to", "staring at",
        "stuck in", "surrounding", "swimming in", "swinging", "talking to",
        "topped with", "touching", "traveling down", "traveling on", "tying",
        "typing on", "underneath", "wading in", "waiting for", "walking across",
        "walking by", "walking down", "walking next to", "walking through",
        "working in", "working on", "worn on", "wrapped around", "wrapped in",
        "by", "of", "near", "next to", "with", "beside", "on the side of",
        "around",
    ]
    df = pd.DataFrame(vgr_records)
    df = df[~df.Relation.isin(symmetric)]
    vgr_metric = df.Accuracy.mean()
    print(f"VG-Relation Macro Accuracy: {vgr_metric}")

    # VG-Attribution 评估
    vga_scores = aro_wrap.get_retrieval_scores_batched(vga_loader)
    vga_records = vga_dataset.evaluate_scores(vga_scores)
    df = pd.DataFrame(vga_records)
    vga_metric = df.Accuracy.mean()
    print(f"VG-Attribution Macro Accuracy: {vga_metric}")

    # COCO Order 评估
    coco_scores = aro_wrap.get_retrieval_scores_batched(coco_loader)
    coco_records = coco_order_dataset.evaluate_scores(coco_scores)
    df = pd.DataFrame(coco_records)
    coco_metric = df["Precision@1"].mean()
    print(f"COCO Precision@1: {coco_metric}")

    # Flickr30k Order 评估
    flickr_scores = aro_wrap.get_retrieval_scores_batched(flickr_loader)
    flickr_records = flickr_order_dataset.evaluate_scores(flickr_scores)
    df = pd.DataFrame(flickr_records)
    flickr_metric = df["Precision@1"].mean()
    print(f"Flickr Precision@1: {flickr_metric}")

    return {
        "vgr_metric": vgr_metric,
        "vga_metric": vga_metric,
        "coco_metric": coco_metric,
        "flickr_metric": flickr_metric,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ARO classification with custom Long-CLIP model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the custom Long-CLIP model checkpoint",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 加载自定义 Long-CLIP 模型和对应的图像预处理器
    clip_model, image_processor = longclip.load(args.model_path)
    clip_model.to("cuda")

    # 使用 Long-CLIP 的 tokenizer 而不是 HuggingFace 的 tokenizer
    # Long-CLIP 模型需要使用其自己的 tokenizer
    from model.longclip import tokenize as clip_tokenize
    
    class LongCLIPTokenizer:
        def __init__(self):
            pass
            
        def __call__(self, texts, truncate=False, **kwargs):
            if isinstance(texts, str):
                texts = [texts]
            return clip_tokenize(texts)
    
    # 创建 tokenizer
    tokenizer = LongCLIPTokenizer()

    # 运行评估
    run_aro_evals(clip_model, image_processor, tokenizer, "cuda")
