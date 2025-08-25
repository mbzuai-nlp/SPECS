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

# 添加额外模块路径
sys.path.append("/workspace/clip-fine-cap")
sys.path.append("/workspace/Long-CLIP")
sys.path.append("/workspace/LaCLIP")  # 添加LaCLIP路径
sys.path.append("/workspace/DCI")     # 添加DCI路径

import pandas as pd
import nltk
import torch
from torch.utils.data import DataLoader

# 可能使用的模型加载方式
try:
    import open_clip
except ImportError:
    print("Warning: open_clip not found. LAION model evaluation may not work.")

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


def load_openclip_model(model_path, base_model_name="ViT-B-32"):
    """加载OpenCLIP兼容的模型，例如LAION-400M"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载模型和预处理函数
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            base_model_name, 
            pretrained=model_path,
            device=device
        )
    except Exception as e:
        print(f"Error loading with pretrained path: {e}")
        print("Trying to load model using standard OpenCLIP method...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            base_model_name, 
            pretrained="laion400m_e32",
            device=device
        )
    
    model = model.eval()
    
    # 创建一个与ARO评估兼容的模型包装
    class OpenCLIPModelWrapper:
        def __init__(self, model):
            self.model = model
            
        def encode_image(self, image_input):
            """
            编码图像
            如果输入是张量直接使用，如果是包含pixel_values的字典，提取其中的值
            """
            if isinstance(image_input, dict) and "pixel_values" in image_input:
                image_input = image_input["pixel_values"]
            return self.model.encode_image(image_input)
            
        def encode_text(self, text_input):
            """
            编码文本
            如果输入是TokenWrapper类型，提取其中的tokens
            """
            # 处理TokenWrapper对象
            if isinstance(text_input, TokenWrapper):
                text_input = text_input.tokens
            # 处理包含input_ids的字典
            elif isinstance(text_input, dict) and "input_ids" in text_input:
                text_input = text_input["input_ids"]
            return self.model.encode_text(text_input)

    # 获取tokenizer并创建包装
    tokenizer_fn = open_clip.get_tokenizer(base_model_name)
    
    class TokenWrapper:
        """包装tokenizer的输出，使其支持.to()方法和类似tensor的操作"""
        
        def __init__(self, tokens, device):
            self.tokens = tokens
            self.device = device
            
        def to(self, device=None):
            """
            实现.to()方法，使TokenWrapper可以被移动到特定设备
            如果tokens已经是tensor并支持.to()，则使用它的方法
            否则返回tokens本身
            """
            if device is None:
                device = self.device
            
            if hasattr(self.tokens, 'to') and callable(getattr(self.tokens, 'to')):
                self.tokens = self.tokens.to(device)
            return self
        
        def __getattr__(self, name):
            """转发所有未知属性和方法到内部的tokens对象"""
            if hasattr(self.tokens, name):
                return getattr(self.tokens, name)
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        # 以下方法使TokenWrapper在AROtoLongCLIPWrap.process_batch中能够像tensor一样工作
        def __len__(self):
            if hasattr(self.tokens, '__len__'):
                return len(self.tokens)
            return 0
        
        def __getitem__(self, key):
            if hasattr(self.tokens, '__getitem__'):
                return self.tokens[key]
            raise TypeError(f"'{self.__class__.__name__}' object is not subscriptable")
        
        def __iter__(self):
            if hasattr(self.tokens, '__iter__'):
                return iter(self.tokens)
            raise TypeError(f"'{self.__class__.__name__}' object is not iterable")

    class OpenCLIPTokenizerWrapper:
        def __init__(self, tokenizer_fn):
            self.tokenizer_fn = tokenizer_fn
            self.device = device
            
        def __call__(self, texts, truncate=False, padding=None, return_tensors=None, **kwargs):
            """
            OpenCLIP tokenizer wrapper，兼容ARO评估需要的接口
            参数:
                texts: 文本或文本列表
                truncate: ARO评估中使用的参数，但在OpenCLIP中不需要
                padding, return_tensors: 保持与HF接口兼容的参数
                **kwargs: 其他可能的参数
            """
            if isinstance(texts, str):
                texts = [texts]
            
            # OpenCLIP的tokenizer会自动处理截断，所以忽略truncate参数
            tokens = self.tokenizer_fn(texts)
            
            # 创建一个TokenWrapper对象，使其支持.to()方法
            return TokenWrapper(tokens, self.device)
    
    # 返回模型包装、处理器和tokenizer包装
    return OpenCLIPModelWrapper(model), preprocess, OpenCLIPTokenizerWrapper(tokenizer_fn)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ARO classification with custom model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="openclip",
        choices=["long-clip", "openclip"],
        help="Type of model (long-clip or openclip)",
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="ViT-B-32",
        help="Base model architecture for OpenCLIP models",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.model_type == "long-clip":
        # 加载原始的Long-CLIP模型
        from model import longclip
        clip_model, image_processor = longclip.load(args.model_path)
        clip_model.to("cuda")
        
        # 使用Long-CLIP的tokenizer
        from model.longclip import tokenize as clip_tokenize
        
        class LongCLIPTokenizer:
            def __init__(self):
                pass
                
            def __call__(self, texts):
                if isinstance(texts, str):
                    texts = [texts]
                return clip_tokenize(texts)
        
        tokenizer = LongCLIPTokenizer()
        
    elif args.model_type == "openclip":
        # 加载OpenCLIP兼容的模型 (例如LAION-400M)
        print(f"Loading OpenCLIP model from: {args.model_path}")
        clip_model, image_processor, tokenizer = load_openclip_model(
            args.model_path, 
            base_model_name=args.base_model_name
        )
    
    # 运行评估
    results = run_aro_evals(clip_model, image_processor, tokenizer, "cuda")
    
    # 打印摘要
    print("\n===== 评估摘要 =====")
    print(f"Model: {args.model_path}")
    print(f"Type: {args.model_type}")
    print(f"VG-Relation: {results['vgr_metric']:.4f}")
    print(f"VG-Attribution: {results['vga_metric']:.4f}")
    print(f"COCO Order: {results['coco_metric']:.4f}")
    print(f"Flickr Order: {results['flickr_metric']:.4f}")
    print("===================") 