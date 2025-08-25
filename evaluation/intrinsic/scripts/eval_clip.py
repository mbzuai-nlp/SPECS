#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import sys
import os

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.abspath(os.path.join(current_dir, "../models"))
metric_dir = os.path.abspath(os.path.join(current_dir, ".."))
utils_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(models_dir)
sys.path.append(metric_dir)
sys.path.append(utils_dir)

from datasets import load_from_disk, Image
from tqdm import tqdm

from transformers import HfArgumentParser, CLIPImageProcessor, AutoTokenizer, CLIPModel
from peft import PeftModel
from accelerate import Accelerator
from accelerate.utils import gather_object

# 修改导入路径 - 使用绝对导入路径
from scores import CLIPScore
from metric import compute_increments
from utils import cosine_similarity



@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="openai/clip-vit-base-patch32",
        metadata={"help": "Model used to compute similarity"},
    )
    model_version: Optional[str] = field(
        default="",
        metadata={"help": "version of clip: pretrained, clip, cliprec, clipdetails"},
    )
    tokenizer_name: Optional[str] = field(
        default="",
        metadata={"help": "Tokenizer used to compute similarity"},
    )
    image_processor_name: Optional[str] = field(
        default="",
        metadata={"help": "Image Processor used to compute similarity"},
    )
    clip_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "clip version to instantiate cliprec"},
    )
    decoder_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "opt version to instantiate cliprec"},
    )
    postfix: Optional[str] = field(
        default="",
        metadata={"help": "postfix for output file"},
    )
    lora: Optional[bool] = field(
        default=True,
        metadata={"help": "If use lora to load the model"},
    )
    custom_model_path: Optional[str] = field(
        default="",
        metadata={"help": "Path to a custom fine-tuned model safetensors file"},
    )
    base_model_for_custom: Optional[str] = field(
        default="openai/clip-vit-base-patch32",
        metadata={"help": "Base model to use with custom fine-tuned weights"},
    )


@dataclass
class DataTrainingArguments:
    data_dir: Optional[Path] = field(
        default=None, metadata={"help": "The data directory containing input files."}
    )
    data_split: Optional[str] = field(
        default=None, metadata={"help": "The split of the dataset to use."}
    )
    image_column: Optional[str] = field(
        default="image",
        metadata={
            "help": "The name of the column in the datasets containing the full image file paths."
        },
    )
    text_column: Optional[str] = field(
        default="captions",
        metadata={
            "help": "The name of the column in the datasets containing the image captions."
        },
    )
    neg_text_column: Optional[str] = field(
        default="neg_captions",
        metadata={"help": "Name of column in dataset with negative captions"},
    )
    output_dir: Optional[Path] = field(
        default=Path("./results/"),
        metadata={"help": "The directory where to save the results."},
    )
    save_file: Optional[bool] = field(
        default=True,
        metadata={"help": "If save or not the file"},
    )


def main():
    # 解析命令行参数
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    # 检查是否使用自定义模型
    if model_args.custom_model_path:
        print(f"使用自定义模型: {model_args.custom_model_path}")
        # 加载基础模型
        base_model = CLIPModel.from_pretrained(model_args.base_model_for_custom)
        # 加载自定义权重
        from safetensors.torch import load_file
        import torch
        
        # 加载自定义模型权重
        custom_weights = load_file(model_args.custom_model_path)
        
        # 将权重加载到模型中
        missing_keys, unexpected_keys = base_model.load_state_dict(custom_weights, strict=False)
        if missing_keys:
            print(f"警告: 模型中缺少以下键: {missing_keys}")
        if unexpected_keys:
            print(f"警告: 权重文件中有未使用的键: {unexpected_keys}")
        
        # 创建CLIPScore实例
        model = CLIPScore(
            model_name_or_path=model_args.base_model_for_custom,
            version="pretrained",  # 使用预训练版本的处理逻辑
        )
        # 替换模型
        model.model = base_model
    else:
        # 初始化模型的 scorer
        if model_args.tokenizer_name and model_args.image_processor_name:
            # 如果显式指定了 tokenizer 和 processor，就从对应路径加载
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.tokenizer_name,
                # local_files_only=True,  # 如果你已经在本地缓存模型，可以打开；否则注释掉
            )
            processor = CLIPImageProcessor.from_pretrained(
                model_args.image_processor_name,
                # local_files_only=True,
            )

            model = CLIPScore(
                model_name_or_path=model_args.model_name_or_path,
                version=model_args.model_version,
                tokenizer=tokenizer,
                processor=processor,
            )

            # LoRA 逻辑
            if model_args.lora:
                # 下面这个路径是示例，需要你自己改成正确路径或注释掉
                clip = CLIPModel.from_pretrained(
                    "/leonardo_work/EUHPC_D12_071/data/HF/clip-base",
                    # local_files_only=True,
                )
                # 假设你的 LoRA 权重保存在 model_name_or_path 里
                model.model = PeftModel.from_pretrained(clip, model_args.model_name_or_path)
            else:
                model.model = CLIPModel.from_pretrained(
                    model_args.model_name_or_path,
                    # local_files_only=True,
                )

        else:
            # 如果没指定 tokenizer_name/image_processor_name，就在 CLIPScore 内部自定义加载
            model = CLIPScore(
                model_name_or_path=model_args.model_name_or_path,
                version=model_args.model_version,
                clip_model_name_or_path=model_args.clip_model_name_or_path,
                decoder_model_name_or_path=model_args.decoder_model_name_or_path,
            )

    # 初始化 accelerate
    accelerator = Accelerator()
    model.to(accelerator.device)  # 使用正确的设备标识符
    device = accelerator.device

    # 加载数据集
    dataset_name = data_args.data_dir.stem
    dataset = load_from_disk(str(data_args.data_dir))

    if data_args.data_split is not None:
        if data_args.data_split not in dataset:
            raise ValueError(f"Dataset split {data_args.data_split} does not exist")
        dataset = dataset[data_args.data_split]

    # 将指定列强制转换为 Image 类型
    dataset = dataset.cast_column(data_args.image_column, Image())

    # 找到 data_args.text_column 最长的 caption 数
    max_split = max(
        dataset.map(lambda sample: {"s": len(sample[data_args.text_column])})["s"]
    )

    # 将数据集按进程切分
    with accelerator.split_between_processes(list(range(len(dataset)))) as dataset_indices:
        ds = dataset.select(dataset_indices)

        # results shape: [num_samples, max_split, 2]
        # 其中 2 表示 (正向相似度, 负向相似度)
        results = np.zeros((len(ds), max_split, 2))

        for r, sample in enumerate(tqdm(ds)):
            # 获取正负文本嵌入
            text_embedding = model.embed_text(sample[data_args.text_column], device)
            neg_text_embedding = model.embed_text(sample[data_args.neg_text_column], device)
            # 获取图像嵌入
            image_embedding = model.embed_image(sample[data_args.image_column], device)
            # 计算正负相似度
            similarity = cosine_similarity(text_embedding, image_embedding)
            neg_similarity = cosine_similarity(neg_text_embedding, image_embedding)

            # 填充结果矩阵
            for c, (sim, neg_sim) in enumerate(zip(similarity, neg_similarity)):
                results[r][c][0] = sim[0]
                results[r][c][1] = neg_sim[0]

    # gather 分布式结果到主进程
    results_gathered = gather_object(results)
    if isinstance(results_gathered, list):
        results_gathered = np.concatenate(results_gathered, axis=0)

    # 主进程保存结果
    if accelerator.is_main_process:
        data_args.output_dir.mkdir(parents=True, exist_ok=True)

        if data_args.data_split is not None:
            output_file = data_args.output_dir / f"{dataset_name}_{data_args.data_split}{model_args.postfix}.npy"
        else:
            output_file = data_args.output_dir / f"{dataset_name}{model_args.postfix}.npy"

        if data_args.save_file:
            np.save(output_file, results_gathered)
            print(f"Results saved in {output_file}")

        # 计算并打印指标
        pos, neg = compute_increments(results_gathered)
        print(f"Results for {dataset_name}")
        print(f"Pos: {np.mean(pos):.4f}  Neg: {np.mean(neg):.4f}")


if __name__ == "__main__":
    main()
