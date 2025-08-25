###########  Code to test OpenCLIP models like LAION-400M for SCPP evaluation  ###
import argparse
from PIL import Image
import numpy as np
import torch, pickle, os, json, sys
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Add project root to Python path
current_file = Path(__file__).resolve()
project_root = current_file.parents[3]  # Go up 3 levels to reach project root
sys.path.append(str(project_root))

# Add additional module paths
sys.path.append("/workspace/clip-fine-cap")
sys.path.append("/workspace/Long-CLIP")
sys.path.append("/workspace/LaCLIP")  # 添加LaCLIP路径
sys.path.append("/workspace/DCI")     # 添加DCI路径

# 导入open_clip
try:
    import open_clip
except ImportError:
    print("Warning: open_clip not found. OpenCLIP model evaluation may not work.")

# 修正图像路径，确保以"/"结尾
img_path = "/workspace/vision-language-models-are-bows/~/.cache/coco/2014/val2014/"
data_path = "/workspace/clip-fine-cap/scpp/data"  #'path to folder with caption files'
fnames = os.listdir(data_path)
image_size = 224
device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SCPP with OpenCLIP models")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the OpenCLIP model checkpoint",
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="ViT-B-32",
        help="Base model architecture for OpenCLIP models (e.g., ViT-B-32)",
    )
    parser.add_argument(
        "--standard_openclip",
        action="store_true",
        help="Use standard OpenCLIP model instead of loading from checkpoint",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="laion400m_e32",
        help="Name of the standard OpenCLIP model to use (if standard_openclip is True)",
    )
    return parser.parse_args()


def load_openclip_model(model_path, base_model_name="ViT-B-32", standard_openclip=False, model_name="laion400m_e32"):
    """加载OpenCLIP兼容的模型，例如LAION-400M"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if standard_openclip:
        # 使用OpenCLIP内置的预训练模型
        print(f"Loading standard OpenCLIP model: {model_name}")
        model, _, preprocess = open_clip.create_model_and_transforms(
            base_model_name, 
            pretrained=model_name,
            device=device
        )
    else:
        # 从指定路径加载模型
        try:
            print(f"Loading OpenCLIP model from: {model_path}")
            model, _, preprocess = open_clip.create_model_and_transforms(
                base_model_name, 
                pretrained=model_path,
                device=device
            )
        except Exception as e:
            print(f"Error loading model from path: {e}")
            print("Falling back to standard OpenCLIP model")
            model, _, preprocess = open_clip.create_model_and_transforms(
                base_model_name, 
                pretrained="laion400m_e32",
                device=device
            )
    
    # 设置为评估模式
    model = model.eval()
    
    # 获取tokenizer
    tokenizer = open_clip.get_tokenizer(base_model_name)
    
    return model, preprocess, tokenizer


def run_scpp_evals(model, image_processor, tokenizer, device):
    all_results = {}
    
    for fname in fnames:
        print(
            "=======================================================================",
            flush=True,
        )
        print(
            "=======================================",
            fname,
            "=====================",
            flush=True,
        )
        json_path = os.path.join(data_path, fname)
        total = 0
        correct_img_p1 = 0
        correct_img_p2 = 0

        correct_full = 0  ###  the main task: P1 and P2 closer to Image than Negative
        correct_text = 0

        f = open(json_path)
        data = json.load(f)

        for line in data:
            p1 = line["caption"]
            ref = line["negative_caption"]
            p2 = line["caption2"]  # discard = fp[6]
            img_fname = line["filename"]
            
            # 修正图像文件名格式为COCO_val2014_XXXXXXXXXXXX.jpg
            # 检查文件名是否已经是COCO格式
            if not img_fname.startswith("COCO_val2014_"):
                # 如果只是数字，则转换为COCO格式
                if img_fname.isdigit() or (img_fname.startswith("0") and img_fname.replace("0", "").isdigit()):
                    img_id = img_fname.lstrip("0")  # 移除前导0
                    img_fname = f"COCO_val2014_{int(img_id):012d}.jpg"
                elif not img_fname.endswith(".jpg"):
                    img_fname = f"{img_fname}.jpg"
            
            # 首先尝试直接加载图像
            ipath = os.path.join(img_path, img_fname)
            
            try:
                image = Image.open(ipath).convert("RGB")
            except FileNotFoundError:
                # 尝试COCO格式的文件名
                if not img_fname.startswith("COCO_val2014_"):
                    coco_path = os.path.join(img_path, f"COCO_val2014_{img_fname}")
                    try:
                        image = Image.open(coco_path).convert("RGB")
                    except FileNotFoundError:
                        print(f"Image not found: {img_fname}, skipping")
                        continue
                else:
                    # 如果已经是COCO格式但仍找不到，跳过该图像
                    print(f"Image not found: {img_fname}, skipping")
                    continue
            
            # 处理图像
            img_processed = image_processor(image).unsqueeze(0).to(device)
            
            # 处理文本
            p1_tokens = tokenizer([p1]).to(device)
            p2_tokens = tokenizer([p2]).to(device)
            neg_tokens = tokenizer([ref]).to(device)
            
            # 获取特征
            with torch.no_grad():
                img_feats = model.encode_image(img_processed)
                p1_feats = model.encode_text(p1_tokens)
                p2_feats = model.encode_text(p2_tokens)
                neg_feats = model.encode_text(neg_tokens)
            
            # 归一化特征
            img_feats = F.normalize(img_feats, dim=-1)
            p1_feats = F.normalize(p1_feats, dim=-1)
            p2_feats = F.normalize(p2_feats, dim=-1)
            neg_feats = F.normalize(neg_feats, dim=-1)

            # 计算余弦相似度
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            cos_p1 = cos(img_feats, p1_feats)  # 图像与P1的相似度
            cos_p2 = cos(img_feats, p2_feats)  # 图像与P2的相似度
            cos_neg = cos(img_feats, neg_feats)  # 图像与负样本的相似度
            cos_p1p2 = cos(p1_feats, p2_feats)  # P1与P2的相似度
            cos_p1_neg = cos(p1_feats, neg_feats)  # P1与负样本的相似度
            cos_p2_neg = cos(p2_feats, neg_feats)  # P2与负样本的相似度

            total += 1

            # 评估不同指标
            if cos_p1 > cos_neg and cos_p2 > cos_neg:
                correct_full += 1
            if cos_p1 > cos_neg:
                correct_img_p1 += 1
            if cos_p2 > cos_neg:
                correct_img_p2 += 1
            if cos_p1p2 > cos_p1_neg and cos_p1p2 > cos_p2_neg:
                correct_text += 1

        # 打印评估结果
        print(f"====== evaluation results ======", flush=True)
        ave_score = float(correct_full) / float(total) if total > 0 else 0
        print(f"Accuracy image-to-text task: {ave_score * 100:.2f}%", flush=True)
        ave_score_orig_p1 = float(correct_img_p1) / float(total) if total > 0 else 0
        print(f"Accuracy Image-P1-Neg: {ave_score_orig_p1 * 100:.2f}%", flush=True)
        ave_score_orig_p2 = float(correct_img_p2) / float(total) if total > 0 else 0
        print(f"Accuracy Image-P2-Neg: {ave_score_orig_p2 * 100:.2f}%", flush=True)
        ave_score_txt = float(correct_text) / float(total) if total > 0 else 0
        print(f"Accuracy text-only task: {ave_score_txt * 100:.2f}%", flush=True)
        
        # 存储这个数据集的结果
        all_results[fname] = {
            "dataset": fname,
            "full_accuracy": ave_score * 100,
            "p1_accuracy": ave_score_orig_p1 * 100,
            "p2_accuracy": ave_score_orig_p2 * 100,
            "text_accuracy": ave_score_txt * 100,
            "total_samples": total
        }
    
    # 返回所有数据集的结果
    return all_results


def main():
    args = parse_args()
    
    # 加载OpenCLIP模型
    model, preprocess, tokenizer = load_openclip_model(
        model_path=args.model_path,
        base_model_name=args.base_model_name,
        standard_openclip=args.standard_openclip,
        model_name=args.model_name
    )
    
    # 运行SCPP评估
    all_results = run_scpp_evals(model, preprocess, tokenizer, device)
    
    # 打印总体摘要
    print("\n===== 总体评估摘要 =====")
    print(f"Model: {args.model_path}")
    print(f"Base architecture: {args.base_model_name}")
    print("-------------------------")
    
    # 计算平均指标
    avg_full = sum(result["full_accuracy"] for result in all_results.values()) / len(all_results)
    avg_p1 = sum(result["p1_accuracy"] for result in all_results.values()) / len(all_results)
    avg_p2 = sum(result["p2_accuracy"] for result in all_results.values()) / len(all_results)
    avg_text = sum(result["text_accuracy"] for result in all_results.values()) / len(all_results)
    total_samples = sum(result["total_samples"] for result in all_results.values())
    
    # 打印每个数据集的结果
    for dataset_name, result in all_results.items():
        print(f"Dataset: {dataset_name}")
        print(f"  Full accuracy: {result['full_accuracy']:.2f}%")
        print(f"  P1 accuracy: {result['p1_accuracy']:.2f}%")
        print(f"  P2 accuracy: {result['p2_accuracy']:.2f}%")
        print(f"  Text accuracy: {result['text_accuracy']:.2f}%")
        print(f"  Samples: {result['total_samples']}")
        print("-------------------------")
    
    # 打印平均指标
    print("Average metrics across all datasets:")
    print(f"  Full accuracy: {avg_full:.2f}%")
    print(f"  P1 accuracy: {avg_p1:.2f}%")
    print(f"  P2 accuracy: {avg_p2:.2f}%")
    print(f"  Text accuracy: {avg_text:.2f}%")
    print(f"  Total samples: {total_samples}")
    print("=========================")


if __name__ == "__main__":
    main() 