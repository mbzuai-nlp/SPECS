###########  Code to test Hugging face multi-modal models  ###
import argparse
from PIL import Image
import numpy as np
import torch, pickle, os, json, sys
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoProcessor, AlignModel
from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor, CLIPTokenizer
from pathlib import Path

# Add project root to Python path
current_file = Path(__file__).resolve()
project_root = current_file.parents[3]  # Go up 3 levels to reach project root
sys.path.append(str(project_root))

# Add additional module paths
sys.path.append("/workspace/clip-fine-cap")
sys.path.append("/workspace/Long-CLIP")

# 直接从Long-CLIP项目导入
sys.path.insert(0, "/workspace/Long-CLIP")
from model import longclip

# 修正图像路径，确保以"/"结尾
img_path = "/workspace/vision-language-models-are-bows/~/.cache/coco/2014/val2014/"
data_path = "/workspace/clip-fine-cap/scpp/data"  #'path to folder with caption files'
fnames = os.listdir(data_path)
image_size = 224
device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ARO classification")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="clip",
        choices=["clip", "longclip", "custom"],
        help="Type of model to load: clip, longclip, or custom",
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Load lora model using Peft",
    )

    args = parser.parse_args()
    return args


def run_scpp_on_lora(image_processor, tokenizer, base_clip_model, lora_weight_path):
    from peft import PeftModel

    loaded = PeftModel.from_pretrained(base_clip_model, lora_weight_path)
    loaded = loaded.merge_and_unload()
    loaded.to("cuda")
    run_scpp_evals(loaded, image_processor, tokenizer, device)


class LongCLIPTokenizer:
    def __init__(self):
        pass
        
    def __call__(self, texts, padding=True, return_tensors="pt", **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        tokens = longclip.tokenize(texts)
        if return_tensors == "pt":
            return {"input_ids": tokens}
        return tokens


def run_scpp_evals(model, image_processor, tokenizer, device):
    # For LongCLIP models, adapt the feature extraction functions
    is_longclip = hasattr(model, "encode_image") and hasattr(model, "encode_text")

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
                    image = Image.open(coco_path).convert("RGB")
                else:
                    # 如果已经是COCO格式但仍找不到，跳过该图像
                    continue
                    
            model.eval()

            # Extract features based on model type
            if is_longclip:
                # LongCLIP interface
                img_feats = model.encode_image(image_processor(image).unsqueeze(0).to(device))
                p1_feats = model.encode_text(longclip.tokenize([p1]).to(device))
                p2_feats = model.encode_text(longclip.tokenize([p2]).to(device))
                neg_feats = model.encode_text(longclip.tokenize([ref]).to(device))
            else:
                # Standard CLIP interface
                inputs = image_processor(images=image, return_tensors="pt").to(device)
                img_feats = model.get_image_features(**inputs)
                
                inputs = tokenizer(p1, padding=True, return_tensors="pt").to(device)
                p1_feats = model.get_text_features(**inputs)
                
                inputs = tokenizer(p2, padding=True, return_tensors="pt").to(device)
                p2_feats = model.get_text_features(**inputs)
                
                inputs = tokenizer(ref, padding=True, return_tensors="pt").to(device)
                neg_feats = model.get_text_features(**inputs)
            
            # Normalize features
            img_feats = F.normalize(img_feats, dim=-1)
            p1_feats = F.normalize(p1_feats, dim=-1)
            p2_feats = F.normalize(p2_feats, dim=-1)
            neg_feats = F.normalize(neg_feats, dim=-1)

            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            cos_p1 = cos(
                img_feats, p1_feats
            )  ###  cosine similarities between image and P1 (positive caption 1)
            cos_p2 = cos(
                img_feats, p2_feats
            )  ###  cosine similarities between image and P2 (positive caption 2)
            cos_neg = cos(
                img_feats, neg_feats
            )  ###  cosine similarities between image and Negative (negative caption)
            cos_p1p2 = cos(
                p1_feats, p2_feats
            )  ###  cosine similarities between P1 and P2 for text-only task
            cos_p1_neg = cos(
                p1_feats, neg_feats
            )  ###  cosine similarities between P1 and Negative for text-only task
            cos_p2_neg = cos(
                p2_feats, neg_feats
            )  ###  cosine similarities between P2 and Negative for text-only task

            total += 1

            if cos_p1 > cos_neg and cos_p2 > cos_neg:
                correct_full += 1
            if cos_p1 > cos_neg:
                correct_img_p1 += 1
            if cos_p2 > cos_neg:
                correct_img_p2 += 1
            if cos_p1p2 > cos_p1_neg and cos_p1p2 > cos_p2_neg:
                correct_text += 1

        print(f"====== evaluation results ======", flush=True)
        ave_score = float(correct_full) / float(total)
        print(f"Accuracy image-to-text task: {ave_score * 100}", flush=True)
        ave_score_orig_p1 = float(correct_img_p1) / float(total)
        print(f"Accuracy Image-P1-Neg: {ave_score_orig_p1 * 100}", flush=True)
        ave_score_orig_p2 = float(correct_img_p2) / float(total)
        print(f"Accuracy Image-P2-Neg: {ave_score_orig_p2 * 100}", flush=True)

        ave_score_txt = float(correct_text) / float(total)
        print(f"Accuracy text-only task: {ave_score_txt * 100}", flush=True)


def main():
    args = parse_args()
    
    if args.model_type == "clip":
        # Load standard CLIP model
        image_processor = CLIPImageProcessor.from_pretrained(
            "/leonardo_work/EUHPC_D12_071/data/HF/clip_processor.hf", local_files_only=True
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            "/leonardo_work/EUHPC_D12_071/data/HF/clip_tokenizer.hf", local_files_only=True
        )

        if args.lora:
            print(f"Using LoRA model from {args.model_path}")
            clip_model = CLIPModel.from_pretrained(
                "/leonardo_work/EUHPC_D12_071/data/HF/clip-base", local_files_only=True
            )
            run_scpp_on_lora(
                image_processor,
                tokenizer,
                clip_model,
                args.model_path,
            )
        else:
            print(f"Using finetuned CLIP model from {args.model_path}")
            loaded = CLIPModel.from_pretrained(args.model_path, local_files_only=True).to(
                device
            )
            run_scpp_evals(loaded, image_processor, tokenizer, device)
    
    elif args.model_type == "longclip":
        # Load LongCLIP model
        print(f"Using LongCLIP model from {args.model_path}")
        loaded, image_processor = longclip.load(args.model_path)
        loaded.to(device)
        tokenizer = LongCLIPTokenizer()
        run_scpp_evals(loaded, image_processor, tokenizer, device)
    
    elif args.model_type == "custom":
        # Load custom model implementation
        print(f"Using custom model from {args.model_path}")
        
        # 直接使用LongCLIP的load方法加载模型
        loaded, image_processor = longclip.load(args.model_path)
        loaded.to(device)
        
        # 创建针对custom模型的tokenizer
        tokenizer = LongCLIPTokenizer()
        
        run_scpp_evals(loaded, image_processor, tokenizer, device)


if __name__ == "__main__":
    main()
