from datasets import load_dataset, load_from_disk
import random
from transformers import CLIPTokenizer
import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from preprocessing_utils import postprocess_text, process_text
from transformers import AutoTokenizer

# ============================================================
# USER CONFIGURATION - PLEASE MODIFY THESE PATHS AS NEEDED
# ============================================================
# Base directory containing ShareGPT4V data
BASE_DATA_PATH = "/ShareGPT4V/data"  # MODIFY THIS PATH

# Input JSON file path
INPUT_JSON_FILE = "/ShareGPT4V/data/share-captioner_coco_lcs_sam_1246k_1107.json"  # MODIFY THIS PATH

# Processed dataset output path
PROCESSED_OUTPUT_PATH = "/ShareGPT4V/data/processed_sharegpt4v.hf"  # MODIFY THIS PATH

# Final output path
FINAL_OUTPUT_PATH = "sharegpt4v_final_longclip_first_90p_shuffle.hf"  # MODIFY THIS PATH

# Local model path (for offline mode)
LOCAL_MODEL_PATH = "/workspace/models/clip-vit-base-patch32"  # MODIFY THIS PATH

# ============================================================

random.seed(42)
IMAGE_PATH = Path(BASE_DATA_PATH)
ONLINE = True

def check_sample_paths():
    print("\n=== Starting Path Check ===")
    dataset = load_dataset(
        "json",
        data_files=INPUT_JSON_FILE,
    )
    dataset = dataset["train"]
    
    path_stats = {}
    for example in dataset:
        prefix = example["image"].split('/')[0]
        path_stats[prefix] = path_stats.get(prefix, 0) + 1
    
    print("\nPath distribution:")
    for prefix, count in path_stats.items():
        print(f"{prefix}: {count} images")

    print("\nActual files in directories:")
    for prefix in path_stats.keys():
        dir_path = IMAGE_PATH / prefix / "images"
        if os.path.exists(dir_path):
            file_count = len([f for f in os.listdir(dir_path) if f.endswith('.jpg')])
            print(f"{prefix} directory: {file_count} files")

def main():
    
    check_sample_paths()
    
    dataset = load_dataset(
        "json",
        data_files=INPUT_JSON_FILE,
    )
    dataset = dataset["train"]
    print(f"Initial dataset size: {len(dataset)}")

    def preprocess_json(example):
        original_path = example["image"]
        if not original_path.startswith('/'):
            new_path = str(IMAGE_PATH / original_path)
        else:
            new_path = original_path
        example["image"] = new_path
        example["caption"] = example["conversations"][-1]["value"]
        return example

    print("Creating dataset from json file...")
    dataset = dataset.map(preprocess_json).remove_columns(["id", "conversations"])

    def filter_corrupt_images(examples):
        valid_images = []
        error_counts = {"not_found": 0, "open_error": 0}
        
        for image_file in examples["image"]:
            try:
                if not os.path.exists(image_file):
                    error_counts["not_found"] += 1
                    valid_images.append(False)
                    continue
                
                Image.open(image_file)
                valid_images.append(True)
            except Exception:
                error_counts["open_error"] += 1
                valid_images.append(False)
        return valid_images

    initial_len = len(dataset)
    print("Filtering non-existing images...")
    dataset = dataset.filter(filter_corrupt_images, batched=True, num_proc=32)
    final_len = len(dataset)
    print(f"Removed images: {initial_len - final_len}")

    print("Adding detail segmentation...")
    segments = []
    for item in tqdm(dataset):
        segments.append(postprocess_text(process_text(item["caption"])))
    dataset = dataset.add_column("segmented_caption", segments)

    def remove_empty_segments(example):
        example["segmented_caption"] = [
            caption for caption in example["segmented_caption"] if caption
        ]
        return example

    dataset = dataset.map(remove_empty_segments)
    
    dataset.save_to_disk(PROCESSED_OUTPUT_PATH)
    print(f"Processed dataset saved to {PROCESSED_OUTPUT_PATH}")

    
    # Load the saved processed dataset
    print("Loading processed dataset...")
    dataset = load_from_disk(PROCESSED_OUTPUT_PATH)
    
    # Split dataset for shuffling strategy - first 90% shuffled, last 10% unshuffled
    dataset_size = len(dataset)
    split_index = int(dataset_size * 0.90)  # 90% split point
    print(f"Dataset size: {dataset_size}, Split at index: {split_index}")
    
    # Add a column to identify which part of the dataset each example belongs to
    indices = list(range(dataset_size))
    dataset = dataset.add_column("dataset_part", ["shuffle" if i < split_index else "no_shuffle" for i in indices])

    # 1. First create neg_details
    print("Adding negative splits...")
    
    def shuffle_words(phrase):
        words = phrase.split()
        random.shuffle(words)
        return " ".join(words)
    
    def obtain_neg_splits(example):
        neg_details = []
        for i in range(len(example["segmented_caption"]) - 1):
            while True:
                sample = random.choice(dataset)
                if sample["image"] != example["image"]:
                    if i >= len(sample["segmented_caption"]):
                        i = len(sample["segmented_caption"]) - 1
                    break
            
            # First 90% gets shuffled captions, last 10% gets original
            if example["dataset_part"] == "shuffle":
                neg_details.append(shuffle_words(sample["segmented_caption"][i].strip()))
            else:
                neg_details.append(sample["segmented_caption"][i].strip())
            
        return {"neg_details": neg_details}

    # Ensure this step is executed first
    dataset = dataset.map(obtain_neg_splits)
    print("Negative splits added successfully")

    # 2. Then load the tokenizer
    print("Loading tokenizer...")
    try:
        if ONLINE:
            # Online mode, allow downloading from huggingface
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        else:
            # Offline mode, try to load from local or use CLIPTokenizer
            try:
                # Try to load from local path
                if os.path.exists(LOCAL_MODEL_PATH):
                    tokenizer = AutoTokenizer.from_pretrained(
                        LOCAL_MODEL_PATH,
                        local_files_only=True
                    )
                else:
                    # If local path doesn't exist, use CLIPTokenizer directly
                    tokenizer = CLIPTokenizer.from_pretrained(
                        "openai/clip-vit-base-patch32",
                        local_files_only=False
                    )
            except Exception as inner_e:
                print(f"Failed to load local model: {inner_e}")
                raise
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        raise

    # 3. Finally process captions
    def join_segmented_captions(example):
        segmented_caption = example["segmented_caption"]
        neg_details = example["neg_details"]  # This field should exist now

        merged_captions = [segmented_caption[0].strip()]
        neg_captions = []
        for caption, neg_caption in zip(segmented_caption[1:], neg_details):
            merged_caption = f"{merged_captions[-1]} {caption.strip()}"
            neg_caption = f"{merged_captions[-1]} {neg_caption.strip()}"
            tokens = tokenizer([merged_caption, neg_caption])["input_ids"]
            if len(tokens[0]) >248 or len(tokens[1]) > 248:
                break
            merged_captions.append(merged_caption)
            neg_captions.append(neg_caption)
        example["captions"] = merged_captions
        example["neg_captions"] = neg_captions
        return example

    # Add progress bar to monitor processing progress
    print("Processing captions...")
    dataset = dataset.map(
        join_segmented_captions,
        desc="Processing captions",
        remove_columns=["segmented_caption", "neg_details", "dataset_part"]
    )
    dataset = dataset.filter(lambda example: len(example["neg_captions"]) > 0)
    
    dataset.save_to_disk(FINAL_OUTPUT_PATH)
    print(f"Final processed dataset saved to {FINAL_OUTPUT_PATH}")

if __name__ == "__main__":
    main()