import json
import os
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import nltk
from datasets import Dataset, DatasetDict, concatenate_datasets
from preprocessing_utils import postprocess_text, process_text

from transformers import CLIPTokenizer

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

random.seed(42)

# =============================================================================
# CONFIGURATION SECTION - MODIFY THESE PATHS ACCORDING TO YOUR SETUP
# =============================================================================

# Path to your DCI dataset directory
# MODIFY THIS: Change to your actual DCI dataset path
DATASET_BASE = Path("/path/to/your/DCI/data/densely_captioned_images")

# Base path for images within the dataset
# MODIFY THIS: Change if your images are in a different subdirectory
BASE_IMAGES_PATH = Path("photos")

# Path to the splits.json file
# MODIFY THIS: Change if your splits.json is in a different location
SPLIT_FILE = DATASET_BASE / "splits.json"

# Output directory where the processed dataset will be saved
# MODIFY THIS: Change to your desired output path
OUTPUT_DIR = Path("/path/to/your/output/dci_train_248.hf")

# Tokenizer configuration
# MODIFY THIS: Set to False if you want to use local tokenizer, True for online
ONLINE = True
# MODIFY THIS: Change to your tokenizer path (local path if ONLINE=False)
TOKENIZER_PATH = "openai/clip-vit-base-patch32"  # HuggingFace model name

# Maximum token length for caption concatenation
# MODIFY THIS: Change according to your model's context length
MAX_LEN = 248

# =============================================================================
# END OF CONFIGURATION SECTION
# =============================================================================

def filter_corrupt_images(examples):
    """Filter out corrupted or unreadable images"""
    valid_images = []
    for image_file in examples["image"]:
        try:
            Image.open(image_file)
            valid_images.append(True)
        except Exception:
            valid_images.append(False)
    return valid_images


def remove_empty_segments(example):
    """Remove empty string segments from caption"""
    example["segmented_caption"] = [
        seg for seg in example["segmented_caption"] if seg.strip()
    ]
    return example


def main():
    # Validate paths before processing
    if not DATASET_BASE.exists():
        raise FileNotFoundError(f"Dataset base directory not found: {DATASET_BASE}")
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"Splits file not found: {SPLIT_FILE}")
    
    print(f"Dataset base: {DATASET_BASE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Max token length: {MAX_LEN}")
    
    # ---------- 1) Read splits.json and build original DatasetDict ----------
    with open(SPLIT_FILE, "r") as f:
        split_metadata = json.load(f)

    dataset_dict = {}
    for split in split_metadata.keys():
        sources = split_metadata[split]
        datasets = []

        # Iterate through all files corresponding to splits.json
        for source_path in tqdm(sources, desc=f"Creating {split} dataset"):
            complete_caption_path = DATASET_BASE / "complete" / source_path
            with open(complete_caption_path, "r") as entry_file:
                base_data = json.load(entry_file)

            # Get image path
            image_path = DATASET_BASE / BASE_IMAGES_PATH / base_data["image"]

            # base_data["summaries"]["base"] is usually a list where each element is a text segment
            annotations = base_data["summaries"]["base"]  # You can modify this key according to your data

            # Build temporary dataset:
            #   Multiple annotations (each corresponds to one row), all pointing to the same image
            tmp_ds = Dataset.from_dict(
                {
                    "image": [str(image_path)] * len(annotations),
                    "caption": annotations,
                }
            )
            datasets.append(tmp_ds)

        # Merge datasets
        dataset_dict[split] = concatenate_datasets(datasets)

    dci = DatasetDict(dataset_dict)

    # ---------- 2) (Optional) Filter invalid images ----------
    for split_name in dci.keys():
        ds = dci[split_name]
        print(f"\nFiltering corrupt images in split: {split_name} ...")
        initial_len = len(ds)
        ds = ds.filter(filter_corrupt_images, batched=True, num_proc=8)
        final_len = len(ds)
        print(f"  Removed {initial_len - final_len} images.")
        dci[split_name] = ds

    # ---------- 3) Segment captions: apply process_text/postprocess_text to each caption ----------
    def segment_caption(example):
        """
        Segment a single caption and return a list stored in the segmented_caption field.
        """
        segments = postprocess_text(process_text(example["caption"]))
        return {"segmented_caption": segments}

    for split_name in dci.keys():
        ds = dci[split_name]
        print(f"\nSegmenting captions in split: {split_name} ...")
        ds = ds.map(segment_caption)
        # Remove empty segments
        ds = ds.map(remove_empty_segments)
        dci[split_name] = ds

    # ---------- 4) Build negative examples: random sampling from segmented_caption ----------
    # First merge all splits to sample negative examples globally; or you can sample within the same split
    # Here we demonstrate sampling within the same split: random.choice(ds)
    # Since dataset.map() doesn't easily allow random.choice(ds) inside, we need to wrap it outside the function
    def build_neg_splits_inplace(ds):
        """
        Given a single split Dataset ds,
        return a Dataset with neg_details column
        """
        def obtain_neg_splits(example):
            # example["segmented_caption"] is a list
            seg_caps = example["segmented_caption"]
            neg_details = []

            for i in range(len(seg_caps) - 1):
                # Keep sampling until we get a different image
                while True:
                    sample_idx = random.randint(0, len(ds) - 1)
                    sample = ds[sample_idx]
                    if sample["image"] != example["image"]:
                        # If the sampled example has shorter segmented_caption and i exceeds its length, truncate
                        if i >= len(sample["segmented_caption"]):
                            i = len(sample["segmented_caption"]) - 1
                        break

                # In neg_details, you can choose the original sentence or apply shuffle_words(sample["..."]) etc.
                # Here we just use it directly
                neg_details.append(sample["segmented_caption"][i].strip())

            return {"neg_details": neg_details}

        ds = ds.map(obtain_neg_splits)
        return ds

    for split_name in dci.keys():
        ds = dci[split_name]
        print(f"\nAdding negative splits in split: {split_name} ...")
        ds = build_neg_splits_inplace(ds)
        dci[split_name] = ds

    # ---------- 5) Concatenate segments => captions, neg_captions ----------
    
    # Initialize tokenizer
    try:
        if ONLINE:
            print("Attempting to load tokenizer from online...")
            tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_PATH)
        else:
            print("Attempting to load tokenizer from local path...")
            tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
    except OSError as e:
        print(f"Local loading failed: {e}")
        print("Automatically switching to online mode...")
        tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_PATH)

    def join_segmented_captions(example):
        seg_caps = example["segmented_caption"]
        neg_details = example["neg_details"]

        merged_captions = []
        neg_captions = []

        if len(seg_caps) > 0:
            # First add the first segment
            merged_captions = [seg_caps[0].strip()]

        # Starting from the second segment, gradually concatenate
        for caption_piece, neg_piece in zip(seg_caps[1:], neg_details):
            merged_candidate = f"{merged_captions[-1]} {caption_piece.strip()}"
            neg_candidate = f"{merged_captions[-1]} {neg_piece.strip()}"

            # Check length using tokenizer
            tokenized = tokenizer([merged_candidate, neg_candidate])["input_ids"]
            if len(tokenized[0]) > MAX_LEN or len(tokenized[1]) > MAX_LEN:
                break

            merged_captions.append(merged_candidate)
            neg_captions.append(neg_candidate)

        return {
            "captions": merged_captions,
            "neg_captions": neg_captions,
        }

    for split_name in dci.keys():
        ds = dci[split_name]
        print(f"\nJoining segmented captions in split: {split_name} ...")

        # Execute map operation
        ds = ds.map(join_segmented_captions)

        # Remove unnecessary columns
        # Here we remove the original "caption", "segmented_caption", "neg_details" columns
        keep_cols = set(ds.column_names) - set(["caption", "segmented_caption", "neg_details"])
        ds = ds.remove_columns(list(set(ds.column_names) - keep_cols))

        # Filter: keep rows with at least one neg_captions
        ds = ds.filter(lambda ex: len(ex["neg_captions"]) > 0)

        dci[split_name] = ds

    # ---------- 6) Save to specified path ----------
    print(f"\nSaving final dataset to {OUTPUT_DIR} ...")
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    dci.save_to_disk(str(OUTPUT_DIR))
    print("Done.")
    
    # Print final statistics
    print("\n" + "="*50)
    print("FINAL DATASET STATISTICS")
    print("="*50)
    for split_name in dci.keys():
        print(f"{split_name}: {len(dci[split_name])} samples")


if __name__ == "__main__":
    main()
