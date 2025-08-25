"""
Prepare ShareGPT4V dataset for LongCLIPDetail training.

This script processes the ShareGPT4V dataset to create a format suitable for training
LongCLIPDetail model with caption, base caption, detail caption, and negative captions.
"""

import os
import argparse
import random
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from tqdm import tqdm
from datasets import load_dataset, Dataset, load_from_disk
from transformers import CLIPTokenizer
from PIL import Image

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler()],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def process_text(text: str) -> List[str]:
    """
    Process a long text caption into segments based on sentence structure.
    
    Args:
        text: A long text caption to process
        
    Returns:
        List of text segments
    """
    # Simple sentence splitting
    sentences = []
    for sent in text.split('. '):
        sent = sent.strip()
        if sent:
            if not sent.endswith('.'):
                sent += '.'
            sentences.append(sent)
    
    return sentences


def postprocess_text(segments: List[str]) -> List[str]:
    """
    Post-process segmented text to ensure they're meaningful and not too short.
    
    Args:
        segments: List of text segments
        
    Returns:
        Processed list of text segments
    """
    # Filter out very short segments (likely not meaningful)
    filtered_segments = [seg for seg in segments if len(seg.split()) > 3]
    
    return filtered_segments


def check_image_file(image_path: str) -> bool:
    """
    Check if an image file exists and is valid.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        True if the image is valid, False otherwise
    """
    if not os.path.exists(image_path):
        return False
        
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def prepare_sharegpt4v_dataset(
    input_json_path: str,
    image_base_path: str,
    output_path: str,
    tokenizer_path: str = "openai/clip-vit-base-patch32",
    max_token_length: int = 77,
    shuffle_first_portion: float = 0.75,
    online: bool = True,
    max_samples: Optional[int] = None,
):
    """
    Prepare the ShareGPT4V dataset for LongCLIPDetail training.
    
    Args:
        input_json_path: Path to the input JSON file
        image_base_path: Base path to the image files
        output_path: Path to save the processed dataset
        tokenizer_path: Path or name of the CLIP tokenizer
        max_token_length: Maximum token length for captions
        shuffle_first_portion: Portion of dataset to use shuffled negative captions
        online: Whether to allow online downloading of the tokenizer
        max_samples: Maximum number of samples to process (for debugging)
    """
    logger.info(f"Loading dataset from {input_json_path}")
    
    # Load the dataset
    dataset = load_dataset("json", data_files=input_json_path)["train"]
    
    if max_samples is not None:
        dataset = dataset.select(range(min(len(dataset), max_samples)))
    
    logger.info(f"Initial dataset size: {len(dataset)}")
    
    # Process image paths
    def preprocess_json(example):
        original_path = example["image"]
        if not original_path.startswith('/'):
            new_path = str(Path(image_base_path) / original_path)
        else:
            new_path = original_path
        example["image"] = new_path
        example["caption"] = example["conversations"][-1]["value"]
        return example
    
    logger.info("Processing image paths and extracting captions...")
    dataset = dataset.map(preprocess_json).remove_columns(["id", "conversations"])
    
    # Filter invalid images
    def filter_corrupt_images(examples):
        valid_images = []
        for image_file in examples["image"]:
            valid_images.append(check_image_file(image_file))
        return valid_images
    
    initial_len = len(dataset)
    logger.info("Filtering invalid images...")
    dataset = dataset.filter(filter_corrupt_images, batched=True, num_proc=16)
    final_len = len(dataset)
    logger.info(f"Removed {initial_len - final_len} invalid images")
    
    # Segment captions
    logger.info("Segmenting captions...")
    segments = []
    for item in tqdm(dataset):
        segments.append(postprocess_text(process_text(item["caption"])))
    dataset = dataset.add_column("segmented_caption", segments)
    
    # Remove examples with empty segments
    def remove_empty_segments(example):
        example["segmented_caption"] = [
            caption for caption in example["segmented_caption"] if caption
        ]
        return example
    
    dataset = dataset.map(remove_empty_segments)
    dataset = dataset.filter(lambda example: len(example["segmented_caption"]) > 1)
    
    # Add dataset part column for shuffling strategy
    dataset_size = len(dataset)
    split_index = int(dataset_size * shuffle_first_portion)
    logger.info(f"Dataset size: {dataset_size}, Split at index: {split_index}")
    
    indices = list(range(dataset_size))
    dataset = dataset.add_column(
        "dataset_part", 
        ["shuffle" if i < split_index else "no_shuffle" for i in indices]
    )
    
    # Create negative captions
    logger.info("Creating negative captions...")
    
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
            
            # First portion gets shuffled captions, last portion gets original
            if example["dataset_part"] == "shuffle":
                neg_details.append(shuffle_words(sample["segmented_caption"][i].strip()))
            else:
                neg_details.append(sample["segmented_caption"][i].strip())
            
        return {"neg_details": neg_details}
    
    dataset = dataset.map(obtain_neg_splits)
    
    # Load tokenizer for length checking
    logger.info("Loading tokenizer...")
    try:
        if online:
            tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
        else:
            tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        raise
    
    # Process captions
    logger.info("Creating caption chains...")
    
    def join_segmented_captions(example):
        segmented_caption = example["segmented_caption"]
        neg_details = example["neg_details"]
        
        merged_captions = [segmented_caption[0].strip()]
        neg_captions = []
        
        for caption, neg_caption in zip(segmented_caption[1:], neg_details):
            merged_caption = f"{merged_captions[-1]} {caption.strip()}"
            neg_caption = f"{merged_captions[-1]} {neg_caption.strip()}"
            
            tokens = tokenizer([merged_caption, neg_caption])["input_ids"]
            if len(tokens[0]) > max_token_length or len(tokens[1]) > max_token_length:
                break
                
            merged_captions.append(merged_caption)
            neg_captions.append(neg_caption)
            
        example["captions"] = merged_captions
        example["neg_captions"] = neg_captions
        return example
    
    logger.info("Processing captions...")
    dataset = dataset.map(
        join_segmented_captions,
        desc="Processing captions",
        remove_columns=["segmented_caption", "neg_details", "dataset_part"]
    )
    
    # Filter examples with no valid caption chains
    dataset = dataset.filter(lambda example: len(example["neg_captions"]) > 0)
    
    # Save the processed dataset
    logger.info(f"Saving processed dataset to {output_path}")
    dataset.save_to_disk(output_path)
    
    # Print some statistics
    logger.info(f"Final dataset size: {len(dataset)}")
    caption_lengths = [len(ex["captions"]) for ex in dataset]
    avg_caption_length = sum(caption_lengths) / len(caption_lengths)
    logger.info(f"Average number of caption levels: {avg_caption_length:.2f}")
    
    return dataset


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Prepare data for LongCLIPDetail training")
    
    parser.add_argument("--input_json", type=str, required=True,
                        help="Path to the input ShareGPT4V JSON file")
    parser.add_argument("--image_base_path", type=str, required=True,
                        help="Base path to image files")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the processed dataset")
    parser.add_argument("--tokenizer_path", type=str, default="openai/clip-vit-base-patch32",
                        help="Path or name of the CLIP tokenizer")
    parser.add_argument("--max_token_length", type=int, default=77,
                        help="Maximum token length for captions")
    parser.add_argument("--shuffle_first_portion", type=float, default=0.75,
                        help="Portion of dataset to use shuffled negative captions")
    parser.add_argument("--offline", action="store_true",
                        help="Run in offline mode (no downloads)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process (for debugging)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Prepare the dataset
    prepare_sharegpt4v_dataset(
        input_json_path=args.input_json,
        image_base_path=args.image_base_path,
        output_path=args.output_path,
        tokenizer_path=args.tokenizer_path,
        max_token_length=args.max_token_length,
        shuffle_first_portion=args.shuffle_first_portion,
        online=not args.offline,
        max_samples=args.max_samples,
    )
    
    logger.info("Data preparation completed!")


if __name__ == "__main__":
    main() 