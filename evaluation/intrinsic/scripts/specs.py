import sys
from pathlib import Path

# Add project root to Python path
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]  # Go up 3 levels to reach project root
sys.path.append(str(project_root))

# Add Long-CLIP to path
sys.path.append('/workspace/Long-CLIP')

try:
    from model import longclip
except ImportError:
    try:
        from model import longclip
    except ImportError:
        print("Error: Could not import longclip module. Please ensure Long-CLIP is installed correctly.")
        sys.exit(1)


from accelerate import Accelerator
from accelerate.utils import gather_object
from dataclasses import dataclass, field
from datasets import load_from_disk, Image
import numpy as np
from typing import Optional
from tqdm import tqdm
import torch
import os

from transformers import HfArgumentParser, CLIPImageProcessor, AutoTokenizer, CLIPModel

# Change this import to use a relative path
sys.path.append(str(current_file.parent.parent))
from models.scores import CLIPScore
from metric import compute_increments

# Import utils from project root
from utils import cosine_similarity
from peft import PeftModel


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="openai/clip-vit-large-patch14",
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
        default="../results/",
        metadata={"help": "The directory where to save the results."},
    )
    save_file: Optional[bool] = field(
        default=True,
        metadata={"help": "If save or not the file"},
    )


@torch.no_grad()
def main():
    # parameters
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    accelerator = Accelerator()
    device = accelerator.device

    clip, processor = longclip.load(model_args.model_name_or_path)
    clip.to(accelerator.process_index)

    # load dataset and process
    dataset_name = data_args.data_dir.stem
    dataset = load_from_disk(data_args.data_dir)
    if data_args.data_split is not None:
        try:
            dataset = dataset[data_args.data_split]
        except:
            raise ValueError(f"Dataset split {data_args.data_split} does not exist")

    dataset = dataset.cast_column(data_args.image_column, Image())

    max_split = max(
        dataset.map(lambda sample: {"s": len(sample[data_args.text_column])})["s"]
    )

    with accelerator.split_between_processes(
        list(range(len(dataset)))
    ) as dataset_indices:
        ds = dataset.select(dataset_indices)
        results = np.zeros((len(ds), max_split, 2))
        for r, sample in enumerate(tqdm(ds)):
            text = longclip.tokenize(sample[data_args.text_column]).to(device)
            text_embedding = clip.encode_text(text)
            neg_text = longclip.tokenize(sample[data_args.neg_text_column]).to(device)
            neg_text_embedding = clip.encode_text(neg_text)
            images = processor(sample[data_args.image_column]).unsqueeze(0).to(device)
            image_embedding = clip.encode_image(images)
            similarity = cosine_similarity(text_embedding, image_embedding)
            neg_similarity = cosine_similarity(neg_text_embedding, image_embedding)

            for c, (sim, neg_sim) in enumerate(zip(similarity, neg_similarity)):
                results[r][c][0] = sim[0]
                results[r][c][1] = neg_sim[0]

    results_gathered = gather_object([results])
    results_gathered = np.concatenate(results_gathered)
    if accelerator.is_main_process:
        if data_args.save_file:
            try:
                if data_args.data_split is not None:
                    output_file = (
                        data_args.output_dir
                        / f"{dataset_name}_{data_args.data_split}{model_args.postfix}.npy"
                    )
                else:
                    output_file = (
                        data_args.output_dir / f"{dataset_name}{model_args.postfix}.npy"
                    )
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                
                np.save(output_file, results_gathered)
                print(f"Results saved in {output_file}")
            except Exception as e:
                print(f"WARNING: Failed to save results file: {e}")
                print("Continuing with evaluation anyway...")

        # Compute metrics
        pos, neg = compute_increments(results_gathered)

        print(f"Results for {dataset_name}")
        print(f"Pos: {np.mean(pos):.4f}  Neg: {np.mean(neg):.4f}")


if __name__ == "__main__":
    main()
