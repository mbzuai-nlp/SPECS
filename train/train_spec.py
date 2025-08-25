#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import os
import sys
import subprocess
from dataclasses import dataclass, field
from typing import Optional, Union, List
import random
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler

import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from PIL import Image
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Import LongCLIP related modules
from model import longclip
from model.model_spec import DetailLossCalculator
from scheduler import cosine_lr

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# Image preprocessing transforms
class Transform(torch.nn.Module):
    def __init__(self, image_size, mean, std):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            Resize(
                [image_size], interpolation=InterpolationMode.BICUBIC, antialias=None
            ),
            CenterCrop(image_size),
            ConvertImageDtype(torch.float),
            Normalize(mean, std),
        )

    def forward(self, x) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
        return x


def transform_images(examples, image_column, image_transformations):
    """Transform images in the dataset batch"""
    images = [
        read_image(image_file, mode=ImageReadMode.RGB)
        for image_file in examples[image_column]
    ]
    examples["pixel_values"] = [image_transformations(image) for image in images]
    return examples


def setup_distributed(backend="nccl", port=None):
    """Initialize distributed training environment"""
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # Specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29522"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank % num_gpus)
    
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    
    return rank, rank % num_gpus


class LongCLIPDetailsTrainer:
    def __init__(self, args, rank, local_rank):
        self.args = args
        self.rank = rank
        self.local_rank = local_rank
        
        # Set up save directories
        self.logdir = args.output_dir
        self.ckptdir = os.path.join(self.logdir, "ckpt")
        os.makedirs(self.ckptdir, exist_ok=True)
        
        # TensorBoard logging
        if rank == 0:
            self.writer = SummaryWriter(self.logdir)
        
        # Load LongCLIP model
        logger.info(f"Loading LongCLIP model from {args.model_path}")
        self.model, self.preprocess = longclip.load(args.model_path, device='cpu')
        
        # Create detail loss calculator
        self.loss_calculator = DetailLossCalculator()
        self.loss_calculator.set_loss_balance(
            lambda_contrast=args.lambda_contrast,
            lambda_details=args.lambda_details,
            lambda_neg=args.lambda_neg,
            epsilon=args.epsilon,
            beta=args.beta
        )
        
        # Prepare for training
        self.model.train()
        self.model = self.model.cuda()
        
        # Distributed training
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model, device_ids=[local_rank]
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay
        )
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Load dataset
        self.load_dataset()
    
    def load_dataset(self):
        """Load and preprocess dataset"""
        logger.info(f"Loading dataset from {self.args.dataset_name}")
        self.dataset = load_from_disk(self.args.dataset_name)
        self.dataset = self.dataset.filter(lambda example: len(example["captions"]) > 1)
        
        # Prepare training dataset
        if "train" not in self.dataset:
            logger.info("Using full dataset as training")
            self.train_dataset = self.dataset
        else:
            self.train_dataset = self.dataset["train"]
        
        # Image transformations
        image_mean = [0.48145466, 0.4578275, 0.40821073]
        image_std = [0.26862954, 0.26130258, 0.27577711]
        image_size = self.model.module.visual.input_resolution
        
        self.image_transformations = Transform(image_size, image_mean, image_std)
        self.image_transformations = torch.jit.script(self.image_transformations)
        
        # Set transforms
        self.train_dataset.set_transform(
            lambda examples: transform_images(examples, "image", self.image_transformations)
        )
    
    def create_dataloader(self):
        """Create training data loader"""
        train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
        
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=train_sampler,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )
        
        return train_loader, train_sampler
    
    def collate_fn(self, examples):
        """Collate batch data together, similar to CLIPDetails approach"""
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        
        # Randomly select a caption index for each sample
        indices = [
            random.choice(range(len(sample["captions"]) - 1)) for sample in examples
        ]
        
        # Collect various texts
        caption = [sample["caption"] for sample in examples]
        base = [sample["captions"][i] for sample, i in zip(examples, indices)]
        neg = [sample["neg_captions"][i] for sample, i in zip(examples, indices)]
        detail = [sample["captions"][i + 1] for sample, i in zip(examples, indices)]
        
        # Encode texts
        caption_ids = longclip.tokenize(caption, truncate=True).to(pixel_values.device)
        base_ids = longclip.tokenize(base, truncate=True).to(pixel_values.device)
        detail_ids = longclip.tokenize(detail, truncate=True).to(pixel_values.device)
        neg_ids = longclip.tokenize(neg, truncate=True).to(pixel_values.device)
        
        return {
            "pixel_values": pixel_values,
            "caption_ids": caption_ids,
            "base_ids": base_ids,
            "detail_ids": detail_ids,
            "neg_ids": neg_ids
        }
    
    def train(self):
        """Training loop"""
        logger.info("Starting training")
        
        # Create data loader
        train_loader, train_sampler = self.create_dataloader()
        
        # Set up learning rate scheduler
        steps_per_epoch = (len(train_loader) + self.args.gradient_accumulation_steps - 1) // self.args.gradient_accumulation_steps
        num_training_steps = steps_per_epoch * self.args.num_train_epochs
        
        logger.info(f"Total training steps: {num_training_steps} (Steps per epoch: {steps_per_epoch})")
        
        self.scheduler = cosine_lr(
            self.optimizer,
            base_lr=self.args.learning_rate,
            warmup_length=self.args.warmup_steps,
            steps=num_training_steps
        )
        
        # Training loop
        global_step = 0
        
        for epoch in range(self.args.num_train_epochs):
            logger.info(f"Starting epoch {epoch}")
            train_sampler.set_epoch(epoch)
            
            self.model.train()
            epoch_loss = 0
            
            # Reset tqdm progress bar at the start of each epoch
            progress_bar = tqdm(
                total=steps_per_epoch,
                disable=(self.rank != 0),
                desc=f"Epoch {epoch}",
                leave=True
            )
            
            optimizer_steps = 0
            
            for step, batch in enumerate(train_loader):
                # Prepare data
                pixel_values = batch["pixel_values"].cuda()
                caption_ids = batch["caption_ids"].cuda()
                base_ids = batch["base_ids"].cuda()
                detail_ids = batch["detail_ids"].cuda()
                neg_ids = batch["neg_ids"].cuda()
                
                # Forward + backward
                with torch.cuda.amp.autocast():
                    # Calculate three types of losses
                    loss, loss_components = self.loss_calculator.calculate_loss(
                        self.model.module,
                        pixel_values,
                        caption_ids,
                        base_ids,
                        detail_ids,
                        neg_ids
                    )
                    
                    # Gradient accumulation - average loss
                    loss = loss / self.args.gradient_accumulation_steps
                
                # Backward propagation
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation - only update after accumulation steps complete
                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (step + 1 == len(train_loader)):
                    # Update learning rate - only call when parameters are updated
                    if self.rank == 0:  # Only let main process update global_step
                        global_step += 1
                    
                    self.scheduler(global_step if self.rank == 0 else optimizer_steps)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    optimizer_steps += 1
                    
                    # Update progress bar
                    if self.rank == 0:
                        progress_bar.update(1)
                    
                    # Logging - only recorded by main process
                    if self.rank == 0 and global_step % self.args.logging_steps == 0:
                        lr = self.optimizer.param_groups[0]['lr']
                        progress_bar.set_postfix(loss=loss.item() * self.args.gradient_accumulation_steps, lr=lr)
                        
                        self.writer.add_scalar("train/loss", loss.item() * self.args.gradient_accumulation_steps, global_step)
                        self.writer.add_scalar("train/learning_rate", lr, global_step)
                        
                        # Record loss components
                        self.writer.add_scalar("train/contrastive_loss", 
                                             self.loss_calculator.last_contrastive_loss, 
                                             global_step)
                        self.writer.add_scalar("train/detail_loss", 
                                             self.loss_calculator.last_detail_loss, 
                                             global_step)
                        self.writer.add_scalar("train/neg_loss", 
                                             self.loss_calculator.last_neg_loss, 
                                             global_step)
                        
                        # Record thresholds
                        self.writer.add_scalar("train/epsilon", self.loss_calculator.epsilon, global_step)
                        self.writer.add_scalar("train/beta", self.loss_calculator.beta, global_step)
                    
                    # Save model
                    if self.rank == 0 and global_step > 0 and global_step % self.args.save_steps == 0:
                        self.save_model(global_step)
                
                epoch_loss += loss.item() * self.args.gradient_accumulation_steps
            
            # Close progress bar at the end of epoch
            if self.rank == 0:
                progress_bar.close()
            
            # Record total loss for each epoch
            if self.rank == 0:
                logger.info(f"Epoch {epoch} average loss: {epoch_loss / len(train_loader)}")
                logger.info(f"Current global step: {global_step}")
            
            # Save model at the end of each epoch
            if self.rank == 0:
                self.save_model(global_step, epoch=epoch)
        
        # Save final model at the end of training
        if self.rank == 0:
            self.save_model(global_step, final=True)
    
    def save_model(self, step, epoch=None, final=False):
        """Save model checkpoint"""
        if final:
            output_dir = os.path.join(self.ckptdir, "final")
            filename = "longclipdetails-final.pt"
        elif epoch is not None:
            output_dir = os.path.join(self.ckptdir, f"epoch-{epoch}")
            filename = f"longclipdetails-epoch-{epoch}.pt"
        else:
            output_dir = os.path.join(self.ckptdir, f"step-{step}")
            filename = f"longclipdetails-step-{step}.pt"
        
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, filename)
        
        logger.info(f"Saving model to {model_path}")
        torch.save(self.model.module.state_dict(), model_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for LongCLIPDetails")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, required=True, 
                      help="Path to the pretrained LongCLIP model")
    
    # Data parameters
    parser.add_argument("--dataset_name", type=str, required=True,
                      help="Path to the dataset")
    
    # Training parameters
    parser.add_argument("--num_train_epochs", type=int, default=3,
                      help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32,
                      help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                      help="Number of update steps to accumulate gradients for")
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                      help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                      help="Weight decay rate")
    parser.add_argument("--warmup_steps", type=int, default=500,
                      help="Number of warmup steps")
    parser.add_argument("--output_dir", type=str, default="./longclipdetails-output",
                      help="Output directory for logs and checkpoints")
    parser.add_argument("--logging_steps", type=int, default=10,
                      help="Log every X steps")
    parser.add_argument("--save_steps", type=int, default=1000,
                      help="Save checkpoint every X steps")
    parser.add_argument("--num_workers", type=int, default=4,
                      help="Number of workers for dataloader")
    
    # Loss balance parameters
    parser.add_argument("--lambda_contrast", type=float, default=1.0,
                      help="Weight for contrastive loss")
    parser.add_argument("--lambda_details", type=float, default=1.0,
                      help="Weight for detail loss")
    parser.add_argument("--lambda_neg", type=float, default=1.0,
                      help="Weight for negative loss")
    parser.add_argument("--epsilon", type=float, default=1e-3,
                      help="Margin for detail loss")
    parser.add_argument("--beta", type=float, default=1e-3,
                      help="Margin for negative loss")
    
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set up distributed training
    rank, local_rank = setup_distributed()
    
    # Create trainer
    trainer = LongCLIPDetailsTrainer(args, rank, local_rank)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main() 