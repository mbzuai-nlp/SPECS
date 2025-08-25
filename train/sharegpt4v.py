import json
from PIL import Image
import torch
import torch.utils.data as data
import os
import numpy as np
import random
from transformers import CLIPProcessor

data4v_root = '/workspace/ShareGPT4V/data/'
json_name = 'share-captioner_coco_lcs_sam_1246k_1107.json'
image_root = '/workspace/ShareGPT4V/data/'

class share4v_val_dataset(data.Dataset):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.data4v_root = data4v_root
        self.json_name = json_name
        self.image_root = image_root
        self.total_len = 1000
        with open(data4v_root + json_name, 'r', encoding='utf8') as fp:
            self.json_data = json.load(fp)[:self.total_len]
        
        # Use HF processor instead of OpenAI CLIP
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.valid_indices = []
        # Preprocess to find valid images
        for i in range(self.total_len):
            image_path = self.image_root + self.json_data[i]['image']
            if os.path.exists(image_path):
                self.valid_indices.append(i)
        print(f"Validation set: Found {len(self.valid_indices)} valid images out of {self.total_len}")
    
    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        # Get the actual index in the json data
        actual_index = self.valid_indices[index]
        caption = self.json_data[actual_index]['conversations'][1]['value']
        caption = caption.replace("\n", " ")
        image_name = self.image_root + self.json_data[actual_index]['image']
        image = Image.open(image_name).convert("RGB")
        
        # Use processor for image only (return_tensors="pt")
        processed = self.processor(images=image, return_tensors="pt")
        image_tensor = processed["pixel_values"].squeeze(0)  # Remove batch dimension
        
        return image_tensor, caption


class share4v_train_dataset(data.Dataset):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.data4v_root = data4v_root
        self.json_name = json_name
        self.image_root = image_root
        self.total_len = 1000
        with open(data4v_root + json_name, 'r', encoding='utf8') as fp:
            self.json_data = json.load(fp)[self.total_len:]
        
        # Use HF processor instead of OpenAI CLIP
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.valid_indices = []
        # Preprocess to find valid images
        for i in range(len(self.json_data)):
            image_path = self.image_root + self.json_data[i]['image']
            if os.path.exists(image_path):
                self.valid_indices.append(i)
        print(f"Training set: Found {len(self.valid_indices)} valid images out of {len(self.json_data)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        # Get the actual index in the json data
        actual_index = self.valid_indices[index]
        caption = self.json_data[actual_index]['conversations'][1]['value']
        caption = caption.replace("\n", " ")
        
        caption_short = caption.split(". ")[0]
        
        image_name = self.image_root + self.json_data[actual_index]['image']
        image = Image.open(image_name).convert("RGB")
        
        # Use processor for image only (return_tensors="pt")
        processed = self.processor(images=image, return_tensors="pt")
        image_tensor = processed["pixel_values"].squeeze(0)  # Remove batch dimension
        
        return image_tensor, caption, caption_short