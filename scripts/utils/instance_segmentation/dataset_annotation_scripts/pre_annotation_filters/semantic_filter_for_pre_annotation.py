"""
================================================================================
Script: YOLO Semantic Annotation Filter (CLIP-Based)
================================================================================

Description:
    A script to filter YOLO-format instance segmentation datasets based on 
    semantic content using the OpenAI CLIP (Contrastive Language-Image Pretraining) 
    model. The script evaluates secondary annotations (Class 1) to determine 
    if the visual content inside the bounding box actually represents a fish, 
    removing false positives triggered by background elements.

    Key Functions:
    1. Bounding Box Extraction: Converts normalized polygon coordinates into 
       absolute pixel bounding boxes to crop the region of interest from the image.
    2. Primary Class Bypass: Automatically retains all primary annotations 
       (Class 0) without processing, assuming they represent verified ground truth.
    3. Semantic Classification: Passes cropped regions of secondary annotations 
       (Class 1) through the CLIP vision-language model.
    4. Text-Prompt Filtering: Compares the cropped image against predefined text 
       prompts (e.g., fish, coral, rock, sand). The polygon is kept only if the 
       highest probability match is "a photo of a fish underwater".

Author:
    Louis Jardinet
    Universiti Teknologi Petronas

Date:
    January 2026 - June 2026

Dependencies:
    - Python standard libraries: os
    - External packages: opencv-python (cv2), numpy, torch, Pillow (PIL), transformers

Inputs:
    - input_dir: Directory containing the YOLO .txt label files (typically the 
      output from a previous morphological filtering step).
    - clean_dir: Directory where the semantically filtered .txt files will be saved.
    - img_dir: Directory containing the source images.
================================================================================
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def get_pixel_bbox(coords, img_width, img_height):
    """Extracts absolute pixel bounding box from normalized polygon vertices."""
    x, y = coords[0::2], coords[1::2]
    # Convert to pixel space
    x_px = np.clip(np.array(x) * img_width, 0, img_width - 1).astype(int)
    y_px = np.clip(np.array(y) * img_height, 0, img_height - 1).astype(int)
    
    x_min, x_max = np.min(x_px), np.max(x_px)
    y_min, y_max = np.min(y_px), np.max(y_px)
    return x_min, y_min, x_max, y_max

def semantic_filter_dataset(input_dir, clean_dir, img_dir):
    os.makedirs(clean_dir, exist_ok=True)
    
    print("Loading OpenAI CLIP Model into VRAM...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # We use the base ViT-B/32 model. It is extremely fast and highly accurate.
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    
    # Define the text prompts. CLIP will score the image against each of these.
    # The wording here is highly critical. Be descriptive.
    labels = [
        "a photo of a fish underwater",      # Index 0
        "a photo of a coral reef",           # Index 1
        "a photo of an underwater rock",     # Index 2
        "a photo of sand",                   # Index 3
        "a photo of empty blue water",       # Index 4
        "a photo of a sea urchin"            # Index 5
    ]
    
    print(f"Model loaded on {device}. Starting Semantic Sweep...")
    print("-" * 50)
    
    total_kept = 0
    total_dropped = 0
    
    for filename in os.listdir(input_dir):
        if not filename.endswith(".txt"): continue
        
        img_name = filename.replace(".txt", ".jpg")
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path): continue
        
        # Load OpenCV image and convert BGR to RGB for the PIL Image pipeline
        cv2_img = cv2.imread(img_path)
        img_h, img_w = cv2_img.shape[:2]
        rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        
        input_filepath = os.path.join(input_dir, filename)
        clean_filepath = os.path.join(clean_dir, filename)
        
        with open(input_filepath, 'r') as f:
            lines = f.readlines()
            
        valid_lines = []
        
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            
            class_id = int(parts[0])
            
            # If it's a Class 0 (Measurable Fish from your XML), we bypass CLIP entirely.
            # We already know it's a fish because you annotated it.
            if class_id == 0:
                valid_lines.append(line.strip())
                continue
                
            # If it's a Class 1 (SAM Candidate), we ask CLIP to verify it.
            if class_id == 1:
                coords = np.array(parts[1:], dtype=float)
                x_min, y_min, x_max, y_max = get_pixel_bbox(coords, img_w, img_h)
                
                # Sanity check: ensure bounding box has actual dimensions
                if (x_max - x_min) < 5 or (y_max - y_min) < 5:
                    total_dropped += 1
                    continue
                
                # Crop the bounding box from the main image
                crop_arr = rgb_img[y_min:y_max, x_min:x_max]
                crop_pil = Image.fromarray(crop_arr)
                
                # Run CLIP Inference
                inputs = processor(text=labels, images=crop_pil, return_tensors="pt", padding=True).to(device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    
                # Convert logits to percentages
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)[0]
                
                # Find the index of the highest probability
                best_match_idx = torch.argmax(probs).item()
                best_match_label = labels[best_match_idx]
                
                # Decision Matrix
                if best_match_idx == 0:  # "a photo of a fish underwater"
                    valid_lines.append(line.strip())
                    total_kept += 1
                else:
                    total_dropped += 1
                    
        # Write clean file
        if valid_lines:
            with open(clean_filepath, 'w') as f:
                f.write("\n".join(valid_lines))
                
    print("Semantic Filtering Complete.")
    print(f"Verified Class 1 Fish: {total_kept}")
    print(f"Deleted Background Objects (Coral/Rocks/Water): {total_dropped}")

if __name__ == '__main__':
    semantic_filter_dataset(
        # The input here should be the OUTPUT folder from your Morphology script!
        input_dir=r'DATASETS FOR INSTANCE SEGMENTATION\UMT_dataset\filtered_labels', 
        clean_dir=r'DATASETS FOR INSTANCE SEGMENTATION\UMT_dataset\semantic_filtered_labels',
        img_dir=r'DATASETS FOR INSTANCE SEGMENTATION\UMT_dataset\UMT images for annotation V1 CLEANED'
    )