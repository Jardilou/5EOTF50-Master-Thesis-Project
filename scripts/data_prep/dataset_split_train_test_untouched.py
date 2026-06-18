"""
================================================================================
Script: YOLO Dataset Train/Validation/Untouched Splitter
================================================================================

Description:
    A utility script to automatically randomly partition a flat directory of 
    images and their corresponding YOLO-format text labels into a standardized 
    training, validation, and untouched holdout directory hierarchy. 

    Key Functions:
    1. Directory Initialization: Generates the standard YOLO subfolder architecture 
       (images/train, images/val, labels/train, labels/val) plus the holdout directories
       (untouched_images, untouched_labels) inside the output root.
    2. Pair Validation: Scans the input directories to ensure every label has a 
       corresponding image file, ignoring orphaned files.
    3. Holdout Extraction: Extracts exactly 10% of the total valid pairs to serve 
       as an untouched test set before any training splits occur.
    4. Proportional Allocation: Splits the remaining 90% of the data based on a 
       user-defined ratio (default 80% training, 20% validation).

Author:
    Louis Jardinet
    Universiti Teknologi Petronas

Date:
    Updated 2026

Dependencies:
    - Python standard libraries: os, shutil, random
================================================================================
"""

import os
import shutil
import random

def split_yolo_dataset(images_dir, labels_dir, output_dir, train_ratio=0.8):
    """
    Splits images and labels into train, val, and untouched holdout directories.
    """
    # 1. Create the YOLO and Holdout directory structure
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)
        
    # Create holdout folders in the output root (same level as 'images' and 'labels')
    os.makedirs(os.path.join(output_dir, 'untouched_images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'untouched_labels'), exist_ok=True)

    # 2. Find all valid Image + Label pairs
    valid_pairs = []
    valid_extensions = ('.jpg', '.jpeg', '.png')

    for label_name in os.listdir(labels_dir):
        if not label_name.endswith('.txt'):
            continue
        
        base_name = os.path.splitext(label_name)[0]
        
        # Look for the matching image file
        image_path = None
        for ext in valid_extensions:
            temp_path = os.path.join(images_dir, base_name + ext)
            if os.path.exists(temp_path):
                image_path = temp_path
                img_ext = ext
                break
        
        if image_path:
            label_path = os.path.join(labels_dir, label_name)
            valid_pairs.append((image_path, label_path, base_name, img_ext))

    if not valid_pairs:
        print("Error: No matching image-label pairs found. Check your folder paths!")
        return

    # 3. Shuffle for randomness
    random.seed(42) # Keeps the split consistent if you run it multiple times
    random.shuffle(valid_pairs)
    
    # 4. Extract 10% for the untouched holdout dataset
    untouched_idx = int(len(valid_pairs) * 0.10)
    untouched_pairs = valid_pairs[:untouched_idx]
    remaining_pairs = valid_pairs[untouched_idx:]
    
    # 5. Calculate split index for the remaining 90% of the data
    split_idx = int(len(remaining_pairs) * train_ratio)
    train_pairs = remaining_pairs[:split_idx]
    val_pairs = remaining_pairs[split_idx:]

    # 6. Copy files to their new homes
    print("Copying Untouched Holdout data (10%)...")
    for img_path, lbl_path, base_name, ext in untouched_pairs:
        shutil.copy(img_path, os.path.join(output_dir, 'untouched_images', base_name + ext))
        shutil.copy(lbl_path, os.path.join(output_dir, 'untouched_labels', base_name + '.txt'))

    def copy_splits(pairs, split_name):
        for img_path, lbl_path, base_name, ext in pairs:
            # Copy Image
            shutil.copy(img_path, os.path.join(output_dir, 'images', split_name, base_name + ext))
            # Copy Label
            shutil.copy(lbl_path, os.path.join(output_dir, 'labels', split_name, base_name + '.txt'))

    print("Copying Training data...")
    copy_splits(train_pairs, 'train')
    
    print("Copying Validation data...")
    copy_splits(val_pairs, 'val')

    # 7. Print Updated Summary
    print("\nDataset Split Complete!")
    print("-" * 40)
    print(f"Total pairs processed: {len(valid_pairs)}")
    print(f"Untouched Holdout:     {len(untouched_pairs)} images (10%)")
    print(f"Training set:          {len(train_pairs)} images ({int((len(train_pairs)/len(valid_pairs))*100)}% of total)")
    print(f"Validation set:        {len(val_pairs)} images ({int((len(val_pairs)/len(valid_pairs))*100)}% of total)")
    print(f"Output saved to:       {output_dir}")

# ==========================================
# USAGE - UPDATE THESE PATHS
# ==========================================
# INPUT_IMAGES = r"DATASETS\DATASETS FOR INSTANCE SEGMENTATION\Deepfish_Segmentation\images\valid"  
# INPUT_LABELS = r"DATASETS\DATASETS FOR INSTANCE SEGMENTATION\Deepfish_Segmentation\mask_labels\valid" 
# OUTPUT_DATASET = r"DATASETS\DATASETS FOR INSTANCE SEGMENTATION\Deepfish_Segmentation\split_dataset" 

INPUT_IMAGES = r"DATASETS\DATASETS FOR INSTANCE SEGMENTATION\Fish4knowledge\yolo_images"  
INPUT_LABELS = r"DATASETS\DATASETS FOR INSTANCE SEGMENTATION\Fish4knowledge\yolo_labels" 
OUTPUT_DATASET = r"DATASETS\DATASETS FOR INSTANCE SEGMENTATION\Fish4knowledge\split_dataset" 


split_yolo_dataset(INPUT_IMAGES, INPUT_LABELS, OUTPUT_DATASET, train_ratio=0.8)