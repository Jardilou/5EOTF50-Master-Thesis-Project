"""
================================================================================
Script: Unified Detection Dataset Builder
================================================================================

Description:
    This script aggregates multiple fish datasets (DEEPFISH, UMT, BezierFusion) 
    that were originally annotated for Pose Estimation / Instance Segmentation, 
    and converts them into a unified, pure Object Detection dataset. 

    Key Operations:
    1. Directory Management: Organizes outputs into nested subfolders 
       (e.g., images/train/DEEPFISH) inside a master directory.
    2. Label Cleaning: Reads YOLO-formatted text files, fixes a known numerical 
       fusion bug via Regex, and strips away all trailing pose keypoints, 
       retaining only the first 5 elements (class, x_center, y_center, w, h).
    3. Windows OS Compatibility: Automatically injects the '\\?\' prefix to 
       bypass the standard Windows 260-character path limit during I/O operations.

Author:
    Louis Jardinet
    Universiti Teknologi Petronas

Date:
    January 2026 - June 2026

Dependencies:
    - Standard Python Library: os, glob, shutil, re

Inputs:
    - Pre-defined paths in the `DATASETS` list pointing to source images and labels.

Outputs:
    - A newly populated `MASTER_OUTPUT_DIR` containing cleaned, ready-to-train 
      YOLO detection labels and copied images, cleanly split by dataset and 
      train/val sets.
================================================================================
"""


import os
import glob
import shutil
import re

# --- 1. SET YOUR MASTER OUTPUT DIRECTORY ---
MASTER_OUTPUT_DIR = r"DATASETS\DATASETS FOR FISH DETECTION ONLY FOR IS"

# --- 2. DEFINE ALL YOUR SOURCE FOLDERS ---
DATASETS = [
    {
        "name": "DEEPFISH_Train",
        "source_images": r"DATASETS\DATASETS FOR INSTANCE SEGMENTATION + POSE ESTIMATION\DEEPFISH Dataset\CURRENT\Head+Tail_DEEPFISH_Annotations\train_test_split\images\train",
        "source_labels": r"DATASETS\DATASETS FOR INSTANCE SEGMENTATION + POSE ESTIMATION\DEEPFISH Dataset\CURRENT\Head+Tail_DEEPFISH_Annotations\train_test_split\labels\train",
        "split": "train" 
    },
    {
        "name": "DEEPFISH_Val",
        "source_images": r"DATASETS\DATASETS FOR INSTANCE SEGMENTATION + POSE ESTIMATION\DEEPFISH Dataset\CURRENT\Head+Tail_DEEPFISH_Annotations\train_test_split\images\val",
        "source_labels": r"DATASETS\DATASETS FOR INSTANCE SEGMENTATION + POSE ESTIMATION\DEEPFISH Dataset\CURRENT\Head+Tail_DEEPFISH_Annotations\train_test_split\labels\val",
        "split": "val"
    },
    {
        "name": "UMT_Train",
        "source_images": r"DATASETS\DATASETS FOR INSTANCE SEGMENTATION + POSE ESTIMATION\UMT Dataset\CURRENT\Head + Tails UMT Annotation V2\TRAIN-TEST\images\train",
        "source_labels": r"DATASETS\DATASETS FOR INSTANCE SEGMENTATION + POSE ESTIMATION\UMT Dataset\CURRENT\Head + Tails UMT Annotation V2\TRAIN-TEST\labels\train",
        "split": "train"
    },
    {
        "name": "UMT_Val",
        "source_images": r"DATASETS\DATASETS FOR INSTANCE SEGMENTATION + POSE ESTIMATION\UMT Dataset\CURRENT\Head + Tails UMT Annotation V2\TRAIN-TEST\images\val",
        "source_labels": r"DATASETS\DATASETS FOR INSTANCE SEGMENTATION + POSE ESTIMATION\UMT Dataset\CURRENT\Head + Tails UMT Annotation V2\TRAIN-TEST\labels\val",
        "split": "val"
    },
    {
        "name": "BezierFusion_1_Train",
        "source_images": r"DATASETS\DATASETS FOR POSE ESTIMATION\BézierFusion Dataset 1\CURRENT\train test split\images\train",
        "source_labels": r"DATASETS\DATASETS FOR POSE ESTIMATION\BézierFusion Dataset 1\CURRENT\train test split\labels\train",
        "split": "train"
    },
    {
        "name": "BezierFusion_1_Val",
        "source_images": r"DATASETS\DATASETS FOR POSE ESTIMATION\BézierFusion Dataset 1\CURRENT\train test split\images\val",
        "source_labels": r"DATASETS\DATASETS FOR POSE ESTIMATION\BézierFusion Dataset 1\CURRENT\train test split\labels\val",
        "split": "val"
    },
    {
        "name": "BezierFusion_2_Train",
        "source_images": r"DATASETS\DATASETS FOR POSE ESTIMATION\Bézierfusion Dataset 2 Reannotated V4\Train Test\images\train",
        "source_labels": r"DATASETS\DATASETS FOR POSE ESTIMATION\Bézierfusion Dataset 2 Reannotated V4\Train Test\labels\train",
        "split": "train"
    },
    {
        "name": "BezierFusion_2_Val",
        "source_images": r"DATASETS\DATASETS FOR POSE ESTIMATION\Bézierfusion Dataset 2 Reannotated V4\Train Test\images\val",
        "source_labels": r"DATASETS\DATASETS FOR POSE ESTIMATION\Bézierfusion Dataset 2 Reannotated V4\Train Test\labels\val",
        "split": "val"
    }
]

def make_long_path(path):
    """Injects the Windows \\?\ prefix to bypass the 260 character limit."""
    abs_path = os.path.abspath(path)
    if os.name == 'nt' and not abs_path.startswith('\\\\?\\'):
        return '\\\\?\\' + abs_path
    return abs_path

def build_detection_dataset():
    print(f"Building Master Detection Dataset at:\n{MASTER_OUTPUT_DIR}\n")

    total_converted = 0
    total_copied = 0

    for ds in DATASETS:
        print(f"Processing {ds['name']}...")
        src_imgs = ds['source_images']
        src_lbls = ds['source_labels']
        split = ds['split']

        # Extract base name to create dynamic subfolders (e.g., "DEEPFISH")
        base_folder_name = ds['name'].replace('_Train', '').replace('_Val', '')

        # Build nested paths: images/train/DEEPFISH/
        dest_imgs = os.path.join(MASTER_OUTPUT_DIR, 'images', split, base_folder_name)
        dest_lbls = os.path.join(MASTER_OUTPUT_DIR, 'labels', split, base_folder_name)

        # Create the subdirectories immediately
        os.makedirs(dest_imgs, exist_ok=True)
        os.makedirs(dest_lbls, exist_ok=True)

        if not os.path.exists(make_long_path(src_imgs)) or not os.path.exists(make_long_path(src_lbls)):
            print(f"  [WARNING] Skipping {ds['name']} - Paths not found.")
            continue

        # --- STEP A: PROCESS LABELS ---
        txt_files = glob.glob(os.path.join(src_lbls, "*.txt"))
        for txt_path in txt_files:
            filename = os.path.basename(txt_path)
            dest_txt_path = os.path.join(dest_lbls, filename)

            safe_read_path = make_long_path(txt_path)
            safe_write_path = make_long_path(dest_txt_path)

            with open(safe_read_path, 'r') as infile:
                content = infile.read()
                
            clean_content = re.sub(r'(2\.0|1\.0|0\.0)0 ', r'\1\n0 ', content)
            
            with open(safe_write_path, 'w') as outfile:
                for line in clean_content.split('\n'):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        box_only = " ".join(parts[:5])
                        outfile.write(box_only + "\n")
            total_converted += 1

        # --- STEP B: PROCESS IMAGES ---
        img_files = glob.glob(os.path.join(src_imgs, "*.jpg")) + glob.glob(os.path.join(src_imgs, "*.png"))
        for img_path in img_files:
            filename = os.path.basename(img_path)
            dest_img_path = os.path.join(dest_imgs, filename)
            
            safe_img_read = make_long_path(img_path)
            safe_img_write = make_long_path(dest_img_path)
            
            if not os.path.exists(safe_img_write):
                shutil.copy2(safe_img_read, safe_img_write)
                total_copied += 1

    print("\n" + "="*50)
    print(" DETECTION DATASET BUILD COMPLETE")
    print("="*50)
    print(f" -> Labels Cleaned & Converted: {total_converted}")
    print(f" -> Images Copied:              {total_copied}")
    print("="*50 + "\n")

if __name__ == '__main__':
    build_detection_dataset()