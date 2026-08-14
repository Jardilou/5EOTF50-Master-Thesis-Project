import cv2
import os
import numpy as np
import random
import itertools

def balance_and_extract_dataset(mask_root, image_root, output_root, target_per_class=2000, train_ratio=0.8, holdout_per_class=2):
    """
    Creates a perfectly balanced training dataset.
    - Extracts a completely untouched holdout set.
    - Splits unique remaining images into Train and Val to prevent Data Leakage.
    - Oversamples the Training set to reach the target capacity for rare classes.
    - Downsamples the Training set for majority classes.
    """
    train_dir = os.path.join(output_root, 'train')
    val_dir = os.path.join(output_root, 'val')
    untouched_dir = os.path.join(output_root, 'untouched_test')
    
    mask_folders = [d for d in os.listdir(mask_root) if os.path.isdir(os.path.join(mask_root, d))]
    mask_folders.sort()
    
    # Calculate exactly how many images the Train set needs to hit the 2000 total target
    target_train_count = int((target_per_class - holdout_per_class) * train_ratio)
    # The Val set gets a maximum cap, but is never artificially oversampled
    target_val_cap = int((target_per_class - holdout_per_class) * (1 - train_ratio))

    print(f"Target Total per Class: {target_per_class}")
    print(f"Target Train Count: {target_train_count} | Max Val Count: {target_val_cap}")
    print("="*60)

    for mask_folder_name in mask_folders:
        image_folder_name = mask_folder_name.replace("mask", "fish")
        class_name = mask_folder_name.replace("mask_", "class_")
        
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(untouched_dir, class_name), exist_ok=True)
        
        species_mask_dir = os.path.join(mask_root, mask_folder_name)
        species_image_dir = os.path.join(image_root, image_folder_name)
        
        valid_files = [f for f in os.listdir(species_mask_dir) if f.endswith('.png')]
        
        if not valid_files:
            continue

        random.seed(42)
        random.shuffle(valid_files)
        
        # --- 1. Extract Untouched Holdout ---
        holdout_files = valid_files[:holdout_per_class]
        unique_remaining = valid_files[holdout_per_class:]
        
        # --- 2. Safe Train/Val Split on UNIQUE files ---
        split_idx = int(len(unique_remaining) * train_ratio)
        unique_train_files = unique_remaining[:split_idx]
        unique_val_files = unique_remaining[split_idx:]

        # --- 3. Balance Validation (Cap only, NO oversampling) ---
        final_val_files = unique_val_files[:target_val_cap]

        # --- 4. Balance Training (Cap Majority OR Oversample Minority) ---
        final_train_tasks = []
        
        if len(unique_train_files) == 0:
            print(f"[Warning] {class_name} has no training images. Skipping oversample.")
            continue
            
        if len(unique_train_files) >= target_train_count:
            # Downsample: Just take the exact number we need
            for f in unique_train_files[:target_train_count]:
                final_train_tasks.append((f, "")) # No suffix needed
        else:
            # Oversample: Cycle through the unique files over and over until target is hit
            copy_counter = 1
            for i, f in enumerate(itertools.islice(itertools.cycle(unique_train_files), target_train_count)):
                # If we've looped through the original list, start appending a copy suffix
                suffix = "" if i < len(unique_train_files) else f"_copy{copy_counter}"
                final_train_tasks.append((f, suffix))
                copy_counter += 1

        print(f"{class_name:<10}: {len(final_train_tasks)} Train | {len(final_val_files)} Val | {len(holdout_files)} Untouched")

        # --- 5. Processing Loop ---
        # Combine all tasks: (original_filename, destination_root, save_suffix)
        processing_queue = []
        for f in holdout_files: processing_queue.append((f, untouched_dir, ""))
        for f in final_val_files: processing_queue.append((f, val_dir, ""))
        for f, suffix in final_train_tasks: processing_queue.append((f, train_dir, suffix))

        for filename, target_root, suffix in processing_queue:
            mask_path = os.path.join(species_mask_dir, filename)
            
            base_name = os.path.splitext(filename)[0]
            image_base_name = base_name.replace("mask_", "fish_")
            
            img_path_jpg = os.path.join(species_image_dir, image_base_name + '.jpg')
            img_path_png = os.path.join(species_image_dir, image_base_name + '.png')
            img_path = img_path_jpg if os.path.exists(img_path_jpg) else img_path_png
            
            if not os.path.exists(img_path):
                continue
                
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if img.shape[:2] != mask.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
                
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            masked_img = cv2.bitwise_and(img, img, mask=binary_mask)
            
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
                
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cropped_fish = masked_img[y:y+h, x:x+w]
            
            # Save the file (applying the suffix if it's a generated duplicate)
            final_save_name = f"{image_base_name}{suffix}.jpg"
            output_path = os.path.join(target_root, class_name, final_save_name)
            cv2.imwrite(output_path, cropped_fish)

    print("="*60)
    print("Dataset Generation Complete!")


# ==========================================
# Run the Extraction
# ==========================================
MASK_DIR = r"DATASETS\DATASETS FOR INSTANCE SEGMENTATION\Fish4knowledge\mask_image"     # Contains the 23 species folders
IMAGE_DIR = r"DATASETS\DATASETS FOR INSTANCE SEGMENTATION\Fish4knowledge\fish_image"   # Contains the 23 species folders
OUTPUT_DIR = r"DATASETS\DATASETS FOR INSTANCE SEGMENTATION\Fish4knowledge\Fish_Species_Classifier_Data"

balance_and_extract_dataset(MASK_DIR, IMAGE_DIR, OUTPUT_DIR)