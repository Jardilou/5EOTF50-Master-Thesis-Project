import os
import shutil
import random

def split_yolo_dataset(images_dir, labels_dir, output_dir, train_ratio=0.8):
    """
    Splits images and labels into train and val directories for YOLO.
    """
    # 1. Create the YOLO directory structure
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

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
        print("❌ Error: No matching image-label pairs found. Check your folder paths!")
        return

    # 3. Shuffle for randomness
    random.seed(42) # Keeps the split consistent if you run it multiple times
    random.shuffle(valid_pairs)
    
    # 4. Calculate split index
    split_idx = int(len(valid_pairs) * train_ratio)
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]

    # 5. Copy files to their new homes
    def copy_files(pairs, split_name):
        for img_path, lbl_path, base_name, ext in pairs:
            # Copy Image
            shutil.copy(img_path, os.path.join(output_dir, 'images', split_name, base_name + ext))
            # Copy Label
            shutil.copy(lbl_path, os.path.join(output_dir, 'labels', split_name, base_name + '.txt'))

    print("Copying Training data...")
    copy_files(train_pairs, 'train')
    
    print("Copying Validation data...")
    copy_files(val_pairs, 'val')

    # 6. Print Summary
    print("\n✅ Dataset Split Complete!")
    print("-" * 30)
    print(f"Total pairs found: {len(valid_pairs)}")
    print(f"Training set:      {len(train_pairs)} images ({(train_ratio*100):.0f}%)")
    print(f"Validation set:    {len(val_pairs)} images ({((1-train_ratio)*100):.0f}%)")
    print(f"Output saved to:   {output_dir}")

# ==========================================
# USAGE - UPDATE THESE PATHS
# ==========================================
# Folder where your raw left images are
INPUT_IMAGES = r"INSTANCE SEGMENTATION + POSE ESTIMATION\DATASETS\UMT Dataset\UMT images annotated V1"  

# Folder where the XML conversion script saved your .txt files
INPUT_LABELS = r"INSTANCE SEGMENTATION + POSE ESTIMATION\DATASETS\UMT Dataset\labels\train" 

# The new root folder where the YOLO-ready data will go
OUTPUT_DATASET = r"INSTANCE SEGMENTATION + POSE ESTIMATION\DATASETS\UMT Dataset\UTM_V1_Train_Test_Dual_Strategy" 

split_yolo_dataset(INPUT_IMAGES, INPUT_LABELS, OUTPUT_DATASET, train_ratio=0.8)