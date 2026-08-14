import os
import shutil

def remove_unpaired_images(images_dir, labels_dir, skipped_dir):
    os.makedirs(skipped_dir, exist_ok=True)
    
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_exts)]
    
    moved_count = 0
    
    print("🔍 Scanning for images without labels...")
    
    for img_filename in images:
        # Get the base name (e.g., "fish_01" from "fish_01.jpg")
        base_name = os.path.splitext(img_filename)[0]
        expected_label = f"{base_name}.txt"
        label_path = os.path.join(labels_dir, expected_label)
        
        # If the label does not exist, move the image out of the training folder!
        if not os.path.exists(label_path):
            src_img_path = os.path.join(images_dir, img_filename)
            dst_img_path = os.path.join(skipped_dir, img_filename)
            
            shutil.move(src_img_path, dst_img_path)
            moved_count += 1
            
    print(f"Clean up complete! Moved {moved_count} unannotated images to: {skipped_dir}")

# --- EXECUTION ---
remove_unpaired_images(
    images_dir=r"DATASETS\DeepFish\First_batch_training",
    labels_dir=r"yolo_labels_overlap_safe_deepfish_V1",
    skipped_dir=r"DATASETS\DeepFish\skipped_unannotated"
)