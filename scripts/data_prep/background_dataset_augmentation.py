import os
import shutil
import math

def add_background_images(source_folders, empty_folder, out_img_dir, out_label_dir, target_percentage=0.05):
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    # 1. Count the total annotated images across all source folders
    total_annotated_images = 0
    for folder in source_folders:
        if os.path.exists(folder):
            images = [f for f in os.listdir(folder) if f.lower().endswith(valid_exts)]
            total_annotated_images += len(images)
        else:
            print(f"Warning: Source folder '{folder}' not found. Skipping.")
            
    print(f"Total annotated images found: {total_annotated_images}")
    
    # 2. Calculate the exact 5% target
    target_bg_count = math.ceil(total_annotated_images * target_percentage)
    print(f"Target background images to add ({target_percentage*100}%): {target_bg_count}")
    
    if target_bg_count == 0:
        print("Not enough annotated images to calculate 5%. Exiting.")
        return

    # 3. Analyze the empty images folder
    if not os.path.exists(empty_folder):
        print(f"Error: Empty images folder '{empty_folder}' not found.")
        return
        
    available_bgs = sorted([f for f in os.listdir(empty_folder) if f.lower().endswith(valid_exts)])
    total_available_bgs = len(available_bgs)
    print(f"Total available empty images to choose from: {total_available_bgs}")
    
    if total_available_bgs == 0:
        print("Error: No images found in the empty images folder.")
        return
        
    # Safety Check: What if we don't have enough empty images to hit 5%?
    if target_bg_count > total_available_bgs:
        print(f"Warning: You need {target_bg_count} images but only have {total_available_bgs}. Using all of them.")
        target_bg_count = total_available_bgs

    # 4. The Math: Select EVENLY SPACED images
    selected_bg_images = []
    for i in range(target_bg_count):
        # This formula guarantees an even spread across the entire list
        idx = int(i * total_available_bgs / target_bg_count)
        selected_bg_images.append(available_bgs[idx])

    # 5. Copy them over and generate the empty .txt files
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)
    
    print("Copying images and generating empty labels...")
    copied_count = 0
    
    for img_name in selected_bg_images:
        src_img_path = os.path.join(empty_folder, img_name)
        
        # Add a prefix so you always know which images are your background ones
        safe_name = f"bg_negative_{img_name}"
        dst_img_path = os.path.join(out_img_dir, safe_name)
        
        # Create the matching .txt file path
        base_name = os.path.splitext(safe_name)[0]
        dst_label_path = os.path.join(out_label_dir, f"{base_name}.txt")
        
        # Action: Copy the image
        shutil.copy(src_img_path, dst_img_path)
        
        # Action: Create the blank 0-byte text file
        with open(dst_label_path, 'w') as f:
            pass 
            
        copied_count += 1

    print(f"\nSuccess! Added {copied_count} evenly spaced background images.")
    print(f"Images saved to: {out_img_dir}")
    print(f"Empty labels saved to: {out_label_dir}")


# ==========================================
# --- EXECUTION PIPELINE ---
# ==========================================
if __name__ == '__main__':
    # List all the folders that currently contain your annotated fish images
    ANNOTATED_FOLDERS = [
        r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\DATASETS\BézierFusion Dataset 1\CORRECT IMAGES\train test split\images\train",
        r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\DATASETS\BézierFusion Dataset 1\CORRECT IMAGES\train test split\images\val",
        r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\DATASETS\Bézierfusion Dataset 2 Reannotated V4\Train Test\images\train",
        r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\DATASETS\Bézierfusion Dataset 2 Reannotated V4\Train Test\images\val",
        r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\DATASETS\DeepFish\First_batch_Train_Test\images\train",
        r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\DATASETS\DeepFish\First_batch_Train_Test\images\val"
    ]
    
    # The folder where all your raw, empty reef/stick images live
    EMPTY_SOURCE_FOLDER = r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\DATASETS\DeepFish\Localization\images\empty"
    
    # Where you want the selected 5% to be placed (Usually right into your training dataset)
    OUTPUT_IMAGES_DIR = r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\DATASETS\DeepFish\Empty_training_dataset\images\train"
    OUTPUT_LABELS_DIR = r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\DATASETS\DeepFish\Empty_training_dataset\labels\train"
    
    add_background_images(
        source_folders=ANNOTATED_FOLDERS,
        empty_folder=EMPTY_SOURCE_FOLDER,
        out_img_dir=OUTPUT_IMAGES_DIR,
        out_label_dir=OUTPUT_LABELS_DIR,
        target_percentage=0.09
    )