import cv2
import os
import random

# Define your 5 target species (Must match your SPECIES_MAP values)
TARGET_SPECIES = [
    "Dascyllus_reticulatus",
    "Abudefduf_vaigiensis",
    "Pomacentrus_moluccensis",
    "Hemigymnus_melapterus",
    "Scaridae"
]

# Your existing mapping
SPECIES_MAP = {
    "mask_01": "Dascyllus_reticulatus",
    "mask_02": "Plectroglyphidodon_dickii",
    "mask_03": "Chromis_chrysura",
    "mask_04": "Amphiprion_clarkii",
    "mask_05": "Chaetodon_lunulatus",
    "mask_06": "Chaetodon_trifascialis",
    "mask_07": "Myripristis_kuntee",
    "mask_08": "Acanthurus_nigrofuscus",
    "mask_09": "Hemigymnus_fasciatus",
    "mask_10": "Neoniphon_sammara",
    "mask_11": "Abudefduf_vaigiensis",
    "mask_12": "Canthigaster_valentini",
    "mask_13": "Pomacentrus_moluccensis",
    "mask_14": "Zebrasoma_scopas",
    "mask_15": "Hemigymnus_melapterus",
    "mask_16": "Lutjanus_fulvus",
    "mask_17": "Scolopsis_bilineata",
    "mask_18": "Scaridae",
    "mask_19": "Pempheris_vanicolensis",
    "mask_20": "Zanclus_cornutus",
    "mask_21": "Neoglyphidodon_nigroris",
    "mask_22": "Balistapus_undulatus",
    "mask_23": "Siganus_fuscescens"
}

def prepare_dino_dataset(mask_root, image_root, output_root, max_train_per_class=14000, max_val_per_class=0, train_ratio=1, holdout_per_class=5):
    train_dir = os.path.join(output_root, 'train')
    val_dir = os.path.join(output_root, 'val')
    untouched_dir = os.path.join(output_root, 'untouched_test')
    
    mask_folders = sorted([d for d in os.listdir(mask_root) if os.path.isdir(os.path.join(mask_root, d))])
    
    for mask_folder_name in mask_folders:
        # Get the mapping, or default to folder name
        mapped_name = SPECIES_MAP.get(mask_folder_name, mask_folder_name)
        
        # --- NEW LOGIC: Assign "Others" folder ---
        class_name = mapped_name if mapped_name in TARGET_SPECIES else "Others"
        
        image_folder_name = mask_folder_name.replace("mask", "fish")
        
        # Create directories based on final class_name
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(untouched_dir, class_name), exist_ok=True)
        
        species_mask_dir = os.path.join(mask_root, mask_folder_name)
        species_image_dir = os.path.join(image_root, image_folder_name)
        
        valid_files = [f for f in os.listdir(species_mask_dir) if f.endswith('.png')]
        if not valid_files: continue

        random.seed(42)
        random.shuffle(valid_files)
        
        holdout_files = valid_files[:holdout_per_class]
        unique_remaining = valid_files[holdout_per_class:]
        
        split_idx = int(len(unique_remaining) * train_ratio)
        train_files = unique_remaining[:split_idx][:max_train_per_class]
        val_files = unique_remaining[split_idx:][:max_val_per_class]

        # The loop now processes into the correctly consolidated folder
        processing_queue = [(f, untouched_dir) for f in holdout_files] + \
                           [(f, val_dir) for f in val_files] + \
                           [(f, train_dir) for f in train_files]

        for filename, target_root in processing_queue:
            mask_path = os.path.join(species_mask_dir, filename)
            base_name = os.path.splitext(filename)[0]
            image_base_name = base_name.replace("mask_", "fish_")
            
            img_path = os.path.join(species_image_dir, image_base_name + '.jpg')
            if not os.path.exists(img_path): img_path = os.path.join(species_image_dir, image_base_name + '.png')
            if not os.path.exists(img_path): continue
                
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                cropped_fish = img[y:y+h, x:x+w]
                # Save into the consolidated folder structure
                cv2.imwrite(os.path.join(target_root, class_name, image_base_name + ".jpg"), cropped_fish)

    print("="*60)
    print("DINO Dataset Generation with 'Others' Class Complete!")
    
    
    
# ==========================================
# Run the Extraction
# ==========================================
MASK_DIR = r"DATASETS\DATASETS FOR INSTANCE SEGMENTATION\Fish4knowledge\mask_image"    
IMAGE_DIR = r"DATASETS\DATASETS FOR INSTANCE SEGMENTATION\Fish4knowledge\fish_image"   

# Create a brand new output folder so it doesn't mix with your YOLO classification data
OUTPUT_DIR = r"DATASETS\DATASETS FOR SPECIES CLASSIFICATION\Fish4knowledge\Fish_Species_Classifier_Data_DINO_All_Samples"

prepare_dino_dataset(MASK_DIR, IMAGE_DIR, OUTPUT_DIR)