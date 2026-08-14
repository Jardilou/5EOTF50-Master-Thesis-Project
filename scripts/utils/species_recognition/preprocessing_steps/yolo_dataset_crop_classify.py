import os
import cv2
import numpy as np
import torch
import random
from collections import defaultdict
from segment_anything import sam_model_registry, SamPredictor

# ==========================================
# 1. Configuration & Setup
# ==========================================

IMAGE_DIR = r"DATASETS\DATASETS FOR SPECIES CLASSIFICATION\UMT For Validation\Images"
LABEL_DIR = r"DATASETS\DATASETS FOR SPECIES CLASSIFICATION\UMT For Validation\updated-labels-with-species"
OUTPUT_DIR = r"DATASETS\DATASETS FOR SPECIES CLASSIFICATION\Fish_Species_Classifier_Data_DINO_All_Samples\train"
SAM_CHECKPOINT = "sam_b.pt"
MODEL_TYPE = "vit_b"

TARGET_SPECIES = {
    "Dascyllus_reticulatus": "Dascyllus_reticulatus",
    "Hemigymnus_melapterus": "Hemigymnus_melapterus",
    "Pomacentrus_moluccensis": "Pomacentrus_moluccensis",
    "Abudefduf_vaigiensis": "Abudefduf_vaigiensis",
    "Scaridae": "Scaridae",
    
}

# UPDATED: Padding as a percentage of the mask's width/height (0.15 = 15% padding on all sides)
PADDING_PERCENTAGE = 0.15 
AUGMENTATIONS_PER_CROP = 3 
SAMPLE_PERCENTAGE = 0.25 

for class_name in TARGET_SPECIES.values():
    os.makedirs(os.path.join(OUTPUT_DIR, class_name), exist_ok=True)

# ==========================================
# 2. Helper & Augmentation Functions
# ==========================================

def yolo_to_bbox(yolo_coords, img_w, img_h):
    x_c, y_c, w, h = yolo_coords
    x_c, y_c = x_c * img_w, y_c * img_h
    w, h = w * img_w, h * img_h
    return np.array([int(x_c - w/2), int(y_c - h/2), int(x_c + w/2), int(y_c + h/2)])

def augment_image_cv2(image):
    aug_img = image.copy()
    if random.random() > 0.5:
        aug_img = cv2.flip(aug_img, 1)
    if random.random() > 0.5:
        alpha = random.uniform(0.8, 1.2) 
        beta = random.uniform(-20, 20)   
        aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=beta)
    if random.random() > 0.5:
        angle = random.uniform(-15, 15)
        scale = random.uniform(0.95, 1.05)
        h, w = aug_img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
        aug_img = cv2.warpAffine(aug_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    if random.random() > 0.7:
        noise = np.zeros(aug_img.shape, np.uint8)
        cv2.randn(noise, 0, random.uniform(10, 30))
        aug_img = cv2.add(aug_img, noise) 
    return aug_img

# ==========================================
# 3. Pass 1: Indexing the Entire Dataset
# ==========================================

print("Pass 1: Scanning labels to map species populations...")
species_registry = defaultdict(list)

for filename in os.listdir(IMAGE_DIR):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    txt_path = os.path.join(LABEL_DIR, os.path.splitext(filename)[0] + ".txt")
    if not os.path.exists(txt_path):
        print(f"Could not find label file for image: {filename}")
        continue
        
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        
    for instance_id, line in enumerate(lines):
        
        data = line.strip().split()
        if not data:
            continue
            
        class_name_key = data[0] 
        print(f"I found: '{class_name_key}'")
        
        if class_name_key in TARGET_SPECIES:
            species_registry[class_name_key].append({
                'filename': filename,
                'instance_id': instance_id,
                'yolo_coords': [float(x) for x in data[1:5]]
            })
# ==========================================
# 4. Pass 2: Exact Stratified Selection
# ==========================================

print("\nPass 2: Selecting exact percentages...")
selected_by_image = defaultdict(list)

for class_name_key, instances in species_registry.items():
    total_found = len(instances)
    num_to_select = max(1, int(total_found * SAMPLE_PERCENTAGE))
    print(f" - {TARGET_SPECIES[class_name_key]}: Found {total_found} instances. Selecting exactly {num_to_select}.")
    
    random.shuffle(instances)
    chosen_instances = instances[:num_to_select]
    
    for inst in chosen_instances:
        selected_by_image[inst['filename']].append({
            'class_name_key': class_name_key,
            'instance_id': inst['instance_id'],
            'yolo_coords': inst['yolo_coords']
        })

# ==========================================
# 5. Execution: SAM Processing & Augmentation
# ==========================================

print("\nLoading SAM model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=device)
predictor = SamPredictor(sam)

print(f"\nProcessing {len(selected_by_image)} unique images containing selected fish...")

for filename, targets in selected_by_image.items():
    img_path = os.path.join(IMAGE_DIR, filename)
    image = cv2.imread(img_path)
    if image is None:
        continue
        
    img_h, img_w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    
    for target in targets:
        class_name_key = target['class_name_key']
        instance_id = target['instance_id']
        input_box = yolo_to_bbox(target['yolo_coords'], img_w, img_h)
        
        masks, _, _ = predictor.predict(
            point_coords=None, point_labels=None, box=input_box, multimask_output=False
        )
        
        mask = masks[0]
        y_indices, x_indices = np.where(mask)
        if len(x_indices) == 0 or len(y_indices) == 0:
            continue
            
        m_x_min, m_x_max = np.min(x_indices), np.max(x_indices)
        m_y_min, m_y_max = np.min(y_indices), np.max(y_indices)
        
        fish_w = m_x_max - m_x_min
        fish_h = m_y_max - m_y_min
        
        pad_w = int(fish_w * PADDING_PERCENTAGE)
        pad_h = int(fish_h * PADDING_PERCENTAGE)
        
        c_x_min = max(0, m_x_min - pad_w)
        c_y_min = max(0, m_y_min - pad_h)
        c_x_max = min(img_w, m_x_max + pad_w)
        c_y_max = min(img_h, m_y_max + pad_h)
        
        cropped_fish = image[c_y_min:c_y_max, c_x_min:c_x_max]
        if cropped_fish.size == 0:
            continue
            
        species_name = TARGET_SPECIES[class_name_key]
        base_save_name = f"{os.path.splitext(filename)[0]}_inst{instance_id}"
        save_dir = os.path.join(OUTPUT_DIR, species_name)
        
        cv2.imwrite(os.path.join(save_dir, f"{base_save_name}_clean.jpg"), cropped_fish)
        
        for aug_idx in range(AUGMENTATIONS_PER_CROP):
            aug_img = augment_image_cv2(cropped_fish)
            cv2.imwrite(os.path.join(save_dir, f"{base_save_name}_aug{aug_idx}.jpg"), aug_img)

print("\nProcessing complete!")
