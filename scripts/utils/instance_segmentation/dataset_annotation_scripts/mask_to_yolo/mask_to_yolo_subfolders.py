import cv2
import os
import numpy as np
import shutil

def mask_to_yolo_polygon(mask_path, txt_path, class_id, min_area=50):
    """
    Converts a binary mask PNG to a YOLO polygon TXT file.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error loading: {mask_path}")
        return False

    height, width = mask.shape
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_polygons = 0
    with open(txt_path, 'w') as f:
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue

            contour_flat = contour.flatten()
            normalized_points = []
            
            for i in range(0, len(contour_flat), 2):
                x_norm = contour_flat[i] / width
                y_norm = contour_flat[i+1] / height
                normalized_points.append(f"{x_norm:.6f} {y_norm:.6f}")

            if len(normalized_points) >= 3:
                polygon_str = " ".join(normalized_points)
                f.write(f"{class_id} {polygon_str}\n")
                valid_polygons += 1
                
    return valid_polygons > 0

def process_species_dataset(root_mask_dir, output_labels_dir, root_image_dir=None, output_images_dir=None):
    """
    Iterates through species subdirectories, extracts the unique image ID,
    and saves both the YOLO labels and copied images as 'fish_01_XXX'.
    """
    os.makedirs(output_labels_dir, exist_ok=True)
    if root_image_dir and output_images_dir:
        os.makedirs(output_images_dir, exist_ok=True)

    # 1. Dynamically map folder names to Class IDs
    species_folders = [d for d in os.listdir(root_mask_dir) if os.path.isdir(os.path.join(root_mask_dir, d))]
    species_folders.sort() 

    class_mapping = {species: idx for idx, species in enumerate(species_folders)}
    
    print("========================================")
    print("YAML Class Mapping (Copy this for data.yaml):")
    print(f"nc: {len(class_mapping)}")
    print("names:")
    for species, idx in class_mapping.items():
        print(f"  {idx}: {species}")
    print("========================================\n")

    # 2. Process each folder
    for species in species_folders:
        species_mask_dir = os.path.join(root_mask_dir, species)
        class_id = class_mapping[species]
        
        # Determine the matching source image folder (e.g., "mask_01" -> "fish_01")
        image_folder_name = species.replace("mask", "fish")
        
        print(f"Processing species: {species} (Class ID: {class_id})...")
        
        for filename in os.listdir(species_mask_dir):
            if not filename.endswith(".png"):
                continue

            # --- Extract the ID and format the new name ---
            # Example filename: "mask_000000009598_05281.png"
            # Extract "000000009598_05281"
            identifier = filename.replace("mask_", "").replace(".png", "")
            
            # Create target name: "fish_01_000000009598_05281"
            new_base_name = f"{image_folder_name}_{identifier}"
            
            # --- Save the YOLO TXT Label ---
            mask_path = os.path.join(species_mask_dir, filename)
            txt_path = os.path.join(output_labels_dir, new_base_name + ".txt")
            mask_to_yolo_polygon(mask_path, txt_path, class_id=class_id)

            # --- Find and Copy the Original Image ---
            if root_image_dir and output_images_dir:
                # The original image we are looking for is named "fish_000000009598_05281.jpg"
                original_img_base = f"fish_{identifier}"
                
                img_extensions = ['.jpg', '.jpeg', '.png']
                image_found = False
                
                for ext in img_extensions:
                    original_img_path = os.path.join(root_image_dir, image_folder_name, original_img_base + ext)
                    
                    if os.path.exists(original_img_path):
                        # Save the copied image with the NEW name so it matches the .txt file perfectly
                        new_img_path = os.path.join(output_images_dir, new_base_name + ext)
                        shutil.copy(original_img_path, new_img_path)
                        image_found = True
                        break
                
                if not image_found:
                    print(f"  [Warning] Could not find original image for mask: {filename}")

    print("\nConversion Complete!")

# ==========================================
# Execution
# ==========================================
if __name__ == "__main__":
    ROOT_MASK_DIR = r"DATASETS\DATASETS FOR INSTANCE SEGMENTATION\Fish4knowledge\mask_image"    
    OUTPUT_LABELS_DIR = r"DATASETS\DATASETS FOR INSTANCE SEGMENTATION\Fish4knowledge\yolo_labels" 
    
    ROOT_IMAGE_DIR = r"DATASETS\DATASETS FOR INSTANCE SEGMENTATION\Fish4knowledge\fish_image"    
    OUTPUT_IMAGES_DIR = r"DATASETS\DATASETS FOR INSTANCE SEGMENTATION\Fish4knowledge\yolo_images" 

    process_species_dataset(ROOT_MASK_DIR, OUTPUT_LABELS_DIR, ROOT_IMAGE_DIR, OUTPUT_IMAGES_DIR)