import cv2
import os
import numpy as np

def mask_to_yolo_polygon(mask_path, txt_path, class_id=0, min_area=50):
    """
    Converts a binary mask PNG to a YOLO polygon TXT file.
    """
    # 1. Load the mask in grayscale
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error loading image: {mask_path}")
        return

    height, width = mask.shape

    # 2. Threshold to ensure strict binary (0 or 255)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 3. Extract contours (EXTERNAL retrieves only the outer boundaries)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. Open the text file to write the polygon data
    with open(txt_path, 'w') as f:
        for contour in contours:
            # Skip noise/tiny artifacts
            if cv2.contourArea(contour) < min_area:
                continue

            # 5. Flatten the contour array: from [[[x,y]], [[x,y]]] to [x, y, x, y...]
            contour_flat = contour.flatten()

            normalized_points = []
            
            # 6. Normalize coordinates (divide x by width, y by height)
            for i in range(0, len(contour_flat), 2):
                x_norm = contour_flat[i] / width
                y_norm = contour_flat[i+1] / height
                
                # Format to 6 decimal places for cleanliness
                normalized_points.append(f"{x_norm:.6f} {y_norm:.6f}")

            # 7. Write to file if it's a valid polygon (at least 3 points)
            if len(normalized_points) >= 3:
                polygon_str = " ".join(normalized_points)
                f.write(f"{class_id} {polygon_str}\n")

# ==========================================
# Batch Processing Example
# ==========================================
if __name__ == "__main__":
    # Update these paths to match your local directories
    mask_directory = r"DATASETS\DATASETS FOR INSTANCE SEGMENTATION\Deepfish_Segmentation\masks\valid"    # Folder containing .png masks
    output_directory = r"DATASETS\DATASETS FOR INSTANCE SEGMENTATION\Deepfish_Segmentation\mask_labels\valid" # Folder to save .txt files

    os.makedirs(output_directory, exist_ok=True)

    for filename in os.listdir(mask_directory):
        if filename.endswith(".png"):
            mask_path = os.path.join(mask_directory, filename)
            
            # Create a matching .txt filename
            txt_filename = filename.replace(".png", ".txt")
            txt_path = os.path.join(output_directory, txt_filename)
            
            # Convert
            mask_to_yolo_polygon(mask_path, txt_path, class_id=0)
            
    print("Conversion complete!")