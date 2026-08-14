"""
================================================================================
Script: YOLO-Pose Dataset Generator and Verification Filter (Single Keypoint)
================================================================================

Description:
    A script designed to generate a strict, verified YOLO-Pose dataset by 
    fusing binary segmentation masks with CVAT XML keypoint annotations. 
    This specific iteration is configured for single-keypoint extraction 
    (Head only) and acts as a rigorous quality control filter. It actively 
    discards compromised data pairs before assembling a clean training directory.

    Key Functions:
    1. XML Parsing: Extracts ground-truth head keypoint coordinates from 
       CVAT XML files.
    2. Mask Processing: Reads grayscale binary masks, extracts external contours, 
       and computes the bounding box for each distinct target.
    3. Strict Verification Pipeline: Automatically drops images or specific 
       targets that fail validation checks (e.g., missing XML head data, 
       blank/empty masks, or masks where the nearest head coordinate exceeds 
       the acceptable Euclidean distance threshold).
    4. Data Normalization: Converts the absolute pixel coordinates of the 
       bounding box and the verified head keypoint into the 0.0 - 1.0 YOLO format.
    5. Clean Dataset Assembly: Populates the output directories exclusively 
       with verified, successfully matched image-label pairs, leaving corrupted 
       source files behind.

Author:
    Louis Jardinet
    Universiti Teknologi Petronas

Date:
    January 2026 - June 2026

Dependencies:
    - Python standard libraries: os, shutil, xml.etree.ElementTree
    - External packages: opencv-python (cv2), numpy, scipy

Inputs:
    - img_dir: Directory containing the source images.
    - mask_dir: Directory containing binary image masks (.png or .jpg).
    - xml_path: Path to the XML file containing the manual head keypoint annotations.

Outputs:
    - output_pose_dir: Directory where the verified YOLO-Pose .txt labels are saved.
    - output_img_dir: Directory where the corresponding verified images are copied.
================================================================================
"""

import os
import cv2
import numpy as np
import shutil
import xml.etree.ElementTree as ET
from scipy.spatial import distance

def get_all_heads(img_name, xml_path):
    """Parses CVAT XML for all Head points in a given image."""
    if not os.path.exists(xml_path):
        return []
        
    tree = ET.parse(xml_path)
    root = tree.getroot()
    heads = []
    
    # Strip extensions to match robustly, just in case CVAT dropped the .jpg
    base_name = os.path.splitext(img_name)[0]
    
    for image in root.findall('image'):
        if base_name in image.attrib.get('name'):
            for point_data in image.findall('.//points'):
                if point_data.attrib.get('label').lower() == 'head':
                    coord_string = point_data.attrib.get('points')
                    x, y = map(float, coord_string.split(','))
                    heads.append((x, y))
            return heads
    return heads

def create_deepfish_pose_dataset(img_dir, mask_dir, xml_path, output_pose_dir, output_img_dir):
    # Create the clean output directories
    os.makedirs(output_pose_dir, exist_ok=True)
    os.makedirs(output_img_dir, exist_ok=True)
    
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg'))]
    
    success_count = 0
    missing_heads = 0
    missing_masks = 0

    print("Starting Strict Verification Filter...")

    for mask_filename in mask_files:
        img_filename = mask_filename.replace('.png', '.jpg') 
        img_path = os.path.join(img_dir, img_filename)
        mask_path = os.path.join(mask_dir, mask_filename)
        
        pose_output_path = os.path.join(output_pose_dir, os.path.splitext(img_filename)[0] + '.txt')
        clean_img_output_path = os.path.join(output_img_dir, img_filename)

        # Verification 1: Does the raw image exist?
        if not os.path.exists(img_path): 
            continue

        # Verification 2: Does the CVAT XML have Head annotations for this image?
        all_heads_xy = get_all_heads(img_filename, xml_path)
        if not all_heads_xy:
            missing_heads += 1
            continue # Discard image: No head annotation found

        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        heads_array = np.array(all_heads_xy)
        pose_lines = []

        # Load the DeepFish Binary Mask
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, binary_mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Verification 3: Does the mask actually contain a valid shape?
        if not contours:
            missing_masks += 1
            continue # Discard image: Mask file exists, but it's completely blank/black

        for contour in contours:
            if cv2.contourArea(contour) < 100:
                continue

            poly = contour.reshape(-1, 2)
            x, y, w, h = cv2.boundingRect(contour)
            
            norm_cx = (x + (w / 2.0)) / img_w
            norm_cy = (y + (h / 2.0)) / img_h
            norm_w = w / img_w
            norm_h = h / img_h

            # Verification 4: Does the Head physically belong to this mask?
            dist_matrix = distance.cdist(heads_array, poly, 'euclidean')
            min_dist_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            closest_head_idx = min_dist_idx[0]
            
            if np.min(dist_matrix) > 300:
                continue # Discard mask: The closest head is too far away

            true_head_x, true_head_y = heads_array[closest_head_idx]
            norm_hx = true_head_x / img_w
            norm_hy = true_head_y / img_h

            pose_str = f"0 {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f} {norm_hx:.6f} {norm_hy:.6f} 2"
            pose_lines.append(pose_str)

        # 5. FINAL VERIFICATION: Write files ONLY if at least one valid fish was found
        if pose_lines:
            # Write the YOLO text file
            with open(pose_output_path, 'w') as f:
                f.write("\n".join(pose_lines))
                
            # COPY the raw image into the clean training folder
            if not os.path.exists(clean_img_output_path):
                shutil.copy(img_path, clean_img_output_path)
                
            success_count += 1

    print("-" * 40)
    print(f"Filter Complete!")
    print(f"[X] Images discarded due to missing Heads: {missing_heads}")
    print(f"[X] Images discarded due to blank Masks: {missing_masks}")
    print(f"[+] Total Verified Images Copied to Dataset: {success_count}")


if __name__ == '__main__':
    # Add your actual DeepFish paths here
    create_deepfish_pose_dataset(
        img_dir=r"INSTANCE SEGMENTATION + POSE ESTIMATION\DATASETS\DEEPFISH Dataset\Segmentation\images\valid", 
        mask_dir=r"INSTANCE SEGMENTATION + POSE ESTIMATION\DATASETS\DEEPFISH Dataset\Segmentation\masks\valid", # The folder with the black & white fish silhouettes
        xml_path=r"POSE ESTIMATION\XML ANNOTATIONS\annotations_DEEPFISH_V1.xml",
        output_pose_dir=r"INSTANCE SEGMENTATION + POSE ESTIMATION\DATASETS\DEEPFISH Dataset\Train_Test_Dual_Strategy\labels", # Output folder
        output_img_dir=r"INSTANCE SEGMENTATION + POSE ESTIMATION\DATASETS\DEEPFISH Dataset\Train_Test_Dual_Strategy\images" # Output folder for the raw images that correspond to the new .txt files
    )