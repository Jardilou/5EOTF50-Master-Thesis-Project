"""
================================================================================
Script: YOLO Segmentation Large and Small Objects Filter
================================================================================

Description:
    A script to filter YOLO-format instance segmentation datasets by removing 
    objects that are too large or too small. The script uses primary annotations 
    (Class 0) to establish size baselines. Secondary annotations (Class 1) are 
    then evaluated and removed if they fall outside the acceptable size range, 
    overlap with primary targets, or exhibit non-target shapes.

    Key Functions:
    1. Size Baseline Extraction: Scans the dataset to determine the minimum 
       and maximum areas of the primary objects (Class 0).
    2. Size Filtering: Removes secondary objects (Class 1) that are smaller 
       than the global minimum or larger than the maximum allowed area.
    3. Overlap Filtering: Calculates Intersection over Union (IoU) to remove 
       secondary objects that overlap with primary objects.
    4. Shape Filtering: Computes aspect ratio and solidity to remove objects 
       with irregular shapes, such as circular rocks or jagged background structures.

Author:
    Louis Jardinet
    Universiti Teknologi Petronas

Date:
    January 2026 - June 2026

Dependencies:
    - Python standard libraries: os
    - External packages: opencv-python (cv2), numpy

Inputs:
    - raw_dir: Directory containing the unfiltered YOLO .txt label files.
    - img_dir: Directory containing the corresponding image files.
    - overlap_thresh: Maximum allowed IoU (default is 0.1) before a polygon 
      is dropped.

Outputs:
    - clean_dir: Directory where the size-filtered YOLO .txt files are saved.
================================================================================
"""

import os
import cv2
import numpy as np

def get_polygon_area(coords):
    """Calculates the normalized area of a polygon using the Shoelace formula."""
    x, y = coords[0::2], coords[1::2]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def get_polygon_bbox(coords):
    """Extracts a normalized bounding box [x_min, y_min, x_max, y_max]."""
    x, y = coords[0::2], coords[1::2]
    return [min(x), min(y), max(x), max(y)]

def calculate_iou(boxA, boxB):
    """Calculates Intersection over Union (IoU) for overlapping."""
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-9)

def get_morphology_metrics(coords, img_width, img_height):
    """Calculates Rotated Aspect Ratio and Solidity of a polygon."""
    pts = coords.reshape(-1, 2)
    pts[:, 0] *= img_width
    pts[:, 1] *= img_height
    pts = np.float32(pts)

    rect = cv2.minAreaRect(pts)
    width, height = rect[1]
    if width == 0 or height == 0: return 1.0, 0.0
    aspect_ratio = max(width, height) / min(width, height)

    area = cv2.contourArea(pts)
    hull = cv2.convexHull(pts)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0: return aspect_ratio, 0.0
    solidity = area / hull_area

    return aspect_ratio, solidity

def filter_dataset_unified(raw_dir, clean_dir, img_dir, overlap_thresh=0.1):
    os.makedirs(clean_dir, exist_ok=True)
    
    # PASS 1: Establish the Global Limits based on Class 0
    global_min_area = float('inf')
    global_max_area = 0.0
    
    for filename in os.listdir(raw_dir):
        if not filename.endswith(".txt"): continue
        with open(os.path.join(raw_dir, filename), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                if int(parts[0]) == 0:
                    area = get_polygon_area(np.array(parts[1:], dtype=float))
                    if area < global_min_area: global_min_area = area
                    if area > global_max_area: global_max_area = area
                        
    if global_min_area == float('inf'): global_min_area = 0.0
    
    print(f"Global Constraints -> Max: {global_max_area:.6f} | Min: {global_min_area:.6f}")
    print("-" * 50)
    
    # MORPHOLOGY TUNING PARAMETERS
    MIN_ASPECT_RATIO = 1.6  # Drops circular/square rocks
    MIN_SOLIDITY = 0.80     # Drops jagged/branching corals
    
    total_kept = 0
    total_dropped = 0
    
    # PASS 2: Filter with all 5 rules
    for filename in os.listdir(raw_dir):
        if not filename.endswith(".txt"): continue
        
        img_name = filename.replace(".txt", ".jpg")
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path): continue
        
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]

        filepath = os.path.join(raw_dir, filename)
        clean_filepath = os.path.join(clean_dir, filename)
        
        with open(filepath, 'r') as f: lines = f.readlines()
            
        class_0_lines = []
        class_0_bboxes = []
        local_max_area = 0.0
        
        # Step A: Extract Class 0 and define the Local Maximum for THIS frame
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            if int(parts[0]) == 0:
                class_0_lines.append(line.strip())
                coords = np.array(parts[1:], dtype=float)
                area = get_polygon_area(coords)
                if area > local_max_area: local_max_area = area
                class_0_bboxes.append(get_polygon_bbox(coords))
                
        # If there are no measurable fish, skip adding background noise
        if local_max_area == 0.0:
            if class_0_lines:
                with open(clean_filepath, 'w') as f: f.write("\n".join(class_0_lines))
            continue
            
        # Step B: Filter Class 1
        valid_class_1_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            if int(parts[0]) == 1:
                coords = np.array(parts[1:], dtype=float)
                area = get_polygon_area(coords)
                
                # Rule 1: Frame-Specific Local Maximum
                if area > local_max_area:
                    total_dropped += 1
                    continue
                    
                # Rule 2: Dataset Global Maximum (Safety Net)
                if area > global_max_area:
                    total_dropped += 1
                    continue
                    
                # Rule 3: Dataset Global Minimum
                if area < global_min_area:
                    total_dropped += 1
                    continue
                    
                # Rule 4: Overlap Constraint
                bbox = get_polygon_bbox(coords)
                is_overlap = any(calculate_iou(bbox, c0_box) > overlap_thresh for c0_box in class_0_bboxes)
                if is_overlap:
                    total_dropped += 1
                    continue
                    
                # Rule 5: Morphology (Shape) Constraints
                aspect_ratio, solidity = get_morphology_metrics(coords, img_w, img_h)
                if aspect_ratio < MIN_ASPECT_RATIO or solidity < MIN_SOLIDITY:
                    total_dropped += 1
                    continue
                    
                # Passes all 5 strict checks
                valid_class_1_lines.append(line.strip())
                total_kept += 1
                
        # Write clean file
        final_lines = class_0_lines + valid_class_1_lines
        if final_lines:
            with open(clean_filepath, 'w') as f:
                f.write("\n".join(final_lines))
                
    print("Filtering Complete.")
    print(f"Kept: {total_kept} | Dropped: {total_dropped}")


                




raw_sam_output_path = r"DATASETS FOR INSTANCE SEGMENTATION\UMT_dataset\DINO+SAM_labels Subset V2"
filtered_sam_output_path = r'DATASETS FOR INSTANCE SEGMENTATION\UMT_dataset\DINO+SAM_filtered_labels Subset V2'
img_directory = r"DATASETS FOR INSTANCE SEGMENTATION\UMT_dataset\UMT images for annotation V1 CLEANED Subset V2"

if __name__ == '__main__':
    filter_dataset_unified(raw_sam_output_path, filtered_sam_output_path, img_directory, overlap_thresh=0.1)