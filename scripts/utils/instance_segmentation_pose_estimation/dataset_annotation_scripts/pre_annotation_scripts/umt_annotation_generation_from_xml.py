"""
================================================================================
Script: YOLO-Pose Dataset Generator (Text-Polygon to Single Keypoint)
================================================================================

Description:
    A script to generate a YOLO-Pose dataset consisting of bounding boxes and 
    a single anatomical keypoint (the head). The script fuses existing YOLO-format 
    instance segmentation labels (.txt polygons) with head coordinates extracted 
    from CVAT XML files. It matches the shape data to the keypoint data and 
    reformats them into the standardized YOLO-Pose structure.

    Key Functions:
    1. XML Parsing: Extracts ground-truth head keypoint coordinates from 
       CVAT XML files.
    2. Polygon Reconstruction: De-normalizes YOLO instance segmentation text 
       coordinates back into absolute pixel polygons to compute accurate bounding boxes.
    3. Keypoint Matching: Links the reconstructed polygon to the nearest XML 
       head coordinate using a Euclidean distance matrix, applying a spatial 
       threshold (300 pixels) to ignore unmatched or hallucinated objects.
    4. Data Normalization: Converts the computed absolute bounding box and the 
       matched head coordinate into normalized YOLO values (0.0 - 1.0).
    5. Formatting: Outputs the data in the YOLO-Pose string format: 
       Class, Center_X, Center_Y, Width, Height, Keypoint_X, Keypoint_Y, Visibility.

Author:
    Louis Jardinet
    Universiti Teknologi Petronas

Date:
    January 2026 - June 2026

Dependencies:
    - Python standard libraries: os, xml.etree.ElementTree
    - External packages: opencv-python (cv2), numpy, scipy

Inputs:
    - img_dir: Directory containing the source images.
    - seg_label_dir: Directory containing YOLO-format instance segmentation (.txt) polygons.
    - xml_path: Path to the XML file containing the manual head keypoint annotations.

Outputs:
    - output_pose_dir: Directory where the generated YOLO-Pose .txt labels are saved.
================================================================================
"""

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial import distance

def get_all_heads(img_name, xml_path):
    """Parses CVAT XML for all Head points in a given image."""
    if not os.path.exists(xml_path):
        return []
        
    tree = ET.parse(xml_path)
    root = tree.getroot()
    heads = []
    
    for image in root.findall('image'):
        if img_name in image.attrib.get('name'):
            for point_data in image.findall('.//points'):
                if point_data.attrib.get('label').lower() == 'head':
                    coord_string = point_data.attrib.get('points')
                    x, y = map(float, coord_string.split(','))
                    heads.append((x, y))
            return heads
    return heads

def create_pose_dataset(img_dir, seg_label_dir, xml_path, output_pose_dir):
    os.makedirs(output_pose_dir, exist_ok=True)
    seg_files = [f for f in os.listdir(seg_label_dir) if f.endswith('.txt')]
    
    success_count = 0

    for seg_filename in seg_files:
        img_filename = seg_filename.replace('.txt', '.jpg')
        img_path = os.path.join(img_dir, img_filename)
        seg_path = os.path.join(seg_label_dir, seg_filename)
        pose_output_path = os.path.join(output_pose_dir, seg_filename)

        if not os.path.exists(img_path): continue

        # Get image dimensions for YOLO normalization
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        
        # 1. Fetch CVAT Heads
        all_heads_xy = get_all_heads(img_filename, xml_path)
        if not all_heads_xy:
            continue # If there are no heads, we skip training pose on this image

        with open(seg_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
            
        pose_lines = []
        heads_array = np.array(all_heads_xy)

        for line in lines:
            parts = line.split()
            class_id = int(parts[0])
            if class_id != 0: continue # Only train on measurable fish
                
            # 2. Rebuild the absolute polygon from normalized SAM text
            coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
            coords[:, 0] *= img_w
            coords[:, 1] *= img_h
            poly = np.int32(coords)

            # 3. Calculate the Bounding Box from the Polygon
            x, y, w, h = cv2.boundingRect(poly)
            
            # Normalize bounding box for YOLO
            norm_cx = (x + (w / 2.0)) / img_w
            norm_cy = (y + (h / 2.0)) / img_h
            norm_w = w / img_w
            norm_h = h / img_h

            # 4. Match the closest Head to this Polygon
            # We measure the distance from all heads to all vertices of the polygon
            dist_matrix = distance.cdist(heads_array, poly, 'euclidean')
            
            # Find the absolute closest head to this specific mask
            min_dist_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            closest_head_idx = min_dist_idx[0]
            
            # 300px capture radius (ignores hallucinated rocks)
            if np.min(dist_matrix) > 300:
                continue 

            true_head_x, true_head_y = heads_array[closest_head_idx]

            # Normalize the Head coordinate
            norm_hx = true_head_x / img_w
            norm_hy = true_head_y / img_h

            # 5. Format the YOLO-Pose string (Class CX CY W H Px Py Visibility)
            pose_str = f"0 {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f} {norm_hx:.6f} {norm_hy:.6f} 2"
            pose_lines.append(pose_str)

        # 6. Save the new pose training file
        if pose_lines:
            with open(pose_output_path, 'w') as f:
                f.write("\n".join(pose_lines))
            success_count += 1

    print(f"[+] Successfully generated YOLO-Pose dataset for {success_count} images.")

if __name__ == '__main__':
    # Add your actual paths here
    create_pose_dataset(
        img_dir=r"DATASETS FOR INSTANCE SEGMENTATION\UMT_dataset\UMT images for annotation V1 CLEANED",
        seg_label_dir=r"DATASETS FOR INSTANCE SEGMENTATION\UMT_dataset\DINO+SAM_filtered_labels",
        xml_path=r"POSE ESTIMATION\XML ANNOTATIONS\UMT Dataset V1 Refixed\annotations_FIXED.xml",
        output_pose_dir=r"INSTANCE SEGMENTATION + POSE ESTIMATION\DATASETS\labels\train" # Output folder for the new .txt files
    )