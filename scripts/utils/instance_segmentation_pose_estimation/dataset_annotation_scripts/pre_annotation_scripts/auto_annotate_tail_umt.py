"""
================================================================================
Script: YOLO-Pose Auto-Annotation Engine (Skeletonization Method)
================================================================================

Description:
    A script to automatically generate YOLO-Pose annotations (bounding box, 
    head keypoint, and tail keypoint) by combining existing instance segmentation 
    polygons with XML head coordinate data. The script rebuilds binary masks 
    from YOLO text files, extracts their morphological skeletons, and uses 
    graph traversal to mathematically locate the tail extremity.

    Key Functions:
    1. XML Parsing: Extracts ground-truth head keypoint coordinates from 
       CVAT XML files.
    2. Mask Reconstruction: De-normalizes YOLO polygon coordinates to generate 
       solid binary masks in memory, applying morphological closing and blurring 
       to prevent edge fragmentation.
    3. Skeletonization & Anchoring: Extracts a 1-pixel wide medial axis (spine) 
       from the mask and anchors it to the nearest verified XML head coordinate.
    4. Pathfinding (Dijkstra): Traverses the skeleton matrix starting from the 
       head anchor to locate the furthest connected pixel, designating it as the tail.
    5. YOLO-Pose Formatting: Computes the bounding box and normalizes all coordinates 
       (bounding box, head, tail) into the standard YOLO-Pose format.

Author:
    Louis Jardinet
    Universiti Teknologi Petronas

Date:
    January 2026 - June 2026

Dependencies:
    - Python standard libraries: os, shutil, heapq, xml.etree.ElementTree
    - External packages: opencv-python (cv2), numpy, scipy

Inputs:
    - img_dir: Directory containing the source images.
    - seg_dir: Directory containing YOLO-format instance segmentation (.txt) polygons.
    - xml_path: Path to the XML file containing the manual head keypoint annotations.

Outputs:
    - output_pose_dir: Directory where the generated YOLO-Pose .txt labels are saved.
    - output_img_dir: Directory where the successfully processed images are copied 
      for direct use in model training.
================================================================================
"""

import os
import cv2
import numpy as np
import shutil
import heapq
import xml.etree.ElementTree as ET
from scipy.spatial import distance

# --- 1. XML Parsing ---
def get_all_heads(img_name, xml_path):
    if not os.path.exists(xml_path): return []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    heads = []
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

# --- 2. Pathfinding Logic ---
def run_dijkstra(skeleton, start_node):
    queue = [(0.0, start_node)]
    distances = {start_node: 0.0}
    furthest_node = start_node
    max_dist = 0.0

    while queue:
        current_dist, current_node = heapq.heappop(queue)
        if current_dist > max_dist:
            max_dist = current_dist
            furthest_node = current_node
            
        y, x = current_node
        for dy, dx in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1] and skeleton[ny, nx] > 0:
                neighbor = (ny, nx)
                step_cost = 1.414 if (dx != 0 and dy != 0) else 1.0
                if neighbor not in distances or current_dist + step_cost < distances[neighbor]:
                    distances[neighbor] = current_dist + step_cost
                    heapq.heappush(queue, (current_dist + step_cost, neighbor))
    return furthest_node

# --- 3. The UMT Auto-Labeler Engine ---
def auto_annotate_umt_dataset(img_dir, seg_dir, xml_path, output_pose_dir, output_img_dir):
    os.makedirs(output_pose_dir, exist_ok=True)
    os.makedirs(output_img_dir, exist_ok=True)
    
    # Notice we are now looking for the SAM .txt segmentation files instead of .png masks
    seg_files = [f for f in os.listdir(seg_dir) if f.endswith('.txt')]
    success_count = 0

    print("Starting UMT Auto-Annotation Engine (Rebuilding masks in memory...)")

    for seg_filename in seg_files:
        img_filename = seg_filename.replace('.txt', '.jpg') 
        img_path = os.path.join(img_dir, img_filename)
        seg_path = os.path.join(seg_dir, seg_filename)
        
        pose_output_path = os.path.join(output_pose_dir, seg_filename)
        clean_img_output_path = os.path.join(output_img_dir, img_filename)

        if not os.path.exists(img_path): continue

        all_heads_xy = get_all_heads(img_filename, xml_path)
        if not all_heads_xy: continue # Skip if no Heads are annotated in CVAT

        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        heads_array = np.array(all_heads_xy)
        pose_lines = []

        # Read the SAM polygons from the text file
        with open(seg_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        for line in lines:
            parts = line.split()
            class_id = int(parts[0])
            if class_id != 0: continue # Only process your target class
                
            # De-normalize coordinates to rebuild the polygon
            coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
            coords[:, 0] *= img_w
            coords[:, 1] *= img_h
            poly = np.int32(coords)

            x, y, w, h = cv2.boundingRect(poly)
            
            # Match Head to this specific SAM Polygon
            dist_matrix = distance.cdist(heads_array, poly, 'euclidean')
            min_dist_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            closest_head_idx = min_dist_idx[0]
            
            if np.min(dist_matrix) > 300: continue # Ignore hallucinated rocks/background

            true_head_x, true_head_y = heads_array[closest_head_idx]

            # --- THE MAGIC: Rebuild Mask in Memory ---
            # Create a blank black canvas
            single_fish_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            # Draw the solid white SAM polygon onto it
            cv2.fillPoly(single_fish_mask, [poly], 255)
            
            # Apply the Morphological Closing fix we discovered earlier to prevent fragmentation
            kernel = np.ones((7,7), np.uint8)
            single_fish_mask = cv2.morphologyEx(single_fish_mask, cv2.MORPH_CLOSE, kernel)
            single_fish_mask = cv2.GaussianBlur(single_fish_mask, (5,5), 0)
            _, single_fish_mask = cv2.threshold(single_fish_mask, 127, 255, cv2.THRESH_BINARY)
            
            # Skeletonize the rebuilt mask
            skeleton = cv2.ximgproc.thinning(single_fish_mask)
            skel_points = np.column_stack(np.where(skeleton > 0))
            if len(skel_points) < 2: continue
            
            # Snap Head to Skeleton (Anchor)
            head_yx = np.array([[true_head_y, true_head_x]])
            dist_to_skel = distance.cdist(head_yx, skel_points, 'euclidean')
            anchor_idx = np.argmin(dist_to_skel)
            anchor_node = tuple(skel_points[anchor_idx])
            
            # Pathfind to the Tail
            true_tail_node = run_dijkstra(skeleton, anchor_node)
            true_tail_x = float(true_tail_node[1])
            true_tail_y = float(true_tail_node[0])

            # --- YOLO Normalization ---
            norm_cx, norm_cy = (x + w/2) / img_w, (y + h/2) / img_h
            norm_w, norm_h = w / img_w, h / img_h
            
            norm_hx, norm_hy = true_head_x / img_w, true_head_y / img_h
            norm_tx, norm_ty = true_tail_x / img_w, true_tail_y / img_h

            # FORMAT: Class CX CY W H HeadX HeadY HeadVis TailX TailY TailVis
            pose_str = f"0 {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f} {norm_hx:.6f} {norm_hy:.6f} 2 {norm_tx:.6f} {norm_ty:.6f} 2"
            pose_lines.append(pose_str)

        # Write output files if valid fish were found
        if pose_lines:
            with open(pose_output_path, 'w') as f:
                f.write("\n".join(pose_lines))
            if not os.path.exists(clean_img_output_path):
                shutil.copy(img_path, clean_img_output_path)
            success_count += 1

    print(f"\n[+] Successfully Auto-Annotated Heads AND Tails for {success_count} UMT images.")


xml_file_path = r"POSE ESTIMATION\XML ANNOTATIONS\UMT Dataset V1 Refixed\annotations_FIXED.xml"
images_directory = r"DATASETS FOR POSE ESTIMATION\UMT_dataset\UMT images for annotation V1 CLEANED"
output_labels_directory = r"DATASETS FOR INSTANCE SEGMENTATION\UMT_dataset\labels"
sam_file_path = r"INSTANCE SEGMENTATION\DATASET ANNOTATION STRATEGIES\sam_vit_h_4b8939.pth"

if __name__ == '__main__':
    auto_annotate_umt_dataset(
        img_dir=r"DATASETS FOR POSE ESTIMATION\UMT_dataset\UMT images for annotation V1 CLEANED", 
        seg_dir=r"DATASETS FOR INSTANCE SEGMENTATION\UMT_dataset\DINO+SAM_filtered_labels", 
        xml_path=r"POSE ESTIMATION\XML ANNOTATIONS\UMT Dataset V1 Refixed\annotations_FIXED.xml",
        
        # Output directly to your main training folder!
        output_pose_dir=r"INSTANCE SEGMENTATION + POSE ESTIMATION\DATASETS\UMT Dataset\Head+Tail UMT Annotations\labels\train",
        output_img_dir=r"INSTANCE SEGMENTATION + POSE ESTIMATION\DATASETS\UMT Dataset\Head+Tail UMT Annotations\images\train" 
    )
  
    
