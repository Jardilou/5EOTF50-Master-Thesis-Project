"""
================================================================================
Script: YOLO-Pose Auto-Annotation Engine (Binary Mask Version)
================================================================================

Description:
    A script to automatically generate YOLO-Pose annotations (bounding box, 
    head keypoint, and tail keypoint) by combining existing binary image masks 
    (.png/.jpg) with XML head coordinate data. The script extracts contours 
    from the segmentation masks, generates morphological skeletons, and uses 
    graph traversal to mathematically locate the tail extremity.

    Key Functions:
    1. XML Parsing: Extracts ground-truth head keypoint coordinates from 
       CVAT XML files.
    2. Mask Ingestion: Reads grayscale image masks, applies binary thresholding, 
       and extracts external contours while filtering out microscopic noise.
    3. Skeletonization & Anchoring: Extracts a 1-pixel wide medial axis (spine) 
       from individual target contours and anchors it to the nearest verified 
       XML head coordinate.
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
    - mask_dir: Directory containing binary image masks (.png or .jpg).
    - xml_path: Path to the XML file containing the manual head keypoint annotations.

Outputs:
    - output_pose_dir: Directory where the generated YOLO-Pose .txt labels are saved.
    - output_img_dir: Directory where the successfully processed images are copied.
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
                new_dist = current_dist + step_cost
                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heapq.heappush(queue, (new_dist, neighbor))
    return furthest_node

# --- 3. The Auto-Labeler Engine ---
def auto_annotate_dataset(img_dir, mask_dir, xml_path, output_pose_dir, output_img_dir):
    os.makedirs(output_pose_dir, exist_ok=True)
    os.makedirs(output_img_dir, exist_ok=True)
    
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg'))]
    success_count = 0

    print("Starting Auto-Annotation Engine (Finding Tails...)")

    for mask_filename in mask_files:
        img_filename = mask_filename.replace('.png', '.jpg') 
        img_path = os.path.join(img_dir, img_filename)
        mask_path = os.path.join(mask_dir, mask_filename)
        
        pose_output_path = os.path.join(output_pose_dir, os.path.splitext(img_filename)[0] + '.txt')
        clean_img_output_path = os.path.join(output_img_dir, img_filename)

        if not os.path.exists(img_path): continue

        all_heads_xy = get_all_heads(img_filename, xml_path)
        if not all_heads_xy: continue # Skip: No Head

        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        heads_array = np.array(all_heads_xy)
        pose_lines = []

        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, binary_mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 100: continue

            poly = contour.reshape(-1, 2)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Match Head to Mask
            dist_matrix = distance.cdist(heads_array, poly, 'euclidean')
            min_dist_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            closest_head_idx = min_dist_idx[0]
            
            if np.min(dist_matrix) > 300: continue # Skip: Mask is a hallucinated rock

            true_head_x, true_head_y = heads_array[closest_head_idx]

            # --- THE MAGIC: Auto-Calculate the Tail ---
            # 1. Create a blank canvas just for this specific fish
            single_fish_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.drawContours(single_fish_mask, [contour], -1, 255, thickness=cv2.FILLED)
            
            # 2. Skeletonize this fish
            skeleton = cv2.ximgproc.thinning(single_fish_mask)
            skel_points = np.column_stack(np.where(skeleton > 0))
            if len(skel_points) < 2: continue
            
            # 3. Snap Head to Skeleton (Anchor)
            head_yx = np.array([[true_head_y, true_head_x]])
            dist_to_skel = distance.cdist(head_yx, skel_points, 'euclidean')
            anchor_idx = np.argmin(dist_to_skel)
            anchor_node = tuple(skel_points[anchor_idx])
            
            # 4. Pathfind to the Tail
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

        if pose_lines:
            with open(pose_output_path, 'w') as f:
                f.write("\n".join(pose_lines))
            if not os.path.exists(clean_img_output_path):
                shutil.copy(img_path, clean_img_output_path)
            success_count += 1

    print(f"\n[+] Successfully Auto-Annotated Heads AND Tails for {success_count} images.")

if __name__ == '__main__':
    auto_annotate_dataset(
        img_dir=r"INSTANCE SEGMENTATION + POSE ESTIMATION\DATASETS\DEEPFISH Dataset\Segmentation\images\valid", 
        mask_dir=r"INSTANCE SEGMENTATION + POSE ESTIMATION\DATASETS\DEEPFISH Dataset\Segmentation\masks\valid", 
        xml_path=r"POSE ESTIMATION\XML ANNOTATIONS\annotations_DEEPFISH_V1.xml",
        output_pose_dir=r"INSTANCE SEGMENTATION + POSE ESTIMATION\DATASETS\DEEPFISH Dataset\Head+Tail DEEPFISH Annotations\images\labels",
        output_img_dir=r"INSTANCE SEGMENTATION + POSE ESTIMATION\DATASETS\DEEPFISH Dataset\Head+Tail DEEPFISH Annotations\images\train" 
    )