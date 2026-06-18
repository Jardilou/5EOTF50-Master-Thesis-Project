"""
================================================================================
Script: Dual-Model Length Estimation Pipeline (YOLO-Pose + SAM)
================================================================================

Description:
    An automated inference pipeline for estimating the anatomical pixel length 
    of fish using a dual-model computer vision approach. The script deploys a 
    trained YOLO-Pose model to predict bounding boxes and head keypoints, and 
    subsequently passes the bounding boxes as prompts to the Segment Anything 
    Model (SAM) to extract high-fidelity instance masks. It then calculates the 
    total anatomical length by skeletonizing the SAM mask and measuring the 
    bridged geodesic path using graph traversal.

    Key Functions:
    1. Dual-Model Inference: Cascades YOLO-Pose predictions with SAM prompt-based 
       segmentation to isolate individual targets without requiring manual XML 
       ground truth annotations.
    2. Mask Processing: Converts boolean SAM outputs into solid binary masks 
       and applies morphological closing to generate contiguous shapes suitable 
       for skeletonization.
    3. Skeletonization & Anchoring: Reduces the generated binary mask to a 
       1-pixel wide medial axis and links it to the YOLO-predicted head keypoint 
       using Euclidean distance thresholds.
    4. Graph Traversal: Executes Dijkstra's algorithm to calculate the curved 
       distance along the skeleton, mathematically locating the tail extremity.
    5. Visual Output: Renders an interactive OpenCV window displaying the 
       overlaid SAM mask, extracted skeleton, predicted keypoints, bridge line, 
       and the final calculated pixel length.

Author:
    Louis Jardinet
    Universiti Teknologi Petronas

Date:
    January 2026 - June 2026

Dependencies:
    - Python standard libraries: os, heapq
    - External packages: opencv-python (cv2), numpy, scipy, ultralytics (YOLO, SAM)

Inputs:
    - img_dir: Directory containing the unseen test images for inference.
    - yolo_weights_path: Path to the trained YOLO-Pose model weights (.pt file).
================================================================================
"""

import os
import cv2
import numpy as np
import heapq
from scipy.spatial import distance
from ultralytics import YOLO, SAM

def pure_cv2_skeletonize(binary_mask):
    """Fallback skeletonization."""
    skel = np.zeros(binary_mask.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    img = binary_mask.copy()
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel

def run_dijkstra(skeleton, start_node):
    """Traces the furthest path along the skeleton."""
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
    return furthest_node, max_dist

def measure_with_yolo_and_sam(img_dir, yolo_weights_path):
    print("\n--- INITIATING DUAL-MODEL PIPELINE (YOLO POSE + SAM) ---")
    
    # 1. Load both models into RAM
    print("[INFO] Loading YOLOv11 Pose Model...")
    yolo_model = YOLO(yolo_weights_path)
    
    print("[INFO] Loading Mobile SAM Model...")
    sam_model = SAM('sam_b.pt') 

    image_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]

    for img_filename in image_files:
        img_path = os.path.join(img_dir, img_filename)
        img = cv2.imread(img_path)
        if img is None: continue
        
        img_h, img_w = img.shape[:2]

        # 2. YOLO detects the Bounding Box and Keypoints
        yolo_results = yolo_model.predict(img, conf=0.25, verbose=False)[0]
        
        if len(yolo_results.boxes) == 0:
            print(f"[{img_filename}] No fish detected.")
            continue

        for i, (box, kpts) in enumerate(zip(yolo_results.boxes, yolo_results.keypoints)):
            # Extract Bounding Box [x1, y1, x2, y2] to prompt SAM
            bbox = box.xyxy[0].cpu().numpy()
            
            # Extract Head Keypoint (Index 0)
            head_xy = kpts.xy[0][0].cpu().numpy()
            if head_xy[0] == 0 and head_xy[1] == 0: continue # Skip if no head found
            yolo_head_yx = (head_xy[1], head_xy[0])

            # 3. Prompt SAM with the YOLO Bounding Box
            sam_results = sam_model.predict(img, bboxes=bbox, verbose=False)[0]
            
            if sam_results.masks is None: continue
            
            # Get the high-fidelity SAM mask
            # Get the high-fidelity SAM mask
            mask_data = sam_results.masks.data[0].cpu().numpy()
            
            # 1. FIX: Cast the boolean (True/False) array to integers (1/0)
            mask_data = mask_data.astype(np.uint8)
            
            # 2. FIX: Resize using INTER_NEAREST to keep the mask boundaries perfectly sharp
            mask_data = cv2.resize(mask_data, (img_w, img_h), interpolation=cv2.INTER_NEAREST) 
            
            # 3. Scale up to 255 for OpenCV drawing functions
            blank_mask = mask_data * 255

            # 4. Process the SAM Mask for Skeletonization
            kernel = np.ones((5,5), np.uint8)
            binary_mask = cv2.morphologyEx(blank_mask, cv2.MORPH_CLOSE, kernel)
            _, binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)

            try:
                skeleton = cv2.ximgproc.thinning(binary_mask)
            except AttributeError:
                skeleton = pure_cv2_skeletonize(binary_mask)

            skel_points = np.column_stack(np.where(skeleton > 0))
            if len(skel_points) < 2: continue

            # 5. Anchor Skeleton to YOLO Head and Run Dijkstra
            dist_matrix = distance.cdist([yolo_head_yx], skel_points, 'euclidean')
            closest_skel_idx = np.argmin(dist_matrix)
            
            if dist_matrix[0, closest_skel_idx] > 300: continue
            
            anchor_node = tuple(skel_points[closest_skel_idx])

            true_tail_node, _ = run_dijkstra(skeleton, anchor_node)
            true_head_node, curved_spine_length = run_dijkstra(skeleton, true_tail_node)

            bridge_distance = distance.euclidean(yolo_head_yx, true_head_node)
            final_length = curved_spine_length + bridge_distance

            print(f"[{img_filename}] Fish {i} -> Bridged Spine Length: {final_length:.2f} px")

            # --- VISUALIZATION ---
            # Draw SAM Mask (Dark Grey)
            img[blank_mask > 0] = img[blank_mask > 0] * 0.5 + np.array([50, 50, 50]) * 0.5
            
            # Draw Skeleton (Red)
            img[skeleton > 0] = [0, 0, 255]
            
            # Draw Bridge (Yellow)
            cv2.line(img, (int(head_xy[0]), int(head_xy[1])), (true_head_node[1], true_head_node[0]), (0, 255, 255), 2)
            
            # Mark Endpoints
            cv2.circle(img, (int(head_xy[0]), int(head_xy[1])), 5, (255, 0, 0), -1) # YOLO Head (Blue)
            cv2.circle(img, (true_tail_node[1], true_tail_node[0]), 5, (0, 255, 255), -1) # Skeleton Tail (Yellow)
            
            # Display Length
            cv2.putText(img, f"Len: {final_length:.1f}px", (int(head_xy[0])-15, int(head_xy[1])-15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.namedWindow("YOLO + SAM + Dijkstra", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLO + SAM + Dijkstra", 1280, 720)
        cv2.imshow("YOLO + SAM + Dijkstra", img)
        
        if cv2.waitKey(0) == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Add your path to your test images and your best.pt pose model
    measure_with_yolo_and_sam(
        img_dir=r"INSTANCE SEGMENTATION + POSE ESTIMATION\INFERENCE AND RESULTS ANALYSIS\Results_Dual_Strategy_Run_1\Best_Cases_Raw",
        yolo_weights_path=r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\ultralytics\runs\pose\INSTANCE SEGMENTATION + POSE ESTIMATION\Runs\ExtremityPose\Dual_Strategy_Run_1\weights\best.pt"
    )