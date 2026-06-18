"""
================================================================================
Script: YOLO-Pose Evaluation B Spline and Dijkstra Analytics Pipeline
================================================================================

Description:
    A comprehensive evaluation and inference pipeline for trained YOLO-Pose models. 
    The script calculates standard quantitative metrics (mAP, OKS), profiles 
    hardware latency, and executes a qualitative visual analysis engine. For each 
    detection, it utilizes the Segment Anything Model (SAM) to generate a binary 
    mask from the YOLO bounding box, extracts the morphological skeleton, and 
    applies Dijkstra's algorithm to calculate the anatomical spine length anchored 
    to the YOLO-predicted head keypoint.

    Key Functions:
    1. Quantitative Evaluation: Computes COCO validation metrics, specifically 
       mean Average Precision (mAP) for bounding boxes and Object Keypoint 
       Similarity (OKS) for pose keypoints.
    2. Hardware Profiling: Measures preprocessing, inference, and post-processing 
       times to estimate total pipeline latency and Frames Per Second (FPS).
    3. Length Estimation Engine: Integrates YOLO bounding boxes with SAM to 
       isolate individual targets, generates a morphological skeleton, and uses 
       graph traversal to measure the bridged pixel length from head to tail.
    4. Qualitative Sorting: Ranks inference results by average confidence scores 
       to automatically isolate the top 10 best and worst prediction cases.
    5. Visual Export: Generates and sorts multiple diagnostic image variants 
       (raw, YOLO-annotated, skeletonized/Dijkstra) and extracts padded, 
       zoomed-in image crops of individual targets for closer inspection.

Author:
    Louis Jardinet
    Universiti Teknologi Petronas

Date:
    January 2026 - June 2026

Dependencies:
    - Python standard libraries: os, shutil, time, heapq
    - External packages: opencv-python (cv2), numpy, scipy, ultralytics (YOLO, SAM)

Inputs:
    - weights_path: Path to the trained YOLO-Pose model weights (.pt).
    - data_yaml: Path to the dataset configuration YAML file used for validation.
    - test_images_dir: Directory containing unseen images for qualitative inference.
    - output_dir: Base directory where all reports, cropped instances, and sorted 
      diagnostic folders will be generated.
================================================================================
"""


import os
import shutil
import time
import numpy as np
import cv2
import heapq
from scipy.spatial import distance
from ultralytics import YOLO, SAM

def pure_cv2_skeletonize(binary_mask):
    """Fallback skeletonization in case ximgproc is missing."""
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
    """Traces the furthest path along the skeleton to find biological extremities."""
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

def build_analytics_directories(base_dir):
    """Clears old analytics and builds comprehensive output directories."""
    dirs = {
        'best_annotated': os.path.join(base_dir, 'Best_Cases_Annotated'),
        'best_raw': os.path.join(base_dir, 'Best_Cases_Raw'),
        'best_masks_pts': os.path.join(base_dir, 'Best_Cases_Masks_Endpoints'), 
        'best_dijkstra': os.path.join(base_dir, 'Best_Cases_Dijkstra'), # NEW
        
        'worst_annotated': os.path.join(base_dir, 'Worst_Cases_Annotated'),
        'worst_raw': os.path.join(base_dir, 'Worst_Cases_Raw'),
        'worst_masks_pts': os.path.join(base_dir, 'Worst_Cases_Masks_Endpoints'),
        'worst_dijkstra': os.path.join(base_dir, 'Worst_Cases_Dijkstra'), # NEW
        
        'zoomed_instances': os.path.join(base_dir, 'Zoomed_Fish_Instances') # NEW
    }
    for path in dirs.values():
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
            except PermissionError:
                print(f"\n[!] Windows locked {path}. Waiting 2 seconds to force unlock...")
                time.sleep(2)
                shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)
    return dirs

def run_pipeline_analysis(weights_path, data_yaml, test_images_dir, output_dir):
    print("\n" + "="*50)
    print(" INITIATING SOTA EXTREMITY POSE & DIJKSTRA ANALYTICS")
    print("="*50 + "\n")

    # 1. Load the Models
    print("[INFO] Loading YOLOv11 Model...")
    model = YOLO(weights_path)
    
    print("[INFO] Loading Mobile SAM Model for high-fidelity masking...")
    sam_model = SAM('sam_b.pt')

    # =================================================================
    # MODULE 1: QUANTITATIVE METRICS
    # =================================================================
    print("\n[INFO] Running COCO Validation Metrics (mAP & OKS)...")
    metrics = model.val(data=data_yaml, split='val', plots=True, project=output_dir, name="Val_Metrics")
    
    box_map50 = metrics.box.map50
    box_map50_95 = metrics.box.map
    pose_map50 = metrics.pose.map50
    pose_map50_95 = metrics.pose.map

    report_text = "\n--- QUANTITATIVE RESULTS ---\n"
    report_text += f"Bounding Box mAP@0.50:       {box_map50:.4f}\n"
    report_text += f"Bounding Box mAP@0.50-0.95:  {box_map50_95:.4f}\n"
    report_text += f"Extremity Pose OKS@0.50:     {pose_map50:.4f}\n"
    report_text += f"Extremity Pose OKS@0.50-0.95:{pose_map50_95:.4f}\n"
    report_text += "----------------------------\n"
    print(report_text)

    # =================================================================
    # MODULE 2 & 3: SPEED PROFILING & QUALITATIVE EDGE-CASE SORTING
    # =================================================================
    print("[INFO] Running Inference Profiling and Dijkstra Generation...")
    dirs = build_analytics_directories(output_dir)
    
    results = model.predict(source=test_images_dir, conf=0.15, stream=False)
    
    image_scores = []
    total_preprocess = total_inference = total_postprocess = 0
    total_images = len(results)

    for res in results:
        total_preprocess += res.speed['preprocess']
        total_inference += res.speed['inference']
        total_postprocess += res.speed['postprocess']
        
        avg_conf = 0.0 if len(res.boxes) == 0 else float(res.boxes.conf.mean().cpu().numpy())
            
        # Extract Standard Images
        annotated_img = res.plot(labels=True, conf=True, line_width=1, kpt_radius=2, font_size=1)
        raw_img = res.orig_img 
        mask_pt_img = res.plot(labels=False, boxes=False, conf=False, kpt_radius=4, line_width=2)
        
        img_h, img_w = raw_img.shape[:2]
        img_filename = os.path.basename(res.path)
        base_filename = os.path.splitext(img_filename)[0]
        
        # Create a copy for the full-frame Dijkstra drawing
        dijkstra_full_img = raw_img.copy()

        # =================================================================
        # THE DIJKSTRA & ZOOMED INSTANCE ENGINE
        # =================================================================
        if len(res.boxes) > 0 and res.keypoints is not None:
            for i, (box, kpts) in enumerate(zip(res.boxes, res.keypoints)):
                # Extract Class Name (e.g., "fish" or "measurable_fish")
                class_id = int(box.cls[0].item())
                species_name = res.names[class_id]

                # Extract Bbox and Keypoint
                bbox = box.xyxy[0].cpu().numpy()
                head_xy = kpts.xy[0][0].cpu().numpy()
                
                if head_xy[0] == 0 and head_xy[1] == 0: 
                    continue # Skip if no keypoints
                    
                yolo_head_yx = (head_xy[1], head_xy[0])

                # Run SAM for this specific bounding box
                sam_results = sam_model.predict(raw_img, bboxes=bbox, verbose=False)[0]
                if sam_results.masks is None: continue
                
                # Format SAM Mask (Fixing the Boolean to Uint8 bug)
                mask_data = sam_results.masks.data[0].cpu().numpy()
                mask_data = mask_data.astype(np.uint8)
                mask_data = cv2.resize(mask_data, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                blank_mask = mask_data * 255

                # Morphology & Skeletonization
                kernel = np.ones((5,5), np.uint8)
                binary_mask = cv2.morphologyEx(blank_mask, cv2.MORPH_CLOSE, kernel)
                _, binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)

                try:
                    skeleton = cv2.ximgproc.thinning(binary_mask)
                except AttributeError:
                    skeleton = pure_cv2_skeletonize(binary_mask)

                skel_points = np.column_stack(np.where(skeleton > 0))
                if len(skel_points) < 2: continue

                # Dijkstra Math
                dist_matrix = distance.cdist([yolo_head_yx], skel_points, 'euclidean')
                closest_skel_idx = np.argmin(dist_matrix)
                if dist_matrix[0, closest_skel_idx] > 300: continue
                
                anchor_node = tuple(skel_points[closest_skel_idx])
                true_tail_node, _ = run_dijkstra(skeleton, anchor_node)
                true_head_node, curved_spine_length = run_dijkstra(skeleton, true_tail_node)

                bridge_distance = distance.euclidean(yolo_head_yx, true_head_node)
                final_length = curved_spine_length + bridge_distance

                # --- DRAWING ON THE FULL FRAME ---
                # Draw SAM Mask Overlay (Dark grey transparent)
                dijkstra_full_img[blank_mask > 0] = dijkstra_full_img[blank_mask > 0] * 0.6 + np.array([50, 150, 50]) * 0.4
                # Draw Skeleton (Red)
                dijkstra_full_img[skeleton > 0] = [0, 0, 255]
                # Draw Bridge (Yellow)
                cv2.line(dijkstra_full_img, (int(head_xy[0]), int(head_xy[1])), (true_head_node[1], true_head_node[0]), (0, 255, 255), 2)
                # Endpoints
                cv2.circle(dijkstra_full_img, (int(head_xy[0]), int(head_xy[1])), 4, (255, 0, 0), -1)
                cv2.circle(dijkstra_full_img, (true_tail_node[1], true_tail_node[0]), 4, (0, 255, 255), -1)
                
                label_text = f"{species_name} | {final_length:.1f}px"
                cv2.putText(dijkstra_full_img, label_text, (int(bbox[0]), int(bbox[1])-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # --- ZOOMED INSTANCE EXTRACTION ---
                # Add a 20 pixel padding around the bounding box so the fish isn't cut off
                pad = 20
                x1 = max(0, int(bbox[0]) - pad)
                y1 = max(0, int(bbox[1]) - pad)
                x2 = min(img_w, int(bbox[2]) + pad)
                y2 = min(img_h, int(bbox[3]) + pad)
                
                # Crop the fully drawn image
                zoomed_crop = dijkstra_full_img[y1:y2, x1:x2]
                
                # Save the specific crop
                crop_filename = f"{base_filename}_fish_{i}.jpg"
                cv2.imwrite(os.path.join(dirs['zoomed_instances'], crop_filename), zoomed_crop)

        # Store for sorting
        image_scores.append((avg_conf, img_filename, annotated_img, raw_img, mask_pt_img, dijkstra_full_img))

    # Calculate average speeds
    avg_infer_ms = total_inference / total_images
    total_latency_ms = (total_preprocess + total_inference + total_postprocess) / total_images
    fps = 1000.0 / total_latency_ms

    speed_text = "\n--- HARDWARE PROFILING ---\n"
    speed_text += f"Average Inference Time: {avg_infer_ms:.2f} ms\n"
    speed_text += f"Total Pipeline Latency: {total_latency_ms:.2f} ms\n"
    speed_text += f"Estimated Real-Time FPS: {fps:.1f} FPS\n"
    speed_text += "--------------------------\n"
    print(speed_text)

    # Export Text Report
    report_file_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_file_path, "w") as f:
        f.write("SOTA EXTREMITY POSE - PIPELINE EVALUATION\n")
        f.write("="*41 + "\n")
        f.write(report_text)
        f.write(speed_text)

    # Sort images by confidence
    image_scores.sort(key=lambda x: x[0], reverse=True)

    # Isolate Best Cases
    for conf, filename, ann_img, raw_img, mask_img, dij_img in image_scores[:10]:
        cv2.imwrite(os.path.join(dirs['best_annotated'], f"CONF_{conf:.3f}_{filename}"), ann_img)
        cv2.imwrite(os.path.join(dirs['best_raw'], f"RAW_{filename}"), raw_img)
        cv2.imwrite(os.path.join(dirs['best_masks_pts'], f"MASK_PTS_{conf:.3f}_{filename}"), mask_img)
        cv2.imwrite(os.path.join(dirs['best_dijkstra'], f"DIJKSTRA_{conf:.3f}_{filename}"), dij_img)

    # Isolate Worst Cases
    for conf, filename, ann_img, raw_img, mask_img, dij_img in image_scores[-10:]:
        cv2.imwrite(os.path.join(dirs['worst_annotated'], f"CONF_{conf:.3f}_{filename}"), ann_img)
        cv2.imwrite(os.path.join(dirs['worst_raw'], f"RAW_{filename}"), raw_img)
        cv2.imwrite(os.path.join(dirs['worst_masks_pts'], f"MASK_PTS_{conf:.3f}_{filename}"), mask_img)
        cv2.imwrite(os.path.join(dirs['worst_dijkstra'], f"DIJKSTRA_{conf:.3f}_{filename}"), dij_img)

    print(f"\n[+] Qualitative analysis complete. {len(image_scores)} images processed.")
    print(f"[+] Output Directories Generated in: {output_dir}")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    
    # Define your paths here to pass into the function
    WEIGHTS = r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\ultralytics\runs\pose\INSTANCE SEGMENTATION + POSE ESTIMATION\Runs\Dual_Strategy_Run_2\weights\best.pt"
    YAML = r"Dual Strategy Fish Pose.yaml"
    TEST_IMAGES = r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\DATASETS\UNTOUCHED_UMT_IMAGES_COMBINED\ADEQUATE UNTOUCHED UMT IMAGES" 
    OUT_DIR = r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\INSTANCE SEGMENTATION + POSE ESTIMATION\INFERENCE AND RESULTS ANALYSIS\Results_Dual_Strategy_Run_2_Bspline_Dijkstra"
    
    run_pipeline_analysis(
        weights_path=WEIGHTS,
        data_yaml=YAML,
        test_images_dir=TEST_IMAGES,
        output_dir=OUT_DIR
    )