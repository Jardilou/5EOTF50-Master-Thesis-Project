"""
================================================================================
Script: Bézierfusion Evaluation Engine (IoU & MARE)
================================================================================

Description:
    A quantitative evaluation script for the custom "Bézierfusion" YOLO-Pose 
    model. This script measures both the spatial accuracy of the bounding boxes 
    (IoU) and the morphological accuracy of the predicted 2D anatomical curves 
    (MARE). It reconstructs cubic Bézier curves from both the ground truth XML 
    annotations and the model predictions, discretizes them into line segments, 
    and calculates the Mean Absolute Relative Error of the total arc length.

    Key Functions:
    1. Bounding Box IoU: Calculates the Intersection over Union between the 
       un-normalized ground truth boxes and the model's predictions to find 
       the best matching detection for evaluation.
    2. Bézier Arc Length Calculation: Converts a set of 4 predicted keypoints 
       (treated as control points) into a continuous cubic Bézier curve using 
       parametric equations. It estimates the true arc length by sampling the 
       curve at 100 discrete intervals and summing the Euclidean distances.
    3. Prediction Matching: Iterates through model detections and explicitly 
       links predicted keypoints to ground truth annotations using an IoU > 0.5 
       threshold.
    4. Metric Aggregation: Computes the overall Mean IoU and the length Mean 
       Absolute Relative Error (MARE) as a percentage across the entire validation 
       dataset.

Author:
    Louis Jardinet
    Universiti Teknologi Petronas

Date:
    January 2026 - June 2026

Dependencies:
    - Python standard libraries: os
    - External packages: opencv-python (cv2), numpy, ultralytics (YOLO)

Inputs:
    - weight_path: Path to the trained Bézierfusion model weights (.pt).
    - images_dir: Directory containing the unseen validation/test images.
    - labels_dir: Directory containing the corresponding ground truth YOLO .txt labels.

Outputs:
    - Terminal report detailing total valid detections, Mean IoU, and the 
      overall 2D Curve Length MARE percentage.
================================================================================
"""


import os
import cv2
import numpy as np
from ultralytics import YOLO

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def bezier_arc_length(points, num_samples=100):
    """
    Calculates the arc length of a cubic Bézier curve by discretizing it into line segments.
    points: Array of 4 control points [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
    """
    length = 0.0
    prev_point = points[0]
    
    for t in np.linspace(0, 1, num_samples):
        # Cubic Bézier Formula
        p = ( (1-t)**3 * points[0] + 
              3 * (1-t)**2 * t * points[1] + 
              3 * (1-t) * t**2 * points[2] + 
              t**3 * points[3] )
              
        if t > 0:
            # Distance between current point and previous point
            dist = np.linalg.norm(p - prev_point)
            length += dist
            
        prev_point = p
        
    return length

def evaluate_model(weight_path, images_dir, labels_dir):
    print("Loading model for evaluation...")
    model = YOLO(weight_path)
    
    total_iou = 0.0
    total_mare = 0.0
    valid_detections = 0

    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
    
    for img_name in image_files:
        img_path = os.path.join(images_dir, img_name)
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        lbl_path = os.path.join(labels_dir, txt_name)
        
        if not os.path.exists(lbl_path):
            continue
            
        # 1. Read Ground Truth (GT)
        img = cv2.imread(img_path)
        h_img, w_img = img.shape[:2]
        
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            data = list(map(float, line.strip().split()))
            if len(data) < 17: continue # Skip malformed lines
            
            # Un-normalize GT Box (cx, cy, w, h -> x1, y1, x2, y2)
            cx, cy, w, h = data[1:5]
            gt_box = [
                (cx - w/2) * w_img, (cy - h/2) * h_img,
                (cx + w/2) * w_img, (cy + h/2) * h_img
            ]
            
            # Un-normalize GT Keypoints (skip visibility flag)
            gt_kpts = []
            for i in range(5, 17, 3):
                gt_kpts.append(np.array([data[i] * w_img, data[i+1] * h_img]))
            gt_kpts = np.array(gt_kpts)
            
            # Calculate GT Curve Length
            gt_length = bezier_arc_length(gt_kpts)
            
            # 2. Get Model Prediction
            results = model.predict(img_path, conf=0.5, verbose=False)
            result = results[0]
            
            if result.boxes is None or len(result.boxes) == 0:
                continue # Model missed this fish
                
            # Find the best matching predicted box using IoU
            best_iou = 0
            best_idx = -1
            for i, pred_box in enumerate(result.boxes.xyxy.cpu().numpy()):
                iou = calculate_iou(gt_box, pred_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
                    
            if best_idx != -1 and best_iou > 0.5: # Threshold for a "True Positive" detection
                # Get predicted keypoints
                pred_kpts = result.keypoints.xy[best_idx].cpu().numpy()
                pred_length = bezier_arc_length(pred_kpts)
                
                # Calculate Relative Error
                relative_error = abs(pred_length - gt_length) / gt_length
                
                total_iou += best_iou
                total_mare += relative_error
                valid_detections += 1

    # 3. Final Aggregation
    if valid_detections > 0:
        mean_iou = total_iou / valid_detections
        mare = (total_mare / valid_detections) * 100 # Convert to percentage
        
        print("\n" + "="*40)
        print("BÉZIERFUSION EVALUATION RESULTS")
        print("="*40)
        print(f"Total Fish Evaluated: {valid_detections}")
        print(f"Mean IoU (Box Accuracy): {mean_iou:.4f}  (Ideal is > 0.75)")
        print(f"MARE (2D Curve Length Error): {mare:.2f}% (Ideal is < 5%)")
        print("="*40)
    else:
        print("No valid detections found to evaluate.")

# --- USAGE ---
weights = r"ultralytics/runs/pose/BezierFusion/eamrf_training_run8/weights/last.pt"
val_images = r"dataset2/dataset2/train-test/images/val"
val_labels = r"dataset2/dataset2/train-test/labels/val"

evaluate_model(weights, val_images, val_labels)