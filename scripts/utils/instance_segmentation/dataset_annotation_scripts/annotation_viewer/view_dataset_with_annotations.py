"""
================================================================================
Script: Interactive YOLO Segmentation Dataset Viewer
================================================================================

Description:
    An interactive, OpenCV-based visualizer for auditing YOLO-formatted instance 
    segmentation datasets. This tool loads images alongside their corresponding 
    .txt polygon labels, de-normalizes the coordinates, and renders semi-transparent 
    masks over the objects. 

    Key Features:
    1. Visual Auditing: Overlays filled polygons and thick outlines to verify 
       segmentation accuracy.
    2. Class Color-Coding: Automatically colors targets based on their class ID:
       - Class 0 (Measurable Targets) -> Green
       - Class 1 (Unmeasurable Auto-Sweep) -> Red
    3. Interactive Navigation: Rapidly page through the dataset using keyboard controls.

Author:
    Louis Jardinet
    Universiti Teknologi Petronas

Date:
    January 2026 - June 2026

Usage:

    Keyboard Controls:
    [D] : Next Image
    [A] : Previous Image
    [Q] : Quit Viewer

Dependencies:
    - Standard Python Library: os
    - Third-Party: opencv-python (cv2), numpy

Inputs:
    - `img_dir`: Absolute path to the folder containing .jpg/.png images.
    - `label_dir`: Absolute path to the folder containing YOLO .txt polygon labels.
================================================================================
"""

import os
import cv2
import numpy as np

def visualize_yolo_polygons(img_dir, label_dir):
    """
    Interactive viewer for YOLO segmentation datasets.
    Controls: 'd' = Next, 'a' = Previous, 'q' = Quit.
    """
    # Grab all text files in the label directory
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    
    if not label_files:
        print(f"No .txt files found in {label_dir}")
        return

    print("--- VIEWER CONTROLS ---")
    print("[D] : Next Image")
    print("[A] : Previous Image")
    print("[Q] : Quit Viewer")
    print("-----------------------")

    index = 0
    total_files = len(label_files)

    cv2.namedWindow("Dataset Viewer", cv2.WINDOW_NORMAL)

    while True:
        label_filename = label_files[index]
        # Assuming your images are .jpg based on the previous output logs
        img_filename = label_filename.replace('.txt', '.jpg') 
        
        img_path = os.path.join(img_dir, img_filename)
        label_path = os.path.join(label_dir, label_filename)

        if not os.path.exists(img_path):
            print(f"Warning: Missing image for label {label_filename}")
            index = (index + 1) % total_files
            continue

        # Load image
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        
        # Create an overlay for transparent fill
        overlay = img.copy()

        # Parse YOLO txt file
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 7: # Minimum parts for class + a 3-point polygon (x,y)
                continue
                
            class_id = int(parts[0])
            
            # Extract coordinates and reshape into [N, 2] array
            coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
            
            # De-normalize from 0.0-1.0 back to absolute pixel coordinates
            coords[:, 0] *= img_w
            coords[:, 1] *= img_h
            points = np.int32(coords)

            # Color coding:
            # Class 0 (Measurable Targets): GREEN
            # Class 1 (Unmeasurable Auto-Sweep): RED
            color = (0, 255, 0) if class_id == 0 else (0, 0, 255) 

            # Draw thick outline on main image
            cv2.polylines(img, [points], isClosed=True, color=color, thickness=2)
            # Draw solid fill on overlay
            cv2.fillPoly(overlay, [points], color=color)

        # Blend the overlay with the original image (40% opacity for the mask)
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

        # Add text HUD
        cv2.putText(img, f"File: {img_filename} ({index + 1}/{total_files})", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, f"Class 0: Green | Class 1: Red", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display
        cv2.imshow("Dataset Viewer", img)

        # Wait for keypress
        key = cv2.waitKey(0) & 0xFF
        if key == ord('d'):   # Next
            index = (index + 1) % total_files
        elif key == ord('a'): # Previous
            index = (index - 1) % total_files
        elif key == ord('q'): # Quit
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    # img_directory = r"INSTANCE SEGMENTATION\DATASETS FOR INSTANCE SEGMENTATION\UMT_dataset\IMAGES\SUBSETS\UMT images for annotation V1 CLEANED Subset V2"
    # labels_directory = r"INSTANCE SEGMENTATION\DATASETS FOR INSTANCE SEGMENTATION\UMT_dataset\LABELS\DINO+SAM_filtered_labels Subset V2"

    img_directory = r"DATASETS\DATASETS FOR INSTANCE SEGMENTATION\Deepfish_Segmentation\images\valid"
    labels_directory = r"DATASETS\DATASETS FOR INSTANCE SEGMENTATION\Deepfish_Segmentation\mask_labels\valid"
    visualize_yolo_polygons(img_dir=img_directory, label_dir=labels_directory)