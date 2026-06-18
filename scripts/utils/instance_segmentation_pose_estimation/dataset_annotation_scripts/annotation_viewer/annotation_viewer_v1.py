"""
================================================================================
Script: YOLO-Pose 2 Key Points Dataset Visualizer
================================================================================

Description:
    An OpenCV-based graphical interface for reviewing YOLO-Pose formatted datasets. 
    The script loads images alongside their corresponding text annotations, 
    de-normalizes bounding box and keypoint coordinates, and overlays them onto 
    the image to facilitate visual verification of the dataset.

    Key Functions:
    1. Coordinate Denormalization: Converts normalized bounding box and keypoint 
       coordinates (0.0 - 1.0) into absolute pixel values based on image dimensions.
    2. Bounding Box Rendering: Draws class-specific colored bounding boxes 
       around the targeted instances (Class 0: Green, Class 1: Red).
    3. Keypoint Visualization: Renders the specified anatomical keypoints, 
       specifically marking the head (blue dot) and tail (yellow dot) locations 
       based on visibility flags.
    4. Interactive Navigation: Enables sequential paging through the dataset 
       using keyboard inputs (D for next, A for previous, Q to quit).

Author:
    Louis Jardinet
    Universiti Teknologi Petronas

Date:
    January 2026 - June 2026

Dependencies:
    - Python standard libraries: os
    - External packages: opencv-python (cv2)

Inputs:
    - img_dir: Directory containing the source images.
    - label_dir: Directory containing the YOLO-Pose .txt label files.
================================================================================
"""

import os
import cv2

# This script provides an interactive viewer for YOLO-Pose formatted datasets. The format
# contains mouth, Tail and Bounding Box information in a single .txt file per image. 
# The viewer allows users to navigate through the dataset using keyboard controls
# and visually inspect the annotations.

def visualize_yolo_pose(img_dir, label_dir):
    """
    Interactive viewer for YOLO-Pose datasets.
    Controls: 'd' = Next, 'a' = Previous, 'q' = Quit.
    """
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

    cv2.namedWindow("Pose Viewer", cv2.WINDOW_NORMAL)

    while True:
        label_filename = label_files[index]
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

        # Parse YOLO-Pose txt file
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            
            # Now we require exactly 11 elements for the 2-point SOTA format
            if len(parts) < 11: 
                continue
                
            class_id = int(parts[0])
            
            # Extract all 10 parameters after the class ID
            cx, cy, w, h, hx, hy, h_vis, tx, ty, t_vis = map(float, parts[1:11])
            
            # 1. Convert Bounding Box
            abs_cx, abs_cy = int(cx * img_w), int(cy * img_h)
            abs_w, abs_h = int(w * img_w), int(h * img_h)
            
            x1 = int(abs_cx - (abs_w / 2))
            y1 = int(abs_cy - (abs_h / 2))
            x2 = int(abs_cx + (abs_w / 2))
            y2 = int(abs_cy + (abs_h / 2))

            color = (0, 255, 0) if class_id == 0 else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 2. Convert and Draw HEAD Keypoint (Solid Blue Dot)
            abs_hx = int(hx * img_w)
            abs_hy = int(hy * img_h)
            if h_vis > 0:
                cv2.circle(img, (abs_hx, abs_hy), 6, (255, 0, 0), -1)
                cv2.circle(img, (abs_hx, abs_hy), 6, (255, 255, 255), 1)

            # 3. Convert and Draw TAIL Keypoint (Solid Yellow Dot)
            abs_tx = int(tx * img_w)
            abs_ty = int(ty * img_h)
            if t_vis > 0:
                cv2.circle(img, (abs_tx, abs_ty), 6, (0, 255, 255), -1) # Yellow
                cv2.circle(img, (abs_tx, abs_ty), 6, (255, 255, 255), 1)

        # Add text HUD
        cv2.putText(img, f"File: {img_filename} ({index + 1}/{total_files})", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Pose Viewer", img)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('d'):   
            index = (index + 1) % total_files
        elif key == ord('a'): 
            index = (index - 1) % total_files
        elif key == ord('q'): 
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    # img_directory = r"INSTANCE SEGMENTATION + POSE ESTIMATION\DATASETS\DEEPFISH Dataset\Train_Test_Dual_Strategy\images"
    # labels_directory = r"INSTANCE SEGMENTATION + POSE ESTIMATION\DATASETS\DEEPFISH Dataset\Train_Test_Dual_Strategy\labels"
    # img_directory = r"INSTANCE SEGMENTATION + POSE ESTIMATION\DATASETS\UMT Dataset\Head+Tail UMT Annotations\images\train"
    # labels_directory = r"INSTANCE SEGMENTATION + POSE ESTIMATION\DATASETS\UMT Dataset\Head+Tail UMT Annotations\labels\train"
    labels_directory=r"INSTANCE SEGMENTATION + POSE ESTIMATION\DATASETS\DEEPFISH Dataset\Head+Tail DEEPFISH Annotations\images\labels"
    img_directory=r"INSTANCE SEGMENTATION + POSE ESTIMATION\DATASETS\DEEPFISH Dataset\Head+Tail DEEPFISH Annotations\images\train" 
    
    visualize_yolo_pose(img_dir=img_directory, label_dir=labels_directory)