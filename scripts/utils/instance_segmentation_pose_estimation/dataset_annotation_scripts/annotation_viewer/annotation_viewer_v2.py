"""
================================================================================
Script: YOLO-Pose 3 Key Points Diagnostic Visualizer and Exporter
================================================================================

Description:
    A script to generate visual diagnostic overlays for YOLO-Pose formatted 
    datasets. It reads image and label pairs, parses bounding box and keypoint 
    coordinates, and exports color-coded verification images. This tool is 
    used to visually audit the accuracy of dataset annotations prior to training.

    Key Functions:
    1. Annotation Parsing: Reads YOLO .txt files and strips inline comments 
       to safely extract numerical class and coordinate data.
    2. Bounding Box Rendering: Draws white bounding boxes for primary targets 
       (Class 0) and red bounding boxes for secondary targets (Class 1).
    3. Keypoint Rendering: Plots up to three distinct anatomical keypoints 
       (head, upper tail, lower tail) and connecting skeletal lines for Class 0 
       targets, conditioned on visibility flags.
    4. Batch Export: Processes the entire dataset, saves the visualized images 
       to a specified directory, and prints a numerical summary report.

Author:
    Louis Jardinet
    Universiti Teknologi Petronas

Date:
    January 2026 - June 2026

Dependencies:
    - Python standard libraries: os
    - External packages: opencv-python (cv2)

Inputs:
    - images_dir: Directory containing the source images.
    - labels_dir: Directory containing the YOLO-Pose .txt label files.

Outputs:
    - output_dir: Directory where the rendered diagnostic images are saved.
================================================================================
"""

import os
import cv2

def visualize_yolo_pose_debug(images_dir, labels_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print("[ERROR] Check your input directories!")
        return

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    print("\n" + "="*50)
    print(" INITIATING COLOR-CODED DIAGNOSTIC VISUALIZER")
    print("="*50 + "\n")

    success_count = 0
    total_measurable = 0
    total_unmeasurable = 0

    for img_name in image_files:
        img_path = os.path.join(images_dir, img_name)
        txt_name = os.path.splitext(img_name)[0] + '.txt'
        txt_path = os.path.join(labels_dir, txt_name)
        
        if not os.path.exists(txt_path):
            continue
            
        img = cv2.imread(img_path)
        if img is None: continue
            
        img_h, img_w = img.shape[:2]
        
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            # --- THE PROACTIVE FIX ---
            # Slice off the inline species comment (#) before trying to convert to numbers
            clean_line = line.split('#')[0].strip()
            if not clean_line: continue
            
            parts = list(map(float, clean_line.split()))
            
            # Failsafe: As long as it has a Class and a Bounding Box (5 numbers), draw it!
            if len(parts) < 5: 
                continue
                
            class_id = int(parts[0])
            cx, cy, w, h = parts[1:5]
            
            x1, y1 = int((cx - w / 2) * img_w), int((cy - h / 2) * img_h)
            x2, y2 = int((cx + w / 2) * img_w), int((cy + h / 2) * img_h)
            
            # --- COLOR CODING & COUNTING LOGIC ---
            if class_id == 0:
                total_measurable += 1
                box_color = (255, 255, 255) # White
                label_text = "Measurable (0)"
                text_color = (0, 255, 0)
            else:
                total_unmeasurable += 1
                box_color = (0, 0, 255) # Red
                label_text = "UNMEASURABLE (1)"
                text_color = (0, 0, 255)

            # Draw Bounding Box and Label
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 1)
            # cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

            # --- ONLY DRAW KEYPOINTS IF THEY EXIST AND IT'S MEASURABLE ---
            if class_id == 0 and len(parts) >= 14:
                hx_px, hy_px = int(parts[5] * img_w), int(parts[6] * img_h)
                tx_px, ty_px = int(parts[8] * img_w), int(parts[9] * img_h)
                t2x_px, t2y_px = int(parts[11] * img_w), int(parts[12] * img_h)

                valid_head, valid_tail, valid_tail2 = (parts[7] > 0), (parts[10] > 0), (parts[13] > 0)

                if valid_head: cv2.circle(img, (hx_px, hy_px), 5, (255, 0, 0), -1) # Blue Head
                if valid_tail: cv2.circle(img, (tx_px, ty_px), 5, (0, 255, 0), -1) # Green Tail 1
                if valid_tail2: cv2.circle(img, (t2x_px, t2y_px), 5, (0, 165, 255), -1) # Orange Tail 2
                    
                if valid_head and valid_tail: cv2.line(img, (hx_px, hy_px), (tx_px, ty_px), (0, 255, 255), 1)
                if valid_head and valid_tail2: cv2.line(img, (hx_px, hy_px), (t2x_px, t2y_px), (0, 255, 255), 1)

        out_path = os.path.join(output_dir, f"VERIFIED_{img_name}")
        cv2.imwrite(out_path, img)
        success_count += 1

    print("\n" + "="*50)
    print(" VERIFICATION REPORT")
    print("="*50)
    print(f" -> Total Images Processed: {success_count}")
    print(f" -> Measurable Fish Found:  {total_measurable}")
    print(f" -> Unmeasurable Fish Found:{total_unmeasurable}")
    print("="*50 + "\n")


if __name__ == '__main__':
    # Add your paths here
    IMAGES_DIR = r"DATASETS\DATASETS FOR INSTANCE SEGMENTATION + POSE ESTIMATION\UMT Dataset\CURRENT\Head + Tails UMT Annotation V2\IMAGES"
    LABELS_DIR = r"DATASETS\DATASETS FOR INSTANCE SEGMENTATION + POSE ESTIMATION\UMT Dataset\CURRENT\Head + Tails UMT Annotation V2\LABELS"
    OUTPUT_DIR = r"INSTANCE SEGMENTATION + POSE ESTIMATION\DATASET ANNOTATION SCRIPTS\ANNOTATION VIEWER\VISUALIZATION_OUTPUTS"
    
    visualize_yolo_pose_debug(IMAGES_DIR, LABELS_DIR, OUTPUT_DIR)