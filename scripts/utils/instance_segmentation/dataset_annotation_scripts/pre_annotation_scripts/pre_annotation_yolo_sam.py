"""
================================================================================
Script: Pre-Annotation using YOLO and SAM
================================================================================

Description:
    A script to automate the generation of YOLO-format instance segmentation 
    labels by combining a pre-trained YOLO object detector with the Segment 
    Anything Model (SAM). The script processes existing manual annotations 
    (Class 0) and autonomously discovers unannotated targets (Class 1) in the 
    background using the YOLO model's predictions.

    Key Functions:
    1. Primary Annotation Processing: Parses XML ground truth data (Class 0) 
       and uses SAM to generate polygon masks from the provided bounding boxes 
       and keypoints.
    2. Automated Detection: Uses a pre-trained YOLO model to scan the image 
       and generate bounding boxes for previously unannotated targets.
    3. Overlap Prevention: Calculates Intersection over Union (IoU) to discard 
       any YOLO detections that overlap with the existing Class 0 annotations, 
       preventing duplicate labels.
    4. Mask-to-Polygon Conversion: Extracts the largest external contour from 
       the SAM-generated masks and converts it into normalized YOLO polygon 
       coordinates.

Author:
    Louis Jardinet
    Universiti Teknologi Petronas

Date:
    January 2026 - June 2026

Dependencies:
    - Python standard libraries: os, xml.etree.ElementTree
    - External packages: opencv-python (cv2), torch, numpy, ultralytics (YOLO), 
      segment_anything (SAM)

Inputs:
    - xml_path: Path to the XML file containing primary ground truth annotations.
    - img_dir: Directory containing the source images.
    - output_dir: Directory where the combined YOLO .txt polygon files are saved.
    - sam_checkpoint: Path to the downloaded SAM model weights (.pth file).
    - yolo_checkpoint: Path to the downloaded YOLO model weights (.pt file).
    - confidence_thresh: Threshold for the YOLO detector (default 0.25).
================================================================================
"""

import os
import cv2
import torch
import numpy as np
import xml.etree.ElementTree as ET
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

def calculate_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-9)

def mask_to_yolo_polygon(mask, img_width, img_height):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    largest_contour = max(contours, key=cv2.contourArea)
    if len(largest_contour) < 3: return None
    polygon = [f"{pt[0][0] / img_width:.6f} {pt[0][1] / img_height:.6f}" for pt in largest_contour]
    return " ".join(polygon)

def build_yolo_sam_dataset(xml_path, img_dir, output_dir, sam_checkpoint, yolo_checkpoint, confidence_thresh=0.25):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading YOLO (Pre-Trained Fish Detector)...")
    yolo_model = YOLO(yolo_checkpoint)
    
    print("Loading SAM (Mask Extractor)...")
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(device)
    sam_predictor = SamPredictor(sam)
    
    print("Parsing Class 0 Ground Truth XML...")
    tree = ET.parse(xml_path)
    ground_truth_data = {}
    
    for image in tree.getroot().findall('image'):
        img_name = image.attrib['name']
        ground_truth_data[img_name] = []
        boxes = {b.attrib['group_id']: b for b in image.findall('box') if 'group_id' in b.attrib}
        skeletons = {s.attrib['group_id']: s for s in image.findall('skeleton') if 'group_id' in s.attrib}
        
        for group_id, box in boxes.items():
            xtl, ytl = float(box.attrib['xtl']), float(box.attrib['ytl'])
            xbr, ybr = float(box.attrib['xbr']), float(box.attrib['ybr'])
            pts = [[float(c) for c in pt.attrib['points'].split(',')] for pt in skeletons.get(group_id, []).findall('points')] if group_id in skeletons else []
            ground_truth_data[img_name].append({'box': [xtl, ytl, xbr, ybr], 'points': pts})

    print("-" * 50)
    
    for img_name, annotations in ground_truth_data.items():
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path): continue
            
        cv2_img = cv2.imread(img_path)
        img_h, img_w = cv2_img.shape[:2]
        rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        
        sam_predictor.set_image(rgb_img)
        yolo_lines = []
        class_0_bboxes = []
        
        # --- PHASE 1: Process Class 0 (XML Annotations) ---
        for ann in annotations:
            box = np.array(ann['box'])
            class_0_bboxes.append(box)
            input_point = np.array(ann['points']) if ann['points'] else None
            input_label = np.ones(len(ann['points'])) if ann['points'] else None
            
            masks, _, _ = sam_predictor.predict(
                point_coords=input_point, point_labels=input_label, box=box, multimask_output=False
            )
            poly_str = mask_to_yolo_polygon(masks[0], img_w, img_h)
            if poly_str: yolo_lines.append(f"0 {poly_str}")
                
        # --- PHASE 2: Process Class 1 (YOLO -> SAM Pipeline) ---
        # Run YOLO inference. verbose=False stops it from spamming the terminal.
        results = yolo_model(rgb_img, imgsz=1024, conf=confidence_thresh, verbose=False)
        
        class_1_count = 0
        for result in results:
            # Extract bounding boxes from YOLO results
            for yolo_box in result.boxes.xyxy.cpu().numpy():
                
                # Constraint 1: Check against known XML fish to prevent duplicates
                is_overlap = any(calculate_iou(yolo_box, c0_box) > 0.1 for c0_box in class_0_bboxes)
                if is_overlap: 
                    continue
                    
                # Constraint 2: Force SAM to extract using the YOLO bounding box
                masks, _, _ = sam_predictor.predict(box=yolo_box, multimask_output=False)
                poly_str = mask_to_yolo_polygon(masks[0], img_w, img_h)
                
                if poly_str:
                    yolo_lines.append(f"1 {poly_str}")
                    class_1_count += 1
                
        # --- Output ---
        if yolo_lines:
            txt_filepath = os.path.join(output_dir, os.path.splitext(img_name)[0] + ".txt")
            with open(txt_filepath, 'w') as f: 
                f.write("\n".join(yolo_lines))
            
        print(f"Processed {img_name} -> Found {class_1_count} unannotated fish.")


xml_file_path = r"POSE ESTIMATION\XML ANNOTATIONS\UMT Dataset V1 Refixed\annotations_FIXED.xml"
images_directory = r"DATASETS FOR POSE ESTIMATION\UMT_dataset\UMT images for annotation V1 CLEANED"
output_labels_directory = r"DATASETS FOR INSTANCE SEGMENTATION\UMT_dataset\YOLO+SAM_labels"
sam_file_path = r"INSTANCE SEGMENTATION\DATASET ANNOTATION STRATEGIES\sam_vit_h_4b8939.pth"
yolo_file_path = r"INSTANCE SEGMENTATION\DATASET ANNOTATION STRATEGIES\cfd-yolov12x-1.00.pt"

if __name__ == '__main__':
    build_yolo_sam_dataset(
        xml_path=xml_file_path,
        img_dir=images_directory,
        output_dir=output_labels_directory,
        sam_checkpoint=sam_file_path,
        yolo_checkpoint=yolo_file_path, # Path to the public fish model you downloaded
        confidence_thresh=0.15 # Tune this based on how aggressive you want the detector to be
    )