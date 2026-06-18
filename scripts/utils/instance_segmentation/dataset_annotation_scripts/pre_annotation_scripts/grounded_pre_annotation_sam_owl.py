"""
================================================================================
Script: Grounded Pre-Annotation using SAM and Owlv2
================================================================================

Description:
    A script to automate the generation of YOLO-format instance segmentation 
    labels by combining a zero-shot vision-language object detector (Owlv2) 
    with the Segment Anything Model (SAM). The script processes existing manual 
    annotations (Class 0) and autonomously discovers unannotated targets (Class 1) 
    in the background using text-based prompts.

    Key Functions:
    1. Primary Annotation Processing: Parses XML ground truth data (Class 0) 
       and uses SAM to generate precise polygon masks from the bounding boxes 
       and keypoints.
    2. Zero-Shot Detection: Uses Google's Owlv2 model to scan the image for 
       the text prompt "a photo of a fish", generating bounding boxes for 
       previously unannotated targets.
    3. Overlap Prevention: Calculates Intersection over Union (IoU) to discard 
       any Owlv2 detections that overlap with the existing Class 0 annotations, 
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
    - External packages: opencv-python (cv2), torch, numpy, Pillow (PIL), 
      transformers (Owlv2), segment_anything (SAM)

Inputs:
    - xml_path: Path to the XML file containing primary ground truth annotations.
    - img_dir: Directory containing the source images.
    - sam_checkpoint: Path to the downloaded SAM model weights (.pth file).
    - confidence_thresh: Threshold for the Owlv2 detector (default 0.15).

Outputs:
    - output_dir: Directory where the combined YOLO .txt polygon files are saved.
================================================================================
"""

import os
import cv2
import torch
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
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

def build_grounded_sam_dataset(xml_path, img_dir, output_dir, sam_checkpoint, confidence_thresh=0.15):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading Owlv2 (Zero-Shot Detector)...")
    owl_processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owl_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device)
    
    print("Loading SAM (Mask Extractor)...")
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(device)
    sam_predictor = SamPredictor(sam)
    
    print("Parsing Class 0 Ground Truth...")
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
    texts = [["a photo of a fish"]]
    
    for img_name, annotations in ground_truth_data.items():
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path): continue
            
        cv2_img = cv2.imread(img_path)
        img_h, img_w = cv2_img.shape[:2]
        rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        
        sam_predictor.set_image(rgb_img)
        yolo_lines = []
        class_0_bboxes = []
        
        # --- PHASE 1: Process Class 0 (XML Annotations) ---
        for ann in annotations:
            box = np.array(ann['box'])
            class_0_bboxes.append(box)
            input_point = np.array(ann['points']) if ann['points'] else None
            input_label = np.ones(len(ann['points'])) if ann['points'] else None
            
            masks, _, _ = sam_predictor.predict(point_coords=input_point, point_labels=input_label, box=box, multimask_output=False)
            poly_str = mask_to_yolo_polygon(masks[0], img_w, img_h)
            if poly_str: yolo_lines.append(f"0 {poly_str}")
                
        # --- PHASE 2: Process Class 1 (Owlv2 -> SAM Pipeline) ---
        inputs = owl_processor(text=texts, images=pil_img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = owl_model(**inputs)
            
        target_sizes = torch.tensor([pil_img.size[::-1]])        
        results = owl_processor.post_process_grounded_object_detection(
            outputs=outputs, 
            target_sizes=target_sizes, 
            text_labels=texts, 
            threshold=confidence_thresh
        )[0]
        class_1_count = 0
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = box.cpu().numpy()
            
            # Constraint 1: Check against known XML fish to prevent duplicates
            is_overlap = any(calculate_iou(box, c0_box) > 0.1 for c0_box in class_0_bboxes)
            if is_overlap: continue
                
            # Constraint 2: Force SAM to extract using the OwlViT bounding box
            masks, _, _ = sam_predictor.predict(box=box, multimask_output=False)
            poly_str = mask_to_yolo_polygon(masks[0], img_w, img_h)
            
            if poly_str:
                yolo_lines.append(f"1 {poly_str}")
                class_1_count += 1
                
        # --- Output ---
        if yolo_lines:
            txt_filepath = os.path.join(output_dir, os.path.splitext(img_name)[0] + ".txt")
            with open(txt_filepath, 'w') as f: f.write("\n".join(yolo_lines))
            
        print(f"Processed {img_name} -> Found {class_1_count} unannotated fish.")


xml_file_path = r"POSE ESTIMATION\XML ANNOTATIONS\UMT Dataset V1 Refixed\annotations_FIXED.xml"
images_directory = r"DATASETS FOR POSE ESTIMATION\UMT_dataset\UMT images for annotation V1 CLEANED"
output_labels_directory = r"DATASETS FOR INSTANCE SEGMENTATION\UMT_dataset\Owl+SAM_labels"
sam_file_path = r"INSTANCE SEGMENTATION\DATASET ANNOTATION STRATEGIES\sam_vit_h_4b8939.pth"

if __name__ == '__main__':
    build_grounded_sam_dataset(
        xml_path=xml_file_path,
        img_dir=images_directory,
        output_dir=output_labels_directory,
        sam_checkpoint=sam_file_path,
        confidence_thresh=0.15 # Lower this to 0.08 if it misses fish, raise to 0.25 if it grabs rocks
    )