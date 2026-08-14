"""
================================================================================
Script: Pre-Annotation via Pose XML and SAM (Segment Anything Model)
================================================================================

Description:
    A script to generate raw YOLO-format instance segmentation labels by utilizing 
    existing pose estimation XML data and the Segment Anything Model (SAM). 
    The script operates in two distinct phases to generate a comprehensive 
    initial dataset: targeted mask generation for known objects, and a blind 
    automated sweep for unannotated objects.

    Key Functions:
    1. XML Parsing: Extracts ground truth bounding box and keypoint (skeleton) 
       coordinates from pose estimation XML files.
    2. Prompted Segmentation (Class 0): Uses the extracted bounding boxes and 
       keypoints as specific prompts for SAM to generate precise polygon masks 
       for the primary known targets.
    3. Automated Sweep (Class 1): Deploys SAM's automatic mask generator to 
       segment all other distinct objects across the entire image. No geometric 
       or semantic filtering is applied at this stage.
    4. YOLO Conversion: Converts the resulting binary masks from both phases 
       into normalized polygon coordinates saved in YOLO .txt format.

Author:
    Louis Jardinet
    Universiti Teknologi Petronas

Date:
    January 2026 - June 2026


Dependencies:
    - Python standard libraries: os, xml.etree.ElementTree
    - External packages: opencv-python (cv2), numpy, segment_anything (SAM)

Inputs:
    - xml_path: Directory containing the XML file with pose annotations.
    - img_dir: Directory containing the corresponding source images.
    - sam_checkpoint: Path to the SAM model weights (.pth file).

Outputs:
    - output_dir: Directory where the unfiltered, raw YOLO .txt polygon 
      files are saved.
================================================================================
"""

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

def mask_to_yolo_polygon(mask, img_width, img_height):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    largest_contour = max(contours, key=cv2.contourArea)
    if len(largest_contour) < 3: return None
        
    polygon = [f"{pt[0][0] / img_width:.6f} {pt[0][1] / img_height:.6f}" for pt in largest_contour]
    return " ".join(polygon)

def generate_raw_dataset(xml_path, img_dir, output_dir, sam_checkpoint):
    print("Loading SAM models...")
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(device="cpu")
    predictor = SamPredictor(sam)
    
    # Highly sensitive generator - grabs everything
    mask_generator = SamAutomaticMaskGenerator(
        model=sam, points_per_side=32, pred_iou_thresh=0.86,
        stability_score_thresh=0.92, min_mask_region_area=100
    )
    os.makedirs(output_dir, exist_ok=True)
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ground_truth_data = {}

    for image in root.findall('image'):
        img_name = image.attrib['name']
        ground_truth_data[img_name] = []
        boxes = {box.attrib['group_id']: box for box in image.findall('box') if 'group_id' in box.attrib}
        skeletons = {skel.attrib['group_id']: skel for skel in image.findall('skeleton') if 'group_id' in skel.attrib}
        
        for group_id, box in boxes.items():
            xtl, ytl = float(box.attrib['xtl']), float(box.attrib['ytl'])
            xbr, ybr = float(box.attrib['xbr']), float(box.attrib['ybr'])
            points_list = []
            if group_id in skeletons:
                for pt in skeletons[group_id].findall('points'):
                    coords = pt.attrib['points'].split(',')
                    points_list.append([float(coords[0]), float(coords[1])])
            ground_truth_data[img_name].append({'box': [xtl, ytl, xbr, ybr], 'points': points_list})

    print(f"Generating raw masks for {len(ground_truth_data)} images...")

    for img_name, annotations in ground_truth_data.items():
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            print(f"FAILED TO FIND: {img_path}")
            continue
            
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        predictor.set_image(img_rgb)
        yolo_lines = []
        
        # 1. Targeted Predictor for Class 0
        for ann in annotations:
            box = np.array(ann['box'])
            input_point = np.array(ann['points']) if ann['points'] else None
            input_label = np.ones(len(ann['points'])) if ann['points'] else None
            
            masks, _, _ = predictor.predict(
                point_coords=input_point, point_labels=input_label, box=box, multimask_output=False
            )
            poly_str = mask_to_yolo_polygon(masks[0], img_width, img_height)
            if poly_str: yolo_lines.append(f"0 {poly_str}") 
                
        # 2. Blind Sweep for Class 1 (NO filtering constraints here)
        auto_masks = mask_generator.generate(img_rgb)
        for auto_mask in auto_masks:
            poly_str = mask_to_yolo_polygon(auto_mask['segmentation'], img_width, img_height)
            if poly_str: yolo_lines.append(f"1 {poly_str}")
                        
        txt_filepath = os.path.join(output_dir, os.path.splitext(img_name)[0] + ".txt")
        if yolo_lines:
            with open(txt_filepath, 'w') as f: f.write("\n".join(yolo_lines))
        print(f"Dumped raw output for {img_name}")


xml_file_path = r"POSE ESTIMATION\XML ANNOTATIONS\UMT Dataset V1 Refixed\annotations_FIXED.xml"
images_directory = r"DATASETS FOR POSE ESTIMATION\UMT_dataset\UMT images for annotation V1 CLEANED"
output_labels_directory = r"DATASETS FOR INSTANCE SEGMENTATION\UMT_dataset\labels"
sam_file_path = r"INSTANCE SEGMENTATION\DATASET ANNOTATION STRATEGIES\sam_vit_h_4b8939.pth"
if __name__ == '__main__':
    generate_raw_dataset(xml_file_path,images_directory, output_labels_directory, sam_file_path)