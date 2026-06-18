import os
import cv2
import torch
import numpy as np
import xml.etree.ElementTree as ET
import torch.nn.functional as F
import torchvision.transforms as T
from segment_anything import sam_model_registry, SamPredictor

def calculate_iou(boxA, boxB):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.
    
    Working Principle: Computes the area of the intersection of two boxes 
    and divides it by the area of their union to determine overlap.
        
    Inputs:
    - boxA (list/tuple): Coordinates of the first box [x_min, y_min, x_max, y_max].
    - boxB (list/tuple): Coordinates of the second box [x_min, y_min, x_max, y_max].

    Output:
    - float: The IoU score ranging from 0.0 to 1.0.
    """
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-9)

def mask_to_yolo_polygon(mask, img_width, img_height):
    """
    Converts a binary segmentation mask into a normalized YOLO-format polygon string.
    
    Working Principle: 
    The function detects the external boundaries (contours) within a binary mask. 
    It isolates the largest object by area, ensures it has enough points to form a 
    valid polygon (at least 3), and normalizes the pixel coordinates based on the 
    image dimensions. The result is formatted as a continuous string of coordinates.
    
    Modules Used: 
    - cv2 (OpenCV)
    - numpy (as np)
    
    Specific Functions Required: 
    - cv2.findContours() (extracts the contour points)
    - cv2.contourArea() (calculates the area to find the largest contour)
    - numpy.ndarray.astype() (ensures the mask is in the correct uint8 format)
    - Python built-ins: max(), len(), float division, f-string formatting, and str.join()
    
    Inputs:
    - mask (numpy.ndarray): A 2D binary array representing the object mask.
    - img_width (int/float): The width of the original image for normalization.
    - img_height (int/float): The height of the original image for normalization.
    
    Output:
    - str or None: A single space-separated string of normalized coordinates 
      ("x1 y1 x2 y2 ..."). Returns None if the mask is empty or invalid.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    largest_contour = max(contours, key=cv2.contourArea)
    # A valid polygon requires at least 3 points
    if len(largest_contour) < 3: return None
    # Normalize contour points to [0, 1] range based on image dimensions
    polygon = [f"{pt[0][0] / img_width:.6f} {pt[0][1] / img_height:.6f}" for pt in largest_contour]
    return " ".join(polygon)

def build_dino_sam_dataset(xml_path, img_dir, output_dir, sam_checkpoint, similarity_thresh=0.75):
    """
    Automates the generation of a YOLO-format polygon segmentation dataset by combining 
    DINOv2 for semantic similarity search and SAM for precise mask extraction.
    
    Working Principle:
    This function operates in three main phases:
    1. Template Extraction (Class 0): Parses XML ground truth bounding boxes, crops 
       these known objects from the image, and passes them through DINOv2 to extract 
       semantic features. These features are averaged to create a "Master Signature".
    2. Zero-Shot Discovery (Class 1): Sweeps the entire image with DINOv2 to extract 
       patch features. It computes cosine similarity against the Master Signature to 
       create a heatmap. "Hot zones" on this heatmap are converted into new bounding 
       boxes representing undiscovered objects of the same class.
    3. SAM Masking: Uses the Segment Anything Model (SAM) to generate pixel-perfect 
       polygon masks for both the known (Class 0) and newly discovered (Class 1) 
       bounding boxes. It filters out Class 1 boxes that overlap heavily with Class 0 
       and saves the final normalized polygons to YOLO-format text files.
       
    Modules Used & Specific Functions:
    - os: makedirs(), path.join(), path.exists(), path.splitext()
    - torch: cuda.is_available(), hub.load(), no_grad(), mean(), stack(), matmul()
    - torch.nn.functional (as F): normalize()
    - torchvision.transforms (as T): Compose(), ToTensor(), Normalize()
    - xml.etree.ElementTree (as ET): parse()
    - cv2 (OpenCV): imread(), cvtColor(), resize(), normalize(), threshold(), 
      findContours(), boundingRect()
    - numpy (as np): array()
    - Custom Dependencies: sam_model_registry, SamPredictor, mask_to_yolo_polygon, calculate_iou
    
    Inputs:
    - xml_path (str): Path to the XML file containing ground truth bounding boxes.
    - img_dir (str): Directory containing the source images.
    - output_dir (str): Destination directory for the generated YOLO .txt files.
    - sam_checkpoint (str): Path to the downloaded SAM model weights (.pth file).
    - similarity_thresh (float): Threshold (0 to 1) for the DINOv2 heatmap. Default: 0.75.
    
    Output:
    - The function writes YOLO format .txt files directly to the output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading DINOv2 (Semantic Feature Extractor)...")
    # Using ViT-Small for speed/memory, it is incredibly powerful
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
    dinov2.eval()
    
    # DINOv2 expects normalized ImageNet tensors : T.Normalize : ImageNet normalization (mean/std)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("Loading SAM (Mask Extractor)...")
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(device)
    sam_predictor = SamPredictor(sam)
    
    print("Parsing XML Ground Truth...")
    tree = ET.parse(xml_path)
    ground_truth_data = {}
    
    for image in tree.getroot().findall('image'):
        img_name = image.attrib['name']
        ground_truth_data[img_name] = []
        boxes = {b.attrib['group_id']: b for b in image.findall('box') if 'group_id' in b.attrib}
        for group_id, box in boxes.items():
            xtl, ytl = float(box.attrib['xtl']), float(box.attrib['ytl'])
            xbr, ybr = float(box.attrib['xbr']), float(box.attrib['ybr'])
            ground_truth_data[img_name].append([xtl, ytl, xbr, ybr])

    print("-" * 50)
    
    for img_name, boxes in ground_truth_data.items():
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path): continue
            
        cv2_img = cv2.imread(img_path)
        img_h, img_w = cv2_img.shape[:2]
        rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        
        # --- PHASE 1: Extract DINO Templates from Class 0 Boxes ---
        template_features = []
        class_0_bboxes = []
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            class_0_bboxes.append(box)
            
            # Crop the annotated fish
            crop = rgb_img[y1:y2, x1:x2]
            # Skip empty crops which can cause DINO to throw errors
            if crop.size == 0: continue
            
            # Resize to a multiple of 14 (DINOv2 patch size)
            crop_resized = cv2.resize(crop, (224, 224))
            # Convert to tensor and normalize for DINOv2
            crop_tensor = transform(crop_resized).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # Extract the CLS (class) token representation of the fish
                features = dinov2(crop_tensor)
                template_features.append(features)
                
        if not template_features:
            continue # No templates in this frame to search with
            
        # Average the features to create a "Master Signature" for fish in this specific frame
        master_template = torch.mean(torch.stack(template_features), dim=0)
        master_template = F.normalize(master_template, p=2, dim=1)
        
        # --- PHASE 2: Dense DINO Sweep of the Full Image ---
        # Resize image dimensions to be divisible by 14
        new_w = (img_w // 14) * 14
        new_h = (img_h // 14) * 14
        img_resized = cv2.resize(rgb_img, (new_w, new_h))
        img_tensor = transform(img_resized).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Extract patch tokens (skipping the CLS token)
            features_dict = dinov2.forward_features(img_tensor)
            patch_tokens = features_dict['x_norm_patchtokens'] # Shape: [1, N, 384]
            
        patch_tokens = F.normalize(patch_tokens[0], p=2, dim=1)
        
        # Calculate Cosine Similarity between the Master Template and every image patch
        similarities = torch.matmul(patch_tokens, master_template.T).squeeze()
        
        # Reshape the 1D similarity array back into a 2D heatmap
        grid_h, grid_w = new_h // 14, new_w // 14
        heatmap = similarities.reshape(grid_h, grid_w).cpu().numpy()
        
        # Normalize heatmap to 0-255
        heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Threshold the heatmap (Find the hot zones)
        _, thresh = cv2.threshold(heatmap_norm, int(similarity_thresh * 255), 255, cv2.THRESH_BINARY)
        
        # Scale the threshold mask back to original image size
        thresh_resized = cv2.resize(thresh, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        
        # Extract Bounding Boxes from the hot zones
        contours, _ = cv2.findContours(thresh_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # --- PHASE 3: SAM Extraction ---
        sam_predictor.set_image(rgb_img)
        yolo_lines = []
        
        # First, process Class 0
        for box in class_0_bboxes:
            masks, _, _ = sam_predictor.predict(box=np.array(box), multimask_output=False)
            poly_str = mask_to_yolo_polygon(masks[0], img_w, img_h)
            if poly_str: yolo_lines.append(f"0 {poly_str}")
                
        # Next, process Class 1 (DINO Heatmap Boxes)
        class_1_count = 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Filter microscopic hotspots
            if w < 15 or h < 15: continue 
                
            dino_box = [x, y, x + w, y + h]
            
            # Check overlap with Class 0
            if any(calculate_iou(dino_box, c0_box) > 0.1 for c0_box in class_0_bboxes):
                continue
                
            masks, _, _ = sam_predictor.predict(box=np.array(dino_box), multimask_output=False)
            poly_str = mask_to_yolo_polygon(masks[0], img_w, img_h)
            
            if poly_str:
                yolo_lines.append(f"1 {poly_str}")
                class_1_count += 1
                
        if yolo_lines:
            txt_filepath = os.path.join(output_dir, os.path.splitext(img_name)[0] + ".txt")
            with open(txt_filepath, 'w') as f: 
                f.write("\n".join(yolo_lines))
                
        print(f"Processed {img_name} -> Found {class_1_count} unannotated fish.")

if __name__ == '__main__':
    build_dino_sam_dataset(
        xml_path=r"POSE ESTIMATION\XML ANNOTATIONS\UMT Dataset V1 Refixed\annotations_FIXED.xml",
        img_dir=r"DATASETS FOR INSTANCE SEGMENTATION\UMT_dataset\UMT images for annotation V1 CLEANED Subset V2",
        output_dir=r"DATASETS FOR INSTANCE SEGMENTATION\UMT_dataset\DINO+SAM_labels Subset V2",
        sam_checkpoint=r"INSTANCE SEGMENTATION\DATASET ANNOTATION STRATEGIES\sam_vit_h_4b8939.pth",
        similarity_thresh=0.40 # Lower to 0.65 to find more camouflaged fish, raise to 0.85 to ignore rocks
    )