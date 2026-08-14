import cv2
import numpy as np
import torch
import os

class SegmentationEngine:
    """
    Unified wrapper for extracting high-precision biological masks.
    Supports either Zero-Shot SAM 2 or Custom YOLO (v8/11) Instance Segmentation.
    """
    
    def __init__(self, mode="sam2", weights_path=None, model_cfg=None):
        """
        Initializes the chosen segmentation engine.
        
        Args:
            mode (str): Either "sam2" or "yolo".
            weights_path (str): Path to the model weights. 
                                Default for SAM2: 'weights/sam2_hiera_large.pt'
                                Default for YOLO: 'weights/best-seg.pt'
            model_cfg (str): Config file required for SAM2.
                             Default: 'sam2_hiera_l.yaml'
        """
        self.mode = mode.lower()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_image_path = None
        
        print(f"Initializing Segmentation Engine in '{self.mode}' mode on {self.device}...")
        
        if self.mode == "sam2":
            # --- SAM 2 INITIALIZATION ---
            try:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
            except ImportError:
                print("CRITICAL ERROR: SAM 2 is not installed.")
                print("Please install it using: pip install git+https://github.com/facebookresearch/sam2.git")
                raise
                
            if weights_path is None:
                weights_path = "weights\segmentation\sam21_hiera_large.pt"
            if model_cfg is None:
                model_cfg = r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\weights\segmentation\segment-anything-2\sam2\configs\sam2.1\sam2.1_hiera_l.yaml"
                
            print(f"Loading SAM 2 weights from {weights_path}...")
            self.sam2_model = build_sam2(model_cfg, weights_path, device=self.device)
            self.predictor = SAM2ImagePredictor(self.sam2_model)
            
        elif self.mode == "yolo":
            # --- YOLO INSTANCE SEGMENTATION INITIALIZATION ---
            try:
                from ultralytics import YOLO
            except ImportError:
                print("CRITICAL ERROR: Ultralytics YOLO is not installed.")
                raise
                
            if weights_path is None:
                weights_path = "weights\segmentation\yolo11n-seg.pt"
                
            print(f"Loading YOLO Segmentation weights from {weights_path}...")
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"YOLO weights not found at {weights_path}")
            
            self.yolo_model = YOLO(weights_path)
            
        else:
            raise ValueError("Invalid segmentation mode. Choose 'sam2' or 'yolo'.")

    def generate_mask(self, image_path, bbox):
        """
        Generates a binary mask for a specific bounding box.
        bbox format: [xmin, ymin, xmax, ymax]
        """
        if self.mode == "sam2":
            return self._generate_mask_sam2(image_path, bbox)
        elif self.mode == "yolo":
            return self._generate_mask_yolo(image_path, bbox)

    def _generate_mask_sam2(self, image_path, bbox):
        """Internal method for SAM 2 mask generation."""
        # Optimize inference by only encoding the image if it's a new frame
        if self.current_image_path != image_path:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error loading image: {image_path}")
                return np.zeros((10, 10), dtype=bool) # Fallback empty mask
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(img_rgb)
            self.current_image_path = image_path
            
        input_box = np.array(bbox)
        
        # SAM 2 prediction based on bounding box prompt
        masks, scores, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=True,
        )
        
        # Take the highest confidence mask
        best_mask_idx = np.argmax(scores)
        binary_mask = masks[best_mask_idx]
        
        return binary_mask

    def _generate_mask_yolo(self, image_path, bbox):
        """
        Internal method for YOLO instance segmentation mask generation.
        Note: YOLO processes the whole image at once. If we call this per-box, 
        we need to find the YOLO mask that best matches the provided bounding box.
        """
        # 1. Run YOLO inference on the image
        results = self.yolo_model(image_path, verbose=False)
        result = results[0] # Get the first (and only) image result
        
        img = cv2.imread(image_path)
        h_orig, w_orig = img.shape[:2]
        
        # Fallback if no masks are found at all
        if result.masks is None:
            return np.zeros((h_orig, w_orig), dtype=bool)

        # Extract predicted boxes and masks
        pred_boxes = result.boxes.xyxy.cpu().numpy()
        
        # Ultralytics scales masks down to 160x160 for speed. 
        # We must scale them back up to the original image size using nearest-neighbor interpolation.
        masks_tensor = result.masks.data
        masks_upscaled = torch.nn.functional.interpolate(
            masks_tensor.unsqueeze(1), 
            size=(h_orig, w_orig), 
            mode="nearest"
        ).squeeze(1).cpu().numpy()
        
        # 2. Find the YOLO detection that matches the input `bbox`
        # We calculate the Intersection over Union (IoU) to find the right mask
        input_xmin, input_ymin, input_xmax, input_ymax = bbox
        
        best_iou = 0
        best_mask_idx = -1
        
        for i, pred_box in enumerate(pred_boxes):
            pxmin, pymin, pxmax, pymax = pred_box
            
            # Calculate Intersection
            ixmin = max(input_xmin, pxmin)
            iymin = max(input_ymin, pymin)
            ixmax = min(input_xmax, pxmax)
            iymax = min(input_ymax, pymax)
            
            if ixmax < ixmin or iymax < iymin:
                inter_area = 0
            else:
                inter_area = (ixmax - ixmin) * (iymax - iymin)
                
            # Calculate Union
            box1_area = (input_xmax - input_xmin) * (input_ymax - input_ymin)
            box2_area = (pxmax - pxmin) * (pymax - pymin)
            union_area = box1_area + box2_area - inter_area
            
            iou = inter_area / union_area if union_area > 0 else 0
            
            if iou > best_iou:
                best_iou = iou
                best_mask_idx = i
                
        # 3. Return the best matching mask, or an empty one if no overlap was found
        if best_mask_idx != -1 and best_iou > 0.3: # Minimum 30% overlap required
            return masks_upscaled[best_mask_idx] > 0.5 # Convert probabilities to binary boolean mask
        else:
             return np.zeros((h_orig, w_orig), dtype=bool)