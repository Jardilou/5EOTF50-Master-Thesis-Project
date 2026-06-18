import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

class SamSegmenter:
    """Wrapper for the Segment Anything Model to extract high-precision biological masks."""
    
    def __init__(self, model_type="vit_b", weights_path="weights/segmentation/sam_b.pt"):
        print(f"Loading SAM ({model_type}) from {weights_path}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam = sam_model_registry[model_type](checkpoint=weights_path)
        self.sam.to(device=device)
        self.predictor = SamPredictor(self.sam)
        
        self.current_image_path = None

    def generate_mask(self, image_path, bbox):
        """
        Generates a binary mask for a specific bounding box.
        bbox format: [xmin, ymin, xmax, ymax]
        """
        # Optimize inference by only encoding the image if it's a new frame
        if self.current_image_path != image_path:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(img)
            self.current_image_path = image_path
            
        input_box = np.array(bbox)
        
        # SAM returns multiple masks (resolutions). We take the one with the highest confidence score.
        masks, scores, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=True,
        )
        
        best_mask_idx = np.argmax(scores)
        binary_mask = masks[best_mask_idx]
        
        return binary_mask