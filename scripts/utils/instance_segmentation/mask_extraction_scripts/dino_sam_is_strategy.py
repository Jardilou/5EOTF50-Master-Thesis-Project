import os
import cv2
import glob
import torch
import numpy as np
from pathlib import Path
import groundingdino 
from PIL import Image

# --- Import DINO and SAM ---
# Adjust these imports based on your specific GroundingDINO/SAM installation
from groundingdino.util.inference import load_model, predict, annotate
import groundingdino.datasets.transforms as T
from segment_anything import sam_model_registry, SamPredictor

# ---------------------------------------------------------
# 1. CONFIGURATION & MODEL LOADING
# ---------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths to your downloaded weights
dino_pkg_dir = os.path.dirname(groundingdino.__file__)
DINO_CONFIG = os.path.join(dino_pkg_dir, "config", "GroundingDINO_SwinT_OGC.py")
DINO_WEIGHTS = "groundingdino_swint_ogc.pth"
SAM_WEIGHTS = "sam_b.pt"
SAM_TYPE = "vit_b"

print(f"Loading Grounding DINO on {DEVICE}...")
dino_model = load_model(DINO_CONFIG, DINO_WEIGHTS).to(DEVICE)

print(f"Loading SAM ({SAM_TYPE}) on {DEVICE}...")
sam = sam_model_registry[SAM_TYPE](checkpoint=SAM_WEIGHTS).to(DEVICE)
sam_predictor = SamPredictor(sam)

# ---------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------
def transform_image_for_dino(image_bgr):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Convert the NumPy array to a PIL Image
    image_pil = Image.fromarray(image_rgb)
    
    # Pass the PIL image and None (for target) as positional arguments
    image_transformed, _ = transform(image_pil, None)
    
    return image_transformed

def extract_fish_mask(image_bgr, text_prompt="fish", box_threshold=0.35, text_threshold=0.25):
    """
    Finds fish using Grounding DINO and segments them using SAM.
    Returns a combined binary mask of all detected fish.
    """
    # 1. DINO Object Detection
    image_transformed = transform_image_for_dino(image_bgr)
    boxes, logits, phrases = predict(
        model=dino_model,
        image=image_transformed,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=DEVICE
    )
    
    if len(boxes) == 0:
        return None # No fish found

    # Convert DINO boxes (cx, cy, w, h) in relative [0,1] coords 
    # to absolute SAM boxes (x1, y1, x2, y2)
    h, w, _ = image_bgr.shape
    boxes_xyxy = boxes * torch.Tensor([w, h, w, h])
    boxes_xyxy[:, :2] -= boxes_xyxy[:, 2:] / 2  # top-left
    boxes_xyxy[:, 2:] += boxes_xyxy[:, :2]      # bottom-right
    
    # 2. SAM Segmentation
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(image_rgb)
    
    # Transform boxes to SAM's tensor format
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_rgb.shape[:2]).to(DEVICE)
    
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False
    )
    
    # Masks are shape (N, 1, H, W). Combine them into a single 2D binary mask.
    combined_mask = torch.any(masks, dim=0).squeeze().cpu().numpy()
    
    # Convert boolean mask to uint8 (0 and 255)
    binary_mask = (combined_mask * 255).astype(np.uint8)
    return binary_mask

# ---------------------------------------------------------
# 3. BATCH PROCESSING LOOP
# ---------------------------------------------------------
# Directories
INPUT_DIR = r"DATASETS\DATASETS FOR POSE ESTIMATION\UMT_dataset\BACKUP\Original Dataset\Images 1-190714\St3-Pantai Vietnam" 
OUTPUT_DIR = r"DATASETS\DATASETS FOR POSE ESTIMATION\UMT_dataset\BACKUP\Original Dataset\Images 1-190714\St3-Pantai Vietnam\right_masks"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find all RIGHT images (Adjust the glob pattern if needed)
right_images = glob.glob(os.path.join(INPUT_DIR, "**", "*(R)*.jpg"), recursive=True)

print(f"Found {len(right_images)} Right images to process.")

for img_path in right_images:
    filename = Path(img_path).name
    print(f"Segmenting: {filename}")
    
    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
        print(f"  -> Error reading image. Skipping.")
        continue
        
    # Get mask
    mask = extract_fish_mask(image_bgr)
    
    if mask is not None:
        # Save mask as PNG (e.g., "Right_Image_01_mask.png")
        stem = Path(img_path).stem
        save_path = os.path.join(OUTPUT_DIR, f"{stem}_mask.png")
        cv2.imwrite(save_path, mask)
        print(f"  -> Saved mask: {save_path}")
    else:
        print("  -> No fish detected.")

print("\nBatch Segmentation Complete! You can now pass these masks into the batch_measure() function.")