import torch
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Grounding DINO specific imports
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model as load_dino_model, predict as predict_dino

class YoloDetector:
    """Wrapper for custom trained YOLO detection models."""
    def __init__(self, weights_path="weights/detection/best.pt", conf_threshold=0.25):
        print(f"Loading YOLO model from {weights_path}...")
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold

    def predict(self, image_path):
        """Returns a list of bounding boxes: [xmin, ymin, xmax, ymax]"""
        results = self.model(image_path, conf=self.conf_threshold, verbose=False)
        boxes = []
        for result in results:
            for box in result.boxes.xyxy:
                boxes.append(box.cpu().numpy().tolist())
        return boxes

class DinoDetector:
    """Wrapper for zero-shot Grounding DINO detection."""
    def __init__(self, config_path, weights_path, box_threshold=0.30, text_threshold=0.25):
        print("Loading Grounding DINO model...")
        self.model = load_dino_model(config_path, weights_path)
        self.box_thresh = box_threshold
        self.text_thresh = text_threshold
        self.prompt = "fish" # Base target

        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def predict(self, image_path):
        """Returns a list of bounding boxes scaled to original image dimensions."""
        img_source = Image.open(image_path).convert("RGB")
        img_tensor, _ = self.transform(img_source, None)
        
        boxes, logits, _ = predict_dino(
            model=self.model,
            image=img_tensor,
            caption=self.prompt,
            box_threshold=self.box_thresh,
            text_threshold=self.text_thresh
        )
        
        # DINO returns normalized [cx, cy, w, h]. Convert to absolute [xmin, ymin, xmax, ymax]
        W, H = img_source.size
        converted_boxes = []
        for box in boxes:
            cx, cy, w, h = box.numpy()
            cx, cy, w, h = cx * W, cy * H, w * W, h * H
            converted_boxes.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
            
        return converted_boxes