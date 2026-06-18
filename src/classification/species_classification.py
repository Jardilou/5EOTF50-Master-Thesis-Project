import os
import cv2
import torch
import pickle
import numpy as np
from PIL import Image
import torchvision.transforms as T

class SpeciesClassifier:
    """Classifies the specific species of a detected fish using DINOv2 embeddings + Logistic Regression."""
    
    def __init__(self, dinov2_model="dinov2_vits14", classifier_weights_path="weights/dino_classifier.pkl"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading DINOv2 foundation backbone ({dinov2_model}) on {self.device}...")
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', dinov2_model)
        self.dinov2.to(self.device)
        self.dinov2.eval() # Set to evaluation mode
        
        print(f"Loading trained Logistic Regression classifier from {classifier_weights_path}...")
        if os.path.exists(classifier_weights_path):
            with open(classifier_weights_path, 'rb') as f:
                # Assuming you saved a dictionary: {'clf': clf, 'classes': database_classes}
                # in your Jupyter notebook after training.
                saved_data = pickle.load(f)
                
                # Handle both raw classifier loads and dictionary loads
                if isinstance(saved_data, dict) and 'clf' in saved_data:
                    self.clf = saved_data['clf']
                    self.database_classes = saved_data.get('classes', [])
                else:
                    self.clf = saved_data
                    self.database_classes = None
        else:
            print(f"Warning: Classifier weights not found at {classifier_weights_path}.")
            self.clf = None

        # Standard inference transforms for DINOv2
        self.inference_transform = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
    def _get_dino_embeddings(self, crop_rgb):
        """Extracts DINOv2 foundation embeddings from the RGB image crop."""
        pil_img = Image.fromarray(crop_rgb)
        img_tensor = self.inference_transform(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.dinov2(img_tensor)
            
        return features.cpu().numpy()
        
    def predict(self, image_path, bbox):
        """
        Crops the bounding box from the image, extracts DINOv2 embeddings, 
        and runs classification. Returns the species string.
        """
        if self.clf is None:
            return "Unknown"

        img = cv2.imread(image_path)
        if img is None:
            return "Unknown"
            
        xmin, ymin, xmax, ymax = map(int, bbox)
        
        # Add safety checks to ensure crop is within image bounds
        h, w = img.shape[:2]
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(w, xmax), min(h, ymax)
        
        # Failsafe for out-of-bounds bounding boxes
        if xmax <= xmin or ymax <= ymin:
            return "Unknown"
            
        # Crop the specific fish
        crop = img[ymin:ymax, xmin:xmax]
        if crop.size == 0:
            return "Unknown"
            
        # Convert BGR to RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Extract embeddings
        emb = self._get_dino_embeddings(crop_rgb)
        
        # Predict using Logistic Regression
        pred_val = self.clf.predict(emb)[0]
        
        # Map prediction back to string if it returned an integer index
        if self.database_classes and isinstance(pred_val, (int, np.integer)):
            pred_species = self.database_classes[pred_val]
        else:
            pred_species = str(pred_val)
            
        return pred_species