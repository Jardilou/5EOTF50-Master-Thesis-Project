"""
================================================================================
Script: YOLO Object Detection Only Training Routine
================================================================================

Description:
    A script to configure and execute the training loop for a YOLO-based 
    bounding box object detection model. The script is configured to train 
    exclusively on fish detection tasks, bypassing pose or segmentation architectures.
    It establishes specific hyperparameters, including the AdamW optimizer, 
    cosine annealing learning rate scheduling, and L2 regularization. It also 
    applies geometric and photometric data augmentations to simulate environmental 
    variations.

    Key Configuration Parameters:
    1. Model Architecture: Initializes a custom YOLO network via a .yaml file.
    2. Optimizer Settings: Utilizes AdamW with a base learning rate of 1e-3, 
       weight decay of 0.0005, and a cosine learning rate scheduler.
    3. Augmentations: Applies scaling, translation, horizontal flipping, mosaic, 
       and HSV (Hue, Saturation, Value) color shifts to increase dataset variance.
    4. Custom Loss Weights: Modifies the penalty multipliers for bounding box 
       regression (CIoU and DFL) and classification accuracy.

Author:
    Louis Jardinet
    Universiti Teknologi Petronas

Date:
    January 2026 - June 2026

Dependencies:
    - External packages: ultralytics (YOLO)

Inputs:
    - Model Configuration: 'yolov11_earmf_detection_only.yaml'
    - Dataset Configuration: 'Detection Only for IS.yaml'

Outputs:
    - Trained model weights (.pt files) and training metrics saved to the 
      'Detection_For_IS/detection_Only_training_run' directory.
================================================================================
"""

from ultralytics import YOLO

def main():
    print("[+] Initializing Baseline Object Detection Training...")
    
    # 1. Load the BASELINE Detection Weights (Not the -pose or -seg versions!)
    model = YOLO(r"YAML FILES\Detection-Only-for-IS-Network.yaml") # Use 'yolo11m.pt' if you want a larger, more accurate model
    
    # 2. Train the Model
    # We specify task='detect' to strictly lock the architecture
    
    results = model.train(
        data=r"YAML FILES\Detection-Only-for-IS-paths.yaml", # Your dataset config
        epochs=500,              # Paper used 500 epochs
        batch=16,                # Paper used batch size 8
        imgsz=1280,              # Minimum image size 1280, can be increased (1920x1080 for native resolution) but more computationally expensive
        optimizer='AdamW',       # Paper used AdamW
        lr0=1e-3,                # Initial learning rate
        weight_decay=0.0005,     # Explicitly set weight decay to 5e-4 (L2 Regularization)
        cos_lr=True,             # Enables Cosine Annealing learning rate scheduler
        lrf=0.01,                # Final learning rate fraction (stops it from hitting absolute 0)
        
        # --- 1. GEOMETRIC AUGMENTATIONS ---
        scale=0.5,        # Random scaling (+/- 50%) to handle size variations
        translate=0.1,    # Random translation (+/- 10%) to handle position variations
        fliplr=0.5,       # 50% chance to flip left-right 
        
        # --- 2. PHOTOMETRIC AUGMENTATIONS ---
        hsv_h=0.015,      # Slight hue shifts (water color changes)
        hsv_s=0.7,        # Saturation adjustments
        hsv_v=0.4,        # Value/Brightness adjustments (simulates deep/dark water vs shallow)
        mosaic=1.0,       # Mosaic augmentation (Default is 1.0)
        
        # =========================================================
        # CUSTOM LOSS WEIGHTS (The "Gains")
        # =========================================================
        box=7.5,     # Weight for Bounding Box CIoU loss (Default is 7.5)
        dfl=1.5,     # Weight for Bounding Box Edge/DFL loss (Default is 1.5)
        cls=3.0,     # Weight for Classifying it as a "fish" (Default is 0.5)
        # ========================================================
        
        project='Detection_For_IS',
        name='detection_Only_training_run',
        val=False                 # Disable validation during training
    )
    
    print("[+] Detection Training Complete!")

if __name__ == '__main__':
    main()