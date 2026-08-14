"""
================================================================================
Script: Bézierfusion YOLO Training Routine (Custom Architecture)
================================================================================

Description:
    A script to configure and execute the training loop for the custom 
    "Bézierfusion" YOLO model from scratch. This script initializes a modified 
    YOLO architecture from a YAML file to accommodate structural changes (such 
    as the EAMRF module) without conflicting with standard pre-trained weights. 
    It features robust data augmentations, the AdamW optimizer, and a custom 
    callback designed to directly bias the network's classification loss.

    Key Functions:
    1. Custom Loss Injection (Callback): Defines and attaches a hook (`set_pos_weight`) 
       that injects a positive weight tensor (3.0) into the Binary Cross-Entropy (BCE) 
       loss. This explicitly forces the model to prioritize recall by penalizing 
       false negatives (missed targets) more heavily than false positives.
    2. Architecture Initialization: Instantiates the model directly from 
       `yolov8_Bézier.yaml` to ensure custom internal modules are built correctly 
       before training begins.
    3. Hyperparameter Configuration: Sets up the AdamW optimizer with an initial 
       learning rate of 1e-3, L2 weight decay, and a cosine annealing scheduler.
    4. Augmentation Pipeline: Applies aggressive geometric (scaling, translation, 
       flipping) and photometric (HSV shifts, high-intensity mosaic) augmentations 
       to handle extreme variations in underwater imagery.
    5. Loss Weight Balancing: Customizes the penalty multipliers across the network, 
       specifically assigning a high weight (12.0) to the custom Bézier geometry 
       (pose) loss.

Author:
    Louis Jardinet
    Universiti Teknologi Petronas

Date:
    January 2026 - June 2026

Dependencies:
    - Python standard libraries: multiprocessing
    - External packages: torch, ultralytics (YOLO)

Inputs:
    - Architecture Configuration: 'yolov8_Bézier.yaml'
    - Dataset Configuration: 'fish_bezier.yaml'

Outputs:
    - Trained model weights (.pt files) and training metrics saved to the 
      'BezierFusion/eamrf_training_run' directory.
================================================================================
"""

from ultralytics import YOLO
import torch

# 1. Define the callback function
def set_pos_weight(trainer):
    """
    Injects a positive weight into the BCE loss.
    A weight of 3.0 means a False Negative (missing a fish) is 
    penalized 3x more heavily than a False Positive (finding an unannotated fish).
    """
    weight_value = 3.0  
    pos_weight = torch.tensor([weight_value], dtype=torch.float32)
    
    # Ensure the loss criterion is initialized before we modify it
    if getattr(trainer.model, "criterion", None) is None:
        trainer.model.criterion = trainer.model.init_criterion()
        
    # Grab the BCE classification loss and inject the custom weight
    bce_loss = getattr(trainer.model.criterion, "bce", None)
    if bce_loss is not None:
        bce_loss.pos_weight = pos_weight.to(trainer.device)
        print(f"\n INJECTED: BCE pos_weight set to {bce_loss.pos_weight.tolist()}")
        print(" BEHAVIOR: Model will forgive False Positives and prioritize Recall.\n")


def main():
    # 1. Load your custom architecture YAML and attach the callback to set the pos_weight at the start of training
    # (Do NOT load a pre-trained .pt file here, because the pre-trained 
    # weights don't have the EAMRF module and it will crash)
    model = YOLO(r'yolov8_Bézier.yaml') 
    model.add_callback("on_train_start", set_pos_weight)
    
    # 2. Train the model using the paper's specifications
    results = model.train(
        data='fish_bezier.yaml', # Your dataset config
        epochs=500,              # Paper used 500 epochs
        batch=8,                 # Paper used batch size 8
        imgsz=640,
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
        mosaic=2.0,       # Mosaic augmentation (Default is 1.0)
        
        # =========================================================
        # CUSTOM LOSS WEIGHTS (The "Gains")
        # =========================================================
        box=7.5,     # Weight for Bounding Box CIoU loss (Default is 7.5)
        dfl=1.5,     # Weight for Bounding Box Edge/DFL loss (Default is 1.5)
        pose=12.0,   # Weight for your custom Bézier Geometry loss (Default is 12.0)
        cls=3.0,     # Weight for Classifying it as a "fish" (Default is 0.5)
        # ========================================================
        
        project='BezierFusion',
        name='eamrf_training_run',
        val=False                 # Disable validation during training
    )

if __name__ == '__main__':
    main()

# from ultralytics import YOLO
# import os

# def main():
#     # =========================================================
#     # PHASE 1: The AdamW Sprint (First 450 Epochs)
#     # =========================================================
#     print("Starting Phase 1: Rapid convergence with AdamW...")
    
#     # Load your custom architecture YAML (no pre-trained weights)
#     model_phase1 = YOLO('yolov8_Bézier.yaml') 
    
#     model_phase1.train(
#         data='fish_bezier.yaml', 
#         epochs=450,              # 90% of the total 500 epochs
#         batch=8,                 
#         imgsz=640,
#         optimizer='AdamW',       # Switched to AdamW for rapid, adaptive learning
#         lr0=1e-3,                # Initial learning rate
#         weight_decay=0.0005,     
#         cos_lr=True,             
#         lrf=0.01,                
        
#         # --- GEOMETRIC AUGMENTATIONS ---
#         scale=0.5, translate=0.1, fliplr=0.5, 
        
#         # --- PHOTOMETRIC AUGMENTATIONS ---
#         hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, 
        
#         # =========================================================
#         # CUSTOM LOSS WEIGHTS
#         # =========================================================
#         box=7.5,     # Weight for Bounding Box CIoU loss (Default is 7.5)
#         dfl=1.5,     # Weight for Bounding Box Edge/DFL loss (Default is 1.5)
#         pose=12.0,   # Weight for your custom Bézier Geometry loss (Default is 12.0)
#         cls=3.0,     # Weight for Classifying it as a "fish" (Default is 0.5)
#         # ========================================================
        
#         project=r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\ultralytics\runs\pose\BezierFusion",
#         name='Phase1_AdamW_run2',
#         val=False                # No validation = no best.pt, only last.pt generated
#     )

#     # =========================================================
#     # PHASE 2: The SGD Marathon (Final 50 Epochs)
#     # =========================================================
#     print("\n Starting Phase 2: Fine-tuning and generalization with SGD...")
    
#     # Grab the final weights from Phase 1 to act as our new starting point
#     phase1_weights = os.path.join(r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\ultralytics\runs\pose\BezierFusion\Phase1_AdamW_run2\weights\last.pt")
    
#     if not os.path.exists(phase1_weights):
#         print(f"Error: Could not find Phase 1 weights at {phase1_weights}")
#         return
        
#     model_phase2 = YOLO(phase1_weights)
    
#     model_phase2.train(
#         data='fish_bezier.yaml', 
#         epochs=50,               # The remaining 10%
#         batch=5,                 
#         imgsz=640,
#         optimizer='SGD',         # Switch to SGD to explore flatter, generalized minima
#         lr0=1e-4,                # CRITICAL: Drop learning rate 10x (from 1e-3 to 1e-4) to prevent shock
#         weight_decay=0.0005,     
#         cos_lr=True,             
#         lrf=0.01,
        
#         # Keep all augmentations and loss weights identical so the environment doesn't shift
#         scale=0.5, translate=0.1, fliplr=0.5,
#         hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
#         box=7.5, dfl=1.5, pose=12.0, cls=3.0,
        
#         project=r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\ultralytics\runs\pose\BezierFusion",
#         name='Phase2_SGD_run2',
#         val=False
#     )
    
#     print("\n Two-Phase Training Complete! Your final, fine-tuned weights are in Phase2_SGD.")

# if __name__ == '__main__':
#     main()