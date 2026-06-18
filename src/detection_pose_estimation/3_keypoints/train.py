"""
================================================================================
Script: YOLO-Pose Extremity Training Routine (Dual Strategy)
================================================================================

Description:
    A script to configure and execute the training loop for a YOLO-Pose model 
    tailored for anatomical extremity (head/upper tail-lower tail) detection. The script utilizes 
    the AdamW optimizer, cosine annealing learning rate scheduling, and aggressive 
    photometric and geometric augmentations. It also features a 
    customizable callback architecture designed to directly modify the model's 
    internal Binary Cross-Entropy (BCE) loss function, allowing researchers to 
    explicitly bias the network toward high recall by heavily penalizing false negatives.

    Key Functions:
    1. Custom Loss Injection (Callback): Defines a hook (`set_pos_weight`) to inject 
       a positive weight tensor into the BCE classification loss, forcing the model 
       to forgive false positives (hallucinations) while prioritizing the discovery 
       of difficult targets.
    2. Hyperparameter Configuration: Initializes the AdamW optimizer with a 
       defined base learning rate (1e-3), weight decay (L2 Regularization), and 
       cosine annealing scheduling (`cos_lr=True`).
    3. Augmentation Pipeline: Applies random scaling, translation, horizontal flipping, 
       mosaic, and HSV (Hue, Saturation, Value) color shifts to simulate variable 
       underwater environmental conditions.
    4. Loss Weight Balancing: Explicitly overrides the default penalty multipliers 
       for bounding box regression (box, dfl), pose keypoint accuracy (pose), and 
       classification confidence (cls).

Author:
    Louis Jardinet
    Universiti Teknologi Petronas

Date:
    January 2026 - June 2026

Dependencies:
    - Python standard libraries: os, multiprocessing
    - External packages: torch, ultralytics (YOLO)

Inputs:
    - Architecture Configuration: 'yolov11_dual_strategy.yaml'
    - Dataset Configuration: 'Dual Strategy Fish Pose.yaml'

Outputs:
    - Trained model weights (.pt files) and comprehensive training metrics saved 
      to the 'INSTANCE SEGMENTATION + POSE ESTIMATION\Runs\Dual_Strategy_Run_2' directory.
================================================================================
"""

import os
import torch
import torch.nn as nn
from ultralytics import YOLO

# --- ULTRALYTICS INJECTION IMPORTS ---
import ultralytics.nn.modules as nn_modules
from ultralytics.utils.loss import v8PoseLoss, xyxy2xywh
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import Bottleneck
import ultralytics.nn.tasks as tasks 


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
    # Make sure these names match your saved files exactly
    architecture_yaml = 'yolov11_dual_strategy.yaml'
    data_yaml = 'Dual Strategy Fish Pose.yaml'
    
    
    print("\n" + "="*50)
    print(" INITIATING ADAMW EXTREMITY SPRINT")
    print("="*50 + "\n")
    
    model_phase1 = YOLO(architecture_yaml) 
    # model_phase1.add_callback("on_train_start", set_pos_weight)
    model_phase1.train(
        data=data_yaml, 
        epochs=500,              
        batch=8,                 
        imgsz=1080,              
        optimizer='AdamW',       
        lr0=1e-3,                
        weight_decay=0.0005,     
        cos_lr=True,             
        lrf=0.01,                
        
        scale=0.5, translate=0.1, fliplr=0.5, 
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, mosaic=1.0,
        box=7.5, dfl=1.5, pose=12.0, cls=3.0,
        
        project=r"INSTANCE SEGMENTATION + POSE ESTIMATION\\Runs",
        name="Dual_Strategy_Run_2",
    )

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()




