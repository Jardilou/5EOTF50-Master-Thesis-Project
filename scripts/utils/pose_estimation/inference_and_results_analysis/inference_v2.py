"""
================================================================================
Script: Multi-Dataset YOLO Simple Inference & Visualization Engine
================================================================================

Description:
    A batch inference script designed to automatically deploy a trained YOLO 
    model across multiple independent validation or test datasets sequentially. 
    The script generates visual prediction overlays and saves them to organized 
    sub-directories. It applies specific Non-Maximum Suppression (NMS) and 
    confidence thresholds to suppress background noise and resolve overlapping 
    duplicate bounding boxes.

    Key Functions:
    1. Model Initialization: Loads the optimal trained weights (`best.pt`) 
       into memory for inference.
    2. Batch Processing: Iterates through a predefined list of target 
       directories containing raw test images from different datasets.
    3. Noise & Overlap Filtering: Applies a 0.35 confidence threshold to ignore 
       weak predictions. Uses class-agnostic NMS with an IoU threshold of 0.45 
       to merge overlapping duplicate bounding boxes, preventing the "double-box" 
       issue on dense targets.
    4. Automated Export: Uses Ultralytics' native plotting capabilities to draw 
       boxes and keypoints, saving the annotated images into separate folders 
       (e.g., Dataset_1, Dataset_2) within a centralized project directory.

Author:
    Louis Jardinet
    Universiti Teknologi Petronas

Date:
    January 2026 - June 2026

Dependencies:
    - External packages: ultralytics (YOLO)

Inputs:
    - model: Path to the trained YOLO model weights.
    - val_folders: A list of directory paths containing the unseen images 
      to be processed.

Outputs:
    - Annotated prediction images sorted into distinct sub-folders within the 
      specified inference results directory.
================================================================================
"""

from ultralytics import YOLO

def predict_multiple_val_sets():
    # 1. Load your best weights
    model = YOLO(r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\ultralytics\runs\pose\BezierFusion\eamrf_training_run24\weights\best.pt")

    # 2. List all your validation folders here
    val_folders = [
        r"DATASETS\DATASETS FOR POSE ESTIMATION\DeepFish\CURRENT\First_batch_Train_Test\images\val",
        r"DATASETS\DATASETS FOR POSE ESTIMATION\UMT_dataset\CURRENT\UMT V1 First batch Train Test Split\images\val"
        
        

    ]

    # 3. Loop through them and let Ultralytics do the work
    for i, folder in enumerate(val_folders):
        print(f"\n🚀 Processing Validation Set {i + 1}...")
        
        model.predict(
            source=folder, 
            save=True, 
            conf=0.35,          # Ignore background noise
            iou=0.45,           # Fix the double-box NMS issue
            agnostic_nms=True,  # Merge boxes across classes
            line_width=2,
            project=r"POSE ESTIMATION\INFERENCE AND RESULTS ANALYSIS\INFERENCE_RESULTS\Inference_results_run24", # Main output folder
            name=f"Dataset_{i + 1}"               # Sub-folder for this specific set
        )

    print("\nAll validation sets processed perfectly!")

if __name__ == '__main__':
    predict_multiple_val_sets()