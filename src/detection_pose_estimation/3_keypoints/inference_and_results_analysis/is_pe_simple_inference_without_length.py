"""
================================================================================
Script: YOLO-Pose Raw Inference Debugger
================================================================================

Description:
    A script to run inference on a directory of test images using a trained 
    YOLO-Pose model and visually output the raw predictions. It utilizes the 
    native plotting functions of the Ultralytics library to draw bounding boxes, 
    confidence scores, and predicted keypoints directly onto the images. The 
    annotated images are then saved to an output directory to allow for a visual 
    review of the model's performance and to identify hallucinated detections 
    or missed targets.

    Key Functions:
    1. Model Initialization: Loads a trained YOLO-Pose weight file (.pt).
    2. Batch Inference: Iterates through a specified folder of raw images, 
       applying the model with a defined confidence threshold (0.25) and using 
       rectangular inference (rect=True) to preserve the original aspect ratios.
    3. Visual Annotation: Extracts the prediction arrays and uses the YOLO 
       `.plot()` method to render the spatial data into visual bounding boxes 
       and keypoint markers.
    4. Diagnostic Overlay: Computes the total number of detections per frame 
       and writes this metric onto the image as a text overlay.
    5. File Export: Saves the fully annotated diagnostic images to a designated 
       output folder.

Author:
    Louis Jardinet
    Universiti Teknologi Petronas

Date:
    January 2026 - June 2026

Dependencies:
    - Python standard libraries: os, glob
    - External packages: opencv-python (cv2), ultralytics (YOLO)

Inputs:
    - YOLO_MODEL_PATH: Path to the trained YOLO-Pose model weights.
    - TEST_IMAGE_FOLDER: Directory containing the raw images to be tested.

Outputs:
    - DEBUG_OUTPUT_DIR: Directory where the annotated diagnostic images are saved.
================================================================================
"""

# Cell: Raw YOLO Pose X-Ray Debugger
import os
import glob
import cv2
from ultralytics import YOLO

# --- 1. SETUP PATHS ---
# Point this to your actual trained Pose model
YOLO_MODEL_PATH = r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\ultralytics\runs\pose\INSTANCE SEGMENTATION + POSE ESTIMATION\Runs\Dual_Strategy_Run_2\weights\last.pt"

# The folder full of your raw test images
TEST_IMAGE_FOLDER = r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\DATASETS\UNTOUCHED_UMT_IMAGES_COMBINED\ADEQUATE UNTOUCHED UMT IMAGES"

# Where you want the debug images to be saved
DEBUG_OUTPUT_DIR = r"INSTANCE SEGMENTATION + POSE ESTIMATION\INFERENCE AND RESULTS ANALYSIS\Raw_Inference_Debug_Images_Dual_Strategy_Run_2"
os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)

# --- 2. LOAD MODEL ---
print("[+] Loading YOLO Pose Model...")
try:
    model = YOLO(YOLO_MODEL_PATH)
except Exception as e:
    print(f"[!] FATAL: Could not load model. Error: {e}")
    raise

# --- 3. GET IMAGES ---
image_files = glob.glob(os.path.join(TEST_IMAGE_FOLDER, "*.jpg")) + \
              glob.glob(os.path.join(TEST_IMAGE_FOLDER, "*.png"))

total_images = len(image_files)
print(f"[+] Found {total_images} images. Starting raw inference engine...\n")

# --- 4. THE INFERENCE LOOP ---
processed_count = 0

for img_path in image_files:
    filename = os.path.basename(img_path)
    
    # 1. Run raw inference
    # conf=0.25: Low enough to see hallucinations, high enough to filter total static.
    # rect=True: Prevents the image from being squished into a square, preserving fish shapes.
    # imgsz=1024: Assuming you are using high-res underwater photos. Change to 640 if you trained on 640.
    results = model.predict(source=img_path, imgsz=1024, conf=0.25, rect=True, verbose=False)
    
    result = results[0]
    
    # 2. Generate YOLO's native visualization (draws boxes, confidences, and keypoints)
    # The plot() function returns a BGR numpy array ready for OpenCV
    annotated_frame = result.plot(line_width=1, font_size=1, kpt_radius=1, txt_color=(255, 0, 0))
    
    # 3. Add a diagnostic text overlay to the image
    num_detections = len(result.boxes)
    status_color = (0, 255, 0) if num_detections > 0 else (0, 0, 255)
    cv2.putText(annotated_frame, f"Detections: {num_detections} | Conf: 0.25 | Rect: True", 
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    
    # 4. Save to the debug folder
    output_path = os.path.join(DEBUG_OUTPUT_DIR, f"DEBUG_{filename}")
    cv2.imwrite(output_path, annotated_frame)
    
    processed_count += 1
    if processed_count % 10 == 0:
        print(f" -> Processed {processed_count}/{total_images} images...")

print("\n" + "="*50)
print(" X-RAY DEBUGGING COMPLETE")
print(f" -> Check your images here: {DEBUG_OUTPUT_DIR}")
print("="*50)