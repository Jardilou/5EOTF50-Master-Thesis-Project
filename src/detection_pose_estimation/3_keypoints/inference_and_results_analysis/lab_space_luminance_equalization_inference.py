"""
================================================================================
Script: LAB Color Space Enhancement and YOLO Inference Pipeline
================================================================================

Description:
    A script to evaluate trained YOLO-Pose models using a specialized image 
    preprocessing pipeline. It converts raw BGR images into the CIELAB color 
    space and applies Contrast Limited Adaptive Histogram Equalization (CLAHE) 
    exclusively to the Lightness (L) channel. This technique enhances underwater 
    edge contrast without introducing chromatic distortion. The script feeds the 
    enhanced images into the YOLO model, records pipeline latency, and exports 
    side-by-side comparative images for visual analysis.

    Key Functions:
    1. Directory Management: Safely clears and rebuilds the output directory 
       to prevent data overlapping from previous runs.
    2. LAB Equalization: Transforms images to CIELAB, applies CLAHE to the 
       L channel to boost contrast while suppressing background particulate noise, 
       and merges the channels back to BGR format.
    3. Inference Integration: Passes the dynamically enhanced image arrays 
       directly into the YOLO model for prediction.
    4. Comparative Export: Generates horizontally stacked images showing the 
       raw input alongside the enhanced and annotated output.
    5. Performance Profiling: Measures total pipeline latency (preprocessing 
       plus inference) to estimate real-time Frames Per Second (FPS).

Author:
    Louis Jardinet
    Universiti Teknologi Petronas

Date:
    January 2026 - June 2026

Dependencies:
    - Python standard libraries: os, time, shutil
    - External packages: opencv-python (cv2), numpy, ultralytics (YOLO)

Inputs:
    - weights_path: Path to the trained YOLO-Pose model weights (.pt).
    - test_images_dir: Directory containing the unseen raw validation/test images.

Outputs:
    - output_dir: Directory where the side-by-side comparative diagnostic 
      images are saved.
================================================================================
"""

import os
import cv2
import numpy as np
import time
import shutil
from ultralytics import YOLO

def build_output_dir(base_dir):
    """Safely builds a fresh output directory."""
    out_path = os.path.join(base_dir, 'LAB_Enhanced_Inference')
    if os.path.exists(out_path):
        try:
            shutil.rmtree(out_path)
        except PermissionError:
            print(f"\n[!] Windows locked {out_path}. Waiting 2 seconds...")
            time.sleep(2)
            shutil.rmtree(out_path, ignore_errors=True)
    os.makedirs(out_path, exist_ok=True)
    return out_path

def apply_lab_equalization(img):
    """
    Converts to LAB space, equalizes the Lightness channel, and returns to BGR.
    This boosts edge contrast without hallucinating bizarre colors.
    """
    # 1. Convert BGR to CIELAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # 2. Split the channels: L (Lightness), A (Color Red/Green), B (Color Blue/Yellow)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # 3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) strictly to the L channel
    # clipLimit prevents over-amplification of background noise (water particulate)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    
    # 4. Merge the enhanced L channel back with the untouched A and B channels
    merged_lab = cv2.merge((cl, a_channel, b_channel))
    
    # 5. Convert back to BGR for YOLO
    enhanced_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_bgr

def run_lab_inference():
    # --- CONFIGURATION ---
    weights_path = r"ultralytics\runs\pose\INSTANCE SEGMENTATION + POSE ESTIMATION\Runs\ExtremityPose\Dual_Strategy_Run_1\weights\best.pt"
    test_images_dir = r"INSTANCE SEGMENTATION + POSE ESTIMATION\DATASETS\UMT Dataset\Head+Tail UMT Annotations\images" 
    output_dir = r"INSTANCE SEGMENTATION + POSE ESTIMATION\INFERENCE AND RESULTS ANALYSIS\Results_Dual_Strategy_Run_1"
    
    print("\n" + "="*50)
    print(" INITIATING LAB-SPACE PREPROCESSED INFERENCE")
    print("="*50 + "\n")

    out_folder = build_output_dir(output_dir)
    model = YOLO(weights_path)

    image_files = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.png', '.jpeg', '.JPG'))]
    
    total_latency_ms = 0

    for img_name in image_files:
        img_path = os.path.join(test_images_dir, img_name)
        
        # Read the raw image
        raw_img = cv2.imread(img_path)
        if raw_img is None:
            continue
            
        start_time = time.time()

        # --- THE PREPROCESSING INJECTION ---
        enhanced_img = apply_lab_equalization(raw_img)
        
        # Feed the processed numpy array directly to YOLO
        results = model.predict(source=enhanced_img, conf=0.15, verbose=False, stream=False)
        
        end_time = time.time()
        total_latency_ms += (end_time - start_time) * 1000

        for res in results:
            # Draw the bounding boxes and keypoints onto the ENHANCED image
            annotated_enhanced_img = res.plot(labels=True, conf=True, line_width=1, kpt_radius=2, font_size=1)
            
            # Create a Side-by-Side comparative image for your thesis
            # hstack horizontally attaches the raw image to the annotated image
            comparison_img = np.hstack((raw_img, annotated_enhanced_img))
            
            # Save the result
            save_path = os.path.join(out_folder, f"LAB_COMP_{img_name}")
            cv2.imwrite(save_path, comparison_img)

    # Calculate Hardware Profiling
    avg_latency = total_latency_ms / len(image_files) if len(image_files) > 0 else 0
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0

    print(f"\n[+] Processed {len(image_files)} images through LAB + YOLO pipeline.")
    print(f"[+] Total Pipeline Latency (Preprocess + YOLO): {avg_latency:.2f} ms")
    print(f"[+] Estimated Real-Time FPS: {fps:.1f} FPS")
    print(f"[+] Side-by-side comparative images saved to: {out_folder}")

if __name__ == '__main__':
    run_lab_inference()