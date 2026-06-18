import cv2
import glob
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# --- FAILSAFE PATH OVERRIDE ---
# Find the exact directory containing the 'src' folder
try:
    current_dir = Path(__file__).resolve().parent
except NameError:
    current_dir = Path.cwd() # Fallback for interactive/Jupyter environments

PROJECT_ROOT = current_dir
for _ in range(5): # Search upwards up to 5 levels
    if (PROJECT_ROOT / "src").is_dir():
        break
    PROJECT_ROOT = PROJECT_ROOT.parent

# Force Python to check the PROJECT_ROOT folder first for all imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Active Imports from our modular structure
from src.detection.object_detection_wrappers import YoloDetector, DinoDetector
from src.segmentation.sam_segmentation_wrapper import SamSegmenter
from src.centerline_extraction.morphological_centerline import extract_centerline
from src.classification.species_classification import SpeciesClassifier
from src.stereovision.stereo_triangulation import load_stereo_matrices, calculate_3d_centerline_length

# Active Imports from Ecology & Visualization Modules
from src.biodiversity_metrics.biodiversity_metrics import ReefMetricsCalculator
from src.visualizer.biodiversity_visualizer import BiodiversityVisualizer

class BiodiversityAssessorPipeline:
    def __init__(self, detector_type="yolo", calibration_npz_path="weights/stereovision/stereo_matrices.npz"):
        """
        Initializes the pipeline. Models are loaded into RAM once to maximize speed.
        """
        print("Initializing Unified Biodiversity Framework...")
        
        # --- RESOLVE ABSOLUTE PATHS ---
        # This guarantees weights are found regardless of where the script is run
        calib_path = str(PROJECT_ROOT / calibration_npz_path)
        dino_config = str(PROJECT_ROOT / "configs" / "networks" / "GroundingDINO_SwinT_OGC.py")
        dino_weights = str(PROJECT_ROOT / "weights" / "groundingdino_swint_ogc.pth")
        classifier_weights = str(PROJECT_ROOT / "weights" / "classification" / "dino_classifier.pkl")
        
        # A. LOAD DETECTION AI
        self.detector = YoloDetector() if detector_type == "yolo" else DinoDetector(
            config_path=dino_config, 
            weights_path=dino_weights
        )
        
        # B. LOAD SEGMENTATION & CLASSIFICATION AI
        self.segmenter = SamSegmenter()
        self.classifier = SpeciesClassifier(classifier_weights_path=classifier_weights)
        
        # D. LOAD STEREO MATRICES WITH AUTO-COMPUTATION
        print(f"Loading stereo calibration from {calib_path}...")
        archive = np.load(calib_path)
        if 'P1' in archive.files and 'P2' in archive.files:
            self.P1, self.P2 = archive['P1'], archive['P2']
        elif 'K1' in archive.files and 'R' in archive.files:
            print("Auto-computing Projection Matrices (P1, P2) from Camera parameters (K1, K2, R, T)...")
            K1, K2 = archive['K1'], archive['K2']
            R, T = archive['R'], archive['T']
            
            # Left Camera Projection Matrix: P1 = K1 * [I | 0]
            self.P1 = np.dot(K1, np.hstack((np.eye(3), np.zeros((3, 1)))))
            # Right Camera Projection Matrix: P2 = K2 * [R | T]
            self.P2 = np.dot(K2, np.hstack((R, T)))
        else:
            print(f"\nCRITICAL ERROR: Calibration file is missing necessary matrices.")
            print(f"Keys found in your file: {archive.files}")
            sys.exit(1)
        
        # E. LOAD ECOLOGY & VISUALIZATION ENGINES
        self.metrics_engine = ReefMetricsCalculator()
        self.visualizer = BiodiversityVisualizer()
        
        self.community_data = []

    def process_stereo_pair(self, left_img_path, right_img_path):
        """
        The core pipeline executed on a single frame pair from a community dataset.
        """
        # --- FAILSAFE: Check if paths are actually valid image files, not folders ---
        if not os.path.isfile(left_img_path) or not os.path.isfile(right_img_path):
            print(f"Skipping invalid path (likely a folder instead of an image):")
            print(f"  L: {left_img_path}\n  R: {right_img_path}")
            return
            
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        if not (left_img_path.lower().endswith(valid_exts) and right_img_path.lower().endswith(valid_exts)):
            print(f"Skipping non-image file format:\n  L: {left_img_path}\n  R: {right_img_path}")
            return

        print(f"Processing stereo pair: {os.path.basename(left_img_path)} | {os.path.basename(right_img_path)}")
        
        # --- A. DETECTION ---
        # Detect fish bounding boxes in BOTH images (Robust matching approach)
        left_boxes = self.detector.predict(left_img_path)
        right_boxes = self.detector.predict(right_img_path)
        
        # Note: A helper function 'match_boxes_epipolar' will be needed here to pair the 
        # left_boxes and right_boxes together based on their Y-coordinates and epipolar lines.
        # For the sake of pipeline flow, we assume a matched list `matched_pairs`:
        matched_pairs = zip(left_boxes, right_boxes) # Placeholder logic
        
        for l_box, r_box in matched_pairs:
            
            # --- B. SEGMENTATION & SPECIES RECOGNITION ---
            # Extract high-precision SAM masks
            l_mask = self.segmenter.generate_mask(left_img_path, l_box)
            r_mask = self.segmenter.generate_mask(right_img_path, r_box)
            
            # Classify species
            species = self.classifier.predict(left_img_path, l_box)
            
            # Optional: Get Area from mask
            area_pixels = np.sum(l_mask) 
            
            # --- C. 2D CENTERLINE EXTRACTION ---
            # Extract discrete points along the morphological centerline
            l_centerline = extract_centerline(l_mask)
            r_centerline = extract_centerline(r_mask)
            
            # --- D. STEREO 3D RECONSTRUCTION ---
            # Triangulate matching points and sum the Euclidean 3D distances
            try:
                length_3d_cm = calculate_3d_centerline_length(l_centerline, r_centerline, self.P1, self.P2)
            except ValueError:
                continue # Skip if centerline points don't perfectly match in number
            
            if length_3d_cm <= 0: continue
            
            # Calculate Ecological Weight (W = aL^b)
            weight_g = self.metrics_engine.compute_weight(length_3d_cm, species)
            
            # Record the evaluated specimen
            self.community_data.append({
                "Species": species,
                "Length_3D_cm": length_3d_cm,
                "Weight_g": weight_g,
                "Area_px": area_pixels
            })

    def generate_ecological_report(self):
        """
        --- E. METRICS COMPUTATION & VISUALIZATION ---
        Aggregates the stored community data, outputs text metrics, and draws graphs.
        """
        df = pd.DataFrame(self.community_data)
        if df.empty:
            print("Notice: No fish evaluated in this dataset.")
            return

        print("\n" + "="*40)
        print(" CORAL REEF HEALTH ASSESSMENT REPORT")
        print("="*40)

        print("\n--- COMMUNITY-LEVEL INDICATORS ---")
        means = self.metrics_engine.community_mean_metrics(df)
        print(f"Mean Community Length (L):  {means['Mean_Community_Length_cm']:.2f} cm")
        print(f"Mean Community Weight (W):  {means['Mean_Community_Weight_g']:.2f} g")
        
        lmax = self.metrics_engine.community_max_length(df)
        print(f"Mean Maximum Length (Lmax): {lmax['Community_Mean_Lmax_cm']:.2f} cm")
        
        spectra = self.metrics_engine.size_spectra(df)
        if spectra.get("Spectra_Slope") is not None:
            print(f"Size Spectra Slope:         {spectra['Spectra_Slope']:.4f}")
            print(f"Size Spectra Intercept:     {spectra['Spectra_Intercept']:.4f}")
            print(f"Size Diversity Spectra (DS):{spectra['Size_Diversity_Spectra_DS']:.4f}")

        print("\n--- POPULATION-LEVEL INDICATORS ---")
        pop_metrics = self.metrics_engine.population_metrics(df)
        for species, m in pop_metrics.items():
            print(f"\nSpecies: {species}")
            print(f"  Mean Length (Li):   {m['Mean_Length_Pop_Li']:.2f} cm")
            print(f"  Mean Weight (Wi):   {m['Mean_Weight_Pop_Wi']:.2f} g")
            print(f"  Max Length (Lmax,i):{m['Max_Length_Pop_Lmax_i']:.2f} cm")
            print(f"  Depletion Tracker (L95%): {m['Robust_Max_Pop_L95']:.2f} cm")
            print(f"  Fulton's Condition (Ki):  {m['Mean_Fultons_K_Ki']:.2f}")

        # GENERATE VISUALS
        print("\nGenerating visual reports...")
        self.visualizer.plot_size_spectra(df)
        self.visualizer.plot_length_weight_curve(df)
        self.visualizer.plot_population_metrics(pop_metrics)



if __name__ == "__main__":
    # 1. Initialize the AI Framework
    pipeline = BiodiversityAssessorPipeline(detector_type="yolo")
    
    # 2. Run over all synchronized frames in a video or dataset folder
    left_frames = sorted(glob.glob(r"data\raw\UMT\Images_1-190714\St3-Pantai-Vietnam\TG4-Black(L)-Images-st3-(P.V)\*.jpg"))
    right_frames = sorted(glob.glob(r"data\raw\UMT\Images_1-190714\St3-Pantai-Vietnam\TG4-Red(R)-Images-st3-(P.V)\*.jpg"))
    
    print(f"\n--- DATASET CHECK ---")
    print(f"Found {len(left_frames)} images in Left folder.")
    print(f"Found {len(right_frames)} images in Right folder.")
    print(f"---------------------\n")
    
    if len(left_frames) == 0 or len(right_frames) == 0:
        print("Pipeline aborted. Please check your folder paths in the glob.glob() function!")
    else:
        for lf, rf in zip(left_frames, right_frames):
            pipeline.process_stereo_pair(lf, rf)
            
        # 3. Generate the final Ecological Analysis and Graphs
        pipeline.generate_ecological_report()