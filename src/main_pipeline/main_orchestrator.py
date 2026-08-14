import cv2
import glob
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import time

# Add the OUTER ultralytics folder to the path, so Python can find the INNER one
custom_yolo_path = r"c:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\ultralytics"
if custom_yolo_path not in sys.path:
    sys.path.insert(0, custom_yolo_path)

from ultralytics import YOLO

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

# Active Imports from the modular structure
from src.detection.object_detection_wrappers import YoloDetector, DinoDetector
from src.segmentation.sam_segmentation_wrapper import SamSegmenter
from src.centerline_extraction.morphological_centerline import extract_centerline
from src.classification.species_classification import SpeciesClassifier
from src.stereovision.stereo_triangulation import load_stereo_matrices, calculate_3d_centerline_length, calculate_3d_point_cloud
from src.stereovision.stereo_matching import match_boxes_epipolar, compute_fundamental_matrix
from src.segmentation.segmentation_engine import SegmentationEngine

# Active Imports from Ecology & Visualization Modules
from src.biodiversity_metrics.biodiversity_metrics import ReefMetricsCalculator
from src.visualizer.biodiversity_visualizer import BiodiversityVisualizer

class BiodiversityAssessorPipeline:
    def __init__(self, detector_type="yolo", segmentation_mode="sam2", calibration_npz_path="weights/stereovision/stereo_matrices.npz", viewer_mode="html", conf_threshold=0.4):
        """
        Initializes the pipeline. Models are loaded into RAM once to maximize speed.
        """
        print("Initializing Unified Biodiversity Framework...")
        self.viewer_mode = viewer_mode.lower() 
        
        # --- RESOLVE ABSOLUTE PATHS ---
        calib_path = str(PROJECT_ROOT / calibration_npz_path)
        dino_config = str(PROJECT_ROOT / "configs" / "networks" / "GroundingDINO_SwinT_OGC.py")
        dino_weights = str(PROJECT_ROOT / "weights" / "groundingdino_swint_ogc.pth")
        classifier_weights = str(PROJECT_ROOT / "weights" / "classification" / "dino_classifier.pkl")
        
        # A. LOAD DETECTION AI 
        if detector_type == "yolo":
            self.detector = YoloDetector(conf_threshold=conf_threshold)
        else:
            self.detector = DinoDetector(
                config_path=dino_config, 
                weights_path=dino_weights,
                box_threshold=conf_threshold
            )
        
        
        # B. LOAD SEGMENTATION & CLASSIFICATION AI
        self.segmenter = SegmentationEngine(mode=segmentation_mode)
        self.classifier = SpeciesClassifier(classifier_weights_path=classifier_weights)
         
        # D. LOAD STEREO MATRICES WITH AUTO-COMPUTATION
        print(f"Loading stereo calibration from {calib_path}...")
        archive = np.load(calib_path)
        
        # Extracting parameters to build the Fundamental Matrix (F)
        try:
            self.K1, self.K2 = archive['K1'], archive['K2']
            self.R, self.T = archive['R'], archive['T']
        except KeyError:
            print("CRITICAL ERROR: 'K1', 'K2', 'R', or 'T' missing from calibration file.")
            sys.exit(1)
            
        # Left Camera Projection Matrix: P1 = K1 * [I | 0]
        self.P1 = np.dot(self.K1, np.hstack((np.eye(3), np.zeros((3, 1)))))
        # Right Camera Projection Matrix: P2 = K2 * [R | T]
        self.P2 = np.dot(self.K2, np.hstack((self.R, self.T)))
        
        # Compute Fundamental Matrix for rigorous matching
        self.F = compute_fundamental_matrix(self.K1, self.K2, self.R, self.T)
        
        # E. LOAD ECOLOGY & VISUALIZATION ENGINES
        self.metrics_engine = ReefMetricsCalculator()
        self.visualizer = BiodiversityVisualizer()
        
        self.community_data = []
        self.timing_stats = {
            "detection": 0.0, "matching": 0.0, "segmentation": 0.0, 
            "classification": 0.0, "centerline": 0.0, "triangulation": 0.0, 
            "total_pipeline": 0.0, "processed_pairs": 0, "processed_fish": 0
        }

    def process_stereo_pair(self, left_img_path, right_img_path):
        """
        The core pipeline executed on a single frame pair from a community dataset.
        """
        list_2d_left, list_2d_right, list_3d_clouds, list_3d_lengths = [], [], [], []
        list_body_clouds = [] 
        
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
        
        t_pair_start = time.time()
        
        # --- A. DETECTION ---
        t_det = time.time()
        left_boxes = self.detector.predict(left_img_path)
        right_boxes = self.detector.predict(right_img_path)
        self.timing_stats["detection"] += (time.time() - t_det)
        
        # Start matching timer (wrap however you have your matching/embeddings currently configured)
        t_match = time.time()
        
        # --- A.2 EXTRACT VISUAL FEATURES (APPEARANCE MATCHING) ---
        print("[DEBUG] Extracting deep visual features for appearance-aware matching...")
        left_embeddings = [self.classifier.get_raw_embedding(left_img_path, box) for box in left_boxes]
        right_embeddings = [self.classifier.get_raw_embedding(right_img_path, box) for box in right_boxes]

        # --- CHANGED: Using Visual-Aware Rigorous Fundamental Matrix Matching ---
        matched_pairs = match_boxes_epipolar(left_boxes, right_boxes, left_embeddings, right_embeddings, self.F)
        
        self.timing_stats["matching"] += (time.time() - t_match)
        
        for l_box, r_box in matched_pairs:
            
            # --- B. SEGMENTATION & SPECIES RECOGNITION ---
            t_seg = time.time()
            l_mask = self.segmenter.generate_mask(left_img_path, l_box)
            r_mask = self.segmenter.generate_mask(right_img_path, r_box)
            self.timing_stats["segmentation"] += (time.time() - t_seg)
            
            t_class = time.time()
            species = self.classifier.predict(left_img_path, l_box)
            self.timing_stats["classification"] += (time.time() - t_class)
            
            area_pixels = np.sum(l_mask) 
            
            # --- C. 2D CENTERLINE EXTRACTION ---
            t_cent = time.time()
            # Extract discrete points along the morphological centerline
            l_centerline = extract_centerline(l_mask)
            r_centerline = extract_centerline(r_mask)
            self.timing_stats["centerline"] += (time.time() - t_cent)
            
            # --- D. STEREO 3D RECONSTRUCTION ---
            t_tri = time.time()
            try:
                print(f"\n[DEBUG] Triangulating match... Left pts: {len(l_centerline)}, Right pts: {len(r_centerline)}")
                length_3d_cm, points_3d = calculate_3d_centerline_length(l_centerline, r_centerline, self.P1, self.P2)
                
                print(f"[DEBUG] -> Triangulation Success! Length: {length_3d_cm:.2f} cm")
                print(f"[DEBUG] -> 3D Points shape: {points_3d.shape if points_3d is not None else 'None'}")
                if points_3d is not None and len(points_3d) > 0:
                    print(f"[DEBUG] -> Sample 3D Point (X,Y,Z): {points_3d[0]}")

                # Calculate Ecological Weight (W = aL^b)
                weight_g = self.metrics_engine.compute_weight(length_3d_cm, species)
                
                # --- Retrieve 3D point cloud for body volume ---
                body_point_cloud = None
                if hasattr(self, 'P1'): 
                    try:
                        from src.stereovision.stereo_triangulation import calculate_3d_point_cloud
                        body_point_cloud = calculate_3d_point_cloud(l_mask, r_mask, self.P1, self.P2)
                        print(f"[DEBUG] -> Body Cloud shape: {body_point_cloud.shape if body_point_cloud is not None else 'None'}")
                    except Exception as e:
                        print(f"[DEBUG] -> Body cloud calculation skipped/failed: {e}")

                list_2d_left.append(l_centerline)
                list_2d_right.append(r_centerline)
                list_3d_clouds.append(points_3d)            
                list_3d_lengths.append(length_3d_cm)
                list_body_clouds.append(body_point_cloud)
                self.timing_stats["triangulation"] += (time.time() - t_tri)
            except ValueError as e:
                print(f"[DEBUG] -> Triangulation ValueError: {e}")
                continue 
            
            if length_3d_cm <= 0: 
                print("[DEBUG] -> Length is <= 0, skipping.")
                continue
            
            
            # Record the evaluated specimen
            self.community_data.append({
                "Frame_ID": os.path.basename(left_img_path),
                "Species": species,
                "Length_3D_cm": length_3d_cm,
                "Weight_g": weight_g,
                "Area_px": area_pixels
            })
            
            self.timing_stats["processed_fish"] += 1
            
        # Record total time taken for this image pair
        self.timing_stats["total_pipeline"] += (time.time() - t_pair_start)
        self.timing_stats["processed_pairs"] += 1
            
        # --- CONDITIONAL VISUALIZER TRIGGER ---
        if list_3d_clouds:
            print(f"\n[DEBUG] Sending {len(list_3d_clouds)} fish to the Visualizer...")
            if self.viewer_mode == "html":
                # Ensure the save name is unique per image so they don't overwrite each other
                save_filename = f"validation_{os.path.basename(left_img_path)}.html"
                
                # --- FIX: Pass the real list instead of the [None] debug override ---
                self.visualizer.plot_interactive_3d_dashboard(
                    left_img_path, right_img_path, 
                    list_2d_left, list_2d_right, list_3d_clouds, 
                    centerlines_3d_lengths=list_3d_lengths,
                    mask_3d_clouds=list_body_clouds,
                    save_name=save_filename
                )
                
                # --- NEW: Save High-Resolution 2D validation images ---
                # Extract the base name (e.g., 'frame_248') without the .jpg extension
                base_name = os.path.splitext(os.path.basename(left_img_path))[0]
                self.visualizer.save_high_res_2d_overlays(
                    left_img_path, right_img_path,
                    list_2d_left, list_2d_right,
                    pair_prefix=base_name
                )
            
            elif self.viewer_mode == "open3d":
                self.visualizer.plot_open3d_native(list_3d_clouds)

    def generate_ecological_report(self):
        """
        --- E. METRICS COMPUTATION & VISUALIZATION ---
        Aggregates the stored community data, outputs text metrics, and draws graphs.
        """
        df = pd.DataFrame(self.community_data)
        if df.empty:
            print("Notice: No fish evaluated in this dataset.")
            return

        # --- ADDED: EXTREME OUTLIER FILTERING ---
        initial_count = len(df)
        df = df[df['Length_3D_cm'] <= 100.0]
        removed_count = initial_count - len(df)
        
        print("\n" + "="*40)
        print(" CORAL REEF HEALTH ASSESSMENT REPORT")
        print("="*40)
        
        if removed_count > 0:
            print(f"[!] FILTER APPLIED: Removed {removed_count} physical anomalies (> 100 cm).")

        print("\n--- COMMUNITY-LEVEL INDICATORS ---")
        means = self.metrics_engine.community_mean_metrics(df)
        print(f"Total Community Abundance:  {means['Total_Community_Abundance']} fish")
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
            print(f"  Total Observations: {m['Total_Observations']}")
            print(f"  MaxN (Max per frame):     {m['MaxN']}")
            print(f"  MeanCount (Avg per frame):{m['MeanCount']:.2f}")
            print(f"  Mean Length (Li):         {m['Mean_Length_Pop_Li']:.2f} cm")
            print(f"  Mean Weight (Wi):         {m['Mean_Weight_Pop_Wi']:.2f} g")
            print(f"  Max Length (Lmax,i):      {m['Max_Length_Pop_Lmax_i']:.2f} cm")
            print(f"  Depletion Tracker (L95%): {m['Robust_Max_Pop_L95']:.2f} cm")
            print(f"  Fulton's Condition (Ki):  {m['Mean_Fultons_K_Ki']:.2f}")

        # --- CSV EXPORT ---
        self.metrics_engine.export_to_csv(df, pop_metrics, means, lmax, spectra)

        # GENERATE VISUALS
        print("\nGenerating visual reports...")
        self.visualizer.plot_size_spectra(df)
        self.visualizer.plot_length_weight_curve(df)
        self.visualizer.plot_population_metrics(pop_metrics)
        # --- PRINT TIMING METRICS ---
        self.print_performance_report()

    def print_performance_report(self):
        pairs = self.timing_stats["processed_pairs"]
        fish = self.timing_stats["processed_fish"]
        if pairs == 0: return
        
        print("\n" + "="*40)
        print(" PIPELINE PERFORMANCE METRICS (AVERAGES)")
        print("="*40)
        print(f"Total Stereo Pairs Processed: {pairs}")
        print(f"Total Valid Fish Reconstructed: {fish}")
        print(f"Total Pipeline Execution Time: {self.timing_stats['total_pipeline']:.2f} seconds")
        print("\n--- Average Inference per Frame Pair ---")
        print(f"  Object Detection:     {self.timing_stats['detection']/pairs:.4f} sec/pair")
        print(f"  Epipolar Matching:    {self.timing_stats['matching']/pairs:.4f} sec/pair")
        
        if fish > 0:
            print("\n--- Average Inference per Fish ---")
            print(f"  Instance Segmentation: {self.timing_stats['segmentation']/fish:.4f} sec/fish")
            print(f"  Species Classificaton: {self.timing_stats['classification']/fish:.4f} sec/fish")
            print(f"  Centerline Extraction: {self.timing_stats['centerline']/fish:.4f} sec/fish")
            print(f"  3D Triangulation:      {self.timing_stats['triangulation']/fish:.4f} sec/fish")
            
            total_per_fish = (self.timing_stats['segmentation'] + self.timing_stats['classification'] + 
                              self.timing_stats['centerline'] + self.timing_stats['triangulation']) / fish
            print(f"  --------------------------------------")
            print(f"  Total secondary AI per fish: {total_per_fish:.4f} sec/fish")
        print("="*40 + "\n")


if __name__ == "__main__":
    # 1. Initialize the AI Framework
    pipeline = BiodiversityAssessorPipeline(detector_type="yolo", segmentation_mode="sam2",viewer_mode="html", conf_threshold=0.25)
    
    # 2. Run over synchronized frames
    left_frames = sorted(glob.glob(r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\data\processed\calibration-v2\sampled_and_synchronized\ground_truth_2_left_frames\GT-Meters-Left-Frame-70.jpg"))
    right_frames = sorted(glob.glob(r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\data\processed\calibration-v2\sampled_and_synchronized\ground_truth_2_right_frames\GT-Meters-Right-Frame-70.jpg"))
    
    # left_frames = sorted(glob.glob(r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\data\processed\calibration-v2\subdatasets_selected\ground_truth_selected_for_pipeline_assessment\left\*.jpg"))
    # right_frames = sorted(glob.glob(r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\data\processed\calibration-v2\subdatasets_selected\ground_truth_selected_for_pipeline_assessment\right\*.jpg"))

    print(f"\n--- DATASET CHECK ---")
    print(f"Found {len(left_frames)} images in Left folder.")
    print(f"Found {len(right_frames)} images in Right folder.")
    print(f"---------------------\n")
    
    if len(left_frames) == 0 or len(right_frames) == 0:
        print("Pipeline aborted. Please check your folder paths in the glob.glob() function!")
    else:
        for lf, rf in zip(left_frames, right_frames):
            pipeline.process_stereo_pair(lf, rf)
        
        # Generate the final ecological report after processing all frames
        pipeline.generate_ecological_report()