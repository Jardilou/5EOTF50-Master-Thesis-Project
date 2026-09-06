# Stereovision Coral Reef Biodiversity Assessor

An advanced, end-to-end computer vision framework for automated non-extractive ecological assessment of coral reefs. This pipeline ingests unrectified stereo camera footage and uses deep learning to perform species classification, pixel-perfect instance segmentation, 3D morphological reconstruction, and population-level metric extraction.

Key Features

Multi-Modal Detection & Segmentation: Supports both zero-shot foundation models (Grounding DINO + SAM 2) and custom-trained highly-optimized models (YOLOv8/11 Instance Segmentation).

Deep Appearance-Aware Stereo Matching: Solves the complex stereo correspondence problem (epipolar collisions) by combining rigorous Epipolar Geometric math with DINOv2 raw visual embeddings and the Hungarian algorithm.

PCA-Based Anatomical Morphometry: Extracts true non-linear fish spines using Principal Component Analysis (PCA) on binary masks and B-Spline strictly interpolated parametric curves, preventing length underestimation due to body curvature.

Ecological Metrics Engine: Automatically computes Total Abundance, MaxN, MeanCount, True 3D Length ($L$), Weight ($W = aL^b$ using FishBase constants), Fulton's Condition Index ($K$), and Size Spectra.

Interactive 3D Validation: Dynamically generates WebGL-based Plotly HTML dashboards projecting translucent 3D point clouds and centerlines with floating length annotations, alongside high-res 2D diagnostic overlays.

# Project Structure

PROJECT_ROOT/
├── data/
│   ├── raw/                 # Raw Left/Right stereo frames
│   └── processed/           # Ground truth and processed datasets
├── weights/
│   ├── stereovision/        # stereo_matrices.npz (Calibration matrices)
│   ├── classification/      # dino_classifier.pkl
│   └── ...                  # YOLO/SAM2/DINO model weights
├── outputs/
│   ├── metrics/             # Exported ecological CSV reports
│   └── metrics_corrected/   # Post-processed updated metrics
├── src/
│   ├── main_pipeline/       # main_orchestrator.py
│   ├── detection/           # YOLO & Grounding DINO wrappers
│   ├── segmentation/        # Unified SAM2/YOLO segmentation engine
│   ├── classification/      # DINOv2 feature extraction & Logistic Regression
│   ├── stereovision/        # Matching, Epipolar geometry, 3D Triangulation
│   ├── morphology/          # PCA centerline and B-spline math
│   ├── biodiversity_metrics/# Ecological calculators (MaxN, Fulton's K)
│   └── visualizer/          # Plotly 3D HTML and OpenCV overlay generators
└── scripts/
    └── utils/               # Calibration checkers, CSV recalculators


# Installation & Setup

Clone the repository and set up a virtual environment:

git clone https://github.comJardilou/5EOTF50-Master-Thesis-Project.git
cd stereovision-reef-assessor
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate



(Ensure you have PyTorch installed with CUDA support for GPU acceleration).

Install Foundation Models:

Install SAM 2 from Meta's repository:

pip install git+https://github.com/facebookresearch/sam2.git


Ensure ultralytics is installed for YOLO support.

Prepare Weights & Calibration:

Place your trained YOLO .pt files, DINOv2 classifier .pkl, and SAM2 .pt weights in the weights/ directory.

Ensure your stereo calibration file (stereo_matrices.npz) containing K1, K2, R, T, P1, and P2 is located in weights/stereovision/.

# Usage

1. Running the Main Orchestrator

The main entry point for the pipeline. It processes pairs of left/right images, performs 3D reconstruction, generates the HTML dashboards, and outputs CSV reports.

python src/main_pipeline/main_orchestrator.py


Note: You can configure the conf_threshold and toggle between YOLO or DINO detection directly within the __init__ call inside this script.

2. Recalculating Ecological Metrics

If you update your species-specific FishBase length-weight constants ($a$ and $b$), you do not need to re-run the heavy AI inference. Simply run the utility script to update your CSVs:

python scripts/utils/recalculate_csv_weights.py


3. Validating Camera Calibration

To check if your underwater checkerboard images are readable before running full calibration:

python scripts/utils/visualize_calibration_board.py


# Output Examples

Running the pipeline will populate your outputs/ folder with:

validation_[frame_id].html: Fully interactive 3D scatter plots showing the triangulated fish volumes and measured spines.

[frame_id]_left_centerlines.jpg: High-resolution 2D visual proof of the PCA centerlines.

metrics/raw_fish_data.csv: Database of every individual fish processed.

metrics/community_metrics.csv: Top-level summary (Abundance, Mean L/W, Size Spectra).

metrics/population_metrics.csv: Species-specific breakdowns (MaxN, MeanCount, L95%, Fulton's K).

# Core Methodology

Deep Appearance-Aware Matching

Traditional stereo vision fails when multiple fish cross the same epipolar line. This framework fixes this by computing the Fundamental Matrix ($F$) to project theoretical epipolar lines, calculating the perpendicular geometric distance, and summing it with a heavy cost penalty derived from the Cosine Similarity of the fish's raw DINOv2 neural embeddings. This is solved optimally via the Hungarian Algorithm.

PCA Morphometrics

Instead of drawing a straight line through a curved fish, the extract_centerline module computes the Covariance Matrix and Eigenvectors of the segmented binary mask. Projecting the mask onto its primary Principal Component guarantees the extraction of the true biological extremes (Head and Tail), which are then connected via a smooth parametric B-Spline interpolation ($s=0, k=2$) passing perfectly through the center of mass.
