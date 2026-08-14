"""
================================================================================
Script: YOLO Training Metrics Comparison Plotter (Ablation Studies)
================================================================================

Description:
    A utility script to visually compare the training progression of multiple 
    YOLO models for ablation studies. The script reads the standard `results.csv` files generated 
    by Ultralytics training runs, extracts specific training or validation 
    metrics (e.g., pose loss, bounding box loss), and plots them onto a single, 
    unified graph to evaluate architectural or dataset improvements.

    Key Functions:
    1. Automated Data Ingestion: Iterates through a provided dictionary of model 
       run paths, locating and reading their respective `results.csv` files.
    2. Data Sanitization: Automatically strips leading and trailing whitespace 
       from YOLO's default CSV column headers to prevent key errors during 
       data extraction.
    3. Multi-Run Plotting: Extracts the specified metric column and plots it 
       against the training epoch count, applying custom labels, gridlines, 
       and legends for comparative analysis.
    4. High-Resolution Export: Renders the final matplotlib graph and saves it 
       to the local directory as a 300 DPI `.png` file suitable for academic 
       publication or reporting.

Author:
    Louis Jardinet
    Universiti Teknologi Petronas

Date:
    January 2026 - June 2026

Dependencies:
    - Python standard libraries: os
    - External packages: pandas (pd), matplotlib.pyplot (plt)

Inputs:
    - run_folders: A dictionary mapping custom graph labels to the local 
      directory paths of the YOLO runs.
    - metric: The specific CSV column name to plot (e.g., 'train/pose_loss').

Outputs:
    - A saved high-resolution image file (`model_comparison_graph.png`) 
      visualizing the comparative metric curves.
================================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_comparison(run_folders, metric='train/pose_loss'):
    plt.figure(figsize=(10, 6))
    
    for label, folder in run_folders.items():
        csv_path = os.path.join(folder, "results.csv")
        
        if not os.path.exists(csv_path):
            print(f"Could not find {csv_path}. Skipping...")
            continue
            
        # Read the CSV
        df = pd.read_csv(csv_path)
        
        # YOLO saves column names with leading spaces, so we strip them
        df.columns = df.columns.str.strip()
        
        # Plot Epoch vs the chosen metric
        plt.plot(df['epoch'], df[metric], label=label, linewidth=2)

    plt.title(f"Model Comparison: {metric}", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss (Lower is better)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save the graph
    save_path = "model_comparison_graph.png"
    plt.savefig(save_path, dpi=300)
    print(f"Graph saved as {save_path}")
    plt.show()

# --- USAGE ---
# Provide a dictionary of the runs you want to compare
# Format: {"Name you want on the graph": "Path to the run folder"}
runs_to_compare = {
    "Baseline (No EAMRF)": r"C:\Users\Work Mode Big Dog\...\runs\pose\baseline_run",
    "Bézierfusion (With EAMRF)": r"C:\Users\Work Mode Big Dog\...\runs\pose\eamrf_run"
}

# You can plot 'train/box_loss', 'train/pose_loss', or 'train/cls_loss'
plot_comparison(runs_to_compare, metric='train/pose_loss')