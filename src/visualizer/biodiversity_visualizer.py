import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

class BiodiversityVisualizer:
    """
    A centralized visualization engine for generating all pertinent visual information
    for the coral reef health assessment framework.
    """
    def __init__(self, output_dir="results/figures"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Set professional seaborn style for academic plots
        sns.set_theme(style="whitegrid", context="paper")

    # ==========================================
    # 1. ECOLOGICAL METRICS PLOTS
    # ==========================================
    
    def plot_size_spectra(self, df, save_name="size_spectra.png"):
        """
        Plots the Size Diversity Spectra (Histogram/KDE) of the community.
        """
        if df.empty or 'Length_3D_cm' not in df:
            return
            
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='Length_3D_cm', hue='Species', multiple="stack", 
                     bins=15, kde=True, palette="viridis")
        
        plt.title("Coral Reef Community Size Spectra", fontsize=16, fontweight='bold')
        plt.xlabel("3D Body Length (cm)", fontsize=12)
        plt.ylabel("Frequency (Count)", fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_name), dpi=300)
        plt.show()

    def plot_length_weight_curve(self, df, save_name="length_weight_curve.png"):
        """
        Plots the Length-Weight relationship scatter plot across species.
        """
        if df.empty or 'Length_3D_cm' not in df or 'Weight_g' not in df:
            return

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='Length_3D_cm', y='Weight_g', hue='Species', 
                        size='Weight_g', sizes=(50, 200), alpha=0.7, palette="deep")
        
        plt.title("Length-Weight Relationship Across Target Species", fontsize=16, fontweight='bold')
        plt.xlabel("Length (cm)", fontsize=12)
        plt.ylabel("Calculated Weight (g)", fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_name), dpi=300)
        plt.show()

    def plot_population_metrics(self, pop_metrics_dict, save_name="pop_metrics.png"):
        """
        Plots a comparative bar chart of Mean Length (Li) vs Max Length (Lmax) per species.
        Expects the dictionary output from ReefMetricsCalculator.population_metrics.
        """
        if not pop_metrics_dict: return
        
        species = list(pop_metrics_dict.keys())
        mean_L = [m["Mean_Length_Pop_Li"] for m in pop_metrics_dict.values()]
        max_L = [m["Max_Length_Pop_Lmax_i"] for m in pop_metrics_dict.values()]

        x = np.arange(len(species))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, mean_L, width, label='Mean Length (Li)', color='skyblue')
        ax.bar(x + width/2, max_L, width, label='Max Length (Lmax)', color='salmon')

        ax.set_ylabel('Length (cm)', fontsize=12)
        ax.set_title('Mean vs Maximum Length per Target Species', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(species, rotation=45, ha="right")
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_name), dpi=300)
        plt.show()

    # ==========================================
    # 2. COMPUTER VISION & PIPELINE PLOTS
    # ==========================================

    def plot_detection_comparison(self, image_path, yolo_boxes, dino_boxes):
        """
        Recreates the visual comparison between YOLO and DINO from the Jupyter notebook.
        """
        img_bgr = cv2.imread(image_path)
        if img_bgr is None: return
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle("Detection Model Comparison", fontsize=18, fontweight='bold')

        # YOLO Plot
        ax1.imshow(img_rgb)
        ax1.set_title(f"Custom YOLO ({len(yolo_boxes)} detections)", fontsize=14)
        ax1.axis('off')
        for box in yolo_boxes:
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, 
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax1.add_patch(rect)

        # DINO Plot
        ax2.imshow(img_rgb)
        ax2.set_title(f"Grounding DINO ({len(dino_boxes)} detections)", fontsize=14)
        ax2.axis('off')
        for box in dino_boxes:
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, 
                                     linewidth=2, edgecolor='blue', facecolor='none')
            ax2.add_patch(rect)

        plt.tight_layout()
        plt.show()

    def plot_stereo_validation(self, left_img_path, right_img_path, distance_cm=None):
        """
        Side-by-side stereo view validation from Step 2 Notebook.
        """
        left_img = cv2.cvtColor(cv2.imread(left_img_path), cv2.COLOR_BGR2RGB)
        right_img = cv2.cvtColor(cv2.imread(right_img_path), cv2.COLOR_BGR2RGB)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        ax1.imshow(left_img)
        ax1.set_title("Left View", fontsize=14)
        ax1.axis('off')

        ax2.imshow(right_img)
        ax2.set_title("Right View", fontsize=14)
        ax2.axis('off')

        title = "Stereo Pair Validation"
        if distance_cm: title += f" | Estimated Length: {distance_cm:.2f} cm"
        plt.suptitle(title, fontsize=18, fontweight='bold')
        
        plt.tight_layout()
        plt.show()