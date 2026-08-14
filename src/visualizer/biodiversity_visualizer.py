import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
# import open3d as o3d

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


    def plot_interactive_3d_dashboard(self, left_img_path, right_img_path, 
                                      centerlines_2d_left, centerlines_2d_right, 
                                      centerlines_3d_list, centerlines_3d_lengths=None,
                                      mask_3d_clouds=None,
                                      save_name="interactive_3d_validation.html"):
        """
        Creates an interactive HTML dashboard with floating length labels and automatic smart-zoom.
        """
        if not centerlines_3d_list:
            return

        fig = make_subplots(
            rows=1, cols=3,
            column_widths=[0.38, 0.24, 0.38],
            specs=[[{"type": "image"}, {"type": "scene"}, {"type": "image"}]],
            subplot_titles=("Left Camera (2D)", "Interactive 3D Point Cloud", "Right Camera (2D)")
        )

        img_l = Image.open(left_img_path)
        img_r = Image.open(right_img_path)
        fig.add_trace(go.Image(z=img_l), row=1, col=1)
        fig.add_trace(go.Image(z=img_r), row=1, col=3) 

        fish_colors = [
            '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
            '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4'
        ]

        for i, (pts_l, pts_r) in enumerate(zip(centerlines_2d_left, centerlines_2d_right)):
            color = fish_colors[i % len(fish_colors)]
            if len(pts_l) > 0:
                x_l, y_l = zip(*pts_l)
                fig.add_trace(go.Scatter(x=x_l, y=y_l, mode='lines+markers', marker=dict(size=4, color=color), line=dict(width=2, color=color), name=f'Fish {i+1} L'), row=1, col=1)
            if len(pts_r) > 0:
                x_r, y_r = zip(*pts_r)
                fig.add_trace(go.Scatter(x=x_r, y=y_r, mode='lines+markers', marker=dict(size=4, color=color), line=dict(width=2, color=color), name=f'Fish {i+1} R'), row=1, col=3)

        lengths = centerlines_3d_lengths if centerlines_3d_lengths is not None else [0] * len(centerlines_3d_list)
        clouds = mask_3d_clouds if mask_3d_clouds is not None else [None] * len(centerlines_3d_list)

        scene_annotations = []
        all_valid_3d_points = [] # Used to calculate the dynamic bounding box

        for i, (pts_3d, length, body_cloud) in enumerate(zip(centerlines_3d_list, lengths, clouds)):
            if pts_3d is None or len(pts_3d) == 0: continue
            
            pts_3d = np.atleast_2d(pts_3d)
            valid_mask = ~np.isnan(pts_3d).any(axis=1) & ~np.isinf(pts_3d).any(axis=1)
            pts_3d = pts_3d[valid_mask]
            
            if pts_3d.shape[0] < 2: continue 
            
            X, Y, Z = pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2]
            color = fish_colors[i % len(fish_colors)]
            all_valid_3d_points.append(pts_3d)
            
            # Draw the translucent colored volume cloud
            if body_cloud is not None and len(body_cloud) > 0:
                body_cloud = np.atleast_2d(body_cloud)
                valid_cloud_mask = ~np.isnan(body_cloud).any(axis=1) & ~np.isinf(body_cloud).any(axis=1)
                clean_cloud = body_cloud[valid_cloud_mask]

                if clean_cloud.shape[0] > 0:
                    all_valid_3d_points.append(clean_cloud)
                    fig.add_trace(
                        go.Scatter3d(
                            x=clean_cloud[:, 0], y=clean_cloud[:, 1], z=clean_cloud[:, 2],
                            mode='markers',
                            marker=dict(size=4, color=color, opacity=0.4),
                            name=f'Fish {i+1} Volume',
                            showlegend=False
                        ), row=1, col=2
                    )
            
            # Draw the strict Black Centerline inside the cloud
            fig.add_trace(
                go.Scatter3d(
                    x=X, y=Y, z=Z,
                    mode='lines+markers',
                    marker=dict(size=3, color='black', opacity=0.9),
                    line=dict(width=2, color='black'),
                    name=f'Fish {i+1} 3D'
                ), row=1, col=2
            )
            
            # Floating Text Label annotations
            if length > 0:
                scene_annotations.append(dict(
                    x=np.mean(X),
                    y=np.mean(Y),
                    z=np.min(Z),
                    text=f"<b>{length:.2f} cm</b>",
                    showarrow=True,
                    arrowhead=1,
                    arrowcolor="black",
                    arrowsize=1,
                    arrowwidth=2,
                    ax=0,   
                    ay=30,  
                    font=dict(color="black", size=15),
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=4
                ))

        if all_valid_3d_points:
            merged_pts = np.vstack(all_valid_3d_points)
            min_bounds = np.min(merged_pts, axis=0)
            max_bounds = np.max(merged_pts, axis=0)
            
            # Center of the cluster
            center = (min_bounds + max_bounds) / 2.0
            
            # Distance to the furthest fish in any axis
            max_spread = np.max(max_bounds - center)
            
            # Multiply by 1.5 to create a generous cubic bounding box margin
            box_radius = max_spread * 1.5
            
            range_x = [center[0] - box_radius, center[0] + box_radius]
            range_y = [center[1] - box_radius, center[1] + box_radius]
            range_z = [center[2] - box_radius, center[2] + box_radius]
        else:
            range_x, range_y, range_z = None, None, None

        custom_camera = dict(
            up=dict(x=0, y=0, z=1),       
            center=dict(x=0, y=0, z=0),   
            eye=dict(x=-1.8, y=0.0, z=0.4) # Slightly pulled back to respect the new box
        )

        fig.update_layout(
            height=800, width=1800,
            title_text="Stereovision Validation Dashboard",
            scene=dict(
                xaxis=dict(title='X (cm)', range=range_x), 
                yaxis=dict(title='Y (cm)', range=range_y), 
                zaxis=dict(title='Depth (cm)', range=range_z), 
                # Forcing a 'cube' combined with perfectly equal numeric ranges guarantees 
                # a 1:1:1 undistorted biological scale while forcing a zoomed-out grid!
                aspectmode='cube', 
                annotations=scene_annotations 
            ),
            scene_camera=custom_camera
        )

        output_path = os.path.join(self.output_dir, save_name)
        fig.write_html(output_path)
        print(f"HTML Dashboard saved to: {output_path}")

    def save_high_res_2d_overlays(self, left_img_path, right_img_path, centerlines_2d_left, centerlines_2d_right, pair_prefix="pair"):
        """
        Saves full-resolution copies of the left and right frames with the 2D centerlines 
        drawn on them using OpenCV (thickness=1) for precise morphological validation.
        """
        import cv2
        import numpy as np

        # Load the original, uncompressed raw frames
        img_l = cv2.imread(left_img_path)
        img_r = cv2.imread(right_img_path)

        if img_l is None or img_r is None:
            print("[VIS-DEBUG] Error loading images for high-res 2D overlays.")
            return

        # Use the exact same hex colors as the dashboard, converted to BGR for OpenCV
        fish_colors = [
            '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
            '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4'
        ]

        def hex_to_bgr(hex_str):
            hex_str = hex_str.lstrip('#')
            r, g, b = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
            return (b, g, r)  # OpenCV uses BGR natively

        # Draw Left Centerlines
        for i, pts in enumerate(centerlines_2d_left):
            if len(pts) > 1:
                color = hex_to_bgr(fish_colors[i % len(fish_colors)])
                # Reshape for cv2.polylines requirements
                pts_arr = np.array(pts, np.int32).reshape((-1, 1, 2))
                cv2.polylines(img_l, [pts_arr], isClosed=False, color=color, thickness=3)

        # Draw Right Centerlines
        for i, pts in enumerate(centerlines_2d_right):
            if len(pts) > 1:
                color = hex_to_bgr(fish_colors[i % len(fish_colors)])
                pts_arr = np.array(pts, np.int32).reshape((-1, 1, 2))
                cv2.polylines(img_r, [pts_arr], isClosed=False, color=color, thickness=3)

        # Save to the results directory
        out_l_path = os.path.join(self.output_dir, f"{pair_prefix}_left_centerlines.jpg")
        out_r_path = os.path.join(self.output_dir, f"{pair_prefix}_right_centerlines.jpg")

        cv2.imwrite(out_l_path, img_l)
        cv2.imwrite(out_r_path, img_r)
        
        print(f"High-Res 2D Overlays saved to: {self.output_dir}")


    def plot_open3d_native(self, centerlines_3d_list):
        """
        Opens a high-performance native desktop window to interact with the 3D data.
        """
        if not centerlines_3d_list:
            return
            
        geometries = []
        
        # Create a coordinate frame for reference (X=Red, Y=Green, Z=Blue)
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])
        geometries.append(coord_frame)

        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]] # Distinct colors for fish
        
        for idx, pts_3d in enumerate(centerlines_3d_list):
            if pts_3d.size < 2: continue
            
            # 1. Create the points (Thin line equivalent)
            points = o3d.utility.Vector3dVector(pts_3d)
            
            # 2. Connect the points sequentially to form the centerline
            lines = [[i, i + 1] for i in range(len(pts_3d) - 1)]
            line_set = o3d.geometry.LineSet(
                points=points,
                lines=o3d.utility.Vector2iVector(lines)
            )
            
            # Apply color
            color = colors[idx % len(colors)]
            line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
            geometries.append(line_set)
            
            # 3. Highlight the Head and Tail with spheres
            head_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
            head_sphere.translate(pts_3d[0])
            head_sphere.paint_uniform_color([0.2, 0.8, 0.2]) # Green Head
            
            tail_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
            tail_sphere.translate(pts_3d[-1])
            tail_sphere.paint_uniform_color([0.8, 0.2, 0.2]) # Red Tail
            
            geometries.extend([head_sphere, tail_sphere])

        print("\nOpening native 3D viewer... (Close the window to continue pipeline)")
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Stereo 3D Centerline Validation (Open3D)",
            width=1024, height=768,
            point_show_normal=False,
            mesh_show_wireframe=False,
            mesh_show_back_face=True
        )

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