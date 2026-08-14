import os
import pandas as pd
import numpy as np
import math

def bind_all_csv_to_yolo(csv_path, yolo_labels_dir, output_labels_dir, threshold=0.02):
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # Map the messy CSV names ONLY for the 5 target classes you care about
    species_mapping = {
        "PomcentDascyllus reticulatus": "Dascyllus_reticulatus",
        "LabHemigymnus melapterus": "Hemigymnus_melapterus",
        "PomcentPomacentrus moluccensis": "Pomacentrus_moluccensis",
        "PomcentAbudefduf vaigiensis": "Abudefduf_vaigiensis",
        "ScarScarus quoyi": "Scaridae",
        "ScarScarus ghobban": "Scaridae",
        "ScarScarus psittacus": "Scaridae",
        "ScarScarus rivulatus": "Scaridae",
        "ScarScarus rubroviolaceus": "Scaridae"
    }
    
    print("Loading metadata...")
    df = pd.read_csv(csv_path)
    # Drop any rows that randomly lack coordinates to prevent crashes
    df = df.dropna(subset=['BBox_CX', 'BBox_CY'])
    
    # Group by image name for fast lookup
    grouped_csv = df.groupby('Image_Name')
    
    total_bound = 0
    total_missed = 0
    
    print("Binding YOLO labels to Species...")
    for filename in os.listdir(yolo_labels_dir):
        if not filename.endswith('.txt'): continue
            
        image_name = filename.replace('.txt', '.jpg')
        
        yolo_path = os.path.join(yolo_labels_dir, filename)
        out_path = os.path.join(output_labels_dir, filename)
        
        # If the image isn't in the CSV at all, skip it
        if image_name not in grouped_csv.groups:
            continue
            
        csv_entries = grouped_csv.get_group(image_name)
        
        with open(yolo_path, 'r') as f:
            yolo_lines = f.readlines()
            
        new_lines = []
        
        for line in yolo_lines:
            parts = line.strip().split()
            if len(parts) < 5: continue
            
            y_cx, y_cy, w, h = map(float, parts[1:5])
            
            best_dist = float('inf')
            best_species = "unknown"
            
            for _, row in csv_entries.iterrows():
                c_cx, c_cy = float(row['BBox_CX']), float(row['BBox_CY'])
                # Euclidean distance
                dist = math.sqrt((y_cx - c_cx)**2 + (y_cy - c_cy)**2)
                
                if dist < best_dist:
                    best_dist = dist
                    best_species = str(row['Species'])
            
            # If the closest point is within our strict threshold
            if best_dist <= threshold:
                # If it's a target, use the clean map. Otherwise, replace spaces with underscores.
                clean_name = species_mapping.get(best_species, best_species.replace(" ", "_"))
                
                new_line = f"{clean_name} {y_cx:.6f} {y_cy:.6f} {w} {h}\n"
                new_lines.append(new_line)
                total_bound += 1
            else:
                total_missed += 1

        if new_lines:
            with open(out_path, 'w') as f:
                f.writelines(new_lines)

    print(f"Binding Complete! Bound {total_bound} instances. Missed {total_missed} (exceeded threshold).")


# Execute
bind_all_csv_to_yolo('species_metadata.csv', r'DATASETS\DATASETS FOR FISH DETECTION ONLY FOR IS\labels\UMT\Train+Val', r'DATASETS\DATASETS FOR SPECIES CLASSIFICATION\UMT For Validation\updated-labels-with-species', threshold=0.1)