import os
import shutil
import re
from pathlib import Path

# --- CONFIGURATION ---
# Define the source (your messy folder) and the destination (the clean folder)
SOURCE_DIR = Path(r"INSTANCE SEGMENTATION\ALGORITHM BENCHMARK")
DEST_DIR = Path("CLEAN_CODE")

# Define the exact professional architecture we want to build
TARGET_FOLDERS = [
    "configs/networks",
    "configs/paths",
    "data/raw",
    "data/processed",
    "data/annotations",
    "notebooks",
    "results/figures",
    "results/metrics",
    "scripts/data_prep",
    "scripts/utils",
    "src/instance_segmentation",
    "src/pose_estimation",
    "src/species_recognition",
    "src/stereo_vision",
    "weights"
]

def to_snake_case(filename):
    """
    Converts 'Custom-IS-Train.py' or 'Triple Strategy Fish Pose.yaml' 
    into 'custom_is_train.py' and 'triple_strategy_fish_pose.yaml'
    """
    name, ext = os.path.splitext(filename)
    
    # Convert to lowercase
    name = name.lower()
    # Replace spaces, hyphens, and plus signs with underscores
    name = re.sub(r'[\s\-+]+', '_', name)
    # Remove any weird special characters (keep letters, numbers, and underscores)
    name = re.sub(r'[^a-z0-9_]', '', name)
    
    # Clean up double underscores if any were created
    name = re.sub(r'_+', '_', name).strip('_')
    
    return name + ext

def determine_routing(filepath):
    """
    Acts as the 'brain' of the script. Looks at the file extension and 
    the folder it came from to decide exactly where it belongs in CLEAN_CODE.
    """
    ext = filepath.suffix.lower()
    path_str = str(filepath).lower()
    
    # 1. Weights -> weights/
    if ext in ['.pt', '.pth', '.npz', '.pth']:
        return "weights"
    
    # 2. Notebooks -> notebooks/
    elif ext == '.ipynb':
        return "notebooks"
        
    # 3. Configs -> configs/
    elif ext in ['.yaml', '.yml']:
        if 'paths' in path_str or 'dataset' in path_str:
            return "configs/paths"
        return "configs/networks"
        
    # 4. XML / JSON Annotations -> data/annotations/
    elif ext in ['.xml', '.json']:
        if 'keypoints' in path_str or 'calibration' in path_str:
            return "configs" # Metadata often belongs near configs
        return "data/annotations"
        
    # 5. Python Scripts -> src/ or scripts/
    elif ext == '.py':
        # Routing based on keywords in the old path or file name
        if 'manipulation' in path_str or 'prep' in path_str or 'split' in path_str:
            return "scripts/data_prep"
        elif 'stereo' in path_str or '3d' in path_str or 'calibration' in path_str:
            return "src/stereo_vision"
        elif 'pose' in path_str or 'pe' in path_str or 'bezier' in path_str:
            return "src/pose_estimation"
        elif 'species' in path_str or 'recognition' in path_str:
            return "src/species_recognition"
        elif 'instance' in path_str or 'is' in path_str or 'segmentation' in path_str:
            return "src/instance_segmentation"
        else:
            return "scripts/utils" # Fallback for unknown scripts
            
    # Ignore raw datasets and images (User must move these manually to preserve structure)
    return None

def main():
    print(f"--- Starting Safe Project Migration ---")
    print(f"Source: {SOURCE_DIR.absolute()}")
    print(f"Destination: {DEST_DIR.absolute()}\n")

    # Step 1: Create the new, clean directory architecture
    if not DEST_DIR.exists():
        DEST_DIR.mkdir()
    
    for folder in TARGET_FOLDERS:
        (DEST_DIR / folder).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {folder}")

    # Step 2: Traverse the messy directory
    copied_files = 0
    skipped_files = 0

    for root, _, files in os.walk(SOURCE_DIR):
        for file in files:
            filepath = Path(root) / file
            
            # Skip git, cache, and hidden files
            if '.git' in str(filepath) or '__pycache__' in str(filepath) or file.startswith('.'):
                continue
                
            # Determine where the file should go
            target_folder = determine_routing(filepath)
            
            if target_folder:
                clean_name = to_snake_case(file)
                dest_path = DEST_DIR / target_folder / clean_name
                
                # Handle duplicate names gracefully
                counter = 1
                while dest_path.exists():
                    name, ext = os.path.splitext(clean_name)
                    dest_path = DEST_DIR / target_folder / f"{name}_v{counter}{ext}"
                    counter += 1
                
                # COPY (do not move/delete) the file to the new location
                shutil.copy2(filepath, dest_path)
                print(f"Copied & Renamed: {file} -> {target_folder}/{dest_path.name}")
                copied_files += 1
            else:
                skipped_files += 1

    print("\n--- Migration Complete ---")
    print(f"Successfully routed and cleaned {copied_files} files.")
    print(f"Skipped {skipped_files} files (Images, videos, and raw data).")
    print("\nNEXT STEP: Manually drag your specific raw image datasets into CLEAN_CODE/data/raw/ to preserve their internal structures.")

if __name__ == "__main__":
    main()