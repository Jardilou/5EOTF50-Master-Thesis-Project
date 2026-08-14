import shutil
import re
from pathlib import Path

def natural_sort_key(path_object):
    """
    Breaks a filename into text and numbers so it sorts numerically.
    e.g., Frame-41 will be sorted before Frame-217.
    """
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', path_object.name)]

def sample_files(source_dir, target_dir, step, start_idx=0, end_idx=None):
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # 1. Ensure the target directory exists
    target_path.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return

    # 2. Get the list of files and apply NATURAL SORTING
    files = [f for f in source_path.iterdir() if f.is_file()]
    files = sorted(files, key=natural_sort_key)
    
    total_files = len(files)

    if total_files == 0:
        print("No files found in the source directory.")
        return

    # 3. Handle the end index
    if end_idx is None or end_idx > total_files:
        end_idx = total_files

    # 4. Slice the list to get exactly what we need
    sampled_files = files[start_idx:end_idx:step]

    print(f"Total files in source: {total_files}")
    print(f"Sampling {len(sampled_files)} files (Start: {start_idx}, End: {end_idx}, Step: {step})...")

    # 5. Copy the selected files
    for file_path in sampled_files:
        target_file_path = target_path / file_path.name
        
        shutil.copy2(file_path, target_file_path)
        print(f"Copied -> {file_path.name}")

    print("\n Sampling complete!")

# ==========================================
# Configuration and Execution
# ==========================================
if __name__ == "__main__":
    # Define your paths (relative or absolute)
    # Using "../" or "./adjacent_folder" creates it right next to your current location
    SOURCE_DIRECTORY_LEFT = r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\data\processed\calibration-v2\sampled_and_synchronized\ground_truth_2_left_frames" 
    SOURCE_DIRECTORY_RIGHT = r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\data\processed\calibration-v2\sampled_and_synchronized\ground_truth_2_right_frames" 
    TARGET_DIRECTORY_LEFT = r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\data\processed\calibration-v2\subdatasets_selected\ground_truth_selected_for_pipeline_assessment\left" 
    TARGET_DIRECTORY_RIGHT = r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\data\processed\calibration-v2\subdatasets_selected\ground_truth_selected_for_pipeline_assessment\right" 
    
    # Define your sampling parameters
    X_STEP = 15          # Take 1 sample every 5 files
    START_SAMPLE = 23   # Start at the 11th file (index is 0-based, so 10)
    END_SAMPLE = 493    # Stop at the 100th file (index 99)

    sample_files(
        source_dir=SOURCE_DIRECTORY_LEFT,
        target_dir=TARGET_DIRECTORY_LEFT,
        step=X_STEP,
        start_idx=START_SAMPLE,
        end_idx=END_SAMPLE
    )

    sample_files(
        source_dir=SOURCE_DIRECTORY_RIGHT,
        target_dir=TARGET_DIRECTORY_RIGHT,
        step=X_STEP,
        start_idx=START_SAMPLE,
        end_idx=END_SAMPLE
    )