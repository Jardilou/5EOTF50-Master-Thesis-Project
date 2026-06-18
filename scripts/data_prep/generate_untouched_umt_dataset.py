import shutil
from pathlib import Path

# 1. Define the source directory (The "Filter" list)
source_dir = Path(r"DATASETS\DATASETS FOR POSE ESTIMATION\UMT_dataset\BACKUP\UMT images for annotation V1 CLEANED")

# 2. Define the new output directory
output_dir = Path(r"DATASETS\UNTOUCHED_UMT_IMAGES_COMBINED")
output_dir.mkdir(parents=True, exist_ok=True)

# 3. Your list of target directories
target_dirs = [
    Path(r"DATASETS\DATASETS FOR POSE ESTIMATION\UMT_dataset\BACKUP\Original Dataset\Images 1-190714\St1-Pantai Pasir Cina\TG4-Red(R) Images st1"),
    Path(r"DATASETS\DATASETS FOR POSE ESTIMATION\UMT_dataset\BACKUP\Original Dataset\Images 1-190714\St2-Pantai Pasir Cina\TG4-Black(L) Images st2 (P.C.2)"),
    Path(r"DATASETS\DATASETS FOR POSE ESTIMATION\UMT_dataset\BACKUP\Original Dataset\Images 1-190714\St3-Pantai Vietnam\TG4-Red(R) Images st3 (P.V)"),
    Path(r"DATASETS\DATASETS FOR POSE ESTIMATION\UMT_dataset\BACKUP\Original Dataset\Images 2-190715\St4-Karang Tengah\TG4 Black(L) images st4 (K.T)"),
    Path(r"DATASETS\DATASETS FOR POSE ESTIMATION\UMT_dataset\BACKUP\Original Dataset\Images 2-190715\St6-Batu Menangis\TG4 Red(R) images st6 (B.M)"),
    Path(r"DATASETS\DATASETS FOR POSE ESTIMATION\UMT_dataset\BACKUP\Original Dataset\Images 3-190716\St7-Teluk Air\TG4 Black(L) images st7 (T.A)"),
    Path(r"DATASETS\DATASETS FOR POSE ESTIMATION\UMT_dataset\BACKUP\Original Dataset\Images 3-190716\St8-Vietnamese Jetty (R)\TG4 Black(L) images st8 (V.J)"),
    Path(r"DATASETS\DATASETS FOR POSE ESTIMATION\UMT_dataset\BACKUP\Original Dataset\Images 4-190717\St9-Geluk West\TG4 Red(R) images st9 (P.G)"),
    Path(r"DATASETS\DATASETS FOR POSE ESTIMATION\UMT_dataset\BACKUP\Original Dataset\Images 4-190717\St10-Batu Payung\TG4 Red(R) images st10 (B.P)"),
    Path(r"DATASETS\DATASETS FOR POSE ESTIMATION\UMT_dataset\BACKUP\Original Dataset\Images 5-190722\St11-Christmas Garden\TG4 Black(L) images st11 (C.G)"),
    Path(r"DATASETS\DATASETS FOR POSE ESTIMATION\UMT_dataset\BACKUP\Original Dataset\Images 5-190722\St12-Dinding Laut\TG4 Black(L) images st12 (D.L)"),
    Path(r"DATASETS\DATASETS FOR POSE ESTIMATION\UMT_dataset\BACKUP\Original Dataset\Images 6-190723\St13-Geluk East\TG4 Black(L) images st13"),
    Path(r"DATASETS\DATASETS FOR POSE ESTIMATION\UMT_dataset\BACKUP\Original Dataset\Images 6-190723\St14-Tengkorak West\TG4 Black(L) Images st14"),
    Path(r"DATASETS\DATASETS FOR POSE ESTIMATION\UMT_dataset\BACKUP\Original Dataset\Images 7-190724\St15-Pulau Karah\TG4 Red(R) Images St15"),
    Path(r"DATASETS\DATASETS FOR POSE ESTIMATION\UMT_dataset\BACKUP\Original Dataset\Images 7-190724\St16-Tengkorak East\TG4 Black(L) Images St16")
]
# Get filenames to exclude
exclude_names = {f.name for f in source_dir.iterdir() if f.is_file()} if source_dir.is_dir() else set()

processed_folders = []

print(f"Starting process. Excluding {len(exclude_names)} filenames.\n")

for target in target_dirs:
    if not target.is_dir():
        continue
    
    # Create a 'Cleaned_Set' subfolder inside the current target folder
    destination_subfolder = output_dir/f"{target.name}_Cleaned_Set"
    destination_subfolder.mkdir(exist_ok=True)
    
    file_count = 0
    for file_path in target.iterdir():
        # Ensure we only process files (and don't try to copy our new subfolder)
        if file_path.is_file() and file_path.name not in exclude_names:
            shutil.copy2(file_path, destination_subfolder / file_path.name)
            file_count += 1
            
    processed_folders.append(target.name)
    print(f"Done: {target.name} ({file_count} files copied to subfolder)")

# Print final summary list
print("\n" + "="*30)
print("LIST OF PROCESSED FOLDERS:")
print("="*30)
for name in processed_folders:
    print(f"- {name}")