import os
import shutil
import re
from pathlib import Path

# --- CONFIGURATION ---
SOURCE_DIR = Path(".")
DEST_ROOT_NAME = "CLEAN_CODE"  # The name of your main target folder
DEST_BASE_DIR = Path(f"{DEST_ROOT_NAME}/scripts/utils")

# Folders to completely ignore to prevent recursion and bloat
IGNORE_FOLDERS = {DEST_ROOT_NAME, '.git', 'venv', '.venv', '__pycache__', '.idea'}

def to_snake_case(name, is_file=False):
    """
    Converts strings like 'FILE MANIPULATION STEPS' to 'file_manipulation_steps'
    and 'Find-File.py' to 'find_file.py'.
    """
    if is_file:
        stem, ext = os.path.splitext(name)
    else:
        stem, ext = name, ""
    
    # 1. Convert to lowercase
    stem = stem.lower()
    # 2. Replace spaces, hyphens, and plus signs with underscores
    stem = re.sub(r'[\s\-+]+', '_', stem)
    # 3. Remove any weird special characters
    stem = re.sub(r'[^a-z0-9_]', '', stem)
    # 4. Clean up consecutive underscores
    stem = re.sub(r'_+', '_', stem).strip('_')
    
    return stem + ext

def main():
    print("--- Starting Safe Python Script Extraction ---")
    
    if not SOURCE_DIR.exists():
        print(f"Error: Source directory '{SOURCE_DIR}' not found.")
        return

    copied_count = 0

    for root, dirs, files in os.walk(SOURCE_DIR):
        # PREVENT RECURSION: 
        # Modify the 'dirs' list in-place so os.walk ignores our target folder and venvs
        dirs[:] = [d for d in dirs if d not in IGNORE_FOLDERS]

        for file in files:
            # We only want to copy Python scripts
            if file.endswith('.py'):
                src_path = Path(root) / file
                
                # Get the relative path
                rel_path = src_path.relative_to(SOURCE_DIR)
                
                # Convert all parent folder names to snake_case
                clean_parent_folders = [to_snake_case(folder) for folder in rel_path.parent.parts]
                
                # Convert the file name to snake_case
                clean_filename = to_snake_case(file, is_file=True)
                
                # Build the new destination directory path
                dest_dir = DEST_BASE_DIR.joinpath(*clean_parent_folders)
                dest_path = dest_dir / clean_filename
                
                # Create the folders if they don't exist
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                # Handle duplicate names gracefully
                counter = 1
                while dest_path.exists():
                    stem, ext = os.path.splitext(clean_filename)
                    dest_path = dest_dir / f"{stem}_v{counter}{ext}"
                    counter += 1
                
                # Copy the file safely
                shutil.copy2(src_path, dest_path)
                
                # Print exactly what happened for visual confirmation
                print(f"COPIED: {rel_path}")
                print(f"    TO: {DEST_BASE_DIR.name}/{'/'.join(clean_parent_folders)}/{dest_path.name}\n")
                
                copied_count += 1

    print("--- Extraction Complete ---")
    print(f"Successfully cleaned, restructured, and safely copied {copied_count} Python scripts.")

if __name__ == "__main__":
    main()