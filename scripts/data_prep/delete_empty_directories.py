import shutil
import os

def quarantine_empty_folders(source_dir, trash_dir):
    # Ensure the trash directory exists
    os.makedirs(trash_dir, exist_ok=True)

    for dirpath, _, _ in os.walk(source_dir, topdown=False):
        # Prevent the script from processing the main folder or the trash folder itself
        if dirpath == source_dir or dirpath.startswith(trash_dir): 
            continue 
            
        if not os.listdir(dirpath): # Checks if the folder is completely empty
            # Recreate the folder's path inside the trash directory to prevent name collisions
            relative_path = os.path.relpath(dirpath, source_dir)
            dest_path = os.path.join(trash_dir, relative_path)
            
            # Create parent directories in the trash if they don't exist yet
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            shutil.move(dirpath, dest_path)
            print(f"Moved to safe deletion folder: {dest_path}")

quarantine_empty_folders(r'C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE', r'C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\to_be_deleted_folder')
