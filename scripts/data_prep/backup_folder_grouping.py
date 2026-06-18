import os
import shutil

def consolidate_backups(root_dir, exclude_folders=None, dest_folder_name="BACKUP-FOLDERS"):
    # Default to an empty list if no exclusions are provided
    if exclude_folders is None:
        exclude_folders = []
        
    # Get absolute path for cleaner parsing
    root_dir = os.path.abspath(root_dir)
    dest_dir_path = os.path.join(root_dir, dest_folder_name)
    
    # Create the destination folder if it doesn't exist
    if not os.path.exists(dest_dir_path):
        os.makedirs(dest_dir_path)

    # Collect paths first before we start moving things around
    backups_to_move = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        
        # 1. Skip the new destination folder AND any user-specified folders
        # Modifying dirnames[:] in-place prevents os.walk from descending into them
        dirnames[:] = [d for d in dirnames if d not in exclude_folders and d != dest_folder_name]

        # 2. Look for folders named 'backup' or 'BACKUP' (case-insensitive)
        for d in list(dirnames): 
            if d.lower() == "backup":
                src_path = os.path.join(dirpath, d)
                
                # Get the name of the parent directory
                parent_name = os.path.basename(dirpath)
                
                # Edge case: If the backup is sitting right in the root folder
                if dirpath == root_dir:
                    parent_name = "Root"
                
                # Construct the new intended name
                new_folder_name = f"Backup-{parent_name}"
                dest_path = os.path.join(dest_dir_path, new_folder_name)
                
                # Add to our list
                backups_to_move.append((src_path, dest_path))
                
                # Remove from dirnames so os.walk doesn't try to look inside the backup we are moving
                dirnames.remove(d)

    # 3. Move the collected folders
    print(f"Found {len(backups_to_move)} backup folder(s) to move...")
    for src, dest in backups_to_move:
        
        # Conflict resolution: If Backup-XXX already exists, add a number (Backup-XXX_1, Backup-XXX_2)
        counter = 1
        final_dest = dest
        while os.path.exists(final_dest):
            final_dest = f"{dest}_{counter}"
            counter += 1
        
        # Move the folder and its contents
        shutil.move(src, final_dest)
        
        # Print a clean summary
        print(f"Moved: '{os.path.relpath(src, root_dir)}' -> '{os.path.basename(final_dest)}'")

if __name__ == "__main__":
    target_directory = "." 
    
    # Add any folders you want the script to completely ignore here
    folders_to_skip = ["DATASETS", "ultralytics"]
    
    consolidate_backups(target_directory, exclude_folders=folders_to_skip)
    print("Done!")