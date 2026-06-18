import os

def rename_files(root_dir, exclude_folders):
    # Walk through the directory tree
    for dirpath, dirnames, filenames in os.walk(root_dir):
        
        # Modify the dirnames list in-place to remove excluded folders
        dirnames[:] = [d for d in dirnames if d not in exclude_folders]

        # Loop through all the files in the current directory
        for filename in filenames:
            
            # Check if the file is a Python, Jupyter Notebook, or YAML file
            if filename.endswith('.py') or filename.endswith('.ipynb') or filename.endswith('.yaml'):
                
                # Check if there's a space or underscore in the filename
                if ' ' in filename or '_' in filename:
                    
                    # Create the new filename
                    new_name = filename.replace(' ', '-').replace('_', '-')
                    
                    # Construct full absolute/relative paths
                    old_file_path = os.path.join(dirpath, filename)
                    new_file_path = os.path.join(dirpath, new_name)
                    
                    # Rename the file
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed: '{filename}' -> '{new_name}'")

if __name__ == "__main__":
    target_directory = "." 
    
    # Folders to skip
    folders_to_skip = ["DATASETS", "ALGORITHM BENCHMARK", "ultralytics"]
    
    print("Starting file rename process...")
    rename_files(target_directory, folders_to_skip)
    print("Done!")