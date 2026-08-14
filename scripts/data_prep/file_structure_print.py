import os



def save_directory_tree(startpath, output_filename="project_structure.txt", exclude_dirs=None, exclude_extensions=None):
    """
    Generates a tree-like structure and saves it directly to a text file.
    """
    # 1. Folders to ignore
    if exclude_dirs is None:
        exclude_dirs = {'.git', 'node_modules', '__pycache__', '.venv', 'venv', '.idea', 'ultralytics', '.vscode', 'EfficientSAM', 'GroundingDINO', 'mmdetection', 'segment-anything-2'}
        
    # 2. File extensions to ignore
    if exclude_extensions is None:
        exclude_extensions = ('.png', '.jpg', '.jpeg', '.txt')

    # Open the text file in "write" mode with UTF-8 encoding for the special drawing characters
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(f"Project Tree for: {os.path.abspath(startpath)}\n\n")

        for root, dirs, files in os.walk(startpath):
            # Modify the dirs list in-place to skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            # Filter out files with the excluded extensions
            filtered_files = [file for file in files if not file.lower().endswith(exclude_extensions)]

            # Calculate the current depth level
            level = root.replace(startpath, '').count(os.sep)
            
            # Create the visual indentation
            indent = '│   ' * (level - 1) + '├── ' if level > 0 else ''
            
            # Write the current directory name to the file
            dir_name = os.path.basename(root) if root != startpath else os.path.basename(os.path.abspath(startpath))
            f.write(f"{indent}{dir_name}/\n")
            
            # Write the FILTERED files within the current directory to the file
            subindent = '│   ' * level + '├── '
            for i, file in enumerate(filtered_files):
                # Use '└──' for the last file in the list for a cleaner look
                if i == len(filtered_files) - 1 and not dirs:
                     subindent_last = '│   ' * level + '└── '
                     f.write(f"{subindent_last}{file}\n")
                else:
                     f.write(f"{subindent}{file}\n")
                     
    print(f"Success! The clean file structure has been saved to: {output_filename}")

if __name__ == '__main__':
    # Run the script starting from the current directory
    save_directory_tree('.')