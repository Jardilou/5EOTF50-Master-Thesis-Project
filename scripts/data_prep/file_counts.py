import os

def count_files_in_subfolders(root_folder):
    print(f"Scanning: {root_folder}\n")
    print(f"{'Subfolder Path':<60} | {'File Count':<10}")
    print("-" * 75)
    
    # os.walk navigates through the directory tree
    for dirpath, dirnames, filenames in os.walk(root_folder):
        # Optional: Skip the root folder itself if you only want subfolders
        if dirpath == root_folder:
            continue
            
        # Count the number of files in the current directory
        file_count = len(filenames)
        
        # Print the path and the count formatted nicely
        print(f"{dirpath:<60} | {file_count:<10}")

if __name__ == "__main__":
    # Replace this with the path to your root folder
    # Use raw string (r"path") on Windows to avoid escape character issues
    # ROOT_PATH = r"DATASETS\DATASETS FOR INSTANCE SEGMENTATION\Fish4knowledge\fish_image" 
    # ROOT_PATH = r"DATASETS\DATASETS FOR POSE ESTIMATION\UMT_dataset\BACKUP\Original Dataset"
    # ROOT_PATH = r"DATASETS\DATASETS FOR POSE ESTIMATION\BézierFusion Dataset 1"
    ROOT_PATH = r"DATASETS\DATASETS FOR POSE ESTIMATION\Bézierfusion Dataset 2 Reannotated V4"
    if os.path.exists(ROOT_PATH):
        count_files_in_subfolders(ROOT_PATH)
    else:
        print("The specified root folder does not exist. Please check the path.")