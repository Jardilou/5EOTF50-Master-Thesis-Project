import os
import shutil

def filter_and_copy_dataset():
    # ==========================================
    # --- 1. CONFIGURATION ---
    # ==========================================
    # Change these to your actual folder paths!
    SOURCE_DIR = r"DATASETS\UTM_dataset\UTM images for annotation V1"
    DEST_DIR = r"DATASETS\UTM_dataset\UTM images for annotation V1 CLEANED"

    # Your exact list of filenames to ignore
    EXCLUDED_FILES = {
        "St2_0-25m(L).MOV-37.jpg",
        "St2_0-25m(L).MOV-34.jpg",
        "St11_50-75m(L).MOV-49.jpg",
        "St11_50-75m(L).MOV-3.jpg",
        "P7240005.MOV-35.jpg",
        "P7240005.MOV-21.jpg",
        "P7240005.MOV-17.jpg",
        "P7240004.MOV-35.jpg",
        "P7240003.MOV-42.jpg",
        "P7240001.MOV-8.jpg",
        "P7240001.MOV-19.jpg",
        "P7230001.MOV-22.jpg",
        "P7230001.MOV-5.jpg",
        "P7230002.MOV-23.jpg",
        "P7230001.MOV-1.jpg",
        "P7230009.MOV-5.jpg",
        "P7230009.MOV-20.jpg",
        "P7230007.MOV-25.jpg",
        "P7230007.MOV-19.jpg",
        "P7230006.MOV-5.jpg",
        "P7230006.MOV-35.jpg",
        "P7230006.MOV-39.jpg",
        "P7230006.MOV-40.jpg",
        "P7230006.MOV-43.jpg",
        "P7230006.MOV-45.jpg",
        "P7230006.MOV-25.jpg",
        "P7230006.MOV-20.jpg",
        "P7230006.MOV-15.jpg",
        "P7230006.MOV-10.jpg",
        "P7230004.MOV-5.jpg",
        "P7230004.MOV-42.jpg",
        "P7230004.MOV-22.jpg",
        "P7230004.MOV-1.jpg",
        "P7230003.MOV-28.jpg",
        "P7230003.MOV-1.jpg",
        "P7230002.MOV-48.jpg",
        "P7230002.MOV-44.jpg",
        "P7230002.MOV-37.jpg",
        "P7230002.MOV-30.jpg",
        "P7230003.MOV-9.jpg",
        "P7230003.MOV-19.jpg",
        "P7230009.MOV-30.jpg",
        "P7230009.MOV-35.jpg",
    }
    # ==========================================

    # Create the clean destination folder if it doesn't exist
    os.makedirs(DEST_DIR, exist_ok=True)
    
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory '{SOURCE_DIR}' not found.")
        return

    print(f" Scanning '{SOURCE_DIR}'...")
    
    copied_count = 0
    skipped_count = 0

    # Iterate through every file in the source folder
    for filename in os.listdir(SOURCE_DIR):
        src_path = os.path.join(SOURCE_DIR, filename)
        
        # Make sure we are only looking at files, not accidental sub-folders
        if os.path.isfile(src_path):
            # Check if the filename is in our blocklist (using O(1) set lookup)
            if filename in EXCLUDED_FILES:
                skipped_count += 1
            else:
                # Safe to copy!
                dst_path = os.path.join(DEST_DIR, filename)
                shutil.copy2(src_path, dst_path)
                copied_count += 1

    print("\nDataset Copying Complete!")
    print(f"Images safely copied to new folder: {copied_count}")
    print(f"Excluded images successfully dropped: {skipped_count}")

if __name__ == '__main__':
    filter_and_copy_dataset()