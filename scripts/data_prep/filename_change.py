# # # import pathlib

# # # # Define the directory ('.' means the current folder)
# # # target_dir = pathlib.Path(r"DATASETS\DATASET FOR STEREO CALIBRATION\Calibration images\Stereo Video Calibration STEREO-DOVs\All_Original_Calibration_Images")



# # # for f in target_dir.iterdir():
# # #     if f.is_file():
# # #         # Check for the ALL CAP "L_"
# # #         if "Left_Image_" in f.name:
# # #             if "Right_Image_" in f.name:
# # #                 new_name = f.name.replace("Left_Image_", "", 1)
# # #                 new_name = f.name.replace("Right_Image_", "Left_Image_", 1)
# # #                 f.rename(f.with_name(new_name))
            
                
    
# # #             # # Remove the first "R_" and add "Right_Image_" to the very front
# # #             # # Note: I added an underscore to "Right_Image_" for consistency
# # #             # new_name =  f.name.replace("-R_", "-", 1)
# # #             # f.rename(f.with_name(new_name))
# # #         else:
            
# # #             print(f"Skipped (already renamed): {f.name}")
# # #     else:
# # #         print(f"Skipped (not a file): {f.name}")
# # import os

# # def clean_filenames(directory):
# #     print(f"Scanning directory: '{directory}'...")
    
# #     # Initialize a counter to keep track of how many files we change
# #     renamed_count = 0
    
# #     # Loop through all files in the target directory
# #     for filename in os.listdir(directory):
# #         new_name = filename
        
# #         # Target and replace the duplicated "Left_Image" text
# #         if "unchangedLeft_Image" in new_name:
# #             new_name = new_name.replace("unchangedLeft_Image", "unchanged")
            
# #         # Target and replace the duplicated "Right_Image" text (just in case!)
# #         if "unchanged_" in new_name:
# #             new_name = new_name.replace("unchange_", "unchanged")
            
# #         # If the name was modified, execute the rename on the operating system
# #         if new_name != filename:
# #             old_path = os.path.join(directory, filename)
# #             new_path = os.path.join(directory, new_name)
            
# #             os.rename(old_path, new_path)
# #             print(f"Renamed:\n  From: {filename}\n  To:   {new_name}\n")
# #             renamed_count += 1
            
# #     print("--- Renaming Complete ---")
# #     print(f"Total files fixed: {renamed_count}")

# # if __name__ == "__main__":
# #     # Update this path if your files are in a different folder
# #     # "." means the current folder where the script is running
# #     TARGET_DIRECTORY = r"DATASETS\DATASET FOR STEREO CALIBRATION\Calibration images\Stereo Video Calibration STEREO-DOVs\All_Original_Calibration_Images" 
    
# #     clean_filenames(TARGET_DIRECTORY)
# import os

# def remove_last_character(directory):
#     print(f"Scanning directory: '{directory}'...")
#     renamed_count = 0
    
#     # Get the name of this script so it doesn't accidentally rename itself!
#     script_name = os.path.basename(__file__)

#     for filename in os.listdir(directory):
#         # Skip folders and the script itself
#         if os.path.isdir(os.path.join(directory, filename)) or filename == script_name:
#             continue
            
#         # Split the file into its base name and its extension (e.g., "Image_" and ".jpg")
#         base_name, extension = os.path.splitext(filename)
        
#         # Check to make sure the base name isn't completely empty
#         if len(base_name) > 0:
#             # Slice off the last character: [:-1] means "everything except the last item"
#             new_base = base_name[:-1]
#             new_name = new_base + extension
            
#             # If the name actually changed, execute the rename
#             if new_name != filename:
#                 old_path = os.path.join(directory, filename)
#                 new_path = os.path.join(directory, new_name)
                
#                 os.rename(old_path, new_path)
#                 print(f"Renamed:\n  From: {filename}\n  To:   {new_name}\n")
#                 renamed_count += 1

#     print("--- Renaming Complete ---")
#     print(f"Total files fixed: {renamed_count}")

# if __name__ == "__main__":
#     # "." means the current folder where the script is running. 
#     # Change this path if your images are somewhere else.
#     TARGET_DIRECTORY = "." 
    
#     remove_last_character(TARGET_DIRECTORY)
# # import pathlib

# # # Define the directory ('.' means the current folder)
# # target_dir = pathlib.Path(r"DATASETS\DATASET FOR STEREO CALIBRATION\Calibration images\Stereo Video Calibration STEREO-DOVs\All_Original_Calibration_Images")



# # for f in target_dir.iterdir():
# #     if f.is_file():
# #         # Check for the ALL CAP "L_"
# #         if "Left_Image_" in f.name:
# #             if "Right_Image_" in f.name:
# #                 new_name = f.name.replace("Left_Image_", "", 1)
# #                 new_name = f.name.replace("Right_Image_", "Left_Image_", 1)
# #                 f.rename(f.with_name(new_name))
            
                
    
# #             # # Remove the first "R_" and add "Right_Image_" to the very front
# #             # # Note: I added an underscore to "Right_Image_" for consistency
# #             # new_name =  f.name.replace("-R_", "-", 1)
# #             # f.rename(f.with_name(new_name))
# #         else:
            
# #             print(f"Skipped (already renamed): {f.name}")
# #     else:
# #         print(f"Skipped (not a file): {f.name}")
# import os

# def clean_filenames(directory):
#     print(f"Scanning directory: '{directory}'...")
    
#     # Initialize a counter to keep track of how many files we change
#     renamed_count = 0
    
#     # Loop through all files in the target directory
#     for filename in os.listdir(directory):
#         new_name = filename
        
#         # Target and replace the duplicated "Left_Image" text
#         if "unchangedLeft_Image" in new_name:
#             new_name = new_name.replace("unchangedLeft_Image", "unchanged")
            
#         # Target and replace the duplicated "Right_Image" text (just in case!)
#         if "unchanged_" in new_name:
#             new_name = new_name.replace("unchange_", "unchanged")
            
#         # If the name was modified, execute the rename on the operating system
#         if new_name != filename:
#             old_path = os.path.join(directory, filename)
#             new_path = os.path.join(directory, new_name)
            
#             os.rename(old_path, new_path)
#             print(f"Renamed:\n  From: {filename}\n  To:   {new_name}\n")
#             renamed_count += 1
            
#     print("--- Renaming Complete ---")
#     print(f"Total files fixed: {renamed_count}")

# if __name__ == "__main__":
#     # Update this path if your files are in a different folder
#     # "." means the current folder where the script is running
#     TARGET_DIRECTORY = r"DATASETS\DATASET FOR STEREO CALIBRATION\Calibration images\Stereo Video Calibration STEREO-DOVs\All_Original_Calibration_Images" 
    
#     clean_filenames(TARGET_DIRECTORY)
import os

def remove_last_character(directory):
    print(f"Scanning directory: '{directory}'...")
    renamed_count = 0
    
    # Get the name of this script so it doesn't accidentally rename itself!
    script_name = os.path.basename(__file__)

    for filename in os.listdir(directory):
        # Skip folders and the script itself
        if os.path.isdir(os.path.join(directory, filename)) or filename == script_name:
            continue
            
        # Split the file into its base name and its extension (e.g., "Image_" and ".jpg")
        base_name, extension = os.path.splitext(filename)
        
        # Check to make sure the base name isn't completely empty
        if len(base_name) > 0:
            # Slice off the last character: [:-1] means "everything except the last item"
            new_base = base_name[:-1]
            new_name = new_base + extension
            
            # If the name actually changed, execute the rename
            if new_name != filename:
                old_path = os.path.join(directory, filename)
                new_path = os.path.join(directory, new_name)
                
                os.rename(old_path, new_path)
                print(f"Renamed:\n  From: {filename}\n  To:   {new_name}\n")
                renamed_count += 1

    print("--- Renaming Complete ---")
    print(f"Total files fixed: {renamed_count}")

if __name__ == "__main__":
    # "." means the current folder where the script is running. 
    # Change this path if your images are somewhere else.
    TARGET_DIRECTORY = r"DATASETS\DATASET FOR STEREO CALIBRATION\Calibration images\Stereo Video Calibration STEREO-DOVs\All_Original_Calibration_Images" 
    
    remove_last_character(TARGET_DIRECTORY)