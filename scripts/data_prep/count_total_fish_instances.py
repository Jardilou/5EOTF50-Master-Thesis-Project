import os

def count_total_yolo_instances(labels_dir):
    print(f"Scanning YOLO labels in: {labels_dir}...\n")
    
    if not os.path.exists(labels_dir):
        print(f"Error: Directory '{labels_dir}' does not exist.")
        return

    total_images = 0
    total_fish = 0
    empty_images = 0

    # Iterate through every .txt file in the folder
    for filename in os.listdir(labels_dir):
        if filename.endswith(".txt"):
            total_images += 1
            filepath = os.path.join(labels_dir, filename)
            
            with open(filepath, 'r') as f:
                # Read all lines and filter out any empty whitespace lines
                lines = [line for line in f.readlines() if line.strip()]
                
                fish_in_this_image = len(lines)
                total_fish += fish_in_this_image
                
                if fish_in_this_image == 0:
                    empty_images += 1

    print("-" * 40)
    print("Dataset Count Complete!")
    print("-" * 40)
    print(f"Total .txt files (Images): {total_images}")
    print(f"Empty background images:   {empty_images}")
    print(f"TOTAL FISH INSTANCES:      {total_fish}")
    print("-" * 40)

if __name__ == '__main__':
    # Point this to the folder where your final .txt files live!
    count_total_yolo_instances(r"DATASETS\UTM_dataset\UTM yolo_dataset V1 Refixed\labels\train")