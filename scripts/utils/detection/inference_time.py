from ultralytics import YOLO

# 1. Load a pretrained YOLO model (e.g., YOLOv8 nano)
model = YOLO(r'weights\detection\detection_Only_training_run-with_medium_pretrained_weights\weights\best.pt') 

# 2. Run inference on an image (can be a local path or URL)
results = model(r'data\processed\UMT\untouched_images_for_assessment\selected_untouched_images\P7230001.MOV-3.jpg')

# 3. Retrieve the time dictionary from the first result object
speed = results[0].speed

# 4. Print the breakdown (times are in milliseconds)
print(f"Pre-process Time:  {speed['preprocess']:.2f} ms")
print(f"Inference Time:    {speed['inference']:.2f} ms")
print(f"Post-process Time: {speed['postprocess']:.2f} ms")
print(f"Total Time:        {sum(speed.values()):.2f} ms")