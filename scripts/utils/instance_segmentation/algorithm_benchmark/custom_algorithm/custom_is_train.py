from ultralytics import YOLO


def main():
    # 1. Initialize the model using your custom YAML configuration
    # (Save your YAML file as 'custom_eamrf_seg.yaml')
    model = YOLO(r"YAML FILES\Custom-IS-Network.yaml")

    # 2. Load standard pre-trained weights to populate the YOLO11 Backbone
    # This prevents you from having to train the feature extractor from scratch
    model.load("yolo11n-seg.pt")

    # 3. Start training your custom model
    results = model.train(
        data=r"YAML FILES\Custom-IS-paths.yaml", 
        epochs=500,               
        imgsz= 320,             
        batch=16,                 
        device="cpu",               
        project="Fish_IS_Comparison",
        name="YOLO_Detection_Run",
        
        # --- Advanced Augmentations for Isolated Fish ---
        degrees=45.0,           # Fish rotate freely; train the model to recognize them at angles
        fliplr=0.5,             # 50% chance to flip left/right (swimming direction)
        flipud=0.1,             # 10% chance to flip upside down (dead/rolling fish)
        scale=0.4,              # Zoom in/out by 50% to simulate distance
        
        # --- Underwater Color Simulation ---
        hsv_h=0.02,             # Shift hue to simulate different water tints (green/blue)
        hsv_s=0.7,              # Heavy saturation shifts (turbid vs clear water)
        hsv_v=0.4,              # Brightness shifts (shallow vs deep water)
        
        # --- Training Dynamics ---
        optimizer='auto',       # Let YOLO select the best optimizer (usually AdamW for classification)
        lr0=0.001,              # Initial learning rate
        cos_lr=True,            # Cosine learning rate scheduling for better convergence
        patience=100,
        save=True
    )

if __name__ == '__main__':
    main()