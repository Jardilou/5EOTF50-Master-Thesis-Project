from ultralytics import YOLO

def train_sota_classifier():
    # 1. Load the Extra-Large Classification Model
    # 'x' has the highest parameter count, capturing the most complex features.
    model = YOLO("yolo11x-cls.pt") 

    print("Initiating High-Accuracy Species Training...")

    # 2. Execute Training with Advanced Hyperparameters
    results = model.train(
        # --- Core Settings ---
        data=r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\DATASETS\DATASETS FOR INSTANCE SEGMENTATION\Fish4knowledge\Fish_Species_Classifier_Data", 
        epochs=500,             # Set high; we will rely on early stopping
        imgsz=384,              # Increased resolution to catch fin/scale details
        batch=8,               # Adjust to 16 if your GPU runs out of VRAM
        device="cpu",             # GPU 0
        project="Fish_Species_Recognition",
        name="YOLO11x_Species_Recognition_Run",
        
        # --- Advanced Augmentations for Isolated Fish ---
        degrees=45.0,           # Fish rotate freely; train the model to recognize them at angles
        fliplr=0.5,             # 50% chance to flip left/right (swimming direction)
        flipud=0.1,             # 10% chance to flip upside down (dead/rolling fish)
        scale=0.5,              # Zoom in/out by 50% to simulate distance
        
        # --- Underwater Color Simulation ---
        hsv_h=0.02,             # Shift hue to simulate different water tints (green/blue)
        hsv_s=0.7,              # Heavy saturation shifts (turbid vs clear water)
        hsv_v=0.4,              # Brightness shifts (shallow vs deep water)
        
        # --- Training Dynamics ---
        optimizer='auto',       # Let YOLO select the best optimizer (usually AdamW for classification)
        lr0=0.001,              # Initial learning rate
        cos_lr=True,            # Cosine learning rate scheduling for better convergence
        patience=50,            # Stop training if validation accuracy doesn't improve for 50 epochs
        save=True               # Ensure best weights are saved
    )

    print("Training Complete! Best weights saved to: Fish_Species_Recognition/YOLO11x_Species_Recognition_Run/weights/best.pt")

if __name__ == '__main__':
    train_sota_classifier()