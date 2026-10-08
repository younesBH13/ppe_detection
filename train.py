from ultralytics import YOLO

if __name__ == '__main__':
    # Load the base model
    model = YOLO("yolov8n.pt") 
    
    # Train the model pointing to the absolute path of the data.yaml
    print("🚀 Starting training. This may take some time depending on your hardware...")
    results = model.train(
        data="c:/Users/GIGABYTE/Desktop/Freelance work/Projects/yolov8_ppe_app/yolov8_ppe/data.yaml", 
        epochs=10, # Lowered to 10 for a quicker portfolio demo (you can increase later)
        imgsz=640,
        project="c:/Users/GIGABYTE/Desktop/Freelance work/Projects/yolov8_ppe_app/runs",
        name="ppe_model",
        workers=0, # Set to 0 to completely fix Windows memory/paging errors
        batch=8    # Lower batch size to reduce RAM/VRAM footprint
    )
    print("✅ Training complete! Your model is saved in runs/ppe_model/weights/best.pt")
