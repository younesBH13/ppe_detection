# YOLOv8 PPE Detection Portfolio Project

This project contains a Streamlit web application that runs YOLOv8 inference on uploaded images. This is the perfect format for a Fiverr portfolio piece.

## Step 1: Install Requirements
Open your terminal and run:
```bash
pip install ultralytics streamlit opencv-python Pillow
```

## Step 2: Test the Web App (Base Model)
Before training, you can test the UI to make sure everything works.
Navigate to this folder in your terminal and run:
```bash
streamlit run app.py
```
*Note: Right now, `app.py` is using the default `yolov8n.pt` model which detects people, cars, etc. We will swap this out for your custom PPE model next.*

## Step 3: Get the PPE Dataset & Train (Custom Model)

1. **Get the Dataset:** 
   Go to [Roboflow Universe](https://universe.roboflow.com/) and search for "PPE" or "Hardhat". 
   Find a dataset you like, click "Download Dataset", select "YOLOv8" format, and it will give you a Python snippet or a zip file.

2. **Train the Model:**
   Create a file called `train.py` in this folder with the following code (update the `data` path to point to your downloaded `data.yaml`):

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
results = model.train(data="path/to/your/dataset/data.yaml", epochs=20, imgsz=640)
```
Run `python train.py`. This will take about 10-20 minutes on a CPU, or 2 minutes on a GPU.

## Step 4: Update the Web App
Once training finishes, your custom model will be saved in `runs/detect/train/weights/best.pt`.
Open `app.py`, find the `load_model()` function, and change `'yolov8n.pt'` to `'runs/detect/train/weights/best.pt'`.

Now, restart Streamlit and screen-record your new custom AI detecting hardhats on the web interface!
