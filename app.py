import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Set page config
st.set_page_config(page_title="PPE Detection App", page_icon="👷‍♂️", layout="centered")

st.title("Industrial PPE Detection 👷‍♂️")
st.write("Upload an image to detect Hardhats and Safety Vests.")

# Load the YOLOv8 model
@st.cache_resource
def load_model():
    # Load your custom trained PPE model
    model = YOLO(r'C:\Users\GIGABYTE\Desktop\Freelance work\Projects\yolov8_ppe_app\runs\ppe_model-4\weights\best.pt') 
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.markdown("---")
    
    if st.button("Run AI Detection 🚀", use_container_width=True):
        with st.spinner("Detecting objects..."):
            # Perform inference
            results = model(image)
            
            # Plot results on image
            res_plotted = results[0].plot()
            
            # Convert BGR to RGB for Streamlit (OpenCV uses BGR)
            res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            st.success("Detection Complete!")
            st.image(res_plotted_rgb, caption='Detected Objects', use_column_width=True)
            
            # Show confidence scores in a clean way
            st.write("### Detection Details")
            boxes = results[0].boxes
            if len(boxes) == 0:
                st.info("No objects detected.")
            else:
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])
                    st.write(f"- **{class_name.capitalize()}**: {confidence:.2%} confidence")
