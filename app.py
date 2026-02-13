import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

st.set_page_config(page_title="PCB Detection System", layout="wide")

MODEL_DIR = "models"

def register_modules():
    try:
        from modules.attention import SE, CBAM 
        setattr(tasks, 'SE', SE)
        setattr(tasks, 'CBAM', CBAM)
    except Exception as e:
        pass

@st.cache_resource
def get_yolo_model(model_name):
    path = os.path.join(MODEL_DIR, model_name)
    if os.path.exists(path):
        return YOLO(path)
    return None

register_modules()

def run_feature_extraction(img_bgr, algo_type):
    H, W = img_bgr.shape[:2]
    display_img = img_bgr.copy()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    for i in range(1, 9):
        cv2.line(display_img, (0, int(i * H / 9)), (W, int(i * H / 9)), (0, 255, 0), 1)
        cv2.line(display_img, (int(i * W / 9), 0), (int(i * W / 9), H), (0, 255, 0), 1)

    if algo_type == "SIFT":
        engine = cv2.SIFT_create(nfeatures=2000)
        color = (255, 0, 0)
    else:
        engine = cv2.ORB_create(nfeatures=5000)
        color = (0, 0, 255)

    kp, des = engine.detectAndCompute(gray, None)
    
    if kp:
        display_img = cv2.drawKeypoints(display_img, kp, None, color=color)

    return display_img, len(kp) if kp else 0

st.title("PCB Intelligent Detection Platform")

st.sidebar.header("Control Center")

ds_option = st.sidebar.selectbox("Select Dataset", ["Dataset 1", "Dataset 2"])
ds_code = "ds1" if ds_option == "Dataset 1" else "ds2"

model_type = st.sidebar.radio("Attention Mechanism", ["Model-SE", "Model-CBAM"])
model_code = "se" if model_type == "Model-SE" else "cbam"

target_model_file = f"{ds_code}_{model_code}.pt"

algo_option = st.sidebar.radio("Analysis Algorithm", ["Algorithm 1", "Algorithm 2"])
real_algo = "SIFT" if "Algorithm 1" in algo_option else "ORB"

conf_val = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.25)

uploaded_file = st.file_uploader("Upload PCB Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1)

    col1, col2 = st.columns([3, 2])

    with col1:
        if st.button("Run Analysis", type="primary"):
            active_model = get_yolo_model(target_model_file)
            
            if active_model:
                yolo_results = active_model.predict(original_img, conf=conf_val)
                res_img = yolo_results[0].plot()

                final_output, kp_count = run_feature_extraction(res_img, real_algo)

                st.image(cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB), 
                         caption=f"Result: {target_model_file} and {real_algo}", 
                         use_container_width=True)

                with col2:
                    st.subheader("Detection Report")
                    
                    report_list = []
                    for box in yolo_results[0].boxes:
                        cls_name = yolo_results[0].names[int(box.cls[0])]
                        conf_score = f"{float(box.conf[0]):.2%}"
                        report_list.append(["YOLO", cls_name, conf_score, "Detected"])
                    
                    report_list.append([real_algo, "Features", f"{kp_count} pts", "Extracted"])
                    
                    df = pd.DataFrame(report_list, columns=["Method", "Target", "Score", "Status"])
                    st.table(df)
            else:
                st.error(f"Model file not found: {target_model_file}")
        else:
            st.image(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), caption="Waiting for analysis", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.write(f"System Ready")
