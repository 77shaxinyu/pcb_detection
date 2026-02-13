import streamlit as st
import cv2
import numpy as np
import os
import sys
import pandas as pd
import shutil
import torch
import gc

# --- 1. 环境初始化 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    import modules_new as modules
    sys.modules['modules.attention'] = modules
    sys.modules['models.common'] = modules
    sys.modules['attention'] = modules
    sys.modules['modules'] = modules
except Exception:
    st.error("Missing modules_new.py")

from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

# 注册注意力机制
try:
    SE_class = getattr(modules, 'SE', getattr(modules, 'SEAttention', None))
    CBAM_class = getattr(modules, 'CBAM', getattr(modules, 'CBAMAttention', None))
    if SE_class: setattr(tasks, 'SE', SE_class)
    if CBAM_class: setattr(tasks, 'CBAM', CBAM_class)
except Exception: pass

@st.cache_resource
def load_yolo_model(model_name):
    path = os.path.join("models", model_name)
    if os.path.exists(path):
        dev = 0 if torch.cuda.is_available() else 'cpu'
        return YOLO(path).to(dev)
    return None

def run_feature_analysis(img_bgr, algo_type):
    canvas = img_bgr.copy()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    feat = cv2.SIFT_create(nfeatures=2000) if algo_type == "Algorithm 1" else cv2.ORB_create(nfeatures=5000)
    kp, _ = feat.detectAndCompute(gray, None)
    if kp: canvas = cv2.drawKeypoints(canvas, kp, None, color=(0, 255, 0))
    return canvas, len(kp) if kp else 0

# --- 2. 界面布局 ---
st.set_page_config(page_title="PCB defect detection", layout="wide")
st.title("PCB defect detection")

st.sidebar.header("Configuration Panel")
ds_select = st.sidebar.selectbox("Test Dataset", ["Dataset 1", "Dataset 2"])
ds_code = "ds1" if ds_select == "Dataset 1" else "ds2"

model_ui = st.sidebar.radio("Inspection Model", ["Model A", "Model B"])
model_code = "se" if model_ui == "Model A" else "cbam"
target_file = f"{ds_code}_{model_code}.pt"

algo_ui = st.sidebar.radio("Analysis Mode", ["Algorithm 1", "Algorithm 2"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.25)

tabs = st.tabs(["Single Image Inspection", "Export all dataset detection results"])

# --- Tab 1: 单图模式 (修复表格显示) ---
with tabs[0]:
    up_file = st.file_uploader("Upload PCB Image", type=["jpg", "png", "jpeg"])
    if up_file:
        raw_img = cv2.imdecode(np.asarray(bytearray(up_file.read()), dtype=np.uint8), 1)
        
        if st.button("Start Analysis", type="primary"):
            model = load_yolo_model(target_file)
            if model:
                # 推理
                res = model.predict(raw_img, conf=conf_threshold, device=0 if torch.cuda.is_available() else 'cpu')
                anno = res[0].plot()
                final, kp_n = run_feature_analysis(anno, algo_ui)
                
                # 创建左右分栏
                col_left, col_right = st.columns([3, 2])
                
                with col_left:
                    st.image(cv2.cvtColor(final, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Detection Result")
                
                with col_right:
                    st.subheader("Detection Summary")
                    # 构建表格数据
                    report_data = []
                    for box in res[0].boxes:
                        label = res[0].names[int(box.cls[0])]
                        conf_val = f"{float(box.conf[0]):.2%}"
                        report_data.append([model_ui, label, conf_val, "Verified"])
                    
                    # 添加算法特征点行
                    report_data.append([algo_ui, "Feature Points", f"{kp_n} pts", "Extracted"])
                    
                    # 转换并显示表格
                    df = pd.DataFrame(report_data, columns=["Method", "Target", "Value", "Status"])
                    st.dataframe(df, hide_index=True, use_container_width=True)

# --- Tab 2: 批量导出模式 (保持防崩溃优化) ---
with tabs[1]:
    st.header("Export all dataset detection results")
    data_path = os.path.join("templates", ds_select.lower().replace(" ", ""))
    if os.path.exists(data_path):
        imgs = [f for f in os.listdir(data_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        st.write(f"Detected **{len(imgs)}** images.")
        if st.button("Execute Batch and Export (ZIP)", type="secondary"):
            # ... 此处省略批量导出的逻辑代码，保持之前的防崩溃版本即可 ...
            st.info("Batch processing starts...")
            # (批量导出代码建议保留上次给你的 gc.collect() 版本)
