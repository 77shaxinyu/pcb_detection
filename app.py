import streamlit as st
import cv2
import numpy as np
import os
import sys
import pandas as pd
import shutil
import torch
import gc

# 1. 环境初始化
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    import modules_new as modules
    sys.modules['modules.attention'] = modules
    sys.modules['models.common'] = modules
    sys.modules['attention'] = modules
    sys.modules['modules'] = modules
except Exception as e:
    st.error("Missing critical modules_new.py file")

from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

# 注册注意力机制
try:
    SE_class = getattr(modules, 'SE', getattr(modules, 'SEAttention', None))
    CBAM_class = getattr(modules, 'CBAM', getattr(modules, 'CBAMAttention', None))
    if SE_class:
        setattr(tasks, 'SE', SE_class)
    if CBAM_class:
        setattr(tasks, 'CBAM', CBAM_class)
except Exception:
    pass

@st.cache_resource
def load_yolo_model(model_name):
    path = os.path.join("models", model_name)
    if os.path.exists(path):
        dev = 0 if torch.cuda.is_available() else 'cpu'
        return YOLO(path).to(dev)
    return None

def run_feature_analysis(img_bgr, algo_type):
    h, w = img_bgr.shape[:2]
    canvas = img_bgr.copy()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if algo_type == "Algorithm 1":
        feat = cv2.SIFT_create(nfeatures=2000)
    else:
        feat = cv2.ORB_create(nfeatures=5000)
    kp, _ = feat.detectAndCompute(gray, None)
    if kp:
        canvas = cv2.drawKeypoints(canvas, kp, None, color=(0, 255, 0))
    return canvas, len(kp) if kp else 0

# 2. 界面设计
st.set_page_config(page_title="PCB defect detection", layout="wide")
st.title("PCB defect detection")

# 侧边栏配置
st.sidebar.header("Configuration Panel")
ds_select = st.sidebar.selectbox("Test Dataset", ["Dataset 1", "Dataset 2"])
ds_code = "ds1" if ds_select == "Dataset 1" else "ds2"

model_ui = st.sidebar.radio("Inspection Model", ["Model A", "Model B"])
model_code = "se" if model_ui == "Model A" else "cbam"
target_file = f"{ds_code}_{model_code}.pt"

algo_ui = st.sidebar.radio("Analysis Mode", ["Algorithm 1", "Algorithm 2"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.25)

tabs = st.tabs(["Single Image Inspection", "Export all dataset detection results"])

# Tab 1: 单图模式 (保持原样)
with tabs[0]:
    up_file = st.file_uploader("Upload PCB Image", type=["jpg", "png", "jpeg"])
    if up_file:
        raw_img = cv2.imdecode(np.asarray(bytearray(up_file.read()), dtype=np.uint8), 1)
        if st.button("Start Analysis", type="primary"):
            model = load_yolo_model(target_file)
            if model:
                res = model.predict(raw_img, conf=conf_threshold, device=0 if torch.cuda.is_available() else 'cpu')
                anno = res[0].plot()
                final, kp_n = run_feature_analysis(anno, algo_ui)
                st.image(cv2.cvtColor(final, cv2.COLOR_BGR2RGB), use_container_width=True)

# Tab 2: 批量导出模式 (防崩溃优化)
with tabs[1]:
    st.header("Export all dataset detection results")
    # 动态匹配路径 templates/dataset1 或 templates/dataset2
    data_path = os.path.join("templates", ds_select.lower().replace(" ", ""))
    
    if not os.path.exists(data_path):
        st.error(f"Directory {data_path} not found")
    else:
        images = [f for f in os.listdir(data_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        st.write(f"Detected **{len(images)}** images in {ds_select}.")

        if st.button("Execute Batch and Export (ZIP)", type="secondary"):
            export_dir = f"{ds_select}_Results_Export"
            if os.path.exists(export_dir): shutil.rmtree(export_dir)
            os.makedirs(export_dir)
            
            # 定义 4 种必须跑的组合
            combos = [
                ("Model A", f"{ds_code}_se.pt", "Algorithm 1"),
                ("Model A", f"{ds_code}_se.pt", "Algorithm 2"),
                ("Model B", f"{ds_code}_cbam.pt", "Algorithm 1"),
                ("Model B", f"{ds_code}_cbam.pt", "Algorithm 2")
            ]
            
            dev = 0 if torch.cuda.is_available() else 'cpu'
            prog = st.progress(0)
            step = 0
            total = len(combos) * len(images)
            
            for m_name, m_file, a_name in combos:
                l1_path = os.path.join(export_dir, f"{m_name} and {a_name}")
                model = load_yolo_model(m_file)
                
                for img_name in images:
                    # 三级目录结构
                    base = os.path.splitext(img_name)[0]
                    l2_path = os.path.join(l1_path, base)
                    os.makedirs(l2_path, exist_ok=True)
                    
                    # 读取与推理 (加速限制 imgsz)
                    img = cv2.imread(os.path.join(data_path, img_name))
                    if img is None: continue
                    
                    res = model.predict(img, conf=conf_threshold, device=dev, imgsz=640, verbose=False)
                    anno = res[0].plot()
                    final, kp_v = run_feature_analysis(anno, a_name)
                    
                    # 写入图片与CSV
                    cv2.imwrite(os.path.join(l2_path, f"{base}_marked.jpg"), final)
                    pd.DataFrame([{"Target": "Defects", "Value": len(res[0].boxes), "Algo": a_name}]).to_csv(os.path.join(l2_path, "report.csv"), index=False)
                    
                    # --- 内存主动释放 (核心防崩溃逻辑) ---
                    del res, anno, final, img
                    step += 1
                    if step % 20 == 0:
                        gc.collect()
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                    prog.progress(step / total)

            shutil.make_archive(export_dir, 'zip', export_dir)
            with open(f"{export_dir}.zip", "rb") as f:
                st.download_button("Download ZIP Package", f, file_name=f"{export_dir}.zip")
            st.success("Batch export complete.")

st.sidebar.markdown("---")
st.sidebar.caption(f"Ready: {ds_select} | GPU: {torch.cuda.is_available()}")
