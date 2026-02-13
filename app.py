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
    pass

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

# --- Tab 1: 单图模式 (添加计数表格) ---
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
                
                col_left, col_right = st.columns([3, 2])
                
                with col_left:
                    st.image(cv2.cvtColor(final, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                with col_right:
                    # 表格 1：详情
                    st.subheader("Detection Summary")
                    report_data = []
                    counts = {} # 用于存储统计数据
                    for box in res[0].boxes:
                        label = res[0].names[int(box.cls[0])]
                        conf_val = f"{float(box.conf[0]):.2%}"
                        report_data.append([model_ui, label, conf_val, "Verified"])
                        # 统计数量
                        counts[label] = counts.get(label, 0) + 1
                    
                    report_data.append([algo_ui, "Feature Points", f"{kp_n} pts", "Extracted"])
                    st.dataframe(pd.DataFrame(report_data, columns=["Method", "Target", "Value", "Status"]), hide_index=True)
                    
                    # 表格 2：计数汇总
                    st.subheader("Component Statistics")
                    if counts:
                        count_df = pd.DataFrame([{"Component": k, "Quantity": v} for k, v in counts.items()])
                        st.table(count_df) # 使用 st.table 显示静态统计表
                    else:
                        st.write("No defects detected.")

# --- Tab 2: 批量导出 (同步更新 CSV 内容) ---
with tabs[1]:
    st.header("Export all dataset detection results")
    data_path = os.path.join("templates", ds_select.lower().replace(" ", ""))
    if os.path.exists(data_path):
        imgs = [f for f in os.listdir(data_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        st.write(f"Detected **{len(imgs)}** images.")
        
        if st.button("Execute Batch and Export (ZIP)", type="secondary"):
            export_root = f"{ds_select}_Results"
            if os.path.exists(export_root): shutil.rmtree(export_root)
            os.makedirs(export_root)
            
            combos = [("Model A", f"{ds_code}_se.pt", "Algorithm 1"), 
                      ("Model A", f"{ds_code}_se.pt", "Algorithm 2"),
                      ("Model B", f"{ds_code}_cbam.pt", "Algorithm 1"), 
                      ("Model B", f"{ds_code}_cbam.pt", "Algorithm 2")]
            
            prog = st.progress(0)
            step = 0
            total = len(combos) * len(imgs)
            dev = 0 if torch.cuda.is_available() else 'cpu'
            
            for m_name, m_file, a_name in combos:
                model = load_yolo_model(m_file)
                l1_dir = os.path.join(export_root, f"{m_name} and {a_name}")
                
                for img_name in imgs:
                    base = os.path.splitext(img_name)[0]
                    target_dir = os.path.join(l1_dir, base)
                    os.makedirs(target_dir, exist_ok=True)
                    
                    img = cv2.imread(os.path.join(data_path, img_name))
                    res = model.predict(img, conf=conf_threshold, device=dev, imgsz=640, verbose=False)
                    anno = res[0].plot()
                    final, kp_val = run_feature_analysis(anno, a_name)
                    
                    # 保存图片
                    cv2.imwrite(os.path.join(target_dir, f"{base}_marked.jpg"), final)
                    
                    # 导出详细 CSV (含统计)
                    csv_data = []
                    batch_counts = {}
                    for box in res[0].boxes:
                        lbl = res[0].names[int(box.cls[0])]
                        csv_data.append({"Type": "Detail", "Target": lbl, "Value": f"{float(box.conf[0]):.2%}"})
                        batch_counts[lbl] = batch_counts.get(lbl, 0) + 1
                    
                    for k, v in batch_counts.items():
                        csv_data.append({"Type": "Summary", "Target": k, "Value": str(v)})
                    
                    pd.DataFrame(csv_data).to_csv(os.path.join(target_dir, "report.csv"), index=False)
                    
                    del img, res, final
                    step += 1
                    if step % 20 == 0:
                        gc.collect()
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                    prog.progress(step / total)

            shutil.make_archive(export_root, 'zip', export_root)
            with open(f"{export_root}.zip", "rb") as f:
                st.download_button("Download ZIP", f, file_name=f"{export_root}.zip")
            st.success("Export finished.")
