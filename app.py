import streamlit as st
import cv2
import numpy as np
import os
import sys
import pandas as pd
import shutil
import tempfile

# --- 1. System Path and Module Mapping ---
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
    st.error("System Error: Critical modules missing")

from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

try:
    SE_class = getattr(modules, 'SE', getattr(modules, 'SEAttention', None))
    CBAM_class = getattr(modules, 'CBAM', getattr(modules, 'CBAMAttention', None))
    if SE_class:
        setattr(tasks, 'SE', SE_class)
        setattr(tasks, 'SEAttention', SE_class)
    if CBAM_class:
        setattr(tasks, 'CBAM', CBAM_class)
        setattr(tasks, 'CBAMAttention', CBAM_class)
except Exception as e:
    pass

@st.cache_resource
def get_yolo_model(model_name):
    path = os.path.join("models", model_name)
    if os.path.exists(path):
        return YOLO(path)
    return None

def run_feature_analysis(img_bgr, algo_type):
    h, w = img_bgr.shape[:2]
    canvas = img_bgr.copy()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if algo_type == "Algorithm 1":
        feat_engine = cv2.SIFT_create(nfeatures=2000)
    else:
        feat_engine = cv2.ORB_create(nfeatures=5000)
    kp, _ = feat_engine.detectAndCompute(gray, None)
    if kp:
        canvas = cv2.drawKeypoints(canvas, kp, None, color=(0, 255, 0))
    return canvas, len(kp) if kp else 0

# --- UI Layout ---
st.title("PCB Batch Inspection System")
st.markdown("---")

# 支持多文件上传
uploaded_files = st.file_uploader("Upload Batch PCB Images (up to 500)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    st.info(f"Loaded {len(uploaded_files)} images. Ready for batch processing.")
    
    if st.button("Generate Full Cross-Analysis Report", type="primary"):
        # 创建临时导出根目录
        export_root = "Batch_Export_Results"
        if os.path.exists(export_root):
            shutil.rmtree(export_root)
        os.makedirs(export_root)
        
        combinations = [
            ("Model A", "ds1_se.pt", "Algorithm 1"),
            ("Model A", "ds1_se.pt", "Algorithm 2"),
            ("Model B", "ds1_cbam.pt", "Algorithm 1"),
            ("Model B", "ds1_cbam.pt", "Algorithm 2")
        ]
        
        progress_bar = st.progress(0)
        total_steps = len(combinations) * len(uploaded_files)
        current_step = 0
        
        for m_name, m_file, a_name in combinations:
            # 第一级目录：Model A and Algorithm 1
            combo_path = os.path.join(export_root, f"{m_name} and {a_name}")
            os.makedirs(combo_path, exist_ok=True)
            
            model = get_yolo_model(m_file)
            
            for up_file in uploaded_files:
                # 获取文件名（二级目录名）
                base_name = os.path.splitext(up_file.name)[0]
                img_dir = os.path.join(combo_path, base_name)
                os.makedirs(img_dir, exist_ok=True)
                
                # 处理图片
                file_bytes = np.asarray(bytearray(up_file.read()), dtype=np.uint8)
                up_file.seek(0) # 重置文件指针供下次读取
                img = cv2.imdecode(file_bytes, 1)
                
                # 推理与标注
                res = model.predict(img, conf=0.25, verbose=False)
                anno_img = res[0].plot()
                final_img, total_kp = run_feature_analysis(anno_img, a_name)
                
                # 保存三级文件：标注图片
                img_save_name = f"{base_name}_{m_name.replace(' ','')}_{a_name.replace(' ','')}.jpg"
                cv2.imwrite(os.path.join(img_dir, img_save_name), final_img)
                
                # 生成并保存三级文件：CSV
                data = []
                for box in res[0].boxes:
                    data.append({
                        "Method": m_name,
                        "Target": res[0].names[int(box.cls[0])],
                        "Confidence": f"{float(box.conf[0]):.2%}",
                        "Original_File": up_file.name,
                        "Algorithm_Info": a_name
                    })
                data.append({"Method": a_name, "Target": "Keypoints", "Confidence": f"{total_kp}", "Original_File": up_file.name, "Algorithm_Info": m_name})
                
                csv_save_name = f"{base_name}_report.csv"
                pd.DataFrame(data).to_csv(os.path.join(img_dir, csv_save_name), index=False)
                
                current_step += 1
                progress_bar.progress(current_step / total_steps)

        # 打包下载
        shutil.make_archive("Full_Inspection_Export", 'zip', export_root)
        with open("Full_Inspection_Export.zip", "rb") as f:
            st.download_button("Download Full Batch Pack (ZIP)", f, file_name="Full_Inspection_Export.zip")
        st.success("Batch processing completed! All combinations exported.")
