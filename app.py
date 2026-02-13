import streamlit as st
import cv2
import numpy as np
import os
import sys
import pandas as pd
import shutil

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

# --- 2. 界面布局 ---
st.title("PCB defect detection") # 修改后的标题
st.markdown("---")

# 侧边栏
st.sidebar.header("Configuration Panel")
ds_select = st.sidebar.selectbox("Test Dataset", ["Dataset 1", "Dataset 2"])
ds_code = "ds1" if ds_select == "Dataset 1" else "ds2"

model_ui = st.sidebar.radio("Inspection Model", ["Model A", "Model B"])
model_code = "se" if model_ui == "Model A" else "cbam"
target_file = f"{ds_code}_{model_code}.pt"

algo_ui = st.sidebar.radio("Analysis Mode", ["Algorithm 1", "Algorithm 2"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.25)

tabs = st.tabs(["Single Image Inspection", "Export all dataset detection results"]) # 修改后的模式名

# --- Tab 1: 单张检测 (保留) ---
with tabs[0]:
    uploaded_file = st.file_uploader("Upload Single PCB Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        base_name = os.path.splitext(uploaded_file.name)[0]
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        raw_img = cv2.imdecode(file_bytes, 1)

        if st.button("Start Analysis", type="primary"):
            yolo_model = get_yolo_model(target_file)
            if yolo_model:
                res = yolo_model.predict(raw_img, conf=conf_threshold)
                render_img = res[0].plot()
                final_img, total_kp = run_feature_analysis(render_img, algo_ui)
                
                col1, col2 = st.columns([3, 2])
                with col1:
                    st.image(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                with col2:
                    st.subheader("Summary")
                    data = []
                    for box in res[0].boxes:
                        label = res[0].names[int(box.cls[0])]
                        data.append([model_ui, label, f"{float(box.conf[0]):.2%}", "Verified"])
                    data.append([algo_ui, "Feature Points", f"{total_kp} pts", "Extracted"])
                    st.dataframe(pd.DataFrame(data, columns=["Method", "Target", "Value", "Status"]), hide_index=True)

# --- Tab 2: 批量导出 (全新精简版) ---
with tabs[1]:
    st.header("Export all dataset detection results")
    st.write("This action will process the entire dataset using all model and algorithm combinations.")
    
    # 设定你服务器上存放 400 张图片的文件夹路径
    # 假设路径是项目根目录下的 'data' 文件夹
    data_path = "data" 
    
    if not os.path.exists(data_path):
        st.warning(f"Please ensure the '{data_path}' folder exists and contains your images.")
    else:
        all_images = [f for f in os.listdir(data_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        st.write(f"Detected **{len(all_images)}** images in '{data_path}' folder.")

        if st.button("Execute and Export (ZIP)", type="secondary"):
            export_root = "PCB_All_Results"
            if os.path.exists(export_root): shutil.rmtree(export_root)
            os.makedirs(export_root)
            
            # 自动跑遍 4 种组合
            combos = [
                ("Model A", f"{ds_code}_se.pt", "Algorithm 1"),
                ("Model A", f"{ds_code}_se.pt", "Algorithm 2"),
                ("Model B", f"{ds_code}_cbam.pt", "Algorithm 1"),
                ("Model B", f"{ds_code}_cbam.pt", "Algorithm 2")
            ]
            
            prog = st.progress(0)
            total = len(combos) * len(all_images)
            step = 0
            
            for m_ui, m_file, a_ui in combos:
                # 1级：组合名
                l1_path = os.path.join(export_root, f"{m_ui} and {a_ui}")
                model = get_yolo_model(m_file)
                
                for img_file in all_images:
                    # 2级：文件名
                    img_base = os.path.splitext(img_file)[0]
                    l2_path = os.path.join(l1_path, img_base)
                    os.makedirs(l2_path, exist_ok=True)
                    
                    # 读取并推理
                    img = cv2.imread(os.path.join(data_path, img_file))
                    res = model.predict(img, conf=conf_threshold, verbose=False)
                    anno = res[0].plot()
                    final, kp = run_feature_analysis(anno, a_ui)
                    
                    # 3级：文件
                    cv2.imwrite(os.path.join(l2_path, f"{img_base}_marked.jpg"), final)
                    # 保存对应的 CSV 数据
                    csv_data = [{"File": img_file, "Model": m_ui, "Algo": a_ui, "KP": kp}]
                    pd.DataFrame(csv_data).to_csv(os.path.join(l2_path, f"{img_base}_info.csv"), index=False)
                    
                    step += 1
                    prog.progress(step / total)

            shutil.make_archive("PCB_Detection_Package", 'zip', export_root)
            with open("PCB_Detection_Package.zip", "rb") as f:
                st.download_button("Download Complete ZIP", f, file_name="PCB_Detection_Package.zip")
            st.success("Batch processing complete.")

st.sidebar.markdown("---")
st.sidebar.caption(f"Ready | {ds_code}")
