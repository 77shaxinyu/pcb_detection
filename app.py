import streamlit as st
import cv2
import numpy as np
import os
import sys
import pandas as pd
import shutil

# --- 1. 环境初始化与模块映射 ---
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
st.title("PCB defect detection")
st.markdown("---")

# 侧边栏配置：决定了当前处理哪个 Dataset
st.sidebar.header("Configuration Panel")
ds_select = st.sidebar.selectbox("Test Dataset", ["Dataset 1", "Dataset 2"])
ds_code = "ds1" if ds_select == "Dataset 1" else "ds2"

# 侧边栏的单图模型选择
model_ui = st.sidebar.radio("Inspection Model", ["Model A", "Model B"])
model_code = "se" if model_ui == "Model A" else "cbam"
target_file = f"{ds_code}_{model_code}.pt"

algo_ui = st.sidebar.radio("Analysis Mode", ["Algorithm 1", "Algorithm 2"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.25)

tabs = st.tabs(["Single Image Inspection", "Export all dataset detection results"])

# --- Tab 1: 单图检测 (保持原样) ---
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
                    st.subheader("Detection Summary")
                    data = []
                    for box in res[0].boxes:
                        label = res[0].names[int(box.cls[0])]
                        data.append([model_ui, label, f"{float(box.conf[0]):.2%}", "Verified"])
                    data.append([algo_ui, "Feature Points", f"{total_kp} pts", "Extracted"])
                    st.dataframe(pd.DataFrame(data, columns=["Method", "Target", "Value", "Status"]), hide_index=True)

# --- Tab 2: 批量导出 (严格匹配 templates 目录与 ds 模型) ---
with tabs[1]:
    st.header("Export all dataset detection results")
    st.write(f"This will process all images in **templates/{ds_select.lower().replace(' ', '')}** using **{ds_select} models**.")

    # 动态构建图片路径：templates/dataset1 或 templates/dataset2
    data_path = os.path.join("templates", ds_select.lower().replace(" ", ""))
    
    if not os.path.exists(data_path):
        st.error(f"Directory not found: {data_path}")
    else:
        all_images = [f for f in os.listdir(data_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        st.write(f"Detected **{len(all_images)}** images for processing.")

        if st.button("Execute Batch and Export (ZIP)", type="secondary"):
            export_root = f"{ds_select}_Full_Report"
            if os.path.exists(export_root): shutil.rmtree(export_root)
            os.makedirs(export_root)
            
            # 定义该 Dataset 对应的 4 种合法组合（Model A/B x Algo 1/2）
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
                # 1级目录：Model A and Algorithm 1
                l1_path = os.path.join(export_root, f"{m_ui} and {a_ui}")
                model = get_yolo_model(m_file)
                
                for img_file in all_images:
                    # 2级目录：图片名
                    img_base = os.path.splitext(img_file)[0]
                    l2_path = os.path.join(l1_path, img_base)
                    os.makedirs(l2_path, exist_ok=True)
                    
                    # 读取图片并推理
                    img = cv2.imread(os.path.join(data_path, img_file))
                    if img is None: continue
                    
                    res = model.predict(img, conf=conf_threshold, verbose=False)
                    anno = res[0].plot()
                    final, kp = run_feature_analysis(anno, a_ui)
                    
                    # 3级文件：标注图与详细CSV
                    cv2.imwrite(os.path.join(l2_path, f"{img_base}_marked.jpg"), final)
                    
                    csv_rows = []
                    for box in res[0].boxes:
                        csv_rows.append({
                            "Model": m_ui,
                            "Target": res[0].names[int(box.cls[0])],
                            "Value": f"{float(box.conf[0]):.2%}",
                            "Algorithm": a_ui,
                            "File": img_file
                        })
                    csv_rows.append({"Model": a_ui, "Target": "Feature Points", "Value": f"{kp}", "Algorithm": m_ui, "File": img_file})
                    pd.DataFrame(csv_rows).to_csv(os.path.join(l2_path, f"{img_base}_analysis.csv"), index=False)
                    
                    step += 1
                    prog.progress(step / total)

            shutil.make_archive(export_root, 'zip', export_root)
            with open(f"{export_root}.zip", "rb") as f:
                st.download_button(f"Download {ds_select} Results (ZIP)", f, file_name=f"{export_root}.zip")
            st.success("Batch Processing Finished Successfully.")

st.sidebar.markdown("---")
st.sidebar.caption(f"Backend Ready | Loaded: {target_file}")
