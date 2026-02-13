import streamlit as st
import cv2
import numpy as np
import os
import sys
import pandas as pd
import shutil
import tempfile

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

# --- 2. 核心算法逻辑 ---
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
    
    # 9x9 参考网格
    for i in range(1, 9):
        cv2.line(canvas, (0, int(i * h / 9)), (w, int(i * h / 9)), (0, 255, 0), 1)
        cv2.line(canvas, (int(i * w / 9), 0), (int(i * w / 9), h), (0, 255, 0), 1)

    if algo_type == "Algorithm 1":
        feat_engine = cv2.SIFT_create(nfeatures=2000)
        pt_color = (255, 0, 0)
    else:
        feat_engine = cv2.ORB_create(nfeatures=5000)
        pt_color = (0, 0, 255)

    kp, _ = feat_engine.detectAndCompute(gray, None)
    if kp:
        canvas = cv2.drawKeypoints(canvas, kp, None, color=pt_color)
    return canvas, len(kp) if kp else 0

# --- 3. UI 界面 ---
st.title("PCB Intelligent Inspection Platform")
st.markdown("---")

# 侧边栏保持原样：用于单张图检测配置
st.sidebar.header("Mode 1: Single Image Configuration")
ds_select = st.sidebar.selectbox("Test Dataset", ["Dataset 1", "Dataset 2"])
ds_code = "ds1" if ds_select == "Dataset 1" else "ds2"

model_ui = st.sidebar.radio("Inspection Model", ["Model A", "Model B"])
model_code = "se" if model_ui == "Model A" else "cbam"
target_file = f"{ds_code}_{model_code}.pt"

algo_ui = st.sidebar.radio("Analysis Mode", ["Algorithm 1", "Algorithm 2"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.25)

# --- 功能模式切换 ---
tabs = st.tabs(["Single Image Inspection", "Batch Processing (400+ Images)"])

# --- 选项卡 1: 原有的单图上传检测功能 (完全保留) ---
with tabs[0]:
    uploaded_file = st.file_uploader("Upload Single PCB Image", type=["jpg", "png", "jpeg"], key="single")
    if uploaded_file:
        base_file_name = os.path.splitext(uploaded_file.name)[0]
        bytes_data = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        raw_img = cv2.imdecode(bytes_data, 1)

        col_view, col_rep = st.columns([3, 2])
        with col_view:
            if st.button("Start Analysis", type="primary"):
                yolo_model = get_yolo_model(target_file)
                if yolo_model:
                    res = yolo_model.predict(raw_img, conf=conf_threshold)
                    render_img = res[0].plot()
                    final_img, total_kp = run_feature_analysis(render_img, algo_ui)
                    st.image(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB), caption=f"{model_ui} + {algo_ui}", use_container_width=True)

                    # 报告数据记录
                    data_rows = []
                    class_counts = {}
                    for box in res[0].boxes:
                        label = res[0].names[int(box.cls[0])]
                        data_rows.append([model_ui, label, f"{float(box.conf[0]):.2%}", "Verified"])
                        class_counts[label] = class_counts.get(label, 0) + 1
                    data_rows.append([algo_ui, "Feature Points", f"{total_kp} pts", "Extracted"])
                    report_df = pd.DataFrame(data_rows, columns=["Method", "Target", "Value", "Status"])

                    with col_rep:
                        st.subheader("Summary")
                        st.dataframe(report_df, hide_index=True)
                        st.subheader("Counts")
                        st.table(pd.DataFrame([{"Component": k, "Quantity": v} for k, v in class_counts.items()]))
                        
                        # 单张导出的三级目录逻辑
                        folder_name = f"{model_ui} and {algo_ui}"
                        img_dir = os.path.join(folder_name, base_file_name)
                        os.makedirs(img_dir, exist_ok=True)
                        cv2.imwrite(os.path.join(img_dir, f"{base_file_name}_annotated.jpg"), final_img)
                        report_df.to_csv(os.path.join(img_dir, f"{base_file_name}_report.csv"), index=False)
                        st.success(f"Archived in {folder_name}/{base_file_name}")

# --- 选项卡 2: 新增的批量检测功能 ---
with tabs[1]:
    st.header("Batch Cross-Analysis Mode")
    st.write("This mode will process all images using ALL 4 combinations (Model A/B x Algo 1/2).")
    batch_files = st.file_uploader("Upload all 400+ images", type=["jpg", "png", "jpeg"], accept_multiple_files=True, key="batch")
    
    if batch_files:
        if st.button("Execute Full Batch & Export 3-Level Zip", type="secondary"):
            export_root = "Full_Batch_Results"
            if os.path.exists(export_root): shutil.rmtree(export_root)
            os.makedirs(export_root)
            
            # 定义 4 种必须跑的组合
            combos = [
                ("Model A", f"{ds_code}_se.pt", "Algorithm 1"),
                ("Model A", f"{ds_code}_se.pt", "Algorithm 2"),
                ("Model B", f"{ds_code}_cbam.pt", "Algorithm 1"),
                ("Model B", f"{ds_code}_cbam.pt", "Algorithm 2")
            ]
            
            progress = st.progress(0)
            total = len(combos) * len(batch_files)
            count = 0
            
            for m_ui, m_file, a_ui in combos:
                # 第一级：模型与算法
                l1_path = os.path.join(export_root, f"{m_ui} and {a_ui}")
                model = get_yolo_model(m_file)
                
                for b_file in batch_files:
                    # 第二级：图片名称
                    img_name = os.path.splitext(b_file.name)[0]
                    l2_path = os.path.join(l1_path, img_name)
                    os.makedirs(l2_path, exist_ok=True)
                    
                    # 运行处理
                    f_bytes = np.asarray(bytearray(b_file.read()), dtype=np.uint8)
                    b_file.seek(0)
                    img = cv2.imdecode(f_bytes, 1)
                    
                    res = model.predict(img, conf=conf_threshold, verbose=False)
                    anno = res[0].plot()
                    final, kp_num = run_feature_analysis(anno, a_ui)
                    
                    # 第三级：文件输出
                    cv2.imwrite(os.path.join(l2_path, f"{img_name}_annotated.jpg"), final)
                    pd.DataFrame([{"File": b_file.name, "Model": m_ui, "Algo": a_ui, "KP": kp_num}]).to_csv(os.path.join(l2_path, "data.csv"), index=False)
                    
                    count += 1
                    progress.progress(count / total)

            shutil.make_archive("Full_Export", 'zip', export_root)
            with open("Full_Export.zip", "rb") as f:
                st.download_button("Download All Combinations (ZIP)", f, file_name="Full_Export.zip")
            st.success("Batch Processing Finished.")

st.sidebar.markdown("---")
st.sidebar.caption(f"System Ready | Mode: {ds_code}")
