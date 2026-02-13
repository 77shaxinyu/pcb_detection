import streamlit as st
import cv2
import numpy as np
import os
import sys
import pandas as pd
import shutil

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

st.set_page_config(page_title="PCB Inspection System", layout="wide")

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
    for i in range(1, 9):
        cv2.line(canvas, (0, int(i * h / 9)), (w, int(i * h / 9)), (0, 255, 0), 1)
        cv2.line(canvas, (int(i * w / 9), 0), (int(i * w / 9), h), (0, 255, 0), 1)

    if algo_type == "Algorithm 1":
        feat_engine = cv2.SIFT_create(nfeatures=2000)
        pt_color = (255, 0, 0)
    else:
        feat_engine = cv2.ORB_create(nfeatures=5000)
        pt_color = (0, 0, 255)

    kp, des = feat_engine.detectAndCompute(gray, None)
    if kp:
        canvas = cv2.drawKeypoints(canvas, kp, None, color=pt_color)
    return canvas, len(kp) if kp else 0

# --- UI Layout ---
st.title("PCB Intelligent Inspection Platform")
st.markdown("---")

st.sidebar.header("Configuration Panel")
ds_select = st.sidebar.selectbox("Test Dataset", ["Dataset 1", "Dataset 2"])
ds_code = "ds1" if ds_select == "Dataset 1" else "ds2"

model_ui = st.sidebar.radio("Inspection Model", ["Model A", "Model B"])
model_code = "se" if model_ui == "Model A" else "cbam"

target_file = f"{ds_code}_{model_code}.pt"
algo_ui = st.sidebar.radio("Analysis Mode", ["Algorithm 1", "Algorithm 2"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.25)

uploaded_file = st.file_uploader("Upload PCB Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 获取原始文件名（不含后缀）
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

                st.image(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB), 
                         caption=f"Process: {model_ui} and {algo_ui}", use_container_width=True)

                # Data processing
                data_rows = []
                class_counts = {}
                for box in res[0].boxes:
                    label = res[0].names[int(box.cls[0])]
                    prob = f"{float(box.conf[0]):.2%}"
                    # 将模型和算法信息直接写入每一行数据
                    data_rows.append([model_ui, label, prob, "Verified", base_file_name, algo_ui])
                    class_counts[label] = class_counts.get(label, 0) + 1
                
                data_rows.append([algo_ui, "Feature Points", f"{total_kp} pts", "Extracted", base_file_name, model_ui])
                
                # 创建带溯源信息的报表
                report_df = pd.DataFrame(data_rows, columns=["Method", "Target", "Value", "Status", "Original_File", "Mode_Info"])

                with col_rep:
                    st.subheader("Detection Summary")
                    st.dataframe(report_df, hide_index=True, use_container_width=True)
                    
                    st.subheader("Component Statistics")
                    if class_counts:
                        stats_df = pd.DataFrame([{"Component": k, "Quantity": v} for k, v in class_counts.items()])
                        st.table(stats_df)

                    # --- EXPORT LOGIC: Strict Naming Convention ---
                    st.markdown("---")
                    # 1. 文件夹名称: Model A and Algorithm 1
                    folder_name = f"{model_ui} and {algo_ui}"
                    if not os.path.exists(folder_name):
                        os.makedirs(folder_name)
                    
                    # 2. 图片名称: [原图名]_[模型]_[算法].jpg
                    clean_model_name = model_ui.replace(" ", "")
                    clean_algo_name = algo_ui.replace(" ", "")
                    out_img_name = f"{base_file_name}_{clean_model_name}_{clean_algo_name}.jpg"
                    cv2.imwrite(os.path.join(folder_name, out_img_name), final_img)
                    
                    # 3. CSV名称: [原图名]_[模型]_[算法].csv
                    out_csv_name = f"{base_file_name}_{clean_model_name}_{clean_algo_name}.csv"
                    report_df.to_csv(os.path.join(folder_name, out_csv_name), index=False)
                    
                    # 打包 ZIP
                    shutil.make_archive(folder_name, 'zip', folder_name)
                    
                    st.info(f"Exported to: {folder_name}")
                    with open(f"{folder_name}.zip", "rb") as fp:
                        st.download_button(
                            label="Download Output Pack",
                            data=fp,
                            file_name=f"{folder_name}.zip",
                            mime="application/zip"
                        )
                    st.success("Files named and saved successfully")
            else:
                st.error("Model Loading Failed")
        else:
            st.image(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB), caption="Waiting for Input", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption(f"Backend Ready: {ds_code} and {model_code}")
