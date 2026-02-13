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

# 状态管理初始化
if 'batch_status' not in st.session_state:
    st.session_state.batch_status = 'IDLE' # 状态: IDLE, RUNNING, PAUSED, STOPPED
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0

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
    if algo_type == "Algorithm 1":
        feat = cv2.SIFT_create(nfeatures=2000)
    else:
        feat = cv2.ORB_create(nfeatures=5000)
    kp, _ = feat.detectAndCompute(gray, None)
    if kp: canvas = cv2.drawKeypoints(canvas, kp, None, color=(0, 255, 0))
    return canvas, len(kp) if kp else 0

# 2. 界面布局
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

# Tab 1: 单图模式
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
                
                col_l, col_r = st.columns([3, 2])
                with col_l:
                    st.image(cv2.cvtColor(final, cv2.COLOR_BGR2RGB), use_container_width=True)
                with col_r:
                    st.subheader("Detection Summary")
                    report_data = []
                    counts = {}
                    for box in res[0].boxes:
                        label = res[0].names[int(box.cls[0])]
                        # --- 修改部分：提取坐标 ---
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        coord_str = f"[{xyxy[0]}, {xyxy[1]}, {xyxy[2]}, {xyxy[3]}]"
                        
                        # 插入坐标到第三列 (Target 和 Value 之间)
                        report_data.append([model_ui, label, coord_str, f"{float(box.conf[0]):.2%}", "Verified"])
                        counts[label] = counts.get(label, 0) + 1
                    
                    # 算法特征点行保持格式一致，坐标位填 "N/A"
                    report_data.append([algo_ui, "Feature Points", "N/A", f"{kp_n} pts", "Extracted"])
                    
                    # 更新 DataFrame 列名，加入 Coordinates
                    st.dataframe(pd.DataFrame(report_data, columns=["Method", "Target", "Coordinates", "Value", "Status"]), hide_index=True)
                    
                    st.subheader("Component Statistics")
                    if counts:
                        st.table(pd.DataFrame([{"Component": k, "Quantity": v} for k, v in counts.items()]))

# Tab 2: 批量导出模式 (含暂停功能)
with tabs[1]:
    st.header("Export all dataset detection results")
    data_path = os.path.join("templates", ds_select.lower().replace(" ", ""))
    
    if os.path.exists(data_path):
        imgs = [f for f in os.listdir(data_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        st.write(f"Detected {len(imgs)} images in {ds_select}")

        # 控制按钮
        c1, c2, c3 = st.columns(3)
        if st.session_state.batch_status in ['IDLE', 'STOPPED']:
            if c1.button("Execute Batch", type="primary"):
                st.session_state.batch_status = 'RUNNING'
                st.session_state.current_step = 0
                st.rerun()
        
        if st.session_state.batch_status == 'RUNNING':
            if c1.button("Pause"):
                st.session_state.batch_status = 'PAUSED'
                st.rerun()
            if c2.button("Stop"):
                st.session_state.batch_status = 'STOPPED'
                st.rerun()

        if st.session_state.batch_status == 'PAUSED':
            if c1.button("Resume"):
                st.session_state.batch_status = 'RUNNING'
                st.rerun()
            if c2.button("Stop"):
                st.session_state.batch_status = 'STOPPED'
                st.rerun()

        # 运行逻辑
        if st.session_state.batch_status == 'RUNNING':
            export_root = f"{ds_select}_Results"
            if st.session_state.current_step == 0:
                if os.path.exists(export_root): shutil.rmtree(export_root)
                os.makedirs(export_root)
            
            combos = [("Model A", f"{ds_code}_se.pt", "Algorithm 1"), 
                      ("Model A", f"{ds_code}_se.pt", "Algorithm 2"),
                      ("Model B", f"{ds_code}_cbam.pt", "Algorithm 1"), 
                      ("Model B", f"{ds_code}_cbam.pt", "Algorithm 2")]
            
            task_list = []
            for m_n, m_f, a_n in combos:
                for im in imgs: task_list.append((m_n, m_f, a_n, im))

            total_tasks = len(task_list)
            prog_bar = st.progress(st.session_state.current_step / total_tasks)
            status_text = st.empty()
            dev = 0 if torch.cuda.is_available() else 'cpu'

            while st.session_state.current_step < total_tasks and st.session_state.batch_status == 'RUNNING':
                m_n, m_f, a_n, im_name = task_list[st.session_state.current_step]
                status_text.text(f"Processing: {im_name} ({m_n} + {a_n})")
                
                base = os.path.splitext(im_name)[0]
                save_dir = os.path.join(export_root, f"{m_n} and {a_n}", base)
                os.makedirs(save_dir, exist_ok=True)
                
                model = load_yolo_model(m_f)
                img = cv2.imread(os.path.join(data_path, im_name))
                res = model.predict(img, conf=conf_threshold, device=dev, imgsz=640, verbose=False)
                anno = res[0].plot()
                final, kp_v = run_feature_analysis(anno, a_n)
                
                cv2.imwrite(os.path.join(save_dir, f"{base}_marked.jpg"), final)
                
                # 生成报告
                csv_data = []
                b_counts = {}
                for box in res[0].boxes:
                    lbl = res[0].names[int(box.cls[0])]
                    csv_data.append({"Type": "Detail", "Target": lbl, "Value": f"{float(box.conf[0]):.2%}"})
                    b_counts[lbl] = b_counts.get(lbl, 0) + 1
                for k, v in b_counts.items():
                    csv_data.append({"Type": "Summary", "Target": k, "Value": str(v)})
                pd.DataFrame(csv_data).to_csv(os.path.join(save_dir, "report.csv"), index=False)
                
                del img, res, final
                st.session_state.current_step += 1
                prog_bar.progress(st.session_state.current_step / total_tasks)
                
                if st.session_state.current_step % 15 == 0:
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                
                if st.session_state.batch_status != 'RUNNING': st.rerun()

            if st.session_state.current_step >= total_tasks:
                shutil.make_archive(export_root, 'zip', export_root)
                st.session_state.batch_status = 'IDLE'
                st.success("Batch processing complete")
                with open(f"{export_root}.zip", "rb") as f:
                    st.download_button("Download ZIP", f, file_name=f"{export_root}.zip")

        elif st.session_state.batch_status == 'PAUSED':
            st.warning(f"Paused at step {st.session_state.current_step} of {len(imgs)*4}")
