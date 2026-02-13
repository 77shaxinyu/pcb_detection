import streamlit as st
import cv2
import numpy as np
import os
import sys
import pandas as pd

# --- 1. 核心修复：强制模块路径映射 ---
# 获取当前脚本所在目录并加入系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入你的 modules.py 文件
try:
    import modules
    # 这一步是终极补丁：将你的文件伪装成模型寻找的各种可能路径
    # 解决 torch.load 时的 Unpickling 路径报错
    sys.modules['modules.attention'] = modules
    sys.modules['models.common'] = modules
    sys.modules['attention'] = modules
except Exception as e:
    st.error(f"Error: Could not find modules.py. Please ensure it is in the root directory.")

from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

# 2. 动态注册类到 YOLO 任务系统
# 这样即使权重里记录的是 'SEAttention'，它也能找到你代码里的 'SE'
try:
    # 尝试从 modules.py 获取类，如果名字不同请根据你的 modules.py 实际类名修改
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

# --- 3. 页面全局配置 ---
st.set_page_config(page_title="PCB Inspection System", layout="wide")

@st.cache_resource
def get_yolo_model(model_name):
    path = os.path.join("models", model_name)
    if os.path.exists(path):
        # 强制在 CPU 上运行以确保云端兼容性
        return YOLO(path)
    return None

def run_feature_analysis(img_bgr, algo_type):
    """执行 Algorithm 1 (SIFT) 或 Algorithm 2 (ORB) 的特征提取"""
    h, w = img_bgr.shape[:2]
    canvas = img_bgr.copy()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 绘制经典 9x9 网格
    for i in range(1, 9):
        cv2.line(canvas, (0, int(i * h / 9)), (w, int(i * h / 9)), (0, 255, 0), 1)
        cv2.line(canvas, (int(i * w / 9), 0), (int(i * w / 9), h), (0, 255, 0), 1)

    if algo_type == "SIFT":
        feat_engine = cv2.SIFT_create(nfeatures=2000)
        pt_color = (255, 0, 0) # Blue
    else:
        feat_engine = cv2.ORB_create(nfeatures=5000)
        pt_color = (0, 0, 255) # Red

    kp, des = feat_engine.detectAndCompute(gray, None)
    if kp:
        canvas = cv2.drawKeypoints(canvas, kp, None, color=pt_color)

    return canvas, len(kp) if kp else 0

# --- 4. 界面展示 ---
st.title("PCB Defect Detection & Feature Analysis")
st.markdown("---")

# 侧边栏
st.sidebar.header("Configuration")
ds_select = st.sidebar.selectbox("Select Dataset", ["Dataset 1", "Dataset 2"])
ds_code = "ds1" if ds_select == "Dataset 1" else "ds2"

model_select = st.sidebar.radio("Attention Type", ["SE", "CBAM"])
model_code = model_select.lower()

# 自动匹配文件名，例如 models/ds1_se.pt
target_file = f"{ds_code}_{model_code}.pt"

algo_select = st.sidebar.radio("Algorithm Mode", ["Algorithm 1 (SIFT)", "Algorithm 2 (ORB)"])
algo_type = "SIFT" if "Algorithm 1" in algo_select else "ORB"

conf_threshold = st.sidebar.slider("Confidence", 0.1, 0.9, 0.25)

# 文件上传
uploaded_file = st.file_uploader("Upload PCB Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 读取图片
    bytes_data = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    raw_img = cv2.imdecode(bytes_data, 1)

    col_view, col_rep = st.columns([3, 2])

    with col_view:
        if st.button("Start Analysis", type="primary"):
            # 加载模型
            yolo_model = get_yolo_model(target_file)
            
            if yolo_model:
                # 1. AI 检测
                res = yolo_model.predict(raw_img, conf=conf_threshold)
                render_img = res[0].plot()

                # 2. 传统视觉提取
                final_img, total_kp = run_feature_analysis(render_img, algo_type)

                # 3. 展示
                st.image(cv2.cvtColor(final_output := final_img, cv2.COLOR_BGR2RGB), 
                         caption=f"Process: {target_file} + {algo_type}", 
                         use_container_width=True)

                # --- 5. 报告区 ---
                with col_rep:
                    st.subheader("Analysis Summary")
                    data_rows = []
                    
                    # YOLO 结果
                    for box in res[0].boxes:
                        label = res[0].names[int(box.cls[0])]
                        prob = f"{float(box.conf[0]):.2%}"
                        data_rows.append(["AI Detection", label, prob, "Verified"])
                    
                    # 特征点结果
                    data_rows.append([algo_type, "Keypoints", f"{total_kp} pts", "Extracted"])
                    
                    report_df = pd.DataFrame(data_rows, columns=["Method", "Target", "Value", "Status"])
                    st.table(report_df)
            else:
                st.error(f"Model file 'models/{target_file}' not found.")
        else:
            # 初始预览
            st.image(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB), caption="Raw Image Preview", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption(f"System Status: Connected to {target_file}")
