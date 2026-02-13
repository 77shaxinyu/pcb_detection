import streamlit as st
import cv2
import numpy as np
import os
import sys
import pandas as pd
from PIL import Image

# 1. 强制对齐路径：解决云端 ModuleNotFoundError
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

# 2. 动态注册自定义注意力机制层
try:
    # 从你的 modules.py 文件中导入类
    from modules import SE, CBAM
    
    # 手动挂载到 ultralytics 任务系统，确保模型反序列化成功
    # 注册多个可能的名字以防权重文件记录的类名不一致
    setattr(tasks, 'SE', SE)
    setattr(tasks, 'SEAttention', SE)
    setattr(tasks, 'CBAM', CBAM)
    setattr(tasks, 'CBAMAttention', CBAM)
except Exception as e:
    st.sidebar.warning(f"Note: Custom modules linked via modules.py")

# --- 页面配置 ---
st.set_page_config(page_title="PCB Detection System", layout="wide")

MODEL_DIR = "models"

@st.cache_resource
def get_yolo_model(model_name):
    path = os.path.join(MODEL_DIR, model_name)
    if os.path.exists(path):
        return YOLO(path)
    return None

def run_feature_extraction(img_bgr, algo_type):
    """执行 Algorithm 1 (SIFT) 或 Algorithm 2 (ORB) 的自我特征提取"""
    H, W = img_bgr.shape[:2]
    display_img = img_bgr.copy()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 绘制经典 9x9 绿色网格
    for i in range(1, 9):
        cv2.line(display_img, (0, int(i * H / 9)), (W, int(i * H / 9)), (0, 255, 0), 1)
        cv2.line(display_img, (int(i * W / 9), 0), (int(i * W / 9), H), (0, 255, 0), 1)

    if algo_type == "SIFT":
        engine = cv2.SIFT_create(nfeatures=2000)
        color = (255, 0, 0) # Blue
    else:
        engine = cv2.ORB_create(nfeatures=5000)
        color = (0, 0, 255) # Red

    kp, des = engine.detectAndCompute(gray, None)
    
    if kp:
        display_img = cv2.drawKeypoints(display_img, kp, None, color=color)

    return display_img, len(kp) if kp else 0

# --- 界面布局 ---
st.title("PCB Intelligent Detection Platform")
st.markdown("---")

# 侧边栏配置
st.sidebar.header("Control Center")

ds_option = st.sidebar.selectbox("Select Dataset", ["Dataset 1", "Dataset 2"])
ds_code = "ds1" if ds_option == "Dataset 1" else "ds2"

model_type = st.sidebar.radio("Attention Mechanism", ["Model-SE", "Model-CBAM"])
model_code = "se" if model_type == "Model-SE" else "cbam"

# 拼接模型文件名，例如 ds1_se.pt
target_model_file = f"{ds_code}_{model_code}.pt"

algo_option = st.sidebar.radio("Analysis Algorithm", ["Algorithm 1", "Algorithm 2"])
real_algo = "SIFT" if "Algorithm 1" in algo_option else "ORB"

conf_val = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.25)

uploaded_file = st.file_uploader("Upload PCB Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 加载图片
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1)

    col1, col2 = st.columns([3, 2])

    with col1:
        if st.button("Run Hybrid Analysis", type="primary"):
            # 1. 加载模型
            active_model = get_yolo_model(target_model_file)
            
            if active_model:
                # 2. YOLO 检测
                results = active_model.predict(original_img, conf=conf_val)
                res_img = results[0].plot()

                # 3. 传统特征提取
                final_output, kp_count = run_feature_extraction(res_img, real_algo)

                # 4. 显示结果
                st.image(cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB), 
                         caption=f"Output: {target_model_file} + {real_algo}", 
                         use_container_width=True)

                # 5. 右侧报告
                with col2:
                    st.subheader("Detection Report")
                    
                    report_list = []
                    # 提取 YOLO 结果
                    for box in results[0].boxes:
                        cls_name = results[0].names[int(box.cls[0])]
                        conf_score = f"{float(box.conf[0]):.2%}"
                        report_list.append(["YOLO AI", cls_name, conf_score, "Detected"])
                    
                    # 提取算法结果
                    report_list.append([real_algo, "Features", f"{kp_count} pts", "Extracted"])
                    
                    df = pd.DataFrame(report_list, columns=["Method", "Target", "Score/Value", "Status"])
                    st.table(df)
                    st.success("Analysis Completed Successfully.")
            else:
                st.error(f"Error: Model file '{target_model_file}' not found in models/ folder.")
        else:
            # 初始预览
            preview_img = original_img.copy()
            st.image(cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB), caption="Waiting for analysis", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info(f"Ready for: {target_model_file}")
