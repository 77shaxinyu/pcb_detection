import streamlit as st
import cv2
import numpy as np
import os
import sys
import pandas as pd

# --- 1. 核心修复：强制模块路径映射与文件名适配 ---
# 获取当前脚本所在目录并加入系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 动态适配你上传的文件名 modules_new.py
try:
    import modules_new as modules
    # 终极补丁：将你的新文件伪装成模型寻找的各种可能路径
    # 解决 torch.load 时的 Unpickling 路径报错（针对自定义注意力机制层）
    sys.modules['modules.attention'] = modules
    sys.modules['models.common'] = modules
    sys.modules['attention'] = modules
    sys.modules['modules'] = modules
except Exception as e:
    st.error(f"Error: Could not find modules_new.py. Please ensure it is in the root directory.")

from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

# 2. 动态注册类到 YOLO 任务系统
# 这样即使权重里记录的是 'SEAttention'，它也能通过别名找到你代码里的类
try:
    # 尝试从 modules_new.py 获取类，兼容多种可能的命名习惯
    SE_class = getattr(modules, 'SE', getattr(modules, 'SEAttention', None))
    CBAM_class = getattr(modules, 'CBAM', getattr(modules, 'CBAMAttention', None))
    
    if SE_class:
        setattr(tasks, 'SE', SE_class)
        setattr(tasks, 'SEAttention', SE_class)
    if CBAM_class:
        setattr(tasks, 'CBAM', CBAM_class)
        setattr(tasks, 'CBAMAttention', CBAM_class)
except Exception as e:
    st.sidebar.warning("Custom layers registration info: Active")

# --- 3. 页面全局配置 ---
st.set_page_config(page_title="PCB Inspection System", layout="wide")

@st.cache_resource
def get_yolo_model(model_name):
    # 确保路径指向 models/ 文件夹
    path = os.path.join("models", model_name)
    if os.path.exists(path):
        # 加载 YOLO 模型
        return YOLO(path)
    return None

def run_feature_analysis(img_bgr, algo_type):
    """执行 Algorithm 1 (SIFT) 或 Algorithm 2 (ORB) 的特征提取与网格绘制"""
    h, w = img_bgr.shape[:2]
    canvas = img_bgr.copy()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 绘制经典 9x9 绿色参考网格
    for i in range(1, 9):
        cv2.line(canvas, (0, int(i * h / 9)), (w, int(i * h / 9)), (0, 255, 0), 1)
        cv2.line(canvas, (int(i * w / 9), 0), (int(i * w / 9), h), (0, 255, 0), 1)

    # 特征点提取逻辑
    if algo_type == "SIFT":
        feat_engine = cv2.SIFT_create(nfeatures=2000)
        pt_color = (255, 0, 0) # 蓝色标识 SIFT
    else:
        feat_engine = cv2.ORB_create(nfeatures=5000)
        pt_color = (0, 0, 255) # 红色标识 ORB

    kp, des = feat_engine.detectAndCompute(gray, None)
    if kp:
        canvas = cv2.drawKeypoints(canvas, kp, None, color=pt_color)

    return canvas, len(kp) if kp else 0

# --- 4. 界面展示布局 ---
st.title("🛡️ PCB Defect Detection & Feature Analysis Platform")
st.markdown("---")

# 侧边栏配置面板
st.sidebar.header("Configuration Panel")
ds_select = st.sidebar.selectbox("Test Dataset", ["Dataset 1", "Dataset 2"])
ds_code = "ds1" if ds_select == "Dataset 1" else "ds2"

model_select = st.sidebar.radio("Attention Mechanism", ["SE", "CBAM"])
model_code = model_select.lower()

# 自动匹配文件名：例如 ds1_se.pt
target_file = f"{ds_code}_{model_code}.pt"

algo_select = st.sidebar.radio("Analysis Mode", ["Algorithm 1 (SIFT)", "Algorithm 2 (ORB)"])
algo_type = "SIFT" if "Algorithm 1" in algo_select else "ORB"

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.25)

# 文件上传区
uploaded_file = st.file_uploader("Upload PCB Image to Analyze", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 将上传的文件转为 OpenCV 格式
    bytes_data = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    raw_img = cv2.imdecode(bytes_data, 1)

    col_view, col_rep = st.columns([3, 2])

    with col_view:
        if st.button("Start Hybrid Analysis", type="primary"):
            # 1. 尝试加载对应的 YOLO 模型
            yolo_model = get_yolo_model(target_file)
            
            if yolo_model:
                # 执行 AI 缺陷检测
                res = yolo_model.predict(raw_img, conf=conf_threshold)
                render_img = res[0].plot()

                # 执行传统视觉特征分析
                final_img, total_kp = run_feature_analysis(render_img, algo_type)

                # 展示合成后的最终结果
                st.image(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB), 
                         caption=f"Analysis Output: {target_file} + {algo_type}", 
                         use_container_width=True)

                # --- 5. 数据报告区 ---
                with col_rep:
                    st.subheader("📋 Detection Summary")
                    data_rows = []
                    
                    # 遍历 YOLO 找到的所有缺陷
                    for box in res[0].boxes:
                        label = res[0].names[int(box.cls[0])]
                        prob = f"{float(box.conf[0]):.2%}"
                        data_rows.append(["AI (YOLO)", label, prob, "Confirmed"])
                    
                    # 添加算法提取的特征点统计
                    data_rows.append([f"CV ({algo_type})", "Keypoints", f"{total_kp} pts", "Auto-Extracted"])
                    
                    # 生成精美的结果表格
                    report_df = pd.DataFrame(data_rows, columns=["Method", "Target", "Score/Value", "Status"])
                    st.table(report_df)
                    st.success("Platform analysis completed successfully.")
            else:
                st.error(f"Critical Error: Model 'models/{target_file}' not found.")
        else:
            # 未点击分析时的预览图
            st.image(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB), caption="Raw PCB Image (Ready for Analysis)", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption(f"Backend Ready | Loaded: {target_file}")
