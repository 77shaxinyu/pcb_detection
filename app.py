import streamlit as st
import pandas as pd
import numpy as np
from ultralytics import YOLO
from PIL import Image
import cv2
import os

# 设置页面配置
st.set_page_config(page_title="PCB Defect Detection System", layout="wide")

# --- 1. 初始化模型 ---
@st.cache_resource
def load_models():
    # 根据你的目录结构配置模型路径
    base_path = r"C:\Users\donghaoran\PycharmProjects\set2\runs\detect"
    models = {
        "YOLO12-SE": YOLO(os.path.join(base_path, "pcb_yolo12_se_dataset2", "weights", "best.pt")),
        "YOLO12-CBAM": YOLO(os.path.join(base_path, "pcb_yolo12_cbam_dataset2", "weights", "best.pt"))
    }
    return models

# --- 2. 检测逻辑 ---
def run_detection(image, model):
    # 将 PIL 图片转为 OpenCV 格式供模型使用
    img_array = np.array(image)
    
    # 执行预测
    results = model.predict(img_array, conf=0.25)
    result = results[0]
    
    # 绘制检测结果图
    annotated_img = result.plot()
    
    # 提取表格数据
    table_data = []
    if result.boxes:
        for box in result.boxes:
            # 类别 ID 和名称
            cls_id = int(box.cls[0])
            label = result.names[cls_id]
            # 置信度
            conf = float(box.conf[0])
            # 坐标提取 (x1, y1, x2, y2)
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            coord_str = f"[{xyxy[0]}, {xyxy[1]}, {xyxy[2]}, {xyxy[3]}]"
            
            # 构造表格行：坐标严格放在第三列
            table_data.append({
                "Method": "YOLO12",
                "Target": label,
                "Coordinates": coord_str, # 第三行/列
                "Confidence": f"{conf:.2%}",
                "Status": "Verified"
            })
            
    return annotated_img, pd.DataFrame(table_data)

# --- 3. 网页 UI 布局 ---
def main():
    st.title("PCB Defect Detection System")
    st.markdown("---")

    # 侧边栏：模型选择
    st.sidebar.header("Settings")
    model_dict = load_models()
    selected_model_name = st.sidebar.selectbox("Select Model", list(model_dict.keys()))
    selected_model = model_dict[selected_model_name]

    # 上传部分
    uploaded_file = st.file_uploader("Upload PCB Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # 分为左右两栏展示
        col1, col2 = st.columns([6, 4])
        
        with col1:
            st.subheader("Visualized Result")
            if st.button("Start Analysis"):
                # 运行检测
                annotated_img, df_results = run_detection(image, selected_model)
                
                # 显示标注后的图片 (转回 RGB)
                st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_column_width=True)
                
                with col2:
                    st.subheader("Detection Summary")
                    if not df_results.empty:
                        # 展示带坐标的表格
                        st.table(df_results) 
                        
                        # 统计部分
                        st.subheader("Component Statistics")
                        stats = df_results['Target'].value_counts().reset_index()
                        stats.columns = ['Component', 'Quantity']
                        st.table(stats)
                    else:
                        st.write("No defects detected.")
        else:
            # 未分析前先预览原图
            st.image(image, caption="Original Image Preview", width=500)

if __name__ == "__main__":
    main()
