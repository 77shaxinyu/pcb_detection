import streamlit as st
import pandas as pd
import numpy as np
from ultralytics import YOLO
from PIL import Image
import cv2
import os

# 页面配置
st.set_page_config(page_title="PCB缺陷检测系统", layout="wide")

# 加载模型
@st.cache_resource
def load_yolo_models():
    # 路径根据您的电脑实际情况配置
    base_path = r"C:\Users\donghaoran\PycharmProjects\set2\runs\detect"
    model_paths = {
        "Dataset1_SE": os.path.join(base_path, "pcb_yolo12_se_dataset1", "weights", "best.pt"),
        "Dataset1_CBAM": os.path.join(base_path, "pcb_yolo12_cbam_dataset1", "weights", "best.pt"),
        "Dataset2_SE": os.path.join(base_path, "pcb_yolo12_se_dataset2", "weights", "best.pt"),
        "Dataset2_CBAM": os.path.join(base_path, "pcb_yolo12_cbam_dataset2", "weights", "best.pt")
    }
    
    loaded_models = {}
    for name, path in model_paths.items():
        if os.path.exists(path):
            loaded_models[name] = YOLO(path)
    return loaded_models

def main():
    st.title("PCB智能检测分析系统")

    # 获取模型
    models = load_yolo_models()
    
    if not models:
        st.error("错误：未找到模型文件，请检查存放路径。")
        return

    # 侧边栏
    st.sidebar.header("控制面板")
    selected_name = st.sidebar.selectbox("选择模型", list(models.keys()))
    conf_val = st.sidebar.slider("置信度阈值", 0.1, 1.0, 0.25)
    
    # 上传文件
    uploaded_file = st.file_uploader("上传PCB图像", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # 布局分栏
        col_left, col_right = st.columns([6, 4])

        with col_left:
            st.subheader("检测结果图")
            if st.button("开始分析"):
                model = models[selected_name]
                results = model.predict(img_array, conf=conf_val)
                result = results[0]

                # 绘制带标签的图并修正颜色
                annotated_frame = result.plot()
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                st.image(annotated_frame, use_container_width=True)

                # 提取数据
                table_list = []
                if result.boxes:
                    for box in result.boxes:
                        # 类别和置信度
                        cls_id = int(box.cls[0])
                        label = result.names[cls_id]
                        conf = float(box.conf[0])
                        
                        # 提取坐标并转为整数
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        coord_str = f"[{xyxy[0]}, {xyxy[1]}, {xyxy[2]}, {xyxy[3]}]"

                        # 组织字典，确保坐标列在第三位
                        table_list.append({
                            "Method": "YOLO12",
                            "Target": label,
                            "Coordinates": coord_str,
                            "Value": f"{conf:.2%}",
                            "Status": "Verified"
                        })

                with col_right:
                    st.subheader("检测汇总表")
                    if table_list:
                        df = pd.DataFrame(table_list)
                        # 网页展示表格
                        st.table(df)

                        # 类别统计
                        st.subheader("成分统计")
                        stats = df["Target"].value_counts().reset_index()
                        stats.columns = ["Component", "Quantity"]
                        st.table(stats)
                    else:
                        st.write("未检测到缺陷。")
            else:
                st.image(image, caption="原始预览", use_container_width=True)

if __name__ == "__main__":
    main()
