import os
import pandas as pd
import numpy as np
from ultralytics import YOLO

def get_detailed_metrics_with_coords(model_path, model_label, yaml_name):
    if not os.path.exists(model_path):
        return []
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, yaml_name)
    
    try:
        model = YOLO(model_path)
        
        # 1. 运行验证获取性能指标 (P, R, mAP50)
        val_results = model.val(data=yaml_path, plots=False, verbose=False, workers=0, cache=False)
        names = model.names
        
        # 2. 运行预测获取坐标信息 (以验证集图片为例)
        # 我们从 yaml 中获取验证集路径
        import yaml
        with open(yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        val_images_path = data_config.get('val')
        
        # 预测验证集图片获取坐标
        pred_results = model.predict(source=val_images_path, save=False, verbose=False)
        
        # 整理坐标：按类别归类坐标点
        coords_dict = {i: [] for i in names.keys()}
        for res in pred_results:
            boxes = res.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                # 获取坐标 [x1, y1, x2, y2] 并转为整数便于查看
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                coords_dict[cls_id].append(str(xyxy))

        # 3. 整合指标与坐标
        class_metrics = []
        present_indices = val_results.box.ap_class_index
        
        for i, class_idx in enumerate(present_indices):
            idx = int(class_idx)
            name = names.get(idx, f"class_{idx}")
            
            # 获取指标
            p, r, ap50, _ = val_results.box.class_result(i)
            
            # 提取该类别的所有坐标，取前几个代表性坐标显示以免表格太长
            all_coords = coords_dict.get(idx, ["N/A"])
            display_coords = "; ".join(all_coords[:2]) + ("..." if len(all_coords) > 2 else "")
            
            f1 = 2 * (float(p) * float(r)) / (float(p) + float(r)) if (float(p) + float(r)) > 0 else 0
            
            class_metrics.append({
                "Model": model_label,
                "Defect_Type": name,
                "Coordinates(x1,y1,x2,y2)": display_coords, # 放在第三列
                "mAP50": round(float(ap50), 4),
                "Precision": round(float(p), 4),
                "Recall": round(float(r), 4),
                "F1_Score": round(float(f1), 4)
            })
        return class_metrics
    except Exception as e:
        print(f"分析 {model_label} 时出错: {e}")
        return []

if __name__ == '__main__':
    base_path = r"C:\Users\donghaoran\PycharmProjects\set2\runs\detect"
    
    model_configs = {
        "Dataset1_SE": (os.path.join(base_path, "pcb_yolo12_se_dataset1", "weights", "best.pt"), "dataset1_data.yaml"),
        "Dataset1_CBAM": (os.path.join(base_path, "pcb_yolo12_cbam_dataset1", "weights", "best.pt"), "dataset1_data.yaml"),
        "Dataset2_SE": (os.path.join(base_path, "pcb_yolo12_se_dataset2", "weights", "best.pt"), "dataset2_data.yaml"),
        "Dataset2_CBAM": (os.path.join(base_path, "pcb_yolo12_cbam_dataset2", "weights", "best.pt"), "dataset2_data.yaml")
    }

    all_data = []
    for label, (path, yaml_file) in model_configs.items():
        print(f"正在提取: {label} (含坐标计算)...")
        all_data.extend(get_detailed_metrics_with_coords(path, label, yaml_file))

    if all_data:
        df = pd.DataFrame(all_data)
        
        # 终端展示 (仅显示前几个字符，防止坐标太长撑破屏幕)
        pd.set_option('display.max_colwidth', 30)
        print("\n" + "="*95)
        print(df.to_string(index=False))
        print("="*95)
        
        # 保存完整数据
        df.to_csv("pcb_metrics_with_coords.csv", index=False, encoding='utf-8-sig')
        print(f"\n完整坐标与指标已存入: {os.path.abspath('pcb_metrics_with_coords.csv')}")
