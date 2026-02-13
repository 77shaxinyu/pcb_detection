import os
import pandas as pd
import numpy as np
from ultralytics import YOLO

def get_detailed_metrics(model_path, model_label, yaml_name):
    if not os.path.exists(model_path):
        return []
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, yaml_name)
    
    try:
        model = YOLO(model_path)
        # cache=False 确保 Dataset2 重新扫描以发现所有 4 种缺陷
        results = model.val(data=yaml_path, plots=False, verbose=False, workers=0, cache=False)
        
        class_metrics = []
        names = model.names 
        
        # 获取当前验证集中存在的类别索引
        # 根据报错信息，新版使用 ap_class_index
        present_classes = results.box.ap_class_index
        
        for i, class_idx in enumerate(present_classes):
            name = names.get(int(class_idx), f"class_{class_idx}")
            
            # 使用官方推荐的 class_result 方法提取各项指标
            # 返回顺序为: (precision, recall, ap50, ap)
            p, r, ap50, _ = results.box.class_result(i)
            
            p = float(p)
            r = float(r)
            m50 = float(ap50)
            f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
            
            class_metrics.append({
                "Model": model_label,
                "Defect_Type": name,
                "mAP50": round(m50, 4),
                "Precision": round(p, 4),
                "Recall": round(r, 4),
                "F1_Score": round(f1, 4)
            })
        return class_metrics
    except Exception as e:
        print(f"处理模型 {model_label} 时出错: {e}")
        return []

if __name__ == '__main__':
    base_path = r"C:\Users\donghaoran\PycharmProjects\set2\runs\detect"
    
    model_configs = {
        "Dataset1_SE": (os.path.join(base_path, "pcb_yolo12_se_dataset1", "weights", "best.pt"), "dataset1_data.yaml"),
        "Dataset1_CBAM": (os.path.join(base_path, "pcb_yolo12_cbam_dataset1", "weights", "best.pt"), "dataset1_data.yaml"),
        "Dataset2_SE": (os.path.join(base_path, "pcb_yolo12_se_dataset2", "weights", "best.pt"), "dataset2_data.yaml"),
        "Dataset2_CBAM": (os.path.join(base_path, "pcb_yolo12_cbam_dataset2", "weights", "best.pt"), "dataset2_data.yaml")
    }

    print("正在提取各模型分缺陷指标...")
    all_data = []
    for label, (path, yaml_file) in model_configs.items():
        print(f"分析中: {label}...")
        all_data.extend(get_detailed_metrics(path, label, yaml_file))

    if all_data:
        df = pd.DataFrame(all_data)
        
        # 终端展示
        print("\n" + "="*75)
        print(f"{'Model':<15} {'Defect Type':<20} {'mAP50':<8} {'P':<8} {'R':<8} {'F1':<8}")
        print("-" * 75)
        for _, row in df.iterrows():
            print(f"{row['Model']:<15} {row['Defect_Type']:<20} {row['mAP50']:<8.4f} {row['Precision']:<8.4f} {row['Recall']:<8.4f} {row['F1_Score']:<8.4f}")
        print("="*75)
        
        # 导出
        df.to_csv("final_pcb_class_metrics.csv", index=False, encoding='utf-8-sig')
        print(f"\n汇总表已保存: {os.path.abspath('final_pcb_class_metrics.csv')}")
    else:
        print("未提取到任何数据，请检查模型文件路径是否正确。")
