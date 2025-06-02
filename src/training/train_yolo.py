import os
import torch
from ultralytics import YOLO

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = YOLO("yolo12n.pt").to(device)

    script_dir = os.path.dirname(__file__)
    project_data = os.path.abspath(os.path.join(script_dir, "..", "..", "data"))
    data_yaml_path = os.path.join(project_data, "wildfire_data.yaml")

    print("--- Starting YOLOv12 Training ---")
    results = model.train(
        data=data_yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        name='wildfire_yolov12_nano',
        project='runs/yolo_train',
        optimizer='AdamW',
        lr0=0.001,
        weight_decay=0.0005,
        pretrained=True,
        workers=os.cpu_count() // 2,
        seed=7
    )

    print("\n--- Running Final Validation ---")
    metrics = model.val()

    print("\n--- Validation Metrics Summary ---")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")

    print("\n--- Running Final Evaluation on 'test' split ---")
    test_metrics = model.val(split='test')

    print("\n--- Test Metrics Summary ---")
    print(f"  mAP50: {test_metrics.box.map50:.4f}")
    print(f"  Precision: {test_metrics.box.mp:.4f}")
    print(f"  Recall: {test_metrics.box.mr:.4f}")

if __name__ == '__main__':
    main()