import os
import yaml
import sys
import cv2
from rapidfuzz.distance import Levenshtein

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.model_trainer import PlateDetectionTrainer
from src.plate_detector import MotorbikePlateDetector

# Ground truth gán bằng tên file (VD: 59B149271.jpg)
def extract_gt_from_filename(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    for part in base.split('_'):
        if any(c.isdigit() for c in part):
            return part.upper().replace('-', '').replace('.', '')
    return None

# Hàm tính độ chính xác ký tự
def char_accuracy(pred, gt):
    if not gt:
        return 0.0
    dist = Levenshtein.distance(pred, gt)
    return 1 - dist / max(len(pred), len(gt))

# Hàm chính

def main():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config.yaml"))
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    paths = config["paths"]
    model_cfg = config["model"]

    # === Tạo file data.yaml cho YOLO val
    data_yaml = {
        "train": os.path.abspath(os.path.join(paths["train_path"], "images")),
        "val": os.path.abspath(os.path.join(paths["dev_path"], "images")),
        "test": os.path.abspath(os.path.join(paths["test_path"], "images")),
        "nc": 1,
        "names": ["license_plate"]
    }
    data_yaml_path = os.path.join(os.path.dirname(__file__), "dev_data.yaml")
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f)

    # === Đánh giá detection
    trainer = PlateDetectionTrainer(model_size=model_cfg["size"])
    if os.path.exists(paths["model_path"]):
        trainer.model = trainer.model.load(paths["model_path"])

    metrics = trainer.evaluate(data_yaml_path)
    print("\n KẾT QUẢ ĐÁNH GIÁ YOLO:")
    print(f"mAP@0.5        : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95   : {metrics.box.map:.4f}")
    print(f"Precision      : {metrics.box.mp:.4f}")
    print(f"Recall         : {metrics.box.mr:.4f}")
    f1 = 2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr + 1e-6)
    print(f"F1 Score       : {f1:.4f}")

    # === Đánh giá OCR nâng cao ===
    print("\n ĐÁNH GIÁ OCR NÂNG CAO:")
    detector = MotorbikePlateDetector(paths["model_path"])
    dev_images = os.listdir(os.path.join(paths["dev_path"], "images"))

    total_char_acc = 0
    total_cer = 0
    exact_match = 0
    count = 0

    for img_name in dev_images:
        img_path = os.path.join(paths["dev_path"], "images", img_name)
        gt_text = extract_gt_from_filename(img_name)
        image = cv2.imread(img_path)
        detections = detector.detect_plates(image)

        if not detections:
            continue

        pred_text = detections[0]['text']
        if not pred_text:
            continue

        pred_clean = pred_text.replace('-', '').replace('.', '')
        gt_clean = gt_text.replace('-', '').replace('.', '') if gt_text else ""

        acc = char_accuracy(pred_clean, gt_clean)
        cer = Levenshtein.distance(pred_clean, gt_clean) / max(len(gt_clean), 1)

        total_char_acc += acc
        total_cer += cer
        exact_match += 1 if pred_clean == gt_clean else 0
        count += 1

        print(f"{img_name}: pred='{pred_clean}' | gt='{gt_clean}' | acc={acc:.2f} | CER={cer:.2f}")

    if count > 0:
        print("\n TỔNG KẾT OCR:")
        print(f"Character-level accuracy: {total_char_acc / count:.4f}")
        print(f"Word-level accuracy     : {exact_match / count:.4f}")
        print(f"Average CER             : {total_cer / count:.4f}")
    else:
        print(" Không có ảnh nào được OCR thành công.")

if __name__ == "__main__":
    main()
