import os
import yaml
from src.model_trainer import PlateDetectionTrainer
from src.plate_detector import MotorbikePlateDetector

# Script xác thực mô hình trên tập dev
def main():
    # Đọc cấu hình từ file config.yaml
    with open("../config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    # Lấy thông tin từ config
    paths_config = config["paths"]
    model_config = config["model"]
    training_config = config["training"]
    
    dev_path = paths_config["dev_path"]
    model_path = paths_config["model_path"]
    
    # Tạo data.yaml cho tập dev (validation)
    data_yaml = {
        "train": os.path.join(paths_config["train_path"], "images"),
        "val": os.path.join(dev_path, "images"),
        "test": os.path.join(paths_config["test_path"], "images"),
        "nc": 1,
        "names": ["license_plate"]
    }
    with open("data.yaml", "w") as yaml_file:
        yaml.dump(data_yaml, yaml_file)
    
    # Đánh giá mô hình
    trainer = PlateDetectionTrainer(model_size=model_config["size"])
    if os.path.exists(model_path):
        trainer.model = PlateDetectionTrainer(model_size=model_config["size"]).model.load(model_path)
    results = trainer.evaluate("data.yaml")
    print(f"Kết quả đánh giá trên tập dev: mAP50 = {results.box.map50}, mAP50-95 = {results.box.map}")

if __name__ == "__main__":
    main()