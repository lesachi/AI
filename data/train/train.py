import os
import yaml
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data_loader import RoboflowDataLoader
from src.model_trainer import PlateDetectionTrainer

# Script huấn luyện chính
def main():
    # Đọc cấu hình từ file config.yaml
    with open("config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    
    # Lấy thông tin từ config
    roboflow_config = config["roboflow"]
    paths_config = config["paths"]
    model_config = config["model"]
    training_config = config["training"]

    api_key = roboflow_config["api_key"]
    workspace = roboflow_config["workspace"]
    project = roboflow_config["project"]
    version = roboflow_config["version"]
    train_path = paths_config["train_path"]
    test_path = paths_config["test_path"]
    dev_path = paths_config["dev_path"]
    model_path = paths_config["model_path"]
    
    # Tải dữ liệu từ Roboflow và sắp xếp vào các thư mục
    loader = RoboflowDataLoader(api_key, workspace, project, version)
    temp_path = f"temp_version_{version}"
    downloaded_path = loader.download_dataset(location=temp_path)
    if not downloaded_path:
        print("Tải dữ liệu thất bại!")
        return
    
    # Tạo cấu trúc thư mục và di chuyển dữ liệu
    for src, dest in [
        ("train", train_path),
        ("test", test_path),
        ("valid", dev_path)  # Roboflow thường dùng "valid" thay cho "dev"
    ]:
        src_path = os.path.join(downloaded_path, src)
        if os.path.exists(src_path):
            for subdir in ["images", "labels"]:
                os.makedirs(os.path.join(dest, subdir), exist_ok=True)
                for file in os.listdir(os.path.join(src_path, subdir)):
                    os.rename(os.path.join(src_path, subdir, file), os.path.join(dest, subdir, file))
    
    # Tạo data.yaml động
    data_yaml = {
        "train": os.path.join(train_path, "images"),
        "val": os.path.join(dev_path, "images"),
        "test": os.path.join(test_path, "images"),
        "nc": 1,  # Số lớp (giả sử chỉ có 1 lớp: biển số)
        "names": ["license_plate"]  # Tên lớp
    }
    with open("data.yaml", "w") as yaml_file:
        yaml.dump(data_yaml, yaml_file)
    
    # Tải và huấn luyện mô hình
    trainer = PlateDetectionTrainer(model_size=model_config["size"])
    trainer.train(
        "data.yaml",
        epochs=training_config["epochs"],            # Lấy epochs từ training
        img_size=model_config["image_size"],
        batch_size=training_config["batch_size"]     # Lấy batch_size từ training
    )
      # TẠO THƯ MỤC CHỨA MODEL TRƯỚC KHI LƯU
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    trainer.save_model(model_path)
    print(f"Huấn luyện hoàn tất! Mô hình được lưu tại: {model_path}")

if __name__ == "__main__":
    main()
