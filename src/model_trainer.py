from ultralytics import YOLO
import torch
import os

class PlateDetectionTrainer:
    # Khởi tạo huấn luyện viên với kích thước mô hình
    def __init__(self, model_size='n'):
        self.model = YOLO(f'yolov8{model_size}.pt')
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_size = model_size
    
    # Huấn luyện mô hình
    def train(self, data_yaml_path, epochs=100, img_size=640, batch_size=16):
        print(f"Kiểm tra GPU: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Tên GPU: {torch.cuda.get_device_name(0)}")
        results = self.model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            name=f'motorbike_plate_yolov8{self.model_size}',
            patience=10,
            save=True
        )
        return results
    
    # Đánh giá mô hình
    def evaluate(self, data_yaml_path):
        return self.model.val(data=data_yaml_path)
    
    # Lưu mô hình
    def save_model(self, path):
        self.model.save(path)