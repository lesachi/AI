import os
import yaml
from roboflow import Roboflow
from pathlib import Path

class RoboflowDataLoader:
    # Khởi tạo bộ tải dữ liệu từ Roboflow
    def __init__(self, api_key, workspace_name, project_name, version=1):
        self.rf = Roboflow(api_key=api_key)
        self.project = self.rf.workspace(workspace_name).project(project_name)
        self.version = version
    
    # Tải dataset từ Roboflow
    def download_dataset(self, format_type="yolov8", location="data"):
        try:
            dataset = self.project.version(self.version).download(format_type, location=location)
            print(f"Dữ liệu đã được tải về tại: {dataset.location}")
            return dataset.location
        except Exception as e:
            print(f"Lỗi khi tải dữ liệu: {e}")
            return None
    
    # Tải cấu hình YAML từ dataset
    def load_yaml_config(self, dataset_path):
        yaml_path = os.path.join(dataset_path, "data.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        return None