import os
import re
import shutil
import yaml
import cv2

from src.plate_detector import MotorbikePlateDetector

# Đọc cấu hình từ config.yaml
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

model_path = config["paths"]["model_path"]
input_dir = config["paths"]["output_path"]
output_dir = os.path.join(input_dir, "filtered")
os.makedirs(output_dir, exist_ok=True)

detector = MotorbikePlateDetector(model_path=model_path, confidence_threshold=0.3)

# Lưu biển số đã gặp: text -> (filename, filesize)
plate_map = {}

# Lưu prefix giống nhau: annotated_XX -> (filename, filesize)
prefix_map = {}

def get_prefix(filename):
    match = re.match(r"annotated_(\d+|IMG\d+)", filename)
    return match.group(0) if match else None

def extract_plate_text(image_path):
    image = cv2.imread(image_path)
    detections = detector.detect_plates(image)
    if not detections or not detections[0]['text']:
        return None
    return detections[0]['text']

# ===== 1. Lọc theo annotated_XX prefix trước =====
for fname in os.listdir(input_dir):
    if not fname.endswith('.jpg') or not fname.startswith("annotated_"):
        continue
    prefix = get_prefix(fname)
    fpath = os.path.join(input_dir, fname)
    fsize = os.path.getsize(fpath)

    if prefix:
        if prefix in prefix_map:
            old_fname, old_size = prefix_map[prefix]
            if fsize > old_size:
                prefix_map[prefix] = (fname, fsize)
        else:
            prefix_map[prefix] = (fname, fsize)

# ===== 2. Lọc theo text biển số OCR =====
for prefix, (fname, _) in prefix_map.items():
    full_path = os.path.join(input_dir, fname)
    text = extract_plate_text(full_path)
    if not text:
        continue

    fsize = os.path.getsize(full_path)

    if text not in plate_map:
        plate_map[text] = (fname, fsize)
    else:
        old_fname, old_size = plate_map[text]
        if fsize > old_size:
            plate_map[text] = (fname, fsize)

# ===== 3. Copy ảnh duy nhất vào thư mục lọc =====
kept_files = {fname for fname, _ in plate_map.values()}

for fname in os.listdir(input_dir):
    if fname in kept_files:
        shutil.copy2(os.path.join(input_dir, fname), os.path.join(output_dir, fname))

print(f"✅ Đã lọc xong {len(kept_files)} ảnh, lưu tại: {output_dir}")
