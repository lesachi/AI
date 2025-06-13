import os

output_dir = 'outputs'
keep = {}
delete = []

for file in os.listdir(output_dir):
    if not file.startswith('annotated_') or not file.endswith('.jpg'):
        continue

    # Tách prefix từ tên ảnh: annotated_11_jpg -> annotated_11_jpg
    prefix = '_'.join(file.split('_')[:2])

    full_path = os.path.join(output_dir, file)
    file_size = os.path.getsize(full_path)

    # So sánh kích thước ảnh (dung lượng) để giữ ảnh rõ hơn
    if prefix not in keep:
        keep[prefix] = (file, file_size)
    else:
        kept_file, kept_size = keep[prefix]
        if file_size > kept_size:
            delete.append(os.path.join(output_dir, kept_file))
            keep[prefix] = (file, file_size)
        else:
            delete.append(full_path)

# Xoá các file thừa
print(f"🧹 Sẽ xoá {len(delete)} ảnh trùng:")
for f in delete:
    os.remove(f)
    print(f"Đã xoá: {f}")
