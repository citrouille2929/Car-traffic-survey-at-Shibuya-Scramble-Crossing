import os
import csv

# ラベルファイルと画像ファイルが存在するディレクトリのパス
label_dir = 'labels'
image_dir = 'images'

# 出力するCSVファイルのパス
output_csv = 'output.csv'

# クラスIDと車両の種類の対応辞書
class_dict = {
    '0': 'ambulance',
    '1': 'bicycle',
    '2': 'bus',
    '3': 'cab',
    '4': 'car',
    '5': 'garbage truck',
    '6': 'go-cart',
    '7': 'go-carts',
    '8': 'motorcycle',
    '9': 'police car',
    '10': 'truck'
}

# CSVファイルに書き込む準備
with open(output_csv, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    # ヘッダーを書き込む
    header = ['写真ファイル名'] + list(class_dict.values())
    csvwriter.writerow(header)
    
    # 画像ディレクトリ内のすべての画像ファイルを処理
    for image_file in os.listdir(image_dir):
        if image_file.endswith('.jpg'):
            image_path = os.path.join(image_dir, image_file)
            label_file = image_file.replace('.jpg', '.txt')
            label_path = os.path.join(label_dir, label_file)
            
            # カウントを保持する辞書を初期化
            vehicle_count = {vehicle: 0 for vehicle in class_dict.values()}
            
            # ラベルファイルが存在する場合、カウントを更新
            if os.path.exists(label_path):
                try:
                    with open(label_path, 'r') as file:
                        for line in file:
                            class_id = line.split()[0]
                            vehicle_type = class_dict.get(class_id)
                            if vehicle_type:
                                vehicle_count[vehicle_type] += 1
                except Exception as e:
                    print(f"Error reading {label_path}: {e}")
            
            # カウントデータを書き込む
            row = [image_file] + list(vehicle_count.values())
            csvwriter.writerow(row)
