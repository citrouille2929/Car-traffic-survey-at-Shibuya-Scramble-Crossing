import os
import shutil
import numpy as np
from sklearn.model_selection import StratifiedKFold
from ultralytics import YOLO

# データセットのパス
image_dir = 'images'
label_dir = 'labels'

# 画像とラベルファイルのリストを取得
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
label_files = [f.replace('.jpg', '.txt') for f in image_files]

# クラス情報を抽出
labels = []
for label_file in label_files:
    with open(os.path.join(label_dir, label_file), 'r') as f:
        label_data = f.readlines()
        # 各ラベルファイルからクラスIDを取得
        class_ids = [int(line.split()[0]) for line in label_data]
        # クラスIDが存在しない場合は-1を挿入
        labels.append(class_ids[0] if class_ids else -1)

# StratifiedKFoldの設定
n_splits = 5  # 分割数
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# 各Foldごとにトレーニングと評価を行う
for fold, (train_index, val_index) in enumerate(skf.split(image_files, labels)):
    print(f"Fold {fold+1}/{n_splits}")
    
    # トレーニングと検証のデータセットを分割
    train_images, val_images = np.array(image_files)[train_index], np.array(image_files)[val_index]
    train_labels, val_labels = np.array(label_files)[train_index], np.array(label_files)[val_index]
    
    # 必要に応じて、画像とラベルをファイルに書き出して、YOLOv8が読み込める形式にする
    fold_dir = os.path.join('folds', f'fold_{fold+1}')
    train_dir = os.path.join(fold_dir, 'train/images')
    val_dir = os.path.join(fold_dir, 'val/images')
    train_label_dir = os.path.join(fold_dir, 'train/labels')
    val_label_dir = os.path.join(fold_dir, 'val/labels')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    
    # トレーニング用の画像とラベルをコピー
    for img_file, lbl_file in zip(train_images, train_labels):
        shutil.copy(os.path.join(image_dir, img_file), train_dir)
        shutil.copy(os.path.join(label_dir, lbl_file), train_label_dir)
    
    # 検証用の画像とラベルをコピー
    for img_file, lbl_file in zip(val_images, val_labels):
        shutil.copy(os.path.join(image_dir, img_file), val_dir)
        shutil.copy(os.path.join(label_dir, lbl_file), val_label_dir)
    
    # 動的にyamlファイルを作成
    yaml_content = f"""
train: {os.path.abspath(train_dir)}
val: {os.path.abspath(val_dir)}

nc: 11
names: ['ambulance','bicycle','bus','cab','car','garbage truck','go-cart','go-carts','motorcycle','police car','truck']
"""
    yaml_path = os.path.join(fold_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    # YOLOv8モデルのトレーニング
    model = YOLO('yolov8n.pt')  # プリトレーニング済みのモデルをロード
    model.train(data=yaml_path, epochs=100, conf=0.5, iou=0.5, batch=8, imgsz=(720, 1280), name=f'yolov8_fold_{fold+1}', project='StratifiedKFold_demo', amp=True, workers=1)
    
    # モデルの評価
    results = model.val(data=yaml_path)  # 検証データで評価

    # 必要に応じて結果を保存または出力
    print(results)
