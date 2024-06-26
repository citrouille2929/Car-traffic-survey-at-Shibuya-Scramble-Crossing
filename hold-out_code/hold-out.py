import os
import shutil
from sklearn.model_selection import train_test_split

# データセットのパスを設定
dataset_path = 'dataset'

# 画像とラベルのリストを初期化
images = []
labels = []

# 画像とラベルのディレクトリパスを設定
images_dir = os.path.join(dataset_path, 'images')
labels_dir = os.path.join(dataset_path, 'labels')

# 画像ファイルとラベルファイルのリストを取得
image_files = os.listdir(images_dir)
label_files = os.listdir(labels_dir)

# 画像ファイルごとに対応するラベルファイルが存在するかチェック
for img_file in image_files:
    img_path = os.path.join(images_dir, img_file)
    label_file = img_file.replace('.jpg', '.txt')  # ラベルファイルは .txt ファイルであると仮定
    label_path = os.path.join(labels_dir, label_file)

    # 対応するラベルファイルが存在する場合、画像とラベルのパスをリストに追加
    if os.path.exists(label_path):
        images.append(img_path)
        labels.append(label_path)

# 画像とラベルのリストをトレーニングセットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# トレーニング用と検証用のディレクトリを作成（既に存在する場合は無視）
os.makedirs(os.path.join(images_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(images_dir, 'val'), exist_ok=True)
os.makedirs(os.path.join(labels_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(labels_dir, 'val'), exist_ok=True)

# トレーニングセットの画像とラベルを対応するディレクトリに移動
for img_train in X_train:
    img_filename = os.path.basename(img_train)
    label_filename = img_filename.replace('.jpg', '.txt')
    
    shutil.move(img_train, os.path.join(images_dir, 'train', img_filename))
    shutil.move(os.path.join(labels_dir, label_filename), os.path.join(labels_dir, 'train', label_filename))

# テストセットの画像とラベルを対応するディレクトリに移動
for img_test in X_test:
    img_filename = os.path.basename(img_test)
    label_filename = img_filename.replace('.jpg', '.txt')
    
    shutil.move(img_test, os.path.join(images_dir, 'val', img_filename))
    shutil.move(os.path.join(labels_dir, label_filename), os.path.join(labels_dir, 'val', label_filename))
