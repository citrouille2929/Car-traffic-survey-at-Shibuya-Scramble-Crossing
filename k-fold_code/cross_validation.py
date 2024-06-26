import datetime
import shutil
from collections import Counter
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import KFold
from ultralytics import YOLO

# データセットのパスを設定
dataset_path = Path("path/to/dataset")
labels = sorted(dataset_path.rglob("*labels/*.txt"))

# YAMLファイルを読み込み、クラス名を取得
yaml_file = "data.yaml"
with open(yaml_file, "r", encoding="utf8") as y:
    classes = yaml.safe_load(y)["names"]

# ラベルファイルのインデックスを取得し、データフレームを初期化
indx = [l.stem for l in labels]
labels_df = pd.DataFrame([], columns=classes, index=indx)

# 各ラベルファイルの内容をカウントし、データフレームに追加
for label in labels:
    lbl_counter = Counter()
    with open(label, "r") as lf:
        lines = lf.readlines()
    for l in lines:
        lbl_counter[int(l.split(" ")[0])] += 1
    row_data = {classes[i]: lbl_counter.get(i, 0) for i in range(len(classes))}
    labels_df.loc[label.stem] = row_data

# NaN値を0.0に置き換え
labels_df = labels_df.replace(to_replace=np.nan, value=0.0)

# K-Foldクロスバリデーションの設定
ksplit = 5
kf = KFold(n_splits=ksplit, shuffle=True, random_state=20)
kfolds = list(kf.split(labels_df))

# 各スプリットのデータフレームを初期化
folds = [f"split_{n}" for n in range(1, ksplit + 1)]
folds_df = pd.DataFrame(index=indx, columns=folds)

# 各スプリットにトレーニングとバリデーションのラベルを設定
for idx, (train, val) in enumerate(kfolds, start=1):
    folds_df[f"split_{idx}"].loc[labels_df.iloc[train].index] = "train"
    folds_df[f"split_{idx}"].loc[labels_df.iloc[val].index] = "val"

# 各スプリットのラベル分布を計算
fold_lbl_distrb = pd.DataFrame(index=folds, columns=classes)
for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
    train_totals = labels_df.iloc[train_indices].sum()
    val_totals = labels_df.iloc[val_indices].sum()
    ratio = val_totals / (train_totals + 1e-7)
    fold_lbl_distrb.loc[f"split_{n}"] = ratio

# サポートされている画像拡張子
supported_extensions = [".jpg", ".jpeg", ".png"]

# 画像ファイルを取得
images = []
for ext in supported_extensions:
    images.extend(sorted((dataset_path / "images").rglob(f"*{ext}")))

# 保存パスを設定し、ディレクトリを作成
save_path = Path(dataset_path / f"{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-val")
save_path.mkdir(parents=True, exist_ok=True)
ds_yamls = []

# 各スプリット用のディレクトリを作成し、YAMLファイルを生成
for split in folds_df.columns:
    split_dir = save_path / split
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (split_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

    # データセットYAMLファイルを作成
    dataset_yaml = split_dir / f"{split}_dataset.yaml"
    ds_yamls.append(dataset_yaml)
    with open(dataset_yaml, "w") as ds_y:
        yaml.safe_dump(
            {
                "path": split_dir.as_posix(),
                "train": "train",
                "val": "val",
                "names": classes,
            },
            ds_y,
        )

# 画像とラベルを対応するディレクトリにコピー
for image, label in zip(images, labels):
    for split, k_split in folds_df.loc[image.stem].items():
        img_to_path = save_path / split / k_split / "images"
        lbl_to_path = save_path / split / k_split / "labels"
        shutil.copy(image, img_to_path / image.name)
        shutil.copy(label, lbl_to_path / label.name)

# YOLOモデルをロード
model = YOLO('yolov8n.pt', task="detect")

# 結果を格納する辞書を初期化
results = {}

# トレーニング設定
batch = 8
project = "kfold_demo"
epochs = 100

# 各スプリットでモデルをトレーニングし、結果を保存
for k in range(ksplit):
    dataset_yaml = ds_yamls[k]
    model.train(
        data=dataset_yaml,    # トレーニングデータセットの設定ファイル。YAML形式でデータセットのパスやクラス名を指定。
        epochs=epochs,        # トレーニングのエポック数。モデルがデータセット全体を何回繰り返して学習するかを指定。
        batch=batch,          # バッチサイズ。一度に処理するデータの数を指定。大きいほどメモリ使用量が増える。
        project=project,      # プロジェクト名。トレーニングの結果やログを保存するディレクトリ名を指定。
        amp=True,             # 自動混合精度 (Automatic Mixed Precision) を有効にするかどうか。Trueにするとトレーニング速度が向上し、メモリ消費量が減少することがある。
        workers=1             # データローディングのためのワーカープロセス数。データの読み込みを並列化することでトレーニング速度を向上させる。
    )
    results[k] = model.metrics  # 各スプリットのトレーニング結果（メトリクス）を保存
    torch.cuda.empty_cache()    # CUDAキャッシュをクリアしてGPUメモリを解放
