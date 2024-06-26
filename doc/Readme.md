# Readme

## 目次

1. プロジェクト名
2. プロジェクト概要
3. プロジェクト詳細
4. リポジトリ内のファイルについて
5. 開発環境
6. 環境構築
7. LabelStudioのインストール
8. Weighted-Boxes-Fusionのインストール

## プロジェクト名

渋谷スクランブル交差点の交通量調査

## プロジェクト概要

- 文化の発信地である渋谷では多種多様な車両を見かけることができます。あなたはデータサイエンティストとして、まずは「車両の種類」を定義し、写真内から「種類ごとに車両を検出・カウントするモデル」を作成してほしいと依頼を受けました。
- そのモデルを用いて、平日・休日の計48時間ぶんの画像から、各時刻にどの種類の車が何台いるかをカウントしてグラフ化したいです。

## プロジェクト詳細

- doc内のプレゼンテーション.mdをご確認ください。
    - プレゼンテーション.md

## 開発環境

- 環境
    - OS  ：windows 10
    - CPU：Intel Core i7-10750H
    - GPU：GeForce RTX 2060
- 導入バージョン
    - conda 24. 5. 0
    - python 3. 8. 19
    - cuda 11. 8
    - cudnn 8. 9. 0
    - YOLOv8(ultralytics)

## リポジトリ内のファイルについて

### doc

- csv
    - 今回の課題の出力したcsvファイルが入っています。
- hold_out_images, k-fold_images, StratifiedKFold_images
    - ホールドアウト法、K-分割交差検証法、層化K分割交差検証法で学習させた結果グラフ画像がそれぞれ入っています。
- プレゼン画像
    - プレゼンテーション内の画像が入っています。
- 検出画像
    - アンサンブルした結果を画像に反映したものです。
- プレゼンテーション.md
    - 今回作成したプレゼンテーションです。
- Readme
    - このファイルです。

### hold-out_code, k-fold_code

- ホールドアウト法, K-分割交差検証法を実行するためのコードが入っています。

### StatifiedKFold_code

- StratifiedKFold.py
    - 層化K-分割交差検証法を実行するためのコードが入っています。
- ensemble.py
    - 層化K-分割交差検証法を用いて分割した学習データで訓練した各モデルの結果をアンサンブルするコードです。

### Other_code

- csv_write.py
    - csvファイルに結果を書き込むためのコードです。
- Graph Drawing.py
    - csvファイルの結果から棒グラフと折れ線グラフを作成するコードです。
- Picture_Drawing.py
    - 画像とUltralytics YOLO フォーマットのラベルを指定する事で画像にBBoxを描く事が出来るコードです。

## 環境構築

下記の順で記載してあります。

- Anacondaのインストール
- 仮想環境の構築
- GPU導入
- YOLOv8のインストール
- 動作確認

### Anacondaのインストール

- Anacondaは、データサイエンスや機械学習に特化したツールキットです。プログラミング言語のPythonやRを使うときに必要なツールやライブラリがすでにセットされているため、それらを一つずつインストールする手間が省けます。また、プロジェクトごとに異なるツールのバージョンを管理できる機能もあります。これにより、異なるプロジェクトで作業する際に互いの設定が干渉しあうことなく、スムーズに作業を進めることができます。
- 下記サイトからダウンロードが出来ます
    
    **Anaconda Installers:** [https://www.anaconda.com/download/success](https://www.anaconda.com/download/success)
    

### 仮想環境の構築

- Anaconda Navigatorを立ち上げた後、EnvironmentsからCreateを選択します
- 仮想環境の名前とPythonのバージョン(ここでは3.8.19)を選択し、createしてください

### GPU導入

- GeForceのGPUで動作させるには、CUDAとcuDNNという物のインストールが必要になります。
- CUDAのインストール
    - 下記サイトから「CUDA Toolkit 11.8.0」を選択してダウンロードしてください。
        
        **CUDA Toolkit Archive:** [https://developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive)
        
- cuDNNのインストール
    - cuDNNの入手にはアカウントが必要となるので持っていない方は作成してください。
    - 「Download cuDNN v8.9.0 (April 11th, 2023), for CUDA 11.x」を以下のサイトから選択してダウンロードしてください。
        
        **cuDNN Archive:** [https://developer.nvidia.com/rdp/cudnn-archive](https://developer.nvidia.com/rdp/cudnn-archive)
        
    - CuDNNはダウンロードしたら解凍します。解凍したフォルダに以下の3つのデータがあると思います。
        - >bin >include >lib
- cuDNNのデータをCUDAに上書きする
    - 解凍したCuDNNデータ(bin,include,libファイル)をCUDA\v11.8.にドラッグアンドドロップで入れます。CUDAをインストールしたときのパスを何も変更していなければ
        - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
        
        にCUDAのデータがあるはずです。移すときにデータは上書きしてください。
        
- 環境変数の設定
    - パスを通しておくため環境変数に「C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8」を登録しておいてください。CUDAの保存場所を変えてる場合はそれに合わせてパスを変えてください。
    - **環境変数に登録した後は再起動してください。**
- CUDAの確認
    - 再起動した後にCUDAが入っているかを確認するため、コマンドプロントを立ち上げてください。
        
        ```python
        nvcc -V
        ```
        
    - 上記のように入力し導入しているバージョンが表示されているか確認してください。
        
        Cuda compilation tools, release 11.8, V11.8.89と表示されていればCUDAが無事に入っています。
        

### YOLOv8のインストール

- Anaconda Promptを立ち上げた後、以下のコマンドを順番に入力します
    
    ```python
    conda activate 仮想環境の名前
    pip install ultralytics
    pip uninstall torch torchvision
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
    

これでYOLOv8のインストールとGPUの適用は完了です。

YOLOv8の仕様については、以下の公式サイトを確認してください

PythonでのYOLO使用方法: [https://docs.ultralytics.com/ja/usage/python/](https://docs.ultralytics.com/ja/usage/python/)

YOLOのデータセット形式について: [https://docs.ultralytics.com/ja/datasets/detect/coco8/](https://docs.ultralytics.com/ja/datasets/detect/coco8/)

### 動作確認

- 次にGPUとYOLOが正常に動作するか確認するため以下のコマンドでテストをします。

```python
yolo detect predict model=yolov8s.pt source="https://ultralytics.com/images/bus.jpg"
```

- GPUが正しく動いていれば
    
    「Ultralytics YOLOv8.2.42 🚀 Python-3.8.19 torch-2.3.1+cu118 CUDA:0 (NVIDIA GeForce RTX 2060, 6144MiB)」と使用しているグラフィックボードの名前が表示されます。
    
- テストしたデータはフォルダ移動をしていなければ
    
    「C:\Users\ユーザー名\runs\detect\predict」に保存されています。
    

## Label Studioのインストール

- 今回はLabel Studioを使いアノテーションを行っています。YOLO形式で出力が出来るものなら他のツールでも問題はありません。
- Label StudioはHeartexという会社が提供しているオープンソースのアノテーションツールです。様々な種類のデータ（音声・画像・テキスト・時系列データ…）、幅広いタスクに対応しています。
    
    ### インストール
    
    - 以下のコマンドでインストールが出来ます
        - 一緒にいろいろなパッケージがインストールされるので、Anacondaを使う場合は別で仮想環境を立ち上げた方が良いです。
    
    ```python
    pip install label-studio
    ```
    
    ### 起動
    
    以下のコマンドで起動が出来ます
    
    ```python
    label-studio
    ```
    
    Label Studioの詳しい使い方は以下のサイトで確認してください
    
    Label Studioの使い方: [https://tech.aru-zakki.com/how-to-use-label-studio-for-object-detection/#google_vignette](https://tech.aru-zakki.com/how-to-use-label-studio-for-object-detection/#google_vignette)
    

### Weighted-Boxes-Fusionのインストール

- 今回は層化K分割交差検証法を用いたモデルの出力結果のアンサンブルを行うため、Weighted-Boxes-Fusionを使用します。
- このリポジトリは、複数の物体検出モデルからの検出ボックスをアンサンブルするための手法を提供するものです。
- 以下のコマンドでインストールが出来ます
    
    ```python
    pip install ensemble-boxes
    ```
    
- 以下のコマンドでインポートが出来ます
    
    ```python
    from ensemble_boxes import *
    ```
    

### 使用例

ボックスの座標は正規化された範囲（例：[0; 1]）であることが期待されます。順序：x1, y1, x2, y2。

- 以下は2つのモデルのボックスアンサンブルの例です。
    
    1つ目のモデルは5つのボックスを予測し、2つ目のモデルは4つのボックスを予測します。
    
    - 各ボックスの信頼度スコア（モデル1）：[0.9, 0.8, 0.2, 0.4, 0.7]
    - 各ボックスの信頼度スコア（モデル2）：[0.5, 0.8, 0.7, 0.3]
    - 各ボックスのラベル（クラス）（モデル1）：[0, 1, 0, 1, 1]
    - 各ボックスのラベル（クラス）（モデル2）：[1, 1, 1, 0]
- 1つ目のモデルの重みを1、2つ目のモデルの重みを1と設定します。
ボックスをマッチさせるためのIoUをiotu_thr = 0.5と設定します。
信頼度がskip_box_thr = 0.0001より低いボックスはスキップします。
- **以下の手法が実装されており、どれか一つを選択します。**
    
    **今回のプロジェクトでは重み付きボックス融合（WBF）を使用しています。**
    
    - **非最大抑制（NMS）**: 重なり合うボックスのうち、信頼度の高いものを残し、他を除去します。
    - **ソフトNMS**: ボックスを削除する代わりに、重なり合うボックスの信頼度を減少させます。
    - **非最大加重（NMW）**: 重なり合うボックスの重み付け平均を取ります。
    - **重み付きボックス融合（WBF）**: 上記手法よりも優れた結果を提供する新しい手法で、複数のモデルからのボックスを重み付けして融合します。

```python
from ensemble_boxes import *

boxes_list = [[
    [0.00, 0.51, 0.81, 0.91],
    [0.10, 0.31, 0.71, 0.61],
    [0.01, 0.32, 0.83, 0.93],
    [0.02, 0.53, 0.11, 0.94],
    [0.03, 0.24, 0.12, 0.35],
],[
    [0.04, 0.56, 0.84, 0.92],
    [0.12, 0.33, 0.72, 0.64],
    [0.38, 0.66, 0.79, 0.95],
    [0.08, 0.49, 0.21, 0.89],
]]
scores_list = [[0.9, 0.8, 0.2, 0.4, 0.7], [0.5, 0.8, 0.7, 0.3]]
labels_list = [[0, 1, 0, 1, 1], [1, 1, 1, 0]]
weights = [1, 1]

iou_thr = 0.5
skip_box_thr = 0.0001
sigma = 0.1

boxes, scores, labels = nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr)
boxes, scores, labels = soft_nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)
boxes, scores, labels = non_maximum_weighted(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
```

必要に応じて公式ページを確認してください

Weighted-Boxes-Fusion: [https://github.com/ZFTurbo/Weighted-Boxes-Fusion/tree/master](https://github.com/ZFTurbo/Weighted-Boxes-Fusion/tree/master)