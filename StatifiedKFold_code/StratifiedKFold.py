import os
import glob
from decimal import Decimal, ROUND_HALF_UP
from ensemble_boxes import *

# 24時間を1分単位で分割した時刻文字列を生成する関数
def generate_time_strings():
    time_strings = []
    for hour in range(24):  # 0時から23時までのループ
        for minute in range(60):  # 0分から59分までのループ
            # 時刻文字列を 'hhmm00' の形式で生成
            time_string = f'{hour:02d}{minute:02d}00'
            time_strings.append(time_string)
    return time_strings

# 四捨五入を行う関数
def round_half_up(value, decimal_places):
    """
    指定された小数点以下の桁数で四捨五入する関数

    Parameters:
    value (float): 四捨五入する数値
    decimal_places (int): 小数点以下の桁数
    """
    context = Decimal('1.' + '0' * decimal_places)  # 丸めの精度を設定
    return float(Decimal(value).quantize(context, rounding=ROUND_HALF_UP))  # 指定された小数点以下の桁数で四捨五入

# ファイルを処理する関数
def process_file(file_path, skipped_files, convert_to_xyxy=True, convert_to_xyxy_rounding=False):
    """
    ファイルを読み込み、バウンディングボックスの情報を処理する関数

    Parameters:
    file_path (str): 処理するファイルのパス
    skipped_files (list): スキップされたファイルのリスト
    convert_to_xyxy (bool): (x_center, y_center, width, height) から (x1, y1, x2, y2) に変換するか
    convert_to_xyxy_rounding (bool): 変換時に四捨五入するか

    Returns:
    boxes (list): ボックスのリスト
    classes (list): クラスラベルのリスト
    scores (list): スコアのリスト
    """
    # ファイルの内容を読み込む
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 結果を格納するリストを初期化
    boxes = []
    classes = []
    scores = []

    # 各行を処理
    for line in lines:
        parts = line.split()  # 行を空白で分割
        if len(parts) < 6:  # 要素数の確認（クラス, x_center, y_center, width, height, スコアの合計6要素が必要）
            skipped_files.append(file_path)  # スキップされたファイルを記録
            continue  # 次の行へ
        cls = int(parts[0])  # クラスラベル
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        score = float(parts[5])  # スコア

        if convert_to_xyxy_rounding:
            # (x1, y1, x2, y2)形式に変換し、小数第三位に丸める
            x1 = round_half_up(x_center - width / 2, 3)
            y1 = round_half_up(y_center - height / 2, 3)
            x2 = round_half_up(x_center + width / 2, 3)
            y2 = round_half_up(y_center + height / 2, 3)
            boxes.append([x1, y1, x2, y2])
        elif convert_to_xyxy:
            # (x1, y1, x2, y2)形式に変換
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            boxes.append([x1, y1, x2, y2])
        else:
            # (x_center, y_center, width, height)形式のまま保持
            boxes.append([x_center, y_center, width, height])
        
        classes.append(cls)
        scores.append(score)

    return boxes, classes, scores

# 指定したルートディレクトリと日付に基づいてファイルを検索して処理する関数
def find_and_process_files(root_dirs, date):
    """
    指定したルートディレクトリと日付に基づいてファイルを検索して処理する関数

    Parameters:
    root_dirs (list): ルートディレクトリのリスト
    date (str): 日付（フォーマットは'YYYYMMDD'）

    Returns:
    all_data (dict): 処理された全データを格納する辞書
    skipped_files (list): スキップされたファイルのリスト
    """
    all_data = {}  # 全てのデータを格納する辞書
    skipped_files = []  # スキップされたファイルを記録するリスト

    # 時刻文字列を生成
    time_strings = generate_time_strings()

    # 各時刻に基づいたファイル名を作成し検索
    for time_string in time_strings:
        filename = f'{date}_{time_string}.txt'  # ファイル名を生成
        for root_dir in root_dirs:
            # ディレクトリ内の全てのサブディレクトリを含むパスを検索
            search_path = os.path.join(root_dir, '**', filename)

            for file_path in glob.glob(search_path, recursive=True):
                # ファイルを処理してボックス、クラス、スコアを取得
                boxes, classes, scores = process_file(file_path, skipped_files)  
                if filename not in all_data:
                    all_data[filename] = {'boxes': [], 'classes': [], 'scores': []}
                all_data[filename]['boxes'].append(boxes)
                all_data[filename]['classes'].append(classes)
                all_data[filename]['scores'].append(scores)

    return all_data, skipped_files

# ボックスを正規化する関数
def normalize_box(box):
    """
    ボックスの座標を0から1の範囲にクランプする関数

    Parameters:
    box (list): ボックスの座標リスト

    Returns:
    list: 正規化されたボックスの座標リスト
    """
    return [max(0, min(1, coord)) for coord in box]  # 座標を0から1の範囲にクランプ

# 各モデルのラベルディレクトリを設定
labels_dir1 = 'model1/labels'
labels_dir2 = 'model2/labels'
labels_dir3 = 'model3/labels'
labels_dir4 = 'model4/labels'
labels_dir5 = 'model5/labels'

# ルートディレクトリをリストにまとめる
root_dirs = [labels_dir1, labels_dir2, labels_dir3, labels_dir4, labels_dir5]
date = '20240512'  # 処理する画像の日付を指定

# ファイルを検索して処理
all_data, skipped_files = find_and_process_files(root_dirs, date)

# アンサンブルボックスのパラメータ設定
iou_thr = 0.5
skip_box_thr = 0.5
sigma = 0.1

# 結果を表示
for filename, data in all_data.items():
    boxes_list = data['boxes']  # 全てのボックス
    scores_list = data['scores']  # 全てのスコア
    labels_list = data['classes']  # 全てのクラスラベル
    weights = [1] * len(scores_list)  # 各モデルの重みを1に設定
    
    # ボックスを正規化
    boxes_list = [[normalize_box(box) for box in boxes] for boxes in boxes_list]
    
    # 重み付きボックス融合（WBF）を実行
    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list, weights=weights, 
        iou_thr=iou_thr, skip_box_thr=skip_box_thr
    )
    
    data = []
    
    # 中心座標とサイズを計算してリストに追加
    for box, score, label in zip(boxes, scores, labels):
        width = box[2] - box[0]
        height = box[3] - box[1]
        x_center = box[0] + width / 2
        y_center = box[1] + height / 2
        data.append([int(label), x_center, y_center, width, height, score])
    
    # 中心座標が近い場合にスコアが高い方のボックスを残す処理
    filtered_data = []
    while data:
        reference = data.pop(0)
        filtered_data.append(reference)
        data = [
            item for item in data 
            if not (
                abs(item[1] - reference[1]) <= 0.01 and 
                abs(item[2] - reference[2]) <= 0.01 and 
                item[5] <= reference[5]
            )
        ]

    # 出力ディレクトリを作成（存在しない場合）
    output_dir = "ensemble_labels"
    os.makedirs(output_dir, exist_ok=True)

    # フィルタリングされたデータをファイルに書き込み
    with open(os.path.join(output_dir, filename), "w", newline="\n") as f:
        for item in filtered_data:
            f.write(" ".join(map(str, item)) + "\n")
