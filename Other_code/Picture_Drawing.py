import cv2
import os
import shutil

def draw_bounding_boxes(img_path, file_path, output_path, relative_coordinates=True):
    """
    画像にバウンディングボックスを描画し、結果を保存する関数

    Parameters:
    img_path (str): 画像ファイルのパス
    file_path (str): バウンディングボックス情報が記載されたテキストファイルのパス
    output_path (str): 出力画像の保存パス
    relative_coordinates (bool): バウンディングボックスの座標が相対座標か絶対座標かを指定
    """
    try:
        # 画像ファイル名を取得
        img_name = os.path.basename(img_path)

        # 画像を読み込む
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read image {img_path}")
            return

        # テキストファイルを開いて各行を読み込む
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist")
            return

        with open(file_path, 'r') as file:
            lines = file.readlines()

        # 各物体の情報をループで処理
        for line in lines:
            try:
                # テキストファイルからクラスID、中心座標、幅、高さ、確信度を読み込む
                class_id, x_center, y_center, width, height, confidence = map(float, line.split())
            except ValueError:
                print(f"Error: Could not parse line {line.strip()}")
                continue

            if relative_coordinates:
                # 相対座標を絶対座標に変換
                abs_x = int(x_center * img.shape[1])
                abs_y = int(y_center * img.shape[0])
                abs_w = int(width * img.shape[1])
                abs_h = int(height * img.shape[0])

                # バウンディングボックスを描画（青紫色）
                cv2.rectangle(img, (abs_x - abs_w // 2, abs_y - abs_h // 2), (abs_x + abs_w // 2, abs_y + abs_h // 2), (255, 0, 255), 2)

                # テキストを描画する座標を計算
                text_x = abs_x - abs_w // 2
                text_y = abs_y - abs_h // 2 - 10  # テキストをバウンディングボックスの上に表示するために少し上にオフセット
            else:
                # 既に絶対座標が指定されている場合
                abs_x = int(x_center)
                abs_y = int(y_center)
                abs_w = int(width)
                abs_h = int(height)

                # バウンディングボックスを描画（青紫色）
                cv2.rectangle(img, (abs_x, abs_y), (abs_x + abs_w, abs_y + abs_h), (255, 0, 255), 2)

                # テキストを描画する座標を計算
                text_x = abs_x
                text_y = abs_y - 10  # テキストをバウンディングボックスの上に表示するために少し上にオフセット

            # 確信度とクラスIDをバウンディングボックスに表示するテキスト（黄色）
            text = f"ID:{int(class_id)} Conf:{confidence:.2f}"

            # テキストを画像に描画
            cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # 描画した画像を表示
        cv2.imshow('Image with Bounding Boxes', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 描画した画像を保存
        cv2.imwrite(output_path, img)
        print(f"Image saved to {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

def process_directory(img_dir, txt_dir, output_dir, relative_coordinates=True):
    """
    指定されたディレクトリ内の画像とテキストファイルを処理する関数

    Parameters:
    img_dir (str): 画像ファイルが格納されたディレクトリのパス
    txt_dir (str): テキストファイルが格納されたディレクトリのパス
    output_dir (str): 出力画像の保存ディレクトリのパス
    relative_coordinates (bool): バウンディングボックスの座標が相対座標か絶対座標かを指定
    """
    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 画像ファイルのリストを取得（.jpgおよび.pngファイル）
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]
    # テキストファイルのリストを取得（.txtファイル）
    txt_files = [f for f in os.listdir(txt_dir) if f.endswith('.txt')]

    # 処理済みのファイル名のセット
    processed_files = set()

    # 画像ファイルをループで処理
    for img_file in img_files:
        base_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(img_dir, img_file)
        txt_path = os.path.join(txt_dir, base_name + '.txt')
        output_path = os.path.join(output_dir, img_file)
        if os.path.exists(txt_path):
            # 対応するテキストファイルが存在する場合はバウンディングボックスを描画
            draw_bounding_boxes(img_path, txt_path, output_path, relative_coordinates)
            processed_files.add(base_name)
        else:
            # 対応するテキストファイルがない場合はそのままコピー
            shutil.copy(img_path, output_path)
            print(f"Image copied to {output_path} (no corresponding txt file)")

    # テキストファイルをループで処理
    for txt_file in txt_files:
        base_name = os.path.splitext(txt_file)[0]
        if base_name not in processed_files:
            txt_path = os.path.join(txt_dir, txt_file)
            # 対応する画像ファイルがない場合はそのままコピー（任意の処理を追加可能）
            output_path = os.path.join(output_dir, txt_file)
            shutil.copy(txt_path, output_path)
            print(f"Text file copied to {output_path} (no corresponding image file)")

# 画像とテキストファイルのディレクトリ
img_dir = 'images'
txt_dir = 'labels'
# 出力ディレクトリ
output_dir = 'output'

# ディレクトリ内のファイルを処理
process_directory(img_dir, txt_dir, output_dir, relative_coordinates=True)
