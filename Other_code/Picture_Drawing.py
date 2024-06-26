import cv2
import os

def draw_bounding_boxes(img_path, file_path, output_path):
    try:
        # 画像ファイル名
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
                class_id, x_center, y_center, width, height, confidence = map(float, line.split())
            except ValueError:
                print(f"Error: Could not parse line {line.strip()}")
                continue

            # 相対座標を絶対座標に変換
            abs_x = int(x_center * img.shape[1])
            abs_y = int(y_center * img.shape[0])
            abs_w = int(width * img.shape[1])
            abs_h = int(height * img.shape[0])

            # バウンディングボックスを描画（青紫色）
            cv2.rectangle(img, (abs_x - abs_w // 2, abs_y - abs_h // 2), (abs_x + abs_w // 2, abs_y + abs_h // 2), (255, 0, 255), 2)

            # 確信度とクラスIDをバウンディングボックスに表示するテキスト（黄色）
            text = f"ID:{int(class_id)} Conf:{confidence:.2f}"

            # テキストを描画する座標
            text_x = abs_x - abs_w // 2
            text_y = abs_y - abs_h // 2 - 10  # テキストをバウンディングボックスの上に表示するために少し上にオフセット

            # テキストを描画
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

# 画像のパス
img_path = '20240513_000000.jpg'
# テキストファイルのパス
file_path = '20240513_000000.txt'
# 出力パス
output_path = f'doc/{os.path.basename(img_path)}'

# 関数の呼び出し
draw_bounding_boxes(img_path, file_path, output_path)
