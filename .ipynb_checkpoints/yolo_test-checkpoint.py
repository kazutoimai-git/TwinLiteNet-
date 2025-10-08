import os
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO
from tqdm import tqdm

def main():
    # 1. モデルのロード
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = YOLO('yolov8n.pt')
    model.to(device)

    # 2. 入出力ディレクトリの設定
    input_dir = Path('inference/images')
    output_dir = Path('inference/yolo_output')

    # 出力ディレクトリがなければ作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. 画像ファイルの一覧を取得
    image_files = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.jpeg')) + list(input_dir.glob('*.png'))

    print(f"Found {len(image_files)} images in {input_dir}")

    # 4. 各画像に対して物体検出を実行
    for image_path in tqdm(image_files, desc="Processing images"):
        # モデルで物体検出を実行
        results = model(image_path, verbose=False)
        
        # 5. 結果の描画と保存
        # results[0]は最初の画像に対する結果オブジェクト
        # .plot()関数でバウンディングボックスが描画された画像(NumPy配列)を取得
        annotated_frame = results[0].plot()

        # OpenCVを使って画像を保存
        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), annotated_frame)

    print(f"Successfully processed all images. Results are saved in: {output_dir}")

if __name__ == '__main__':
    main()