import torch
import time
import numpy as np
from argparse import ArgumentParser, Namespace
from pathlib import Path
from tqdm import tqdm
import os

# モデルとデータローダーのインポート
from model.model import TwinLiteNetPlus
from demoDataset import LoadImages

def benchmark_real_data(args):
    # -------------------------------------------------------------------------
    # 1. 初期設定とモデルロード
    # -------------------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    print(f"Loading model config: {args.config}")
    model_args = Namespace(config=args.config)
    model = TwinLiteNetPlus(model_args)
    
    # 重みのロード
    if os.path.exists(args.weight):
        model.load_state_dict(torch.load(args.weight, map_location=device))
        print(f"Weights loaded from: {args.weight}")
    else:
        print(f"Error: Weight file not found at {args.weight}")
        return

    model.to(device)
    model.eval()

    # --- 精度設定 (デフォルト: FP32) ---
    # FP16を使用したい場合は、以下のコメントアウトを外してください
    # if device.type == 'cuda':
    #     model.half()
    #     print("Using Half Precision (FP16)")
    # else:
    print("Using Single Precision (FP32)")

    # -------------------------------------------------------------------------
    # 2. 画像データの準備 (メモリへのプリロード)
    # -------------------------------------------------------------------------
    print(f"\nLoading images from: {args.source}")
    try:
        # LoadImagesクラスで画像を読み込む
        dataset = LoadImages(args.source, img_size=args.img_size)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 推論用のテンソルリストを作成（I/O時間を計測から除外するため）
    input_batch = []
    print("Pre-processing images to memory...")
    
    for _, img, _, _, _ in dataset:
        # demo.py と同様の前処理
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # GPUへ転送・正規化
        img = img.to(device)

        # --- FP32 (デフォルト) ---
        img = img.float() / 255.0
        
        # --- FP16 (使用する場合はコメントアウトを外す) ---
        # img = img.half() / 255.0
        
        input_batch.append(img)

    if len(input_batch) == 0:
        print("No images found in the source directory.")
        return
    
    print(f"Loaded {len(input_batch)} unique images.")

    # -------------------------------------------------------------------------
    # 3. FPS計測ループ (100回実行)
    # -------------------------------------------------------------------------
    num_runs = 100  # 実行回数
    latencies = []

    print(f"\nStarting benchmark ({num_runs} iterations)...")
    
    # ウォームアップ (GPUの初期化コストを吸収)
    with torch.no_grad():
        _ = model(input_batch[0])
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # 計測開始
    for i in tqdm(range(num_runs), desc="Inferencing"):
        # 画像を選択 (リストを循環)
        img_tensor = input_batch[i % len(input_batch)]

        # --- 計測区間 開始 ---
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_start = time.time()

        with torch.no_grad():
            _ = model(img_tensor)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_end = time.time()
        # --- 計測区間 終了 ---

        latencies.append((t_end - t_start) * 1000) # ms単位で保存

    # -------------------------------------------------------------------------
    # 4. 結果の集計と表示
    # -------------------------------------------------------------------------
    latencies = np.array(latencies)
    
    # 外れ値（最初の数回や突発的なラグ）の影響を除くため、中央値も見る
    avg_latency = np.mean(latencies)
    median_latency = np.median(latencies)
    
    avg_fps = 1000.0 / avg_latency
    median_fps = 1000.0 / median_latency
    
    print("\n" + "="*50)
    print(f" BENCHMARK RESULTS (Real Data, {num_runs} runs)")
    print("-" * 50)
    print(f" Model: TwinLiteNet+ ({args.config})")
    print(f" Precision: FP32") # FP16使用時はここも変更してください
    print(f" Image Size: {args.img_size}x{args.img_size}")
    print("-" * 50)
    print(f" Average Latency: {avg_latency:.2f} ms")
    print(f" Average FPS:     {avg_fps:.2f} FPS")
    print("-" * 50)
    print(f" Median Latency:  {median_latency:.2f} ms")
    print(f" Median FPS:      {median_fps:.2f} FPS")
    print("="*50 + "\n")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='nano', choices=["nano", "small", "medium", "large"], help='Model configuration')
    parser.add_argument('--weight', type=str, default='./result/large/model_0.pth', help='Path to model weights')
    parser.add_argument('--source', type=str, default='inference/images', help='Path to image directory')
    parser.add_argument('--img-size', type=int, default=640, help='Inference size')
    
    opt = parser.parse_args()
    benchmark_real_data(opt)