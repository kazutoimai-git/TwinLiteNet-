import torch
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser, Namespace
from pathlib import Path
import time
import numpy as np
from tqdm import tqdm

# 既存のモジュールをインポート
from model.model import TwinLiteNetPlus
from demoDataset import LoadImages

def benchmark(args):
    # 1. デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. モデルのロード
    # Namespaceを使ってargs.configをモデルに渡せるように整形
    model_args = Namespace(config=args.config)
    model = TwinLiteNetPlus(model_args)
    
    # 重みのロード
    print(f"Loading weights from: {args.weight}")
    try:
        model.load_state_dict(torch.load(args.weight, map_location=device))
    except FileNotFoundError:
        print("Error: Weight file not found.")
        return

    model.to(device)
    model.eval()

    # ハーフ精度の設定（demo.pyに準拠、高速化のため）
    half = device.type != 'cpu'  # GPUならFP16を使用
    if half:
        model.half()

    # 3. データローダーの設定
    print(f"Loading data from: {args.source}")
    try:
        dataset = LoadImages(args.source, img_size=args.img_size)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 4. ウォームアップ (GPUの初期化オーバーヘッドを計測から除外するため)
    print("Warming up...")
    img_warmup = torch.zeros((1, 3, args.img_size, args.img_size), device=device)
    if half:
        img_warmup = img_warmup.half()
    for _ in range(10):
        _ = model(img_warmup)

    # 5. 計測開始
    print("Starting benchmark...")
    latencies = []

    # tqdmで進捗表示
    for path, img, img_det, vid_cap, shapes in tqdm(dataset, desc="Inferencing"):
        
        # 前処理 (demo.pyと同様)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        if half:
            img = img.cuda().half() / 255.0
        else:
            img = img.cuda().float() / 255.0

        # CUDAの同期（正確な時間計測のため必須）
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        t_start = time.time()

        # 推論実行
        with torch.no_grad():
            da_seg_out, ll_seg_out = model(img)

        # CUDAの同期
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        t_end = time.time()

        # レイテンシを記録 (秒 -> ミリ秒)
        latencies.append((t_end - t_start) * 1000)

    # 6. 結果の計算と表示
    if len(latencies) > 0:
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        fps = 1000.0 / avg_latency

        print("\n" + "="*40)
        print(f" Model Config : {args.config}")
        print(f" Input Source : {args.source}")
        print(f" Total Frames : {len(latencies)}")
        print("-" * 40)
        print(f" Average Latency : {avg_latency:.2f} ms ± {std_latency:.2f} ms")
        print(f" Average FPS     : {fps:.2f} FPS")
        print("="*40 + "\n")
    else:
        print("No images/frames found to process.")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--weight', type=str, default='./result/large/model_0.pth', help='Path to model weights')
    # imagesフォルダとmovieフォルダの両方を一度に指定したい場合は親ディレクトリを指定するか、個別に実行してください
    parser.add_argument('--source', type=str, default='inference/images', help='Path to image/video file or directory')
    parser.add_argument('--img-size', type=int, default=640, help='Inference size (pixels)')
    parser.add_argument('--config', type=str, default='nano', choices=["nano", "small", "medium", "large"], help='Model configuration')
    
    opt = parser.parse_args()
    benchmark(opt)