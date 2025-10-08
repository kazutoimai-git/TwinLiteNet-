import torch
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser, Namespace
from pathlib import Path
import cv2
from model.model import TwinLiteNetPlus
from demoDataset import LoadImages, LoadStreams
from tqdm import tqdm
import time
import numpy as np
import os
import shutil
from ultralytics import YOLO

def show_seg_result(img, result):
    """
    白線検出の結果（セグメンテーションマスク）を画像に描画する関数。
    """
    palette = np.array([[0, 0, 0], [0, 255, 0], [255, 0, 0]], dtype=np.uint8)
    color_area = np.zeros((result[0].shape[0], result[0].shape[1], 3), dtype=np.uint8)
    color_area[result[0] == 1] = palette[1]
    color_area[result[1] == 1] = palette[2]

    color_seg = color_area[..., ::-1] # BGRに変換
    color_mask = np.mean(color_seg, 2)
    img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    img = img.astype(np.uint8)
    return img

def detect(args):
    # 1. モデルのロード
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    model_args = Namespace(config=args.config)
    model = TwinLiteNetPlus(model_args)
    model.load_state_dict(torch.load(args.weight, map_location=device))
    model.to(device)
    model.eval()

    yolo_model = YOLO('yolov8n.pt')
    yolo_model.to(device)

    # 2. データローダーと出力ディレクトリの設定
    # LoadImagesは画像と動画の両方を自動で判定してくれます
    dataset = LoadImages(args.source, img_size=args.img_size)
    save_dir = Path(args.save_dir)
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 動画書き出し用の変数を初期化
    vid_path, vid_writer = None, None

    # 3. パイプライン処理の実行
    for path, img, img_det, vid_cap, shapes in tqdm(dataset, desc="Processing"):
        # 画像の前処理 (TwinLiteNet+用)
        img = img.to(device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # --- パイプライン処理: 手順 1: 白線検出 ---
        da_seg_out, ll_seg_out = model(img)
        
        # 白線検出結果の後処理
        _, _, height, width = img.shape
        pad_w, pad_h = shapes[1][1]
        ratio = shapes[1][0][1]
        
        da_predict = da_seg_out[:, :, int(pad_h):int(height-pad_h), int(pad_w):int(width-pad_w)]
        da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
        
        ll_predict = ll_seg_out[:, :, int(pad_h):int(height-pad_h), int(pad_w):int(width-pad_w)]
        ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()

        # オリジナルのフレームに白線検出の結果を描画
        img_vis = show_seg_result(img_det.copy(), (da_seg_mask, ll_seg_mask))

        # --- パイプライン処理: 手順 2: 物体検出 ---
        yolo_results = yolo_model(img_vis, verbose=False)
        
        # 物体検出の結果を描画
        annotated_frame = yolo_results[0].plot()

        # 4. 結果の保存
        if dataset.mode == 'images':
            save_path = str(save_dir / Path(path).name)
            cv2.imwrite(save_path, annotated_frame)
        elif dataset.mode == 'video':
            save_path = str(save_dir / Path(path).name)
            if vid_path != save_path:  # 新しいビデオの最初のフレームの場合
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # 前のビデオライターを解放
                
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer.write(annotated_frame)

    if isinstance(vid_writer, cv2.VideoWriter):
        vid_writer.release()

    print(f"Successfully processed. Results are saved in: {save_dir}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--weight', type=str, default='testv3/nano.pth', help='TwinLiteNet+ model.pth path')
    parser.add_argument('--source', type=str, default='inference/images', help='Input file or folder')
    parser.add_argument('--img-size', type=int, default=640, help='Inference size (pixels)')
    parser.add_argument('--config', type=str, required=True, choices=["nano", "small", "medium", "large"], help='TwinLiteNet+ model configuration')
    parser.add_argument('--save-dir', type=str, default='inference/pipeline_output', help='Directory to save results')
    opt = parser.parse_args()
    
    with torch.no_grad():
        detect(opt)