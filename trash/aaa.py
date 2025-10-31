import argparse
import torch
from model.model import TwinLiteNetPlus # 元のモデル
# from model_modified.model import TwinLiteNetPlus as TwinLiteNetPlusModified # 変更後のモデル定義を別ファイルにした場合
# もしくは、変更したmodel.pyを一時的に別名でインポートするなど工夫が必要です
from utils import netParams

# --- 元のモデルのパラメータ数 ---
print("Calculating parameters for the original model...")
# ダミーの引数オブジェクトを作成 (config='nano'を指定)
parser_orig = argparse.ArgumentParser()
parser_orig.add_argument('--config', default='nano')
args_orig, _ = parser_orig.parse_known_args() # 未知の引数を無視

# 元のモデルをインスタンス化
model_original = TwinLiteNetPlus(args=args_orig)
params_original = netParams(model_original)
print(f"Original Nano Model Parameters: {params_original}")

# --- 変更後のモデルのパラメータ数 ---
print("\nCalculating parameters for the modified model...")
# 上記で model/model.py を変更した状態で再度読み込む必要があります
# 最も簡単なのは、一度Pythonを終了し、再度起動して以下の部分を実行することです。
# あるいは、変更を別ファイル (例: model_modified.py) に保存してインポートします。

# 再度、ダミー引数とモデルをインスタンス化（変更後のmodel.pyが読み込まれるように）
parser_mod = argparse.ArgumentParser()
parser_mod.add_argument('--config', default='nano')
args_mod, _ = parser_mod.parse_known_args()

# ----- 注意： -----
# ここで model.py の変更が反映されている必要があります。
# Pythonインタプリタを再起動するか、
# import importlib; import model.model; importlib.reload(model.model)
# のようにモジュールをリロードするテクニックを使う必要があります。
# 下記は変更が反映されている前提で進めます。
# ------------------
try:
    # 変更を反映させるため再度インポート (環境によってはリロードが必要)
    import importlib
    import model.model
    importlib.reload(model.model)
    from model.model import TwinLiteNetPlus as TwinLiteNetPlusModified

    model_modified = TwinLiteNetPlusModified(args=args_mod)
    params_modified = netParams(model_modified)
    print(f"Modified Nano Model Parameters: {params_modified}")

    # --- 削減数の計算 ---
    reduction = params_original - params_modified
    reduction_percent = (reduction / params_original) * 100
    print(f"\nParameter Reduction: {reduction} ({reduction_percent:.2f}%)")

except Exception as e:
    print("\nError calculating modified model parameters.")
    print("Please ensure 'model/model.py' has been modified and reloaded correctly.")
    print(f"Error details: {e}")