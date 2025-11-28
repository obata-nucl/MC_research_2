import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

from src.dataset import IBM2Dataset
from src.model import IBM2FlexibleNet, IBM2PESDecoder
from src.utils import load_config, set_seed
from src.visualize import IBM2Visualizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optuna", action="store_true", help="Visualize Optuna best result")
    # type引数で描画対象を選べるようにする (デフォルトは全部)
    parser.add_argument("--type", type=str, default="all", choices=["pes", "params", "loss", "all"])
    args = parser.parse_args()

    # 1. 設定ロード
    cfg = load_config()
    device = torch.device(cfg.get("device", "cpu"))
    
    # ディレクトリ設定
    output_dir = cfg["dirs"]["output_dir"]
    model_dir = output_dir / "models"
    plot_dir = output_dir / "plots"
    
    # Visualizer準備
    vis = IBM2Visualizer(save_dir=plot_dir)

    # ==========================================
    # 2. ファイルパスの決定 (通常 vs Optuna)
    # ==========================================
    if args.optuna:
        print("Mode: Visualizing OPTUNA BEST result")
        model_path = model_dir / "optuna_best_model.pth"
        history_path = model_dir / "optuna_best_history.csv"
        config_path = model_dir / "optuna_best_config.yaml"
        prefix = "optuna_"
        if config_path.exists():
            print(f"Loading model config from {config_path}")
            with open(config_path, 'r') as f:
                model_config = yaml.safe_load(f)
        else:
            print("Warning: Config file not found. Using default config (Running may fail).")
            model_config = cfg["default"]["nn"].copy()
            model_config["fixed_chi_pi"] = cfg["nuclei"]["fixed_chi_pi"]
    else:
        print("Mode: Visualizing NORMAL training result")
        model_path = model_dir / "best_model.pth"
        history_path = model_dir / "training_history.csv" # train.pyの保存名に合わせる
        prefix = "normal_"
        
        # 通常時はデフォルト設定を使用
        model_config = cfg["default"]["nn"].copy()
        model_config["fixed_chi_pi"] = cfg["nuclei"]["fixed_chi_pi"]

    # ==========================================
    # 3. 学習曲線 (Loss) のプロット
    # ==========================================
    if args.type in ["loss", "all"]:
        if history_path.exists():
            print(f"Plotting Learning Curve from {history_path} ...")
            df = pd.read_csv(history_path)
            
            vis.plot_loss_history(
                train_loss=df["train_loss"],
                val_loss=df["val_loss"],
                filename=f"{prefix}learning_curve.png"
            )
        else:
            print(f"Warning: History file not found at {history_path}")

    # Lossのプロットだけならここで終了しても良いが、PES等も見る場合は続行
    if args.type == "loss":
        return

    # ==========================================
    # 4. モデルとデータの準備 (PES/Params用)
    # ==========================================
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return

    # Dataset (全データ)
    dataset = IBM2Dataset(cfg)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Model構築
    # model_config は上で設定済み
    model = IBM2FlexibleNet(model_config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Decoder
    decoder = IBM2PESDecoder(beta_f_grid=dataset.beta_grid).to(device)

    # ==========================================
    # 5. 推論 & プロットループ
    # ==========================================
    n_list = []
    params_history = {"epsilon": [], "kappa": [], "chi_nu": [], "chi_pi": [], "C_beta": []}
    pes_data_list = [] # まとめてプロット用
    
    print("Generating PES & Parameter plots...")
    
    with torch.no_grad():
        for inputs, targets, n_pi, n_nu in loader:
            inputs = inputs.to(device)
            n_pi = n_pi.to(device)
            n_nu = n_nu.to(device)
            
            # 予測
            params = model(inputs)
            preds = decoder(params, n_pi, n_nu)
            
            # 値取り出し
            p = params.cpu().numpy()[0] # [eps, kap, chi_nu, chi_pi, C_beta]
            n_val = int(inputs[0, 0].item())
            
            # --- PESデータ収集 ---
            if args.type in ["pes", "all"]:
                target_y = targets[0].cpu().numpy()
                pred_y = preds[0].cpu().numpy()
                
                pes_data_list.append({
                    "N": n_val,
                    "target": target_y,
                    "pred": pred_y
                })

            # --- パラメータ収集 ---
            n_list.append(n_val)
            params_history["epsilon"].append(p[0])
            params_history["kappa"].append(p[1])
            params_history["chi_nu"].append(p[2])
            params_history["chi_pi"].append(p[3])
            params_history["C_beta"].append(p[4])

    # --- PESまとめてプロット ---
    if args.type in ["pes", "all"] and pes_data_list:
        vis.plot_all_pes(
            dataset.beta_grid, 
            pes_data_list, 
            filename=f"{prefix}PES_all.png"
        )

    # --- パラメータ推移プロット ---
    if args.type in ["params", "all"]:
        vis.plot_parameters_evolution(
            n_list, params_history, 
            filename=f"{prefix}params_trend.png"
        )
        
    print("All plots generated.")

if __name__ == "__main__":
    main()