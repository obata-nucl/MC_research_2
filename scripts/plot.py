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
    parser.add_argument("--type", type=str, default="all", choices=["pes", "params", "loss", "spectra", "ratio", "all"])
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

    mode_name = "optuna" if args.optuna else "normal"
    prefix = f"{mode_name}_"

    # ==========================================
    # 2. ファイルパスの決定 (通常 vs Optuna)
    # ==========================================
    if args.optuna:
        print("Mode: Visualizing OPTUNA BEST result")
        model_path = model_dir / "optuna_best_model.pth"
        history_path = model_dir / "optuna_best_history.csv"
        config_path = model_dir / "optuna_best_config.yaml"
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
        
        # 通常時はデフォルト設定を使用
        model_config = cfg["default"]["nn"].copy()
        model_config["fixed_chi_pi"] = cfg["nuclei"]["fixed_chi_pi"]

    analysis_file = output_dir / f"analysis_{mode_name}.csv"
    analysis_df = None
    allowed_n_values = None
    if analysis_file.exists():
        try:
            analysis_df = pd.read_csv(analysis_file)
            if "N" in analysis_df.columns:
                allowed_n_values = set(analysis_df["N"].astype(int).tolist())
                print(f"Found {len(allowed_n_values)} nuclei in {analysis_file.name}.")
            else:
                print(f"Warning: Column 'N' missing in {analysis_file}. Cannot filter nuclei.")
                analysis_df = None
        except Exception as exc:
            print(f"Warning: Failed to read {analysis_file}: {exc}")
            analysis_df = None
    else:
        print(f"Warning: {analysis_file} not found. Run analyze.py to generate it.")

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

            if allowed_n_values and n_val not in allowed_n_values:
                continue
            
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
    elif args.type in ["pes", "all"]:
        print("Warning: No nuclei available for PES plot.")

    # --- パラメータ推移プロット ---
    if args.type in ["params", "all"] and n_list:
        vis.plot_parameters_evolution(
            n_list, params_history, 
            filename=f"{prefix}params_trend.png"
        )
    elif args.type in ["params", "all"]:
        print("Warning: No nuclei available for parameter plot.")

    # ==========================================
    # 6. Spectra & Ratio (from Analysis results)
    # ==========================================
    if args.type in ["spectra", "ratio", "all"]:
        # Load Analysis Results
        expt_file = cfg["dirs"]["raw_dir"] / "expt.csv"
        n_min = cfg["nuclei"].get("n_min")
        n_max = cfg["nuclei"].get("n_max")
        
        if analysis_df is None and analysis_file.exists():
            try:
                analysis_df = pd.read_csv(analysis_file)
            except Exception as exc:
                print(f"Warning: Failed to read {analysis_file}: {exc}")
                analysis_df = None

        if analysis_df is not None and expt_file.exists():
            print(f"Plotting Spectra & Ratio from {analysis_file} ...")
            pred_df = analysis_df.copy()
            expt_df = pd.read_csv(expt_file)

            if "N" in pred_df.columns and n_min is not None and n_max is not None:
                pred_df = pred_df[(pred_df["N"] >= n_min) & (pred_df["N"] <= n_max)]
            if "N" in expt_df.columns and n_min is not None and n_max is not None:
                expt_df = expt_df[(expt_df["N"] >= n_min) & (expt_df["N"] <= n_max)]
            
            if args.type in ["spectra", "all"]:
                vis.plot_spectra(pred_df, expt_df, filename=f"{prefix}spectra.png")
                
            if args.type in ["ratio", "all"]:
                vis.plot_ratio(pred_df, expt_df, filename=f"{prefix}ratio.png")
        else:
            if analysis_df is None:
                print(f"Warning: Analysis file not available at {analysis_file}. Run analyze.py first.")
            if not expt_file.exists():
                print(f"Warning: Expt file not found at {expt_file}")
        
    print("All plots generated.")

if __name__ == "__main__":
    main()