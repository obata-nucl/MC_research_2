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
            model_config["fixed_C_beta"] = cfg["nuclei"]["fixed_C_beta"]
    else:
        print("Mode: Visualizing NORMAL training result")
        model_path = model_dir / "best_model.pth"
        history_path = model_dir / "training_history.csv" # train.pyの保存名に合わせる
        
        # 通常時はデフォルト設定を使用
        model_config = cfg["default"]["nn"].copy()
        model_config["fixed_C_beta"] = cfg["nuclei"]["fixed_C_beta"]

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
    z_list = [] # Zのリストを追加
    params_history = {"epsilon": [], "kappa": [], "chi_nu": [], "chi_pi": [], "C_beta": []}
    pes_data_list = [] # まとめてプロット用
    
    print("Generating PES & Parameter plots...")
    
    with torch.no_grad():
        for i, (inputs, targets, n_pi, n_nu) in enumerate(loader):
            inputs = inputs.to(device)
            n_pi = n_pi.to(device)
            n_nu = n_nu.to(device)
            
            # 予測
            params = model(inputs)
            preds = decoder(params, n_pi, n_nu)
            
            # 値取り出し
            p = params.cpu().numpy()[0] # [eps, kap, chi_pi, chi_nu, C_beta]
            
            # Nを取得 (shuffle=Falseなのでindexでアクセス可能)
            n_val = dataset.data[i]["N"]
            z_val = dataset.data[i]["Z"]

            if allowed_n_values and n_val not in allowed_n_values:
                continue
            
            # --- PESデータ収集 ---
            if args.type in ["pes", "all"]:
                target_y = targets[0].cpu().numpy()
                pred_y = preds[0].cpu().numpy()
                
                pes_data_list.append({
                    "Z": z_val,
                    "N": n_val,
                    "target": target_y,
                    "pred": pred_y
                })

            # --- パラメータ収集 ---
            n_list.append(n_val)
            z_list.append(z_val) # Zを保存
            params_history["epsilon"].append(p[0])
            params_history["kappa"].append(p[1])
            params_history["chi_pi"].append(p[2])
            params_history["chi_nu"].append(p[3])
            params_history["C_beta"].append(p[4])

    # ==========================================
    # 6. プロット (Zごとにフォルダ分け)
    # ==========================================
    
    # 解析データのロード (Spectra/Ratio用)
    pred_df_all = None
    expt_df_all = None
    
    if args.type in ["spectra", "ratio", "all"]:
        expt_file = cfg["dirs"]["raw_dir"] / "expt.csv"
        
        # analysis_dfが未ロードならロード
        if analysis_df is None and analysis_file.exists():
            try:
                analysis_df = pd.read_csv(analysis_file)
            except Exception as exc:
                print(f"Warning: Failed to read {analysis_file}: {exc}")
                analysis_df = None
        
        pred_df_all = analysis_df
        
        if expt_file.exists():
            expt_df_all = pd.read_csv(expt_file)
        else:
            print(f"Warning: Expt file not found at {expt_file}")

    # Zのリストを取得
    unique_zs = sorted(list(set(z_list)))
    # もしPES/Paramsのデータがない場合でも、Analysis結果があればそこからZを取得
    if not unique_zs and pred_df_all is not None and "Z" in pred_df_all.columns:
        unique_zs = sorted(pred_df_all["Z"].unique())

    print(f"Generating plots for Z: {unique_zs}")

    for z in unique_zs:
        print(f"--- Plotting for Z={z} ---")
        z_plot_dir = plot_dir / str(z)
        z_vis = IBM2Visualizer(save_dir=z_plot_dir)
        
        # 1. PES
        z_pes_data = [d for d in pes_data_list if d.get("Z") == z]
        if args.type in ["pes", "all"] and z_pes_data:
            z_vis.plot_all_pes(
                dataset.beta_grid, 
                z_pes_data, 
                filename=f"{prefix}PES_all.png"
            )

        # 2. Params
        z_indices = [i for i, val in enumerate(z_list) if val == z]
        if args.type in ["params", "all"] and z_indices:
            z_n_list = [n_list[i] for i in z_indices]
            z_z_list_sub = [z_list[i] for i in z_indices]
            z_params = {k: [v[i] for i in z_indices] for k, v in params_history.items()}
            
            z_vis.plot_parameters_evolution(
                z_n_list, z_z_list_sub, z_params, 
                filename=f"{prefix}params_trend.png"
            )

        # 3. Spectra & Ratio
        if args.type in ["spectra", "ratio", "all"]:
            # Filter Pred
            z_pred_df = None
            if pred_df_all is not None and "Z" in pred_df_all.columns:
                z_pred_df = pred_df_all[pred_df_all["Z"] == z]
            
            # Filter Expt
            z_expt_df = pd.DataFrame()
            if expt_df_all is not None:
                if "Z" in expt_df_all.columns:
                    z_expt_df = expt_df_all[expt_df_all["Z"] == z]
                else:
                    # Zカラムがない場合はフィルタリングできないため、そのまま使うか、警告を出す
                    # ここでは安全のため空にするか、あるいは全データを使うか...
                    # ユーザーの意図としてはZごとに分けたいはずなので、Zカラムがないと困る。
                    # とりあえずそのまま渡してみる（もしNが被ってなければ動く）
                    # しかし、他のZのデータが混ざるとおかしくなる。
                    # ここでは「Zカラムがあればフィルタ、なければ空」とするのが安全。
                    pass

            if z_pred_df is not None and not z_pred_df.empty:
                if args.type in ["spectra", "all"]:
                    z_vis.plot_spectra(z_pred_df, z_expt_df, filename=f"{prefix}spectra.png")
                
                if args.type in ["ratio", "all"]:
                    z_vis.plot_ratio(z_pred_df, z_expt_df, filename=f"{prefix}ratio.png")

    print("All plots generated.")

if __name__ == "__main__":
    main()