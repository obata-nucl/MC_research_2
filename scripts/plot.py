import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

# Allow direct execution: `python scripts/plot.py ...`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import IBM2Dataset
from src.model import IBM2FlexibleNet, IBM2PESDecoder
from src.utils import load_config, set_seed
from src.visualize import IBM2Visualizer


def _resolve_experiment_dir(dir_arg, output_dir):
    raw = Path(dir_arg)
    candidates = [
        raw,
        output_dir.parent / raw,
        Path.cwd() / raw,
    ]
    for cand in candidates:
        if cand.exists() and cand.is_dir():
            return cand.resolve()
    return None


def _load_allowed_n_values(analysis_file):
    if not analysis_file.exists():
        return None
    try:
        df = pd.read_csv(analysis_file)
    except Exception as exc:
        print(f"Warning: Failed to read {analysis_file}: {exc}")
        return None

    if "N" not in df.columns:
        print(f"Warning: Column 'N' missing in {analysis_file}. Cannot filter nuclei.")
        return None
    return set(df["N"].astype(int).tolist())


def _build_model_label(model_config, exp_dir):
    c_beta = model_config.get("fixed_C_beta")
    if c_beta is None:
        return f"IBM-2 ({exp_dir.name})"
    return f"IBM-2 ($C_{{\\beta}}={c_beta:.1f}$)"


def _spectra_panel_labels(c_beta):
    if c_beta is None:
        left = "(a) IBM-2"
    else:
        left = rf"(a) IBM-2 ($C_{{\beta}}={float(c_beta):.1f}$)"
    right = "(b) Expt."
    return left, right


def _z_panel_label(z):
    mapping = {
        60: "(a) Nd",
        62: "(b) Sm",
        64: "(c) Gd",
    }
    return mapping.get(z)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optuna", action="store_true", help="Visualize Optuna best result")
    # type引数で描画対象を選べるようにする (デフォルトは全部)
    parser.add_argument("--type", type=str, default="all", choices=["pec", "pes", "params", "loss", "spectra", "ratio", "all"])
    parser.add_argument(
        "--element",
        type=str,
        default="all",
        choices=["all", "Nd", "Sm", "Gd"],
        help="Plot only selected element (Nd=60, Sm=62, Gd=64)."
    )
    parser.add_argument(
        "--z",
        type=int,
        default=None,
        help="Plot only selected proton number Z (overrides --element). Example: --z 62 for Sm."
    )
    parser.add_argument(
        "--compare_dirs",
        nargs="+",
        default=None,
        help="Compare multiple experiment dirs in one PES figure. Example: --compare_dirs 4 25"
    )
    parser.add_argument(
        "--compare_filename",
        type=str,
        default="PES_compare_cbeta.pdf",
        help="Output filename in compare mode."
    )
    args = parser.parse_args()

    if args.type == "pec":
        args.type = "pes"

    element_to_z = {"Nd": 60, "Sm": 62, "Gd": 64}
    selected_z = args.z if args.z is not None else element_to_z.get(args.element)

    # 1. 設定ロード
    cfg = load_config()
    device = torch.device(cfg.get("device", "cpu"))

    # Smなど特定元素のみ指定された場合は、Dataset読み込み範囲をZ固定にする
    if selected_z is not None:
        cfg["nuclei"]["z_min"] = selected_z
        cfg["nuclei"]["z_max"] = selected_z
        cfg["nuclei"]["z_step"] = 1
        print(f"Dataset Z-range fixed to Z={selected_z}")
    
    # ディレクトリ設定
    output_dir = cfg["dirs"]["output_dir"]
    model_dir = output_dir / "models"
    plot_dir = output_dir / "plots"
    
    # Visualizer準備
    vis = IBM2Visualizer(save_dir=plot_dir)

    mode_name = "optuna" if args.optuna else "normal"
    prefix = f"{mode_name}_"

    if args.compare_dirs:
        if args.type not in ["pes", "all"]:
            print("Warning: compare mode supports only --type pes/pec/all.")
            return

        print(f"Mode: Comparing C_beta models from dirs: {args.compare_dirs}")

        dataset = IBM2Dataset(cfg)
        decoder = IBM2PESDecoder(beta_f_grid=dataset.beta_grid).to(device)

        compare_records = {}
        model_labels = []

        for dir_arg in args.compare_dirs:
            exp_dir = _resolve_experiment_dir(dir_arg, output_dir)
            if exp_dir is None:
                print(f"Warning: compare dir not found: {dir_arg}")
                continue

            model_dir_cmp = exp_dir / "models"
            model_path = model_dir_cmp / ("optuna_best_model.pth" if args.optuna else "best_model.pth")
            config_path = model_dir_cmp / ("optuna_best_config.yaml" if args.optuna else "best_config.yaml")
            analysis_file = exp_dir / f"analysis_{mode_name}.csv"

            if not model_path.exists():
                print(f"Warning: model file not found: {model_path}")
                continue

            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    model_config = yaml.safe_load(f)
            else:
                print(f"Warning: config file not found: {config_path}. Fallback to default config.")
                model_config = cfg["default"]["nn"].copy()
                model_config["fixed_C_beta"] = cfg["nuclei"].get("fixed_C_beta")

            label = _build_model_label(model_config, exp_dir)
            if label in model_labels:
                label = f"{label}#{len(model_labels)}"
            model_labels.append(label)

            allowed_n_values_cmp = _load_allowed_n_values(analysis_file)
            if allowed_n_values_cmp is not None:
                print(f"{exp_dir.name}: filter by {len(allowed_n_values_cmp)} nuclei from {analysis_file.name}")

            model = IBM2FlexibleNet(model_config).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            with torch.no_grad():
                for i in range(len(dataset)):
                    inputs, target, n_pi, n_nu = dataset[i]
                    z_val = dataset.data[i]["Z"]
                    n_val = dataset.data[i]["N"]

                    if selected_z is not None and z_val != selected_z:
                        continue
                    if allowed_n_values_cmp and n_val not in allowed_n_values_cmp:
                        continue

                    inputs = inputs.unsqueeze(0).to(device)
                    n_pi = n_pi.unsqueeze(0).to(device)
                    n_nu = n_nu.unsqueeze(0).to(device)

                    params = model(inputs)
                    preds = decoder(params, n_pi, n_nu)

                    key = (z_val, n_val)
                    if key not in compare_records:
                        compare_records[key] = {
                            "Z": z_val,
                            "N": n_val,
                            "target": target.cpu().numpy(),
                            "preds": {}
                        }

                    compare_records[key]["preds"][label] = preds[0].cpu().numpy()

        if len(compare_records) == 0:
            print("Error: No comparison data collected. Check --compare_dirs and model files.")
            return

        compare_plot_root = plot_dir / "compare_cbeta"
        unique_zs_cmp = sorted({v["Z"] for v in compare_records.values()})
        print(f"Generating comparison PES for Z: {unique_zs_cmp}")

        for z in unique_zs_cmp:
            z_plot_dir = compare_plot_root / str(z)
            z_vis = IBM2Visualizer(save_dir=z_plot_dir)

            z_data = [v for v in compare_records.values() if v["Z"] == z]
            labels_for_z = [
                label for label in model_labels
                if any(label in d.get("preds", {}) for d in z_data)
            ]
            if len(labels_for_z) == 0:
                continue

            z_vis.plot_all_pes_compare_models(
                dataset.beta_grid,
                z_data,
                model_labels=labels_for_z,
                filename=f"{prefix}{args.compare_filename}"
            )

        print("All comparison PES plots generated.")
        return

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
                lr=df.get("lr"),
                filename=f"{prefix}learning_curve.pdf"
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
    params_history = {"epsilon": [], "kappa": [], "chi_nu": [], "chi_pi": []}
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

            if selected_z is not None and z_val != selected_z:
                continue

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

    # ==========================================
    # 6. プロット (Zごとにフォルダ分け)
    # ==========================================
    
    # 解析データのロード (Spectra/Ratio用)
    pred_df_all = None
    
    if args.type in ["spectra", "ratio", "all"]:
        # analysis_dfが未ロードならロード
        if analysis_df is None and analysis_file.exists():
            try:
                analysis_df = pd.read_csv(analysis_file)
            except Exception as exc:
                print(f"Warning: Failed to read {analysis_file}: {exc}")
                analysis_df = None
        
        pred_df_all = analysis_df

    # Zのリストを取得
    unique_zs = sorted(list(set(z_list)))
    # もしPES/Paramsのデータがない場合でも、Analysis結果があればそこからZを取得
    if not unique_zs and pred_df_all is not None and "Z" in pred_df_all.columns:
        unique_zs = sorted(pred_df_all["Z"].unique())

    if selected_z is not None:
        unique_zs = [z for z in unique_zs if z == selected_z]
        if len(unique_zs) == 0:
            print(f"Warning: No data found for selected Z={selected_z}.")

    print(f"Generating plots for Z: {unique_zs}")

    # ==========================================
    # 7. 全核種まとめてパラメータ推移プロット
    # ==========================================
    if args.type in ["params", "all"] and len(n_list) > 0:
        print(f"--- Plotting combined parameters evolution for {len(set(z_list))} isotopes ---")
        # 直接 plot_dir を指定して保存を確実にする
        vis.plot_parameters_evolution(
            n_list, z_list, params_history, 
            filename=f"{prefix}params_trend_all.pdf"
        )
    else:
        print("Warning: No parameter data collected. Skipping combined plot.")

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
                filename=f"{prefix}PES_all.pdf"
            )

        # 2. Params
        z_indices = [i for i, val in enumerate(z_list) if val == z]
        if args.type in ["params", "all"] and z_indices:
            z_n_list = [n_list[i] for i in z_indices]
            z_z_list_sub = [z_list[i] for i in z_indices]
            z_params = {k: [v[i] for i in z_indices] for k, v in params_history.items()}
            
            z_vis.plot_parameters_evolution(
                z_n_list, z_z_list_sub, z_params, 
                filename=f"{prefix}params_trend.pdf"
            )

        # 3. Spectra & Ratio
        if args.type in ["spectra", "ratio", "all"]:
            # Filter Pred
            z_pred_df = None
            if pred_df_all is not None and "Z" in pred_df_all.columns:
                z_pred_df = pred_df_all[pred_df_all["Z"] == z]
            
            # Load Expt for this Z
            z_expt_file = cfg["dirs"]["raw_dir"] / str(z) / "expt.csv"
            z_expt_df = pd.DataFrame()
            
            if z_expt_file.exists():
                try:
                    z_expt_df = pd.read_csv(z_expt_file)
                    print(f"Loaded experimental data for Z={z} from {z_expt_file}")
                    
                    # Filter by N range from config
                    n_min = cfg["nuclei"].get("n_min")
                    n_max = cfg["nuclei"].get("n_max")
                    
                    if "N" in z_expt_df.columns and n_min is not None and n_max is not None:
                        z_expt_df = z_expt_df[(z_expt_df["N"] >= n_min) & (z_expt_df["N"] <= n_max)]
                        print(f"Filtered experimental data: N in [{n_min}, {n_max}] -> {len(z_expt_df)} records")
                        
                except Exception as e:
                    print(f"Warning: Failed to load {z_expt_file}: {e}")
            else:
                print(f"Warning: Experimental data not found for Z={z} at {z_expt_file}")

            if z_pred_df is not None and not z_pred_df.empty:
                if args.type in ["spectra", "all"]:
                    panel_labels = _spectra_panel_labels(model_config.get("fixed_C_beta"))
                    z_vis.plot_spectra(
                        z_pred_df,
                        z_expt_df,
                        filename=f"{prefix}spectra.pdf",
                        panel_labels=panel_labels,
                    )
                    # Specify levels without 0+_2 for optuna_spectra_g
                    if args.optuna:
                        z_vis.plot_spectra(
                            z_pred_df,
                            z_expt_df,
                            filename=f"{prefix}spectra_g.pdf",
                            levels=["2+_1", "4+_1", "6+_1"],
                            panel_labels=panel_labels,
                        )
                
                if args.type in ["ratio", "all"]:
                    z_vis.plot_ratio(
                        z_pred_df,
                        z_expt_df,
                        filename=f"{prefix}ratio.pdf",
                        panel_label=_z_panel_label(z),
                    )

    print("All plots generated.")

if __name__ == "__main__":
    main()