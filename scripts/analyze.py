import argparse
import subprocess
import pandas as pd
import torch
import yaml
import sys
from pathlib import Path
import numpy as np

from src.model import IBM2FlexibleNet
from src.dataset import IBM2Dataset
from src.utils import load_config

def run_npbos(script_path, npbos_dir, mass_num, n_nu, params):
    """
    src/analyze.sh を実行して npbos の計算結果を取得する
    params: [epsilon, kappa, chi_pi, chi_nu]
    """
    # analyze.sh arguments: NPBOS_DIR MASS_NUM N_NU eps kappa chi_pi chi_n
    
    cmd = [
        "bash", str(script_path),
        str(npbos_dir),
        str(mass_num),
        str(n_nu),
        f"{params[0]:.4f}", # eps
        f"{params[1]:.4f}", # kappa
        f"{params[2]:.4f}", # chi_pi
        f"{params[3]:.4f}"  # chi_nu
    ]
    
    try:
        # タイムアウトを設定して実行
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            print(f"Error running npbos for A={mass_num}: {result.stderr}")
            return None
        
        # Output format from analyze.sh: "2_1 4_1 6_1 0_2"
        # 空白区切りで数値が返ってくることを期待
        output_str = result.stdout.strip()
        if not output_str:
            return None
            
        energies = list(map(float, output_str.split()))
        return energies
    except subprocess.TimeoutExpired:
        print(f"Timeout running npbos for A={mass_num}")
        return None
    except Exception as e:
        print(f"Exception running npbos: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optuna", action="store_true", help="Use Optuna best model")
    args = parser.parse_args()

    cfg = load_config()
    device = torch.device(cfg.get("device", "cpu"))
    
    # Paths
    output_dir = cfg["dirs"]["output_dir"]
    model_dir = output_dir / "models"
    analyze_script = Path("src/analyze.sh").resolve()
    
    # NPBOSディレクトリの取得 (configから)
    npbos_dir = Path(cfg["dirs"].get("npbos_dir", "./NPBOS")).resolve()
    
    if not analyze_script.exists():
        print(f"Error: {analyze_script} not found.")
        sys.exit(1)
        
    if not npbos_dir.exists():
        print(f"Error: NPBOS directory not found at {npbos_dir}")
        sys.exit(1)

    # ==========================================
    # モデルのロード (plot.pyと同様のロジック)
    # ==========================================
    if args.optuna:
        print("Mode: Analyzing OPTUNA BEST result")
        model_path = model_dir / "optuna_best_model.pth"
        config_path = model_dir / "optuna_best_config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                model_config = yaml.safe_load(f)
        else:
            print("Warning: Config file not found. Using default config.")
            model_config = cfg["default"]["nn"].copy()
            model_config["fixed_chi_pi"] = cfg["nuclei"]["fixed_chi_pi"]
    else:
        print("Mode: Analyzing NORMAL training result")
        model_path = model_dir / "best_model.pth"
        model_config = cfg["default"]["nn"].copy()
        model_config["fixed_chi_pi"] = cfg["nuclei"]["fixed_chi_pi"]

    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)

    # Model構築
    model = IBM2FlexibleNet(model_config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ==========================================
    # データセット準備 & 解析ループ
    # ==========================================
    dataset = IBM2Dataset(cfg)
    results = []
    
    print(f"Starting analysis...")
    print(f"NPBOS directory: {npbos_dir}")
    
    # データセット内の全核種についてループ
    for item in dataset.data:
        z = item["Z"]
        n = item["N"]
        n_pi = item["n_pi"]
        n_nu = item["n_nu"]
        mass_num = z + n
        
        # モデル入力作成 [N, n_nu, N^2]
        # dataset.pyの実装に合わせて入力を作成
        # dataset.py: inputs = torch.tensor([norm_N, norm_n_nu, norm_N_sq], dtype=torch.float32)
        norm_N = float(n) / 126.0
        norm_n_nu = float(n_nu) / 30.0
        norm_N_sq = norm_N ** 2
        
        inp = torch.tensor([[norm_N, norm_n_nu, norm_N_sq]], device=device)
        
        with torch.no_grad():
            # Output: [epsilon, kappa, chi_nu, chi_pi, C_beta]
            preds = model(inp).cpu().numpy()[0]
        
        epsilon = preds[0]
        kappa = preds[1]
        chi_nu = preds[2]
        chi_pi = preds[3]
        c_beta = preds[4]
        
        # NPBOS実行
        # analyze.sh args: NPBOS_DIR MASS_NUM N_NU eps kappa chi_pi chi_n
        energies = run_npbos(analyze_script, npbos_dir, mass_num, n_nu, [epsilon, kappa, chi_pi, chi_nu])
        
        row = {
            "Z": z,
            "N": n,
            "n_pi": n_pi,
            "n_nu": n_nu,
            "epsilon": epsilon,
            "kappa": kappa,
            "chi_nu": chi_nu,
            "chi_pi": chi_pi,
            "C_beta": c_beta
        }
        
        if energies and len(energies) >= 4:
            row["2+_1"] = energies[0]
            row["4+_1"] = energies[1]
            row["6+_1"] = energies[2]
            row["0+_2"] = energies[3]
            
            if energies[0] != 0:
                row["R_4/2"] = energies[1] / energies[0]
            else:
                row["R_4/2"] = 0.0
        else:
            row["2+_1"] = None
            row["4+_1"] = None
            row["6+_1"] = None
            row["0+_2"] = None
            row["R_4/2"] = None

        results.append(row)
        
        r42_str = f"{row['R_4/2']:.3f}" if row['R_4/2'] is not None else "N/A"
        print(f"Z={z}, N={n} | eps={epsilon:.3f}, kap={kappa:.3f} | R4/2={r42_str}")

    # ==========================================
    # 結果保存
    # ==========================================
    df = pd.DataFrame(results)
    
    # 保存先
    save_name = "analysis_optuna.csv" if args.optuna else "analysis_normal.csv"
    save_path = output_dir / save_name
    
    df.to_csv(save_path, index=False)
    print(f"\nAnalysis completed. Results saved to: {save_path}")

if __name__ == "__main__":
    main()
