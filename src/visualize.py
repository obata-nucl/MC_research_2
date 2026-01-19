import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.ticker import MaxNLocator


class IBM2Visualizer:
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        plt.rcParams['font.family'] = "serif"
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['font.size'] = 12

    @staticmethod
    def _resolve_column(df, candidates):
        for name in candidates:
            if name in df.columns:
                return name
        return None

    def plot_all_pes(self, beta, pes_data_list, filename="PES_all.png"):
        """
        全核種のPESをまとめてプロット
        pes_data_list: [{"N": n, "target": target_E, "pred": pred_E}, ...]
        """
        n_panels = len(pes_data_list)
        cols = int(np.ceil(np.sqrt(n_panels)))
        rows = int(np.ceil(n_panels / cols))
        
        base_w, base_h = 5.0, 4.0
        fig, axes = plt.subplots(rows, cols, figsize=(base_w * cols, base_h * rows), sharex=True, sharey=True)
        
        # 軸ラベルは外側のプロットのみに表示
        for ax in axes[-1, :]:
            ax.set_xlabel(r"$\beta$", fontsize=14)
        for ax in axes[:, 0]:
            ax.set_ylabel("Energy [MeV]", fontsize=14)

        # ソート (Z, Nの昇順)
        pes_data_list.sort(key=lambda x: (x.get("Z", 0), x["N"]))
        
        # 元素記号のマッピング
        element_symbols = {
            60: "Nd",
            62: "Sm",
            64: "Gd"
        }
        
        for i, data in enumerate(pes_data_list):
            ax = axes.ravel()[i]
            n_val = data["N"]
            z_val = data.get("Z", 62) # デフォルトはSm (62)
            target_E = data["target"]
            pred_E = data["pred"]
            
            # HFB (Target): オレンジ破線 + 青丸 (最小点)
            ax.plot(beta, target_E, linestyle="--", color="tab:orange", label="HFB PES")
            idx_min_expt = np.argmin(target_E)
            ax.plot(beta[idx_min_expt], target_E[idx_min_expt], 'bo', markersize=6)
            
            # IBM (Pred): 黒実線 + 赤丸 (最小点)
            ax.plot(beta, pred_E, linestyle="-", color="black", label="IBM PES")
            idx_min_calc = np.argmin(pred_E)
            ax.plot(beta[idx_min_calc], pred_E[idx_min_calc], 'ro', markersize=6)
            
            # タイトル (Zに応じて変更)
            mass_number = z_val + n_val
            symbol = element_symbols.get(z_val, "X")
            ax.set_title(rf"$^{{{mass_number}}}\mathrm{{{symbol}}}$", fontsize=18)
            
            ax.tick_params(axis="both", which="major", labelsize=12)
            if i == 0: # 凡例は最初だけ
                ax.legend(loc="best", fontsize=12)
            
        # 余ったサブプロットを非表示
        for j in range(i + 1, rows * cols):
            axes.ravel()[j].axis('off')
            
        plt.tight_layout()
        save_path = self.save_dir / filename
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

    def plot_parameters_evolution(self, n_list, z_list, params_dict, filename="params.png"):
        """
        中性子数Nに対する各パラメータの変化をプロット (Zごとに色分け)
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        keys = ["epsilon", "kappa", "chi_pi", "chi_nu"]
        keys_labels = [r"$\epsilon$", r"$\kappa$", r"$\chi_{\pi}$", r"$\chi_{\nu}$"]
        
        # Zごとの色設定
        unique_z = sorted(list(set(z_list)))
        # Zが少ない場合は固定色、多い場合はカラーマップ
        if len(unique_z) <= 3:
            colors = ["blue", "red", "green"]
        else:
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_z)))
            
        element_symbols = {60: "Nd", 62: "Sm", 64: "Gd"}

        param_limits = {
            "epsilon": (0.0, 3.5),
            "kappa": (-1.0, 0.0),
            "chi_pi": (-1.5, 0.0),
            "chi_nu": (-1.5, 0.0)
        }

        n_arr = np.array(n_list)
        z_arr = np.array(z_list)

        for i, key in enumerate(keys):
            if key not in params_dict:
                axes[i].axis("off") # データがない場合は枠を消す
                continue
            
            ax = axes[i]
            vals = np.array(params_dict[key])
            
            # Zごとにプロット
            for j, z in enumerate(unique_z):
                mask = (z_arr == z)
                if not np.any(mask):
                    continue
                    
                z_n = n_arr[mask]
                z_vals = vals[mask]
                
                # Nでソート
                sort_idx = np.argsort(z_n)
                sorted_n = z_n[sort_idx]
                sorted_vals = z_vals[sort_idx]
                
                symbol = element_symbols.get(z, f"Z={z}")
                color = colors[j % len(colors)]
                ax.plot(sorted_n, sorted_vals, "o-", color=color, label=f"{symbol}")

            ax.set_xlabel("Neutron Number N")
            ax.set_ylabel(keys_labels[i])
            ax.set_title(f"Evolution of {keys_labels[i]}")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if key in param_limits:
                ax.set_ylim(*param_limits[key])
            
            # 凡例を表示
            ax.legend(fontsize=10)
        
        plt.tight_layout()
        save_path = self.save_dir / filename
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

    def plot_loss_history(self, train_loss, val_loss, filename="loss.png"):
        """
        学習曲線のプロット
        """
        epochs = range(1, len(train_loss) + 1)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(epochs, train_loss, label='Train Loss', linewidth=2)
        ax.plot(epochs, val_loss, label='Val Loss', linewidth=2)
        
        ax.set_xlabel("Epochs", fontsize=14)
        ax.set_ylabel("Weighted MSE Loss", fontsize=14)
        ax.set_title("Learning Curve", fontsize=16)
        ax.set_yscale("log")
        ax.legend(fontsize=12)
        
        # グリッドと目盛りを細かく設定
        ax.grid(True, which="major", linestyle="-", alpha=0.5)
        ax.grid(True, which="minor", linestyle=":", alpha=0.3)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        save_path = self.save_dir / filename
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

    def plot_spectra(self, pred_df, expt_df, filename="spectra.png"):
        """
        エネルギー準位の比較プロット (Project 1 style)
        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        levels = ["2+_1", "4+_1", "6+_1", "0+_2"]
        markers = ['o', 's', '^', 'D']
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        level_aliases = {
            "2+_1": ["2+_1", "E2_1"],
            "4+_1": ["4+_1", "E4_1"],
            "6+_1": ["6+_1", "E6_1"],
            "0+_2": ["0+_2", "E0_2"]
        }
        
        # Theory (Pred)
        ax[0].set_title("Theory (IBM-2)", fontsize=16)
        for i, level in enumerate(levels):
            col = self._resolve_column(pred_df, level_aliases[level])
            if col is None:
                continue
            valid = pred_df[col].notna()
            if not valid.any():
                continue
            ax[0].plot(pred_df.loc[valid, "N"], pred_df.loc[valid, col], 
                       marker=markers[i], color=colors[i], label=level)

        # Expt
        ax[1].set_title("Experiment", fontsize=16)
        for i, level in enumerate(levels):
            col = self._resolve_column(expt_df, level_aliases[level])
            if col is None:
                continue
            valid = expt_df[col].notna()
            if not valid.any():
                continue
            ax[1].plot(expt_df.loc[valid, "N"], expt_df.loc[valid, col], 
                       marker=markers[i], color=colors[i], label=level)

        for a in ax:
            a.set_xlabel("Neutron Number", fontsize=14)
            a.set_ylabel("Energy [MeV]", fontsize=14)
            a.legend(loc="best")
            a.grid(True, linestyle='--', alpha=0.5)
            a.tick_params(labelsize=12)
            a.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Calculate max Y for shared axis
        max_y = 2.0
        for df in [pred_df, expt_df]:
            for level in levels:
                col = self._resolve_column(df, level_aliases[level])
                if col is not None and col in df.columns:
                    val_max = df[col].max()
                    if not np.isnan(val_max) and val_max > max_y:
                        max_y = val_max
        
        limit_y = max_y * 1.1

        ax[1].set_ylim(0.0, 2.0)
            
        plt.tight_layout()
        save_path = self.save_dir / filename
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

        # Additional plot: Shared Y-axis (aligned)
        ax[0].set_ylim(0.0, limit_y)
        ax[1].set_ylim(0.0, limit_y)
        
        stem = Path(filename).stem
        suffix = Path(filename).suffix
        new_filename = f"{stem}_common_scale{suffix}"
        save_path_fixed = self.save_dir / new_filename
        
        plt.savefig(save_path_fixed)
        plt.close()
        print(f"Saved: {save_path_fixed}")

    def plot_ratio(self, pred_df, expt_df, filename="ratio.png"):
        """
        R4/2比のプロット
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ratio_aliases = ["R_4/2", "R4_2"]
        pred_col = self._resolve_column(pred_df, ratio_aliases)
        expt_col = self._resolve_column(expt_df, ratio_aliases)

        # Pred
        if pred_col is not None:
            valid = pred_df[pred_col].notna()
            if valid.any():
                ax.plot(pred_df.loc[valid, "N"], pred_df.loc[valid, pred_col], marker='D', color="#2A23F3", 
                        linewidth=2.0, label="Theory Ratio")
        
        # Expt
        if expt_col is not None:
            valid = expt_df[expt_col].notna()
            if valid.any():
                ax.plot(expt_df.loc[valid, "N"], expt_df.loc[valid, expt_col], marker='D', color="#5C006E", 
                        linestyle="--", linewidth=1.8, label="Expt. Ratio")
        
        ax.set_title(r"$E(4^+_1)/E(2^+_1)$ Ratio", fontsize=16)
        ax.set_ylim(1.0, 3.5)
        ax.set_xlabel("Neutron Number", fontsize=14)
        ax.set_ylabel("Ratio", fontsize=14)
        ax.legend(loc="best", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(labelsize=12)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        save_path = self.save_dir / filename
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")