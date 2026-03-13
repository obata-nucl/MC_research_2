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
        plt.rcParams['font.size'] = 15

    @staticmethod
    def _resolve_column(df, candidates):
        for name in candidates:
            if name in df.columns:
                return name
        return None

    def _pdf_path(self, filename):
        stem = Path(filename).stem
        return self.save_dir / f"{stem}.pdf"

    HFB_COLOR = "#6A0DAD"

    def plot_all_pes(self, beta, pes_data_list, filename="PES_all.pdf"):
        """
        全核種のPESをまとめてプロット
        pes_data_list: [{"N": n, "target": target_E, "pred": pred_E}, ...]
        """
        n_panels = len(pes_data_list)
        # 横4列固定: 8核種なら縦2 x 横4レイアウトになる
        cols = 4
        rows = int(np.ceil(n_panels / cols))
        
        base_w, base_h = 4.8, 4.4
        fig, axes = plt.subplots(rows, cols, figsize=(base_w * cols, base_h * rows), sharex=True, sharey=True)
        axes = np.array(axes).reshape(rows, cols)
        
        # 軸ラベルは外側のプロットのみに表示
        for ax in axes[-1, :]:
            ax.set_xlabel(r"$\beta$", fontsize=18)
        for ax in axes[:, 0]:
            ax.set_ylabel("Energy [MeV]", fontsize=18)

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
            ax.plot(beta, target_E, linestyle="--", color=self.HFB_COLOR, label="HFB", linewidth=2.6)
            idx_min_expt = np.argmin(target_E)
            ax.plot(
                beta[idx_min_expt],
                target_E[idx_min_expt],
                marker="o",
                markersize=11,
                markerfacecolor="white",
                markeredgecolor=self.HFB_COLOR,
                markeredgewidth=2.4,
                zorder=5,
            )
            
            # IBM (Pred): 黒実線 + 赤丸 (最小点)
            ax.plot(beta, pred_E, linestyle="-", color="black", label="IBM-2", linewidth=2.6)
            idx_min_calc = np.argmin(pred_E)
            ax.plot(
                beta[idx_min_calc],
                pred_E[idx_min_calc],
                marker="o",
                markersize=10,
                markerfacecolor="red",
                markeredgecolor="black",
                markeredgewidth=1.2,
                zorder=6,
            )
            
            # タイトル (Zに応じて変更)
            mass_number = z_val + n_val
            symbol = element_symbols.get(z_val, "X")
            ax.set_title(rf"$^{{{mass_number}}}\mathrm{{{symbol}}}$", fontsize=22)
            
            ax.tick_params(axis="both", which="major", labelsize=16, width=1.4)
            if i == 0: # 凡例は最初だけ
                ax.legend(loc="best", fontsize=13)
            
        # 余ったサブプロットを非表示
        for j in range(i + 1, rows * cols):
            axes.ravel()[j].axis('off')
            
        plt.tight_layout()
        save_path = self._pdf_path(filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

    def plot_all_pes_compare_models(self, beta, pes_data_list, model_labels, filename="PES_compare.pdf"):
        """
        全核種のPESをまとめてプロット (複数モデル比較)
        pes_data_list: [{"Z": z, "N": n, "target": target_E, "preds": {label: pred_E, ...}}, ...]
        """
        if len(pes_data_list) == 0:
            print("Warning: No PES data to plot for model comparison.")
            return

        n_panels = len(pes_data_list)
        cols = 4
        rows = int(np.ceil(n_panels / cols))

        base_w, base_h = 4.8, 4.4
        fig, axes = plt.subplots(rows, cols, figsize=(base_w * cols, base_h * rows), sharex=True, sharey=True)
        axes = np.array(axes).reshape(rows, cols)

        for ax in axes[-1, :]:
            ax.set_xlabel(r"$\beta$", fontsize=18)
        for ax in axes[:, 0]:
            ax.set_ylabel("Energy [MeV]", fontsize=18)

        pes_data_list.sort(key=lambda x: (x.get("Z", 0), x["N"]))

        element_symbols = {60: "Nd", 62: "Sm", 64: "Gd"}
        # HFBは紫固定。IBM-2比較線は暖色も含めて視認性を上げる
        model_palette = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#8C564B"]
        color_map = {label: model_palette[i % len(model_palette)] for i, label in enumerate(model_labels)}

        for i, data in enumerate(pes_data_list):
            ax = axes.ravel()[i]
            n_val = data["N"]
            z_val = data.get("Z", 62)
            target_E = data["target"]
            preds_dict = data.get("preds", {})

            ax.plot(beta, target_E, linestyle="--", color=self.HFB_COLOR, label="HFB", linewidth=2.8)
            idx_min_expt = np.argmin(target_E)
            ax.plot(
                beta[idx_min_expt],
                target_E[idx_min_expt],
                marker="o",
                markersize=11,
                markerfacecolor="white",
                markeredgecolor=self.HFB_COLOR,
                markeredgewidth=2.4,
                zorder=7,
            )

            for label in model_labels:
                pred_E = preds_dict.get(label)
                if pred_E is None:
                    continue
                ax.plot(beta, pred_E, linestyle="-", color=color_map[label], label=label, linewidth=2.8, alpha=0.95)
                idx_min_pred = np.argmin(pred_E)
                ax.plot(
                    beta[idx_min_pred],
                    pred_E[idx_min_pred],
                    marker="o",
                    markersize=10,
                    markerfacecolor=color_map[label],
                    markeredgecolor="black",
                    markeredgewidth=1.0,
                    zorder=8,
                )

            mass_number = z_val + n_val
            symbol = element_symbols.get(z_val, "X")
            ax.set_title(rf"$^{{{mass_number}}}\mathrm{{{symbol}}}$", fontsize=22)
            ax.tick_params(axis="both", which="major", labelsize=14, width=1.4)

            if i == 0:
                ax.legend(loc="best", fontsize=14)

        for j in range(i + 1, rows * cols):
            axes.ravel()[j].axis("off")

        plt.tight_layout()
        save_path = self._pdf_path(filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

    def plot_parameters_evolution(self, n_list, z_list, params_dict, filename="params.pdf"):
        """
        中性子数Nに対する各パラメータの変化をプロット (Zごとに色分け)
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        keys = ["epsilon", "kappa", "chi_pi", "chi_nu"]
        keys_labels = [
            r"$\epsilon$ (MeV)",
            r"$\kappa$ (MeV)",
            r"$\chi_{\pi}$",
            r"$\chi_{\nu}$",
        ]
        panel_labels = ["(a)", "(b)", "(c)", "(d)"]
        
        # Zごとの色設定
        unique_z = sorted(list(set(z_list)))
        # Zが少ない場合は固定色、多い場合はカラーマップ
        if len(unique_z) <= 3:
            colors = ["blue", "red", "green"]
        else:
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_z)))
            
        line_styles = ["-", "--", ":"]
        marker_styles = ["o", "s", "^"]
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
                ls = line_styles[j % len(line_styles)]
                mk = marker_styles[j % len(marker_styles)]
                
                # Jitterなしでプロット。透明度を調整し、マーカーと線種で区別。
                ax.plot(sorted_n, sorted_vals, marker=mk, linestyle=ls, 
                        color=color, label=f"{symbol}", alpha=0.7,
                        linewidth=2.6, markersize=7)

            ax.text(
                0.03,
                0.95,
                panel_labels[i],
                transform=ax.transAxes,
                fontsize=24,
                va="top",
                ha="left",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.2},
            )

            if i >= 2:
                ax.set_xlabel("Neutron Number N", fontsize=22)
            else:
                ax.set_xlabel("")
            ax.set_ylabel(keys_labels[i], fontsize=22)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.tick_params(labelsize=18, width=1.4)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            if key in param_limits:
                ax.set_ylim(*param_limits[key])
            
            # 凡例を表示
            ax.legend(fontsize=20)
        
        plt.tight_layout()
        save_path = self._pdf_path(filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

    def plot_loss_history(self, train_loss, val_loss, filename="loss.pdf", lr=None):
        """
        学習曲線のプロット
        """
        epochs = range(1, len(train_loss) + 1)
        
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax1.set_title("Learning Curve", fontsize=18)
        ax1.set_xlabel("Epochs", fontsize=16)
        ax1.set_ylabel("MAE", fontsize=16)
        ax1.grid(True, alpha=0.3, linestyle=":")
        ax1.tick_params(labelsize=16)
        ax1.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))

        ax1.text(
            0.80,
            0.80,
            "(a) Model 1",
            transform=ax1.transAxes,
            fontsize=16,
            va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
        )
        
        ax1.plot(epochs, train_loss, label='train MAE', color="#1f77b4", linewidth=2.8)
        ax1.plot(epochs, val_loss, label='validation MAE', color="#ff7f0e", linewidth=2.8)
        ax1.legend(loc="best", fontsize=20)
        
        if lr is not None:
            ax2 = ax1.twinx()
            lr_arr = np.asarray(lr, dtype=float)
            valid = np.isfinite(lr_arr)
            if valid.any():
                # Show LR in 1e-3 units to avoid a single 10^-3 tick label.
                lr_scaled = lr_arr * 1e3
                ax2.plot(epochs, lr_scaled, color="#2ca02c", alpha=0.45, linewidth=2.0, label="lr")
                y_min = float(np.nanmin(lr_scaled[valid]))
                y_max = float(np.nanmax(lr_scaled[valid]))
                pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.05
                ax2.set_ylim(y_min - pad, y_max + pad)

            ax2.set_ylabel(r"Learning Rate ($\times 10^{-3}$)", fontsize=22)
            ax2.tick_params(labelsize=22)
            ax2.yaxis.set_major_locator(MaxNLocator(nbins=4))
        
        plt.tight_layout()
        save_path = self._pdf_path(filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

    def plot_spectra(self, pred_df, expt_df, filename="spectra.pdf", levels=None, panel_labels=None):
        """
        エネルギー準位の比較プロット (Project 1 style)
        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 5.2), sharey=False)
        
        if levels is None:
            levels = ["2+_1", "4+_1", "6+_1", "0+_2"]
        
        markers = ['o', 's', '^', 'D']
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        level_aliases = {
            "2+_1": ["2+_1", "E2_1"],
            "4+_1": ["4+_1", "E4_1"],
            "6+_1": ["6+_1", "E6_1"],
            "0+_2": ["0+_2", "E0_2"]
        }
        level_labels_tex = {
            "2+_1": r"$2^+_1$",
            "4+_1": r"$4^+_1$",
            "6+_1": r"$6^+_1$",
            "0+_2": r"$0^+_2$",
        }
        
        # マーカーと色の対応を固定するために辞書化 (Subset指定に対応)
        all_levels = ["2+_1", "4+_1", "6+_1", "0+_2"]
        marker_map = {l: m for l, m in zip(all_levels, markers)}
        color_map = {l: c for l, c in zip(all_levels, colors)}
        
        # Theory (Pred)
        for level in levels:
            col = self._resolve_column(pred_df, level_aliases[level])
            if col is None:
                continue
            valid = pred_df[col].notna()
            if not valid.any():
                continue
            label = level_labels_tex.get(level, level)
            ax[0].plot(pred_df.loc[valid, "N"], pred_df.loc[valid, col], 
                       marker=marker_map[level], color=color_map[level], label=label,
                       linewidth=2.8, markersize=7)

        # Expt
        for level in levels:
            col = self._resolve_column(expt_df, level_aliases[level])
            if col is None:
                continue
            valid = expt_df[col].notna()
            if not valid.any():
                continue
            label = level_labels_tex.get(level, level)
            ax[1].plot(expt_df.loc[valid, "N"], expt_df.loc[valid, col], 
                       marker=marker_map[level], color=color_map[level], label=label,
                       linewidth=2.8, markersize=7)

        if panel_labels is not None:
            left_label, right_label = panel_labels
            ax[0].text(
                0.03,
                0.96,
                left_label,
                transform=ax[0].transAxes,
                fontsize=16,
                va="top",
                ha="left",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
            )
            ax[1].text(
                0.03,
                0.96,
                right_label,
                transform=ax[1].transAxes,
                fontsize=16,
                va="top",
                ha="left",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
            )

        for a in ax:
            a.set_xlabel("Neutron Number", fontsize=16)
            a.set_ylabel("Energy [MeV]", fontsize=16)
            a.legend(loc="best", fontsize=12)
            a.grid(True, linestyle='--', alpha=0.5)
            a.tick_params(labelsize=16)
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
        save_path = self._pdf_path(filename)
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

        # Additional plot: Shared Y-axis (aligned)
        ax[0].set_ylim(0.0, limit_y)
        ax[1].set_ylim(0.0, limit_y)
        
        stem = Path(filename).stem
        save_path_fixed = self.save_dir / f"{stem}_common_scale.pdf"
        
        plt.savefig(save_path_fixed)
        plt.close()
        print(f"Saved: {save_path_fixed}")

    def plot_ratio(self, pred_df, expt_df, filename="ratio.pdf", panel_label=None):
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
                        linewidth=2.8, markersize=8, label="IBM-2")
        
        # Expt
        if expt_col is not None:
            valid = expt_df[expt_col].notna()
            if valid.any():
                ax.plot(expt_df.loc[valid, "N"], expt_df.loc[valid, expt_col], marker='D', color="#5C006E", 
                        linestyle="--", linewidth=2.6, markersize=8, label="Expt.")

        if panel_label is not None:
            ax.text(
                0.96,
                0.07,
                panel_label,
                transform=ax.transAxes,
                fontsize=24,
                ha="right",
                va="bottom",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.2},
            )
        
        ax.set_ylim(1.5, 3.5)
        ax.set_xlabel("Neutron Number", fontsize=14)
        ax.set_ylabel(r"$R_{4/2}$", fontsize=14)
        ax.legend(loc="best", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(labelsize=12)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        save_path = self._pdf_path(filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")