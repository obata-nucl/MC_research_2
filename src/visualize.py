import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class IBM2Visualizer:
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        plt.rcParams['font.family'] = "serif"
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['font.size'] = 12

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

        # ソート (Nの昇順)
        pes_data_list.sort(key=lambda x: x["N"])
        
        for i, data in enumerate(pes_data_list):
            ax = axes.ravel()[i]
            n_val = data["N"]
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
            
            # タイトル (Sm固定)
            mass_number = 62 + n_val
            ax.set_title(rf"$^{{{mass_number}}}\mathrm{{Sm}}$", fontsize=18)
            
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

    def plot_parameters_evolution(self, n_list, params_dict, filename="params.png"):
        """
        中性子数Nに対する各パラメータの変化をプロット
        params_dict: {"epsilon": [v1, v2...], "kappa": [...], ...}
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        keys = ["epsilon", "kappa", "chi_pi", "chi_nu", "C_beta"]
        keys_labels = [r"$\epsilon$", r"$\kappa$", r"$\chi_{\pi}$", r"$\chi_{\nu}$", r"$C_{\beta}$"]
        colors = ["blue", "green", "orange", "red", "purple"]

        # n_listに基づいてソート順を決定
        sorted_indices = np.argsort(n_list)
        sorted_n = np.array(n_list)[sorted_indices]
        
        for i, key in enumerate(keys):
            if key not in params_dict:
                axes[i].axis("off") # データがない場合は枠を消す
                continue
            
            ax = axes[i]
            vals = np.array(params_dict[key])
            sorted_vals = vals[sorted_indices]
            
            ax.plot(sorted_n, sorted_vals, "o-", color=colors[i], label=key)
            ax.set_xlabel("Neutron Number N")
            ax.set_ylabel(keys_labels[i])
            ax.set_title(f"Evolution of {keys_labels[i]}")
            ax.grid(True, linestyle="--", alpha=0.5)
        
        if len(keys) < 6:
            for j in range(len(keys), 6):
                axes[j].axis('off')
            
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
        
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(epochs, train_loss, label='Train Loss')
        ax.plot(epochs, val_loss, label='Val Loss')
        
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Weighted MSE Loss")
        ax.set_title("Learning Curve")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        
        save_path = self.save_dir / filename
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")