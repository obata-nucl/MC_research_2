import torch
import torch.nn as nn

class FlexiblePESLoss(nn.Module):
    def __init__(self, loss_type=None):
        """
        Args:
            loss_type (str): 
                - "normalized": 新しいLoss。PESを0~1に正規化して「形」を見る。
                - "absolute":   通常のLoss。エネルギーの絶対値(MeV)で「深さ」を見る。
        """
        super().__init__()
        self.loss_type = loss_type

    def normalize(self, tensor):
        """
        バッチごとに [0, 1] に正規化する関数
        """
        min_val, _ = torch.min(tensor, dim=1, keepdim=True)
        max_val, _ = torch.max(tensor, dim=1, keepdim=True)
        eps = 1e-6
        return (tensor - min_val) / (max_val - min_val + eps)

    def forward(self, pred, target):
        """
        pred:   [Batch, Grid] (NN出力)
        target: [Batch, Grid] (正解データ)
        """
        
        # --- 1. モード分岐: 正規化するかどうか ---
        if self.loss_type == "normalized":
            # 新しいLoss: 両方を正規化して「形」だけ比較
            input_pred = self.normalize(pred)
            input_target = self.normalize(target)
        else:
            # 通常のLoss: 生の値をそのまま比較
            input_pred = pred
            input_target = target

        # --- 2. 誤差の計算 ---
        diff_sq = (input_pred - input_target) ** 2
        
        # 重みなしで平均をとる
        loss = torch.mean(diff_sq)
        
        return loss