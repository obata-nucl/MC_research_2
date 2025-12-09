import torch
import torch.nn as nn

class FlexiblePESLoss(nn.Module):
    def __init__(self, loss_type="normalized", weight_type="reciprocal", alpha=5.0):
        """
        Args:
            loss_type (str): 
                - "normalized": 新しいLoss。PESを0~1に正規化して「形」を見る。
                - "absolute":   通常のLoss。エネルギーの絶対値(MeV)で「深さ」を見る。
            weight_type (str): "reciprocal" (1/|E|) or "exp" (e^-E) or "none"
            alpha (float): 重みの強さ係数。
                           ※ loss_typeによって適切な値が変わる点に注意してください。
                           (normalizedなら5.0~10.0, absoluteなら0.1~0.5くらいが目安)
        """
        super().__init__()
        self.loss_type = loss_type
        self.weight_type = weight_type
        self.alpha = alpha

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
        
        # --- 3. 重みの計算 ---
        # 重みは常に「正解データ(input_target)」の安定度に基づいて決める
        
        if self.weight_type == "reciprocal":
            # 逆数型: 谷底付近を重視
            # normalizedの場合: targetは0.0~1.0 -> alphaは大きめ(5.0など)推奨
            # absoluteの場合: targetは-10.0~5.0とかなので絶対値をとる
            weights = 1.0 / (self.alpha * torch.abs(input_target) + 1.0)
            
        elif self.weight_type == "exp":
            # 指数型
            weights = torch.exp(-self.alpha * torch.abs(input_target))
            
        else:
            # 重みなし
            weights = torch.ones_like(input_target)

        # 重み付き平均を返す
        loss = torch.mean(weights * diff_sq)
        
        return loss