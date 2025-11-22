import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self, weight_type="reciprocal", alpha=0.5):
        """
        Args:
            weight_type (str): "reciprocal" (1/|E|+c) or "exp" (e^-E)
            alpha (float): 重みの強さを調整する係数
        """
        super().__init__()
        self.weight_type = weight_type
        self.alpha = alpha

    def forward(self, pred, target):
        """
        pred:   [Batch, Grid] (予測されたPES)
        target: [Batch, Grid] (正解のPES)
        """
        # 誤差の二乗 (基本のMSE成分)
        diff_sq = (pred - target) ** 2
        
        # 重みの計算 (ターゲットのエネルギー値に基づく)
        # targetは正規化済みで、基底状態=0、他は正の値になっている前提
        
        if self.weight_type == "reciprocal":
            # 逆数型: E=0付近で重みが最大(1.0)になり、Eが増えると重みが減る
            # weight = 1 / (alpha * |E| + 1)
            # alphaが大きいほど、高エネルギー部分の無視度合いが強くなる
            weights = 1.0 / (self.alpha * torch.abs(target) + 1.0)
            
        elif self.weight_type == "exp":
            # 指数型: 急激に重みが減る (厳しめの設定)
            weights = torch.exp(-self.alpha * torch.abs(target))
            
        else:
            # 重みなし (通常のMSE)
            weights = torch.ones_like(target)

        # 重み付き平均を計算
        loss = torch.mean(weights * diff_sq)
        
        return loss