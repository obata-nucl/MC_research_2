import numpy as np
import torch
import torch.nn as nn

def _get_activation(name: str) -> nn.Module:
    """ 活性化関数の取得 """
    if name == "ReLU" : return nn.ReLU()
    elif name == "Tanh": return nn.Tanh()
    elif name == "SiLU": return nn.SiLU()
    raise ValueError(f"Unknown activation {name}")

class FlexibleMLP(nn.Module):
    """ Optunaで「層の数」や「各層のノード数」が変わっても対応できるクラス """
    def __init__(self, input_dim: int, hidden_sizes: list[int], activation: str):
        """ Args:
                input_dim (int): 入力次元
                hidden_sizes (list[int]): 各隠れ層のノード数のリスト
                activation (str): 活性化関数の名前
        """
        super().__init__()
        layers = []
        act_fn = _get_activation(activation)

        current_dim = input_dim
        for h_size in hidden_sizes:
            layers.append(nn.Linear(current_dim, h_size))
            layers.append(act_fn)
            current_dim = h_size
        
        self.net = nn.Sequential(*layers)
        self.output_dim = current_dim
    
    def forward(self, x):
        return self.net(x)

class IBM2FlexibleNet(nn.Module):
    """ メインのネットワーク(Isotope専用) """
    def __init__(self, config: dict):
        super().__init__()

        input_dim = config["input_dim"]
        hidden_sizes = config["hidden_sizes"]
        act_name = config["activation"]

        # --------------------
        # 1. 特徴抽出
        # --------------------
        self.neutron_branch = FlexibleMLP(input_dim, hidden_sizes, act_name)
        branch_output_dim = self.neutron_branch.output_dim

        # [Future] 陽子数Zの変化に対応する場合
        # input_dim を陽子用(Z, n_pi)にも用意する
        # self.proton_branch = FlexibleMLP(proton_input_dim, hidden_sizes, act_name)を追加

        # --------------------
        # 2. パラメータ予測ヘッド
        # --------------------
        # (A) chi_nu: 中性子の特徴から抽出
        self.head_chi_nu = nn.Linear(branch_output_dim, 1)

        # (B) chi_pi: 陽子の特徴から抽出 (現在は同位体のみ対応)
        fixed_chi_pi = config["fixed_chi_pi"]
        self.register_buffer("fixed_chi_pi", torch.tensor(float(fixed_chi_pi)))

        # (C) epsilon, kappa : ハミルトニアンのパラメータ
        self.head_hamiltonian = nn.Linear(branch_output_dim, 2)

        # (D) C_beta : 固定値 (PESの座標スケールパラメータ)
        fixed_C_beta = config["fixed_C_beta"]
        self.register_buffer("fixed_C_beta", torch.tensor(float(fixed_C_beta)))

        # [Future] 陽子数変化に対応する場合
        # self.head_chi_pi = nn.Linear(proton_branch_output_dim, 1)を追加
        # 固定値バッファを削除
        # self.head_interaction = nn.Sequential(
        #     nn.Linear(neutron_branch_output_dim + proton_branch_output_dim, neuron_branch_output_dim),
        #     _get_activation(act_name),
        #     nn.Linear(neuron_branch_output_dim, 2)
        # )
        # self.head_c_beta = ...
        self.softplus = nn.Softplus()
    
    def forward(self, x):
        """ Args:
                x; [batch_size, 2] -> [n_nu, P]
            
            [Future] 陽子数Zも考慮する場合 (neutron_x, proton_x)を受け取るよう変更 neutron_x: [N, n_nu, N^2], proton_x: [Z, n_pi, Z^2]
        """
        h = self.neutron_branch(x)

        # [Future] Y字型にする場合
        # h_nu = self.neutron_branch(neutron_x)
        # h_pi = self.proton_branch(proton_x)
        # h = torch.cat([h_nu, h_pi], dim=-1)

        chi_nu = self.head_chi_nu(h)
        chi_nu = - 1.5 * torch.sigmoid(chi_nu)
        chi_pi = self.fixed_chi_pi.expand_as(chi_nu)
        
        # Hamiltonian parameters
        hamiltonian = self.head_hamiltonian(h)      # [batch_size, 2]
        epsilon = hamiltonian[:, 0:1]               # [batch_size, 1]
        kappa   = hamiltonian[:, 1:2]               # [batch_size, 1]
        
        # Scale parameter
        C_beta = self.fixed_C_beta.expand_as(epsilon) # [batch_size, 1]

        epsilon = self.softplus(epsilon)
        kappa   = - self.softplus(kappa)
        
        return torch.cat([epsilon, kappa, chi_nu, chi_pi, C_beta], dim=1)  # [batch_size, 5]
    
class IBM2PESDecoder(nn.Module):
    def __init__(self, beta_f_grid: np.ndarray, gamma_grid: np.ndarray = None):
        super().__init__()
        self.register_buffer("beta_f_grid", torch.tensor(beta_f_grid))
        if gamma_grid is None:
            gamma_grid = torch.zeros_like(self.beta_f_grid)
        self.register_buffer("gamma_grid", gamma_grid)
    
    @staticmethod
    def calc_energy(n_pi, n_nu, epsilon, kappa, chi_pi, chi_nu, beta_b, gamma=None):
        """ IBM-2 のポテンシャルエネルギー計算式 """
        if gamma is None:
            gamma = torch.zeros_like(beta_b)

        beta2 = beta_b ** 2
        deno1 = 1 + beta2
        deno2 = deno1 ** 2
        cos_3gamma = torch.cos(3 * gamma)

        SQRT_2_7 = 0.5345224838248488

        term1 = epsilon*(n_pi + n_nu)*(beta2 / deno1)
        term2 = (n_pi*n_nu*kappa*(beta2 / deno2)) * (4.0 - 2.0*SQRT_2_7*(chi_pi + chi_nu)*beta_b*cos_3gamma + 2.0*chi_pi*chi_nu*beta2 / 7.0)

        return term1 + term2
    
    def forward(self, params, n_pi, n_nu):
        epsilon = params[:, 0:1]
        kappa   = params[:, 1:2]
        chi_nu  = params[:, 2:3]
        chi_pi  = params[:, 3:4]
        C_beta  = params[:, 4:5]

        # [Future] gammaも考慮する場合
        # beta, gamma のメッシュグリッドを作成して計算するロジックが必要

        beta_b = self.beta_f_grid.unsqueeze(0) * C_beta
        gamma = self.gamma_grid.unsqueeze(0)

        if n_pi.dim() == 1: n_pi = n_pi.view(-1, 1)
        if n_nu.dim() == 1: n_nu = n_nu.view(-1, 1)

        return self.calc_energy(n_pi, n_nu, epsilon, kappa, chi_pi, chi_nu, beta_b, gamma)  # [batch_size, grid_size]