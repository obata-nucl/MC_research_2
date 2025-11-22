import itertools
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BosonCounter:
    """ 魔法数に基づいてボソン数を計算するクラス """
    def __init__(self, magic_numbers):
        self.magic_numbers = sorted(magic_numbers)

    def get_bosons(self, number: int) -> int:
        closest_magic = min(self.magic_numbers, key=lambda x: abs(x - number))
        boson_num = abs(number - closest_magic) // 2
        return boson_num


class IBM2Dataset(Dataset):
    def __init__(self, config):
        """
        Args:
            config (dict): load_config() で読み込まれた辞書全体
        """
        # 1. 設定の読み込み
        self.nuclei_conf = config["nuclei"]
        self.raw_dir = config["dirs"]["raw_dir"]
        self.magic_nums = self.nuclei_conf["magic_numbers"]
        
        # 2. グリッド設定
        self.beta_min = self.nuclei_conf["beta_min"]
        self.beta_max = self.nuclei_conf["beta_max"]
        self.beta_step = self.nuclei_conf["beta_step"]

        self.beta_grid = np.arange(
            self.beta_min, 
            self.beta_max + 0.5 * self.beta_step,
            self.beta_step
        )

        self.beta_zero_idx = np.argmin(np.abs(self.beta_grid))
        
        print(f"Target Grid: {self.beta_min} ~ {self.beta_max} (Step: {self.beta_step})")
        print(f" -> Shape: {self.beta_grid.shape}")

        # 3. 学習範囲の設定
        self.z_range = range(
            self.nuclei_conf["z_min"],
            self.nuclei_conf["z_max"] + 1,
            self.nuclei_conf["z_step"]
        )
        self.n_range = range(
            self.nuclei_conf["n_min"], 
            self.nuclei_conf["n_max"] + 1, 
            self.nuclei_conf["n_step"]
        )

        # ボソン計算機の準備
        self.counter = BosonCounter(self.magic_nums)
        
        self.data = self._load_data_specific()

    def _load_data_specific(self):
        dataset = []
        print(f"Loading data from: {self.raw_dir}")
        # 範囲の確認表示
        print(f"Target Range: Z={list(self.z_range)}, N={list(self.n_range)}")
        
        for z, n in itertools.product(self.z_range, self.n_range):
            # ファイル名: "84.csv" のような形式 (Zはファイル名に含まれない前提)
            # もし将来 "62_84.csv" になるなら f"{z}_{n}.csv" に変更
            filename = f"{n}.csv"
            file_path = self.raw_dir / filename
            
            # 指定範囲のファイルが無い場合はスキップ
            if not file_path.exists():
                raise FileNotFoundError(f"file not found : {file_path}")
            
            try:
                df = pd.read_csv(file_path, header=0, names=["Beta", "Energy"])
                df = df.sort_values(by="Beta")
                raw_beta = df["Beta"].values
                raw_energy = df["Energy"].values

                diff_matrix = np.abs(raw_beta[:, None] - self.beta_grid[None, :])
                min_diff_idx = np.argmin(diff_matrix, axis=0)
                min_diffs = np.min(diff_matrix, axis=0)
                if not np.all(min_diffs < 1e-4):
                    failed_betas = self.beta_grid[min_diffs >= 1e-4]
                    raise ValueError(f"Failed to find close beta values for: {failed_betas}")
                target_energy = raw_energy[min_diff_idx]

                # エネルギーの基準をbeta = 0に合わせる
                target_pes = target_energy - target_energy[self.beta_zero_idx]

                n_pi = self.counter.get_bosons(z)
                n_nu = self.counter.get_bosons(n)
                
                dataset.append({
                    "Z": z,
                    "N": n,
                    "n_pi": n_pi,
                    "n_nu": n_nu,
                    "target_pes": target_pes.astype(np.float32),
                })
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        print(f"Successfully loaded {len(dataset)} nuclei data.")
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 入力データ: [N, n_nu]
        # [Future] 同位体以外も学習する場合 [Z, N, n_pi, n_nu]に変更し
        inputs = torch.tensor([float(item["N"]), float(item["n_nu"])], dtype=torch.float32)
        
        # 教師データ
        target = torch.tensor(item["target_pes"], dtype=torch.float32)
        
        # 物理層(Decoder)用のスカラー値 (入力が[Z, N, n_pi, n_nu]の場合は不要)
        n_pi = torch.tensor(float(item["n_pi"]), dtype=torch.float32)
        n_nu = torch.tensor(float(item["n_nu"]), dtype=torch.float32)
        
        return inputs, target, n_pi, n_nu