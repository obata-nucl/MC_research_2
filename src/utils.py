import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(config_dir: str = "./configs", nuclei_file: str = "nuclei", training_file: str = "training") -> dict:
    """ 設定ファイルを読み込み、パス解決を行って結合した辞書を返す """
    config = {}
    config_dir = Path(config_dir)

    def _load_yaml(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    # 各種ファイルのロード
    base_config = _load_yaml(config_dir / "base.yml")
    config.update(base_config)

    nuclei_path = config_dir / f"{nuclei_file}.yml"
    nuclei_config = _load_yaml(nuclei_path)
    config.update(nuclei_config)

    training_path = config_dir / f"{training_file}.yml"
    training_config = _load_yaml(training_path)
    config.update(training_config)

    # パスの解決と結合
    dirs = config.get("dirs", {})

    # 起点となるディレクトリ
    data_root = Path(dirs.get("data_dir", "./data"))
    output_root = Path(dirs.get("output_dir", "./output"))
    scripts_root = Path(dirs.get("scripts_dir", "./scripts"))
    src_root = Path(dirs.get("src_dir", "./src"))
    
    # サブディレクトリの結合
    raw_dir = data_root / dirs.get("raw_dir", "raw")
    processed_dir = data_root / dirs.get("processed_dir", "processed")

    config["dirs"]["data_dir"] = data_root
    config["dirs"]["raw_dir"] = raw_dir
    config["dirs"]["processed_dir"] = processed_dir
    config["dirs"]["output_dir"] = output_root
    config["dirs"]["scripts_dir"] = scripts_root
    config["dirs"]["src_dir"] = src_root

    return config

def set_seed(seed: int = 42) -> None:
    """ 再現性のためのシード設定 """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"global seed set to {seed}")
    return