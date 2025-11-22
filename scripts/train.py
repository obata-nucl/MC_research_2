import os
import argparse

# ============================================================
# 並列化設定
# ============================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from src.dataset import IBM2Dataset
from src.losses import WeightedMSELoss
from src.model import IBM2FlexibleNet, IBM2PESDecoder
from src.utils import load_config, set_seed

# =============================================================================
# Helper Functions
# =============================================================================

def train_one_epoch(model, decoder, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    
    for inputs, targets, n_pi, n_nu in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        n_pi = n_pi.to(device)
        n_nu = n_nu.to(device)
        
        optimizer.zero_grad()
        params = model(inputs)
        preds = decoder(params, n_pi, n_nu)
        loss = criterion(preds, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        
    return total_loss / len(loader.dataset)


def evaluate(model, decoder, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets, n_pi, n_nu in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            n_pi = n_pi.to(device)
            n_nu = n_nu.to(device)
            
            params = model(inputs)
            preds = decoder(params, n_pi, n_nu)
            loss = criterion(preds, targets)
            
            total_loss += loss.item() * inputs.size(0)
            
    return total_loss / len(loader.dataset)


def get_manual_split(dataset, val_n_list):
    """
    datasetの中身を見て、Nが val_n_list に含まれるものをValidationにする
    """
    train_indices = []
    val_indices = []
    
    # dataset.data はリスト形式の辞書
    for idx, item in enumerate(dataset.data):
        n_val = item["N"]
        if n_val in val_n_list:
            val_indices.append(idx)
        else:
            train_indices.append(idx)
            
    # 安全確認
    if len(train_indices) == 0:
        raise ValueError(f"Train set is empty! val_n_list={val_n_list} covers all data.")
    
    # Subsetを使って分割
    return Subset(dataset, train_indices), Subset(dataset, val_indices)

# =============================================================================
# 1. 通常学習モード
# =============================================================================

def run_normal_training(cfg):
    print("=== Start Normal Training ===")
    
    device = torch.device(cfg.get("device", "cpu"))
    train_conf = cfg["default"]["training"]
    nn_conf = cfg["default"]["nn"]
    
    # Dataset準備
    full_dataset = IBM2Dataset(cfg)

    # training.yml から "validation_n" (リスト) を取得
    # 設定がなければデフォルトとして [88, 94] などを検証用にする
    val_n_list = train_conf.get("validation_n", [88, 94])
    
    train_set, val_set = get_manual_split(full_dataset, val_n_list)
    
    # データローダー (Valはシャッフル不要)
    train_loader = DataLoader(train_set, batch_size=train_conf["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=train_conf["batch_size"], shuffle=False)
    
    print(f"Data Split: Train={len(train_set)}, Val={len(val_set)}")
    print(f"  Validation Neutrons: {val_n_list}")

    # モデル構築
    model_config = nn_conf.copy()
    model_config["fixed_chi_pi"] = cfg["nuclei"]["fixed_chi_pi"]
    
    model = IBM2FlexibleNet(model_config).to(device)
    
    # Decoder
    beta_grid = full_dataset.beta_grid
    decoder = IBM2PESDecoder(beta_f_grid=beta_grid).to(device)
    
    # Optimizer & Loss
    lr_conf = train_conf["lr"]
    optimizer = optim.Adam(model.parameters(), lr=lr_conf["initial"])
    criterion = WeightedMSELoss(weight_type="reciprocal", alpha=0.5)
    
    scheduler = None
    if lr_conf.get("scheduler") == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=lr_conf["patience"], 
            gamma=lr_conf["factor"]
        )

    # 学習ループ
    epochs = train_conf["epochs"]
    best_loss = float('inf')
    output_dir = cfg["dirs"]["output_dir"] / "models"
    
    early_stop_conf = train_conf.get("early_stopping", {})
    es_enabled = early_stop_conf.get("enabled", False)
    es_patience = early_stop_conf.get("patience", 20)
    es_counter = 0
    
    print(f"Training on {device} for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, decoder, train_loader, optimizer, criterion, device)
        # Valデータが無い場合(空リスト)のエラー回避
        if len(val_set) > 0:
            val_loss = evaluate(model, decoder, val_loader, criterion, device)
        else:
            val_loss = 0.0
        
        if scheduler:
            scheduler.step()
            
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            es_counter = 0
            save_path = output_dir / "best_model.pth"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
        else:
            es_counter += 1
            if es_enabled and es_counter >= es_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Finished. Best Val Loss: {best_loss:.6f}")

# =============================================================================
# 2. Optuna探索モード
# =============================================================================

def run_optuna_optimization(cfg):
    import optuna
    print("=== Start Optuna Optimization ===")
    opt_conf = cfg["optuna"]
    
    output_dir = cfg["dirs"]["output_dir"]
    model_save_dir = output_dir / "models"
    storage_url = f"sqlite:///{output_dir / opt_conf['db_name']}"

    search_space = opt_conf["search_space"]
    device = torch.device(cfg.get("device", "cpu"))
    
    full_dataset = IBM2Dataset(cfg)
    beta_grid = full_dataset.beta_grid
    fixed_chi_pi = cfg["nuclei"]["fixed_chi_pi"]
    
    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 
                                     search_space["hidden_layers_min"], 
                                     search_space["hidden_layers_max"])
        hidden_sizes = []
        for i in range(n_layers):
            size = trial.suggest_categorical(f"l{i}_size", search_space["node_candidates"])
            hidden_sizes.append(size)
            
        act_name = trial.suggest_categorical("activation", search_space["activation_list"])
        lr_init = trial.suggest_float("lr", float(search_space["lr_initial_min"]), float(search_space["lr_initial_max"]), log=True)
        batch_size = trial.suggest_categorical("batch_size", search_space["batch_size_list"])
        
        # --- ★修正: 探索時も固定Validation ---
        # 探索時こそ、汎化性能(内挿能力)を正しく評価するために固定セットを使うべき
        val_n_list = cfg["default"]["training"].get("validation_n", [88, 94])
        train_set, val_set = get_manual_split(full_dataset, val_n_list)
        
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        
        model_config = {
            "input_dim": 2,
            "hidden_sizes": hidden_sizes,
            "activation": act_name,
            "fixed_chi_pi": fixed_chi_pi
        }
        
        model = IBM2FlexibleNet(model_config).to(device)
        decoder = IBM2PESDecoder(beta_f_grid=beta_grid).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr_init)
        criterion = WeightedMSELoss(weight_type="reciprocal", alpha=0.5)
        
        n_epochs = 50
        
        for epoch in range(n_epochs):
            train_one_epoch(model, decoder, train_loader, optimizer, criterion, device)
            val_loss = evaluate(model, decoder, val_loader, criterion, device)
            
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        try:
            best_so_far = study.best_value
        except ValueError:
            best_so_far = float('inf')

        if val_loss < best_so_far:
            save_path = model_save_dir / "optuna_best_model.pth"
            torch.save(model.state_dict(), save_path)
                
        return val_loss

    total_cores = os.cpu_count() or 1
    n_jobs = max(1, total_cores // 2)
    print(f"Running Optuna with {n_jobs} jobs")

    study = optuna.create_study(
        study_name=opt_conf["study_name"],
        storage=storage_url,
        direction=opt_conf["direction"],
        load_if_exists=True
    )
    
    study.optimize(objective, n_trials=opt_conf["n_trials"], n_jobs=n_jobs)
    
    print("Best trial:")
    print(f"  Value: {study.best_trial.value}")
    print(f"  Params: {study.best_trial.params}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optuna", action="store_true", help="Enable Optuna optimization")
    args = parser.parse_args()
    
    cfg = load_config()
    set_seed(cfg.get("seed", 42))
    
    if args.optuna:
        run_optuna_optimization(cfg)
    else:
        run_normal_training(cfg)