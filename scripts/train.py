import os
import argparse
import yaml
import pandas as pd

# ============================================================
# 並列化設定 (必ずimport torchより前に書く)
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
    train_indices = []
    val_indices = []
    
    for idx, item in enumerate(dataset.data):
        n_val = item["N"]
        if n_val in val_n_list:
            val_indices.append(idx)
        else:
            train_indices.append(idx)
            
    if len(train_indices) == 0:
        raise ValueError(f"Train set is empty! val_n_list={val_n_list} covers all data.")
    
    return Subset(dataset, train_indices), Subset(dataset, val_indices)

# =============================================================================
# 1. 通常学習モード
# =============================================================================

def run_normal_training(cfg):
    print("=== Start Normal Training ===")
    
    device = torch.device(cfg.get("device", "cpu"))
    train_conf = cfg["default"]["training"]
    nn_conf = cfg["default"]["nn"]
    
    # Dataset
    full_dataset = IBM2Dataset(cfg)
    val_n_list = train_conf.get("validation_n", [88, 94])
    train_set, val_set = get_manual_split(full_dataset, val_n_list)
    
    train_loader = DataLoader(train_set, batch_size=train_conf["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=train_conf["batch_size"], shuffle=False)
    
    print(f"Data Split: Train={len(train_set)}, Val={len(val_set)}")
    print(f"  Validation Neutrons: {val_n_list}")

    # Model
    model_config = nn_conf.copy()
    # fixed_chi_pi の取得
    if "nuclei" in cfg["nuclei"]:
        fixed_chi_pi = cfg["nuclei"]["nuclei"]["fixed_chi_pi"]
        fixed_C_beta = cfg["nuclei"]["nuclei"]["fixed_C_beta"]
    else:
        fixed_chi_pi = cfg["nuclei"]["fixed_chi_pi"]
        fixed_C_beta = cfg["nuclei"]["fixed_C_beta"]
    model_config["fixed_chi_pi"] = fixed_chi_pi
    model_config["fixed_C_beta"] = fixed_C_beta
    
    model = IBM2FlexibleNet(model_config).to(device)
    decoder = IBM2PESDecoder(beta_f_grid=full_dataset.beta_grid).to(device)
    
    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=train_conf["lr"]["initial"])
    criterion = WeightedMSELoss(weight_type="reciprocal", alpha=0.5)
    
    scheduler = None
    if train_conf["lr"].get("scheduler") == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=train_conf["lr"]["patience"], 
            gamma=train_conf["lr"]["factor"]
        )

    # Training Loop
    epochs = train_conf["epochs"]
    best_loss = float('inf')
    output_dir = cfg["dirs"]["output_dir"] / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    early_stop_conf = train_conf.get("early_stopping", {})
    es_enabled = early_stop_conf.get("enabled", False)
    es_patience = early_stop_conf.get("patience", 20)
    es_counter = 0
    
    history = {"epoch": [], "train_loss": [], "val_loss": [], "lr": []}
    
    print(f"Training on {device} for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, decoder, train_loader, optimizer, criterion, device)
        if len(val_set) > 0:
            val_loss = evaluate(model, decoder, val_loader, criterion, device)
        else:
            val_loss = 0.0
        
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        
        if scheduler:
            scheduler.step()
            
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            es_counter = 0
            save_path = output_dir / "best_model.pth"
            torch.save(model.state_dict(), save_path)
        else:
            es_counter += 1
            if es_enabled and es_counter >= es_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Finished. Best Val Loss: {best_loss:.6f}")
    
    # History CSV
    history_df = pd.DataFrame(history)
    history_path = cfg["dirs"]["output_dir"] / "models" / "training_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"History saved to: {history_path}")

# =============================================================================
# 2. Optuna探索モード
# =============================================================================

def run_optuna_optimization(cfg):
    import optuna
    print("=== Start Optuna Optimization ===")
    opt_conf = cfg["optuna"]
    
    output_dir = cfg["dirs"]["output_dir"]
    model_save_dir = output_dir / "models"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    storage_url = f"sqlite:///{output_dir / opt_conf['db_name']}"

    search_space = opt_conf["search_space"]
    device = torch.device(cfg.get("device", "cpu"))
    
    full_dataset = IBM2Dataset(cfg)
    beta_grid = full_dataset.beta_grid
    
    if "nuclei" in cfg["nuclei"]:
        fixed_chi_pi = cfg["nuclei"]["nuclei"]["fixed_chi_pi"]
        fixed_C_beta = cfg["nuclei"]["nuclei"]["fixed_C_beta"]
    else:
        fixed_chi_pi = cfg["nuclei"]["fixed_chi_pi"]
        fixed_C_beta = cfg["nuclei"]["fixed_C_beta"]
    
    # ★修正: Configからinput_dimを取得 (デフォルト値)
    default_input_dim = cfg["default"]["nn"]["input_dim"]
    
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
        
        # Validation Split
        val_n_list = cfg["default"]["training"].get("validation_n", [88, 94])
        train_set, val_set = get_manual_split(full_dataset, val_n_list)
        
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        
        # ★修正: input_dim をConfigから参照
        model_config = {
            "input_dim": default_input_dim,
            "hidden_sizes": hidden_sizes,
            "activation": act_name,
            "fixed_chi_pi": fixed_chi_pi,
            "fixed_C_beta": fixed_C_beta
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
        
        return val_loss

    # Execution
    total_cores = os.cpu_count() or 1
    n_jobs = max(1, total_cores // 2)
    print(f"Running Optuna with {n_jobs} jobs (Storage: {storage_url})")

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

    # ==========================================
    # Retraining with best parameters
    # ==========================================
    print("\nRetraining with best parameters...")
    best_params = study.best_trial.params
    
    n_layers = best_params["n_layers"]
    hidden_sizes = [best_params[f"l{i}_size"] for i in range(n_layers)]
    act_name = best_params["activation"]
    lr_init = best_params["lr"]
    batch_size = best_params["batch_size"]
    
    # ★修正: Configから参照
    model_config = {
        "input_dim": default_input_dim,
        "hidden_sizes": hidden_sizes,
        "activation": act_name,
        "fixed_chi_pi": fixed_chi_pi,
        "fixed_C_beta": fixed_C_beta
    }
    
    # DataLoader
    val_n_list = cfg["default"]["training"].get("validation_n", [88, 94])
    train_set, val_set = get_manual_split(full_dataset, val_n_list)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    # Model
    model = IBM2FlexibleNet(model_config).to(device)
    decoder = IBM2PESDecoder(beta_f_grid=beta_grid).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr_init)
    criterion = WeightedMSELoss(weight_type="reciprocal", alpha=0.5)
    
    # Scheduler
    train_conf = cfg["default"]["training"]
    scheduler = None
    if train_conf["lr"].get("scheduler") == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=train_conf["lr"]["patience"], 
            gamma=train_conf["lr"]["factor"]
        )

    # Loop
    epochs = train_conf["epochs"]
    best_loss = float('inf')
    early_stop_conf = train_conf.get("early_stopping", {})
    es_enabled = early_stop_conf.get("enabled", False)
    es_patience = early_stop_conf.get("patience", 20)
    es_counter = 0
    
    history = {"epoch": [], "train_loss": [], "val_loss": [], "lr": []}
    save_path = model_save_dir / "optuna_best_model.pth"
    
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, decoder, train_loader, optimizer, criterion, device)
        if len(val_set) > 0:
            val_loss = evaluate(model, decoder, val_loader, criterion, device)
        else:
            val_loss = 0.0
        
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        
        if scheduler:
            scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Retrain Epoch {epoch}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            es_counter = 0
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
        else:
            es_counter += 1
            if es_enabled and es_counter >= es_patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Save History
    history_df = pd.DataFrame(history)
    history_path = model_save_dir / "optuna_best_history.csv"
    history_df.to_csv(history_path, index=False)
    
    # Save Config (YAML)
    config_save_path = model_save_dir / "optuna_best_config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(model_config, f, sort_keys=False)
    
    print(f"Best model saved to {save_path}")
    print(f"Config saved to {config_save_path}")
    print(f"History saved to {history_path}")

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