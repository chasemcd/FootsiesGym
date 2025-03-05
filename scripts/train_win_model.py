import glob
import gzip
import json
from itertools import product
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

DATA_NAME = "test-trajectory-export"


class WinPredictionModel(nn.Module):
    def __init__(self, input_size: int, layer_sizes: List[int]):
        super().__init__()

        layers = []
        prev_size = input_size
        for size in layer_sizes:
            layers.extend(
                [
                    nn.Linear(prev_size, size),
                    nn.ReLU(),
                ]
            )
            prev_size = size

        layers.extend([nn.Linear(prev_size, 1), nn.Sigmoid()])

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TrajectoryDataset(Dataset):
    def __init__(self, data_path: str):
        self.all_obs = []
        self.all_labels = []

        # Load all trajectory files
        trajectory_files = glob.glob(f"{data_path}/*.json.gz")
        for file_path in tqdm(trajectory_files, desc="Loading data"):
            with gzip.open(file_path, "rt", encoding="utf-8") as f:
                # Read the JSON directly from the gzipped file
                trajectories = json.load(f)

                for trajectory in trajectories:
                    # Combine p1 and p2 observations for each timestep
                    for t in range(len(trajectory["t"])):
                        obs = trajectory["last_encoding"][t]
                        self.all_obs.append(obs)
                        self.all_labels.append(trajectory["p1_win"][t])

        self.all_obs = torch.FloatTensor(self.all_obs)
        self.all_labels = torch.FloatTensor(self.all_labels)

    def __len__(self):
        return len(self.all_obs)

    def __getitem__(self, idx):
        return self.all_obs[idx], self.all_labels[idx]


def evaluate_model(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: str
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for obs, labels in dataloader:
            obs = obs.to(device)
            labels = labels.to(device)

            predictions = model(obs)
            loss = criterion(predictions, labels.unsqueeze(1))
            total_loss += loss.item()

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    auc_score = roc_auc_score(all_labels, all_preds)
    return avg_loss, auc_score


def train_win_model(
    config: Dict[str, Any],
    data_path: str,
    patience: int = 20,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[float, float, nn.Module, Dict[str, List[float]]]:
    # Load dataset
    full_dataset = TrajectoryDataset(data_path)

    # Split into train and validation sets
    val_size = int(len(full_dataset) * config["val_split"])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        # persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        # persistent_workers=True,
    )

    # Initialize model
    input_size = full_dataset.all_obs.shape[1]
    model = WinPredictionModel(input_size, config["layer_sizes"]).to(device)

    # Setup training
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # Keep track of best model
    best_val_loss = float("inf")
    best_model_state = None

    # Track metrics
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    # Training loop
    for epoch in range(config["num_epochs"]):
        # Training phase
        model.train()
        total_train_loss = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config['num_epochs']}",
            leave=False,
        )

        for obs, labels in progress_bar:
            obs = obs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            predictions = model(obs)
            loss = criterion(predictions, labels.unsqueeze(1))

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Evaluation phase
        val_loss, val_auc = evaluate_model(
            model, val_loader, criterion, device
        )

        # Record metrics
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        # Save best model based on validation loss instead of AUC
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Load best model state
    model.load_state_dict(best_model_state)
    return val_auc, best_val_loss, model, history


def plot_training_history(history: Dict[str, List[float]], save_path: str):
    """Plot and save training history."""
    plt.figure(figsize=(15, 5))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    # Plot validation AUC
    plt.subplot(1, 2, 2)
    plt.plot(history["val_auc"], label="Validation AUC", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("Validation AUC")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    # Create models directory if it doesn't exist
    import os

    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # Define parameter grid
    param_grid = {
        "lr": [1e-4, 3e-4],
        "batch_size": [32, 128, 512, 1024],
        "num_epochs": [200],
        "val_split": [0.2],
        "layer_sizes": [
            [64, 64],
            [128, 128],
            [128, 64],
            [256, 128],
            [512, 256],
            [128, 64, 32],
        ],
    }

    # cur_best = {
    #     "lr": 0.0003,
    #     "batch_size": 32,
    #     "num_epochs": 200,
    #     "val_split": 0.2,
    #     "layer_sizes": [64, 64],
    # }

    # Generate all combinations of parameters
    param_combinations = [
        dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())
    ]
    print(f"Total number of combinations: {len(param_combinations)}")

    # Track best parameters and results
    best_loss = float("inf")
    best_config = None
    best_model = None
    results = []

    # Run grid search
    for i, config in enumerate(param_combinations, 1):
        print(f"\nTrying combination {i}/{len(param_combinations)}:")
        print(json.dumps(config, indent=2))

        val_auc, val_loss, model, history = train_win_model(
            config, data_path=f"FootsiesTrajectories/{DATA_NAME}"
        )

        # Plot and save training history
        plot_name = f"training_history_run_{i}.png"
        plot_training_history(history, f"plots/{plot_name}")

        results.append(
            {
                "config": config,
                "val_auc": val_auc,
                "val_loss": val_loss,
                "plot_path": plot_name,
                "history": history,
            }
        )

        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation AUC: {val_auc:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_config = config
            best_model = model
            # Save the best run's plot separately
            plot_training_history(history, "plots/best_training_history.png")
            print(f"New best model found! Loss: {val_loss:.4f}")

    # Save the best model
    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "config": best_config,
            "val_auc": val_auc,
        },
        "models/win_predictor.pt",
    )

    # Save all results
    with open("models/grid_search_results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Save best configuration
    with open("models/best_config.json", "w") as f:
        json.dump(best_config, f, indent=4)

    print("\nGrid search completed!")
    print(f"Best configuration: {json.dumps(best_config, indent=2)}")
    print(f"Best validation loss: {best_loss:.4f}")
