import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

from data_loader import main as load_data
from dataset import TrajectoryDataset
from model import TrajectoryLSTM


def get_data():
    # reuse your loader logic
    from data_loader import (
        load_json, extract_pedestrian_instances,
        build_trajectories, create_windows
    )

    sample_annotations = load_json("sample_annotation")
    instances = load_json("instance")
    categories = load_json("category")

    ped_instances = extract_pedestrian_instances(
        sample_annotations, instances, categories
    )

    trajectories = build_trajectories(sample_annotations, ped_instances)
    samples = create_windows(trajectories)

    return samples


def compute_ade(pred, gt):
    return torch.mean(torch.norm(pred - gt, dim=2))


def compute_fde(pred, gt):
    return torch.mean(torch.norm(pred[:, -1] - gt[:, -1], dim=1))


def train():
    samples = get_data()
    dataset = TrajectoryDataset(samples)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)

    model = TrajectoryLSTM()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        model.train()
        total_loss = 0

        for obs, future in train_loader:
            pred, probs = model(obs)
            loss = best_of_k_loss(pred, future)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()
        ade, fde = 0, 0

        with torch.no_grad():
            for obs, future in val_loader:
                pred, _ = model(obs)

                gt = future.unsqueeze(1)  # (B,1,6,2)

                error = torch.norm(pred - gt, dim=3).mean(dim=2)  # (B,K)

                best_idx = torch.argmin(error, dim=1)

                best_pred = pred[torch.arange(pred.size(0)), best_idx]

                ade += compute_ade(best_pred, future).item()
                fde += compute_fde(best_pred, future).item()

        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {total_loss:.4f}")
        print(f"ADE: {ade:.4f}, FDE: {fde:.4f}")
        print("-" * 40)

    torch.save(model.state_dict(), "model_phase1.pth")


def best_of_k_loss(pred, gt):
    # pred: (B, K, 6, 2)
    # gt: (B, 6, 2)

    gt = gt.unsqueeze(1)  # (B, 1, 6, 2)

    # compute error for each K
    error = torch.norm(pred - gt, dim=3)  # (B, K, 6)
    error = torch.mean(error, dim=2)      # (B, K)

    # pick best trajectory
    min_error, _ = torch.min(error, dim=1)

    return torch.mean(min_error)


if __name__ == "__main__":
    train()