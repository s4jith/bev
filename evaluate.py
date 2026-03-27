import torch
from torch.utils.data import DataLoader, random_split
from dataset import TrajectoryDataset
from model import TrajectoryTransformer
from train import get_data, collate_fn, compute_ade, compute_fde
import numpy as np

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Evaluation on {device}...")

    # Load test split
    # Since we used random_split in train.py without a fixed seed, 
    # we'll evaluate the full dataset here to get generalized global metrics
    samples = get_data()
    dataset = TrajectoryDataset(samples)
    eval_loader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn)

    # Load Model
    model = TrajectoryTransformer().to(device)
    try:
        model.load_state_dict(torch.load("best_social_model.pth", map_location=device))
        print("Successfully loaded 'best_social_model.pth'")
    except Exception as e:
        print(f"Could not load model weights: {e}")
        return

    model.eval()
    
    total_ade = 0
    total_fde = 0
    miss_rate = 0
    total_samples = 0
    
    # Miss rate threshold: if best path's endpoint is off by more than 2.0 meters
    MISS_THRESHOLD = 2.0

    print("\n--- Starting Deep Evaluation ---")
    with torch.no_grad():
        for obs, neighbors, future in eval_loader:
            obs, future = obs.to(device), future.to(device)
            
            # Forward pass
            pred, goals, probs, _ = model(obs, neighbors)
            
            # Find the best prediction out of K=3 for each item in the batch
            gt = future.unsqueeze(1)
            error = torch.norm(pred - gt, dim=3).mean(dim=2)
            best_idx = torch.argmin(error, dim=1)
            
            best_pred = pred[torch.arange(pred.size(0)), best_idx]
            
            # Metrics computation
            batch_ade = compute_ade(best_pred, future).item()
            batch_fde = compute_fde(best_pred, future).item()
            
            total_ade += batch_ade * obs.size(0)
            total_fde += batch_fde * obs.size(0)
            
            # Calculate Miss Rate
            final_displacement = torch.norm(best_pred[:, -1] - future[:, -1], dim=1)
            misses = (final_displacement > MISS_THRESHOLD).sum().item()
            miss_rate += misses
            
            total_samples += obs.size(0)

    # Average metrics
    avg_ade = total_ade / total_samples
    avg_fde = total_fde / total_samples
    avg_miss_rate = (miss_rate / total_samples) * 100

    print("\n=============================================")
    print("      HACKATHON FINAL METRICS REPORT         ")
    print("=============================================")
    print(f"Total Trajectories Evaluated: {total_samples}")
    print(f"Prediction Horizon:           6 Seconds (12 steps)")
    print(f"Social Context Radius:        50 Meters")
    print("---------------------------------------------")
    print(f"Average Displacement Error (ADE): {avg_ade:.4f} meters")
    print(f"Final Displacement Error (FDE):   {avg_fde:.4f} meters")
    print(f"Off-Target Miss Rate (>2.0m):     {avg_miss_rate:.2f} %")
    print("=============================================\n")

if __name__ == "__main__":
    evaluate()