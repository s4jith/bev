import torch
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    def __init__(self, samples):
        self.obs = []
        self.future = []

        for obs, future in samples:
            self.obs.append(obs)
            self.future.append(future)

        self.obs = torch.tensor(self.obs, dtype=torch.float32)
        self.future = torch.tensor(self.future, dtype=torch.float32)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.future[idx]