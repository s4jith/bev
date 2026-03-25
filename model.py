import torch
import torch.nn as nn

class TrajectoryLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed = nn.Linear(4, 64)
        self.relu = nn.ReLU()

        self.encoder = nn.LSTM(
            input_size=64,
            hidden_size=128,
            batch_first=True
        )

        self.K = 3  # number of trajectories

        self.traj_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.K * 12)
        )

        self.prob_head = nn.Linear(128, self.K)

    def forward(self, x):
        # x: (B, 4, 4)

        x = self.relu(self.embed(x))  # (B, 4, 64)

        _, (h, _) = self.encoder(x)   # h: (1, B, 128)
        h = h.squeeze(0)              # (B, 128)

        traj = self.traj_head(h)              # (B, K*12)
        traj = traj.view(-1, self.K, 6, 2)    # (B, K, 6, 2)

        probs = self.prob_head(h)             # (B, K)
        probs = torch.softmax(probs, dim=1)

        return traj, probs