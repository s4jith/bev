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

        # 🔥 IMPORTANT CHANGE (128 → 256)
        self.traj_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.K * 12)
        )

        self.prob_head = nn.Linear(256, self.K)

    # ----------------------------
    # SOCIAL POOLING
    # ----------------------------
    def social_pool(self, neighbor_h_list, device):
        if len(neighbor_h_list) == 0:
            return torch.zeros(128, device=device)

        return torch.mean(torch.stack(neighbor_h_list), dim=0)

    # ----------------------------
    # FORWARD PASS
    # ----------------------------
    def forward(self, x, neighbors):
        """
        x: (B, 4, 4)
        neighbors: list of length B
        """

        B = x.size(0)
        device = x.device

        # Encode main trajectory
        x = self.relu(self.embed(x))
        _, (h, _) = self.encoder(x)
        h = h.squeeze(0)  # (B, 128)

        final_h = []

        # Loop through batch (important)
        for i in range(B):
            h_target = h[i]  # (128)

            neighbor_h_list = []

            for n in neighbors[i]:
                n_tensor = torch.tensor(n, dtype=torch.float32, device=device).unsqueeze(0)

                n_tensor = self.relu(self.embed(n_tensor))
                _, (h_n, _) = self.encoder(n_tensor)

                neighbor_h_list.append(h_n.squeeze(0).squeeze(0))  # (128)

            # Social pooling
            h_social = self.social_pool(neighbor_h_list, device)

            # Combine
            h_combined = torch.cat([h_target, h_social], dim=0)  # (256)

            final_h.append(h_combined)

        h_final = torch.stack(final_h)  # (B, 256)

        # Multi-modal output
        traj = self.traj_head(h_final)
        traj = traj.view(-1, self.K, 6, 2)

        probs = self.prob_head(h_final)
        probs = torch.softmax(probs, dim=1)

        return traj, probs