import torch
from model import TrajectoryLSTM


# ----------------------------
# LOAD MODEL
# ----------------------------
model = TrajectoryLSTM()
model.load_state_dict(torch.load("model_phase1.pth", map_location="cpu"))
model.eval()


# ----------------------------
# PREPROCESS INPUT
# ----------------------------
def prepare_input(points):
    """
    points: list of (x, y)
    returns: (4,4) → [x, y, dx, dy]
    """

    # normalize
    x0, y0 = points[0]
    norm = [(x - x0, y - y0) for x, y in points]

    obs = []
    for i in range(len(norm)):
        if i == 0:
            dx, dy = 0, 0
        else:
            dx = norm[i][0] - norm[i-1][0]
            dy = norm[i][1] - norm[i-1][1]

        obs.append([norm[i][0], norm[i][1], dx, dy])

    return obs, (x0, y0)


# ----------------------------
# PREDICTION FUNCTION
# ----------------------------
def predict(points):
    """
    points: list of (x, y) → length must be 4
    """

    obs, origin = prepare_input(points)

    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # (1,4,4)

    with torch.no_grad():
        pred, probs = model(obs)

    pred = pred.squeeze(0)   # (K,6,2)
    probs = probs.squeeze(0)

    # convert back to real coordinates
    x0, y0 = origin
    pred_real = pred.clone()

    pred_real[:, :, 0] += x0
    pred_real[:, :, 1] += y0

    return pred_real, probs


# ----------------------------
# DEMO RUN
# ----------------------------
if __name__ == "__main__":

    # Example input (your format)
    points = [
        (2, 3),
        (3, 3),
        (4, 3),
        (5, 3)
    ]

    pred, probs = predict(points)

    print("\nInput Points:")
    print(points)

    print("\nPredicted Trajectories (Real Coordinates):")
    for i in range(pred.shape[0]):
        print(f"\nTrajectory {i+1} (prob={probs[i].item():.2f}):")
        print(pred[i])