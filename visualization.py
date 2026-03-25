import matplotlib.pyplot as plt
from inference import predict


def plot_scene(points):
    pred, probs = predict(points)

    # extract x, y from input
    x_obs = [p[0] for p in points]
    y_obs = [p[1] for p in points]

    plt.figure(figsize=(8, 8))

    # 🚗 Car (origin)
    plt.scatter(points[0][0], points[0][1], c='black', s=100, label="Car (Origin)")

    # 👤 Past trajectory
    plt.plot(x_obs, y_obs, 'bo-', label="Observed Path")

    # 🎯 Predicted trajectories
    colors = ['r', 'g', 'orange']

    for i in range(pred.shape[0]):
        x_pred = pred[i][:, 0].numpy()
        y_pred = pred[i][:, 1].numpy()

        plt.plot(x_pred, y_pred, color=colors[i],
                 label=f"Pred {i+1} (p={probs[i]:.2f})")

    plt.title("Trajectory Prediction Demo")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    points = [
        (2, 3),
        (3, 3),
        (4, 3),
        (5, 3)
    ]

    plot_scene(points)