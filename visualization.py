import matplotlib.pyplot as plt
from inference import predict


def plot_scene(points):
    pred, probs = predict(points)

    x_obs = [p[0] for p in points]
    y_obs = [p[1] for p in points]

    plt.figure(figsize=(8, 8))

    # 🚗 Car (origin reference)
    plt.scatter(points[0][0], points[0][1],
                c='black', s=120, marker='s', label="Car (Origin)")

    # 👤 Past trajectory
    plt.plot(x_obs, y_obs, 'bo-', linewidth=2, label="Observed Path")

    # 🔥 Mark last observed point
    plt.scatter(x_obs[-1], y_obs[-1],
                c='blue', s=100, edgecolors='black', label="Current Position")

    # 🎯 Predicted trajectories
    colors = ['red', 'green', 'orange']

    for i in range(pred.shape[0]):
        x_pred = pred[i][:, 0].numpy()
        y_pred = pred[i][:, 1].numpy()

        # 🔥 fading effect (uncertainty)
        for t in range(len(x_pred)):
            alpha = 0.3 + (t / len(x_pred)) * 0.7
            plt.scatter(x_pred[t], y_pred[t],
                        color=colors[i], alpha=alpha)

        plt.plot(x_pred, y_pred,
                 color=colors[i],
                 linewidth=2,
                 label=f"Pred {i+1} (p={probs[i]:.2f})")

    plt.title("Multi-Modal Trajectory Prediction (Social-Aware)",
              fontsize=14)

    plt.xlabel("X position (meters)")
    plt.ylabel("Y position (meters)")

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.axis("equal")  # 🔥 important for real-world scale

    plt.show()


if __name__ == "__main__":
    points = [
        (2, 3),
        (3, 3),
        (4, 3),
        (5, 3)
    ]

    plot_scene(points)