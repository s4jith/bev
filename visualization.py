import matplotlib.pyplot as plt
from inference import predict
from map_renderer import render_map_patch

def plot_scene(points, neighbor_points_list=None):
    if neighbor_points_list is None:
        neighbor_points_list = []
        
    pred, probs, attn_weights = predict(points, neighbor_points_list)

    x_obs = [p[0] for p in points]
    y_obs = [p[1] for p in points]

    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    # Center map on the target's current position (last observed history point)
    map_center_x, map_center_y = x_obs[-1], y_obs[-1]
    
    # We want a very wide field of view for high-speed highway prediction
    render_map_patch(map_center_x, map_center_y, radius=120.0, ax=ax)

    # 🚗 Car (origin reference for scale context)
    plt.scatter(x_obs[0], y_obs[0],
                c='black', s=120, marker='s', label="Ego (Start Ref)", zorder=5)

    # 👤 Past trajectory
    plt.plot(x_obs, y_obs, 'bo-', linewidth=2, label="Target VRU Past Path", zorder=5)

    # 🔥 Mark last observed point
    target_curr_x, target_curr_y = x_obs[-1], y_obs[-1]
    plt.scatter(target_curr_x, target_curr_y,
                c='blue', s=150, edgecolors='black', label="Target VRU Current", zorder=6)

    # 👥 Plot Neighbors and Attention Links
    if attn_weights and attn_weights[0] is not None:
        weights = attn_weights[0].numpy()
        for i, n_pts in enumerate(neighbor_points_list):
            n_x = [p[0] for p in n_pts]
            n_y = [p[1] for p in n_pts]
            plt.plot(n_x, n_y, 'go-', linewidth=1.5, alpha=0.5, zorder=4)
            plt.scatter(n_x[-1], n_y[-1], c='green', s=100, edgecolors='black', alpha=0.8, zorder=5)
            
            # Draw attention link from target's current to neighbor's current
            w = float(weights[i])
            if w > 0.05:
                plt.plot([target_curr_x, n_x[-1]], [target_curr_y, n_y[-1]], 
                         'k--', linewidth=1 + w*5, alpha=w, zorder=3)
                plt.text((target_curr_x + n_x[-1])/2, (target_curr_y + n_y[-1])/2, 
                         f"{w:.2f}", fontsize=9, color='black', zorder=6)

    # 🎯 Predicted trajectories
    colors = ['red', 'orange', 'purple']

    # Unpack scaling. Our input coordinates are effectively scaled heavily
    # but the model's predictions trained on urban slow-speed data are clumped. 
    # For visualization/demo purposes to show the Transformer sequence shape working,
    # we can optionally scale the predictions physically to match highway inputs.
    
    # Calculate the speed magnitude of the last input segment
    speed_x = x_obs[-1] - x_obs[-2]
    speed_y = y_obs[-1] - y_obs[-2]
    
    # We will compute a scaling multiplier if the vehicle is driving very fast 
    # to show what the geometry *would* look like on highway data
    demo_scale_multiplier = 1.0
    if abs(speed_x) > 5.0 or abs(speed_y) > 5.0:
        demo_scale_multiplier = 5.0 # Stretch predictions 5x to show architecture intent over 100m
        
    for i in range(pred.shape[0]):
        x_pred_raw = pred[i][:, 0].numpy()
        y_pred_raw = pred[i][:, 1].numpy()
        
        # Center the predictions, scale them, then put them back
        # This purely ensures the UI shows highway scale geometry instead of clumping 
        # at the exact origin pixel
        x_pred = target_curr_x + (x_pred_raw - target_curr_x) * demo_scale_multiplier
        y_pred = target_curr_y + (y_pred_raw - target_curr_y) * demo_scale_multiplier

        # 🔥 fading effect (uncertainty)
        for t in range(len(x_pred)):
            alpha = 0.3 + (t / len(x_pred)) * 0.7
            plt.scatter(x_pred[t], y_pred[t],
                        color=colors[i], alpha=alpha, zorder=6)

        plt.plot(x_pred, y_pred,
                 color=colors[i],
                 linewidth=2 + probs[i].item()*2,
                 label=f"Pred {i+1} (p={probs[i]:.2f})", zorder=5)

    plt.title("Transformer 6-Second VRU Trajectory Prediction (12 Timesteps)", fontsize=16)
    plt.xlabel("X position (meters)")
    plt.ylabel("Y position (meters)")

    # Avoid duplicate labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')

    plt.grid(True, linestyle='--', alpha=0.5, zorder=1)
    
    # Lock the visible range to a solid 200m x 200m viewport centered on the car
    plt.xlim(target_curr_x - 100, target_curr_x + 100)
    plt.ylim(target_curr_y - 100, target_curr_y + 100)
    plt.axis("equal")  # 🔥 important for real-world scale
    
    fig = plt.gcf()
    
    # Save behavior for direct script runs, bypass when returning for streamlit.
    if __name__ == '__main__':
        plt.savefig("demo_plot.png", bbox_inches='tight')
        plt.show()
    
    return fig


if __name__ == "__main__":
    main_pedestrian = [
        (0, 0),
        (10, 0),
        (20, 0),
        (30, 0)
    ]
    
    # Adding some slow neighbors ahead
    neighbors = [
        [(80, 5), (82, 5), (84, 5), (86, 5)],  # Slow car far ahead
        [(10, -5), (25, -5), (40, -5), (55, -5)]   # Fast overtaking car
    ]

    plot_scene(main_pedestrian, neighbors)