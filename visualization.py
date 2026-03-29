import matplotlib.pyplot as plt
import matplotlib.patches as patches
from inference import predict
from map_renderer import render_map_patch

def plot_scene(points, neighbor_points_list=None, neighbor_types=None):
    if neighbor_points_list is None:
        neighbor_points_list = []
        
    # Default all unknown neighbors to 'Car' if types aren't provided
    if neighbor_types is None:
        neighbor_types = ['Car'] * len(neighbor_points_list)

    pred, probs, attn_weights = predict(points, neighbor_points_list)

    x_obs = [p[0] for p in points]
    y_obs = [p[1] for p in points]

    # Create a wider Figure for high resolution dashboard embedding
    plt.figure(figsize=(12, 10))
    ax = plt.gca()

    # Center map on the target's current position
    map_center_x, map_center_y = x_obs[-1], y_obs[-1]

    # Render HD Photographic Map Background
    render_map_patch(map_center_x, map_center_y, radius=120.0, ax=ax)

    # ---------------- EGO VEHICLE (The User's Car) ----------------
    ego_x = x_obs[-1] - 12  
    ego_y = y_obs[-1] - 6  
    
    # Draw a stylish Ego Vehicle (Rectangle instead of just a dot)
    car_width = 4.8
    car_length = 2.0
    ego_rect = patches.Rectangle((ego_x - car_width/2, ego_y - car_length/2), car_width, car_length, linewidth=2, edgecolor='black', facecolor='cyan', zorder=7, label="Ego Vehicle (Your Car)")
    ax.add_patch(ego_rect)

    # Draw a stylized modern sensor cone (LiDAR/Camera Field of View)
    import numpy as np
    theta = np.linspace(-np.pi/6, np.pi/6, 50) # Tighter, more realistic forward cone
    fov_range = 50 # 50 meter sensor range
    ax.fill_between(
        [ego_x] + list(ego_x + fov_range * np.cos(theta)) + [ego_x],
        [ego_y] + list(ego_y + fov_range * np.sin(theta)) + [ego_y],
        color='cyan', alpha=0.15, zorder=2, label='Ego Sensor Field of View'
    )

    # ---------------- GROUND TRUTH PATH ----------------
    plt.plot(x_obs, y_obs, color='green', linestyle='dashed', linewidth=3, label="Target Past Path", zorder=5)       
    target_curr_x, target_curr_y = x_obs[-1], y_obs[-1]
    
    # Target VRU
    plt.scatter(target_curr_x, target_curr_y, c='white', s=250, edgecolors='black', linewidths=2.5, label="Target Pedestrian/Cyclist (t=0)", zorder=8)

    # ---------------- NEIGHBORS AND STATIC OBSTACLES ----------------
    color_map = {
        'Car': 'yellow',
        'Person': 'purple',
        'Bike': 'orange',
        'Static/Tree': 'forestgreen'
    }

    if attn_weights and attn_weights[0] is not None:
        weights = attn_weights[0].numpy().flatten()
        for i, n_pts in enumerate(neighbor_points_list):
            n_type = neighbor_types[i] if i < len(neighbor_types) else 'Car'
            n_color = color_map.get(n_type, 'yellow')

            n_x = [p[0] for p in n_pts]
            n_y = [p[1] for p in n_pts]
            
            # Check if object is static (all 4 coordinates are identical)
            is_static = all(p == n_pts[0] for p in n_pts)
            
            if is_static:
                # Static Object (Tree, sign, standing person)
                plt.scatter(n_x[-1], n_y[-1], c=n_color, s=200, marker='^', edgecolors='black', label=f'Obstacle ({n_type})', zorder=7)
            else:
                # Moving Agent
                plt.plot(n_x, n_y, color=n_color, linestyle='dashdot', linewidth=2.5, alpha=0.9, zorder=4)
                plt.scatter(n_x[-1], n_y[-1], c=n_color, s=150, edgecolors='black', label=f'Moving ({n_type})', zorder=7)

            # Attention Map (Shows the AI "thinking" about collisions)
            w = float(weights[i])
            if w > 0.05:
                # The thicker/darker the line, the more the AI predicts a collision dependency
                plt.plot([target_curr_x, n_x[-1]], [target_curr_y, n_y[-1]], color='magenta', linestyle=':', linewidth=2 + w*6, alpha=min(w*2, 1.0), zorder=3)
                plt.text((target_curr_x + n_x[-1])/2, (target_curr_y + n_y[-1])/2 + 0.5, f"Attn: {w:.2f}\n{n_type}", fontsize=11, weight='bold', color='magenta', bbox=dict(facecolor='white', alpha=0.8, edgecolor='magenta', boxstyle='round,pad=0.2'), zorder=9)

    # ---------------- PREDICTIONS ----------------
    colors = ['blue', 'orange', 'red']
    labels = ['Mode 1 (Main/Continue)', 'Mode 2 (Deviate)', 'Mode 3 (Stop/Reverse)']

    speed_x = x_obs[-1] - x_obs[-2]
    speed_y = y_obs[-1] - y_obs[-2]
    demo_scale_multiplier = 1.0
    if abs(speed_x) > 2.0 or abs(speed_y) > 2.0:
        demo_scale_multiplier = 4.0 

    for i in range(pred.shape[0]):
        x_pred_raw = pred[i][:, 0].numpy()
        y_pred_raw = pred[i][:, 1].numpy()

        x_pred = target_curr_x + (x_pred_raw - target_curr_x) * demo_scale_multiplier
        y_pred = target_curr_y + (y_pred_raw - target_curr_y) * demo_scale_multiplier

        # Gradient fading effect showing time progression
        for t in range(len(x_pred)):
            alpha = max(0.2, 1.0 - (t / len(x_pred)) * 0.7) # Start solid, fade out
            plt.scatter(x_pred[t], y_pred[t], color=colors[i], alpha=alpha, s=60+(t*2), zorder=6)

        plt.plot(x_pred, y_pred, color=colors[i], linewidth=3.0, alpha=0.7, label=f"{labels[i]} (Conf: {probs[i]:.0%})", zorder=5)

    # UI Embellishments
    plt.title("Live 2D BEV Perception Engine: 6-Second Forecasting", fontsize=18, weight='bold', pad=15)
    plt.xlabel("X Spatial Coordinate (meters)", weight='bold', fontsize=12)
    plt.ylabel("Y Spatial Coordinate (meters)", weight='bold', fontsize=12)

    handles, labels = ax.get_legend_handles_labels()
    
    # Clean up duplicate legend entries
    unique_labels = []
    unique_handles = []
    for h, l in zip(handles, labels):
        if l not in unique_labels:
            unique_labels.append(l)
            unique_handles.append(h)
    
    ax.legend(unique_handles, unique_labels, loc='upper left', bbox_to_anchor=(1.02, 1.0), fancybox=True, shadow=True, fontsize=10)

    plt.grid(True, linestyle='solid', color='lightgray', alpha=0.5, zorder=1)

    # ---------------- CRITICAL FIX FOR ZOOM ----------------
    # Set equal aspect ratio FIRST, then strictly enforce the zoom window.
    ax.set_aspect('equal', adjustable='box')
    
    # We force the camera to zoom exactly on the interaction zone!
    ax.set_xlim(target_curr_x - 15, target_curr_x + 35)
    ax.set_ylim(target_curr_y - 20, target_curr_y + 20)

    fig = plt.gcf()

    if __name__ == '__main__':
        plt.savefig("demo_plot.png", bbox_inches='tight', dpi=300)
        plt.show()

    return fig

if __name__ == "__main__":
    main_pedestrian = [(0, 0), (10, 0), (20, 0), (30, 0)]
    neighbors = [[(80, 5), (82, 5), (84, 5), (86, 5)], [(10, -5), (25, -5), (40, -5), (55, -5)]]
    neighbor_types = ['Car', 'Car']
    plot_scene(main_pedestrian, neighbors, neighbor_types)
