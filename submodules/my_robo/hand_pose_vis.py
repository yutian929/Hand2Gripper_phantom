import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_axes(ax, length=0.1):
    """Draw a world coordinate frame at the origin."""
    # X axis (Red)
    ax.quiver(0, 0, 0, length, 0, 0, color='r', arrow_length_ratio=0.1)
    ax.text(length, 0, 0, 'X', color='r')
    # Y axis (Green)
    ax.quiver(0, 0, 0, 0, length, 0, color='g', arrow_length_ratio=0.1)
    ax.text(0, length, 0, 'Y', color='g')
    # Z axis (Blue)
    ax.quiver(0, 0, 0, 0, 0, length, color='b', arrow_length_ratio=0.1)
    ax.text(0, 0, length, 'Z', color='b')

def visualize_hand_centroids():
    # Load data
    try:
        data_left = np.load("hand_data_left.npz")
        data_right = np.load("hand_data_right.npz")
    except FileNotFoundError:
        print("Error: hand_data_left.npz or hand_data_right.npz not found in current directory.")
        return

    # Calculate centroids (N, 3)
    # kpts_3d shape is typically (N, 21, 3)
    left_centroids = np.mean(data_left["kpts_3d"], axis=1)
    right_centroids = np.mean(data_right["kpts_3d"], axis=1)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot World Frame Axes at origin
    plot_axes(ax, length=0.2)

    def plot_traj(centroids, label, color):
        # Downsample: every 10th frame
        indices = np.arange(0, len(centroids), 10)
        points = centroids[indices]
        
        if len(points) == 0:
            return

        # Plot connecting lines
        ax.plot(points[:, 0], points[:, 1], points[:, 2], label=f'{label} Path', linewidth=1, color='gray', alpha=0.5)

        # Plot spheres (scatter)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, s=30, marker='o', depthshade=True)

        # Add number labels 1, 2, 3...
        for i, point in enumerate(points):
            ax.text(point[0], point[1], point[2], str(i + 1), color='black', fontsize=9, fontweight='bold')

    plot_traj(left_centroids, "Left Hand", 'blue')
    plot_traj(right_centroids, "Right Hand", 'red')
    print(f"left_centroids: {left_centroids}")
    print(f"right_centroids: {right_centroids}")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Hand Centroid Trajectories (Every 10 frames)')
    ax.legend()
    
    # Adjust aspect ratio to be roughly equal
    all_points = np.concatenate([left_centroids, right_centroids], axis=0)
    if len(all_points) > 0:
        max_range = np.array([
            all_points[:,0].max()-all_points[:,0].min(), 
            all_points[:,1].max()-all_points[:,1].min(), 
            all_points[:,2].max()-all_points[:,2].min()
        ]).max() / 2.0
        
        mid_x = (all_points[:,0].max()+all_points[:,0].min()) * 0.5
        mid_y = (all_points[:,1].max()+all_points[:,1].min()) * 0.5
        mid_z = (all_points[:,2].max()+all_points[:,2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()

if __name__ == "__main__":
    visualize_hand_centroids()
