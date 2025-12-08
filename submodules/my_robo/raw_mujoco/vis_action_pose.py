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

def visualize_actions(data_left, data_right):
    # Extract points and orientations (Rotation Matrices)
    left_pts = data_left["ee_pts"]
    left_oris = data_left["ee_oris"]
    
    right_pts = data_right["ee_pts"]
    right_oris = data_right["ee_oris"]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot World Frame Axes at origin
    plot_axes(ax, length=0.2)

    def plot_traj(pts, oris, label, color):
        # Downsample: every 10th frame
        indices = np.arange(0, len(pts), 10)
        points = pts[indices]
        orientations = oris[indices]
        
        if len(points) == 0:
            return

        # Plot connecting lines
        ax.plot(points[:, 0], points[:, 1], points[:, 2], label=f'{label} Path', linewidth=1, color='gray', alpha=0.5)

        # Plot spheres (scatter)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, s=30, marker='o', depthshade=True)

        # Add number labels and orientation axes
        axis_len = 0.05
        for i, (point, R) in enumerate(zip(points, orientations)):
            ax.text(point[0], point[1], point[2], str(indices[i]), color='black', fontsize=9, fontweight='bold')
            
            # Plot orientation axes (RGB for XYZ)
            # X axis (Red)
            ax.quiver(point[0], point[1], point[2], R[0, 0], R[1, 0], R[2, 0], length=axis_len, color='r')
            # Y axis (Green)
            ax.quiver(point[0], point[1], point[2], R[0, 1], R[1, 1], R[2, 1], length=axis_len, color='g')
            # Z axis (Blue)
            ax.quiver(point[0], point[1], point[2], R[0, 2], R[1, 2], R[2, 2], length=axis_len, color='b')

    plot_traj(left_pts, left_oris, "Left Hand", 'blue')
    plot_traj(right_pts, right_oris, "Right Hand", 'red')
    
    print(f"Left points shape: {left_pts.shape}")
    print(f"Right points shape: {right_pts.shape}")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Action Trajectories (Every 10 frames)')
    ax.legend()
    
    # Adjust aspect ratio to be roughly equal
    all_points = np.concatenate([left_pts, right_pts], axis=0)
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
    # Load data
    try:
        data_left = dict(np.load("free_hand_smoothed_actions_left_in_camera_optical_frame.npz"))
        data_right = dict(np.load("free_hand_smoothed_actions_right_in_camera_optical_frame.npz"))
    except FileNotFoundError:
        print("Error: smoothed_actions_left_shoulders.npz or smoothed_actions_right_shoulders.npz not found in current directory.")
        exit()

    visualize_actions(data_left, data_right)

    # Apply transformation Optical -> Camera_link
    # Original (Optical): Z-forward, X-right, Y-down
    # Target (Link): X-forward, Y-left, Z-up
    # Transformation: x' = z, y' = -x, z' = -y
    R_transform = np.array([
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0]
    ])

    # Transform positions
    data_left["ee_pts"] = (R_transform @ data_left["ee_pts"].T).T
    data_right["ee_pts"] = (R_transform @ data_right["ee_pts"].T).T

    # Transform orientations
    data_left["ee_oris"] = np.matmul(R_transform, data_left["ee_oris"])
    data_right["ee_oris"] = np.matmul(R_transform, data_right["ee_oris"])

    visualize_actions(data_left, data_right)
