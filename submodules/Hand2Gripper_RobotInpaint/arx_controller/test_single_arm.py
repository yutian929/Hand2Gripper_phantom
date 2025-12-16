import numpy as np
import cv2
import os
from scipy.spatial.transform import Rotation as R
from single_arm_controller import SingleArmController
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_and_transform_data(filepath):
    """
    Load .npz data and transform from Optical Frame to Camera Link Frame.
    Returns:
        target_seq_world (np.array): Shape (N, 6) -> [x, y, z, rx, ry, rz]
    """
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        return None

    try:
        data = np.load(filepath)
        ee_pts = data["ee_pts"]   # (N, 3)
        ee_oris = data["ee_oris"] # (N, 3, 3)
    except Exception as e:
        print(f"Error loading npz: {e}")
        return None

    # Transformation Matrix: Optical -> Camera Link
    # x' = z, y' = -x, z' = -y
    R_transform = np.array([
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0]
    ])

    # 1. Transform Positions
    # P_new = R_transform * P_old
    ee_pts_transformed = (R_transform @ ee_pts.T).T

    # 2. Transform Orientations
    # R_new = R_transform * R_old
    ee_oris_transformed = np.matmul(R_transform, ee_oris)

    # Additional 180 degree rotation around Z axis
    R_z180 = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ])
    ee_oris_transformed = np.matmul(ee_oris_transformed, R_z180)

    # 3. Convert Rotation Matrices to Euler Angles (xyz)
    # We need (N, 6) for the controller: [x, y, z, rx, ry, rz]
    N = len(ee_pts)
    target_seq = np.zeros((N, 6))
    
    target_seq[:, :3] = ee_pts_transformed
    
    # Batch convert rotations
    r = R.from_matrix(ee_oris_transformed)
    euler_angles = r.as_euler('xyz', degrees=False)
    target_seq[:, 3:] = euler_angles

    return target_seq

# ---------------------------------------------------------
# Helper Functions: 矩阵与Pose互转
# ---------------------------------------------------------
def pose_to_matrix(pose):
    """
    [x, y, z, rx, ry, rz] -> 4x4 Homogeneous Matrix
    """
    t = pose[:3]
    euler = pose[3:]
    
    r = R.from_euler('xyz', euler, degrees=False)
    mat = np.eye(4)
    mat[:3, :3] = r.as_matrix()
    mat[:3, 3] = t
    return mat

def matrix_to_pose(mat):
    """
    4x4 Homogeneous Matrix -> [x, y, z, rx, ry, rz]
    """
    t = mat[:3, 3]
    rot_mat = mat[:3, :3]
    
    r = R.from_matrix(rot_mat)
    euler = r.as_euler('xyz', degrees=False)
    
    return np.concatenate([t, euler])

def visualize_trajectory(target_seq, title="Trajectory Visualization"):
    """
    Visualize the trajectory using Matplotlib.
    Args:
        target_seq (np.array): Shape (N, 6) -> [x, y, z, rx, ry, rz]
    """
    points = target_seq[:, :3]
    euler_angles = target_seq[:, 3:]
    
    # Convert Euler to Rotation Matrices for visualization
    r = R.from_euler('xyz', euler_angles, degrees=False)
    matrices = r.as_matrix()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot World Frame Axes at origin
    ax.quiver(0, 0, 0, 0.1, 0, 0, color='r', arrow_length_ratio=0.3)
    ax.text(0.1, 0, 0, 'X', color='r')
    ax.quiver(0, 0, 0, 0, 0.1, 0, color='g', arrow_length_ratio=0.3)
    ax.text(0, 0.1, 0, 'Y', color='g')
    ax.quiver(0, 0, 0, 0, 0, 0.1, color='b', arrow_length_ratio=0.3)
    ax.text(0, 0, 0.1, 'Z', color='b')

    # Plot path line
    ax.plot(points[:, 0], points[:, 1], points[:, 2], label='Trajectory Path', linewidth=1, color='gray')
    
    # Plot points and orientation
    # Downsample for clarity (every 5th point)
    indices = np.arange(0, len(points), 5) 
    
    for i in indices:
        pt = points[i]
        rot = matrices[i]
        
        # Plot point
        ax.scatter(pt[0], pt[1], pt[2], c='r', marker='o', s=20)
        
        # Plot orientation axes
        length = 0.02
        # X axis (Red)
        ax.quiver(pt[0], pt[1], pt[2], rot[0, 0], rot[1, 0], rot[2, 0], length=length, color='r')
        # Y axis (Green)
        ax.quiver(pt[0], pt[1], pt[2], rot[0, 1], rot[1, 1], rot[2, 1], length=length, color='g')
        # Z axis (Blue)
        ax.quiver(pt[0], pt[1], pt[2], rot[0, 2], rot[1, 2], rot[2, 2], length=length, color='b')
        
        # Label start and end
        if i == 0:
            ax.text(pt[0], pt[1], pt[2], 'Start', color='black', fontweight='bold')
        if i == indices[-1]:
            ax.text(pt[0], pt[1], pt[2], 'End', color='black', fontweight='bold')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Equal aspect ratio hack
    all_points = points
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
    
    plt.legend()
    plt.show()

if __name__ == "__main__":
    camera_pose_in_base = np.array([0.0, 0.0, 0.5, 0.0, 1.0, 0.0])
    Mat_base_T_camera = pose_to_matrix(camera_pose_in_base)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "R5/R5a/meshes/single_arm_scene.xml")
    data_path = os.path.join(current_dir, "free_hand_N_to_F_smoothed_actions_right_in_camera_optical_frame.npz")

    robot = SingleArmController(xml_path)
    robot_base_pose_world = robot._get_base_pose_world()
    Mat_world_T_base = pose_to_matrix(robot_base_pose_world)
    print(f"Mat_world_T_base:\n{Mat_world_T_base}")
    
    seqs_in_camera_link = load_and_transform_data(data_path)
    Mat_camera_T_seqs = np.array([pose_to_matrix(pose) for pose in seqs_in_camera_link])
    print(f"Mat_camera_T_seqs shape: {Mat_camera_T_seqs.shape}")
    # visualize_trajectory(seqs_in_camera_link, title="Trajectory in Camera Link Frame")

    Mat_world_T_seqs = Mat_world_T_base @ Mat_base_T_camera @ Mat_camera_T_seqs
    seqs_in_world = np.array([matrix_to_pose(mat) for mat in Mat_world_T_seqs])
    print(f"seqs_in_world shape: {seqs_in_world.shape}")
    # visualize_trajectory(seqs_in_world, title="Trajectory in World Frame")

    # assume camera base fixed
    Mat_world_T_camera = Mat_world_T_base @ Mat_base_T_camera
    camera_poses_in_world = np.array([matrix_to_pose(Mat_world_T_camera) for _ in range(len(seqs_in_world))])
    print(f"camera_poses_in_world shape: {camera_poses_in_world.shape}")
    # visualize_trajectory(camera_poses_in_world, title="Camera Poses in World Frame")

    if seqs_in_world is not None:
        try:
            print("########## Executing move_trajectory ##########")
            robot.move_trajectory(seqs_in_world, kinematic_only=True)
            print("########## Executing move_trajectory_with_camera ##########")
            frames, masks = robot.move_trajectory_with_camera(seqs_in_world, camera_poses_in_world, kinematic_only=True)
            
            for i, (frame, mask) in enumerate(zip(frames, masks)):
                cv2.imshow("Frame", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                
                geom_ids = mask[:, :, 0]
                # ID 0 -> Black [0,0,0], Others -> White [255,255,255]
                mask_vis = np.zeros((geom_ids.shape[0], geom_ids.shape[1], 3), dtype=np.uint8)
                mask_vis[geom_ids > 0] = [255, 255, 255]
                
                cv2.imshow("Mask", mask_vis)
                cv2.waitKey(100)
            
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Execution Error: {e}")
    else:
        print("Failed to load trajectory data.")
