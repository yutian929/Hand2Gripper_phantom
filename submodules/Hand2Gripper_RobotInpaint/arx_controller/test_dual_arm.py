import cv2
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import mediapy as media
from scipy.spatial.transform import Rotation as R
from hand2gripper_robot_inpaint.arx_controller.mujoco_dual_arm_controller import DualArmController

def load_calibration_matrix(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return np.array(data["Mat_base_T_camera_link"])

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
    # R_z180 = np.array([
    #     [-1, 0, 0],
    #     [0, -1, 0],
    #     [0, 0, 1]
    # ])
    # ee_oris_transformed = np.matmul(ee_oris_transformed, R_z180)

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

    # if nan in points or euler_angles, skip visualization
    if np.isnan(points).any() or np.isnan(euler_angles).any():
        print("Warning: NaN values found in trajectory data. Skipping visualization.")
        return
    
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
# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    # Load calibration matrices from JSON
    calib_path_L = "/home/yutian/Hand2Gripper_phantom/phantom/camera/eye_to_hand_result_left_latest.json"
    calib_path_R = "/home/yutian/Hand2Gripper_phantom/phantom/camera/eye_to_hand_result_right_latest.json"

    try:
        Mat_base_L_T_camera = load_calibration_matrix(calib_path_L)
        print(f"Loaded Left Calibration from {calib_path_L}")
    except Exception as e:
        print(f"Failed to load Left Calibration: {e}")
        # Fallback or exit
        exit(1)

    try:
        Mat_base_R_T_camera = load_calibration_matrix(calib_path_R)
        print(f"Loaded Right Calibration from {calib_path_R}")
    except Exception as e:
        print(f"Failed to load Right Calibration: {e}")
        exit(1)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "R5", "R5a", "meshes", "dual_arm_scene.xml")
    
    # Data paths
    data_path_L = "/home/yutian/Hand2Gripper_phantom/data/processed/epic/0/smoothing_processor/smoothed_actions_left_shoulders.npz"
    data_path_R = "/home/yutian/Hand2Gripper_phantom/data/processed/epic/0/smoothing_processor/smoothed_actions_right_shoulders.npz"
    human_inpaint_video_path = "/home/yutian/Hand2Gripper_phantom/data/processed/epic/0/inpaint_processor/video_human_inpaint.mkv"
    human_video_path = "/home/yutian/Hand2Gripper_phantom/data/processed/epic/0/video_L.mp4"

    dual_robot = DualArmController(xml_path)
    base_pose_world_L = dual_robot._get_base_pose_world("L")
    base_pose_world_R = dual_robot._get_base_pose_world("R")
    Mat_world_T_base_L = pose_to_matrix(base_pose_world_L)
    Mat_world_T_base_R = pose_to_matrix(base_pose_world_R)
    print("Mat_world_T_base_L:\n", np.round(Mat_world_T_base_L, 2))
    print("Mat_world_T_base_R:\n", np.round(Mat_world_T_base_R, 2))
    
    seqs_L_in_camera_link = load_and_transform_data(data_path_L)
    seqs_R_in_camera_link = load_and_transform_data(data_path_R)
    Mat_camera_T_seqs_L = np.array([pose_to_matrix(pose) for pose in seqs_L_in_camera_link])
    Mat_camera_T_seqs_R = np.array([pose_to_matrix(pose) for pose in seqs_R_in_camera_link])
    print(f"Mat_camera_T_seqs_L shape: {Mat_camera_T_seqs_L.shape}")
    print(f"Mat_camera_T_seqs_R shape: {Mat_camera_T_seqs_R.shape}")
    visualize_trajectory(seqs_L_in_camera_link, title="Left Arm Trajectory in Camera Link Frame")
    visualize_trajectory(seqs_R_in_camera_link, title="Right Arm Trajectory in Camera Link Frame")

    Mat_world_T_seqs_L = Mat_world_T_base_L @ Mat_base_L_T_camera @ Mat_camera_T_seqs_L
    Mat_world_T_seqs_R = Mat_world_T_base_R @ Mat_base_R_T_camera @ Mat_camera_T_seqs_R
    seqs_L_in_world = np.array([matrix_to_pose(mat) for mat in Mat_world_T_seqs_L])
    seqs_R_in_world = np.array([matrix_to_pose(mat) for mat in Mat_world_T_seqs_R])
    print(f"seqs_L_in_world shape: {seqs_L_in_world.shape}")
    print(f"seqs_R_in_world shape: {seqs_R_in_world.shape}")
    visualize_trajectory(seqs_L_in_world, title="Left Arm Trajectory in World Frame")
    visualize_trajectory(seqs_R_in_world, title="Right Arm Trajectory in World Frame")

    # assume camera base fixed
    Mat_world_T_camera_L = Mat_world_T_base_L @ Mat_base_L_T_camera
    Mat_world_T_camera_R = Mat_world_T_base_R @ Mat_base_R_T_camera
    # These 2 Mat should be very close
    diff = np.linalg.norm(Mat_world_T_camera_L - Mat_world_T_camera_R)
    assert diff < 0.01, f"Camera world transforms from two arms differ significantly! Difference: {diff}"
    camera_poses_world = np.array([matrix_to_pose(Mat_world_T_camera_L) for _ in range(len(seqs_L_in_world))])

    if seqs_L_in_world is not None and seqs_R_in_world is not None:
        try:
            print("########## Executing dual arm move_trajectory ##########")
            # dual_robot.move_trajectory(seqs_L_in_world, seqs_R_in_world, 50, kinematic_only=True)
            print("########## Executing dual arm move_trajectory_with_camera ##########")
            frames, masks = dual_robot.move_trajectory_with_camera(
                seqs_L_in_world, seqs_R_in_world, camera_poses_world)
            rgbs_human = media.read_video(human_video_path)
            rgbs_inpaint = media.read_video(human_inpaint_video_path)
            for i, (rgb_human, rgb_inpaint, frame, mask) in enumerate(zip(rgbs_human, rgbs_inpaint, frames, masks)):
                cv2.imshow("Human Original Video", cv2.cvtColor(rgb_human, cv2.COLOR_RGB2BGR))
                cv2.imshow("Human Inpainted Video", cv2.cvtColor(rgb_inpaint, cv2.COLOR_RGB2BGR))
                cv2.imshow("Frame in mujoco", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                
                geom_ids = mask[:, :, 0]
                # ID 0 -> Black [0,0,0], Others -> White [255,255,255]
                mask_vis = np.zeros((geom_ids.shape[0], geom_ids.shape[1], 3), dtype=np.uint8)
                mask_vis[geom_ids > 0] = [255, 255, 255]
                
                cv2.imshow("Mask in mujoco", mask_vis)

                mask_bool = geom_ids > 0
                blended = rgb_inpaint.copy()
                blended[mask_bool] = frame[mask_bool]
                cv2.imshow("Blended Frame", cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
                
                cv2.waitKey(100)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error during dual arm trajectory execution: {e}")
    else:
        print("Failed to load trajectory data for one or both arms.")
