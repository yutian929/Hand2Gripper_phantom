import numpy as np
import os
import mujoco
from scipy.spatial.transform import Rotation as R
from dual_arm_controller import DualArmController
from test_single_arm import visualize_trajectory, pose_to_matrix, matrix_to_pose

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
# Define Camera to Base transforms
# TODO: Fill in the actual calibration matrices

# Camera relative to Left Arm Base
Mat_base_L_T_camera = np.array([
    [1.0, 0.0, 0.0, 0.0,],
    [0.0, 1.0, 0.0, -0.25,],
    [0.0, 0.0, 1.0, 0.2,],
    [0.0, 0.0, 0.0, 1.0,],
])

# Camera relative to Right Arm Base
Mat_base_R_T_camera = np.array([
    [1.0, 0.0, 0.0, 0.0,],
    [0.0, 1.0, 0.0, 0.25,],
    [0.0, 0.0, 1.0, 0.2,],
    [0.0, 0.0, 0.0, 1.0,],
])

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
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
    ee_pts_transformed = (R_transform @ ee_pts.T).T

    # 2. Transform Orientations
    ee_oris_transformed = np.matmul(R_transform, ee_oris)

    # Additional 180 degree rotation around Z axis
    R_z180 = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ])
    ee_oris_transformed = np.matmul(ee_oris_transformed, R_z180)

    # 3. Convert Rotation Matrices to Euler Angles (xyz)
    N = len(ee_pts)
    target_seq = np.zeros((N, 6))
    
    target_seq[:, :3] = ee_pts_transformed
    
    # Batch convert rotations
    r = R.from_matrix(ee_oris_transformed)
    euler_angles = r.as_euler('xyz', degrees=False)
    target_seq[:, 3:] = euler_angles

    return target_seq

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "R5", "R5a", "meshes", "dual_arm_scene.xml")
    
    # Data paths
    data_path_L = os.path.join(current_dir, "free_hand_N_to_F_smoothed_actions_left_in_camera_optical_frame.npz")
    data_path_R = os.path.join(current_dir, "free_hand_N_to_F_smoothed_actions_right_in_camera_optical_frame.npz")

    dual_robot = DualArmController(xml_path, max_steps=100)
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
    # visualize_trajectory(seqs_L_in_camera_link, title="Left Arm Trajectory in Camera Link Frame")
    # visualize_trajectory(seqs_R_in_camera_link, title="Right Arm Trajectory in Camera Link Frame")

    Mat_world_T_seqs_L = Mat_world_T_base_L @ Mat_base_L_T_camera @ Mat_camera_T_seqs_L
    Mat_world_T_seqs_R = Mat_world_T_base_R @ Mat_base_R_T_camera @ Mat_camera_T_seqs_R
    seqs_L_in_world = np.array([matrix_to_pose(mat) for mat in Mat_world_T_seqs_L])
    seqs_R_in_world = np.array([matrix_to_pose(mat) for mat in Mat_world_T_seqs_R])
    print(f"seqs_L_in_world shape: {seqs_L_in_world.shape}")
    print(f"seqs_R_in_world shape: {seqs_R_in_world.shape}")
    visualize_trajectory(seqs_L_in_world, title="Left Arm Trajectory in World Frame")
    visualize_trajectory(seqs_R_in_world, title="Right Arm Trajectory in World Frame")

    if seqs_L_in_world is not None and seqs_R_in_world is not None:
        try:
            dual_robot.move_trajectory(seqs_L_in_world, seqs_R_in_world, 50, kinematic_only=True)
        except Exception as e:
            print(f"Error during dual arm trajectory execution: {e}")
    else:
        print("Failed to load trajectory data for one or both arms.")