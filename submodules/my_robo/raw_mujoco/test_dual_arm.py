import numpy as np
import os
import mujoco
from scipy.spatial.transform import Rotation as R
from dual_arm_controller import DualArmController

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
# Define Camera to Base transforms
# TODO: Fill in the actual calibration matrices

# Camera relative to Left Arm Base
Mat_base_L_T_camera = np.array([
    [1.0, 0.0, 0.0, 0.0,],
    [0.0, 1.0, 0.0, 0.0,],
    [0.0, 0.0, 1.0, 0.2,],
    [0.0, 0.0, 0.0, 1.0,],
])

# Camera relative to Right Arm Base
Mat_base_R_T_camera = np.array([
    [1.0, 0.0, 0.0, 0.0,],
    [0.0, 1.0, 0.0, 0.0,],
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

def pose_to_matrix(pose):
    """ [x, y, z, rx, ry, rz] -> 4x4 Homogeneous Matrix """
    t = pose[:3]
    euler = pose[3:]
    r = R.from_euler('xyz', euler, degrees=False)
    mat = np.eye(4)
    mat[:3, :3] = r.as_matrix()
    mat[:3, 3] = t
    return mat

def matrix_to_pose(mat):
    """ 4x4 Homogeneous Matrix -> [x, y, z, rx, ry, rz] """
    t = mat[:3, 3]
    rot_mat = mat[:3, :3]
    r = R.from_matrix(rot_mat)
    euler = r.as_euler('xyz', degrees=False)
    return np.concatenate([t, euler])

def get_body_pose_world(model, data, body_name):
    """ Get the pose of a body in world frame from MuJoCo data """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise ValueError(f"Body {body_name} not found")
    
    pos = data.xpos[body_id]
    mat = data.xmat[body_id].reshape(3, 3)
    r = R.from_matrix(mat)
    euler = r.as_euler('xyz', degrees=False)
    return np.concatenate([pos, euler])

def transform_seq_to_world(seq_camera, T_world_camera):
    """ Transform a sequence of poses from Camera frame to World frame """
    Mat_camera_T_seq = np.array([pose_to_matrix(p) for p in seq_camera])
    Mat_world_T_seq = T_world_camera @ Mat_camera_T_seq
    return np.array([matrix_to_pose(m) for m in Mat_world_T_seq])

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "R5", "R5a", "meshes", "dual_arm_scene.xml")
    
    # Data paths
    data_path_L = os.path.join(current_dir, "free_hand_N_to_F_smoothed_actions_left_in_camera_optical_frame.npz")
    data_path_R = os.path.join(current_dir, "free_hand_N_to_F_smoothed_actions_right_in_camera_optical_frame.npz")

    print("Initializing DualArmController...")
    try:
        robot = DualArmController(xml_path, arm_names=['L', 'R'])
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit()

    # 1. Get Reference Base Poses for Left and Right Arms
    try:
        base_pose_world_L = get_body_pose_world(robot.model, robot.data, "base_link_L")
        base_pose_world_R = get_body_pose_world(robot.model, robot.data, "base_link_R")
        print(f"Base L Pose: {np.round(base_pose_world_L, 3)}")
        print(f"Base R Pose: {np.round(base_pose_world_R, 3)}")
    except ValueError as e:
        print(f"Error getting base poses: {e}")
        exit()

    Mat_world_T_base_L = pose_to_matrix(base_pose_world_L)
    Mat_world_T_base_R = pose_to_matrix(base_pose_world_R)
    
    # 2. Calculate World -> Camera Transforms
    # Camera pose derived from Left Base
    Mat_world_T_camera_L = Mat_world_T_base_L @ Mat_base_L_T_camera
    
    # Camera pose derived from Right Base
    Mat_world_T_camera_R = Mat_world_T_base_R @ Mat_base_R_T_camera
    
    print("Mat_world_T_camera (derived from L):\n", np.round(Mat_world_T_camera_L, 2))
    print("Mat_world_T_camera (derived from R):\n", np.round(Mat_world_T_camera_R, 2))

    # 3. Load and Transform Data
    print(f"Loading Left trajectory: {data_path_L}")
    seq_camera_L = load_and_transform_data(data_path_L)
    
    print(f"Loading Right trajectory: {data_path_R}")
    seq_camera_R = load_and_transform_data(data_path_R)
    
    if seq_camera_L is None or seq_camera_R is None:
        print("Failed to load trajectory data.")
        exit()

    print("Transforming trajectories to World frame...")
    # Transform Left trajectory using Left-derived camera pose
    seq_world_L = transform_seq_to_world(seq_camera_L, Mat_world_T_camera_L)
    # Transform Right trajectory using Right-derived camera pose
    seq_world_R = transform_seq_to_world(seq_camera_R, Mat_world_T_camera_R)

    target_world_seqs = {
        'L': seq_world_L,
        'R': seq_world_R
    }

    # 4. Execute
    print("Starting execution (Kinematic Only)...")
    try:
        robot.move_trajectory(target_world_seqs, kinematic_only=True)
    except Exception as e:
        print(f"Execution Error: {e}")
