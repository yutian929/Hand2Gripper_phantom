import json
import numpy as np
from scipy.spatial.transform import Rotation as R

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_T_from_left_json(data):
    return np.array(data["Mat_base_T_camera_link"])

def get_T_from_right_json(data):
    return np.array(data["Mat_base_T_camera_link"])

def main():
    # Paths to the JSON files provided
    left_json_path = "/home/yutian/Hand2Gripper_phantom/phantom/camera/eye_to_hand_result_left_latest.json"
    right_json_path = "/home/yutian/Hand2Gripper_phantom/phantom/camera/eye_to_hand_result_right_latest.json"

    try:
        left_data = load_json(left_json_path)
        right_data = load_json(right_json_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    T_LB_C = get_T_from_left_json(left_data)
    T_RB_C = get_T_from_right_json(right_data)

    # We want T_LB_RB (LeftBase to RightBase)
    # T_LB_C = T_LB_RB @ T_RB_C  => T_LB_RB = T_LB_C @ inv(T_RB_C)
    T_RB_C_inv = np.linalg.inv(T_RB_C)
    T_LB_RB = T_LB_C @ T_RB_C_inv

    # World Frame: Left Base is at (0, 0, 1.0) with Identity rotation
    T_W_LB = np.eye(4)
    T_W_LB[:3, 3] = [0.0, 0.2, 1.0]

    # T_W_RB = T_W_LB @ T_LB_RB
    T_W_RB = T_W_LB @ T_LB_RB

    pos = T_W_RB[:3, 3]
    
    # Scipy returns (x, y, z, w)
    quat = R.from_matrix(T_W_RB[:3, :3]).as_quat() 
    
    # MuJoCo uses (w, x, y, z)
    # We manually reorder to ensure consistency with MuJoCo
    mujoco_quat = [quat[3], quat[0], quat[1], quat[2]]

    print("Calculated Right Arm Base Pose in World Frame:")
    print(f"Position: {pos}")
    print(f"Quaternion (w,x,y,z): {mujoco_quat}")
    
    print("\nCopy the following attributes to the 'base_link_R' body in your XML:")
    print(f'pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" quat="{mujoco_quat[0]:.4f} {mujoco_quat[1]:.4f} {mujoco_quat[2]:.4f} {mujoco_quat[3]:.4f}"')

if __name__ == "__main__":
    main()
