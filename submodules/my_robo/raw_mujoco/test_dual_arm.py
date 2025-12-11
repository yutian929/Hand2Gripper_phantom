import cv2
import numpy as np
import os
import mujoco
from scipy.spatial.transform import Rotation as R
from dual_arm_controller import DualArmController
from test_single_arm import load_and_transform_data, visualize_trajectory, pose_to_matrix, matrix_to_pose

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
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
    # visualize_trajectory(seqs_L_in_world, title="Left Arm Trajectory in World Frame")
    # visualize_trajectory(seqs_R_in_world, title="Right Arm Trajectory in World Frame")

    # assume camera base fixed
    Mat_world_T_camera_L = Mat_world_T_base_L @ Mat_base_L_T_camera
    Mat_world_T_camera_R = Mat_world_T_base_R @ Mat_base_R_T_camera
    # These 2 Mat should be very close
    diff = np.linalg.norm(Mat_world_T_camera_L - Mat_world_T_camera_R)
    assert diff < 1e-6, f"Camera world transforms from two arms differ significantly! Difference: {diff}"
    camera_poses_world = np.array([matrix_to_pose(Mat_world_T_camera_L) for _ in range(len(seqs_L_in_world))])

    if seqs_L_in_world is not None and seqs_R_in_world is not None:
        try:
            print("########## Executing dual arm move_trajectory ##########")
            # dual_robot.move_trajectory(seqs_L_in_world, seqs_R_in_world, 50, kinematic_only=True)
            print("########## Executing dual arm move_trajectory_with_camera ##########")
            frames, masks = dual_robot.move_trajectory_with_camera(
                seqs_L_in_world, seqs_R_in_world, camera_poses_world, 50, kinematic_only=True)
            
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
            print(f"Error during dual arm trajectory execution: {e}")
    else:
        print("Failed to load trajectory data for one or both arms.")