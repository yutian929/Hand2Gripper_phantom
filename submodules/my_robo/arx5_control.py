import time
import numpy as np
import cv2
import tqdm
import os
import json
from robosuite.robots import MobileRobot
from robosuite.utils.input_utils import *
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
import robosuite.utils.transform_utils as T
from scipy.spatial.transform import Rotation

MAX_FR = 25  # max frame rate for running simluation

USE_LEFT = False
USE_RIGHT = True

def _load_hand2gripper_params(key: str):
    hand2gripper_config_path = os.environ.get("HAND2GRIPPER_CONFIG_PATH", "hand2gripper_config.json")
    if os.path.exists(hand2gripper_config_path): 
        with open(hand2gripper_config_path, 'r') as f:
            config = json.load(f)
            val = config.get(key, None)
            if val is not None:
                return val
    # Default values if config file or key does not exist
    default_params = {
        "gripper-tip-offset": 0.2,
    }
    return default_params.get(key, None)

def transform_points(points, trans_mat):
    """
    Transform points using a 4x4 transformation matrix.
    points: (N, 3)
    trans_mat: (4, 4)
    """
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack([points, ones])
    points_transformed_h = (trans_mat @ points_h.T).T
    return points_transformed_h[:, :3]

def transform_coordinates(points):
    """
    Transform coordinates:
    Original (HaMeR): Z-forward, X-right, Y-down
    Target: X-forward, Y-left, Z-up
    Transformation: x' = z, y' = -x, z' = -y
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    return np.stack([z, -x, -y], axis=1)

class DualArmInpaintor:
    def __init__(self, env_name="TwoArmPhantom", robot_name="Arx5", render=True):
        # Create dict to hold options that will be passed to env creation call
        options = {}
        options["env_name"] = env_name
        options["env_configuration"] = "phantom_parallel"

        if "TwoArm" in options["env_name"]:
            options["robots"] = [robot_name, robot_name]
        else:
            options["robots"] = [robot_name]

        # Load OSC_POSE controller configuration for Cartesian control
        controller_name = "OSC_POSE"
        arm_controller_config = suite.load_part_controller_config(default_controller=controller_name)
        
        options["controller_configs"] = refactor_composite_controller_config(
            arm_controller_config, robot_name, ["left", "right"]
        )

        # initialize the task
        self.env = suite.make(
            **options,
            has_renderer=render,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_camera_obs=True,
            camera_names="zed",
            camera_segmentations="instance",
            camera_heights=480,
            camera_widths=640,
            control_freq=20,
        )
        self.obs = self.env.reset()
        
        if render:
            self.env.viewer.set_camera(camera_id=0)
            self.env.render()
            
        for robot in self.env.robots:
            if isinstance(robot, MobileRobot):
                robot.enable_parts(legs=False, base=False)

        # Get robot base poses for coordinate transformation
        self.base_pos = [self.env.robots[0].base_pos, self.env.robots[1].base_pos]
        self.base_ori = [self.env.robots[0].base_ori, self.env.robots[1].base_ori]
        
        self.gripper_tip_offset = _load_hand2gripper_params("gripper-tip-offset")
        if self.gripper_tip_offset is None:
            self.gripper_tip_offset = 0.0

        print(f"Robot 0 Base: {self.base_pos[0]}")
        print(f"Robot 1 Base: {self.base_pos[1]}")
        print(f"Gripper Tip Offset: {self.gripper_tip_offset} (along X-axis)")

        # Calculate action dimensions
        total_action_dim = self.env.action_dim
        num_robots = len(self.env.robots)
        self.action_per_robot = total_action_dim // num_robots
        self.controller_dim = 6 # OSC_POSE (x, y, z, ax, ay, az)
        self.gripper_dim = self.action_per_robot - self.controller_dim
        print(f"Action dim per robot: {self.action_per_robot}, Gripper dim: {self.gripper_dim}")

    def _update_sphere(self, name, pos):
        """Helper to update visual sphere position in MuJoCo"""
        try:
            # Find joint address for the free joint of the sphere
            joint_name = f"{name}_joint0"
            joint_id = self.env.sim.model.joint_name2id(joint_name)
            qpos_addr = self.env.sim.model.jnt_qposadr[joint_id]
            # Update position (x, y, z)
            self.env.sim.data.qpos[qpos_addr:qpos_addr+3] = pos
            # Ensure orientation is valid (identity quaternion)
            self.env.sim.data.qpos[qpos_addr+3:qpos_addr+7] = [1, 0, 0, 0]
        except ValueError:
            pass # Joint not found

    def execute_trajectory(self, left_pos_seq, left_ori_seq, right_pos_seq, right_ori_seq, 
                           left_gripper_seq=None, right_gripper_seq=None, steps_per_waypoint=30):
        """
        Execute a sequence of targets for both arms.
        
        Args:
            left_pos_seq: List or array of (N, 3) target positions for left arm (Gripper Tip, World Frame)
            left_ori_seq: List or array of (N, 3, 3) target rotation matrices for left arm (Gripper Tip, World Frame)
            right_pos_seq: List or array of (N, 3) target positions for right arm (Gripper Tip, World Frame)
            right_ori_seq: List or array of (N, 3, 3) target rotation matrices for right arm (Gripper Tip, World Frame)
            left_gripper_seq: Optional (N,) array of gripper values (1=open, -1=closed)
            right_gripper_seq: Optional (N,) array of gripper values
            steps_per_waypoint: Number of simulation steps to hold each waypoint
        """
        traj_len = len(left_pos_seq)
        
        # Default grippers if not provided
        if left_gripper_seq is None:
            left_gripper_seq = np.ones(traj_len) # Open
        if right_gripper_seq is None:
            right_gripper_seq = np.zeros(traj_len) # Closed (as per previous example default)

        total_steps = traj_len * steps_per_waypoint
        
        # PID Gains
        kp_pos, ki_pos, kd_pos = 16.0, 0.1, 1.0
        kp_ori, ki_ori, kd_ori = 8.0, 0.04, 0.2

        # PID State Initialization
        # Left Robot
        integral_pos_0 = np.zeros(3)
        prev_err_pos_0 = np.zeros(3)
        integral_ori_0 = np.zeros(3)
        prev_err_ori_0 = np.zeros(3)

        # Right Robot
        integral_pos_1 = np.zeros(3)
        prev_err_pos_1 = np.zeros(3)
        integral_ori_1 = np.zeros(3)
        prev_err_ori_1 = np.zeros(3)

        for i in tqdm.tqdm(range(total_steps)):
            start = time.time()
            
            # Determine current waypoint index
            idx = i // steps_per_waypoint
            if idx >= traj_len:
                idx = traj_len - 1
            
            # Get Targets (Gripper Tip)
            target_pos_0_tip = left_pos_seq[idx]
            target_ori_0 = left_ori_seq[idx]
            target_pos_1_tip = right_pos_seq[idx]
            target_ori_1 = right_ori_seq[idx]

            # Calculate Flange Targets for Controller
            # Offset along X-axis of the end-effector frame
            offset_vec = np.array([self.gripper_tip_offset, 0, 0])
            
            target_pos_0_flange = target_pos_0_tip - target_ori_0 @ offset_vec
            target_pos_1_flange = target_pos_1_tip - target_ori_1 @ offset_vec
            
            # Get Current State (Flange)
            current_pos_0 = self.obs["robot0_eef_pos"]
            current_pos_1 = self.obs["robot1_eef_pos"]
            current_ori_0 = T.quat2mat(self.obs["robot0_eef_quat"])
            current_ori_1 = T.quat2mat(self.obs["robot1_eef_quat"])
            
            # --- PID Calculation for Left Robot ---
            # Position (Error relative to Flange Target)
            err_pos_0 = target_pos_0_flange - current_pos_0
            integral_pos_0 += err_pos_0
            derivative_pos_0 = err_pos_0 - prev_err_pos_0
            prev_err_pos_0 = err_pos_0
            action_pos_0 = (kp_pos * err_pos_0) + (ki_pos * integral_pos_0) + (kd_pos * derivative_pos_0)

            # Orientation
            err_ori_mat_0 = np.dot(target_ori_0, current_ori_0.T)
            err_ori_0 = T.quat2axisangle(T.mat2quat(err_ori_mat_0))
            integral_ori_0 += err_ori_0
            derivative_ori_0 = err_ori_0 - prev_err_ori_0
            prev_err_ori_0 = err_ori_0
            action_ori_0 = (kp_ori * err_ori_0) + (ki_ori * integral_ori_0) + (kd_ori * derivative_ori_0)

            # --- PID Calculation for Right Robot ---
            # Position (Error relative to Flange Target)
            err_pos_1 = target_pos_1_flange - current_pos_1
            integral_pos_1 += err_pos_1
            derivative_pos_1 = err_pos_1 - prev_err_pos_1
            prev_err_pos_1 = err_pos_1
            action_pos_1 = (kp_pos * err_pos_1) + (ki_pos * integral_pos_1) + (kd_pos * derivative_pos_1)

            # Orientation
            err_ori_mat_1 = np.dot(target_ori_1, current_ori_1.T)
            err_ori_1 = T.quat2axisangle(T.mat2quat(err_ori_mat_1))
            integral_ori_1 += err_ori_1
            derivative_ori_1 = err_ori_1 - prev_err_ori_1
            prev_err_ori_1 = err_ori_1
            action_ori_1 = (kp_ori * err_ori_1) + (ki_ori * integral_ori_1) + (kd_ori * derivative_ori_1)

            # Gripper action
            g_action_0 = np.array([left_gripper_seq[idx]] * self.gripper_dim)
            g_action_1 = np.array([right_gripper_seq[idx]] * self.gripper_dim)

            # Construct Action: [dx, dy, dz, dax, day, daz, gripper]
            action_0 = np.concatenate([
                action_pos_0,
                action_ori_0,
                g_action_0
            ])
            action_1 = np.concatenate([
                action_pos_1, 
                action_ori_1, 
                g_action_1
            ])

            if not USE_LEFT:
                action_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [0.0]*self.gripper_dim)
            if not USE_RIGHT:
                action_1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [0.0]*self.gripper_dim)
            action_1[3] = 0.0  # Disable rotation around x-axis for right arm
            action_1[4] = 0.0  # Disable rotation around y-axis for right arm
            action_1[5] = 0.0  # Disable rotation around z-axis for right arm
            action = np.concatenate([action_0, action_1])

            # Update visual spheres to match targets (Tip)
            self._update_sphere("target_sphere_0", target_pos_0_tip)
            self._update_sphere("target_sphere_1", target_pos_1_tip)

            # Step Environment
            self.obs, reward, done, _ = self.env.step(action)
            
            # Calculate and print distance error (Tip Error)
            # Current Tip Position = Current Flange Pos + Current Ori @ Offset
            curr_pos_0_tip = self.obs["robot0_eef_pos"] + T.quat2mat(self.obs["robot0_eef_quat"]) @ offset_vec
            curr_pos_1_tip = self.obs["robot1_eef_pos"] + T.quat2mat(self.obs["robot1_eef_quat"]) @ offset_vec
            
            dist_err_0 = np.linalg.norm(curr_pos_0_tip - target_pos_0_tip)
            dist_err_1 = np.linalg.norm(curr_pos_1_tip - target_pos_1_tip)
            print(f"Step {i} | Dist Err - Left: {dist_err_0:.4f}, Right: {dist_err_1:.4f}")
            
            # Visualization
            self._visualize_camera()
            self.env.render()

            # Limit frame rate
            elapsed = time.time() - start
            diff = 1 / MAX_FR - elapsed
            if diff > 0:
                time.sleep(diff)
                
    def _visualize_camera(self):
        # Get RGB and Instance Segmentation images
        rgb_img = self.obs.get("zed_image")  # Shape: (H, W, 3)
        if rgb_img is not None:
            rgb_img = cv2.flip(rgb_img, 1)  # Flip image horizontally
            cv2.imshow("RGB Image", cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

        seg_img = self.obs.get("zed_segmentation_instance")  # Shape: (H, W, 1)
        if seg_img is not None:
            seg_img = cv2.flip(seg_img, 1)
            seg_max = seg_img.max()
            if seg_max > 0:
                seg_vis = (seg_img / seg_max).astype(np.float32)
            else:
                seg_vis = seg_img.astype(np.float32)
            cv2.imshow("Instance Segmentation", seg_vis)
        
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()
        self.env.close()


if __name__ == "__main__":
    # Load smoothed actions
    try:
        data_left = np.load("free_hand_smoothed_actions_left_in_camera_optical_frame.npz")
        data_right = np.load("free_hand_smoothed_actions_right_in_camera_optical_frame.npz")
    except FileNotFoundError:
        print("Error: smoothed_actions.npz not found.")
        exit()

    # Extract data (Optical Frame)
    left_pts = data_left["ee_pts"]
    left_oris = data_left["ee_oris"]
    left_widths = data_left["ee_widths"]
    
    right_pts = data_right["ee_pts"]
    right_oris = data_right["ee_oris"]
    right_widths = data_right["ee_widths"]

    # Transform Optical -> Camera Link
    # Transformation: x' = z, y' = -x, z' = -y
    R_optical_to_link = np.array([
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0]
    ])

    # Transform positions
    left_pts_link = (R_optical_to_link @ left_pts.T).T
    right_pts_link = (R_optical_to_link @ right_pts.T).T

    # Transform orientations
    left_oris_link = np.matmul(R_optical_to_link, left_oris)
    right_oris_link = np.matmul(R_optical_to_link, right_oris)

    # Camera World Pose (Camera Link Frame)
    cam_pos = np.array([
        _load_hand2gripper_params("camera-zed-pos_x"),
        _load_hand2gripper_params("camera-zed-pos_y"),
        _load_hand2gripper_params("camera-zed-pos_z")
    ])
    cam_quat = np.array([
        _load_hand2gripper_params("camera-zed-quat_x"),
        _load_hand2gripper_params("camera-zed-quat_y"),
        _load_hand2gripper_params("camera-zed-quat_z"),
        _load_hand2gripper_params("camera-zed-quat_w")
    ])
    T_world_cam = np.eye(4)
    T_world_cam[:3, 3] = cam_pos
    T_world_cam[:3, :3] = T.quat2mat(cam_quat)

    # Transform points to World Frame directly
    left_pos_world = transform_points(left_pts_link, T_world_cam)
    right_pos_world = transform_points(right_pts_link, T_world_cam)
    
    # Transform orientations to World Frame directly
    left_ori_world = np.matmul(T_world_cam[:3, :3], left_oris_link)
    right_ori_world = np.matmul(T_world_cam[:3, :3], right_oris_link)

    # Initialize Controller
    controller = DualArmInpaintor()

    # Map gripper widths to actions (1=open, -1=closed)
    # Assuming width > 0.04 is open
    left_gripper_seq = np.where(left_widths > 0.04, 1.0, -1.0)
    right_gripper_seq = np.where(right_widths > 0.04, 1.0, -1.0)
    
    # Execute
    try:
        controller.execute_trajectory(left_pos_world, left_ori_world, right_pos_world, right_ori_world, left_gripper_seq, right_gripper_seq)
    finally:
        controller.close()