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

MAX_FR = 25  # max frame rate for running simluation

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
            self.env.viewer.set_camera(camera_id=1)
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
                           left_gripper_seq=None, right_gripper_seq=None, steps_per_waypoint=50):
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
                np.clip(action_pos_0, -1, 1), 
                np.clip(action_ori_0, -0.5, 0.5), 
                g_action_0
            ])
            action_1 = np.concatenate([
                np.clip(action_pos_1, -1, 1), 
                np.clip(action_ori_1, -0.5, 0.5), 
                g_action_1
            ])
            
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
    data_left_hand = np.load("hand_data_left.npz")
    data_right_hand = np.load("hand_data_right.npz")
    
    # Calculate centroids (N, 3)
    left_centroids = np.mean(data_left_hand["kpts_3d"], axis=1)
    right_centroids = np.mean(data_right_hand["kpts_3d"], axis=1)
    
    # Apply coordinate transformation
    left_centroids = transform_coordinates(left_centroids)
    right_centroids = transform_coordinates(right_centroids)

    # Ensure same length
    min_len = min(len(left_centroids), len(right_centroids))
    left_centroids = left_centroids[:min_len]
    right_centroids = right_centroids[:min_len]
    left_centroids = left_centroids[::10, :]
    right_centroids = right_centroids[::10, :]
    min_len = min(len(left_centroids), len(right_centroids))

    # Camera World Pose
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

    # Left Robot Base World Pose
    left_base_pos = np.array([
        _load_hand2gripper_params("robots-left-base_x"),
        _load_hand2gripper_params("robots-left-base_y"),
        _load_hand2gripper_params("robots-left-base_z")
    ])
    T_world_base_left = np.eye(4)
    T_world_base_left[:3, 3] = left_base_pos

    # Right Robot Base World Pose
    right_base_pos = np.array([
        _load_hand2gripper_params("robots-right-base_x"),
        _load_hand2gripper_params("robots-right-base_y"),
        _load_hand2gripper_params("robots-right-base_z")
    ])
    T_world_base_right = np.eye(4)
    T_world_base_right[:3, 3] = right_base_pos

    # Calculate T_cam2base = inv(T_world_base) @ T_world_cam
    T_cam2base_left = np.linalg.inv(T_world_base_left) @ T_world_cam
    T_cam2base_right = np.linalg.inv(T_world_base_right) @ T_world_cam

    # Transform centroids to robot base frame (Local)
    left_pos_local = transform_points(left_centroids, T_cam2base_left)
    right_pos_local = transform_points(right_centroids, T_cam2base_right)
    # breakpoint()
    # Initialize Controller
    controller = DualArmInpaintor()

    # --- Define Trajectory Sequences ---
    steps = min_len
    
    # Fixed orientation for now (Horizontal)
    target_ori_0 = T.euler2mat([1.5, 0.0, 0.0]) 
    target_ori_1 = T.euler2mat([0.0, 1.5, 0.0]) 

    # Create orientation sequences
    left_ori_seq = [target_ori_0] * steps
    right_ori_seq = [target_ori_1] * steps
    
    # Convert Local to World Frame for execution
    left_pos_seq = []
    right_pos_seq = []
    for i in range(steps):
        # Apply Base -> World
        p0 = controller.base_pos[0] + controller.base_ori[0] @ left_pos_local[i]
        p1 = controller.base_pos[1] + controller.base_ori[1] @ right_pos_local[i]
        left_pos_seq.append(np.array([0.5, 0, 0.8]))
        right_pos_seq.append(p1)
    
    # Execute
    try:
        controller.execute_trajectory(left_pos_seq, left_ori_seq, right_pos_seq, right_ori_seq)
    finally:
        controller.close()