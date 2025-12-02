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

        print(f"Robot 0 Base: {self.base_pos[0]}")
        print(f"Robot 1 Base: {self.base_pos[1]}")

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
            left_pos_seq: List or array of (N, 3) target positions for left arm (World Frame)
            left_ori_seq: List or array of (N, 3, 3) target rotation matrices for left arm (World Frame)
            right_pos_seq: List or array of (N, 3) target positions for right arm (World Frame)
            right_ori_seq: List or array of (N, 3, 3) target rotation matrices for right arm (World Frame)
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
            
            # Get Targets
            target_pos_0 = left_pos_seq[idx]
            target_ori_0 = left_ori_seq[idx]
            target_pos_1 = right_pos_seq[idx]
            target_ori_1 = right_ori_seq[idx]
            
            # Get Current State
            current_pos_0 = self.obs["robot0_eef_pos"]
            current_pos_1 = self.obs["robot1_eef_pos"]
            current_ori_0 = T.quat2mat(self.obs["robot0_eef_quat"])
            current_ori_1 = T.quat2mat(self.obs["robot1_eef_quat"])
            
            # --- PID Calculation for Left Robot ---
            # Position
            err_pos_0 = target_pos_0 - current_pos_0
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
            # Position
            err_pos_1 = target_pos_1 - current_pos_1
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

            # Update visual spheres to match targets
            self._update_sphere("target_sphere_0", target_pos_0)
            self._update_sphere("target_sphere_1", target_pos_1)

            # Step Environment
            self.obs, reward, done, _ = self.env.step(action)
            
            # Calculate and print distance error
            curr_pos_0 = self.obs["robot0_eef_pos"]
            curr_pos_1 = self.obs["robot1_eef_pos"]
            dist_err_0 = np.linalg.norm(curr_pos_0 - target_pos_0)
            dist_err_1 = np.linalg.norm(curr_pos_1 - target_pos_1)
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
    # Initialize Controller
    controller = DualArmInpaintor()

    # --- Define Trajectory Sequences ---
    steps = 500
    
    # Left Robot Target (Local -> World)
    target_pos_0_local = np.array([0.2, -0.05, 0.2])
    target_pos_0 = controller.base_pos[0] + controller.base_ori[0] @ target_pos_0_local
    target_ori_0 = T.euler2mat([0.2, 0.3, 0.1]) # Horizontal
    
    # Right Robot Target (Local -> World)
    target_pos_1_local = np.array([0.0, 0.1, 0.2])
    target_pos_1 = controller.base_pos[1] + controller.base_ori[1] @ target_pos_1_local
    target_ori_1 = T.euler2mat([0.15, 0.15, 0]) # Horizontal

    # Create sequences (repeating the same target for demonstration)
    # In a real scenario, these would be varying arrays
    left_pos_seq = [target_pos_0] * steps
    left_ori_seq = [target_ori_0] * steps
    right_pos_seq = [target_pos_1] * steps
    right_ori_seq = [target_ori_1] * steps
    
    # Execute
    try:
        controller.execute_trajectory(left_pos_seq, left_ori_seq, right_pos_seq, right_ori_seq)
    finally:
        controller.close()