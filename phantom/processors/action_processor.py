"""
Action Processor Module

This module processes hand motion capture data and converts it into robot-executable actions.
It handles both single-arm and bimanual robotic setups, converting detected hand keypoints
into end-effector positions, orientations, and gripper widths that can be used for robot control.

Key Features:
- Converts hand keypoints from camera frame to robot frame
- Supports both unconstrained and physically constrained hand models
- Handles missing hand detections with interpolation
- Processes bimanual data with union-based frame selection
- Generates neutral poses when no hand data is available

The processor follows this pipeline:
1. Load hand sequence data (keypoints, detection flags)
2. Convert keypoints to robot coordinate frame
3. Apply hand model constraints (optional)
4. Extract end-effector poses and gripper states
5. Refine actions to handle missing detections
6. Save processed actions for robot execution
"""

import os 
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import logging
from scipy.spatial.transform import Rotation

from phantom.processors.base_processor import BaseProcessor
from phantom.processors.phantom_data import HandSequence
from phantom.processors.paths import Paths
from phantom.hand import HandModel, PhysicallyConstrainedHandModel, get_list_finger_pts_from_skeleton

logger = logging.getLogger(__name__)

# >>> Hand2Gripper >>>
import mediapy as media
# from hand2gripper_vslam._orb_slam3 import ORB_SLAM3_RGBD_VO
from hand2gripper_vslam.rtab_map_client import RTABMapClient
# <<< Hand2Gripper <<<

@dataclass
class EEActions:
    """
    Container for bimanual end-effector action data.
    
    This dataclass holds the processed robot actions for a sequence of timesteps,
    including 3D positions, 3D orientations, and gripper opening widths.
    
    Attributes:
        ee_pts (np.ndarray): End-effector positions, shape (N, 3) in robot frame coordinates
        ee_oris (np.ndarray): End-effector orientations as rotation matrices, shape (N, 3, 3)
        ee_widths (np.ndarray): Gripper opening widths in meters, shape (N,)
    """
    ee_pts: np.ndarray      # End-effector positions (N, 3)
    ee_oris: np.ndarray     # End-effector orientations (N, 3, 3) as rotation matrices
    ee_widths: np.ndarray   # Gripper widths (N,)

class ActionProcessor(BaseProcessor): 
    """
    Processor for converting hand motion capture data into robot-executable actions.
    
    This class handles the complete pipeline from raw hand keypoints to refined robot actions.
    It supports both single-arm and bimanual robotic setups, with intelligent handling of
    missing hand detections and physically realistic constraints.
    
    The processor can operate in different modes:
    - Single arm: Processes only left or right hand data
    - Bimanual: Processes both hands with union-based frame selection
    
    Key processing steps:
    1. Load hand sequences with 3D keypoints and detection flags
    2. Transform keypoints from camera frame to robot frame
    3. Fit hand model (optionally with physical constraints)
    4. Extract end-effector poses and gripper states
    5. Refine actions using last-valid-value interpolation
    6. Generate neutral poses for undetected periods
    
    Attributes:
        dt (float): Time delta between frames (1/15 seconds for 15Hz processing)
        bimanual_setup (str): Setup type ("single_arm", "shoulders", etc.)
        target_hand (str): Which hand to process in single-arm mode ("left"/"right")
        constrained_hand (bool): Whether to use physically constrained hand model
        T_cam2robot (np.ndarray): 4x4 transformation matrix from camera to robot frame
    """
    def __init__(self, args):
        # Set processing frequency to 15Hz 
        self.dt = 1/15 
        super().__init__(args)
        self.visualize_traj = False

    def process_one_demo(self, data_sub_folder: str) -> None:
        """
        Process a single demonstration recording into robot actions.
        
        This is the main entry point for processing one demo. It handles both
        single-arm and bimanual processing modes, loading the raw hand data,
        converting it to robot actions, and saving the results.
        
        Args:
            data_sub_folder (str): Path to the folder containing this demo's data
        """
        save_folder = self.get_save_folder(data_sub_folder)
        paths = self.get_paths(save_folder)

        # Load hand sequence data for both hands
        left_sequence, right_sequence = self._load_sequences(paths)

        # >>> Hand2Gripper >>>
        # Load RGB video frames
        imgs_rgb = media.read_video(getattr(paths, f"video_left"))
        self.H, self.W, _ = imgs_rgb[0].shape
        # Load bounding box data
        bbox_data = np.load(paths.bbox_data)
        # <<< Hand2Gripper <<<

        # Handle single-arm processing mode
        if self.bimanual_setup == "single_arm":
            self._process_single_arm(left_sequence, right_sequence, paths)
        else:
            # >>> Hand2Gripper >>>
            # Origin
            # self._process_bimanual(left_sequence, right_sequence, paths)
            left_actions, right_actions = self._process_bimanual_hand2gripper(left_sequence, right_sequence, paths, imgs_rgb, bbox_data)
            # <<< Hand2Gripper <<<
        
        # >>> Hand2Gripper >>>
        # using RTAB-Map VSLAM to get camera trajectory
        rtab_map_client = RTABMapClient(paths.data_path.resolve())
        process = rtab_map_client.launch(use_new_terminal=False, capture_output=True)
        target_text = ">>> VSLAM PLAYBACK COMPLETE <<<" 
        rtab_map_client.wait_for_text(process, target_text, timeout=len(imgs_rgb)*0.1)
        # self.visualize_trajectories(paths.hand2gripper_vslam_camera_link_traj)
        # <<< Hand2Gripper <<<

    # >>> Hand2Gripper >>>
    def _process_bimanual_hand2gripper(
        self, 
        left_sequence: HandSequence, 
        right_sequence: HandSequence, 
        paths,
        imgs_rgb: np.ndarray,
        bbox_data: np.ndarray
    ) -> None:
        # Process both hand sequences
        assert len(left_sequence.frame_indices) == len(right_sequence.frame_indices) == len(imgs_rgb), "Frame count mismatch among left hand, right hand, and RGB images."
        assert len(bbox_data['left_bboxes']) == len(bbox_data['right_bboxes']) == len(imgs_rgb), "Frame count mismatch among bounding boxes and RGB images."
        
        if self.my_robo:
            print("\nmy_robo is True: Using Hand2Gripper processing pipeline.")
            self.T_cam2robot = np.eye(4)  # Identity matrix for Hand2Gripper setup
        left_actions = self._process_hand_sequence_hand2gripper(left_sequence, self.T_cam2robot, imgs_rgb, bbox_data['left_bboxes'], "left")
        right_actions = self._process_hand_sequence_hand2gripper(right_sequence, self.T_cam2robot, imgs_rgb, bbox_data['right_bboxes'], "right")
        # Combine detection results using OR logic - frame is valid if either hand detected
        union_indices = np.where(left_sequence.hand_detected | right_sequence.hand_detected)[0]
        
        # Refine actions for both hands using the union indices
        left_actions_refined = self._refine_actions(left_sequence, left_actions, union_indices, "left")
        right_actions_refined = self._refine_actions(right_sequence, right_actions, union_indices, "right")
        
        # Save results for both hands
        self._save_results(paths, union_indices, left_actions_refined, right_actions_refined)

        return left_actions_refined, right_actions_refined
    
    def _process_hand_sequence_hand2gripper(
        self, 
        sequence: HandSequence, 
        T_cam2robot: np.ndarray,
        imgs_rgb: np.ndarray,
        bboxes: np.ndarray,
        hand_side: str
    ) -> EEActions:
        """
        """
        # Convert keypoints from camera frame to robot frame coordinates
        kpts_3d_cf = sequence.kpts_3d  # Camera frame keypoints
        kpts_3d_rf = ActionProcessor._convert_pts_to_robot_frame(
            kpts_3d_cf, 
            T_cam2robot
        )

        # Create and fit hand model to the keypoint sequence
        hand_model = self._get_hand_model_hand2gripper(kpts_3d_rf, sequence.hand_detected, imgs_rgb, bboxes, sequence.contact_logits, hand_side, kpts_3d_cf)
        
        # 加入对应的可视化代码
        # self.hand2gripper_show_traj(hand_model, 10, title=f"{hand_side.capitalize()} Hand EE Trajectory(v2)")

        return EEActions(
            ee_pts=np.array(hand_model.ee_pts),
            ee_oris=np.array(hand_model.ee_oris),
            ee_widths=np.array(hand_model.ee_widths),
        )
    
    def hand2gripper_show_traj(self, hand_model, interval: int = 10, title: str = None):
        """
        可视化 Hand2Gripper 的 EE 轨迹（position + orientation）

        - 轨迹：黑色折线连接所有点
        - 坐标系：每隔 interval 帧画一个 3D 坐标轴（R的三列分别当作 x/y/z 轴方向）
        - 起点：绿色点
        - 终点：红色叉

        Args:
            hand_model: 需要包含 ee_pts (N,3) 和 ee_oris (N,3,3)
            interval: 间隔多少帧画一个坐标系
            title: 图标题（可选）
        """
        import numpy as np
        import matplotlib.pyplot as plt

        pts = np.asarray(hand_model.ee_pts, dtype=float)
        Rs  = np.asarray(hand_model.ee_oris, dtype=float)

        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"ee_pts shape expected (N,3), got {pts.shape}")
        if Rs.ndim != 3 or Rs.shape[1:] != (3, 3):
            raise ValueError(f"ee_oris shape expected (N,3,3), got {Rs.shape}")
        if len(pts) == 0:
            print("[hand2gripper_show_traj] Empty trajectory.")
            return

        # 过滤非法点（NaN/Inf）
        valid = np.isfinite(pts).all(axis=1)
        pts_v = pts[valid]
        Rs_v  = Rs[valid]

        if len(pts_v) == 0:
            print("[hand2gripper_show_traj] All points are invalid (NaN/Inf).")
            return

        # 自动决定坐标轴箭头长度：用轨迹尺度的某个比例
        extent = np.ptp(pts_v, axis=0)  # max-min per axis
        max_range = float(np.max(extent))
        axis_len = (0.08 * max_range) if max_range > 1e-6 else 0.02

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # 1) 画轨迹折线
        ax.plot(pts_v[:, 0], pts_v[:, 1], pts_v[:, 2], linewidth=2)  # 不指定颜色，matplotlib默认即可

        # 2) 起点 / 终点
        ax.scatter(pts_v[0, 0],  pts_v[0, 1],  pts_v[0, 2],  s=60, marker="o", label="Start")
        ax.scatter(pts_v[-1, 0], pts_v[-1, 1], pts_v[-1, 2], s=80, marker="x", label="End")

        # 3) 每隔 interval 画一个坐标系
        interval = max(1, int(interval))
        idxs = np.arange(0, len(pts_v), interval)

        for i in idxs:
            p = pts_v[i]
            R = Rs_v[i]

            # 约定：R 的三列分别为 x/y/z 轴在世界坐标（机器人坐标）下的方向
            x_axis = R[:, 0]
            y_axis = R[:, 1]
            z_axis = R[:, 2]

            # 画三根轴（这里用常见RGB配色更直观，你也可以去掉颜色参数）
            ax.quiver(p[0], p[1], p[2], x_axis[0], x_axis[1], x_axis[2], length=axis_len, normalize=True, color="r")
            ax.quiver(p[0], p[1], p[2], y_axis[0], y_axis[1], y_axis[2], length=axis_len, normalize=True, color="g")
            ax.quiver(p[0], p[1], p[2], z_axis[0], z_axis[1], z_axis[2], length=axis_len, normalize=True, color="b")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title if title is not None else f"EE Trajectory (interval={interval})")
        ax.legend()

        # 4) 等比例显示（两种方式：新版本用 set_box_aspect；否则手动设lim）
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

        # 手动设定等比例范围（更稳）
        mins = pts_v.min(axis=0)
        maxs = pts_v.max(axis=0)
        mid = 0.5 * (mins + maxs)
        r = 0.5 * max(maxs - mins) if max_range > 1e-6 else 0.1
        ax.set_xlim(mid[0] - r, mid[0] + r)
        ax.set_ylim(mid[1] - r, mid[1] + r)
        ax.set_zlim(mid[2] - r, mid[2] + r)

        plt.tight_layout()
        plt.show()


    def _get_hand_model_hand2gripper(self, kpts_3d_rf: np.ndarray, hand_detected: np.ndarray, imgs_rgb: np.ndarray, bboxes: np.ndarray, contact_logits: np.array, hand_side: str, kpts_3d_cf:np.array) -> HandModel | PhysicallyConstrainedHandModel:
        """
        点都是optical frame下的坐标
        """
        # Choose hand model type based on configuration
        if self.constrained_hand:
            hand_model = PhysicallyConstrainedHandModel(self.robot)
        else:
            hand_model = HandModel(self.robot)
        
        # Add each frame to the model for trajectory fitting
        for t_idx in range(len(kpts_3d_rf)):
            hand_model.add_frame_hand2gripper(
                kpts_3d_rf[t_idx], 
                t_idx * self.dt,  # Convert frame index to time
                hand_detected[t_idx],
                np.array(imgs_rgb[t_idx]).astype(np.uint8),
                bboxes[t_idx],
                contact_logits[t_idx],
                hand_side,
                kpts_3d_cf[t_idx],
                version='v2' if self.my_robo else 'v1'
            )
        return hand_model
    # <<< Hand2Gripper <<<

    @staticmethod
    def visualize_trajectories(camera_link_traj_path: str):
        import json
        import matplotlib.pyplot as plt
        
        with open(camera_link_traj_path, 'r') as f:
            data = json.load(f)
            
        # Sort keys numerically to ensure correct order
        sorted_keys = sorted(data.keys(), key=lambda x: int(x))
        
        positions = []
        quats = []
        
        for key in sorted_keys:
            pose = data[key]['pose']
            positions.append([pose['tx'], pose['ty'], pose['tz']])
            quats.append([pose['qx'], pose['qy'], pose['qz'], pose['qw']])
            
        positions = np.array(positions)
        quats = np.array(quats)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectory path
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='Path', color='black', linewidth=2)
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='green', marker='o', s=50, label='Start')
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='red', marker='x', s=50, label='End')
        
        # Determine scale for coordinate frames based on trajectory extent
        if len(positions) > 0:
            max_range = np.ptp(positions, axis=0).max()
            # Increase scale factor to 0.3 (30% of trajectory size) for better visibility
            scale = max_range * 0.3 if max_range > 1e-6 else 0.01
        else:
            scale = 0.01

        # Plot orientation frames at intervals
        step = max(1, len(positions) // 15)  # Show ~15 frames
        for i in range(0, len(positions), step):
            pos = positions[i]
            # Convert quaternion [x, y, z, w] to rotation matrix
            R = Rotation.from_quat(quats[i]).as_matrix()
            
            # Plot RGB axes: X (Red), Y (Green), Z (Blue)
            # normalize=True ensures arrows are unit length before scaling
            ax.quiver(pos[0], pos[1], pos[2], R[0, 0], R[1, 0], R[2, 0], color='r', length=scale, normalize=True)
            ax.quiver(pos[0], pos[1], pos[2], R[0, 1], R[1, 1], R[2, 1], color='g', length=scale, normalize=True)
            ax.quiver(pos[0], pos[1], pos[2], R[0, 2], R[1, 2], R[2, 2], color='b', length=scale, normalize=True)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Camera Trajectory Visualization')
        
        # Set equal aspect ratio for better visualization
        mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
        mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
        mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        
        ax.legend()
        plt.show()


    def _process_single_arm(self, left_sequence: HandSequence, right_sequence: HandSequence, paths) -> None:
        """Process single-arm setup with one target hand."""
        # Select target hand based on configuration
        target_sequence = left_sequence if self.target_hand == "left" else right_sequence
        
        # Process the selected hand sequence
        target_actions = self._process_hand_sequence(target_sequence, self.T_cam2robot)
        
        # Get indices where hand was detected for this sequence
        union_indices = np.where(target_sequence.hand_detected)[0]
        
        # Refine actions to handle missing detections
        target_actions_refined = self._refine_actions(target_sequence, target_actions, union_indices, self.target_hand)
        
        # Save results for the selected hand only
        if self.target_hand == "left":
            self._save_results(paths, union_indices=union_indices, left_actions=target_actions_refined)
        else:
            self._save_results(paths, union_indices=union_indices, right_actions=target_actions_refined)

    def _process_bimanual(self, left_sequence: HandSequence, right_sequence: HandSequence, paths) -> None:
        """Process bimanual setup with both hands."""
        # Process both hand sequences
        left_actions = self._process_hand_sequence(left_sequence, self.T_cam2robot)
        right_actions = self._process_hand_sequence(right_sequence, self.T_cam2robot)
        
        # Combine detection results using OR logic - frame is valid if either hand detected
        union_indices = np.where(left_sequence.hand_detected | right_sequence.hand_detected)[0]

        # Refine actions for both hands using the union indices
        left_actions_refined = self._refine_actions(left_sequence, left_actions, union_indices, "left")
        right_actions_refined = self._refine_actions(right_sequence, right_actions, union_indices, "right")

        # Save results for both hands
        self._save_results(paths, union_indices, left_actions_refined, right_actions_refined)
    

    def _load_sequences(self, paths) -> Tuple[HandSequence, HandSequence]:
        """
        Load hand sequences from disk for both left and right hands.
        
        HandSequence objects contain the processed keypoint data, detection flags,
        and other metadata needed for action processing.
        
        Args:
            paths: Paths object containing file locations for hand data
            
        Returns:
            Tuple[HandSequence, HandSequence]: Left and right hand sequences
        """
        return (
            HandSequence.load(paths.hand_data_left),
            HandSequence.load(paths.hand_data_right)
        )
    
    def _process_hand_sequence(
        self, 
        sequence: HandSequence, 
        T_cam2robot: np.ndarray,
    ) -> EEActions:
        """
        Process a single hand sequence into end-effector actions.
        
        This method performs the following processing pipeline for one hand:
        1. Transform keypoints from camera frame to robot frame
        2. Fit a hand model to the keypoint sequence
        3. Extract end-effector poses and gripper states
        
        Args:
            sequence (HandSequence): Hand keypoint sequence with detection flags
            T_cam2robot (np.ndarray): 4x4 transformation matrix from camera to robot frame
            
        Returns:
            EEActions: Processed end-effector positions, orientations, and gripper widths
        """
        # Convert keypoints from camera frame to robot frame coordinates
        kpts_3d_cf = sequence.kpts_3d  # Camera frame keypoints
        kpts_3d_rf = ActionProcessor._convert_pts_to_robot_frame(
            kpts_3d_cf, 
            T_cam2robot
        )

        # Create and fit hand model to the keypoint sequence
        hand_model = self._get_hand_model(kpts_3d_rf, sequence.hand_detected)
        
        # Extract end-effector poses and gripper states from fitted model
        kpts_3d, ee_pts, ee_oris = self._get_model_keypoints(hand_model)
        
        # Compute gripper opening distances from fingertip positions
        ee_widths = self._compute_gripper_distances(
            kpts_3d, 
            sequence.hand_detected
        )
        
        return EEActions(
            ee_pts=ee_pts,
            ee_oris=ee_oris,
            ee_widths=ee_widths,
        )

    def _get_hand_model(self, kpts_3d_rf: np.ndarray, hand_detected: np.ndarray) -> HandModel | PhysicallyConstrainedHandModel:
        """
        Create and fit a hand model to the keypoint sequence.
        
        The hand model can be either unconstrained (simple fitting) or physically
        constrained (enforces realistic hand poses and robot constraints).
        
        Args:
            kpts_3d_rf (np.ndarray): Hand keypoints in robot frame, shape (N, 21, 3)
            hand_detected (np.ndarray): Boolean array indicating valid detections, shape (N,)
            
        Returns:
            HandModel | PhysicallyConstrainedHandModel: Fitted hand model with trajectory data
        """
        # Choose hand model type based on configuration
        if self.constrained_hand:
            hand_model = PhysicallyConstrainedHandModel(self.robot)
        else:
            hand_model = HandModel(self.robot)
        
        # Add each frame to the model for trajectory fitting
        for t_idx in range(len(kpts_3d_rf)):
            hand_model.add_frame(
                kpts_3d_rf[t_idx], 
                t_idx * self.dt,  # Convert frame index to time
                hand_detected[t_idx]
            )
        return hand_model
    
    def _get_model_keypoints(self, model: HandModel | PhysicallyConstrainedHandModel) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract keypoints and end-effector data from fitted hand model.
        
        Args:
            model (HandModel | PhysicallyConstrainedHandModel): Fitted hand model
            
        Returns:
            Tuple containing:
                - kpts_3d (np.ndarray): Model keypoint positions, shape (N, 21, 3)
                - ee_pts (np.ndarray): End-effector positions, shape (N, 3)
                - ee_oris (np.ndarray): End-effector orientations, shape (N, 3, 3)
        """
        kpts_3d = np.array(model.vertex_positions)   # All hand keypoints
        ee_pts = np.array(model.grasp_points)        # End-effector positions (palm center)
        ee_oris = np.array(model.grasp_oris)         # End-effector orientations (rotation matrices)
        return kpts_3d, ee_pts, ee_oris

    def _compute_gripper_distances(
        self, 
        kpts_3d_rf: np.ndarray, 
        hand_detected: np.ndarray
    ) -> np.ndarray:
        """
        Compute gripper opening distances for all frames in the sequence.
        
        The gripper distance is calculated as the Euclidean distance between
        the thumb tip and index finger tip, providing a proxy for gripper state.
        
        Args:
            kpts_3d_rf (np.ndarray): Hand keypoints in robot frame, shape (N, 21, 3)
            hand_detected (np.ndarray): Boolean flags for valid detections, shape (N,)
            
        Returns:
            np.ndarray: Gripper distances for each frame, shape (N,)
        """
        gripper_dists = np.zeros(len(kpts_3d_rf))
        
        for idx in range(len(kpts_3d_rf)):
            if hand_detected[idx]:
                # Only compute distance for frames with valid hand detection
                gripper_dists[idx] = ActionProcessor._compute_gripper_opening(
                    kpts_3d_rf[idx]
                )
            # Note: Invalid frames remain at 0.0, will be refined later
        return gripper_dists

    def _refine_actions(
        self, 
        sequence: HandSequence, 
        actions: EEActions,
        union_indices: np.ndarray,
        hand_side: str
    ) -> EEActions:
        """
        Refine actions to handle missing hand detections using last-valid-value interpolation.
        
        When hand detection fails, this method fills in missing values by carrying forward
        the last valid pose and gripper state. This creates smooth, executable trajectories
        even when the vision system temporarily loses tracking.
        
        Args:
            sequence (HandSequence): Original hand sequence with detection flags
            actions (EEActions): Raw actions from hand model
            union_indices (np.ndarray): Frame indices to include in final trajectory
            hand_side (str): "left" or "right" for neutral pose generation
            
        Returns:
            EEActions: Refined actions with interpolated values for missing detections
        """
        # Find frames where this hand was actually detected
        hand_detected_indices = np.where(sequence.hand_detected)[0]
        
        # If no valid detections, return neutral pose for entire sequence
        if len(hand_detected_indices) == 0:
            return self._get_neutral_actions(hand_side, len(union_indices))

        # Apply carry-forward interpolation
        return self._apply_carry_forward_interpolation(sequence, actions, union_indices, hand_detected_indices)

    def _apply_carry_forward_interpolation(
        self, 
        sequence: HandSequence, 
        actions: EEActions,
        union_indices: np.ndarray,
        hand_detected_indices: np.ndarray
    ) -> EEActions:
        """Apply last-valid-value interpolation to fill missing detections."""
        # Initialize with first valid detection values
        first_valid_idx = hand_detected_indices[0]
        last_valid_pt = actions.ee_pts[first_valid_idx]
        last_valid_ori = actions.ee_oris[first_valid_idx]
        last_valid_width = actions.ee_widths[first_valid_idx]
        
        # Process each frame in the union sequence
        ee_pts_refined = []
        ee_oris_refined = []
        ee_widths_refined = []
        
        for idx in union_indices:
            if sequence.hand_detected[idx]:
                # Update with new valid values when available
                last_valid_pt = actions.ee_pts[idx]
                last_valid_ori = actions.ee_oris[idx]
                last_valid_width = actions.ee_widths[idx]
            
            # Always append the last valid values (carry-forward for missing frames)
            ee_pts_refined.append(last_valid_pt)
            ee_oris_refined.append(last_valid_ori)
            ee_widths_refined.append(last_valid_width)
        
        return EEActions(
            ee_pts=np.array(ee_pts_refined),
            ee_oris=np.array(ee_oris_refined),
            ee_widths=np.array(ee_widths_refined),
        )
    
    def _get_neutral_actions(self, hand_side: str, n_frames: int) -> EEActions:
        """
        Generate neutral pose actions when no hand detection is available.
        
        Neutral poses place the robot arms in out-of-frame positions.
        
        Args:
            hand_side (str): "left" or "right" to determine which neutral pose to use
            n_frames (int): Number of frames to generate
            
        Returns:
            EEActions: Neutral pose actions for the specified number of frames
        """
        # Define neutral pose configurations
        neutral_configs = {
            "single_arm": {
                "right": {"pos": [0.2, -0.8, 0.3], "quat": [1, 0.0, 0.0, 0.0]},
                "left": {"pos": [0.2, 0.8, 0.3], "quat": [1, 0.0, 0.0, 0.0]}
            },
            # >>> Hand2Gripper >>>
            # "shoulders": {
            #     "right": {"pos": [0.4, -0.5, 0.3], "quat": [-0.7071, 0.0, 0.0, 0.7071]},
            #     "left": {"pos": [0.4, 0.5, 0.3], "quat": [0.7071, 0.0, 0.0, 0.7071]}
            # }
            "shoulders": {
                "right": {"pos": [None, None, None], "quat": [1, 0.0, 0.0, 0.0]},
                "left": {"pos": [None, None, None], "quat": [1, 0.0, 0.0, 0.0]}
            }
            # <<< Hand2Gripper <<<
        }
        
        # Get configuration for current setup and hand
        config = neutral_configs[self.bimanual_setup][hand_side]
        
        # Convert to numpy arrays and create rotation matrix
        neutral_pos = np.array(config["pos"])
        neutral_ori = Rotation.from_quat(config["quat"], scalar_first=False).as_matrix()
        neutral_width = 0.085  # Standard gripper opening (8.5cm)
        
        # Create arrays replicated for all frames
        return EEActions(
            ee_pts=np.repeat(neutral_pos.reshape(1, 3), n_frames, axis=0),
            ee_oris=np.repeat(neutral_ori.reshape(1, 3, 3), n_frames, axis=0),
            ee_widths=np.full(n_frames, neutral_width)
        )

    def _save_results(
        self, 
        paths: Paths, 
        union_indices: np.ndarray,
        left_actions: Optional[EEActions] = None,
        right_actions: Optional[EEActions] = None,
    ) -> None:
        """
        Save processed action results to disk in NPZ format.
        
        The saved files contain all necessary data for robot execution:
        - union_indices: Valid frame indices in the original sequence
        - ee_pts: End-effector positions
        - ee_oris: End-effector orientations (rotation matrices)
        - ee_widths: Gripper opening widths
        
        Args:
            paths (Paths): File path configuration object
            union_indices (np.ndarray): Valid frame indices
            left_actions (Optional[EEActions]): Left hand actions to save
            right_actions (Optional[EEActions]): Right hand actions to save
        """
        # Create output directory if it doesn't exist
        os.makedirs(paths.action_processor, exist_ok=True)
        
        # Save actions for each hand if provided
        if left_actions is not None:
            self._save_hand_actions(paths.actions_left, union_indices, left_actions)
        if right_actions is not None:
            self._save_hand_actions(paths.actions_right, union_indices, right_actions)

    def _save_hand_actions(self, base_path: str, union_indices: np.ndarray, actions: EEActions) -> None:
        """Save actions for a single hand to NPZ file."""
        file_path = str(base_path).split(".npz")[0] + f"_{self.bimanual_setup}.npz"
        np.savez(
            file_path,
            union_indices=union_indices,
            ee_pts=actions.ee_pts,
            ee_oris=actions.ee_oris,
            ee_widths=actions.ee_widths
        )
    
    @staticmethod
    def _compute_gripper_opening(skeleton_pts: np.ndarray) -> float:
        """
        Compute gripper opening distance from hand keypoints for a single frame.
        
        The gripper distance is calculated as the Euclidean distance between
        the thumb tip and index finger tip.
        
        Args:
            skeleton_pts (np.ndarray): Hand keypoints for one frame, shape (21, 3)
            
        Returns:
            float: Distance between thumb tip and index finger tip in meters
        """
        # Extract finger tip positions from the hand skeleton
        finger_dict = get_list_finger_pts_from_skeleton(skeleton_pts)
        
        # Compute distance between thumb tip and index finger tip
        return np.linalg.norm(finger_dict["thumb"][-1] - finger_dict["index"][-1])
    
    @staticmethod
    def _convert_pts_to_robot_frame(skeleton_poses_cf: np.ndarray, T_cam2robot: np.ndarray) -> np.ndarray:
        """
        Convert hand keypoints from camera frame to robot frame coordinates.
        
        Args:
            skeleton_poses_cf (np.ndarray): Hand poses in camera frame, shape (N, 21, 3).
                                          Format: (Batch, Joints, XYZ).
                                          Units: Meters.
                                          Coordinate System: Camera Frame (Right-handed, Z-forward usually).
            T_cam2robot (np.ndarray): 4x4 transformation matrix from camera to robot frame
            
        Returns:
            np.ndarray: Hand poses in robot frame, shape (N, 21, 3)
        """
        # Convert to homogeneous coordinates by adding ones
        pts_h = np.ones((skeleton_poses_cf.shape[0], skeleton_poses_cf.shape[1], 1))
        skeleton_poses_cf_h = np.concatenate([skeleton_poses_cf, pts_h], axis=-1)
        
        # Apply transformation matrix to convert coordinate frames
        skeleton_poses_rf_h0 = np.einsum('ij,bpj->bpi', T_cam2robot, skeleton_poses_cf_h)
        
        # Remove homogeneous coordinate and return 3D points
        return skeleton_poses_rf_h0[..., :3]