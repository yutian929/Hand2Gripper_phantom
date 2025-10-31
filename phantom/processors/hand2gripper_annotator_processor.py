"""
Hand2Gripper Annotator Module

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
import mediapy as media
from scipy.spatial.transform import Rotation
from typing import List

from phantom.processors.base_processor import BaseProcessor
from phantom.processors.phantom_data import HandSequence
from phantom.processors.paths import Paths
from phantom.hand import HandModel, PhysicallyConstrainedHandModel, get_list_finger_pts_from_skeleton

logger = logging.getLogger(__name__)

@dataclass
class Hand2GripperLabel:
    """
    """
    crop_rgb_img: np.ndarray      # Cropped RGB image (H, W, 3)
    kpts_3d: np.ndarray     # 2D keypoints (21, 3)
    is_right: np.ndarray      # Is right hand flag (bool)
    contact_logits: np.ndarray  # Contact logits (21,)

@dataclass
class Hand2GripperLabelSequnence:
    """
    """
    crop_rgb_imgs: np.ndarray      # Cropped RGB images (N, H, W, 3)
    kpts_3d: np.ndarray     # 2D keypoints (N, 21, 3)
    is_right: np.ndarray      # Is right hand flag (N, bool)
    contact_logits: np.ndarray  # Contact logits (N, 21,)

class Hand2GripperAnnotator(BaseProcessor):
    """
    """
    def __init__(self, cfg):
        super().__init__(cfg)

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

        # Load RGB video frames
        imgs_rgb = media.read_video(getattr(paths, f"video_left"))
        self.H, self.W, _ = imgs_rgb[0].shape

        # Load hand sequence data for both hands
        left_sequence, right_sequence = self._load_sequences(paths)
        breakpoint()
        # Handle single-arm processing mode
        if self.bimanual_setup == "single_arm":
            self._process_single_arm(left_sequence, right_sequence, paths)
        else:
            self._process_bimanual(left_sequence, right_sequence, imgs_rgb, paths)

    # def _process_single_arm(self, left_sequence: HandSequence, right_sequence: HandSequence, paths) -> None:
    #     """Process single-arm setup with one target hand."""
    #     # Select target hand based on configuration
    #     target_sequence = left_sequence if self.target_hand == "left" else right_sequence
        
    #     # Process the selected hand sequence
    #     target_actions = self._process_hand_sequence(target_sequence, self.T_cam2robot)
        
    #     # Get indices where hand was detected for this sequence
    #     union_indices = np.where(target_sequence.hand_detected)[0]
        
    #     # Refine actions to handle missing detections
    #     target_actions_refined = self._refine_actions(target_sequence, target_actions, union_indices, self.target_hand)
        
    #     # Save results for the selected hand only
    #     if self.target_hand == "left":
    #         self._save_results(paths, union_indices=union_indices, left_actions=target_actions_refined)
    #     else:
    #         self._save_results(paths, union_indices=union_indices, right_actions=target_actions_refined)

    def _process_bimanual(self, left_sequence: HandSequence, right_sequence: HandSequence, imgs_rgb: List[np.ndarray], paths) -> None:
        """"""
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
 

    # def _save_results(
    #     self, 
    #     paths: Paths, 
    #     union_indices: np.ndarray,
    #     left_actions: Optional[EEActions] = None,
    #     right_actions: Optional[EEActions] = None,
    # ) -> None:
    #     """
    #     Save processed action results to disk in NPZ format.
        
    #     The saved files contain all necessary data for robot execution:
    #     - union_indices: Valid frame indices in the original sequence
    #     - ee_pts: End-effector positions
    #     - ee_oris: End-effector orientations (rotation matrices)
    #     - ee_widths: Gripper opening widths
        
    #     Args:
    #         paths (Paths): File path configuration object
    #         union_indices (np.ndarray): Valid frame indices
    #         left_actions (Optional[EEActions]): Left hand actions to save
    #         right_actions (Optional[EEActions]): Right hand actions to save
    #     """
    #     # Create output directory if it doesn't exist
    #     os.makedirs(paths.action_processor, exist_ok=True)
        
    #     # Save actions for each hand if provided
    #     if left_actions is not None:
    #         self._save_hand_actions(paths.actions_left, union_indices, left_actions)
    #     if right_actions is not None:
    #         self._save_hand_actions(paths.actions_right, union_indices, right_actions)

    # def _save_hand_actions(self, base_path: str, union_indices: np.ndarray, actions: EEActions) -> None:
    #     """Save actions for a single hand to NPZ file."""
    #     file_path = str(base_path).split(".npz")[0] + f"_{self.bimanual_setup}.npz"
    #     np.savez(
    #         file_path,
    #         union_indices=union_indices,
    #         ee_pts=actions.ee_pts,
    #         ee_oris=actions.ee_oris,
    #         ee_widths=actions.ee_widths
    #     )
