import os
import cv2
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import logging
import mediapy as media
from typing import List
import tqdm

from phantom.processors.base_processor import BaseProcessor
from phantom.processors.phantom_data import HandSequence
from phantom.utils.hand2gripper_visualize import vis_hand_2D_skeleton_without_bbox, vis_selected_gripper

logger = logging.getLogger(__name__)

CHECKING_EXISTING = True

@dataclass
class Hand2GripperLabel:
    """
    """
    img_rgb: np.ndarray       # Full RGB image (H, W, 3)
    bbox: np.ndarray         # Hand bounding box (4,)
    crop_img_rgb: np.ndarray      # Cropped RGB image (256, 256, 3)
    kpts_2d: np.ndarray     # 2D keypoints (21, 2)
    kpts_3d: np.ndarray     # 2D keypoints (21, 3)
    is_right: np.ndarray      # Is right hand flag (bool)
    contact_logits: np.ndarray  # Contact logits (21,)
    selected_gripper_blr_ids: np.ndarray  # Selected gripper bottom-left-right IDs (3,)

class Hand2GripperAnnotator(BaseProcessor):
    """
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.last_chosen_gripper_joints_seq = [0, 0, 0]  # 初始化为无效值

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
        for dir_path in [paths.hand2gripper_labels_left, paths.hand2gripper_labels_right]:
            os.makedirs(dir_path, exist_ok=True)
        # Load RGB video frames
        imgs_rgb = media.read_video(getattr(paths, f"video_left"))
        self.H, self.W, _ = imgs_rgb[0].shape

        # Load hand sequence data for both hands
        left_sequence, right_sequence = self._load_sequences(paths)
        # Load bounding box data
        bbox_data = np.load(paths.bbox_data)

        # Handle bimanual processing mode
        self._process_bimanual(left_sequence, right_sequence, imgs_rgb, bbox_data, paths)

    def _process_bimanual(self, left_sequence: HandSequence, right_sequence: HandSequence, imgs_rgb: List[np.ndarray], bbox_data: np.ndarray, paths) -> None:
        """
        """
        # Process both hand sequences
        assert len(left_sequence.frame_indices) == len(right_sequence.frame_indices) == len(imgs_rgb), "Frame count mismatch among left hand, right hand, and RGB images."
        assert len(bbox_data['left_bboxes']) == len(bbox_data['right_bboxes']) == len(imgs_rgb), "Frame count mismatch among bounding boxes and RGB images."
        self._process_single_arm("left", left_sequence, imgs_rgb, bbox_data['left_bboxes'], paths)
        self._process_single_arm("right", right_sequence, imgs_rgb, bbox_data['right_bboxes'], paths)

    def _process_single_arm(self, hand_side: str, hand_sequence: HandSequence, imgs_rgb: List[np.ndarray], bbox_sequence: np.ndarray, paths) -> None:
        """
        """
        for frame_indice in tqdm.tqdm(range(len(hand_sequence.frame_indices)), desc=f"Labeling {hand_side} hand"):
            if CHECKING_EXISTING and self._check_label(hand_side, frame_indice, paths):
                continue  # Skip if label already exists
            
            # Extract data for current frame
            hand_detected = hand_sequence.hand_detected[frame_indice]  # (bool)
            if not hand_detected:
                # Skip frames without detected hand
                continue

            bbox = bbox_sequence[frame_indice].astype(np.int32)  # (4,)
            kpts_2d = hand_sequence.kpts_2d[frame_indice]  # (21, 2)
            kpts_3d = hand_sequence.kpts_3d[frame_indice]  # (21, 3)
            contact_logits = hand_sequence.contact_logits[frame_indice]  # (21,)
            crop_img_rgb = hand_sequence.crop_img_rgb[frame_indice]  # (256, 256, 3)
            img_rgb = np.array(imgs_rgb[frame_indice]).astype(np.uint8)  # (H, W, 3)

            # Select gripper IDs
            window_name = f"{paths.hand2gripper_labels_left}_{frame_indice}" if hand_side == "left" else f"{paths.hand2gripper_labels_right}_{frame_indice}"
            # selected_gripper_blr_ids = np.array([0, 4, 8]) if hand_side == "right" else np.array([0, 8, 4])
            selected_gripper_blr_ids = self._select_gripper_ids(img_rgb, crop_img_rgb, kpts_2d, contact_logits, hand_side, window_name)

            # Create Hand2GripperLabel
            label = Hand2GripperLabel(
                img_rgb=img_rgb,
                crop_img_rgb=crop_img_rgb,
                bbox=bbox,
                kpts_2d=kpts_2d,
                kpts_3d=kpts_3d,
                is_right=np.array([hand_side == "right"]).astype(np.bool_),
                contact_logits=contact_logits,
                selected_gripper_blr_ids=selected_gripper_blr_ids
            )

            # Save label to disk
            self._save_label(label, hand_side, frame_indice, paths)
    

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

    def _save_label(self, label: Hand2GripperLabel, hand_side: str, frame_indice: int, paths) -> None:
        """
        Save Hand2GripperLabel to disk.

        Args:
            label (Hand2GripperLabel): The label data to save
            hand_side (str): The hand side ("left" or "right")
            frame_indice (int): The frame index
        """
        if hand_side == "left":
            save_path = paths.hand2gripper_labels_left / f"{frame_indice}.npz"
        else:
            save_path = paths.hand2gripper_labels_right / f"{frame_indice}.npz"

        np.savez(
            save_path,
            img_rgb=label.img_rgb,  # (H, W, 3)
            bbox=label.bbox,  # (4,)
            crop_img_rgb=label.crop_img_rgb,  # (256, 256, 3)
            kpts_2d=label.kpts_2d,  # (21, 2)
            kpts_3d=label.kpts_3d,  # (21, 3)
            is_right=label.is_right,  # (1,)
            contact_logits=label.contact_logits,  # (21,)
            selected_gripper_blr_ids=label.selected_gripper_blr_ids  # (3,)
        )
    
    def _check_label(self, hand_side: str, frame_indice: int, paths) -> bool:
        """
        Check if Hand2GripperLabel already exists on disk.
        Args:
            hand_side (str): The hand side ("left" or "right")
            frame_indice (int): The frame index
        Returns:
            bool: True if label file exists, False otherwise
        """
        if hand_side == "left":
            save_path = paths.hand2gripper_labels_left / f"{frame_indice}.npz"
        else:
            save_path = paths.hand2gripper_labels_right / f"{frame_indice}.npz"
        return os.path.exists(save_path)

    def _resize_image_to_match_height(self, img_to_resize: np.ndarray, target_height: int) -> np.ndarray:
        """Resizes an image to a target height while maintaining aspect ratio."""
        original_height, original_width, _ = img_to_resize.shape
        if original_height == target_height:
            return img_to_resize
        
        aspect_ratio = original_width / original_height
        new_width = int(target_height * aspect_ratio)
        
        return cv2.resize(img_to_resize, (new_width, target_height), interpolation=cv2.INTER_AREA)

    def _select_gripper_ids(self, img_rgb: np.ndarray, crop_img_rgb: np.ndarray, kpts_2d: np.ndarray, contact_logits: np.ndarray, hand_side: str, window_name: str) -> np.ndarray:
        """
        Select gripper bottom-left-right IDs based on hand keypoints and contact logits.

        Args:
            img_rgb (np.ndarray): Full RGB image (H, W, 3)
            crop_img_rgb (np.ndarray): Cropped RGB image (256, 256, 3)
            kpts_2d (np.ndarray): 2D keypoints (21, 2)
            contact_logits (np.ndarray): Contact logits (21,)
            hand_side (str): The hand side ("left" or "right")

        Returns:
            np.ndarray: Selected gripper IDs (3,)
        """
        # --- Prepare base images ---
        UI_left_image = img_rgb.copy()
        UI_middle_image_base = vis_hand_2D_skeleton_without_bbox(img_rgb.copy(), kpts_2d, is_right=(hand_side=="right"))
        UI_right_image_orig = crop_img_rgb.copy()
        
        # --- Resize images to the same height ---
        target_height = UI_left_image.shape[0]
        UI_middle_image_base = self._resize_image_to_match_height(UI_middle_image_base, target_height)
        UI_right_image_resized = self._resize_image_to_match_height(UI_right_image_orig, target_height)

        # --- Create the initial combined image ---
        UI_base_images = np.concatenate([UI_left_image, UI_middle_image_base, UI_right_image_resized], axis=1)
        
        gripper_joints_seq = []
        while True:
            cv2.imshow(window_name, cv2.cvtColor(UI_base_images, cv2.COLOR_RGB2BGR))
            cv2.waitKey(100)
            try:
                user_input = input(f"Enter the gripper joints sequence (3 integers separated by blank, last chosen(Base, Left, Right): {self.last_chosen_gripper_joints_seq}): ")
                if user_input == '':
                    gripper_joints_seq = self.last_chosen_gripper_joints_seq
                else:
                    gripper_joints_seq = [int(x) for x in user_input.split(' ')]
                
                if len(gripper_joints_seq) != 3 or not all(0 <= joint_id <= 20 for joint_id in gripper_joints_seq):
                    print("Invalid input. Please enter 3 joint IDs (base, left, right) between 0 and 20 (e.g., 1 2 3).")
                    continue
                
                # Draw on a copy of the base middle image for confirmation
                UI_middle_image_confirm = vis_selected_gripper(UI_middle_image_base.copy(), kpts_2d, np.array(gripper_joints_seq))
                UI_selected_images = np.concatenate([UI_left_image, UI_middle_image_confirm, UI_right_image_resized], axis=1)
                cv2.imshow(window_name, cv2.cvtColor(UI_selected_images, cv2.COLOR_RGB2BGR))
                cv2.waitKey(100)

                confirm = input("Confirm the selection? (y/n or press Enter): ").lower().strip()
                if confirm in ['', 'y', 'yes']:
                    self.last_chosen_gripper_joints_seq = gripper_joints_seq
                    break
                print("Selection cancelled. Please try again.")
                
            except ValueError:
                print("Invalid input. Please enter 3 integers (base, left, right) separated by blank (e.g., 1 2 3).")
        
        cv2.destroyWindow(window_name)
        return np.array(gripper_joints_seq).astype(np.int32)