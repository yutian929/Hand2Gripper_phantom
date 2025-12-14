from __future__ import annotations
import cv2
import numpy as np
import os
import mujoco
import mediapy as media
from scipy.spatial.transform import Rotation as R
from typing import TYPE_CHECKING, Tuple
from pathlib import Path

# Fix: Import Paths class
from phantom.processors.paths import Paths 

# Fix: Import MuJoCo control modules using absolute path from project root
from submodules.my_robo.raw_mujoco.dual_arm_controller import DualArmController
from submodules.my_robo.raw_mujoco.test_single_arm import load_and_transform_data, pose_to_matrix, matrix_to_pose

# Conditional import for type checking if needed
if TYPE_CHECKING:
    from omegaconf import DictConfig 

# ---------------------------------------------------------
# Robot Inpainting Processor 
# ---------------------------------------------------------

class RobotInpaintProcessor:
    """
    Uses MuJoCo simulation (DualArmController) to overlay robot onto
    human inpainted videos based on trajectory data.
    """

    def __init__(self, args: DictConfig):
        """
        Initialize the processor.
        """
        self.args = args
        self.robot = args.robot
        self.bimanual_setup = args.bimanual_setup
        self.square = args.square
        self.output_resolution = args.output_resolution
        self.skip_existing = args.skip_existing
        
        # Assume data_root_dir is correctly set in cfg (e.g., data/raw/)
        self.data_root_dir = args.data_root_dir
        
        # Fixed Camera relative to Left Arm Base (from original A code)
        self.Mat_base_L_T_camera = np.array([
            [1.0, 0.0, 0.0, 0.0,],
            [0.0, 1.0, 0.0, -0.25,],
            [0.0, 0.0, 1.0, 0.2,],
            [0.0, 0.0, 0.0, 1.0,],
        ])
        
        # Fixed Camera relative to Right Arm Base (from original A code)
        self.Mat_base_R_T_camera = np.array([
            [1.0, 0.0, 0.0, 0.0,],
            [0.0, 1.0, 0.0, 0.25,],
            [0.0, 0.0, 1.0, 0.2,],
            [0.0, 0.0, 0.0, 1.0,],
        ])

    def process_one_demo(self, data_sub_folder: str):
        """
        Processes a single demonstration by simulating robot trajectory and creating an overlay video.
        
        Args:
            data_sub_folder: The name of the demonstration folder (string).
        """
        # ----------------------------------------------------
        # STEP 1: Create Paths Object (FIXED)
        # ----------------------------------------------------
        
        # Construct absolute path to the demo data folder
        data_abs_path = os.path.join(self.data_root_dir, data_sub_folder)
        
        # Use CORRECT parameter names: data_path and robot_name (from paths.py definition)
        paths = Paths(
            data_path=Path(data_abs_path),
            robot_name=self.robot,
        )

        output_video_path = str(paths.video_overlay).split(".mkv")[0] + f"_{self.robot}_{self.bimanual_setup}.mkv"
        
        if self.skip_existing and os.path.exists(output_video_path):
            print(f"Skipping existing demo {output_video_path}")
            return
        
        # ----------------------------------------------------
        # STEP 2: Initialize MuJoCo Controller and Load XML
        # ----------------------------------------------------
        
        # Get project root assuming current working directory is project root
        project_root = os.getcwd() 
        xml_path = os.path.join(
            project_root,
            "submodules",
            "my_robo",
            "raw_mujoco",
            "R5",
            "R5a",
            "meshes",
            "dual_arm_scene.xml")
            
        if not os.path.exists(xml_path):
            print(f"MuJoCo XML file not found: {xml_path}")
            return
        
        print(f"Loading MuJoCo model: {xml_path}")
        dual_robot = DualArmController(xml_path, max_steps=100)
        
        base_pose_world_L = dual_robot._get_base_pose_world("L")
        base_pose_world_R = dual_robot._get_base_pose_world("R")
        Mat_world_T_base_L = pose_to_matrix(base_pose_world_L)
        Mat_world_T_base_R = pose_to_matrix(base_pose_world_R)
        
        # ----------------------------------------------------
        # STEP 3: Load and Transform Trajectory Data
        # ----------------------------------------------------

        try:
            rgbs_inpaint = media.read_video(str(paths.video_human_inpaint))
            print(f"Loaded inpaint video: {len(rgbs_inpaint)} frames")
        except Exception as e:
            print(f"Could not load inpaint video: {e}")
            return

        try:
            seqs_L_in_camera_link = load_and_transform_data(str(paths.smoothed_actions_left))
            seqs_R_in_camera_link = load_and_transform_data(str(paths.smoothed_actions_right))
        except Exception as e:
            print(f"Failed to load A-format trajectory data, attempting B-format fallback: {e}")
            seqs_L_in_camera_link, seqs_R_in_camera_link = self._load_b_format_data(paths)
        
        if seqs_L_in_camera_link is None or seqs_R_in_camera_link is None:
            print("Trajectory data loading failed")
            return
        
        # Transform poses to World frame
        Mat_camera_T_seqs_L = np.array([pose_to_matrix(pose) for pose in seqs_L_in_camera_link])
        Mat_camera_T_seqs_R = np.array([pose_to_matrix(pose) for pose in seqs_R_in_camera_link])
        
        Mat_world_T_seqs_L = Mat_world_T_base_L @ self.Mat_base_L_T_camera @ Mat_camera_T_seqs_L
        Mat_world_T_seqs_R = Mat_world_T_base_R @ self.Mat_base_R_T_camera @ Mat_camera_T_seqs_R
        
        seqs_L_in_world = np.array([matrix_to_pose(mat) for mat in Mat_world_T_seqs_L])
        seqs_R_in_world = np.array([matrix_to_pose(mat) for mat in Mat_world_T_seqs_R])
        
        Mat_world_T_camera_L = Mat_world_T_base_L @ self.Mat_base_L_T_camera
        camera_poses_world = np.array([matrix_to_pose(Mat_world_T_camera_L) for _ in range(len(seqs_L_in_world))])
        
        # ----------------------------------------------------
        # STEP 4: Execution, Rendering, and Blending
        # ----------------------------------------------------

        min_frames = min(len(seqs_L_in_world), len(seqs_R_in_world), len(rgbs_inpaint))
        if min_frames == 0:
            print("No valid frame data available")
            return
        
        print(f"Processing {min_frames} frames")
        
        seqs_L_in_world = seqs_L_in_world[:min_frames]
        seqs_R_in_world = seqs_R_in_world[:min_frames]
        camera_poses_world = camera_poses_world[:min_frames]
        
        print("Executing dual arm trajectory and capturing images...")
        frames, masks = dual_robot.move_trajectory_with_camera(
            seqs_L_in_world, seqs_R_in_world, camera_poses_world, 
            50, 
            kinematic_only=True,
            cam_name="camera",
            width=640,
            height=480
        )
        
        if len(frames) == 0:
            print("No frames captured from MuJoCo")
            return
        
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        
        blended_frames = []
        
        for i, (rgb_inpaint, frame, mask) in enumerate(zip(rgbs_inpaint[:min_frames], frames, masks)):
            if frame.shape[:2] != rgb_inpaint.shape[:2]:
                frame = cv2.resize(frame, (rgb_inpaint.shape[1], rgb_inpaint.shape[0]))
                mask = cv2.resize(mask, (rgb_inpaint.shape[1], rgb_inpaint.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            geom_ids = mask[:, :, 0]
            mask_bool = geom_ids > 0
            blended = rgb_inpaint.copy()
            blended[mask_bool] = frame[mask_bool]
            
            if self.square and self.output_resolution > 0:
                blended = self._crop_to_square(blended, self.output_resolution)
            
            blended_frames.append(blended)
            
            if i % 50 == 0:
                print(f"Processing frame {i}/{min_frames}")
            
        print(f"Saving video to: {output_video_path}")
        self._save_video(output_video_path, blended_frames)
        
        print("Processing complete!")
    
    def _load_data(self, paths: Paths) -> dict | None:
        """
        Loads trajectory data in B-code format (ee_pts and ee_oris).
        """
        try:
            left_data = np.load(str(paths.smoothed_actions_left))
            right_data = np.load(str(paths.smoothed_actions_right))
            
            if "ee_pts" in left_data and "ee_oris" in left_data:
                return {
                    'ee_pts_left': left_data["ee_pts"],
                    'ee_oris_left': left_data["ee_oris"],
                    'ee_pts_right': right_data["ee_pts"],
                    'ee_oris_right': right_data["ee_oris"],
                }
            return None
        except Exception as e:
            print(f"Error loading B-format data structure: {e}")
            return None
    
    def _load_b_format_data(self, paths: Paths) -> Tuple[np.ndarray | None, np.ndarray | None]:
        """
        Loads B-code data (pos + rot matrix) and converts it to A-code format (pos + euler).
        """
        try:
            data = self._load_data(paths)
            if data is None:
                return None, None
            
            ee_pts_left = data["ee_pts_left"]
            ee_oris_left = data["ee_oris_left"]
            ee_pts_right = data["ee_pts_right"]
            ee_oris_right = data["ee_oris_right"]
            
            from scipy.spatial.transform import Rotation
            
            seqs_L, seqs_R = [], []
            
            for i in range(len(ee_pts_left)):
                r_left = Rotation.from_matrix(ee_oris_left[i])
                euler_left = r_left.as_euler('xyz', degrees=False)
                seq_L = np.concatenate([ee_pts_left[i], euler_left])
                seqs_L.append(seq_L)
                
                r_right = Rotation.from_matrix(ee_oris_right[i])
                euler_right = r_right.as_euler('xyz', degrees=False)
                seq_R = np.concatenate([ee_pts_right[i], euler_right])
                seqs_R.append(seq_R)
            
            return np.array(seqs_L), np.array(seqs_R)
            
        except Exception as e:
            print(f"Could not convert B-format data to A-format: {e}")
            return None, None
    
    def _crop_to_square(self, image: np.ndarray, size: int) -> np.ndarray:
        """
        Crops an image to a square shape from the center and resizes.
        """
        h, w = image.shape[:2]
        
        if h > w:
            start = (h - w) // 2
            cropped = image[start:start+w, :]
        else:
            start = (w - h) // 2
            cropped = image[:, start:start+h]
        
        if cropped.shape[0] != size or cropped.shape[1] != size:
            cropped = cv2.resize(cropped, (size, size))
        
        return cropped
    
    def _save_video(self, path: str, frames: list[np.ndarray]):
        """
        Saves video using mediapy with specific codec and FPS.
        """
        try:
            media.write_video(
                path,
                frames,
                fps=15,
                codec="ffv1"
            )
            print(f"Video saved to: {path}")
        except Exception as e:
            print(f"Error saving video: {e}")
