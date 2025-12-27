import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import cv2
import json
import bimanual
from scipy.spatial.transform import Rotation as R

# 特定补偿值
FLANGE_MUJOCO_REAL_GAP = np.array([-0.02, -0.01, -0.066])
FLANGE_GRIPPER_GAP = np.array([-0.16, 0.0, 0.0])  # Gripper to Flange offset

class DualArmController:
    def __init__(self, xml_path, arm_names=None, end_effector_site_names=None, base_link_names=None, position_threshold=0.02):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML file not found: {xml_path}")
            
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.ARM_NAMES = ['L', 'R'] if not arm_names else arm_names
        self.EE_SITE_NAMES = {name: f"end_effector_{name}" for name in self.ARM_NAMES} if end_effector_site_names is None else end_effector_site_names
        self.BASE_LINK_NAMES = {name: f"base_link_{name}" for name in self.ARM_NAMES} if base_link_names is None else base_link_names
        
        self.GRIPPER_CLOSE_VAL = 0.0
        self.GRIPPER_OPEN_VAL = 0.044
        self.POSITION_THRESHOLD = position_threshold
        
        self.arm_params = {}
        self._init_kinematics_params()
        
        self.reset()

    def _init_kinematics_params(self):
        for name in self.ARM_NAMES:
            params = {}
            
            # 1. Kinematic Parameters (基座位置、EE 偏移)
            base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.BASE_LINK_NAMES[name])
            params['base_pos_world'] = self.model.body_pos[base_id].copy() if base_id != -1 else np.zeros(3)

            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.EE_SITE_NAMES[name])
            params['ee_offset_local'] = self.model.site_pos[site_id].copy() if site_id != -1 else np.zeros(3)

            # 2. Joint Indices (QPOS)
            # 假设关节命名为 joint1_L, joint2_L ...
            j_names_6dof = [f"joint{i}_{name}" for i in range(1, 7)]
            j_ids_6dof = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j_name) for j_name in j_names_6dof]
            
            j_names_gripper = [f"joint7_{name}", f"joint8_{name}"]
            j_ids_gripper = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j_name) for j_name in j_names_gripper]
            
            params['qpos_indices_6dof'] = [self.model.jnt_qposadr[j_id] for j_id in j_ids_6dof if j_id != -1]
            params['qpos_indices_gripper'] = [self.model.jnt_qposadr[j_id] for j_id in j_ids_gripper if j_id != -1]
            
            # Dynamics related code removed (Actuator Indices)

            self.arm_params[name] = params
            
            # Debug info
            print(f"[{name}] QPOS Indices: {params['qpos_indices_6dof']}")

    def reset(self):
        self.data.qpos[:] = 0
        # Dynamics related code removed (ctrl)
        mujoco.mj_forward(self.model, self.data)

    def _ik_base(self, target_pose_base):
        try:
            return bimanual.inverse_kinematics(target_pose_base)
        except Exception:
            return None
    
    def _get_base_pose_world(self, arm_name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.BASE_LINK_NAMES[arm_name])
        if body_id == -1: 
            print(f"[Error] Base link '{self.BASE_LINK_NAMES[arm_name]}' not found.")
            return np.zeros(6)
        pos = self.data.xpos[body_id]
        mat = self.data.xmat[body_id].reshape(3, 3)
        r = R.from_matrix(mat)
        euler = r.as_euler('xyz', degrees=False)
        return np.concatenate([pos, euler])
            
    def _get_ee_pose_world(self, arm_name):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.EE_SITE_NAMES[arm_name])
        if site_id == -1: return np.zeros(6)
             
        pos = self.data.site_xpos[site_id]
        mat = self.data.site_xmat[site_id].reshape(3, 3)
        r = R.from_matrix(mat)
        euler = r.as_euler('xyz', degrees=False)
        return np.concatenate([pos, euler])

    def _pose_tf_world2base(self, target_pose_world, arm_name):
        # 1. Construct T_world_target
        pos_target = target_pose_world[:3]
        euler_target = target_pose_world[3:6]
        r_target = R.from_euler('xyz', euler_target, degrees=False)
        
        T_world_target = np.eye(4)
        T_world_target[:3, :3] = r_target.as_matrix()
        T_world_target[:3, 3] = pos_target

        # 2. Get T_world_base from MuJoCo data
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.BASE_LINK_NAMES[arm_name])
        if body_id == -1:
            print(f"[Error] Base link for {arm_name} not found.")
            return T_world_target

        pos_base = self.data.xpos[body_id]
        mat_base = self.data.xmat[body_id].reshape(3, 3)

        T_world_base = np.eye(4)
        T_world_base[:3, :3] = mat_base
        T_world_base[:3, 3] = pos_base

        # 3. Compute T_base_target = inv(T_world_base) @ T_world_target
        # Using explicit inverse for SE(3) is slightly faster/cleaner but linalg.inv is fine here
        R_wb = T_world_base[:3, :3]
        t_wb = T_world_base[:3, 3]
        
        T_base_world = np.eye(4)
        T_base_world[:3, :3] = R_wb.T
        T_base_world[:3, 3] = -R_wb.T @ t_wb
        
        T_base_target = T_base_world @ T_world_target
        
        # Convert to xyzrpy
        xyz = T_base_target[:3, 3]
        r = R.from_matrix(T_base_target[:3, :3])
        rpy = r.as_euler('xyz', degrees=False)
        
        return np.concatenate([xyz, rpy])
        

    def _update_mocap_marker(self, pos, arm_name):
        marker_name = f"target_marker_{arm_name}"
        try:
            target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, marker_name)
            if target_body_id != -1:
                mocap_id = self.model.body_mocapid[target_body_id]
                self.data.mocap_pos[mocap_id] = pos
        except: pass

    # ========================================================
    # 【新增】辅助函数：设置相机位姿
    # ========================================================
    def _set_camera_pose(self, cam_name, pose_world):
        """
        设置指定相机的位姿。
        """
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam_id == -1:
            print(f"[Warning] Camera '{cam_name}' not found.")
            return

        # 1. 设置位置
        self.model.cam_pos[cam_id] = pose_world[:3]

        # 2. 处理姿态旋转
        r_input = R.from_euler('xyz', pose_world[3:], degrees=False)
        mat_input = r_input.as_matrix()

        mat_link_to_optical = np.array([
            [ 0,  0, -1],
            [-1,  0,  0],
            [ 0,  1,  0]
        ])

        mat_final = mat_input @ mat_link_to_optical

        r_final = R.from_matrix(mat_final)
        quat_scipy = r_final.as_quat() # [x, y, z, w]
        
        # 转换为 MuJoCo 顺序 [w, x, y, z]
        quat_mujoco = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])
        
        self.model.cam_quat[cam_id] = quat_mujoco

    # ========================================================
    # 【实现】轨迹跟踪 + 相机拍摄 (同步修改支持 (N,7) 输入)
    # ========================================================
    def move_trajectory_with_camera(self, target_seqs_L_world, target_seqs_R_world, camera_poses_world, cam_name="camera", width=640, height=480):
        """
        连续执行一系列双臂目标点(Flange)，并同步移动相机进行拍摄。
        输入形状需为 (N, 7),第7位为夹爪宽度。
        """
        if not self.ARM_NAMES: return [], []
        
        # 1. 验证输入数据
        target_world_seqs = {
            'L': target_seqs_L_world,
            'R': target_seqs_R_world
        }
        
        try:
            num_points = len(target_world_seqs['L'])
            if len(target_world_seqs['R']) != num_points or len(camera_poses_world) != num_points:
                print("[Error] All trajectory sequences must have the same length.")
                return [], []
        except Exception as e:
            print(f"[Error] Invalid target sequence format: {e}")
            return [], []

        print(f"\n[Trajectory & Cam] Processing {num_points} frames...")

        # 2. 初始化 Renderer
        renderer = mujoco.Renderer(self.model, height=height, width=width)

        # 3. 预计算：解算所有点的 IK 和 Gripper
        joint_targets_queue = {name: [] for name in self.ARM_NAMES}
        gripper_targets_queue = {name: [] for name in self.ARM_NAMES}
        last_valid_joints = {}
        for name in self.ARM_NAMES:
            qpos_indices = self.arm_params[name]['qpos_indices_6dof']
            if qpos_indices:
                last_valid_joints[name] = self.data.qpos[qpos_indices].copy()
            else:
                last_valid_joints[name] = np.zeros(6)

        for i in range(num_points):
            for name in self.ARM_NAMES:
                full_data = target_world_seqs[name][i]
                pose = full_data[:6].copy()  # Make a copy to avoid modifying original target for visualization
                pose[:3] += FLANGE_MUJOCO_REAL_GAP  # mujoco flange dismatch compensation
                # now pose is flange in world frame
                
                # Gripper Logic
                if len(full_data) >= 7:
                    raw_width = full_data[6]
                    clipped_width = np.clip(raw_width, self.GRIPPER_CLOSE_VAL, self.GRIPPER_OPEN_VAL)
                else:
                    clipped_width = self.GRIPPER_OPEN_VAL
                gripper_targets_queue[name].append(clipped_width)

                # IK Logic
                pose_base = self._pose_tf_world2base(pose, name)  # flange in base frame
                # breakpoint()
                joints = self._ik_base(pose_base)
                
                if joints is None:
                    joints = last_valid_joints[name].copy()
                else:
                    last_valid_joints[name] = joints.copy()
                joint_targets_queue[name].append(joints)

        # 4. 执行循环 (Kinematic Replay)
        rgb_list = []
        mask_list = []
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            for i in range(num_points):
                if not viewer.is_running():
                    break
                
                # 获取当前帧目标
                current_joint_targets = {name: joint_targets_queue[name][i] for name in self.ARM_NAMES}
                current_gripper_targets = {name: gripper_targets_queue[name][i] for name in self.ARM_NAMES}
                
                # Retrieve original target pose (World Frame) for visualization and error checking
                # Note: target_world_seqs contains the original input sequence (unmodified)
                current_pose_targets = {name: target_world_seqs[name][i][:6] for name in self.ARM_NAMES}
                current_cam_pose = camera_poses_world[i]
                
                # 更新 Mocap Marker (Visualize the input target sequence in World Frame)
                for name in self.ARM_NAMES:
                    self._update_mocap_marker(current_pose_targets[name][:3], name)
                
                # 设置关节角度 (Kinematic)
                for name in self.ARM_NAMES:
                    params = self.arm_params[name]
                    if params['qpos_indices_6dof']:
                        self.data.qpos[params['qpos_indices_6dof']] = current_joint_targets[name]
                    if params['qpos_indices_gripper']:
                        self.data.qpos[params['qpos_indices_gripper']] = current_gripper_targets[name]
                
                # 设置相机位姿
                self._set_camera_pose(cam_name, current_cam_pose)
                
                # 刷新模型状态
                mujoco.mj_forward(self.model, self.data)
                
                # --- Calculate and Print Flange Error ---
                for name in self.ARM_NAMES:
                    # Current Flange (Link6) Position
                    flange_body_name = f"link6_{name}"
                    bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, flange_body_name)
                    curr_flange = self.data.xpos[bid] if bid != -1 else np.zeros(3)
                    
                    # Target Flange Position = Input Target (Gripper) + Offset
                    targ_flange = current_pose_targets[name][:3]
                    
                    diff = curr_flange - targ_flange
                    dist = np.linalg.norm(diff)
                    
                    print(f"[{name}] Flange Err: {dist:.5f} | XYZ: {np.round(diff, 5)}")

                # 渲染 RGB
                renderer.update_scene(self.data, camera=cam_name)
                rgb_list.append(renderer.render().copy())
                
                # 渲染 Mask
                renderer.enable_segmentation_rendering()
                renderer.update_scene(self.data, camera=cam_name)
                mask_list.append(renderer.render().copy())
                renderer.disable_segmentation_rendering()
                
                # 同步 Viewer
                viewer.sync()
                
                # 可选：稍微延时以便肉眼观察
                time.sleep(0.05)

            print("Trajectory finished. Viewer is open. Close the window to continue...")
            # while viewer.is_running():
            #     time.sleep(10)
            #     viewer.sync()

        print("[Trajectory & Cam] All frames captured.")
        return rgb_list, mask_list

# ========================================================
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "R5", "R5a", "meshes", "dual_arm_scene.xml")
    
    aruco_pic = "/home/yutian/Hand2Gripper_phantom/data/raw/epic/0/frames/frame_0005.png"
    calib_path_L = "/home/yutian/Hand2Gripper_phantom/phantom/camera/eye_to_hand_result_left_latest.json"
    calib_path_R = "/home/yutian/Hand2Gripper_phantom/phantom/camera/eye_to_hand_result_right_latest.json"
    
    def load_calibration_matrix(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return np.array(data["Mat_base_T_camera_link"])

    try:
        # 1. Load Calibration
        Mat_base_L_T_camera = load_calibration_matrix(calib_path_L)  # 左臂基座坐标下下相机的位姿
        Mat_base_R_T_camera = load_calibration_matrix(calib_path_R)  # 右臂基座坐标下下相机的位姿

        # 2. Load Image
        if not os.path.exists(aruco_pic):
            raise FileNotFoundError(f"Image not found: {aruco_pic}")
        img = cv2.imread(aruco_pic)
        if img is None:
            raise RuntimeError(f"Failed to read image: {aruco_pic}")

        # 3. Detect ArUco
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        if hasattr(cv2.aruco, "ArucoDetector"):
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            corners, ids, _ = detector.detectMarkers(img)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
        
        target_pos_world = None
        target_euler_world = None

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(img, corners, ids)
            
            # Camera Intrinsics (from provided JSON)
            fx, fy = 606.0810546875, 605.1178588867188
            cx, cy = 327.5788879394531, 245.88775634765625
            camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
            dist_coeffs = np.zeros(5)
            marker_length = 0.05  # 5cm

            # Transformation Matrix: Optical -> Camera Link
            # Link X (Fwd) = Optical Z
            # Link Y (Left) = -Optical X
            # Link Z (Up) = -Optical Y
            Mat_link_T_optical = np.array([
                [0, 0, 1, 0],
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 0, 1]
            ])

            # Estimate Pose using solvePnP (compatible with new OpenCV)
            half_size = marker_length / 2.0
            # Marker corners in object frame (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
            obj_points = np.array([
                [-half_size, half_size, 0],
                [half_size, half_size, 0],
                [half_size, -half_size, 0],
                [-half_size, -half_size, 0]
            ], dtype=np.float32)

            for i in range(len(ids)):
                success, rvec, tvec = cv2.solvePnP(obj_points, corners[i], camera_matrix, dist_coeffs)
                
                if success:
                    # Draw Axis (in Optical Frame for visualization on image)
                    cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, marker_length)
                    
                    # Draw Center Point
                    c = corners[i][0]
                    center_x = int(np.mean(c[:, 0]))
                    center_y = int(np.mean(c[:, 1]))
                    cv2.circle(img, (center_x, center_y), 5, (0, 255, 255), -1)
                    
                    # --- Calculate Full Pose in World Frame ---
                    
                    # 1. T_optical_marker
                    R_opt_marker, _ = cv2.Rodrigues(rvec)
                    T_opt_marker = np.eye(4)
                    T_opt_marker[:3, :3] = R_opt_marker
                    T_opt_marker[:3, 3] = tvec.flatten()
                    
                    # 2. T_world_baseL (Base L is at 0,0,1.0)
                    T_world_baseL = np.eye(4)
                    T_world_baseL[:3, 3] = [0.0, 0.2, 1.0]
                    
                    # 3. Chain: World -> BaseL -> CamLink -> CamOpt -> Marker
                    # T_world_marker = T_world_baseL @ T_baseL_camLink @ T_camLink_camOpt @ T_opt_marker
                    T_world_marker = T_world_baseL @ Mat_base_L_T_camera @ Mat_link_T_optical @ T_opt_marker
                    
                    pos_world = T_world_marker[:3, 3]
                    
                    # Extract Euler Angles (xyz)
                    r_world = R.from_matrix(T_world_marker[:3, :3])
                    euler_world = r_world.as_euler('xyz', degrees=False)
                    
                    # Store the first marker position/orientation as target
                    if target_pos_world is None:
                        target_pos_world = pos_world
                        target_euler_world = euler_world

                    text = f"ID:{ids[i][0]} World:({pos_world[0]:.2f}, {pos_world[1]:.2f}, {pos_world[2]:.2f})"
                    cv2.putText(img, text, (center_x + 10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Detected Markers", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if target_pos_world is not None:
            print(f"Target World Position: {target_pos_world}")
            print(f"Target World Euler (rad): {target_euler_world}")
            
            # Initialize Robot
            robot = DualArmController(xml_path)
            
            # --- Calculate Camera Pose in World Frame ---
            def pose_to_matrix(pose):
                t = pose[:3]
                euler = pose[3:]
                r = R.from_euler('xyz', euler, degrees=False)
                mat = np.eye(4)
                mat[:3, :3] = r.as_matrix()
                mat[:3, 3] = t
                return mat

            def matrix_to_pose(mat):
                t = mat[:3, 3]
                rot_mat = mat[:3, :3]
                r = R.from_matrix(rot_mat)
                euler = r.as_euler('xyz', degrees=False)
                return np.concatenate([t, euler])

            # Get Base Poses in World Frame
            pose_base_L = robot._get_base_pose_world("L")
            pose_base_R = robot._get_base_pose_world("R")
            T_W_BL = pose_to_matrix(pose_base_L)
            T_W_BR = pose_to_matrix(pose_base_R)

            # Calculate Camera Pose (Camera Link Frame) in World Frame
            T_W_C_L = T_W_BL @ Mat_base_L_T_camera
            T_W_C_R = T_W_BR @ Mat_base_R_T_camera

            # Check Consistency
            diff_t = np.linalg.norm(T_W_C_L[:3, 3] - T_W_C_R[:3, 3])
            print(f"Camera World Pos from L: {T_W_C_L[:3, 3]}")
            print(f"Camera World Pos from R: {T_W_C_R[:3, 3]}")
            print(f"Camera Position Discrepancy: {diff_t:.4f} m")

            if diff_t > 0.05: # 5cm threshold
                print("[Warning] Large discrepancy in camera pose calculation!")
            
            # Use Left result for simulation camera
            cam_pose_world = matrix_to_pose(T_W_C_L)
            camera_poses = [cam_pose_world]
            
            # Apply Gripper Offset to get Flange Target
            # Target is marker position, we want gripper to be there.
            # Flange = Marker + FLANGE_GRIPPER_GAP
            flange_target_pos = target_pos_world + FLANGE_GRIPPER_GAP
            
            # Define Pose: [x, y, z, r, p, y, gripper]
            # Orientation: Use detected marker orientation
            # Gripper: 0.044 (Open)
            target_pose = np.concatenate([flange_target_pos, target_euler_world, [0.044]])
            
            seqs_L = [target_pose]
            seqs_R = [target_pose]
            
            print("Moving arms to ArUco marker...")
            
            # Get image dimensions for rendering
            h, w = img.shape[:2]
            
            rgb_frames, mask_frames = robot.move_trajectory_with_camera(seqs_L, seqs_R, camera_poses, width=w, height=h)
            
            if len(rgb_frames) > 0:
                print("Overlaying simulation result on original image...")
                # Use the last frame
                rgb_sim = rgb_frames[-1]
                mask_sim = mask_frames[-1]
                
                # Convert RGB to BGR for OpenCV
                bgr_sim = cv2.cvtColor(rgb_sim, cv2.COLOR_RGB2BGR)
                
                # Create mask (assuming ID > 0 is robot/objects, 0 is background)
                geom_ids = mask_sim[:, :, 0]
                mask_bool = geom_ids > 0
                
                # Overlay
                blended = img.copy()
                blended[mask_bool] = bgr_sim[mask_bool]
                
                cv2.imshow("Simulation Overlay", blended)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        else:
            print("No ArUco marker detected to move to.")

    except Exception as e:
        print(f"[Error] {e}")
        exit(1)