import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import cv2
import bimanual
from scipy.spatial.transform import Rotation as R

# 特定补偿值 (与单臂代码一致)
FLANGE_POS_OFFSET = np.array([-0.1, 0.0, -0.16])

class DualArmController:
    def __init__(self, xml_path, arm_names=None, end_effector_site_names=None, base_link_names=None, position_threshold=0.02, max_steps=10_000):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML file not found: {xml_path}")
            
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.ARM_NAMES = ['L', 'R'] if not arm_names else arm_names
        self.EE_SITE_NAMES = {name: f"end_effector_{name}" for name in self.ARM_NAMES} if end_effector_site_names is None else end_effector_site_names
        self.BASE_LINK_NAMES = {name: f"base_link_{name}" for name in self.ARM_NAMES} if base_link_names is None else base_link_names
        
        self.GRIPPER_OPEN_VAL = 0.044
        self.POSITION_THRESHOLD = position_threshold
        self.MAX_STEPS = max_steps
        
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
            
            # 3. Actuator Indices (CTRL) - 用于动力学模式
            # 假设执行器命名为 act_j1_L, act_j2_L ...
            act_names_6dof = [f"act_j{i}_{name}" for i in range(1, 7)]
            act_ids_6dof = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, a_name) for a_name in act_names_6dof]
            params['ctrl_indices_6dof'] = [a_id for a_id in act_ids_6dof if a_id != -1]
            
            act_names_gripper = [f"act_j7_{name}", f"act_j8_{name}"]
            act_ids_gripper = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, a_name) for a_name in act_names_gripper]
            params['ctrl_indices_gripper'] = [a_id for a_id in act_ids_gripper if a_id != -1]

            self.arm_params[name] = params
            
            # Debug info
            print(f"[{name}] QPOS Indices: {params['qpos_indices_6dof']}, CTRL Indices: {params['ctrl_indices_6dof']}")

    def reset(self):
        self.data.qpos[:] = 0
        self.data.ctrl[:] = 0
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
        params = self.arm_params[arm_name]
        target_pos_world = target_pose_world[:3]
        target_euler_world = target_pose_world[3:]
        
        r = R.from_euler('xyz', target_euler_world, degrees=False)
        rot_matrix = r.as_matrix()
        
        offset_world = rot_matrix @ params['ee_offset_local']
        flange_pos_world = target_pos_world - offset_world
        flange_pos_base = flange_pos_world - params['base_pos_world']
        
        flange_pos_base += FLANGE_POS_OFFSET
        
        return np.concatenate([flange_pos_base, target_euler_world])

    def _update_mocap_marker(self, pos, arm_name):
        marker_name = f"target_marker_{arm_name}"
        try:
            target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, marker_name)
            if target_body_id != -1:
                mocap_id = self.model.body_mocapid[target_body_id]
                self.data.mocap_pos[mocap_id] = pos
        except: pass

    # ========================================================
    # 轨迹跟踪函数
    # ========================================================
    def move_trajectory(self, target_seqs_L_world, target_seqs_R_world, max_steps_per_point=None, kinematic_only=False):
        """
        连续执行一系列双臂目标点。
        
        Args:
            target_seqs_L_world (np.array): 左臂目标序列，形状 (N, 6)
            target_seqs_R_world (np.array): 右臂目标序列，形状 (N, 6)
            max_steps_per_point (int, optional): 每个目标点的最大步数。默认使用 self.MAX_STEPS。
            kinematic_only (bool): 是否仅使用运动学模式（直接设置关节位置），忽略动力学和碰撞。
            
        Returns:
            success (bool): 是否成功完成所有目标点。
        """
        if not self.ARM_NAMES: return True
        
        print(f"\n[Trajectory] Dual Arm. Kinematic Only: {kinematic_only}")
        
        # 1. 验证输入数据
        # ----------------------------------------------------
        target_world_seqs = {
            'L': target_seqs_L_world,
            'R': target_seqs_R_world
        }

        try:
            num_points = len(target_world_seqs['L'])
            if len(target_world_seqs['R']) != num_points:
                print("[Error] Left and Right trajectories must have the same length.")
                return False
        except Exception as e:
            print(f"[Error] Invalid target sequence format: {e}")
            return False

        if max_steps_per_point is None:
            max_steps_per_point = self.MAX_STEPS

        # 2. 预计算：解算所有点的 IK
        # ----------------------------------------------------
        joint_targets_queue = {name: [] for name in self.ARM_NAMES}
        
        # 初始化 Fallback 值 (防止 IK 无解时飞车)
        last_valid_joints = {}
        for name in self.ARM_NAMES:
            qpos_indices = self.arm_params[name]['qpos_indices_6dof']
            if qpos_indices:
                last_valid_joints[name] = self.data.qpos[qpos_indices].copy()
            else:
                last_valid_joints[name] = np.zeros(6)

        failed_counts = {name: 0 for name in self.ARM_NAMES}

        # 遍历每一个时间步
        for i in range(num_points):
            for name in self.ARM_NAMES:
                pose = target_world_seqs[name][i]
                
                # 坐标变换: World -> Base Frame
                pose_base = self._pose_tf_world2base(pose, name)
                
                # 解算 IK
                joints = self._ik_base(pose_base)
                
                if joints is None:
                    failed_counts[name] += 1
                    # IK 失败，沿用上一个有效姿态
                    joints = last_valid_joints[name].copy()
                else:
                    last_valid_joints[name] = joints.copy()
                        
                joint_targets_queue[name].append(joints)

        for name in self.ARM_NAMES:
            if failed_counts[name] > 0:
                print(f"[Warning] Arm {name}: IK Failed for {failed_counts[name]} waypoints (used fallback).")

        print("[Trajectory] All IK solved. Starting execution...")

        # 3. 执行循环
        # ----------------------------------------------------
        current_idx = 0
        
        # 初始化第一个目标
        current_joint_targets = {name: joint_targets_queue[name][0] for name in self.ARM_NAMES}
        current_pose_targets = {name: target_world_seqs[name][0] for name in self.ARM_NAMES}
        
        # 更新可视化 Marker
        for name in self.ARM_NAMES:
            self._update_mocap_marker(current_pose_targets[name][:3], name)
        
        success_count = 0
        is_finished = False  # 标记轨迹是否执行完毕
        debug = 0
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            step_count_for_current = 0
            
            while viewer.is_running():
                step_start = time.time()
                
                # --- A. 下发控制 (Kinematic 或 Dynamic) ---
                if kinematic_only:
                    # [运动学模式] 直接修改关节位置 (Teleport)
                    for name in self.ARM_NAMES:
                        params = self.arm_params[name]
                        if params['qpos_indices_6dof']:
                            self.data.qpos[params['qpos_indices_6dof']] = current_joint_targets[name]
                        if params['qpos_indices_gripper']:
                            self.data.qpos[params['qpos_indices_gripper']] = self.GRIPPER_OPEN_VAL
                    
                    # 强制刷新几何体位置
                    mujoco.mj_forward(self.model, self.data)
                else:
                    # [动力学模式] 设置控制信号 (Torque/Position Control)
                    for name in self.ARM_NAMES:
                        params = self.arm_params[name]
                        if params['ctrl_indices_6dof']:
                            self.data.ctrl[params['ctrl_indices_6dof']] = current_joint_targets[name]
                        if params['ctrl_indices_gripper']:
                            self.data.ctrl[params['ctrl_indices_gripper']] = self.GRIPPER_OPEN_VAL
                
                # --- B. 物理步进 ---
                if not kinematic_only:
                    mujoco.mj_step(self.model, self.data)

                # 只有未完成时才进行误差检测和目标切换
                if not is_finished:
                    step_count_for_current += 1
                    
                    # --- C. 误差检测 ---
                    all_reached = True
                    max_error = 0.0
                    
                    for name in self.ARM_NAMES:
                        curr = self._get_ee_pose_world(name)[:3]
                        targ = current_pose_targets[name][:3]
                        err = np.linalg.norm(curr - targ)
                        max_error = max(max_error, err)
                        
                        if err > self.POSITION_THRESHOLD:
                            all_reached = False
                    
                    # --- D. 切换目标逻辑 ---
                    # 第一个点通常距离较远，给更多时间 (MAX_STEPS)
                    if current_idx == 0:
                        timeout = step_count_for_current >= self.MAX_STEPS
                    else:
                        timeout = step_count_for_current >= max_steps_per_point

                    # 如果到达或超时，切换下一个点
                    if all_reached or timeout:
                        status = "✅ Reached" if all_reached else "⚠️ Timeout"
                        # 可选：打印每个点的状态
                        print(f"Waypoint {current_idx}: {status} | Max Error: {max_error:.4f}m")
                        
                        if all_reached: success_count += 1
                        
                        current_idx += 1
                        if current_idx >= num_points:
                            print(f"\n[Trajectory] All waypoints completed! Holding final position...")
                            is_finished = True # 标记完成，但不退出循环
                        else:
                            # 更新目标
                            current_joint_targets = {name: joint_targets_queue[name][current_idx] for name in self.ARM_NAMES}
                            current_pose_targets = {name: target_world_seqs[name][current_idx] for name in self.ARM_NAMES}
                            
                            for name in self.ARM_NAMES:
                                self._update_mocap_marker(current_pose_targets[name][:3], name)
                            
                            step_count_for_current = 0

                # --- E. 渲染 ---
                viewer.sync()

                # --- F. 帧率控制 ---
                time_until_next = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next > 0:
                    time.sleep(time_until_next)
                
        return success_count == num_points
    
    # ========================================================
    # 【新增】辅助函数：设置相机位姿
    # ========================================================
    def _set_camera_pose(self, cam_name, pose_world):
        """
        设置指定相机的位姿。
        
        输入 pose_world 假定为 Robotics Link Frame:
          X = Forward (前方)
          Y = Left    (左方)
          Z = Up      (上方)
          
        MuJoCo 内部相机坐标系为 Optical Frame:
          -Z = Forward
           X = Right
           Y = Up
           
        该函数会自动应用变换矩阵，使得输入直观的位姿能正确映射到 MuJoCo 相机。
        """
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam_id == -1:
            print(f"[Warning] Camera '{cam_name}' not found.")
            return

        # 1. 设置位置
        self.model.cam_pos[cam_id] = pose_world[:3]

        # 2. 处理姿态旋转
        # (1) 将输入的 Euler/Pose 转换为旋转矩阵 R_input (World -> Link)
        r_input = R.from_euler('xyz', pose_world[3:], degrees=False)
        mat_input = r_input.as_matrix()

        # (2) 定义转换矩阵 R_offset (Link -> Optical)
        mat_link_to_optical = np.array([
            [ 0,  0, -1],
            [-1,  0,  0],
            [ 0,  1,  0]
        ])

        # (3) 计算最终传给 MuJoCo 的旋转矩阵 (World -> Optical)
        mat_final = mat_input @ mat_link_to_optical

        # (4) 转换回 MuJoCo 需要的四元数 [w, x, y, z]
        r_final = R.from_matrix(mat_final)
        quat_scipy = r_final.as_quat() # [x, y, z, w]
        
        # 转换为 MuJoCo 顺序 [w, x, y, z]
        quat_mujoco = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])
        
        self.model.cam_quat[cam_id] = quat_mujoco

    # ========================================================
    # 【实现】轨迹跟踪 + 相机拍摄
    # ========================================================
    def move_trajectory_with_camera(self, target_seqs_L_world, target_seqs_R_world, camera_poses_world, max_steps_per_point=None, kinematic_only=False, cam_name="camera", width=640, height=480):
        """
        连续执行一系列双臂目标点，并同步移动相机进行拍摄。
        
        Args:
            target_seqs_L_world (np.array): 左臂目标序列，形状 (N, 6)
            target_seqs_R_world (np.array): 右臂目标序列，形状 (N, 6)
            camera_poses_world (np.array): 相机位姿序列，形状 (N, 6)
            max_steps_per_point (int, optional): 每个目标点的最大步数。默认使用 self.MAX_STEPS。
            kinematic_only (bool): 是否仅使用运动学模式（直接设置关节位置），忽略动力学和碰撞。
            cam_name (str): XML中的相机名称
            width, height (int): 渲染分辨率
            
        Returns:
            captured_frames (list): 录制的图像列表。
            captured_masks (list): 录制的掩码列表。
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
        if max_steps_per_point is None:
            max_steps_per_point = self.MAX_STEPS

        # 2. 初始化 Renderer
        renderer = mujoco.Renderer(self.model, height=height, width=width)

        # 3. 预计算：解算所有点的 IK
        joint_targets_queue = {name: [] for name in self.ARM_NAMES}
        last_valid_joints = {}
        for name in self.ARM_NAMES:
            qpos_indices = self.arm_params[name]['qpos_indices_6dof']
            if qpos_indices:
                last_valid_joints[name] = self.data.qpos[qpos_indices].copy()
            else:
                last_valid_joints[name] = np.zeros(6)

        for i in range(num_points):
            for name in self.ARM_NAMES:
                pose = target_world_seqs[name][i]
                pose_base = self._pose_tf_world2base(pose, name)
                joints = self._ik_base(pose_base)
                
                if joints is None:
                    joints = last_valid_joints[name].copy()
                else:
                    last_valid_joints[name] = joints.copy()
                joint_targets_queue[name].append(joints)

        # 4. 执行循环
        rgb_list = []
        mask_list = []
        current_idx = 0
        
        # 初始化第一个目标
        current_joint_targets = {name: joint_targets_queue[name][0] for name in self.ARM_NAMES}
        current_pose_targets = {name: target_world_seqs[name][0] for name in self.ARM_NAMES}
        current_cam_pose = camera_poses_world[0]
        
        for name in self.ARM_NAMES:
            self._update_mocap_marker(current_pose_targets[name][:3], name)
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            step_count = 0
            
            while viewer.is_running() and current_idx < num_points:
                step_start = time.time()
                
                # --- A. 下发控制 ---
                if kinematic_only:
                    for name in self.ARM_NAMES:
                        params = self.arm_params[name]
                        if params['qpos_indices_6dof']:
                            self.data.qpos[params['qpos_indices_6dof']] = current_joint_targets[name]
                        if params['qpos_indices_gripper']:
                            self.data.qpos[params['qpos_indices_gripper']] = self.GRIPPER_OPEN_VAL
                    mujoco.mj_forward(self.model, self.data)
                else:
                    for name in self.ARM_NAMES:
                        params = self.arm_params[name]
                        if params['ctrl_indices_6dof']:
                            self.data.ctrl[params['ctrl_indices_6dof']] = current_joint_targets[name]
                        if params['ctrl_indices_gripper']:
                            self.data.ctrl[params['ctrl_indices_gripper']] = self.GRIPPER_OPEN_VAL
                    mujoco.mj_step(self.model, self.data)

                if not kinematic_only:
                    mujoco.mj_step(self.model, self.data)
                
                step_count += 1
                
                # --- B. 误差检测 ---
                all_reached = True
                for name in self.ARM_NAMES:
                    curr = self._get_ee_pose_world(name)[:3]
                    targ = current_pose_targets[name][:3]
                    if np.linalg.norm(curr - targ) > self.POSITION_THRESHOLD:
                        all_reached = False
                        break
                
                timeout = step_count >= max_steps_per_point
                
                # --- C. 拍摄逻辑 ---
                if all_reached or timeout:
                    # 1. 更新相机
                    self._set_camera_pose(cam_name, current_cam_pose)
                    mujoco.mj_forward(self.model, self.data)
                    
                    # 2. 渲染 RGB
                    renderer.update_scene(self.data, camera=cam_name)
                    rgb_list.append(renderer.render().copy())
                    
                    # 3. 渲染 Mask
                    renderer.enable_segmentation_rendering()
                    renderer.update_scene(self.data, camera=cam_name)
                    mask_list.append(renderer.render().copy())
                    renderer.disable_segmentation_rendering()
                    
                    # 4. 切换下一个点
                    current_idx += 1
                    if current_idx < num_points:
                        current_joint_targets = {name: joint_targets_queue[name][current_idx] for name in self.ARM_NAMES}
                        current_pose_targets = {name: target_world_seqs[name][current_idx] for name in self.ARM_NAMES}
                        current_cam_pose = camera_poses_world[current_idx]
                        
                        for name in self.ARM_NAMES:
                            self._update_mocap_marker(current_pose_targets[name][:3], name)
                        step_count = 0
                    else:
                        print("[Trajectory & Cam] All frames captured.")
                        break
                
                # --- D. 渲染 ---
                viewer.sync()
                
                time_until_next = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next > 0:
                    time.sleep(time_until_next)
                    
        return rgb_list, mask_list

# ========================================================
# 使用示例
# ========================================================
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "R5", "R5a", "meshes", "dual_arm_scene.xml")
    
    try:
        robot = DualArmController(xml_path, max_steps=5000)
        
        # 1. 定义轨迹
        target_seq_L = np.array([
            [0.3, 0.5, 1.3, np.pi/2, 0.0, np.pi/2], 
            [0.3, 0.3, 1.3, 0.0, 0.0, 0.0],  
        ])
        
        target_seq_R = np.array([
            [0.3, -0.5, 1.3, np.pi/2, 0.0, -np.pi/2], 
            [0.3, -0.3, 1.3, 0.0, 0.0, 0.0], 
        ])
        
        # 2. 定义相机轨迹 (俯视)
        camera_poses = np.zeros((len(target_seq_L), 6))
        for i in range(len(target_seq_L)):
             camera_poses[i] = [0.0, 0.0, 2.0, 0.0, 1.57, 0.0]

        # 3. 执行并录制
        print("Testing Camera Recording...")
        rgb_frames, mask_frames = robot.move_trajectory_with_camera(
            target_seq_L, target_seq_R, camera_poses, 
            kinematic_only=True, cam_name="camera"
        )
        
        # 4. 可视化结果
        if len(rgb_frames) > 0:
            print(f"Captured {len(rgb_frames)} frames.")
            
            # 随机颜色板
            np.random.seed(42)
            color_palette = np.random.randint(0, 255, (100, 3), dtype=np.uint8)
            color_palette[0] = [0, 0, 0]

            for i, (rgb, mask) in enumerate(zip(rgb_frames, mask_frames)):
                bgr_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                
                geom_ids = mask[:, :, 0]
                mask_vis = color_palette[geom_ids]
                
                cv2.imshow(f"RGB Frame {i}", bgr_img)
                cv2.imshow(f"Seg Mask {i}", mask_vis)
                
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")
