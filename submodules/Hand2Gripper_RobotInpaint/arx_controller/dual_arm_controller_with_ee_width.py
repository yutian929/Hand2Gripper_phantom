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
        
        self.GRIPPER_CLOSE_VAL = 0.0
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
    # 轨迹跟踪函数 (已修改支持 (N,7) 输入)
    # ========================================================
    def move_trajectory(self, target_seqs_L_world, target_seqs_R_world, max_steps_per_point=None, kinematic_only=False):
        """
        连续执行一系列双臂目标点。
        
        Args:
            target_seqs_L_world (np.array): 左臂目标序列，形状 (N, 7) -> [x, y, z, rx, ry, rz, ee_width]
            target_seqs_R_world (np.array): 右臂目标序列，形状 (N, 7) -> [x, y, z, rx, ry, rz, ee_width]
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
            # 检查形状是否包含 ee_width
            if target_world_seqs['L'].shape[1] < 7 or target_world_seqs['R'].shape[1] < 7:
                 print("[Warning] Input shape is less than 7. Assuming default gripper width.")

            if len(target_world_seqs['R']) != num_points:
                print("[Error] Left and Right trajectories must have the same length.")
                return False
        except Exception as e:
            print(f"[Error] Invalid target sequence format: {e}")
            return False

        if max_steps_per_point is None:
            max_steps_per_point = self.MAX_STEPS

        # 2. 预计算：解算所有点的 IK 和 提取 Gripper Width
        # ----------------------------------------------------
        joint_targets_queue = {name: [] for name in self.ARM_NAMES}
        gripper_targets_queue = {name: [] for name in self.ARM_NAMES} # 新增：存储夹爪宽度
        
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
                full_data = target_world_seqs[name][i]
                
                # 提取位姿 (前6个)
                pose = full_data[:6]
                
                # 提取并 Clip 夹爪宽度 (第7个，如果存在)
                if len(full_data) >= 7:
                    raw_width = full_data[6]
                    clipped_width = np.clip(raw_width, self.GRIPPER_CLOSE_VAL, self.GRIPPER_OPEN_VAL)
                else:
                    clipped_width = self.GRIPPER_OPEN_VAL # 默认全开
                
                gripper_targets_queue[name].append(clipped_width)
                
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
        current_gripper_targets = {name: gripper_targets_queue[name][0] for name in self.ARM_NAMES}
        current_pose_targets = {name: target_world_seqs[name][0][:6] for name in self.ARM_NAMES}
        
        # 更新可视化 Marker
        for name in self.ARM_NAMES:
            self._update_mocap_marker(current_pose_targets[name][:3], name)
        
        success_count = 0
        is_finished = False  # 标记轨迹是否执行完毕
        
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
                            # 使用当前帧计算出的夹爪值
                            self.data.qpos[params['qpos_indices_gripper']] = current_gripper_targets[name]
                    
                    # 强制刷新几何体位置
                    mujoco.mj_forward(self.model, self.data)
                else:
                    # [动力学模式] 设置控制信号 (Torque/Position Control)
                    for name in self.ARM_NAMES:
                        params = self.arm_params[name]
                        if params['ctrl_indices_6dof']:
                            self.data.ctrl[params['ctrl_indices_6dof']] = current_joint_targets[name]
                        if params['ctrl_indices_gripper']:
                            # 使用当前帧计算出的夹爪值
                            self.data.ctrl[params['ctrl_indices_gripper']] = current_gripper_targets[name]
                
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
                        print(f"Waypoint {current_idx}: {status} | Max Error: {max_error:.4f}m | Grip: L={current_gripper_targets['L']:.3f}, R={current_gripper_targets['R']:.3f}")
                        
                        if all_reached: success_count += 1
                        
                        current_idx += 1
                        if current_idx >= num_points:
                            print(f"\n[Trajectory] All waypoints completed! Holding final position...")
                            is_finished = True # 标记完成，但不退出循环
                        else:
                            # 更新目标
                            current_joint_targets = {name: joint_targets_queue[name][current_idx] for name in self.ARM_NAMES}
                            current_gripper_targets = {name: gripper_targets_queue[name][current_idx] for name in self.ARM_NAMES}
                            current_pose_targets = {name: target_world_seqs[name][current_idx][:6] for name in self.ARM_NAMES}
                            
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
    def move_trajectory_with_camera(self, target_seqs_L_world, target_seqs_R_world, camera_poses_world, max_steps_per_point=None, kinematic_only=False, cam_name="camera", width=640, height=480):
        """
        连续执行一系列双臂目标点，并同步移动相机进行拍摄。
        输入形状需为 (N, 7)，第7位为夹爪宽度。
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
                pose = full_data[:6]
                
                # Gripper Logic
                if len(full_data) >= 7:
                    raw_width = full_data[6]
                    clipped_width = np.clip(raw_width, self.GRIPPER_CLOSE_VAL, self.GRIPPER_OPEN_VAL)
                else:
                    clipped_width = self.GRIPPER_OPEN_VAL
                gripper_targets_queue[name].append(clipped_width)

                # IK Logic
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
        current_gripper_targets = {name: gripper_targets_queue[name][0] for name in self.ARM_NAMES}
        current_pose_targets = {name: target_world_seqs[name][0][:6] for name in self.ARM_NAMES}
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
                            self.data.qpos[params['qpos_indices_gripper']] = current_gripper_targets[name]
                    mujoco.mj_forward(self.model, self.data)
                else:
                    for name in self.ARM_NAMES:
                        params = self.arm_params[name]
                        if params['ctrl_indices_6dof']:
                            self.data.ctrl[params['ctrl_indices_6dof']] = current_joint_targets[name]
                        if params['ctrl_indices_gripper']:
                            self.data.ctrl[params['ctrl_indices_gripper']] = current_gripper_targets[name]
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
                        current_gripper_targets = {name: gripper_targets_queue[name][current_idx] for name in self.ARM_NAMES}
                        current_pose_targets = {name: target_world_seqs[name][current_idx][:6] for name in self.ARM_NAMES}
                        current_cam_pose = camera_poses_world[current_idx]
                        
                        for name in self.ARM_NAMES:
                            self._update_mocap_marker(current_pose_targets[name][:3], name)
                        step_count = 0
                    else:
                        print("[Trajectory & Cam] All frames captured.")
                        break
                
                # --- D. 渲染 ---
                viewer.sync()
                time.sleep(0.1)
                time_until_next = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next > 0:
                    time.sleep(time_until_next)
                    
        return rgb_list, mask_list

# ========================================================
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "R5", "R5a", "meshes", "dual_arm_scene.xml")
    def generate_smooth_trajectory(keyframes, steps_per_segment=30):
        """
        在关键帧之间进行线性插值，生成平滑轨迹
        """
        full_traj = []
        for i in range(len(keyframes) - 1):
            start_pt = np.array(keyframes[i])
            end_pt = np.array(keyframes[i+1])
            
            # 生成 steps_per_segment 个中间点
            # linspace 返回形状 (steps, 7)
            segment = np.linspace(start_pt, end_pt, steps_per_segment)
            
            # 除去最后一个点，避免与下一段的起点重复（除了最后一段）
            if i < len(keyframes) - 2:
                segment = segment[:-1]
                
            full_traj.append(segment)
            
        return np.vstack(full_traj)

    try:
        # 初始化控制器
        robot = DualArmController(xml_path, max_steps=5000)
        
        # -------------------------------------------------------------------
        # 1. 定义动作关键帧 (Keyframes)
        # 格式: [x, y, z, rx, ry, rz, gripper]
        # rx, ry, rz 使用欧拉角 (rad)
        # gripper: 0.044 (Open), 0.0 (Close)
        # -------------------------------------------------------------------
        
        # 定义常用姿态角度
        euler_L_down = [np.pi, 0, 0]      # 左臂手心向下 
        euler_R_down = [np.pi, 0, 0]      # 右臂手心向下
        
        # 或者沿用你之前的侧向抓取角度
        euler_L_side = [np.pi/2, 0, np.pi/2] 
        euler_R_side = [np.pi/2, 0, -np.pi/2]

        # --- 左臂关键帧序列 ---
        keys_L = [
            # 1. 准备 (高处, 张开)
            [0.4,  0.4, 1.3, *euler_L_side, 0.044],
            # 2. 下探 (低处, 张开)
            [0.4,  0.2, 1.1, *euler_L_side, 0.044],
            # 3. 抓取 (低处, 闭合)
            [0.4,  0.2, 1.1, *euler_L_side, 0.0],
            # 4. 举起 (高处, 闭合)
            [0.4,  0.2, 1.4, *euler_L_side, 0.0],
            # 5. 侧移展示 (向左移, 闭合)
            [0.4,  0.5, 1.4, *euler_L_side, 0.0],
            # 6. 复位 (回到开始, 张开)
            [0.4,  0.4, 1.3, *euler_L_side, 0.044],
        ]
        
        # --- 右臂关键帧序列 ---
        keys_R = [
            # 1. 准备 (高处, 张开) - Y是负的
            [0.4, -0.4, 1.3, *euler_R_side, 0.044],
            # 2. 下探 (低处, 张开)
            [0.4, -0.2, 1.1, *euler_R_side, 0.044],
            # 3. 抓取 (低处, 闭合)
            [0.4, -0.2, 1.1, *euler_R_side, 0.0],
            # 4. 举起 (高处, 闭合)
            [0.4, -0.2, 1.4, *euler_R_side, 0.0],
            # 5. 侧移展示 (向右移, 闭合)
            [0.4, -0.5, 1.4, *euler_R_side, 0.0],
            # 6. 复位 (回到开始, 张开)
            [0.4, -0.4, 1.3, *euler_R_side, 0.044],
        ]

        # -------------------------------------------------------------------
        # 2. 生成平滑轨迹 (插值)
        # -------------------------------------------------------------------
        print("Generating smooth trajectory...")
        # 每一段动作生成 20 帧，动作会显得比较连贯
        traj_L = generate_smooth_trajectory(keys_L, steps_per_segment=20)
        traj_R = generate_smooth_trajectory(keys_R, steps_per_segment=20)
        
        print(f"Total frames generated: {len(traj_L)}")

        # -------------------------------------------------------------------
        # 3. 生成相机轨迹 (简单的环绕或固定视角)
        # -------------------------------------------------------------------
        num_frames = len(traj_L)
        camera_poses = np.zeros((num_frames, 6))
        
        # 让相机稍微动一点点，模拟一种运镜效果
        # 从 (0, 0, 2.0) 慢慢移动
        for i in range(num_frames):
            ratio = i / num_frames
            cam_y = 0.0 - ratio * 0.001
            cam_z = 2.0 - ratio * 0.001
            # 俯视角度
            camera_poses[i] = [0.3, cam_y, cam_z, 0.0, 1.57, 0.0]

        # -------------------------------------------------------------------
        # 4. 执行并录制
        # -------------------------------------------------------------------
        print("Starting Simulation...")
        rgb_frames, mask_frames = robot.move_trajectory_with_camera(
            traj_L, traj_R, camera_poses, 
            kinematic_only=True, cam_name="camera"
        )
        
        # -------------------------------------------------------------------
        # 5. 播放结果
        # -------------------------------------------------------------------
        if len(rgb_frames) > 0:
            print(f"Captured {len(rgb_frames)} frames. Playing back...")
            
            # 随机颜色板用于 Mask 显示
            np.random.seed(42)
            color_palette = np.random.randint(0, 255, (256, 3), dtype=np.uint8)
            color_palette[0] = [0, 0, 0]

            for i, (rgb, mask) in enumerate(zip(rgb_frames, mask_frames)):
                # 转换 RGB -> BGR (OpenCV格式)
                bgr_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                
                # 处理 Mask 颜色
                geom_ids = mask[:, :, 0]
                # 防止 id 越界
                geom_ids[geom_ids >= 256] = 0
                mask_vis = color_palette[geom_ids]
                
                # 在图片上显示帧数
                cv2.putText(bgr_img, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("Dual Arm RGB", bgr_img)
                cv2.imshow("Segmentation Mask", mask_vis)
                
                # 每帧显示 50ms，产生约 20fps 的动画效果
                key = cv2.waitKey(50)
                if key == 27: # ESC 退出
                    break
            
            cv2.destroyAllWindows()

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")
