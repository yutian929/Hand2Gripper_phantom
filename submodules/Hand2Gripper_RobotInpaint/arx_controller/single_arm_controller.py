import mujoco
import mujoco.viewer
import numpy as np
import time
import cv2
import os
import bimanual
from scipy.spatial.transform import Rotation as R

FLANGE_POS_OFFSET = np.array([-0.1, 0.0, -0.16])  # 特定补偿值

class SingleArmController:
    def __init__(self, xml_path, end_effector_site_name="end_effector", base_link_name="base_link", position_threshold=0.02, max_steps=10_000):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML file not found: {xml_path}")
            
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.ee_site_name = end_effector_site_name
        self.base_link_name = base_link_name
        
        
        # 配置参数
        self.GRIPPER_OPEN_VAL = 0.044
        self.POSITION_THRESHOLD = position_threshold
        self.MAX_STEPS = max_steps
        
        # 自动读取参数
        self._init_kinematics_params()
        self.reset()

    def _init_kinematics_params(self):
        try:
            base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.base_link_name)
            self.base_pos_world = self.model.body_pos[base_id].copy()
        except:
            self.base_pos_world = np.zeros(3)

        try:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site_name)
            self.ee_offset_local = self.model.site_pos[site_id].copy()
        except:
            self.ee_offset_local = np.zeros(3)

    def reset(self):
        self.data.qpos[:] = 0
        self.data.ctrl[:] = 0
        mujoco.mj_step(self.model, self.data)

    def _ik_base(self, target_pose_base):
        try:
            return bimanual.inverse_kinematics(target_pose_base)
        except:
            return None
    
    def _get_base_pose_world(self):
            """
            获取基座在世界坐标系下的位姿
            Returns:
                np.array: [x, y, z, rx, ry, rz] (单位: 米, 弧度)
            """
            # 1. 获取 Body ID
            base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.base_link_name)
            if base_id == -1:
                print(f"[Error] Base link '{self.base_link_name}' not found.")
                return np.zeros(6)

            # 2. 获取位置 (xpos 是世界坐标系下的位置)
            pos = self.data.xpos[base_id]

            # 3. 获取姿态 (xmat 是世界坐标系下的旋转矩阵，展平的9元素数组)
            # MuJoCo 的 xmat 是行优先还是列优先并不重要，scipy 能自动处理 reshape(3,3)
            mat = self.data.xmat[base_id].reshape(3, 3)
            
            # 4. 转换为欧拉角 (保持和你其他函数一致的 XYZ 顺序)
            r = R.from_matrix(mat)
            euler = r.as_euler('xyz', degrees=False)

            return np.concatenate([pos, euler])

    def _get_ee_pose_world(self):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site_name)
        pos = self.data.site_xpos[site_id]
        mat = self.data.site_xmat[site_id].reshape(3, 3)
        r = R.from_matrix(mat)
        euler = r.as_euler('xyz', degrees=False)
        return np.concatenate([pos, euler])

    def _pose_tf_world2base(self, target_pose_world):
        target_pos_world = target_pose_world[:3]
        target_euler_world = target_pose_world[3:]
        
        r = R.from_euler('xyz', target_euler_world, degrees=False)
        rot_matrix = r.as_matrix()
        offset_world = rot_matrix @ self.ee_offset_local
        
        flange_pos_world = target_pos_world - offset_world
        flange_pos_base = flange_pos_world - self.base_pos_world
        
        # 特定补偿 (根据你的上一版代码保留)
        flange_pos_base += FLANGE_POS_OFFSET
        
        return np.concatenate([flange_pos_base, target_euler_world])

    def _update_mocap_marker(self, pos):
        try:
            target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_marker")
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
        Args:
            cam_name (str): XML中定义的相机名称
            pose_world (np.array): [x, y, z, rx, ry, rz]
        """
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam_id == -1:
            print(f"[Warning] Camera '{cam_name}' not found.")
            return

        # 1. 设置位置
        self.model.cam_pos[cam_id] = pose_world[:3]

        # 2. 设置姿态 (MuJoCo 使用四元数 [w, x, y, z])
        r = R.from_euler('xyz', pose_world[3:], degrees=False)
        quat_scipy = r.as_quat() # 返回 [x, y, z, w]
        # 转换为 MuJoCo 顺序 [w, x, y, z]
        quat_mujoco = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])
        
        self.model.cam_quat[cam_id] = quat_mujoco

    # ========================================================
    # 【新增】轨迹跟踪函数
    # ========================================================
    def move_trajectory(self, target_world_seq, max_steps_per_point=None, kinematic_only=False):
        """
        连续执行一系列目标点
        
        Args:
            target_world_seq (np.array): 形状 (N, 6) 的数组，包含 N 个目标位姿
            max_steps_per_point (int): 每个点的最大等待步数
            kinematic_only (bool): 是否仅进行运动学控制 (无物理/碰撞/力矩)
        """
        print(f"\n[Trajectory] Received {len(target_world_seq)} waypoints. Kinematic Only: {kinematic_only}")
        if max_steps_per_point is None:
            max_steps_per_point = self.MAX_STEPS
        
        # 1. 预计算：先解算所有点的 IK，确保路径可行
        # ----------------------------------------------------
        joint_targets_queue = []
        
        # 初始 fallback 为当前关节角度 (通常是0或上一帧的位置)
        last_valid_joints = self.data.qpos[:6].copy()
        
        # 记录 IK 失败的索引
        failed_indices = []

        for i, pose in enumerate(target_world_seq):
            pose_base = self._pose_tf_world2base(pose)
            joints = self._ik_base(pose_base)
            
            if joints is None:
                # print(f"[Warning] IK Failed at waypoint {i}. Using last valid joints.")
                failed_indices.append(i)
                # 使用上一个有效值 (如果是第0个，则是初始状态)
                joints = last_valid_joints.copy()
            else:
                # 更新有效值
                last_valid_joints = joints.copy()
                
            joint_targets_queue.append(joints)
            
        if failed_indices:
            print(f"[Warning] IK Failed for {len(failed_indices)} waypoints (used fallback). Indices: {failed_indices}")

        print("[Trajectory] All IK solved (with fallbacks). Starting execution...")

        # 2. 执行循环：只启动一次 Viewer
        # ----------------------------------------------------
        current_idx = 0
        total_points = len(joint_targets_queue)
        
        # 初始化第一个目标
        current_joint_target = joint_targets_queue[0]
        current_pose_target_world = target_world_seq[0]
        self._update_mocap_marker(current_pose_target_world[:3])
        
        success_count = 0
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            step_count_for_current = 0
            
            while viewer.is_running():
                step_start = time.time()
                
                # --- A. 下发当前目标的控制 ---
                if kinematic_only:
                    # [运动学模式] 直接修改关节位置，无视物理
                    self.data.qpos[:6] = current_joint_target
                    if self.model.nu >= 8:
                        # 假设 gripper 对应 qpos 的 6, 7 (基于当前 XML 结构)
                        self.data.qpos[6] = self.GRIPPER_OPEN_VAL
                        self.data.qpos[7] = self.GRIPPER_OPEN_VAL
                    
                    # 强制更新几何体位置 (不进行物理步进)
                    mujoco.mj_forward(self.model, self.data)
                else:
                    # [动力学模式] 设置控制信号，由物理引擎驱动
                    self.data.ctrl[:6] = current_joint_target
                    if self.model.nu >= 8:
                        self.data.ctrl[6:] = self.GRIPPER_OPEN_VAL

                # --- B. 物理步进 ---
                if not kinematic_only:
                    mujoco.mj_step(self.model, self.data)
                
                step_count_for_current += 1
                
                # --- C. 误差检测 ---
                current_ee_pose = self._get_ee_pose_world()
                # 只计算位置误差 (欧氏距离)
                error = np.linalg.norm(current_ee_pose[:3] - current_pose_target_world[:3])
                
                # 状态检查
                # 如果是 kinematic_only，IK 准确的话 error 应该接近 0，reached 立即为 True
                reached = error < self.POSITION_THRESHOLD
                
                if current_idx == 0:
                    timeout = step_count_for_current >= self.MAX_STEPS
                else:
                    timeout = step_count_for_current >= max_steps_per_point
                
                # --- D. 切换目标逻辑 ---
                if reached or timeout:
                    # 如果是 kinematic_only 且 dense trajectory，这里会每一帧切换一个点，形成动画播放效果
                    status = "✅ Reached" if reached else "⚠️ Timeout (Skipping)"
                    print(f"Waypoint {current_idx}: {status} | Error: {error:.4f}m")
                    
                    if reached: success_count += 1
                    
                    # 切换到下一个点
                    current_idx += 1
                    if current_idx >= total_points:
                        print("\n[Trajectory] All waypoints completed!")
                        break # 结束整个任务
                    
                    # 更新目标
                    current_joint_target = joint_targets_queue[current_idx]
                    current_pose_target_world = target_world_seq[current_idx]
                    self._update_mocap_marker(current_pose_target_world[:3])
                    
                    # 重置计数器
                    step_count_for_current = 0
                
                # --- E. 渲染 ---
                viewer.sync()
                
                time_until_next = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next > 0:
                    time.sleep(time_until_next)

        return success_count == total_points

    # ========================================================
    # 【修改】辅助函数：设置相机位姿（自动处理坐标系转换）
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

        # 1. 设置位置 (位置是点，不受旋转坐标系定义影响，直接赋值)
        self.model.cam_pos[cam_id] = pose_world[:3]

        # 2. 处理姿态旋转
        # (1) 将输入的 Euler/Pose 转换为旋转矩阵 R_input (World -> Link)
        r_input = R.from_euler('xyz', pose_world[3:], degrees=False)
        mat_input = r_input.as_matrix()

        # (2) 定义转换矩阵 R_offset (Link -> Optical)
        # Columns correspond to Optical axes expressed in Link frame:
        # Col 0 (Opt X/Right) = Link -Y
        # Col 1 (Opt Y/Up)    = Link +Z
        # Col 2 (Opt Z/Back)  = Link -X
        mat_link_to_optical = np.array([
            [ 0,  0, -1],
            [-1,  0,  0],
            [ 0,  1,  0]
        ])

        # (3) 计算最终传给 MuJoCo 的旋转矩阵 (World -> Optical)
        # R_final = R_input * R_offset
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
    def move_trajectory_with_camera(self, target_seqs_world, camera_poses_world, max_steps_per_point=None, kinematic_only=False, cam_name="camera", width=640, height=480):
        """
        连续执行一系列目标点，并同步移动相机进行拍摄。
        
        Args:
            target_seqs_world (np.array): (N, 6) 机器人目标位姿
            camera_poses_world (np.array): (N, 6) 相机目标位姿 [x, y, z, rx, ry, rz]
            max_steps_per_point (int): 超时步数
            kinematic_only (bool): 仅运动学
            cam_name (str): XML中的相机名称
            width, height (int): 渲染分辨率

        Returns:
            rgb_list (list of np.array): 捕获的RGB图像列表
            mask_list (list of np.array): 捕获的Segmentation Mask列表
        """
        if len(target_seqs_world) != len(camera_poses_world):
            raise ValueError("Target sequence and Camera sequence must have the same length.")

        print(f"\n[Trajectory & Cam] Processing {len(target_seqs_world)} frames...")
        if max_steps_per_point is None:
            max_steps_per_point = self.MAX_STEPS

        # 1. 初始化 Renderer
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        
        # 2. IK 预计算 (同 move_trajectory)
        joint_targets_queue = []
        last_valid_joints = self.data.qpos[:6].copy()
        
        for i, pose in enumerate(target_seqs_world):
            pose_base = self._pose_tf_world2base(pose)
            joints = self._ik_base(pose_base)
            if joints is None:
                joints = last_valid_joints.copy()
            else:
                last_valid_joints = joints.copy()
            joint_targets_queue.append(joints)

        # 3. 执行并拍摄
        rgb_list = []
        mask_list = []
        
        current_idx = 0
        total_points = len(joint_targets_queue)
        
        # 为了看到过程，我们依然可以启动 viewer，但为了采集数据的准确性，
        # 这里的 viewer 仅作为“监视器”，实际图像数据来自 renderer。
        # 如果不需要看动画，可以去掉 viewer 上下文，直接写 while 循环。
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            
            # 初始化第一个点
            current_joint_target = joint_targets_queue[0]
            current_pose_target = target_seqs_world[0]
            current_cam_pose = camera_poses_world[0]
            
            self._update_mocap_marker(current_pose_target[:3])
            step_count = 0

            while viewer.is_running() and current_idx < total_points:
                step_start = time.time()

                # --- Control ---
                if kinematic_only:
                    self.data.qpos[:6] = current_joint_target
                    if self.model.nu >= 8:
                        self.data.qpos[6:8] = self.GRIPPER_OPEN_VAL
                    mujoco.mj_forward(self.model, self.data)
                else:
                    self.data.ctrl[:6] = current_joint_target
                    if self.model.nu >= 8:
                        self.data.ctrl[6:] = self.GRIPPER_OPEN_VAL
                    mujoco.mj_step(self.model, self.data)

                step_count += 1
                
                # --- Check Status ---
                current_ee_pose = self._get_ee_pose_world()
                error = np.linalg.norm(current_ee_pose[:3] - current_pose_target[:3])
                
                reached = error < self.POSITION_THRESHOLD
                timeout = step_count >= max_steps_per_point
                
                # --- Capture Logic (当到达当前点 或 超时时进行拍摄) ---
                if reached or timeout:
                    # 1. 强制更新相机位置 (对应当前帧的相机位姿)
                    # 注意：如果机器人还没到，相机是否应该提前动？
                    # 通常在收集数据时，我们希望“机器人到位”且“相机到位”是同一时刻。
                    # 这里我们在机器人到位的那一瞬间，设置相机位姿并截图。
                    self._set_camera_pose(cam_name, current_cam_pose)
                    
                    # 必须调用 mj_forward 确保相机参数在内部更新生效
                    mujoco.mj_forward(self.model, self.data)
                    
                    # 2. 渲染 RGB
                    renderer.update_scene(self.data, camera=cam_name)
                    rgb = renderer.render() # Returns (H, W, 3) uint8
                    rgb_list.append(rgb.copy())
                    
                    # 3. 渲染 Segmentation Mask
                    renderer.enable_segmentation_rendering()
                    renderer.update_scene(self.data, camera=cam_name) # 通常不需要再次 update，但为了保险
                    mask = renderer.render() # Returns (H, W, 2) int32 usually
                    # mask[:,:,0] 是 geom ID, mask[:,:,1] 是 body ID (取决于设置)
                    mask_list.append(mask.copy())
                    renderer.disable_segmentation_rendering()
    
                    # --- Move to next waypoint ---
                    current_idx += 1
                    if current_idx < total_points:
                        current_joint_target = joint_targets_queue[current_idx]
                        current_pose_target = target_seqs_world[current_idx]
                        current_cam_pose = camera_poses_world[current_idx]
                        self._update_mocap_marker(current_pose_target[:3])
                        step_count = 0
                    else:
                        print("[Trajectory & Cam] All frames captured.")
                        break

                # --- Viewer Sync ---
                viewer.sync()
                # 稍微加一点延时让Viewer看起来顺滑，但在纯数据采集中可以去掉
                time_until_next = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next > 0:
                    time.sleep(time_until_next)

        return rgb_list, mask_list

# ========================================================
# 使用示例
# ========================================================

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "R5/R5a/meshes/single_arm_scene.xml")
    
    try:
        robot = SingleArmController(xml_path)
        
        target_seq_world = np.array([
            [0.3, 0.0, 1.3, 0.0, 0.0, 0.0],
            [0.3, 0.0, 1.3, 0.0, 0.0, 0.0],
            [0.3, 0.0, 1.3, 0.0, 0.0, 0.0]
        ])
        
        # 示例：开启 kinematic_only=True
        robot.move_trajectory(target_seq_world, kinematic_only=True)

        camera_poses_world = np.array([
            [0.0, 0.0, 2.0, 0.0, 0.78, 0.0],
            [0.0, 0.0, 2.0, 0.0, 1.57, 0.0],
            [0.0, 0.0, 2.0, 0.0, 2.34, 0.0]
        ])
        
        rgb_frames, mask_frames = robot.move_trajectory_with_camera(
            target_seq_world,
            camera_poses_world,
            kinematic_only=True,
            cam_name="camera",
            width=640,
            height=480
        )
        # === 可视化部分 ===
        print(f"Captured {len(rgb_frames)} frames.")
        
        # 预先生成一个随机颜色板 (假设最大 ID 不超过 100)
        # 形状: (100, 3)，每个 ID 对应一个 [B, G, R] 颜色
        np.random.seed(42) # 固定种子保证颜色一致
        color_palette = np.random.randint(0, 255, (100, 3), dtype=np.uint8)
        color_palette[0] = [0, 0, 0] # 强制让 ID 0 (背景) 为黑色

        for i, (rgb, mask) in enumerate(zip(rgb_frames, mask_frames)):
            # 1. 处理 RGB (MuJoCo 是 RGB, OpenCV 需要 BGR)
            bgr_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            
            # 2. 处理 Mask
            # mask[:, :, 0] 通常是 Geom ID (几何体粒度)
            # mask[:, :, 1] 通常是 Body ID (物体/连杆粒度)
            geom_ids = mask[:, :, 0]
            
            # 【方法 A】彩色可视化 (推荐)
            # 使用 numpy 的高级索引，直接把 ID 映射成颜色
            # 假如 geom_ids 里有值 5，它会去 color_palette[5] 取颜色
            mask_vis = color_palette[geom_ids] 

            # 3. 显示
            cv2.imshow(f"RGB Frame {i}", bgr_img)
            cv2.imshow(f"Seg Mask {i} (Geom ID)", mask_vis)
            
            # 这里的 waitKey(0) 会暂停，按任意键看下一帧
            # 改成 waitKey(1) 可以自动播放
        cv2.waitKey(0) 
                
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")