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
        self.GRIPPER_CLOSE_VAL = 0.0
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
            # ... (保持不变) ...
            base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.base_link_name)
            if base_id == -1:
                return np.zeros(6)
            pos = self.data.xpos[base_id]
            mat = self.data.xmat[base_id].reshape(3, 3)
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
        
        flange_pos_base += FLANGE_POS_OFFSET
        
        return np.concatenate([flange_pos_base, target_euler_world])

    def _update_mocap_marker(self, pos):
        try:
            target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_marker")
            if target_body_id != -1:
                mocap_id = self.model.body_mocapid[target_body_id]
                self.data.mocap_pos[mocap_id] = pos
        except: pass
    
    def _set_camera_pose(self, cam_name, pose_world):
        # ... (保持不变，已包含坐标系转换) ...
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam_id == -1:
            print(f"[Warning] Camera '{cam_name}' not found.")
            return

        self.model.cam_pos[cam_id] = pose_world[:3]

        r_input = R.from_euler('xyz', pose_world[3:], degrees=False)
        mat_input = r_input.as_matrix()

        mat_link_to_optical = np.array([
            [ 0,  0, -1],
            [-1,  0,  0],
            [ 0,  1,  0]
        ])

        mat_final = mat_input @ mat_link_to_optical
        r_final = R.from_matrix(mat_final)
        quat_scipy = r_final.as_quat() 
        quat_mujoco = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])
        
        self.model.cam_quat[cam_id] = quat_mujoco

    # ========================================================
    # 【修改】轨迹跟踪函数：支持 (N, 7) 输入
    # ========================================================
    def move_trajectory(self, target_world_seq, max_steps_per_point=None, kinematic_only=False):
        """
        连续执行一系列目标点
        target_world_seq: (N, 7) -> [x, y, z, rx, ry, rz, ee_width]
        """
        print(f"\n[Trajectory] Received {len(target_world_seq)} waypoints. Kinematic Only: {kinematic_only}")
        if max_steps_per_point is None:
            max_steps_per_point = self.MAX_STEPS
        
        # 1. 预计算：IK 和 Gripper
        joint_targets_queue = []
        gripper_targets_queue = [] # 新增
        
        last_valid_joints = self.data.qpos[:6].copy()
        failed_indices = []

        for i, full_data in enumerate(target_world_seq):
            # 1. 提取 Gripper
            if len(full_data) >= 7:
                raw_width = full_data[6]
                clipped_width = np.clip(raw_width, self.GRIPPER_CLOSE_VAL, self.GRIPPER_OPEN_VAL)
            else:
                clipped_width = self.GRIPPER_OPEN_VAL
            gripper_targets_queue.append(clipped_width)

            # 2. 提取 Pose 并计算 IK
            pose = full_data[:6]
            pose_base = self._pose_tf_world2base(pose)
            joints = self._ik_base(pose_base)
            
            if joints is None:
                failed_indices.append(i)
                joints = last_valid_joints.copy()
            else:
                last_valid_joints = joints.copy()
                
            joint_targets_queue.append(joints)
            
        if failed_indices:
            print(f"[Warning] IK Failed for {len(failed_indices)} waypoints.")

        print("[Trajectory] All IK solved. Starting execution...")

        # 2. 执行循环
        current_idx = 0
        total_points = len(joint_targets_queue)
        
        current_joint_target = joint_targets_queue[0]
        current_gripper_target = gripper_targets_queue[0] # 新增
        current_pose_target_world = target_world_seq[0][:6]
        
        self._update_mocap_marker(current_pose_target_world[:3])
        success_count = 0
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            step_count_for_current = 0
            
            while viewer.is_running():
                step_start = time.time()
                
                # --- A. 下发控制 ---
                if kinematic_only:
                    self.data.qpos[:6] = current_joint_target
                    if self.model.nu >= 8:
                        # 应用计算出的 clipped width
                        self.data.qpos[6] = current_gripper_target
                        self.data.qpos[7] = current_gripper_target
                    
                    mujoco.mj_forward(self.model, self.data)
                else:
                    self.data.ctrl[:6] = current_joint_target
                    if self.model.nu >= 8:
                        self.data.ctrl[6:] = current_gripper_target
                    mujoco.mj_step(self.model, self.data)

                if not kinematic_only:
                    mujoco.mj_step(self.model, self.data)
                
                step_count_for_current += 1
                
                # --- C. 误差检测 ---
                current_ee_pose = self._get_ee_pose_world()
                error = np.linalg.norm(current_ee_pose[:3] - current_pose_target_world[:3])
                
                reached = error < self.POSITION_THRESHOLD
                if current_idx == 0:
                    timeout = step_count_for_current >= self.MAX_STEPS
                else:
                    timeout = step_count_for_current >= max_steps_per_point
                
                # --- D. 切换 ---
                if reached or timeout:
                    status = "✅ Reached" if reached else "⚠️ Timeout"
                    print(f"Waypoint {current_idx}: {status} | Grip: {current_gripper_target:.3f}")
                    
                    if reached: success_count += 1
                    
                    current_idx += 1
                    if current_idx >= total_points:
                        print("\n[Trajectory] All waypoints completed!")
                        break 
                    
                    current_joint_target = joint_targets_queue[current_idx]
                    current_gripper_target = gripper_targets_queue[current_idx]
                    current_pose_target_world = target_world_seq[current_idx][:6]
                    self._update_mocap_marker(current_pose_target_world[:3])
                    
                    step_count_for_current = 0
                
                viewer.sync()
                time_until_next = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next > 0:
                    time.sleep(time_until_next)

        return success_count == total_points

    # ========================================================
    # 【修改】带相机函数：支持 (N, 7) 输入
    # ========================================================
    def move_trajectory_with_camera(self, target_seqs_world, camera_poses_world, max_steps_per_point=None, kinematic_only=False, cam_name="camera", width=640, height=480):
        if len(target_seqs_world) != len(camera_poses_world):
            raise ValueError("Target sequence and Camera sequence must have the same length.")

        print(f"\n[Trajectory & Cam] Processing {len(target_seqs_world)} frames...")
        if max_steps_per_point is None:
            max_steps_per_point = self.MAX_STEPS

        renderer = mujoco.Renderer(self.model, height=height, width=width)
        
        # IK & Gripper Pre-calc
        joint_targets_queue = []
        gripper_targets_queue = []
        last_valid_joints = self.data.qpos[:6].copy()
        
        for i, full_data in enumerate(target_seqs_world):
            # Gripper
            if len(full_data) >= 7:
                clipped_width = np.clip(full_data[6], self.GRIPPER_CLOSE_VAL, self.GRIPPER_OPEN_VAL)
            else:
                clipped_width = self.GRIPPER_OPEN_VAL
            gripper_targets_queue.append(clipped_width)

            # Pose
            pose = full_data[:6]
            pose_base = self._pose_tf_world2base(pose)
            joints = self._ik_base(pose_base)
            if joints is None:
                joints = last_valid_joints.copy()
            else:
                last_valid_joints = joints.copy()
            joint_targets_queue.append(joints)

        rgb_list = []
        mask_list = []
        current_idx = 0
        total_points = len(joint_targets_queue)
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            
            current_joint_target = joint_targets_queue[0]
            current_gripper_target = gripper_targets_queue[0]
            current_pose_target = target_seqs_world[0][:6]
            current_cam_pose = camera_poses_world[0]
            
            self._update_mocap_marker(current_pose_target[:3])
            step_count = 0

            while viewer.is_running() and current_idx < total_points:
                step_start = time.time()

                # --- Control ---
                if kinematic_only:
                    self.data.qpos[:6] = current_joint_target
                    if self.model.nu >= 8:
                        self.data.qpos[6:8] = current_gripper_target
                    mujoco.mj_forward(self.model, self.data)
                else:
                    self.data.ctrl[:6] = current_joint_target
                    if self.model.nu >= 8:
                        self.data.ctrl[6:] = current_gripper_target
                    mujoco.mj_step(self.model, self.data)

                step_count += 1
                
                # --- Check Status ---
                current_ee_pose = self._get_ee_pose_world()
                error = np.linalg.norm(current_ee_pose[:3] - current_pose_target[:3])
                
                reached = error < self.POSITION_THRESHOLD
                timeout = step_count >= max_steps_per_point
                
                # --- Capture Logic ---
                if reached or timeout:
                    self._set_camera_pose(cam_name, current_cam_pose)
                    mujoco.mj_forward(self.model, self.data)
                    
                    renderer.update_scene(self.data, camera=cam_name)
                    rgb_list.append(renderer.render().copy())
                    
                    renderer.enable_segmentation_rendering()
                    renderer.update_scene(self.data, camera=cam_name)
                    mask_list.append(renderer.render().copy())
                    renderer.disable_segmentation_rendering()
                    
                    # --- Next ---
                    current_idx += 1
                    if current_idx < total_points:
                        current_joint_target = joint_targets_queue[current_idx]
                        current_gripper_target = gripper_targets_queue[current_idx]
                        current_pose_target = target_seqs_world[current_idx][:6]
                        current_cam_pose = camera_poses_world[current_idx]
                        self._update_mocap_marker(current_pose_target[:3])
                        step_count = 0
                    else:
                        print("[Trajectory & Cam] All frames captured.")
                        break

                viewer.sync()
                time_until_next = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next > 0:
                    time.sleep(time_until_next)

        return rgb_list, mask_list

# ========================================================
# 使用示例：插值与动作演示
# ========================================================
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "R5/R5a/meshes/single_arm_scene.xml")
    
    # 辅助插值函数
    def generate_smooth_trajectory(keyframes, steps_per_segment=20):
        full_traj = []
        for i in range(len(keyframes) - 1):
            start_pt = np.array(keyframes[i])
            end_pt = np.array(keyframes[i+1])
            # 生成 steps_per_segment 个中间点 (包括起点，不包括终点以防重复)
            segment = np.linspace(start_pt, end_pt, steps_per_segment, endpoint=False)
            full_traj.append(segment)
            
        # 补上最后一个点
        full_traj.append([keyframes[-1]])
        return np.vstack(full_traj)

    try:
        robot = SingleArmController(xml_path)
        
        # 1. 定义关键帧 [x, y, z, rx, ry, rz, gripper_width]
        # rx, ry, rz = Euler (rad)
        # gripper = 0.044(Open) -> 0.0(Close)
        
        # 姿态定义: 假设手心向下为 [pi, 0, 0] 或者侧向
        euler_down = [np.pi, 0, 0] 
        
        keyframes = [
            # 1. 准备 (高, 张开)
            [0.4, 0.0, 1.3, *euler_down, 0.044],
            # 2. 下探 (低, 张开)
            [0.4, 0.0, 1.1, *euler_down, 0.044],
            # 3. 抓取 (低, 闭合)
            [0.4, 0.0, 1.1, *euler_down, 0.0],
            # 4. 举起 (高, 闭合)
            [0.4, 0.0, 1.3, *euler_down, 0.0],
            # 5. 移开 (侧, 闭合)
            [0.4, 0.3, 1.3, *euler_down, 0.0],
            # 6. 释放 (侧, 张开)
            [0.4, 0.3, 1.3, *euler_down, 0.044]
        ]
        
        print("Generating smooth trajectory...")
        trajectory_smooth = generate_smooth_trajectory(keyframes, steps_per_segment=30)
        print(f"Total frames: {len(trajectory_smooth)}")

        # 2. 生成相机轨迹 (简单的俯视旋转)
        num_frames = len(trajectory_smooth)
        camera_poses = np.zeros((num_frames, 6))
        for i in range(num_frames):
            # 相机绕着Z轴慢慢转一点
            angle = 1.57 + (i / num_frames) * 0.5 
            camera_poses[i] = [0.0, -1.0, 2.0, 0.0, 1.0, 1.57] 

        # 3. 执行
        rgb_frames, mask_frames = robot.move_trajectory_with_camera(
            trajectory_smooth,
            camera_poses,
            kinematic_only=True,
            cam_name="camera",
            width=640, height=480
        )
        
        # 4. 可视化
        print(f"Captured {len(rgb_frames)} frames. Playing back...")
        
        np.random.seed(42)
        color_palette = np.random.randint(0, 255, (256, 3), dtype=np.uint8)
        color_palette[0] = [0, 0, 0]

        for i, (rgb, mask) in enumerate(zip(rgb_frames, mask_frames)):
            bgr_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            
            geom_ids = mask[:, :, 0]
            geom_ids[geom_ids >= 256] = 0
            mask_vis = color_palette[geom_ids]
            
            cv2.putText(bgr_img, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow(f"Single Arm RGB", bgr_img)
            cv2.imshow(f"Segmentation Mask", mask_vis)
            
            # 使用 waitKey(50) 自动播放，按 ESC 退出
            if cv2.waitKey(50) == 27:
                break
                
        cv2.destroyAllWindows()

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")
