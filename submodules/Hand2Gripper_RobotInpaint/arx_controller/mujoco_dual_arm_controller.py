import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import cv2
import bimanual
from scipy.spatial.transform import Rotation as R

# 特定补偿值
FLANGE_MUJOCO_REAL_GAP = np.array([-0.02, -0.01, -0.066])
FLANGE_GRIPPER_GAP = np.array([-0.1, 0.0, 0.0])  # Gripper to Flange offset

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
    try:
        # 初始化控制器
        robot = DualArmController(xml_path)
        
        # -------------------------------------------------------------------
        # 1. 定义动作关键帧 (Keyframes)
        # 格式: [x, y, z, rx, ry, rz, gripper]
        # rx, ry, rz 使用欧拉角 (rad)
        # gripper: 0.044 (Open), 0.0 (Close)
        # -------------------------------------------------------------------
        # --- 左臂关键帧序列 ---
        seqs_L = [
            # 1. 准备 (高处, 张开)
            [0.4,  0.0, 1.4, 0, 0, 0, 0.00],
        ]
        
        # --- 右臂关键帧序列 ---
        seqs_R = [
            # 1. 准备 (高处, 张开) - Y是负的
            [0.4,  -0.4, 1.4, 0, 0, 0, 0.00],
        ]

        # -------------------------------------------------------------------
        # 3. 生成相机轨迹 (简单的环绕或固定视角)
        # -------------------------------------------------------------------
        camera_poses = [
            [0.0, -0.2, 2.0, 0.0, np.pi*0.25, 0.0],  # Frame 0
        ]

        # -------------------------------------------------------------------
        # 4. 执行并录制
        # -------------------------------------------------------------------
        print("Starting Simulation...")
        rgb_frames, mask_frames = robot.move_trajectory_with_camera(
            seqs_L, seqs_R, camera_poses, 
            cam_name="camera"
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
                key = cv2.waitKey(0)
                if key == 27: # ESC 退出
                    break
            
            cv2.destroyAllWindows()
    except Exception as e:
        print(f"[Error] {e}")
