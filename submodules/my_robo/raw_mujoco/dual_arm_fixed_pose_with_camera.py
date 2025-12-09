import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import cv2  # [新增] 引入OpenCV
import bimanual
from scipy.spatial.transform import Rotation as R

# 假设 bimanual.inverse_kinematics 模块可用

# 特定补偿值 (与单臂代码一致)
FLANGE_POS_OFFSET = np.array([-0.1, 0.0, -0.16])

class DualArmKinematicController:
    def __init__(self, xml_path, arm_names=['L', 'R']):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML file not found: {xml_path}")
            
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.ARM_NAMES = arm_names
        self.EE_SITE_NAMES = {name: f"end_effector_{name}" for name in arm_names}
        self.BASE_LINK_NAMES = {name: f"base_link_{name}" for name in arm_names}
        
        # 运动学模式的配置参数
        self.GRIPPER_OPEN_VAL = 0.044
        self.POSITION_THRESHOLD = 0.02
        self.MAX_STEPS = 10_000 
        
        self.arm_params = {}
        self._init_kinematics_params()

        # ================= [新增] 相机与渲染器初始化 =================
        # 初始化离屏渲染器，用于获取图像数据 (RGB + Segmentation)
        self.render_height = 480
        self.render_width = 640
        self.renderer = mujoco.Renderer(self.model, height=self.render_height, width=self.render_width)
        
        # 设定观察者相机 (Free Camera)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        
        # 设定相机位置 (观察者视角)
        # 这里设定一个俯视且稍微偏向正前方的视角，你可以根据需要修改 lookat, distance, azimuth, elevation
        self.camera.fixedcamid = -1
        self.camera.lookat = [0.0, 0.0, 0.8]  # 看向工作空间中心
        self.camera.distance = 2.5            # 距离
        self.camera.azimuth = 90              # 方位角
        self.camera.elevation = -40           # 俯仰角
        
        # 预先获取属于机械臂的 Geom ID，用于生成掩码
        self.robot_geom_ids = self._get_robot_geom_ids()
        # ==========================================================
        
        self.reset()

    def _get_robot_geom_ids(self):
        """[新增] 获取所有属于机械臂的几何体 ID，用于 Instance Segmentation"""
        ids = []
        for i in range(self.model.ngeom):
            # 获取该 geom 所属的 body ID
            body_id = self.model.geom_bodyid[i]
            # 获取 body 名称
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            
            # 简单的启发式逻辑：如果 body 名字里包含 'L' 或 'R' (且不是基座以外的无关物体)
            # 或者你可以根据 xml 的层级结构更精确地筛选
            if body_name:
                # 假设机械臂的 body 命名包含 arm_names 中的字符 (比如 link_L, link_R 等)
                if any(name in body_name for name in self.ARM_NAMES):
                    ids.append(i)
        return np.array(ids)

    def _init_kinematics_params(self):
        """为每个机械臂初始化运动学参数和关节/执行器 QPOS 索引"""
        for name in self.ARM_NAMES:
            params = {}
            
            # 1. Kinematic Parameters (基座位置、EE 偏移)
            base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.BASE_LINK_NAMES[name])
            params['base_pos_world'] = self.model.body_pos[base_id].copy() if base_id != -1 else np.zeros(3)

            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.EE_SITE_NAMES[name])
            params['ee_offset_local'] = self.model.site_pos[site_id].copy() if site_id != -1 else np.zeros(3)

            # 2. Index Mapping
            j_names_6dof = [f"joint{i}_{name}" for i in range(1, 7)]
            j_ids_6dof = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j_name) for j_name in j_names_6dof]
            
            j_names_gripper = [f"joint7_{name}", f"joint8_{name}"]
            j_ids_gripper = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j_name) for j_name in j_names_gripper]
            
            params['qpos_indices_6dof'] = [self.model.jnt_qposadr[j_id] for j_id in j_ids_6dof if j_id != -1]
            params['qpos_indices_gripper'] = [self.model.jnt_qposadr[j_id] for j_id in j_ids_gripper if j_id != -1]
            
            self.arm_params[name] = params

    def reset(self):
        """将所有关节位置归零，并调用 mj_forward 刷新几何体"""
        self.data.qpos[:] = 0
        self.data.ctrl[:] = 0 
        mujoco.mj_forward(self.model, self.data)

    def _ik_base(self, target_pose_base):
        try:
            return bimanual.inverse_kinematics(target_pose_base)
        except Exception as e:
            return None
            
    def _get_ee_pose_world(self, arm_name):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.EE_SITE_NAMES[arm_name])
        if site_id == -1:
             raise ValueError(f"EE site {self.EE_SITE_NAMES[arm_name]} not found.")
             
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

    # ================= [新增] 可视化相机与掩码 =================
    def _visualize_camera(self):
        """
        模仿 AAAA 中的逻辑：
        1. 获取 RGB 图像
        2. 获取 Segmentation 图像
        3. 提取 Robot 的 Instance Mask
        4. 使用 cv2 显示
        """
        # 更新渲染器场景
        self.renderer.update_scene(self.data, camera=self.camera)
        
        # 1. 获取 RGB
        rgb_img = self.renderer.render() # 默认是 RGB
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR) # 转为 OpenCV 格式
        
        # 2. 获取 Segmentation
        self.renderer.enable_segmentation_rendering()
        seg_img = self.renderer.render()
        self.renderer.disable_segmentation_rendering()
        
        # seg_img shape 是 (H, W, 2)。channel 0 是 geom ID, channel 1 是 object ID
        geom_ids = seg_img[:, :, 0]
        
        # 3. 提取 Mask (Instance Level for Robot)
        # 检查像素的 geom_id 是否在我们的 robot_geom_ids 列表中
        mask = np.isin(geom_ids, self.robot_geom_ids).astype(np.uint8) * 255
        
        # 4. 显示
        cv2.imshow("Observer RGB", rgb_img)
        cv2.imshow("Robot Instance Mask", mask)
        cv2.waitKey(1)
    # ==========================================================

    def move_trajectory(self, target_world_seqs, delay_per_point=0.5):
        if not self.ARM_NAMES: return True
        
        try:
            num_points = len(target_world_seqs[self.ARM_NAMES[0]])
            if not all(len(seq) == num_points for seq in target_world_seqs.values()):
                 print("[Error] Trajectories for all arms must have the same number of waypoints.")
                 return False
        except KeyError:
            print("[Error] Target sequence dictionary is incomplete.")
            return False

        joint_targets_queue = {name: [] for name in self.ARM_NAMES}
        ik_status = {name: [] for name in self.ARM_NAMES}
        ik_failed_count = 0
        last_valid_joints = {}
        
        for name in self.ARM_NAMES:
             qpos_indices = self.arm_params[name]['qpos_indices_6dof']
             last_valid_joints[name] = self.data.qpos[qpos_indices].copy() if qpos_indices else np.zeros(6)

        # 预解算 IK
        for i in range(num_points):
            for name in self.ARM_NAMES:
                pose = target_world_seqs[name][i]
                pose_base = self._pose_tf_world2base(pose, name)
                joints = self._ik_base(pose_base)
                
                if joints is None:
                    joints = last_valid_joints[name].copy()
                    ik_failed_count += 1
                    ik_status[name].append("FALLBACK")
                else:
                    last_valid_joints[name] = joints.copy()
                    ik_status[name].append("SOLVED")
                        
                joint_targets_queue[name].append(joints)

        print(f"\n[Trajectory] All IK solved (total {ik_failed_count} fallbacks). Starting execution...")

        success_count = 0
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            
            for current_idx in range(num_points):
                if not viewer.is_running():
                    break
                    
                current_pose_targets_world = {name: target_world_seqs[name][current_idx] for name in self.ARM_NAMES}
                current_joint_targets = {name: joint_targets_queue[name][current_idx] for name in self.ARM_NAMES}

                # --- A. 设置 QPOS (瞬移) ---
                for name in self.ARM_NAMES:
                    params = self.arm_params[name]
                    
                    if params['qpos_indices_6dof']:
                         self.data.qpos[params['qpos_indices_6dof']] = current_joint_targets[name]
                    if params['qpos_indices_gripper']:
                        self.data.qpos[params['qpos_indices_gripper']] = self.GRIPPER_OPEN_VAL

                    self._update_mocap_marker(current_pose_targets_world[name][:3], name)
                
                # --- B. 运动学更新 ---
                mujoco.mj_forward(self.model, self.data)
                
                # --- [新增] C. 相机采集与可视化 ---
                # 在每次物理状态更新后，提取掩码并显示
                self._visualize_camera()
                # ------------------------------

                # --- D. 运行检验和输出 ---
                output = ""
                reached_total = True
                
                for name in self.ARM_NAMES:
                    target_pos = current_pose_targets_world[name][:3]
                    current_pos = self._get_ee_pose_world(name)[:3]
                    error = np.linalg.norm(current_pos - target_pos)
                    
                    reached = error < self.POSITION_THRESHOLD
                    
                    status_symbol = "✅" if reached else "❌"
                    status_text = "Reached" if reached else "IK Failed (Fallback)"
                    
                    if ik_status[name][current_idx] == "FALLBACK":
                         status_text = "IK Failed (Fallback)"
                         reached_total = False 
                    
                    output += f" {name}: {status_symbol} {status_text} | Error: {error:.4f}m |"
                    
                    if not reached:
                        reached_total = False
                
                if reached_total:
                    success_count += 1
                
                print(f"Waypoint {current_idx} / {num_points-1}: {output}")

                # --- E. 渲染和延迟 ---
                viewer.sync()
                if delay_per_point > 0:
                    time.sleep(delay_per_point)

            print(f"\n[Trajectory] All waypoints completed! ({success_count} / {num_points} perfect steps).")
                
        # [新增] 关闭 OpenCV 窗口
        cv2.destroyAllWindows() 
        return success_count == num_points

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "R5", "R5a", "meshes", "dual_arm_model.xml")
    
    try:
        robot = DualArmKinematicController(xml_path, arm_names=['L', 'R'])
        
        target_seq_L = np.array([
            [-0.3, 0.3, 1.3, np.pi/2, 0.0, np.pi/2], 
            [-0.3, -0.3, 1.3, 0.0, 0.0, 0.0],       
            [-0.4, -0.2, 1.2, 0.0, np.pi/4, 0.0]    
        ])
        
        target_seq_R = np.array([
            [0.3, 0.3, 1.3, np.pi/2, 0.0, -np.pi/2], 
            [0.3, -0.3, 1.3, np.pi/2, 0.0, -np.pi/2], 
            [0.4, -0.2, 1.2, 0.0, np.pi/4, 0.0]       
        ])
        
        target_world_seqs = {
            'L': target_seq_L,
            'R': target_seq_R
        }
        
        print("Starting dual arm trajectory playback (Kinematic) with Camera Segmentation...")
        robot.move_trajectory(target_world_seqs, delay_per_point=0.5) 
        
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the XML file is at the specified path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
