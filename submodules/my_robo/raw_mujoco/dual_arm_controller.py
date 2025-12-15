import mujoco
import mujoco.viewer
import numpy as np
import time
import os
from .ARX_R5_python import bimanual
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
        # 运动学模式下，POSITION_THRESHOLD 和 MAX_STEPS 仅用于调试和轨迹播放速度控制，
        # 实际 Kinematic 模式下，IK 成功则误差应该极小。
        self.POSITION_THRESHOLD = 0.02
        self.MAX_STEPS = 10_000 # 保持大数值，但Kinematic模式下通常在 1 步内完成。
        
        self.arm_params = {}
        self._init_kinematics_params()
        
        self.reset()

    def _init_kinematics_params(self):
        """为每个机械臂初始化运动学参数和关节/执行器 QPOS 索引"""
        for name in self.ARM_NAMES:
            params = {}
            
            # 1. Kinematic Parameters (基座位置、EE 偏移)
            base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.BASE_LINK_NAMES[name])
            params['base_pos_world'] = self.model.body_pos[base_id].copy() if base_id != -1 else np.zeros(3)

            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.EE_SITE_NAMES[name])
            params['ee_offset_local'] = self.model.site_pos[site_id].copy() if site_id != -1 else np.zeros(3)

            # 2. Index Mapping (修正 API)
            
            # 前 6 个旋转关节的 ID
            j_names_6dof = [f"joint{i}_{name}" for i in range(1, 7)]
            j_ids_6dof = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j_name) for j_name in j_names_6dof]
            
            # 夹爪 2 个滑动关节的 ID
            j_names_gripper = [f"joint7_{name}", f"joint8_{name}"]
            j_ids_gripper = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j_name) for j_name in j_names_gripper]
            
            # Qpos 地址：使用 self.model.jnt_qposadr (已修复)
            params['qpos_indices_6dof'] = [self.model.jnt_qposadr[j_id] for j_id in j_ids_6dof if j_id != -1]
            params['qpos_indices_gripper'] = [self.model.jnt_qposadr[j_id] for j_id in j_ids_gripper if j_id != -1]
            
            self.arm_params[name] = params

    def reset(self):
        """将所有关节位置归零，并调用 mj_forward 刷新几何体"""
        self.data.qpos[:] = 0
        self.data.ctrl[:] = 0 # 虽然是运动学，但习惯上清空 ctrl
        mujoco.mj_forward(self.model, self.data)

    def _ik_base(self, target_pose_base):
        """调用 IK 模型"""
        try:
            return bimanual.inverse_kinematics(target_pose_base)
        except Exception as e:
            # print(f"[IK Error]: {e}")
            return None
            
    def _get_ee_pose_world(self, arm_name):
        """获取指定机械臂的 EE 位姿 (世界坐标系)"""
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.EE_SITE_NAMES[arm_name])
        if site_id == -1:
             raise ValueError(f"EE site {self.EE_SITE_NAMES[arm_name]} not found.")
             
        pos = self.data.site_xpos[site_id]
        mat = self.data.site_xmat[site_id].reshape(3, 3)
        r = R.from_matrix(mat)
        euler = r.as_euler('xyz', degrees=False)
        return np.concatenate([pos, euler])

    def _pose_tf_world2base(self, target_pose_world, arm_name):
        """将世界坐标系的 EE 目标位姿转换为基座坐标系的法兰位姿 (IK 输入)"""
        params = self.arm_params[arm_name]
        target_pos_world = target_pose_world[:3]
        target_euler_world = target_pose_world[3:]
        
        r = R.from_euler('xyz', target_euler_world, degrees=False)
        rot_matrix = r.as_matrix()
        
        offset_world = rot_matrix @ params['ee_offset_local']
        flange_pos_world = target_pos_world - offset_world
        flange_pos_base = flange_pos_world - params['base_pos_world']
        
        # 特定补偿
        flange_pos_base += FLANGE_POS_OFFSET
        
        return np.concatenate([flange_pos_base, target_euler_world])

    def _update_mocap_marker(self, pos, arm_name):
        """更新目标标记"""
        marker_name = f"target_marker_{arm_name}"
        try:
            target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, marker_name)
            if target_body_id != -1:
                mocap_id = self.model.body_mocapid[target_body_id]
                self.data.mocap_pos[mocap_id] = pos
        except: pass

    # ========================================================
    # 【核心功能: 轨迹跟踪】 (纯运动学模式)
    # ========================================================
    def move_trajectory(self, target_world_seqs, delay_per_point=0.5):
        """
        连续执行一系列双臂目标点 (纯运动学模式)。
        Args:
            target_world_seqs (dict): key 为 'L'/'R'，value 为形状 (N, 6) 的目标位姿数组。
            delay_per_point (float): 每切换一个目标点后的延迟时间 (秒)。
        """
        if not self.ARM_NAMES: return True
        
        # 1. 预计算和 IK 检验 (与单臂类似)
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
        
        # 初始化 Fallback 值 (使用当前 QPOS)
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

        # 2. 轨迹播放和运行检验 (与单臂类似)
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
                
                # --- C. 运行检验和输出 ---
                output = ""
                reached_total = True
                
                for name in self.ARM_NAMES:
                    target_pos = current_pose_targets_world[name][:3]
                    current_pos = self._get_ee_pose_world(name)[:3]
                    error = np.linalg.norm(current_pos - target_pos)
                    
                    # 在运动学模式下，IK 成功则误差应该极小。
                    reached = error < self.POSITION_THRESHOLD
                    
                    # 根据 IK 预解算结果和实际误差生成报告
                    status_symbol = "✅" if reached else "❌"
                    status_text = "Reached" if reached else "IK Failed (Fallback)"
                    
                    # 如果 IK 本身失败了，我们报告 IK 状态
                    if ik_status[name][current_idx] == "FALLBACK":
                         status_text = "IK Failed (Fallback)"
                         reached_total = False # 如果任何一个臂是 Fallback，则总体不是完美 Reached
                    
                    output += f" {name}: {status_symbol} {status_text} | Error: {error:.4f}m |"
                    
                    if not reached:
                        reached_total = False
                
                if reached_total:
                    success_count += 1
                
                print(f"Waypoint {current_idx} / {num_points-1}: {output}")

                # --- D. 渲染和延迟 ---
                viewer.sync()
                if delay_per_point > 0:
                    time.sleep(delay_per_point)

            print(f"\n[Trajectory] All waypoints completed! ({success_count} / {num_points} perfect steps).")
                
        return success_count == num_points

# ========================================================
# 使用示例
# ========================================================
if __name__ == "__main__":
    # 【修改处】: 使用 os 获取当前脚本所在目录，并拼接相对路径
    # 假设当前脚本位于 raw_mujoco 文件夹下
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "R5", "R5a", "meshes", "dual_arm_model.xml")
    
    try:
        # 使用新的纯运动学控制器
        robot = DualArmKinematicController(xml_path, arm_names=['L', 'R'])
        
        # 目标点定义 (世界坐标系下的 EE 位姿: [x, y, z, rx, ry, rz])
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
        
        print("Starting dual arm trajectory playback (Kinematic).")
        # 纯运动学播放，设置 0.5 秒的延迟，让每个点停留一段时间
        robot.move_trajectory(target_world_seqs, delay_per_point=0.5) 
        
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the XML file is at the specified path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
