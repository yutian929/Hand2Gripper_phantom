import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import bimanual
from scipy.spatial.transform import Rotation as R

# 特定补偿值 (与单臂代码一致)
FLANGE_POS_OFFSET = np.array([-0.1, 0.0, -0.16])

class DualArmController:
    def __init__(self, xml_path, arm_names=['L', 'R'], position_threshold=0.02, max_steps=10_000):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML file not found: {xml_path}")
            
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.ARM_NAMES = arm_names
        self.EE_SITE_NAMES = {name: f"end_effector_{name}" for name in arm_names}
        self.BASE_LINK_NAMES = {name: f"base_link_{name}" for name in arm_names}
        
        # 配置参数
        self.GRIPPER_OPEN_VAL = 0.044
        self.POSITION_THRESHOLD = position_threshold
        self.MAX_STEPS = max_steps
        
        self.arm_params = {}
        self._init_kinematics_params()
        
        self.reset()

    def _init_kinematics_params(self):
        """为每个机械臂初始化运动学参数、关节索引和执行器索引"""
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
        """将所有关节位置归零"""
        self.data.qpos[:] = 0
        self.data.ctrl[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def _ik_base(self, target_pose_base):
        try:
            return bimanual.inverse_kinematics(target_pose_base)
        except Exception:
            return None
            
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
    def move_trajectory(self, target_world_seqs, max_steps_per_point=None, kinematic_only=False):
        """
        连续执行一系列双臂目标点。
        Args:
            target_world_seqs (dict): key 为 'L'/'R'，value 为形状 (N, 6) 的目标位姿数组。
            max_steps_per_point (int): 每个点的最大等待步数。
            kinematic_only (bool): 是否仅进行运动学控制 (无物理/碰撞/力矩)。
        """
        if not self.ARM_NAMES: return True
        
        print(f"\n[Trajectory] Dual Arm. Kinematic Only: {kinematic_only}")
        
        # 检查数据长度一致性
        try:
            num_points = len(list(target_world_seqs.values())[0])
            if not all(len(seq) == num_points for seq in target_world_seqs.values()):
                 print("[Error] Trajectories for all arms must have the same number of waypoints.")
                 return False
        except Exception as e:
            print(f"[Error] Invalid target sequence: {e}")
            return False

        if max_steps_per_point is None:
            max_steps_per_point = self.MAX_STEPS

        # 1. 预计算：解算所有点的 IK
        # ----------------------------------------------------
        joint_targets_queue = {name: [] for name in self.ARM_NAMES}
        last_valid_joints = {}
        
        # 初始化 Fallback 值
        for name in self.ARM_NAMES:
             qpos_indices = self.arm_params[name]['qpos_indices_6dof']
             if qpos_indices:
                 last_valid_joints[name] = self.data.qpos[qpos_indices].copy()
             else:
                 last_valid_joints[name] = np.zeros(6)

        failed_indices = {name: [] for name in self.ARM_NAMES}

        for i in range(num_points):
            for name in self.ARM_NAMES:
                pose = target_world_seqs[name][i]
                pose_base = self._pose_tf_world2base(pose, name)
                joints = self._ik_base(pose_base)
                
                if joints is None:
                    failed_indices[name].append(i)
                    joints = last_valid_joints[name].copy()
                else:
                    last_valid_joints[name] = joints.copy()
                        
                joint_targets_queue[name].append(joints)

        for name in self.ARM_NAMES:
            if len(failed_indices[name]) > 0:
                print(f"[Warning] Arm {name}: IK Failed for {len(failed_indices[name])} waypoints (used fallback).")

        print("[Trajectory] All IK solved. Starting execution...")

        # 2. 执行循环
        # ----------------------------------------------------
        current_idx = 0
        
        # 初始化第一个目标
        current_joint_targets = {name: joint_targets_queue[name][0] for name in self.ARM_NAMES}
        current_pose_targets = {name: target_world_seqs[name][0] for name in self.ARM_NAMES}
        
        for name in self.ARM_NAMES:
            self._update_mocap_marker(current_pose_targets[name][:3], name)
        
        success_count = 0
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            step_count_for_current = 0
            
            while viewer.is_running():
                step_start = time.time()
                
                # --- A. 下发控制 ---
                if kinematic_only:
                    # [运动学模式] 直接修改关节位置
                    for name in self.ARM_NAMES:
                        params = self.arm_params[name]
                        if params['qpos_indices_6dof']:
                            self.data.qpos[params['qpos_indices_6dof']] = current_joint_targets[name]
                        if params['qpos_indices_gripper']:
                            self.data.qpos[params['qpos_indices_gripper']] = self.GRIPPER_OPEN_VAL
                    
                    mujoco.mj_forward(self.model, self.data)
                else:
                    # [动力学模式] 设置控制信号
                    for name in self.ARM_NAMES:
                        params = self.arm_params[name]
                        if params['ctrl_indices_6dof']:
                            self.data.ctrl[params['ctrl_indices_6dof']] = current_joint_targets[name]
                        if params['ctrl_indices_gripper']:
                            self.data.ctrl[params['ctrl_indices_gripper']] = self.GRIPPER_OPEN_VAL
                
                # --- B. 物理步进 ---
                if not kinematic_only:
                    mujoco.mj_step(self.model, self.data)
                
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
                # 第一个点通常距离较远，给予更多时间 (MAX_STEPS)
                if current_idx == 0:
                     timeout = step_count_for_current >= self.MAX_STEPS
                else:
                     timeout = step_count_for_current >= max_steps_per_point

                if all_reached or timeout:
                    status = "✅ Reached" if all_reached else "⚠️ Timeout"
                    print(f"Waypoint {current_idx}: {status} | Max Error: {max_error:.4f}m")
                    
                    if all_reached: success_count += 1
                    
                    current_idx += 1
                    if current_idx >= num_points:
                        print("\n[Trajectory] All waypoints completed!")
                        break
                    
                    # 更新目标
                    current_joint_targets = {name: joint_targets_queue[name][current_idx] for name in self.ARM_NAMES}
                    current_pose_targets = {name: target_world_seqs[name][current_idx] for name in self.ARM_NAMES}
                    
                    for name in self.ARM_NAMES:
                        self._update_mocap_marker(current_pose_targets[name][:3], name)
                    
                    step_count_for_current = 0

                # --- E. 渲染 ---
                viewer.sync() 
                time_until_next = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next > 0:
                    time.sleep(time_until_next)
            
        return success_count == num_points

# ========================================================
# 使用示例
# ========================================================
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "R5", "R5a", "meshes", "dual_arm_scene.xml")
    
    try:
        robot = DualArmController(xml_path, arm_names=['L', 'R'])
        
        target_seq_L = np.array([
            [0.3, -0.5, 1.3, np.pi/2, 0.0, np.pi/2], 
            # [-0.3, -0.3, 1.3, 0.0, 0.0, 0.0],       
            # [-0.4, -0.2, 1.2, 0.0, np.pi/4, 0.0]    
        ])
        
        target_seq_R = np.array([
            [0.3, 0.5, 1.3, np.pi/2, 0.0, -np.pi/2], 
            # [0.3, -0.3, 1.3, np.pi/2, 0.0, -np.pi/2], 
            # [0.4, -0.2, 1.2, 0.0, np.pi/4, 0.0]      
        ])
        
        target_world_seqs = {
            'L': target_seq_L,
            'R': target_seq_R
        }
        
        # 示例：开启 kinematic_only=True
        robot.move_trajectory(target_world_seqs, kinematic_only=True)

    except Exception as e:
        print(f"Error: {e}")
