import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import bimanual
from scipy.spatial.transform import Rotation as R

class SingleArmController:
    def __init__(self, xml_path, end_effector_site="end_effector", base_link_name="base_link", position_threshold=0.01, max_steps=10_000):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML file not found: {xml_path}")
            
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.ee_site_name = end_effector_site
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
        flange_pos_base[0] -= 0.1
        flange_pos_base[2] -= 0.16
        
        return np.concatenate([flange_pos_base, target_euler_world])

    def _update_mocap_marker(self, pos):
        try:
            target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_marker")
            if target_body_id != -1:
                mocap_id = self.model.body_mocapid[target_body_id]
                self.data.mocap_pos[mocap_id] = pos
        except: pass

    # ========================================================
    # 【新增】轨迹跟踪函数
    # ========================================================
    def move_trajectory(self, target_world_seq, max_steps_per_point=None):
        """
        连续执行一系列目标点
        
        Args:
            target_world_seq (np.array): 形状 (N, 6) 的数组，包含 N 个目标位姿
            max_steps_per_point (int): 每个点的最大等待步数
        """
        print(f"\n[Trajectory] Received {len(target_world_seq)} waypoints.")
        if max_steps_per_point is None:
            max_steps_per_point = self.MAX_STEPS
        
        # 1. 预计算：先解算所有点的 IK，确保路径可行
        # ----------------------------------------------------
        joint_targets_queue = []
        for i, pose in enumerate(target_world_seq):
            pose_base = self._pose_tf_world2base(pose)
            joints = self._ik_base(pose_base)
            
            if joints is None:
                print(f"[Error] IK Failed at waypoint {i}: {pose}")
                return False
            joint_targets_queue.append(joints)
            
        print("[Trajectory] All IK solved. Starting execution...")

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
                self.data.ctrl[:6] = current_joint_target
                if self.model.nu >= 8:
                    self.data.ctrl[6:] = self.GRIPPER_OPEN_VAL

                # --- B. 物理步进 ---
                mujoco.mj_step(self.model, self.data)
                step_count_for_current += 1
                
                # --- C. 误差检测 ---
                current_ee_pose = self._get_ee_pose_world()
                # 只计算位置误差 (欧氏距离)
                error = np.linalg.norm(current_ee_pose[:3] - current_pose_target_world[:3])
                
                # 状态检查
                reached = error < self.POSITION_THRESHOLD
                timeout = step_count_for_current >= max_steps_per_point
                
                # --- D. 切换目标逻辑 ---
                if reached or timeout:
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
# 使用示例
# ========================================================
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "R5/R5a/meshes/mjmodel.xml")
    
    try:
        robot = SingleArmController(xml_path)
        
        target_seq_world = np.array([
            [0.3, 0.3, 1.3, np.pi/2, 0.0, np.pi/2],
            [0.3, 0.0, 1.3, 0.0, 0.0, 0.0],
            [0.4, -0.2, 1.2, 0.0, np.pi/4, 0.0]
        ])
        
        robot.move_trajectory(target_seq_world)
        
    except Exception as e:
        print(f"Error: {e}")