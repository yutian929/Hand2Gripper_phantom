import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import bimanual
from scipy.spatial.transform import Rotation as R

class RobotController:
    def __init__(self, xml_path, end_effector_site="end_effector", base_link_name="base_link"):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML file not found: {xml_path}")
            
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.ee_site_name = end_effector_site
        
        # ========================================================
        # 1. 自动从 XML 读取参数 (不再硬编码)
        # ========================================================
        
        # A. 获取基座位置 (World Frame -> Base Frame)
        try:
            base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, base_link_name)
            if base_id == -1:
                raise ValueError(f"Body '{base_link_name}' not found.")
            # model.body_pos 获取的是 XML 中定义的 pos 属性 (相对于父级，这里父级是 world)
            self.base_pos_world = self.model.body_pos[base_id]
            print(f"[Init] Detected Base Position (World): {self.base_pos_world}")
        except Exception as e:
            print(f"[Error] Failed to read base position: {e}")
            self.base_pos_world = np.zeros(3)

        # B. 获取末端偏移 (Flange -> Tip)
        try:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, end_effector_site)
            if site_id == -1:
                raise ValueError(f"Site '{end_effector_site}' not found.")
            # model.site_pos 获取的是 site 相对于其附着 body (Link6) 的局部偏移
            self.ee_offset_local = self.model.site_pos[site_id]
            print(f"[Init] Detected EE Offset (Local): {self.ee_offset_local}")
        except Exception as e:
            print(f"[Error] Failed to read EE offset: {e}")
            self.ee_offset_local = np.zeros(3)

        # ========================================================
        
        # Constants
        self.GRIPPER_OPEN = 0.044
        self.POSITION_THRESHOLD = 0.01
        self.MAX_STEPS = 10000 # 适当减少，100万步太久了
        
        self.reset()

    def reset(self):
        """Reset robot."""
        self.data.qpos[:] = 0
        self.data.ctrl[:] = 0
        mujoco.mj_step(self.model, self.data)
        print("Robot reset.")

    def get_current_ee_pos_world(self):
        """获取当前末端执行器的【世界坐标】"""
        try:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site_name)
            # data.site_xpos 返回的是仿真运行时的全局世界坐标
            return self.data.site_xpos[site_id]
        except Exception:
            return np.zeros(3)

    def transform_target_for_ik(self, ee_target_pose_world):
        """
        坐标系转换流水线：
        输入: 指尖目标 (世界坐标)
        输出: 法兰盘目标 (基座坐标) -> 供 IK 使用
        """
        # 1. 解包目标
        target_pos_world = ee_target_pose_world[:3]
        target_euler = ee_target_pose_world[3:]
        
        # 2. 计算末端偏移在世界坐标系下的向量
        #    Offset_World = Rotation_Matrix * Offset_Local
        r = R.from_euler('xyz', target_euler, degrees=False)
        rot_matrix = r.as_matrix()
        offset_vector_world = rot_matrix @ self.ee_offset_local
        
        # 3. 计算法兰盘在【世界坐标系】的位置
        #    Flange_World = Tip_World - Offset_World
        flange_pos_world = target_pos_world - offset_vector_world
        
        # 4. 计算法兰盘在【基座坐标系】的位置
        #    Flange_Base = Flange_World - Base_Position
        flange_pos_base = flange_pos_world - self.base_pos_world
        flange_pos_base[0] -= 0.1   # 特定机械臂的 X 轴偏移补偿
        flange_pos_base[-1] -= 0.16  # 特定机械臂的 Z 轴偏移补偿
        
        # 5. 组合结果 (假设姿态在基座系和世界系一致，如果不一致需额外左乘基座逆旋转)
        #    通常基座只平移不旋转，所以欧拉角可以直接复用
        flange_target_pose_base = np.concatenate([flange_pos_base, target_euler])
        
        print(f"  [Debug] EE(World):     {np.round(target_pos_world, 3)}")
        print(f"  [Debug] Flange(World): {np.round(flange_pos_world, 3)}")
        print(f"  [Debug] Flange(Base):  {np.round(flange_pos_base, 3)} (Send to IK)")
        
        return flange_target_pose_base

    def move_to_pose(self, target_pose_world):
        print(f"\nTarget EE Pose (World): {target_pose_world[:3]}")
        
        # 1. 坐标转换：世界系指尖 -> 基座系法兰
        ik_input_pose = self.transform_target_for_ik(target_pose_world)

        # 2. IK 计算 (使用基座系坐标)
        try:
            target_joints = bimanual.inverse_kinematics(ik_input_pose)
            if target_joints is None:
                print("IK failed: No solution found.")
                return
        except Exception as e:
            print(f"IK Error: {e}")
            return
        
        print(f"Solved Joints: {np.round(target_joints, 4)}")
        
        # 3. 可视化 (更新小球位置到世界坐标目标)
        self.update_mocap_marker(target_pose_world[:3]) 

        # 4. 控制循环
        q_end = target_joints
        step_count = 0
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                step_start = time.time()
                
                # --- 控制 ---
                self.data.ctrl[:6] = q_end
                if self.model.nu >= 8:
                    self.data.ctrl[6:] = self.GRIPPER_OPEN

                # --- 物理 ---
                mujoco.mj_step(self.model, self.data)
                step_count += 1
                
                # --- 误差计算 (在世界坐标系下比较) ---
                # A. 获取当前指尖真实世界坐标
                current_ee_world = self.get_current_ee_pos_world()
                # B. 获取目标指尖世界坐标
                target_ee_world = target_pose_world[:3]
                
                # C. 计算误差
                error = np.linalg.norm(current_ee_world - target_ee_world)

                if step_count % 100 == 0:
                    print(f"Step {step_count}: Error = {error:.4f} m")

                # --- 判定 ---
                if error < self.POSITION_THRESHOLD:
                    print(f"\n✅ 成功到达! 最终误差: {error:.5f}m")
                    break
                
                if step_count >= self.MAX_STEPS:
                    print(f"\n⚠️ 超时! 最终误差: {error:.5f}m")
                    break

                viewer.sync()
                
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    def update_mocap_marker(self, pos):
        """移动绿色小球 (输入世界坐标)"""
        try:
            target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_marker")
            if target_body_id != -1:
                mocap_id = self.model.body_mocapid[target_body_id]
                self.data.mocap_pos[mocap_id] = pos
        except:
            pass

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "R5/R5a/meshes/mjmodel.xml")
    
    # 【注意】这里的目标是世界坐标
    # 因为基座在 z=1.0，所以如果我们想让机械臂去一个合理的位置，z 应该在 1.0 附近或之上
    target_pose_world = np.array([0.2, 0.0, 1.4, 0.0, 0.0, 0.0])
    
    try:
        robot = RobotController(xml_path, end_effector_site="end_effector")
        robot.move_to_pose(target_pose_world)
    except Exception as e:
        print(f"Error: {e}")