import numpy as np
from scipy.spatial.transform import Rotation as R

def inv(T):
    Rm, t = T[:3,:3], T[:3,3]
    Ti = np.eye(4)
    Ti[:3,:3] = Rm.T
    Ti[:3,3]  = -Rm.T @ t
    return Ti

def quat_wxyz_from_R(Rm):
    q_xyzw = R.from_matrix(Rm).as_quat()   # [x,y,z,w]
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

T_L = np.load("/home/yutian/Hand2Gripper_phantom/phantom/camera/eye_to_hand_result_left_latest.npz")["T_base_link"]   # base_T_cam
T_R = np.load("/home/yutian/Hand2Gripper_phantom/phantom/camera/eye_to_hand_result_right_latest.npz")["T_base_link"]

# 方案A：固定左臂 base 在你想要的位置（这里填你 XML 里的左臂 base）
T_world_base_L = np.eye(4)
T_world_base_L[:3,3] = np.array([0.0, 0.0, 1.0])  # <-- 改成你的左臂 base 位置
# 如果左臂 base 不是单位朝向，也要把旋转写进 T_world_base_L[:3,:3]

T_world_base_R = T_world_base_L @ T_L @ inv(T_R)

pos_R  = T_world_base_R[:3,3]
quat_R = quat_wxyz_from_R(T_world_base_R[:3,:3])

print("base_link_R pos =", pos_R)
print("base_link_R quat(wxyz) =", quat_R)

# 顺便验证 camera check 会是 0
T_world_cam_L = T_world_base_L @ T_L
T_world_cam_R = T_world_base_R @ T_R
pos_diff = np.linalg.norm(T_world_cam_L[:3,3] - T_world_cam_R[:3,3])
rot_diff = np.linalg.norm(R.from_matrix(T_world_cam_L[:3,:3].T @ T_world_cam_R[:3,:3]).as_rotvec())
print("pos_diff =", pos_diff, "rot_diff_deg =", np.degrees(rot_diff))
