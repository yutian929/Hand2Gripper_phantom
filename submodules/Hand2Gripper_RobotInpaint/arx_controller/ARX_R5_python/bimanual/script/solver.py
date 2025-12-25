import kinematic_solver as solver
import numpy as np

_instance = None
# Base frame offset: Initial flange is at (0.09, 0.0, 0.10) in Base frame
BASE_OFFSET = np.array([0.09, 0.0, 0.10])

def forward_kinematics(joint_angles):
    global _instance
    if _instance is None:
        _instance = solver.KinematicSolver()
    
    # Get pose in InitFrame
    pose = np.array(_instance.forward_kinematics(joint_angles))
    
    # Convert to BaseFrame
    if pose.shape == (4, 4):
        pose[:3, 3] += BASE_OFFSET
    else:
        pose[:3] += BASE_OFFSET
        
    return pose

def inverse_kinematics(target_pose):
    global _instance
    if _instance is None:
        _instance = solver.KinematicSolver()
        
    # Convert from BaseFrame to InitFrame
    pose = np.array(target_pose, dtype=float)
    if pose.shape == (4, 4):
        pose[:3, 3] -= BASE_OFFSET
    else:
        pose[:3] -= BASE_OFFSET
        
    return _instance.inverse_kinematics(pose)

