import time
import numpy as np
import cv2
import tqdm
from robosuite.robots import MobileRobot
from robosuite.utils.input_utils import *
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
import robosuite.utils.transform_utils as T

MAX_FR = 25  # max frame rate for running simluation

if __name__ == "__main__":

    # Create dict to hold options that will be passed to env creation call
    options = {}

    # Choose environment and add it to options
    # options["env_name"] = choose_environment()
    options["env_name"] = "TwoArmPhantom"
    # options["env_name"] = "TwoArmPegInHole"
    options["env_configuration"] = "phantom_parallel"

    if "TwoArm" in options["env_name"]:
        options["robots"] = ["Arx5", "Arx5"]
    else:
        options["robots"] = ["Arx5"]

    # Load OSC_POSE controller configuration for Cartesian control
    controller_name = "OSC_POSE"
    arm_controller_config = suite.load_part_controller_config(default_controller=controller_name)
    robot_name = options["robots"][0]
    options["controller_configs"] = refactor_composite_controller_config(
        arm_controller_config, robot_name, ["left", "right"]
    )

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        camera_names="zed",
        camera_segmentations="instance",
        camera_heights=480,
        camera_widths=640,
        control_freq=20,
    )
    obs = env.reset()
    env.viewer.set_camera(camera_id=0)
    for robot in env.robots:
        if isinstance(robot, MobileRobot):
            robot.enable_parts(legs=False, base=False)

    # Get robot base poses for coordinate transformation
    base_pos_0 = env.robots[0].base_pos
    base_ori_0 = env.robots[0].base_ori
    base_pos_1 = env.robots[1].base_pos
    base_ori_1 = env.robots[1].base_ori

    print(f"Robot 0 Base: {base_pos_0}")
    print(f"Robot 0 Ori: {base_ori_0}")
    print(f"Robot 1 Base: {base_pos_1}")
    print(f"Robot 1 Ori: {base_ori_1}")

    # Capture initial orientation to maintain during movement
    init_ori_0 = T.quat2mat(obs["robot0_eef_quat"])
    init_ori_1 = T.quat2mat(obs["robot1_eef_quat"])

    # Define targets relative to base
    # Left Robot
    target_pos_0_local = np.array([0.2, -0.1, 0.2])
    target_pos_0 = base_pos_0 + base_ori_0 @ target_pos_0_local
    
    # Set orientation to horizontal (World Frame)
    # Rotate 90 degrees around Y axis to point gripper forward (along World X)
    target_ori_0 = T.euler2mat([0, 0, 0])

    # Right Robot
    target_pos_1_local = np.array([0.2, 0.1, 0.2])
    target_pos_1 = base_pos_1 + base_ori_1 @ target_pos_1_local
    
    # Set orientation to horizontal (World Frame)
    target_ori_1 = T.euler2mat([0, 0, 0])

    # Calculate action dimensions
    total_action_dim = env.action_dim
    num_robots = len(env.robots)
    action_per_robot = total_action_dim // num_robots
    controller_dim = 6 # OSC_POSE (x, y, z, ax, ay, az)
    gripper_dim = action_per_robot - controller_dim
    print(f"Action dim per robot: {action_per_robot}, Gripper dim: {gripper_dim}")

    # Render once to initialize viewer
    env.render()

    # do visualization
    total_steps = 500

    for i in tqdm.tqdm(range(total_steps)):
        start = time.time()
        
        # Control Logic
        current_pos_0 = obs["robot0_eef_pos"]
        current_pos_1 = obs["robot1_eef_pos"]
        current_ori_0 = T.quat2mat(obs["robot0_eef_quat"])
        current_ori_1 = T.quat2mat(obs["robot1_eef_quat"])
        
        kp = 2.0 # Proportional gain for position
        kp_ori = 1.0 # Proportional gain for orientation
        
        # Calculate position error
        err_pos_0 = target_pos_0 - current_pos_0
        err_pos_1 = target_pos_1 - current_pos_1
        
        # Calculate orientation error (orientation difference -> axis-angle)
        # R_diff = R_target * R_current^T
        err_ori_mat_0 = np.dot(target_ori_0, current_ori_0.T)
        err_ori_0 = T.quat2axisangle(T.mat2quat(err_ori_mat_0))

        err_ori_mat_1 = np.dot(target_ori_1, current_ori_1.T)
        err_ori_1 = T.quat2axisangle(T.mat2quat(err_ori_mat_1))

        # Gripper action: Open (0.0)
        gripper_action_0 = np.array([1] * gripper_dim)
        gripper_action_1 = np.array([0] * gripper_dim)

        # Action: [dx, dy, dz, dax, day, daz, gripper]
        action_0 = np.concatenate([
            np.clip(err_pos_0 * kp, -1, 1), 
            np.clip(err_ori_0 * kp_ori, -0.5, 0.5), 
            gripper_action_0
        ])
        action_1 = np.concatenate([
            np.clip(err_pos_1 * kp, -1, 1), 
            np.clip(err_ori_1 * kp_ori, -0.5, 0.5), 
            gripper_action_1
        ])
        
        action = np.concatenate([action_0, action_1])

        obs, reward, done, _ = env.step(action)
        
        # Calculate and print distance error
        curr_pos_0 = obs["robot0_eef_pos"]
        curr_pos_1 = obs["robot1_eef_pos"]
        dist_err_0 = np.linalg.norm(curr_pos_0 - target_pos_0)
        dist_err_1 = np.linalg.norm(curr_pos_1 - target_pos_1)
        print(f"Step {i} | Dist Err - Left: {dist_err_0:.4f}, Right: {dist_err_1:.4f}")

        # breakpoint()
        # Get RGB and Instance Segmentation images
        # Keys are formatted as {camera_name}_{modality}
        rgb_img = obs.get("zed_image")  # Shape: (H, W, 3)
        rgb_img = cv2.flip(rgb_img, 1)  # Flip image horizontally for correct visualization
        seg_img = obs.get("zed_segmentation_instance")  # Shape: (H, W, 1)
        seg_img = cv2.flip(seg_img, 1)  # Flip image horizontally for correct visualization

        cv2.imshow("RGB Image", cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

        # cv2.flip squeezes the channel dimension if it is 1, so seg_img is (H, W)
        seg_max = seg_img.max()
        if seg_max > 0:
            seg_vis = (seg_img / seg_max).astype(np.float32)
        else:
            seg_vis = seg_img.astype(np.float32)

        cv2.imshow("Instance Segmentation", seg_vis)
        cv2.waitKey(1)
        env.render()

        # limit frame rate if necessary
        elapsed = time.time() - start
        diff = 1 / MAX_FR - elapsed
        if diff > 0:
            time.sleep(diff)
    cv2.destroyAllWindows()