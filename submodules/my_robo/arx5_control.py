import time
import numpy as np
import cv2
import tqdm
from robosuite.robots import MobileRobot
from robosuite.utils.input_utils import *
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config

MAX_FR = 25  # max frame rate for running simluation

if __name__ == "__main__":

    # Create dict to hold options that will be passed to env creation call
    options = {}

    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    # Choose environment and add it to options
    # options["env_name"] = choose_environment()
    options["env_name"] = "TwoArmPhantom"
    # options["env_name"] = "TwoArmPegInHole"
    options["env_configuration"] = "phantom_parallel"

    # If a multi-arm environment has been chosen, choose configuration and appropriate robot(s)
    if "TwoArm" in options["env_name"]:
        # # Choose env config and add it to options
        # options["env_configuration"] = choose_multi_arm_config()

        # # If chosen configuration was bimanual, the corresponding robot must be Baxter. Else, have user choose robots
        # if options["env_configuration"] == "single-robot":
        #     options["robots"] = choose_robots(exclude_bimanual=False, use_humanoids=True, exclude_single_arm=True)
        # else:
        #     options["robots"] = []

        #     # Have user choose two robots
        #     for i in range(2):
        #         print("Please choose Robot {}...\n".format(i))
        #         options["robots"].append(choose_robots(exclude_bimanual=False, use_humanoids=True))
        options["robots"] = ["Arx5", "Arx5"]
        # options["robots"] = ["Baxter", "Baxter"]
    # If a humanoid environment has been chosen, choose humanoid robots
    # elif "Humanoid" in options["env_name"]:
    #     options["robots"] = choose_robots(use_humanoids=True)
    else:
        # options["robots"] = choose_robots(exclude_bimanual=False, use_humanoids=True)
        options["robots"] = ["Arx5"]

    # Load OSC_POSE controller configuration for Cartesian control
    controller_name = "OSC_POSE"
    arm_controller_config = suite.load_part_controller_config(default_controller=controller_name)
    robot_name = options["robots"][0]
    options["controller_configs"] = refactor_composite_controller_config(
        arm_controller_config, robot_name, ["right", "left"]
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

    # Define target positions (relative to initial positions for safety)
    # We assume robot0 is left and robot1 is right (or vice versa depending on config)
    init_pos_0 = obs["robot0_eef_pos"]
    init_pos_1 = obs["robot1_eef_pos"]
    
    # Example: Move both arms 20cm forward (x) and 10cm up (z)
    target_pos_0 = init_pos_0 + np.array([0.2, 0.0, 0.1]) 
    target_pos_1 = init_pos_1 + np.array([0.2, 0.0, 0.1])

    print(f"Initial Pos 0: {init_pos_0}")
    print(f"Target Pos 0: {target_pos_0}")

    # Calculate action dimensions
    total_action_dim = env.action_dim
    num_robots = len(env.robots)
    action_per_robot = total_action_dim // num_robots
    controller_dim = 6 # OSC_POSE (x, y, z, ax, ay, az)
    gripper_dim = action_per_robot - controller_dim
    print(f"Action dim per robot: {action_per_robot}, Gripper dim: {gripper_dim}")

    # do visualization
    for i in tqdm.tqdm(range(300)):
        start = time.time()
        
        # Control Logic
        current_pos_0 = obs["robot0_eef_pos"]
        current_pos_1 = obs["robot1_eef_pos"]
        
        kp = 2.0 # Proportional gain
        
        # Calculate position error
        err_0 = target_pos_0 - current_pos_0
        err_1 = target_pos_1 - current_pos_1
        
        # Action: [dx, dy, dz, dax, day, daz, gripper]
        # We only control position here, orientation kept 0 (maintain)
        # Gripper 0 (open)
        gripper_action = np.zeros(gripper_dim)
        action_0 = np.concatenate([np.clip(err_0 * kp, -1, 1), [0, 0, 0], gripper_action])
        action_1 = np.concatenate([np.clip(err_1 * kp, -1, 1), [0, 0, 0], gripper_action])
        
        action = np.concatenate([action_0, action_1])

        obs, reward, done, _ = env.step(action)
        # breakpoint()
        # Get RGB and Instance Segmentation images
        # Keys are formatted as {camera_name}_{modality}
        rgb_img = obs.get("zed_image")  # Shape: (H, W, 3)
        seg_img = obs.get("zed_segmentation_instance")  # Shape: (H, W, 1)
        cv2.imshow("RGB Image", cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

        seg_max = seg_img[:, :, 0].max()
        if seg_max > 0:
            seg_vis = (seg_img[:, :, 0] / seg_max).astype(np.float32)
        else:
            seg_vis = seg_img[:, :, 0].astype(np.float32)

        cv2.imshow("Instance Segmentation", seg_vis)
        cv2.waitKey(1)
        env.render()

        # limit frame rate if necessary
        elapsed = time.time() - start
        diff = 1 / MAX_FR - elapsed
        if diff > 0:
            time.sleep(diff)
    cv2.destroyAllWindows()