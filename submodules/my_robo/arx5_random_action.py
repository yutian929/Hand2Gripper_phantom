import time
import numpy as np
import cv2
import tqdm
from robosuite.robots import MobileRobot
from robosuite.utils.input_utils import *

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
    env.reset()
    env.viewer.set_camera(camera_id=0)
    for robot in env.robots:
        if isinstance(robot, MobileRobot):
            robot.enable_parts(legs=False, base=False)

    # do visualization
    for i in tqdm.tqdm(range(300)):
        start = time.time()
        action = np.random.randn(*env.action_spec[0].shape)
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