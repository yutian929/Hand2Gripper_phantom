import torch
import cv2
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float16

pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)
img_path = "epic_kitch_demo.jpg"
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
outputs = pipe.predict(image)
breakpoint()
print(outputs)

# (Pdb) type(outputs)
# <class 'list'>
# (Pdb) len(outputs)
# 2
# (Pdb) type(outputs[0])
# <class 'dict'>
# (Pdb) outputs[0].keys()
# dict_keys(['hand_bbox', 'is_right', 'wilor_preds'])
# (Pdb) outputs[0]['hand_bbox']
# [638.0, 732.0, 882.0, 1035.0]
# (Pdb) outputs[0]['is_right']
# 0.0
# (Pdb) type(outputs[0]['wilor_preds'])
# <class 'dict'>
# (Pdb) outputs[0]['wilor_preds'].keys()
# dict_keys(['global_orient', 'hand_pose', 'betas', 'pred_cam', 'pred_keypoints_3d', 'pred_vertices', 'pred_cam_t_full', 'scaled_focal_length', 'pred_keypoints_2d'])