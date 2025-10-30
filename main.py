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
# breakpoint()
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


pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)
img_path = "epic_kitch_demo.jpg"
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
detections = pipe.hand_detector(image, conf=0.3, verbose=True)[0]
breakpoint()
print(detections)

# (Pdb) type(detections)
# <class 'ultralytics.engine.results.Results'>
# (Pdb) dir(detections)
# ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattr__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__len__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_apply', '_keys', 'boxes', 'cpu', 'cuda', 'keypoints', 'masks', 'names', 'new', 'numpy', 'obb', 'orig_img', 'orig_shape', 'path', 'plot', 'probs', 'save', 'save_crop', 'save_dir', 'save_txt', 'show', 'speed', 'summary', 'to', 'tojson', 'update', 'verbose']
# (Pdb) len(detections)
# 2
# (Pdb) type(detections[0])
# <class 'ultralytics.engine.results.Results'>
# (Pdb) dir(detections[0])
# ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattr__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__len__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_apply', '_keys', 'boxes', 'cpu', 'cuda', 'keypoints', 'masks', 'names', 'new', 'numpy', 'obb', 'orig_img', 'orig_shape', 'path', 'plot', 'probs', 'save', 'save_crop', 'save_dir', 'save_txt', 'show', 'speed', 'summary', 'to', 'tojson', 'update', 'verbose']
# (Pdb) detections[0]._keys
# ('boxes', 'masks', 'probs', 'keypoints', 'obb')
# (Pdb) detections[0].boxes
# ultralytics.engine.results.Boxes object with attributes:
# cls: tensor([0.], device='cuda:0')
# conf: tensor([0.8516], device='cuda:0')
# data: tensor([[6.3800e+02, 7.3200e+02, 8.8200e+02, 1.0350e+03, 8.5157e-01, 0.0000e+00]], device='cuda:0')
# id: None
# is_track: False
# orig_shape: (1080, 1920)
# shape: torch.Size([1, 6])
# xywh: tensor([[760.0000, 883.5000, 244.0000, 303.0000]], device='cuda:0')
# xywhn: tensor([[0.3958, 0.8181, 0.1271, 0.2806]], device='cuda:0')
# xyxy: tensor([[ 638.,  732.,  882., 1035.]], device='cuda:0')
# xyxyn: tensor([[0.3323, 0.6778, 0.4594, 0.9583]], device='cuda:0')
