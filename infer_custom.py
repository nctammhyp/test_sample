import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import cv2
import random
import h5py
import torch
import sys
from google.colab.patches import cv2_imshow

sys.path.append('/kaggle/working/test_sample/metric_depth')

from depth_anything_v2.dpt import DepthAnythingV2

model_weights_path =  '/kaggle/working/depth_anything_v2_vitl.pth'
model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
model_encoder = 'vitl'

model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
model.load_state_dict(torch.load('/kaggle/working/depth_anything_v2_vitl.pth', map_location='cuda:0'))

model.to('cuda:0')
model.eval()

# Read the image
raw_img = cv2.imread('/kaggle/working/test_sample/dataset/frame_rgb_2.png')

# raw_img = cv2.resize(raw_img, (160, 128))

# raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

# Perform inference
depth = model.infer_image(raw_img)  # HxW raw depth map
# depth = 255 - depth

# # Optionally, visualize the depth map
# depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
# depth_colormap = cv2.applyColorMap(depth_normalized.astype('uint8'), cv2.COLORMAP_PLASMA)
# cv2_imshow(depth_colormap)

print(depth)