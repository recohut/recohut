---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="7IIpgpFc01kz" -->
# Human Post Estimation with Keypoint RCNN model
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 103, "referenced_widgets": ["2afea985ce864250bdeff518d11a73d1", "b137eb92212e4e99aa9a6a7d8fcd12b6", "aafe8c53753c4bf2b04474a4e25b28b9", "a99d167497bb41c4af9d673335275fe0", "b95a30781c6046a8808787f79a38f586", "66d7998454a04658862bf2d3d1aaf77d", "7783d6cf9f3440c6941f8ea19795770e", "c62f9886ffa8401784da98a5cd31e8d5"]} id="1NRvJeQRXxR2" executionInfo={"status": "ok", "timestamp": 1608555408177, "user_tz": -330, "elapsed": 19796, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d7196e1a-6a2c-48a6-8cac-7d4a808e394d"
import os
from os.path import exists, join, basename, splitext

import random
import PIL
import torchvision
import cv2
import numpy as np
import torch
torch.set_grad_enabled(False)
  
import time
import matplotlib
import matplotlib.pylab as plt
plt.rcParams["axes.grid"] = False

model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
model = model.eval().cuda()
```

```python colab={"base_uri": "https://localhost:8080/"} id="eC_YMOpSXyCH" executionInfo={"status": "ok", "timestamp": 1608555408181, "user_tz": -330, "elapsed": 19794, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ba76d2fa-fcc8-418d-e0b8-41cebf16fd35"
if not exists('keypoint.py'):
  !wget https://raw.githubusercontent.com/facebookresearch/maskrcnn-benchmark/e0a525a0139baf7086117b7ed3fd318a4878d71c/maskrcnn_benchmark/structures/keypoint.py
    
from keypoint import PersonKeypoints
def vis_keypoints(img, kps, kp_thresh=2, alpha=0.7):
    """Visualizes keypoints (adapted from vis_one_image).
    kps has shape (4, #keypoints) where 4 rows are (x, y, logit, prob).
    """
    dataset_keypoints = PersonKeypoints.NAMES
    kp_lines = PersonKeypoints.CONNECTIONS

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw mid shoulder / mid hip first for better visualization.
    mid_shoulder = (
        kps[:2, dataset_keypoints.index('right_shoulder')] +
        kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
    sc_mid_shoulder = np.minimum(
        kps[2, dataset_keypoints.index('right_shoulder')],
        kps[2, dataset_keypoints.index('left_shoulder')])
    mid_hip = (
        kps[:2, dataset_keypoints.index('right_hip')] +
        kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
    sc_mid_hip = np.minimum(
        kps[2, dataset_keypoints.index('right_hip')],
        kps[2, dataset_keypoints.index('left_hip')])
    nose_idx = dataset_keypoints.index('nose')
    if sc_mid_shoulder > kp_thresh and kps[2, nose_idx] > kp_thresh:
        cv2.line(
            kp_mask, tuple(mid_shoulder), tuple(kps[:2, nose_idx]),
            color=colors[len(kp_lines)], thickness=2, lineType=cv2.LINE_AA)
    if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
        cv2.line(
            kp_mask, tuple(mid_shoulder), tuple(mid_hip),
            color=colors[len(kp_lines) + 1], thickness=2, lineType=cv2.LINE_AA)

    # Draw the keypoints.
    for l in range(len(kp_lines)):
        i1 = kp_lines[l][0]
        i2 = kp_lines[l][1]
        p1 = kps[0, i1], kps[1, i1]
        p2 = kps[0, i2], kps[1, i2]
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)
  
  
def overlay_keypoints(image, kps, scores):
  kps = torch.cat((kps[:, :, 0:2], scores[:, :, None]), dim=2).cpu().numpy()
  for region in kps:
    image = vis_keypoints(image, region.transpose((1, 0)))
  return image
```

```python colab={"base_uri": "https://localhost:8080/", "height": 249} id="uuNbi8s4X0Dr" executionInfo={"status": "ok", "timestamp": 1608555410708, "user_tz": -330, "elapsed": 13612, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a8839783-b67d-477a-bd74-4d2d87df6bad"
IMAGE_URL = 'https://raw.githubusercontent.com/facebookresearch/DensePose/master/DensePoseData/demo_data/demo_im.jpg'


image_file = basename(IMAGE_URL)
!wget -q -O {image_file} {IMAGE_URL}
plt.imshow(matplotlib.image.imread(image_file))
```

```python colab={"base_uri": "https://localhost:8080/"} id="NZrVr73KX3-j" executionInfo={"status": "ok", "timestamp": 1608555411587, "user_tz": -330, "elapsed": 5222, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="10ea2483-ccf3-4fb0-d72d-918f85554404"
t = time.time()
image = PIL.Image.open(image_file)
image_tensor = torchvision.transforms.functional.to_tensor(image).cuda()
output = model([image_tensor])[0]
print('executed in %.3fs' % (time.time() - t))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 668} id="bMviPqAUX6PQ" executionInfo={"status": "ok", "timestamp": 1608555421682, "user_tz": -330, "elapsed": 7272, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cd1955ae-988c-4101-9cf8-9b5160ddac99"
result_image = np.array(image.copy())
result_image = overlay_keypoints(result_image, output['keypoints'], output['keypoints_scores'])

plt.figure(figsize=(20, 15))
plt.imshow(result_image)
```
