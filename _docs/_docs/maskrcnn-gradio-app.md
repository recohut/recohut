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

<!-- #region id="PTiG8grE1K7E" -->
# MaskRCNN Object Detection Gradio App
<!-- #endregion -->

```python id="JTTy0zx6yrZE"
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
```

```python id="pDUAhA5Fy8UU" colab={"base_uri": "https://localhost:8080/", "height": 103, "referenced_widgets": ["cbb845940574454abcf26d0da72820ed", "2b823f5d08844f43b8a05c99a7aed91d", "a9ac8c7fdeeb4c97aba5034cc67c35cd", "e26cc1b0422345a7814dcba46a948950", "ff928f305e6e4be5b0cbea2ad2b9c04c", "a486f075818048228952f30300c09a06", "d92d7f88c6944afb94ee3f8ce65c2e0b", "d5a3060ce82d4da5b9ddd42426babda9"]} executionInfo={"status": "ok", "timestamp": 1595392411655, "user_tz": -330, "elapsed": 21466, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="de1c4e0a-087a-4164-b912-003818341f0a"
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model = model.eval().cuda()
```

```python id="THCvkna-y8yS" colab={"base_uri": "https://localhost:8080/", "height": 286} executionInfo={"status": "ok", "timestamp": 1595392431522, "user_tz": -330, "elapsed": 5356, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6b431ee2-1278-498c-ad8d-83a7d5def507"
IMAGE_URL = 'https://raw.githubusercontent.com/tugstugi/dl-colab-notebooks/master/resources/dog.jpg'

image_file = basename(IMAGE_URL)
!wget -q -O {image_file} {IMAGE_URL}
plt.imshow(matplotlib.image.imread(image_file))
```

```python id="vX1hMBYyzIAl" colab={"base_uri": "https://localhost:8080/", "height": 156} executionInfo={"status": "ok", "timestamp": 1595392434046, "user_tz": -330, "elapsed": 1917, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5ead2489-ae39-4c05-e85a-b81fbcab3d45"
t = time.time()
image = PIL.Image.open(image_file)
image_tensor = torchvision.transforms.functional.to_tensor(image).cuda()
output = model([image_tensor])[0]
print('executed in %.3fs' % (time.time() - t))
```

```python id="6G3XWoumzJeG" colab={"base_uri": "https://localhost:8080/", "height": 884} executionInfo={"status": "ok", "timestamp": 1595392443764, "user_tz": -330, "elapsed": 5596, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8b890966-b2c5-4746-be76-60b24f57b07b"
coco_names = ['unlabeled', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
colors = [[random.randint(0, 255) for _ in range(3)] for _ in coco_names]

result_image = np.array(image.copy())
for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
  if score > 0.5:
    color = random.choice(colors)
    
    # draw box
    tl = round(0.002 * max(result_image.shape[0:2])) + 1  # line thickness
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(result_image, c1, c2, color, thickness=tl)
    # draw text
    display_txt = "%s: %.1f%%" % (coco_names[label], 100*score)
    tf = max(tl - 1, 1)  # font thickness
    t_size = cv2.getTextSize(display_txt, 0, fontScale=tl / 3, thickness=tf)[0]
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    cv2.rectangle(result_image, c1, c2, color, -1)  # filled
    cv2.putText(result_image, display_txt, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
  
plt.figure(figsize=(20, 15))
plt.imshow(result_image)
```

```python id="LqSwGCgjzK8F" colab={"base_uri": "https://localhost:8080/", "height": 286} executionInfo={"status": "ok", "timestamp": 1595392446253, "user_tz": -330, "elapsed": 1599, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="091f4fef-6add-4d94-eb5d-5bf3cf4379ac"
masks = None
for score, mask in zip(output['scores'], output['masks']):
  if score > 0.5:
    if masks is None:
      masks = mask
    else:
      masks = torch.max(masks, mask)

plt.imshow(masks.squeeze(0).cpu().numpy())
```

<!-- #region id="8CosLJwezNY0" -->
---
<!-- #endregion -->

```python id="QEOWSefE0BH1"
def func(image):
  
  image_tensor = torchvision.transforms.functional.to_tensor(image).cuda()
  output = model([image_tensor])[0]
  coco_names = ['unlabeled', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
  colors = [[random.randint(0, 255) for _ in range(3)] for _ in coco_names]

  result_image = np.array(image.copy())
  for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
    if score > 0.5:
      color = random.choice(colors)
      
      # draw box
      tl = round(0.002 * max(result_image.shape[0:2])) + 1  # line thickness
      c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
      cv2.rectangle(result_image, c1, c2, color, thickness=tl)
      # draw text
      display_txt = "%s: %.1f%%" % (coco_names[label], 100*score)
      tf = max(tl - 1, 1)  # font thickness
      t_size = cv2.getTextSize(display_txt, 0, fontScale=tl / 3, thickness=tf)[0]
      c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
      cv2.rectangle(result_image, c1, c2, color, -1)  # filled
      cv2.putText(result_image, display_txt, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    
  return result_image
```

```python id="DbvrThoB0sWO" colab={"base_uri": "https://localhost:8080/", "height": 71} executionInfo={"status": "ok", "timestamp": 1595393069091, "user_tz": -330, "elapsed": 1366, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8e162035-4156-4391-9e90-07d8bfae285b"
xx = func(PIL.Image.open(image_file))
```

```python id="1-aMzOi_zMgY" colab={"base_uri": "https://localhost:8080/", "height": 102} executionInfo={"status": "ok", "timestamp": 1595392675265, "user_tz": -330, "elapsed": 15169, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="be6446ab-da71-4736-e4fc-5001dcbb018f"
! pip install -q gradio
import gradio as gr
```

```python id="UTx7qJVs1mUg"
input = gr.inputs.Image()
output = gr.outputs.Image()
```

```python id="sYzLIbGa10Q7" colab={"base_uri": "https://localhost:8080/", "height": 623} executionInfo={"status": "ok", "timestamp": 1595393278987, "user_tz": -330, "elapsed": 13515, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="256b2828-4477-40a7-f9d3-7da48bf6252a"
gr.Interface(fn=func, inputs=input, outputs=output).launch()
```

<!-- #region id="w8v2MiTm5I6b" -->
<!-- #endregion -->

```python id="uTFQFVs42TmU"

```
