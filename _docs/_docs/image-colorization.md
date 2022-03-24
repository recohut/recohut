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

<!-- #region id="vDM_0zW6uM24" -->
# Image Colorization
<!-- #endregion -->

<!-- #region id="KnMeI8ow1203" -->
## Environment Setting
<!-- #endregion -->

```python id="xccN7_1QU-ac"
!pip install -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html
!pip install cython pyyaml==5.1
!pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
!pip install dominate==2.4.0
!pip install detectron2==0.1.2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/index.html
```

```python id="tWOgZIl_sxlq"
!git clone https://github.com/sparsh-ai/InstColorization.git
```

<!-- #region id="L65jjMKVpfkH" -->
## Start Colorization
<!-- #endregion -->

```python id="lIzoQKQhdTIl" outputId="c3067d61-0a79-43f2-da71-8da1e71836d8" colab={"base_uri": "https://localhost:8080/", "height": 33} executionInfo={"status": "ok", "timestamp": 1590259775388, "user_tz": -330, "elapsed": 55, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
cd InstColorization/
```

<!-- #region id="RMSD5cwV57Yt" -->
You can also follow the step below to get results directly.


```
!sh scripts/test_mask.sh
```


<!-- #endregion -->

<!-- #region id="92YnbNtU2cdx" -->
### Detect Object bounding box

<!-- #endregion -->

<!-- #region id="3Z8V8s3f3cs0" -->
Setting the Detectron2.
<!-- #endregion -->

```python id="ngV-n2MbvvTZ" outputId="cf75247d-3032-44ae-c3ce-bceb3e5ff828" colab={"base_uri": "https://localhost:8080/", "height": 33} executionInfo={"status": "ok", "timestamp": 1590259807980, "user_tz": -330, "elapsed": 32604, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
from os.path import join, isfile, isdir
from os import listdir
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from argparse import ArgumentParser

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

import torch

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
```

<!-- #region id="HUfLSSQf32SJ" -->
Let's create a bounding box folder to save our prediction results.
<!-- #endregion -->

```python id="b62LZaSTxkiQ"
!rm -rf /content/InstColorization/example/*
!wget -O '/content/InstColorization/example/example.jpg' 'https://www.theawl.com/wp-content/uploads/2015/09/0j3kZ7OI-xi51YDPw.jpg'
```

```python id="Y5RSCOwA4Cum"
input_dir = "example"
image_list = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
output_npz_dir = "{0}_bbox".format(input_dir)
if os.path.isdir(output_npz_dir) is False:
    print('Create path: {0}'.format(output_npz_dir))
    os.makedirs(output_npz_dir)
```

<!-- #region id="hBlhTIoA4YSB" -->
Here we simply take L channel as our input and make sure that we can get consistent box prediction results even though the original image is color images.
<!-- #endregion -->

```python id="60Z0uIQH4ztv"
for image_path in image_list:
    img = cv2.imread(join(input_dir, image_path))
    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    l_stack = np.stack([l_channel, l_channel, l_channel], axis=2)
    outputs = predictor(l_stack)
    save_path = join(output_npz_dir, image_path.split('.')[0])
    pred_bbox = outputs["instances"].pred_boxes.to(torch.device('cpu')).tensor.numpy()
    pred_scores = outputs["instances"].scores.cpu().data.numpy()
    np.savez(save_path, bbox = pred_bbox, scores = pred_scores)
```

<!-- #region id="4Z4y8giu48hV" -->
Now we have all the images' prediction results.
<!-- #endregion -->

```python id="541hJilN5Iox" outputId="27acc5b9-dca7-452d-9564-fc4f799af764" colab={"base_uri": "https://localhost:8080/", "height": 33} executionInfo={"status": "ok", "timestamp": 1590260392262, "user_tz": -330, "elapsed": 4098, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
!ls example_bbox
```

<!-- #region id="fYvWpoUl5VKX" -->
### Colorize Images
<!-- #endregion -->

<!-- #region id="_0qhXEQ45bxc" -->
We first set up some libraries and options
<!-- #endregion -->

```python id="g80xLXzi9tOB"
import sys
import time
from options.train_options import TestOptions
from models import create_model

import torch
from tqdm import tqdm_notebook

from fusion_dataset import Fusion_Testing_Dataset
from util import util
import multiprocessing
multiprocessing.set_start_method('spawn', True)

torch.backends.cudnn.benchmark = True

sys.argv = [sys.argv[0]]
opt = TestOptions().parse()
```

<!-- #region id="xsMYnRQeKQEw" -->
Then we need to create a results folder to save our predicted color images and read the dataset loader.
<!-- #endregion -->

```python id="KTWCeb2iEWFM" cellView="code" outputId="2960d56d-c6be-4a7b-9b46-06d35aa16a01" colab={"base_uri": "https://localhost:8080/", "height": 50} executionInfo={"status": "ok", "timestamp": 1590260396368, "user_tz": -330, "elapsed": 1449, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
save_img_path = opt.results_img_dir
if os.path.isdir(save_img_path) is False:
    print('Create path: {0}'.format(save_img_path))
    os.makedirs(save_img_path)
opt.batch_size = 1
dataset = Fusion_Testing_Dataset(opt)
dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size)

dataset_size = len(dataset)
print('#Testing images = %d' % dataset_size)
```

<!-- #region id="6aBEbi-vKgLG" -->
Load the pre-trained model.
<!-- #endregion -->

```python id="fM9oF4OdwKjZ" colab={"base_uri": "https://localhost:8080/", "height": 233} outputId="0f0df2fb-9c84-412e-f404-7e1cb594acb0" executionInfo={"status": "ok", "timestamp": 1590260418845, "user_tz": -330, "elapsed": 23621, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
!sh scripts/download_model.sh
```

```python id="kpw5UGUWImIq" outputId="0a90574f-a12b-42ac-b4ba-3681797d93f7" colab={"base_uri": "https://localhost:8080/", "height": 100} executionInfo={"status": "ok", "timestamp": 1590260420126, "user_tz": -330, "elapsed": 24762, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
model = create_model(opt)
model.setup_to_test('coco_finetuned_mask_256_ffs')
```

<!-- #region id="vornjFjuKlzu" -->
Start to colorize every images in `dataset_loader`.
<!-- #endregion -->

```python id="mwy1Tvh8Iuzm" outputId="78687c66-b77e-4787-deb7-27c8be45eeb9" colab={"base_uri": "https://localhost:8080/", "height": 185, "referenced_widgets": ["4bfd691461484f9bbdea9b467a8fbde7", "3924236a1b824671b8a31f104a317b3c", "b10bf890019f4b38a679818a71f75bfb", "54fb2c2bf1b44cdf928a05f5c8509d42", "f28866d5435f4c298f1b88856f094f93", "d4a7f24daf694a55b0eaa4885801fdd3", "1916e14cb6414946aeee24094684423c", "b1cefce0f48944a18bcae044e4998350"]} executionInfo={"status": "ok", "timestamp": 1590260433893, "user_tz": -330, "elapsed": 10061, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
count_empty = 0
for data_raw in tqdm_notebook(dataset_loader):
    data_raw['full_img'][0] = data_raw['full_img'][0].cuda()
    if data_raw['empty_box'][0] == 0:
        data_raw['cropped_img'][0] = data_raw['cropped_img'][0].cuda()
        box_info = data_raw['box_info'][0]
        box_info_2x = data_raw['box_info_2x'][0]
        box_info_4x = data_raw['box_info_4x'][0]
        box_info_8x = data_raw['box_info_8x'][0]
        cropped_data = util.get_colorization_data(data_raw['cropped_img'], opt, ab_thresh=0, p=opt.sample_p)
        full_img_data = util.get_colorization_data(data_raw['full_img'], opt, ab_thresh=0, p=opt.sample_p)
        model.set_input(cropped_data)
        model.set_fusion_input(full_img_data, [box_info, box_info_2x, box_info_4x, box_info_8x])
        model.forward()
    else:
        count_empty += 1
        full_img_data = util.get_colorization_data(data_raw['full_img'], opt, ab_thresh=0, p=opt.sample_p)
        model.set_forward_without_box(full_img_data)
    model.save_current_imgs(join(save_img_path, data_raw['file_id'][0] + '.png'))
print('{0} images without bounding boxes'.format(count_empty))
```

```python id="j1AI2ydNJ5yu" outputId="40068a55-6d6d-4c71-8a1c-a11ccf1e5459" colab={"base_uri": "https://localhost:8080/", "height": 357} executionInfo={"status": "ok", "timestamp": 1590260604466, "user_tz": -330, "elapsed": 2860, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
from google.colab.patches import cv2_imshow
# img_name_list = ['000000022969', '000000023781', '000000046872', '000000050145']
# show_index = 1

img_name_list = ['example']
show_index = 0

img = cv2.imread('example/'+img_name_list[show_index]+'.jpg')
lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l_channel, _, _ = cv2.split(lab_image)

img = cv2.imread('results/'+img_name_list[show_index]+'.png')
lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
_, a_pred, b_pred = cv2.split(lab_image)
a_pred = cv2.resize(a_pred, (l_channel.shape[1], l_channel.shape[0]))
b_pred = cv2.resize(b_pred, (l_channel.shape[1], l_channel.shape[0]))
gray_color = np.ones_like(a_pred) * 128

gray_image = cv2.cvtColor(np.stack([l_channel, gray_color, gray_color], 2), cv2.COLOR_LAB2BGR)
color_image = cv2.cvtColor(np.stack([l_channel, a_pred, b_pred], 2), cv2.COLOR_LAB2BGR)

cv2_imshow(np.concatenate([gray_image, color_image], 1))
```
