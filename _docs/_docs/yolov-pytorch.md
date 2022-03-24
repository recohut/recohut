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

<!-- #region id="w5KSKmal7FtZ" colab_type="text" -->
# Object Detection with YOLO v3

This notebook uses a [PyTorch port](https://github.com/ayooshkathuria/pytorch-yolo-v3) of [YOLO v3](https://pjreddie.com/darknet/yolo/) to detect objects on a given image.

For other deep-learning Colab notebooks, visit [tugstugi/dl-colab-notebooks](https://github.com/tugstugi/dl-colab-notebooks).


## Install ayooshkathuria/pytorch-yolo-v3
<!-- #endregion -->

```python id="ap5kDp1X69_s" colab_type="code" colab={}
import os
from os.path import exists, join, basename, splitext

git_repo_url = 'https://github.com/ayooshkathuria/pytorch-yolo-v3.git'
project_name = splitext(basename(git_repo_url))[0]
if not exists(project_name):
  # clone and install dependencies
  !git clone -q $git_repo_url
  #!cd $project_name && pip install -q -r requirement.txt
  
import sys
sys.path.append(project_name)
import time
import matplotlib
import matplotlib.pylab as plt
plt.rcParams["axes.grid"] = False
```

<!-- #region id="KEcjFp7U9aV6" colab_type="text" -->
## Download official YOLO v3 pretrained weights
<!-- #endregion -->

```python id="KSl-69n98Kpc" colab_type="code" colab={}
if not exists('yolov3.weights'):
  !wget -q https://pjreddie.com/media/files/yolov3.weights
```

<!-- #region id="wMxlaoWoBhHF" colab_type="text" -->
## Detect objects on a test image

First, dowload a test image from internet:
<!-- #endregion -->

```python id="P6ETX8scB7oj" colab_type="code" outputId="3e707fb6-fed6-45fe-8b2d-3695fdf8c56b" colab={"base_uri": "https://localhost:8080/", "height": 369}
IMAGE_URL = 'https://raw.githubusercontent.com/tugstugi/dl-colab-notebooks/master/resources/dog.jpg'


image_file = basename(IMAGE_URL)
!wget -q -O $image_file $IMAGE_URL
plt.imshow(matplotlib.image.imread(image_file))
```

<!-- #region id="FRQu6AkbC9w-" colab_type="text" -->
Execute `detect.py` on that image and show the result:
<!-- #endregion -->

```python id="RQZUaDTo8dvr" colab_type="code" outputId="3defd891-528f-4094-af52-f30894d4cacc" colab={"base_uri": "https://localhost:8080/", "height": 1227}
!cd pytorch-yolo-v3 && python detect.py --weights ../yolov3.weights --images ../$image_file --det ..

plt.figure(figsize=(20, 15))
plt.imshow(matplotlib.image.imread('det_%s' % image_file))
```
