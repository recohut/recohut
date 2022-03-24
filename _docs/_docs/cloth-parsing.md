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

<!-- #region id="J1WkGAIk1dNh" -->
# Cloth Parsing
<!-- #endregion -->

```python id="CCZiP5ryTKGX"
from scipy.io import loadmat
from matplotlib import pyplot as plt
import numpy as np
from os.path import basename
import shutil
import random
import os
import h5py
from PIL import Image
import pandas as pd
from glob import glob
import pickle

!apt install -y caffe-cuda
!pip install pydensecrf

import caffe
from pydensecrf import densecrf as dcrf
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, create_pairwise_gaussian, softmax_to_unary
import skimage.io as io

import warnings
warnings.filterwarnings('ignore')
```

```python id="yz0T0gWR9Faq" colab={"base_uri": "https://localhost:8080/", "height": 102} outputId="1d986163-9cd0-474c-86d8-660bff653299" executionInfo={"status": "ok", "timestamp": 1587661230785, "user_tz": -330, "elapsed": 25422, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
!git clone https://github.com/bearpaw/clothing-co-parsing.git
data_path = '/content/clothing-co-parsing/'
```

```python id="oT-ofHDo-kPd" colab={"base_uri": "https://localhost:8080/", "height": 187} outputId="0516919a-f043-4fb0-887b-207ea1cf3025" executionInfo={"status": "ok", "timestamp": 1587661604099, "user_tz": -330, "elapsed": 1776, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
loadmat(data_path+'annotations/pixel-level/0001.mat')
```

```python id="lqTuTNMv_Nze" colab={"base_uri": "https://localhost:8080/", "height": 487} outputId="b5d2c82e-1836-4ec3-9a57-23d3a57ef9c8" executionInfo={"status": "ok", "timestamp": 1587661777599, "user_tz": -330, "elapsed": 3256, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
plt.figure(figsize=(12, 8))

plt.subplot(1, 2, 1)
img1 = Image.open(data_path+'photos/0001.jpg')
plt.imshow(img1)

plt.subplot(1, 2, 2)
mask1 = loadmat(data_path+'annotations/pixel-level/0001.mat')['groundtruth']
plt.imshow(mask1)

plt.show()
```

```python id="v19lJxkfAzni" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="a2dbde8d-2505-4edd-cb7e-69c01c42e573" executionInfo={"status": "ok", "timestamp": 1587661950613, "user_tz": -330, "elapsed": 1474, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# reading names of categories in the ccp dataset and saving it as csv
labels = loadmat(data_path+'label_list.mat')
ccp_categories = []
for i in labels['label_list'][0]:
    ccp_categories.append(str(i[0]))
color_map = pd.Series(ccp_categories)
color_map
```
