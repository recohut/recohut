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

# Basic techniques for Image segmentation

```python colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" executionInfo={"elapsed": 1059, "status": "ok", "timestamp": 1585722608225, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="ig5H4QC5c5hW" outputId="f2db1fae-2a8e-4525-dea5-99afb9bec282"
%tensorflow_version 1.x
from skimage.color import rgb2gray
import numpy as np
import cv2
import matplotlib.pyplot as plt
%matplotlib inline
from scipy import ndimage
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} colab_type="code" executionInfo={"elapsed": 6008, "status": "ok", "timestamp": 1585722616350, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="Kc7WWfHpfatL" outputId="2811813d-1ee5-421b-f878-369c27059d50"
!wget 'https://www.dike.lib.ia.us/images/sample-1.jpg'
```

```python colab={"base_uri": "https://localhost:8080/", "height": 286} colab_type="code" executionInfo={"elapsed": 4344, "status": "ok", "timestamp": 1585722616351, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="9oKZbmzufh8k" outputId="10f3297c-12cc-4956-ee0c-d886045582d8"
image = plt.imread('sample-1.jpg')/255
image.shape
plt.imshow(image)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 286} colab_type="code" executionInfo={"elapsed": 3848, "status": "ok", "timestamp": 1585722616352, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="4sK0LN06f4zX" outputId="cd5f962b-dfaa-4bf3-c9d3-56fb62b02131"
gray = rgb2gray(image)
plt.imshow(gray, cmap='gray')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 286} colab_type="code" executionInfo={"elapsed": 10362, "status": "ok", "timestamp": 1585722623357, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="Q6bjQMrvf7sD" outputId="fa68708f-a219-4f98-a015-edd4aa8f68a6"
gray_r = gray.reshape(gray.shape[0]*gray.shape[1])
for i in range(gray_r.shape[0]):
    if gray_r[i] > gray_r.mean():
        gray_r[i] = 1
    else:
        gray_r[i] = 0
gray = gray_r.reshape(gray.shape[0],gray.shape[1])
plt.imshow(gray, cmap='gray')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 286} colab_type="code" executionInfo={"elapsed": 17803, "status": "ok", "timestamp": 1585722631222, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="y-5_fcsYgweB" outputId="df733c50-c05f-4e40-d300-0d843a33322e"
gray = rgb2gray(image)
gray_r = gray.reshape(gray.shape[0]*gray.shape[1])
for i in range(gray_r.shape[0]):
    if gray_r[i] > gray_r.mean():
        gray_r[i] = 3
    elif gray_r[i] > 0.5:
        gray_r[i] = 2
    elif gray_r[i] > 0.25:
        gray_r[i] = 1
    else:
        gray_r[i] = 0
gray = gray_r.reshape(gray.shape[0],gray.shape[1])
plt.imshow(gray, cmap='gray')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" executionInfo={"elapsed": 17427, "status": "ok", "timestamp": 1585722631225, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="c8-1lqU9js5C" outputId="3b369896-6862-4a32-cf38-fa5646aca625"
pic_n = image.reshape(image.shape[0]*image.shape[1], image.shape[2])
pic_n.shape
```

```python colab={} colab_type="code" id="gfOCAoomj0Ek"
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=0).fit(pic_n)
pic2show = kmeans.cluster_centers_[kmeans.labels_]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 286} colab_type="code" executionInfo={"elapsed": 18276, "status": "ok", "timestamp": 1585722632989, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="ilHNtqSlj2t3" outputId="51748ce7-2ec1-45bd-a830-0325f0774741"
cluster_pic = pic2show.reshape(image.shape[0], image.shape[1], image.shape[2])
plt.imshow(cluster_pic)
```

<!-- #region colab_type="text" id="wbexAmohVzvU" -->
## Mask R-CNN
<!-- #endregion -->

```python colab={} colab_type="code" id="yMzTkxRZW40z"
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
```

```python colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" executionInfo={"elapsed": 3272, "status": "ok", "timestamp": 1585722638238, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="gFSL2Ajmj4-I" outputId="a98fbbf8-3b36-4439-8c74-dcc3f31d2272"
!git clone https://github.com/matterport/Mask_RCNN.git
sys.path.append('Mask_RCNN/')
sys.path.append('Mask_RCNN/samples/coco/')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" executionInfo={"elapsed": 5261, "status": "ok", "timestamp": 1585722643130, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="qwnZlxDQV4xc" outputId="1eb500c6-db5f-43e6-870c-200a87c0a835"
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
import coco
```

```python colab={"base_uri": "https://localhost:8080/", "height": 309} colab_type="code" executionInfo={"elapsed": 8647, "status": "ok", "timestamp": 1585722648483, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="Uzod4CaqaMk2" outputId="348ad9d7-9cf3-41d8-f917-361c4260bdb8"
!wget 'https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5'
COCO_MODEL_PATH = 'mask_rcnn_coco.h5'
```

```python colab={} colab_type="code" id="QJfsehfra815"
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
# config.display()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 700} colab_type="code" executionInfo={"elapsed": 14459, "status": "ok", "timestamp": 1585722657834, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="0UJGKZc7Z393" outputId="b0f16325-32f2-4961-b2e6-701a06a75dd7"
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir='mask_rcnn_coco.hy', config=config)

# Load weights trained on MS-COCO
model.load_weights('mask_rcnn_coco.h5', by_name=True)
```

```python colab={} colab_type="code" id="IoV8WfkwZygd"
# COCO Class names
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
```

```python colab={"base_uri": "https://localhost:8080/", "height": 680} colab_type="code" executionInfo={"elapsed": 2340, "status": "ok", "timestamp": 1585722665524, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="IzFg_FhfZO6A" outputId="bbc74abf-91c3-4be7-dda5-b3ce23b1580a"
# Load a random image from the images folder
image = skimage.io.imread('sample-1.jpg')

# original image
plt.figure(figsize=(12,10))
skimage.io.imshow(image)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 796} colab_type="code" executionInfo={"elapsed": 18514, "status": "ok", "timestamp": 1585722691621, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="5QIwpNC1Xcqq" outputId="4e42abc2-af80-4cba-8a95-5d684bef8d31"
# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" executionInfo={"elapsed": 1201, "status": "ok", "timestamp": 1585724831412, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="OWknJyB7b8dG" outputId="0f4425c4-feca-4573-f3a1-36d6c57307b5"
import cv2
import numpy as np

ann_img = np.zeros((300,300,3)).astype('uint8')
ann_img[100:200, 40:100] = 100
ann_img[200:250, 150:250] = 200

cv2.imwrite( "ann_1.png" ,ann_img )
```

<!-- #region colab_type="text" id="bgkhppmjnoDy" -->
## Keras Segment Library
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 803} colab_type="code" executionInfo={"elapsed": 8897, "status": "ok", "timestamp": 1585725805421, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="gWJGCeStjWRu" outputId="9d603afd-9905-4970-e550-538b1bb646a9"
!pip install git+https://github.com/divamgupta/image-segmentation-keras
```

```python colab={} colab_type="code" id="4XuJoGqynq0S"
# from keras_segmentation.pretrained import pspnet_101_cityscapes
# model = pspnet_101_cityscapes() # load the pretrained model trained on Cityscapes dataset
out = model.predict_segmentation(inp="sample-1.jpg", out_fname="out.png")
```
