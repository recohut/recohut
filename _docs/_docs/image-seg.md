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

# Image segmentation and Background removal

<!-- #region colab_type="text" id="IIDafmcXSNns" -->
An editted version of the demo notebook in https://github.com/matterport/Mask_RCNN
<!-- #endregion -->

<!-- #region colab_type="text" id="shQBS8Cd7qMr" -->
## Install libraries
<!-- #endregion -->

```python colab={} colab_type="code" id="N46-SNnB7kdT"
!pip install imgaug
!pip install Cython
!pip install pycocotools
!pip install kaggle
```

<!-- #region colab_type="text" id="9uB4cF2D7t_5" -->
## Clone Repo
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 121} colab_type="code" id="ttgRcakjuQ4P" outputId="4170e1cd-f9e8-45dd-f0aa-0858bd050193"
!git clone https://github.com/matterport/Mask_RCNN
```

```python colab={} colab_type="code" id="n2aH_LFLudLM"
import os 
os.chdir('Mask_RCNN/samples')
```

<!-- #region colab_type="text" id="FyO_T7_h7xHq" -->
## Prepare Model
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 69} colab_type="code" id="q3TqaKKzukha" outputId="e57fdbc2-2d64-4ef1-d7ef-5f0a103af458"
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

%matplotlib inline 

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
```

<!-- #region colab_type="text" id="pRNzMHg08Uvn" -->
## Create Inference Object
<!-- #endregion -->

```python colab={} colab_type="code" id="x-SLKOPeu0PY"
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 89} colab_type="code" id="j-C88yeKvWaa" outputId="c98ace76-8525-4b77-8896-8bbbe45219e2"
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)
```

```python colab={} colab_type="code" id="2rEErre3vY5m"
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
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

<!-- #region colab_type="text" id="6PTdUjUC76A8" -->
## Prediction and Visualization
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 653} colab_type="code" id="YBaILxtBPyBu" outputId="e9fc1bca-e7cc-48e3-e3ce-2c98bd76d778"
path = '../images/8829708882_48f263491e_z.jpg'
image = skimage.io.imread(path)

# Run detection
results = model.detect([image], verbose=0)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])
```

<!-- #region colab_type="text" id="15MmM7qjP4ZO" -->
## Background Removal
<!-- #endregion -->

```python colab={} colab_type="code" id="iHa0sU0oaa9E"
def segment(image, r):
  idx = r['scores'].argmax()
  mask = r['masks'][:,:,idx]
  mask = np.stack((mask,)*3, axis=-1)
  mask = mask.astype('uint8')
  bg = 255 - mask * 255
  mask_img = image*mask
  result = mask_img+ bg
  return result
```

```python colab={"base_uri": "https://localhost:8080/", "height": 366} colab_type="code" id="pzLmfWZdvg3E" outputId="885d7781-ff3a-4e88-e8c4-19e92e80f7a0"
segmentation = segment(image, r)
plt.subplots(1, figsize=(16, 16))
plt.axis('off')
plt.imshow(np.concatenate([image, segmentation], axis = 1))
```
