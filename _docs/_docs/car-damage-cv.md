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
    language: python
    name: python3
---

<!-- #region id="8du19s48c_OJ" -->
# MaskRCNN Car Damage Prediction
<!-- #endregion -->

<!-- #region id="tYnnncKKcoG-" -->
> Note: This jupyter notebook contains data visualization of car damage images and automated car damage detection example. First we need to import all the packages including custom functions of Matterport Mask R-CNN’ repository 
<!-- #endregion -->

```python id="5UgA2omYcoHD"
import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon

# Import Mask RCNN
#sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn import model
import mrcnn.model as modellib
from mrcnn.model import log
import cv2
import custom,custom_1
import imgaug,h5py,IPython

%matplotlib inline
```

<!-- #region id="FmAc1JsgcoHH" -->
Setting up the configuration - root directory,data path setting up the ,log file path and model object(weight matrix)for inference (prediction) 
<!-- #endregion -->

```python id="KVULJKXJcoHH" outputId="d8e24325-6149-45da-adbf-405e822b8d3b"
# Root directory of the project
ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)  # To find local version of the library
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
custom_WEIGHTS_PATH = "mask_rcnn_scratch_0013.h5"  # TODO: update this path for best performing iteration weights
config = custom.CustomConfig()
custom_DIR = os.path.join(ROOT_DIR, "custom/")
custom_DIR
```

<!-- #region id="qfA1ISDecoHJ" -->
loading the data
<!-- #endregion -->

```python id="xYjqdN3fcoHJ" outputId="d663ad8f-6909-4f68-be6a-ca29e764a228"
# Load dataset
dataset = custom_1.CustomDataset()
dataset.load_custom(custom_DIR, "train")

# Must call before using the dataset
dataset.prepare()

print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))
```

<!-- #region id="IJCVJ8kLcoHL" -->
We will visualize few car damage(scratch) images
<!-- #endregion -->

```python id="NmpX9WVQcoHM" outputId="f5fc587b-6cd5-4f10-ea42-503fe6c772d6"
# Load and display random samples
image_ids = np.random.choice(dataset.image_ids, 5)
for image_id in image_ids:
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset.class_names)
```

<!-- #region id="qdvdFSTDcoHM" -->
Next we will see Bounding Box(BB)with annotated damage mask for a typical car image.
<!-- #endregion -->

```python id="Sa05AFO1coHN" outputId="c52d88d8-91a2-49e8-f2de-027105166216"
image_id = random.choice(dataset.image_ids)
image = dataset.load_image(image_id)
mask, class_ids = dataset.load_mask(image_id)
# Compute Bounding box
bbox = utils.extract_bboxes(mask)

# Display image and additional stats
print("image_id ", image_id, dataset.image_reference(image_id))
log("image", image)
log("mask", mask)
log("class_ids", class_ids)
log("bbox", bbox)
# Display image and instances
visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)
```

<!-- #region id="B8a4tscmcoHP" -->
We see some the components of image annotations. Mainly it has x and y co-ordinate of all labeled damages('polygon') and class name(here 'scratch') for respective car image.
<!-- #endregion -->

```python id="deZane_PcoHP" outputId="9a09fee1-fa29-4476-e605-fe18ed5466c2"
#Annotation file load
annotations1 = json.load(open(os.path.join(ROOT_DIR, "via_region_data.json"),encoding="utf8"))
annotations = list(annotations1.values()) 
annotations = [a for a in annotations if a['regions']]
annotations[0]
```

<!-- #region id="nRKw0twacoHQ" -->
If we have to quantify a car damage,we need to know the x and y coordinates of the polygon to calculate area of the marked/detected damage.This is for 2nd damage polygon of 'image2.jpg'
<!-- #endregion -->

```python id="a39ZBvLacoHR" outputId="ec13d61d-f0de-4b5f-dc60-cf73395c51d3"
annotations[1]['regions']['0']['shape_attributes']
l = []
for d in annotations[1]['regions']['0']['shape_attributes'].values():
    l.append(d)
display('x co-ordinates of the damage:',l[1])    
display('y co-ordinates of the damage:',l[2])
```

<!-- #region id="j4QELyZDcoHS" -->
For prediction or damage detection we need to use the model as inference mode. Model description is consists of important model information like CNN architecture name('resnet101'), ROI threshold(0.9 as defined),configuration description, weightage of different loss components, mask shape, WEIGHT_DECAY etc. 

<!-- #endregion -->

```python id="Kg3afNmwcoHT" outputId="0574999c-cff1-44ea-f7a0-862c201893bd"
config = custom.CustomConfig()
ROOT_DIR = 'C:/Users/Sourish/Mask_RCNN'
CUSTOM_DIR = os.path.join(ROOT_DIR + "/custom/")
print(CUSTOM_DIR)
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"
```

<!-- #region id="e9RZW0f1coHU" -->
Helper function to visualize predicted damage masks and loading the model weights for prediction
<!-- #endregion -->

```python id="gg_X-nkGcoHU" outputId="7aaa7b3e-62cc-4acd-da56-26483e3a2459"
def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

from importlib import reload # was constantly changin the visualization, so I decided to reload it instead of notebook
reload(visualize)

# Create model in inference mode
import tensorflow as tf
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

# load the last best model you trained
# weights_path = model.find_last()[1]
custom_WEIGHTS_PATH = 'C:/Users/Sourish/Mask_RCNN/logs/scratch20190612T2046/mask_rcnn_scratch_0013.h5'
# Load weights
print("Loading weights ", custom_WEIGHTS_PATH)
model.load_weights(custom_WEIGHTS_PATH, by_name=True)    
```

<!-- #region id="2WESvvZRcoHV" -->
Loading validation data-set for prediction
<!-- #endregion -->

```python id="trdEzyC2coHc" outputId="462f89a7-f93b-4a68-a487-041c0e7271f7"
dataset = custom_1.CustomDataset()
dataset.load_custom(CUSTOM_DIR,'val')
dataset.prepare()
print('Images: {}\nclasses: {}'.format(len(dataset.image_ids), dataset.class_names))
```

<!-- #region id="NsJiLesGcoHf" -->
Visualize model weight matrix descriptive statistics(shapes, histograms)  
<!-- #endregion -->

```python id="UIQzMlZzcoHf" outputId="d7ccb76c-9d5e-4f5c-e6d7-09f1f56b3244"
visualize.display_weight_stats(model)
```

<!-- #region id="fFx2Zi0bcoHg" -->
Prediction on a random validation image(image53.jpeg)
<!-- #endregion -->

```python id="74UNwRYDcoHh" outputId="e6656422-c738-4ac5-d57c-9af5a1875a73"
image_id = random.choice(dataset.image_ids)
image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
info = dataset.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                       dataset.image_reference(image_id)))

# Run object detection
results = model.detect([image], verbose=1)

# Display results
ax = get_ax(1)
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            dataset.class_names, r['scores'], ax=ax,
                            title="Predictions")
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)
print('The car has:{} damages'.format(len(dataset.image_info[image_id]['polygons'])))
```

<!-- #region id="Nf6VnDF6coHi" -->
On another image
<!-- #endregion -->

```python id="oO-lT9hYcoHi" outputId="86351bd8-f448-4100-ade8-26b7d5a4505b"
image_id = random.choice(dataset.image_ids)
image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
info = dataset.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                       dataset.image_reference(image_id)))

# Run object detection
results = model.detect([image], verbose=1)

# Display results
ax = get_ax(1)
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            dataset.class_names, r['scores'], ax=ax,
                            title="Predictions")
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)
print('The car has:{} damages'.format(len(dataset.image_info[image_id]['polygons'])))
```

<!-- #region id="MdAVJPB1coHj" -->
Pretty decent prediction considering training with only 49 images and 15 epochs.
<!-- #endregion -->
