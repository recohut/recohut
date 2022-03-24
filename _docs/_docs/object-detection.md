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

<!-- #region id="4fTBjfPFxWWH" -->
# Various Object Detection Models
<!-- #endregion -->

<!-- #region id="SSuS6d4dCtPw" -->
## TFHub
<!-- #endregion -->

```python id="05c06pw9AadO"
import tensorflow as tf
import tensorflow_hub as hub

import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO

import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

import time
```

<!-- #region id="LNgZ1nQrA341" -->
Helper functions for downloading images and for visualization.

Visualization code adapted from TF object detection API for the simplest required functionality.
<!-- #endregion -->

```python id="zE5-51yZAxoe"
def display_image(image):
  fig = plt.figure(figsize=(20, 15))
  plt.grid(False)
  plt.imshow(image)


def download_and_resize_image(url, new_width=256, new_height=256,
                              display=False):
  _, filename = tempfile.mkstemp(suffix=".jpg")
  response = urlopen(url)
  image_data = response.read()
  image_data = BytesIO(image_data)
  pil_image = Image.open(image_data)
  pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
  pil_image_rgb = pil_image.convert("RGB")
  pil_image_rgb.save(filename, format="JPEG", quality=90)
  print("Image downloaded to %s." % filename)
  if display:
    display_image(pil_image)
  return filename


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
  """Adds a bounding box to an image."""
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
  draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = top + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom)],
                   fill=color)
    draw.text((left + margin, text_bottom - text_height - margin),
              display_str,
              fill="black",
              font=font)
    text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
  """Overlay labeled boxes on an image with formatted scores and label names."""
  colors = list(ImageColor.colormap.values())

  try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                              25)
  except IOError:
    print("Font not found, using default font.")
    font = ImageFont.load_default()

  for i in range(min(boxes.shape[0], max_boxes)):
    if scores[i] >= min_score:
      ymin, xmin, ymax, xmax = tuple(boxes[i])
      display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                     int(100 * scores[i]))
      color = colors[hash(class_names[i]) % len(colors)]
      image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
      draw_bounding_box_on_image(
          image_pil,
          ymin,
          xmin,
          ymax,
          xmax,
          color,
          font,
          display_str_list=[display_str])
      np.copyto(image, np.array(image_pil))
  return image

def load_img(path):
  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img, channels=3)
  return img

def run_detector(detector, path):
  img = load_img(path)

  converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
  start_time = time.time()
  result = detector(converted_img)
  end_time = time.time()

  result = {key:value.numpy() for key,value in result.items()}

  print("Found %d objects." % len(result["detection_scores"]))
  print("Inference time: ", end_time-start_time)

  image_with_boxes = draw_boxes(
      img.numpy(), result["detection_boxes"],
      result["detection_class_entities"], result["detection_scores"])

  display_image(image_with_boxes)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 815} id="eS2hXoadA9O4" executionInfo={"status": "ok", "timestamp": 1608549426696, "user_tz": -330, "elapsed": 8094, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f6902132-3a64-48c4-82be-7484da06bcfe"
image_url = "https://upload.wikimedia.org/wikipedia/commons/6/60/Naxos_Taverna.jpg"
downloaded_image_path = download_and_resize_image(image_url, 1280, 856, True)
```

```python colab={"base_uri": "https://localhost:8080/"} id="uZ_jJ-POBEXp" executionInfo={"status": "ok", "timestamp": 1608549756869, "user_tz": -330, "elapsed": 74840, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="28805904-0c93-4a20-f076-97d381d5d423"
module_handle = "Inception ResNet" #@param ["Inception ResNet", "SSD MobileNet"]

tfhub_module = {"Inception ResNet":"https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1",
                "SSD MobileNet":"https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1",
                }

module_handle = tfhub_module[module_handle]

detector = hub.load(module_handle).signatures['default']
```

```python colab={"base_uri": "https://localhost:8080/", "height": 832} id="zx_CH6qtCEqK" executionInfo={"status": "ok", "timestamp": 1608549798316, "user_tz": -330, "elapsed": 39466, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c0130e11-59a8-4f76-bb2b-a58d93308b19"
run_detector(detector, downloaded_image_path)
```

<!-- #region id="Csf1eE1oCwIS" -->
## PyTorch YOLO3
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="KVl3R9W9C_lF" executionInfo={"status": "ok", "timestamp": 1608549930679, "user_tz": -330, "elapsed": 1226, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1c44703e-5f7d-466f-da93-8ee6ff6d8061"
%cd /content
```

<!-- #region id="qKxxU5TXDE0j" -->
Install ayooshkathuria/pytorch-yolo-v3
<!-- #endregion -->

```python id="Set7H1ezCXb9"
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

<!-- #region id="raY-klmPDHCx" -->
Download official YOLO v3 pretrained weights
<!-- #endregion -->

```python id="yicVQJpxDB91"
if not exists('yolov3.weights'):
  !wget -q https://pjreddie.com/media/files/yolov3.weights
```

<!-- #region id="CHOvqJAGDSzq" -->
Download a test image
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 815} id="K0QnrODyDKEF" executionInfo={"status": "ok", "timestamp": 1608550043926, "user_tz": -330, "elapsed": 7586, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cedf5aff-956c-4aef-96cd-abbe489e572f"
image_url = "https://upload.wikimedia.org/wikipedia/commons/6/60/Naxos_Taverna.jpg"
downloaded_image_path = download_and_resize_image(image_url, 1280, 856, True)
```

<!-- #region id="TsgGkF3ODioH" -->
Execute detect.py on that image and show the result:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="kYeBKZxpDbLf" executionInfo={"status": "ok", "timestamp": 1608550251086, "user_tz": -330, "elapsed": 25677, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5f24e517-f8b6-451d-c959-f920dd963f94"
image_file = 'tmpnj5it70r.jpg'

!cd pytorch-yolo-v3 && python detect.py --weights ../yolov3.weights --images /tmp/$image_file --det ..

plt.figure(figsize=(20, 15))
plt.imshow(matplotlib.image.imread('det_%s' % image_file))
```

<!-- #region id="ufeYxwTXEi3p" -->
## PyTorch SSD
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="mH1VrVJlDtTx" executionInfo={"status": "ok", "timestamp": 1608550347035, "user_tz": -330, "elapsed": 927, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="148f32c2-9497-4ae4-e375-5cdc022243b3"
%cd /content
```

```python id="PkbboE_OHDtk"
import torch
precision = 'fp32'
ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision)

# For convenient and comprehensive formatting of input and output of the model, load a set of utility methods
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

# Prepare the model for inference
ssd_model.to('cuda')
ssd_model.eval()
```

<!-- #region id="l0MR72jfIe-7" -->
Load sample images
<!-- #endregion -->

```python id="FDTa0mCKIf06"
uris = [
    'http://images.cocodataset.org/val2017/000000397133.jpg',
    'http://images.cocodataset.org/val2017/000000037777.jpg',
    'http://images.cocodataset.org/val2017/000000252219.jpg'
]

inputs = [utils.prepare_input(uri) for uri in uris]
tensor = utils.prepare_tensor(inputs, precision == 'fp16')
```

```python id="er8qHVyyIfxQ"
# run the SSD model
with torch.no_grad():
    detections_batch = ssd_model(tensor)
```

```python colab={"base_uri": "https://localhost:8080/"} id="8PqkbnR3JFhL" executionInfo={"status": "ok", "timestamp": 1608551587585, "user_tz": -330, "elapsed": 13160, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b6abb32a-2670-45ab-91ee-d8572317226b"
# By default, raw output from SSD network per input image contains 8732 boxes with localization 
# and class probability distribution. Let's filter this output to only get reasonable detections 
# (confidence>40%) in a more comprehensive format.
results_per_input = utils.decode_results(detections_batch)
best_results_per_input = [utils.pick_best(results, 0.40) for results in results_per_input]

# The model was trained on COCO dataset, which we need to access in order to translate class IDs 
# into object names. For the first time, downloading annotations may take a while.
classes_to_labels = utils.get_coco_object_dictionary()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 773} id="JlyFCzOIJSiR" executionInfo={"status": "ok", "timestamp": 1608551596577, "user_tz": -330, "elapsed": 4570, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a1238550-3fe2-4b48-cad8-06d295ee6356"
from matplotlib import pyplot as plt
import matplotlib.patches as patches

for image_idx in range(len(best_results_per_input)):
    fig, ax = plt.subplots(1)
    # Show original, denormalized image...
    image = inputs[image_idx] / 2 + 0.5
    ax.imshow(image)
    # ...with detections
    bboxes, classes, confidences = best_results_per_input[image_idx]
    for idx in range(len(bboxes)):
        left, bot, right, top = bboxes[idx]
        x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, "{} {:.0f}%".format(classes_to_labels[classes[idx] - 1], confidences[idx]*100), bbox=dict(facecolor='white', alpha=0.5))
plt.show()
```

<!-- #region id="MNdNtDEfJ7s6" -->
## PyTorch Mask R-CNN
<!-- #endregion -->

```python id="CRjd3krWFlD8"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 103, "referenced_widgets": ["a7c4a0c1911f4b6caf2289a64a216913", "301b2654f4884dc9b56725e160944edb", "c89e09fd5eea4249b0dce0982b30b060", "c309397e7c8d4e5184e27ce90b4ae660", "faf3c7dedd124af7844c34d82149a52e", "2273fdb2caaa4a4db69c650e32b34fb1", "d05210b529484fc9b6b96f173c43882a", "7f5476287f32436fa723e614f5dcd106"]} id="7lexKa0dJ-9I" executionInfo={"status": "ok", "timestamp": 1608551763569, "user_tz": -330, "elapsed": 1882, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="354b57ff-17a2-4f0c-c389-8579bf626dd1"
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model = model.eval().cuda()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 815} id="XMgforlUKAaI" executionInfo={"status": "ok", "timestamp": 1608551790868, "user_tz": -330, "elapsed": 7957, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b918c7d8-72ed-44ec-b71f-18b3bee08f30"
image_url = "https://upload.wikimedia.org/wikipedia/commons/6/60/Naxos_Taverna.jpg"
downloaded_image_path = download_and_resize_image(image_url, 1280, 856, True)
```

```python colab={"base_uri": "https://localhost:8080/"} id="fwYHACVxKFlj" executionInfo={"status": "ok", "timestamp": 1608551832365, "user_tz": -330, "elapsed": 1831, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4fb7a9bd-93c6-4f79-ecd4-58965fb7a145"
t = time.time()
image = PIL.Image.open(downloaded_image_path)
image_tensor = torchvision.transforms.functional.to_tensor(image).cuda()
output = model([image_tensor])[0]
print('executed in %.3fs' % (time.time() - t))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 815} id="wYaR6gx7KIht" executionInfo={"status": "ok", "timestamp": 1608551849054, "user_tz": -330, "elapsed": 7871, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="20a0af98-321a-4e05-fd75-bc0e16b8440e"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 286} id="kEtNK4unKTzn" executionInfo={"status": "ok", "timestamp": 1608551857991, "user_tz": -330, "elapsed": 1838, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c8f0551c-ddce-4488-9782-984c84530319"
masks = None
for score, mask in zip(output['scores'], output['masks']):
  if score > 0.5:
    if masks is None:
      masks = mask
    else:
      masks = torch.max(masks, mask)

plt.imshow(masks.squeeze(0).cpu().numpy())
```

<!-- #region id="vAgODuLqKslr" -->
## DETR
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="0N5hySBhKxlW" executionInfo={"status": "ok", "timestamp": 1608551972793, "user_tz": -330, "elapsed": 1249, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="03803321-064a-444d-f702-ce6d1e41098e"
%cd /content
```

```python id="-Q9HmEUGKXdw"
from PIL import Image
import requests
import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'retina'

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False);
```

```python id="4NUJtN5bKzsE"
class DETRdemo(nn.Module):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        # construct positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1)
        
        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_logits': self.linear_class(h), 
                'pred_boxes': self.linear_bbox(h).sigmoid()}
```

```python colab={"base_uri": "https://localhost:8080/", "height": 83, "referenced_widgets": ["3c5822dbe50040a5a8ae775ead73c167", "a617cd174c0844fa8421e8f0095708ee", "0dffa75484d34280b9e1bfdf4e5cb256", "db31d9ba13c14b40810caa74a91e595d", "17f3fa891fd94adeb23d7bcb5d83281d", "9d7b1695763a4611a1ecc9126db7f382", "c99b0f47941c487ebcd522fefa8c56a7", "f84b9b40ab31444c9b25687c1b7ffef2"]} id="pUMtx_vhK4V1" executionInfo={"status": "ok", "timestamp": 1608552232909, "user_tz": -330, "elapsed": 5115, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fdbd19cc-d3ea-49b3-a4bd-59a15a68269d"
detr = DETRdemo(num_classes=91)
state_dict = torch.hub.load_state_dict_from_url(
    url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
    map_location='cpu', check_hash=True)
detr.load_state_dict(state_dict)
detr.eval();
```

```python id="AWIihDjaLyL_"
# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
```

```python id="AFTO2sdRL5At"
# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b
```

```python id="Njz0QekSL6wW"
def detect(im, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled
```

<!-- #region id="WxUdVfbQL_FU" -->
Let's Use DETR now
<!-- #endregion -->

```python id="8-iavKiZL8iT"
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
im = Image.open(requests.get(url, stream=True).raw)

scores, boxes = detect(im, detr, transform)
```

<!-- #region id="ijC78Uq2xR6h" -->
## OpenCV MobileNet SSD TFHub
<!-- #endregion -->

```python id="XWesBtPy9irb"
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

def display_image(image):
    fig = plt.figure(figsize=(20, 15))
    plt.grid(False)
    plt.imshow(image)

def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
    """Adds a bounding box to an image."""
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height
    
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom)],
                   fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
              display_str,
              fill="black",
              font=font)
        text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
    """Overlay labeled boxes on an image with formatted scores and label names."""
    colors = list(ImageColor.colormap.values())

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                              25)
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                     int(100 * scores[i]))
            color = colors[hash(class_names[i]) % len(colors)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box_on_image(
              image_pil,
              ymin,
              xmin,
              ymax,
              xmax,
              color,
              font,
              display_str_list=[display_str])
        np.copyto(image, np.array(image_pil))
    return image

def run_detector(detector, img):
    converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]

    result = detector(converted_img)
    result = {key:value.numpy() for key,value in result.items()}

    image_with_boxes = draw_boxes(
      img, result["detection_boxes"],
      result["detection_class_entities"], result["detection_scores"])
    
    display_image(image_with_boxes)
```

<!-- #region id="eLI_OI8y9mJq" -->
---
<!-- #endregion -->

```python id="Q9Oa_SXe809l"
import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub
```

```python colab={"base_uri": "https://localhost:8080/"} id="D2F0O0xQ9nbn" executionInfo={"status": "ok", "timestamp": 1608548567315, "user_tz": -330, "elapsed": 23160, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="907d6d10-e3a4-4f04-f3ff-880aeab8fd69"
module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
detector = hub.load(module_handle).signatures['default']
```

```python colab={"base_uri": "https://localhost:8080/"} id="ytK4rd_r9u37" executionInfo={"status": "ok", "timestamp": 1608548750631, "user_tz": -330, "elapsed": 10962, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9ebc6d71-d156-48fa-b1f6-842c26700bd3"
!pip install -q youtube-dl
!youtube-dl -o '%(title)s.%(ext)s' 1sk_xkww4AQ --restrict-filenames -f mp4
```

```python colab={"base_uri": "https://localhost:8080/"} id="E2M9Vpel-enl" executionInfo={"status": "ok", "timestamp": 1608548793919, "user_tz": -330, "elapsed": 1152, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fd324e7d-6df6-4bfd-b333-87184e8e6c3f"
cap = cv2.VideoCapture('/content/Spectre_opening_highest_for_a_James_Bond_film_in_India.mp4')
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print("Width x Height = %d x %d, Frames = %d, Frames/second = %d\n"%(width,height,total_frames,fps))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="il6k2IuV-mGj" executionInfo={"status": "ok", "timestamp": 1608548915651, "user_tz": -330, "elapsed": 38016, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8966e2d9-4873-4ac3-adcd-a24a7dec8553"
cap = cv2.VideoCapture('/content/Spectre_opening_highest_for_a_James_Bond_film_in_India.mp4')
for i in range(1,total_frames,200):
    cap.set(cv2.CAP_PROP_POS_FRAMES,i)
    ret,frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    run_detector(detector,frame)
```

```python id="0LCYGlaT-uhN"

```

```python id="-Z-t1u-wMBnY"

```
