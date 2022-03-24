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

<!-- #region id="_twZRD4WW5qT" -->
# R(2+1)D Video Action Recognition
<!-- #endregion -->

```python id="6gNAzR5cWa2B"
!git clone https://github.com/microsoft/computervision-recipes.git
%cd computervision-recipes
!pip install decord ipywebrtc einops
```

```python colab={"base_uri": "https://localhost:8080/"} id="A2fyVjvrbSdT" executionInfo={"status": "ok", "timestamp": 1607616781656, "user_tz": -330, "elapsed": 1370, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4b1f000a-ce38-4ea4-d61d-97f50538b02b"
%cd /content/video_input/
```

```python id="3RJLpQ_AZ5yk"
# !pip install -q pytube
from pytube import YouTube
# YouTube('https://www.youtube.com/watch?v=9P7JzTRHz5g').streams.first().download()
# YouTube('https://www.youtube.com/watch?v=0Cl_Q8RjmfI').streams.first().download()
# YouTube('https://www.youtube.com/watch?v=k2eCJ2XI1IA').streams.first().download()
# YouTube('https://www.youtube.com/watch?v=6mhRTDBNQ-M').streams.first().download()
# YouTube('https://www.youtube.com/watch?v=xm9c5HAUBpY').streams.first().download()
# YouTube('https://www.youtube.com/watch?v=K1FPxvdB_to').streams.first().download()
```

```python id="O9_yC43CWk2d" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1605251149124, "user_tz": -330, "elapsed": 14304, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f7b23c1c-e02c-4fd1-9211-235b21557b7e"
# Regular Python libraries
import sys
from collections import deque #
import io
import requests
import os
from time import sleep, time
from threading import Thread
from IPython.display import HTML
from base64 import b64encode

# Third party tools
import decord #
import IPython.display #
from ipywebrtc import CameraStream, ImageRecorder
from ipywidgets import HBox, HTML, Layout, VBox, Widget, Label
import numpy as np
from PIL import Image
import torch
import torch.cuda as cuda
import torch.nn as nn
from torchvision.transforms import Compose

# utils_cv
sys.path.append("/content/computervision-recipes")
from utils_cv.action_recognition.data import KINETICS, Urls
from utils_cv.action_recognition.dataset import get_transforms
from utils_cv.action_recognition.model import VideoLearner
from utils_cv.action_recognition.references import transforms_video as transforms
from utils_cv.common.gpu import system_info, torch_device
from utils_cv.common.data import data_path

%reload_ext autoreload
%autoreload 2

system_info()
```

```python id="awhjOGIPXJTb"
NUM_FRAMES = 8  # 8 or 32.
IM_SCALE = 128  # resize then crop
INPUT_SIZE = 112  # input clip size: 3 x NUM_FRAMES x 112 x 112

# video sample to download
sample_video_url = Urls.webcam_vid

# file path to save video sample
video_fpath = data_path() / "sample_video.mp4"

# prediction score threshold
SCORE_THRESHOLD = 0.01

# Averaging 5 latest clips to make video-level prediction (or smoothing)
AVERAGING_SIZE = 5  
```

```python id="6ypzuSrwXXEo" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1605251150816, "user_tz": -330, "elapsed": 13378, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7f850c39-e9a8-4f98-b9fb-a879e72d5390"
learner = VideoLearner(base_model="kinetics", sample_length=NUM_FRAMES)
```

```python id="m-sjp13YXamZ" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1605251150820, "user_tz": -330, "elapsed": 12043, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3340d45a-5865-4a8f-9797-41517d71c008"
LABELS = KINETICS.class_names
LABELS[:10]
```

```python id="NWQhUQKHXr1a" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1605251150823, "user_tz": -330, "elapsed": 11904, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fc11c4db-8f16-4ac2-eddb-ae7ccd1c2fcd"
TARGET_LABELS = [
    "assembling computer",
    "applying cream",
    "brushing teeth",
    "clapping",
    "cleaning floor",
    "cleaning windows",
    "drinking",
    "eating burger",
    "eating chips",
    "eating doughnuts",
    "eating hotdog",
    "eating ice cream",
    "fixing hair",
    "hammer throw",
    "high kick",
    "jogging",
    "laughing",
    "mopping floor",
    "moving furniture",
    "opening bottle",
    "plastering",
    "punching bag",
    "punching person (boxing)",
    "pushing cart",
    "reading book",
    "reading newspaper",
    "rock scissors paper",
    "running on treadmill",
    "shaking hands",
    "shaking head",
    "side kick",
    "slapping",
    "smoking",
    "sneezing",
    "spray painting",
    "spraying",
    "stretching arm",
    "stretching leg",
    "sweeping floor",
    "swinging legs",
    "texting",
    "throwing axe",
    "throwing ball",
    "unboxing",
    "unloading truck",
    "using computer",
    "using remote controller (not gaming)",
    "welding",
    "writing",
    "yawning",
]
len(TARGET_LABELS)
```

```python id="oIOZA3LMXwTd" colab={"base_uri": "https://localhost:8080/", "height": 246} executionInfo={"elapsed": 59628, "status": "ok", "timestamp": 1605250442894, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} outputId="393de987-6231-4821-8d72-fd65e8ddaaa4"
# !curl $sample_video_url --output video.mp4

# path = 'video.mp4'
# mp4 = open(path,'rb').read()
# data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
# HTML("""<video width=400 controls><source src="%s" type="video/mp4"></video>""" % data_url)
```

```python id="AI313OJ7YI02" colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3497, "status": "ok", "timestamp": 1605250769603, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} outputId="eb40aab2-1b35-440c-beea-b35cb5a46b26"
r = requests.get(sample_video_url)
open(video_fpath, 'wb').write(r.content)
```

```python id="3uu2FnW4YtaA" colab={"background_save": true}
video = str(data_path()/"sample_video.mp4")
learner.predict_video(
    video,
    LABELS,
    averaging_size=AVERAGING_SIZE,
    score_threshold=SCORE_THRESHOLD,
    target_labels=TARGET_LABELS,
)
```

```python id="XzoHdSXxY2kN"
# Webcam settings
w_cam = CameraStream(
    constraints={
        "facing_mode": "user",
        "audio": False,
        "video": {"width": 400, "height": 400},
    },
    layout=Layout(width="400px"),
)

# Image recorder for taking a snapshot
w_imrecorder = ImageRecorder(
    format="jpg", stream=w_cam, layout=Layout(padding="0 0 0 100px")
)

# Text widget to show our classification results
w_text = HTML(layout=Layout(padding="0 0 0 100px"))
```

```python id="f6VNh57WaUzV"
def predict_webcam_frames():
    """ Predict activity by using a pretrained model
    """
    global w_imrecorder, w_text, is_playing
    global device, model

    # Use deque for sliding window over frames
    window = deque()
    scores_cache = deque()
    scores_sum = np.zeros(len(LABELS))

    while is_playing:
        try:
            # Get the image (RGBA) and convert to RGB
            im = Image.open(io.BytesIO(w_imrecorder.image.value)).convert("RGB")
            window.append(np.array(im))

            # update println func
            def update_println(println):
                w_text.value = println
            
            if len(window) == NUM_FRAMES:
                learner.predict_frames(
                    window,
                    scores_cache,
                    scores_sum,
                    None,
                    AVERAGING_SIZE,
                    SCORE_THRESHOLD,
                    LABELS,
                    TARGET_LABELS,
                    get_transforms(train=False), 
                    update_println,
                )
            else:
                w_text.value = "Preparing..."
        except OSError:
            # If im_recorder doesn't have valid image data, skip it.
            pass
        except BaseException as e:
            w_text.value = "Exception: " + str(e)
            break

        # Taking the next snapshot programmatically
        w_imrecorder.recording = True
        sleep(0.02)
```

```python id="6LDymgjEaqMS"
is_playing = False
#  Once prediciton started, hide image recorder widget for faster fps
def start(_):
    global is_playing
    # Make sure this get called only once
    if not is_playing:
        w_imrecorder.layout.display = "none"
        is_playing = True
        Thread(target=predict_webcam_frames).start()


w_imrecorder.image.observe(start, "value")
```

```python id="o3YtkOWAasxz" colab={"base_uri": "https://localhost:8080/", "height": 17, "referenced_widgets": ["b13c9d2ea4ab42cf83f6421761b5422f", "72807c6a9e66423fb90ea7a6651025c9", "ed5a10296e6c4421a2ae5dbb7a1b22ae", "427d20cb850041e98e3fd213ca74042b", "b8e9702b67f7407dafed01288b1879b3", "8d3ad86b79d04203ad77dae732bcc39c", "168131607c9c47c0bbea699faadb49cf"]} executionInfo={"status": "ok", "timestamp": 1605251197427, "user_tz": -330, "elapsed": 931, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c68e1557-bafa-4b5c-89cb-d4f2a9e3b8f3"
HBox([w_cam, w_imrecorder, w_text])
```

```python id="CyYCEda-av46"
is_playing = False
Widget.close_all()
```
