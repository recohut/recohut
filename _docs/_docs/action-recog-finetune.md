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

# Finetune the pretrained R(2+1)D model

```python id="STPjKB9Gd7jw"
# !git clone https://github.com/microsoft/computervision-recipes.git
# %cd computervision-recipes
# !pip install decord ipywebrtc einops nteract-scrapbook
```

<!-- #region id="xQm3rqyLd30y" -->
Objective - to finetune the pretrained R(2+1)D model

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 5516, "status": "ok", "timestamp": 1605252132924, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="9me1KhpPd301" outputId="0ffe8454-fcc4-4f00-b9c7-bb5d648906ca"
import sys

sys.path.append("/content/computervision-recipes")

import numpy as np
import os
from pathlib import Path
import time
import warnings

from sklearn.metrics import accuracy_score
import scrapbook as sb
import torch
import torchvision

from utils_cv.action_recognition.data import Urls
from utils_cv.action_recognition.dataset import VideoDataset
from utils_cv.action_recognition.model import VideoLearner 
from utils_cv.common.gpu import system_info
from utils_cv.common.data import data_path, unzip_url

system_info()
warnings.filterwarnings('ignore')
```

```python executionInfo={"elapsed": 1404, "status": "ok", "timestamp": 1605252157310, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="eGZzZShvd31B"
# Ensure edits to libraries are loaded and plotting is shown in the notebook.
%reload_ext autoreload
%autoreload 2
%matplotlib inline
```

<!-- #region id="NWlZHRhBd31J" -->
Next, set some model runtime parameters. We use the `unzip_url` helper function to download and unzip the data used in this example notebook.
<!-- #endregion -->

```python executionInfo={"elapsed": 7157, "status": "ok", "timestamp": 1605252180517, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="DscEZhaNd31L" tags=["parameters"]
# Your data
DATA_PATH = unzip_url(Urls.milk_bottle_action_path, exist_ok=True)

# Number of consecutive frames used as input to the DNN. Use: 32 for high accuracy, 8 for inference speed.
MODEL_INPUT_SIZE = 8

# Number of training epochs
EPOCHS = 16

# Batch size. Reduce if running out of memory.
BATCH_SIZE = 8

# Learning rate
LR = 0.0001
```

<!-- #region id="0-NUrK5ed31U" -->
## Prepare Action Recognition Dataset

In this notebook, we use a toy dataset called *Milk Bottle Actions*, which consists of 60 clips of 2 actions: `{opening, pouring}`. The same objects appear in both classes. 

After downloading the dataset, the `unzip_url` helper function will also unzip the dataset to the `data` directory.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1329, "status": "ok", "timestamp": 1605252208185, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="QWWr5ujsd31W" outputId="b0cff45b-57e3-4e5a-dfd7-b3640420315a"
os.listdir(Path(DATA_PATH))
```

<!-- #region id="x9sdCAD_d31g" -->
You'll notice that we have two different folders inside:
- `/pouring`
- `/opening`

Action videos can be stored as follows:

```
/data
+-- action_class_1
|   +-- video_01.mp4
|   +-- video_02.mp4
|   +-- ...
+-- action_class_2
|   +-- video_11.mp4
|   +-- video_12.mp4
|   +-- ...
+-- ...
```

For action recognition, the way data is stored can be as straight forward as putting the videos for each action inside a folder named after the action. 
<!-- #endregion -->

<!-- #region id="Cq0Sv5T4d31i" -->
## Load Images

To load the data, we need to create a VideoDataset object using the `VideoDataset` helper class. This class knows how to extract the dataset based on the above format. We simply pass it the path to the root dir of our dataset.
<!-- #endregion -->

```python executionInfo={"elapsed": 1385, "status": "ok", "timestamp": 1605252287217, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="fjrwDGQcd31k"
data = VideoDataset(DATA_PATH, batch_size=BATCH_SIZE, sample_length=MODEL_INPUT_SIZE)
```

<!-- #region id="zdwmckWUd31q" -->
The `VideoDataset` will automatically divide the data into a training/validation set and set up the dataloaders that PyTorch uses. Lets inspect our Datasets/DataLoaders to make sure the train/test split looks right.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1444, "status": "ok", "timestamp": 1605252299827, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="6XnONWKOd31r" outputId="1d62441d-1539-4ddb-8514-88456d312c17"
print(
    f"Training dataset: {len(data.train_ds)} | Training DataLoader: {data.train_dl} \
    \nTesting dataset: {len(data.test_ds)} | Testing DataLoader: {data.test_dl}"
)
```

<!-- #region id="0NZm69z8d31y" -->
Lets take a look at the data by briefly looking at the frames.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 623} executionInfo={"elapsed": 7908, "status": "ok", "timestamp": 1605252357084, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="_1SjdJGzd311" outputId="2f687ae0-e1fc-422a-d608-4f60c401bf15"
data.show_batch(rows=5, train_or_test="test")
```

<!-- #region id="H6z-xCayd32C" -->
## Finetune a Pretrained Model

By default, our VideoLearner's R(2+1)D model is pretrained on __ig65m__ which is based of 65 million instagram videos. You can find more information on the dataset in this paper: https://arxiv.org/pdf/1905.00561.pdf

When we initialize the VideoLearner, we simply pass in the dataset. By default, the object will set the model to torchvision's ig65m R(2+1)D pre-trained model. Alternatively, we can also select the R(2+1)D model pretrained on the __kinetics__ dataset or even pass in the model we want to use. 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 22110, "status": "ok", "timestamp": 1605252407770, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="WGhCRFTxd32D" outputId="a72e679d-2deb-4c43-8dd0-71e0ba2ca755"
learner = VideoLearner(data, num_classes=2)
```

<!-- #region id="6agyCO6ld32I" -->
The dataset we're using only has two actions: __opening__ and __pouring__. This means that our fully connected (FC) layer must only have an output of two. We can check that this is the case by inspecting the model's FC layer.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1378, "status": "ok", "timestamp": 1605252443798, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="g8E-ee_Rd32J" outputId="b2f646ab-45a3-4350-ee75-1387eae304a4"
learner.model.fc
```

<!-- #region id="4vTcnHU_d32P" -->
Fine-tune the model using the `learner`'s `fit` function. Here we pass in the learning rate.
<!-- #endregion -->

<!-- #region id="tSy6UNmGhtYF" -->
To resolve error: "RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead"

replace line 37 with this: correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)

[reference](https://github.com/cezannec/capsule_net_pytorch/issues/4)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 156841, "status": "ok", "timestamp": 1605252984255, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="XX9MFGWHd32R" outputId="d721e2e8-e00a-41f0-b6e5-fede666ee90b"
learner.fit(lr=LR, epochs=EPOCHS)
```

<!-- #region id="jzDhgMG-d32Z" -->
## Evaluate

Video datasets, composed of multiple videos, have varying video lengths. Yet our model only takes in clips of a set length. This is why we need to take into consideration two types of accuracies when evaluating our model: clip-level accuracy and video-level accuracy. 

Our classifications are set at the video level, which mean that each video is assigned 1 classification. However, each video could have hundreds or thousands of frames. To make sure we're getting a good range of clips across those frames, we can sample a single video (uniformly) a bunch of times. By default, our evaluation tool will sample 10 clips uniformly from each test video and pass it into the model in batches of [10 x 3 x (8 or 32) x 112 x 112]. This is where clip-level accuracy comes in. The direct results of the model will show us how many of those clips are correctly classified. 

Since our classifications are at the video level, we'll also want to see what our video-level accuracy is. We do this simply by average across the clip-level accuracy for each video.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 7835, "status": "ok", "timestamp": 1605253074209, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="JtCzD_rNd32b" outputId="5a8c7ad9-2d96-4b01-92f5-b8222c3c0ffc"
ret = learner.evaluate()
```

<!-- #region id="Xg81brARd32h" -->
## Predict

Now that we've developed a model that works, lets run a prediction on one of our videos and see how it works. We'll choose an image that's already downloaded to our disk, and run it through our `predict_video()` function.
<!-- #endregion -->

```python executionInfo={"elapsed": 3373, "status": "ok", "timestamp": 1605253133256, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="EipeBFaFd32i"
test_vid = str(data_path()/"milkBottleActions"/"opening"/"01.mp4")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 407} executionInfo={"elapsed": 13763, "status": "ok", "timestamp": 1605253148193, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="9SVbad8Ld32p" outputId="2619b004-3a59-464c-a9fe-18863063459a"
learner.predict_video(test_vid)
```

<!-- #region id="V1ZM39B-d32w" -->
## Conclusion
Using the concepts introduced in this notebook, you can bring your own dataset and train an action recognition model to detect specific objects of interest for your scenarios.
<!-- #endregion -->

```python id="tVropIOxd32x"
# Preserve some of the notebook outputs
sb.glue("vid_pred_accuracy", accuracy_score(ret["video_trues"], ret["video_preds"]))
sb.glue("clip_pred_accuracy", accuracy_score(ret["clip_trues"], ret["clip_preds"]))
```
