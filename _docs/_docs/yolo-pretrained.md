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

```python colab={"base_uri": "https://localhost:8080/"} id="U5oJJJatsuvL" executionInfo={"status": "ok", "timestamp": 1607405179612, "user_tz": -330, "elapsed": 5463, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b75454dc-9bfb-4ecd-fdf4-4a806cf1d2e6"
# clone darknet repo
!git clone https://github.com/AlexeyAB/darknet
```

```python colab={"base_uri": "https://localhost:8080/"} id="byawkb4s0Dds" executionInfo={"status": "ok", "timestamp": 1607405184629, "user_tz": -330, "elapsed": 2546, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a04b494f-e3b2-459e-ad2b-6ca5d808f065"
# change makefile to have GPU and OPENCV enabled
%cd darknet
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
!sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
```

```python colab={"base_uri": "https://localhost:8080/"} id="W231N4-s0JwI" executionInfo={"status": "ok", "timestamp": 1607405190773, "user_tz": -330, "elapsed": 2496, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="12907e9d-a5e1-42aa-cacb-cc715be1a112"
# verify CUDA
!/usr/local/cuda/bin/nvcc --version
```

```python colab={"base_uri": "https://localhost:8080/"} id="wyxiuOEx0LQg" executionInfo={"status": "ok", "timestamp": 1607405282608, "user_tz": -330, "elapsed": 43863, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d49c9c25-6d13-47c9-dd50-2a7f54899e73"
# builds darknet so that you can then use the darknet executable file to run or train object detectors
!make
```

<!-- #region id="JSGqsL3E0abJ" -->
YOLOv4 has been trained already on the coco dataset which has 80 classes that it can predict. We will grab these pretrained weights so that we can run YOLOv4 on these pretrained classes and get detections
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="tc1bY1Es0O2h" executionInfo={"status": "ok", "timestamp": 1607405290565, "user_tz": -330, "elapsed": 3638, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5531479c-3fe1-416b-da37-193f7049bf78"
# Download pre-trained YOLOv4 weights
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
```

```python id="qpmQK2L20jWF"
# define helper functions
def imShow(path):
  import cv2
  import matplotlib.pyplot as plt
  %matplotlib inline

  image = cv2.imread(path)
  height, width = image.shape[:2]
  resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

  fig = plt.gcf()
  fig.set_size_inches(18, 10)
  plt.axis("off")
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  plt.show()

# use this to upload files
def upload():
  from google.colab import files
  uploaded = files.upload() 
  for name, data in uploaded.items():
    with open(name, 'wb') as f:
      f.write(data)
      print ('saved file', name)

# use this to download a file  
def download(path):
  from google.colab import files
  files.download(path)
```

```python colab={"base_uri": "https://localhost:8080/"} id="-7Y02KpL0l7W" executionInfo={"status": "ok", "timestamp": 1607405373111, "user_tz": -330, "elapsed": 10795, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0735e307-091a-4af9-bc98-5af2e87065ab"
# run darknet detection on test images
!./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights data/person.jpg
```

```python colab={"base_uri": "https://localhost:8080/", "height": 575} id="Y1Lu4w4Z01v5" executionInfo={"status": "ok", "timestamp": 1607405389939, "user_tz": -330, "elapsed": 6089, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="080f1a5b-5737-4bb0-9539-c2b9992acc82"
# show image using our helper function
imShow('predictions.jpg')
```

```python id="jC89kiLD07At"
!./darknet detector demo cfg/coco.data cfg/yolov4.cfg yolov4.weights -dont_show test.mp4 -i 0 -out_filename results.avi
```

<!-- #region id="2kWuPh983Iq-" -->
Darknet and YOLOv4 have a lot of command line flags you can add to your '!./darknet detector ...' to allow it to be customizeable and flexible.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="6alRG6kp3JOJ" executionInfo={"status": "ok", "timestamp": 1607406420780, "user_tz": -330, "elapsed": 11624, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e7619698-9f8a-462b-e604-bfd7cca2c58e"
# Threshold flag -thresh
# Only detections with a confidence level above the threshold you set will be returned

!./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights data/person.jpg -thresh 0.5
imShow('predictions.jpg')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="R5Fo8LGb3yRY" executionInfo={"status": "ok", "timestamp": 1607406476129, "user_tz": -330, "elapsed": 11255, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4538fc53-19db-4053-bc5e-7d44d66f5ec6"
# darknet run with external output flag to print bounding box coordinates
!./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights data/person.jpg -ext_output
imShow('predictions.jpg')
```

<!-- #region id="HQ89tV7-55CG" -->
### Multiple Images at Once
YOLOv4 object detections can be run on multiple images at once. This is done through having a text file which has the paths to several images that you want to have the detector run on.

<!-- #endregion -->

```python id="Cb9XREum5C6Y"
# save the file to .JSON
!./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights -ext_output -dont_show -out result.json < /mydrive/images.txt
```

```python id="LpuX402z7M_D"
# save the file to .txt
!./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights -dont_show -ext_output < /mydrive/images.txt > result.txt
```

<!-- #region id="FT2uS68C_aZ0" -->
---
### References
https://github.com/theAIGuysCode/YOLOv4-Cloud-Tutorial
<!-- #endregion -->

```python id="AS_W2jTy_dsq"

```
