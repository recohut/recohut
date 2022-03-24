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

<!-- #region id="XhBeDBXQLygm" -->
# Dlib and MTCNN face detectors

<!-- #endregion -->

```python id="Z9NQkWMGE07P"
# No need to clone the repo, but in case someone wants to inspect the code.
!git clone https://github.com/ipazc/mtcnn.git
!git clone https://github.com/ageitgey/face_recognition.git  
```

```python id="vDk5ZBMPHMq2"
!apt-get install build-essential cmake
!apt-get install libopenblas-dev liblapack-dev 
!pip3 install dlib
!pip3 install face_recognition
```

```python id="n7IBNZcRE38G"
!pip3 install mtcnn
!pip3 install opencv-contrib-python
```

```python id="VLDNw9PBEHBE" outputId="6d36f12b-b7db-44a1-8ca7-64545bea161d" executionInfo={"status": "ok", "timestamp": 1587881116870, "user_tz": -330, "elapsed": 17547, "user": {"displayName": "", "photoUrl": "", "userId": ""}} colab={"base_uri": "https://localhost:8080/", "height": 187}
!ls -la face_recognition/tests/test_images/
```

<!-- #region id="UJn9Eu8sME6M" -->
**Timing the MTCNN Detector**
<!-- #endregion -->

```python id="gA9B5akoFAKY" outputId="f24db519-2ea2-495f-ea96-1ae0ccb947b3" executionInfo={"status": "ok", "timestamp": 1587881131856, "user_tz": -330, "elapsed": 11161, "user": {"displayName": "", "photoUrl": "", "userId": ""}} colab={"base_uri": "https://localhost:8080/", "height": 34}
from mtcnn.mtcnn import MTCNN
import cv2

filename = "face_recognition/tests/test_images/obama2.jpg"
img = cv2.imread(filename)
detector = MTCNN()
```

```python id="aqS0aflMFEiH" outputId="75b4cf0d-50d9-4920-8dee-74835e0c1a0c" executionInfo={"status": "ok", "timestamp": 1587881141850, "user_tz": -330, "elapsed": 8192, "user": {"displayName": "", "photoUrl": "", "userId": ""}} colab={"base_uri": "https://localhost:8080/", "height": 51}
%%timeit

face_locations = detector.detect_faces(img)
```

```python id="YBGJVzV-M2pN" outputId="7d67987c-fb88-49c2-a260-f7a7e40fa7a3" executionInfo={"status": "ok", "timestamp": 1587881142434, "user_tz": -330, "elapsed": 1159, "user": {"displayName": "", "photoUrl": "", "userId": ""}} colab={"base_uri": "https://localhost:8080/", "height": 136}
detector.detect_faces(img)
```

<!-- #region id="ojhAeyrhMTcf" -->
**Timing the Dlib-based Face Detector**
<!-- #endregion -->

```python id="9T4peShSRJOg"
from PIL import Image
import face_recognition

# Load the jpg file into a numpy array
image = face_recognition.load_image_file(filename)
```

```python id="Ppkr-GSpRNqq" outputId="a47dc4a5-2aa2-45b9-a878-f179b41e7ee3" executionInfo={"status": "ok", "timestamp": 1587881163604, "user_tz": -330, "elapsed": 4334, "user": {"displayName": "", "photoUrl": "", "userId": ""}} colab={"base_uri": "https://localhost:8080/", "height": 34}
%%timeit 

# Find all the faces in the image using the default HOG-based model.
# This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
# See also: find_faces_in_picture_cnn.py
face_locations = face_recognition.face_locations(image)
```

```python id="m3kNTGwZ_zBC" outputId="d6d6daa8-9474-4373-8f38-424cba2e8a50" executionInfo={"status": "ok", "timestamp": 1587881187399, "user_tz": -330, "elapsed": 2817, "user": {"displayName": "", "photoUrl": "", "userId": ""}} colab={"base_uri": "https://localhost:8080/", "height": 628}
%matplotlib inline

from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 10]

face_locations = face_recognition.face_locations(image)

print("Found {} face(s) in this photograph.".format(len(face_locations)))

for face_location in face_locations:

    # Print the location of each face in this image
    top, right, bottom, left = face_location
    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

    # You can access the actual face itself like this:
    face_image = image[top:bottom, left:right]
    plt.imshow(face_image)
    plt.show()
```
