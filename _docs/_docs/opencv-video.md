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

# OpenCV Video Basics

<!-- #region id="nBEMIyr_zS6C" -->
Download Youtube Video
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 8728, "status": "ok", "timestamp": 1608545751978, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="SSBLzR-ayJ3c" outputId="1bc27b62-70a6-4cb0-b63e-760048c96aec"
!pip install youtube-dl
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4672, "status": "ok", "timestamp": 1608545757751, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="r4lCXS4yU4EG" outputId="0be9a18e-f7cf-4901-820a-92a0bd7a6f4b"
!youtube-dl -o '%(title)s.%(ext)s' Zbl3n2qQ-iQ --restrict-filenames -f mp4
```

<!-- #region id="9otIf5uXzW1A" -->
Display Youtube Video
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 321} executionInfo={"elapsed": 4152, "status": "ok", "timestamp": 1608545776863, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="GDa38pobzDEm" outputId="06dbe2a3-b64e-4db4-8fb8-a08d51607e5a"
from IPython.lib.display import YouTubeVideo
YouTubeVideo('Y9w0cVVmJsk')
```

<!-- #region id="GvP_8oqQzaQS" -->
Display Video Metadata
<!-- #endregion -->

```python id="swUor1l28QbO"
import cv2
import matplotlib.pyplot as plt
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2017, "status": "ok", "timestamp": 1608548160504, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="f-ER9fJ58F5W" outputId="4c23b65e-dda1-417f-c038-98096daf1670"
#cap = cv2.VideoCapture('HGTV_Dream_Home_2020_-_Designing_for_an_Open-Concept_Space.mp4')
cap = cv2.VideoCapture('/content/HGTV_Dream_Home_2020_-_Designing_for_an_Open-Concept_Space.mp4')
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print("Width x Height = %d x %d, Frames = %d, Frames/second = %d\n"%(width,height,total_frames,fps))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000, "output_embedded_package_id": "1wha1c2h1396x14903BCCgFCwZM_nGDzT"} executionInfo={"elapsed": 12406, "status": "ok", "timestamp": 1608548207831, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="dEOjJzKF8OLD" outputId="d954a5d6-4c60-4416-dfc9-29caeefb1e2f"
cap = cv2.VideoCapture('HGTV_Dream_Home_2020_-_Designing_for_an_Open-Concept_Space.mp4')
for i in range(1,total_frames,5000):
  for i in range(1,total_frames,2000):
    cap.set(cv2.CAP_PROP_POS_FRAMES,i)
    ret,frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(frame)
    plt.grid(False)
    plt.axis('off')
    plt.show()
```

```python id="WGHYVQGo8XpU"

```
