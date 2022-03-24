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

<!-- #region id="FaYUSvX3WpoA" -->
# Easy OCR
<!-- #endregion -->

```python id="euSqn0XwCWyR"
!pip install easyocr
```

```python colab={"base_uri": "https://localhost:8080/"} id="tfJKkB_JCXnD" executionInfo={"status": "ok", "timestamp": 1607509753184, "user_tz": -330, "elapsed": 22224, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="18cc5b97-9b5b-455f-8ab9-cc0ec0d75ae1"
# import library
import easyocr


#specify shortform of language you want to extract,
# I am using Hindi(hi) and English(en) here by list of language ids
reader = easyocr.Reader(['hi','en'])
```

```python id="GTR3EAasCdxu"
!wget -O 'image.jpg' "https://miro.medium.com/max/391/1*zvN3oQiI6YFtV7hBPz4ATA.png"
```

```python id="RVUwAmBNDLUt"
from PIL import Image, ImageDraw
```

```python colab={"base_uri": "https://localhost:8080/", "height": 369} id="xrDJGdkYDDXn" executionInfo={"status": "ok", "timestamp": 1607509795078, "user_tz": -330, "elapsed": 1538, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8ddbc203-b193-4294-c61d-e5064af7fd13"
# Read Image
im = Image.open("image.jpg")
im
```

```python colab={"base_uri": "https://localhost:8080/"} id="drMcbU0_DNpn" executionInfo={"status": "ok", "timestamp": 1607509844133, "user_tz": -330, "elapsed": 22931, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4681e172-562d-4de8-fdc0-9650e230d434"
# Doing OCR. Get bounding boxes.
bounds = reader.readtext('image.jpg', detail=0) #detail=0 argument will only give text in array
print("Output:")
print(bounds)
```

```python colab={"base_uri": "https://localhost:8080/"} id="3OakQ4ZNDUZ6" executionInfo={"status": "ok", "timestamp": 1607509914881, "user_tz": -330, "elapsed": 21861, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3eff9080-6a1e-4b21-c41e-3584401d6d77"
# Doing OCR. Get bounding boxes. bounds = reader.readtext('Hindi_fonts.png', detail=1) #detail=1 
bounds = reader.readtext('image.jpg', detail=1)
print("Output:")
print(bounds)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 369} id="z-f990keDcvM" executionInfo={"status": "ok", "timestamp": 1607510076128, "user_tz": -330, "elapsed": 2447, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="165a08a8-b71d-41c6-f07c-2e8a8b1145e7"
# Draw bounding boxes
def draw_boxes(image, bounds, color='yellow', width=2):
    draw = ImageDraw.Draw(image)
    for bound in bounds: # iterate though all the tuples of output
        p0, p1, p2, p3 = bound[0] # get coordinates 
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
    return image

draw_boxes(im, bounds)
```

<!-- #region id="ouSjb6_3QWDz" -->
### References
- https://www.pyimagesearch.com/2020/09/14/getting-started-with-easyocr-for-optical-character-recognition/
<!-- #endregion -->
