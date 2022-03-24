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

```python colab={"base_uri": "https://localhost:8080/"} id="VWV_-VW46b4X" executionInfo={"status": "ok", "timestamp": 1607507622162, "user_tz": -330, "elapsed": 13577, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="06e2a9b8-511d-4daf-9cad-bdd8763f24d9"
!sudo apt install tesseract-ocr
!pip install pytesseract
```

```python id="tX5HjAiY618H" executionInfo={"status": "ok", "timestamp": 1607507631875, "user_tz": -330, "elapsed": 1501, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import pytesseract
import shutil
import os
import random
from PIL import Image
```

```python colab={"base_uri": "https://localhost:8080/"} id="wwKqiZzI69im" executionInfo={"status": "ok", "timestamp": 1607508299257, "user_tz": -330, "elapsed": 1312, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a59ae84b-6b9d-4b6a-9abd-1b3a8b82e61e"
# !wget -O "sample.jpg" "https://talkerscode.com/webtricks/images/text_over_image.jpg"
# !wget -O "sample2.jpg" "https://i.ytimg.com/vi/fIVFH08ZPRE/maxresdefault.jpg"
# !wget -O "sample3.jpg" "https://alinguistinfrance.files.wordpress.com/2017/10/screen-shot-2017-10-10-at-08-56-10.png?w=329&h=188&crop=1"
```

```python colab={"base_uri": "https://localhost:8080/", "height": 332} id="k--WAakc7lsj" executionInfo={"status": "ok", "timestamp": 1607508153964, "user_tz": -330, "elapsed": 2528, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c221c1e6-f469-4e38-9d29-54f37e086cce"
image_path= 'sample.jpg'
Image.open(image_path)
```

```python colab={"base_uri": "https://localhost:8080/"} id="dNcH_zdJ80tP" executionInfo={"status": "ok", "timestamp": 1607508154824, "user_tz": -330, "elapsed": 2964, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="882da48c-5fb0-4313-ad65-b225b7f01eb9"
# text extraction
extractedInformation = pytesseract.image_to_string(Image.open(image_path))
print(extractedInformation)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 717} id="8Y6CyNqD7_UO" executionInfo={"status": "ok", "timestamp": 1607508165198, "user_tz": -330, "elapsed": 3394, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fc5fd116-1d14-4d3c-aa87-c75b1439c058"
image_path= 'sample2.jpg'
Image.open(image_path)
```

```python colab={"base_uri": "https://localhost:8080/"} id="zdTriQRP8Uhi" executionInfo={"status": "ok", "timestamp": 1607508168063, "user_tz": -330, "elapsed": 1242, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d4db640e-d113-4380-8cc5-1f489f9ce08a"
# text extraction
extractedInformation = pytesseract.image_to_string(Image.open(image_path))
print(extractedInformation)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 205} id="GBhPulOM8kXO" executionInfo={"status": "ok", "timestamp": 1607508318901, "user_tz": -330, "elapsed": 2534, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1653fe2a-f990-4e35-9a45-0a8b250a136e"
image_path= 'sample3.jpg'
Image.open(image_path)
```

```python colab={"base_uri": "https://localhost:8080/"} id="VImBPItR9lAG" executionInfo={"status": "ok", "timestamp": 1607508323581, "user_tz": -330, "elapsed": 1298, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4a7d88b4-e43e-4d8d-9ee0-25902475aedf"
# text extraction
extractedInformation = pytesseract.image_to_string(Image.open(image_path))
print(extractedInformation)
```

```python id="m5yoKZ2s9mdd"
# text extraction for foreign language
extractedInformation = pytesseract.image_to_string(Image.open(image_path), lang='fra')
print(extractedInformation)
```

```python colab={"base_uri": "https://localhost:8080/"} id="12OfP2NJ9ufJ" executionInfo={"status": "ok", "timestamp": 1607509263529, "user_tz": -330, "elapsed": 1610, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8ff1cdf6-f4df-4f4d-e4b9-376f2ff322ed"
# get bounding boxes
print(pytesseract.image_to_boxes(Image.open(image_path_in_colab)))
```

<!-- #region id="pZ3esbkPEyYm" -->
### Page segmentation modes:
- 0    Orientation and script detection (OSD) only.
- 1    Automatic page segmentation with OSD.
- 2    Automatic page segmentation, but no OSD, or OCR.
- 3    Fully automatic page segmentation, but no OSD. (Default)
- 4    Assume a single column of text of variable sizes.
- 5    Assume a single uniform block of vertically aligned text.
- 6    Assume a single uniform block of text.
- 7    Treat the image as a single text line.
- 8    Treat the image as a single word.
- 9    Treat the image as a single word in a circle.
- 10    Treat the image as a single character.
- 11    Sparse text. Find as much text as possible in no particular order.
- 12    Sparse text with OSD.
- 13    Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.
<!-- #endregion -->

```python id="Xgk9ETZpBL3R"
pytesseract.image_to_string(img_cv, lang='eng', config='-psm 1')
```
