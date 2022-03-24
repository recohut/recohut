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

<!-- #region id="cViuxI5-gQgo" -->
# License Plate Detection
<!-- #endregion -->

```python id="XnrZ2mpbxpG4"
# !git clone https://github.com/GuiltyNeuron/ANPR.git
# !wget -O weights.zip https://bit.ly/39RYf1u
# !unzip weights.zip
# !cp '/content/classes.names' '/content/ANPR/Licence_plate_detection/'
# %cd '/content/ANPR/Licence_plate_detection/'
# !python detector.py --image test.jpg
```

```python id="K5jNrjZuz-mD"
!git clone https://github.com/quangnhat185/Plate_detect_and_recognize.git
%cd Plate_detect_and_recognize

import cv2
import numpy as np
import matplotlib.pyplot as plt
from local_utils import detect_lp
from os.path import splitext,basename
from keras.models import model_from_json
import glob

def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)

wpod_net_path = "wpod-net.json"
wpod_net = load_model(wpod_net_path)

def preprocess_image(image_path,resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img
```

```python id="y4yQZvIj1CVM"
# !wget -q -O /content/plates/plate1.jpg 'https://images.squarespace-cdn.com/content/v1/5c981f3d0fb4450001fdde5d/1563727260863-E9JQC4UVO8IYCE6P19BO/ke17ZwdGBToddI8pDm48kDHPSfPanjkWqhH6pl6g5ph7gQa3H78H3Y0txjaiv_0fDoOvxcdMmMKkDsyUqMSsMWxHk725yiiHCCLfrh8O1z4YTzHvnKhyp6Da-NYroOW3ZGjoBKy3azqku80C789l0mwONMR1ELp49Lyc52iWr5dNb1QJw9casjKdtTg1_-y4jz4ptJBmI9gQmbjSQnNGng/cars+1.jpg'
# !wget -q -O /content/plates/plate2.jpg 'https://www.cars24.com/blog/wp-content/uploads/2018/12/High-Security-Registration-Plates-Feature-Cars24.com_.png'
```

```python id="j99fzMJd0MAv" colab={"base_uri": "https://localhost:8080/", "height": 191} executionInfo={"status": "ok", "timestamp": 1596550820996, "user_tz": -330, "elapsed": 5897, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7767425f-015f-43a0-8c8d-6bbd12992f1a"
# Create a list of image paths 
image_paths = glob.glob("/content/plates/*.jpg")
print("Found %i images..."%(len(image_paths)))

# Visualize data in subplot 
fig = plt.figure(figsize=(12,8))
cols = 5
rows = 4
fig_list = []
for i in range(len(image_paths)):
    fig_list.append(fig.add_subplot(rows,cols,i+1))
    title = splitext(basename(image_paths[i]))[0]
    fig_list[-1].set_title(title)
    img = preprocess_image(image_paths[i],True)
    plt.axis(False)
    plt.imshow(img)

plt.tight_layout(True)
plt.show()
```

```python id="SEd06yRh0Npc" colab={"base_uri": "https://localhost:8080/", "height": 291} executionInfo={"status": "ok", "timestamp": 1596550868236, "user_tz": -330, "elapsed": 4181, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5f58f645-f364-4249-a7d4-0e051a6253ab"
# forward image through model and return plate's image and coordinates
# if error "No Licensese plate is founded!" pop up, try to adjust Dmin
def get_plate(image_path, Dmax=608, Dmin=256):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return LpImg, cor

# Obtain plate image and its coordinates from an image
test_image = image_paths[0]
LpImg,cor = get_plate(test_image)
print("Detect %i plate(s) in"%len(LpImg),splitext(basename(test_image))[0])
print("Coordinate of plate(s) in image: \n", cor)

# Visualize our result
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.axis(False)
plt.imshow(preprocess_image(test_image))
plt.subplot(1,2,2)
plt.axis(False)
plt.imshow(LpImg[0])

#plt.savefig("part1_result.jpg",dpi=300)
```

```python id="zsJTq37L0UFk" colab={"base_uri": "https://localhost:8080/", "height": 138} executionInfo={"status": "ok", "timestamp": 1596550897173, "user_tz": -330, "elapsed": 5100, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="61ab2f68-2045-4647-d9ee-df864569cbb1"
# Viualize all obtained plate images 
fig = plt.figure(figsize=(12,6))
cols = 5
rows = 4
fig_list = []

for i in range(len(image_paths)):
    fig_list.append(fig.add_subplot(rows,cols,i+1))
    title = splitext(basename(image_paths[i]))[0]
    fig_list[-1].set_title(title)
    LpImg,_ = get_plate(image_paths[i])
    plt.axis(False)
    plt.imshow(LpImg[0])

plt.tight_layout(True)
plt.show()
```
