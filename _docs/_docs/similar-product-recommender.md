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

<!-- #region id="4eU7FehIJDmq" -->
# Similar Product Recommender system using Deep Learning for an online e-commerce store
> A tutorial on building a recommender that will allow users to select a specific type of shirt and search for similar pattern of shirts from the inventory

- toc: true
- badges: true
- comments: true
- categories: [similarity, visual, retail]
- image: 
<!-- #endregion -->

<!-- #region id="PKYBbXgTHVif" -->
<!-- #endregion -->

<!-- #region id="PdAIqV9lNgle" -->
## Import libraries required for file operations
<!-- #endregion -->

```python executionInfo={"elapsed": 2597, "status": "ok", "timestamp": 1619154989654, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="crmoVAelGCQs"
import os
import pickle
from glob import glob

# import basic numerical libraries
import numpy as np
import pandas as pd

# import keras libraries for image recognition
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image as kimage
```

<!-- #region id="r4Lr1hV0Nn-7" -->
## Data preparation
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3392, "status": "ok", "timestamp": 1619155050954, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="0lRyXnY5GgOY" outputId="bd02cded-c646-4327-b7b2-eba213eb24dd"
#hide-output
# download and unzip shirts folder from the directory
!wget https://raw.githubusercontent.com/sparsh-ai/rec-data-public/master/shirts.zip
!unzip shirts.zip
```

```python executionInfo={"elapsed": 5477, "status": "ok", "timestamp": 1619155061939, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="wakhpPg9Gj8H"
# Create a dictionary of shirts for feeding to the image recognition model
shirts_dict = dict()
for shirt in glob('shirts/*.jpg'):  # load all shirts
  img = kimage.load_img(shirt, target_size=(224, 224))   # VGG accepts images in 224 X 224 pixels
  img = preprocess_input(np.expand_dims(kimage.img_to_array(img), axis=0))  # so some preprocessing
  id = shirt.split('/')[-1].split('.')[0]
  shirts_dict[id] = img  # map image & shirt id
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3227, "status": "ok", "timestamp": 1619155061941, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="9FbMKho4I0Vi" outputId="752117d5-67fb-4282-d214-f08601f61c05"
#hide-input
no_of_shirts = len(shirts_dict.keys())
print('Number of shirts = {}'.format(no_of_shirts))
```

<!-- #region id="1vRBMptsHwQN" -->
<!-- #endregion -->

<!-- #region id="j0nYJIRUNuiC" -->
## Model training
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 165444, "status": "ok", "timestamp": 1619155229580, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="l2U2Az3pJBJ3" outputId="2c6bf7e6-9962-43f9-b6c2-812ef70e5fa2"
# Train on the VGG Model
model = VGG16(include_top=False, weights='imagenet')

shirts_matrix = np.zeros([no_of_shirts, 25088])   # initialize the matrix with zeros
for i, (id, img) in enumerate(shirts_dict.items()):  
  shirts_matrix[i, :] = model.predict(img).ravel()  # flatten the matrix
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 163481, "status": "ok", "timestamp": 1619155229581, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="P59hcvIL7beX" outputId="bc7ed66f-8ade-4b5a-ddba-5bf254466dbe"
model.summary()
```

<!-- #region id="LMiCiXryId2o" -->
<!-- #endregion -->

<!-- #region id="vrGtrXxxN3Fq" -->
## Inference pipeline
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 154318, "status": "ok", "timestamp": 1619155235496, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="Mbn072FYLvho" outputId="c1925c2e-f8e8-41ef-a4bc-77aaa77bcf3d"
#hide
# Create a corelation between shirts
dot_product = shirts_matrix.dot(shirts_matrix.T)
norms = np.array([np.sqrt(np.diagonal(dot_product))])
similarity = dot_product / (norms * norms.T)
print(similarity.shape)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 152292, "status": "ok", "timestamp": 1619155235497, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="yNoU8ViNg_f0" outputId="e29156e8-3250-4896-b07f-9688dc550c94"
#hide
type(similarity)
```

```python executionInfo={"elapsed": 150322, "status": "ok", "timestamp": 1619155235498, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="8LMdYswtToRG"
# create a cross reference matrix with shirts and matrix
matrix_id_to_shirt_id = dict()
shirt_id_to_matrix_id = dict()
for i, (id, img) in enumerate(shirts_dict.items()):  
    matrix_id_to_shirt_id[i] = id
    shirt_id_to_matrix_id[id] = i
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 149133, "status": "ok", "timestamp": 1619155235499, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="4ihwxzeXnMpW" outputId="24176644-db9a-48ea-838e-5751941e1087"
#hide
type(matrix_id_to_shirt_id)
```

<!-- #region id="OUsP9FKrIlE4" -->
<!-- #endregion -->

<!-- #region id="Nyet1CqiLzMD" -->
##Finding top 10 similar shirts 
<!-- #endregion -->

<!-- #region id="obtRVZ2DMbqv" -->
### Display the sample shirt
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 259} executionInfo={"elapsed": 144556, "status": "ok", "timestamp": 1619155235504, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="uA4eOva4Rg1x" outputId="ad36d35e-d7ad-479e-ca08-4933272444fd"
# Display the sample shirt
from IPython.display import Image
Image('shirts/1015.jpg')
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 144722, "status": "ok", "timestamp": 1619155235502, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="pmnUxoGhMNnK" outputId="b9262a07-0332-4d2e-f0eb-1ecb2bfd6d94"
#hide
# evaluate on shirt "1015"
target_shirt_id = '1015'
target_id = shirt_id_to_matrix_id[target_shirt_id]

# Sort 10 shirts based on their closest corelation
closest_ids = np.argsort(similarity[target_id, :])[::-1][0:10]
closest_shirts = [matrix_id_to_shirt_id[matrix_id] for matrix_id in closest_ids]

closest_shirts
```

<!-- #region id="69OqnYdHMhCa" -->
### Print images of top-10 similar shirts
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 346} executionInfo={"elapsed": 145123, "status": "ok", "timestamp": 1619155237825, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="u2S_z1VENyWb" outputId="53a1851f-2a4b-4409-92e9-4acd1322a215"
#collapse-input
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

images = []

for shirt in closest_shirts:
  shirt = 'shirts/'+shirt+'.jpg'
  for img_path in glob.glob(shirt):
      images.append(mpimg.imread(img_path))

plt.figure(figsize=(20,10))
columns = 5
for i, image in enumerate(images):
    plt.subplot(len(images) / columns + 1, columns, i + 1)
    plt.imshow(image)
```

<!-- #region id="srIyD4bJNGas" -->
## Model persistence
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 145642, "status": "ok", "timestamp": 1619155238576, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="CRJA_MpngVah" outputId="2f2651f3-91f7-47a4-eab8-5a12dc8fd3bb"
#hide-output
from sklearn.externals import joblib
joblib.dump(similarity, 'similarity.pkl')
joblib.dump(shirt_id_to_matrix_id, 'shirt_id_to_matrix_id.pkl')
joblib.dump(matrix_id_to_shirt_id, 'matrix_id_to_shirt_id.pkl')
```

```python executionInfo={"elapsed": 144063, "status": "ok", "timestamp": 1619155238579, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="Y30DwJR17RQx"
# load the model from disk
loaded_model = joblib.load('similarity.pkl')
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 143922, "status": "ok", "timestamp": 1619155238581, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="pVgPsiTm7nhZ" outputId="87787fd1-7f07-46d4-a931-1c54e21fbe55"
#hide
type(loaded_model)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 142103, "status": "ok", "timestamp": 1619155238582, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="nK8OIRmz7a87" outputId="77746b3b-730a-451e-f302-c3d232032996"
# Sort 10 shirts based on their closest corelation
closest_ids = np.argsort(loaded_model[target_id, :])[::-1][0:10]
closest_shirts = [matrix_id_to_shirt_id[matrix_id] for matrix_id in closest_ids]
closest_shirts
```

<!-- #region id="NwyH7eBqIz2O" -->
<!-- #endregion -->

<!-- #region id="TbY8E_EiK93C" -->
[Credits](https://www.analyticsvidhya.com/)
<!-- #endregion -->
