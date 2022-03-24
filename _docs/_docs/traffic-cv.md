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

<!-- #region id="f-MqEdPjWH-x" -->
# Traffic Sign Classification
<!-- #endregion -->

```python id="VUr1jvpHABBm"
# !gdown --id 1_7v1SNHVp2BK_49WywyQbwllAJE8n31J
# !unzip traffic-signs-data.zip
```

```python id="Tw_SJrNu-fVY"
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import pickle
import random
```

```python id="c0x4DgD3-fVZ"
with open("./traffic-signs-data/train.p", mode='rb') as training_data:
    train = pickle.load(training_data)
with open("./traffic-signs-data/valid.p", mode='rb') as validation_data:
    valid = pickle.load(validation_data)
with open("./traffic-signs-data/test.p", mode='rb') as testing_data:
    test = pickle.load(testing_data)
```

```python id="-uEFeTF3rPl0"
X_train, y_train = train['features'], train['labels']
X_validation, y_validation = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
```

```python colab={"base_uri": "https://localhost:8080/"} id="LTLHnPTGrRkc" executionInfo={"status": "ok", "timestamp": 1609455404231, "user_tz": -330, "elapsed": 1356, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f0bb60a8-b51c-4ec3-eadf-81cf0a4e460e"
X_train.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="1jLck0QwrSNz" executionInfo={"status": "ok", "timestamp": 1609455404232, "user_tz": -330, "elapsed": 1170, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b975a6ac-62fc-4343-c550-41125ed5eab7"
y_train.shape
```

<!-- #region id="LlszUhNNyrl_" -->
VISUALIZATION
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 283} id="ronVvCdJsYc5" executionInfo={"status": "ok", "timestamp": 1609455428708, "user_tz": -330, "elapsed": 1305, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3c82572b-8101-4b77-a4db-f2e5e4de0d56"
i = np.random.randint(1, len(X_train))
plt.imshow(X_train[i])
y_train[i]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 592} id="ZUUvpXsmlwbb" executionInfo={"status": "ok", "timestamp": 1609455433655, "user_tz": -330, "elapsed": 3316, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="718cede0-7a2f-4fba-bc64-3e86812a7e51"
# Let's view more images in a grid format
# Define the dimensions of the plot grid 
W_grid = 5
L_grid = 5

# fig, axes = plt.subplots(L_grid, W_grid)
# subplot return the figure object and axes object
# we can use the axes object to plot specific figures at various locations

fig, axes = plt.subplots(L_grid, W_grid, figsize = (10,10))

axes = axes.ravel() # flaten the 15 x 15 matrix into 225 array

n_training = len(X_train) # get the length of the training dataset

# Select a random number from 0 to n_training
for i in np.arange(0, W_grid * L_grid): # create evenly spaces variables 

    # Select a random number
    index = np.random.randint(0, n_training)
    # read and display an image with the selected index    
    axes[i].imshow( X_train[index])
    axes[i].set_title(y_train[index], fontsize = 15)
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.4)
```

<!-- #region id="Y0GmpAjG3GiH" -->
CONVERT IMAGES TO GRAYSCALE AND PERFORM NORMALIZATION
<!-- #endregion -->

```python id="YI1QcjORsq2G"
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)
```

```python id="u2-GkZo0riel"
X_train_gray = np.sum(X_train/3, axis = 3, keepdims = True)
X_test_gray = np.sum(X_test/3, axis = 3, keepdims = True)
X_validation_gray = np.sum(X_validation/3, axis = 3, keepdims = True)
```

```python colab={"base_uri": "https://localhost:8080/"} id="a2thgGKwricb" executionInfo={"status": "ok", "timestamp": 1609455465513, "user_tz": -330, "elapsed": 2689, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b0532983-278b-4cd2-b577-4929086dc460"
X_train_gray.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="4yZu7n28riaV" executionInfo={"status": "ok", "timestamp": 1609455465514, "user_tz": -330, "elapsed": 2534, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a4bf4bb1-0339-4b03-a90c-5aa05de15fc1"
X_test_gray.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="Nar_imirriYX" executionInfo={"status": "ok", "timestamp": 1609455471213, "user_tz": -330, "elapsed": 1118, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="346c8fdb-e297-48d1-d930-2b0eb1505cd2"
X_validation_gray.shape
```

```python id="-xRVL8FpriWI"
X_train_gray_norm = (X_train_gray - 128)/128
X_test_gray_norm = (X_test_gray - 128)/128
X_validation_gray_norm = (X_validation_gray - 128)/128
```

```python id="vep8YTC1riUI"
# X_train_gray_norm
```

```python colab={"base_uri": "https://localhost:8080/", "height": 781} id="__hg6A5yrq2c" executionInfo={"status": "ok", "timestamp": 1609455478614, "user_tz": -330, "elapsed": 1488, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="54db27a5-632e-4055-cd21-16275614721d"
i = random.randint(1, len(X_train_gray))
plt.imshow(X_train_gray[i].squeeze(), cmap = 'gray')
plt.figure()
plt.imshow(X_train[i])
plt.figure()
plt.imshow(X_train_gray_norm[i].squeeze(), cmap = 'gray')
```

<!-- #region id="zmxxcT4P-fVg" -->
BUILD DEEP CONVOLUTIONAL NEURAL NETWORK MODEL
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="kFsd6rYsr9uW" executionInfo={"status": "ok", "timestamp": 1609455506290, "user_tz": -330, "elapsed": 6588, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ba10d375-3994-448d-b016-9d6747bdb691"
from tensorflow.keras import datasets, layers, models

CNN = models.Sequential()

CNN.add(layers.Conv2D(6, (5,5), activation = 'relu', input_shape = (32,32,1)))
CNN.add(layers.AveragePooling2D())

#CNN.add(layers.Dropout(0.2))

CNN.add(layers.Conv2D(16, (5,5), activation = 'relu'))
CNN.add(layers.AveragePooling2D())

CNN.add(layers.Flatten())

CNN.add(layers.Dense(120, activation = 'relu'))

CNN.add(layers.Dense(84, activation = 'relu'))

CNN.add(layers.Dense(43, activation = 'softmax'))
CNN.summary()
```

<!-- #region id="wr9IYeR6acWf" -->
COMPILE AND TRAIN DEEP CNN MODEL
<!-- #endregion -->

```python id="XDTDxCwgsBI1"
CNN.compile(optimizer = 'Adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
```

```python colab={"base_uri": "https://localhost:8080/"} id="g3IyLXbdsCyj" executionInfo={"status": "ok", "timestamp": 1609455544400, "user_tz": -330, "elapsed": 9449, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a54466a3-2a79-4491-bc3e-9715449a963d"
history = CNN.fit(X_train_gray_norm,
                 y_train, 
                 batch_size = 500,
                 epochs = 2,
                 verbose = 1,
                 validation_data = (X_validation_gray_norm, y_validation))
```

<!-- #region id="-wRQqOeB5Zh5" -->
ASSESS TRAINED CNN MODEL PERFORMANCE 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="_RpB2PAd9SE9" executionInfo={"status": "ok", "timestamp": 1609455559376, "user_tz": -330, "elapsed": 1933, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0c2bcdd0-02b1-4110-f4c9-ff66f684cc4a"
score = CNN.evaluate(X_test_gray_norm, y_test)
print('Test Accuracy: {}'.format(score[1]))
```

```python colab={"base_uri": "https://localhost:8080/"} id="FM1WY_Q_sMkL" executionInfo={"status": "ok", "timestamp": 1609455564799, "user_tz": -330, "elapsed": 1126, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d3c39ba9-83ee-4dc2-f63a-0ea3d1d02afe"
history.history.keys()
```

```python id="xBBJ9WlpsMiH"
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
```

```python colab={"base_uri": "https://localhost:8080/", "height": 298} id="c8oHEt0OsMf2" executionInfo={"status": "ok", "timestamp": 1609455568317, "user_tz": -330, "elapsed": 1610, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e0f793dd-1d36-4399-da8e-212b5b6df52a"
epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 298} id="liiGj4HdsMa3" executionInfo={"status": "ok", "timestamp": 1609455572357, "user_tz": -330, "elapsed": 1371, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a29c9b50-e82a-4b63-f5ba-87aa514266c3"
plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
```

```python id="RKZxo6khsMYv"
predicted_classes = CNN.predict_classes(X_test_gray_norm)
y_true = y_test
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="3knUnry5sMWd" outputId="103a614b-dee1-49a8-903c-f5bf320e3e8e"
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, predicted_classes)
plt.figure(figsize = (25, 25))
sns.heatmap(cm, annot = True)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 676} id="3WLTbXkysVdv" outputId="f40338bb-1fe6-4934-cbe3-2a2908f382f3"
L = 5
W = 5

fig, axes = plt.subplots(L, W, figsize = (12, 12))
axes = axes.ravel()

for i in np.arange(0, L*W):
    axes[i].imshow(X_test[i])
    axes[i].set_title('Prediction = {}\n True = {}'.format(predicted_classes[i], y_true[i]))
    axes[i].axis('off')

plt.subplots_adjust(wspace = 1)    
```
