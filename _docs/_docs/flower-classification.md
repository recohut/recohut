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

<!-- #region id="v09zCTsGDQVd" -->
# Flower classification
> Building end-to-end flower classification system using tensorflow keras

- toc: true
- badges: true
- comments: true
- categories: [ComputerVision, ImageClassification, Gradio]
- image:
<!-- #endregion -->

<!-- #region id="YT7JbAqMDFNp" -->
### Process flow
<!-- #endregion -->

<!-- #region id="90zne8TpDIEi" -->
<!-- #endregion -->

<!-- #region id="zF9uvbXNVrVY" -->
Import libraries
<!-- #endregion -->

```python id="L1WtoaOHVrVh"
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
```

<!-- #region id="UZZI6lNkVrVm" -->
Download and explore the dataset
<!-- #endregion -->

```python id="57CcilYSG0zv" colab={"base_uri": "https://localhost:8080/"} outputId="9654153a-2dc3-4340-f399-dff91c954e4c"
import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)
```

```python id="SbtTDYhOHZb6" colab={"base_uri": "https://localhost:8080/"} outputId="ad23fcca-50bb-4bc6-bd98-820906e108b5"
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)
```

<!-- #region id="PVmwkOSdHZ5A" -->
Here are some roses:
<!-- #endregion -->

```python id="N1loMlbYHeiJ" colab={"base_uri": "https://localhost:8080/", "height": 257} outputId="2ab439f7-3f43-4b51-9713-9fec3d6ccd68"
roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[0]))
```

```python id="RQbZBOTLHiUP" colab={"base_uri": "https://localhost:8080/", "height": 230} outputId="bdea02a9-bb84-4180-8a61-d6fcaada494f"
PIL.Image.open(str(roses[1]))
```

<!-- #region id="DGEqiBbRHnyI" -->
And some tulips:
<!-- #endregion -->

```python id="HyQkfPGdHilw" colab={"base_uri": "https://localhost:8080/", "height": 257} outputId="5b4adba6-413c-4a66-d459-88c08d2fa82a"
tulips = list(data_dir.glob('tulips/*'))
PIL.Image.open(str(tulips[0]))
```

```python id="wtlhWJPAHivf" colab={"base_uri": "https://localhost:8080/", "height": 257} outputId="e3869848-8233-4e9f-a7b9-e0762ab62625"
PIL.Image.open(str(tulips[1]))
```

<!-- #region id="gIjgz7_JIo_m" -->
### Load using keras.preprocessing

Let's load these images off disk using the helpful [image_dataset_from_directory](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory) utility. This will take you from a directory of images on disk to a `tf.data.Dataset` in just a couple lines of code. If you like, you can also write your own data loading code from scratch by visiting the [load images](https://www.tensorflow.org/tutorials/load_data/images) tutorial.
<!-- #endregion -->

<!-- #region id="anqiK_AGI086" -->
Define some parameters for the loader:
<!-- #endregion -->

```python id="H74l2DoDI2XD"
batch_size = 32
img_height = 180
img_width = 180
```

<!-- #region id="pFBhRrrEI49z" -->
It's good practice to use a validation split when developing your model. Let's use 80% of the images for training, and 20% for validation.
<!-- #endregion -->

```python id="fIR0kRZiI_AT" colab={"base_uri": "https://localhost:8080/"} outputId="d67349a5-f57e-40b7-86d3-9a4d54cd8890"
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
```

```python id="iscU3UoVJBXj" colab={"base_uri": "https://localhost:8080/"} outputId="faae1844-08aa-4ec2-f062-11a2b3bdaf07"
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
```

<!-- #region id="WLQULyAvJC3X" -->
You can find the class names in the `class_names` attribute on these datasets. These correspond to the directory names in alphabetical order.
<!-- #endregion -->

```python id="ZHAxkHX5JD3k" colab={"base_uri": "https://localhost:8080/"} outputId="f2358377-653b-483f-a630-67b2291aa13a"
class_names = train_ds.class_names
print(class_names)
```

<!-- #region id="_uoVvxSLJW9m" -->
Visualize the data

Here are the first 9 images from the training dataset.
<!-- #endregion -->

```python id="wBmEA9c0JYes" colab={"base_uri": "https://localhost:8080/", "height": 591} outputId="92a91451-83f8-49ae-b237-8bc11035e72b"
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
```

<!-- #region id="5M6BXtXFJdW0" -->
You will train a model using these datasets by passing them to `model.fit` in a moment. If you like, you can also manually iterate over the dataset and retrieve batches of images:
<!-- #endregion -->

```python id="2-MfMoenJi8s" colab={"base_uri": "https://localhost:8080/"} outputId="acb7ccff-19bb-4e08-b02b-17eaf836b924"
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break
```

<!-- #region id="Wj4FrKxxJkoW" -->
The `image_batch` is a tensor of the shape `(32, 180, 180, 3)`. This is a batch of 32 images of shape `180x180x3` (the last dimension refers to color channels RGB). The `label_batch` is a tensor of the shape `(32,)`, these are corresponding labels to the 32 images. 

You can call `.numpy()` on the `image_batch` and `labels_batch` tensors to convert them to a `numpy.ndarray`.

<!-- #endregion -->

<!-- #region id="4Dr0at41KcAU" -->
### Configure the dataset for performance

Let's make sure to use buffered prefetching so you can yield data from disk without having I/O become blocking. These are two important methods you should use when loading data.

`Dataset.cache()` keeps the images in memory after they're loaded off disk during the first epoch. This will ensure the dataset does not become a bottleneck while training your model. If your dataset is too large to fit into memory, you can also use this method to create a performant on-disk cache.

`Dataset.prefetch()` overlaps data preprocessing and model execution while training. 

Interested readers can learn more about both methods, as well as how to cache data to disk in the [data performance guide](https://www.tensorflow.org/guide/data_performance#prefetching).
<!-- #endregion -->

```python id="nOjJSm7DKoZA"
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
```

<!-- #region id="8GUnmPF4JvEf" -->
## Standardize the data
<!-- #endregion -->

<!-- #region id="e56VXHMWJxYT" -->
The RGB channel values are in the `[0, 255]` range. This is not ideal for a neural network; in general you should seek to make your input values small. Here, you will standardize values to be in the `[0, 1]` range by using a Rescaling layer.
<!-- #endregion -->

```python id="PEYxo2CTJvY9"
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
```

<!-- #region id="8aGpkwFaIw4i" -->
Note: The Keras Preprocessing utilities and layers introduced in this section are currently experimental and may change.
<!-- #endregion -->

<!-- #region id="Bl4RmanbJ4g0" -->
There are two ways to use this layer. You can apply it to the dataset by calling map:
<!-- #endregion -->

```python id="X9o9ESaJJ502" colab={"base_uri": "https://localhost:8080/"} outputId="42bb8ab4-4e8a-4062-a207-7dea150068a7"
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image)) 
```

<!-- #region id="XWEOmRSBJ9J8" -->
Or, you can include the layer inside your model definition, which can simplify deployment. Let's use the second approach here.
<!-- #endregion -->

<!-- #region id="XsRk1xCwKZR4" -->
Note: you previously resized images using the `image_size` argument of `image_dataset_from_directory`. If you want to include the resizing logic in your model as well, you can use the [Resizing](https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/Resizing) layer.
<!-- #endregion -->

<!-- #region id="WcUTyDOPKucd" -->
## Create the model

The model consists of three convolution blocks with a max pool layer in each of them. There's a fully connected layer with 128 units on top of it that is activated by a `relu` activation function. This model has not been tuned for high accuracy, the goal of this tutorial is to show a standard approach. 
<!-- #endregion -->

```python id="QR6argA1K074"
num_classes = 5

model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])
```

<!-- #region id="EaKFzz72Lqpg" -->
## Compile the model

For this tutorial, choose the `optimizers.Adam` optimizer and `losses.SparseCategoricalCrossentropy` loss function. To view training and validation accuracy for each training epoch, pass the `metrics` argument.
<!-- #endregion -->

```python id="jloGNS1MLx3A"
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

<!-- #region id="aMJ4DnuJL55A" -->
## Model summary

View all the layers of the network using the model's `summary` method:
<!-- #endregion -->

```python id="llLYH-BXL7Xe" colab={"base_uri": "https://localhost:8080/"} outputId="70a25660-23bf-40b9-ea5a-b7b98fcbc62f"
model.summary()
```

<!-- #region id="NiYHcbvaL9H-" -->
## Train the model
<!-- #endregion -->

```python id="5fWToCqYMErH" colab={"base_uri": "https://localhost:8080/"} outputId="63dece9a-c20a-43aa-bf7f-d9e3397f83b8"
epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
```

<!-- #region id="SyFKdQpXMJT4" -->
## Visualize training results
<!-- #endregion -->

<!-- #region id="dFvOvmAmMK9w" -->
Create plots of loss and accuracy on the training and validation sets.
<!-- #endregion -->

```python id="jWnopEChMMCn" colab={"base_uri": "https://localhost:8080/", "height": 499} outputId="ec75287f-8fcd-4418-f83d-6fda953ef643"
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
```

<!-- #region id="hO_jT7HwMrEn" -->
As you can see from the plots, training accuracy and validation accuracy are off by large margin and the model has achieved only around 60% accuracy on the validation set.

Let's look at what went wrong and try to increase the overall performance of the model.
<!-- #endregion -->

<!-- #region id="hqtyGodAMvNV" -->
## Overfitting
<!-- #endregion -->

<!-- #region id="ixsz9XFfMxcu" -->
In the plots above, the training accuracy is increasing linearly over time, whereas validation accuracy stalls around 60% in the training process. Also, the difference in accuracy between training and validation accuracy is noticeable—a sign of [overfitting](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit).

When there are a small number of training examples, the model sometimes learns from noises or unwanted details from training examples—to an extent that it negatively impacts the performance of the model on new examples. This phenomenon is known as overfitting. It means that the model will have a difficult time generalizing on a new dataset.

There are multiple ways to fight overfitting in the training process. In this tutorial, you'll use *data augmentation* and add *Dropout* to your model.
<!-- #endregion -->

<!-- #region id="BDMfYqwmM1C-" -->
## Data augmentation
<!-- #endregion -->

<!-- #region id="GxYwix81M2YO" -->
Overfitting generally occurs when there are a small number of training examples. [Data augmentation](https://www.tensorflow.org/tutorials/images/data_augmentation) takes the approach of generating additional training data from your existing examples by augmenting them using random transformations that yield believable-looking images. This helps expose the model to more aspects of the data and generalize better.

You will implement data augmentation using experimental [Keras Preprocessing Layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/?version=nightly). These can be included inside your model like other layers, and run on the GPU.
<!-- #endregion -->

```python id="9J80BAbIMs21"
data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(img_height, 
                                                              img_width,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)
```

<!-- #region id="PN4k1dK3S6eV" -->
Let's visualize what a few augmented examples look like by applying data augmentation to the same image several times:
<!-- #endregion -->

```python id="7Z90k539S838" colab={"base_uri": "https://localhost:8080/", "height": 575} outputId="372e4f3b-2793-4d7b-f434-3684cf40b045"
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
  for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")
```

<!-- #region id="tsjXCBLYYNs5" -->
You will use data augmentation to train a model in a moment.
<!-- #endregion -->

<!-- #region id="ZeD3bXepYKXs" -->
## Dropout

Another technique to reduce overfitting is to introduce [Dropout](https://developers.google.com/machine-learning/glossary#dropout_regularization) to the network, a form of *regularization*.

When you apply Dropout to a layer it randomly drops out (by setting the activation to zero) a number of output units from the layer during the training process. Dropout takes a fractional number as its input value, in the form such as 0.1, 0.2, 0.4, etc. This means dropping out 10%, 20% or 40% of the output units randomly from the applied layer.

Let's create a new neural network using `layers.Dropout`, then train it using augmented images.
<!-- #endregion -->

```python id="2Zeg8zsqXCsm"
model = Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])
```

<!-- #region id="L4nEcuqgZLbi" -->
## Compile and train the model
<!-- #endregion -->

```python id="EvyAINs9ZOmJ"
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

```python id="wWLkKoKjZSoC" colab={"base_uri": "https://localhost:8080/"} outputId="576a75ee-98c3-414e-a59c-5f25de288a3b"
model.summary()
```

```python id="LWS-vvNaZDag" colab={"base_uri": "https://localhost:8080/"} outputId="c850c4d1-c986-49ff-87ce-f301642d7dcb"
epochs = 15
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
```

<!-- #region id="Lkdl8VsBbZOu" -->
## Visualize training results

After applying data augmentation and Dropout, there is less overfitting than before, and training and validation accuracy are closer aligned. 
<!-- #endregion -->

```python id="dduoLfKsZVIA" colab={"base_uri": "https://localhost:8080/", "height": 499} outputId="dcc7be5f-81c7-48d7-9cae-0f44e4e1c0a9"
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
```

<!-- #region id="dtv5VbaVb-3W" -->
## Predict on new data
<!-- #endregion -->

<!-- #region id="10buWpJbcCQz" -->
Finally, let's use our model to classify an image that wasn't included in the training or validation sets.
<!-- #endregion -->

<!-- #region id="NKgMZ4bDcHf7" -->
Note: Data augmentation and Dropout layers are inactive at inference time.
<!-- #endregion -->

```python id="dC40sRITBSsQ" colab={"base_uri": "https://localhost:8080/"} outputId="f9ea340e-894f-4b12-860e-eccb603c668a"
sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

img = keras.preprocessing.image.load_img(
    sunflower_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
```

```python colab={"base_uri": "https://localhost:8080/"} id="W13IC5AzAt1c" outputId="3c9b6c75-c18b-4584-dab2-53381dab4859"
# !wget -O 'rose.jpg' "https://images-na.ssl-images-amazon.com/images/I/81kNpvvxsdL._SX466_.jpg"
!wget -O 'rose2.jpg' "https://cdn.britannica.com/99/96099-050-96F791B5/tea-rose-garden-roses-plants-stem-flowers.jpg"
```

```python id="DeuqAlfN-fpZ"
!pip install gradio
import gradio as gr
from PIL import Image
```

```python id="upO-2Rqd-qX6"
def classify_images(im):
  im = Image.fromarray(im.astype('uint8'), 'RGB')
  im = im.resize((img_height, img_width))
  img_array = np.array(im).reshape((-1, img_height, img_width, 3))
  # img_array = im.reshape((-1, img_height, img_width, 3))
  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])
  return {class_names[i]: float(score[i]) for i in range(5)}
```

```python colab={"base_uri": "https://localhost:8080/"} id="IkqTkspfABVh" outputId="2a7545d3-2b5c-4819-98f1-3b85bd93e5b6"
img = keras.preprocessing.image.load_img(
    sunflower_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
# img_array.shape
classify_images(img_array)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 609} id="e6ZH34Rs_mYa" outputId="9450e2c2-f210-4d62-8cf6-f9dafa153688"
imagein = gr.inputs.Image()
label = gr.outputs.Label(num_top_classes=5)
sample_images = [
                 ["rose.jpg"],
                 ["rose2.jpg"],
]
gr.Interface(
    [classify_images],
    imagein,
    label,
    title="Flower Classification",
    description="This model identifies the flower type.",
    examples=sample_images).launch();
```

<!-- #region id="yeEvf9oACnWe" -->
<!-- #endregion -->
