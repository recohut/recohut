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

<!-- #region colab_type="text" id="Za8-Nr5k11fh" -->
##### Copyright 2018 The TensorFlow Authors.
<!-- #endregion -->

```python cellView="form" colab_type="code" id="Eq10uEbw0E4l" colab={}
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

<!-- #region colab_type="text" id="06ndLauQxiQm" -->
# Train Your Own Model and Convert It to TFLite
<!-- #endregion -->

<!-- #region colab_type="text" id="Dtav_aq2xh6n" -->
<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_lite/tflite_c04_exercise_convert_model_to_tflite_solution.ipynb">
    <img src="https://www.tensorflow.org/images/colab_logo_32px.png" />
    Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_lite/tflite_c04_exercise_convert_model_to_tflite_solution.ipynb">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub</a>
  </td>
</table>
<!-- #endregion -->

<!-- #region colab_type="text" id="Ka96-ajYzxVU" -->
This notebook uses the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset which contains 70,000 grayscale images in 10 categories. The images show individual articles of clothing at low resolution (28 by 28 pixels), as seen here:

<table>
  <tr><td>
    <img src="https://tensorflow.org/images/fashion-mnist-sprite.png"
         alt="Fashion MNIST sprite"  width="600">
  </td></tr>
  <tr><td align="center">
    <b>Figure 1.</b> <a href="https://github.com/zalandoresearch/fashion-mnist">Fashion-MNIST samples</a> (by Zalando, MIT License).<br/>&nbsp;
  </td></tr>
</table>

Fashion MNIST is intended as a drop-in replacement for the classic [MNIST](http://yann.lecun.com/exdb/mnist/) dataset—often used as the "Hello, World" of machine learning programs for computer vision. The MNIST dataset contains images of handwritten digits (0, 1, 2, etc.) in a format identical to that of the articles of clothing we'll use here.

This uses Fashion MNIST for variety, and because it's a slightly more challenging problem than regular MNIST. Both datasets are relatively small and are used to verify that an algorithm works as expected. They're good starting points to test and debug code.

We will use 60,000 images to train the network and 10,000 images to evaluate how accurately the network learned to classify images. You can access the Fashion MNIST directly from TensorFlow. Import and load the Fashion MNIST data directly from TensorFlow:
<!-- #endregion -->

<!-- #region colab_type="text" id="rjOAfhgd__Sp" -->
# Setup
<!-- #endregion -->

```python colab_type="code" id="pfyZKowNAQ4j" colab={}
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pathlib


print(tf.__version__)
```

<!-- #region colab_type="text" id="tadPBTEiAprt" -->
# Download Fashion MNIST Dataset
<!-- #endregion -->

```python colab_type="code" id="jmSkLCyRKqKB" colab={}
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
```

```python colab_type="code" id="XcNwi6nFKneZ" colab={}
splits, info = tfds.load('fashion_mnist', with_info=True, as_supervised=True, 
                         split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'])

(train_examples, validation_examples, test_examples) = splits

num_examples = info.splits['train'].num_examples
num_classes = info.features['label'].num_classes
```

```python colab_type="code" id="-eAv71FRm4JE" colab={}
class_names = ['T-shirt_top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```

```python colab_type="code" id="hXe6jNokqX3_" colab={}
with open('labels.txt', 'w') as f:
  f.write('\n'.join(class_names))
```

```python colab_type="code" id="iubWCThbdN8K" colab={}
IMG_SIZE = 28
```

<!-- #region colab_type="text" id="ZAkuq0V0Aw2X" -->
# Preprocessing data
<!-- #endregion -->

<!-- #region colab_type="text" id="_5SIivkunKCC" -->
## Preprocess
<!-- #endregion -->

```python colab_type="code" id="BwyhsyGydHDl" colab={}
def format_example(image, label):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  image = image / 255.0
  return image, label
```

```python colab_type="code" id="HAlBlXOUMwqe" colab={}
BATCH_SIZE = 32
```

<!-- #region colab_type="text" id="JM4HfIJtnNEk" -->
## Create a Dataset from images and labels
<!-- #endregion -->

```python colab_type="code" id="uxe2I3oxLDhq" colab={}
train_batches = train_examples.cache().shuffle(num_examples//4).batch(BATCH_SIZE).map(format_example).prefetch(1)
validation_batches = validation_examples.cache().batch(BATCH_SIZE).map(format_example).prefetch(1)
test_batches = test_examples.cache().batch(1).map(format_example)
```

<!-- #region colab_type="text" id="M-topQaOm_LM" -->
# Building the model
<!-- #endregion -->

```python colab_type="code" id="kDqcwksFB1bh" colab={}
model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(16, 3, activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(10)
])

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])
```

<!-- #region colab_type="text" id="zEMOz-LDnxgD" -->
## Train
<!-- #endregion -->

```python colab_type="code" id="1fk2faPsjqfU" colab={}
validation_batches
```

```python colab_type="code" id="DGJe_CNvjnhT" colab={}
train_batches, 
```

```python colab_type="code" id="JGlNoRtzCP4_" colab={}
model.fit(train_batches, 
          epochs=10,
          validation_data=validation_batches)
```

<!-- #region colab_type="text" id="TZT9-7w9n4YO" -->
# Exporting to TFLite
<!-- #endregion -->

```python colab_type="code" id="9dq78KBkCV2_" colab={}
export_dir = 'saved_model/1'
tf.saved_model.save(model, export_dir)
```

```python cellView="form" colab_type="code" id="EDGiYrBdE6fl" colab={}
#@title Select mode of optimization
mode = "Speed" #@param ["Default", "Storage", "Speed"]

if mode == 'Storage':
  optimization = tf.lite.Optimize.OPTIMIZE_FOR_SIZE
elif mode == 'Speed':
  optimization = tf.lite.Optimize.OPTIMIZE_FOR_LATENCY
else:
  optimization = tf.lite.Optimize.DEFAULT
```

```python colab_type="code" id="RbcS9C00CzGe" colab={}
# Convert the model.
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
converter.optimizations = [optimization]
tflite_model = converter.convert()
```

```python colab_type="code" id="q5PWCDsTC3El" colab={}
tflite_model_file = 'model.tflite'

with open(tflite_model_file, "wb") as f:
  f.write(tflite_model)
```

<!-- #region colab_type="text" id="SR6wFcQ1Fglm" -->
# Test the model with TFLite interpreter 
<!-- #endregion -->

```python colab_type="code" id="rKcToCBEC-Bu" colab={}
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]
```

```python colab_type="code" id="E8EpFpIBFkq8" colab={}
# Gather results for the randomly sampled test images
predictions = []
test_labels = []
test_images = []

for img, label in test_batches.take(50):
  interpreter.set_tensor(input_index, img)
  interpreter.invoke()
  predictions.append(interpreter.get_tensor(output_index))
  test_labels.append(label[0])
  test_images.append(np.array(img))
```

```python cellView="form" colab_type="code" id="kSjTmi05Tyod" colab={}
#@title Utility functions for plotting
# Utilities for plotting

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  img = np.squeeze(img)

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label.numpy():
    color = 'green'
  else:
    color = 'red'
    
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks(list(range(10)), class_names, rotation='vertical')
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array[0], color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array[0])

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('green')
```

```python cellView="form" colab_type="code" id="ZZwg0wFaVXhZ" colab={}
#@title Visualize the outputs { run: "auto" }
index = 12 #@param {type:"slider", min:1, max:50, step:1}
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(index, predictions, test_labels, test_images)
plt.show()
plot_value_array(index, predictions, test_labels)
plt.show()
```

<!-- #region colab_type="text" id="076bo3FMpRDb" -->
# Download TFLite model and assets

**NOTE: You might have to run to the cell below twice**
<!-- #endregion -->

```python colab_type="code" id="XsPXqPlgZPjE" colab={}
try:
  from google.colab import files

  files.download(tflite_model_file)
  files.download('labels.txt')
except:
  pass
```

<!-- #region colab_type="text" id="H8t7_jRiz9Vw" -->
# Prepare the test images for download (Optional)
<!-- #endregion -->

```python colab_type="code" id="Fi09nIps0gBu" colab={}
!mkdir -p test_images
```

```python colab_type="code" id="sF7EZ63J0hZs" colab={}
from PIL import Image

for index, (image, label) in enumerate(test_batches.take(50)):
  image = tf.cast(image * 255.0, tf.uint8)
  image = tf.squeeze(image).numpy()
  pil_image = Image.fromarray(image)
  pil_image.save('test_images/{}_{}.jpg'.format(class_names[label[0]].lower(), index))
```

```python colab_type="code" id="uM35O-uv0iWS" colab={}
!ls test_images
```

```python colab_type="code" id="aR20r4qW0jVm" colab={}
!zip -qq fmnist_test_images.zip -r test_images/
```

```python colab_type="code" id="tjk4537X0kWN" colab={}
try:
  files.download('fmnist_test_images.zip')
except:
  pass
```
