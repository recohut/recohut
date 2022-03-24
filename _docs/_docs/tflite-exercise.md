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

<!-- #region colab_type="text" id="KlUrRaN4w3ct" -->
# Train Your Own Model and Convert It to TFLite
<!-- #endregion -->

<!-- #region colab_type="text" id="H3UojxdNw8J1" -->
<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_lite/tflite_c03_exercise_convert_model_to_tflite.ipynb">
    <img src="https://www.tensorflow.org/images/colab_logo_32px.png" />
    Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_lite/tflite_c03_exercise_convert_model_to_tflite.ipynb">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub</a>
  </td>
</table>
<!-- #endregion -->

<!-- #region colab_type="text" id="pXX-pi1r6NfG" -->
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

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pathlib

print(tf.__version__)
```

<!-- #region colab_type="text" id="tadPBTEiAprt" -->
# Download Fashion MNIST Dataset

<!-- #endregion -->

```python colab_type="code" id="Ds9gfZKzAnkX" colab={}
splits = tfds.Split.ALL.subsplit(weighted=(80, 10, 10))

splits, info = tfds.load('fashion_mnist', with_info=True, as_supervised=True, split=splits)

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

```python colab_type="code" id="q0RxpwTmQN-y" colab={}
IMG_SIZE = 28
```

<!-- #region colab_type="text" id="ZAkuq0V0Aw2X" -->
# Preprocessing data
<!-- #endregion -->

<!-- #region colab_type="text" id="_5SIivkunKCC" -->
## Preprocess
<!-- #endregion -->

```python colab_type="code" id="nQMIkJf9AvJ4" colab={}
# Write a function to normalize and resize the images

def format_example(image, label):
  # Cast image to float32
  image = # YOUR CODE HERE
  # Resize the image if necessary
  image = # YOUR CODE HERE
  # Normalize the image in the range [0, 1]
  image = # YOUR CODE HERE
  return image, label
```

```python colab_type="code" id="oEQP743aMv4C" colab={}
# Set the batch size to 32

BATCH_SIZE = 32
```

<!-- #region colab_type="text" id="JM4HfIJtnNEk" -->
## Create a Dataset from images and labels
<!-- #endregion -->

```python colab_type="code" id="zOL4gSUARFjM" colab={}
# Prepare the examples by preprocessing the them and then batching them (and optionally prefetching them)

# If you wish you can shuffle train set here
train_batches = # YOUR CODE HERE

validation_batches = # YOUR CODE HERE
test_batches = # YOUR CODE HERE
```

<!-- #region colab_type="text" id="M-topQaOm_LM" -->
# Building the model
<!-- #endregion -->

```python colab_type="code" id="4gsYqdIlEFVg" colab={}
"""
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 26, 26, 16)        160       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 16)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 32)        4640      
_________________________________________________________________
flatten (Flatten)            (None, 3872)              0         
_________________________________________________________________
dense (Dense)                (None, 64)                247872    
_________________________________________________________________
dense_1 (Dense)              (None, 10)                650       
=================================================================
Total params: 253,322
Trainable params: 253,322
Non-trainable params: 0
"""
```

```python colab_type="code" id="kDqcwksFB1bh" colab={}
# Build the model shown in the previous cell


model = tf.keras.Sequential([
  # Set the input shape to (28, 28, 1), kernel size=3, filters=16 and use ReLU activation,  
  tf.keras.layers.Conv2D(# YOUR CODE HERE),    
  tf.keras.layers.MaxPooling2D(),
  # Set the number of filters to 32, kernel size to 3 and use ReLU activation 
  tf.keras.layers.Conv2D(# YOUR CODE HERE),
  # Flatten the output layer to 1 dimension
  tf.keras.layers.Flatten(),
  # Add a fully connected layer with 64 hidden units and ReLU activation
  tf.keras.layers.Dense(# YOUR CODE HERE),
  # Attach a final softmax classification head
  tf.keras.layers.Dense(# YOUR CODE HERE)])

# Set the loss and accuracy metrics
model.compile(
    optimizer='adam', 
    loss=# YOUR CODE HERE, 
    metrics=# YOUR CODE HERE)
      
```

<!-- #region colab_type="text" id="zEMOz-LDnxgD" -->
## Train
<!-- #endregion -->

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

# Use the tf.saved_model API to export the SavedModel

# Your Code Here
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

```python colab_type="code" id="SLskPWHsG4Nj" colab={}
optimization
```

```python colab_type="code" id="RbcS9C00CzGe" colab={}
# Use the TFLiteConverter SavedModel API to initialize the converter
converter = # YOUR CODE HERE

# Set the optimzations
converter.optimizations = # YOUR CODE HERE

# Invoke the converter to finally generate the TFLite model
tflite_model = # YOUR CODE HERE
```

```python colab_type="code" id="q5PWCDsTC3El" colab={}
tflite_model_file = 'model.tflite'

with open(tflite_model_file, "wb") as f:
  f.write(tflite_model)
```

<!-- #region colab_type="text" id="SR6wFcQ1Fglm" -->
# Test if your model is working
<!-- #endregion -->

```python colab_type="code" id="O3IFOcUEIzQx" colab={}
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]
```

```python colab_type="code" id="rKcToCBEC-Bu" colab={}
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
index = 49 #@param {type:"slider", min:1, max:50, step:1}
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

<!-- #region colab_type="text" id="VyBVNwAzH3Oe" -->
# Deploying TFLite model
<!-- #endregion -->

<!-- #region colab_type="text" id="pdfa5L6wH87u" -->
Now once you've the trained TFLite model downloaded, you can ahead and deploy this on an Android/iOS application by placing the model assets in the appropriate location.
<!-- #endregion -->

<!-- #region colab_type="text" id="iLY6X8P90L0P" -->
# Prepare the test images for download (Optional)
<!-- #endregion -->

```python colab_type="code" id="G3bjzLj10OJv" colab={}
!mkdir -p test_images
```

```python colab_type="code" id="pVrBZv1-0Py-" colab={}
from PIL import Image

for index, (image, label) in enumerate(test_batches.take(50)):
  image = tf.cast(image * 255.0, tf.uint8)
  image = tf.squeeze(image).numpy()
  pil_image = Image.fromarray(image)
  pil_image.save('test_images/{}_{}.jpg'.format(class_names[label[0]].lower(), index))
```

```python colab_type="code" id="nX0N0M8u0R2s" colab={}
!ls test_images
```

```python colab_type="code" id="LvLht1QM0W8k" colab={}
!zip -qq fmnist_test_images.zip -r test_images/
```

```python colab_type="code" id="FdOq-4sT0X95" colab={}
try:
  files.download('fmnist_test_images.zip')
except:
  pass
```
