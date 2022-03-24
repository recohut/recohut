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

<!-- #region colab_type="text" id="UysiGN3tGQHY" -->
# Running TFLite models
<!-- #endregion -->

<!-- #region colab_type="text" id="2hOrvdmswy5O" -->
<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_lite/tflite_c01_linear_regression.ipynb">
    <img src="https://www.tensorflow.org/images/colab_logo_32px.png" />
    Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_lite/tflite_c01_linear_regression.ipynb">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub</a>
  </td>
</table>
<!-- #endregion -->

<!-- #region colab_type="text" id="W-VhTkyTGcaQ" -->
## Setup
<!-- #endregion -->

```python colab_type="code" id="dy4BcTjBFTWx" colab={}
import tensorflow as tf

import pathlib
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
```

<!-- #region colab_type="text" id="ceibQLDeGhI4" -->
## Create a basic model of the form y = mx + c
<!-- #endregion -->

```python colab_type="code" id="YIBCsjQNF46Z" colab={}
# Create a simple Keras model.
x = [-1, 0, 1, 2, 3, 4]
y = [-3, -1, 1, 3, 5, 7]

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(x, y, epochs=200, verbose=1)
```

<!-- #region colab_type="text" id="EjsB-QICGt6L" -->
## Generate a SavedModel
<!-- #endregion -->

```python colab_type="code" id="a9xcbK7QHOfm" colab={}
export_dir = 'saved_model/1'
tf.saved_model.save(model, export_dir)
```

<!-- #region colab_type="text" id="RRtsNwkiGxcO" -->
## Convert the SavedModel to TFLite
<!-- #endregion -->

```python colab_type="code" id="TtM8yKTVTpD3" colab={}
# Convert the model.
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
tflite_model = converter.convert()
```

```python colab_type="code" id="4idYulcNHTdO" colab={}
tflite_model_file = pathlib.Path('model.tflite')
tflite_model_file.write_bytes(tflite_model)
```

<!-- #region colab_type="text" id="HgGvp2yBG25Q" -->
## Initialize the TFLite interpreter to try it out
<!-- #endregion -->

```python colab_type="code" id="DOt94wIWF8m7" colab={}
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
```

```python colab_type="code" id="JGYkEK08F8qK" colab={}
# Test the TensorFlow Lite model on random input data.
input_shape = input_details[0]['shape']
inputs, outputs = [], []
for _ in range(100):
  input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
  interpreter.set_tensor(input_details[0]['index'], input_data)

  interpreter.invoke()
  tflite_results = interpreter.get_tensor(output_details[0]['index'])

  # Test the TensorFlow model on random input data.
  tf_results = model(tf.constant(input_data))
  output_data = np.array(tf_results)
  
  inputs.append(input_data[0][0])
  outputs.append(output_data[0][0])
```

<!-- #region colab_type="text" id="t1gQGH1KWAgW" -->
## Visualize the model
<!-- #endregion -->

```python colab_type="code" id="ccvQ1mEJVrqo" colab={}
plt.plot(inputs, outputs, 'r')
plt.show()
```

<!-- #region colab_type="text" id="WbugMH6yKvtd" -->
## Download the TFLite model file
<!-- #endregion -->

```python colab_type="code" id="FOAIMETeJmkc" colab={}
try:
  from google.colab import files
  files.download(tflite_model_file)
except:
  pass
```
