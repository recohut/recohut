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

<!-- #region id="uHigh05D2Pw3" colab_type="text" -->
## Installation
<!-- #endregion -->

```python id="5QAA0ZyCzGrh" colab_type="code" colab={}
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
```

<!-- #region id="kjNKh3kU2cwf" colab_type="text" -->
## Dataset
<!-- #endregion -->

<!-- #region id="X_gioF2X2eZq" colab_type="text" -->
#### The weather dataset
This tutorial uses a <a href="https://www.bgc-jena.mpg.de/wetter/" class="external">[weather time series dataset</a> recorded by the <a href="https://www.bgc-jena.mpg.de" class="external">Max Planck Institute for Biogeochemistry</a>.

This dataset contains 14 different features such as air temperature, atmospheric pressure, and humidity. These were collected every 10 minutes, beginning in 2003. For efficiency, you will use only the data collected between 2009 and 2016. This section of the dataset was prepared by François Chollet for his book [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python).
<!-- #endregion -->

```python id="oS1ZQ2ei2XAp" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 51} outputId="e1a0b267-1b88-44a7-e6f8-18f690f133f4" executionInfo={"status": "ok", "timestamp": 1586434291823, "user_tz": -330, "elapsed": 4290, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)
```

```python id="hhQln-uV2idp" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 306} outputId="266e6808-b1da-403f-c9bd-48ef827e3898" executionInfo={"status": "ok", "timestamp": 1586434300743, "user_tz": -330, "elapsed": 2057, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
df = pd.read_csv(csv_path)
df.head()
```

<!-- #region id="47ednkzv2yE9" colab_type="text" -->
## Data preparation
<!-- #endregion -->

<!-- #region id="4MwHfNze20vB" colab_type="text" -->
As you can see above, an observation is recorded every 10 mintues. This means that, for a single hour, you will have 6 observations. Similarly, a single day will contain 144 (6x24) observations.

Given a specific time, let's say you want to predict the temperature 6 hours in the future. In order to make this prediction, you choose to use 5 days of observations. Thus, you would create a window containing the last 720(5x144) observations to train the model. Many such configurations are possible, making this dataset a good one to experiment with.

The function below returns the above described windows of time for the model to train on. The parameter history_size is the size of the past window of information. The target_size is how far in the future does the model need to learn to predict. The target_size is the label that needs to be predicted.
<!-- #endregion -->

```python id="rqUjujYC2kuX" colab_type="code" colab={}
def univariate_data(dataset, start_index, end_index, history_size, target_size):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i)
    # Reshape data from (history_size,) to (history_size, 1)
    data.append(np.reshape(dataset[indices], (history_size, 1)))
    labels.append(dataset[i+target_size])
  return np.array(data), np.array(labels)
```

<!-- #region id="M2oF7vEh28sA" colab_type="text" -->
In both the following tutorials, the first 300,000 rows of the data will be the training dataset, and there remaining will be the validation dataset. This amounts to ~2100 days worth of training data.
<!-- #endregion -->

```python id="oPdgdolM2352" colab_type="code" colab={}
TRAIN_SPLIT = 300000
```

```python id="SU4F_OaP2-N5" colab_type="code" colab={}
tf.random.set_seed(13)
```

<!-- #region id="ZrV2SqkC3Mg0" colab_type="text" -->
## Univariate forecasting
<!-- #endregion -->

<!-- #region id="_k64_kiJ3P5m" colab_type="text" -->
First, you will train a model using only a single feature (temperature), and use it to make predictions for that value in the future.
<!-- #endregion -->

```python id="XdtUJeMJ2_cv" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 136} outputId="6202accb-de5f-4137-a34b-cdeda9d31601" executionInfo={"status": "ok", "timestamp": 1586434484572, "user_tz": -330, "elapsed": 1498, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
uni_data = df['T (degC)']
uni_data.index = df['Date Time']
uni_data.head()
```

```python id="sQ9YU8f-3SM5" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 443} outputId="9e5b8bfb-6807-4c59-bc9b-6bc46b218c75" executionInfo={"status": "ok", "timestamp": 1586434490217, "user_tz": -330, "elapsed": 2506, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
uni_data.plot(subplots=True)
```

```python id="znZQUWac3TVW" colab_type="code" colab={}
uni_data = uni_data.values
```

<!-- #region id="w3_M9KGk3WNW" colab_type="text" -->
It is important to scale features before training a neural network. Standardization is a common way of doing this scaling by subtracting the mean and dividing by the standard deviation of each feature.You could also use a tf.keras.utils.normalize method that rescales the values into a range of [0,1].
<!-- #endregion -->

```python id="EogISl8-3UuQ" colab_type="code" colab={}
uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
uni_train_std = uni_data[:TRAIN_SPLIT].std()

uni_data = (uni_data-uni_train_mean)/uni_train_std
```

<!-- #region id="J-nYWQFk3c4o" colab_type="text" -->
Let's now create the data for the univariate model. For part 1, the model will be given the last 20 recorded temperature observations, and needs to learn to predict the temperature at the next time step.
<!-- #endregion -->

```python id="KRAUhCki3Xda" colab_type="code" colab={}
univariate_past_history = 20
univariate_future_target = 0

x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                           univariate_past_history,
                                           univariate_future_target)
x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                       univariate_past_history,
                                       univariate_future_target)
```

```python id="tLNMCFUL3eSf" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 425} outputId="7caf7e95-1b8a-4d97-c57e-693a3517aa57" executionInfo={"status": "ok", "timestamp": 1586434540936, "user_tz": -330, "elapsed": 1386, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
print ('Single window of past history')
print (x_train_uni[0])
print ('\n Target temperature to predict')
print (y_train_uni[0])
```

```python id="lfNp_QVc3fwW" colab_type="code" colab={}

```
