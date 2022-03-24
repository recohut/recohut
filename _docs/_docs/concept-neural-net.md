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

<!-- #region id="0KvHjPhpI07Z" -->
# Concept - Neural Network
> Training a simple neural net to solve a quadratic equation. Visualizing the actual vs predicted function

- toc: true
- badges: true
- comments: true
- categories: [Concept, Keras, 3D, Visualization]
- image:
<!-- #endregion -->

```python id="RpAal5ZwHd5Z"
import numpy as np
import pandas as pd

import altair as alt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import keras
from keras.preprocessing.image import ImageDataGenerator

%matplotlib inline
```

```python id="Hsy9HpZ7Hbe0"
# Plot a 3d 
def plot3d(X,Y,Z):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, color='y')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    
        
# Visualise the metrics from the model
def metrics(history):
    df = pd.DataFrame(history)
    df.reset_index()
    df["batch"] = df.index + 1
    df = df.melt("batch", var_name="name")
    df["val"] = df.name.str.startswith("val")
    df["type"] = df["val"]
    df["metrics"] = df["val"]
    df.loc[df.val == False, "type"] = "training"
    df.loc[df.val == True, "type"] = "validation"
    df.loc[df.val == False, "metrics"] = df.name
    df.loc[df.val == True, "metrics"] = df.name.str.split("val_", expand=True)[1]
    df = df.drop(["name", "val"], axis=1)
    
    base = alt.Chart().encode(
        x = "batch:Q",
        y = "value:Q",
        color = "type"
    ).properties(width = 300, height = 300)

    layers = base.mark_circle(size = 50).encode(tooltip = ["batch", "value"]) + base.mark_line()
    chart = layers.facet(column='metrics:N', data=df).resolve_scale(y='independent')    
    
    return chart
```

<!-- #region id="a3go5JB2HNFB" -->
Classical programming is all about creating a function that helps us to process input data and get the desired output.

In the learning paradigm, we change the process so that given a set of examples of input data and desired output, we aim to learn the function that can process the data.

- In machine learning, we end up handcrafting the features and then learn the function to get the desired output
- In deep learning, we want to both learn the features and the function together to get the desired output

## Theory of Deep Learning
<!-- #endregion -->

<!-- #region id="aba8WE_XHNFD" -->
We will start with why deep learning works and explain the basis of Universal Approximation

Let us take a non-linear function - a saddle function

$$ Z = 2X^2 - 3Y^2  + 1 + \epsilon $$
<!-- #endregion -->

<!-- #region id="LsEU6oc2HNFE" -->
## Problem: A Noisy Function
<!-- #endregion -->

```python id="g5Y_10jSHNFI"
x = np.arange(-1,1,0.01)
y = np.arange(-1,1,0.01)
```

```python id="Y-8bTGZsHNFK"
X, Y = np.meshgrid(x, y)
c = np.ones((200,200))
e = np.random.rand(200,200)*0.1
```

```python id="LddO9CHnHNFL"
Z = 2*X*X - 3*Y*Y + 5*c + e
```

```python id="YUDquwLZHNFM" colab={"base_uri": "https://localhost:8080/", "height": 466} outputId="1d26429f-345e-47bb-cb8a-436c76ff5c09"
plot3d(X,Y,Z)
```

<!-- #region id="qV1qhm-eHNFO" -->
## Using Neural Network
<!-- #endregion -->

<!-- #region id="paNd5iOJHNFP" -->
### Step 0: Load the Keras Model
<!-- #endregion -->

```python id="v3oOdmN0HNFP"
from keras.models import Sequential
from keras.layers import Dense
```

<!-- #region id="Ir4HH9WYHNFQ" -->
### Step 1: Create the input and output
<!-- #endregion -->

```python id="goIbn963HNFR"
input_xy = np.c_[X.reshape(-1),Y.reshape(-1)]
output_z = Z.reshape(-1)
```

```python id="QBm9v2sEHNFR" colab={"base_uri": "https://localhost:8080/"} outputId="6ff0da45-937a-4223-9e1e-60b12c01417f"
output_z.shape, input_xy.shape
```

<!-- #region id="qP2SKVNAHNFS" -->
### Step 2: Create the Transformation & Prediction Model
<!-- #endregion -->

```python id="k9P9BQn0HNFT"
model = Sequential()
model.add(Dense(64, input_dim=2,  activation="relu"))
model.add(Dense(32, input_dim=2,  activation="relu"))
model.add(Dense(1))
```

```python id="ueDmVRrqHNFU" colab={"base_uri": "https://localhost:8080/"} outputId="1a068518-9132-4639-87a6-4898800c2dd6"
model.summary()
```

<!-- #region id="edVNuJwwHNFV" -->
### Step 3: Compile the Model - Loss, Optimizer and Fit the Model
<!-- #endregion -->

```python id="4bZyQucHHNFW"
model.compile(loss='mean_squared_error', optimizer="sgd", metrics=["mse"])
```

```python id="-TIZr-MZHNFX" colab={"base_uri": "https://localhost:8080/"} outputId="130701c1-762d-4bae-e2d7-9ec7e5041fac"
%%time
output = model.fit(input_xy, output_z, epochs=10, validation_split=0.2, shuffle=True, verbose=1)
```

<!-- #region id="ULemGoLhHNFY" -->
### Step 4: Evaluate Model Performance
<!-- #endregion -->

```python id="FJdvcekvHNFZ" colab={"base_uri": "https://localhost:8080/", "height": 426} outputId="6068d043-5bd2-4449-ba28-da639c39f8b2"
metrics(output.history)
```

<!-- #region id="HCrOMKlTHNFa" -->
### Step 5: Make Prediction from the model
<!-- #endregion -->

```python id="w640wIuPHNFb"
Z_pred = model.predict(input_xy).reshape(200,200)
```

```python id="u6u8UydGHNFe" colab={"base_uri": "https://localhost:8080/", "height": 466} outputId="310ae52e-e5c2-483d-e880-bb3ba6e60174"
plot3d(X,Y,Z_pred)
```

<!-- #region id="6VOQQBABHNFf" -->
## Experimentation / Questions
<!-- #endregion -->

<!-- #region id="dngjULU3HNFf" -->
- Try changing the activation to a "linear" and see whether you can predict the function or not 
- Try adding more layers to the network
- Try changing the number of layers in the network
<!-- #endregion -->
