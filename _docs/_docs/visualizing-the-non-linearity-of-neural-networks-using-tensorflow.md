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

<!-- #region id="UA9MqQE7UKMJ" -->
In this notebook, we will be looking at a toy dataset and a simple neural network to demonstrate the advantage of the neural networks over linear classifiers. 
<!-- #endregion -->

```python id="fmmYrxJNlwDx" executionInfo={"status": "ok", "timestamp": 1637384794030, "user_tz": -330, "elapsed": 729, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import numpy as np
import matplotlib.pyplot as plt
```

<!-- #region id="YNXQ6oyHUvoC" -->
The following utility functions are for creating random data and plotting. 
<!-- #endregion -->

```python id="qSMBv9QBo9VU" executionInfo={"status": "ok", "timestamp": 1637384794853, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def create_data():
  x_1 = np.random.randn(100,2) * 0.1 + 0.5
  x_2 = np.random.randn(100,2) * 0.1 + np.array([0.5,-0.5])
  x_4 = np.random.randn(100,2) * 0.1 + np.array([-0.5,0.5])
  x_3 = np.random.randn(100,2) * 0.1 - 0.5
  x = np.concatenate((x_1, x_2, x_3, x_4))
  
  y_1 = np.ones(100)
  y_2 = np.ones(100) * 0
  y_4 = np.ones(100) * 0
  y_3 = np.ones(100)
  y = np.concatenate((y_1,y_2,y_3, y_4))
  y = y.reshape((-1,1))
  
  return x, y
```

```python id="w8Oaj_qJlzp_" executionInfo={"status": "ok", "timestamp": 1637384794854, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def plot_data(x, y):

  plt.plot(x[y[:,0] == 1,0], x[y[:,0]==1,1], 'bx')
  plt.plot(x[y[:,0]==0,0], x[y[:,0]==0,1], 'ro')

def plot_line(c_1, c_2, c):
  # c_1*x + c_2*y + c = 0 

  plt.ylim(-1.2,1.2)
  plt.xlim(-1.2,1.2)
  x = np.linspace(-2,2,100)
  y = (-c_1*x-c)/c_2
  
  plt.plot(x, y)

def plot_lines(W, b):
  for col, c in zip(W.T, b):
    plot_line(col[0], col[1], c)
```

```python id="Y8qFg3wBmRrN" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637384794855, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a33e5d16-1c0b-4e15-8ab6-7adf2b8c2bf3"
x_train, y_train = create_data()
plot_data(x_train,y_train)
```

<!-- #region id="yqQPRAAIVaDI" -->
There are two classes in the data. In the above plot, blue points correspond to the positive class (label 1) and the red points correspond to the negative class (label 0). As you can see from the plot, this data is not linearly separable. In a 2-d input space, this means there is no line that separates the two classes.
<!-- #endregion -->

<!-- #region id="uhUgJ5U7WY0h" -->
Now we will train a neural network on this data and visualize the features learned by the network. 
<!-- #endregion -->

```python id="GXLs7b5SDJlH" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637384799013, "user_tz": -330, "elapsed": 2482, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a6bf8f4b-2fa8-4cdc-8f6e-7d581ff9e4eb"
import tensorflow as tf
print(tf.__version__)
```

```python id="KEB7DEwlDOUn" executionInfo={"status": "ok", "timestamp": 1637384799014, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
tf.keras.backend.clear_session()
```

```python id="2Rr-MKiqA_pD" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637384799014, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2eba041f-1624-4fcb-ad49-7938662ee80f"
tf.random.set_seed(10)
inputs = tf.keras.Input(shape=(2,))
x = tf.keras.layers.Dense(2, activation=tf.nn.tanh)(inputs)
outputs = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()
```

```python id="l2GveRU5cQcE" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637384800052, "user_tz": -330, "elapsed": 1044, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e27cd5fb-f9da-47a9-a45b-1f5e30712022"
plot_data(x_train ,y_train)
b = model.weights[1].numpy()
w = model.weights[0].numpy()
plot_lines(w,b)
```

```python id="9tUxF2QQBxK5" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637384801172, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5b140518-e365-4e20-c417-e1b7761b790f"
plot_data(x_train ,y_train)
b = model.weights[1].numpy().reshape((1,-1))
w = model.weights[0].numpy()
plot_line(w[0,0], w[1,0], b[0,0])    
plot_line(w[0,1], w[1,1], b[0,1]) 
```

```python id="6j0SlgipBTtT" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637384810859, "user_tz": -330, "elapsed": 5919, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e81575ba-20a2-4a76-e634-2c6e7ba5913f"
model.fit(x_train, y_train, batch_size = 16, epochs=100)
```

```python id="58g9B60_HSC6" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637384841869, "user_tz": -330, "elapsed": 713, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="99c249e0-a367-4ee8-f58c-8decb372de11"
model.evaluate(x_train, y_train)
```

```python id="SbmExrOnchB5" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637384812957, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="13175aaf-ca80-4315-baf5-3761150aafcf"
plot_data(x_train ,y_train)
b = model.weights[1].numpy()
w = model.weights[0].numpy()
plot_lines(w,b)
```

```python id="2yE7jY-yO-Qc" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637384822374, "user_tz": -330, "elapsed": 447, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fd67da04-d5dd-4f66-abfc-ab205d3e1b78"
model.weights
```

```python id="h2sb7VpfDold" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637384836725, "user_tz": -330, "elapsed": 538, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e6f12153-6fbc-4d16-9cc8-fc0de5760ba7"
activations = model.get_layer(name='dense')(x_train)
plot_data(activations.numpy(), y_train)
b = model.weights[3].numpy().reshape((1,-1))
w = model.weights[2].numpy()
plot_line(w[0,0], w[1,0], b[0,0]) 
```

<!-- #region id="Dot1k0vivjoJ" -->
Note that in this new feature space our data becomes linearly separable and the line defined by the output layer separates the two classes. The blue points has both a₁ and a₂ coordinates positive because those points in the (x₁, x₂) space are on the positive side of both lines defined by the hidden layer parameters and after applying tanh both coordinates are positive. For the red points one of a₁ and a₂ is positive because in the (x₁, x₂)-space the red points are on the positive side of only one of the lines defined by the hidden layer parameters and depending on this line they have only one of their coordinates in the new feature space positive and the other is negative. This explains the data plot in (a₁,a₂)-space above.
<!-- #endregion -->

<!-- #region id="y3CK73FVrF_6" -->
**Conclusion**: Neural networks learn a new representation of the data which makes it easy to classify with respect to this new representation.
<!-- #endregion -->
