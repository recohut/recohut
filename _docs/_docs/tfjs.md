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

# Tensorflow js model
> Creating and serving a Tensorflow javascript model in the browser.

- toc: false
- badges: true
- comments: true
- categories: [tfjs, keras, serving]

<!-- #region colab_type="text" id="taEElfKSBHge" -->
In this tutorial we learn how to 


1.   Train a model with Keras with GPU
2.   Convert a model to web format 
3.   Upload the model to GitHub Pages 
4.   Prediction using TensorFlow.js 


<!-- #endregion -->

<!-- #region colab_type="text" id="b1JCrGrePvKp" -->
We will create a simple model that models XOR operation. Given two inputs $(x_0, x_1)$ it outputs $y$

$$\left[\begin{array}{cc|c}  
 x_0 & x_1 & y\\
 0 & 0 & 0\\  
 0 & 1 & 1\\
 1 & 0 & 1\\
 1 & 1 & 0
\end{array}\right]$$
<!-- #endregion -->

## Build the model

<!-- #region colab_type="text" id="WKYiL-oYR0yk" -->
Imports 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="a-UbSG-DR3ID" outputId="c8c804fc-d97d-45df-dff4-6cc729f21951"
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np 
```

<!-- #region colab_type="text" id="UvBySAixR4Ca" -->
Initialize the inputs 
<!-- #endregion -->

```python colab={} colab_type="code" id="Hj65iQS6R6pO"
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])
```

<!-- #region colab_type="text" id="TTyQKnEgSBQb" -->
Create the model 
<!-- #endregion -->

```python colab={} colab_type="code" id="ivnpyw3ZSAF9"
model = Sequential()
model.add(Dense(8, input_dim=2))
model.add(Activation('tanh'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=sgd)
```

<!-- #region colab_type="text" id="zzrpHO1XSIeJ" -->
Train the model 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="jRwYsPJxRrYT" outputId="733d8a80-8589-4d5e-cbc9-9e8f4c8f2c1e"
model.fit(X, y, batch_size=1, epochs=1000, verbose= 0)
```

<!-- #region colab_type="text" id="VHlJ2cmpSbZ7" -->
Predict the output 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 85} colab_type="code" id="ky1bM2EiSHYt" outputId="cd528c04-469d-479f-eda5-ed610716a9f0"
print(model.predict_proba(X))
```

<!-- #region colab_type="text" id="vvdWZCRslZUz" -->
Save the model 
<!-- #endregion -->

```python colab={} colab_type="code" id="mxRke-l9lXfY"
model.save('saved_model/keras.h5')
```

<!-- #region colab_type="text" id="glkP5CvySfgK" -->
## Convert the model 
<!-- #endregion -->

<!-- #region colab_type="text" id="q30sPc63lbvw" -->
Download the library 
<!-- #endregion -->

```python colab={} colab_type="code" id="-FSJVtS9SiVi"
!pip install tensorflowjs
```

<!-- #region colab_type="text" id="HWCP02udldLr" -->
Convert the model 
<!-- #endregion -->

```python colab={} colab_type="code" id="DuQP_mkeSkKL"
!tensorflowjs_converter --input_format keras saved_model/keras.h5 web_model
```

<!-- #region colab_type="text" id="dr8MnQUbUBY7" -->
## Create a web page to serve the model
<!-- #endregion -->

<!-- #region colab_type="text" id="SC9QaDQreTDr" -->
Import TensorFlow.js 
<!-- #endregion -->

```python colab={} colab_type="code" id="iwJQerK2eA_u"
header = '<head><script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.5.2/dist/tf.min.js"> </script>\n'
```

<!-- #region colab_type="text" id="GE_Y5U3UeW6U" -->
Code for loading the web model. We predict a tensor of zeros and show the result in the page. 
<!-- #endregion -->

```python colab={} colab_type="code" id="kpGEMkjJecBM"
script = '\
<script>\n\
          async function loadModel(){ \n\
              model = await tf.loadLayersModel(\'web_model/model.json\') \n\
              y = model.predict(tf.zeros([1,2])) \n\
              document.getElementById(\'out\').innerHTML = y.dataSync()[0] \n\
          } \n\
          loadModel() \n\
</script>\n\
</head> \n'
```

<!-- #region colab_type="text" id="0TDOfXR6f9tp" -->
Body of the page
<!-- #endregion -->

```python colab={} colab_type="code" id="cf5VErepf9H0"
body = '\
<body>\n\
        <p id =\'out\'></p> \n\
</body>'
```

<!-- #region colab_type="text" id="2DaBOiA-jTER" -->
Save the code as html file
<!-- #endregion -->

```python colab={} colab_type="code" id="pM6JIkRCglMu"
with open('index.html','w') as f:
  f.write(header+script+body)
  f.close()
```
