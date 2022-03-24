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

<!-- #region id="ZWazUhOh8_3Y" -->
Automated machine learning (AutoML) provides methods to find the optimal neural architecture and the best hyperparameter settings for a given neural network. 
<!-- #endregion -->

<!-- #region id="HWkqlzdM9X-Z" -->
One way to think of machine learning algorithms is that they automate the process of learning relationships between given inputs and outputs. In traditional software engineering, we would have to explicitly write/code these relationships in the form of functions that take in input and return output. In the machine learning world, machine learning models find such functions for us. Although we automate to a certain extent, there is still a lot to be done. Besides mining and cleaning data, here are a few routine tasks to be performed in order to get those functions:
- Choosing a machine learning model (or a model family and then a model)
- Deciding the model architecture (especially in the case of deep learning)
- Choosing hyperparameters
- Adjusting hyperparameters based on validation set performance
- Trying different models (or model families)
<!-- #endregion -->

<!-- #region id="NpLiJ4iE9hdr" -->
These are the kinds of tasks that justify the requirement of a human machine learning expert. Most of these steps are manual and either take a lot of time or need a lot of expertise to discount the required time, and we have far fewer machine learning experts than needed to create and deploy machine learning models that are increasingly popular, valuable, and useful across both industries and academia.

This is where AutoML comes to the rescue. AutoML has become a discipline within the field of machine learning that aims to automate the previously listed steps and beyond.
<!-- #endregion -->

<!-- #region id="bqiEvapr9In8" -->
we will look more broadly at the AutoML tool for PyTorch—Auto-PyTorch—which performs both neural architecture search and hyperparameter search. We will first load the dataset, then define an Auto-PyTorch model search instance, and finally run the model searching routine, which will provide us with a best-performing model.


We will also look at another AutoML tool called Optuna that performs hyperparameter search for a PyTorch model.
<!-- #endregion -->

```python id="Xfxxe7XO9MUM"
!pip install git+https://github.com/shukon/HpBandSter.git
!pip install autoPyTorch==0.0.2
!pip install torchviz==0.0.1
!pip install configspace==0.4.12
```

```python colab={"base_uri": "https://localhost:8080/"} id="3lEXFTX692bJ" executionInfo={"status": "ok", "timestamp": 1631265557564, "user_tz": -330, "elapsed": 1361, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6d9f2e6f-dc66-4f50-e745-d5f1527705c4"
import torch
from torchviz import make_dot
from torchvision import datasets, transforms
from autoPyTorch import AutoNetClassification

import matplotlib.pyplot as plt
import numpy as np
```

```python colab={"base_uri": "https://localhost:8080/"} id="lhKEYr2--Rka" executionInfo={"status": "ok", "timestamp": 1631265557572, "user_tz": -330, "elapsed": 36, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="16112d1b-cb00-453c-e24b-6ad102f6aa89"
train_ds = datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1302,), (0.3069,))]))

test_ds = datasets.MNIST('../data', train=False, 
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1302,), (0.3069,))]))
```

```python id="bBnjQcaN-U1Q"
X_train, X_test, y_train, y_test = train_ds.data.numpy().reshape(-1, 28*28), test_ds.data.numpy().reshape(-1, 28*28) ,train_ds.targets.numpy(), test_ds.targets.numpy()
```

```python colab={"base_uri": "https://localhost:8080/"} id="01t84nHd-Wbx" executionInfo={"status": "ok", "timestamp": 1631266344426, "user_tz": -330, "elapsed": 786876, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c7b35a47-3c65-40e6-eece-8ecc9ac6fb73"
# running Auto-PyTorch
autoPyTorch = AutoNetClassification("tiny_cs",  # config preset
                                    log_level='info',
                                    max_runtime=2000,
                                    min_budget=100,
                                    max_budget=1500)

autoPyTorch.fit(X_train, y_train, validation_split=0.1)
```

```python colab={"base_uri": "https://localhost:8080/"} id="uG83BUEuAQM8" executionInfo={"status": "ok", "timestamp": 1631266373180, "user_tz": -330, "elapsed": 1306, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5cd0cfff-9f31-4b49-8327-875f7d669a43"
y_pred = autoPyTorch.predict(X_test)
print("Accuracy score", np.mean(y_pred.reshape(-1) == y_test))
```

```python colab={"base_uri": "https://localhost:8080/"} id="yBmPCJGsAavW" executionInfo={"status": "ok", "timestamp": 1631266373183, "user_tz": -330, "elapsed": 28, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9c9e2870-de5a-4b31-f16f-2d7e4abd883e"
pytorch_model = autoPyTorch.get_pytorch_model()
print(pytorch_model)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="SRKun9ziAed4" executionInfo={"status": "ok", "timestamp": 1631266373191, "user_tz": -330, "elapsed": 30, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="983aefb2-c42d-4bb9-aa3f-bcc42a59a784"
x = torch.randn(1, pytorch_model[0].in_features)
y = pytorch_model(x)
arch = make_dot(y.mean(), params=dict(pytorch_model.named_parameters()))
arch.format="pdf"
arch.filename = "convnet_arch"
arch.render(view=False)
```

```python colab={"base_uri": "https://localhost:8080/"} id="oOs3P_CcAifi" executionInfo={"status": "ok", "timestamp": 1631266376874, "user_tz": -330, "elapsed": 432, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fcdff773-b9d4-443a-b484-40d6f823a9a3"
autoPyTorch.get_hyperparameter_search_space()
```

```python id="tv6JRIg7DYhS"

```
