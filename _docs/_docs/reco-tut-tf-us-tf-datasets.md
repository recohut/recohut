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

```python id="snKVvcmG5hyA"
import os
project_name = "reco-tut-tf"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python id="uWlMPZHD56HS" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629955388481, "user_tz": -330, "elapsed": 2022, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f96d3a3b-9883-43ab-a93e-f753408078da"
if not os.path.exists(project_path):
    !cp /content/drive/MyDrive/mykeys.py /content
    import mykeys
    !rm /content/mykeys.py
    path = "/content/" + project_name; 
    !mkdir "{path}"
    %cd "{path}"
    import sys; sys.path.append(path)
    !git config --global user.email "recotut@recohut.com"
    !git config --global user.name  "reco-tut"
    !git init
    !git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout main
else:
    %cd "{project_path}"
```

```python id="A1pwh_0H56HT"
!git status
```

```python id="pO2Ubylb56HU"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

<!-- #region id="VYohuYT86W5h" -->
---
<!-- #endregion -->

<!-- #region id="wXCNxR40ZEzW" -->
An **input pipeline** is a sequence of data processing components that manipulate and apply data transformations. Pipelines are very common in machine learning and deep learning systems because these systems are data-rich. That is, they demand large volumes of data to perform. Input pipelines are the best way to transform large datasets because they break down processing into manageable components.

Each component of an input pipeline pulls in a large amount of data, processes it in some manner, and spits out the result. The next component pulls in the resultant data, processes it in another manner, and spits out its own output. The pipeline continues until all of its components have finished their work.
<!-- #endregion -->

```python id="zdOlU8O86yKR"
import tensorflow as tf
```

```python colab={"base_uri": "https://localhost:8080/"} id="FCay0zQh68QG" executionInfo={"status": "ok", "timestamp": 1629955543527, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="90c5e586-d935-49d7-b6f4-49ee920db1b6"
X = tf.range(5)
X
```

```python colab={"base_uri": "https://localhost:8080/"} id="qn83vDdW64NT" executionInfo={"status": "ok", "timestamp": 1629955548815, "user_tz": -330, "elapsed": 504, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="aae41c0c-9c93-411c-acb6-5bb71251feb7"
dataset = tf.data.Dataset.from_tensor_slices(X)
dataset
```

```python id="PgqkVSRG65rO"

```
