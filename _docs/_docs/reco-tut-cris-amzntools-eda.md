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

```python id="Dg8frDmMWhHA" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628060822437, "user_tz": -330, "elapsed": 520, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="30bba15d-e18e-4aab-c651-f62549470cbc"
import os
project_name = "reco-tut-cris"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)

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

```python id="0wm-JFXm03Jg" executionInfo={"status": "ok", "timestamp": 1628060833674, "user_tz": -330, "elapsed": 1032, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%reload_ext google.colab.data_table
```

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="DwXrbaOS5X_6" executionInfo={"status": "ok", "timestamp": 1628060834691, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="88c4cc65-2c9a-498a-c6ee-ffe6fc7f9d76"
train = pd.read_csv('./data/silver/amazon_tools/train.csv', header=None,
                    names=['USERID','ITEMID','RATING','TIMESTAMP'])
train
```

```python colab={"base_uri": "https://localhost:8080/", "height": 194} id="udbDviPp6OGw" executionInfo={"status": "ok", "timestamp": 1628060834692, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="082ce172-58dc-4fd1-ba8a-22c1dfb34ba8"
train['TIMESTAMP'] = pd.to_datetime(train['TIMESTAMP'], unit='s')
train.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="0PCb6xrh5fu3" executionInfo={"status": "ok", "timestamp": 1628060834693, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9d037f88-efa4-4841-e8c2-f4ddba9a6961"
train.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 189} id="7XHG_I1b56Px" executionInfo={"status": "ok", "timestamp": 1628060835321, "user_tz": -330, "elapsed": 637, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2ea92802-16a4-4c0a-a117-415427ec2d0a"
train.describe(include='all', datetime_is_numeric=True).T
```

```python colab={"base_uri": "https://localhost:8080/", "height": 279} id="vSAxJeVC59dY" executionInfo={"status": "ok", "timestamp": 1628060836217, "user_tz": -330, "elapsed": 903, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a99ebc81-5c2f-4f83-fd9f-68590b2f7c07"
sns.kdeplot(train['RATING']);
```

```python colab={"base_uri": "https://localhost:8080/", "height": 334} id="Beldhn4O7JvA" executionInfo={"status": "ok", "timestamp": 1628060840645, "user_tz": -330, "elapsed": 1256, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="67ddd03e-ff24-4f4c-96f7-2c701a1751c7"
fig, ax = plt.subplots(1, 2, figsize=(18, 5))
train.groupby(train["TIMESTAMP"].dt.year)["RATING"].count().plot(kind='bar', rot=0, ax=ax[0])
train.groupby(train["TIMESTAMP"].dt.month)["RATING"].count().plot(kind='bar', rot=0, ax=ax[1])
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 388} id="RBxBxGl18w2O" executionInfo={"status": "ok", "timestamp": 1628060878598, "user_tz": -330, "elapsed": 2432, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c2acb699-565b-4932-d003-a8b8abc57d97"
fig, ax = plt.subplots(figsize=(18, 5))
train.USERID.value_counts()[:50].plot(kind='bar', ax=ax, rot=45)
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 367} id="YOGtLEyC90Fw" executionInfo={"status": "ok", "timestamp": 1628060891332, "user_tz": -330, "elapsed": 1416, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="98053fa9-a76b-4b4e-9332-8a99ef2cabfa"
fig, ax = plt.subplots(figsize=(18, 5))
train.ITEMID.value_counts()[:50].plot(kind='bar', ax=ax, rot=45)
plt.show()
```
