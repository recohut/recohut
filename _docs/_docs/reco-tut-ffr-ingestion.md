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

```python id="-p45xus_9Lxv" executionInfo={"status": "ok", "timestamp": 1627741555842, "user_tz": -330, "elapsed": 438, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
project_name = "reco-tut-ffr"; branch = "main"; account = "sparsh-ai"
```

```python id="D03Mx8Df9Lx1"
!cp /content/drive/MyDrive/mykeys.py /content
import mykeys
!rm /content/mykeys.py
path = "/content/" + project_name; 
!mkdir "{path}"
%cd "{path}"
import sys; sys.path.append(path)
!git config --global user.email "nb@recohut.com"
!git config --global user.name  "colab-sparsh"
!git init
!git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git
!git pull origin "{branch}"
!git checkout main
```

<!-- #region id="dfMsBJhGW6ld" -->
## Load dataset
<!-- #endregion -->

<!-- #region id="6ZCD5IvKXEwH" -->
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="nz2l-vNj91Ft" executionInfo={"status": "ok", "timestamp": 1627741709993, "user_tz": -330, "elapsed": 661, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="249e5e39-d2a8-4c55-c613-b405742c2a9e"
%cd /content
```

```python colab={"base_uri": "https://localhost:8080/"} id="vWLoLTTQXIG0" executionInfo={"status": "ok", "timestamp": 1627741721304, "user_tz": -330, "elapsed": 9650, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="53b97ecf-236d-4763-bf24-e019a7b26390"
!pip install -q -U kaggle
!pip install --upgrade --force-reinstall --no-deps kaggle
!mkdir ~/.kaggle
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c FacebookRecruiting
```

```python colab={"base_uri": "https://localhost:8080/"} id="fxu9z1aJXJ5Y" executionInfo={"status": "ok", "timestamp": 1627741727792, "user_tz": -330, "elapsed": 3410, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="254f2597-61b3-42f8-ad78-96ace12ed38d"
!unzip FacebookRecruiting.zip
```

```python id="LSBo_Qu49qOy" executionInfo={"status": "ok", "timestamp": 1627741734303, "user_tz": -330, "elapsed": 456, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import pandas as pd
```

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="TQwEgya098AB" executionInfo={"status": "ok", "timestamp": 1627741762818, "user_tz": -330, "elapsed": 2531, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="36b5f7ec-2ca3-4077-b973-bf26da834532"
train = pd.read_csv('train.csv')
train
```

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="L48kKhaJ-bUY" executionInfo={"status": "ok", "timestamp": 1627741872363, "user_tz": -330, "elapsed": 424, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6dce3356-8151-44fc-bdfa-fa9f6ea66ac4"
test = pd.read_csv('test.csv')
test
```

```python id="8854h7Gt-CPC" executionInfo={"status": "ok", "timestamp": 1627741936443, "user_tz": -330, "elapsed": 11584, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os

data_path_bronze = '/content/reco-tut-ffr/data/bronze'
if not os.path.exists(data_path_bronze):
    os.makedirs(data_path_bronze)

train.to_parquet(os.path.join(data_path_bronze, 'train.parquet.gzip'), compression='gzip', hea)
test.to_parquet(os.path.join(data_path_bronze, 'test.parquet.gzip'), compression='gzip')
```

```python colab={"base_uri": "https://localhost:8080/"} id="-bKfK82Y-w2X" executionInfo={"status": "ok", "timestamp": 1627741970453, "user_tz": -330, "elapsed": 581, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c4c0d6f6-73f6-4b32-f3f6-e4ced824c144"
!cd reco-tut-ffr && git status
```

```python id="GhWDvHQUBeBX"
train.to_csv('x.csv', )
```

```python colab={"base_uri": "https://localhost:8080/"} id="srEDp542-1sP" executionInfo={"status": "ok", "timestamp": 1627742030031, "user_tz": -330, "elapsed": 7475, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="049be677-1ad9-4ce2-c968-4b224cddd6e0"
!cd reco-tut-ffr && git add . && git commit -m 'commit' && git push origin "{branch}"
```
