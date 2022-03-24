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

```python id="YAl5xP7kFxNg" executionInfo={"status": "ok", "timestamp": 1628315419856, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-mcp"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python id="86MgMsi_GD70" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628315422847, "user_tz": -330, "elapsed": 3000, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d4c36590-9fe2-4ca8-a5c8-69b34e49ef47"
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

```python id="zzFVExkIFzDe" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628314299306, "user_tz": -330, "elapsed": 696, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="93badc2c-c209-406b-aa97-6a95bed28503"
!git status
```

```python id="pXWJ6RWXjvEx" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628314303224, "user_tz": -330, "elapsed": 1177, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="08952117-15f8-4608-e708-e796daeb3eef"
!git add . && git commit -m 'commit' && git push origin main
```

<!-- #region id="DqVtQ4T7Fz_l" -->
---
<!-- #endregion -->

```python id="bJBfoPZRLXrt" executionInfo={"status": "ok", "timestamp": 1628315703598, "user_tz": -330, "elapsed": 556, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
%reload_ext autoreload
%autoreload 2
```

```python id="gaONFtJSKY_4" executionInfo={"status": "ok", "timestamp": 1628315446971, "user_tz": -330, "elapsed": 567, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import sys
sys.path.insert(0, './code')
```

```python id="LXwazh2YKelq" executionInfo={"status": "ok", "timestamp": 1628315770690, "user_tz": -330, "elapsed": 566, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
from utils import load_data
load_data()
```

```python id="ZnmsTonSKiOd" executionInfo={"status": "ok", "timestamp": 1628315827534, "user_tz": -330, "elapsed": 3, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import pandas as pd
```

<!-- #region id="7qVsvNlcLvj-" -->
## Users
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 253} id="DvcrO1B4L9Pn" executionInfo={"status": "ok", "timestamp": 1628315881050, "user_tz": -330, "elapsed": 2511, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0504c3e3-31d9-4e70-e58f-6964a8566eed"
users = pd.read_pickle('./data/bronze/users.pickle.gzip', compression='gzip')
users.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="bdaP2WwoMFm0" executionInfo={"status": "ok", "timestamp": 1628315888307, "user_tz": -330, "elapsed": 537, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6960f027-a7d7-4e09-ee5a-0bcad6850199"
users.shape
```

<!-- #region id="ouhiqknkMKTx" -->
It is a dataset of `974 960` fully anonymized Deezer users. Each user is described by a *96-dimensional* embedding vector (fields dim_0 to dim_95), and a bias term, summarizing the user's musical preferences.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="Fw1U6DS8ayOS" executionInfo={"status": "ok", "timestamp": 1628319753838, "user_tz": -330, "elapsed": 19773, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="64b98106-6540-420d-e04a-5e8068c60196"
users.sample(10000).hist(figsize=(30, 40));
```

```python colab={"base_uri": "https://localhost:8080/"} id="YWs5DfivMeBu" executionInfo={"status": "ok", "timestamp": 1628316646683, "user_tz": -330, "elapsed": 1645, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a487debf-4e53-40f0-8cff-b75dbad9d6d6"
users.segment.astype('str').describe()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 277} id="9HOKskIdO-YK" executionInfo={"status": "ok", "timestamp": 1628316719595, "user_tz": -330, "elapsed": 756, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a622f191-5938-45e3-a2b1-9834ebfaff25"
users.segment.value_counts()[:10].plot(kind='bar');
```

<!-- #region id="G0AGMl_gPKYA" -->
Segment column represent the user-segment. Each user belongs to one out of 100 cluster/segment. The reason of doing this clustering is because instead of personalization at user-level, we will personalize at segment-level (with an assumption that users within the segment have similar taste of music). A k-means clustering with Q = 100 clusters was performed to assign each user to a single cluster. Here is the snip from the paper discussing it in more detail:
<!-- #endregion -->

<!-- #region id="nVAkY12aQdem" -->
<!-- #endregion -->

<!-- #region id="xCfvPJNWPqys" -->
## Playlists
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 253} id="qligNAb3YC98" executionInfo={"status": "ok", "timestamp": 1628319019516, "user_tz": -330, "elapsed": 844, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d18c8ce2-9895-480c-e72f-037c2a8de823"
playlists = pd.read_pickle('./data/bronze/playlists.pickle.gzip', compression='gzip')
playlists.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="0xvflGuQZW66" executionInfo={"status": "ok", "timestamp": 1628319657104, "user_tz": -330, "elapsed": 19431, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="94b94481-ea5b-4702-c532-690c9015b12f"
playlists.hist(figsize=(30, 40));
```

```python colab={"base_uri": "https://localhost:8080/"} id="Kihfb7IfYGrD" executionInfo={"status": "ok", "timestamp": 1628319036797, "user_tz": -330, "elapsed": 836, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5b037793-5aa6-43a9-a741-3bc7ea0e8da4"
playlists.shape
```

<!-- #region id="5LjVsviLYK4H" -->
It is a dataset of 862 playlists. Each playlist i is described by:
a 97-dimensional weight vector. For each user-playlist pair (u,i), the "ground-truth" display-to-stream probability is there. 97-dimensional x_u vector corresponds to the concatenation of the 96-dim embedding vector of user u and of the bias term, and where sigma denotes the sigmoid activation function. Here is the snip from the paper discussing it in more detail:
<!-- #endregion -->

<!-- #region id="-lOSM3JFY9FG" -->
<!-- #endregion -->
