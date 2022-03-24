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

```python id="xWTTsFsu3idp"
import os
project_name = "reco-tut-itr"; branch = "main"; account = "sparsh-ai"
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

```python id="2dWj-TPW392r" executionInfo={"status": "ok", "timestamp": 1628092520869, "user_tz": -330, "elapsed": 408, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import pandas as pd
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="1JFS2DOZ4FHu" executionInfo={"status": "ok", "timestamp": 1628092624622, "user_tz": -330, "elapsed": 478, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b0f0d1b5-cc27-4a65-b593-39e7ed8f0b5b"
ratings = pd.read_csv("./data/bronze/ratings.csv",encoding='ISO-8859-1')
ratings.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="P5Fc5q-N4epu" executionInfo={"status": "ok", "timestamp": 1628092638761, "user_tz": -330, "elapsed": 409, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5233c9eb-86fa-4a21-ba43-cb565cadbaf3"
ratings.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 297} id="lQY4xfMG4iEr" executionInfo={"status": "ok", "timestamp": 1628092647696, "user_tz": -330, "elapsed": 466, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f510f631-a62a-4696-e56f-3cd09ad69889"
ratings.describe()
```

```python id="NhKX4kuw4kHO" executionInfo={"status": "ok", "timestamp": 1628092719743, "user_tz": -330, "elapsed": 423, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
ratings.to_parquet('./data/bronze/ratings.parquet.gz', compression='gzip')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 394} id="MGKrtU5e47Q1" executionInfo={"status": "ok", "timestamp": 1628092752992, "user_tz": -330, "elapsed": 513, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="889d1afa-0206-42d3-c05d-5367e3977745"
items = pd.read_csv("./data/bronze/items.csv",encoding='ISO-8859-1')
items.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="K4OKTG3r47Q3" executionInfo={"status": "ok", "timestamp": 1628092762870, "user_tz": -330, "elapsed": 517, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4fb98139-9ea1-4b11-e045-a62d0cb119cb"
items.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 669} id="JxEKuIf047Q4" executionInfo={"status": "ok", "timestamp": 1628092782851, "user_tz": -330, "elapsed": 495, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="afa60210-9c28-48ff-a6f3-39a2bb1bd62d"
items.describe().T
```

```python id="NGnNvm_F47Q4" executionInfo={"status": "ok", "timestamp": 1628092792794, "user_tz": -330, "elapsed": 443, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
items.to_parquet('./data/bronze/items.parquet.gz', compression='gzip')
```

```python id="9SDv15JF5Uib" executionInfo={"status": "ok", "timestamp": 1628092939175, "user_tz": -330, "elapsed": 463, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
!mv ./data/*.gz ./data/bronze
```

```python colab={"base_uri": "https://localhost:8080/"} id="PVLm7SRL5YHb" executionInfo={"status": "ok", "timestamp": 1628092945683, "user_tz": -330, "elapsed": 613, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="835af143-8895-4507-986e-f794463083a3"
!git status
```

```python colab={"base_uri": "https://localhost:8080/"} id="Ea50VQKF5Yxi" executionInfo={"status": "ok", "timestamp": 1628092964116, "user_tz": -330, "elapsed": 1185, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ef3a212d-5ee7-4e92-b5a3-8a5d06beffe5"
!git add . && git commit -m 'commit' && git push origin main
```
