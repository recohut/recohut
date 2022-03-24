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

```python id="Jep4VQyz3ZzE" executionInfo={"status": "ok", "timestamp": 1625767349784, "user_tz": -330, "elapsed": 490, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
project_name="reco-wikirecs"; branch="master"; account="sparsh-ai"
```

```python id="ailHP5gi3ZzP"
!cp /content/drive/MyDrive/mykeys.py /content
import mykeys
!rm /content/mykeys.py
path = "/content/" + project_name; 
!mkdir "{path}"
%cd "{path}"
import sys; sys.path.append(path)
!git config --global user.email "sparsh@recohut.com"
!git config --global user.name  "colab-sparsh"
!git init
!git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git
!git pull origin "{branch}"
```

```python id="WWDDXhuK9klF"
%cd /content/reco-wikirecs/
```

```python id="6HVnZkVW3ZzQ"
!git status
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python id="LLMOakVK7lZg"
!pip install -r requirements.txt
```

<!-- #region id="wlWx6OrY3n_A" -->
---
<!-- #endregion -->

```python id="ZsdBdQU4SlPy" executionInfo={"status": "ok", "timestamp": 1625768153384, "user_tz": -330, "elapsed": 472, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
import yaml
import pandas as pd
from pyarrow import feather
```

```python id="wXZfzk8rTYX6" executionInfo={"status": "ok", "timestamp": 1625767830838, "user_tz": -330, "elapsed": 441, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
```

```python id="8k7gkgbdSYrF" executionInfo={"status": "ok", "timestamp": 1625768041365, "user_tz": -330, "elapsed": 45394, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
all_histories = []
for fname in os.listdir(config['file_save_path']):
    if 'feather' in  fname:
        all_histories.append(feather.read_feather(os.path.join(config['file_save_path'],fname)))
```

```python id="gMtXFbWLTpJT" executionInfo={"status": "ok", "timestamp": 1625768163665, "user_tz": -330, "elapsed": 1813, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
all_histories = pd.concat(all_histories, ignore_index=True)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="Zg-ZRYUFVW33" executionInfo={"status": "ok", "timestamp": 1625768191836, "user_tz": -330, "elapsed": 415, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="76249d15-9f6e-45ef-8863-41cb6eef25d8"
all_histories.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="hU_SbzmmVeG1" executionInfo={"status": "ok", "timestamp": 1625768201872, "user_tz": -330, "elapsed": 634, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1820caac-cb05-4c4a-a62d-064a88338d00"
all_histories.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="UYjNXm6bVy2o" executionInfo={"status": "ok", "timestamp": 1625768283454, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="17921817-c2e3-47b5-a75a-4ff45afe8ebc"
all_histories.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 297} id="Z6lerT-PVgXn" executionInfo={"status": "ok", "timestamp": 1625768227959, "user_tz": -330, "elapsed": 2973, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b2ba7815-305e-49b5-e527-8591fbcefe16"
all_histories.describe()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 173} id="mvU0BTqOVmXs" executionInfo={"status": "ok", "timestamp": 1625768275900, "user_tz": -330, "elapsed": 27586, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cea875e4-9b28-4a45-f68c-cf6f9fd15562"
all_histories.describe(include=['O'])
```

```python id="muyaq_HuVsBL" executionInfo={"status": "ok", "timestamp": 1625768383000, "user_tz": -330, "elapsed": 46065, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
all_histories.to_parquet('wikirecs.parquet.gzip', compression='gzip')
```

<!-- #region id="q_s-Fib4XF31" -->
> Note: Data is also registered with [recochef](https://github.com/sparsh-ai/recochef/blob/master/src/recochef/datasets/wikirecs.py) for easy access
<!-- #endregion -->
