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

```python id="9QJdP4lptz2J" executionInfo={"status": "ok", "timestamp": 1629197146889, "user_tz": -330, "elapsed": 24, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-bfr"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python colab={"base_uri": "https://localhost:8080/"} id="KKsYb6rUtwQI" executionInfo={"status": "ok", "timestamp": 1629197148733, "user_tz": -330, "elapsed": 1860, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d447ed66-c45b-488b-9a49-3997c509d0bc"
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

```python colab={"base_uri": "https://localhost:8080/"} id="EjMYuXIStwQM" executionInfo={"status": "ok", "timestamp": 1629197609825, "user_tz": -330, "elapsed": 445, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="dfdb6ba5-328d-4259-e4da-498f434d5a74"
!git status
```

```python colab={"base_uri": "https://localhost:8080/"} id="nBUkpOX6twQN" executionInfo={"status": "ok", "timestamp": 1629197616882, "user_tz": -330, "elapsed": 2339, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="eaff75da-8f32-442a-cc32-3f1ce0ff9b60"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python id="FG3IFXjdtwQO"
import glob
files = sorted(glob.glob(f'/content/drive/MyDrive/Colab Notebooks/{project_name}*.ipynb'))
files
```

```python id="PdPj5cH5twQP"
from shutil import copy
for fpath in files[1:]:
    copy(fpath, './notebooks')
```

```sh id="DZRJLnh0twQQ"
mkdir -p ./code/nbs
for file in ./notebooks/*.ipynb
do
    jupyter nbconvert --output-dir="./code/nbs" --to python "$file"
done
```

```python id="8InyOeXstwQQ"
!sudo apt-get install -qq tree
```

```python id="ywg9_WTStwQR"
!tree -L 3 .
```

````python id="tNzWOtNDtwQR"
%%writefile README.md
# 

## Project structure
```
.
├── artifacts   
```
````

```python id="mO5yZHiktwQT"
xproject_name = "reco-nb-stage"; xbranch = "queued"; xaccount = "recohut"
xproject_path = os.path.join('/content', xproject_name)

if not os.path.exists(xproject_path):
    !cp /content/drive/MyDrive/mykeys.py /content
    import mykeys
    !rm /content/mykeys.py
    path = "/content/" + xproject_name; 
    !mkdir "{path}"
    %cd "{path}"
    import sys; sys.path.append(path)
    !git config --global user.email "recotut@recohut.com"
    !git config --global user.name  "reco-tut"
    !git init
    !git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{xaccount}"/"{xproject_name}".git
    !git pull origin "{xbranch}"
else:
    %cd "{xproject_path}"
```

```python id="irCnbQ5jtwQU"
!git checkout -b queued
!cp $project_path/notebooks/*.ipynb ./_notebooks
!git add . && git commit -m 'commit' && git push origin queued
```

<!-- #region id="O3zVNVCJuBbX" -->
---
<!-- #endregion -->

<!-- #region id="VU9h-9Y3vVeQ" -->
Track 1 - AutoDebias
<!-- #endregion -->

```python id="jgn935xRVTO4"
# !cd /content && git clone https://github.com/DongHande/AutoDebias.git
# !mkdir ./data/bronze && mv /content/AutoDebias/datasets/* ./data/bronze
# !mv /content/AutoDebias/baselines/*.py ./code
# mkdir ./code/models && mv ./code/*.py ./code/models
# !mv /content/AutoDebias/utils ./code
```

<!-- #region id="fL5IzF-4uAgd" -->
Track 2 - DICE
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="g0SfCRDOuMvk" executionInfo={"status": "ok", "timestamp": 1629199205823, "user_tz": -330, "elapsed": 6782, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f3a77b7f-8e76-4733-b97d-adc4b9041f4d"
!cd /content && git clone https://github.com/tsinghua-fib-lab/DICE.git
```

```python colab={"base_uri": "https://localhost:8080/"} id="5nYFa1Zd1uf9" executionInfo={"status": "ok", "timestamp": 1629199235164, "user_tz": -330, "elapsed": 1235, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1f74ecb1-c5e1-4e09-9252-56eddffa315c"
!cd /content && unzip /content/DICE/data/ml10m.zip
```

<!-- #region id="LAq2ttqn13Cr" -->
Track 4 - MACR
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="AaHh5Wvm4Vik" executionInfo={"status": "ok", "timestamp": 1629199895358, "user_tz": -330, "elapsed": 3377, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="71f4fdf3-9dc5-412d-ef32-a5ad3e5a56cd"
!cd /content && git clone https://github.com/weitianxin/MACR.git
```

```python colab={"base_uri": "https://localhost:8080/"} id="VD-hqZiD4XtF" executionInfo={"status": "ok", "timestamp": 1629200034361, "user_tz": -330, "elapsed": 626, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="591f0d33-3884-4eec-eb0e-82d90ca3295a"
# !zip ./data/bronze/adressa.zip /content/MACR/data/addressa/*
# !zip ./data/bronze/globe.zip /content/MACR/data/globe/*
# !zip ./data/bronze/gowalla.zip /content/MACR/data/gowalla/*
# !zip ./data/bronze/ml10m.zip /content/MACR/data/ml_10m/*
# !zip ./data/bronze/yelp18.zip /content/MACR/data/yelp2018/*
```

```python id="UvQ5enN64pmY"

```
