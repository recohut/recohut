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

```python id="OvSMGB9a8onG" executionInfo={"status": "ok", "timestamp": 1628949435488, "user_tz": -330, "elapsed": 682, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-itr"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python colab={"base_uri": "https://localhost:8080/"} id="CAhfBMPR86sn" executionInfo={"status": "ok", "timestamp": 1628949437860, "user_tz": -330, "elapsed": 1113, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b5d1a18b-8b8a-40dd-ce9a-7c548d7bbe89"
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

```python colab={"base_uri": "https://localhost:8080/"} id="hfeS7Z-g86ss" executionInfo={"status": "ok", "timestamp": 1628949514178, "user_tz": -330, "elapsed": 840, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="211a3dee-409c-484f-f41d-661b88a1ac74"
!git status
```

```python colab={"base_uri": "https://localhost:8080/"} id="_vOCLJeI86st" executionInfo={"status": "ok", "timestamp": 1628949519508, "user_tz": -330, "elapsed": 1319, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5f72875a-654b-4b52-f078-40d4f4b7555f"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python colab={"base_uri": "https://localhost:8080/"} id="gkHUri3Q86su" executionInfo={"status": "ok", "timestamp": 1628949439078, "user_tz": -330, "elapsed": 488, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b71066a6-cf27-413d-8b20-62bee3d0c722"
import glob
files = sorted(glob.glob(f'/content/drive/MyDrive/Colab Notebooks/{project_name}*.ipynb'))
files
```

```python id="4OP1eS8P86sv" executionInfo={"status": "ok", "timestamp": 1628949447077, "user_tz": -330, "elapsed": 1191, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
from shutil import copy
for fpath in files[1:]:
    copy(fpath, './notebooks')
```

```sh colab={"base_uri": "https://localhost:8080/"} id="upK5jVN186sw" executionInfo={"status": "ok", "timestamp": 1628949467734, "user_tz": -330, "elapsed": 2964, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="98c5e1d9-ec9e-47a1-acfc-9f282bc77b32"
for file in ./notebooks/*.ipynb
do
    jupyter nbconvert --output-dir="./code/nbs" --to python "$file"
done
```

```python colab={"base_uri": "https://localhost:8080/"} id="xloR43Ti86sw" executionInfo={"status": "ok", "timestamp": 1628949473859, "user_tz": -330, "elapsed": 6132, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e6ffb794-67d7-4709-fd03-eadced24fd5b"
!sudo apt-get install -qq tree
```

```python colab={"base_uri": "https://localhost:8080/"} id="KpPv4ykl86sx" executionInfo={"status": "ok", "timestamp": 1628949473860, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e75fcb0d-17fa-49e8-ee82-6eb62bc41be5"
!tree -L 3 .
```

````python colab={"base_uri": "https://localhost:8080/"} id="ADfMoIJN86sy" executionInfo={"status": "ok", "timestamp": 1628949508966, "user_tz": -330, "elapsed": 656, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="af408341-e98c-492c-a701-dba9f762afcd"
%%writefile README.md
# Indian Travel Recommender

## Project structure
```
.
├── code
│   ├── __init__.py
│   └── nbs
│       ├── reco-tut-itr-01-data-ingestion.py
│       ├── reco-tut-itr-02-eda.py
│       └── reco-tut-itr-03-modeling-collaborative-knn.py
├── data
│   ├── bronze
│   │   ├── items.parquet.gz
│   │   └── ratings.parquet.gz
│   └── silver
│       ├── items.parquet.gz
│       └── rating.parquet.gz
├── docs
├── LICENSE
├── model
├── notebooks
│   ├── reco-tut-itr-01-data-ingestion.ipynb
│   ├── reco-tut-itr-02-eda.ipynb
│   └── reco-tut-itr-03-modeling-collaborative-knn.ipynb
└── README.md
```
````

```python colab={"base_uri": "https://localhost:8080/"} id="jUzir-G286s0" executionInfo={"status": "ok", "timestamp": 1628949534271, "user_tz": -330, "elapsed": 9072, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3993c6d0-e3bf-4053-db99-82a134f3a1e5"
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

```python colab={"base_uri": "https://localhost:8080/"} id="SM8JRgWp86s1" executionInfo={"status": "ok", "timestamp": 1628949538344, "user_tz": -330, "elapsed": 4082, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a8ccaaa1-26fa-41a1-a1ce-fe84cf66f3aa"
!git checkout -b queued
!cp $project_path/notebooks/*.ipynb ./_notebooks
!git add . && git commit -m 'commit' && git push origin queued
```

```python id="gesxp4Kx9TCP"

```
