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

```python colab={"base_uri": "https://localhost:8080/"} id="Dg8frDmMWhHA" executionInfo={"status": "ok", "timestamp": 1628161697158, "user_tz": -330, "elapsed": 6515, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ad78155f-85a2-4879-d469-736270db3b51"
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

```python colab={"base_uri": "https://localhost:8080/"} id="I0dUu8Y5AR_q" executionInfo={"status": "ok", "timestamp": 1628161966399, "user_tz": -330, "elapsed": 517, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="53506cff-ce8d-4a48-a9f4-474a85ef84a5"
import glob
files = sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/reco-tut-cris*.ipynb'))
files
```

```python id="RrndMHFeAu7a" executionInfo={"status": "ok", "timestamp": 1628162157181, "user_tz": -330, "elapsed": 2880, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
from shutil import copy
for fpath in files[1:]:
    copy(fpath, './notebooks')
```

```python id="ZvBJn1_kBzqB" executionInfo={"status": "ok", "timestamp": 1628162182416, "user_tz": -330, "elapsed": 706, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="08e6e145-f7b3-4e68-ff3d-db6f554a81cc" colab={"base_uri": "https://localhost:8080/"}
!git status
```

```python id="__1yG7T9B0Rp" executionInfo={"status": "ok", "timestamp": 1628162201500, "user_tz": -330, "elapsed": 1527, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d1860480-1add-48b9-d2ca-59fbb0f2bec6" colab={"base_uri": "https://localhost:8080/"}
!git add . && git commit -m 'commit' && git push origin main
```
