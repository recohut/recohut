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

```python id="jVkAVV4pixpb"
import os
project_name = "reco-tut-mal"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python colab={"base_uri": "https://localhost:8080/"} id="RDSfrKdHi4C8" executionInfo={"status": "ok", "timestamp": 1629177543489, "user_tz": -330, "elapsed": 2679, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e986f92b-d1fd-442e-863a-6fc9fecfa5bd"
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

```python id="ien61F96uHVw"
!git pull --rebase origin main
```

```python id="22P-ZOjbi4C_" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629180671751, "user_tz": -330, "elapsed": 839, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="eb27568f-b53f-4924-e81e-b31c4120f4ad"
!git status
```

```python id="9LDKaBYRi4DA" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629180714907, "user_tz": -330, "elapsed": 2223, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b7ff8097-434e-475c-8495-c5b7a6fc7db8"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python id="5D5IPzCFi4DB"
import glob
files = sorted(glob.glob(f'/content/drive/MyDrive/Colab Notebooks/{project_name}*.ipynb'))
files
```

```python id="5QAPq817i4DB"
from shutil import copy
for fpath in files[1:]:
    copy(fpath, './notebooks')
```

```sh id="lY4UqclTi4DC"
mkdir -p ./code/nbs
for file in ./notebooks/*.ipynb
do
    jupyter nbconvert --output-dir="./code/nbs" --to python "$file"
done
```

```python id="FYUyAG9Ri4DC"
!sudo apt-get install -qq tree
```

```python id="HW2_XhQmi4DD"
!tree -L 3 .
```

````python id="Vthg49ONi4DE"
%%writefile README.md
# 

## Project structure
```
.
├── artifacts   
```
````

```python id="D9nR9-Dai4DF"
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

```python id="IduFiV19i4DG"
!git checkout -b queued
!cp $project_path/notebooks/*.ipynb ./_notebooks
!git add . && git commit -m 'commit' && git push origin queued
```
