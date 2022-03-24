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

```python id="UixHtPjEyLlP" executionInfo={"status": "ok", "timestamp": 1629013780860, "user_tz": -330, "elapsed": 2628, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-mba"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Fl3T7IcJyVm6" executionInfo={"status": "ok", "timestamp": 1629013782717, "user_tz": -330, "elapsed": 1899, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ba99d204-2955-4af1-fe19-31a58dd13a64"
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

```python id="rqbbjIKkyVnE"
!git status
```

```python colab={"base_uri": "https://localhost:8080/"} id="5VrCY8YzyVnG" executionInfo={"status": "ok", "timestamp": 1629014314060, "user_tz": -330, "elapsed": 1567, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5381b72a-8ea4-418e-b5c9-afc3f8991706"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python id="mlDqwpFGyVnI"
import glob
files = sorted(glob.glob(f'/content/drive/MyDrive/Colab Notebooks/{project_name}*.ipynb'))
files
```

```python id="kjc7KYcSyVnJ"
from shutil import copy
for fpath in files[1:]:
    copy(fpath, './notebooks')
```

```sh id="znvfJ7edyVnK"
mkdir -p ./code/nbs
for file in ./notebooks/*.ipynb
do
    jupyter nbconvert --output-dir="./code/nbs" --to python "$file"
done
```

```python id="58FMG3CyyVnK"
!sudo apt-get install -qq tree
```

```python id="uxSgUyMoyVnL"
!tree -L 3 .
```

````python id="I6IjqnB2yVnM"
%%writefile README.md
# 

## Project structure
```
.
├── artifacts   
```
````

```python id="ERKcNjTzyVnO"
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

```python id="Ptko7KJMyVnP"
!git checkout -b queued
!cp $project_path/notebooks/*.ipynb ./_notebooks
!git add . && git commit -m 'commit' && git push origin queued
```

<!-- #region id="Z_M04HmLyeA3" -->
## Data Ingestions
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="5mHpaMVQyflV" executionInfo={"status": "ok", "timestamp": 1629013862671, "user_tz": -330, "elapsed": 693, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="75e5dd93-db78-46ed-ce92-aa45459cc71e"
!wget -O ./data/telco.csv -q --show-progress "https://raw.githubusercontent.com/nchelaru/data-prep/master/telco_cleaned_yes_no.csv"
```
