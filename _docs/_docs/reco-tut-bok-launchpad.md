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

```python id="4sv1j8lRtwKQ" executionInfo={"status": "ok", "timestamp": 1630069531557, "user_tz": -330, "elapsed": 851, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-bok"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python id="n8RiXVLstuoO" executionInfo={"status": "ok", "timestamp": 1630069533175, "user_tz": -330, "elapsed": 1630, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7185e8e7-aa13-4c54-e6ab-e1864aa79575" colab={"base_uri": "https://localhost:8080/"}
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

```python id="drkDUesKtuoT"
!git status
```

```python id="pWkwGc0ltuoT"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python id="KQoB06RmtuoU"
import glob
files = sorted(glob.glob(f'/content/drive/MyDrive/Colab Notebooks/{project_name}*.ipynb'))
files
```

```python id="h7le8ERJtuoV"
from shutil import copy
for fpath in files[1:]:
    copy(fpath, './notebooks')
```

```sh id="zWGLaL1-tuoV"
mkdir -p ./code/nbs
for file in ./notebooks/*.ipynb
do
    jupyter nbconvert --output-dir="./code/nbs" --to python "$file"
done
```

```python id="A07k37WmtuoV"
!sudo apt-get install -qq tree
```

```python id="Fc0TebaStuoW"
!tree -L 3 .
!tree --du -h
```

````python id="-enykZkUtuoW"
%%writefile README.md
# 

## Project structure
```
.
├── artifacts   
```
````

```python id="lkyg1LhVtuoW"
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

```python id="0-b3EHsItuoX"
!git checkout -b queued
!cp $project_path/notebooks/*.ipynb ./_notebooks
!git add . && git commit -m 'commit' && git push origin queued
```
