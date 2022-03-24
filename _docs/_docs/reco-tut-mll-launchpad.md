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

```python id="1uICbB4nDexm" executionInfo={"status": "ok", "timestamp": 1628498218089, "user_tz": -330, "elapsed": 766, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-mll"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python colab={"base_uri": "https://localhost:8080/"} id="EFnuEM16DqQd" executionInfo={"status": "ok", "timestamp": 1628498221375, "user_tz": -330, "elapsed": 2476, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="60a7611b-35df-478f-e753-fe7f891422b1"
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

```python id="7bkm0Tb0DqQq"
!git status
```

```python id="9nEA2fSADqQr"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python id="j5stdXNwDqQr"
import glob
files = sorted(glob.glob(f'/content/drive/MyDrive/Colab Notebooks/{project_name}*.ipynb'))
files
```

```python id="PAaobbVrDqQt"
from shutil import copy
for fpath in files[1:]:
    copy(fpath, './notebooks')
```
