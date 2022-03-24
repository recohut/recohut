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

<!-- #region id="5kUqaQUfM0EG" -->
# Personalized Transfer of User Preferences for Cross-domain Recommendation
<!-- #endregion -->

<!-- #region id="FCHdanvhJghm" -->
## API Run
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="fGZFDlfoIylZ" executionInfo={"status": "ok", "timestamp": 1635431603514, "user_tz": -330, "elapsed": 1512, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d3db991b-c821-4bdd-88d3-d6ca2837fc9b"
!git clone https://github.com/easezyc/WSDM2022-PTUPCDR.git
%cd WSDM2022-PTUPCDR
```

```python colab={"base_uri": "https://localhost:8080/"} id="jCX-tdeaJoda" executionInfo={"status": "ok", "timestamp": 1635431610433, "user_tz": -330, "elapsed": 6923, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6c849dae-dca0-47fa-f8ff-664c3e5b7e00"
!apt-get -qq install tree
```

```python colab={"base_uri": "https://localhost:8080/"} id="DG3MVqstJqkR" executionInfo={"status": "ok", "timestamp": 1635431610434, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d6828ed0-34a8-4e12-fc10-980ab75c080d"
!tree .
```

```python colab={"base_uri": "https://localhost:8080/"} id="mH32qqugILOX" executionInfo={"status": "ok", "timestamp": 1635431803491, "user_tz": -330, "elapsed": 193062, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4514511e-7ac0-495f-a837-6cc9e5f76343"
!gdown --id 1B1eIpDpecongoygrXdFIrgy1qIAabpq7
!gdown --id 10j3J18W9RdmEtreOq9TyF0ndJGf-USN6
!gdown --id 1l-EKL0JVZ7vgIxNx3oMmC-vuxQcyHtCz
```

```python id="VH5kXRGLJufQ" executionInfo={"status": "ok", "timestamp": 1635431803492, "user_tz": -330, "elapsed": 33, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
!mv *.gz data/raw
```

```python colab={"base_uri": "https://localhost:8080/"} id="ikmTTFtMJ2MC" executionInfo={"status": "ok", "timestamp": 1635431803492, "user_tz": -330, "elapsed": 26, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="50d30cd0-af0f-4ae9-d904-6ec6b86def04"
!tree --du -h -C .
```

```python id="SEIFE1yVVMLf" executionInfo={"status": "ok", "timestamp": 1635431806770, "user_tz": -330, "elapsed": 452, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import warnings
warnings.filterwarnings('ignore')
```

```python colab={"base_uri": "https://localhost:8080/"} id="5qIU01dDJ61j" outputId="e109f80f-fef6-48be-a5ac-6ba708fdb6b7"
!python entry.py --process_data_mid 1 --process_data_ready 1
```

```python id="mAdjd312KHkr"
import os
project_name = "coldstart-recsys"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', branch)

if not os.path.exists(project_path):
    !cp -r /content/drive/MyDrive/git_credentials/. ~
    !mkdir "{project_path}"
    %cd "{project_path}"
    !git init
    !git remote add origin https://github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout -b "{branch}"
else:
    %cd "{project_path}"
```
