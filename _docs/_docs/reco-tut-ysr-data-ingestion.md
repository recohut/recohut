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

```python id="86MgMsi_GD70" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628264124272, "user_tz": -330, "elapsed": 2128, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d16f83f0-6f40-43df-cea5-ad0173c3a4d4"
import os
project_name = "reco-tut-ysr"; branch = "main"; account = "sparsh-ai"
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

```python id="zzFVExkIFzDe" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628264296374, "user_tz": -330, "elapsed": 465, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2345d832-fcfe-4b91-a1c3-a803d787b573"
!git status
```

```python id="pXWJ6RWXjvEx" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628264307358, "user_tz": -330, "elapsed": 6247, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f243f5d1-1bf3-4066-cc7c-33fb45acd5ad"
!git add . && git commit -m 'commit' && git push origin main
```

<!-- #region id="DqVtQ4T7Fz_l" -->
---
<!-- #endregion -->

```python id="XJhLU7p1l0rL" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628264167431, "user_tz": -330, "elapsed": 31264, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9566d06e-8fa3-43d5-a35e-f23398647152"
!cd /content && git clone https://github.com/fafilia/dss_song2vec_recsys.git
```

```python id="QiPQimwZGtqJ" executionInfo={"status": "ok", "timestamp": 1628264279234, "user_tz": -330, "elapsed": 430, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
!mkdir ./data/bronze && cp /content/dss_song2vec_recsys/dataset/yes_complete/* ./data/bronze
```
