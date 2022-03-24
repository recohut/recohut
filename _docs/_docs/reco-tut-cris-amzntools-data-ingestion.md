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

```python colab={"base_uri": "https://localhost:8080/"} id="Dg8frDmMWhHA" executionInfo={"status": "ok", "timestamp": 1628016670043, "user_tz": -330, "elapsed": 4677, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b9e28437-0c4d-4575-c58a-ebda60c65c2f"
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

```python id="vOtRu2rBfJ_Z" executionInfo={"status": "ok", "timestamp": 1628018961364, "user_tz": -330, "elapsed": 1225, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
%%writefile ./data/bronze/load.sh
wget -O amazon_tools.json.gz http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Tools_and_Home_Improvement_5.json.gz
gunzip amazon_tools.json.gz
```

```python colab={"base_uri": "https://localhost:8080/"} id="K3pBgYbEg-e0" executionInfo={"status": "ok", "timestamp": 1628019404476, "user_tz": -330, "elapsed": 20270, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="242afcfe-5f98-409c-cfbd-ece6ecaeac2b"
!cd ./data/bronze && sh load.sh
```

```python colab={"base_uri": "https://localhost:8080/"} id="kasG57llg4lC" executionInfo={"status": "ok", "timestamp": 1628019461938, "user_tz": -330, "elapsed": 534, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6b1361a7-bc07-4f16-ac10-fe73c5310229"
!ls ./data/bronze
```

```python id="n2x9gqVBgbih" executionInfo={"status": "ok", "timestamp": 1628019267765, "user_tz": -330, "elapsed": 626, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
!echo 'data/bronze/*.gz' >> .gitignore
!echo 'data/bronze/*.json' >> .gitignore
!echo 'data/bronze/*.csv' >> .gitignore
```

```python colab={"base_uri": "https://localhost:8080/"} id="Pe-5Iol6cd5N" executionInfo={"status": "ok", "timestamp": 1628019469662, "user_tz": -330, "elapsed": 548, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fffdf37b-6d76-4344-f295-e7263498efc2"
!git status
```

```python colab={"base_uri": "https://localhost:8080/"} id="Ua-Nm_h0cesF" executionInfo={"status": "ok", "timestamp": 1628019609993, "user_tz": -330, "elapsed": 1340, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ec4efc6e-c88e-429b-983a-bd22676b6960"
!git add . && git commit -m 'commit' && git push origin main
```
