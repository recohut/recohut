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

```python colab={"base_uri": "https://localhost:8080/"} id="F5rLoSxIocB-" executionInfo={"status": "ok", "timestamp": 1628189605155, "user_tz": -330, "elapsed": 2291, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1534c24c-ed1b-445c-d69f-6e6631f4fd7a"
import os
project_name = "reco-tut-sjr"; branch = "main"; account = "sparsh-ai"
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

```python id="z4Lt7QJdqcUc" executionInfo={"status": "ok", "timestamp": 1628189636503, "user_tz": -330, "elapsed": 436, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import glob
import pandas as pd
```

```python colab={"base_uri": "https://localhost:8080/"} id="B77UA_hbqf60" executionInfo={"status": "ok", "timestamp": 1628189650824, "user_tz": -330, "elapsed": 429, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="73b5f6c8-4730-49ea-abb1-a839754cf749"
files = glob.glob('./data/bronze/*.csv')
files
```

```python colab={"base_uri": "https://localhost:8080/", "height": 479} id="3yk01JJVqmZe" executionInfo={"status": "ok", "timestamp": 1628189671810, "user_tz": -330, "elapsed": 1064, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6c58d7d1-6c1d-4f79-e087-c11008dc4675"
df1 = pd.read_csv(files[0])
df1.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="CDtHtVOxqrTh" executionInfo={"status": "ok", "timestamp": 1628189686228, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f22890b8-3292-4a8f-92dc-6aa45e6c22c9"
df1.info()
```

```python id="vjFxZiG7qu5p" executionInfo={"status": "ok", "timestamp": 1628189770287, "user_tz": -330, "elapsed": 443, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
df1.to_parquet('./data/bronze/experience.parquet.gz', compression='gzip')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 598} id="k1EgFILcrGK6" executionInfo={"status": "ok", "timestamp": 1628189796865, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="38a1b663-426f-4559-dac9-c5069806f3bc"
df2 = pd.read_csv(files[1])
df2.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="8K2e01MkrGLH" executionInfo={"status": "ok", "timestamp": 1628189815964, "user_tz": -330, "elapsed": 522, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="191ee353-2661-4938-f4f2-a31b5cdf1bdf"
df2.info()
```

```python id="mqpab4KqrGLL" executionInfo={"status": "ok", "timestamp": 1628189833811, "user_tz": -330, "elapsed": 624, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
df2.to_parquet('./data/bronze/job_views.parquet.gz', compression='gzip')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="GutK2kMIrG2r" executionInfo={"status": "ok", "timestamp": 1628189839443, "user_tz": -330, "elapsed": 633, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="659ed457-1465-4894-b8cc-cb5a22db5fec"
df3 = pd.read_csv(files[2])
df3.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="g49Ol7KfrG2t" executionInfo={"status": "ok", "timestamp": 1628189857296, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9c7ca8c9-c0db-4718-9f79-20543c1f7b50"
df3.info()
```

```python id="C5ADhH0TrG2v" executionInfo={"status": "ok", "timestamp": 1628189870980, "user_tz": -330, "elapsed": 767, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
df3.to_parquet('./data/bronze/poi.parquet.gz', compression='gzip')
```

```python colab={"base_uri": "https://localhost:8080/"} id="h_-HjjjHrDyj" executionInfo={"status": "ok", "timestamp": 1628189935099, "user_tz": -330, "elapsed": 626, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="565c3bc2-59c7-4493-d312-ece2ac2b1a8f"
!cd ./data/bronze && zip -m raw_csv.zip ./*.csv
```

```python id="8S4rEww3rr_Y"
!git status
```

```python id="Luo9OP5VryCS"
!git add . && git commit -m 'ADD notebooks' && git push origin main
```
