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

```python id="2Kkh41eapDyt"
import os
project_name = "reco-tut-rec"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python id="gLlY3Kntpldf" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1630781939080, "user_tz": -330, "elapsed": 949, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f83a6b82-c6a6-416b-ae79-9875cda671d2"
if not os.path.exists(project_path):
    !cp -r /content/drive/MyDrive/git_credentials/. ~
    # !pip install -U -q dvc dvc[gdrive]
    path = "/content/" + project_name; 
    !mkdir "{path}"
    %cd "{path}"
    !git init
    !git remote add origin https://github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout main
else:
    %cd "{project_path}"
```

```python id="Y0k-sht3pldj" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1630784183047, "user_tz": -330, "elapsed": 439, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d12af4a1-321d-4c92-9cb8-8100708bf246"
!git status
```

```python id="zbD-LYR9pldj" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1630784186901, "user_tz": -330, "elapsed": 587, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d44aa7f4-d4cb-4634-d077-35ec165735e2"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python id="4b2eSxV5Tpts"
!git pull --rebase origin "{branch}"
```

<!-- #region id="RPKFyM_0r1o6" -->
---
<!-- #endregion -->

```python id="0kdoCsayQGpW"
!rm -r /content/reco-tut-rec/data/bronze/olx/df_subset_users.parquet.snappy
```

```python id="X5KAqqRbR--1"
!dvc init
!dvc remote 
```

```python colab={"base_uri": "https://localhost:8080/"} id="v-3fz8ivNDgf" executionInfo={"status": "ok", "timestamp": 1630784173812, "user_tz": -330, "elapsed": 45815, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="840a649e-043a-4937-dba4-eef4f362a1af"
!mkdir -p ./data/bronze/olx
!dvc import -o ./data/bronze/olx/df_subset_items.parquet.snappy https://github.com/sparsh-ai/reco-data olx/v1/df_subset_items.parquet.snappy
!dvc import -o ./data/bronze/olx/df_subset_users.parquet.snappy https://github.com/sparsh-ai/reco-data olx/v1/df_subset_users.parquet.snappy
```

```python colab={"base_uri": "https://localhost:8080/"} id="RXVM_SjfL1aa" executionInfo={"status": "ok", "timestamp": 1630783215189, "user_tz": -330, "elapsed": 36572, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="96303de8-c589-437c-e1b6-ae8d66a59f96"
!cd ./data/bronze/olx && dvc get https://github.com/sparsh-ai/reco-data olx/v1/df_subset_users.parquet.snappy
```

```python colab={"base_uri": "https://localhost:8080/"} id="ZeuAoC9MPM2C" executionInfo={"status": "ok", "timestamp": 1630782990647, "user_tz": -330, "elapsed": 521, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6d112c8e-75af-4faf-a555-b56d30a025bb"
!make
```

```python id="vi43j5CfPNwL"
!rm -r /content/reco-tut-rec/data/bronze/olx/*
```

```python colab={"base_uri": "https://localhost:8080/"} id="w_khmIZeQnGT" executionInfo={"status": "ok", "timestamp": 1630783587637, "user_tz": -330, "elapsed": 30411, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7baeeb1c-11bf-45b0-99d3-2777718d9f34"
!make olx
```

```python id="BpyH5piEQpWo"
!rm -r /content/reco-tut-rec/data/bronze/olx/df_subset_items.parquet.snappy.dGB6iEgb7KZVKeBvU7GDb7.tmp
```

```python id="QKzZcA96R0C5"

```
