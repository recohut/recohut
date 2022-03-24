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

```python id="1uICbB4nDexm" executionInfo={"status": "ok", "timestamp": 1628498309425, "user_tz": -330, "elapsed": 730, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-mll"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python colab={"base_uri": "https://localhost:8080/"} id="EFnuEM16DqQd" executionInfo={"status": "ok", "timestamp": 1628498311297, "user_tz": -330, "elapsed": 1888, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="72ddc027-cf0f-4cbc-f83e-c0f31ee3630e"
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

```python id="7bkm0Tb0DqQq" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628498989203, "user_tz": -330, "elapsed": 537, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0e3f2aeb-52bc-46f8-d4c2-76587675c31f"
!git status
```

```python id="9nEA2fSADqQr" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628498995320, "user_tz": -330, "elapsed": 2456, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9e239cc3-4e8e-4b72-99af-a46be801f564"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

<!-- #region id="Bsl5CHbCD2zJ" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="2eSvllfxEA0U" executionInfo={"status": "ok", "timestamp": 1628498367349, "user_tz": -330, "elapsed": 503, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5ca18a79-ec23-4c70-df5b-f1ddb140e6ef"
!cd /content && wget http://files.grouplens.org/datasets/movielens/ml-latest-small.zip
```

```python colab={"base_uri": "https://localhost:8080/"} id="lwiuWX4BEQ6y" executionInfo={"status": "ok", "timestamp": 1628498391524, "user_tz": -330, "elapsed": 454, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="707734ab-c9fc-4994-e7a8-bfe1d3f37756"
!cd /content && unzip ml-latest-small.zip
```

```python id="ildYBgGgEWsD" executionInfo={"status": "ok", "timestamp": 1628498443264, "user_tz": -330, "elapsed": 672, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import glob
import pandas as pd
```

```python id="BgCJ9VCmEaMP" executionInfo={"status": "ok", "timestamp": 1628498424241, "user_tz": -330, "elapsed": 427, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
!mkdir ./data/bronze
```

```python colab={"base_uri": "https://localhost:8080/"} id="XcRhriTREdFz" executionInfo={"status": "ok", "timestamp": 1628498475292, "user_tz": -330, "elapsed": 460, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="18991c23-424f-410e-d3a3-9de1e7c9cd41"
files = sorted(glob.glob('/content/ml-latest-small/*.csv'))
files
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="WFgY4JhcErSn" executionInfo={"status": "ok", "timestamp": 1628498500930, "user_tz": -330, "elapsed": 23, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="51fabbcb-94bc-4b4e-ba80-b8f716039278"
df1 = pd.read_csv(files[0])
df1.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="uqUtw8FhExkd" executionInfo={"status": "ok", "timestamp": 1628498508787, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a8587e70-a8ba-4b52-fc74-151eb0c47b7c"
df1.info()
```

```python id="MxKBKIWqEzgh" executionInfo={"status": "ok", "timestamp": 1628498573022, "user_tz": -330, "elapsed": 447, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
df1.to_parquet('./data/bronze/links.parquet.gzip', compression='gzip')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="r2p0xvt7FGqX" executionInfo={"status": "ok", "timestamp": 1628498595728, "user_tz": -330, "elapsed": 492, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ef1c28aa-ce16-4ec1-9078-aa5d90a48b1d"
df2 = pd.read_csv(files[1])
df2.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="W8Pe4i_oFGqj" executionInfo={"status": "ok", "timestamp": 1628498602181, "user_tz": -330, "elapsed": 422, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4da94f6e-3136-463d-f18f-30581db2ddc0"
df2.info()
```

```python id="k6GCN2B0FGqk" executionInfo={"status": "ok", "timestamp": 1628498620534, "user_tz": -330, "elapsed": 418, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
df2.to_parquet('./data/bronze/movies.parquet.gzip', compression='gzip')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="qvb6xM1eFPbM" executionInfo={"status": "ok", "timestamp": 1628498631582, "user_tz": -330, "elapsed": 513, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d47ea304-ce2e-4fbf-ecb3-d70d2ad95ddf"
df3 = pd.read_csv(files[2])
df3.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="0dG-hOhmFPbN" executionInfo={"status": "ok", "timestamp": 1628498641119, "user_tz": -330, "elapsed": 422, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c885f252-eab6-44a7-8e9e-af65fe7505e5"
df3.info()
```

```python id="TWPkMLoIFPbO" executionInfo={"status": "ok", "timestamp": 1628498651765, "user_tz": -330, "elapsed": 708, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
df3.to_parquet('./data/bronze/ratings.parquet.gzip', compression='gzip')
```

```python id="-r8sap0zFDIw"

```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="8aKGgdMfFXL9" executionInfo={"status": "ok", "timestamp": 1628498670207, "user_tz": -330, "elapsed": 529, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="125a2592-ea27-4daa-b95e-2d44242e4092"
df4 = pd.read_csv(files[3])
df4.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="DrCG8hZEFXL-" executionInfo={"status": "ok", "timestamp": 1628498676344, "user_tz": -330, "elapsed": 523, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="722b1d5a-1220-4fe9-aeff-dd87f1d9f1a9"
df4.info()
```

```python id="fg5oqj3eFXL_" executionInfo={"status": "ok", "timestamp": 1628498688223, "user_tz": -330, "elapsed": 512, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
df4.to_parquet('./data/bronze/tags.parquet.gzip', compression='gzip')
```

```python id="JGu1vqJDGcmi" executionInfo={"status": "ok", "timestamp": 1628498969301, "user_tz": -330, "elapsed": 449, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
!mv /content/reco-tut-mll/data/bronze/small/* /content/reco-tut-mll/data/bronze
```
