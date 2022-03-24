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

```python id="86MgMsi_GD70" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628247309149, "user_tz": -330, "elapsed": 2053, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e2daa42c-914b-46e6-c03f-c17b56c751f2"
import os
project_name = "reco-tut-mcp"; branch = "main"; account = "sparsh-ai"
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

```python id="Y8Zw_6uTGD71" executionInfo={"status": "ok", "timestamp": 1628247358577, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import pandas as pd
```

```python colab={"base_uri": "https://localhost:8080/"} id="eUkYpgziGvhA" executionInfo={"status": "ok", "timestamp": 1628247408810, "user_tz": -330, "elapsed": 3109, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f4fed81f-ec91-4278-83b6-e4436931e2f9"
!wget -O /content/playlists.csv https://zenodo.org/record/4048678/files/playlist_features.csv?download=1
```

```python colab={"base_uri": "https://localhost:8080/", "height": 253} id="k3e9JL8PG6-P" executionInfo={"status": "ok", "timestamp": 1628247435919, "user_tz": -330, "elapsed": 597, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="30040136-a44d-4dc0-bfaf-2e5efa068c15"
df1 = pd.read_csv('/content/playlists.csv')
df1.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="ZuaqmfxuHCHA" executionInfo={"status": "ok", "timestamp": 1628247462751, "user_tz": -330, "elapsed": 521, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="756e39d6-eef5-4018-b726-310196d9cc40"
df1.T.head().T.info()
```

```python colab={"base_uri": "https://localhost:8080/"} id="-qBMnhmQHI1g" executionInfo={"status": "ok", "timestamp": 1628247529021, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a62c752f-a4c9-4a9c-b24c-69de940a0305"
df1 = df1.astype('float16')
df1.T.head().T.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 253} id="RfA5mvoeHxuy" executionInfo={"status": "ok", "timestamp": 1628247633547, "user_tz": -330, "elapsed": 526, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a89d52d4-fcc7-460a-c2b8-be34f9ffdd66"
df1.head()
```

```python id="orm-LeUNHZGo" executionInfo={"status": "ok", "timestamp": 1628247604335, "user_tz": -330, "elapsed": 780, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
df1.to_pickle('/content/playlists.pickle.gzip', compression='gzip')
```

```python colab={"base_uri": "https://localhost:8080/"} id="1FRT_6OfIImF" executionInfo={"status": "ok", "timestamp": 1628247848985, "user_tz": -330, "elapsed": 102151, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="56538dbc-198f-4899-f240-91a5eb2b3d08"
!wget -O /content/users.csv https://zenodo.org/record/4048678/files/user_features.csv?download=1
```

```python colab={"base_uri": "https://localhost:8080/", "height": 253} id="_ia9b2YlIImZ" executionInfo={"status": "ok", "timestamp": 1628249526586, "user_tz": -330, "elapsed": 24278, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1645ba74-d4cc-4539-d58d-934b6bb42332"
df2 = pd.read_csv('/content/users.csv')
df2.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="75EtaVBOIIma" executionInfo={"status": "ok", "timestamp": 1628249526587, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1540a7cb-3dbd-46d0-eb78-b422dba8d11f"
df2.T.head().T.info()
```

```python colab={"base_uri": "https://localhost:8080/"} id="2QQQy5mvIIme" executionInfo={"status": "ok", "timestamp": 1628247889324, "user_tz": -330, "elapsed": 772, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7cb01abf-5770-4365-95a6-21c73fb0cce3"
df2 = df2.astype('float16')
df2.T.head().T.info()
```

```python id="gNtdLYciIImg" executionInfo={"status": "ok", "timestamp": 1628247942548, "user_tz": -330, "elapsed": 12277, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
df2.to_pickle('/content/users.pickle.gzip', compression='gzip')
```

```python id="kVs03ofVJJul" executionInfo={"status": "ok", "timestamp": 1628248040205, "user_tz": -330, "elapsed": 518, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
!mkdir ./data/bronze && mv /content/*.pickle.gzip ./data/bronze
```

```python id="h5XxhhxqLJG4" executionInfo={"status": "ok", "timestamp": 1628249721469, "user_tz": -330, "elapsed": 653, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
!mv "/content/reco-tut-mcp/data/bronze/users.pickle.gzip" "/content/drive/MyDrive/shareddata"
```

```python colab={"base_uri": "https://localhost:8080/"} id="BzVqyGonQ3Gx" executionInfo={"status": "ok", "timestamp": 1628250015276, "user_tz": -330, "elapsed": 470, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0ff319f9-8572-45e0-a18c-4b659004e62b"
!gdown --help
```

```python colab={"base_uri": "https://localhost:8080/"} id="oeCDo3DxRHbi" executionInfo={"status": "ok", "timestamp": 1628251519057, "user_tz": -330, "elapsed": 526, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="63115e2f-9636-445e-fcb6-557561d57af2"
%%writefile ./code/load_data.py
import os
import subprocess  
from pathlib import Path

filepath = os.path.join(Path(__file__).parent.parent, 'data/bronze/users.pickle.gzip')

cmd = 'gdown'
temp = subprocess.Popen([cmd,'--id','1-9PSkNwQZAlfIHWq4aUqV1NHO3JUyR_X','-O',filepath],
                        stdout = subprocess.PIPE) 
output = str(temp.communicate()) 
print(output)
```

```python colab={"base_uri": "https://localhost:8080/"} id="guVkKBZ-RIgM" executionInfo={"status": "ok", "timestamp": 1628251560831, "user_tz": -330, "elapsed": 529, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="98f72b46-8d52-454b-c17a-564f5d7d09be"
!git status
```

```python id="nFdmtr3fVT7_" executionInfo={"status": "ok", "timestamp": 1628251450290, "user_tz": -330, "elapsed": 991, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
!echo ./data/bronze/users.pickle.gzip >> .gitignore
```

```python colab={"base_uri": "https://localhost:8080/"} id="0CYxoodOWFkh" executionInfo={"status": "ok", "timestamp": 1628251384304, "user_tz": -330, "elapsed": 19, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e2ef3a78-f950-433f-e1c4-14b9b5894a25"
!git status
```

```python id="KeyKdcQ5UVw1" executionInfo={"status": "ok", "timestamp": 1628251575929, "user_tz": -330, "elapsed": 437, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
!git add .
```

```python colab={"base_uri": "https://localhost:8080/"} id="mOYnrKRXUhdz" executionInfo={"status": "ok", "timestamp": 1628251581020, "user_tz": -330, "elapsed": 1019, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b64d4cb6-4fa8-4c5d-c839-8d65af6bd794"
!git commit -m 'commit'
```

```python colab={"base_uri": "https://localhost:8080/"} id="Ysw-JMveUlvX" executionInfo={"status": "ok", "timestamp": 1628251591871, "user_tz": -330, "elapsed": 1645, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cdf74eeb-41eb-4a83-d0ff-e702cbc169eb"
!git push origin main
```
