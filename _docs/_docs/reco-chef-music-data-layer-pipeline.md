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

```python id="rxL9QQCZPx8Q"
import os
project_name = "reco-chef"; branch = "30music"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python id="dwLTtRQK88Yz" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1630822646650, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="da36eafa-aed6-41e5-a178-4617a74ef004"
if not os.path.exists(project_path):
    !pip install -U -q dvc dvc[gdrive]
    !cp -r /content/drive/MyDrive/git_credentials/. ~
    path = "/content/" + project_name; 
    !mkdir "{path}"
    %cd "{path}"
    !git init
    !git remote add origin https://github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout "{branch}"
else:
    %cd "{project_path}"
```

```python id="2jrtc9Bg88Y0" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1630823389128, "user_tz": -330, "elapsed": 621, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="99229c81-5324-495d-a783-d6b00b10fc83"
!git status
```

```python id="2G4iErkK88Y1" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1630823400489, "user_tz": -330, "elapsed": 1187, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ef256801-37b8-48f6-f3b5-eefee1cbeee7"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python colab={"base_uri": "https://localhost:8080/"} id="h8c4LGrkmbNy" executionInfo={"status": "ok", "timestamp": 1630822593254, "user_tz": -330, "elapsed": 11031, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="238ce727-afac-4739-b48d-993f1fb88162"
!make setup
```

```python id="Tv4THPUQb3FL"
%reload_ext autoreload
%autoreload 2
```

<!-- #region id="0JF9rKloSN-B" -->
## Prototyping
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="5HtCujPxVJXK" executionInfo={"status": "ok", "timestamp": 1630818381500, "user_tz": -330, "elapsed": 17021, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4d7d3c92-e8bf-48e8-c225-59bd18f8b4d7"
!dvc pull ./data/bronze/30music/sessions_sample_10.parquet.snappy.dvc
```

```python id="Ykc1FmxXXMBv"
import os
import pandas as pd

from src.dataset import SessionDataset
```

```python id="C6yt9eETagyX"
data_source_path = './data/bronze/30music/sessions_sample_10.parquet.snappy'
data_target_path = './data/silver/30music/'
filter_last_months = 1
filter_topk = 1000
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="Ednl2i0OXh3K" executionInfo={"status": "ok", "timestamp": 1630819273474, "user_tz": -330, "elapsed": 564, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="882b596c-a58c-451f-ca05-4a2d3bd2e701"
df = pd.read_parquet(data_source_path)
df.head()
```

```python id="PmVfup5rXyXX"
sess_ds = SessionDataset(df)
```

```python id="iMdhF0jyZ8JB"
sess_ds.filter_by_time(filter_last_months)
sess_ds.convert_to_sequence(filter_topk)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="dyxbyOzsXyPn" executionInfo={"status": "ok", "timestamp": 1630819541759, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c6571077-0133-4c37-9eb4-49e4151b6ded"
sess_ds.data.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="3B2myuHna1yd" executionInfo={"status": "ok", "timestamp": 1630819572402, "user_tz": -330, "elapsed": 4, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9d6d6aa5-a169-4ea0-833c-dcf35a03377c"
stats = sess_ds.get_stats()
print(stats)
```

```python colab={"base_uri": "https://localhost:8080/"} id="jEoXquaja9Mg" executionInfo={"status": "ok", "timestamp": 1630819680229, "user_tz": -330, "elapsed": 669, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8b786978-5c94-4b72-b4ae-7e1be66f0df7"
sess_ds.last_session_out_split()
print("Train sessions: {} - Test sessions: {}".format(len(sess_ds.train), len(sess_ds.test)))
```

```python id="smmd-IDFcdCM"
!mkdir -p ./data/silver/30music
```

```python id="DdWvtaBfc_7q"
sess_ds.train.to_parquet(os.path.join(data_target_path, 'train.parquet.snappy'), compression='snappy')
sess_ds.test.to_parquet(os.path.join(data_target_path, 'test.parquet.snappy'), compression='snappy')
```

```python colab={"base_uri": "https://localhost:8080/"} id="hj3_i7KaeEI-" executionInfo={"status": "ok", "timestamp": 1630820419412, "user_tz": -330, "elapsed": 667, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2e7ace0a-1e96-4a83-b243-ba3a6019ba4c"
!echo $data_target_path*.snappy
```

```python id="xolCqKVLddbm"
!dvc -q add $data_target_path*.snappy
!dvc -q commit $data_target_path*.snappy
!dvc -q push $data_target_path*.snappy
```

```python id="Jtg4q8r7nFji"
!rm -r ./data/silver/30music/*
```

<!-- #region id="hUWbyI9oeWuf" -->
## DVC Pipeline
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="YmKTZQ47el_I" executionInfo={"status": "ok", "timestamp": 1630822889721, "user_tz": -330, "elapsed": 730, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="eb5e1c87-3e76-41bc-f77d-0b3f4b9e8f90"
%%writefile params.yaml

prepare_30music:
  filter_last_months: 1
  filter_topk: 1000
```

```python colab={"base_uri": "https://localhost:8080/"} id="aD_JFlUVjpG5" executionInfo={"status": "ok", "timestamp": 1630823238836, "user_tz": -330, "elapsed": 413, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8e47e3e1-0e4a-44aa-e4cf-94a52955ad76"
%%writefile src/dataprep/30music.py

import os
import sys
import yaml
import pandas as pd

from src.dataset import SessionDataset

params = yaml.safe_load(open("params.yaml"))["prepare_30music"]

if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython {dataprep}.py {data-source-filepath}\n")
    sys.exit(1)

filter_last_months = params['filter_last_months']
filter_topk = params['filter_topk']

data_source_path = str(sys.argv[1])
data_target_path = os.path.join("data", "silver", "30music")
data_target_path_train = os.path.join(data_target_path, 'train.parquet.snappy')
data_target_path_test = os.path.join(data_target_path, 'test.parquet.snappy')


def prepare_data(data_source_path,
                 filter_last_months, filter_topk,
                 data_target_path_train, data_target_path_test):
    
    df = pd.read_parquet(data_source_path)
    sess_ds = SessionDataset(df)

    sess_ds.filter_by_time(filter_last_months)
    sess_ds.convert_to_sequence(filter_topk)
    stats = sess_ds.get_stats()
    print(stats)
    sess_ds.last_session_out_split()
    print("Train sessions: {} - Test sessions: {}".format(len(sess_ds.train), len(sess_ds.test)))
    sess_ds.train.to_parquet(data_target_path_train, compression='snappy')
    sess_ds.test.to_parquet(data_target_path_test, compression='snappy')


os.makedirs(data_target_path, exist_ok=True)

prepare_data(data_source_path,
             filter_last_months, filter_topk,
             data_target_path_train, data_target_path_test)
```

```python colab={"base_uri": "https://localhost:8080/"} id="rI1tWch_g8IH" executionInfo={"status": "ok", "timestamp": 1630823281937, "user_tz": -330, "elapsed": 2607, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5534d4d7-d5b6-4294-d738-dfe945a06462"
!dvc run \
    -n prepare_30music \
    -p prepare_30music.filter_last_months,prepare_30music.filter_topk \
    -d src/dataprep/30music.py -d data/bronze/30music/sessions_sample_10.parquet.snappy \
    -o data/silver/30music \
    python src/dataprep/30music.py data/bronze/30music/sessions_sample_10.parquet.snappy
```

```python id="722DedyfpTc2"

```
