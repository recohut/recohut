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

```python id="rxL9QQCZPx8Q" executionInfo={"status": "ok", "timestamp": 1631029847417, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-chef"; branch = "30music"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python id="dwLTtRQK88Yz" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631029855846, "user_tz": -330, "elapsed": 8438, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="32d07ffe-8119-4f1e-c58c-7cfb5866293a"
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

```python id="2jrtc9Bg88Y0" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631036426969, "user_tz": -330, "elapsed": 506, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="10676076-e99a-4a68-ae41-a184982b5e24"
!git status
```

```python id="2G4iErkK88Y1" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631036458857, "user_tz": -330, "elapsed": 1156, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a2d0537e-1067-4c7c-cd71-e4ce4934306f"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python id="h8c4LGrkmbNy" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631029935998, "user_tz": -330, "elapsed": 10952, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a80e3775-f66e-42c1-9168-fad606a77fab"
!make setup
```

```python id="Tv4THPUQb3FL" executionInfo={"status": "ok", "timestamp": 1631029936001, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
%reload_ext autoreload
%autoreload 2
```

```python id="5HtCujPxVJXK"
!dvc pull ./data/bronze/30music/sessions_sample_10.parquet.snappy.dvc
```

```python id="722DedyfpTc2"
!dvc repro
```

```python colab={"base_uri": "https://localhost:8080/"} id="siRN938DTFXz" executionInfo={"status": "ok", "timestamp": 1631036025041, "user_tz": -330, "elapsed": 440, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d46a7da9-8537-4859-fa5f-9bb0fa4f837e"
%%writefile ./src/prepare.py

import os
import sys
import yaml
import pandas as pd

from src.dataset import SessionDataset

params = yaml.safe_load(open("params.yaml"))["prepare"]

if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython {dataprep}.py {data-source-filepath} {data-target-dirpath}\n")
    sys.exit(1)

filter_last_months = params['filter_last_months']
filter_topk = params['filter_topk']

data_source_path = str(sys.argv[1])
data_target_path = str(sys.argv[2])
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

```python colab={"base_uri": "https://localhost:8080/"} id="mWuGmD_wEcoO" executionInfo={"status": "ok", "timestamp": 1631035774845, "user_tz": -330, "elapsed": 403, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d2102f40-baa2-49d0-df89-d26c2cf218d4"
%%writefile ./src/train.py
import os
import sys
import yaml
import pickle
import pandas as pd

from src.models import PopularityRecommender

params = yaml.safe_load(open("params.yaml"))["train"]

models = {
    'itempop': PopularityRecommender()
}


def load_data(path):
    path_train = os.path.join(path, 'train.parquet.snappy')
    path_test = os.path.join(path, 'test.parquet.snappy')
    train = pd.read_parquet(path_train)
    test = pd.read_parquet(path_test)
    return train, test


def load_model(name):
    model = models[name]
    return model


if __name__ == "__main__":
    model_name = params['model_name']
    data_path = str(sys.argv[1])
    model_path = str(sys.argv[2])
    train, test = load_data(data_path)
    model = load_model(model_name)
    model.fit(train)
    pickle.dump(model, open(os.path.join(model_path, model_name+'.pkl'), 'wb'))
```

<!-- #region id="cUVAFl9d-bXW" -->
## Scripts
<!-- #endregion -->

```python id="nffjTR9IH_Fh" executionInfo={"status": "ok", "timestamp": 1631034478145, "user_tz": -330, "elapsed": 1214, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
!python ./src/train.py ./data/silver/30music ./artifacts/30music/models
```

```python id="FYvYvzIB-IvL"
!python ./src/eval/itempop.py \
./data/silver/30music \
./artifacts/30music/models/itempop.pkl
```

```python id="sxQpwbx1-T0L"
import pickle
results = pickle.load(open('./artifacts/30music/results/itempop.pkl', 'rb'))
results
```

<!-- #region id="zyMZsab7_NJW" -->
## DVC Pipeline
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="tTq83sNd_OfC" executionInfo={"status": "ok", "timestamp": 1631033916267, "user_tz": -330, "elapsed": 477, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f892e52a-f825-4925-9584-ea061a51d94a"
%%writefile params.yaml

prepare:
  filter_last_months: 1
  filter_topk: 1000

train:
  model_name: itempop
```

```python colab={"base_uri": "https://localhost:8080/"} id="OtkaNESt_WyK" executionInfo={"status": "ok", "timestamp": 1631036421317, "user_tz": -330, "elapsed": 496, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="892008df-869e-4ece-d71d-1546b78cd57d"
%%writefile dvc.yaml
stages:
    prepare:
        cmd: python src/prepare.py data/bronze/30music/sessions_sample_10.parquet.snappy data/silver/30music
        deps:
        - data/bronze/30music/sessions_sample_10.parquet.snappy
        - src/dataprep/30music.py
        params:
        - prepare.filter_last_months
        - prepare.filter_topk
        outs:
        - data/silver/30music
    train:
        cmd: python src/train.py data/silver/30music artifacts/30music/models
        deps:
        - data/silver/30music/test.parquet.snappy
        - data/silver/30music/train.parquet.snappy
        - src/train.py
        params:
        - train.model_name
    evaluate:
        cmd: python
```

```python id="yE7q7TpuOQb1"

```
