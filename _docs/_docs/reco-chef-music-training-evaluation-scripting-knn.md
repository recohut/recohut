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

<!-- #region id="cZ73HjGo4kE8" -->
## Setup
<!-- #endregion -->

```python id="rxL9QQCZPx8Q" executionInfo={"status": "ok", "timestamp": 1631079029841, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-chef"; branch = "30music"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python id="dwLTtRQK88Yz" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631079038368, "user_tz": -330, "elapsed": 7750, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="399fde88-b654-44ab-e7fd-9062d3505e2a"
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

```python id="2jrtc9Bg88Y0" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631085995518, "user_tz": -330, "elapsed": 862, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1512aa2e-de3d-4bc5-bc07-bf23e454dc84"
!git status
```

```python id="2G4iErkK88Y1" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631086005229, "user_tz": -330, "elapsed": 2168, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2d2f3f83-257c-4487-feb5-4752c1bf340b"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python id="o7iCtIG5CZiR"
!dvc status
```

```python id="Klqi_zTLF_1d" executionInfo={"status": "ok", "timestamp": 1631079059947, "user_tz": -330, "elapsed": 11630, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
!pip install -q -U -e .
```

```python id="Tv4THPUQb3FL" executionInfo={"status": "ok", "timestamp": 1631079065675, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
%reload_ext autoreload
%autoreload 2
```

```python id="5HtCujPxVJXK"
!dvc pull ./data/bronze/30music/sessions_sample_10.parquet.snappy.dvc
```

```python colab={"base_uri": "https://localhost:8080/"} id="Fa9UMRTBCFda" executionInfo={"status": "ok", "timestamp": 1631081624657, "user_tz": -330, "elapsed": 1302, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4f07efe2-e444-4191-8d2b-8da576333338"
!python src/prepare.py data/bronze/30music/sessions_sample_10.parquet.snappy data/silver/30music
```

<!-- #region id="sssecbG64iew" -->
## Development
<!-- #endregion -->

```python id="JgqF4_JIJf6y"

import operator

from src.models.abstract import ISeqRecommender


class PopularityRecommender(ISeqRecommender):

    def __init__(self):
        super(PopularityRecommender, self).__init__()

    def fit(self, train_data):
        sequences = train_data['sequence'].values

        count_dict = {}
        for s in sequences:
            for item in s:
                if item not in count_dict:
                    count_dict[item] = 1
                else:
                    count_dict[item] += 1

        self.top = sorted(count_dict.items(), key=operator.itemgetter(1), reverse=True)
        self.top = [([x[0]], x[1]) for x in self.top]

    def recommend(self, user_profile, user_id=None):
        return self.top

    def get_popular_list(self):
        return self.top
```

<!-- #region id="r_uefzI5DKEQ" -->
### Train
<!-- #endregion -->

```python id="Eu4XJdUn7dvQ" executionInfo={"status": "ok", "timestamp": 1631081638374, "user_tz": -330, "elapsed": 602, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
import sys
import pickle
import pandas as pd

from src.models import PopularityRecommender, KNNRecommender


# import yaml
# params = yaml.safe_load(open("params.yaml"))["train"]
# modelname = params['model_name']


def load_data(datapath):
    path_train = os.path.join(datapath, 'train.parquet.snappy')
    path_test = os.path.join(datapath, 'test.parquet.snappy')
    train = pd.read_parquet(path_train)
    test = pd.read_parquet(path_test)
    return train, test


def train_model(modelname, train):
    models = {'itempop': PopularityRecommender()}
    model = models[modelname]
    model.fit(train)
    return model


def save_model(model, modelname, modelpath):
    pickle.dump(model, open(os.path.join(modelpath, modelname+'.pkl'), 'wb'))


if __name__ == "__main__":
    # load the params
    modelname = 'itempop'
    datapath = './data/silver/30music'
    modelpath = './artifacts/30music/models'
    # load the data
    train, test = load_data(datapath)
    # train the model
    model = train_model(modelname, train)
    # save the model
    save_model(model, modelname, modelpath)
```

<!-- #region id="2bYidLV1DLZs" -->
### Evaluate
<!-- #endregion -->

```python id="rrtlT-cs4jX4"
import os
import sys
import pickle
import pandas as pd

from src.eval import SequentialEvaluator


def load_data(datapath):
    path_train = os.path.join(datapath, 'train.parquet.snappy')
    path_test = os.path.join(datapath, 'test.parquet.snappy')
    train = pd.read_parquet(path_train)
    test = pd.read_parquet(path_test)
    return train, test


def load_model(modelpath):
    model = pickle.load(open(modelpath, 'rb'))
    return model


def save_results(evaluator, resultspath):
    results = {}
    results['seq_reveal'] = evaluator.eval_seqreveal()
    results['static_profile'] = evaluator.eval_staticprofile()
    results['rec_length'] = evaluator.eval_reclength()
    results['profile_length'] = evaluator.eval_profilelength()
    pickle.dump(results, open(resultspath, 'wb'))


if __name__ == "__main__":
    # load the params
    datapath = './data/silver/30music'
    modelpath = './artifacts/30music/models/itempop.pkl'
    resultspath = './artifacts/30music/results/itempop.pkl'
    # load the data
    train, test = load_data(datapath)
    # load the model
    model = load_model(modelpath)
    # evaluate and save the results
    evaluator = SequentialEvaluator(train, test, model)
    save_results(evaluator, resultspath)
```

<!-- #region id="Z3rmReHm4mhx" -->
## Scripting
<!-- #endregion -->

```python id="roFrzEtG4nK3"
import os
import sys
import yaml
import pickle
import pandas as pd

from src.models import PopularityRecommender, KNNRecommender


params = yaml.safe_load(open("params.yaml"))["train"]


def load_data(datapath):
    path_train = os.path.join(datapath, 'train.parquet.snappy')
    path_test = os.path.join(datapath, 'test.parquet.snappy')
    train = pd.read_parquet(path_train)
    test = pd.read_parquet(path_test)
    return train, test


def train_model(modelname, train):
    models = {'itempop': PopularityRecommender(),
              'knn': KNNRecommender(model='sknn', k=10)}
    model = models[modelname]
    model.fit(train)
    return model


def save_model(model, modelname, modelpath):
    pickle.dump(model, open(os.path.join(modelpath, modelname+'.pkl'), 'wb'))


if __name__ == "__main__":
    # load the params
    modelname = params['model_name']
    datapath = str(sys.argv[1])
    modelpath = str(sys.argv[2])
    # load the data
    train, test = load_data(datapath)
    # train the model
    model = train_model(modelname, train)
    # save the model
    save_model(model, modelname, modelpath)
```

```python id="j3dgGg5uSoRk"
import os
import sys
import pickle
import pandas as pd

from src.eval import SequentialEvaluator


def load_data(datapath):
    path_train = os.path.join(datapath, 'train.parquet.snappy')
    path_test = os.path.join(datapath, 'test.parquet.snappy')
    train = pd.read_parquet(path_train)
    test = pd.read_parquet(path_test)
    return train, test


def load_model(modelpath):
    model = pickle.load(open(modelpath, 'rb'))
    return model


def save_results(evaluator, resultspath):
    results = {}
    results['seq_reveal'] = evaluator.eval_seqreveal()
    results['static_profile'] = evaluator.eval_staticprofile()
    results['rec_length'] = evaluator.eval_reclength()
    results['profile_length'] = evaluator.eval_profilelength()
    pickle.dump(results, open(resultspath, 'wb'))


if __name__ == "__main__":
    # load the params
    datapath = str(sys.argv[1])
    modelpath = str(sys.argv[2])
    resultspath = str(sys.argv[3])
    # load the data
    train, test = load_data(datapath)
    # load the model
    model = load_model(modelpath)
    # evaluate and save the results
    evaluator = SequentialEvaluator(train, test, model)
    save_results(evaluator, resultspath)
```
