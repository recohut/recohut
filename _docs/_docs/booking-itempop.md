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

# Booking Popularity-based RecSys

<!-- #region id="MRrUvPKgE0rk" -->
## Setup
<!-- #endregion -->

```python id="0G3R3Ab3p1Wb"
import os
project_name = "chef-session"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python id="cfu34YGOp1Wc"
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

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 527, "status": "ok", "timestamp": 1631210654024, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="Bq_pdLr1p1Wf" outputId="09c45062-aeda-47e2-88cc-b3055e578ede"
!git status
```

```python id="EKT3dcx0p1Wg"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python id="Ez3BYNp6JExF"
!dvc pull ./data/bronze/booking/*
```

<!-- #region id="SZGyQmNZFlPH" -->
## Context
<!-- #endregion -->

<!-- #region id="b9CFffH-FnFA" -->
- Booking.com dataset
    - Popularity recommender, hit rate evaluation

<!-- #endregion -->

<!-- #region id="lCEBxCM1FcLW" -->
## Prototype
<!-- #endregion -->

```python id="JyJVyJgPNFZ8"
import pandas as pd
```

```python colab={"base_uri": "https://localhost:8080/", "height": 419} executionInfo={"elapsed": 1460, "status": "ok", "timestamp": 1631188781550, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="wkaGHauFMz3D" outputId="130a108a-d7c7-4761-f74e-d9d0f0a0e42a"
train = pd.read_parquet('./data/bronze/booking/train.parquet.snappy')
train = train.sort_values(by=['utrip_id','checkin'])
train
```

```python colab={"base_uri": "https://localhost:8080/", "height": 589} executionInfo={"elapsed": 719, "status": "ok", "timestamp": 1631134796016, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="X34UvBzRNZF3" outputId="cb1727d4-cfa1-4db7-d0ba-458ee7f70241"
test = pd.read_parquet('./data/bronze/booking/test.parquet.snappy')
test = test.sort_values(by=['utrip_id','checkin'])
test
```

```python colab={"base_uri": "https://localhost:8080/", "height": 419} executionInfo={"elapsed": 975, "status": "ok", "timestamp": 1631135416115, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="wo1l1GuHNHjl" outputId="9eaad82f-a5f3-4846-b8fd-bff72f7bf787"
# Generate Dummy Predictions - use top 4 cities in the trainset as benchmark recommendation
topcities = train.city_id.value_counts().index[:4]

test_trips = (test[['utrip_id']].drop_duplicates()).reset_index().drop('index', axis=1)

cities_prediction = pd.DataFrame([topcities]*test_trips.shape[0], columns=['city_id_1','city_id_2','city_id_3','city_id_4'])

cities_prediction = pd.concat([test_trips, cities_prediction], axis =1)
cities_prediction
```

```python colab={"base_uri": "https://localhost:8080/", "height": 450} executionInfo={"elapsed": 549, "status": "ok", "timestamp": 1631135189445, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="ENsq0VxUOHP9" outputId="b2113edb-1cf2-4e59-971a-dd9d4a292722"
ground_truth = pd.read_parquet('./data/bronze/booking/ground_truth.parquet.snappy')
ground_truth.set_index('utrip_id', inplace=True)
ground_truth
```

```python id="Q2maoXWbN-S_"
def evaluate_accuracy_at_4(predicted, actual):
    '''checks if the true city is within the four recommended cities'''
    data = predicted.join(actual, on='utrip_id')

    hits = ((data['city_id']==data['city_id_1'])|(data['city_id']==data['city_id_2'])|
        (data['city_id']==data['city_id_3'])|(data['city_id']==data['city_id_4']))*1
    return hits.mean()
```

```python id="EbeGDIoyN_mp"
evaluate_accuracy_at_4(cities_prediction, ground_truth)
```

<!-- #region id="gvEoJau1KaiU" -->
## Tests
<!-- #endregion -->

```python id="9TnNISyzKcY_"
!pip install -q ipytest
import ipytest
ipytest.autoconfig()
```

```python id="ctvELvsJcrO7"
# %%ipytest

```

```python id="3pd8Iwxbh08i"
train = Dataset(path: str)
test = Dataset(path: str)
model = Model()
model.fit(train: pd.DataFrame)
model.recommend(test: pd.DataFrame, topk=4)

metrics = Metrics()
hr = metrics.HitRate(k=4)

eval = Evaluator(model,
                 data = test,
                 metrics=[hr])
eval.evaluate()
eval.save_results(path: str)
```

<!-- #region id="5Ewbbn7gYlo7" -->
## Dev
<!-- #endregion -->

```python id="5PeruxlZg6rW"
import numpy as np
import pandas as pd
from typing import List
```

```python id="Sw8Ij75jJJro"
class Dataset:
    def __init__(self, data=None):
        self.data = data

    def load(self, path, type='parquet'):
        if type=='parquet':
            self.data = pd.read_parquet(path)
        return self

    def sort(self, by: List):
        self.data.sort_values(by=by)
        return self

    def filter(self, by='cols', keep=[]):
        if by=='cols':
            self.data = self.data[keep]
        return self
    
    def rename(self, rename_map):
        self.data = self.data.rename(columns=rename_map)
        return self
    
    def cast(self, schema_map):
        self.data = self.data.astype(schema_map)
        return self

    def __repr__(self):
        return '{}\n{}\n{}\n{}'\
        .format(
            self.data.info(),
            '='*100,
            self.data.head(),
            '='*100
            )
```

```python id="eIBJmW8tJdeN"
class Model:
    def __init__(self):
        self.items_by_popularity = []

    def fit(self, train):
        self.items_by_popularity = train.data['ITEM_ID'].value_counts().index.tolist()

    def recommend(self, uid=None, topk=4):
        return self.items_by_popularity[:topk]
```

```python id="AKHsp25GdIFn"
class HitRate:
    def __init__(self, k=4):
        self.k = k

    def calculate(self, recommended_list, actual_list):
        actual_list = np.array(actual_list) 
        recommended_list = np.array(recommended_list)[:self.k]
        flags = np.isin(actual_list, recommended_list) 
        return (flags.sum() > 0) * 1

    def __repr__(self):
        return 'HR@{}'.format(self.k)
```

```python id="Zg6ZlHGMKVjc"
class Evaluate:
    def __init__(self, model, test_ids, ground_truth, metrics):
        self.model = model
        self.test_ids = test_ids
        self.ground_truth = ground_truth
        self.metrics = metrics
        self.results = {}
        self.recommendations = {}
        self._calculate_recommendations()
    
    def _calculate_recommendations(self):
        for test_id in self.test_ids:
            self.recommendations[test_id] = self.model.recommend(test_id)

    def evaluate(self):
        for metric in self.metrics:
            self.results[metric] = 0
            scores = []
            for test_id in self.test_ids:
                actual_list = self.ground_truth[test_id]
                recommended_list = self.recommendations[test_id]
                score = metric.calculate(recommended_list=recommended_list,
                                         actual_list=actual_list)
                scores.append(score)
            self.results[metric] = np.mean(scores)
        return self

    def save_results(self, path):
        with open(path, 'wt') as handle:
            self.results.write(str(handle))
    
    def __repr__(self):
        return str(self.results)
```

<!-- #region id="s43SXdMzgmVw" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2796, "status": "ok", "timestamp": 1631208121988, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="k22cNx-fieRJ" outputId="a54ea4c6-75d0-4ffd-b4af-93d83b55ca30"
train = Dataset()

train_info = train.load('./data/bronze/booking/train.parquet.snappy')\
                        .sort(by=['utrip_id','checkin'])\
                        .filter(by='cols', keep=['utrip_id','city_id'])\
                        .rename({'utrip_id':'USER_ID','city_id':'ITEM_ID'})\
                        .cast({'USER_ID':'str', 'ITEM_ID':'str'})
train_info
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2132, "status": "ok", "timestamp": 1631208202839, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="tdZT8bW6jVGg" outputId="8c9bd86c-2f85-4edf-f330-f137dba854a3"
test = Dataset()

test_info = test.load('./data/bronze/booking/test.parquet.snappy')\
                        .sort(by=['utrip_id','checkin'])\
                        .filter(by='cols', keep=['utrip_id','city_id'])\
                        .rename({'utrip_id':'USER_ID','city_id':'ITEM_ID'})\
                        .cast({'USER_ID':'str', 'ITEM_ID':'str'})
test_info
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 393, "status": "ok", "timestamp": 1631208377078, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="FPtEFAe_cUps" outputId="b8b429a9-f17a-43a7-91e7-da116ee72eda"
model = Model()
model.fit(train)
model.recommend('1000066_2')
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 552, "status": "ok", "timestamp": 1631208808409, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="B7W5XpXGcXYu" outputId="c561c167-598b-4bb4-82eb-84f3bae37cc4"
hitrate = HitRate(k=4)
hitrate
print(hitrate.calculate(recommended_list=['1','2','3','4','5'], actual_list = ['4']))
print(hitrate.calculate(recommended_list=['1','2','3','4','5'], actual_list = ['5']))
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 605, "status": "ok", "timestamp": 1631209914026, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="iqaekUZ-o7k-" outputId="d55e798c-e9bb-4c97-a37c-7a7cb892be2a"
ground_truth = Dataset()

gt_info = ground_truth.load('./data/bronze/booking/ground_truth.parquet.snappy')\
                            .filter(by='cols', keep=['utrip_id','city_id'])\
                            .rename({'utrip_id':'USER_ID','city_id':'ITEM_ID'})\
                            .cast({'USER_ID':'str', 'ITEM_ID':'str'})

ground_truth = ground_truth.data\
                    .drop_duplicates(subset='USER_ID', keep='last')\
                    .set_index('USER_ID')\
                    .to_dict()['ITEM_ID']

print(type(ground_truth), len(ground_truth.keys()))
```

```python id="glYtKR5SmDWM"
eval = Evaluate(model=model,
                test_ids=test.data.USER_ID.unique(),
                ground_truth=ground_truth,
                metrics=[hitrate])
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2339, "status": "ok", "timestamp": 1631210346836, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="mF3AndoQm2iS" outputId="7e930c5b-9dad-4573-cfd7-50df39752d8b"
eval.evaluate()
```
