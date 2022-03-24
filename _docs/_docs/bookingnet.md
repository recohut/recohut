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

# BookingNet

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

```python id="wkaGHauFMz3D"
train = pd.read_parquet('./data/bronze/booking/train.parquet.snappy')
train = train.sort_values(by=['utrip_id','checkin'])
train
```

```python id="X34UvBzRNZF3"
test = pd.read_parquet('./data/bronze/booking/test.parquet.snappy')
test = test.sort_values(by=['utrip_id','checkin'])
test
```

```python id="wo1l1GuHNHjl"
# Generate Dummy Predictions - use top 4 cities in the trainset as benchmark recommendation
topcities = train.city_id.value_counts().index[:4]

test_trips = (test[['utrip_id']].drop_duplicates()).reset_index().drop('index', axis=1)

cities_prediction = pd.DataFrame([topcities]*test_trips.shape[0], columns=['city_id_1','city_id_2','city_id_3','city_id_4'])

cities_prediction = pd.concat([test_trips, cities_prediction], axis =1)
cities_prediction
```

```python id="ENsq0VxUOHP9"
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

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 11670, "status": "ok", "timestamp": 1631266839746, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="9TnNISyzKcY_" outputId="12c977c6-e8cc-4d5d-b62d-01ade714ed35"
!pip install -q ipytest
import ipytest
ipytest.autoconfig()
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

```python executionInfo={"elapsed": 407, "status": "ok", "timestamp": 1631270453089, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="5PeruxlZg6rW"
import numpy as np
import pandas as pd
from typing import List
```

```python executionInfo={"elapsed": 600, "status": "ok", "timestamp": 1631273370583, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="cE7tL_8mdIvy"
FEATURES_TO_ENCODE = ['city_id', 'device_class', 'affiliate_id',
                      'booker_country', 'hotel_country', 'checkin_year',
                      'days_stay', 'checkin_day', 'checkin_month',
                      'transition_days']

NEXT_CITY_COLUMNS = ['city_id', 'affiliate_id',
                      'booker_country', 'days_stay',
                      'checkin_day']
```

```python executionInfo={"elapsed": 618, "status": "ok", "timestamp": 1631273150164, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="07r6TLDJdNFn"
import logging
from typing import List, Dict

import numpy as np
import pandas as pd


class LabelEncoder:
    """
    LabelEncoder similar to `sklearn.preprocessing.LabelEncoder`
    with the exception it ignores `NaN` values.
    .. todo:: Enhance this encoder with the option to set a `min_frequency`.
    """

    def fit_transform(self, col: pd.Series) -> pd.Series:
        self.rev_classes_ = dict(enumerate(sorted(col.dropna().unique())))
        self.classes_ = {v: k for k, v in self.rev_classes_.items()}
        return col.apply(lambda k: self.classes_.get(k, np.nan))

    def inverse_transform(self, col: pd.Series) -> pd.Series:
        return col.apply(lambda k: self.rev_classes_.get(k, np.nan))


class DatasetEncoder:
    """
    DatasetEncoder looks to encapsulate multiple LabelEncoder objects
    to fully transform a dataset.
    """

    def __init__(self, features_embedding: List[str]):
        self.label_encoders = {c: LabelEncoder() for c in features_embedding}

    def fit_transform(self, df: pd.DataFrame) -> None:
        """
        Transform columns in all columns given by feature_embedding.
         df:
        :return:
        """
        logging.info("Running LabelEncoder on columns")
        for column, encoder in self.label_encoders.items():
            # reserve zero index for OOV elements
            df[column] = encoder.fit_transform(df[column]) + 1
            logging.info(f"{column}: {len(encoder.classes_)}")


def get_embedding_complexity_proxy(dataset_encoder: DatasetEncoder) -> Dict:
    """
    Get embedding complexity proxy
    The idea is to find out how many bits (dimension) we need to naively encode each element in the encoder.
    It's a proxy since we have no idea which is the dimension of the underlying manifold for every feature.
    """
    return {k: (len(v.classes_), np.ceil(np.log2(len(v.classes_))))
            for k, v in dataset_encoder.label_encoders.items()}
```

```python executionInfo={"elapsed": 538, "status": "ok", "timestamp": 1631271326132, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="Sw8Ij75jJJro"
class Dataset:
    def __init__(self, data=None, is_train=True):
        self.data = data
        self.is_train = is_train

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

    def preprocess(self):
        pass

    def __repr__(self):
        return '{}\n{}\n{}\n{}'\
        .format(
            self.data.info(),
            '='*100,
            self.data.head(),
            '='*100
            )
```

```python executionInfo={"elapsed": 431, "status": "ok", "timestamp": 1631270555370, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="pMeB4-BCS40f"
class BookingDataset(self.data):

    def encode(self):
        de = DatasetEncoder(config.FEATURES_TO_ENCODE)
        de.fit_transform(self.data)

    def set_future_features(self):
        # Add features about the next city to the dataframe.
        for column in NEXT_CITY_COLUMNS:
            self.data['next_' + column] = self.data.groupby('utrip_id')[column].shift(periods=-1)

    def min_sequence_length(self):
        # Constrains the minimum trip length to `sequence_length`.
        sequence_length = 3
        self.data.groupby('utrip_id').filter(lambda x: len(x) >= sequence_length)

    def preprocess(self):
        self.data['city_id'] = self.data['city_id'].replace({0: np.nan})
    
    def featurize(self):
        # create some time features
        self.data['days_stay'] = (self.data['checkout'] - self.data['checkin']).dt.days - 1
        self.data['checkin_day'] = self.data['checkin'].dt.dayofweek
        self.data['checkin_month'] = self.data['checkin'].dt.month
        self.data['checkin_year'] = self.data['checkin'].dt.year

        # create transition time feature
        self.data['prev_checkout'] = self.data.groupby('utrip_id')['checkout'].shift(periods=1)
        self.data['transition_days'] = (self.data['checkout'] - self.data['prev_checkout']).dt.days - 1
        self.data['transition_days'].fillna(0, inplace=True)
        self.data.drop(columns="prev_checkout", inplace=True)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2157, "status": "ok", "timestamp": 1631270624996, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="91AGmHAFTWV4" outputId="edb093a4-c774-4ccd-f9c5-c12d379ea098"
ds = BookingDataset()
ds.load('./data/bronze/booking/train.parquet.snappy')
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

```python id="k22cNx-fieRJ"
train = Dataset()

train_info = train.load('./data/bronze/booking/train.parquet.snappy')\
                        .sort(by=['utrip_id','checkin'])\
                        .filter(by='cols', keep=['utrip_id','city_id'])\
                        .rename({'utrip_id':'USER_ID','city_id':'ITEM_ID'})\
                        .cast({'USER_ID':'str', 'ITEM_ID':'str'})
train_info
```

```python id="tdZT8bW6jVGg"
test = Dataset()

test_info = test.load('./data/bronze/booking/test.parquet.snappy')\
                        .sort(by=['utrip_id','checkin'])\
                        .filter(by='cols', keep=['utrip_id','city_id'])\
                        .rename({'utrip_id':'USER_ID','city_id':'ITEM_ID'})\
                        .cast({'USER_ID':'str', 'ITEM_ID':'str'})
test_info
```

```python id="FPtEFAe_cUps"
model = Model()
model.fit(train)
model.recommend('1000066_2')
```

```python id="B7W5XpXGcXYu"
hitrate = HitRate(k=4)
hitrate
print(hitrate.calculate(recommended_list=['1','2','3','4','5'], actual_list = ['4']))
print(hitrate.calculate(recommended_list=['1','2','3','4','5'], actual_list = ['5']))
```

```python id="iqaekUZ-o7k-"
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

```python id="mF3AndoQm2iS"
eval.evaluate()
```
