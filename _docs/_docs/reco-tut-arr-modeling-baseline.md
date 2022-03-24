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

```python id="-UOOzCs9ukul" executionInfo={"status": "ok", "timestamp": 1628004655220, "user_tz": -330, "elapsed": 420, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
project_name = "reco-tut-arr"; branch = "main"; account = "sparsh-ai"
```

```python colab={"base_uri": "https://localhost:8080/"} id="PYvHGli8ukum" executionInfo={"status": "ok", "timestamp": 1628004657725, "user_tz": -330, "elapsed": 549, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c7089d8c-7459-4f84-a887-c773c94792f0"
import os

if not os.path.exists('/content/reco-tut-arr'):
    !cp /content/drive/MyDrive/mykeys.py /content
    import mykeys
    !rm /content/mykeys.py
    path = "/content/" + project_name; 
    !mkdir "{path}"
    %cd "{path}"
    import sys; sys.path.append(path)
    !git config --global user.email "arr@recohut.com"
    !git config --global user.name  "reco-tut-arr"
    !git init
    !git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout main
else:
    %cd '/content/reco-tut-arr'
```

```python id="rpddT_FwnhMb" executionInfo={"status": "ok", "timestamp": 1628004661774, "user_tz": -330, "elapsed": 1977, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score,classification_report
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier
# from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold,StratifiedKFold, GroupKFold,train_test_split

import gc
import datetime
from tqdm.notebook import tqdm

import warnings
warnings.filterwarnings("ignore")

np.random.seed(0)
```

```python colab={"base_uri": "https://localhost:8080/"} id="UExHTcpinstU" executionInfo={"status": "ok", "timestamp": 1628004661778, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e87347f9-bd7d-4b6a-b488-a0302c487c10"
data_path = dict()

for dirname, _, filenames in os.walk('./data/gold'):
    for filename in filenames:
        if filename.endswith('.parquet.gzip'):
            name = filename.split('.')[0]
            data_path[name] = os.path.join(dirname, filename)

data_path
```

```python colab={"base_uri": "https://localhost:8080/"} id="kzMgQlvynwIu" executionInfo={"status": "ok", "timestamp": 1628004673409, "user_tz": -330, "elapsed": 9134, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3128bf1f-6548-4f47-d4e8-ff9e0bcde968"
train = pd.read_parquet('./data/gold/train.parquet.gzip')
test = pd.read_parquet('./data/gold/test.parquet.gzip')

train.shape, test.shape
```

<!-- #region id="qY73ZFjQoUhA" -->
Data sampling for quick modeling
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="1YZoVfnfowHd" executionInfo={"status": "ok", "timestamp": 1628004756674, "user_tz": -330, "elapsed": 1710, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="88a5fe36-848c-4282-a694-bee9b4a0b31f"
train = train.sample(frac=0.1)
train.shape
```

```python id="Vlbg9MkLn879" executionInfo={"status": "ok", "timestamp": 1628004767015, "user_tz": -330, "elapsed": 904, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b

def bearing_array(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))
```

```python id="Kdcj4DbBoBz4" executionInfo={"status": "ok", "timestamp": 1628004767690, "user_tz": -330, "elapsed": 684, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
train['center_latitude'] = (train['c_latitude'].values + train['v_latitude'].values) / 2
train['center_longitude'] = (train['c_longitude'].values + train['v_longitude'].values) / 2
train['harvesine_dist']=haversine_array(train['c_latitude'], train['c_longitude'], train['v_latitude'], train['v_longitude'])
train['manhattan_dist']=dummy_manhattan_distance(train['c_latitude'], train['c_longitude'], train['v_latitude'], train['v_longitude'])
train['bearing']=bearing_array(train['c_latitude'], train['c_longitude'], train['v_latitude'], train['v_longitude'])
```

```python id="Kv8fsU_aoC-F" executionInfo={"status": "ok", "timestamp": 1628004771882, "user_tz": -330, "elapsed": 1165, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
test['center_latitude'] = (test['c_latitude'].values + test['v_latitude'].values) / 2
test['center_longitude'] = (test['c_longitude'].values + test['v_longitude'].values) / 2
test['harvesine_dist']=haversine_array(test['c_latitude'], test['c_longitude'], test['v_latitude'], test['v_longitude'])
test['manhattan_dist']=dummy_manhattan_distance(test['c_latitude'], test['c_longitude'], test['v_latitude'], test['v_longitude'])
test['bearing']=bearing_array(test['c_latitude'], test['c_longitude'], test['v_latitude'], test['v_longitude'])
```

```python id="bnZZJVBToDe-" executionInfo={"status": "ok", "timestamp": 1628004783839, "user_tz": -330, "elapsed": 597, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
y = train['target']
del train['target']
```

```python colab={"base_uri": "https://localhost:8080/"} id="9MMbhkAnpgXz" executionInfo={"status": "ok", "timestamp": 1628004838517, "user_tz": -330, "elapsed": 931, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6073280f-34b3-45dc-bc22-d9d457161be0"
y.value_counts(dropna=False)
```

```python id="yRG5G-79pZFT" executionInfo={"status": "ok", "timestamp": 1628004812022, "user_tz": -330, "elapsed": 2429, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.20, random_state=42, stratify=y)
```

<!-- #region id="xZT6KHLGpncK" -->
Model Training on Validation Data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="osm8NyAQpt2H" executionInfo={"status": "ok", "timestamp": 1628005045769, "user_tz": -330, "elapsed": 59394, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ff961bad-4d6e-40a1-eb96-3a383a4632cb"
%%time
m = LGBMClassifier(n_estimators = 300, objective = "binary", metric = "auc", learning_rate = 0.1,
                   random_state = 42, scale_pos_weight = 7.3, bagging_fraction = 0.8,
                   bagging_freq = 1, feature_fraction = 0.8, n_jobs=-1)

m.fit(X_train, y_train, eval_set=[(X_train, y_train),(X_test, y_test)], 
      early_stopping_rounds = 50, verbose = 100)

pred = m.predict(X_test)

print('F1 Score: ', f1_score(y_test, pred))
```

```python colab={"base_uri": "https://localhost:8080/"} id="4zMy6W9iqKn4" executionInfo={"status": "ok", "timestamp": 1628005055410, "user_tz": -330, "elapsed": 521, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="316645f6-09a2-43af-b10e-1c8593bbc73a"
print(classification_report(y_test, pred))
```

<!-- #region id="s3v6mU71qcPO" -->
Model Training on Complete Data
<!-- #endregion -->

```python id="bN1zsGgcqjy5"
m = LGBMClassifier(n_estimators = 231, objective = "binary", metric = "auc", learning_rate = 0.1,
                   random_state = 42, scale_pos_weight = 7.3, bagging_fraction = 0.8,
                   bagging_freq = 1, feature_fraction = 0.8, n_jobs=-1)

m.fit(train, y)

full_pred = m.predict(test)
```

```python colab={"base_uri": "https://localhost:8080/"} id="90cynIJlrj0H" executionInfo={"status": "ok", "timestamp": 1628005403276, "user_tz": -330, "elapsed": 526, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="661888b1-9ba2-43a1-982d-0f0fa35d7a83"
final = pd.DataFrame()
final['target'] = full_pred
final.target.value_counts()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 729} id="eyQN98MZrE8p" executionInfo={"status": "ok", "timestamp": 1628005409850, "user_tz": -330, "elapsed": 3206, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3e16f052-b8f9-4bc8-c084-c87610f8a4ae"
feature_imp = pd.DataFrame(sorted(zip(m.feature_importances_, train.columns), reverse=True)[:], columns=['Value','Feature'])
plt.figure(figsize=(10,10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('LGBM Features')
plt.tight_layout()
plt.show()
```
