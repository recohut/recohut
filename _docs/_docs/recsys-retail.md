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

<!-- #region id="8ByI1KspixsF" -->
# Various Recommender models on Retail data
<!-- #endregion -->

<!-- #region id="h5ZpupY2cA_V" -->
## Environment Setup
<!-- #endregion -->

```python id="rtPR102pblgZ"
import tensorflow as tf

## loading packages
import sys
import random
import datetime
import numpy as np
import pandas as pd
from math import ceil
from tqdm import trange
from subprocess import call
from itertools import islice
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, dok_matrix
from sklearn.model_selection import ParameterGrid

import matplotlib.pyplot as plt
import seaborn as sns

import math
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
import heapq ## for retrieval topK
import multiprocessing
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime
from pprint import pprint 
from time import time
from scipy.sparse.linalg import svds, eigs

from functools import wraps
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    roc_auc_score,
    log_loss,
)

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import os
import pandas as pd
import scipy.sparse
import time
import sys

from fastai.collab import *
from fastai.tabular import *
from fastai.text import *

!pip install git+https://github.com/maciejkula/spotlight.git@master#egg=spotlight

!git clone https://github.com/microsoft/recommenders.git
sys.path.append('/content/recommenders/')
```

```python id="7_V3zcOJb038" outputId="de8a67db-032c-43e5-a9fd-527ad558a9b6" executionInfo={"status": "ok", "timestamp": 1584883194792, "user_tz": -330, "elapsed": 1860, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "05005557606489372709"}} colab={"base_uri": "https://localhost:8080/", "height": 204}
## loading data
file_path =  '/content/drive/My Drive/Recommendation/'
file_name = 'rawdata.csv'
df = pd.read_csv(file_path+file_name, header = 0,
                 names = ['event','userid','itemid','timestamp'],
                 dtype={0:'category', 1:'category', 2:'category'},
                 parse_dates=['timestamp'])

## dropping exact duplicates
## df = df.drop_duplicates()

## userid normalization
userid_encoder = preprocessing.LabelEncoder()
df.userid = userid_encoder.fit_transform(df.userid)

## itemid normalization
itemid_encoder = preprocessing.LabelEncoder()
df.itemid = itemid_encoder.fit_transform(df.itemid)

df.head()
```

```python id="iu4lbjr1FKNv"
df.info()
df.shape[0]/df.userid.nunique()
df.describe().T
df.describe(exclude='int').T
df.userid.cat.codes
df.event.value_counts()/df.userid.nunique()
df.timestamp.max() - df.timestamp.min()
```

```python id="A1AdzL2mzB4p"
grouped_df = df.groupby(['userid', 'itemid'])['event'].sum().reset_index()
```

<!-- #region id="m91Oa1bAOKYo" -->
## Data Transformation
- Count
- Weighted Count
- Time dependent Count
- Negative Sampling
<!-- #endregion -->

<!-- #region id="krV7TuzcOllV" -->
### A. Count
<!-- #endregion -->

```python id="FpDCdi71OlDe" outputId="bf341044-a73f-4f69-f0c7-30afeb584be3" executionInfo={"status": "ok", "timestamp": 1584881169137, "user_tz": -330, "elapsed": 22556, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "05005557606489372709"}} colab={"base_uri": "https://localhost:8080/", "height": 204}
data_count = df.groupby(['userid', 'itemid']).agg({'timestamp': 'count'}).reset_index()
data_count.columns = ['userid', 'itemid', 'affinity']
data_count.head()
```

<!-- #region id="lnmAYVKKPFl6" -->
### B. Weighted Count
<!-- #endregion -->

```python id="fQhgPg6SOlB4"
data_w = df.loc[df.event!='remove_from_cart',:]
```

```python id="TxnAy08HOk_d"
affinity_weights = {
    'view_item': 1,
    'add_to_cart': 3, 
    'begin_checkout': 5, 
    'purchase': 6,
    'remove_from_cart': 3
}
```

```python id="oo7K3EggOk87"
data_w['weight'] = data_w['event'].apply(lambda x: affinity_weights[x])
```

```python id="aUee5ctfNHQm" outputId="e724c345-0a47-40b7-ef9a-a0d0d0176211" executionInfo={"status": "ok", "timestamp": 1584881169140, "user_tz": -330, "elapsed": 21165, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "05005557606489372709"}} colab={"base_uri": "https://localhost:8080/", "height": 204}
data_wcount = data_w.groupby(['userid', 'itemid'])['weight'].sum().reset_index()
data_wcount.columns = ['userid', 'itemid', 'affinity']
data_wcount.head()
```

<!-- #region id="v6I8VR3nQ3kp" -->
### C. Time dependent Count
<!-- #endregion -->

```python id="5UDJpjL4Q1BP"
T = 30
t_ref = datetime.utcnow()
```

```python id="vIjR5hIiRCTT"
data_w['timedecay'] = data_w.apply(
    lambda x: x['weight'] * math.exp(-math.log2((t_ref - pd.to_datetime(x['timestamp']).tz_convert(None)).days / T)), 
    axis=1
)
```

```python id="HmUPODxsRCSJ" outputId="8c34b11b-96ce-4fae-f972-e35ea22255b8" executionInfo={"status": "ok", "timestamp": 1584881182761, "user_tz": -330, "elapsed": 33105, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "05005557606489372709"}} colab={"base_uri": "https://localhost:8080/", "height": 204}
data_w.head()
```

```python id="ZiOgHFkgRCRI" outputId="aec7657e-fee6-4e83-db7c-652e8f0728bb" executionInfo={"status": "ok", "timestamp": 1584881182761, "user_tz": -330, "elapsed": 32922, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "05005557606489372709"}} colab={"base_uri": "https://localhost:8080/", "height": 204}
data_wt = data_w.groupby(['userid', 'itemid'])['timedecay'].sum().reset_index()
data_wt.columns = ['userid', 'itemid', 'affinity']
data_wt.head()
```

<!-- #region id="I9_Ul8u4UB1q" -->
## Negative Sampling
<!-- #endregion -->

```python id="Rf24M-2CUE3o"
data_b = df[['userid', 'itemid']].copy()
data_b['feedback'] = 1
data_b = data_b.drop_duplicates()
```

```python id="HhFCSINPUKi6"
users = df['userid'].unique()
items = df['itemid'].unique()
```

```python id="VXWcyesRUKf6"
interaction_lst = []
for user in users:
    for item in items:
        interaction_lst.append([user, item, 0])

data_all = pd.DataFrame(data=interaction_lst, columns=["userid", "itemid", "feedbackAll"])
```

```python id="LYh1dptDUhsN" outputId="777e40e4-ef49-4318-f581-0bf4af398444" executionInfo={"status": "ok", "timestamp": 1584865151631, "user_tz": -330, "elapsed": 9911, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "05005557606489372709"}} colab={"base_uri": "https://localhost:8080/", "height": 204}
data_ns = pd.merge(data_all, data_b, on=['userid', 'itemid'], how='outer').fillna(0).drop('feedbackAll', axis=1)
data_ns.head()
```

<!-- #region id="prQONRV5Q2FS" -->
## Other
<!-- #endregion -->

```python id="Bi3W1krjOPK2" outputId="682c97f2-eceb-4af2-e549-33fefaf0fe23" executionInfo={"status": "ok", "timestamp": 1584861740506, "user_tz": -330, "elapsed": 737, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "05005557606489372709"}} colab={"base_uri": "https://localhost:8080/", "height": 204}
dfx = df[['userid','itemid','eventStrength','timestamp']]
dfx.head()
```

<!-- #region id="EqukSqbz6sXK" -->
## Train Test Split
<!-- #endregion -->

```python id="jjAZKENcXCSV"
data = data_w[['userid','itemid','timedecay','timestamp']]
```

```python id="WYcIOEK0Xq6e"
col = {
  'col_user': 'userid',
  'col_item': 'itemid',
  'col_rating': 'timedecay',
  'col_timestamp': 'timestamp',
}

col3 = {
  'col_user': 'userid',
  'col_item': 'itemid',
  'col_timestamp': 'timestamp',
}
```

```python id="rIGrXXykAtqx"
from reco_utils.dataset.python_splitters import python_chrono_split
train, test = python_chrono_split(data, ratio=0.75, min_rating=10, 
                                  filter_by='user', **col3)
```

```python id="A_xMyKw9IA4F" outputId="7040bc26-29c2-42b1-e467-8c7c5454dc72" executionInfo={"status": "ok", "timestamp": 1584881754461, "user_tz": -330, "elapsed": 6254, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "05005557606489372709"}} colab={"base_uri": "https://localhost:8080/", "height": 204}
train.head()
```

```python id="zjpp0UsaHy-X" outputId="7cbfe9d9-282b-4739-95a4-d2cbbe9554a6" executionInfo={"status": "ok", "timestamp": 1584881754461, "user_tz": -330, "elapsed": 4625, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "05005557606489372709"}} colab={"base_uri": "https://localhost:8080/", "height": 297}
train.loc[train.userid==7,:]
```

```python id="yPDMEf0SIZcb" outputId="59a663a4-001f-44f5-a093-489a8813c69a" executionInfo={"status": "ok", "timestamp": 1584881754462, "user_tz": -330, "elapsed": 4449, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "05005557606489372709"}} colab={"base_uri": "https://localhost:8080/", "height": 111}
test.loc[test.userid==7,:]
```

<!-- #region id="gpHpUY40YlaN" -->
## Baseline
<!-- #endregion -->

```python id="xMSr6rZkYnOa" outputId="db12caef-5bc2-4752-a657-fd92229fe763" executionInfo={"status": "ok", "timestamp": 1584869623288, "user_tz": -330, "elapsed": 1370, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "05005557606489372709"}} colab={"base_uri": "https://localhost:8080/", "height": 204}
## Recommending the most popular items is intuitive and simple approach
item_counts = train['itemid'].value_counts().to_frame().reset_index()
item_counts.columns = ['itemid', 'count']
item_counts.head()
```

```python id="SBIh1h85l01P" outputId="e41e90c4-2f8f-4df9-aecd-3f044d8882ad" executionInfo={"status": "ok", "timestamp": 1584869748446, "user_tz": -330, "elapsed": 7828, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "05005557606489372709"}} colab={"base_uri": "https://localhost:8080/", "height": 51}
user_item_col = ['userid', 'itemid']

## Cross join users and items
test_users = test['userid'].unique()
user_item_list = list(itertools.product(test_users, item_counts['itemid']))
users_items = pd.DataFrame(user_item_list, columns=user_item_col)

print("Number of user-item pairs:", len(users_items))

## Remove seen items (items in the train set) as we will not recommend those again to the users
from reco_utils.dataset.pandas_df_utils import filter_by
users_items_remove_seen = filter_by(users_items, train, user_item_col)

print("After remove seen items:", len(users_items_remove_seen))
```

```python id="utLRr7iWl0xr" outputId="cd668023-94f9-4302-e20c-afba1a4db961" executionInfo={"status": "ok", "timestamp": 1584869782826, "user_tz": -330, "elapsed": 1359, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "05005557606489372709"}} colab={"base_uri": "https://localhost:8080/", "height": 204}
## Generate recommendations
baseline_recommendations = pd.merge(item_counts, users_items_remove_seen, 
                                    on=['itemid'], how='inner')
baseline_recommendations.head()
```

```python id="gzREQqueL3Y0"
from reco_utils.evaluation.python_evaluation import map_at_k
from reco_utils.evaluation.python_evaluation import precision_at_k
from reco_utils.evaluation.python_evaluation import ndcg_at_k 
from reco_utils.evaluation.python_evaluation import recall_at_k
from reco_utils.evaluation.python_evaluation import get_top_k_items
```

```python id="MhU9enormcBH" outputId="ab267b7b-8176-49ad-d415-0490aad4bc01" executionInfo={"status": "ok", "timestamp": 1584869959664, "user_tz": -330, "elapsed": 20050, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "05005557606489372709"}} colab={"base_uri": "https://localhost:8080/", "height": 85}
k = 10

cols = {
  'col_user': 'userid',
  'col_item': 'itemid',
  'col_rating': 'timedecay',
  'col_prediction': 'count',
}

eval_map = map_at_k(test, baseline_recommendations, k=k, **cols)
eval_ndcg = ndcg_at_k(test, baseline_recommendations, k=k, **cols)
eval_precision = precision_at_k(test, baseline_recommendations, k=k, **cols)
eval_recall = recall_at_k(test, baseline_recommendations, k=k, **cols)

print("MAP:\t%f" % eval_map,
      "NDCG@K:\t%f" % eval_ndcg,
      "Precision@K:\t%f" % eval_precision,
      "Recall@K:\t%f" % eval_recall, sep='\n')
```

```python id="uwjJcikzoKwd"
from reco_utils.common.notebook_utils import is_jupyter
if is_jupyter():
    ## Record results with papermill for unit-tests
    import papermill as pm
    pm.record("map", eval_map)
    pm.record("ndcg", eval_ndcg)
    pm.record("precision", eval_precision)
    pm.record("recall", eval_recall)
```

<!-- #region id="ozvrFi3coZh4" -->
## Model 1 - BPR
<!-- #endregion -->

```python id="YnqZyl1iocI7" outputId="55cbc554-213b-4dd4-b8ca-0cc679b1154d" executionInfo={"status": "ok", "timestamp": 1584872165670, "user_tz": -330, "elapsed": 7936, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "05005557606489372709"}} colab={"base_uri": "https://localhost:8080/", "height": 173}
!pip install cornac
import cornac
from reco_utils.recommender.cornac.cornac_utils import predict_ranking
```

```python id="qRaIWD49rrGT" outputId="cc407182-cb91-4b90-c0c5-b87f9a3a91c1" executionInfo={"status": "ok", "timestamp": 1584872165674, "user_tz": -330, "elapsed": 4989, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "05005557606489372709"}} colab={"base_uri": "https://localhost:8080/", "height": 204}
train.head()
```

```python id="rKjdmDjDocG_"
TOP_K = 10
NUM_FACTORS = 200
NUM_EPOCHS = 100
SEED = 40
```

```python id="Hek3_jiKocEr" outputId="df3e2f8b-3447-4bde-ccef-e459d6a6a746" executionInfo={"status": "ok", "timestamp": 1584872228847, "user_tz": -330, "elapsed": 1483, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "05005557606489372709"}} colab={"base_uri": "https://localhost:8080/", "height": 51}
train_set = cornac.data.Dataset.from_uir(train.itertuples(index=False), seed=SEED)
print('Number of users: {}'.format(train_set.num_users))
print('Number of items: {}'.format(train_set.num_items))
```

```python id="Kkrtf84PocC7"
bpr = cornac.models.BPR(
    k=NUM_FACTORS,
    max_iter=NUM_EPOCHS,
    learning_rate=0.01,
    lambda_reg=0.001,
    verbose=True,
    seed=SEED
)
```

```python id="KQBN6vlhocB7" outputId="4afd30a9-0cf5-4e4e-ba71-04d4c0b5dc13" executionInfo={"status": "ok", "timestamp": 1584872282631, "user_tz": -330, "elapsed": 3183, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "05005557606489372709"}} colab={"base_uri": "https://localhost:8080/", "height": 68}
from reco_utils.common.timer import Timer
with Timer() as t:
    bpr.fit(train_set)
print("Took {} seconds for training.".format(t))
```

```python id="ULm4veZlob-a" outputId="5abd4cda-c0a6-4803-e896-04d69a30d4e3" executionInfo={"status": "ok", "timestamp": 1584872322420, "user_tz": -330, "elapsed": 6582, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "05005557606489372709"}} colab={"base_uri": "https://localhost:8080/", "height": 34}
with Timer() as t:
    all_predictions = predict_ranking(bpr, train, usercol='userid', itemcol='itemid', remove_seen=True)
print("Took {} seconds for prediction.".format(t))
```

```python id="wGjuWSufob7v" outputId="61b69044-bb06-4dfb-a765-3313942dee0e" executionInfo={"status": "ok", "timestamp": 1584872343749, "user_tz": -330, "elapsed": 3575, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "05005557606489372709"}} colab={"base_uri": "https://localhost:8080/", "height": 204}
all_predictions.head()
```

```python id="U2cc5FBFwMC9" outputId="25e18fb5-e965-4e9b-d2f4-30e768ae6547" executionInfo={"status": "ok", "timestamp": 1584872512135, "user_tz": -330, "elapsed": 21429, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "05005557606489372709"}} colab={"base_uri": "https://localhost:8080/", "height": 85}
k = 10
cols = {
  'col_user': 'userid',
  'col_item': 'itemid',
  'col_rating': 'timedecay',
  'col_prediction': 'prediction',
}

eval_map = map_at_k(test, all_predictions, k=k, **cols)
eval_ndcg = ndcg_at_k(test, all_predictions, k=k, **cols)
eval_precision = precision_at_k(test, all_predictions, k=k, **cols)
eval_recall = recall_at_k(test, all_predictions, k=k, **cols)

print("MAP:\t%f" % eval_map,
      "NDCG:\t%f" % eval_ndcg,
      "Precision@K:\t%f" % eval_precision,
      "Recall@K:\t%f" % eval_recall, sep='\n')
```

```python id="gA60Ljx6wMAF"
## Record results with papermill for tests
pm.record("map", eval_map)
pm.record("ndcg", eval_ndcg)
pm.record("precision", eval_precision)
pm.record("recall", eval_recall)
```

<!-- #region id="RsoB5-ESIr8G" -->
## NCF
<!-- #endregion -->

```python id="vS7dfL1G70Zo"
TOP_K = 10
EPOCHS = 20
BATCH_SIZE = 256
SEED = 42
```

```python id="KUP57FLo9ALS"
from reco_utils.recommender.ncf.ncf_singlenode import NCF
from reco_utils.recommender.ncf.dataset import Dataset as NCFDataset
```

```python id="jbIY5B27Aych"
cols = {
  'col_user': 'userid',
  'col_item': 'itemid',
  'col_rating': 'timedecay',
  'col_timestamp': 'timestamp',
}

data = NCFDataset(train=train, test=test, seed=SEED, **cols)
```

```python id="2eAMDb0eCFlW"
model = NCF (
    n_users=data.n_users, 
    n_items=data.n_items,
    model_type="NeuMF",
    n_factors=4,
    layer_sizes=[16,8,4],
    n_epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=1e-3,
    verbose=10,
    seed=SEED
)
```

```python id="1xaF_4P6C_SD" outputId="a936e367-9acc-4fbd-b9c5-f90a8820aa29" executionInfo={"status": "ok", "timestamp": 1584872828479, "user_tz": -330, "elapsed": 112835, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "05005557606489372709"}} colab={"base_uri": "https://localhost:8080/", "height": 34}
with Timer() as t:
    model.fit(data)
print("Took {} seconds for training.".format(t))
```

```python id="EXJpKqRnDGDw"
users, items, preds = [], [], []
item = list(train.itemid.unique())
for user in train.userid.unique():
    user = [user] * len(item) 
    users.extend(user)
    items.extend(item)
    preds.extend(list(model.predict(user, item, is_list=True)))
all_predictions = pd.DataFrame(data={'userid': users, 'itemid':items, "prediction":preds})
merged = pd.merge(train, all_predictions, on=['userid','itemid'], how="outer")
all_predictions = merged[merged[col['col_rating']].isnull()].drop(col['col_rating'], axis=1)
```

```python id="qBK5GXJQDnbU" outputId="b209bf00-f5ce-43ca-f7c5-6604acabd0e6" executionInfo={"status": "ok", "timestamp": 1584873055819, "user_tz": -330, "elapsed": 53772, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "05005557606489372709"}} colab={"base_uri": "https://localhost:8080/", "height": 68}
k = 10
cols = {
  'col_user': 'userid',
  'col_item': 'itemid',
  'col_rating': 'timedecay',
  'col_prediction': 'prediction',
}

eval_map = map_at_k(test, all_predictions, k=k, **cols)
eval_ndcg = ndcg_at_k(test, all_predictions, k=k, **cols)
eval_precision = precision_at_k(test, all_predictions, k=k, **cols)
eval_recall = recall_at_k(test, all_predictions, k=k, **cols)

print("NDCG:\t%f" % eval_ndcg,
      "Precision@K:\t%f" % eval_precision,
      "Recall@K:\t%f" % eval_recall, sep='\n')
```

<!-- #region id="AvOPjrjO0Bk2" -->
## Model - SARS
<!-- #endregion -->

```python id="JvnyQXhEMLoQ"
from reco_utils.recommender.sar.sar_singlenode import SARSingleNode
```

```python id="KvTibDUN0Emf"
TOP_K = 10
```

```python id="8PuV4_D_0Ezr"
header = {
    "col_user": "userid",
    "col_item": "itemid",
    "col_rating": "timedecay",
    "col_timestamp": "timestamp",
    "col_prediction": "prediction",
}
```

```python id="JJPQflyW1t8b"
SARSingleNode?
```

```python id="2cVRTg1z0Tgj"
model = SARSingleNode(
    similarity_type="jaccard", 
    time_decay_coefficient=0, 
    time_now=None, 
    timedecay_formula=False, 
    **header
)
```

```python id="sfkFkxj71kdc"
model.fit(train)
```

```python id="Zqe79vek1mvl" outputId="b9c59ebe-3501-4170-db9d-1ff702bc53e8" executionInfo={"status": "ok", "timestamp": 1584873998445, "user_tz": -330, "elapsed": 16090, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "05005557606489372709"}} colab={"base_uri": "https://localhost:8080/", "height": 119}
## all ranking metrics have the same arguments
args = [test, top_k]
kwargs = dict(col_user='userid', 
              col_item='itemid', 
              col_rating='timedecay', 
              col_prediction='prediction', 
              relevancy_method='top_k', 
              k=TOP_K)

eval_map = map_at_k(*args, **kwargs)
eval_ndcg = ndcg_at_k(*args, **kwargs)
eval_precision = precision_at_k(*args, **kwargs)
eval_recall = recall_at_k(*args, **kwargs)

print(f"Model:",
      f"Top K:\t\t {TOP_K}",
      f"MAP:\t\t {eval_map:f}",
      f"NDCG:\t\t {eval_ndcg:f}",
      f"Precision@K:\t {eval_precision:f}",
      f"Recall@K:\t {eval_recall:f}", sep='\n')
```

```python id="Fw_7auE16ZB8"
## Instantiate the recommender models to be compared
gmf = cornac.models.GMF(
    num_factors=8,
    num_epochs=10,
    learner="adam",
    batch_size=256,
    lr=0.001,
    num_neg=50,
    seed=123,
)
mlp = cornac.models.MLP(
    layers=[64, 32, 16, 8],
    act_fn="tanh",
    learner="adam",
    num_epochs=10,
    batch_size=256,
    lr=0.001,
    num_neg=50,
    seed=123,
)
neumf1 = cornac.models.NeuMF(
    num_factors=8,
    layers=[64, 32, 16, 8],
    act_fn="tanh",
    learner="adam",
    num_epochs=10,
    batch_size=256,
    lr=0.001,
    num_neg=50,
    seed=123,
)
neumf2 = cornac.models.NeuMF(
    name="NeuMF_pretrained",
    learner="adam",
    num_epochs=10,
    batch_size=256,
    lr=0.001,
    num_neg=50,
    seed=123,
    num_factors=gmf.num_factors,
    layers=mlp.layers,
    act_fn=mlp.act_fn,
).pretrain(gmf, mlp)
```

```python id="G6kf9FNh6Z19" outputId="c4b6137b-feb6-4964-cef7-c3013c32c066" executionInfo={"status": "ok", "timestamp": 1584875543262, "user_tz": -330, "elapsed": 156795, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "05005557606489372709"}} colab={"base_uri": "https://localhost:8080/", "height": 51}
with Timer() as t:
    gmf.fit(train_set)
print("Took {} seconds for training.".format(t))
```

```python id="Ra0saVNv6dpq" outputId="e28efb95-5892-45b5-a688-4734a32c6487" executionInfo={"status": "ok", "timestamp": 1584875693690, "user_tz": -330, "elapsed": 306845, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "05005557606489372709"}} colab={"base_uri": "https://localhost:8080/", "height": 51}
with Timer() as t:
    mlp.fit(train_set)
print("Took {} seconds for training.".format(t))
```

```python id="54lrFJaD7tV9" outputId="f282ba4f-6034-4bf3-864e-c7c2248f42bc" executionInfo={"status": "ok", "timestamp": 1584875844418, "user_tz": -330, "elapsed": 438466, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "05005557606489372709"}} colab={"base_uri": "https://localhost:8080/", "height": 51}
with Timer() as t:
    neumf1.fit(train_set)
print("Took {} seconds for training.".format(t))
```

```python id="0nl0C1iq7t2A" outputId="34ec2586-83b2-4566-8520-78b602575c34" executionInfo={"status": "ok", "timestamp": 1584876151423, "user_tz": -330, "elapsed": 160057, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "05005557606489372709"}} colab={"base_uri": "https://localhost:8080/", "height": 51}
with Timer() as t:
    neumf2.fit(train_set)
print("Took {} seconds for training.".format(t))
```

```python id="Dg7Whepk74NG"
def rec_eval(model):
  with Timer() as t:
      all_predictions = predict_ranking(model, train, usercol='userid', itemcol='itemid', remove_seen=True)

  k = 10
  cols = {
    'col_user': 'userid',
    'col_item': 'itemid',
    'col_rating': 'timedecay',
    'col_prediction': 'prediction',
  }

  eval_map = map_at_k(test, all_predictions, k=k, **cols)
  eval_ndcg = ndcg_at_k(test, all_predictions, k=k, **cols)
  eval_precision = precision_at_k(test, all_predictions, k=k, **cols)
  eval_recall = recall_at_k(test, all_predictions, k=k, **cols)

  print("MAP:\t%f" % eval_map,
        "NDCG:\t%f" % eval_ndcg,
        "Precision@K:\t%f" % eval_precision,
        "Recall@K:\t%f" % eval_recall, sep='\n')
```

```python id="5rKo3NWc-P0V" outputId="0efde74f-e136-44f6-d45e-783eec78794a" executionInfo={"status": "ok", "timestamp": 1584876259241, "user_tz": -330, "elapsed": 100414, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "05005557606489372709"}} colab={"base_uri": "https://localhost:8080/", "height": 357}
rec_eval(gmf);
rec_eval(mlp);
rec_eval(neumf1);
rec_eval(neumf2);
```

<!-- #region id="fX_C0MrxCG_V" -->
## DeepRec Ranking Models
<!-- #endregion -->

```python id="978lFfbD_p-H"
!git clone https://github.com/cheungdaven/DeepRec.git
sys.path.append('/content/DeepRec/')
## sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
```

```python id="1zlili2xCQKd"
## from models.item_ranking.cdae import ICDAE
from models.item_ranking.bprmf import BPRMF
from models.item_ranking.cml import CML
from models.item_ranking.neumf import NeuMF
from models.item_ranking.gmf import GMF
from models.item_ranking.jrl import JRL
from models.item_ranking.mlp import MLP
from models.item_ranking.lrml import LRML
```

```python id="B5lT1SYvC8MQ"
epochs = 10
num_factors = 10
display_step = 1000
batch_size = 256
learning_rate = 1e-3
reg_rate = 0.1
```

```python id="2d216f-sDqqX"
try:
  gpus = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_memory_growth(gpus[0], True)
except:
  pass
```

```python id="XJ1FUZRADsaS"
n_users = df.userid.unique().shape[0]
n_items = df.itemid.unique().shape[0]

train_row = []
train_col = []
train_rating = []
for line in train.itertuples():
    train_row.append(line[1])
    train_col.append(line[2])
    train_rating.append(line[3])   
train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))

test_row = []
test_col = []
test_rating = []
for line in test.itertuples():
    test_row.append(line[1])
    test_col.append(line[2])
    test_rating.append(line[3])
test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))

test_dict = {}
for u in range(n_users):
    test_dict[u] = test_matrix.getrow(u).nonzero()[1]
```

```python id="EoxRDaGYG0vF"
train_data, test_data, n_user, n_item = train_matrix.todok(), test_dict, n_users, n_items
```

```python id="d4d-FIVdM9Hk"
model = GMF(n_user, n_item)
model.build_network()
model.execute(train_data, test_data)
```

```python id="VKdk5zD6M9Br"
model = JRL(n_user, n_item)
model.build_network()
model.execute(train_data, test_data)
```

```python id="sroc_CCoM9Ah"
model = MLP(n_user, n_item)
model.build_network()
model.execute(train_data, test_data)
```

```python id="mmrOX5pmSVX8"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
```

```python id="EuC208S-SZrK" outputId="c1b79f85-c4e7-4a78-8000-a01fe057c548" executionInfo={"status": "ok", "timestamp": 1584881891758, "user_tz": -330, "elapsed": 32687, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "05005557606489372709"}} colab={"base_uri": "https://localhost:8080/", "height": 768}
with tf.Session(config=config) as sess:
  train_data, test_data, n_user, n_item = train_matrix.todok(), test_dict, n_users, n_items
  model = LRML(sess, n_user, n_item, epoch=epochs, batch_size=batch_size)
  model.build_network()
  model.execute(train_data, test_data)
```
