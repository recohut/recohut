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

<!-- #region id="6h05QWKyNs4C" -->
# Simple Recommenders on Retail Product Data
<!-- #endregion -->

<!-- #region id="J5ufegc44wEt" -->
## Setup
<!-- #endregion -->

```python id="wuSExkaEnz_9"
!pip install -q implicit
```

```python id="LYrh1wL831CB"
!git clone https://github.com/RecoHut-Stanzas/S593234
%cd S593234
```

```python id="B-ncyxeRniZV"
import sys
sys.path.insert(0, './code')
```

```python id="hRA492SKnWIF"
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from scipy.sparse import csr_matrix, coo_matrix
from sklearn.model_selection import GridSearchCV

from implicit.nearest_neighbours import *
from implicit.evaluation import * 
from implicit.evaluation import train_test_split

from metrics import precision_at_k, recall_at_k

import warnings
warnings.filterwarnings("ignore")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 224} id="E510rfm0pnF_" executionInfo={"status": "ok", "timestamp": 1628357180284, "user_tz": -330, "elapsed": 888, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="40993a65-5f04-4f04-f5cf-d1d4a40f94e6"
data = pd.read_parquet('./data/bronze/train.parquet.gzip')
data.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="oTWdilyGplQp" executionInfo={"status": "ok", "timestamp": 1628357216952, "user_tz": -330, "elapsed": 1741, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="735d6e7b-f24e-4733-b1bc-1b39dd3f97e8"
test_size_weeks = 3
data_train = data[data['week_no'] < data['week_no'].max() - test_size_weeks]
data_test = data[data['week_no'] >= data['week_no'].max() - test_size_weeks]
data_train.shape, data_test.shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="sksKsql7qr1A" executionInfo={"status": "ok", "timestamp": 1628357460908, "user_tz": -330, "elapsed": 832, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1501aad2-4c41-4770-f6be-10d91a6a109d"
result = data_test.groupby('user_id')['item_id'].unique().reset_index()
result.columns = ['user_id', 'actual']
result.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="55FOLv-3pg_s" executionInfo={"status": "ok", "timestamp": 1628357468924, "user_tz": -330, "elapsed": 617, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a9b5a9b5-ff3e-48dd-a900-8693045519d8"
test_users = result.shape[0]
new_test_users = len(set(data_test['user_id']) - set(data_train['user_id']))
print('There are {} users in test set out of which {} are new.'.format(test_users, new_test_users))
```

<!-- #region id="v3gbTRHdqhYq" -->
## Random recommender
<!-- #endregion -->

```python id="UNXW3UOCpEca"
def random_recommendation(items, n=5):
    """randomly picks n items from given list without replacement"""
    
    items = np.array(items)
    recs = np.random.choice(items, size=n, replace=False)
    
    return recs.tolist()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="Wzzf3lyHqf1i" executionInfo={"status": "ok", "timestamp": 1628357516142, "user_tz": -330, "elapsed": 5217, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cb201902-6144-4ef3-c976-51d347dc522e"
items = data_train.item_id.unique()
result['random_recommendation'] = result['user_id'].apply(lambda x: random_recommendation(items, n=5))
result.head()
```

<!-- #region id="FjFNJ9OTrYHe" -->
## Weighted random recommender
<!-- #endregion -->

```python id="dloa3aZDpSGq"
def weighted_random_recommendation(items_weights, n=5):
    """picks n items from given list in proportion of their weights"""

    recs = np.random.choice(item_weights['item_id'], p=item_weights['weight'], size=n, replace=False)
    
    return recs.tolist()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="ezfFcL8zqfx6" executionInfo={"status": "ok", "timestamp": 1628357771797, "user_tz": -330, "elapsed": 596, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="37d0e349-8c77-4b77-b2b5-746c6d9b407d"
# weight for item is equivalent to the popularity of that item
# popularity is measured by the percentage purchase of that item
item_weights = data.groupby('item_id')['user_id'].count().reset_index()
item_weights.sort_values('user_id', ascending=False, inplace=True)
total_count = item_weights['user_id'].sum()
item_weights['weight'] = item_weights['user_id'] / total_count
item_weights.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 289} id="RBWNnSwjqfuW" executionInfo={"status": "ok", "timestamp": 1628357813843, "user_tz": -330, "elapsed": 3884, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3292ef2a-22e8-4150-a36c-0c68adf5fa23"
result['weighted_random_recommendation'] = result['user_id'].apply(lambda x: weighted_random_recommendation(item_weights, n=5))
result.head()
```

<!-- #region id="dqB0FPLhqfrZ" -->
## Popularity-based recommender
<!-- #endregion -->

```python id="Uk7uMM8isg1v"
def popularity_recommendation(data, n=5):
    """Top-n popular items by their sales values"""
    
    popular = data.groupby('item_id')['sales_value'].sum().reset_index()
    popular.sort_values('sales_value', ascending=False, inplace=True)
    
    recs = popular.head(n).item_id
    
    return recs.tolist()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 374} id="vA2cGZ3Dsyhj" executionInfo={"status": "ok", "timestamp": 1628358029967, "user_tz": -330, "elapsed": 757, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2ddff8e5-488e-4d9c-b7e1-a3a3dddb4412"
popular_recs = popularity_recommendation(data_train, n=5)
result['popular_recommendation'] = result['user_id'].apply(lambda x: popular_recs)
result.head()
```

<!-- #region id="TmbBoDLts5Ks" -->
## Item-item recommender
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="rAyjzKsduMDH" executionInfo={"status": "ok", "timestamp": 1628358366183, "user_tz": -330, "elapsed": 768, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fbd5110d-241d-45e2-e97c-5c2c0938cf1a"
item_qty = data_train.groupby('item_id')['quantity'].sum().reset_index()
item_qty.rename(columns={'quantity': 'n_sold'}, inplace=True)
item_qty.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 224} id="u36tORt-uPts" executionInfo={"status": "ok", "timestamp": 1628358468920, "user_tz": -330, "elapsed": 647, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cfe564b6-e221-46dd-931e-1088881f1d65"
item_qty_top5k = item_qty.sort_values('n_sold', ascending=False).head(5000).item_id.tolist()
# Let's create a fictitious item_id (if the user bought goods from the top 5000, then he "bought" such a product)
data_train.loc[ ~ data_train['item_id'].isin(item_qty_top5k), 'item_id'] = 6666
data_train.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="fea5j8lHut5T" executionInfo={"status": "ok", "timestamp": 1628358565583, "user_tz": -330, "elapsed": 4362, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="82260a8b-485e-48d5-bb36-3c618286ec5f"
user_item_matrix = pd.pivot_table(data_train, 
                                  index='user_id', columns='item_id', 
                                  values='quantity',
                                  aggfunc='count', 
                                  fill_value=0
                                 )
user_item_matrix[user_item_matrix > 0] = 1
user_item_matrix = user_item_matrix.astype(float)
sparse_user_item = csr_matrix(user_item_matrix).tocsr()
sparse_user_item
```

```python id="QPmgpC6Wu8GE"
userids = user_item_matrix.index.values
itemids = user_item_matrix.columns.values

matrix_userids = np.arange(len(userids))
matrix_itemids = np.arange(len(itemids))

id_to_itemid = dict(zip(matrix_itemids, itemids))
id_to_userid = dict(zip(matrix_userids, userids))

itemid_to_id = dict(zip(itemids, matrix_itemids))
userid_to_id = dict(zip(userids, matrix_userids))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 66, "referenced_widgets": ["bda3d8d193ad4c55a81c8eb365ce98c5", "3d92770394ca4ef5ba396f0244f09fe3", "c2b83e3380fe4c6bb998133f05812113", "f006076cb4ae491c8a193e01428a331c", "06e24a2211a441f0a788595cdb430b90", "37798f6be4af45188948b1295d7eeef1", "cf20d5e3ff3e40b5a17ef4c22d0f4e94", "55202913025945bea1954dd0cf6e7871"]} id="hxPKSArqtBmN" executionInfo={"status": "ok", "timestamp": 1628358617962, "user_tz": -330, "elapsed": 2371, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="56dbacab-857f-4e4f-8067-0d5b8472f614"
model = ItemItemRecommender(K=5, num_threads=4)
model.fit(csr_matrix(user_item_matrix).T.tocsr(), show_progress=True)
recs = model.recommend(userid=userid_to_id[2],
                        user_items=csr_matrix(user_item_matrix).tocsr(),
                        N=5,
                        filter_already_liked_items=False, 
                        filter_items=None, 
                        recalculate_user=True)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 281} id="TmmihbX2vJ7X" executionInfo={"status": "ok", "timestamp": 1628358853999, "user_tz": -330, "elapsed": 590, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="da9d69b3-526d-41f6-953f-56f75308620a"
result['itemitem'] = result['user_id'].\
    apply(lambda x: [id_to_itemid[rec[0]] for rec in 
                    model.recommend(userid=userid_to_id[x], 
                                    user_items=sparse_user_item,
                                    N=5, 
                                    filter_already_liked_items=False, 
                                    filter_items=None, 
                                    recalculate_user=True)])
    
result.head(2)
```

<!-- #region id="C1Hl31BJvXaD" -->
## Cosine recommender
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 330, "referenced_widgets": ["1fae4f33bc0f4a90bef35864fabd5a0c", "30fe66b910a64ac68c033c0e8046165a", "5c29a0a2a85c4cad848bbea353464901", "87205fdd0b044dd1a00d82e77c2f84cd", "600e7554983e4525b285df0213e04b55", "fbfe9206ff0e4205a8c56f2e090b4b5b", "4573da9b6a1848a0a7bb2f1fe8241825", "2268ff8575a7403da84b735e1cea8dd6"]} id="yRf9BRSYvj0-" executionInfo={"status": "ok", "timestamp": 1628358913743, "user_tz": -330, "elapsed": 2872, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b992c1d4-695b-46cb-b02b-098f3b63382c"
model = CosineRecommender(K=5, num_threads=4)
model.fit(csr_matrix(user_item_matrix).T.tocsr(), show_progress=True)
recs = model.recommend(userid=userid_to_id[1], 
                        user_items=csr_matrix(user_item_matrix).tocsr(),
                        N=5, 
                        filter_already_liked_items=False, 
                        filter_items=None, 
                        recalculate_user=False)

result['cosine'] = result['user_id'].\
    apply(lambda x: [id_to_itemid[rec[0]] for rec in 
                    model.recommend(userid=userid_to_id[x], 
                                    user_items=sparse_user_item,
                                    N=5, 
                                    filter_already_liked_items=False, 
                                    filter_items=None, 
                                    recalculate_user=True)])
result.head(2)
```

<!-- #region id="W2OHuUvCw_e_" -->
## TF-IDF recommender
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 66, "referenced_widgets": ["7b7d98d117d44018a160dfae75e92095", "9c6e7e0171684806b4becded01815e66", "4e229e89d8024e9391427dcefb82fb07", "bac2c3d671ff45a5b7d2cda083453036", "f4ea5d9f27764b049c6058e6bc224dfe", "b91aa44570fc495a983322eb5e4285ee", "9994cbfc0f9d41a59f76c35c8272499c", "5095a336545b45c4bd6324b959e9be5b"]} id="rGkGz1wExBoq" executionInfo={"status": "ok", "timestamp": 1628359115415, "user_tz": -330, "elapsed": 2613, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ab0418dc-9eea-42ca-abf4-4fa113909cbb"
model = TFIDFRecommender(K=5, num_threads=4)
model.fit(csr_matrix(user_item_matrix).T.tocsr(), show_progress=True)
recs = model.recommend(userid=userid_to_id[1], 
                        user_items=csr_matrix(user_item_matrix).tocsr(),
                        N=5, 
                        filter_already_liked_items=False, 
                        filter_items=None, 
                        recalculate_user=False)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 301} id="37Mq3Qx-xE9D" executionInfo={"status": "ok", "timestamp": 1628359127169, "user_tz": -330, "elapsed": 605, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a60fcc21-7005-40fe-8a02-7a1734a37ceb"
result['tfidf'] = result['user_id'].\
    apply(lambda x: [id_to_itemid[rec[0]] for rec in 
                    model.recommend(userid=userid_to_id[x], 
                                    user_items=sparse_user_item,   # на вход user-item matrix
                                    N=5, 
                                    filter_already_liked_items=False, 
                                    filter_items=None, 
                                    recalculate_user=False)])
result.head(2)
```

<!-- #region id="k6iisdi-yCe9" -->
## Item-item recommender with filtering + (K=1)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 350, "referenced_widgets": ["663811d24e74491aa15c64564f270c0b", "8134b53d324d4f10b937d9d44f80557f", "8d4f05a77be94333bd9ef37833b72481", "c42f007e05f44a81bc8d1873b692c5ca", "b82f516d14d04a92bda42f5bb7641a25", "c2ca84a609484be69f59816cb1389990", "a48ef1c2d0244767a8a9428c28b578ac", "2ff7a2b0734c490e831d00895b59f995"]} id="Jcr9WNKoyMXM" executionInfo={"status": "ok", "timestamp": 1628359416110, "user_tz": -330, "elapsed": 1553, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3c945c0d-5a31-482d-d7eb-22217a3aa043"
model = ItemItemRecommender(K=1, num_threads=4)
model.fit(csr_matrix(user_item_matrix).T.tocsr(), show_progress=True)

result['own_purchases'] = result['user_id'].\
    apply(lambda x: [id_to_itemid[rec[0]] for rec in 
                    model.recommend(userid=userid_to_id[x], 
                                    user_items=sparse_user_item,
                                    N=5, 
                                    filter_already_liked_items=False, 
                                    filter_items=[itemid_to_id[6666]], 
                                    recalculate_user=True)])
result.head(2)
```

<!-- #region id="lGcFRCJNwSAZ" -->
## Comparison
<!-- #endregion -->

```python id="hhWZczvnypwd"
results_precision_at_k = {}
results_recall_at_k = {}

for name_col in result.columns[1:]:
    results_precision_at_k[name_col] = round(result.apply(lambda row: precision_at_k(row[name_col], row['actual']), axis=1).mean(),4)
    results_recall_at_k[name_col] = round(result.apply(lambda row: recall_at_k(row[name_col], row['actual']), axis=1).mean(),4)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 297} id="9EYNcI1LzacK" executionInfo={"status": "ok", "timestamp": 1628360233795, "user_tz": -330, "elapsed": 699, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7839a98c-4ecc-4a12-eada-268a58b648dc"
results_precision_at_k = pd.DataFrame.from_dict(results_precision_at_k, orient='index', columns=['precision_at_k'])
results_recall_at_k = pd.DataFrame.from_dict(results_recall_at_k, orient='index', columns=['recall_at_k'])
pd.concat([results_precision_at_k, results_recall_at_k], axis=1).sort_values(by='precision_at_k', ascending=False)
```

<!-- #region id="Y2Rgd6df46yx" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="IMdtXnU_46yz" executionInfo={"status": "ok", "timestamp": 1638712746135, "user_tz": -330, "elapsed": 3583, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="29877176-23b7-4964-c655-3e92d7d907fb"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d -p implicit
```

<!-- #region id="JWllKC8846y0" -->
---
<!-- #endregion -->

<!-- #region id="sLRWrVYL46y2" -->
**END**
<!-- #endregion -->
