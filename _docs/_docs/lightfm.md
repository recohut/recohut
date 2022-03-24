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

# LightFM Hybrid Recommender

```python id="wuSExkaEnz_9"
!pip install -q implicit
!pip install -q lightfm
```

```python id="lrQJLasUKVtR"
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from scipy.sparse import csr_matrix, coo_matrix
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight, tfidf_weight

from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k

from metrics import precision_at_k as custom_precision, recall_at_k
from utils import prefilter_items

import warnings
warnings.filterwarnings('ignore')
```

```python id="gJ81tjyYK1bY"
data = pd.read_parquet('./data/bronze/train.parquet.gzip')
item_features = pd.read_parquet('./data/bronze/products.parquet.gzip')
user_features = pd.read_parquet('./data/bronze/demographics.parquet.gzip')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 224} id="BZ8F_TWsKeGP" outputId="1a6ff533-c859-4469-acaa-ae55b90a0511"
# column processing
item_features.columns = [col.lower() for col in item_features.columns]
user_features.columns = [col.lower() for col in user_features.columns]

item_features.rename(columns={'product_id': 'item_id'}, inplace=True)
user_features.rename(columns={'household_key': 'user_id'}, inplace=True)

# train test split
test_size_weeks = 3

data_train = data[data['week_no'] < data['week_no'].max() - test_size_weeks]
data_test = data[data['week_no'] >= data['week_no'].max() - test_size_weeks]

data_train.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 111} id="yr_LNtXgK3YC" outputId="55bba456-23f6-4c75-a7ee-062d86a53c8f"
result = data_test.groupby('user_id')['item_id'].unique().reset_index()
result.columns=['user_id', 'actual']
result.head(2)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 128} id="mxXurTlZK_9F" outputId="58967583-2a5a-4b41-b292-5f2c5ac7f2aa"
item_features.head(2)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 165} id="WUSjA2ohLAkK" outputId="d1e9f167-b0cc-435d-801b-e6b7a64a5d48"
user_features.head(2)
```

<!-- #region id="WOwjVTZTLA6W" -->
## Filter items
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="wNV2d1SkLXRC" outputId="d7adeecf-2c8a-4e46-f5db-e8b045fa9561"
n_items_before = data_train['item_id'].nunique()

data_train_filtered = prefilter_items(data_train, take_n_popular=5000, item_features=item_features)

n_items_after = data_train_filtered['item_id'].nunique()
print('Decreased # items from {} to {}'.format(n_items_before, n_items_after))
```

<!-- #region id="yVNmTb7oLi9R" -->
## Prepare data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="o4eF_-KdMRCV" outputId="7fd8df12-92c1-41fb-e979-969864c2fa36"
### Prepare csr train matrix
user_item_matrix = pd.pivot_table(data_train_filtered, 
                                  index='user_id', columns='item_id', 
                                  values='quantity',
                                  aggfunc='count', 
                                  fill_value=0
                                 )

user_item_matrix = user_item_matrix.astype(float)
sparse_user_item = csr_matrix(user_item_matrix).tocsr()
sparse_user_item
```

```python id="Hx9yoxi-MZ6Z"
### Prepare CSR test matrix
data_test = data_test[data_test['item_id'].isin(data_train['item_id'].unique())]

test_user_item_matrix = pd.pivot_table(data_test, 
                                  index='user_id', columns='item_id', 
                                  values='quantity',
                                  aggfunc='count', 
                                  fill_value=0
                                 )

test_user_item_matrix = test_user_item_matrix.astype(float)
userids = user_item_matrix.index.values
itemids = user_item_matrix.columns.values

matrix_userids = np.arange(len(userids))
matrix_itemids = np.arange(len(itemids))

id_to_itemid = dict(zip(matrix_itemids, itemids))
id_to_userid = dict(zip(matrix_userids, userids))

itemid_to_id = dict(zip(itemids, matrix_itemids))
userid_to_id = dict(zip(userids, matrix_userids))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 159} id="gkWX4pYeMepX" outputId="ac1809b0-eacf-4170-be7c-09d89317eb5e"
### Prepare user and item features
user_feat = pd.DataFrame(user_item_matrix.index)
user_feat = user_feat.merge(user_features, on='user_id', how='left')
user_feat.set_index('user_id', inplace=True)
user_feat.head(2)
```

```python colab={"base_uri": "https://localhost:8080/"} id="bqpp3kFwMnku" outputId="9ac49212-9170-404f-8854-38f92ad5ea98"
user_feat.shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 142} id="a6G95GugMDfQ" outputId="7d928884-4e02-434c-86ce-3add804e4fa0"
item_feat = pd.DataFrame(user_item_matrix.columns)
item_feat = item_feat.merge(item_features, on='item_id', how='left')
item_feat.set_index('item_id', inplace=True)
item_feat.head(2)
```

```python colab={"base_uri": "https://localhost:8080/"} id="JDnFW4WMMo4P" outputId="733d4500-0b77-4377-fc0f-b84b301e3922"
item_feat.shape
```

<!-- #region id="NQLJR9rcMpJ_" -->
## Encoding features
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 179} id="VJ-ZGRT-MwDu" outputId="d279eb4b-e395-4fcd-bfc1-ee24eb28d1a8"
user_feat_lightfm = pd.get_dummies(user_feat, columns=user_feat.columns.tolist())
item_feat_lightfm = pd.get_dummies(item_feat, columns=item_feat.columns.tolist())
user_feat_lightfm.head(2)
```

<!-- #region id="-axEab5gMygk" -->
## Model training
<!-- #endregion -->

```python id="9mpL1_5FM1VL"
model = LightFM(no_components=40,
                loss='bpr', # "logistic","bpr"
                learning_rate=0.01, 
                item_alpha=0.4,
                user_alpha=0.1, 
                random_state=42,
                k=5,
                n=15,
                max_sampled=100)
```

```python colab={"base_uri": "https://localhost:8080/"} id="9YghTs_iM3eA" outputId="7a13dc7f-e440-42e5-90b4-24509442cde0"
model.fit((sparse_user_item > 0) * 1,  # user-item matrix из 0 и 1
          sample_weight=coo_matrix(user_item_matrix),
          user_features=csr_matrix(user_feat_lightfm.values).tocsr(),
          item_features=csr_matrix(item_feat_lightfm.values).tocsr(),
          epochs=20, 
          num_threads=20,
          verbose=True) 
```

<!-- #region id="KsoD-362NV71" -->
## Getting embeddings
<!-- #endregion -->

<!-- #region id="jzXvubVkNW-y" -->
### Vectors by users
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="UnclCeLsNYbs" outputId="fb4858b7-5687-4df9-e615-f99531c2a91f"
user_emb = model.get_user_representations(features=csr_matrix(user_feat_lightfm.values).tocsr())
user_emb[0].shape # biases
```

```python colab={"base_uri": "https://localhost:8080/"} id="yKEPJneHNZ4m" outputId="0cd551ad-4e56-4308-eef8-8d0283767871"
user_emb[1].shape # users vectors
```

<!-- #region id="vNU5ob4PNdVG" -->
### Vectors by products
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="zaTIwuWsNcfj" outputId="8ea9ca3f-0e86-40a2-fa7e-85b6544560a8"
item_emb = model.get_item_representations(features=csr_matrix(item_feat_lightfm.values).tocsr())
item_emb[0].shape # biases
```

```python colab={"base_uri": "https://localhost:8080/"} id="ZkFSY2twNe06" outputId="c592306f-19db-43c0-e670-2728f0d4c8c3"
item_emb[1].shape # items vectors
```

<!-- #region id="fO_sNiX_Nf1x" -->
### Evaluation -> Train precision
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="MwSCEQsjNVCO" outputId="ac4ddc22-7684-4a84-cedb-7b6c8a542674"
# we can use built-in lightFM metrics
train_precision = precision_at_k(model, sparse_user_item, 
                                 user_features=csr_matrix(user_feat_lightfm.values).tocsr(),
                                 item_features=csr_matrix(item_feat_lightfm.values).tocsr(),
                                 k=5).mean()

print(f"Train precision {train_precision}")
```

<!-- #region id="Qzu-x2ZTNi9_" -->
## Predict
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="AAWrkHtzNpMD" outputId="5926cbcd-4a66-4739-e217-d62fdd368139"
# prepare id for users and products in order of user-item pairs
users_ids_row = data_train_filtered['user_id'].apply(lambda x: userid_to_id[x]).values.astype(int)
items_ids_row = data_train_filtered['item_id'].apply(lambda x: itemid_to_id[x]).values.astype(int)
users_ids_row[:10]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 224} id="OjfcSbYuNwMD" outputId="6d23c91d-bdc6-40b6-a88d-e5c6ded1689a"
# the model returns a measure / rate of similarity between the corresponding user and the product
predictions = model.predict(user_ids=users_ids_row,
                            item_ids=items_ids_row,
                            user_features=csr_matrix(user_feat_lightfm.values).tocsr(),
                            item_features=csr_matrix(item_feat_lightfm.values).tocsr(),
                            num_threads=10)

# add to the train dataframe
data_train_filtered['score'] = predictions
data_train_filtered.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="a7YB_FMuOFEo" outputId="ed16bff9-a9e9-418c-f801-3253ee041380"
# create a predicate dataframe in list format
predict_result = data_train_filtered[['user_id','item_id','score']][data_train_filtered.item_id != 999999].drop_duplicates().sort_values(by=['user_id','score'], ascending=False).groupby('user_id')['item_id']. \
            unique().reset_index()
predict_result.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="7Ll95wEHOJfJ" outputId="98b95ec6-1e37-4205-cfb4-46213469ddd2"
# combine the predicate and test dataset to calculate precision
df_result_for_metrics = result.merge(predict_result, on='user_id', how='inner')
df_result_for_metrics.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="mCpffU75M5TU" outputId="73eee0a4-cf7e-45eb-f3b6-60adc7c48dea"
# Test with custom precision func
precision = df_result_for_metrics.apply(lambda row: custom_precision(row['item_id'], row['actual'],k=5), axis=1).mean()
print(f"Precision: {precision}")
```

<!-- #region id="7L6bJapQPAtZ" -->
## Hyperparameter tuning
<!-- #endregion -->

```python id="wQMxI_wQPCQG"
# prepare id for users and products in order of user-item pairs
users_ids_row = data_train_filtered['user_id'].apply(lambda x: userid_to_id[x]).values.astype(int)
items_ids_row = data_train_filtered['item_id'].apply(lambda x: itemid_to_id[x]).values.astype(int)
```

```python id="ntaK9lOtPHZs"
params_grid = {'item_alpha': [0.2, 0.4, 0.6],
               'user_alpha': [0.1, 0.2],
               'learning_rate' : [0.01, 0.05],
               'n': [10 , 100],
               'max_sampled': [10, 100]}
```

```python id="aDms1FFBPHsK"
result_searh = {}
index = 0

for item_alp in params_grid['item_alpha']:
    for user_alp in params_grid['n']:
        for learn_rate in params_grid['learning_rate']:
            for n in params_grid['n']:
                for max_sampl in params_grid['max_sampled']:

                    model = LightFM(no_components=40,
                          loss="bpr", # "logistic","bpr"
                          learning_rate=learn_rate, 
                          item_alpha=item_alp,
                          user_alpha=user_alp, 
                          random_state=42,
                          k=5,
                          n=n,
                          max_sampled=max_sampl)
                    model.fit((sparse_user_item > 0) * 1,  # user-item matrix из 0 и 1
                    sample_weight=coo_matrix(user_item_matrix),
                    user_features=csr_matrix(user_feat_lightfm.values).tocsr(),
                    item_features=csr_matrix(item_feat_lightfm.values).tocsr(),
                    epochs=20, 
                    num_threads=20,
                    verbose=True) 

                    predictions = model.predict(user_ids=users_ids_row,
                                      item_ids=items_ids_row,
                                      user_features=csr_matrix(user_feat_lightfm.values).tocsr(),
                                      item_features=csr_matrix(item_feat_lightfm.values).tocsr(),
                                      num_threads=10)

                    data_train_filtered['score'] = predictions

                    predict_result = data_train_filtered[['user_id','item_id','score']][data_train_filtered.item_id != 999999].drop_duplicates().sort_values(by=['user_id','score'], ascending=False).groupby('user_id')['item_id']. \
                      unique().reset_index()

                    df_result_for_metrics = result.merge(predict_result, on='user_id', how='inner')

                    precision = df_result_for_metrics.apply(lambda row: custom_precision(row['item_id'], row['actual'],k=5), axis=1).mean()


                    result_searh[index] = [item_alp,
                                        user_alp,
                                        learn_rate,
                                        n,
                                        max_sampl,                                      
                                        df_result_for_metrics.apply(lambda row: custom_precision(row['item_id'], row['actual'],k=5), axis=1).mean()]
                    index += 1
```

```python colab={"base_uri": "https://localhost:8080/"} id="w72FYPr4aKHA" outputId="4ecdcbb9-c5a6-4f62-d992-29a8b2bfea7c"
list(params_grid.keys())
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="j2PZoRLPPKRc" outputId="2d901c63-9b3b-4cf1-a0f4-eb081ed2f167"
bpr_results = pd.DataFrame.from_dict(result_searh, orient='index',
                       columns=['item_alpha', 'user_alpha', 'learning_rate',
                                'n', 'max_sampled', 'score'])
bpr_results.sort_values(by='score', ascending=False).head()
```
