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

<!-- #region id="mZyTTTRg5Nnt" -->
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
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight, tfidf_weight

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from metrics import precision_at_k, recall_at_k

import warnings
warnings.filterwarnings('ignore')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 224} id="E510rfm0pnF_" executionInfo={"status": "ok", "timestamp": 1628362298842, "user_tz": -330, "elapsed": 939, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f8d9bf4a-f443-4905-f3ab-66d729ebc987"
data = pd.read_parquet('./data/bronze/transactions.parquet.gzip')
data.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 379} id="7QAs-z296tj8" executionInfo={"status": "ok", "timestamp": 1628362301059, "user_tz": -330, "elapsed": 675, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="486d126a-5d13-4701-8f84-d82bffdf657d"
data.columns = [col.lower() for col in data.columns]
data.rename(columns={'household_key': 'user_id',
                    'product_id': 'item_id'},
           inplace=True)


test_size_weeks = 3

data_train = data[data['week_no'] < data['week_no'].max() - test_size_weeks]
data_test = data[data['week_no'] >= data['week_no'].max() - test_size_weeks]

data_train.head(10)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 221} id="5B2YMa0p6xWz" executionInfo={"status": "ok", "timestamp": 1628362301767, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="dfe30238-a95d-44b3-b2ff-5b1770e27aa3"
item_features = pd.read_parquet('./data/bronze/products.parquet.gzip')
item_features.columns = [col.lower() for col in item_features.columns]
item_features.rename(columns={'product_id': 'item_id'}, inplace=True)
item_features.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="NMpo4GXP63lg" executionInfo={"status": "ok", "timestamp": 1628361705371, "user_tz": -330, "elapsed": 734, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="998e04be-98b1-43a4-b3b0-c939f18c9fe0"
item_features.department.unique()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="Feo6Scst68E0" executionInfo={"status": "ok", "timestamp": 1628362305195, "user_tz": -330, "elapsed": 523, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b6d1bf5a-c65d-426c-e9ef-f7477f7da512"
result = data_test.groupby('user_id')['item_id'].unique().reset_index()
result.columns=['user_id', 'actual']
result.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="GCjxi-yz7FnC" executionInfo={"status": "ok", "timestamp": 1628362310227, "user_tz": -330, "elapsed": 4206, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0fd95a5f-9bfb-4cc7-95b8-000fb87835ed"
popularity = data_train.groupby('item_id')['quantity'].sum().reset_index()
popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

top_5000 = popularity.sort_values('n_sold', ascending=False).head(5000).item_id.tolist()
data_train.loc[~data_train['item_id'].isin(top_5000), 'item_id'] = 999_999

user_item_matrix = pd.pivot_table(data_train, 
                                  index='user_id', columns='item_id', 
                                  values='quantity',
                                  aggfunc='count', 
                                  fill_value=0
                                 )
user_item_matrix = user_item_matrix.astype(float)
sparse_user_item = csr_matrix(user_item_matrix)
sparse_user_item
```

```python id="T6QBRcMg7WNe"
userids = user_item_matrix.index.values
itemids = user_item_matrix.columns.values

matrix_userids = np.arange(len(userids))
matrix_itemids = np.arange(len(itemids))

id_to_itemid = dict(zip(matrix_itemids, itemids))
id_to_userid = dict(zip(matrix_userids, userids))

itemid_to_id = dict(zip(itemids, matrix_itemids))
userid_to_id = dict(zip(userids, matrix_userids))
```

<!-- #region id="DgnQs5tT7hvX" -->
## ALS
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 103, "referenced_widgets": ["0342bfbb40164ac4ac8f9a3a1d7e29ad", "595b94e97f11420a86a0f837296f90b5", "33ea1852f5e449dcbc7e5e35bba29a9f", "7fc8e1ab9ab44c3ebed0e3a767b05e4f", "8cb51d3c2d9b4c9982556fdb20751956", "1a845dad93da408581e5fc29fbef2b11", "d17572a19e784e8b9077d8d59fe6740c", "a8231c3cc310418c8153423ad9afe8d0"]} id="CmMG1Ry77igl" executionInfo={"status": "ok", "timestamp": 1628362343246, "user_tz": -330, "elapsed": 24243, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="81b17d14-ba18-4eac-ba59-9d1e9adcc075"
model = AlternatingLeastSquares(factors=64, 
                                regularization=0.05,
                                iterations=15,  
                                calculate_training_loss=True,
                                use_gpu=False)
model.fit(csr_matrix(user_item_matrix).T.tocsr(), show_progress=True)
recs = model.recommend(userid=userid_to_id[2], 
                        user_items=csr_matrix(user_item_matrix).tocsr(),
                        N=5,
                        filter_already_liked_items=False, 
                        filter_items=None, 
                        recalculate_user=True)
```

```python id="vSb6au6b707o"
def get_recommendations(user, model, N=5):
    res = [id_to_itemid[rec[0]] for rec in 
                    model.recommend(userid=userid_to_id[user], 
                                    user_items=sparse_user_item,
                                    N=N, 
                                    filter_already_liked_items=False, 
                                    filter_items=None, 
                                    recalculate_user=True)]
    return res
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="8ig5Dtc8703f" executionInfo={"status": "ok", "timestamp": 1628362370289, "user_tz": -330, "elapsed": 27060, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d5808c36-8b2b-4ad9-b94c-40f3dce2f3dd"
result['als'] = result['user_id'].apply(lambda x: get_recommendations(x, model=model, N=5))
result.apply(lambda row: precision_at_k(row['als'], row['actual']), axis=1).mean()
result.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="p6ycg2XQ8Tg0" executionInfo={"status": "ok", "timestamp": 1628362370298, "user_tz": -330, "elapsed": 45, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f687c776-ed9f-4154-b995-cd549f986af1"
model.item_factors.shape, model.user_factors.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="Xi50he-08gaE" executionInfo={"status": "ok", "timestamp": 1628362403845, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2bbd04b9-b7db-4162-e706-3fc229ae0fbd"
%%time
# we can calculate predictions very quickly by multuplying these matrices
fast_recs = model.user_factors @ model.item_factors.T 
fast_recs.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="uJ_QCocr70yk" executionInfo={"status": "ok", "timestamp": 1628362388296, "user_tz": -330, "elapsed": 18, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2723c559-73dd-4e0c-a8ff-c598f5c312ce"
fast_recs[0,:]
```

<!-- #region id="3iTMfgGc_5MR" -->
### Embeddings visualization
<!-- #endregion -->

```python id="r7w28p_F70u1"
def reduce_dims(df, dims=2, method='pca'):
    
    assert method in ['pca', 'tsne'], 'Неверно указан метод'
    
    if method=='pca':
        pca = PCA(n_components=dims)
        components = pca.fit_transform(df)
    elif method == 'tsne':
        tsne = TSNE(n_components=dims, learning_rate=250, random_state=42)
        components = tsne.fit_transform(df)
    else:
        print('Error')
        
    colnames = ['component_' + str(i) for i in range(1, dims+1)]
    return pd.DataFrame(data = components, columns = colnames) 
```

```python id="Zyg7Upt4-Wrk"
def display_components_in_2D_space(components_df, labels='category', marker='D'):
    
    groups = components_df.groupby(labels)

    # Plot
    fig, ax = plt.subplots(figsize=(12,8))
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    for name, group in groups:
        ax.plot(group.component_1, group.component_2, 
                marker='o', ms=6,
                linestyle='',
                alpha=0.7,
                label=name)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

    plt.xlabel('component_1')
    plt.ylabel('component_2') 
    plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="CV8ENXKF-fp2" executionInfo={"status": "ok", "timestamp": 1628362642245, "user_tz": -330, "elapsed": 930, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="577e8d11-910b-4e01-93cb-d0f031f2ffa0"
model.item_factors.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="6jH3x7Zo-YwS" executionInfo={"status": "ok", "timestamp": 1628362681657, "user_tz": -330, "elapsed": 4702, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a125b268-56fc-4469-a282-e0744c493ba9"
category = []

for idx in range(model.item_factors.shape[0]):
    try:
        cat = item_features.loc[item_features['item_id'] == id_to_itemid[idx], 'department'].values[0]
        category.append(cat)
    except:
        category.append('UNKNOWN')
        
print(category[:10])
```

```python id="2xPLnpZ2-yno"
item_emb_tsne = reduce_dims(model.item_factors, dims=2, method='tsne') # 5001 х 64  ---> 5001 x 2
item_emb_tsne['category'] = category
item_emb_tsne = item_emb_tsne[item_emb_tsne['category'] != 'UNKNOWN']
```

```python colab={"base_uri": "https://localhost:8080/", "height": 498} id="29UQeYkf-pYh" executionInfo={"status": "ok", "timestamp": 1628362769355, "user_tz": -330, "elapsed": 62195, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d8de9261-ed7e-4531-e57e-3d58a57b0ae3"
display_components_in_2D_space(item_emb_tsne, labels='category')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 498} id="9zsYyIhg-wyh" executionInfo={"status": "ok", "timestamp": 1628362796492, "user_tz": -330, "elapsed": 3516, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="696c1058-b34c-42b6-8858-950bcf1a3d94"
# grocery taking lots of points in this space
# let's draw again without grocery to get more clear picture
display_components_in_2D_space(item_emb_tsne[item_emb_tsne['category'] != 'GROCERY'], labels='category')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 498} id="zB03YYpK_L9i" executionInfo={"status": "ok", "timestamp": 1628362846560, "user_tz": -330, "elapsed": 3917, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3f0373ec-35d7-4a36-e072-0b816a4bb915"
# let's draw some specific categories
interesting_catgs = ['PASTRY', 'PRODUCE', 'DRUG GM', 'FLORAL']
display_components_in_2D_space(item_emb_tsne[item_emb_tsne['category'].isin(interesting_catgs)], 
                                             labels='category')
```

<!-- #region id="j6ZNN9AZ_vAj" -->
### Similar items
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="mo0t0owM_Rxt" executionInfo={"status": "ok", "timestamp": 1628362918541, "user_tz": -330, "elapsed": 1148, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ecc28cf0-5a4d-41fd-8a93-c9889e316ed7"
example_item_row_id = 3606
closest_items = [id_to_itemid[row_id] for row_id, score in model.similar_items(example_item_row_id, N=5)]
item_features[item_features.item_id.isin(closest_items)]
```

<!-- #region id="W1B_3eqb_yCG" -->
### Similar users
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="dj89v3mP_po_" executionInfo={"status": "ok", "timestamp": 1628363027706, "user_tz": -330, "elapsed": 1050, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b1688afd-4ca3-4d71-9349-55aac8de624a"
model.similar_users(userid_to_id[10], N=5)
```

<!-- #region id="pAW0Y4vDAGOv" -->
## TF-IDF weighting
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 355, "referenced_widgets": ["6454b7c8f95944f7a147ea1208c6d2bb", "260050051a4244f19db0d89b8b70f75c", "2827f57a96bd42039925df0370100cc3", "996a46b023ff40e59613942c9fbd43d6", "ac4827b1efd34618a55ef57551ea7be5", "5c957ac218fc47d9b792a538bb8ba535", "b3fabde82bba4734b031047f3f2e1b3b", "9f27aa9a2eb645cc93b34f4b7a807e82"]} id="0HNzn_TSAIGD" executionInfo={"status": "ok", "timestamp": 1628363341865, "user_tz": -330, "elapsed": 48147, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ebfd3c8c-b880-48e6-9473-04733bb5e951"
user_item_matrix = tfidf_weight(user_item_matrix.T).T

model = AlternatingLeastSquares(factors=64, 
                                regularization=0.05,
                                iterations=15, 
                                calculate_training_loss=True,
                                use_gpu=False)

model.fit(csr_matrix(user_item_matrix).T.tocsr(), show_progress=True)

result['als_tfidf'] = result['user_id'].apply(lambda x: get_recommendations(x, model=model, N=5))
print(result.apply(lambda row: precision_at_k(row['als_tfidf'], row['actual']), axis=1).mean())

result.head()
```

<!-- #region id="ytJSn3WqAUKm" -->
## BM25 weighting
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 211, "referenced_widgets": ["d09c236a50884587a0b3021e75fdaf7f", "9f4043251c6c4834a3ed7c874d46626e", "d541934cab034673a3fb70a93d75202b", "91f4687fe891435aaa1b080f16bc98b6", "517b0e8c5ee146bd89fd50fffb34a6c6", "8be88fc67ac5464b8f50bab2570154c3", "e79f64ebf47749d5b1d853148a816a97", "302641470f17479399832c36c1b3f713"]} id="pOwEoszFAm9I" executionInfo={"status": "ok", "timestamp": 1628363396572, "user_tz": -330, "elapsed": 53291, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1a55c0e1-02bf-41ab-fe29-8e8329187b28"
data_train.loc[~data_train['item_id'].isin(top_5000), 'item_id'] = 999999
user_item_matrix = pd.pivot_table(data_train, 
                                  index='user_id', columns='item_id', 
                                  values='quantity',
                                  aggfunc='count', 
                                  fill_value=0
                                 )

user_item_matrix = user_item_matrix.astype(float)
sparse_user_item = csr_matrix(user_item_matrix).tocsr()
user_item_matrix = bm25_weight(user_item_matrix.T).T 

model = AlternatingLeastSquares(factors=64, 
                                regularization=0.05,
                                iterations=15, 
                                calculate_training_loss=True, 
                                use_gpu=False)

model.fit(csr_matrix(user_item_matrix).T.tocsr(), show_progress=True)

result['als_bm25'] = result['user_id'].apply(lambda x: get_recommendations(x, model=model, N=5))
print(result.apply(lambda row: precision_at_k(row['als_bm25'], row['actual']), axis=1).mean())
result.head(2)
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
