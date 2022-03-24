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

```python id="UV_mis-jdwLd" executionInfo={"status": "ok", "timestamp": 1629281631404, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-mlh"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python id="KRGLEjqMd3dV" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629281634476, "user_tz": -330, "elapsed": 3081, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ff0fd268-492a-47f5-aebe-e51c337e4654"
if not os.path.exists(project_path):
    !cp /content/drive/MyDrive/mykeys.py /content
    import mykeys
    !rm /content/mykeys.py
    path = "/content/" + project_name; 
    !mkdir "{path}"
    %cd "{path}"
    import sys; sys.path.append(path)
    !git config --global user.email "recotut@recohut.com"
    !git config --global user.name  "reco-tut"
    !git init
    !git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout main
else:
    %cd "{project_path}"
```

```python id="wpdAcaW80D3o" executionInfo={"status": "ok", "timestamp": 1629282662637, "user_tz": -330, "elapsed": 622, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8e2933a5-fa9c-4dc5-e8ac-9ae84aff954e" colab={"base_uri": "https://localhost:8080/"}
!git pull --rebase origin "{branch}"
```

```python colab={"base_uri": "https://localhost:8080/"} id="Aa6AQmftAovn" executionInfo={"status": "ok", "timestamp": 1629282668910, "user_tz": -330, "elapsed": 444, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b3bdebd7-f593-4514-f400-85589d20504e"
!git status
```

```python colab={"base_uri": "https://localhost:8080/"} id="aG5PN_2EAovn" executionInfo={"status": "ok", "timestamp": 1629282673079, "user_tz": -330, "elapsed": 615, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="57ff8b49-7f5e-463b-9d37-d391463f0207"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python id="7J9mKTyRyFwQ" executionInfo={"status": "ok", "timestamp": 1629282146336, "user_tz": -330, "elapsed": 532, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import sys
sys.path.insert(0,'./code')
```

<!-- #region id="y1Fne9fbwMl5" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Dq_dZSZuwXN9" executionInfo={"status": "ok", "timestamp": 1629281771477, "user_tz": -330, "elapsed": 85565, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e10c86e7-6712-424e-c71c-5145aa8c752a"
!pip install -q implicit
```

```python id="laKkuV4RwM4R" executionInfo={"status": "ok", "timestamp": 1629282159124, "user_tz": -330, "elapsed": 423, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import pandas as pd
import numpy as np
import implicit
import scipy.sparse as sparse

from metricsv2 import *
```

```python id="BROZFpzNwVw4" executionInfo={"status": "ok", "timestamp": 1629281837683, "user_tz": -330, "elapsed": 501, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
columns = ['movieid', 'title', 'release_date', 'video_release_date', \
           'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', \
           'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', \
           'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', \
           'Thriller', 'War', 'Western']
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="4e9BSTWxw92C" executionInfo={"status": "ok", "timestamp": 1629281901522, "user_tz": -330, "elapsed": 456, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="71a826af-df04-4e82-d2e7-4f08512dfa62"
df_train = pd.read_csv('./data/bronze/u2.base',
                         sep='\t',
                         names=['userid', 'itemid', 'rating', 'timestamp'],
                         header=None)

# rating> = 3 is relevant (1) and rating less than 3 is not relevant (0)
df_train.rating = [1 if x >=3 else 0 for x in df_train.rating ]

df_train.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 391} id="6YKLBSeCxNdZ" executionInfo={"status": "ok", "timestamp": 1629281924895, "user_tz": -330, "elapsed": 773, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3a863f1b-b2c6-4301-c461-000c85a1e56b"
df_items = pd.read_csv('./data/bronze/u.item',
                        sep='|',
                        index_col=0,
                        names = columns,
                        header=None, 
                        encoding='latin-1')

df_items.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="OPOEIOTCxQ4r" executionInfo={"status": "ok", "timestamp": 1629281992831, "user_tz": -330, "elapsed": 633, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0118c234-c259-4c81-e810-35ae82929d5d"
df_test = pd.read_csv('./data/bronze/u2.test',
                      sep='\t',
                      names=['userid', 'itemid', 'rating', 'timestamp'],
                      header=None)
df_test.rating = [1 if x >=3 else 0 for x in df_test.rating ]
df_test.head()
```

```python id="DvLBsMkmxjnr" executionInfo={"status": "ok", "timestamp": 1629282209325, "user_tz": -330, "elapsed": 409, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
user_items_test = {}
for row in df_test.itertuples():
    if row[1] not in user_items_test:
        user_items_test[row[1]] = []
    user_items_test[row[1]].append(row[2])
```

```python id="161ZL_eUyTPG" executionInfo={"status": "ok", "timestamp": 1629282211454, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
user_items = {}
itemset = set()

for row in df_train.itertuples():
    if row[1] not in user_items:
        user_items[row[1]] = []
        
    user_items[row[1]].append(row[2])
    itemset.add(row[2])

itemset = np.sort(list(itemset))

sparse_matrix = np.zeros((len(user_items), len(itemset)))

for i, items in enumerate(user_items.values()):
    sparse_matrix[i] = np.isin(itemset, items, assume_unique=True).astype(int)
    
matrix = sparse.csr_matrix(sparse_matrix.T)

user_ids = {key: i for i, key in enumerate(user_items.keys())}
user_item_matrix = matrix.T.tocsr()
```

```python colab={"base_uri": "https://localhost:8080/"} id="w0Hxna_Hyqut" executionInfo={"status": "ok", "timestamp": 1629282286315, "user_tz": -330, "elapsed": 491, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="22994624-fbcc-49b2-f2d9-124ccc1df080"
user_item_matrix
```

```python id="-tbvAwqzyZFT" executionInfo={"status": "ok", "timestamp": 1629282229586, "user_tz": -330, "elapsed": 430, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def evaluate_model(model, n):
  mean_map = 0.
  mean_ndcg = 0.
  for u in user_items_test.keys():
    rec = [t[0] for t in model.recommend(u, user_item_matrix, n)]
    rel_vector = [np.isin(user_items_test[u], rec, assume_unique=True).astype(int)]
    mean_map += mean_average_precision(rel_vector)
    mean_ndcg += ndcg_at_k(rel_vector, n)

  mean_map /= len(user_items_test)
  mean_ndcg /= len(user_items_test)
  
  return mean_map, mean_ndcg
```

```python id="51_RJk3jydZ8" executionInfo={"status": "ok", "timestamp": 1629282234098, "user_tz": -330, "elapsed": 623, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def show_recommendations(model, user, n):
  recommendations = [t[0] for t in model.recommend(user, user_item_matrix, n)]
  return df_items.loc[recommendations]['title']
```

```python id="oXG_3-BJyeiM" executionInfo={"status": "ok", "timestamp": 1629282235163, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def show_similar_movies(model, item, n=10):
  sim_items = [t[0] for t in model.similar_items(item, n)]
  return df_items.loc[sim_items]['title']
```

```python id="jP9cQZC6zjcn" executionInfo={"status": "ok", "timestamp": 1629282572624, "user_tz": -330, "elapsed": 818, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# custom NDCG
def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.
    
def ndcg_at_k(r, k):
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg
```

<!-- #region id="q98ohvSkye26" -->
## ALS
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 86, "referenced_widgets": ["78ba09edd24b4e5984ca2533cb0cfa46", "5251e91e43f44376a31cab28f2185b62", "bcc1760cdbef476c88b4dcf465a3ce5d", "5a62bd68a3bc471a9bbd29e35f5d90b2", "dff823a1ccbe4db099408c16a667ab73", "b3edb91d8ab14d188239edaa5539e452", "d52c348b68d64476a05ebdb8046f9f37", "d5fbce2e94014eaaa909160185d7dab8", "b3324a339ca0471eb6b0ce6ef03f943f", "19d71d80f2534cd9bc61c05cfa1742eb", "6f49dff0cf304daaa27808f9248240e1"]} id="seKyNTxoylbZ" executionInfo={"status": "ok", "timestamp": 1629282302492, "user_tz": -330, "elapsed": 2798, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ca4e4595-163d-4e21-8951-e5b85b01ef56"
model_als = implicit.als.AlternatingLeastSquares(factors=100, iterations=10, use_gpu=False)
model_als.fit(matrix)
```

```python colab={"base_uri": "https://localhost:8080/"} id="bVpcYFUkyuuD" executionInfo={"status": "ok", "timestamp": 1629282302493, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="92ae19fd-26e5-4809-f767-24affd1e89f2"
show_recommendations(model_als, user=77, n=10)
```

```python colab={"base_uri": "https://localhost:8080/"} id="wMLyUWwTyvPp" executionInfo={"status": "ok", "timestamp": 1629282577244, "user_tz": -330, "elapsed": 1456, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="378a23ee-3773-426b-9d33-f941e5d0e7a1"
maprec, ndcg = evaluate_model(model_als, n=10)
print('map: {}\nndcg: {}'.format(maprec, ndcg))
```

<!-- #region id="_DIKNWLlyxba" -->
## BPR
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["f2b57f39511f47ccac2588c1dd724a59", "c31552748beb4a6498e15b19594976a3", "7909855dc880492d83971668ea2df76d", "cd4672023e404106845e34b7b3836ba9", "bba957d4e9f540d69dc2e24e0ae64ef3", "6c8f6bd5599846079d3718d338751125", "0c68e21e09d0423dbb9358e39ffa7822", "413557e723e34feeac31722471d7307c", "07d17687383246b599e34cc0635aebac", "781792950f974880b3d23d1543e51442", "fa8ed30e21634f7089a6abdbc3c23a37"]} id="cGQWWu7pz4kE" executionInfo={"status": "ok", "timestamp": 1629282618767, "user_tz": -330, "elapsed": 2798, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="309ceb4c-1fa5-425f-cfb3-387bdf875a4f"
model_bpr = implicit.bpr.BayesianPersonalizedRanking(factors=400, iterations=40, use_gpu=False)
model_bpr.fit(matrix)
```

```python colab={"base_uri": "https://localhost:8080/"} id="MalphiRWz7_N" executionInfo={"status": "ok", "timestamp": 1629282624784, "user_tz": -330, "elapsed": 32, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2ba94274-c274-4141-c36b-96dab2001a1b"
show_recommendations(model_bpr, user=77, n=10)
```

```python colab={"base_uri": "https://localhost:8080/"} id="BJZ51vQ7z95G" executionInfo={"status": "ok", "timestamp": 1629282634707, "user_tz": -330, "elapsed": 1158, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8a8f178e-87b4-4de4-a0b4-b53bc90366d2"
maprec, ndcg = evaluate_model(model_bpr, n=10)
print('map: {}\nndcg: {}'.format(maprec, ndcg))
```
