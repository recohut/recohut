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

```python id="oGHZx10vJg9R"
import os
project_name = "reco-chef"; branch = "ml1m"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python colab={"base_uri": "https://localhost:8080/"} id="M9shE0jRJg9a" executionInfo={"status": "ok", "timestamp": 1630837255225, "user_tz": -330, "elapsed": 5270, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b4541528-c0f5-4385-d376-fbe17290e68b"
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

```python id="5iDYrd0-Jg9c"
!git status
```

```python id="iIIi0R89iWLz"
!dvc status
```

```python id="-9DWAoQyJg9d"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python id="kI4SpIr2JqGJ"
!dvc pull ./data/bronze/ml-1m/*.dvc
```

```python id="FwYoeMVyJvzL"
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import os,sys,inspect
import gc
from tqdm import tqdm
import random

import heapq

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks, layers, losses
from tensorflow.keras.layers import Dense, Concatenate, Activation, Add, BatchNormalization, Dropout, Input, Embedding, Flatten, Multiply
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

from sklearn.metrics.pairwise import cosine_similarity
```

```python id="sB_OJdPrYZSK"
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
gpus = tf.config.experimental.list_physical_devices('GPU')
```

```python id="0fxfIpXLTe-B"
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
```

```python colab={"base_uri": "https://localhost:8080/"} id="EpIyG5zgtaai" executionInfo={"status": "ok", "timestamp": 1630837378146, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="eb91aac6-4f42-4020-f013-97b3b39993df"
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

```python id="4DbVJCtKPuBF"
def mish(x):
    return x*tf.math.tanh(tf.math.softplus(x))

def leakyrelu(x, factor=0.2):
    return tf.maximum(x, factor*x)
```

```python id="XOfeaOEgLsGG"
def load_data(filepath, threshold=0):
    df = pd.read_csv(filepath,
                     sep="::",
                     header=None,
                     engine='python',
                     names=['userId', 'movieId', 'rating', 'time'])
    df = df.drop('time', axis=1)
    df['userId'] = df['userId'].astype(int)
    df['movieId'] = df['movieId'].astype(int)
    df['rating'] = df['rating'].astype(float)
    
    df = df[['userId', 'movieId', 'rating']]
    if threshold > 0:
        df['rating'] = np.where(df['rating']>threshold, 1, 0)  
    else:
        df['rating'] = 1.
    m_codes = df['movieId'].astype('category').cat.codes
    u_codes = df['userId'].astype('category').cat.codes
    df['movieId'] = m_codes
    df['userId'] = u_codes

    return df
    

def add_negative(df, uiid, times=4):
    df_ = df.copy()
    user_id = df_['userId'].unique()
    item_id = df_['movieId'].unique()
    
    for i in tqdm(user_id):
        cnt = 0
        n = len(df_[df_['userId']==i])
        n_negative = min(n*times, len(item_id)-n-1)
        available_negative = list(set(uiid) - set(df[df['userId']==i]['movieId'].values))
        
        new = np.random.choice(available_negative, n_negative, replace=False)
        new = [[i, j, 0] for j in new]
        df_ = df_.append(pd.DataFrame(new, columns=df.columns), ignore_index=True)
    
    return df_

def extract_from_df(df, n_positive, n_negative):
    df_ = df.copy()
    rtd = []
    
    user_id = df['userId'].unique()
    
    for i in tqdm(user_id):
        rtd += list(np.random.choice(df[df['userId']==i][df['rating']==1]['movieId'].index, n_positive, replace=False))
        rtd += list(np.random.choice(df[df['userId']==i][df['rating']==0]['movieId'].index, n_negative, replace=False))
        
    return rtd
```

```python id="GFQjwqAPMH8C"
def eval_hit(model, df, test, user_id, item_ids, top_k):
    df = pd.concat([df, test])
    items = list(set(item_ids) - set(df[df['userId']==user_id][df['rating']==1]['movieId'].values))
    np.random.shuffle(items)
    items = items[:99]
    items.append(test[test['userId']==user_id]['movieId'].values[0])
    items = np.array(items).reshape(-1, 1)

    user = np.full(len(items), user_id).reshape(-1, 1)

    preds = model.predict([user, items]).flatten()
    item_to_pred = {item: pred for item, pred in zip(items.flatten(), preds)}

    top_k = heapq.nlargest(top_k, item_to_pred, key=item_to_pred.get)
    
    if items[-1][0] in top_k:
            return 1
    return 0

def eval_NDCG(model, df, test, user_id, item_ids, top_k):
    df = pd.concat([df, test])
    items = list(set(item_ids) - set(df[df['userId']==user_id][df['rating']==1]['movieId'].values))
    np.random.shuffle(items)
    items = items[:99]
    items.append(test[test['userId']==user_id]['movieId'].values[0])
    items = np.array(items).reshape(-1, 1)

    user = np.full(len(items), user_id).reshape(-1, 1)

    preds = model.predict([user, items]).flatten()
    item_to_pred = {item: pred for item, pred in zip(items.flatten(), preds)}

    top_k = heapq.nlargest(top_k, item_to_pred, key=item_to_pred.get)
    
    for i, item in enumerate(top_k, 1):
        if item == test[test['userId']==user_id]['movieId'].values:
            return 1 / np.log2(i+1)
    return 0

def eval_hit_wrapper(model, df, test, item_ids, top_k):
    def f(user_id):
        return eval_hit(model, df, test, user_id, item_ids, top_k)
    return f

def eval_NDCG_wrapper(model, df, test, item_ids, top_k):
    def f(user_id):
        return eval_NDCG(model, df, test, user_id, item_ids, top_k)
    return f
```

<!-- #region id="gMR1AHL5cw1j" -->
## CML
<!-- #endregion -->

<!-- #region id="1sAQYk5eYv1j" -->
### Load data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="rCg36MGFTue5" executionInfo={"status": "ok", "timestamp": 1630837422281, "user_tz": -330, "elapsed": 6178, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ae9fdb59-51fa-4e4d-e1dc-9bc2c9ffd038"
df = load_data('./data/bronze/ml-1m/ratings.dat', threshold=0)
df.head()
```

<!-- #region id="uo-H-H5yZjSI" -->
### Preprocessing
<!-- #endregion -->

```python id="4_0IEsjWM4nI"
uuid = df['userId'].unique()
uiid = df['movieId'].unique()

rtd = extract_from_df(df, 1, 0)

train = df.drop(rtd)
test = df.loc[rtd]

u_i = pd.pivot_table(train, index='userId', columns='movieId', values='rating').fillna(0)
u_i
```

```python id="c3h8GhpqfLza"
groups = []
for i in range(len(u_i)):
    groups.append(list(np.argwhere(u_i.values[i]).flatten()))
# groups = np.array(groups)
```

<!-- #region id="fyhWH5RRfZR_" -->
### Gensim model
<!-- #endregion -->

```python id="XY4Yr5y5fn2o"
from gensim.models import Word2Vec

model = Word2Vec(
      np.array(groups),
      size = 32,
      window=10,
      min_count=1,
      sg=1,
      negative=5)

model.build_vocab(np.array(groups))
model.train(np.array(groups),
         total_examples = model.corpus_count,
         epochs=100,
         compute_loss=True)
```

```python id="-kMZAcEZf_ER"
embedding_matrix = model.wv[model.wv.key_to_index.keys()]
embedding_matrix.shape
```

```python id="LqjPxTC5fkxz"
from sklearn.metrics.pairwise import cosine_similarity

def get_average(user_id, model=model, embedding=embedding_matrix):
    seen_movies = train[train['userId']==user_id]['movieId'].values
    kdx = []
    for i in seen_movies:
        kdx.append(model.wv.key_to_index[i])
        
    vec = embedding_matrix[kdx]
    vec = np.mean(vec, 0)
        
    return vec

def top_n(user_id, k=10, uiid=uiid, model=model):
    seen_movies = train[train['userId']==user_id]['movieId'].values
    unseen_movies = list(set(uiid) - set(seen_movies))
    
    user_vec = get_average(user_id)
    
    kdx = []
    for i in unseen_movies:
        kdx.append(model.wv.key_to_index[i])
        
    unseen_vec = embedding_matrix[kdx]
    
    res = sorted(unseen_movies, key=lambda x: cosine_similarity([embedding_matrix[model.wv.key_to_index[x]]], [user_vec]), reverse=True)
    return np.array(res[:k])
```

```python id="6WPIlwcEgCEz"
cnt = 0
for i in range(len(test)):
    user, item, _ = test.values[i]
    pred = top_n(user, 10)
    if item in pred:
        cnt += 1
        
cnt / len(test)
```
