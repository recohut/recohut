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

```python colab={"base_uri": "https://localhost:8080/"} id="kctCMXhsTVqp" executionInfo={"status": "ok", "timestamp": 1638110419753, "user_tz": -330, "elapsed": 5355, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c23c1135-8c17-4eab-881e-f2044d40c97d"
!pip install -q tensorflow_addons
```

```python id="Cj4mS1f2RoZh" executionInfo={"status": "ok", "timestamp": 1638110422539, "user_tz": -330, "elapsed": 2796, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import os
import heapq  
import gc
from tqdm import tqdm
import random
from sklearn.metrics import mean_squared_error

from tensorflow import keras
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import optimizers, callbacks, layers, losses
from tensorflow.keras.layers import Dense, Concatenate, Activation, Add, BatchNormalization, Dropout, Input, Embedding, Flatten, Multiply, Dot
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.utils import to_categorical

import warnings
warnings.filterwarnings('ignore')
```

```python id="4gILXTyjRyUx" executionInfo={"status": "ok", "timestamp": 1638110423008, "user_tz": -330, "elapsed": 476, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
gpus = tf.config.experimental.list_physical_devices('GPU')
```

```python id="0fxfIpXLTe-B" executionInfo={"status": "ok", "timestamp": 1638110423010, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
```

```python id="4DbVJCtKPuBF" executionInfo={"status": "ok", "timestamp": 1638110423011, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def mish(x):
    return x*tf.math.tanh(tf.math.softplus(x))

def leakyrelu(x, factor=0.2):
    return tf.maximum(x, factor*x)
```

<!-- #region id="kZ9wjIupPbbS" -->
## Matrix Factorization from scratch - SGD method
<!-- #endregion -->

<!-- #region id="tvIqdo56hyWe" -->
### Data Loading
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ivMV3Edr_Wln" executionInfo={"status": "ok", "timestamp": 1638110427525, "user_tz": -330, "elapsed": 1707, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0a2eb449-c8fd-43b4-93d7-9840c5ab7576"
!wget -q --show-progress -O movies.dat https://github.com/RecoHut-Datasets/movielens_1m/raw/main/ml1m_items.dat
!wget -q --show-progress -O ratings.dat https://github.com/RecoHut-Datasets/movielens_1m/raw/main/ml1m_ratings.dat
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="rCg36MGFTue5" executionInfo={"status": "ok", "timestamp": 1638110472199, "user_tz": -330, "elapsed": 6230, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0afbc6ff-c0fa-475f-bab4-c80731136148"
df = pd.read_csv('ratings.dat',
                     sep="\t",
                     header=None,
                     engine='python',
                     names=['userId', 'movieId', 'rating', 'time'])

df.head()
```

```python id="Pr9OvQ4Jg35r" executionInfo={"status": "ok", "timestamp": 1638110472201, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
uuid = df['userId'].unique()
uiid = df['movieId'].unique()
```

```python id="V9XHKoYUL_Fq" executionInfo={"status": "ok", "timestamp": 1638110474390, "user_tz": -330, "elapsed": 1697, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
um = pd.pivot_table(df, values='rating', index='userId', columns='movieId').fillna(0)
```

<!-- #region id="Dc46pHBjPCVY" -->
### RMSE eval
<!-- #endregion -->

```python id="2EsEU9fjMMwU" executionInfo={"status": "ok", "timestamp": 1638110474391, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def rmse(true, pred):
    user_idx, item_idx = np.nonzero(true)
    trues = [true[i][j] for i, j in zip(user_idx, item_idx)]
    preds = [pred[i][j] for i, j in zip(user_idx, item_idx)]
    return np.sqrt(mean_squared_error(trues, preds))
```

<!-- #region id="be6cyWfZQFd_" -->
### Algorithm
<!-- #endregion -->

```python id="S_MBa3NcMEK_" executionInfo={"status": "ok", "timestamp": 1638110474913, "user_tz": -330, "elapsed": 4, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def matrix_fatorization(M, k, epochs, lr=0.01):
    n_user, n_item = M.shape
    
    U = np.random.normal(0., 1./k, (n_user, k))
    V = np.random.normal(0., 1./k, (n_item, k))
    
    u_idx, i_idx = np.nonzero(M)
    
    for e in (range(epochs)):
        for i, j in zip(u_idx, i_idx):
            e_ij = M[i][j] - np.dot(U[i,:], V[j,:].T)
            
            U[i, :] = U[i, :] + lr*(e_ij*V[j, :] - 0.01*U[i,:])
            V[j, :] = V[j, :] + lr*(e_ij*U[i, :] - 0.01*V[j,:])
            
        recon = np.dot(U, V.T)
        print(f'epochs: {e}:', rmse(M, recon))
    return U, V.T
```

```python colab={"base_uri": "https://localhost:8080/"} id="L51cmwAxN9hH" executionInfo={"status": "ok", "timestamp": 1638110582455, "user_tz": -330, "elapsed": 107024, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0885923a-4daf-4877-f7b0-784659db6172"
U, V = matrix_fatorization(um.values, 16, 5)
```

```python colab={"base_uri": "https://localhost:8080/"} id="pYsATJWPON97" executionInfo={"status": "ok", "timestamp": 1638110584867, "user_tz": -330, "elapsed": 2427, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9e7e181a-f665-4ec5-d168-d5d4b3209bbd"
recon = np.dot(U, V)
rmse(um.values, recon)
```

<!-- #region id="bdYX0lr1QBnF" -->
### Inference
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="X6d2dkMFOwyQ" executionInfo={"status": "ok", "timestamp": 1638110584868, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="72ddf58a-368c-41cc-9aa3-8a3f82d8fb00"
def get_best(record, U, V=V, top_k=10):
    prev = np.nonzero(record[0])[0]
    candidates = np.argsort(-np.dot(U, V))
    
    res = []
    cnt = 0
    for c in candidates:
        if c not in prev:
            res.append(c)
            cnt += 1
        if cnt == top_k:
            return res
get_best(um.values, U[0], V, 10)
```

<!-- #region id="cLw5CpTpPtKz" -->
## Matrix Factorization from scratch - ALS method
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="2ETdFDuZQJQJ" executionInfo={"elapsed": 7023, "status": "ok", "timestamp": 1630264555956, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} outputId="a2330e26-a423-45ae-92b3-910bf3c0e5b6"
df = pd.read_csv('ratings.dat',
                     sep="\t",
                     header=None,
                     engine='python',
                     names=['userId', 'movieId', 'rating', 'time'])

df.head()
```

```python id="CdkovDx3QJQL"
uuid = df['userId'].unique()
uiid = df['movieId'].unique()
```

```python id="NK8o3kIAVZ-A"
def extract_from_df(df, n_positive):
    df_ = df.copy()
    rtd = []
    user_id = df['userId'].unique()
    for i in tqdm(user_id):
        rtd += list(np.random.choice(df[df['userId']==i]['movieId'].index, n_positive, replace=False))
    return rtd
```

```python colab={"base_uri": "https://localhost:8080/"} id="67nSMyjoQJQM" executionInfo={"elapsed": 7822, "status": "ok", "timestamp": 1630264587917, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} outputId="e5843f7a-1a13-4cf2-bd1d-e6d3e9e8cc5a"
rtd = extract_from_df(df, 1)
train = df.drop(rtd)
test = df.loc[rtd]
```

```python id="JOv9CosPXEfv"
R = pd.pivot_table(train, index='userId', values='rating', columns='movieId').fillna(0)

P = np.where(R>0, 1, 0)
R = R.values
n_u = R.shape[0]
n_i = R.shape[1]

k = 20
alpha = 40
lamda = 150
epochs = 10
X = np.random.rand(n_u, k)*0.01
Y = np.random.rand(n_i, k)*0.01

C = (1 + alpha*R)
```

```python colab={"background_save": true, "base_uri": "https://localhost:8080/"} id="Lq1cdOI2V06m" outputId="dfea8953-5d16-4ba9-e42e-cb6c5da120eb"
def loss_function(C, P, X, Y, r_lambda):
    predict_error = np.square(P - np.matmul(X, Y.T))
    
    regularization = r_lambda * (np.mean(np.square(X)) + np.mean(np.square(Y)))
    confidence_error = np.mean(C * predict_error)
    total_loss = confidence_error + regularization
    predict_error = np.mean(predict_error)
    return predict_error, confidence_error, regularization, total_loss

def update(x, y, p, c=C):
    xt = x.T
    yt = y.T
    
    for u in range(n_u):
        c_ = C[u, :]
        p_ = P[u, :]
        cu = np.diag(c_)
        
        ycy = y.T.dot(cu).dot(y)
        ycyi = ycy+lamda*np.identity(ycy.shape[0])
        ycp = y.T.dot(cu).dot(p_.T)
        
        x[u] = np.linalg.solve(ycyi, ycp)
        
    for i in range(n_i):
        c_ = C[:, i]
        p_ = P[:, i]
        ci = np.diag(c_)
        
        xcx = x.T.dot(ci).dot(x)
        xcxi = xcx+lamda*np.identity(xcx.shape[0])
        xcp = x.T.dot(ci).dot(p_.T)
        
        y[i] = np.linalg.solve(xcxi, xcp)
        
    return x, y

for e in tqdm(range(epochs)):
    X, Y = update(X, Y, C)
    predict_error, confidence_error, regularization, total_loss = loss_function(C, P, X, Y, lamda)
    print('----------------step %d----------------' %e)
    print("predict error: %f" % predict_error)
    print("confidence error: %f" % confidence_error)
    print("regularization: %f" % regularization)
    print("total loss: %f" % total_loss)
```

```python id="qstXQRVrXP0y"
def eval_hit(X, y, df, test, user_id, item_ids, top_k):
    df = pd.concat([df, test])
    items = list(set(item_ids) - set(df[df['userId']==user_id][df['rating']==1]['movieId'].values))
    np.random.shuffle(items)
    items = items[:99]
    items.append(test[test['userId']==user_id]['movieId'].values[0])
    items = np.array(items).reshape(-1, 1)

    user = np.full(len(items), user_id).reshape(-1, 1)

    preds = np.dot(X[user_id], Y[items].squeeze(1).T)
    item_to_pred = {item: pred for item, pred in zip(items.flatten(), preds)}

    top_k = heapq.nlargest(top_k, item_to_pred, key=item_to_pred.get)
    
    if items[-1][0] in top_k:
            return 1
    return 0

def eval_NDCG(X, Y, df, test, user_id, item_ids, top_k):
    df = pd.concat([df, test])
    items = list(set(item_ids) - set(df[df['userId']==user_id][df['rating']==1]['movieId'].values))
    np.random.shuffle(items)
    items = items[:99]
    items.append(test[test['userId']==user_id]['movieId'].values[0])
    items = np.array(items).reshape(-1, 1)

    user = np.full(len(items), user_id).reshape(-1, 1)

    preds = np.dot(X[user_id], Y[items].squeeze(1).T)
    item_to_pred = {item: pred for item, pred in zip(items.flatten(), preds)}

    top_k = heapq.nlargest(top_k, item_to_pred, key=item_to_pred.get)
    
    for i, item in enumerate(top_k, 1):
        if item == test[test['userId']==user_id]['movieId'].values:
            return np.log(i) / np.log(i+2)
    return 0

def eval_hit_wrapper(X, Y, df, test, item_ids, top_k):
    def f(user_id):
        return eval_hit(X, Y, df, test, user_id, item_ids, top_k)
    return f

def eval_NDCG_wrapper(X, Y, df, test, item_ids, top_k):
    def f(user_id):
        return eval_NDCG(X, Y, df, test, user_id, item_ids, top_k)
    return f
```

```python id="wWpByuKdXkIj"
hits10 = list(map(eval_hit_wrapper(X, Y, train, test, uiid, 10), uuid))
print(sum(hits10)/len(hits10))
```

```python id="yYcUsX1MXlov"
ndcg10 = list(map(eval_NDCG_wrapper(X, Y, train, test, uiid, 10), uuid))
print(sum(ndcg10)/len(ndcg10))
```

<!-- #region id="Ypy7SvqbAJxa" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="1Ukmo4-MAJxc" executionInfo={"status": "ok", "timestamp": 1638110652821, "user_tz": -330, "elapsed": 4162, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3b2ae774-72b1-4c7d-b8f7-82a595377c53"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="bsfbbzuiAJxc" -->
---
<!-- #endregion -->

<!-- #region id="UJGdst-ZAJxd" -->
**END**
<!-- #endregion -->
