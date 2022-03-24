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

<!-- #region id="q3fEie3LylIu" -->
# BPR Triplet on ML-1m in Tensorflow
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="kctCMXhsTVqp" executionInfo={"status": "ok", "timestamp": 1638110751550, "user_tz": -330, "elapsed": 4547, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f870ded4-a841-4f04-8350-7f275ab470cf"
!pip install -q tensorflow_addons
```

```python id="Cj4mS1f2RoZh"
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

import os
import heapq  
import gc
from tqdm import tqdm
import random

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

```python id="4gILXTyjRyUx"
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

<!-- #region id="lReIb0MfCBY0" -->
### Utils
<!-- #endregion -->

```python id="4DbVJCtKPuBF"
def mish(x):
    return x*tf.math.tanh(tf.math.softplus(x))

def leakyrelu(x, factor=0.2):
    return tf.maximum(x, factor*x)
```

```python id="4hQnVCmDBtw3"
def load_data(filepath):
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
    df['rating'] = 1.
    m_codes = df['movieId'].astype('category').cat.codes
    u_codes = df['userId'].astype('category').cat.codes
    df['movieId'] = m_codes
    df['userId'] = u_codes
    
    return df


def make_triplet(df):
    df_ = df.copy()
    user_id = df['userId'].unique()
    item_id = df['movieId'].unique()
    
    negs = np.zeros(len(df), dtype=int)
    for u in tqdm(user_id):
        user_idx = list(df[df['userId']==u].index)
        n_choose = len(user_idx)
        available_negative = list(set(item_id) - set(df[df['userId']==u]['movieId'].values))
        new = np.random.choice(available_negative, n_choose, replace=True)
        
        negs[user_idx] = new
    df_['negative'] = negs
    
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

<!-- #region id="e6jjXYGYB-Rd" -->
### Model
<!-- #endregion -->

```python id="pVTqjvC2Bblf"
class BPR_Triplet(keras.Model):
    def __init__(self, u_dim, i_dim, latent_dim):
        super(BPR_Triplet, self).__init__()
        
        self.u_dim = u_dim
        self.i_dim = i_dim
        self.latent_dim = latent_dim
        
        self.model = self.build_model()

    def compile(self, optim):
        super(BPR_Triplet, self).compile()
        self.optim = optim
    
    def build_model(self):
        u_input = Input(shape=(1, ))
        i_input = Input(shape=(1, ))

        u_emb = Flatten()(Embedding(self.u_dim, self.latent_dim, input_length=u_input.shape[1])(u_input))
        i_emb = Flatten()(Embedding(self.i_dim, self.latent_dim, input_length=i_input.shape[1])(i_input))

        mul = Dot(1)([u_emb, i_emb])

#         out = Dense(1)(mul)
        
        return Model([u_input, i_input], mul)
    
    def train_step(self, data):
        user, pos, neg = data[0]

        with tf.GradientTape() as tape:
            pos_d = self.model([user, pos])
            neg_d = self.model([user, neg])
            
            loss = -tf.reduce_mean(tf.math.log(tf.sigmoid(pos_d - neg_d)))

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optim.apply_gradients(zip(grads, self.model.trainable_weights))
        
        return {'loss': loss}
    
    def call(self, data):
        user, item = data
        return self.model([user, item])
```

<!-- #region id="tvIqdo56hyWe" -->
### Data Loading
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="1WxUTg4qB0gv" executionInfo={"status": "ok", "timestamp": 1638111074245, "user_tz": -330, "elapsed": 1514, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8ea9d32b-5b16-461d-f573-68c1b6bb174f"
!wget -q --show-progress -O movies.dat https://github.com/RecoHut-Datasets/movielens_1m/raw/main/ml1m_items.dat
!wget -q --show-progress -O ratings.dat https://github.com/RecoHut-Datasets/movielens_1m/raw/main/ml1m_ratings.dat
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="rCg36MGFTue5" executionInfo={"status": "ok", "timestamp": 1638111086126, "user_tz": -330, "elapsed": 5199, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d4fc1382-7f73-41b6-c810-f772299ccde4"
df = pd.read_csv('ratings.dat',
                     sep="\t",
                     header=None,
                     engine='python',
                     names=['userId', 'movieId', 'rating', 'time'])

df.head()
```

```python id="Pr9OvQ4Jg35r"
uuid = df['userId'].unique()
uiid = df['movieId'].unique()
```

<!-- #region id="dzI_0mEfhu_e" -->
### Data Preparation
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="akuYNKxIUElb" executionInfo={"status": "ok", "timestamp": 1638111119203, "user_tz": -330, "elapsed": 33086, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="77ae837f-8f7b-458f-ad9f-fa1c8a192010"
# [user_id, positive_item_id, negative_item_id]
df = make_triplet(df)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="a9U4fVPLV62f" executionInfo={"status": "ok", "timestamp": 1638111119205, "user_tz": -330, "elapsed": 24, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3f564d67-c45d-4bfa-c682-33a093039cef"
df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="GfxKxSd2UDS0" executionInfo={"status": "ok", "timestamp": 1630166561844, "user_tz": -330, "elapsed": 58002, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c1f82f7c-bb50-4070-88be-ed553af26707"
# randomly select just one pair
rtd = extract_from_df(df, 1, 0)
```

```python id="ClLe8t18UDPm"
train = df.drop(rtd)
test = df.loc[rtd]

tr_X = [
    train['userId'].values, 
    train['movieId'].values,
    train['negative'].values
]
```

<!-- #region id="gofo4l1Ef7KE" -->
### BPR Triplet model
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ymeL3NcDgw0T" executionInfo={"status": "ok", "timestamp": 1630168363748, "user_tz": -330, "elapsed": 1575226, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="757ae426-247d-4993-9d5a-7a793433fbfc"
bpr = BPR_Triplet(len(uuid), len(uiid), 32)
bpr.compile(optim=optimizers.Adam())
bpr.fit(tr_X, epochs = 10)
```

<!-- #region id="dgNXl-yyR0f1" -->
### Evaluate
<!-- #endregion -->

```python id="h1n1jsSqnw2H"
def eval_hit(model, test, user_id, item_ids, top_k):
    # TODO(maybe): remove negative used in train
    items = list(set(uiid) - set(df[df['userId']==user_id][df['rating']==1]['movieId'].values) - set(df[df['userId']==user_id]['negative'].values))
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

def eval_hit_wrapper(model, test, item_ids, top_k):
    def f(user_id):
        return eval_hit(model, test, user_id, item_ids, top_k)
    return f

def eval_NDCG(model, test,user_id, item_ids, top_k):
    items = list(set(uiid) - set(df[df['userId']==user_id][df['rating']==1]['movieId'].values) - set(df[df['userId']==user_id]['negative'].values))
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
            return np.log(i) / np.log(i+2)
    return 0

def eval_NDCG_wrapper(model, test, item_ids, top_k):
    def f(user_id):
        return eval_NDCG(model, test, user_id, item_ids, top_k)
    return f
```

```python colab={"base_uri": "https://localhost:8080/"} id="TKJym-2ihK6F" executionInfo={"status": "ok", "timestamp": 1630169060183, "user_tz": -330, "elapsed": 415348, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8d007095-663c-45e1-b346-34904d291086"
hr10 = list(map(eval_hit_wrapper(bpr, test, uiid, 10), uuid))
sum(hr10)/len(hr10)
```

```python colab={"base_uri": "https://localhost:8080/"} id="V6guq7UohKyF" executionInfo={"status": "ok", "timestamp": 1630169497351, "user_tz": -330, "elapsed": 437176, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2d719531-9062-4f5f-82da-edac779a8e9a"
ndcg10 = list(map(eval_NDCG_wrapper(bpr, test, uiid, 10), uuid))
sum(ndcg10)/len(ndcg10)
```

<!-- #region id="Dmfq9ln8CG5p" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="FEMlB8X1CG5r" executionInfo={"status": "ok", "timestamp": 1638111165210, "user_tz": -330, "elapsed": 3506, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e1a25019-39ae-400c-f288-d0ae14b710e6"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="ytYxFSQCCG5r" -->
---
<!-- #endregion -->

<!-- #region id="n76EywsgCG5r" -->
**END**
<!-- #endregion -->
