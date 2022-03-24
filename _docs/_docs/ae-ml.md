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

<!-- #region id="tT6lcgsxsYzT" -->
# Autoencoder RecSys Models on ML-1m
<!-- #endregion -->

<!-- #region id="MugCaZzrsrM9" -->
## Setup
<!-- #endregion -->

```python id="FwYoeMVyJvzL"
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import os,sys,inspect
import gc
from tqdm.notebook import tqdm
import random
import heapq

from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks, layers, losses
from tensorflow.keras.layers import Dense, Concatenate, Activation, Add, BatchNormalization, Dropout, Input, Embedding, Flatten, Multiply
from tensorflow.keras.models import Model, Sequential, load_model
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

```python colab={"base_uri": "https://localhost:8080/"} id="EpIyG5zgtaai" executionInfo={"status": "ok", "timestamp": 1639716148938, "user_tz": -330, "elapsed": 28, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9d5c6806-8dde-45d4-a715-1639547315ab"
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

```python colab={"base_uri": "https://localhost:8080/"} id="UlyLzNq1sooD" executionInfo={"status": "ok", "timestamp": 1639716176022, "user_tz": -330, "elapsed": 1132, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="bdabe7d6-81af-4772-c63e-7b466287e619"
!wget -q --show-progress https://files.grouplens.org/datasets/movielens/ml-1m.zip
!unzip ml-1m.zip
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
def eval_NDCG(true, pred):
    top_k = pred

    for i, item in enumerate(top_k, 1):
        if item == true:
            return 1 / np.log2(i+1)
    return 0
```

<!-- #region id="S78F5a7AYgQz" -->
## CDAE
<!-- #endregion -->

<!-- #region id="1sAQYk5eYv1j" -->
### Load data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 215} id="rCg36MGFTue5" executionInfo={"status": "ok", "timestamp": 1639716202713, "user_tz": -330, "elapsed": 5930, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d87757a9-77d6-4c1f-e057-5adab8f37e60"
df = load_data('./ml-1m/ratings.dat', threshold=3)
df.head()
```

<!-- #region id="uo-H-H5yZjSI" -->
### Preprocessing
<!-- #endregion -->

```python id="4_0IEsjWM4nI"
df = df[df['rating']==1].reset_index(drop=True)
tdf = pd.pivot_table(df, index='userId', values='rating', columns='movieId').fillna(0)

cnt = tdf.sum(1)
df = df[df['userId'].isin(np.where(cnt >= 10)[0])].reset_index(drop=True)
tdf = pd.pivot_table(df, index='userId', values='rating', columns='movieId').fillna(0)
tdf.iloc[:,:] = 0

test_idx = []
for i in tdf.index:
    test_idx += list(np.random.choice(df[df['userId']==i].index, 1))
    
train = df.loc[list(set(df.index)-set(test_idx)),:]
test = df.loc[test_idx, :]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 424} id="QCEA3JWYNgHT" executionInfo={"status": "ok", "timestamp": 1639716243239, "user_tz": -330, "elapsed": 49, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3077ce6b-b907-4e0d-b2b2-94c6e019dc42"
df
```

```python colab={"base_uri": "https://localhost:8080/"} id="Ov0ZmaZabiVe" executionInfo={"status": "ok", "timestamp": 1639716249445, "user_tz": -330, "elapsed": 425, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f4c9d415-cd86-4274-8d0e-dca78697f41d"
df.shape, train.shape, test.shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 470} id="AyFVjZiUNtBV" executionInfo={"status": "ok", "timestamp": 1630835048249, "user_tz": -330, "elapsed": 51728, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e9cbcb22-0779-4089-a265-085b72948158"
for uid, iid in zip(train['userId'].values, train['movieId'].values):
    tdf.loc[uid, iid] = 1
train =  tdf.copy()
train
```

<!-- #region id="_br-gyTNd40x" -->
### Model architecture
<!-- #endregion -->

```python id="UK-_v59OYjHz"
class CDAE(tf.keras.models.Model):
    def __init__(self, input_dim, latent_dim, n_user, lamda=1e-4):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lamda = lamda
        self.n_user = n_user
        self.embedding = Embedding(n_user, latent_dim, )        

        self.model = self.build()

    def compile(self, optimizer, loss_fn=None):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        
    def build(self):
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        
        rating = Input(shape=(self.input_dim, ), name='rating_input')
        user_id = Input(shape=(1, ), name='user_input')
        
        emb = self.embedding(user_id)
        emb = tf.squeeze(emb, 1)
        enc = self.encoder(rating) + emb
        enc = tf.nn.tanh(enc)
        outputs = self.decoder(enc)
    
        return Model([rating, user_id], outputs)
    
    def build_encoder(self):
        inputs = Input(shape = (self.input_dim, ))
        
        encoder = Sequential()
        encoder.add(Dropout(0.2))
        encoder.add(Dense(self.latent_dim, activation='tanh'))
        
        outputs = encoder(inputs)
        
        return Model(inputs, outputs)
    
    def build_decoder(self):
        inputs = Input(shape = (self.latent_dim, ))
        
        encoder = Sequential()
        encoder.add(Dense(self.input_dim, activation='sigmoid'))
        
        outputs = encoder(inputs)
        
        return Model(inputs, outputs)
    
    def train_step(self, data):
        x = data['rating']
        user_ids = data['id']
        with tf.GradientTape() as tape:
            pred = self.model([x, user_ids])
            
            rec_loss = tf.losses.binary_crossentropy(x, pred)
            loss = rec_loss

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        
        return {'loss': loss}
```

<!-- #region id="-R-qDHX_dD3T" -->
### Training
<!-- #endregion -->

```python id="Fjtaq3RkcIO6" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1630835242291, "user_tz": -330, "elapsed": 194051, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a69c188d-f725-4553-ee6d-d6df3484f19d"
loader = tf.data.Dataset.from_tensor_slices({'rating': train.values, 'id': np.arange(len(train))})
loader = loader.batch(32, drop_remainder=True).shuffle(len(train))
model = CDAE(train.shape[1], 200, len(train))
model.compile(optimizer=tf.optimizers.Adam())
model.fit(loader, epochs=25)
```

<!-- #region id="NfDBp4XndBDM" -->
### Evaluation
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 66, "referenced_widgets": ["0bc4610585dc4b6e89d3bc4598293dc4", "0e1c65f3d0a04bbfa4d53d8287faf7b0", "707a58e866a9416eaf94f9aea50bb9f5", "35c412197fb6462b9d03d9fe8f65e3cb", "a89f0935d9c846ae92716177f9a3da31", "ae8bf7ee34674a0e9fe0b9aef3b8063d", "eea7588021c24de29cd5018918139907", "b997a6a3013f4f649bbba23c10d1bbe3", "c04be49d501e4f70a3ed182c6253a336", "ee609560d3104683b742c9ed66e782d7", "4e3a404483954fa3916ac4bbf4bd4929"]} id="4vR5WPgGOwbn" executionInfo={"status": "ok", "timestamp": 1630835457454, "user_tz": -330, "elapsed": 128721, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6ad93a00-03d5-4158-bb59-0babb6fac1b5"
top_k = 10
np.random.seed(42)

scores = []
for idx, i in tqdm(enumerate(np.random.choice(train.index, 100))):
    item_to_pred = {item: pred for item, pred in zip(train.columns, model.model.predict([train.values, np.arange(len(train))])[idx])}
    test_ = test[(test['userId']==i) & (test['rating']==1)]['movieId'].values
    items = list(np.random.choice(list(filter(lambda x: x not in np.argwhere(train.values[idx]).flatten(), item_to_pred.keys())), 100)) + list(test_)
    top_k_items = heapq.nlargest(top_k, items, key=item_to_pred.get)
    
    score = eval_NDCG(test_, top_k_items)
    scores.append(score)
    
np.mean(scores)
```

<!-- #region id="cqBW2VetZvne" -->
## EASE
<!-- #endregion -->

<!-- #region id="R-XpG8cMZvnf" -->
### Load data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="D_oXGFDhZvnf" executionInfo={"status": "ok", "timestamp": 1630834671821, "user_tz": -330, "elapsed": 5997, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="66c7d829-c8f3-4e7b-f03c-1c832040c3f5"
df = load_data('./ml-1m/ratings.dat', threshold=3)
df.head()
```

<!-- #region id="OfHD5JjfZvnh" -->
### Preprocessing
<!-- #endregion -->

```python id="KRzrwrHpZvnh"
test_idx = []
user_id = df
for i in df['userId'].unique():
    test_idx += list(np.random.choice(df[df['userId']==i].index, 1))
    
train = df.iloc[list(set(df.index)-set(test_idx)),:]
test = df.iloc[test_idx, :]
```

```python colab={"base_uri": "https://localhost:8080/"} id="5hbeGQeyZvni" executionInfo={"status": "ok", "timestamp": 1630834902542, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="19ad5720-8a92-47bc-b018-b360fe911342"
df.shape, train.shape, test.shape
```

<!-- #region id="REsycBTbZvni" -->
### Model architecture
<!-- #endregion -->

```python id="N8qLutMpZvnj"
class EASE:
    def __init__(self):
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

    def _get_users_and_items(self, df):
        users = self.user_enc.fit_transform(df.loc[:, 'userId'])
        items = self.item_enc.fit_transform(df.loc[:, 'movieId'])
        return users, items

    def fit(self, df, lambda_: float = 0.5, implicit=True):
        """
        df: pandas.DataFrame with columns user_id, item_id and (rating)
        lambda_: l2-regularization term
        implicit: if True, ratings are ignored and taken as 1, else normalized ratings are used
        """
        users, items = self._get_users_and_items(df)
        values = np.ones(df.shape[0]) if implicit else df['rating'].to_numpy() / df['rating'].max()

        X = csr_matrix((values, (users, items)))
        self.X = X

        G = X.T.dot(X).toarray()
        diagIndices = np.diag_indices(G.shape[0])
        G[diagIndices] += lambda_
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        B[diagIndices] = 0

        self.B = B
        self.pred = X.dot(B)

    def predict(self, train, users, items, k):
        df = pd.DataFrame()
        items = self.item_enc.transform(items)
        dd = train.loc[train['userId'].isin(users)]
        dd['ci'] = self.item_enc.transform(dd['movieId'])
        dd['cu'] = self.user_enc.transform(dd['userId'])
        g = dd.groupby('userId')
        for user, group in tqdm(g):
            watched = set(group['ci'])
            candidates = [item for item in items if item not in watched]
            u = group['cu'].iloc[0]
            pred = np.take(self.pred[u, :], candidates)
            res = np.argpartition(pred, -k)[-k:]
            r = pd.DataFrame({
                "userId": [user] * len(res),
                "movieId": np.take(candidates, res),
                "score": np.take(pred, res)
            }).sort_values('score', ascending=False)
            df = df.append(r, ignore_index=True)
        df['movieId'] = self.item_enc.inverse_transform(df['movieId'])
        return df
```

<!-- #region id="KwfcFM2yZvnj" -->
### Training
<!-- #endregion -->

```python id="zEzkqTbWVCzD"
ease = EASE()
ease.fit(train)
```

```python colab={"base_uri": "https://localhost:8080/"} id="5I5877tBU7yB" executionInfo={"status": "ok", "timestamp": 1630834999318, "user_tz": -330, "elapsed": 644, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="dcb8117a-e70e-42b8-d713-4da7dd8cc4e4"
uid = 0
ease.user_enc.inverse_transform([0])[0]
```

```python colab={"base_uri": "https://localhost:8080/"} id="ZdpFQsXgV1Mp" executionInfo={"status": "ok", "timestamp": 1630835007002, "user_tz": -330, "elapsed": 551, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="80c3d15c-99a2-4603-b91f-f5c7bb2045f7"
ease.item_enc.inverse_transform(np.argsort(ease.pred[0]))
```

```python colab={"base_uri": "https://localhost:8080/"} id="d9fCVWvOV2sK" executionInfo={"status": "ok", "timestamp": 1630835012996, "user_tz": -330, "elapsed": 785, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8b52e238-bc6f-47d4-ce62-2e5602f4c4f7"
np.argsort(-ease.pred[0])
```

```python colab={"base_uri": "https://localhost:8080/"} id="j-arIMZHV4e5" executionInfo={"status": "ok", "timestamp": 1630835020095, "user_tz": -330, "elapsed": 508, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="257943c7-262d-4aee-99fa-a97e53ec23fe"
ease.pred[0][np.argsort(-ease.pred[0])]
```

```python colab={"base_uri": "https://localhost:8080/"} id="GTdoYucRU30u" executionInfo={"status": "ok", "timestamp": 1630835023395, "user_tz": -330, "elapsed": 738, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7a35e7f5-8f17-4200-b8bb-a18c5cf8b4d9"
np.unique(train[train['userId']==0]['movieId'])
```

<!-- #region id="pV3bdkthZvnn" -->
### Evaluation
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 451, "referenced_widgets": ["2f741acf5a964291af3d1314627871e7", "508e94b448894be08342221ff5ca5515", "4de8e9ae4b8e4e4c954bec8acd09d992", "c5882e7ec03a4e978341e8406e7ec604", "23ef87acab4e46bf9eb6e11870fbb6fc", "ee3e841f043d404fa1a8a8409cc3f421", "386ab749cd954d36a9c7ecebccf5f2df", "eacf302707954850878334ff6a5c1fae", "ad83565fcfd34dfc824ab4c16ad6cd7a", "f9ef7106318146e89ecbcde0ad71bcd7", "6c3ee40cc767411d98efd6d4af1ca2eb"]} id="01b3WkGrV9Nk" executionInfo={"status": "ok", "timestamp": 1630835080341, "user_tz": -330, "elapsed": 40605, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b6635be8-427b-4694-84bf-90d422c9d30e"
pred = ease.predict(train, train['userId'].unique(), train['movieId'].unique(), 100)
pred
```

```python colab={"base_uri": "https://localhost:8080/", "height": 49} id="ggRtLpiNV_c2" executionInfo={"status": "ok", "timestamp": 1630835091546, "user_tz": -330, "elapsed": 588, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="744a2a4c-d13f-4211-eafa-1dea1e4671ee"
uid = 1
df[(df['userId']==uid) & (df['movieId'].isin(pred[pred['userId']==uid]['movieId']))]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 49} id="QPkVz-gnWCQW" executionInfo={"status": "ok", "timestamp": 1630835091967, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="def7da36-b6ea-4131-a7ac-bbf81ae15a3a"
train[(train['userId']==uid) & (train['movieId'].isin(pred[pred['userId']==uid]['movieId']))]
```

```python id="UcilNIv2WD_n"
for uid in range(942):
    pdf = df[(df['userId']==uid) & (df['movieId'].isin(pred[pred['userId']==uid]['movieId']))]
```

```python colab={"base_uri": "https://localhost:8080/"} id="8WvaXEwZWFTu" executionInfo={"status": "ok", "timestamp": 1630835114633, "user_tz": -330, "elapsed": 726, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3eb88b98-6150-41d2-fad5-c97043016ef2"
ease.pred.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="WudrTlyWZvnq" executionInfo={"status": "ok", "timestamp": 1630835116746, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="528edaa4-2162-45fe-e108-41d0c5de33bb"
train['userId'].unique().shape, train['movieId'].unique().shape, 
```

<!-- #region id="dQUI23C-WWWO" -->
## MultiVAE
<!-- #endregion -->

<!-- #region id="hA2o80xvWlah" -->
### Load data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="7NKMTvN_Wlai" executionInfo={"status": "ok", "timestamp": 1630835218743, "user_tz": -330, "elapsed": 6555, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a5b11636-ea6a-458e-e482-49b72a12d954"
df = load_data('./ml-1m/ratings.dat', threshold=3)
df.head()
```

<!-- #region id="vm3oUHOcWlaj" -->
### Preprocessing
<!-- #endregion -->

```python id="nxvgVlESWlaj"
df = df[df['rating']==1].reset_index(drop=True)
tdf = pd.pivot_table(df, index='userId', values='rating', columns='movieId').fillna(0)

cnt = tdf.sum(1)
df = df[df['userId'].isin(np.where(cnt >= 10)[0])].reset_index(drop=True)
tdf = pd.pivot_table(df, index='userId', values='rating', columns='movieId').fillna(0)
tdf.iloc[:,:] = 0
test_idx = []

for i in tdf.index:
    test_idx += list(np.random.choice(df[df['userId']==i].index, 1))
    
train = df.iloc[list(set(df.index)-set(test_idx)),:]
test = df.iloc[test_idx, :]

for uid, iid in zip(train['userId'].values, train['movieId'].values):
    tdf.loc[uid, iid] = 1

train =  tdf.copy()

def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch, dim), stddev=0.01)
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon
```

```python colab={"base_uri": "https://localhost:8080/"} id="Lr6hS321Wlak" executionInfo={"status": "ok", "timestamp": 1630835304957, "user_tz": -330, "elapsed": 20, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7e9aa5fa-d0ee-47c7-c909-daca50c8b412"
df.shape, train.shape, test.shape
```

<!-- #region id="Q3XcxcpoWlak" -->
### Model architecture
<!-- #endregion -->

```python id="25ZXRGkfXLjd"
class MultVAE(tf.keras.models.Model):
    def __init__(self, input_dim, latent_dim, lamda=1e-4):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.anneal = 0.
        
        self.model = self.build()

    def compile(self, optimizer, loss_fn=None):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        
    def build(self):
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        
        inputs = self.encoder.input
        
        mu, log_var = self.encoder(inputs)
        h = sampling([mu, log_var])
        
        outputs = self.decoder(h)
    
        return Model(inputs, outputs)
    
    def build_encoder(self):
        inputs = Input(shape = (self.input_dim, ))
        h = Dropout(0.2)(inputs)
        
        mu = Dense(self.latent_dim)(h)
        log_var = Dense(self.latent_dim)(h)
        
        return Model(inputs, [mu, log_var])
    
    def build_decoder(self):
        inputs = Input(shape = (self.latent_dim, ))
        
        outputs = Dense(self.input_dim, activation='sigmoid')(inputs)

        return Model(inputs, outputs)
    
    def train_step(self, data):
        x = data
        with tf.GradientTape() as tape:
            mu, log_var = self.encoder(x)
            pred = self.model(x)
            
            kl_loss = tf.reduce_mean(tf.reduce_sum(0.5*(log_var + tf.exp(log_var) + tf.pow(mu, 2)-1), 1, keepdims=True))
            ce_loss = -tf.reduce_mean(tf.reduce_sum(tf.nn.log_softmax(pred) * x, -1))
            
            loss = ce_loss + kl_loss*self.anneal
            
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        
        return {'loss': loss}
    
    def predict(self, data):
        mu, log_var = self.encoder(data)
        return self.decoder(mu)
```

<!-- #region id="zh7u-CfiXNwo" -->
### Training
<!-- #endregion -->

```python id="OMgAqDDAXQ0s"
loader = tf.data.Dataset.from_tensor_slices(train.values.astype(np.float32))
loader = loader.batch(8, drop_remainder=True).shuffle(len(train))

model = MultVAE(train.shape[1], 200)
model.compile(optimizer=tf.optimizers.Adam())
```

```python id="rRtvceiYXXmX"
class AnnealCallback(callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.anneal_cap = 0.3
        
    def on_train_batch_end(self, batch, logs=None):
        self.model.anneal =  min(self.anneal_cap, self.model.anneal+1e-4)
```

```python colab={"base_uri": "https://localhost:8080/"} id="vPj4AythXUo6" executionInfo={"status": "ok", "timestamp": 1630835847202, "user_tz": -330, "elapsed": 429687, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="23552135-8011-439d-a61e-22d007fb7f44"
model.fit(loader, epochs=25, callbacks=[AnnealCallback()])
```

<!-- #region id="OutR1NzaXZ4p" -->
### Evaluation
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 66, "referenced_widgets": ["356fbd553cd74283bd40f67e43b24e2f", "4e71c9f1b0a74ffbae2c04f977efd6a6", "2b4e62ee18cf43a59467818ebb76289e", "39515edfa1324e2e8b3ded22fd6f1384", "a7a16d4486fb4a9ab1ff40b1b61a41a4", "e2d3d44b598242ed84c780fab3cea768", "0f589d8eceff4716a2029420f1da243c", "c0dc1db826314529aadd1884683d4eda", "59269807a50b4aecac1472fdece6a0be", "ef2b338adc7140a8856263406b28c8d3", "388c8dca4a624437a32b04844a53e84d"]} id="EGuXlAcpXtCA" executionInfo={"status": "ok", "timestamp": 1630835990749, "user_tz": -330, "elapsed": 143578, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9dc168b8-fc8d-46f8-ff4f-3aef4f32cc5b"
top_k = 10
np.random.seed(42)

scores = []
for idx, i in tqdm(enumerate(np.random.choice(train.index, 100))):
    item_to_pred = {item: pred for item, pred in zip(train.columns, model.model.predict(train.values)[idx])}
    test_ = test[(test['userId']==i) & (test['rating']==1)]['movieId'].values
    items = list(np.random.choice(list(filter(lambda x: x not in np.argwhere(train.values[idx]).flatten(), item_to_pred.keys())), 100)) + list(test_)
    top_k_items = heapq.nlargest(top_k, items, key=item_to_pred.get)
    
    score = eval_NDCG(test_, top_k_items)
    scores.append(score)
    
np.mean(scores)
```

<!-- #region id="gyrB_wB1aXGf" -->
## DAE
<!-- #endregion -->

<!-- #region id="p27z1a9RaXGg" -->
### Load data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="FUCzohpraXGh" executionInfo={"status": "ok", "timestamp": 1630833675434, "user_tz": -330, "elapsed": 5144, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="28528ff8-0aec-41dd-8355-823359e5f830"
df = load_data('./ml-1m/ratings.dat', threshold=3)
df.head()
```

<!-- #region id="VYGBPKuKaXGj" -->
### Preprocessing
<!-- #endregion -->

```python id="95emN_MmaXGk"
df = df[df['rating']==1].reset_index(drop=True)
tdf = pd.pivot_table(df, index='userId', values='rating', columns='movieId').fillna(0)

cnt = tdf.sum(1)
df = df[df['userId'].isin(np.where(cnt >= 10)[0])].reset_index(drop=True)
tdf = pd.pivot_table(df, index='userId', values='rating', columns='movieId').fillna(0)
tdf.iloc[:,:] = 0

test_idx = []
for i in tdf.index:
    test_idx += list(np.random.choice(df[df['userId']==i].index, 1))
    
train = df.loc[list(set(df.index)-set(test_idx)),:]
test = df.loc[test_idx, :]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="3Qjyrw5SaXGl" executionInfo={"status": "ok", "timestamp": 1630833683218, "user_tz": -330, "elapsed": 31, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6286da55-c46b-42d7-de7e-bcfda08b3227"
df
```

```python colab={"base_uri": "https://localhost:8080/"} id="tFJKSzzfaXGm" executionInfo={"status": "ok", "timestamp": 1630833683220, "user_tz": -330, "elapsed": 27, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bc4ae887-d201-46be-be65-6fab4a39d805"
df.shape, train.shape, test.shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 470} id="24m_wWovaXGo" executionInfo={"status": "ok", "timestamp": 1630833732782, "user_tz": -330, "elapsed": 49579, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0bc3342b-926b-41be-c4f6-19764a138450"
for uid, iid in zip(train['userId'].values, train['movieId'].values):
    tdf.loc[uid, iid] = 1
train =  tdf.copy()
train
```

<!-- #region id="jJQnYRyGaXGp" -->
### Model architecture
<!-- #endregion -->

```python id="k0osEM5vaXGq"
class DAE(tf.keras.models.Model):
    def __init__(self, input_dim, latent_dim, lamda=1e-4):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lamda = lamda
        self.model = self.build()
        
    def compile(self, optimizer, loss_fn=None):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        
    def build(self):
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        inputs = self.encoder.input
        outputs = self.decoder(self.encoder(inputs))
        
        return Model(inputs, outputs)
    
    def build_encoder(self):
        inputs = Input(shape = (self.input_dim, ))
        
        encoder = Sequential()
        encoder.add(Dropout(0.2))
        encoder.add(Dense(self.latent_dim, activation='tanh'))
        
        outputs = encoder(inputs)
        
        return Model(inputs, outputs)
    
    def build_decoder(self):
        inputs = Input(shape = (self.latent_dim, ))
        
        encoder = Sequential()
        encoder.add(Dense(self.input_dim, activation='sigmoid'))
        
        outputs = encoder(inputs)
        
        return Model(inputs, outputs)
    
    def train_step(self, x):
        with tf.GradientTape() as tape:
            pred = self.model(x)
            
            rec_loss = tf.losses.binary_crossentropy(x, pred)
            loss = rec_loss

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        
        return {'loss': loss}
```

<!-- #region id="0Nx3cmNKaXGr" -->
### Training
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="_O0JYFMeaXGr" executionInfo={"status": "ok", "timestamp": 1630833850254, "user_tz": -330, "elapsed": 113676, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="00b48b7c-859a-478d-f0a9-4349db9ca679"
loader = tf.data.Dataset.from_tensor_slices(train.values)
loader = loader.batch(32, drop_remainder=True).shuffle(len(df))
model = DAE(train.shape[1], 200)
model.compile(optimizer=tf.optimizers.Adam())
model.fit(loader, epochs = 25)
```

<!-- #region id="UG7OWzoCaXGt" -->
### Evaluation
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 66, "referenced_widgets": ["54ec6362c43f4c9fb0da5fae155dc21e", "b5bae646ec9141fa858b0f24e55c3919", "26275847d59a42ef9a5199401a59e6d5", "f4e0639d4d194b15aeeb3e8fcf17b0ee", "79e2ba356105419aba51015935b1206c", "a569e072c0dc409eb91f27bab7ad4c4f", "4276f2e7a5aa4042ac0fea7a4433341a", "e0c831a414ab443ab1b2d5053d34ca43", "55d8f25f6840463689954ae7061546cf", "bb840c41a45e44b78d9dffb837ef2872", "ac9ba7893cbd4c44aacd670a5ab2ac1b"]} id="zaBPGF_paXGt" executionInfo={"status": "ok", "timestamp": 1630836102534, "user_tz": -330, "elapsed": 131749, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a05cf1da-f629-41b6-8a38-755acaf0ad39"
top_k = 10
np.random.seed(42)

scores = []
for idx, i in tqdm(enumerate(np.random.choice(train.index, 100))):
    item_to_pred = {item: pred for item, pred in zip(train.columns, model.model.predict(train.values)[idx])}
    test_ = test[(test['userId']==i) & (test['rating']==1)]['movieId'].values
    items = list(np.random.choice(list(filter(lambda x: x not in np.argwhere(train.values[idx]).flatten(), item_to_pred.keys())), 100)) + list(test_)
    top_k_items = heapq.nlargest(top_k, items, key=item_to_pred.get)
    
    score = eval_NDCG(test_, top_k_items)
    scores.append(score)
    
np.mean(scores)
```

<!-- #region id="3WzMlytjXENu" -->
## RecVAE
<!-- #endregion -->

<!-- #region id="3JGgb4cpX-8S" -->
### Load data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="iJY1toVZX-8S" executionInfo={"status": "ok", "timestamp": 1630835582915, "user_tz": -330, "elapsed": 6008, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5d82f1ef-db9e-4ce1-a201-ec5f3ed4b04e"
df = load_data('./ml-1m/ratings.dat', threshold=3)
df.head()
```

<!-- #region id="_er5KwLBX-8T" -->
### Preprocessing
<!-- #endregion -->

```python id="STkKDPOiX-8U"
df = df[df['rating']==1].reset_index(drop=True)
tdf = pd.pivot_table(df, index='userId', values='rating', columns='movieId').fillna(0)

cnt = tdf.sum(1)
df = df[df['userId'].isin(np.where(cnt >= 10)[0])].reset_index(drop=True)
tdf = pd.pivot_table(df, index='userId', values='rating', columns='movieId').fillna(0)
tdf.iloc[:,:] = 0
test_idx = []
for i in tdf.index:
    test_idx += list(np.random.choice(df[df['userId']==i].index, 1))
    
train = df.iloc[list(set(df.index)-set(test_idx)),:]
test = df.iloc[test_idx, :]

for uid, iid in zip(train['userId'].values, train['movieId'].values):
    tdf.loc[uid, iid] = 1
train =  tdf.copy().astype(np.float32)

loader = tf.data.Dataset.from_tensor_slices(train.values.astype(np.float32))
loader = loader.batch(8, drop_remainder=True).shuffle(len(train))
```

<!-- #region id="6ZDpx3i9X-8V" -->
### Model architecture
<!-- #endregion -->

```python id="axGJNldkYLEe"
def log_norm_pdf(x, mu, logvar):
    return -0.5*(logvar + tf.math.log(2 * np.pi) + tf.pow((x - mu), 2) / tf.exp(logvar))

def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch, dim), stddev=0.01)
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon
```

```python id="48AlS5OtX-8W"
class CompositePrior(tf.keras.models.Model):
    def __init__(self, x_dim, latent_dim, mixture_weights = [3/20, 15/20, 2/20]):
        super().__init__()
        self.encoder_old = Encoder(x_dim, latent_dim, dropout_rate=0)
        self.latent_dim = latent_dim
        self.mixture_weights = mixture_weights
        
        self.mu_prior = self.add_weight(shape=(self.latent_dim, ), initializer = tf.zeros_initializer(), trainable=False)
        self.logvar_prior  = self.add_weight(shape=(self.latent_dim, ), initializer = tf.zeros_initializer(), trainable=False)
        self.logvar_unif_prior = self.add_weight(shape=(self.latent_dim, ), initializer = tf.constant_initializer(10), trainable=False)
        
    def call(self, x, z):
        post_mu, post_logvar = self.encoder_old(x)
        
        stnd_prior = log_norm_pdf(z, self.mu_prior, self.logvar_prior)
        post_prior = log_norm_pdf(z, post_mu, post_logvar)
        unif_prior = log_norm_pdf(z, self.mu_prior, self.logvar_unif_prior)
        
        gaussians = [stnd_prior, post_prior, unif_prior]
        gaussians = [g+tf.math.log(w) for g, w in zip(gaussians, self.mixture_weights)]
        
        density = tf.stack(gaussians, -1)
        return tf.math.log(tf.reduce_sum(tf.exp(density), -1)) # logsumexp
```

```python id="CQG9eIBxYPWl"
class Encoder(tf.keras.models.Model):
    def __init__(self, x_dim, latent_dim, dropout_rate = 0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.x_dim = x_dim
        self.dropout_rate = dropout_rate
        self.model = self.build_model()
        
    def build_model(self): # now just shallow net
        x_in = Input(shape=(self.x_dim, ))
        
        h = Dense(1024, activation='relu')(x_in)
        mu = Dense(self.latent_dim)(h)
        logvar = Dense(self.latent_dim)(h)
        
        return Model(x_in, [mu, logvar])
        
    def call(self, x):
        norm = tf.sqrt(tf.reduce_sum(tf.pow(x, 2), -1, keepdims=True))
        x = x/norm
        if self.dropout_rate>0:
            x = Dropout(self.dropout_rate)(x)
        
        return self.model(x)

class RecVAE(tf.keras.models.Model):
    def __init__(self, x_dim, latent_dim):
        super().__init__()
        
        self.encoder = Encoder(x_dim, latent_dim)
        self.decoder = Dense(x_dim)
        self.prior = CompositePrior(x_dim, latent_dim)
        
    def call(self, data):
        mu, logvar = self.encoder(data)
        z = sampling([mu, logvar])
        recon = self.decoder(z)
        
        return mu, logvar, z, recon
    
    def predict(self, data):
        mu, logvar = self.encoder(data)
        z = sampling([mu, logvar])
        recon = self.decoder(z)
        
        return recon
    
    def update_prior(self):
        self.prior.encoder_old.set_weights(self.encoder.get_weights())
```

<!-- #region id="HdkKDWyXX-8W" -->
### Training
<!-- #endregion -->

```python id="peb46Hd9X-8W"
def tf_train(model, loader, optimizer, target, gamma=1.):
    total_loss = 0.
    for x in loader:
        norm = tf.reduce_sum(x, -1, keepdims=True)
        kl_weight = gamma*norm
        
        with tf.GradientTape() as tape:
            mu, logvar, z, pred = model(x)
            
#             kl_loss = tf.reduce_mean(tf.reduce_sum(0.5*(logvar + tf.exp(logvar) + tf.pow(mu, 2)-1), 1, keepdims=True))
            kl_loss = tf.reduce_mean(log_norm_pdf(z, mu, logvar) - tf.multiply(model.prior(x, z), kl_weight))
            ce_loss = -tf.reduce_mean(tf.reduce_sum(tf.nn.log_softmax(pred) * x, -1))
            
            loss = ce_loss + kl_loss*kl_weight
            
        if target == 'encoder':
            grads = tape.gradient(loss, model.encoder.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.encoder.trainable_weights))
        else:
            grads = tape.gradient(loss, model.decoder.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.decoder.trainable_weights))
            
        total_loss += tf.reduce_sum(loss)
    return total_loss
```

```python id="FzTZPD9CYfpj"
epochs = 25

model = RecVAE(train.shape[1], 200)
enc_opt = optimizers.Adam()
dec_opt = optimizers.Adam()

for e in range(epochs):
    # alternating 
    ## train step
    tf_train(model, loader, enc_opt, 'encoder')
    model.update_prior()
    tf_train(model, loader, dec_opt, 'decoder')
    ## eval step
```

<!-- #region id="bhZuJedDX-8X" -->
### Evaluation
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 66, "referenced_widgets": ["87c9bc69718d40c0acd55be8b3d028c3", "592a360bb8f74fd690223f2e4bb14f0e", "291c4790efeb43b68b78eaef6a99ced7", "8d0ed0b8a3e94733aad8d23ae0265d3c", "4e554fb4f30a48faa6859d457d08ba1e", "32727136c19e46558e6016eea4fa6fec", "fa3e40aea0f14e4db3be250b51c8ede0", "7a486090a89343168b6c82943865733c", "57b0ddec4be6490b8da427622de4ebac", "2d1191aacad24eba93d354e26e6ee37b", "9940ceebfb384b6da8e6d5deffcce3a4"]} id="KgYQGSPBYUCb" executionInfo={"status": "ok", "timestamp": 1630838390198, "user_tz": -330, "elapsed": 184666, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b760dcd0-9831-49d3-c3ee-3ab95dc24eda"
top_k = 10
np.random.seed(42)

scores = []
for idx, i in tqdm(enumerate(np.random.choice(train.index, 100))):
    item_to_pred = {item: pred.numpy() for item, pred in zip(train.columns, model.predict(train.values)[idx])}
    test_ = test[(test['userId']==i) & (test['rating']==1)]['movieId'].values
    items = list(np.random.choice(list(filter(lambda x: x not in np.argwhere(train.values[idx]).flatten(), item_to_pred.keys())), 100)) + list(test_)
    top_k_items = heapq.nlargest(top_k, items, key=item_to_pred.get)
    
    score = eval_NDCG(test_, top_k_items)
    scores.append(score)
#     break
np.mean(scores)
```

<!-- #region id="RAaqLy1UtcIC" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Jit1oP3jtd7k" executionInfo={"status": "ok", "timestamp": 1639716362410, "user_tz": -330, "elapsed": 4112, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8470ae76-a435-4a4f-f909-351ed5e37fed"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="qrYL9Jx-tcIF" -->
---
<!-- #endregion -->

<!-- #region id="pZR6MBOZtcIG" -->
**END**
<!-- #endregion -->
