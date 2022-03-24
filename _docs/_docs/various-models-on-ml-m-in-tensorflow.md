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

<!-- #region id="bklQwz_fyqtk" -->
# Various models on ML-1m in Tensorflow
<!-- #endregion -->

<!-- #region id="B7cCmTtsYo0T" -->
## Load dependencies
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="gPKHf71guxg_" executionInfo={"status": "ok", "timestamp": 1638118410361, "user_tz": -330, "elapsed": 5455, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e2aca490-387d-418c-9a65-2d080e6518b0"
!pip install -q tensorflow_addons
```

```python colab={"base_uri": "https://localhost:8080/"} id="hFAyisZrdOgW" executionInfo={"status": "ok", "timestamp": 1638118411227, "user_tz": -330, "elapsed": 877, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8f5e80b8-c859-4cc0-c314-250160451a6c"
!wget -q --show-progress -O ratings.dat https://github.com/RecoHut-Datasets/movielens_1m/raw/main/ml1m_ratings.dat
```

```python id="Qvj6T0wNXhdZ"
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import os,sys,inspect
import gc
from tqdm.notebook import tqdm
import random
import heapq

from sklearn.metrics import precision_score, recall_score,  roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from tensorflow import keras
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import optimizers, callbacks, layers, losses
from tensorflow.keras.layers import Dense, Concatenate, Activation, Add, BatchNormalization, Dropout, Input, Embedding, Flatten, Multiply
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

import warnings
warnings.filterwarnings('ignore')
```

```python id="e0fQRwwYX_NG"
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

```python id="y7VRxuiUYLzO"
def load_data(filepath, threshold=0):
    df = pd.read_csv(filepath,
                     sep="\t",
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

```python id="MRXY87gWs_Xf"
tf.config.experimental.set_memory_growth(gpus[0], True)
```

```python colab={"base_uri": "https://localhost:8080/"} id="EpIyG5zgtaai" executionInfo={"status": "ok", "timestamp": 1638118428748, "user_tz": -330, "elapsed": 23, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4d852851-cd37-41b6-aa92-dd3db00df3de"
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

```python id="4DbVJCtKPuBF"
def mish(x):
    return x*tf.math.tanh(tf.math.softplus(x))

def leakyrelu(x, factor=0.2):
    return tf.maximum(x, factor*x)
```

<!-- #region id="S78F5a7AYgQz" -->
## NeuMF
<!-- #endregion -->

<!-- #region id="1sAQYk5eYv1j" -->
### Load data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="rCg36MGFTue5" executionInfo={"status": "ok", "timestamp": 1638118438771, "user_tz": -330, "elapsed": 4553, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6153a39c-2e3b-4c33-de05-0ed13cbe60bb"
df = load_data('ratings.dat')
df.head()
```

```python id="Pr9OvQ4Jg35r"
uuid = df['userId'].unique()
uiid = df['movieId'].unique()
```

<!-- #region id="uo-H-H5yZjSI" -->
### Preprocessing
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 113, "referenced_widgets": ["eb45f9b9baed4cc39ebc119d251a9f5c", "76701e445b12402d8ce0060363973ed2", "10073156f51044d48d74b8cc628e0dcc", "ab7d1c15851f49dbb9c49760878e2316", "439d8b0a88cf4ba5b6a75bdacee1f501", "f860e82c6fa347e4bed821451201bd35", "9addcb1ad2004ecda0fff57e7a47dd9c", "3ce0a6e67c6d489a877bb328a77de456", "bf937f4e2a79403cbb56b9944dccd854", "2edd2b91b21b43a2ad2ddf28e321618b", "736c1bc31c9c47a98a052273acd1cf69", "d42608df350441dc8d58f27f116844d9", "ec675481b8264301aee6eb4a07fce708", "334c645175f24763b405d8c22188fc04", "dadda1302e624e7ca8b7bf40c85f5b2b", "fcf5871cc716462a8e3d30af073708bb", "cce0bbcc7fef4adcbff295d918feada3", "1e2f40e931fb400d9544ad1b970a9670", "a8da8b66a52f46c8a296f820c7ddf3ca", "ffac85fc959b4275b22a90558b71b993", "4cb05edcd65a4b17a52f16667f6f2e67", "bf20ebd5e90241fd98014119bd12d434", "a4f4e592484f4824bf058aa0836bda96", "4e8be57da414433084a23f204e311c69", "deb58aa02fe74575ad92b4bf2e4b4371", "0af13e04222140f5885202f7b9fbdd13", "3ec731c921be48a9bc780216e5c72a2b", "87945dd5f3514e93b5139eedf3b109b9", "55023f1c07ae4233968d662aa4721a1f", "aca1ff3d1f1a4693a7ebcf31c2a679f2", "d4b8fd274eb545e585fd8735bddbbc26", "626dca090cd74246856974a78264d3ad", "1f93316e32ff48c4bb4e60715b85f651"]} id="Tf5XXE0RZkgs" executionInfo={"status": "ok", "timestamp": 1630488548941, "user_tz": -330, "elapsed": 463586, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a232b81c-d277-40c5-92ad-aeecf566da10"
# Negative Resampling
df = add_negative(df, uiid, 4)

# Reshuffling
idx = list(df.index)
np.random.shuffle(idx)
df = df.loc[idx]

# Train/Val/Test Split
rtd = extract_from_df(df, 2, 0)
train = df.drop(rtd)
val = df.loc[rtd]
rtd = extract_from_df(val, 1, 0)
test = val.drop(rtd)
val = val.loc[rtd]
```

```python colab={"base_uri": "https://localhost:8080/"} id="Ov0ZmaZabiVe" executionInfo={"status": "ok", "timestamp": 1630488662830, "user_tz": -330, "elapsed": 716, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="56680693-12e6-4b83-a1d2-74d9f441610a"
df.shape, train.shape, val.shape, test.shape
```

<!-- #region id="_br-gyTNd40x" -->
### Model architecture
<!-- #endregion -->

```python id="UK-_v59OYjHz"
def build_gmf(u_dim, i_dim, gmf_dim):
    u_input = Input(shape=(1, ))
    i_input = Input(shape=(1, ))

    u_emb_gmf = Flatten()(Embedding(u_dim, gmf_dim, input_length=u_input.shape[1])(u_input))
    i_emb_gmf = Flatten()(Embedding(i_dim, gmf_dim, input_length=i_input.shape[1])(i_input))

    # gmf
    mul = Multiply()([u_emb_gmf, i_emb_gmf])

    out = Dense(1)(mul)

    return Model([u_input, i_input], out, name='GMF')

def build_mlp(u_dim, i_dim, mlp_dim):
    u_input = Input(shape=(1, ))
    i_input = Input(shape=(1, ))
    
    u_emb_mlp = Flatten()(Embedding(u_dim, mlp_dim, input_length=u_input.shape[1])(u_input))
    i_emb_mlp = Flatten()(Embedding(i_dim, mlp_dim, input_length=i_input.shape[1])(i_input))

    # mlp
    concat = Concatenate()([u_emb_mlp, i_emb_mlp])
    h = Dense(128, activation='relu')(concat)
    h = Dense(64, activation='relu')(h)
    h = Dropout(0.2)(h)

    out = Dense(1)(h)

    return Model([u_input, i_input], out, name='MLP')

def build_nmf(u_dim, i_dim, gmf_dim, mlp_dim):
    u_input = Input(shape=(1, ))
    i_input = Input(shape=(1, ))

    u_emb_gmf = Flatten()(Embedding(u_dim, gmf_dim, input_length=u_input.shape[1])(u_input))
    i_emb_gmf = Flatten()(Embedding(i_dim, gmf_dim, input_length=i_input.shape[1])(i_input))

    u_emb_mlp = Flatten()(Embedding(u_dim, mlp_dim, input_length=u_input.shape[1])(u_input))
    i_emb_mlp = Flatten()(Embedding(i_dim, mlp_dim, input_length=i_input.shape[1])(i_input))

    # gmf
    mul = Multiply()([u_emb_gmf, i_emb_gmf])

    # mlp
    concat = Concatenate()([u_emb_mlp, i_emb_mlp])
    h = Dense(128, activation='relu')(concat)
    h = Dense(64, activation='relu')(h)
    h = Dropout(0.2)(h)

    con = Concatenate()([mul, h])
    out = Dense(1)(con)

    return Model([u_input, i_input], out, name='NMF')
```

<!-- #region id="-R-qDHX_dD3T" -->
### Training
<!-- #endregion -->

```python id="Fjtaq3RkcIO6"
class ValCallback(callbacks.Callback):
    def __init__(self):
        self.best_score = 0
        self.bw = None
        self.score_hist = []
        self.score_hist_ = []
    
    def on_epoch_end(self, epochs, logs=None):
        if epochs%1 == 0:
            uid = np.random.choice(uuid, int(len(uuid)*0.3), replace=False)
            hits = list(map(eval_hit_wrapper(self.model, train, val, uiid, 10), uid))
            score = sum(hits)
            self.score_hist.append(score)
            self.score_hist_.append(score/len(hits))
            if self.best_score < score:
                self.bw = self.model.get_weights()
                self.best_score = score
```

```python id="RyfEAn2ucBtX"
gmf = build_gmf(len(uuid), len(uiid), 16)
gmf.compile(optimizer=optimizers.Adam(2e-4), loss=losses.BinaryCrossentropy(from_logits=True))

gmf_cb = ValCallback()

hist_gmf = gmf.fit([train['userId'].values, train['movieId'].values],
                   train['rating'].values,
                   shuffle=True,
                   epochs=10,
                  callbacks=[gmf_cb],
                  validation_split=0.1)
```

```python id="Dcc2G_aGcN3_"
mlp = build_mlp(len(uuid), len(uiid), 16)
mlp.compile(optimizer=optimizers.Adam(2e-4), loss=losses.BinaryCrossentropy(from_logits=True))

mlp_cb = ValCallback()

hist_mlp = mlp.fit([train['userId'].values, train['movieId'].values],
                   train['rating'].values,
                   shuffle=True,
                   epochs=10,
                  callbacks=[mlp_cb],
                  validation_split=0.1)
```

```python id="fDfTVfWtcc37"
nmf = build_nmf(len(uuid), len(uiid), 16, 16)
nmf.compile(optimizer=optimizers.Adam(2e-4), loss=losses.BinaryCrossentropy(from_logits=True))

nmf_cb = ValCallback()

hist_nmf = nmf.fit([train['userId'].values, train['movieId'].values],
                   train['rating'].values,
                   shuffle=True,
                   epochs=10,
                  callbacks=[nmf_cb],
                  validation_split=0.1)
```

<!-- #region id="NfDBp4XndBDM" -->
### Losses
<!-- #endregion -->

```python id="GSqehAi7c7tf"
plt.plot(hist_gmf.history['loss'], label='GMF')
plt.plot(hist_mlp.history['loss'], label='MLP')
plt.plot(hist_nmf.history['loss'], label='NMF')
plt.legend()
```

```python id="bepmgCCGc6Ir"
plt.plot(gmf_cb.score_hist, label='GMF')
plt.plot(mlp_cb.score_hist, label='MLP')
plt.plot(nmf_cb.score_hist, label='NMF')
plt.legend()
```

<!-- #region id="L7yzZdAncyNp" -->
### Hit@10
<!-- #endregion -->

```python id="bc8oyZsBc07x"
# GMF
hits_gmf10 = list(map(eval_hit_wrapper(gmf, train, test, uiid, 10), uuid))
print(sum(hits_gmf10)/len(hits_gmf10))

# MLP
hits_mlp10 = list(map(eval_hit_wrapper(mlp, train, test, uiid, 10), uuid))
print(sum(hits_mlp10)/len(hits_mlp10))

# NMF
hits_nmf10 = list(map(eval_hit_wrapper(nmf, train, test, uiid, 10), uuid))
print(sum(hits_nmf10)/len(hits_nmf10))
```

<!-- #region id="tML9Vcrjc2ip" -->
### NDCG@10
<!-- #endregion -->

```python id="D1q-E2YkcdqT"
# GMF
ndcg_gmf10 = list(map(eval_NDCG_wrapper(gmf, train, test, uiid, 10), uuid))
print(sum(ndcg_gmf10)/len(ndcg_gmf10))

# MLP
ndcg_mlp10 = list(map(eval_NDCG_wrapper(mlp, train, test, uiid, 10), uuid))
print(sum(ndcg_mlp10)/len(ndcg_mlp10))

# NMF
ndcg_nmf10 = list(map(eval_NDCG_wrapper(nmf, train, test, uiid, 10), uuid))
print(sum(ndcg_nmf10)/len(ndcg_nmf10))
```

<!-- #region id="c-FehElNdZbO" -->
## AFM
<!-- #endregion -->

<!-- #region id="RAZkYEiGfhHE" -->
### Abstract
<!-- #endregion -->

<!-- #region id="fflTHIrqfnDq" -->
[Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](https://www.ijcai.org/proceedings/2017/0435.pdf)
<!-- #endregion -->

<!-- #region id="3marmjhcf8to" -->
<!-- #endregion -->

<!-- #region id="NmeVwOfqd-1T" -->
### Load data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="hLyHCnqrd-1d" executionInfo={"status": "ok", "timestamp": 1630488902663, "user_tz": -330, "elapsed": 8173, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e7a2506f-1eac-4c8b-ea0a-435d02d431f6"
df = load_data('ratings.dat')
df.head()
```

```python id="3L3jJxFTd-1f"
uuid = df['userId'].unique()
uiid = df['movieId'].unique()
```

<!-- #region id="K_vafvLfd-1g" -->
### Preprocessing
<!-- #endregion -->

```python id="FkKNo_j-eEAg"
train, test = train_test_split(df, test_size=0.15, random_state=SEED, stratify=df['userId'].values)
```

<!-- #region id="YhWpuDhse3jS" -->
### Model architecture
<!-- #endregion -->

```python id="KsqeaVvce5pA"
class AFM_layer(keras.Model):
    def __init__(self, x_dims, latent_dim, att_dim, l2_emb=1e-4):
        super(AFM_layer, self).__init__()
        self.x_dims = x_dims
        self.latent_dim = latent_dim
        self.att_dim = att_dim
        
        
        self.embedding = Embedding(sum(x_dims)+1, latent_dim, input_length=1, embeddings_regularizer=l2(l2_emb))
        
        self.linear = Dense(1)
        self.att_dense1 = Dense(att_dim)
        self.att_dense2 = Dense(1)
        self.final_dense = Dense(1)

    def call(self, inputs):
        cat_ = [tf.squeeze(tf.one_hot(feat, self.x_dims[i]), 1) for i, feat in enumerate(inputs)]
        X_cat = tf.concat(cat_, 1)
        X = tf.concat(inputs, 1)

        linear_terms = self.linear(X_cat)
        non_zero_emb = self.embedding(X + tf.constant((0, *np.cumsum(self.x_dims)))[:-1])

        n = len(self.x_dims)
        r = []; c = []
        for i in range(n-1):
            for j in range(i+1, n):
                r.append(i), c.append(j)
        p = tf.gather(non_zero_emb, r, axis=1)
        q = tf.gather(non_zero_emb, c, axis=1)
        pairwise = p*q
        
        att_score = tf.nn.relu(self.att_dense1(pairwise))
        att_score = tf.nn.softmax(self.att_dense2(att_score), axis=1)

        att_output = tf.reduce_sum(att_score * pairwise, axis=1)

        att_output = self.final_dense(att_output)
        
        y_hat = att_output + linear_terms

        return y_hat
```

```python id="I_NLszZDe-4c"
class AFM(tf.keras.Model):
    def __init__(self, x_dim, latnt_dim, att_dim):
        super(AFM, self).__init__()
        self.afm = AFM_layer(x_dim, latnt_dim, att_dim)

    def call(self, inputs):
        outputs = self.afm(inputs)
#         outputs = tf.nn.sigmoid(outputs)
        return outputs
```

<!-- #region id="Pxz_faJJfC_x" -->
### Training
<!-- #endregion -->

```python id="pH78n5D3fD0v"
afm = AFM((len(uuid), len(uiid)), 8, 8)
afm.compile(loss=losses.BinaryCrossentropy(from_logits=True), optimizer=optimizers.Adam())
afm.fit([train['userId'].values.astype(np.int32), train['movieId'].values.astype(np.int32)], 
       train['rating'].values,
      epochs=10,
      shuffle=True,
      validation_split=0.1)
```

<!-- #region id="8AGd2L03fHrl" -->
### Evaluation
<!-- #endregion -->

```python id="GNYhM-3UfItM"
pred = afm.predict([test['userId'].values.astype(np.int32), test['movieId'].values.astype(np.int32)])
np.sum(np.where(pred>0., 1, 0).flatten() == test['rating'].values) / len(pred)
```

```python id="vUUlYXj9fMO9"
print(roc_auc_score(test['rating'].values, pred))
print(precision_score(test['rating'].values, np.where(pred>0., 1, 0)))
print(recall_score(test['rating'].values, np.where(pred>0., 1, 0)))
```

<!-- #region id="j4FvXUq0gM9m" -->
## AutoInt
<!-- #endregion -->

<!-- #region id="0fbeasMZgO78" -->
### Abstract
<!-- #endregion -->

<!-- #region id="F5s9JDcAgViO" -->
[AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/pdf/1810.11921.pdf)
<!-- #endregion -->

<!-- #region id="vTNGvcrAgmVI" -->
<!-- #endregion -->

<!-- #region id="RlACV6jVg1VX" -->
### Load data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="rufSlGREg1VY" executionInfo={"status": "ok", "timestamp": 1630489325868, "user_tz": -330, "elapsed": 5942, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="66d22fb5-75e8-49ca-b0fd-855ce50c0667"
df = load_data('ratings.dat')
df.head()
```

```python id="hN5WYRJLg1VZ"
uuid = df['userId'].unique()
uiid = df['movieId'].unique()
```

<!-- #region id="5ziISY0Wg1Va" -->
### Preprocessing
<!-- #endregion -->

```python id="Wub7F9iUg1Vb"
train, test = train_test_split(df, test_size=0.15, random_state=SEED, stratify=df['userId'].values)
```

```python colab={"base_uri": "https://localhost:8080/"} id="KJHkyf_Yg9MX" executionInfo={"status": "ok", "timestamp": 1630489328856, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f4dce34f-76f9-4639-a723-7a13034ca5bc"
tr_X = np.stack([train['userId'].values.astype(np.int32), train['movieId'].values.astype(np.int32)], 1)
test_X = np.stack([test['userId'].values.astype(np.int32), test['movieId'].values.astype(np.int32)], 1)

tr_X.shape, test_X.shape
```

<!-- #region id="gy-fHQ9Eg1Vc" -->
### Model architecture
<!-- #endregion -->

```python id="N7EAGXXDg3wT"
class MHA(layers.Layer):
    def __init__(self, emb_size, head_num, use_resid=True):
        super(MHA, self).__init__()
        
        self.emb_size = emb_size
        self.head_num = head_num
        self.use_resid = use_resid
        
        self.flatten = Flatten()
        
        self.att = tfa.layers.MultiHeadAttention(emb_size, head_num)
        
    def build(self, input_shape):
        units = self.emb_size * self.head_num
        
        self.W_q = Dense(units)
        self.W_k = Dense(units)
        self.W_v = Dense(units)
        if self.use_resid:
            self.W_res = Dense(units)
            
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        q = self.W_q(inputs)
        k = self.W_k(inputs)
        v = self.W_v(inputs)
        
        out = self.att([q, k, v])

        if self.use_resid:
            out = out + self.W_res((inputs))
            
        out = tf.nn.relu(out)
        
        return out
```

```python id="mrzX0enUhnX9"
class AutoInt(tf.keras.Model):
    def __init__(self, x_dims, latent_dim, att_sizes, att_heads, l2_emb=1e-4):
        super(AutoInt, self).__init__()
        
        self.x_dims = x_dims
        self.latent_dim = latent_dim

        self.embedding = Embedding(sum(x_dims)+1, latent_dim, input_length=1, embeddings_regularizer=l2(l2_emb))
        
        self.linear = Dense(1)
        
        self.att_layers = [MHA(a, h) for a, h in zip(att_sizes, att_heads)]
        
        self.flatten =  Flatten()
        
    def call(self, inputs):
        emb = self.embedding(inputs + tf.constant((0, *np.cumsum(self.x_dims)))[:-1])
        
        att = emb
        for att_layer in self.att_layers:
            att = att_layer(att)
        
        out = self.linear(self.flatten(att))
        
        return out
```

<!-- #region id="YXnHBM1BhrBI" -->
### Training
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="-_Z_Ya_fhtm6" outputId="af79f449-e783-40f9-a5da-d9f51d1163b7"
autoint = AutoInt((len(uuid), len(uiid)), 8, [8, 8], [2, 2])
autoint.compile(loss=losses.BinaryCrossentropy(from_logits=True), 
            optimizer=optimizers.Adam(2e-4))
hist = autoint.fit(tr_X, 
           train['rating'].values,
          epochs=10,
          shuffle=True,
          validation_split=0.1)
```

<!-- #region id="ChL8rMz9hx5q" -->
### Evaluation
<!-- #endregion -->

```python id="LmH0iNKch1DN"
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.show()
```

```python id="kbfA9k_1h2CE"
pred = autoint.predict(test_X)
np.sum(np.where(pred>0., 1, 0).flatten() == test['rating'].values) / len(pred)
```

```python id="Tk4bHOGihy9q"
print(roc_auc_score(test['rating'].values, pred))
print(precision_score(test['rating'].values, np.where(pred>0., 1, 0)))
print(recall_score(test['rating'].values, np.where(pred>0., 1, 0)))
```

<!-- #region id="XSVWv303iKOf" -->
## CDN
<!-- #endregion -->

<!-- #region id="1mwtUbKeiK5D" -->
### Abstract
<!-- #endregion -->

<!-- #region id="9YLsdcGjiM02" -->
[A hybrid recommender system for suggesting CDN (content delivery network)](https://github.com/lucashu1/CDN-RecSys)
<!-- #endregion -->

<!-- #region id="9-DVDSjiiyPR" -->
### Load data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="UL_iLSbUiyPS" executionInfo={"status": "ok", "timestamp": 1630486240558, "user_tz": -330, "elapsed": 6770, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1a0c8f44-caed-4d12-e9c9-4bae0e347cad"
df = load_data('ratings.dat')
df.head()
```

```python id="Y2L4c8N4iyPS"
uuid = df['userId'].unique()
uiid = df['movieId'].unique()
```

<!-- #region id="5IcLzyssiyPT" -->
### Preprocessing
<!-- #endregion -->

```python id="3XKY_tniiyPT"
train, test = train_test_split(df, test_size=0.15, random_state=SEED, stratify=df['userId'].values)
```

```python colab={"base_uri": "https://localhost:8080/"} id="ftTKK4amiyPT" executionInfo={"status": "ok", "timestamp": 1630486241405, "user_tz": -330, "elapsed": 856, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="eb5f4425-1326-4a3e-b9b1-7971cc6409a9"
tr_X = np.stack([train['userId'].values.astype(np.int32), train['movieId'].values.astype(np.int32)], 1)
test_X = np.stack([test['userId'].values.astype(np.int32), test['movieId'].values.astype(np.int32)], 1)

tr_X.shape, test_X.shape
```

<!-- #region id="NTeRNqGriyPU" -->
### Model architecture
<!-- #endregion -->

```python id="jnaunuYRi4yn"
class CrossNetwork(layers.Layer):
    def __init__(self, n_layers):
        super(CrossNetwork, self).__init__()
        self.n_layers = n_layers
    
    def build(self, input_shape):
        dim = input_shape[-1]
        self.cross_weights = [self.add_weight(shape=(dim, 1), 
                                        initializer=tf.random_normal_initializer(),
                                       trainable=True,
                                       name=f'cross_weight_{i}') for i in range(self.n_layers)]
    
        self.cross_biases = [self.add_weight(shape=(dim, 1),
                                      initializer=tf.random_normal_initializer(),
                                      trainable=True,
                                      name=f'cross_bias_{i}') for i in range(self.n_layers)]
    def call(self, inputs):
        x_0 = tf.expand_dims(inputs, -1)
        x_l = x_0
        for i in range(self.n_layers):
            x_l1 = tf.tensordot(x_l, self.cross_weights[i], axes=[1, 0])
            x_l = tf.matmul(x_0, x_l1) + self.cross_biases[i]
            
        x_l = tf.squeeze(x_l, -1)
        
        return x_l

class DeepNetwork(layers.Layer):
    def __init__(self, units, activation='relu'):
        super(DeepNetwork, self).__init__()
        
        self.layers = [Dense(unit, activation=activation) for unit in units]
    
    def call(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
            
        return x
```

```python id="clNYzCoUjicp"
class DCN(Model):
    def __init__(self, x_dims, latent_dim, n_cross_layers, hidden_units, activation='relu', l2_emb=1e-4):
        super(DCN, self).__init__()
        self.x_dims = x_dims
        self.latent_dim = latent_dim
        
        self.cross_layers = CrossNetwork(n_cross_layers)
        self.deep_layers = DeepNetwork(hidden_units, activation)
        
        self.embedding = Embedding(sum(x_dims)+1, latent_dim, input_length=1, embeddings_regularizer=l2(l2_emb))
    
        self.flatten = Flatten()
        self.final_dense = Dense(1)
        
    def call(self, inputs):
        embed = self.embedding(inputs + tf.constant((0, *np.cumsum(self.x_dims)))[:-1])
        embed = self.flatten(embed)
        
        # if continuous, concat with embed
        
        cross_out = self.cross_layers(embed)
        deep_out = self.deep_layers(embed)
        
        out = tf.concat([cross_out, deep_out], 1)
        out = self.final_dense(out)
        
        return out
```

<!-- #region id="hIFfCfagjnP4" -->
### Training
<!-- #endregion -->

```python id="fnrgaeV0joO8"
dcn = DCN((len(uuid), len(uiid)), 8, 2, [128,64])
dcn.compile(optimizer='adam',
           loss = losses.BinaryCrossentropy(from_logits=True))

dcn.fit(tr_X,
       train['rating'].values,
       epochs=10,
       shuffle=True)
```

<!-- #region id="bU8wYyEAjqM4" -->
### Evaluation
<!-- #endregion -->

```python id="Vz8rUnjOjuRF"
pred = dcn.predict(test_X)
np.sum(np.where(pred>0., 1, 0).flatten() == test['rating'].values) / len(pred)
```

```python id="Ga0pW9gPjrCY"
print(roc_auc_score(test['rating'].values, pred))
print(precision_score(test['rating'].values, np.where(pred>0., 1, 0)))
print(recall_score(test['rating'].values, np.where(pred>0., 1, 0)))
```

<!-- #region id="XWbPC6kGjwuI" -->
## DeepFM
<!-- #endregion -->

<!-- #region id="6vujimZ6j5qo" -->
### Abstract
<!-- #endregion -->

<!-- #region id="xIblRKPPkAfB" -->
[DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/pdf/1703.04247.pdf)
<!-- #endregion -->

<!-- #region id="GZqlkN9-kdzH" -->
<!-- #endregion -->

<!-- #region id="uIxRyZC1j5qp" -->
### Load data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="wFbCcZjkj5qq" executionInfo={"status": "ok", "timestamp": 1630486555164, "user_tz": -330, "elapsed": 6685, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3b375b28-df9c-4809-924c-be94ea45ee3d"
df = load_data('ratings.dat')
df.head()
```

```python id="-Gf_eH4Jj5qr"
uuid = df['userId'].unique()
uiid = df['movieId'].unique()
```

<!-- #region id="AqeSbda_j5qs" -->
### Preprocessing
<!-- #endregion -->

```python id="waifMCOlj5qs"
train, test = train_test_split(df, test_size=0.15, random_state=SEED, stratify=df['userId'].values)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Wv-41g_Gj5qt" executionInfo={"status": "ok", "timestamp": 1630486555174, "user_tz": -330, "elapsed": 40, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7f3e9623-1601-44cd-8a25-95f8ff6b1bb1"
tr_X = np.stack([train['userId'].values.astype(np.int32), train['movieId'].values.astype(np.int32)], 1)
test_X = np.stack([test['userId'].values.astype(np.int32), test['movieId'].values.astype(np.int32)], 1)

tr_X.shape, test_X.shape
```

<!-- #region id="yHs-DE2Lj5qt" -->
### Model architecture
<!-- #endregion -->

```python id="6g4-Qhn2klIs"
class FM_layer(keras.Model):
    def __init__(self, latent_dim, w_reg=1e-4, v_reg=1e-4):
        super(FM_layer, self).__init__()
        self.latent_dim = latent_dim
        
        self.w_reg = w_reg
        self.v_reg = v_reg

    def build(self, input_shape):
        self.w_0 = self.add_weight(shape=(1, ),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        
        self.w = self.add_weight(shape=(input_shape[-1], 1), 
                             initializer=tf.random_normal_initializer(),
                                trainable=True,
                                regularizer=l2(self.w_reg))
        
        self.V = self.add_weight(shape=(input_shape[-1], self.latent_dim), 
                             initializer=tf.random_normal_initializer(),
                                trainable=True,
                                regularizer=l2(self.v_reg))

    def call(self, inputs):
        linear_terms = tf.reduce_sum(tf.matmul(inputs, self.w), axis=1)

        interactions = 0.5 * tf.reduce_sum(
            tf.pow(tf.matmul(inputs, self.V), 2)
            - tf.matmul(tf.pow(inputs, 2), tf.pow(self.V, 2)),
            1,
            keepdims=False
        )

        y_hat = (self.w_0 + linear_terms + interactions)

        return y_hat
```

```python id="7mDwv-n6lJIl"
class DeepFM(tf.keras.Model):
    def __init__(self, x_dims, latent_dim, l2_emb=1e-4):
        super(DeepFM, self).__init__()
        
        self.x_dims = x_dims
        self.latent_dim = latent_dim

        self.embedding = Embedding(sum(x_dims)+1, latent_dim, input_length=1, embeddings_regularizer=l2(l2_emb))
        self.fm_layer = FM_layer(latent_dim)
        self.dnn_layers = self.build_dnn()
        self.flatten =  Flatten()

    def build_dnn(self):
        model = Sequential()
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1))
        
        return model
        
    def call(self, inputs):        
        emb = self.embedding(inputs + tf.constant((0, *np.cumsum(self.x_dims)))[:-1])
        embed = self.flatten(emb)

        fm_out = self.fm_layer(embed)

        deep_out = self.dnn_layers(embed)

        outputs = fm_out + tf.squeeze(deep_out)
        
        return outputs
```

<!-- #region id="h3pl1mc6lT0f" -->
### Training
<!-- #endregion -->

```python id="TVEiZEiBlU2g"
dfm = DeepFM((len(uuid), len(uiid)), 8)
dfm.compile(loss=losses.BinaryCrossentropy(from_logits=True), 
            optimizer=optimizers.Adam())

dfm.fit(tr_X, 
       train['rating'].values,
      epochs=10,
      shuffle=True,
      validation_split=0.1)
```

<!-- #region id="OdPxifaMlW4Q" -->
### Evaluation
<!-- #endregion -->

```python id="_PiyGliQlaX4"
pred = dfm.predict(test_X)
np.sum(np.where(pred>0., 1, 0).flatten() == test['rating'].values) / len(pred)
```

```python id="c4ln2Z29lX-Z"
print(roc_auc_score(test['rating'].values, pred))
print(precision_score(test['rating'].values, np.where(pred>0., 1, 0)))
print(recall_score(test['rating'].values, np.where(pred>0., 1, 0)))
```

<!-- #region id="NbgeAyl5ldCi" -->
## FM
<!-- #endregion -->

<!-- #region id="lkYLJuWKmD6C" -->
### Abstract
<!-- #endregion -->

<!-- #region id="_st5ynA_mUVr" -->
[Factorization Machines for Data with Implicit Feedback](https://arxiv.org/abs/1812.08254)
<!-- #endregion -->

<!-- #region id="2zmZExCwmiVS" -->
<!-- #endregion -->

<!-- #region id="zmW-PO2Xli6X" -->
### Load data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="nbxQBgq_li6Y" executionInfo={"status": "ok", "timestamp": 1630486555164, "user_tz": -330, "elapsed": 6685, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3b375b28-df9c-4809-924c-be94ea45ee3d"
df = load_data('ratings.dat')
df.head()
```

```python id="807xQV6vli6Y"
uuid = df['userId'].unique()
uiid = df['movieId'].unique()
```

<!-- #region id="hXYx23Lxli6Y" -->
### Preprocessing
<!-- #endregion -->

```python id="zZ2m4lP1li6Y"
train, test = train_test_split(df, test_size=0.15, random_state=SEED, stratify=df['userId'].values)
```

```python colab={"base_uri": "https://localhost:8080/"} id="KzZmHBICli6Z" executionInfo={"status": "ok", "timestamp": 1630486555174, "user_tz": -330, "elapsed": 40, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7f3e9623-1601-44cd-8a25-95f8ff6b1bb1"
tr_X = np.hstack([to_categorical(train['userId'], len(uuid)), to_categorical(train['movieId'], len(uiid))])
test_X = np.hstack([to_categorical(test['userId'], len(uuid)), to_categorical(test['movieId'], len(uiid))])

tr_X.shape, test_X.shape
```

<!-- #region id="tGOu9GVwli6Z" -->
### Model architecture
<!-- #endregion -->

```python id="OQnEgzxtlvjV"
class FM_layer(keras.Model):
    def __init__(self, x_dim, latent_dim, w_reg=1e-4, v_reg=1e-4):
        super(FM_layer, self).__init__()
        self.x_dim = x_dim
        self.latent_dim = latent_dim
        
        self.w_reg = w_reg
        self.v_reg = v_reg

    def build(self, input_shape):
        self.w_0 = self.add_weight(shape=(1, ),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        
        self.w = self.add_weight(shape=(self.x_dim, 1), 
                             initializer=tf.random_normal_initializer(),
                                trainable=True,
                                regularizer=l2(self.w_reg))
        
        self.V = self.add_weight(shape=(self.x_dim, self.latent_dim), 
                             initializer=tf.random_normal_initializer(),
                                trainable=True,
                                regularizer=l2(self.v_reg))

    def call(self, inputs):
        linear_terms = tf.reduce_sum(tf.matmul(inputs, self.w), axis=1)

        interactions = 0.5 * tf.reduce_sum(
            tf.pow(tf.matmul(inputs, self.V), 2)
            - tf.matmul(tf.pow(inputs, 2), tf.pow(self.V, 2)),
            1,
            keepdims=False
        )

        y_hat = (self.w_0 + linear_terms + interactions)

        return y_hat
```

```python id="mFAIMXHmlwj3"
class FM(tf.keras.Model)
    def __init__(self, x_dim, latnt_dim):
        super(FM, self).__init__()
        self.fm = FM_layer(x_dim, latnt_dim)

    def call(self, inputs):
        fm_outputs = self.fm(inputs)
        outputs = tf.nn.sigmoid(fm_outputs)
        return outputs
```

<!-- #region id="bcSfjjR0l0xn" -->
### Training
<!-- #endregion -->

```python id="SLUh3ALNl1qK"
fm = FM(tr_X.shape[1], 8)
fm.compile(loss='binary_crossentropy', optimizer='adam')
fm.fit(tr_X, train['rating'].values,
      epochs=10,
      shuffle=True,
      validation_split=0.1)
```

<!-- #region id="GrZJ9y-ll4JN" -->
### Evaluation
<!-- #endregion -->

```python id="lBDc-ofMl5Al"
pred = fm.predict(test_X)
np.sum(np.where(pred>0.5, 1, 0).flatten() == test['rating'].values) / len(pred)
```

```python id="Y9sBKX2ol9jO"
print(roc_auc_score(test['rating'].values, pred))
print(precision_score(test['rating'].values, np.where(pred>0.5, 1, 0)))
print(recall_score(test['rating'].values, np.where(pred>0.5, 1, 0)))
```

<!-- #region id="gR10GewNmmXm" -->
## PNN
<!-- #endregion -->

<!-- #region id="5qmIGTQFmm8S" -->
### Abstract
<!-- #endregion -->

<!-- #region id="ZLLVNoz5moGm" -->
[Product-based Neural Networks for User Response Prediction](https://arxiv.org/pdf/1611.00144.pdf)
<!-- #endregion -->

<!-- #region id="obydHcIHm_fU" -->
<!-- #endregion -->

<!-- #region id="9dvnWiKgnHzr" -->
### Load data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="sWFfN_IHnHzs" executionInfo={"status": "ok", "timestamp": 1630486555164, "user_tz": -330, "elapsed": 6685, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3b375b28-df9c-4809-924c-be94ea45ee3d"
df = load_data('ratings.dat')
df.head()
```

```python id="DoZfcufNnHzt"
uuid = df['userId'].unique()
uiid = df['movieId'].unique()
```

<!-- #region id="KP03Kub0nHzu" -->
### Preprocessing
<!-- #endregion -->

```python id="9lx2BCJRnHzu"
train, test = train_test_split(df, test_size=0.15, random_state=SEED, stratify=df['userId'].values)
```

```python colab={"base_uri": "https://localhost:8080/"} id="QfYNzzQ8nHzv" executionInfo={"status": "ok", "timestamp": 1630486555174, "user_tz": -330, "elapsed": 40, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7f3e9623-1601-44cd-8a25-95f8ff6b1bb1"
tr_X = np.stack([train['userId'].values.astype(np.int32), train['movieId'].values.astype(np.int32)], 1)
test_X = np.stack([test['userId'].values.astype(np.int32), test['movieId'].values.astype(np.int32)], 1)

tr_X.shape, test_X.shape
```

<!-- #region id="nrCkfxbvnHzv" -->
### Model architecture
<!-- #endregion -->

```python id="2XOXppXfnJkt"
class InnerProduct(layers.Layer):
    def __init__(self, x_dims):
        super().__init__()
        self.x_dims = x_dims
        
    def call(self, inputs):
        n = len(self.x_dims)
        
        p = []
        q = []
        for i in range(n):
            for j in range(i+1, n):
                p.append(i)
                q.append(j)
                
        p = tf.gather(inputs, p, axis=1)
        q = tf.gather(inputs, q, axis=1)
        
        out = p*q
        out = tf.squeeze(out, 1)
#         out = tf.reduce_sum(out, axis=2)
        return out
    
    
class OuterProduct(layers.Layer):
    def __init__(self, x_dims, kernel_type='mat'):
        super().__init__()
        self.x_dims = x_dims
        self.kernel_type = kernel_type
        
    def build(self, input_shape):
        n, m, k = input_shape
        
        if self.kernel_type == 'mat':
            self.kernel = self.add_weight(shape=(k, (m*(m-1)//2), k), 
                                         initializer = tf.zeros_initializer())
        else:
            self.kernel = self.add_weight(shape=((m*(m-1)//2), k),
                                         initializer = tf.zeros_initializer())
        
    def call(self, inputs):
        n = len(self.x_dims)
        
        p = []
        q = []
        for i in range(n):
            for j in range(i+1, n):
                p.append(i)
                q.append(j)
                
        p = tf.gather(inputs, p, axis=1)
        q = tf.gather(inputs, q, axis=1)
        
        if self.kernel_type == 'mat':
            kp = tf.transpose(tf.reduce_sum(tf.expand_dims(p, 1) * self.kernel, -1), [0, 2, 1])
            out = tf.reduce_sum(kp * q, -1)
        else:
            out = tf.reduce_sum(p * q * tf.expand_dims(self.kernel, 0), -1)
            
        return out
```

```python id="JPvtSXbhnQk_"
class PNN(Model):
    def __init__(self, x_dims, latent_dim, dnn_layers, model_type='inner', l2_emb=1e-4):
        super().__init__()
        self.x_dims = x_dims
        self.latent_dim = latent_dim

        self.embedding = Embedding(sum(x_dims)+1, latent_dim, input_length=1, embeddings_regularizer=l2(l2_emb))

        self.linear = Dense(latent_dim)

        if model_type == 'inner':
            self.pnn = InnerProduct(x_dims)
        elif model_type == 'outer':
            self.pnn = OuterProduct(x_dims)
        else:
            raise ValueError('no available model type')
        
        self.dnn = [Dense(unit, activation='relu') for unit in dnn_layers]
        
        self.final = Dense(1)
        
        self.flatten = Flatten()
        
    def call(self, inputs):
        emb = self.embedding(inputs + tf.constant((0, *np.cumsum(self.x_dims)))[:-1])
        
        linear = self.flatten(self.linear(emb))
        quadratic = self.pnn(emb)

        concat = tf.concat([linear, quadratic], -1)
        
        out = concat
        for layer in self.dnn:
            out = layer(out)
        
        out = self.final(out)
        return out
```

<!-- #region id="ThQHXiOGnVOf" -->
### Training
<!-- #endregion -->

```python id="4MqNWV35nWN5"
ipnn = PNN((len(uuid), len(uiid)), 8, [64, 32])
opnn = PNN((len(uuid), len(uiid)), 8, [64, 32], 'outer')
ipnn.compile(loss=losses.BinaryCrossentropy(from_logits=True), 
            optimizer=optimizers.Adam())

ipnn.fit(tr_X, 
       train['rating'].values,
      epochs=10,
      shuffle=True,
      validation_split=0.1)
```

```python id="ro0XvIqbnZzc"
opnn.compile(loss=losses.BinaryCrossentropy(from_logits=True), 
            optimizer=optimizers.Adam())

opnn.fit(tr_X, 
       train['rating'].values,
      epochs=10,
      shuffle=True,
      validation_split=0.1)
```

<!-- #region id="j-xTOgTence_" -->
### Evaluation
<!-- #endregion -->

```python id="GQDDf-2Hndiq"
pred_i = ipnn.predict(test_X)
pred_o = opnn.predict(test_X)
print(np.sum(np.where(pred_i>0., 1, 0).flatten() == test['rating'].values) / len(pred_i))
print(np.sum(np.where(pred_o>0., 1, 0).flatten() == test['rating'].values) / len(pred_o))
```

```python id="B_ULcGghnhAt"
# inner
print(roc_auc_score(test['rating'].values, pred_i))
print(precision_score(test['rating'].values, np.where(pred_i>0., 1, 0)))
print(recall_score(test['rating'].values, np.where(pred_i>0., 1, 0)))
```

```python id="4PIHiDxIng9i"
# outer
print(roc_auc_score(test['rating'].values, pred_o))
print(precision_score(test['rating'].values, np.where(pred_o>0., 1, 0)))
print(recall_score(test['rating'].values, np.where(pred_o>0., 1, 0)))
```

<!-- #region id="g42IU-Fcng4m" -->
## W&D
<!-- #endregion -->

<!-- #region id="Yk0Dp4wansYI" -->
### Abstract
<!-- #endregion -->

<!-- #region id="Jp0vW40RntW1" -->
[Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)
<!-- #endregion -->

<!-- #region id="yTQavWgCoFpD" -->
<!-- #endregion -->

<!-- #region id="LSASjl_lpqk2" -->
### Load data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="til-J1dZpqlF" executionInfo={"status": "ok", "timestamp": 1630486555164, "user_tz": -330, "elapsed": 6685, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3b375b28-df9c-4809-924c-be94ea45ee3d"
df = load_data('ratings.dat')
df.head()
```

```python id="tB9feWDzpqlG"
uuid = df['userId'].unique()
uiid = df['movieId'].unique()
```

<!-- #region id="k295K2rfpqlH" -->
### Preprocessing
<!-- #endregion -->

```python id="CBpJPPQdpqlI"
train, test = train_test_split(df, test_size=0.15, random_state=SEED, stratify=df['userId'].values)
```

```python colab={"base_uri": "https://localhost:8080/"} id="w1WOmIvbpqlI" executionInfo={"status": "ok", "timestamp": 1630486555174, "user_tz": -330, "elapsed": 40, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7f3e9623-1601-44cd-8a25-95f8ff6b1bb1"
tr_X = np.stack([train['userId'].values.astype(np.int32), train['movieId'].values.astype(np.int32)], 1)
test_X = np.stack([test['userId'].values.astype(np.int32), test['movieId'].values.astype(np.int32)], 1)

tr_X.shape, test_X.shape
```

<!-- #region id="oKH3anIFpqlJ" -->
### Model architecture
<!-- #endregion -->

```python id="wAeOQmg4n0Qf"
class WideAndDeep(keras.Model):
    def __init__(self, u_dim, i_dim, u_emb_dim=4, i_emb_dim=4):
        super(WideAndDeep, self).__init__()
        
        self.u_dim = u_dim
        self.i_dim = i_dim
        self.u_emb_dim = u_emb_dim
        self.i_emb_dim = i_emb_dim
        
        self.deep_model = self.build_deep_model()
        self.wide_model = self.build_wide_model()


    def compile(self, wide_optim, deep_optim, loss_fn):
        super(WideAndDeep, self).compile()
        self.wide_optim = wide_optim
        self.deep_optim = deep_optim
        self.loss_fn = loss_fn
    
    def build_deep_model(self):
        u_input = Input(shape=(1, ))
        i_input = Input(shape=(1, ))

        u_emb = Flatten()(Embedding(self.u_dim, self.u_emb_dim, input_length=u_input.shape[1])(u_input))
        i_emb = Flatten()(Embedding(self.i_dim, self.i_emb_dim, input_length=i_input.shape[1])(i_input))

        concat = Concatenate()([u_emb, i_emb])
        
        h = Dense(256, activation='relu')(concat)
        h = Dense(128, activation='relu')(h)
        h = Dense(64, activation='relu')(h)
        h = Dropout(0.2)(h)

        out = Dense(1)(h)
        
        return Model([u_input, i_input], out, name='DeepModel')
    
    def build_wide_model(self):
        u_input = Input(shape=(self.u_dim, ))
        i_input = Input(shape=(self.i_dim, ))

        concat = Concatenate()([u_input, i_input])
        
        out = Dense(1)(concat)
        
        return Model([u_input, i_input], out, name='WideModel')
        
    
    def train_step(self, data):
        X, y = data
        user, item, user_ohe, item_ohe = X
        
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            wide_logit = self.wide_model([user_ohe, item_ohe])
            deep_logit = self.deep_model([user, item])
            logit = 0.5*(wide_logit + deep_logit)
            
            loss = self.loss_fn(y, logit)
            
        wide_grads = tape1.gradient(loss, self.wide_model.trainable_weights)
        self.wide_optim.apply_gradients(zip(wide_grads, self.wide_model.trainable_weights))
        
        deep_grads = tape2.gradient(loss, self.deep_model.trainable_weights)
        self.deep_optim.apply_gradients(zip(deep_grads, self.deep_model.trainable_weights))
        
        return {'loss': loss}
    
    def call(self, data):
        user, item, user_ohe, item_ohe = data
        wide_logit = self.wide_model([user_ohe, item_ohe])
        deep_logit = self.deep_model([user, item])
        return 0.5*(wide_logit + deep_logit)
```

<!-- #region id="2t1wp6zBoOOb" -->
### Training
<!-- #endregion -->

```python id="aTWRR26joPUP"
wnd = WideAndDeep(len(uuid), len(uiid))
wnd.compile(
    optimizers.Adam(1e-3),
    optimizers.Adam(1e-3),
    losses.BinaryCrossentropy(from_logits=True)
           )

hist = wnd.fit([train['userId'].values, train['movieId'].values, to_categorical(train['userId']), to_categorical(train['movieId'])],
                   train['rating'].values,
                   shuffle=True,
                   epochs=10,
                   validation_split=0.1
                  )
```

<!-- #region id="oYMh5Gq3oSS0" -->
### Evaluation
<!-- #endregion -->

```python id="DLphTiJCoTTF"
pred = wnd.predict([test['userId'].values, test['movieId'].values, to_categorical(test['userId'], len(uuid)), to_categorical(test['movieId'], len(uiid))])
np.sum(np.where(pred>0, 1, 0).flatten() == test['rating'].values) / len(pred)
```

```python id="W4uS0jz_oVxv"
print(roc_auc_score(test['rating'].values, pred.flatten()))
print(precision_score(test['rating'].values, np.where(pred>0, 1, 0).flatten()))
print(recall_score(test['rating'].values, np.where(pred>0, 1, 0).flatten()))
```

<!-- #region id="GsA-gR3xoYec" -->
## xDFM
<!-- #endregion -->

<!-- #region id="O6Z46hWUopBu" -->
### Abstract

[xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/pdf/1803.05170.pdf)
<!-- #endregion -->

<!-- #region id="dxYmOHEKpWhD" -->
<!-- #endregion -->

<!-- #region id="tJ0JRQxFpspl" -->
### Load data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="LkPZReQMpspm" executionInfo={"status": "ok", "timestamp": 1630486555164, "user_tz": -330, "elapsed": 6685, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3b375b28-df9c-4809-924c-be94ea45ee3d"
df = load_data('ratings.dat')
df.head()
```

```python id="lELSstXnpspn"
uuid = df['userId'].unique()
uiid = df['movieId'].unique()
```

<!-- #region id="WSrzne2upspo" -->
### Preprocessing
<!-- #endregion -->

```python id="c9JWH7aZpspo"
train, test = train_test_split(df, test_size=0.15, random_state=SEED, stratify=df['userId'].values)
```

```python colab={"base_uri": "https://localhost:8080/"} id="pzUab1eApspo" executionInfo={"status": "ok", "timestamp": 1630486555174, "user_tz": -330, "elapsed": 40, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7f3e9623-1601-44cd-8a25-95f8ff6b1bb1"
tr_X = np.stack([train['userId'].values.astype(np.int32), train['movieId'].values.astype(np.int32)], 1)
test_X = np.stack([test['userId'].values.astype(np.int32), test['movieId'].values.astype(np.int32)], 1)

tr_X.shape, test_X.shape
```

<!-- #region id="UNz6scgHpspp" -->
### Model architecture
<!-- #endregion -->

```python id="tYnDiimXoanI"
class CIN(layers.Layer):
    def __init__(self, cross_layer_sizes, activation=None):
        super(CIN, self).__init__()
        self.cross_layer_sizes = cross_layer_sizes
        self.n_layers = len(cross_layer_sizes)
        self.activation = None
        
        if activation:
            self.activation = Activation(activation)
        
        self.cross_layers = []
        for corss_layer_size in cross_layer_sizes:
            self.cross_layers.append(Conv1D(corss_layer_size, 1, data_format='channels_first'))
            
        self.linear = Dense(1)
    
    def call(self, inputs): # embedding is input
        batch_size, field_size, emb_size = inputs.shape
        xs = [inputs]

        for i, layer in enumerate(self.cross_layers):
            x = tf.einsum('nie,nje->nije', xs[i], xs[0])
            x = tf.reshape(x, (-1, field_size*xs[i].shape[1] , emb_size))

            x = layer(x)
            if self.activation:
                x = self.activation(x)
            
            xs.append(x)
            
        res = tf.reduce_sum(tf.concat(xs, axis=1), -1)
        return res
```

```python id="EyDkpVGMpxWH"
class xDFM(Model):
    def __init__(self, x_dims, latent_dim, cin_layers, dnn_layers, activation=None, l2_emb=1e-4):
        super(xDFM, self).__init__()
        self.x_dims = x_dims
        
        self.embedding = Embedding(sum(x_dims)+1, latent_dim, input_length=1, embeddings_regularizer=l2(l2_emb))
        
        self.linear = Dense(1)
        
        self.dnn_layers = [Dense(n, activation=activation) for n in dnn_layers]
        self.dnn_final = Dense(1)
        
        self.cin_layers = CIN(cin_layers, activation=activation)
        self.cin_final = Dense(1)
        
    def call(self, inputs):
        # only apply ohe for categorical
        n_feat = inputs.shape[-1]
        sparse = [(tf.one_hot(inputs[:,i], self.x_dims[i])) for i in range(n_feat)]
        sparse = tf.concat(sparse, 1)

        emb = self.embedding(inputs + tf.constant((0, *np.cumsum(self.x_dims)))[:-1])

        dnn_input = Flatten()(emb)

        linear_out = self.linear(sparse)
            
        dnn_out = dnn_input
        for dnn_layer in self.dnn_layers:
            dnn_out = dnn_layer(dnn_out)
        dnn_out = self.dnn_final(dnn_out)

        cin_out = self.cin_layers(emb)
        cin_out = self.cin_final(cin_out)

        out = linear_out + dnn_out + cin_out
        
        return out
```

<!-- #region id="mUqVYinJp0Qq" -->
### Training
<!-- #endregion -->

```python id="-3BeDQ1Jp0NF"
xdfm = xDFM((len(uuid), len(uiid)), 8, [32, 32], [128, 64], 'relu')
# easily overfitting, reduce epochs
xdfm.compile(loss=losses.BinaryCrossentropy(from_logits=True), 
            optimizer=optimizers.Adam())

xdfm.fit(tr_X, 
       train['rating'].values,
      epochs=5,
      shuffle=True,
      validation_split=0.1)
```

<!-- #region id="9vMNUaZKp0I1" -->
### Evaluation
<!-- #endregion -->

```python id="6GErPNxZp6Ht"
pred = xdfm.predict(test_X)
np.sum(np.where(pred>0., 1, 0).flatten() == test['rating'].values) / len(pred)
```

```python id="Q7foGuxZp8Mh"
print(roc_auc_score(test['rating'].values, pred))
print(precision_score(test['rating'].values, np.where(pred>0., 1, 0)))
print(recall_score(test['rating'].values, np.where(pred>0., 1, 0)))
```

<!-- #region id="DaAfZdHxeGS-" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="H9O3IujqeGTB" executionInfo={"status": "ok", "timestamp": 1638118497142, "user_tz": -330, "elapsed": 3707, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9cba785b-9406-4eb6-b913-c82fca23a2c8"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="xQ7mmpxteGTC" -->
---
<!-- #endregion -->

<!-- #region id="w_Oqi7kqeGTD" -->
**END**
<!-- #endregion -->
