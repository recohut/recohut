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

<!-- #region id="Fh5MPHxmhlaB" -->
# Deep Variational Models on FilmTrust, ML-1m, and MyAnimeList Dataset in Tensorflow

[Link to Report](https://recohut.notion.site/Learning-better-Latent-spaces-with-Variational-Autoencoder-Plugin-b2043aad71ad4073a7c2e3a7bb2426f8)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="peOW_bNJnkFH" executionInfo={"status": "ok", "timestamp": 1638031646894, "user_tz": -330, "elapsed": 4386, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="483025ec-5568-4007-e6c0-b41a007a82dd"
!wget -q --show-progress https://github.com/RecoHut-Datasets/filmtrust/raw/v1/ft.csv
!wget -q --show-progress https://github.com/RecoHut-Datasets/movielens_1m/raw/v3/ml1m.csv
!wget -q --show-progress https://github.com/RecoHut-Datasets/myanimelist/raw/v1/anime.zip
!unzip anime.zip
```

```python id="9r8-VYWyS0rK"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dropout, Dense, Concatenate, Dot, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

from sklearn.metrics import precision_score, recall_score, ndcg_score, mean_absolute_error, mean_squared_error, r2_score
```

```python id="T3ILQ6MYTaZK"
class Args:
    def __init__(self, dataset):
        self.dataset = dataset
        if dataset == 'ml1m':
            self.latent_dim = 5
            self.like_threshold = 4
            self.steps_per_epoch = None
            self.deepmf_epochs = 10
            self.ncf_epochs = 10
            self.vdeepmf_epochs = 6
            self.vncf_epochs = 9
        elif dataset == 'ft':
            self.latent_dim = 5
            self.like_threshold = 3
            self.steps_per_epoch = None
            self.deepmf_epochs = 15
            self.ncf_epochs = 8
            self.vdeepmf_epochs = 10
            self.vncf_epochs = 6
        elif dataset == 'anime':
            self.latent_dim = 7
            self.like_threshold = 8
            self.steps_per_epoch = None
            self.deepmf_epochs = 20
            self.ncf_epochs = 15
            self.vdeepmf_epochs = 9
            self.vncf_epochs = 9
```

```python id="Qlef78HkVR6h"
args = Args(dataset='ft')
```

```python id="P_5s5i_ETUui"
df = pd.read_csv(args.dataset+'.csv', delimiter = ',')

num_users = df.user.max() + 1
num_items = df.item.max() + 1

X = df[['user', 'item']].to_numpy()
y = df[['rating']].to_numpy()
```

```python id="VrfhN3gXUGTK"
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

X_train = [X_train[:,0], X_train[:,1]]
X_test = [X_test[:,0], X_test[:,1]]
```

<!-- #region id="KTJjlOuUWgVB" -->
DeepMF
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="w_H1BYPaUzLr" executionInfo={"status": "ok", "timestamp": 1638033531331, "user_tz": -330, "elapsed": 42445, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3ab67241-f225-47c2-ccaa-4102ee3b799e"
user_input = Input(shape=[1])
user_embedding = Embedding(num_users, args.latent_dim)(user_input)
user_vec = Flatten()(user_embedding)

item_input = Input(shape=[1])
item_embedding = Embedding(num_items, args.latent_dim)(item_input)
item_vec = Flatten()(item_embedding) 
        
dot = Dot(axes=1)([item_vec, user_vec])
    
DeepMF = Model([user_input, item_input], dot)

DeepMF.compile(optimizer='adam', metrics=['mae'], loss='mean_squared_error')
DeepMF.summary()

deepmf_report = DeepMF.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=args.deepmf_epochs, steps_per_epoch=args.steps_per_epoch, verbose=1)
```

<!-- #region id="Iz_qgh3DV-AT" -->
NCF
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="aEED3Ar1Wi8r" executionInfo={"status": "ok", "timestamp": 1638033547992, "user_tz": -330, "elapsed": 16704, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="289b742f-dde6-493c-bdf0-16aa8a87c06e"
item_input = Input(shape=[1], name='item-input')
item_embedding = Embedding(num_items, args.latent_dim, name='item-embedding')(item_input)
item_vec = Flatten(name='item-flatten')(item_embedding)

user_input = Input(shape=[1], name='user-input')
user_embedding = Embedding(num_users, args.latent_dim, name='user-embedding')(user_input)
user_vec = Flatten(name='user-flatten')(user_embedding)

concat = Concatenate(axis=1, name='item-user-concat')([item_vec, user_vec])
fc_1 = Dense(70, name='fc-1', activation='relu')(concat)
fc_1_dropout = Dropout(0.5, name='fc-1-dropout')(fc_1)
fc_2 = Dense(30, name='fc-2', activation='relu')(fc_1_dropout)
fc_2_dropout = Dropout(0.4, name='fc-2-dropout')(fc_2)
fc_3 = Dense(1, name='fc-3', activation='relu')(fc_2_dropout)

NCF = Model([user_input, item_input], fc_3)

NCF.compile(optimizer='adam', metrics=['mae'], loss='mean_squared_error')
NCF.summary()
    
ncf_report = NCF.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=args.ncf_epochs, steps_per_epoch=args.steps_per_epoch, verbose=1)
```

<!-- #region id="tIAMDQT1W9W8" -->
VDeepMF
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="QYeUI-9SXCzR" executionInfo={"status": "ok", "timestamp": 1638033598532, "user_tz": -330, "elapsed": 42818, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7ddb99a2-f1ae-476c-a908-71b11bfd9e8a"
batch_size = 32

def sampling(qargs):
    z_mean, z_var = qargs
    epsilon = K.random_normal(shape=(batch_size, args.latent_dim), mean=0., stddev=1)
    return z_mean + K.exp(z_var) * epsilon

user_input = Input(shape=[1])
user_embedding = Embedding(num_users, args.latent_dim)(user_input)
user_embedding_mean = Dense(args.latent_dim)(user_embedding)
user_embedding_var = Dense(args.latent_dim)(user_embedding)
user_embedding_z = Lambda(sampling)([user_embedding_mean, user_embedding_var])
user_vec = Flatten()(user_embedding_z)

item_input = Input(shape=[1])
item_embedding = Embedding(num_items, args.latent_dim)(item_input)
item_embedding_mean = Dense(args.latent_dim)(item_embedding)
item_embedding_var = Dense(args.latent_dim)(item_embedding)
item_embedding_z = Lambda(sampling)([item_embedding_mean, item_embedding_var], args.latent_dim)
item_vec = Flatten()(item_embedding_z)

dot = Dot(axes=1)([item_vec, user_vec])

VDeepMF = Model([user_input, item_input], dot)

VDeepMF.compile(optimizer='adam', metrics=['mae'], loss='mean_squared_error')
VDeepMF.summary()

vdeepmf_report = VDeepMF.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=args.vdeepmf_epochs, batch_size=batch_size, steps_per_epoch=args.steps_per_epoch, verbose=1)
```

<!-- #region id="ZLOPNDDQXRrJ" -->
VNCF
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ECYpJH2dXzrv" executionInfo={"status": "ok", "timestamp": 1638033620636, "user_tz": -330, "elapsed": 22160, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="74ac3d7c-c92e-46f5-b5d5-adf85007a932"
batch_size = 32

def sampling(qargs):
    z_mean, z_var = qargs
    epsilon = K.random_normal(shape=(batch_size, args.latent_dim), mean=0., stddev=1)
    return z_mean + K.exp(z_var) * epsilon

user_input = Input(shape=[1])
user_embedding = Embedding(num_users, args.latent_dim)(user_input)
user_embedding_mean = Dense(args.latent_dim)(user_embedding)
user_embedding_var = Dense(args.latent_dim)(user_embedding)
user_embedding_z = Lambda(sampling)([user_embedding_mean, user_embedding_var])
user_vec = Flatten()(user_embedding_z)

item_input = Input(shape=[1])
item_embedding = Embedding(num_items + 1, args.latent_dim)(item_input)
item_embedding_mean = Dense(args.latent_dim)(item_embedding)
item_embedding_var = Dense(args.latent_dim)(item_embedding)
item_embedding_z = Lambda(sampling)([item_embedding_mean, item_embedding_var], args.latent_dim)
item_vec = Flatten()(item_embedding_z)

concat = Concatenate(axis=1)([item_vec, user_vec])

fc_1 = Dense(80, name='fc-1', activation='relu')(concat)
fc_1_dropout = Dropout(0.6, name='fc-1-dropout')(fc_1)
fc_2 = Dense(25, name='fc-2', activation='relu')(fc_1_dropout)
fc_2_dropout = Dropout(0.4, name='fc-2-dropout')(fc_2)
fc_3 = Dense(1, name='fc-3', activation='relu')(fc_2_dropout)

VNCF = Model([user_input, item_input], fc_3)

VNCF.compile(optimizer='adam', metrics=['mae'], loss='mean_squared_error')
VNCF.summary()

vncf_report = VNCF.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=args.vncf_epochs, batch_size=batch_size, steps_per_epoch=args.steps_per_epoch, verbose=1)
```

<!-- #region id="1AtuV3ilYBMf" -->
### Results
<!-- #endregion -->

```python id="53UidfIsYNR8"
methods = ['vdeepmf', 'deepmf', 'vncf', 'ncf']

preds = pd.DataFrame()
preds['user'] = X_test[0]
preds['item'] = X_test[1]
preds['y_test'] = y_test

preds['deepmf'] = DeepMF.predict(X_test)
preds['ncf'] = NCF.predict(X_test)
```

<!-- #region id="6rq97FGbYX9n" -->
Due to the variational approachs of the proposed methods, the same model can generates different predictions for the same <user, item> input. To avoid this, we compute the predictions of the proposed models as the average of 10 repetitions of the same prediction.
<!-- #endregion -->

```python id="nM4JUlE8YYhr"
n_repeats = 10
```

```python id="WyfenU6VYfEz"
predictions = None
for i in range(n_repeats):
    if i == 0:
        predictions = VDeepMF.predict(X_test)
    else:
        predictions = np.append(predictions, VDeepMF.predict(X_test), axis=1)
        
preds['vdeepmf'] = np.mean(predictions, axis=1)
predictions = None
```

```python id="bnWxFfIpYgFx"
for i in range(n_repeats):
    if i == 0:
        predictions = VNCF.predict(X_test)
    else:
        predictions = np.append(predictions, VNCF.predict(X_test), axis=1)

preds['vncf'] = np.mean(predictions, axis=1)
```

<!-- #region id="fUal5rS3YgqW" -->
Quality of predictions
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="q2li0wMeYjj2" executionInfo={"status": "ok", "timestamp": 1638033628163, "user_tz": -330, "elapsed": 32, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a3ee1de2-866a-43e4-be53-ac8e4c08090b"
print('MAE:')
for m in methods:
    print('-', m, ':', mean_absolute_error(preds['y_test'], preds[m]))

print('MSD:')
for m in methods:
    print('-', m, ':', mean_squared_error(preds['y_test'], preds[m]))

print('R2:')
for m in methods:
    print('-', m, ':', r2_score(preds['y_test'], preds[m]))
```

<!-- #region id="6Wf1Rqx_Yj5C" -->
Quality of the recommendations
<!-- #endregion -->

```python id="vFKA6BTXYnQh"
num_recommendations = [2,3,4,5,6,7,8,9,10]

def recommender_precision_recall(X, y_true, y_pred, N, threshold):
    precision = 0
    recall = 0
    count = 0
    
    rec_true = np.array([1 if rating >= threshold else 0 for rating in y_true])
    rec_pred = np.zeros(y_pred.size)
    
    for user_id in np.unique(X[:,0]):
        indices = np.where(X[:,0] == user_id)[0]
        
        rec_true = np.array([1 if y_true[i] >= threshold else 0 for i in indices])

        if (np.count_nonzero(rec_true) > 0): # ignore test users without relevant ratings
        
            user_pred = np.array([y_pred[i] for i in indices])
            rec_pred = np.zeros(indices.size)

            for pos in np.argsort(user_pred)[-N:]:
                if user_pred[pos] >= threshold:
                    rec_pred[pos] = 1
            
            precision += precision_score(rec_true, rec_pred, zero_division=0)
            recall += recall_score(rec_true, rec_pred)
            count += 1
        
    return precision/count, recall/count
```

```python colab={"base_uri": "https://localhost:8080/", "height": 314} id="UNqiMbU3YtSA" executionInfo={"status": "ok", "timestamp": 1638033812297, "user_tz": -330, "elapsed": 71714, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="415c9111-c9d4-4fe2-b6bc-7707f0faa909"
for m in methods:
    precision = np.zeros(len(num_recommendations))
    recall = np.zeros(len(num_recommendations))
    
    for i, n in enumerate(num_recommendations):
        ids = preds[['user', 'item']].to_numpy()
        y_true = preds['y_test'].to_numpy()
        y_pred = preds[m].to_numpy()
        precision[i], recall[i] = recommender_precision_recall(ids, y_true, y_pred, n, args.like_threshold) 

    c = 'blue' if 'deepmf' in m else 'red'
    alpha = 1 if m[0] == 'v' else 0.6
    ls = '-' if m[0] == 'v' else '--'
        
    plt.plot(recall, precision, c=c, ls=ls, alpha=alpha, label=m)

    if m == 'vdeepmf':
        for i,(r,p) in enumerate(zip(recall, precision)):
            plt.annotate(num_recommendations[i], (r,p), textcoords="offset points", xytext=(5,5), ha='center')
    
plt.xlabel('Recall', fontsize=15); 
plt.ylabel('Precision', fontsize=15)

plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

plt.legend(bbox_to_anchor=(0,1.02,1,0.2), fontsize=12, loc="lower left", mode="expand", borderaxespad=0, ncol=len(methods))

plt.grid(True)

ylim_min, ylim_max = plt.ylim()
plt.ylim((ylim_min, ylim_max * 1.02))

plt.show()
```

```python id="YeJEUig1ZIcg"
def recommender_ndcg(X, y_true, y_pred, N):
    ndcg = 0
    count = 0
    
    for user_id in np.unique(X[:,0]):
        indices = np.where(X[:,0] == user_id)[0]
        
        user_true = np.array([y_true[i] for i in indices])
        user_pred = np.array([y_pred[i] for i in indices])  
        
        user_true = np.expand_dims(user_true, axis=0)
        user_pred = np.expand_dims(user_pred, axis=0)
                
        if user_true.size > 1:
            ndcg += ndcg_score(user_true, user_pred, k=N, ignore_ties=False)
            count += 1
    
    return ndcg / count
```

```python colab={"base_uri": "https://localhost:8080/", "height": 314} id="eFSm0Fpga4zp" executionInfo={"status": "ok", "timestamp": 1638033866884, "user_tz": -330, "elapsed": 17739, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="deab8d77-3faf-45f0-be4c-704a37f3a7cf"
for m in methods:
    ndcg = np.zeros(len(num_recommendations))
    
    for i, n in enumerate(num_recommendations):
        ids = preds[['user', 'item']].to_numpy()
        y_true = preds['y_test'].to_numpy()
        y_pred = preds[m].to_numpy()
        ndcg[i] = recommender_ndcg(ids, y_true, y_pred, n) 
        
    c = 'blue' if 'deepmf' in m else 'red'
    alpha = 1 if m[0] == 'v' else 0.6
    ls = '-' if m[0] == 'v' else '--'
 
    plt.plot(num_recommendations, ndcg, c=c, ls=ls, alpha=alpha, label=m)

plt.xlabel('Number of recommendations', fontsize=15); 
plt.ylabel('NDCG', fontsize=15)

plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

plt.legend(bbox_to_anchor=(0,1.02,1,0.2), fontsize=12, loc="lower left", mode="expand", borderaxespad=0, ncol=len(methods))

plt.grid(True)

plt.show()
```

<!-- #region id="xNDy6k12bVxP" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="LgrJODmhbVxS" executionInfo={"status": "ok", "timestamp": 1638033897655, "user_tz": -330, "elapsed": 2944, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f74fd630-b0bb-49c2-9c0e-98aa53bba458"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="hKleh6bfbVxT" -->
---
<!-- #endregion -->

<!-- #region id="9MbAhyUhbVxU" -->
**END**
<!-- #endregion -->
