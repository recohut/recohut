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

<!-- #region id="Rgy-wkK33glM" -->
# Deep Variational Models on FilmTrust Dataset in Tensorflow
<!-- #endregion -->

<!-- #region id="lF29_ys12dy1" -->
<img src='https://github.com/RecoHut-Stanzas/S394070/raw/main/images/flow.svg'>
<!-- #endregion -->

<!-- #region id="ILGqY7FA2o8P" -->
## **Step 1 - Setup the environment**
<!-- #endregion -->

<!-- #region id="9CEauQyaqpzo" -->
### **1.1 Install libraries**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="mI-Hbwwv25Z1" executionInfo={"status": "ok", "timestamp": 1639887181043, "user_tz": -330, "elapsed": 7061, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="150b9dfd-f164-4dc4-917a-36e0992d49cb"
!pip install -q -U git+https://github.com/RecoHut-Projects/recohut.git -b v0.0.2
```

<!-- #region id="vRf7z5fC24vu" -->
### **1.2 Download datasets**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="sQI_J4hQ28hk" executionInfo={"status": "ok", "timestamp": 1639887185554, "user_tz": -330, "elapsed": 745, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="aa6c08b9-b590-4648-c870-43aa3f3a621f"
!wget -q --show-progress https://github.com/RecoHut-Datasets/filmtrust/raw/v1/ft.csv
# !wget -q --show-progress https://github.com/RecoHut-Datasets/movielens_1m/raw/v3/ml1m.csv
# !wget -q --show-progress https://github.com/RecoHut-Datasets/myanimelist/raw/v1/anime.zip
```

<!-- #region id="JQ6u2WK73Awv" -->
### **1.3 Import libraries**
<!-- #endregion -->

```python id="9r8-VYWyS0rK" executionInfo={"status": "ok", "timestamp": 1639887204366, "user_tz": -330, "elapsed": 7855, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, ndcg_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

```python id="-0mNDMpqqx80" executionInfo={"status": "ok", "timestamp": 1639887211953, "user_tz": -330, "elapsed": 7611, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# models
from recohut.models.tf.deepmf import DeepMF
from recohut.models.tf.ncf import NCF
from recohut.models.tf.vdeepmf import VDeepMF
from recohut.models.tf.vncf import VNCF

# metrics
from recohut.metrics.utils import calculate_precision_recall
from recohut.metrics.utils import calculate_ndcg
```

<!-- #region id="5lye2WyJ5OGv" -->
### **1.4 Set params**
<!-- #endregion -->

```python id="T3ILQ6MYTaZK" executionInfo={"status": "ok", "timestamp": 1639887225480, "user_tz": -330, "elapsed": 621, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
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

```python id="Qlef78HkVR6h" executionInfo={"status": "ok", "timestamp": 1639887230208, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
args = Args(dataset='ft')
```

<!-- #region id="gYytf-uj5Z1t" -->
## **Step 2 - Data preparation**
<!-- #endregion -->

```python id="P_5s5i_ETUui" executionInfo={"status": "ok", "timestamp": 1639887430124, "user_tz": -330, "elapsed": 460, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
df = pd.read_csv(args.dataset+'.csv', delimiter = ',')

num_users = df.user.max() + 1
num_items = df.item.max() + 1

X = df[['user', 'item']].to_numpy()
y = df[['rating']].to_numpy()
```

```python id="VrfhN3gXUGTK" executionInfo={"status": "ok", "timestamp": 1639887450352, "user_tz": -330, "elapsed": 733, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

X_train = [X_train[:,0], X_train[:,1]]
X_test = [X_test[:,0], X_test[:,1]]
```

<!-- #region id="ygNPCx5o5yAv" -->
## **Step 3 - Model training**
<!-- #endregion -->

<!-- #region id="KTJjlOuUWgVB" -->
### **3.1 DeepMF**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="w_H1BYPaUzLr" outputId="9df601fb-a024-43aa-97b5-675933af0959" executionInfo={"status": "ok", "timestamp": 1639887483444, "user_tz": -330, "elapsed": 30499, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
model = DeepMF(num_users=num_users, num_items=num_items, latent_dim=args.latent_dim)
deepmf = model.build()

deepmf.compile(optimizer='adam', metrics=['mae'], loss='mean_squared_error')
deepmf.summary()

deepmf_report = deepmf.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=args.deepmf_epochs, steps_per_epoch=args.steps_per_epoch, verbose=1)
```

<!-- #region id="Iz_qgh3DV-AT" -->
### **3.2 NCF**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="aEED3Ar1Wi8r" outputId="27d23eb1-07d7-4bfb-b90b-d402913aabb2" executionInfo={"status": "ok", "timestamp": 1639887525780, "user_tz": -330, "elapsed": 42351, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
model = NCF(num_users=num_users, num_items=num_items, latent_dim=args.latent_dim)
ncf = model.build()

ncf.compile(optimizer='adam', metrics=['mae'], loss='mean_squared_error')
ncf.summary()

ncf_report = ncf.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=args.ncf_epochs, steps_per_epoch=args.steps_per_epoch, verbose=1)
```

<!-- #region id="tIAMDQT1W9W8" -->
### **3.3 VDeepMF**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="QYeUI-9SXCzR" outputId="b007f023-6299-437e-e2e2-6c03b8655995" executionInfo={"status": "ok", "timestamp": 1639887568242, "user_tz": -330, "elapsed": 42484, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
args.batch_size = 32

model = VDeepMF(num_users=num_users, num_items=num_items, latent_dim=args.latent_dim, batch_size=args.batch_size)
vdeepmf = model.build()

vdeepmf.compile(optimizer='adam', metrics=['mae'], loss='mean_squared_error')
vdeepmf.summary()

vdeepmf_report = vdeepmf.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=args.vdeepmf_epochs, batch_size=args.batch_size, steps_per_epoch=args.steps_per_epoch, verbose=1)
```

<!-- #region id="ZLOPNDDQXRrJ" -->
### **3.4 VNCF**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ECYpJH2dXzrv" outputId="19afc1e7-5c35-4918-8c37-b358167c9136" executionInfo={"status": "ok", "timestamp": 1639887610437, "user_tz": -330, "elapsed": 42217, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
model = VNCF(num_users=num_users, num_items=num_items, latent_dim=args.latent_dim, batch_size=args.batch_size)
vncf = model.build()

vncf.compile(optimizer='adam', metrics=['mae'], loss='mean_squared_error')
vncf.summary()

vncf_report = vncf.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=args.vncf_epochs, batch_size=args.batch_size, steps_per_epoch=args.steps_per_epoch, verbose=1)
```

<!-- #region id="MPFpnqYP6UZY" -->
## **Step 4 - Performance analysis**
<!-- #endregion -->

<!-- #region id="1AtuV3ilYBMf" -->
### **4.1 Model predictions**
<!-- #endregion -->

```python id="53UidfIsYNR8" executionInfo={"status": "ok", "timestamp": 1639887611105, "user_tz": -330, "elapsed": 682, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
methods = ['vdeepmf', 'deepmf', 'vncf', 'ncf']

preds = pd.DataFrame()
preds['user'] = X_test[0]
preds['item'] = X_test[1]
preds['y_test'] = y_test

preds['deepmf'] = deepmf.predict(X_test)
preds['ncf'] = ncf.predict(X_test)
```

<!-- #region id="6rq97FGbYX9n" -->
Due to the variational approachs of the proposed methods, the same model can generates different predictions for the same <user, item> input. To avoid this, we compute the predictions of the proposed models as the average of 10 repetitions of the same prediction.
<!-- #endregion -->

```python id="nM4JUlE8YYhr" executionInfo={"status": "ok", "timestamp": 1639887611107, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
n_repeats = 10
```

```python id="WyfenU6VYfEz" executionInfo={"status": "ok", "timestamp": 1639887614797, "user_tz": -330, "elapsed": 3700, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
predictions = None
for i in range(n_repeats):
    if i == 0:
        predictions = vdeepmf.predict(X_test)
    else:
        predictions = np.append(predictions, vdeepmf.predict(X_test), axis=1)
        
preds['vdeepmf'] = np.mean(predictions, axis=1)
predictions = None
```

```python id="bnWxFfIpYgFx" executionInfo={"status": "ok", "timestamp": 1639887620284, "user_tz": -330, "elapsed": 5498, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
for i in range(n_repeats):
    if i == 0:
        predictions = vncf.predict(X_test)
    else:
        predictions = np.append(predictions, vncf.predict(X_test), axis=1)

preds['vncf'] = np.mean(predictions, axis=1)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 424} id="lXHPo52J8QD2" executionInfo={"status": "ok", "timestamp": 1639887620292, "user_tz": -330, "elapsed": 52, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3304871b-60d1-481e-837b-3b1d1a0b0a82"
preds
```

<!-- #region id="fUal5rS3YgqW" -->
### **4.2 Quality of predictions**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="q2li0wMeYjj2" outputId="1ab16510-9537-48e5-ca39-1414fa5794d5" executionInfo={"status": "ok", "timestamp": 1639887620295, "user_tz": -330, "elapsed": 30, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
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
### **4.3 Quality of the recommendations**
<!-- #endregion -->

```python id="bA6npDTgFzNz" executionInfo={"status": "ok", "timestamp": 1639887620298, "user_tz": -330, "elapsed": 24, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
num_recommendations = [2,3,4,5,6,7,8,9,10]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 314} id="UNqiMbU3YtSA" outputId="57c50dd5-08de-42e8-b71d-b4419092719d" executionInfo={"status": "ok", "timestamp": 1639887737527, "user_tz": -330, "elapsed": 117252, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
for m in methods:
    precision = np.zeros(len(num_recommendations))
    recall = np.zeros(len(num_recommendations))
    
    for i, n in enumerate(num_recommendations):
        ids = preds[['user', 'item']].to_numpy()
        y_true = preds['y_test'].to_numpy()
        y_pred = preds[m].to_numpy()
        precision[i], recall[i] = calculate_precision_recall(ids, y_true, y_pred, n, args.like_threshold) 

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

```python colab={"base_uri": "https://localhost:8080/", "height": 314} id="eFSm0Fpga4zp" outputId="6a9a482e-b559-4bfb-fbe5-93ec5788c042" executionInfo={"status": "ok", "timestamp": 1639887760385, "user_tz": -330, "elapsed": 22873, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
for m in methods:
    ndcg = np.zeros(len(num_recommendations))
    
    for i, n in enumerate(num_recommendations):
        ids = preds[['user', 'item']].to_numpy()
        y_true = preds['y_test'].to_numpy()
        y_pred = preds[m].to_numpy()
        ndcg[i] = calculate_ndcg(ids, y_true, y_pred, n) 
        
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

<!-- #region id="wycnKnps68l3" -->
## **Closure**
<!-- #endregion -->

<!-- #region id="BMk2vGOh6-cp" -->
For more details, you can refer to https://github.com/RecoHut-Stanzas/S394070.
<!-- #endregion -->

<!-- #region id="f-_pvwaI7bNX" -->
<a href="https://github.com/RecoHut-Stanzas/S394070/blob/main/reports/S394070_Report.ipynb" alt="S394070_Report"> <img src="https://img.shields.io/static/v1?label=report&message=active&color=green" /></a> <a href="https://github.com/RecoHut-Stanzas/S394070" alt="S394070"> <img src="https://img.shields.io/static/v1?label=code&message=github&color=blue" /></a>
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="LgrJODmhbVxS" outputId="d26f7a61-743a-46a3-b7c8-baaea8cede02" executionInfo={"status": "ok", "timestamp": 1639887914676, "user_tz": -330, "elapsed": 4319, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
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
