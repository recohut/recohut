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

<!-- #region id="1rCqqXdh6HLD" -->
## BPR-MF model
<!-- #endregion -->

```python id="uAom_5nIl8nj"
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime

from tqdm.notebook import tqdm

import warnings, random
warnings.filterwarnings('ignore')
```

```python id="aB3rWtCGOm0U"
!wget -q --show-progress https://github.com/sparsh-ai/stanza/raw/S629908/rec/CDL/data/ml_100k_train.npy
```

```python id="uv_UfT-9naHl"
# data loading
train = np.load('ml_100k_train.npy')
```

```python colab={"base_uri": "https://localhost:8080/"} id="Jo0qyfpmofOx" executionInfo={"status": "ok", "timestamp": 1638114512302, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4f670070-0ab9-466f-b2d0-f27e552097c3"
train.shape
```

```python id="H8See0uhoti2"
train = np.array(train > 0, dtype=float)
```

```python id="GnD7LCKho8Tu"
class Config:
    learning_rate = 0.01/2
    weight_decay = 0.1/2
    early_stopping_round = 0
    epochs = 50
    seed = 1995
    dim_f = 20
    alpha = 100
    bootstrap_proportion = 0.5
    
config = Config()
```

```python id="vJ5NDG_MqBHa"
def item_per_user_dict(data):
    item_per_user = defaultdict(list)
    user_pos = np.nonzero(data != 0)[0]
    item_pos = np.nonzero(data != 0)[1]
    
    for u, i in zip(user_pos, item_pos):
        item_per_user[u].append(i)

    return item_per_user
```

```python id="ojZ4LdrwqEFg"
class BPR_MF:
    def __init__(self, data):
        self.data = data
        self.user_num = data.shape[0]
        self.item_num = data.shape[1]
        self.user_pos = data.nonzero()[0]
        self.item_pos = data.nonzero()[1]

        self.train_hist = defaultdict(list) 
        self.valid_hist = defaultdict(list)

        self.W = np.random.standard_normal((self.user_num, config.dim_f))
        self.H = np.random.standard_normal((self.item_num, config.dim_f))

    def sampling_uij(self, item_per_user):
        u = np.random.choice(self.user_num)
        rated_items = item_per_user[u]
        i = np.random.choice(rated_items)
        j = np.random.choice(self.item_num)

        return u, i, j, rated_items

    def fit(self):
        train_per_user, test_per_user = self.train_test_split(self.data)

        n = len(self.user_pos)
        for epoch in range(config.epochs):
            preds = []
            num_update_per_epoch = 0
            while num_update_per_epoch <= n*config.bootstrap_proportion:
                u, i, j, rated_items = self.sampling_uij(train_per_user)
                if j not in rated_items:
                    xuij = self.gradient_descent(u, i, j)        
                    num_update_per_epoch += 1    
                    preds.append(xuij)

            auc = np.where(np.array(preds) > 0, 1, 0).mean()
            auc_vl = self.evaluate(train_per_user, test_per_user)
            self.train_hist[epoch] = auc; self.valid_hist[epoch] = auc_vl
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(f'EPOCH {epoch+1} TRAIN AUC: {auc}, TEST AUC {auc_vl}')
    
    def scoring(self, u, i, j):
        xui = np.dot(self.W[u, :], self.H[i, :])
        xuj = np.dot(self.W[u, :], self.H[j, :])
        xuij = np.clip(xui - xuj, -500, 500)
        
        return xui, xuj, xuij 
    
    def gradient(self, u, i, j):
        xui, xuj, xuij = self.scoring(u, i, j)
        common_term = np.exp(-xuij) / (np.exp(-xuij) + 1)
        
        dw = common_term * (self.H[i, :] - self.H[j, :]) + config.weight_decay*self.W[u, :]
        dhi = common_term * self.W[u, :] + config.weight_decay*self.H[i, :]
        dhj = common_term * -self.W[u, :] + config.weight_decay*self.H[j, :]
        return dw, dhi, dhj, xuij
    
    def gradient_descent(self, u, i, j):
        dw, dhi, dhj, xuij = self.gradient(u, i, j)

        self.W[u, :] = self.W[u, :] + config.learning_rate*dw
        self.H[i, :] = self.H[i, :] + config.learning_rate*dhi
        self.H[j, :] = self.H[j, :] + config.learning_rate*dhj
        return xuij

    def train_test_split(self, data):
        train_per_user = item_per_user_dict(data)
        test_per_user = {}
        for u in range(self.user_num):
            temp = train_per_user[u]
            length = len(temp)
            test_per_user[u] = temp.pop(np.random.choice(length))
            train_per_user[u] = temp

        return train_per_user, test_per_user

    def evaluate(self, train_per_user, test_per_user):
        X = np.dot(self.W, self.H.T)
        item_idx = set(np.arange(self.item_num))
        auc = []
        for u in range(self.user_num):
            i = test_per_user[u]
            j_s = list(item_idx - set(train_per_user[u]))
            auc.append(np.mean(np.where(X[u, i] - X[u, j_s] > 0, 1, 0)))
        return np.mean(auc)

    def plot_loss(self):
        fig, ax = plt.subplots(1,1, figsize=(10, 5))
        
        ax.plot(list(self.train_hist.keys()), list(self.train_hist.values()), color='orange', label='train')
        ax.plot(list(self.valid_hist.keys()), list(self.valid_hist.values()), color='green', label='valid')
        plt.legend()
        plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="XZPHz0zdua8t" executionInfo={"status": "ok", "timestamp": 1638114897798, "user_tz": -330, "elapsed": 302276, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="63b25393-672c-4ddd-b385-069de707f004"
model = BPR_MF(train)
model.fit()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 320} id="kR7rfWxMuv59" executionInfo={"status": "ok", "timestamp": 1638114898671, "user_tz": -330, "elapsed": 904, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1ee40f84-0e0d-4755-b23a-34e3000e1b69"
model.plot_loss()
```

<!-- #region id="vhB5APuTQNQI" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="jpKEP2FGQSKE" executionInfo={"status": "ok", "timestamp": 1638115069735, "user_tz": -330, "elapsed": 4232, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8fd7f015-d498-4686-e1bf-b9c6f1444fff"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="-k-HLCIjQNQR" -->
---
<!-- #endregion -->

<!-- #region id="Mxkw17lSQNQS" -->
**END**
<!-- #endregion -->
