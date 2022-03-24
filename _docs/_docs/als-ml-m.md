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

<!-- #region id="PlRmMjIaDBr6" -->
# ALS on ML-1m
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
    learning_rate = 0.01
    weight_decay = 0.1
    early_stopping_round = 0
    epochs = 10
    seed = 1995
    dim_f = 30
    alpha = 100
    
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
class ALS():
    
    def __init__(self, data, dim_f, seed):
        self.dim_f = dim_f
        self.user_num = data.shape[0]
        self.item_num = data.shape[1]
        
        self.R_tr = data
        self.P_tr = np.array(data > 0, dtype=np.float16)
        self.R_tst = data
        self.P_tst = np.array(data > 0, dtype=np.float16)
        self.C = self.R_tr * config.alpha + 1
        self.C_u = np.zeros((self.item_num, self.item_num))
        self.C_i = np.zeros((self.user_num, self.user_num))
            
        np.random.seed(seed)
        self.X = np.random.standard_normal((self.user_num, dim_f))
        self.Y = np.random.standard_normal((self.item_num, dim_f))
        
        self.loss_tr = defaultdict(float)
        self.loss_tst = defaultdict(float)

    def fit(self):
        start = datetime.now()
        for epoch in range(config.epochs):
            start_epoch = datetime.now()
            # stochastic
            n = 0
            for u in range(self.user_num):
                yty = np.dot(self.Y.T, self.Y)
                self.X[u, :] = self.update_user_vector(u, yty)

            for i in range(self.item_num):
                xtx = np.dot(self.X.T, self.X)
                self.Y[i, :] = self.update_item_vector(i, xtx)
            
            phat = self.scoring()
            train_loss = self.evaluate(train_eval=True)
            test_loss = self.evaluate(train_eval=False)

            self.loss_tr[epoch] = train_loss
            self.loss_tst[epoch] = test_loss
            print(f'EPOCH {epoch+1} : TRAINING RANK {self.loss_tr[epoch]:.5f}, VALID RANK {self.loss_tst[epoch]:.5f}')

            print(f'Time per one epoch {datetime.now() - start_epoch}')
        end = datetime.now()
        print(f'Training takes time {end-start}')
        
    def scoring(self):
        return np.dot(self.X, self.Y.T)
    
    def update_user_vector(self, u, yty):
        np.fill_diagonal(self.C_u, (self.C[u, :] - 1))
        comp1 = yty
        comp2 = np.dot(self.Y.T, self.C_u).dot(self.Y)
        comp3 = np.identity(config.dim_f) * config.weight_decay
        comp = np.linalg.inv(comp1 + comp2 + comp3)
        comp = np.dot(comp, self.Y.T).dot(self.C_u)
        
        return np.dot(comp, self.P_tr[u, :])

    def update_item_vector(self, i, xtx): 
        np.fill_diagonal(self.C_i, (self.C[:, i] - 1))
        comp1 = xtx
        comp2 = np.dot(self.X.T, self.C_i).dot(self.X)
        comp3 = np.identity(config.dim_f) * config.weight_decay
        comp = np.linalg.inv(comp1 + comp2 + comp3)
        comp = np.dot(comp, self.X.T).dot(self.C_i)
        
        return np.dot(comp, self.P_tr[:, i])
    

    def evaluate(self, train_eval):
        if train_eval:
            R = self.R_tr
        else:
            R = self.R_tst

        phat = self.scoring()
        rank_mat = np.zeros(phat.shape)
        for u in range(self.user_num):
            pred_u = phat[u, :] * -1
            rank = pred_u.argsort().argsort()
            rank = rank / self.item_num
            rank_mat[u, :] = rank

        return np.sum(R * rank_mat) / np.sum(R)

    def plot_loss(self):
        fig, ax = plt.subplots(1,1, figsize=(10, 5))
        
        ax.plot(list(self.loss_tr.keys()), list(self.loss_tr.values()), color='orange', label='train')
        ax.plot(list(self.loss_tst.keys()), list(self.loss_tst.values()), color='green', label='valid')
        plt.legend()
        plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="BrpsJmRTys6H" executionInfo={"status": "ok", "timestamp": 1630608273143, "user_tz": -330, "elapsed": 39539, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1bf5c172-205e-4873-d94e-f7c134ac9486"
model = ALS(train, config.dim_f, config.seed)
model.fit()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 320} id="PSyO4K3SzMpR" executionInfo={"status": "ok", "timestamp": 1630608273147, "user_tz": -330, "elapsed": 18, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5be8d282-2550-4a42-94fa-b59ad622a0bb"
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
