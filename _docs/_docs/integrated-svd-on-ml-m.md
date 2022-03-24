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

```python id="RuiURj-F17mq" executionInfo={"status": "ok", "timestamp": 1638116889957, "user_tz": -330, "elapsed": 694, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm.notebook import tqdm

import warnings
warnings.filterwarnings('ignore')
```

```python id="aB3rWtCGOm0U"
!wget -q --show-progress https://github.com/sparsh-ai/stanza/raw/S629908/rec/CDL/data/ml_100k_train.npy
!wget -q --show-progress https://github.com/sparsh-ai/stanza/raw/S629908/rec/CDL/data/ml_100k_test.npy
```

```python id="oGgWfwh7T8oe" executionInfo={"status": "ok", "timestamp": 1638116892206, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
train = np.load('ml_100k_train.npy')
test = np.load('ml_100k_test.npy')

train_imp = (train > 0).astype(float)
test_imp = (test > 0).astype(float)
```

```python id="N3cj-gQQB-ce" executionInfo={"status": "ok", "timestamp": 1638116892209, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class Config:
    learning_rate = 0.01
    weight_decay = 0.1
    early_stopping_round = 0
    epochs = 100
    seed = 1995
    dim_f = 15
    K = 30
    
config = Config()
```

```python colab={"base_uri": "https://localhost:8080/"} id="8WhLB-jSCgLa" executionInfo={"status": "ok", "timestamp": 1638119631829, "user_tz": -330, "elapsed": 2706533, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f3e1da4b-328e-4124-f77c-3cca9b162589"
def get_neighbor(train_e, train_i, lambda_=100):
    neighbor = []
    for data in [train_e, train_i]:
        RHO = np.corrcoef(data.T)
        RHO = np.nan_to_num(RHO, nan=-1)
        n_ij = np.dot(train_i.T, train_i)

        S = n_ij / (n_ij + lambda_) * RHO
        np.fill_diagonal(S, -1)    

        R_u = {u: data[u, :].nonzero()[0] for u in range(len(data))}
        S_k = {i: np.argsort(S[i, :])[-config.K:] for i in range(len(S))}
        neighbor.append((R_u, S_k))
    
    return neighbor

(R_u, S_k_r), (N_u, S_k_n) = get_neighbor(train, train_imp)

dd = []
for u in range(train.shape[0]):
    for i in range(train.shape[1]):
        if train[u, i] != 0:
            dd.append(len(np.intersect1d(R_u[u], S_k_r[i])))

class SVD_integrated():    
    def __init__(self, train_exp, train_imp, test_exp, test_imp, dim_f, seed):
        self.R_tr = train_exp
        self.N_tr = train_imp
        self.R_tst = test_exp
        self.N_tst = test_imp
        
        self.dim_f = dim_f
        self.user_num = train_exp.shape[0]
        self.item_num = train_exp.shape[1]

        np.random.seed(seed)
        self.P = np.random.standard_normal((self.user_num, dim_f))
        self.Q = np.random.standard_normal((self.item_num, dim_f))
        self.Y = np.random.standard_normal((self.item_num, dim_f))
        self.W = np.random.standard_normal((self.item_num, self.item_num))
        self.C = np.random.standard_normal((self.item_num, self.item_num))

        self.B_u = np.random.standard_normal(self.user_num)
        self.B_i = np.random.standard_normal(self.item_num)
        self.mu = np.mean(train_exp[train_exp != 0])
        
        (self.R_u, self.S_k_r), (self.N_u, self.S_k_n) = get_neighbor(train_exp, train_imp)

        self.loss_tr = defaultdict(float)
        self.loss_tst = defaultdict(float)

    def fit(self):
        start = datetime.now() 
        for epoch in range(config.epochs):
            # stochastic 
            n = 0
            for u in range(self.user_num):
                N_u = np.where(self.N_tr[u, :] != 0)[0]

                for i in range(self.item_num):
                    # rating 있는 애들만
                    if self.R_tr[u, i] != 0:                 
                        # p, q, bu, bi, y update
                        R_k_iu = np.intersect1d(self.R_u[u], self.S_k_r[i])
                        N_k_iu = np.intersect1d(self.N_u[u], self.S_k_n[i])
                        self.loss_tr[epoch] += self.gradient_descent(u, i, N_u, R_k_iu, N_k_iu)
                        n += 1

            self.loss_tr[epoch] = np.sqrt(self.loss_tr[epoch]/n )
            self.loss_tst[epoch] = self.evaluate()
            if epoch % 10 == 0 or epoch == config.epochs-1:
                print(f'EPOCH {epoch+1} : TRAINING RMSE {self.loss_tr[epoch]:.5f}, VALID RMSE {self.loss_tst[epoch]:.5f}')
        end = datetime.now()
        print(f'Training takes time {end-start}')
        
    def scoring(self, u, i, N_u, R_k_iu, N_k_iu):
        p = self.P[u] + np.sum(self.Y[N_u], axis=0)/np.sqrt(len(N_u))
        MF_part = np.dot(p, self.Q[i].T)
        if len(R_k_iu) > 0:
            bias_loss = self.R_tr[u, R_k_iu] - (self.mu + self.B_u[u] + self.B_i[R_k_iu])
            NB_part_exp = np.dot(bias_loss, self.W[i, R_k_iu]) / np.sqrt(len(R_k_iu))
        else:
            bias_loss = 0
            NB_part_exp = 0
        if len(N_k_iu) > 0:
            NB_part_imp = np.sum(self.C[i, N_k_iu]) / np.sqrt(len(N_k_iu))
        else: 
            NB_part_imp = 0

        return self.mu + self.B_u[u] + self.B_i[i] + MF_part + NB_part_exp + NB_part_imp, bias_loss
    
    def gradient(self, u, i, N_u, R_k_iu, N_k_iu):
        score, bias_loss = self.scoring(u, i, N_u, R_k_iu, N_k_iu)
        loss =  self.R_tr[u, i] - score
        added = np.sum(self.Y[N_u], axis=0)/np.sqrt(len(N_u))

        dp = loss*self.Q[i] - config.weight_decay*self.P[u]
        dq = loss*(self.P[u] + added) - config.weight_decay*self.Q[i]
        dbu = loss - config.weight_decay*self.B_u[u]
        dbi = loss - config.weight_decay*self.B_i[i]
        dyj = (loss*self.Q[i]/np.sqrt(len(N_u))).reshape(1, -1) - config.weight_decay*self.Y[N_u]
        dw = loss*(bias_loss)/np.sqrt(len(R_k_iu)) - config.weight_decay*self.W[i, R_k_iu]
        dc = loss/np.sqrt(len(N_k_iu)) - config.weight_decay*self.C[i, N_k_iu]
        return dp, dq, dbu, dbi, dyj, dw, dc, loss**2

    def gradient_descent(self, u, i, N_u, R_k_iu, N_k_iu):
        dp, dq, dbu, dbi, dyj, dw, dc, loss = self.gradient(u, i, N_u, R_k_iu, N_k_iu)
        
        self.P[u] = self.P[u] + config.learning_rate * dp
        self.Q[i] = self.Q[i] + config.learning_rate * dq
        self.B_u[u] = self.B_u[u] + config.learning_rate * dbu
        self.B_i[i] = self.B_i[i] + config.learning_rate * dbi
        self.Y[N_u] = self.Y[N_u] + config.learning_rate * dyj
        if len(R_k_iu) > 0:
            self.W[i, R_k_iu] = self.W[i, R_k_iu] + config.learning_rate * dw
        if len(N_k_iu) > 0:
            self.C[i, N_k_iu] = self.C[i, N_k_iu] + config.learning_rate * dc
        return loss

    def predict(self):
        pred = np.zeros((self.user_num, self.item_num))
        for u in range(self.user_num):
            N_u = np.where(self.N_tr[u, :] != 0)[0]
            for i in range(self.item_num):
                # rating 있는 애들만
                if self.R_tst[u, i] != 0:                 
                    # p, q, bu, bi, y update
                    R_k_iu = np.intersect1d(self.R_u[u], self.S_k_r[i])
                    N_k_iu = np.intersect1d(self.N_u[u], self.S_k_n[i])
                    pred[u, i], _ = self.scoring(u, i, N_u, R_k_iu, N_k_iu)
        return pred

    def evaluate(self):
        pred = self.predict()
        rating_idx = self.R_tst != 0
        
        loss_pred = np.sqrt(np.mean(np.power((self.R_tst - pred)[rating_idx], 2)))

        return loss_pred

    def plot_loss(self):
        fig, ax = plt.subplots(1,1, figsize=(10, 5))
        
        ax.plot(list(self.loss_tr.keys()), list(self.loss_tr.values()), color='orange', label='train')
        ax.plot(list(self.loss_tst.keys()), list(self.loss_tst.values()), color='green', label='valid')
        plt.legend()
        plt.show()

mf = SVD_integrated(train, train_imp, test, test_imp, config.dim_f, config.seed)

mf.fit()
```

<!-- #region id="vhB5APuTQNQI" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="jpKEP2FGQSKE" executionInfo={"status": "ok", "timestamp": 1638116703733, "user_tz": -330, "elapsed": 4043, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="063116a7-3c20-4668-b019-e7055a1b84c7"
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
