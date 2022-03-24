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

<!-- #region id="SR5QPHU6lQug" -->
# SVD++ on ML-1m
<!-- #endregion -->

```python id="RuiURj-F17mq"
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime
```

```python id="aB3rWtCGOm0U"
!wget -q --show-progress https://github.com/sparsh-ai/stanza/raw/S629908/rec/CDL/data/ml_100k_train.npy
!wget -q --show-progress https://github.com/sparsh-ai/stanza/raw/S629908/rec/CDL/data/ml_100k_test.npy
```

```python id="oGgWfwh7T8oe"
train = np.load('ml_100k_train.npy')
test = np.load('ml_100k_test.npy')

train_imp = (train > 0).astype(float)
test_imp = (test > 0).astype(float)
```

```python id="N3cj-gQQB-ce"
class Config:
    learning_rate = 0.01
    weight_decay = 0.1
    early_stopping_round = 0
    epochs = 100
    seed = 1995
    dim_f = 15

config = Config()
```

```python colab={"base_uri": "https://localhost:8080/"} id="8WhLB-jSCgLa" executionInfo={"status": "ok", "timestamp": 1630646966392, "user_tz": -330, "elapsed": 260446, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2d74b630-f722-48c7-e143-47b8d3ee3b72"
class SVDpp():
    
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

        self.B_u = np.random.standard_normal(self.user_num)
        self.B_i = np.random.standard_normal(self.item_num)
        self.mu = np.mean(train_exp[train_exp != 0])
        
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
                        self.loss_tr[epoch] += self.gradient_descent(u, i, N_u)
                        n += 1
            
            self.loss_tr[epoch] = np.sqrt(self.loss_tr[epoch]/n )
            self.loss_tst[epoch] = self.evaluate()
            if epoch % 10 == 0 or epoch == config.epochs-1:
                print(f'EPOCH {epoch+1} : TRAINING RMSE {self.loss_tr[epoch]:.5f}, VALID RMSE {self.loss_tst[epoch]:.5f}')
        end = datetime.now()
        print(f'Training takes time {end-start}')
        
    def scoring(self, u, i, N_u):
        p = self.P[u] + np.sum(self.Y[N_u], axis=0)/np.sqrt(len(N_u))
        return self.mu + self.B_u[u] + self.B_i[i] + np.dot(p, self.Q[i].T)
    
    def gradient(self, u, i, N_u):
        loss =  self.R_tr[u, i] - self.scoring(u, i, N_u)
        added = np.sum(self.Y[N_u], axis=0)/np.sqrt(len(N_u))

        dp = loss*self.Q[i] - config.weight_decay*self.P[u]
        dq = loss*(self.P[u] + added) - config.weight_decay*self.Q[i]
        dbu = loss - config.weight_decay*self.B_u[u]
        dbi = loss - config.weight_decay*self.B_i[i]
        dyj = (loss*self.Q[i]/np.sqrt(len(N_u))).reshape(1, -1) - config.weight_decay*self.Y[N_u]
        
        return dp, dq, dbu, dbi, dyj, loss**2

    def gradient_descent(self, u, i, N_u):
        dp, dq, dbu, dbi, dyj, loss = self.gradient(u, i, N_u)
        self.P[u] = self.P[u] + config.learning_rate * dp
        self.Q[i] = self.Q[i] + config.learning_rate * dq
        self.B_u[u] = self.B_u[u] + config.learning_rate * dbu
        self.B_i[i] = self.B_i[i] + config.learning_rate * dbi
        self.Y[N_u] = self.Y[N_u] + config.learning_rate * dyj
        
        return loss

    def predict(self):
        P = np.zeros(self.P.shape)
        for u in range(self.user_num):
            N_u = np.where(self.N_tr[u, :] != 0)[0]
            P[u] = self.P[u] + np.sum(self.Y[N_u], axis=0) / np.sqrt(len(N_u))

        return self.mu + self.B_u.reshape(-1, 1) + self.B_i.reshape(1, -1) + np.dot(P, self.Q.T)

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

mf = SVDpp(train, train_imp, test, test_imp, config.dim_f, config.seed)

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
