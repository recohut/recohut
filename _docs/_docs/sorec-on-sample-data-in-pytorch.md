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

```python colab={"base_uri": "https://localhost:8080/"} id="upf8sCUcb9_8" executionInfo={"status": "ok", "timestamp": 1630652093572, "user_tz": -330, "elapsed": 424, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="96c3a1e4-23c6-4954-f513-e65a1937d9e9"
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime

# toy example Thanks to NK
trust=np.array([[0.0,0.0,0.0,0.0,0.0,0.0],
            [0.0,0.0,0.0,1.0,0.0,0.8],
            [0.8,0.0,0.0,0.0,0.0,0.0],
            [0.8,1.0,0.0,0.0,0.6,0.0],
            [0.0,0.0,0.4,0.0,0.0,0.8],
            [0.0,0.0,0.0,0.0,0.0,0.0]])

train=np.array([[5.0,2.0,0.0,3.0,0.0,4.0,0.0,0.0],
            [4.0,3.0,0.0,0.0,5.0,0.0,0.0,0.0],
            [4.0,0.0,2.0,0.0,0.0,0.0,2.0,4.0],
            [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
            [5.0,1.0,2.0,0.0,4.0,3.0,0.0,0.0],
            [4.0,3.0,0.0,2.0,4.0,0.0,3.0,5.0]])

class Config:
    learning_rate = 0.01
    weight_decay = 0.1
    early_stopping_round = 0
    epochs = 10
    seed = 1995
    dim_f = 15

config = Config()

class SoRec():
    
    def __init__(self, train_exp, network, test_exp, dim_f, seed):
        self.R_tr = train_exp
        self.R_tst = test_exp
        self.trust = network
        self.adj = np.where(self.trust > 0, 1, 0)
        
        self.rate_max = self.R_tr.max()

        self.r_tr = self.scale_rating(train_exp)
        self.r_tst = self.scale_rating(test_exp)

        self.dim_f = dim_f
        self.user_num = train_exp.shape[0]
        self.item_num = train_exp.shape[1]

        np.random.seed(seed)
        self.U = np.random.standard_normal((self.user_num, dim_f))
        self.V = np.random.standard_normal((self.item_num, dim_f))
        self.Z = np.random.standard_normal((self.user_num, dim_f))
        
        self.loss_tr = defaultdict(float)
        self.loss_tst = defaultdict(float)

        self.lambda_c = 0.1
        self.lambda_u = 0.1
        self.lambda_v = 0.1
        self.lambda_z = 0.1


    def fit(self):
        weight = self.modified_trust()
        self.trust_s = self.trust * weight

        start = datetime.now()
        for epoch in range(config.epochs):
            # stochastic 
            n = 0
            for u in range(self.user_num):
                for i in range(self.item_num):
                    # rating 있는 애들만
                    if self.R_tr[u, i] != 0:                 
                        # p, q, bu, bi, y update
                        self.loss_tr[epoch] += self.gradient_descent(u, i)
                        n += 1
            
            self.loss_tr[epoch] = np.sqrt(self.loss_tr[epoch]/n )
            print(f'EPOCH {epoch+1} : TRAINING RMSE {self.loss_tr[epoch]:.5f}, VALID RMSE {self.loss_tr[epoch]:.5f}')
        end = datetime.now()
        print(f'Training takes time {end-start}')
    
    def modified_trust(self):
        out_degree = self.adj.sum(axis=1)
        in_degree = self.adj.sum(axis=0)
        weight = np.zeros((len(out_degree), len(out_degree)))
        for i in range(len(out_degree)):
            for j in range(len(out_degree)):
                weight[i, j] = np.sqrt(in_degree[j] / (out_degree[i]+in_degree[j]))
        return weight

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def scale_rating(self, x):
        return (x-1) / (self.rate_max-1)
    
    def rescale_rating(self, x):
        return (self.rate_max - 1)*x + 1

    def scoring_rate(self, u, i):
        return  self.rescale_rating(self.sigmoid(self.U[u, :].dot(self.V[i, :])))
    
    def scoring_adj(self, u, i):
        return  self.rescale_rating(self.sigmoid(self.U[u, :].dot(self.Z[u, :])))


    def loss_fn(self, u, i):
        pred = self.scoring_rate(u, i)
        adj = self.scoring_adj(u, i)

        comp1 = self.r_tr[u, i] - pred
        comp2 = self.trust_s[u,u] - adj
        loss = comp1**2 + comp2**2 * self.lambda_c/2 + (self.U[u, :]**2).sum()/2*self.lambda_u + (self.Z[u, :]**2).sum()/2*self.lambda_z + (self.V[:, i]**2).sum()/2*self.lambda_z
        return comp1, comp2, loss


    def gradient(self, u, i):
        comp1, comp2, loss =  self.loss_fn(u, i)

        du = comp1*self.scoring_rate(u, i)*self.V[i,:] + self.lambda_c*self.scoring_adj(u, i)*comp2*self.Z[u, :] + self.lambda_u*self.U[u, :]
        dv = comp1*self.scoring_rate(u, i)*self.U[u,:] + self.lambda_v*self.V[i, :]
        dz = self.lambda_c*self.scoring_adj(u, i)*comp2*self.U[u, :] + self.lambda_z*self.Z[u, :]
        
        return du, dv, dz, loss

    def gradient_descent(self, u, i):
        du, dv, dz, loss = self.gradient(u, i)
        self.U[u] = self.U[u] + config.learning_rate * du
        self.V[i] = self.V[i] + config.learning_rate * dv
        self.Z[u] = self.Z[u] + config.learning_rate * dz
        
        return loss

    def predict(self):
        score = self.rescale_rating(self.U.dot(self.V.T))
        return score

if __name__ == '__main__':
    mf = SoRec(train, trust, train, config.dim_f, config.seed)
    mf.fit()
```
