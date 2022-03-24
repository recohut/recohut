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

<!-- #region id="8OiTWf8-UHea" -->
# Equal Experience in Recommender Systems on ML-1m and Synthetic biased data
<!-- #endregion -->

<!-- #region id="2TVkI_xCUHb8" -->
## Executive summary

| | |
| --- | --- |
| Problem | Biased data due to inherent stereotypes of particular groups (e.g., male students’ average rating on mathematics is often higher than that on humanities, and vice versa for females) may yield a limited scope of suggested items to a certain group of users. |
| Solution | The novel fairness notion, coined ***Equal Experience***, tries to capture the degree of the equal experience of item recommendations across distinct groups. Specifically, the prediction $\hat{Y}$ should be independent of the 1) user group $Z_{user}$, and 2) item group $Z_{item}$. Formally, $I(\hat{Y};Z_{user},Z_{item}) = I(\hat{Y};Z_{item}) + I(\hat{Y};Z_{user}|Z_{item}) = I(\hat{Y};Z_{user}) + I(\hat{Y};Z_{item}|Z_{user})$ |
| Dataset | ML-1m, LastFM-360k, Synthetic. |
| Preprocessing | For ML-1m, we divide user and item groups based on gender and genre, respectively. Action, crime, film-noir, war are selected as male-preferred genre, whereas children, fantasy, musical, romance are selected as female-preferred genre. We can select male-preferred and female-preferred genres in a variety of ways based on ratings and observations. In case of LastFM-360k, the associated task is to predict whether the user likes the artist or not. The data for play counts is converted to binary rating. We divide user and item groups based on gender and genre, respectively. We also randomly select 5000 male and 5000 female users. Among 10 genres, we choose hip-hop and musical for male and female preferred genres, respectively. The final rating matrix of 10,000 users and 5,706 artists is 0.55% full. We randomly split the real datasets into 90% train set and 10% test set. In case of MovieLens data, the rating is five-star based, so we set the threshold $\tau$ = 3, on the other hand, for LastFM and for synthetic dataset, we set $\tau$ = 0 as $M_{ij} \in \{+1, −1\}$. |
| Metrics | RMSE, DEE, VAL, UGF, CVS |
| Models | MF class models, AE class models |
| Cluster | Python 3.6+, PyTorch |
| Tags | `Fairness`, `MatrixFactorization`, `AutoEncoder`,  `ExposureBias`, `PopulationBias` |
| Credits | cjw2525 |
<!-- #endregion -->

<!-- #region id="qDzuVLpuUHZv" -->
## Process flow

![](https://github.com/RecoHut-Stanzas/S035564/raw/main/images/process_flow.svg)
<!-- #endregion -->

<!-- #region id="iuYv-2Vn4DIk" -->
## Setup
<!-- #endregion -->

```python id="7b_o3k0T4HIu"
!pip install livelossplot
```

```python id="URl1qly74HGN"
import numpy as np
import pandas as pd
import random
from tqdm.notebook import tqdm
import math
import collections
import itertools

import os
import sys

import torch
import torch.nn as nn
from torch import optim
from torch.nn import utils
import torch.nn.functional as F

import matplotlib.pyplot as plt
from time import sleep
from livelossplot import PlotLosses  

%matplotlib inline
```

<!-- #region id="ppialzra4HD0" -->
## Data
<!-- #endregion -->

<!-- #region id="fs6mK_pG-LCj" -->
### ML-1m
<!-- #endregion -->

<!-- #region id="RkfFSurY4YtM" -->
Download
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="AYqYdfh64HBC" executionInfo={"status": "ok", "timestamp": 1639234435404, "user_tz": -330, "elapsed": 1804, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5e29d399-8902-4a9c-e169-e55cc7a340d0"
!wget -q --show-progress http://files.grouplens.org/datasets/movielens/ml-1m.zip
!unzip ml-1m.zip
```

<!-- #region id="007w6cqU4G-l" -->
Preprocessing
<!-- #endregion -->

```python id="S46eIOrG4bP8"
def data_loader_movielens():
    path = './ml-1m/'
    num_users, num_items = 6040, 3952
      
    data = load_data(path, num_users, num_items, train_ratio=.9)
    user, _ = load_users(path)
    genre = load_items(path, option='single')
    item = {}
    item['M'] = genre['War']+genre['Crime']+genre['Film-Noir']+genre['Sci-Fi']
    item['F'] = genre['Children\'s']+genre['Fantasy']+genre['Musical']+genre['Romance']
    
    return data, user, item
```

```python id="yEzkuTdz4hhn"
def load_users(path):
    f = open(path + "users.dat")
    lines = f.readlines()

    gender, age = {}, {} # generate dictionaries
    gender_index, age_index = ['M', 'F'], [1, 18, 25, 35, 45, 50, 56]

    for i in gender_index:
        gender[i] = []
    for i in age_index:
        age[i] = []  
    for line in lines:
        user, g, a, *args = line.split("::")
        gender[g].append(int(user) - 1)
        age[int(a)].append(int(user) - 1) 

    return gender, age
```

```python id="bm_fVrwe4s-_"
def load_items(path, option='multiple_genre'):
    f = open(path + "movies.dat", encoding = "ISO-8859-1")
    lines = f.readlines()

    genre={}
    genre_index = ['Action', 'Adventure', 'Animation', 'Children\'s', 
                   'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                   'Film-Noir', 'Horror', 'Musical', 'Mystery', 
                   'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    for idx in genre_index:
        genre[idx] = []

    for line in lines:
        item, _, tags = line.split("::")
        tags = tags.split('|')
        tags[-1] = tags[-1][:-1]
        if option=='multiple_genre':
            for tag in tags:
                genre[tag].append(int(item) - 1)
        else:
            tag = tags[0]
            genre[tag].append(int(item)-1)
    return genre
```

```python id="DuZ3Eh6n4vI4"
def load_data(path, num_users, num_items, train_ratio):
    '''
    Read data by lines and produce train/test data matrices.
    '''

    f = open(path + "ratings.dat")
    lines = f.readlines()
    random.shuffle(lines)  

    num_ratings = len(lines)

    X_train = np.zeros((num_users, num_items))
    X_test = np.zeros((num_users, num_items))

    for i, line in enumerate(lines):
        user, item, rating, _ = line.split("::")
        user_idx = int(user) - 1
        item_idx = int(item) - 1
        if i < int(num_ratings * train_ratio):
            X_train[user_idx, item_idx] = float(rating)
        else:
            X_test[user_idx, item_idx] = float(rating)

    return (X_train, X_test)
```

<!-- #region id="We9HslPj-bvL" -->
### Synthetic dataset
<!-- #endregion -->

```python id="0FFfZaP5-esD"
def data_loader_synthetic(p=.2, q=.2, r=.2, s=.2, rank=10, seed=42):
    '''
    ground truth matrix Y
    '''
    num_users, num_items = 600, 400
    n1, n2 = num_users // 2, num_items // 2
    np.random.seed(42)
    Y_1 = np.where(np.random.random((rank, n2)) < p, 1, -1)
    np.random.seed(42)
    Y_2 = np.where(np.random.random((rank, n2)) < q, 1, -1)
    Y_rank = np.concatenate((Y_1, Y_2), axis = 1)

    Y_m = Y_rank.copy()

    for i in range(num_users // (rank * 2) -1):
        Y_m = np.concatenate((Y_m, Y_rank))
    np.random.seed(43)
    Y_1 = np.where(np.random.random((rank, n2)) < q, 1, -1)
    np.random.seed(43)
    Y_2 = np.where(np.random.random((rank, n2)) < p, 1, -1)
    Y_rank = np.concatenate((Y_1, Y_2), axis = 1)

    Y_f = Y_rank.copy()

    for i in range(num_users // (rank * 2) -1):
        Y_f = np.concatenate((Y_f, Y_rank))
    
    np.random.shuffle(Y_m)
    np.random.shuffle(Y_f)
    Y = np.concatenate((Y_m, Y_f))
    
    I_obs_mm = np.where(np.random.random((n1, n2)) < r, 1, 0)
    I_obs_mf = np.where(np.random.random((n1, n2)) < s, 1, 0)
    I_obs_fm = np.where(np.random.random((n1, n2)) < s, 1, 0)
    I_obs_ff = np.where(np.random.random((n1, n2)) < r, 1, 0)

    I_obs_m = np.concatenate((I_obs_mm, I_obs_mf), axis = 1)
    I_obs_f = np.concatenate((I_obs_fm, I_obs_ff), axis = 1)

    I_obs = np.concatenate((I_obs_m, I_obs_f))
    
    Y_obs = Y * I_obs
    
    Y_train, Y_test = Y_obs, (Y-Y_obs)
#     Y_train, Y_test = np.zeros((num_users, num_items)), np.zeros((num_users, num_items))

#     for i in tqdm(range(num_users)):
#         for j in range(num_items):
#             if Y_obs[i, j] != 0:
#                 k = np.random.random()
#                 if k > 0.9:
#                     Y_test[i, j] = Y_obs[i, j]
#                 else:
#                     Y_train[i, j] = Y_obs[i, j]
                    
                    
    user, item = {}, {}
    user['M'] = [x for x in range(n1)]
    user['F'] = [x for x in range(n1, n1*2)]
    item['M'] = [x for x in range(n2)]
    item['F'] = [x for x in range(n2, n2*2)]
                    
    return (Y_train, Y_test), user, item
```

<!-- #region id="Ud6B5UnK5CFF" -->
## Metrics
<!-- #endregion -->

```python id="W9TbcwrE5LEB"
def metrics(model, data_tuple, device, model_type='AE', tau=3):
    
    data, gender, item = data_tuple
    measures = {}
    
    with torch.no_grad(): 
        model.eval()
        Y_train, Y_test = data[0], data[1]
        if model_type=='PQ':
            identity = torch.from_numpy(np.eye(Y_train.shape[0])).float().to(device)
            pred = model(identity).cpu().detach().numpy()
        else:
            pred = model(torch.tensor(Y_train).float().to(device)).cpu().detach().numpy()   
        # 1. rmse
        test_rmse = np.sqrt(np.mean((Y_test[Y_test != 0] - pred[Y_test != 0]) ** 2))
        # Y_tilde 
        pred_hat = np.where(pred > tau, 1, 0)
        
        # 2. DEE
        DEE = 0
        for g in ['M', 'F']:
            for i in ['M', 'F']:
                DEE += np.abs(np.mean(pred_hat)-np.mean(pred_hat[gender[g]][:, item[i]]))
        # 3. value_fairness
        VAL = VAL_measure(pred, data, gender, device)
        # 4. DP_user
        UGF = 0
        for g in ['M', 'F']:
            UGF += np.abs(np.mean(pred_hat)-np.mean(pred_hat[gender[g]]))
        # 4. DP_item
        CVS = 0
        for i in ['M', 'F']:
            CVS += np.abs(np.mean(pred_hat)-np.mean(pred_hat[:, item[i]]))
        measures['RMSE'] = test_rmse
        measures['DEE'] = DEE
        measures['VAL'] = VAL 
        measures['UGF'] = UGF
        measures['CVS'] = CVS
    return measures
```

```python id="rQ9GDLXZ5Lt0"
def VAL_measure(pred, data, gender, device):
    train_data = data[0]
    mask = np.where(train_data!=0, 1, 0)

    y_m = train_data[gender['M']]
    y_f = train_data[gender['F']]
    y_hat_m = pred[gender['M']]
    y_hat_f = pred[gender['F']]

    #average ratings
    d_m = np.abs(np.sum(y_m, axis=0)/(np.sum(mask[gender['M']], axis=0)+1e-8)-np.sum(y_hat_m, axis=0)/len(gender['M']))
    d_f = np.abs(np.sum(y_f, axis=0)/(np.sum(mask[gender['F']], axis=0)+1e-8)-np.sum(y_hat_f, axis=0)/len(gender['F']))

    v_fairness = np.mean(np.abs(d_m-d_f))
    return v_fairness
```

<!-- #region id="8W9k77hT4-v8" -->
## Regularizers
<!-- #endregion -->

```python id="K_dXakNu5fBq"
def normal_pdf(x):
    import math
    return torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

def normal_cdf(y, h=0.01, tau=0.5):
    # Approximation of Q-function given by López-Benítez & Casadevall (2011)
    # based on a second-order exponential function & Q(x) = 1 - Q(-x):
    Q_fn = lambda x: torch.exp(-0.4920*x**2 - 0.2887*x - 1.1893)
    m = y.shape[0]*y.shape[1]
    y_prime = (tau - y) / h
    sum_ = torch.sum(Q_fn(y_prime[y_prime > 0])) \
           + torch.sum(1 - Q_fn(torch.abs(y_prime[y_prime < 0]))) \
           + 0.5 * len(y_prime[y_prime == 0])
    return sum_ / m

def Huber_loss(x, delta):
    if abs(x) < delta:
        return (x ** 2) / 2
    return delta * (x.abs() - delta / 2)

def Huber_loss_derivative(x, delta):
    if x > delta:
        return delta/2
    elif x < -delta:
        return -delta/2
    return x
```

```python id="moZ6xr6G5iPB"
class FairnessLoss():
    def __init__(self, h, tau, delta, device, data_tuple, type_='EqualExp'):
        self.h = h
        self.tau = tau
        self.delta = delta
        self.device = device
        self.type_ = type_
        self.data_tuple = data_tuple

    def DEE(self, y_hat, gender, item):
        backward_loss = 0
        logging_loss_ = 0 
        
        for gender_key in ['M','F']:
            for item_key in ['M', 'F']:
                gender_idx = gender[gender_key] 
                item_idx = item[item_key]
                m_gi = len(gender_idx)*len(item_idx)
                y_hat_gender_item = y_hat[gender_idx][:, item_idx]

                Prob_diff_Z = normal_cdf(y_hat.detach(), self.h, self.tau)-normal_cdf(y_hat_gender_item.detach(), self.h, self.tau)
                
                _dummy = Huber_loss_derivative(Prob_diff_Z, self.delta)
                _dummy *= \
                    torch.dot(
                        normal_pdf((self.tau - y_hat.detach()) / self.h).reshape(-1), 
                        y_hat.reshape(-1)
                    ) / (self.h * y_hat.shape[0]*y_hat.shape[1]) -\
                    torch.dot(
                        normal_pdf((self.tau - y_hat_gender_item.detach()) / self.h).reshape(-1), 
                        y_hat_gender_item.reshape(-1)
                    ) / (self.h * m_gi)
                backward_loss += _dummy
        return backward_loss
        
    def VAL(self, y_hat, gender, item):
        device = self.device
        
        backward_loss = 0
        
        data = self.data_tuple[0]
        train_data = data[0]
        mask = np.where(train_data!=0, 1, 0)

        train_data = torch.from_numpy(train_data).to(device)
        mask = torch.from_numpy(mask).to(device)

        y_m = train_data[gender['M']]
        y_f = train_data[gender['F']]
        y_hat_m = y_hat[gender['M']]
        y_hat_f = y_hat[gender['F']]

        #average ratings
        d_m = torch.abs(torch.sum(y_m, axis=0)/(torch.sum(mask[gender['M']], axis=0)+1e-8)
        -torch.sum(y_hat_m, axis=0)/len(gender['M']))

        d_f = torch.abs(torch.sum(y_f, axis=0)/(torch.sum(mask[gender['F']], axis=0)+1e-8)
        -torch.sum(y_hat_f, axis=0)/len(gender['F']))


        backward_loss = torch.mean(torch.abs(d_m-d_f))
        
        return backward_loss
    
    def UGF(self, y_hat, gender, item):
        backward_loss = 0
        
        for key in ['M', 'F']:
            
            gender_idx = gender[key]
            m_i = y_hat.shape[1]*len(gender_idx)
            y_hat_group = y_hat[gender_idx]
            
            Prob_diff_Z = normal_cdf(y_hat.detach(), self.h, self.tau)-normal_cdf(y_hat_group.detach(), self.h, self.tau)

            _dummy = Huber_loss_derivative(Prob_diff_Z, self.delta)
            _dummy *= \
                torch.dot(
                    normal_pdf((self.tau - y_hat.detach()) / self.h).reshape(-1), 
                    y_hat.reshape(-1)
                ) / (self.h * y_hat.shape[0]*y_hat.shape[1]) -\
                torch.dot(
                    normal_pdf((self.tau - y_hat_group.detach()) / self.h).reshape(-1), 
                    y_hat_group.reshape(-1)
                ) / (self.h * m_i)
            backward_loss += _dummy
        return backward_loss
    
    def CVS(self, y_hat, gender, item):
        backward_loss = 0
        
        for key in ['M', 'F']:
            item_idx = item[key]
            m_i = y_hat.shape[0]*len(item_idx)
            y_hat_group = y_hat[:, item_idx]

            Prob_diff_Z = normal_cdf(y_hat.detach(), self.h, self.tau)-normal_cdf(y_hat_group.detach(), self.h, self.tau)

            _dummy = Huber_loss_derivative(Prob_diff_Z, self.delta)
            _dummy *= \
                torch.dot(
                    normal_pdf((self.tau - y_hat.detach()) / self.h).reshape(-1), 
                    y_hat.reshape(-1)
                ) / (self.h * y_hat.shape[0]*y_hat.shape[1]) -\
                torch.dot(
                    normal_pdf((self.tau - y_hat_group.detach()) / self.h).reshape(-1), 
                    y_hat_group.reshape(-1)
                ) / (self.h * m_i)
            backward_loss += _dummy
        return backward_loss
        
    
    def __call__(self, y_hat, gender, item):
        if self.type_ == 'EqualExp':
            return self.DEE(y_hat, gender, item)
        elif self.type_ == 'VAL':
            return self.VAL(y_hat, gender, item)
        elif self.type_ == 'UGF':
            return self.UGF(y_hat, gender, item)
        elif self.type_ == 'CVS':
            return self.CVS(y_hat, gender, item)
```

<!-- #region id="2ne-ZTmK5jrW" -->
## Models
<!-- #endregion -->

<!-- #region id="U1e_aTnS5nIo" -->
### Matrix factorization
<!-- #endregion -->

```python id="W98hqtpw5uze"
class PQ(nn.Module):
    def __init__(self, rating, num_users, num_items, rank):
        super(PQ, self).__init__()
        
        self.rating = rating
        
        self.encoder = nn.Sequential(nn.Linear(num_users, rank, bias=False))
        self.decoder = nn.Sequential(nn.Linear(rank, num_items, bias=False))

    def forward(self, x):
        if self.rating == 'binary':
            x = self.decoder(self.encoder(x))
            x = torch.tanh(x)
        elif self.rating == 'five-stars':
            x = self.decoder(self.encoder(x))
            x = torch.clamp(x, 0, 5.0)
        else:
            raise KeyError("unavailable rating scale")
    
        return x
```

<!-- #region id="EuGauSWg5vJ-" -->
### Autoencoder
<!-- #endregion -->

```python id="NIxPMCss5yKa"
class AE(nn.Module):
    def __init__(self, rating, num_user):
        super(AE, self).__init__()
        
        self.rating = rating
        self.encoder = nn.Sequential(
          nn.Linear(num_user, 512),
          nn.ReLU(),
          nn.Linear(512, 512),
          nn.Dropout(0.7),
          nn.ReLU(),
        )
        self.decoder = nn.Sequential(
          nn.Linear(512, num_user),
        )
        
    def forward(self, x):
        x = torch.transpose(x,0,1)
        if self.rating == 'binary':
            x = self.decoder(self.encoder(x))
            x = torch.tanh(x)
        elif self.rating == 'five-stars':
            x = (x - 1) / 4.0
            x = self.decoder(self.encoder(x))
            x = torch.clamp(x, 0, 1.0)
            x = 4.0 * x + 1
        x = torch.transpose(x,0,1)
        return x
```

<!-- #region id="Yadz1mj55z0U" -->
## Trainer
<!-- #endregion -->

```python id="XAGDJQWV53_f"
def mklogs():
    logs = {'train_loss':[], 
            'train_f_loss':[], 
            'RMSE':[],
            'acc':[], 
            'DEE':[], 
            'DP_user':[], 
            'DP_item':[], 
            'v_fairness':[]
           }
    return logs 
```

```python id="xSYdzaOm59c1"
def get_test_logs(logs, log):
    measures=['RMSE',
              'acc', 
              'DEE', 
              'DP_user', 
              'DP_item', 
              'v_fairness']
    for measure in measures:
        logs[measure].append(log[measure])
```

```python id="P6CGzYZy5_EM"
def train_PQ(data_tuple, model, optimizer, 
             num_epochs, device, l_value=0., lambda_=0., f_criterion=None, tau=3):

    logs = mklogs()
    data, gender, item = data_tuple
    
    # data_input 
    train_data = torch.from_numpy(data[0]).float().to(device) 
    test_data = data[1]
    
    identity = torch.from_numpy(np.eye(data[0].shape[0])).float().to(device)
    
    x = train_data
    mask = x.clone().detach()
    mask = torch.where(mask != 0, torch.ones(1).to(device), torch.zeros(1).to(device)).float().to(device)
    count = torch.sum(mask).item()
    
    #losses
    criterion = nn.MSELoss(reduction='sum')
    
    for epoch in range(num_epochs):
        rmse, cost = 0, 0
        model.train()
        W, V = model.encoder[0].weight, model.decoder[0].weight
        W_fro, V_fro = torch.sum(W ** 2), torch.sum(V ** 2)
        
        x_hat = model(identity)
        loss = 0 
        loss += (1-lambda_)*(criterion(x * mask, x_hat * mask)/count + l_value / 2 * ( W_fro + V_fro ))
        if f_criterion!=None: 
            f_loss = f_criterion(x_hat, gender, item)
            logs['train_f_loss'].append(f_loss.item())
            loss += lambda_*f_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        logs['train_loss'].append(loss.item())
        
    return logs
```

```python id="3zMMz4wX6Br2"
def train_AE(data_tuple, model, optimizer, 
             num_epochs, device, l_value=0., lambda_=0., f_criterion=None, tau=3):

    logs = mklogs()
    data, gender, item = data_tuple
    
    # data_input 
    train_data = torch.from_numpy(data[0]).float().to(device) 
    test_data = data[1]
    
    x = train_data
    mask = x.clone().detach()
    mask = torch.where(mask != 0, torch.ones(1).to(device), torch.zeros(1).to(device)).float().to(device)
    count = torch.sum(mask).item()
    
    #losses
    criterion = nn.MSELoss(reduction='sum')
    
    for epoch in range(num_epochs):
        rmse, cost = 0, 0
        model.train()
        W, V = model.encoder[0].weight, model.decoder[0].weight
        W_fro, V_fro = torch.sum(W ** 2), torch.sum(V ** 2)
        
        x_hat = model(x)
        loss = 0 
        loss += (1-lambda_)*(criterion(x * mask, x_hat * mask)/count + l_value / 2 * ( W_fro + V_fro ))
        if f_criterion!=None: 
            f_loss = f_criterion(x_hat, gender, item)
            logs['train_f_loss'].append(f_loss.item())
            loss += lambda_*f_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        logs['train_loss'].append(loss.item())
        
    return logs
```

<!-- #region id="7Ss_Tjop6DYA" -->
## Run
<!-- #endregion -->

```python id="5mNqwf4g67rH"
class Args:
    def __init__(self, dataset='movielens'):
        self.dataset = dataset
        self.p0, self.p1 = 0.4, 0.1
        self.q0, self.q1 = 0.2, 0.2
        self.data_type = 'binary'
        self.model_type='PQ' # Choose model type: 'PQ' or 'AE'
        self.algorithm_type = 'EqualExp' # Choose algorithm: 'unfair', 'EqualExp', 'VAL', 'UGF', 'CVS'
        if self.dataset=='movielens':
            self.data_type = 'five-stars'
            self.data_tuple = (data_loader_movielens())
        # elif self.dataset=='lastfm':
        #     self.data_tuple = (data_loader_lastfm())
        elif self.dataset=='synthetic':
            self.data_tuple = (data_loader_synthetic(self.p0, self.p1, self.q0, self.q1)) # return ((train_data, test_data), user attribute, item attribute)
        self.learning_rate = 1e-3
        self.l_value = 0
        self.num_epochs = 1000
        self.lambda_ = 0.99
        self.tau=0
        self.n, self.m = self.data_tuple[0][0].shape[0], self.data_tuple[0][0].shape[1]
        self.r = 20
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if self.algorithm_type == 'unfair':
            self.f_criterion = None 
        else: 
            self.f_criterion = FairnessLoss(h=0.01, tau=self.tau, delta=0.01,
                                            device=self.device, 
                                            data_tuple=self.data_tuple, 
                                            type_=self.algorithm_type)
```

<!-- #region id="Ngd5Il6H_Yfw" -->
### Experiment on ML-1m
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="EO2rNtM__cAW" outputId="6961d4cd-dc9e-47f5-90e0-e999d0b426a2" executionInfo={"status": "ok", "timestamp": 1639240054142, "user_tz": -330, "elapsed": 231355, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
args = Args(dataset='movielens')

results = []

model_type = ['PQ','AE']
algorithm_type = ['unfair', 'EqualExp', 'VAL', 'UGF', 'CVS']

for m in model_type:
    for algo in algorithm_type:
        try:
            args.model_type = m
            args.algorithm_type = algo
            # train the model
            if args.model_type == 'PQ':
                model = PQ(args.data_type, args.n, args.m, 20).to(args.device)
                optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)
                logs = train_PQ(args.data_tuple, model, optimizer, args.num_epochs,
                                args.device, l_value=args.l_value, lambda_=args.lambda_,
                                f_criterion=args.f_criterion, tau=args.tau)
            elif args.model_type == 'AE':
                model = AE(args.data_type, args.n).to(args.device)
                optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)
                logs = train_AE(args.data_tuple, model, optimizer, args.num_epochs,
                                args.device, l_value=args.l_value, lambda_=args.lambda_,
                                f_criterion=args.f_criterion, tau=args.tau)
                
            result = metrics(model, args.data_tuple, args.device, args.model_type, args.tau)
            print(result)
            results.append((m, algo, result))
        except:
            pass
```

```python colab={"base_uri": "https://localhost:8080/", "height": 363} id="X0Nn457P6Izo" executionInfo={"status": "ok", "timestamp": 1639240444307, "user_tz": -330, "elapsed": 677, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e9fff065-af7f-4047-c2a3-47c6e589048c"
df_ml = pd.DataFrame.from_records(results)
df_ml.columns = ['Model','Algorithm','Metrics']
df_ml = pd.concat([df_ml.drop('Metrics', axis=1), pd.DataFrame(df_ml['Metrics'].tolist())], axis=1)
df_ml
```

```python colab={"base_uri": "https://localhost:8080/"} id="xT93cF71Vsyx" executionInfo={"status": "ok", "timestamp": 1639240679604, "user_tz": -330, "elapsed": 169936, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="bb4a48f7-505b-417e-c32f-97912e084291"
args = Args(dataset='synthetic')

results = []

model_type = ['PQ','AE']
algorithm_type = ['unfair', 'EqualExp', 'VAL', 'UGF', 'CVS']

for m in model_type:
    for algo in algorithm_type:
        try:
            args.model_type = m
            args.algorithm_type = algo
            # train the model
            if args.model_type == 'PQ':
                model = PQ(args.data_type, args.n, args.m, 20).to(args.device)
                optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)
                logs = train_PQ(args.data_tuple, model, optimizer, args.num_epochs,
                                args.device, l_value=args.l_value, lambda_=args.lambda_,
                                f_criterion=args.f_criterion, tau=args.tau)
            elif args.model_type == 'AE':
                model = AE(args.data_type, args.n).to(args.device)
                optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)
                logs = train_AE(args.data_tuple, model, optimizer, args.num_epochs,
                                args.device, l_value=args.l_value, lambda_=args.lambda_,
                                f_criterion=args.f_criterion, tau=args.tau)
                
            result = metrics(model, args.data_tuple, args.device, args.model_type, args.tau)
            print(result)
            results.append((m, algo, result))
        except:
            pass
```

```python colab={"base_uri": "https://localhost:8080/", "height": 363} id="t7ok9CM_WK5v" executionInfo={"status": "ok", "timestamp": 1639240711022, "user_tz": -330, "elapsed": 493, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5647825b-3802-4e0a-b9c5-0e6c493f6a43"
df_synthetic = pd.DataFrame.from_records(results)
df_synthetic.columns = ['Model','Algorithm','Metrics']
df_synthetic = pd.concat([df_synthetic.drop('Metrics', axis=1), pd.DataFrame(df_synthetic['Metrics'].tolist())], axis=1)
df_synthetic
```

<!-- #region id="OSbwthtbXK3Z" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="vgwQKjJdXK3a" executionInfo={"status": "ok", "timestamp": 1639240756506, "user_tz": -330, "elapsed": 3550, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="cf020ca7-97e7-4997-cf71-09285d4caefe"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="PMrh5sONXK3b" -->
---
<!-- #endregion -->

<!-- #region id="XTDWWX2IXK3c" -->
**END**
<!-- #endregion -->
