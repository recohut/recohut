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

```python id="RuiURj-F17mq"
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tqdm.notebook import tqdm
```

```python colab={"base_uri": "https://localhost:8080/"} id="aB3rWtCGOm0U" executionInfo={"status": "ok", "timestamp": 1638115805771, "user_tz": -330, "elapsed": 1255, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c00a910d-f4e2-4f29-d461-e3e7bed154a7"
!wget -q --show-progress https://github.com/sparsh-ai/stanza/raw/S629908/rec/CDL/data/ml_100k_train.npy
!wget -q --show-progress https://github.com/sparsh-ai/stanza/raw/S629908/rec/CDL/data/ml_100k_test.npy
!wget -q --show-progress https://github.com/sparsh-ai/stanza/raw/S629908/rec/CDL/data/movies.csv
```

```python id="oGgWfwh7T8oe"
train = np.load('ml_100k_train.npy')
test = np.load('ml_100k_test.npy')
item = pd.read_csv('movies.csv').iloc[:,5:].values.astype(np.float)

train = (train >= 4).astype(float)
test = (test >= 4).astype(float)
```

```python id="im82cGBg-IB2"
class Config():
    learning_rate = 0.01
    weight_decay = 5e-4
    neg_sample_size = 20
    batch_size = 1024
    margin = 0.5
    embed_dim = 15
    n_user, n_item = train.shape
    epochs = 300
    device = 'cuda:2' if torch.cuda.is_available else 'cpu'
    rankM = 50

config = Config()
```

```python id="o45_S5XH-JTq"
idx = np.arange(config.n_item)
config.neg_item_tr = {i :idx[~train[i, :].astype(bool)] for i in range(config.n_user)}
config.neg_item_tst = {i :idx[~test[i, :].astype(bool)] for i in range(config.n_user)}

config.pos_item_tr_bool = {i :train[i, :].astype(bool) for i in range(config.n_user)}
config.pos_item_tst_bool = {i :test[i, :].astype(bool) for i in range(config.n_user)}
```

```python id="xYbvJYa09j2I"
class CMLData(Dataset):
    def __init__(self, implicit_data, item_data, neg_sample_size, neg_item):
        self.implicit_data = implicit_data
        self.item_data = item_data
        self.user_pos = implicit_data.nonzero()[0]
        self.item_pos = implicit_data.nonzero()[1]
        self.n_user, self.n_item = implicit_data.shape

        self.neg_sample_size = neg_sample_size
        idx = np.arange(self.n_item)
        self.neg_item_per_user = neg_item

    def __len__(self):
        return len(self.user_pos)
    
    def __getitem__(self, idx):
        user = self.user_pos[idx]
        item = self.item_pos[idx]
        neg_item = np.random.choice(self.neg_item_per_user[user], self.neg_sample_size)
        item_feature = self.item_data[item, :]
        return {'user_idx':user,
                'item_idx':item,
                'neg_item_idx':neg_item,
                'item_x':item_feature}

class CML(nn.Module):
    def __init__(self, user_n, item_n, input_dim, embed_dim, neg_sample_size, margin, neg_item):
        super(CML, self).__init__()
        self.user_n = user_n
        self.item_n = item_n
        self.embed_dim = embed_dim
        self.input_dim= input_dim
        self.neg_sample_size = neg_sample_size
        
        self.embed_u = nn.utils.weight_norm(nn.Embedding(user_n, embed_dim))
        self.embed_v = nn.utils.weight_norm(nn.Embedding(item_n, embed_dim))
        self.mlp1 = nn.Linear(input_dim, input_dim)
        self.mlp2 = nn.Linear(input_dim, embed_dim)
        self.dropout1 = nn.Dropout(0)
        self.dropout2 = nn.Dropout(0)
    
        self.rank_ij = torch.rand(user_n, item_n).to(config.device) * 10
        self.margin = margin
        self.neg_item = neg_item
        
        self.lambda_f = torch.FloatTensor([1]).to(config.device)
        self.lambda_c = torch.FloatTensor([10]).to(config.device)        

        self.embed_u.weight.data.normal_(mean=0, std=1/embed_dim**0.5)
        self.embed_v.weight.data.normal_(mean=0, std=1/embed_dim**0.5)


    def forward(self, batch_data):
        user, item, neg_item, item_x = batch_data['user_idx'], batch_data['item_idx'], batch_data['neg_item_idx'], batch_data['item_x'].float()
        batch_size = user.size(0)
        rank = self.rank_ij[user, item].unsqueeze(-1)
        ui, vj, vk = self.embed_u(user), self.embed_v(item), self.embed_v(neg_item)

        pos_distance = self.pos_distance(ui, vj)
        neg_distance = self.neg_distance(ui, vk)
        temp = torch.relu(pos_distance - neg_distance + self.margin)
        loss_m = torch.sum(rank*temp, axis=1)
        
        item_x = self.item_feature_extractor(item_x)
        loss_f = torch.sum((item_x - vj)**2, axis=1)

        C = self.get_cov_mat(ui, vj, batch_size)
        loss_c = (torch.norm(C, p='fro') - torch.norm(torch.diagonal(C, 0), 2))/batch_size

        self.update_rank(pos_distance, neg_distance, user, item)        

        return torch.sum(loss_m), torch.sum(loss_f), loss_c, torch.sum(loss_m) + torch.sum(loss_f) * self.lambda_f + loss_c*self.lambda_c


    def update_rank(self, pos_d, neg_d, user, item):
        impost = torch.sum((pos_d + self.margin - neg_d) > 0, axis=1)
        self.rank_ij[user, item] = torch.log(impost / self.neg_sample_size * self.item_n + 1)

    def pos_distance(self, ui, vj):
        return torch.sum((ui-vj)**2, axis=1).unsqueeze(-1)

    def neg_distance(self, ui, vk):
        return torch.sum((ui.unsqueeze(axis=1) - vk)**2, axis=2)
    
    def item_feature_extractor(self, item_feature):
        item_feature = self.dropout1(item_feature)
        item_feature = torch.relu(self.mlp1(item_feature))
        item_feature = self.dropout2(item_feature)
        item_feature = torch.relu(self.mlp2(item_feature))
        return item_feature

    def get_cov_mat(self, ui, vj, batch_size):
        cat_emb = torch.cat((ui, vj), axis=0)
        mu = torch.mean(cat_emb, axis=0)
        cat_emb = cat_emb - mu
        C = torch.matmul(cat_emb.T, cat_emb) / batch_size
        return C

def eval_recallM(model, config):
    recall_tr = []
    recall_tst = []

    for u in torch.arange(config.n_user).to(config.device):
        ui = model.embed_u(u)
        scores = torch.sum((ui - model.embed_v.weight.data)**2, axis=1).detach().cpu().numpy()
        rank = scores.argsort().argsort()
        topM_mask = (rank <= config.rankM)
        pos_item_mask_tr = config.pos_item_tr_bool[u.item()] 
        pos_item_mask_tst = config.pos_item_tst_bool[u.item()] 
        if pos_item_mask_tr.sum() > 0:
            recall_tr.append((topM_mask * pos_item_mask_tr).sum() / pos_item_mask_tr.sum())
        if pos_item_mask_tst.sum() >= 5:
            recall_tst.append((topM_mask * pos_item_mask_tst).sum() / pos_item_mask_tst.sum())    
    return np.mean(recall_tr), np.mean(recall_tst)

user_pos = train.nonzero()[0]
item_pos = train.nonzero()[1]
train_dataset = CMLData(train, item, config.neg_sample_size, config.neg_item_tr)
valid_dataset = CMLData(test, item, config.neg_sample_size, config.neg_item_tr)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, drop_last=False, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size*10, drop_last=False, shuffle=False)

model = CML(config.n_user, 
            config.n_item, 
            item.shape[1],
            config.embed_dim, 
            config.neg_sample_size, 
            config.margin, 
            config.neg_item_tr,
            )

model = model.to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

history = defaultdict(list)
for epoch in range(config.epochs):
    losses = []
    model.train()
    for batch_data in train_loader:
        # model.censor_norm_()
        optimizer.zero_grad()
        batch_data = {k:v.to(config.device) for k,v in batch_data.items()}
        loss_m, loss_f, loss_c, loss = model(batch_data)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        
    recall_tr, recall_tst = eval_recallM(model, config)
    if (epoch+1) % 10 == 0 or epoch==0 or (epoch+1)==config.epochs:
        print(f'EPOCH {epoch+1} : train loss {np.sum(losses) : .0f}, train recall@{config.rankM} {recall_tr: .4f}, valid recall@{config.rankM} {recall_tst: .4f}')
    history['loss'].append(np.sum(losses))
    history['recall_tr'].append(recall_tr)
    history['recall_tst'].append(recall_tst)
```

```python id="PPfNEiFM-OIN"
fig, axes = plt.subplots(1,2)
axes[0].plot(history['loss'], label='loss')
axes[1].plot(history['recall_tr'], label='recall_train')
axes[1].plot(history['recall_tst'], label='recall_valid')
axes[0].legend()
axes[1].legend()
plt.plot()
```

<!-- #region id="vhB5APuTQNQI" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="jpKEP2FGQSKE" executionInfo={"status": "ok", "timestamp": 1638116033956, "user_tz": -330, "elapsed": 3644, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="901783a7-623e-49c7-e7d8-6569c7c567a1"
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
