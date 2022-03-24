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
```

```python id="aB3rWtCGOm0U"
!wget -q --show-progress https://github.com/sparsh-ai/stanza/raw/S629908/rec/CDL/data/ml_100k_train.npy
!wget -q --show-progress https://github.com/sparsh-ai/stanza/raw/S629908/rec/CDL/data/ml_100k_test.npy
```

```python id="oGgWfwh7T8oe"
train = np.load('ml_100k_train.npy')
test = np.load('ml_100k_test.npy')

train = (train > 0).astype(float)
test = (test > 0).astype(float)
```

```python id="_wGFp4aZB-ce"
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
```

```python id="N3cj-gQQB-ce"
class Config:
    learning_rate = 0.001
    weight_decay = 0.01
    early_stopping_round = 0
    epochs = 20
    seed = 1995
    embed_dim = 50
    hidden_dim = [64, 32, 16]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 128

config = Config()
```

```python id="0J-y-sYmB-ce"
class TrainDataset(Dataset):
    def __init__(self, data, neg_data_per_pos_data):
        super(TrainDataset).__init__()
        self.M = data.shape[0]
        self.N = data.shape[1]
        self.data = data
        idx_mat = np.arange(self.M * self.N).reshape(self.M, self.N)
        pos_n = np.sum(data, dtype=np.int16)
        
        neg_idx = idx_mat[data == 0]
        pos_idx = idx_mat[data == 1]

        neg_sampled_idx = np.random.choice(neg_idx, pos_n*neg_data_per_pos_data, replace=False)
        self.total_rate = np.sort(np.union1d(pos_idx, neg_sampled_idx))

    def __len__(self):
        return len(self.total_rate)
        
    def __getitem__(self, i):
        idx = self.total_rate[i]
        u = int(idx // self.N)
        i = int(idx % self.M)
        r = self.data[u, i]

        return (u, i, r)

class TestDataset(Dataset):
    def __init__(self, data):
        super(TestDataset).__init__()
        self.M = data.shape[0]
        self.N = data.shape[1]
        self.data = data

    def __len__(self):
        return self.M * self.N
        
    def __getitem__(self, idx):
        u = int(idx // self.N)
        i = int(idx % self.M)
        r = self.data[u, i]
        
        return (u, i, r)
```

```python id="dNr8qTvhB-cf"
class MLP(nn.Module):
    def __init__(self, user_dim, item_dim, embed_dim, hidden_dim):
        super(MLP, self).__init__()
        self.embed_dim = embed_dim

        self.user_embed_mlp = nn.Embedding(user_dim, embed_dim)
        self.item_embed_mlp = nn.Embedding(item_dim, embed_dim)
        self.mlp_1 = nn.Linear(embed_dim*2, hidden_dim[0])
        self.mlp_2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.mlp_3 = nn.Linear(hidden_dim[1], hidden_dim[2])

        self.out = nn.Linear(hidden_dim[2], 1)
        self.sig = nn.Sigmoid()

        self._init_weight_()

    def _init_weight_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(0, 0.02)

    def forward(self, user_idx, item_idx):
        p = self.user_embed_mlp(user_idx)
        q = self.item_embed_mlp(item_idx)
        interact = torch.cat((p, q), axis=1)
        # print(interact.shape) # (batch_size, embed_dim*2)
        
        out = F.relu(self.mlp_1(interact))
        out = F.relu(self.mlp_2(out))
        out = F.relu(self.mlp_3(out))

        out = self.out(out)
        out = self.sig(out)

        return out
```

```python colab={"base_uri": "https://localhost:8080/"} id="SLx7bxeVB-cf" outputId="6b500b30-4a0a-40ea-f172-774772b52546"
seed_everything(config.seed)

train_data = TrainDataset(train, neg_data_per_pos_data=4)
test_data = TestDataset(test)

train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=config.batch_size*100, shuffle=False)

model = MLP(train.shape[0], train.shape[1], config.embed_dim, config.hidden_dim)
model.to(config.device)

optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
loss_fn = nn.BCEWithLogitsLoss()

start = datetime.now()
history = defaultdict(list)
history['best_loss'] = np.inf
for epoch in range(config.epochs):
    
    model.train()
    losses = 0
    for batch_data in train_loader:
        user = batch_data[0].to(config.device, dtype=torch.long)
        item = batch_data[1].to(config.device, dtype=torch.long)
        rate = batch_data[2].to(config.device, dtype=torch.float)

        optimizer.zero_grad()
        
        pred = model(user, item)
        loss = loss_fn(pred, rate.unsqueeze(-1))
        loss.backward()
        optimizer.step()

        losses += loss.item()
    losses /= len(train_loader) 
    history['train_losses'].append(losses)

    losses_val = 0
    for bacth_data in test_loader:
        user = batch_data[0].to(config.device, dtype=torch.long)
        item = batch_data[1].to(config.device, dtype=torch.long)
        rate = batch_data[2].to(config.device, dtype=torch.float)

        with torch.no_grad():

            pred = model(user, item)
            loss = loss_fn(pred, rate.unsqueeze(-1))
            losses_val += loss.item()

    losses_val /= len(test_loader)
    
    print(f'EPOCH {epoch+1} TRAIN LogLoss : {losses:.6f}, TEST LogLoss : {losses_val:.6f}')

    history['val_losses'].append(losses_val)
    if history['best_loss'] > losses_val:
        history['best_loss'] = losses_val
        torch.save(model.state_dict(), f'./artifacts/models/pretrained/pytorch/MLP.pth')
        print('The Model Saving...')
    # if epoch==0 or (epoch + 1) % 10 == 0 or epoch == config.epochs:

end = datetime.now()
print(f'Training takes time {end-start}')

```

```python id="PbQJLyfmB-ch"
history['best_loss']
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
