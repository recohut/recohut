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
import os, sys
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
```

```python colab={"base_uri": "https://localhost:8080/"} id="aB3rWtCGOm0U" executionInfo={"status": "ok", "timestamp": 1638115577496, "user_tz": -330, "elapsed": 829, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7d0471b0-b255-4f79-8c5f-57b5e660de59"
!wget -q --show-progress https://github.com/sparsh-ai/stanza/raw/S629908/rec/CDL/data/ml_100k_train.npy
!wget -q --show-progress https://github.com/sparsh-ai/stanza/raw/S629908/rec/CDL/data/ml_100k_test.npy
```

```python id="GHovGk_c3G5F"
class AutoRecData(Dataset):
    def __init__(self, train, based_on):
        super(AutoRecData, self).__init__()
        self.train = train
        self.based_on = based_on
        self.n_user, self.n_item = train.shape

    def __len__(self):
        if self.based_on == 'item':
            return self.n_item
        elif self.based_on == 'user':
            return self.n_user
    
    def __getitem__(self, idx):
        if self.based_on == 'item':
            return torch.tensor(self.train[:, idx]).float()
        elif self.based_on == 'user':
            return torch.tensor(self.train[idx, :]).float()
```

```python id="vqUF5wkE29xX"
class Config:
    lr = 0.01
    weight_decay = 5e-4
    based_on = 'item'
    batch_size = 64
    input_dim = train.shape[0] if based_on == 'item' else train.shape[1]
    hidden_dim = 15
    epochs = 30
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Config()
```

```python id="IZ4ZkIx63i6J"
class AutoRec(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AutoRec, self).__init__()
        self.enc = nn.Linear(input_dim, hidden_dim)
        self.dec = nn.Linear(hidden_dim, output_dim)
        self.activate = F.sigmoid

    def forward(self, x):
        x = self.activate(self.enc(x))
        x = self.dec(x)
        return x
```

```python id="3O40WzXX2y4C"
train = np.load('ml_100k_train.npy')
test = np.load('ml_100k_test.npy')
```

```python colab={"base_uri": "https://localhost:8080/"} id="r_UcJzwu4cM7" executionInfo={"status": "ok", "timestamp": 1630609225991, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="708c528b-9701-4922-d563-78a36a1cdc4b"
train.shape, test.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="ibYSEe5S4oZY" executionInfo={"status": "ok", "timestamp": 1630609266057, "user_tz": -330, "elapsed": 440, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="05bd2df5-1c6d-481a-8856-ce70542a5a02"
for x in trainloader:
    print(x)
    break
```

```python colab={"base_uri": "https://localhost:8080/"} id="WRdOimHdzs59" executionInfo={"status": "ok", "timestamp": 1630609230304, "user_tz": -330, "elapsed": 3724, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="16a52976-42dd-4ee1-b34b-9a373994349f"
trainset = AutoRecData(train, config.based_on)
testset = AutoRecData(test, config.based_on)
trainloader = DataLoader(trainset, batch_size=config.batch_size, shuffle=False, drop_last=False)
testloader = DataLoader(testset, batch_size=config.batch_size*100, shuffle=False, drop_last=False)

model = AutoRec(input_dim=config.input_dim, hidden_dim=config.hidden_dim, output_dim=config.input_dim)
model = model.to(config.device)
optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

history = defaultdict(list)
for epoch in range(config.epochs):
    model.train()
    losses = []
    for x in trainloader:
        optimizer.zero_grad()
        x = x.to(config.device)
        mask = x > 0
        pred = model(x)
        loss = torch.mean(((x - pred)[mask])**2)
        loss.backward()
        optimizer.step()
        losses.append(np.sqrt(loss.item()))
    history['tr'].append(np.mean(losses))

    model.eval()
    with torch.no_grad():
        for x in testloader:
            x = x.to(config.device)
            mask = x > 0
            pred = model(x)
            loss = torch.sqrt(torch.mean(((x - pred)[mask])**2))
            losses.append(loss.item())
    history['test'].append(np.mean(losses))
    print(f'EPOCH {epoch+1}: TRAINING loss {history["tr"][-1]} VALID loss {history["test"][-1]}')
```

<!-- #region id="vhB5APuTQNQI" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="jpKEP2FGQSKE" executionInfo={"status": "ok", "timestamp": 1638115654441, "user_tz": -330, "elapsed": 3216, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c36674e7-df1a-4143-f355-9786cdfdf417"
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
