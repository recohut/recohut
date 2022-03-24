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

<!-- #region id="NfkGGOZ0xpLG" -->
# GCN on CORA in PyTorch
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="8S8VFwCjxNZa" executionInfo={"status": "ok", "timestamp": 1638106724565, "user_tz": -330, "elapsed": 6245, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a1a43f7f-b20b-4e2c-9551-8b5c73e68d59"
!pip install -q dgl
```

```python colab={"base_uri": "https://localhost:8080/"} id="W8_VhUsDYzLE" executionInfo={"status": "ok", "timestamp": 1638106767240, "user_tz": -330, "elapsed": 702, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c12b7b04-08d7-4651-e9f8-fa71294e79fa"
import numpy as np
import pandas as pd
import random
import os, sys, pickle
import random, math, gc

from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F

from datetime import datetime
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import dgl
from dgl.data import CoraGraphDataset
```

```python id="VxRBHd8aZYEL"
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
```

```python id="DmZ_eW-vZGHK"
class GCNDataset(Dataset):
    def __init__(self, graph, is_train):
        super(GCNDataset, self).__init__()
        self.graph = graph
        self.mask = graph.ndata['train_mask'] if is_train else graph.ndata['test_mask']
        self.label = graph.ndata['label']
        self.node = graph.nodes()
        self.feat = graph.ndata['feat'].float()

    def __len__(self):
        return self.graph.num_nodes()

    def __getitem__(self, idx):
        return {
            'node': self.node[idx],
            'y': self.label[idx],
            'mask': self.mask[idx],
            'x': self.feat[idx]
        }

def get_A_mat(graph, config):
    A = np.zeros((graph.num_nodes(), graph.num_nodes()))
    for src, dst in zip(graph.edges()[0].numpy(), graph.edges()[1].numpy()):
        A[src, dst] += 1
    A = A + np.identity(graph.num_nodes())
    D = np.sum(A, axis=1)
    D = np.diag(np.power(D, -0.5))
    Ahat = np.dot(D, A).dot(D)
    return torch.tensor(Ahat).float().to(config.device)
```

```python id="gcUJ5zCWZS-W"
class GCNLayer(nn.Module):
    def __init__(self, input, output, dropout):
        super(GCNLayer, self).__init__()
        self.input = input
        self.output = output
        self.W = nn.Linear(input, output)
        self.dropout = nn.Dropout(dropout)
        # torch.nn.init.uniform_(self.W.weight, -1/math.sqrt(output), 1/math.sqrt(output))
        torch.nn.init.uniform_(self.W.weight)        
    
    def forward(self, x, adj):
        output = torch.spmm(adj, x)
        output = self.dropout(output)
        output = self.W(output)
        return output

class GCN(nn.Module):
    def __init__(self, config):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(config.input_dim, config.hidden_dim, dropout=0.1) 
        self.gcn2 = GCNLayer(config.hidden_dim, config.output_dim, dropout=0.1) 
        
    def forward(self, batch_data, A):
        label, data, mask = batch_data['y'], batch_data['x'], batch_data['mask']
        data = F.relu(self.gcn1(data, A))
        data = self.gcn2(data, A)
        return data[mask], label[mask]
```

```python colab={"base_uri": "https://localhost:8080/"} id="rU1R_3VSYxGn" executionInfo={"status": "ok", "timestamp": 1630651560310, "user_tz": -330, "elapsed": 143233, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="387dcde0-a501-4618-c22b-ec56d0403e4e"
class Config:
    learning_rate = 0.01
    weight_decay = 5e-4
    hidden_dim = 16
    epochs = 200
    early_stopping_round = None
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed = 1995

config = Config()

dataset = CoraGraphDataset()
graph = dataset[0]
config.batch_size = graph.num_nodes()
config.input_dim = graph.ndata['feat'].shape[1]
config.output_dim = graph.ndata['label'].unique().shape[0]

seed_everything(config.seed)
train_set = GCNDataset(graph, True)
valid_set = GCNDataset(graph, False)
train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=False)
valid_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False)

A = get_A_mat(graph, config)
model = GCN(config)
model = model.to(config.device)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
loss_fn = nn.CrossEntropyLoss()
history = defaultdict(list)

start = datetime.now()
best_loss, early_step, best_epoch = 0, 0, 0
for epoch in range(config.epochs):
    model.train()
    for batch_data in train_loader:
        optimizer.zero_grad()
        batch_data = {k:v.to(config.device) for k,v in batch_data.items()}
        output, true = model(batch_data, A)
        acc_tr = torch.sum(true == torch.argmax(output, axis=1)) / len(true)
        loss = loss_fn(output, true)
        loss.backward()
        optimizer.step()

    history['train_loss'].append(loss.item())
    history['train_acc'].append(acc_tr)

    model.eval()
    with torch.no_grad():
        for batch_data in valid_loader:
            batch_data = {k:v.to(config.device) for k,v in batch_data.items()}
            output, true = model(batch_data, A)
            acc = torch.sum(true == torch.argmax(output, axis=1)) / len(true)
            loss = loss_fn(output, true)

    history['valid_loss'].append(loss.item())
    history['valid_acc'].append(acc)

    if epoch == 0 or epoch == config.epochs-1 or (epoch+1)%10 == 0:
        print(f'EPOCH {epoch+1} : TRAINING loss {history["train_loss"][-1]:.3f}, TRAINING ACC {history["train_acc"][-1]:.3f}, VALID loss {history["valid_loss"][-1]:.3f}, VALID ACC {history["valid_acc"][-1]:.3f}')
    
    if history['valid_acc'][-1] > best_loss:
        best_loss = history['valid_acc'][-1]
        best_epoch = epoch

    elif(config.early_stopping_round is not None):
        
        early_step += 1
        if (early_step >= config.early_stopping_round):
            break
end = datetime.now()
print(end-start)
print(f'At EPOCH {best_epoch + 1}, We have Best Acc {best_loss}')
```

<!-- #region id="EsFzSRvnxb3X" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="QKXM9U7sxb3Z" executionInfo={"status": "ok", "timestamp": 1638106792273, "user_tz": -330, "elapsed": 3002, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="729e3d64-42f2-4de7-9685-304a7dfbd0dc"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="Bn4s9s_Axb3Z" -->
---
<!-- #endregion -->

<!-- #region id="jFXf3bKdxb3a" -->
**END**
<!-- #endregion -->
