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

```python colab={"base_uri": "https://localhost:8080/"} id="qAY1qpJdyBgK" executionInfo={"status": "ok", "timestamp": 1630651401173, "user_tz": -330, "elapsed": 4842, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3a303c9f-772c-45be-e4b1-98aa09288f69"
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

```python id="ANmISYaJyBgN"
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
```

```python id="9ausYME5Z5nA"
def load_data():

    graph = CoraGraphDataset()[0]
    train_mask = ~(graph.ndata['test_mask'] |  graph.ndata['val_mask'])
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']

    feat = graph.ndata['feat']
    label = graph.ndata['label']
    n_nodes = graph.num_nodes()
    edges = graph.edges()
    adj = np.zeros((n_nodes, n_nodes))
    for src, dst in zip(edges[0].numpy(), edges[1].numpy()):
        adj[src, dst] += 1
    
    return train_mask, val_mask, test_mask, feat, label, torch.LongTensor(adj)

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
```

```python id="Jkok3V-DZ1LQ"
class GATLayer(nn.Module):
    def __init__(self, input_dim, out_dim, device):
        super(GATLayer, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        
        self.W = nn.Linear(input_dim, out_dim)
        self.a = nn.Linear(2*self.out_dim, 1)
        
        self.device = device
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a.weight, gain=1.414)
        
    def forward(self, h, adj):
        # h : (batch_size, input_dim)
        # adj : (batch_size, batch_size) => batch에 해당하는 adj matrix
        batch_size = h.size(0)
        wh = self.W(h) # wh : (batch_size, hidden_dim)

        repeat_wh = wh.repeat_interleave(batch_size, dim=0)
        tile_wh = wh.repeat(batch_size, 1)
        
        wh_concat = torch.cat([repeat_wh, tile_wh], dim=1) # whwh : (batch_size*batch_size, 2*hidden_dim)
        wh_concat = F.leaky_relu(self.a(wh_concat), negative_slope=0.2) # awhwh : (batch_size*batch_size, 1)
        wh_concat = wh_concat.view(batch_size, batch_size, -1).squeeze() # awhwh : (batch_size, batch_size, 1)

        small = -9e15 * torch.ones(batch_size, batch_size).to(self.device)
        
        masked_attention = torch.where(adj > 0, wh_concat, small) # masked_attention : (batch_size, batch_size, n_heads)
        attention_weight = F.softmax(masked_attention, dim=1) # attention_weight : (n_heads, batch_size, batch_size)
        
        return torch.mm(attention_weight, wh).squeeze()

class MultiHeadGATLayer(nn.Module):
    '''
    Attention is all you need 에서 한 방식으로 multihead attention 을 구현해봄
    계속 에러가 나는데 원인을 찾지 못했다... ㅜㅜ
    '''
    def __init__(self, input_dim, out_dim, n_heads, device, concat):
        super(MultiHeadGATLayer, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.concat = concat
        self.W = nn.Linear(input_dim, out_dim)
        self.a = nn.Linear(2*self.n_heads, 1)
        
        self.device = device
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a.weight, gain=1.414)
        
    def forward(self, h, adj):
        # h : (batch_size, input_dim)
        # adj : (batch_size, batch_size) => batch에 해당하는 adj matrix
        batch_size = h.size(0)
        wh = self.W(h) # wh : (batch_size, hidden_dim)
        wh_head = wh.view(batch_size, self.n_heads, self.head_dim)

        repeat_wh = wh_head.repeat_interleave(batch_size, dim=0)
        tile_wh = wh_head.repeat(batch_size, 1, 1)
        
        wh_concat = torch.cat([repeat_wh, tile_wh], dim=2) # whwh : (batch_size*batch_size, 2*hidden_dim)
        wh_concat = F.leaky_relu(self.a(wh_concat), negative_slope=0.2) # awhwh : (batch_size*batch_size, 1)
        wh_concat = wh_concat.view(batch_size, batch_size, -1).squeeze() # awhwh : (batch_size, batch_size, 1)

        small = -9e15 * torch.ones_like(wh_concat).to(self.device)
        adj = adj.repeat(self.n_heads, 1, 1).permute(1,2,0)

        masked_attention = torch.where(adj > 0, wh_concat, small) # masked_attention : (batch_size, batch_size, n_heads)
        attention_weight = F.softmax(masked_attention, dim=1).permute(2,0,1) # attention_weight : (n_heads, batch_size, batch_size)
        
        if self.concat:
            return F.elu(torch.bmm(attention_weight, wh_head.permute(1,0,2)).squeeze()).view(-1, self.out_dim)
        else:
            return torch.bmm(attention_weight, wh_head.permute(1,0,2)).squeeze().mean(dim=0)


class GAT(nn.Module):
    def __init__(self, config):
        super(GAT, self).__init__()
        
        self.multihead_attention = [GATLayer(config.input_dim, config.hidden_dim, config.device) for _ in range(config.n_heads)]
        for i, mha in enumerate(self.multihead_attention):
            self.add_module(f'attention_head{i}', mha)
        
        self.outgat = GATLayer(config.n_heads*config.hidden_dim, config.output_dim, config.device)

        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, h, adj):
        h = self.dropout1(h)
        out = torch.cat([F.elu(mha(h, adj)) for mha in self.multihead_attention], axis=1)
        out = self.dropout1(out)
        out = F.elu(self.outgat(out, adj))
        return out
```

```python colab={"base_uri": "https://localhost:8080/"} id="Io0S9f4-aCmo" executionInfo={"status": "ok", "timestamp": 1630651723038, "user_tz": -330, "elapsed": 404, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="673f4fb4-ff0e-44d1-bc59-3ceefc163b0b"
train_mask, val_mask, test_mask, feat, label, adj = load_data()

batch_size = len(train_mask)
input_dim = feat.shape[1]
output_dim = label.unique().shape[0]
```

```python id="ndxAuO4JaAV8"
class Config:
    lr = 0.005
    weight_decay = 5e-4
    hidden_dim = 32
    epochs = 200
    early_stopping_round = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 1995
    n_heads = 8
    dropout = 0.6
    alpha = 0.2
    bs = batch_size
    input_dim = input_dim
    output_dim = output_dim

args = Config()
```

```python id="3fumyMgIZmzd"
seed_everything(args.seed)

model = GAT(args)
model = model.to(args.device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
loss_fn = nn.CrossEntropyLoss()

if 'cuda' in args.device:
    feat = feat.to(args.device)
    label = label.to(args.device)
    adj = adj.to(args.device)
    train_mask = train_mask.to(args.device)
    test_mask = test_mask.to(args.device)
    
train_label = label[train_mask]
test_label = label[test_mask]


history = defaultdict(list)
start = datetime.now()
best_loss, early_step, best_epoch = 0, 0, 0
for epoch in range(args.epochs):
    model.train()
    optimizer.zero_grad()
    output = model(feat, adj)
    acc = torch.sum(train_label == torch.argmax(output[train_mask], axis=1)) / len(train_label)
    loss = loss_fn(output[train_mask], train_label)
    loss.backward()
    optimizer.step()

    history['train_loss'].append(loss.item())
    history['train_acc'].append(acc)

    model.eval()
    with torch.no_grad():    
        output = model(feat, adj)
        acc = torch.sum(test_label == torch.argmax(output[test_mask], axis=1)) / len(test_label)
        loss = loss_fn(output[test_mask], test_label)

    history['valid_loss'].append(loss.item())
    history['valid_acc'].append(acc)

    if epoch == 0 or epoch == args.epochs-1 or (epoch+1)%10 == 0:
        print(f'EPOCH {epoch+1} : TRAINING loss {history["train_loss"][-1]:.3f}, TRAINING ACC {history["train_acc"][-1]:.3f}, VALID loss {history["valid_loss"][-1]:.3f}, VALID ACC {history["valid_acc"][-1]:.3f}')
    
    if history['valid_acc'][-1] > best_loss:
        best_loss = history['valid_acc'][-1]
        best_epoch = epoch

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
