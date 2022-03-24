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

<!-- #region id="o_ne_gC5a-I1" -->
# Attention GNN PyTorch
<!-- #endregion -->

```python id="U4fNVWWSTgWa" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637824784717, "user_tz": -330, "elapsed": 1460, "user": {"displayName": "sparsh agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "00322518567794762549"}} outputId="b66ac9fb-16e8-4397-b584-8714c5898233"
!wget -q --show-progress https://github.com/sparsh-ai/stanza/raw/S969796/datasets/yoochoose1_64/raw/train.txt
```

```python id="oTkBob7DTs2L"
import torch
from torch import nn, optim
from torch.nn import Module, Parameter
import torch.nn.functional as F

from collections import Iterable
from tqdm import tqdm
import datetime
import math
import numpy as np
import pickle
import time
import sys
```

```python id="Uk_7AdvECZxr"
import warnings
warnings.filterwarnings('ignore')
```

```python id="KNFB6Rl293Zc"
def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le)
               for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max
```

```python id="bLRy6o3196VW"
class Dataset():
    def __init__(self, data, shuffle=False, graph=None):
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i):
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)
        for u_input in inputs:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        return alias_inputs, A, items, mask, targets
```

```python id="0qCNCvko-fb4"
train_data = pickle.load(open('train.txt', 'rb'))

train_data = Dataset(train_data, shuffle=True)

n_node = 37484
```

```python id="xMhrwkxC9Z1q"
class Attention_GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(Attention_GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(
            self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(
            self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(
            self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]],
                                self.linear_edge_in(hidden)) + self.b_iah

        input_out = torch.matmul(
            A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah

        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden
```

```python id="OBrHhi2M9Zzu"
class Attention_SessionGraph(Module):
    def __init__(self, n_node):
        super(Attention_SessionGraph, self).__init__()
        self.hidden_size = 4
        self.batch_size = 5
        self.n_node = n_node
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.tagnn = Attention_GNN(self.hidden_size, 1)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
        self.parameters(), lr=0.001, weight_decay=1e-5)

    def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden = self.tagnn(A, hidden)
        hidden = hidden.permute(1, 0, 2)
        return hidden
```

```python id="LISE5EPV9Zxg"
def forward(model, i, data):
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    alias_inputs = torch.Tensor(alias_inputs).long()
    items = torch.Tensor(items).long()
    A = torch.Tensor(A).float()
    mask = torch.Tensor(mask).long()
    hidden = model(items, A)
    return targets, hidden


def train_test(model, train_data):
    print('Start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)

    for i, j in tqdm(zip(slices, np.arange(len(slices))), total=1):
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)
        print('Targets:{}\n\n'.format(targets))
        print('Scores:{}\n\n'.format(scores))
        break;
```

```python id="ZhU6HsqE-tY5"
model = Attention_SessionGraph(n_node)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Q1jeiDjh-ubz" executionInfo={"status": "ok", "timestamp": 1637826058780, "user_tz": -330, "elapsed": 9, "user": {"displayName": "sparsh agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "00322518567794762549"}} outputId="0caa6f33-ca99-4273-b4d1-947cc87adbc7"
model
```

```python id="uCfErpBd9ZrY" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637826059505, "user_tz": -330, "elapsed": 731, "user": {"displayName": "sparsh agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "00322518567794762549"}} outputId="95ca209a-5c1b-42ea-d860-37faf3a9888f"
train_test(model, train_data)
```

```python id="Objv_mXf9ZpV"

```
