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

<!-- #region id="hHf7UeaK9uDD" -->
## Theory
<!-- #endregion -->

<!-- #region id="P1jFw4KwJd1q" -->
### PyTorch Geometric

We had mentioned before that implementing graph networks with adjacency matrix is simple and straight-forward but can be computationally expensive for large graphs. Many real-world graphs can reach over 200k nodes, for which adjacency matrix-based implementations fail. There are a lot of optimizations possible when implementing GNNs, and luckily, there exist packages that provide such layers. The most popular packages for PyTorch are [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) and the [Deep Graph Library](https://www.dgl.ai/) (the latter being actually framework agnostic). Which one to use depends on the project you are planning to do and personal taste. In this tutorial, we will look at PyTorch Geometric as part of the PyTorch family.
<!-- #endregion -->

<!-- #region id="WKGs-CAqoIt1" -->
A graph is used to model pairwise relations (edges) between objects (nodes). A single graph in PyG is described by an instance of ```torch_geometric.data.Data```, which holds the following attributes by default:

- ```data.x```: Node feature matrix with shape [num_nodes, num_node_features]

- ```data.edge_index```: Graph connectivity in COO format with shape [2, num_edges] and type torch.long

- ```data.edge_attr```: Edge feature matrix with shape [num_edges, num_edge_features]

- ```data.y```: Target to train against (may have arbitrary shape), e.g., node-level targets of shape [num_nodes, *] or graph-level targets of shape [1, *]

- ```data.pos```: Node position matrix with shape [num_nodes, num_dimensions]

None of these attributes are required. In fact, the Data object is not even restricted to these attributes. We can, e.g., extend it by data.face to save the connectivity of triangles from a 3D mesh in a tensor with shape [3, num_faces] and type torch.long.
<!-- #endregion -->

<!-- #region id="uehWg91U0UAm" -->
## Setup
<!-- #endregion -->

<!-- #region id="GvUA0__E-gt0" -->
### Version Control
<!-- #endregion -->

```python id="0nexADPxY4e8"
import os
project_name = "recobase"; branch = "US739178"; account = "recohut"
project_path = os.path.join('/content', branch)

if not os.path.exists(project_path):
    !pip install -U -q dvc dvc[gdrive]
    !cp -r /content/drive/MyDrive/git_credentials/. ~
    !mkdir "{project_path}"
    %cd "{project_path}"
    !git init
    !git remote add origin https://github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout -b "{branch}"
    %reload_ext autoreload
    %autoreload 2
else:
    %cd "{project_path}"
```

```python id="wDQ4KSqraU-_"
!git status -u
```

```python id="p4GIMVp1Y4e9"
!git add .
!git commit -m 'commit'
!git push origin "{branch}"
```

<!-- #region id="-hunnaOC0Vrl" -->
### Installation
<!-- #endregion -->

```python id="_EbEqI1s0dKH"
!pip install -q torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
!pip install -q torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
!pip install -q torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
!pip install -q torch-geometric
```

<!-- #region id="jkGBD-eGKBfO" -->
While the theory and math behind GNNs might first seem complicated, the implementation of those models is quite simple and helps in understanding the methodology. Therefore, we will discuss the implementation of basic network layers of a GNN, namely graph convolutions, and attention layers. Finally, we will apply a GNN on a node-level, edge-level, and graph-level tasks.
<!-- #endregion -->

<!-- #region id="7Hx9nLYyZOnt" -->
## Data Ingestion
<!-- #endregion -->

<!-- #region id="aCb0C5OSZx5p" -->
### CORA
<!-- #endregion -->

```python id="yoIoWlOAZX4p"
!mkdir /content/x && git clone https://github.com/AntonioLonga/PytorchGeometricTutorial.git /content/x
```

```python id="tcYBE1-wZe3T"
!mkdir -p data/bronze/cora
!mkdir -p data/silver/cora

!cp -r /content/x/Tutorial1/tutorial1/Cora/raw/* data/bronze/cora
!cp -r /content/x/Tutorial1/tutorial1/Cora/processed/* data/silver/cora
```

```python id="tzC2eA7kZzx2"
!dvc add data/bronze/cora/*
!dvc add data/silver/cora/*
```

```python id="c5iLLY_-amHG"
!dvc commit data/bronze/cora/*
!dvc push data/bronze/cora/*

!dvc commit data/silver/cora/*
!dvc push data/silver/cora/*
```

<!-- #region id="_uEuTcIAPND-" -->
## Computation Graph
<!-- #endregion -->

<!-- #region id="JavXabtuP9TN" -->
The neighbors of a node define its computation graph
<!-- #endregion -->

<!-- #region id="H5QZWhOHQV-L" -->
Every node has its own computation graph
<!-- #endregion -->

<!-- #region id="ddpYUOPqQV2H" -->
## Information Aggregation Function
<!-- #endregion -->

<!-- #region id="h5IGoI0mSUdK" -->
A slightly general representation of this function is to replace the summation with aggregation. This will give us flexibility to aggregate the neighborhood weights.
<!-- #endregion -->

<!-- #region id="pFlFkvOlO7d6" -->
### Shared Parameters
<!-- #endregion -->

<!-- #region id="m1jx-rSsR6rT" -->
> Note: When a new node come, it will get pre-trained weights in this way.
<!-- #endregion -->

<!-- #region id="8P3I5aWSRuPN" -->
### Aggregation Functions
<!-- #endregion -->

<!-- #region id="fBaONxHzS-um" -->
### Code Practice
<!-- #endregion -->

<!-- #region id="P9l2KngKgWks" -->
#### Prototype
<!-- #endregion -->

```python id="7Ao3nX6DWZY1"
import torch_geometric
from torch_geometric.datasets import Planetoid
```

```python id="LPpv44qsaqb3"
use_cuda_if_available = False
```

```python id="yeTUJmArasRM"
dataset = Planetoid(root="/content/cora", name= "Cora")
```

```python colab={"base_uri": "https://localhost:8080/"} id="21xVO3ula0UI" executionInfo={"status": "ok", "timestamp": 1631524231285, "user_tz": -330, "elapsed": 512, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2332afca-4b92-4163-cc22-2aefab151c07"
print(dataset)
print("number of graphs:\t\t",len(dataset))
print("number of classes:\t\t",dataset.num_classes)
print("number of node features:\t",dataset.num_node_features)
print("number of edge features:\t",dataset.num_edge_features)
```

```python colab={"base_uri": "https://localhost:8080/"} id="1ISVVYE5bEug" executionInfo={"status": "ok", "timestamp": 1631524247316, "user_tz": -330, "elapsed": 422, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="30bfee48-a8db-4416-f062-5dcd75dca485"
print(dataset.data)
```

```python colab={"base_uri": "https://localhost:8080/"} id="0_qeav8sbEo4" executionInfo={"status": "ok", "timestamp": 1631524289177, "user_tz": -330, "elapsed": 1840, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0268022c-df33-417f-d6e8-ac6dcd3e8cd4"
print("edge_index:\t\t",dataset.data.edge_index.shape)
print(dataset.data.edge_index)
print("\n")
print("train_mask:\t\t",dataset.data.train_mask.shape)
print(dataset.data.train_mask)
print("\n")
print("x:\t\t",dataset.data.x.shape)
print(dataset.data.x)
print("\n")
print("y:\t\t",dataset.data.y.shape)
print(dataset.data.y)
```

```python id="h_DPSL4zbOqF"
import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
```

```python id="4lRviHFBbXIo"
data = dataset[0]
```

```python id="AIm93HpdbcTF"
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv = SAGEConv(dataset.num_features,
                             dataset.num_classes,
                             aggr="max") # max, mean, add ...)

    def forward(self):
        x = self.conv(data.x, data.edge_index)
        return F.log_softmax(x, dim=1)
```

```python id="KZfAEl7ubcO1"
device = torch.device('cuda' if torch.cuda.is_available() and use_cuda_if_available else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Bq6K-QAvbcH-" executionInfo={"status": "ok", "timestamp": 1631524496613, "user_tz": -330, "elapsed": 515, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="75dac9a6-4e1c-4bfa-bb75-debf5d9b6b1e"
device
```

```python id="eXwCdpHycB-h"
def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs
```

```python colab={"base_uri": "https://localhost:8080/"} id="ISVIQ-qFcyoa" executionInfo={"status": "ok", "timestamp": 1631524785625, "user_tz": -330, "elapsed": 40182, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="14678a64-ce4e-4d86-fbfc-937af3c013d7"
best_val_acc = test_acc = 0

for epoch in range(1,100):
    train()
    _, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Val: {:.4f}, Test: {:.4f}'
    
    if epoch % 10 == 0:
        print(log.format(epoch, best_val_acc, test_acc))
```

<!-- #region id="Wwi-MOgkgS7U" -->
#### Scripting
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="8MtmyZrVfTvA" executionInfo={"status": "ok", "timestamp": 1631525453623, "user_tz": -330, "elapsed": 1072, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8398a4bb-2b4b-417c-fbe1-07b2ce90a1f3"
%%writefile src/datasets/vectorial.py
import torch.nn as nn
import torch


#%% Dataset to manage vector to vector data
class VectorialDataset(torch.utils.data.Dataset):
    def __init__(self, input_data, output_data):
        super(VectorialDataset, self).__init__()
        self.input_data = torch.tensor(input_data.astype('f'))
        self.output_data = torch.tensor(output_data.astype('f'))
        
    def __len__(self):
        return self.input_data.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = (self.input_data[idx, :], 
                  self.output_data[idx, :])  
        return sample 
```

```python colab={"base_uri": "https://localhost:8080/"} id="P94yQLIDgA_S" executionInfo={"status": "ok", "timestamp": 1631525593073, "user_tz": -330, "elapsed": 1603, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="973f21cd-d82e-4c7d-e84d-6434549e11f1"
%%writefile src/datasets/__init__.py
from .vectorial import VectorialDataset
```

```python colab={"base_uri": "https://localhost:8080/"} id="58I29mSRc-yv" executionInfo={"status": "ok", "timestamp": 1631525536407, "user_tz": -330, "elapsed": 1300, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6929a0d0-1612-46bd-e7c1-e4f56b16978a"
%%writefile src/models/linear.py
import torch.nn as nn
import torch

#%% Linear layer
class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear = nn.Linear(self.input_dim, self.output_dim, bias=True)

    def forward(self, x):
        out = self.linear(x)
        return out
    
    def reset(self):
        self.linear.reset_parameters()
```

```python colab={"base_uri": "https://localhost:8080/"} id="uWGKn9Tjf9kW" executionInfo={"status": "ok", "timestamp": 1631525608126, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="78e1bac4-3209-43a3-e453-8fafef38d9dc"
%%writefile src/models/__init__.py
from .linear import LinearModel
```

<!-- #region id="BrQyCrq-nFYA" -->
## Graph attention networks (GAT)
<!-- #endregion -->

<!-- #region id="986OYYSwqjCt" -->
### Overview
<!-- #endregion -->

<!-- #region id="i3P3cuVJqWpF" -->
### Code Practice
<!-- #endregion -->

```python id="fIxCZoQBsXGf"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
```

<!-- #region id="escsS-iusy5t" -->
#### Structure
<!-- #endregion -->

```python id="CdxIh9-msXGh"
class GATLayer(nn.Module):
    """
    Simple PyTorch Implementation of the Graph Attention layer.
    """
    def __init__(self):
        super(GATLayer, self).__init__()
      
    def forward(self, input, adj):
        print("")
```

<!-- #region id="9pHUmxNzsXGi" -->
Let's start from the forward method
<!-- #endregion -->

<!-- #region id="4V3WzW7vsXGj" -->
#### Linear Transformation

$$
\bar{h'}_i = \textbf{W}\cdot \bar{h}_i
$$
with $\textbf{W}\in\mathbb R^{F'\times F}$ and $\bar{h}_i\in\mathbb R^{F}$.

$$
\bar{h'}_i \in \mathbb{R}^{F'}
$$
<!-- #endregion -->

```python id="vHxrLNcvsXGk" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631528877454, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b2d06c43-421b-40bb-a000-c2ea454d4cf5"
in_features = 5
out_features = 2
nb_nodes = 3

W = nn.Parameter(torch.zeros(size=(in_features, out_features))) #xavier paramiter inizializator
nn.init.xavier_uniform_(W.data, gain=1.414)

input = torch.rand(nb_nodes,in_features) 


# linear transformation
h = torch.mm(input, W)
N = h.size()[0]

print(h.shape)
```

<!-- #region id="BmEJAQYrsXGn" -->
#### Attention Mechanism
<!-- #endregion -->

```python id="mM952UIQsXGp" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631528997529, "user_tz": -330, "elapsed": 752, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3ea2b14c-9395-4471-b595-0cae2e27136a"
a = nn.Parameter(torch.zeros(size=(2*out_features, 1))) #xavier parameter inizializator
nn.init.xavier_uniform_(a.data, gain=1.414)
print(a.shape)

leakyrelu = nn.LeakyReLU(0.2)  # LeakyReLU
```

```python id="kgco_qXqsXGq"
a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * out_features)
```

<!-- #region id="WD0v0CCHtMqr" -->
<!-- #endregion -->

```python id="yfjy06mDsXGr"
e = leakyrelu(torch.matmul(a_input, a).squeeze(2))
```

```python id="xd5Tgf7MsXGs" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631529040574, "user_tz": -330, "elapsed": 127, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="091b6e77-71ea-4dfd-d386-94ad172824ba"
print(a_input.shape,a.shape)
print("")
print(torch.matmul(a_input,a).shape)
print("")
print(torch.matmul(a_input,a).squeeze(2).shape)
```

<!-- #region id="2JEveCgDsXGs" -->
#### Masked Attention
<!-- #endregion -->

```python id="90nbf6gZsXGs" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631529088098, "user_tz": -330, "elapsed": 448, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="cf6d4722-d255-45d9-cc5a-468983a7195c"
# Masked Attention
adj = torch.randint(2, (3, 3))

zero_vec  = -9e15*torch.ones_like(e)
print(zero_vec.shape)
```

```python id="MC7wZsDCsXGt" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631529094634, "user_tz": -330, "elapsed": 1930, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="13f26258-7e3f-4962-be17-7eac198bb17f"
attention = torch.where(adj > 0, e, zero_vec)
print(adj,"\n",e,"\n",zero_vec)
attention
```

```python id="TmbtuV2hsXGu"
attention = F.softmax(attention, dim=1)
h_prime   = torch.matmul(attention, h)
```

```python id="_kqS_t-esXGv" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631529104801, "user_tz": -330, "elapsed": 33, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="13f0e4c4-cdfe-4101-b471-d703f5592cab"
attention
```

```python id="H7fwMik-sXGw" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631529112734, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4c84027d-f8a0-4f50-c80a-456011b7ce3c"
h_prime
```

<!-- #region id="xp49RxTQsXGx" -->
h_prime vs h
<!-- #endregion -->

```python id="dwu9p5tQsXGy" outputId="8d60105e-7f92-44a0-c105-a0f47492382c"
33print(h_prime,"\n",h)
```

<!-- #region id="MELcnTAlwvfl" -->
### Loading the dataset
<!-- #endregion -->

```python id="T_l8_z4GsXG1"
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

import matplotlib.pyplot as plt

name_data = 'Cora'
dataset = Planetoid(root= '/content/' + name_data, name = name_data)
dataset.transform = T.NormalizeFeatures()

print(f"Number of Classes in {name_data}:", dataset.num_classes)
print(f"Number of Node Features in {name_data}:", dataset.num_node_features)
```

<!-- #region id="Vd67RsopsXGy" -->
### Assembling the components
<!-- #endregion -->

```python id="LGQg18xYsXGz"
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.dropout       = dropout        # drop prob = 0.6
        self.in_features   = in_features    # 
        self.out_features  = out_features   # 
        self.alpha         = alpha          # LeakyReLU with negative input slope, alpha = 0.2
        self.concat        = concat         # conacat = True for all layers except the output layer.

        
        # Xavier Initialization of Weights
        # Alternatively use weights_init to apply weights of choice 
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        # LeakyReLU
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        # Linear Transformation
        h = torch.mm(input, self.W) # matrix multiplication
        N = h.size()[0]
        print(N)

        # Attention Mechanism
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e       = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        # Masked Attention
        zero_vec  = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime   = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
```

```python id="HjTPUBsfw_e4"
class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.hid = 8
        self.in_head = 8
        self.out_head = 1
        
        
        self.conv1 = GATConv(dataset.num_features, self.hid, heads=self.in_head, dropout=0.6)
        self.conv2 = GATConv(self.hid*self.in_head, dataset.num_classes, concat=False,
                             heads=self.out_head, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
                
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)
```

<!-- #region id="lx7QPj9SsXG0" -->
### Use it
<!-- #endregion -->

```python id="LD79VrMosXG3" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631530058182, "user_tz": -330, "elapsed": 73148, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d7467f8c-3a1a-4efa-ff77-7144c78bc7ec"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"

model = GAT().to(device)
data = dataset[0].to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

model.train()
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    
    if epoch%200 == 0:
        print(loss)
    
    loss.backward()
    optimizer.step()
```

```python id="bW0OMhKqsXG4" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631530130604, "user_tz": -330, "elapsed": 475, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a567917b-c1e1-43b1-d80f-215870be3bac"
model.eval()
_, pred = model(data).max(dim=1)
correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / data.test_mask.sum().item()
print('Accuracy: {:.4f}'.format(acc))
```

```python id="Hf1iQvHYsXG5"

```

<!-- #region id="e6J6bpLXLqlD" -->
## Graph representation

Before starting the discussion of specific neural network operations on graphs, we should consider how to represent a graph. Mathematically, a graph $\mathcal{G}$ is defined as a tuple of a set of nodes/vertices $V$, and a set of edges/links $E$: $\mathcal{G}=(V,E)$. Each edge is a pair of two vertices, and represents a connection between them. For instance, let's look at the following graph:
<!-- #endregion -->

<!-- #region id="PhodeYd0LSes" -->
<center width="100%" style="padding:10px"><img src="data:image/png;base64,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" width="250px"></center>
<!-- #endregion -->

<!-- #region id="KZHGr561Jd07" -->
The vertices are $V=\{1,2,3,4\}$, and edges $E=\{(1,2), (2,3), (2,4), (3,4)\}$. Note that for simplicity, we assume the graph to be undirected and hence don't add mirrored pairs like $(2,1)$. In application, vertices and edge can often have specific attributes, and edges can even be directed. The question is how we could represent this diversity in an efficient way for matrix operations. Usually, for the edges, we decide between two variants: an adjacency matrix, or a list of paired vertex indices. 

The **adjacency matrix** $A$ is a square matrix whose elements indicate whether pairs of vertices are adjacent, i.e. connected, or not. In the simplest case, $A_{ij}$ is 1 if there is a connection from node $i$ to $j$, and otherwise 0. If we have edge attributes or different categories of edges in a graph, this information can be added to the matrix as well. For an undirected graph, keep in mind that $A$ is a symmetric matrix ($A_{ij}=A_{ji}$). For the example graph above, we have the following adjacency matrix:

$$
A = \begin{bmatrix}
    0 & 1 & 0 & 0\\
    1 & 0 & 1 & 1\\
    0 & 1 & 0 & 1\\
    0 & 1 & 1 & 0
\end{bmatrix}
$$

While expressing a graph as a list of edges is more efficient in terms of memory and (possibly) computation, using an adjacency matrix is more intuitive and simpler to implement. In our implementations below, we will rely on the adjacency matrix to keep the code simple. However, common libraries use edge lists, which we will discuss later more.
Alternatively, we could also use the list of edges to define a sparse adjacency matrix with which we can work as if it was a dense matrix, but allows more memory-efficient operations. PyTorch supports this with the sub-package `torch.sparse` ([documentation](https://pytorch.org/docs/stable/sparse.html)) which is however still in a beta-stage (API might change in future).
<!-- #endregion -->

<!-- #region id="jZQwHaxZM9n1" -->
## Graph Convolutions

Graph Convolutional Networks have been introduced by [Kipf et al.](https://openreview.net/pdf?id=SJU4ayYgl) in 2016 at the University of Amsterdam. He also wrote a great [blog post](https://tkipf.github.io/graph-convolutional-networks/) about this topic, which is recommended if you want to read about GCNs from a different perspective. GCNs are similar to convolutions in images in the sense that the "filter" parameters are typically shared over all locations in the graph. At the same time, GCNs rely on message passing methods, which means that vertices exchange information with the neighbors, and send "messages" to each other. Before looking at the math, we can try to visually understand how GCNs work. The first step is that each node creates a feature vector that represents the message it wants to send to all its neighbors. In the second step, the messages are sent to the neighbors, so that a node receives one message per adjacent node. Below we have visualized the two steps for our example graph. 
<!-- #endregion -->

<!-- #region id="RoAqgLUINDQ5" -->
<center width="80%"><img src="data:image/png;base64,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" width="700px"></center>
<!-- #endregion -->

<!-- #region id="v3LT905AJd1J" -->
If we want to formulate that in more mathematical terms, we need to first decide how to combine all the messages a node receives. As the number of messages vary across nodes, we need an operation that works for any number. Hence, the usual way to go is to sum or take the mean. Given the previous features of nodes $H^{(l)}$, the GCN layer is defined as follows:

$$H^{(l+1)} = \sigma\left(\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}H^{(l)}W^{(l)}\right)$$

$W^{(l)}$ is the weight parameters with which we transform the input features into messages ($H^{(l)}W^{(l)}$). To the adjacency matrix $A$ we add the identity matrix so that each node sends its own message also to itself: $\hat{A}=A+I$. Finally, to take the average instead of summing, we calculate the matrix $\hat{D}$ which is a diagonal matrix with $D_{ii}$ denoting the number of neighbors node $i$ has. $\sigma$ represents an arbitrary activation function, and not necessarily the sigmoid (usually a ReLU-based activation function is used in GNNs). 

When implementing the GCN layer in PyTorch, we can take advantage of the flexible operations on tensors. Instead of defining a matrix $\hat{D}$, we can simply divide the summed messages by the number of neighbors afterward. Additionally, we replace the weight matrix with a linear layer, which additionally allows us to add a bias. Written as a PyTorch module, the GCN layer is defined as follows:
<!-- #endregion -->

```python id="QGoxbcxb-FTO"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
```

```python id="pXhfaEuGKNP2"
class GCNLayer(nn.Module):
    
    def __init__(self, c_in, c_out):
        super().__init__()
        self.projection = nn.Linear(c_in, c_out)

    def forward(self, node_feats, adj_matrix):
        """
        Inputs:
            node_feats - Tensor with node features of shape [batch_size, num_nodes, c_in]
            adj_matrix - Batch of adjacency matrices of the graph. If there is an edge from i to j, adj_matrix[b,i,j]=1 else 0.
                         Supports directed edges by non-symmetric matrices. Assumes to already have added the identity connections. 
                         Shape: [batch_size, num_nodes, num_nodes]
        """
        # Num neighbours = number of incoming edges
        num_neighbours = adj_matrix.sum(dim=-1, keepdims=True)
        node_feats = self.projection(node_feats)
        node_feats = torch.bmm(adj_matrix, node_feats)
        node_feats = node_feats / num_neighbours
        return node_feats
```

<!-- #region id="bhJEKQIjJd1R" -->
To further understand the GCN layer, we can apply it to our example graph above. First, let's specify some node features and the adjacency matrix with added self-connections:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="SRny94_XOgou" executionInfo={"status": "ok", "timestamp": 1631336482352, "user_tz": -330, "elapsed": 400, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="11bb492e-b59f-4728-c24d-b3972cd87b06"
node_feats = torch.arange(8, dtype=torch.float32).view(1, 4, 2)
adj_matrix = torch.Tensor([[[1, 1, 0, 0],
                            [1, 1, 1, 1],
                            [0, 1, 1, 1],
                            [0, 1, 1, 1]]])

print("Node features:\n", node_feats)
print("\nAdjacency matrix:\n", adj_matrix)
```

<!-- #region id="0p7_bfgPJd1X" -->
Next, let's apply a GCN layer to it. For simplicity, we initialize the linear weight matrix as an identity matrix so that the input features are equal to the messages. This makes it easier for us to verify the message passing operation.
<!-- #endregion -->

```python id="rKvNBJ6zJd1Z" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631336586980, "user_tz": -330, "elapsed": 529, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ecfc0711-5090-42df-9095-a54eef34299d"
layer = GCNLayer(c_in=2, c_out=2)
layer.projection.weight.data = torch.Tensor([[1., 0.], [0., 1.]])
layer.projection.bias.data = torch.Tensor([0., 0.])

with torch.no_grad():
    out_feats = layer(node_feats, adj_matrix)

print("Adjacency matrix", adj_matrix)
print("Input features", node_feats)
print("Output features", out_feats)
```

<!-- #region id="4-iXNxPTPScd" -->
Next, let's apply a GCN layer to it. For simplicity, we initialize the linear weight matrix as an identity matrix so that the input features are equal to the messages. This makes it easier for us to verify the message passing operation.
<!-- #endregion -->

```python id="EUGQ3PjhPSce" outputId="f9f400b9-9adb-4c00-bcf5-e6d40ced69fc"
layer = GCNLayer(c_in=2, c_out=2)
layer.projection.weight.data = torch.Tensor([[1., 0.], [0., 1.]])
layer.projection.bias.data = torch.Tensor([0., 0.])

with torch.no_grad():
    out_feats = layer(node_feats, adj_matrix)

print("Adjacency matrix", adj_matrix)
print("Input features", node_feats)
print("Output features", out_feats)
```

<!-- #region id="L_JuW8BgPULc" -->
As we can see, the first node's output values are the average of itself and the second node. Similarly, we can verify all other nodes. However, in a GNN, we would also want to allow feature exchange between nodes beyond its neighbors. This can be achieved by applying multiple GCN layers, which gives us the final layout of a GNN. The GNN can be build up by a sequence of GCN layers and non-linearities such as ReLU. For a visualization, see below (figure credit - [Thomas Kipf, 2016](https://tkipf.github.io/graph-convolutional-networks/)).
<!-- #endregion -->

<!-- #region id="i_nHrF0YJd1f" -->
However, one issue we can see from looking at the example above is that the output features for nodes 3 and 4 are the same because they have the same adjacent nodes (including itself). Therefore, GCN layers can make the network forget node-specific information if we just take a mean over all messages. Multiple possible improvements have been proposed. While the simplest option might be using residual connections, the more common approach is to either weigh the self-connections higher or define a separate weight matrix for the self-connections. Alternatively, we can re-visit a familiar concept: attention. 
<!-- #endregion -->

<!-- #region id="G2RHXjTISmo8" -->
## Graph Attention 

If you remember from the last tutorial, attention describes a weighted average of multiple elements with the weights dynamically computed based on an input query and elements' keys (if you haven't read Tutorial 6 yet, it is recommended to at least go through the very first section called [What is Attention?](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html#What-is-Attention?)). This concept can be similarly applied to graphs, one of such is the Graph Attention Network (called GAT, proposed by [Velickovic et al., 2017](https://arxiv.org/abs/1710.10903)). Similarly to the GCN, the graph attention layer creates a message for each node using a linear layer/weight matrix. For the attention part, it uses the message from the node itself as a query, and the messages to average as both keys and values (note that this also includes the message to itself). The score function $f_{attn}$ is implemented as a one-layer MLP which maps the query and key to a single value. The MLP looks as follows (figure credit - [Velickovic et al.](https://arxiv.org/abs/1710.10903)):
<!-- #endregion -->

<!-- #region id="kvCeqy8OSpHS" -->
<center width="100%" style="padding:10px"><img src="data:image/png;base64,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" width="250px"></center>
<!-- #endregion -->

<!-- #region id="neiG2SqzJd1h" -->
$h_i$ and $h_j$ are the original features from node $i$ and $j$ respectively, and represent the messages of the layer with $\mathbf{W}$ as weight matrix. $\mathbf{a}$ is the weight matrix of the MLP, which has the shape $[1,2\times d_{\text{message}}]$, and $\alpha_{ij}$ the final attention weight from node $i$ to $j$. The calculation can be described as follows:

$$\alpha_{ij} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}\left[\mathbf{W}h_i||\mathbf{W}h_j\right]\right)\right)}{\sum_{k\in\mathcal{N}_i} \exp\left(\text{LeakyReLU}\left(\mathbf{a}\left[\mathbf{W}h_i||\mathbf{W}h_k\right]\right)\right)}$$

The operator $||$ represents the concatenation, and $\mathcal{N}_i$ the indices of the neighbors of node $i$. Note that in contrast to usual practice, we apply a non-linearity (here LeakyReLU) before the softmax over elements. Although it seems like a minor change at first, it is crucial for the attention to depend on the original input. Specifically, let's remove the non-linearity for a second, and try to simplify the expression:

$$
\begin{split}
    \alpha_{ij} & = \frac{\exp\left(\mathbf{a}\left[\mathbf{W}h_i||\mathbf{W}h_j\right]\right)}{\sum_{k\in\mathcal{N}_i} \exp\left(\mathbf{a}\left[\mathbf{W}h_i||\mathbf{W}h_k\right]\right)}\\[5pt]
    & = \frac{\exp\left(\mathbf{a}_{:,:d/2}\mathbf{W}h_i+\mathbf{a}_{:,d/2:}\mathbf{W}h_j\right)}{\sum_{k\in\mathcal{N}_i} \exp\left(\mathbf{a}_{:,:d/2}\mathbf{W}h_i+\mathbf{a}_{:,d/2:}\mathbf{W}h_k\right)}\\[5pt]
    & = \frac{\exp\left(\mathbf{a}_{:,:d/2}\mathbf{W}h_i\right)\cdot\exp\left(\mathbf{a}_{:,d/2:}\mathbf{W}h_j\right)}{\sum_{k\in\mathcal{N}_i} \exp\left(\mathbf{a}_{:,:d/2}\mathbf{W}h_i\right)\cdot\exp\left(\mathbf{a}_{:,d/2:}\mathbf{W}h_k\right)}\\[5pt]
    & = \frac{\exp\left(\mathbf{a}_{:,d/2:}\mathbf{W}h_j\right)}{\sum_{k\in\mathcal{N}_i} \exp\left(\mathbf{a}_{:,d/2:}\mathbf{W}h_k\right)}\\
\end{split}
$$

We can see that without the non-linearity, the attention term with $h_i$ actually cancels itself out, resulting in the attention being independent of the node itself. Hence, we would have the same issue as the GCN of creating the same output features for nodes with the same neighbors. This is why the LeakyReLU is crucial and adds some dependency on $h_i$ to the attention. 

Once we obtain all attention factors, we can calculate the output features for each node by performing the weighted average:

$$h_i'=\sigma\left(\sum_{j\in\mathcal{N}_i}\alpha_{ij}\mathbf{W}h_j\right)$$

$\sigma$ is yet another non-linearity, as in the GCN layer. Visually, we can represent the full message passing in an attention layer as follows (figure credit - [Velickovic et al.](https://arxiv.org/abs/1710.10903)):
<!-- #endregion -->

<!-- #region id="2jwDUpz9StK8" -->
To increase the expressiveness of the graph attention network, [Velickovic et al.](https://arxiv.org/abs/1710.10903) proposed to extend it to multiple heads similar to the Multi-Head Attention block in Transformers. This results in $N$ attention layers being applied in parallel. In the image above, it is visualized as three different colors of arrows (green, blue, and purple) that are afterward concatenated. The average is only applied for the very final prediction layer in a network. 

After having discussed the graph attention layer in detail, we can implement it below:
<!-- #endregion -->

```python id="o_169MUj-KOr"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
```

```python id="8GKQh008O0HT"
class GATLayer(nn.Module):
    
    def __init__(self, c_in, c_out, num_heads=1, concat_heads=True, alpha=0.2):
        """
        Inputs:
            c_in - Dimensionality of input features
            c_out - Dimensionality of output features
            num_heads - Number of heads, i.e. attention mechanisms to apply in parallel. The 
                        output features are equally split up over the heads if concat_heads=True.
            concat_heads - If True, the output of the different heads is concatenated instead of averaged.
            alpha - Negative slope of the LeakyReLU activation.
        """
        super().__init__()
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        if self.concat_heads:
            assert c_out % num_heads == 0, "Number of output features must be a multiple of the count of heads."
            c_out = c_out // num_heads
        
        # Sub-modules and parameters needed in the layer
        self.projection = nn.Linear(c_in, c_out * num_heads)
        self.a = nn.Parameter(torch.Tensor(num_heads, 2 * c_out)) # One per head
        self.leakyrelu = nn.LeakyReLU(alpha)
        
        # Initialization from the original implementation
        nn.init.xavier_uniform_(self.projection.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
    def forward(self, node_feats, adj_matrix, print_attn_probs=False):
        """
        Inputs:
            node_feats - Input features of the node. Shape: [batch_size, c_in]
            adj_matrix - Adjacency matrix including self-connections. Shape: [batch_size, num_nodes, num_nodes]
            print_attn_probs - If True, the attention weights are printed during the forward pass (for debugging purposes)
        """
        batch_size, num_nodes = node_feats.size(0), node_feats.size(1)
        
        # Apply linear layer and sort nodes by head
        node_feats = self.projection(node_feats)
        node_feats = node_feats.view(batch_size, num_nodes, self.num_heads, -1)
        
        # We need to calculate the attention logits for every edge in the adjacency matrix 
        # Doing this on all possible combinations of nodes is very expensive
        # => Create a tensor of [W*h_i||W*h_j] with i and j being the indices of all edges
        edges = adj_matrix.nonzero(as_tuple=False) # Returns indices where the adjacency matrix is not 0 => edges
        node_feats_flat = node_feats.view(batch_size * num_nodes, self.num_heads, -1)
        edge_indices_row = edges[:,0] * num_nodes + edges[:,1]
        edge_indices_col = edges[:,0] * num_nodes + edges[:,2]
        a_input = torch.cat([
            torch.index_select(input=node_feats_flat, index=edge_indices_row, dim=0),
            torch.index_select(input=node_feats_flat, index=edge_indices_col, dim=0)
        ], dim=-1) # Index select returns a tensor with node_feats_flat being indexed at the desired positions along dim=0
        
        # Calculate attention MLP output (independent for each head)
        attn_logits = torch.einsum('bhc,hc->bh', a_input, self.a) 
        attn_logits = self.leakyrelu(attn_logits)
        
        # Map list of attention values back into a matrix
        attn_matrix = attn_logits.new_zeros(adj_matrix.shape+(self.num_heads,)).fill_(-9e15)
        attn_matrix[adj_matrix[...,None].repeat(1,1,1,self.num_heads) == 1] = attn_logits.reshape(-1)
        
        # Weighted average of attention
        attn_probs = F.softmax(attn_matrix, dim=2)
        if print_attn_probs:
            print("Attention probs\n", attn_probs.permute(0, 3, 1, 2))
        node_feats = torch.einsum('bijh,bjhc->bihc', attn_probs, node_feats)
        
        # If heads should be concatenated, we can do this by reshaping. Otherwise, take mean
        if self.concat_heads:
            node_feats = node_feats.reshape(batch_size, num_nodes, -1)
        else:
            node_feats = node_feats.mean(dim=2)
        
        return node_feats 
```

<!-- #region id="16j4vshTJd1l" -->
Again, we can apply the graph attention layer on our example graph above to understand the dynamics better. As before, the input layer is initialized as an identity matrix, but we set $\mathbf{a}$ to be a vector of arbitrary numbers to obtain different attention values. We use two heads to show the parallel, independent attention mechanisms working in the layer.
<!-- #endregion -->

```python id="BiP4hn7HJd1m" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631337954766, "user_tz": -330, "elapsed": 639, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="93369719-35e3-4bf4-a6b0-544a3d4e0366"
layer = GATLayer(2, 2, num_heads=2)
layer.projection.weight.data = torch.Tensor([[1., 0.], [0., 1.]])
layer.projection.bias.data = torch.Tensor([0., 0.])
layer.a.data = torch.Tensor([[-0.2, 0.3], [0.1, -0.1]])

with torch.no_grad():
    out_feats = layer(node_feats, adj_matrix, print_attn_probs=True)

print("Adjacency matrix", adj_matrix)
print("Input features", node_feats)
print("Output features", out_feats)
```

<!-- #region id="EX__hkkOJd1o" -->
We recommend that you try to calculate the attention matrix at least for one head and one node for yourself. The entries are 0 where there does not exist an edge between $i$ and $j$. For the others, we see a diverse set of attention probabilities. Moreover, the output features of node 3 and 4 are now different although they have the same neighbors.
<!-- #endregion -->

<!-- #region id="kNvD63-qmKBK" -->
## Convolution Fundamentals
<!-- #endregion -->

<!-- #region id="-s6Oz4zujU1l" -->
**Why convolution in ML?**

- Weight sharing
- Detection of translational invariant and local features
<!-- #endregion -->

<!-- #region id="31nP7bAUmoOO" -->
### Imports
<!-- #endregion -->

```python id="5EZLWtKTmncC"
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
```

<!-- #region id="1-SFGEbOjU1u" -->
### Definition
<!-- #endregion -->

<!-- #region id="6tc5FeJAjU1v" -->
\begin{align*}
c[n] = (v * w)[n] = \sum_{m=0}^{N-1} v[m] \cdot w[n-m]
\end{align*}
<!-- #endregion -->

```python id="P-w3eN4njU1x"
def conv(v, w):
    c = np.zeros(v.shape)
    for n in range(len(v)):
        c[n] = 0
        for m in range(len(v)):
            c[n] += v[m] * w[n - m]  
    return c
```

```python id="yWuC2SVijU1z" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1631594649331, "user_tz": -330, "elapsed": 587, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8f30bb14-a141-49ca-9f2f-f88cf7612c63"
N = 20
v = np.zeros(N)
v[8:12] = 1
w = np.zeros(N)
w[1:5] = 1
c = conv(v, w)

fig = plt.figure()
ax = fig.gca()
ax.plot(v, '.-')
ax.plot(w, '.-')
ax.plot(c, '.-')
ax.legend(['v', 'w', 'c'])
ax.grid(True)
```

<!-- #region id="_ub3xokljU11" -->
### Fourier transform
<!-- #endregion -->

<!-- #region id="MMc7h-07jU12" -->
Transformation $\mathcal F: \mathbb{R}^N \to \mathbb{R}^N$ with

\begin{align*}
\mathcal F^{-1}(\mathcal F (v)) &= v\\
\mathcal F(v * w) &= \mathcal F(v) \cdot \mathcal F(w).
\end{align*}
<!-- #endregion -->

<!-- #region id="eiFAIAZnjU12" -->
This implies
\begin{align*}
v * w &= \mathcal F^{-1}(\mathcal F (v * w))\\
&= \mathcal F^{-1}(\mathcal F(v) \cdot \mathcal F(w))
\end{align*}
<!-- #endregion -->

```python id="wZBnJFtXjU13" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631594702690, "user_tz": -330, "elapsed": 1028, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="dcb0378a-5082-4a35-a01e-a8fb2d9875a8"
v, w = np.random.rand(N), np.random.rand(N)
conv(v, w)
```

```python id="S6y-jQyljU14" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631594702692, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="70ac4870-753e-4561-d529-9b25f9ebf32b"
from scipy.fft import fft, ifft # Fast Fourier Transform / Inverse FFT
np.abs(ifft(fft(v) * fft(w)))
```

<!-- #region id="twDIgZFFjU16" -->
#### Definition of the Fourier transform
<!-- #endregion -->

<!-- #region id="sW7sZJ08jU17" -->
The Fourier transform can be computed as

\begin{align*}
\mathcal F(v) = U\cdot v, \;\;\mathcal F^{-1}(v) = \frac{1}{N}\ U^H \cdot v
\end{align*}

where the $N\times N$ matrix $U$ is defined as
\begin{align*}
\\
U = 
\begin{bmatrix}
u_0(0) & u_1(0) & \dots & u_{N-1}(0)\\
u_0(1) & u_1(1) & \dots & u_{N-1}(1)\\
\vdots & \vdots& & \vdots\\
u_0(N-1) & u_1(N-1) & \dots & u_{N-1}(N-1)\\
\end{bmatrix} 
\end{align*}

and $u_0, \dots, u_{N-1}$ are functions defined as

\begin{align*}
u_n(x)&:= \cos\left(2 \pi \frac{n}{N} x\right) - i \sin\left(2 \pi \frac{n}{N} x\right).
\end{align*}
<!-- #endregion -->

```python id="UF6lObDzjU18"
def matrix_U(N):
    u = lambda n, N: np.cos(2 * np.pi / N * n * np.arange(N)) - 1j * np.sin(2 * np.pi / N * n * np.arange(N))
    U = np.empty((N, 0))
    for n in range(N):
        U = np.c_[U, u(n, N)]
    return U


def fourier_transform(v):
    N = v.shape[0]
    U = matrix_U(N)
    return U @ v


def inverse_fourier_transform(v):
    N = v.shape[0]
    U = matrix_U(N)
    return (U.conj().transpose() @ v) / N
```

```python id="NGjk8BnrjU19" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631594708328, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f5bdbd3e-7afb-4271-a931-d38ad7f2dd4f"
fft(v) - fourier_transform(v)
```

```python id="3bYkgQCVjU1-" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631594709217, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="30cc2d55-63fe-4050-cc8d-791865a5000c"
ifft(v) - inverse_fourier_transform(v)
```

<!-- #region id="5h5jdqCujU1_" -->
#### Connection with the Laplacian
<!-- #endregion -->

<!-- #region id="ABnRkGpVjU2A" -->
The functions $u_n$ (the columns of the Fourier transform matrix) are eigenvectors of the Laplacian:

\begin{align*}
u_n(x)&:= \cos\left(2 \pi \frac{n}{N} x\right) - i \sin\left(2 \pi \frac{n}{N} x\right)\\
\Delta u_n(x)&:= \left(-4 \pi\frac{n^2}{N^2}\right) u_n(x)
\end{align*}
<!-- #endregion -->

<!-- #region id="uETONhRcjU2B" -->
#### Summary
<!-- #endregion -->

<!-- #region id="IeeGZUUEjU2D" -->
\begin{align*}
v * w 
= U^H ((U  w) \odot (U  v))
\end{align*}

or if $g_w=\mbox{diag}(U w)$ is  filter
\begin{align*}
v * w 
= U^H g_w U  w
\end{align*}
<!-- #endregion -->

```python id="INwk9GfVjU2D" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631594730512, "user_tz": -330, "elapsed": 406, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f1e846a3-9110-46e0-9167-af428c1e5278"
U = matrix_U(N)
np.abs((U.conj().transpose() / N) @ ((U @ v) * (U @ w)))
```

```python id="uYiFMzD0jU2E" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631594733736, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="eed7a679-fd6b-4d82-99d0-753edc7e9acb"
conv(v, w)
```

<!-- #region id="81CxozvcjU2F" -->
### Convolution on graphs
<!-- #endregion -->

<!-- #region id="XTLi4iv4jU2F" -->
**Plan**:
- Define the graph Laplacian
- Compute the spectrum
- Define a Fourier transform
- Define convolution on a graph
<!-- #endregion -->

<!-- #region id="llJtDpsqjU2G" -->
**Note:** From now on $G = (V, E)$ is an undirected, unweighted, simple graph.
<!-- #endregion -->

<!-- #region id="-xGl79_djU2J" -->
#### Graph Laplacian
<!-- #endregion -->

<!-- #region id="krHNV6BDjU2K" -->
Adjacency matrix
\begin{align*}
A_{ij} = \left\{
    \begin{array}{ll}
    1 &\text{ if } e_{ij}\in E\\
    0 &\text{ if } e_{ij}\notin E
    \end{array}
    \right.
\end{align*}

Degree matrix
\begin{align*}
D_{ij} = \left\{
    \begin{array}{ll}
    \mbox{deg}(v_i) &\text{ if } i=j\\
    0 &\text{ if } i\neq j
    \end{array}
    \right.
\end{align*}

Laplacian
\begin{align*}
L &= D - A.
\end{align*}

Normalized Laplacian
\begin{align*}
L &= I - D^{-1/2} A D^{-1/2}.
\end{align*}
<!-- #endregion -->

<!-- #region id="2cd_QoNejU2K" -->
#### Graph spectrum, Fourier transform, and convolution
<!-- #endregion -->

<!-- #region id="HuDedISkjU2L" -->
1. Spectral decomposition of the Laplacian:
\begin{align*}
L = U \Lambda U^T\\
\end{align*}


2. Fourier transform: if $v$ is a vector of features on the graph, then
\begin{align*}
\mathcal F (v) = U \cdot v, \;\;\mathcal F^{-1} (v) = U^T \cdot v\\
\end{align*}


3. Convolution with a filter $U \cdot w$
\begin{align*}
v * w = U ((U^T  w) \odot (U^T  v) )
\end{align*}


Or $g_w = \mbox{diag}(U^T w)$ is a filter, then
\begin{align*}
v * w = U g_w U^T  v
\end{align*}

<!-- #endregion -->

<!-- #region id="o-4jTYNqjU2M" -->
## Spectral-convolutional layers in PyTorch Geometric
<!-- #endregion -->

<!-- #region id="ymGdvbfpjU2M" -->
**Problem:** Computing the spectrum is a global and very expensive property.

**Goal:** Implementation as message passing.
<!-- #endregion -->

<!-- #region id="BiwOrRmpjU2N" -->
### ChebConv
<!-- #endregion -->

<!-- #region id="prNjVo4mjU2N" -->
- Original [paper](https://arxiv.org/pdf/1606.09375.pdf)
- PyTorch [doc](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.ChebConv)
<!-- #endregion -->

<!-- #region id="sbn9saMpjU2O" -->
#### Goal: 
Compute $U g_w U^T x$ with $g_w = g_w(\Lambda)$ a filter.
<!-- #endregion -->

<!-- #region id="81fAsO0_jU2O" -->
#### Chebyshev approximation

Chebyshev polynomials $T_k$:
\begin{align*}
T_{k}(x) = 2 x T_{k-1}(x) - T_{k-2}(x), \;\; T_0(x) = 1, T_1(x) = x
\end{align*}

#### Chebyshev approximation of the filter
Aproximation of the filter:
\begin{align*}
g_w(\Lambda) = \sum_{k=0}^K \theta_k T_k(\tilde \Lambda),\;\;\;\;\tilde \Lambda = \frac{2}{\lambda_\max} \Lambda - I \cdot \lambda_\max
\end{align*}


#### Property
If $L = U \Lambda U^T$ then $T_k(L) = U T_k(\Lambda) U^T$.

<!-- #endregion -->

<!-- #region id="T3iEp4oZjU2O" -->
#### Fast approximated convolution 
\begin{align*}
v * w &= U g_w U^T x
= U \left(\sum_{k=0}^K \theta_k T_k(\tilde \Lambda) \right)U^T x
=\sum_{k=0}^K  \theta_k U  T_k(\tilde \Lambda) U^T x\\ 
&=\sum_{k=0}^K  \theta_k T_k(\tilde L) x 
\end{align*}

\begin{align*}
\tilde L = \frac{2}{\lambda_\max} L - I
\end{align*}
<!-- #endregion -->

<!-- #region id="XT68q77zjU2P" -->
#### Properties:
- Depends on $L$ and $\lambda_\max$, not on $U, \Sigma$
- Uses only $K$-powers $\Rightarrow$ only the $K$-th neighborhood of each node, localized filter
<!-- #endregion -->

<!-- #region id="Xuz5GKV0jU2R" -->
### GCNConv
<!-- #endregion -->

<!-- #region id="bjhjOvQMjU2S" -->
- Original [paper](https://arxiv.org/pdf/1609.02907.pdf)
- PyTorch [doc](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv)
<!-- #endregion -->

<!-- #region id="ec6WKPXRjU2S" -->
Start from `ChebConv` and assume 
1. $K=1$ (linear approximation) so
\begin{align*}
v * w 
&=\sum_{k=0}^1  \theta_k T_k(\tilde L) x
= \theta_0 x + \theta_1 \tilde L x\\
\end{align*}

2. $\lambda_\max =2$ so
\begin{align*}
v * w 
&= \theta_0 x + \theta_1 (L - I) x\\
&= \theta_0 x - \theta_1 D^{-1/2} A D^{1/2} x\\
\end{align*}


3. $\theta_0=-\theta_1= \theta$ so 
\begin{align*}
v * w = \left(I + D^{-1/2} A D^{1/2}\right) x \theta
\end{align*}

4. Renormalization of $\theta$ by using 
\begin{align*}
\tilde A&:= I + A\\
\tilde D_{ii}&:= \sum_j \tilde A_{ij}
\end{align*}
so 
\begin{align*}
v * w = \left(D^{-1/2} A D^{1/2}\right) x \theta
\end{align*}

If $x$ is a $F$-dimensional feature vector, and we want an $F'$-dimensional feature vector as output:
use $W'\in \mathbb{R}^{F\times F'}$
\begin{align*}
v * w = \left(D^{-1/2} A D^{1/2}\right) x \Theta
\end{align*}

<!-- #endregion -->

<!-- #region id="OiSa_k7Mr7ov" -->
## Aggregation Functions in GNNs
<!-- #endregion -->

<!-- #region id="CJGFIV83tIfx" -->
### Context
<!-- #endregion -->

<!-- #region id="EIkXNFKmsXhr" -->
We explore how to perform neighborhood aggregation in GNNs, describing the GIN model and other recent techniques for selecting the right aggregation (PNA) or learn it (LAF).

We will override the aggregation method of the GIN convolution module of Pytorch Geometric implementing the following methods:

- Principal Neighborhood Aggregation (PNA)
- Learning Aggregation Functions (LAF)
<!-- #endregion -->

<!-- #region id="_Z30Q0IeuVQp" -->
### WL Isomorphism Test
<!-- #endregion -->

<!-- #region id="lDpSF5orswqp" -->
### Imports
<!-- #endregion -->

```python id="PwrdyvfbseoG" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631598091026, "user_tz": -330, "elapsed": 390, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1e41f55f-fde7-4062-fbad-97f7af1cca67"
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GINConv
from torch.nn import Linear
from torch_geometric.nn import MessagePassing, SAGEConv, GINConv, global_add_pool
import torch_scatter
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch.nn import Parameter, Module, Sigmoid
import os.path as osp

torch.manual_seed(42)
```

<!-- #region id="lplheNqFseoS" -->
### Message Passing Class
<!-- #endregion -->

<!-- #region id="Cvl2EURpseoW" -->
We are interested in the <span style='color:Blue'>aggregate</span> method, or, if you are using a sparse adjacency matrix, in the <span style='color:Blue'>message_and_aggregate</span> method. Convolutional classes in PyG extend MessagePassing, we construct our custom convoutional class extending GINConv.
<!-- #endregion -->

<!-- #region id="D220S_OrseoY" -->
Scatter operation in <span style='color:Blue'>aggregate</span>:
<!-- #endregion -->

<!-- #region id="80xKlB8oseoZ" -->
<img src="https://raw.githubusercontent.com/rusty1s/pytorch_scatter/master/docs/source/_figures/add.svg?sanitize=true" width="500">
<!-- #endregion -->

<!-- #region id="YtHRWKDdtn7K" -->
### LAF Aggregation Module
<!-- #endregion -->

<!-- #region id="C3R487ld0vHM" -->
**LAF Layer**
<!-- #endregion -->

```python id="mpsMsIF00wcV"
class AbstractLAFLayer(Module):
    def __init__(self, **kwargs):
        super(AbstractLAFLayer, self).__init__()
        assert 'units' in kwargs or 'weights' in kwargs
        if 'device' in kwargs.keys():
            self.device = kwargs['device']
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ngpus = torch.cuda.device_count()
        
        if 'kernel_initializer' in kwargs.keys():
            assert kwargs['kernel_initializer'] in [
                'random_normal',
                'glorot_normal',
                'he_normal',
                'random_uniform',
                'glorot_uniform',
                'he_uniform']
            self.kernel_initializer = kwargs['kernel_initializer']
        else:
            self.kernel_initializer = 'random_normal'

        if 'weights' in kwargs.keys():
            self.weights = Parameter(kwargs['weights'].to(self.device), \
                                     requires_grad=True)
            self.units = self.weights.shape[1]
        else:
            self.units = kwargs['units']
            params = torch.empty(12, self.units, device=self.device)
            if self.kernel_initializer == 'random_normal':
                torch.nn.init.normal_(params)
            elif self.kernel_initializer == 'glorot_normal':
                torch.nn.init.xavier_normal_(params)
            elif self.kernel_initializer == 'he_normal':
                torch.nn.init.kaiming_normal_(params)
            elif self.kernel_initializer == 'random_uniform':
                torch.nn.init.uniform_(params)
            elif self.kernel_initializer == 'glorot_uniform':
                torch.nn.init.xavier_uniform_(params)
            elif self.kernel_initializer == 'he_uniform':
                torch.nn.init.kaiming_uniform_(params)
            self.weights = Parameter(params, \
                                     requires_grad=True)
        e = torch.tensor([1,-1,1,-1], dtype=torch.float32, device=self.device)
        self.e = Parameter(e, requires_grad=False)
        num_idx = torch.tensor([1,1,0,0], dtype=torch.float32, device=self.device).\
                                view(1,1,-1,1)
        self.num_idx = Parameter(num_idx, requires_grad=False)
        den_idx = torch.tensor([0,0,1,1], dtype=torch.float32, device=self.device).\
                                view(1,1,-1,1)
        self.den_idx = Parameter(den_idx, requires_grad=False)
        

class LAFLayer(AbstractLAFLayer):
    def __init__(self, eps=1e-7, **kwargs):
        super(LAFLayer, self).__init__(**kwargs)
        self.eps = eps
    
    def forward(self, data, index, dim=0, **kwargs):
        eps = self.eps
        sup = 1.0 - eps 
        e = self.e

        x = torch.clamp(data, eps, sup)
        x = torch.unsqueeze(x, -1)
        e = e.view(1,1,-1)        

        exps = (1. - e)/2. + x*e 
        exps = torch.unsqueeze(exps, -1)
        exps = torch.pow(exps, torch.relu(self.weights[0:4]))

        scatter = torch_scatter.scatter_add(exps, index.view(-1), dim=dim)
        scatter = torch.clamp(scatter, eps)

        sqrt = torch.pow(scatter, torch.relu(self.weights[4:8]))
        alpha_beta = self.weights[8:12].view(1,1,4,-1)
        terms = sqrt * alpha_beta

        num = torch.sum(terms * self.num_idx, dim=2)
        den = torch.sum(terms * self.den_idx, dim=2)
        
        multiplier = 2.0*torch.clamp(torch.sign(den), min=0.0) - 1.0

        den = torch.where((den < eps) & (den > -eps), multiplier*eps, den)

        res = num / den
        return res
```

```python id="XgskJll0tn7M"
class GINLAFConv(GINConv):
    def __init__(self, nn, units=1, node_dim=32, **kwargs):
        super(GINLAFConv, self).__init__(nn, **kwargs)
        self.laf = LAFLayer(units=units, kernel_initializer='random_uniform')
        self.mlp = torch.nn.Linear(node_dim*units, node_dim)
        self.dim = node_dim
        self.units = units
    
    def aggregate(self, inputs, index):
        x = torch.sigmoid(inputs)
        x = self.laf(x, index)
        x = x.view((-1, self.dim * self.units))
        x = self.mlp(x)
        return x
```

<!-- #region id="q0Qwqbrvtn7N" -->
### PNA Aggregation
<!-- #endregion -->

```python id="ysP9SOV5tn7P"
class GINPNAConv(GINConv):
    def __init__(self, nn, node_dim=32, **kwargs):
        super(GINPNAConv, self).__init__(nn, **kwargs)
        self.mlp = torch.nn.Linear(node_dim*12, node_dim)
        self.delta = 2.5749
    
    def aggregate(self, inputs, index):
        sums = torch_scatter.scatter_add(inputs, index, dim=0)
        maxs = torch_scatter.scatter_max(inputs, index, dim=0)[0]
        means = torch_scatter.scatter_mean(inputs, index, dim=0)
        var = torch.relu(torch_scatter.scatter_mean(inputs ** 2, index, dim=0) - means ** 2)
        
        aggrs = [sums, maxs, means, var]
        c_idx = index.bincount().float().view(-1, 1)
        l_idx = torch.log(c_idx + 1.)
        
        amplification_scaler = [c_idx / self.delta * a for a in aggrs]
        attenuation_scaler = [self.delta / c_idx * a for a in aggrs]
        combinations = torch.cat(aggrs+ amplification_scaler+ attenuation_scaler, dim=1)
        x = self.mlp(combinations)
    
        return x
```

<!-- #region id="dRiavOjHtn7Q" -->
### Test the new classes
<!-- #endregion -->

```python id="8L6L2VEltn7R" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631598131970, "user_tz": -330, "elapsed": 1255, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="dcbe9ec6-2185-4885-c678-8451b4a8b4d8"
path = osp.join('./', 'data', 'TU')
dataset = TUDataset(path, name='MUTAG').shuffle()
test_dataset = dataset[:len(dataset) // 10]
train_dataset = dataset[len(dataset) // 10:]
test_loader = DataLoader(test_dataset, batch_size=128)
train_loader = DataLoader(train_dataset, batch_size=128)
```

<!-- #region id="KfmUuo0F2H5o" -->
### LAF Model
<!-- #endregion -->

```python id="9UJS6CRltn7S"
class LAFNet(torch.nn.Module):
    def __init__(self):
        super(LAFNet, self).__init__()

        num_features = dataset.num_features
        dim = 32
        units = 3
        
        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINLAFConv(nn1, units=units, node_dim=num_features)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINLAFConv(nn2, units=units, node_dim=dim)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINLAFConv(nn3, units=units, node_dim=dim)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINLAFConv(nn4, units=units, node_dim=dim)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINLAFConv(nn5, units=units, node_dim=dim)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

```

<!-- #region id="O4QzD1Z72FKD" -->
### PNA Model
<!-- #endregion -->

```python id="Zw63xP4ctn7T"
class PNANet(torch.nn.Module):
    def __init__(self):
        super(PNANet, self).__init__()

        num_features = dataset.num_features
        dim = 32

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINPNAConv(nn1, node_dim=num_features)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINPNAConv(nn2, node_dim=dim)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINPNAConv(nn3, node_dim=dim)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINPNAConv(nn4, node_dim=dim)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINPNAConv(nn5, node_dim=dim)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)
```

<!-- #region id="K1XJf1R42CmD" -->
### GIN Model
<!-- #endregion -->

```python id="PsbosIB4tn7V"
class GINNet(torch.nn.Module):
    def __init__(self):
        super(GINNet, self).__init__()

        num_features = dataset.num_features
        dim = 32

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)
```

<!-- #region id="dP8Ca1Of2ZMW" -->
### Training
<!-- #endregion -->

```python id="E_uoVQX-2YFW"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = "PNA"
if net == "LAF":
    model = LAFNet().to(device)
elif net == "PNA":
    model = PNANet().to(device)
elif net == "GIN":
    GINNet().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    model.train()

    if epoch == 51:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_dataset)


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)
```

```python id="jd8O2SB_tn7W"
for epoch in range(1, 101):
    train_loss = train(epoch)
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print('Epoch: {:03d}, Train Loss: {:.7f}, '
          'Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss,
                                                       train_acc, test_acc))
```

```python id="m0H3TkSptn7Z"

```

```python id="1MUjl0Sf6iym"
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
```

<!-- #region id="bO7Cksrg6iyt" -->
## Graph AutoEncoders - GAE & VGAE    
<!-- #endregion -->

<!-- #region id="B5Ffn8dJ7Ukv" -->
[paper](https://arxiv.org/pdf/1611.07308.pdf)  
[code](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/autoencoder.py)
<!-- #endregion -->

<!-- #region id="Geskh6iL8iIm" -->
### Context
<!-- #endregion -->

<!-- #region id="bHIpDgbW8_Cv" -->
#### Loss function
<!-- #endregion -->

<!-- #region id="oSlE-8HH7cBM" -->
### Imports
<!-- #endregion -->

```python id="ROnCeb8m7dHz"
from torch_geometric.nn import GAE
from torch_geometric.nn import VGAE
from torch.utils.tensorboard import SummaryWriter
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
```

<!-- #region id="kqtK6kIl6iyx" -->
### Load the CiteSeer data
<!-- #endregion -->

```python id="xJcHe3-U6iyy"
dataset = Planetoid("\..", "CiteSeer", transform=T.NormalizeFeatures())
dataset.data
```

```python id="wFajQVdR6iyz" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631600026265, "user_tz": -330, "elapsed": 28, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c5690042-afb3-41a3-986f-7254fbc0446c"
data = dataset[0]
data.train_mask = data.val_mask = data.test_mask = None
data
```

```python id="ilpI-cCL6iy0" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631600027952, "user_tz": -330, "elapsed": 703, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="43d89ce4-49f8-4c9a-89ea-eb349a605db6"
data = train_test_split_edges(data)
data
```

<!-- #region id="DTHs2luw6iy4" -->
### Define the Encoder
<!-- #endregion -->

```python id="e-9gsNuE6iy6"
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True) # cached only for transductive learning

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)
```

<!-- #region id="A5H_VoaH6iy9" -->
### Define the Autoencoder
<!-- #endregion -->

```python id="ZYYew3im6izA"
# parameters
out_channels = 2
num_features = dataset.num_features
epochs = 100

# model
model = GAE(GCNEncoder(num_features, out_channels))

# move to GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = data.x.to(device)
train_pos_edge_index = data.train_pos_edge_index.to(device)

# inizialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
```

```python id="hqrC2jls6izC"
def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    #if args.variational:
    #   loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)


def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)
```

```python id="SFPciyaw6izD"
for epoch in range(1, epochs + 1):
    loss = train()

    auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
```

```python id="v-m09mHT6izD" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631600048621, "user_tz": -330, "elapsed": 577, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9e52c43f-4606-40e0-c2e9-d370dcc2f79e"
Z = model.encode(x, train_pos_edge_index)
Z
```

<!-- #region id="2KqlZtCd6izF" -->
### Result analysis with Tensorboard
<!-- #endregion -->

```python id="UJctXayE6izG"
# parameters
out_channels = 2
num_features = dataset.num_features
epochs = 100

# model
model = GAE(GCNEncoder(num_features, out_channels))

# move to GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = data.x.to(device)
train_pos_edge_index = data.train_pos_edge_index.to(device)

# inizialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
```

```python id="FVqv-NxA6izH"
writer = SummaryWriter('runs/GAE1_experiment_'+'2d_100_epochs')
```

```python id="MSrnWNIq6izH"
for epoch in range(1, epochs + 1):
    loss = train()
    auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
    
    
    writer.add_scalar('auc train',auc,epoch) # new line
    writer.add_scalar('ap train',ap,epoch)   # new line
```

<!-- #region id="vrjYGm2L6izI" -->
### Graph Variational AutoEncoder (GVAE)
<!-- #endregion -->

```python id="ATQTuF5Z6izJ" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631600056892, "user_tz": -330, "elapsed": 40, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c8a017c0-042d-41b9-df5c-01b3ec48bd93"
dataset = Planetoid("\..", "CiteSeer", transform=T.NormalizeFeatures())
data = dataset[0]
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data)


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
```

```python id="ORqNPw6v6izK"
out_channels = 2
num_features = dataset.num_features
epochs = 300


model = VGAE(VariationalGCNEncoder(num_features, out_channels))  # new line

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = data.x.to(device)
train_pos_edge_index = data.train_pos_edge_index.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
```

```python id="QXaCVkwY6izL"
def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    
    loss = loss + (1 / data.num_nodes) * model.kl_loss()  # new line
    loss.backward()
    optimizer.step()
    return float(loss)


def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)
```

```python id="vS3PKukE6izN"
writer = SummaryWriter('runs/VGAE_experiment_'+'2d_100_epochs')

for epoch in range(1, epochs + 1):
    loss = train()
    auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
    
    
    writer.add_scalar('auc train',auc,epoch) # new line
    writer.add_scalar('ap train',ap,epoch)   # new line
```

<!-- #region id="Hw-WFrHO6izO" -->
## End
<!-- #endregion -->
