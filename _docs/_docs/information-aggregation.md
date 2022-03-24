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

<!-- #region id="ij48zhee8Gfb" -->
# Information Aggregation
> Understanding Information Aggregation by applying SAGE convolution layer on Cora dataset
<!-- #endregion -->

```python id="33HhJOZ_6wxl"
# torch geometric
try: 
    import torch_geometric
except ModuleNotFoundError:
    # Installing torch geometric packages with specific CUDA+PyTorch version. 
    # See https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html for details 
    import torch
    TORCH = torch.__version__.split('+')[0]
    CUDA = 'cu' + torch.version.cuda.replace('.','')

    !pip install torch-scatter     -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
    !pip install torch-sparse      -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
    !pip install torch-cluster     -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
    !pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
    !pip install torch-geometric 
    import torch_geometric
import torch_geometric.nn as geom_nn
import torch_geometric.data as geom_data
```

<!-- #region id="VHKkioYa8K5F" -->
### Prototype
<!-- #endregion -->

```python id="5qiU3KMV6zFr"
import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SAGEConv
```

```python id="GBx6zeSa7B4P"
use_cuda_if_available = False
```

```python colab={"base_uri": "https://localhost:8080/"} id="6FVfZweF7DT8" executionInfo={"status": "ok", "timestamp": 1638730071488, "user_tz": -330, "elapsed": 3985, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="14cb5517-dc09-43a7-b5ab-c8481df45c58"
dataset = Planetoid(root="/content/cora", name= "Cora")
```

```python colab={"base_uri": "https://localhost:8080/"} id="ljx3DrLS7GrR" executionInfo={"status": "ok", "timestamp": 1638730071490, "user_tz": -330, "elapsed": 48, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1a129c18-9b06-4e8b-d5bc-d0bb02492e00"
print(dataset)
print("number of graphs:\t\t",len(dataset))
print("number of classes:\t\t",dataset.num_classes)
print("number of node features:\t",dataset.num_node_features)
print("number of edge features:\t",dataset.num_edge_features)
```

```python colab={"base_uri": "https://localhost:8080/"} id="mSgjzQ1G7G9p" executionInfo={"status": "ok", "timestamp": 1638730087701, "user_tz": -330, "elapsed": 516, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4876c74e-6b09-4887-9797-02182e59ab8c"
print(dataset.data)
```

```python colab={"base_uri": "https://localhost:8080/"} id="_2cP4-IH7Lfr" executionInfo={"status": "ok", "timestamp": 1638730116565, "user_tz": -330, "elapsed": 499, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6dbae112-a82f-443a-c7d6-ce88e41cd031"
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

```python colab={"base_uri": "https://localhost:8080/"} id="Bq6K-QAvbcH-" executionInfo={"status": "ok", "timestamp": 1638730214359, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8d846ec9-8fbd-41ff-a012-a66ec56dabc1"
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

```python colab={"base_uri": "https://localhost:8080/"} id="ISVIQ-qFcyoa" executionInfo={"status": "ok", "timestamp": 1638730263431, "user_tz": -330, "elapsed": 46858, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9c8ebdca-d970-454b-f58b-cb1adf1d2f7e"
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
### Scripting
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

<!-- #region id="n83l4fRt9luz" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="trEfQuxo9lu5" executionInfo={"status": "ok", "timestamp": 1638730732578, "user_tz": -330, "elapsed": 3418, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="86f4bbba-495f-4d90-b6f1-e5303bfd5d13"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="xe6ycIM89lu6" -->
---
<!-- #endregion -->

<!-- #region id="Cy6OjZxd9lu6" -->
**END**
<!-- #endregion -->
