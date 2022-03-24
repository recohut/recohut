---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: Python 2
    language: python
    name: python2
---

<!-- #region id="TECBk0nwkUCj" -->
# NCF from scratch in pytorch
> Building neural collaborative filtering model from scratch in pytorch and plotting movielens rating matrix before and after rating prediction

- toc: false
- badges: true
- comments: true
- categories: [Pytorch, Visualization, Movie, NCF]
- image:
<!-- #endregion -->

```python id="0ZN73VFNkPur"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
```

```python id="hVhGAA2qkPuw" outputId="278a6aca-69bd-4ba1-ae7b-17e8e066d101"
rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table('u1.base', sep='\t', header=None, names=rnames)
ratings
```

```python id="kKm1yQDKkPuy" outputId="ee207912-cc0e-46e0-c5e3-106e02e1df41"
M = 1682 #number of movies
N = 943 #number of users

R = np.zeros((M,N)) #datamatrix

for k in range(len(ratings)):
    i = ratings['movie_id'][k] - 1
    j = ratings['user_id'][k] - 1
    r = ratings['rating'][k]
    R[i,j] = 1.0

print R
```

```python id="m6U4ZfqNkPuz" outputId="c5cec9ed-cfee-420c-9466-d21677e8f30a"
Data = R[3:20,5:16] #Sample from original data matrix

plt.imshow(Data, interpolation='nearest',vmax=1,vmin=0)
plt.colorbar()
plt.set_cmap('binary')
plt.xlabel('Users')
plt.ylabel('Movies')
plt.show()
```

```python id="EJHL1yHGkPu0"
x_train = np.zeros((M*N,M+N))
for i in range(M):
    for j in range(N):
        x_train[i*N+j,i] = 1.0
        x_train[i*N+j,j+M] = 1.0
```

```python id="Lne2XzTpkPu1" outputId="0216789d-59d6-4ba4-8d4c-009b007e5129"
x_train
```

```python id="xzsMa-HckPu2"
y_train = R.reshape(M*N)
```

```python id="pmEZRdAgkPu3" outputId="f408024e-cbf5-4e08-8e60-45d26ed164ca"
print y_train
```

```python id="2Z2hOSJWkPu3"
x_train, y_train = map(torch.tensor, (np.float32(x_train), np.float32(y_train)))
```

```python id="nkfdZ41VkPu4"
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear_m = nn.Linear(M,64)
        self.linear_u = nn.Linear(N,64)
        self.linear1 = nn.Linear(128,32)
        self.linear2 = nn.Linear(32,16)
        self.linear3 = nn.Linear(16,1)

    def forward(self, x):
        x_m = x[:,0:M]
        x_u = x[:,M:M+N]
        x_m = self.linear_m(x_m)
        x_u = self.linear_u(x_u)
        x = torch.cat((x_m,x_u),-1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        return torch.squeeze(x)
```

```python id="pXfLza5XkPu5" outputId="d561d5b4-ea2f-45fd-818b-0cd4f8692e9c"
net = Net()
print(net)
```

```python id="wV51Gl0skPu6" outputId="7e5bfca2-8647-49c9-e928-f4f480115899"
out = net(x_train)
print(out)
```

```python id="VmPvBcqHkPu6"
#binary_cross_entropy_with_logits kullanırsak sonda yaptığımız sigmoid'e
#gerek kalmıyor. cross_entropy de aynı şekilde softmax istemiyor 
loss_func = F.binary_cross_entropy
```

```python id="_6rIsvFekPu7" outputId="60eabc39-88f2-408a-fca7-9b8073f5ecdf"
loss_func(net(x_train), y_train)
```

```python id="t0ZGaor1kPu8"
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=100, shuffle=True)
```

```python id="2ELPKqdVkPu8"
from torch import optim
def get_model():
    model = Net()
    return model, optim.SGD(model.parameters(), lr=0.05)
```

```python id="q6gKA7ACkPu9" outputId="baa9219e-9655-467d-9201-a6160e7fc426"
model, opt = get_model()

for epoch in range(5):
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()
    print(loss)

print(loss_func(model(x_train), y_train))
```

```python id="R-STN7YUkPu9"
pred = model(x_train)
```

```python id="F6ksuOBjkPu-"
R_pred = pred.detach().numpy().reshape(M,N)
```

```python id="tSRlRKs1kPu-" outputId="762d9469-8f31-4beb-9010-6b5f75ee27a5"
Data = R_pred[3:20,5:16] #Sample from original data matrix

plt.imshow(Data, interpolation='nearest',vmax=1,vmin=0)
plt.colorbar()
plt.set_cmap('binary')
plt.xlabel('Users')
plt.ylabel('Movies')
plt.show()
```

```python id="VU9OnctakPu_"

```
