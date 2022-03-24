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

<!-- #region id="k_6fIKUKnhDX" -->
# ListNet Learning-to-rank model
> Training ListNet on synthetic data in pytorch

- toc: false
- badges: true
- comments: false
- categories: [LTR]
- image:
<!-- #endregion -->

```python id="JUE9Su3Skl4U"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
```

```python id="zfhg5_MMknm_"
class RankNet(nn.Module):
    def __init__(self, num_feature):
        super(RankNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_feature, 512),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.output_sig = nn.Sigmoid()

    def forward(self, input_1, input_2):
        # relevant document score
        s1 = self.model(input_1)
        # irrelevant document score
        s2 = self.model(input_2)
        # subtract scores
        out = self.output_sig(s1-s2)
        return out
    
    def predict(self, input_):
        s = self.model(input_)
        return s
```

```python id="WnphmhIfljUN"
# generating synthetic data

n_sample = 30000
n_feature = 300

data1 = torch.rand((n_sample, n_feature))
data2 = torch.rand((n_sample, n_feature))

y = np.random.random((n_sample, 1))
y = y > 0.9
y = 1. * y
y = torch.Tensor(y)
```

```python id="lED0zxRDlpXp"
rank_model = RankNet(num_feature = n_feature)
optimizer = torch.optim.Adam(rank_model.parameters())
loss_fun = torch.nn.BCELoss()
```

```python id="qC0pGjI8l7Pj"
# putting to GPU for faster learning
rank_model.cuda()
loss_fun.cuda()
data1 = data1.cuda()
data2 = data2.cuda()
y = y.cuda()
```

```python colab={"base_uri": "https://localhost:8080/"} id="dbpBBjYVmKsu" outputId="03ba1c53-6eea-425d-b952-e29a7b7a5a04"
epoch = 5000

losses = []

for i in range(epoch):
    
    rank_model.zero_grad()
    
    y_pred = rank_model(data1, data2)
    
    loss = loss_fun(y_pred,y)
    
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())

    if i % 200 == 0:
        print('Epoch{}, loss : {}'.format(i, loss.item()))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 282} id="G1HI8tsI39F8" outputId="2fb3073e-0832-44e6-83bd-35be97c4065a"
x = list(range(5000))
plt.plot(x, losses)
```
