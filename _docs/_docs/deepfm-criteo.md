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

<!-- #region id="2VZ8fr0qMt5t" -->
# DeepFM on Criteo DAC sample dataset in PyTorch
<!-- #endregion -->

```python id="EToD4LnRLgyY"
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import sys
import os

import torch.nn as nn
import numpy as np
import torch.utils.data as Data
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import log_loss, roc_auc_score
from collections import OrderedDict, namedtuple, defaultdict
import random
```

```python colab={"base_uri": "https://localhost:8080/"} id="L60wRz3KLutF" executionInfo={"status": "ok", "timestamp": 1641536229646, "user_tz": -330, "elapsed": 1342, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1a380f02-5cf1-43d2-8a40-974119bbecdc"
!wget -q --show-progress https://github.com/RecoHut-Datasets/criteo/raw/v1/dac_sample.txt
```

```python id="D_slY7KGN44C"
class FM(nn.Module):
    def __init__(self, p, k):
        super(FM, self).__init__()
        self.p = p
        self.k = k
        self.linear = nn.Linear(self.p, 1, bias=True)
        self.v = nn.Parameter(torch.Tensor(self.p, self.k), requires_grad=True)
        self.v.data.uniform_(-0.01, 0.01)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        linear_part = self.linear(x)
        inter_part1 = torch.pow(torch.mm(x, self.v), 2)
        inter_part2 = torch.mm(torch.pow(x, 2), torch.pow(self.v, 2))
        pair_interactions = torch.sum(torch.sub(inter_part1, inter_part2), dim=1)
        self.drop(pair_interactions)
        output = linear_part.transpose(1, 0) + 0.5 * pair_interactions
        return output.view(-1, 1)
```

```python id="ZiDRlS-GO9Y1"
class deepfm(nn.Module):
    def __init__(self, feat_sizes, sparse_feature_columns, dense_feature_columns,dnn_hidden_units=[400, 400,400], dnn_dropout=0.0, ebedding_size=4,
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                 device='cpu'):
        super(deepfm, self).__init__()
        self.feat_sizes = feat_sizes
        self.device = device
        self.dense_feature_columns = dense_feature_columns
        self.sparse_feature_columns = sparse_feature_columns
        self.embedding_size = ebedding_size
        self.l2_reg_linear = l2_reg_linear

        self.bias = nn.Parameter(torch.zeros((1, )))
        self.init_std = init_std
        self.dnn_dropout = dnn_dropout

        self.embedding_dic = nn.ModuleDict({feat:nn.Embedding(self.feat_sizes[feat], self.embedding_size, sparse=False)
                                            for feat in self.sparse_feature_columns})
        for tensor in self.embedding_dic.values():
            nn.init.normal_(tensor.weight, mean=0, std=self.init_std)
        self.embedding_dic.to(self.device)

        self.feature_index = defaultdict(int)
        start = 0
        for feat in self.feat_sizes:
            if feat in self.feature_index:
                continue
            self.feature_index[feat] = start
            start += 1

        # 输入维度 fm层与DNN层共享嵌入层， 输入维度应该是一样的
        self.input_size = self.embedding_size * len(self.sparse_feature_columns)+len(self.dense_feature_columns)
        # fm
        self.fm = FM(self.input_size, 10)

        # DNN
        self.dropout = nn.Dropout(self.dnn_dropout)
        self.hidden_units = [self.input_size] + dnn_hidden_units
        self.Linears = nn.ModuleList([nn.Linear(self.hidden_units[i], self.hidden_units[i+1]) for i in range(len(self.hidden_units)-1)])
        self.relus = nn.ModuleList([nn.ReLU() for i in range(len(self.hidden_units)-1)])
        for name, tensor in self.Linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=self.init_std)
        self.dnn_outlayer = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(self.device)


    def forward(self, x):
        # x shape 1024*39

        sparse_embedding = [self.embedding_dic[feat](x[:, self.feature_index[feat]].long()) for feat in self.sparse_feature_columns]
        sparse_embedding = torch.cat(sparse_embedding, dim=-1)
        # print(sparse_embedding.shape)  # batch * 208

        dense_value = [x[:, self.feature_index[feat]] for feat in
                            self.dense_feature_columns]

        dense_value = torch.cat(dense_value, dim=0)
        dense_value = torch.reshape(dense_value, (len(self.dense_feature_columns), -1))
        dense_value = dense_value.T
        # print(dense_value.shape) # batch * 13

        input_x = torch.cat((dense_value, sparse_embedding), dim=1)
        # print(input_x.shape) # batch * 221

        fm_logit = self.fm(input_x)

        for i in range(len(self.Linears)):
            fc = self.Linears[i](input_x)
            fc = self.relus[i](fc)
            fc = self.dropout(fc)
            input_x = fc
        dnn_logit = self.dnn_outlayer(input_x)

        y_pre = torch.sigmoid(fm_logit+dnn_logit+self.bias)
        return y_pre
```

```python id="g-zB2cR0PJQ_"
def get_auc(loader, model):
    pred, target = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).float()
            y_hat = model(x)
            pred += list(y_hat.numpy())
            target += list(y.numpy())
    auc = roc_auc_score(target, pred)
    return auc
```

```python colab={"base_uri": "https://localhost:8080/"} id="ZH69aNy6LmvA" executionInfo={"status": "ok", "timestamp": 1641537334473, "user_tz": -330, "elapsed": 181659, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9a504fcb-fe7b-49c4-bf15-2ccece85893d"
batch_size = 1024
lr = 0.00005
wd = 0.00001
epoches = 10

seed = 1024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I' + str(i) for i in range(1, 14)]
col_names = ['label'] + dense_features + sparse_features
df = pd.read_csv('dac_sample.txt', names=col_names, sep='\t')
feature_names = dense_features + sparse_features

df[sparse_features] = df[sparse_features].fillna('-1', )
df[dense_features] = df[dense_features].fillna(0, )
target = ['label']

for feat in sparse_features:
    lbe = LabelEncoder()
    df[feat] = lbe.fit_transform(df[feat])

mms = MinMaxScaler(feature_range=(0, 1))
df[dense_features] = mms.fit_transform(df[dense_features])

feat_size1 = {feat: 1 for feat in dense_features}
feat_size2 = {feat: len(df[feat].unique()) for feat in sparse_features}
feat_sizes = {}
feat_sizes.update(feat_size1)
feat_sizes.update(feat_size2)

# print(df.head(5))
# print(feat_sizes)

train, test =train_test_split(df, test_size=0.2, random_state=2021)
train_model_input = {name: train[name] for name in feature_names}
test_model_input = {name: test[name] for name in feature_names}

device = 'cpu'

model = deepfm(feat_sizes, sparse_feature_columns=sparse_features, dense_feature_columns=dense_features,
                dnn_hidden_units=[1000, 500, 250], dnn_dropout=0.9, ebedding_size=16,
                l2_reg_linear=1e-3, device=device)

train_label = pd.DataFrame(train['label'])
train_data = train.drop(columns=['label'])
#print(train.head(5))
train_tensor_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(train_data)), torch.from_numpy(np.array(train_label)))
train_loader = DataLoader(dataset=train_tensor_data, shuffle=True, batch_size=batch_size)

test_label = pd.DataFrame(test['label'])
test_data = test.drop(columns=['label'])
test_tensor_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(test_data)),
                                                    torch.from_numpy(np.array(test_label)))
test_loader = DataLoader(dataset=test_tensor_data, shuffle=False, batch_size=batch_size)

loss_func = nn.BCELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

for epoch in range(epoches):
    total_loss_epoch = 0.0
    total_tmp = 0

    model.train()
    for index, (x, y) in enumerate(train_loader):
        x = x.to(device).float()
        y = y.to(device).float()

        y_hat = model(x)

        optimizer.zero_grad()
        loss = loss_func(y_hat, y)
        loss.backward()
        optimizer.step()
        total_loss_epoch += loss.item()
        total_tmp += 1

    auc = get_auc(test_loader, model)
    print('epoch/epoches: {}/{}, train loss: {:.3f}, test auc: {:.3f}'.format(epoch, epoches, total_loss_epoch / total_tmp, auc))
```

```python colab={"base_uri": "https://localhost:8080/"} id="tmhWlpwiL0bb" executionInfo={"status": "ok", "timestamp": 1641537342905, "user_tz": -330, "elapsed": 3754, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="57e12b81-7f55-4ce0-8aa4-b43e46a80887"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```
