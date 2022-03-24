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

```python colab={"base_uri": "https://localhost:8080/"} id="qmB6NJSPQd88" executionInfo={"status": "ok", "timestamp": 1631463146947, "user_tz": -330, "elapsed": 489, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5e6f1c63-20e9-47b4-e373-c2b5c6cc7e13"
import os
project_name = "recobase"; branch = "US987772"; account = "recohut"
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

```python colab={"base_uri": "https://localhost:8080/"} id="nEywuPvdySWp" executionInfo={"status": "ok", "timestamp": 1631463287311, "user_tz": -330, "elapsed": 411, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0959f5bd-6d48-412b-cf67-ec58530b8942"
import glob

glob.glob('./data/silver/**/*', recursive=True)
```

```python id="x3Zkt5Rpywqs"
!rm ./data/silver/processed/book-crossing/book-crossing.statinfo
```

```python id="N5m1GB_dnWCB"
!git status -u
```

```python id="gkw68JpUQd9N" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631463751613, "user_tz": -330, "elapsed": 1514, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="de22f0ba-4ef7-4578-d136-cc47bf77d1c9"
!git add .
!git commit -m 'commit'
!git push origin "{branch}"
```

```python colab={"base_uri": "https://localhost:8080/"} id="uh5nTSxTQnsS" executionInfo={"status": "ok", "timestamp": 1631455455367, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0a03c307-ea1c-4f81-b611-c134ccee8104"
%%writefile requirements.txt
icecream
torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
torch-geometric
wget
libarchive
```

```python id="wHwcTiebToWO"
!pip install -r requirements.txt
```

<!-- #region id="9wpCL7AzQ3nY" -->
## Data
<!-- #endregion -->

```python id="ULTxUeVZrVZm"
import torch
import random
import numpy as np
import time
import argparse
from sklearn.model_selection import train_test_split
from torch_geometric.data import DataLoader

from src.datasets import GMCFDataset
from src.trainers import GMCFTrainer
```

```python id="otCb4cNPrT9v"
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ml-1m', help='which dataset to use')
parser.add_argument('--rating_file', type=str, default='implicit_ratings.csv', help='which dataset to use')
parser.add_argument('--dim', type=int, default=64, help='dimension of entity and relation embeddings')
parser.add_argument('--l2_weight', type=float, default=1e-5, help='weight of the l2 regularization term')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--n_epoch', type=int, default=50, help='the number of epochs')
parser.add_argument('--hidden_layer', type=int, default=256, help='neural hidden layer')
parser.add_argument('--num_user_features', type=int, default=3, help='the number of user attributes')
parser.add_argument('--random_seed', type=int, default=2019, help='size of common item be counted')
args = parser.parse_args(args={})
```

```python colab={"base_uri": "https://localhost:8080/"} id="tXCf_gFmeew5" outputId="a95233fb-a6d8-4099-b6a3-cb27de1c3716"
dataset = GMCFDataset('data/silver/', args.dataset, args.rating_file)
```

```python id="BIzRcYwhtnPO"
data_num = dataset.data_N()
feature_num = dataset.feature_N()
train_index, val_index = dataset.stat_info['train_test_split_index']
print(np.concatenate((dataset[0][0:5], dataset[0][6:10])))
```

```python id="z2NvUfNuu027"
# split inner graphs
train_dataset = dataset[:train_index]
val_dataset = dataset[train_index:val_index]
test_dataset = dataset[val_index:]
```

```python id="PPxHSnQXgnAC"
n_workers = 4
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=n_workers)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=n_workers)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=n_workers)
```

```python id="wi76qvhLu2wF"
show_loss = True
print(f"""
datast: {args.dataset}
vector dim: {args.dim}
batch_size: {args.batch_size}
lr: {args.lr}
""")

```

```python id="TZtik8A_XPBB"
datainfo = {}
datainfo['train'] = train_loader
datainfo['val'] = val_loader 
datainfo['test'] = test_loader
datainfo['feature_num'] = feature_num 
datainfo['data_num'] = [len(train_dataset), len(val_dataset), len(test_dataset)]
```

```python id="ppDuzyEDZs02"
train(args, datainfo, show_loss)
```
