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
    language: python
    name: python3
---

<!-- #region id="UxWgUAp7vnIM" -->
# Neural Matrix Factorization on Movielens
> Experiments with different variations of Neural matrix factorization model in PyTorch on movielens dataset

- toc: true
- badges: true
- comments: true
- categories: [movie, pytorch]
- image: 
<!-- #endregion -->

```python id="hfac3W-Z4yEs"
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="74OaJCfn5H4Z" outputId="9ec26cd4-a004-49cf-aefb-9a24c670bf11"
data = pd.read_csv("https://raw.githubusercontent.com/sparsh-ai/reco-data/master/MovieLens_LatestSmall_ratings.csv")
data.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="F0Irmpk2oUna" outputId="082a3deb-3f36-4d93-8917-77c27db5fc55"
data.shape
```

<!-- #region id="AsSMbrTG6LQr" -->
Data encoding
<!-- #endregion -->

```python id="m_wEgrHx5U93"
np.random.seed(3)
msk = np.random.rand(len(data)) < 0.8
train = data[msk].copy()
valid = data[~msk].copy()
```

```python id="fmeQCQXP6Vtv"
# here is a handy function modified from fast.ai
def proc_col(col, train_col=None):
    """Encodes a pandas column with continous ids. 
    """
    if train_col is not None:
        uniq = train_col.unique()
    else:
        uniq = col.unique()
    name2idx = {o:i for i,o in enumerate(uniq)}
    return name2idx, np.array([name2idx.get(x, -1) for x in col]), len(uniq)
```

```python id="OUfpCvFJ6W72"
def encode_data(df, train=None):
    """ Encodes rating data with continous user and movie ids. 
    If train is provided, encodes df with the same encoding as train.
    """
    df = df.copy()
    for col_name in ["userId", "movieId"]:
        train_col = None
        if train is not None:
            train_col = train[col_name]
        _,col,_ = proc_col(df[col_name], train_col)
        df[col_name] = col
        df = df[df[col_name] >= 0]
    return df
```

```python id="TZDUY2rt6Z9B"
# encoding the train and validation data
df_train = encode_data(train)
df_valid = encode_data(valid, train)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="c9VIAfR6otTe" outputId="dc5ad891-2004-4fda-acfa-22d04525df3b"
df_train.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="iy77qSeCo2WY" outputId="c4ce1748-6c39-44f6-c094-476873135fdb"
df_train.shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="sOSEDMQPo2Tq" outputId="4327e4a5-b01b-4d03-c829-0220e7c8b36c"
df_valid.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="2uiymy6po2Lo" outputId="f1ebfdf2-a139-46d4-d8e7-850701cb57a2"
df_valid.shape
```

<!-- #region id="y3QSDyZj61Iy" -->
Matrix factorization model
<!-- #endregion -->

```python id="HBPnUZl-6z1g"
class MF(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100):
        super(MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.user_emb.weight.data.uniform_(0, 0.05)
        self.item_emb.weight.data.uniform_(0, 0.05)
        
    def forward(self, u, v):
        u = self.user_emb(u)
        v = self.item_emb(v)
        return (u*v).sum(1)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="dBQhfy2l7AAn" outputId="d667d3a3-2baa-467b-e8ab-905c3282780f"
# unit testing the architecture
sample = encode_data(train.sample(5))
display(sample)
```

```python id="9tmmtDTuqnIB"
num_users = 5
num_items = 5
emb_size = 3

user_emb = nn.Embedding(num_users, emb_size)
item_emb = nn.Embedding(num_items, emb_size)

users = torch.LongTensor(sample.userId.values)
items = torch.LongTensor(sample.movieId.values)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 102} id="H557JwqSqimK" outputId="b45ad07e-0cf7-4cb0-f9ad-5de2f8a86c08"
U = user_emb(users)
display(U)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 102} id="IsdtlmtBq3cj" outputId="418f529d-1481-475e-ad2c-39578baa4cdf"
V = item_emb(items)
display(V)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 102} id="TKtMvkNfq0q2" outputId="c7bb19a7-3a7b-45db-f4c1-b95f0994750f"
display(U*V) # element wise multiplication
```

```python colab={"base_uri": "https://localhost:8080/", "height": 34} id="5_AN_dhQq0nE" outputId="7cc9d053-c9f3-49d0-e2a6-f0ea809ecd8f"
display((U*V).sum(1))
```

<!-- #region id="W01e58dr86WY" -->
Model training
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="VC5vARcP7QAc" outputId="b4cea803-bbac-4301-bb1c-12683689948d"
num_users = len(df_train.userId.unique())
num_items = len(df_train.movieId.unique())
print("{} users and {} items in the training set".format(num_users, num_items))
```

```python id="yRi5sy-K8-fr"
model = MF(num_users, num_items, emb_size=100) # .cuda() if you have a GPU
```

```python id="4nAGJ4l08_83"
def train_epocs(model, epochs=10, lr=0.01, wd=0.0, unsqueeze=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model.train()
    for i in range(epochs):
        users = torch.LongTensor(df_train.userId.values) # .cuda()
        items = torch.LongTensor(df_train.movieId.values) #.cuda()
        ratings = torch.FloatTensor(df_train.rating.values) #.cuda()
        if unsqueeze:
            ratings = ratings.unsqueeze(1)
        y_hat = model(users, items)
        loss = F.mse_loss(y_hat, ratings)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item()) 
    test_loss(model, unsqueeze)
```

```python id="7l_3G5gn9GH3"
def test_loss(model, unsqueeze=False):
    model.eval()
    users = torch.LongTensor(df_valid.userId.values) #.cuda()
    items = torch.LongTensor(df_valid.movieId.values) #.cuda()
    ratings = torch.FloatTensor(df_valid.rating.values) #.cuda()
    if unsqueeze:
        ratings = ratings.unsqueeze(1)
    y_hat = model(users, items)
    loss = F.mse_loss(y_hat, ratings)
    print("test loss %.3f " % loss.item())
```

```python colab={"base_uri": "https://localhost:8080/"} id="EztQtZKl9M53" outputId="25713f3c-edff-4333-e45b-ffb2c79375fb"
train_epocs(model, epochs=10, lr=0.1)
```

```python colab={"base_uri": "https://localhost:8080/"} id="AoSgUhWV9O1q" outputId="48e887fa-b55c-4465-e2d0-05789b5a7419"
train_epocs(model, epochs=10, lr=0.01)
```

```python colab={"base_uri": "https://localhost:8080/"} id="erRnsApY9Q7e" outputId="1e6a68f3-13b1-4963-b187-5d1b1bc3e438"
train_epocs(model, epochs=10, lr=0.01)
```

<!-- #region id="HAwA9Rts9UI1" -->
MF with bias
<!-- #endregion -->

```python id="Dur1n3lo9S3C"
class MF_bias(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100):
        super(MF_bias, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.item_bias = nn.Embedding(num_items, 1)
        self.user_emb.weight.data.uniform_(0,0.05)
        self.item_emb.weight.data.uniform_(0,0.05)
        self.user_bias.weight.data.uniform_(-0.01,0.01)
        self.item_bias.weight.data.uniform_(-0.01,0.01)
        
    def forward(self, u, v):
        U = self.user_emb(u)
        V = self.item_emb(v)
        b_u = self.user_bias(u).squeeze()
        b_v = self.item_bias(v).squeeze()
        return (U*V).sum(1) +  b_u  + b_v
```

```python id="WAyaietL9ZAq"
model = MF_bias(num_users, num_items, emb_size=100) #.cuda()
```

```python colab={"base_uri": "https://localhost:8080/"} id="5nEO-IVp9acn" outputId="773ee576-ea2e-40ac-97c7-7d78606595e1"
train_epocs(model, epochs=10, lr=0.05, wd=1e-5)
```

```python colab={"base_uri": "https://localhost:8080/"} id="nD2fD5A59cK4" outputId="03df230c-70ff-4767-e088-928d54f6cd37"
train_epocs(model, epochs=10, lr=0.01, wd=1e-5)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Os53hZxr9e_T" outputId="d9bf93d3-8411-4535-dcc7-ebcf4a3fbc48"
train_epocs(model, epochs=10, lr=0.001, wd=1e-5)
```

<!-- #region id="-XhFy6bU9h48" -->
Note that these models are susceptible to weight initialization, optimization algorithm and regularization.


<!-- #endregion -->

<!-- #region id="NugoowzF9kCk" -->
### Neural Network Model
Note here there is no matrix multiplication, we could potentially make the embeddings of different sizes. Here we could get better results by keep playing with regularization.
<!-- #endregion -->

```python id="qLVWHOxQ9fVX"
class CollabFNet(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100, n_hidden=10):
        super(CollabFNet, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.lin1 = nn.Linear(emb_size*2, n_hidden)
        self.lin2 = nn.Linear(n_hidden, 1)
        self.drop1 = nn.Dropout(0.1)
        
    def forward(self, u, v):
        U = self.user_emb(u)
        V = self.item_emb(v)
        x = F.relu(torch.cat([U, V], dim=1))
        x = self.drop1(x)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x
```

```python id="ljjju7Yy9x7b"
model = CollabFNet(num_users, num_items, emb_size=100) #.cuda()
```

```python colab={"base_uri": "https://localhost:8080/"} id="YuG2Hz5e9yyl" outputId="67b716ed-84e4-4dcf-eaf0-f06609542d0d"
train_epocs(model, epochs=15, lr=0.05, wd=1e-6, unsqueeze=True)
```

```python colab={"base_uri": "https://localhost:8080/"} id="JYyXb1qO90vb" outputId="b3bcc463-51a8-4625-f547-38ebbf105a7f"
train_epocs(model, epochs=10, lr=0.001, wd=1e-6, unsqueeze=True)
```

```python colab={"base_uri": "https://localhost:8080/"} id="o0d-WRvW92h-" outputId="20f81203-7fbb-44db-b421-c6c20239cd22"
train_epocs(model, epochs=10, lr=0.001, wd=1e-6, unsqueeze=True)
```

<!-- #region id="6fpmHCaYAn2d" -->
### Neural network model - different approach
> youtube: https://youtu.be/MVB1cbe923A
<!-- #endregion -->

<!-- #region id="SMuCwuPWPfGz" -->
### Ethan Rosenthal

Ref - https://github.com/EthanRosenthal/torchmf
<!-- #endregion -->

```python id="J31f-camBorB"
import os
import requests
import zipfile
import collections

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

import torch
from torch import nn
import torch.multiprocessing as mp
import torch.utils.data as data
from tqdm import tqdm
```

```python id="ahu8EWCGQJkI"
def _get_data_path():
    """
    Get path to the movielens dataset file.
    """
    data_path = '/content/data'
    if not os.path.exists(data_path):
        print('Making data path')
        os.mkdir(data_path)
    return data_path


def _download_movielens(dest_path):
    """
    Download the dataset.
    """

    url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
    req = requests.get(url, stream=True)

    print('Downloading MovieLens data')

    with open(os.path.join(dest_path, 'ml-100k.zip'), 'wb') as fd:
        for chunk in req.iter_content(chunk_size=None):
            fd.write(chunk)

    with zipfile.ZipFile(os.path.join(dest_path, 'ml-100k.zip'), 'r') as z:
        z.extractall(dest_path)
```

```python id="DouK7x1nPsNb"
def read_movielens_df():
    path = _get_data_path()
    zipfile = os.path.join(path, 'ml-100k.zip')
    if not os.path.isfile(zipfile):
        _download_movielens(path)
    fname = os.path.join(path, 'ml-100k', 'u.data')
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(fname, sep='\t', names=names)
    return df


def get_movielens_interactions():
    df = read_movielens_df()

    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]

    interactions = np.zeros((n_users, n_items))
    for row in df.itertuples():
        interactions[row[1] - 1, row[2] - 1] = row[3]
    return interactions


def train_test_split(interactions, n=10):
    """
    Split an interactions matrix into training and test sets.
    Parameters
    ----------
    interactions : np.ndarray
    n : int (default=10)
        Number of items to select / row to place into test.

    Returns
    -------
    train : np.ndarray
    test : np.ndarray
    """
    test = np.zeros(interactions.shape)
    train = interactions.copy()
    for user in range(interactions.shape[0]):
        if interactions[user, :].nonzero()[0].shape[0] > n:
            test_interactions = np.random.choice(interactions[user, :].nonzero()[0],
                                                 size=n,
                                                 replace=False)
            train[user, test_interactions] = 0.
            test[user, test_interactions] = interactions[user, test_interactions]

    # Test and training are truly disjoint
    assert(np.all((train * test) == 0))
    return train, test


def get_movielens_train_test_split(implicit=False):
    interactions = get_movielens_interactions()
    if implicit:
        interactions = (interactions >= 4).astype(np.float32)
    train, test = train_test_split(interactions)
    train = sp.coo_matrix(train)
    test = sp.coo_matrix(test)
    return train, test
```

```python colab={"base_uri": "https://localhost:8080/"} id="0x9xMyW6PsK6" outputId="00096a84-7542-4d07-920d-48830410244e"
%%writefile metrics.py

import numpy as np
from sklearn.metrics import roc_auc_score
from torch import multiprocessing as mp
import torch

def get_row_indices(row, interactions):
    start = interactions.indptr[row]
    end = interactions.indptr[row + 1]
    return interactions.indices[start:end]


def auc(model, interactions, num_workers=1):
    aucs = []
    processes = []
    n_users = interactions.shape[0]
    mp_batch = int(np.ceil(n_users / num_workers))

    queue = mp.Queue()
    rows = np.arange(n_users)
    np.random.shuffle(rows)
    for rank in range(num_workers):
        start = rank * mp_batch
        end = np.min((start + mp_batch,  n_users))
        p = mp.Process(target=batch_auc,
                       args=(queue, rows[start:end], interactions, model))
        p.start()
        processes.append(p)

    while True:
        is_alive = False
        for p in processes:
            if p.is_alive():
                is_alive = True
                break
        if not is_alive and queue.empty():
            break

        while not queue.empty():
            aucs.append(queue.get())

    queue.close()
    for p in processes:
        p.join()
    return np.mean(aucs)


def batch_auc(queue, rows, interactions, model):
    n_items = interactions.shape[1]
    items = torch.arange(0, n_items).long()
    users_init = torch.ones(n_items).long()
    for row in rows:
        row = int(row)
        users = users_init.fill_(row)

        preds = model.predict(users, items)
        actuals = get_row_indices(row, interactions)

        if len(actuals) == 0:
            continue
        y_test = np.zeros(n_items)
        y_test[actuals] = 1
        queue.put(roc_auc_score(y_test, preds.data.numpy()))


def patk(model, interactions, num_workers=1, k=5):
    patks = []
    processes = []
    n_users = interactions.shape[0]
    mp_batch = int(np.ceil(n_users / num_workers))

    queue = mp.Queue()
    rows = np.arange(n_users)
    np.random.shuffle(rows)
    for rank in range(num_workers):
        start = rank * mp_batch
        end = np.min((start + mp_batch, n_users))
        p = mp.Process(target=batch_patk,
                       args=(queue, rows[start:end], interactions, model),
                       kwargs={'k': k})
        p.start()
        processes.append(p)

    while True:
        is_alive = False
        for p in processes:
            if p.is_alive():
                is_alive = True
                break
        if not is_alive and queue.empty():
            break

        while not queue.empty():
            patks.append(queue.get())

    queue.close()
    for p in processes:
        p.join()
    return np.mean(patks)


def batch_patk(queue, rows, interactions, model, k=5):
    n_items = interactions.shape[1]

    items = torch.arange(0, n_items).long()
    users_init = torch.ones(n_items).long()
    for row in rows:
        row = int(row)
        users = users_init.fill_(row)

        preds = model.predict(users, items)
        actuals = get_row_indices(row, interactions)

        if len(actuals) == 0:
            continue

        top_k = np.argpartition(-np.squeeze(preds.data.numpy()), k)
        top_k = set(top_k[:k])
        true_pids = set(actuals)
        if true_pids:
            queue.put(len(top_k & true_pids) / float(k))
```

```python colab={"base_uri": "https://localhost:8080/"} id="I6IqWKWFhAQh" outputId="3bef0869-34b5-488c-ede7-bf9be08c115f"
import metrics
import importlib
importlib.reload(metrics)
```

```python id="UEhEE_GhPsH2"
class Interactions(data.Dataset):
    """
    Hold data in the form of an interactions matrix.
    Typical use-case is like a ratings matrix:
    - Users are the rows
    - Items are the columns
    - Elements of the matrix are the ratings given by a user for an item.
    """

    def __init__(self, mat):
        self.mat = mat.astype(np.float32).tocoo()
        self.n_users = self.mat.shape[0]
        self.n_items = self.mat.shape[1]

    def __getitem__(self, index):
        row = self.mat.row[index]
        col = self.mat.col[index]
        val = self.mat.data[index]
        return (row, col), val

    def __len__(self):
        return self.mat.nnz


class PairwiseInteractions(data.Dataset):
    """
    Sample data from an interactions matrix in a pairwise fashion. The row is
    treated as the main dimension, and the columns are sampled pairwise.
    """

    def __init__(self, mat):
        self.mat = mat.astype(np.float32).tocoo()

        self.n_users = self.mat.shape[0]
        self.n_items = self.mat.shape[1]

        self.mat_csr = self.mat.tocsr()
        if not self.mat_csr.has_sorted_indices:
            self.mat_csr.sort_indices()

    def __getitem__(self, index):
        row = self.mat.row[index]
        found = False

        while not found:
            neg_col = np.random.randint(self.n_items)
            if self.not_rated(row, neg_col, self.mat_csr.indptr,
                              self.mat_csr.indices):
                found = True

        pos_col = self.mat.col[index]
        val = self.mat.data[index]

        return (row, (pos_col, neg_col)), val

    def __len__(self):
        return self.mat.nnz

    @staticmethod
    def not_rated(row, col, indptr, indices):
        # similar to use of bsearch in lightfm
        start = indptr[row]
        end = indptr[row + 1]
        searched = np.searchsorted(indices[start:end], col, 'right')
        if searched >= (end - start):
            # After the array
            return False
        return col != indices[searched]  # Not found

    def get_row_indices(self, row):
        start = self.mat_csr.indptr[row]
        end = self.mat_csr.indptr[row + 1]
        return self.mat_csr.indices[start:end]


class BaseModule(nn.Module):
    """
    Base module for explicit matrix factorization.
    """
    
    def __init__(self,
                 n_users,
                 n_items,
                 n_factors=40,
                 dropout_p=0,
                 sparse=False):
        """

        Parameters
        ----------
        n_users : int
            Number of users
        n_items : int
            Number of items
        n_factors : int
            Number of latent factors (or embeddings or whatever you want to
            call it).
        dropout_p : float
            p in nn.Dropout module. Probability of dropout.
        sparse : bool
            Whether or not to treat embeddings as sparse. NOTE: cannot use
            weight decay on the optimizer if sparse=True. Also, can only use
            Adagrad.
        """
        super(BaseModule, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.user_biases = nn.Embedding(n_users, 1, sparse=sparse)
        self.item_biases = nn.Embedding(n_items, 1, sparse=sparse)
        self.user_embeddings = nn.Embedding(n_users, n_factors, sparse=sparse)
        self.item_embeddings = nn.Embedding(n_items, n_factors, sparse=sparse)
        
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=self.dropout_p)

        self.sparse = sparse
        
    def forward(self, users, items):
        """
        Forward pass through the model. For a single user and item, this
        looks like:

        user_bias + item_bias + user_embeddings.dot(item_embeddings)

        Parameters
        ----------
        users : np.ndarray
            Array of user indices
        items : np.ndarray
            Array of item indices

        Returns
        -------
        preds : np.ndarray
            Predicted ratings.

        """
        ues = self.user_embeddings(users)
        uis = self.item_embeddings(items)

        preds = self.user_biases(users)
        preds += self.item_biases(items)
        preds += (self.dropout(ues) * self.dropout(uis)).sum(dim=1, keepdim=True)

        return preds.squeeze()
    
    def __call__(self, *args):
        return self.forward(*args)

    def predict(self, users, items):
        return self.forward(users, items)


def bpr_loss(preds, vals):
    sig = nn.Sigmoid()
    return (1.0 - sig(preds)).pow(2).sum()


class BPRModule(nn.Module):
    
    def __init__(self,
                 n_users,
                 n_items,
                 n_factors=40,
                 dropout_p=0,
                 sparse=False,
                 model=BaseModule):
        super(BPRModule, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.dropout_p = dropout_p
        self.sparse = sparse
        self.pred_model = model(
            self.n_users,
            self.n_items,
            n_factors=n_factors,
            dropout_p=dropout_p,
            sparse=sparse
        )

    def forward(self, users, items):
        assert isinstance(items, tuple), \
            'Must pass in items as (pos_items, neg_items)'
        # Unpack
        (pos_items, neg_items) = items
        pos_preds = self.pred_model(users, pos_items)
        neg_preds = self.pred_model(users, neg_items)
        return pos_preds - neg_preds

    def predict(self, users, items):
        return self.pred_model(users, items)


class BasePipeline:
    """
    Class defining a training pipeline. Instantiates data loaders, model,
    and optimizer. Handles training for multiple epochs and keeping track of
    train and test loss.
    """

    def __init__(self,
                 train,
                 test=None,
                 model=BaseModule,
                 n_factors=40,
                 batch_size=32,
                 dropout_p=0.02,
                 sparse=False,
                 lr=0.01,
                 weight_decay=0.,
                 optimizer=torch.optim.Adam,
                 loss_function=nn.MSELoss(reduction='sum'),
                 n_epochs=10,
                 verbose=False,
                 random_seed=None,
                 interaction_class=Interactions,
                 hogwild=False,
                 num_workers=0,
                 eval_metrics=None,
                 k=5):
        self.train = train
        self.test = test

        if hogwild:
            num_loader_workers = 0
        else:
            num_loader_workers = num_workers
        self.train_loader = data.DataLoader(
            interaction_class(train), batch_size=batch_size, shuffle=True,
            num_workers=num_loader_workers)
        if self.test is not None:
            self.test_loader = data.DataLoader(
                interaction_class(test), batch_size=batch_size, shuffle=True,
                num_workers=num_loader_workers)
        self.num_workers = num_workers
        self.n_users = self.train.shape[0]
        self.n_items = self.train.shape[1]
        self.n_factors = n_factors
        self.batch_size = batch_size
        self.dropout_p = dropout_p
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_function = loss_function
        self.n_epochs = n_epochs
        if sparse:
            assert weight_decay == 0.0
        self.model = model(self.n_users,
                           self.n_items,
                           n_factors=self.n_factors,
                           dropout_p=self.dropout_p,
                           sparse=sparse)
        self.optimizer = optimizer(self.model.parameters(),
                                   lr=self.lr,
                                   weight_decay=self.weight_decay)
        self.warm_start = False
        self.losses = collections.defaultdict(list)
        self.verbose = verbose
        self.hogwild = hogwild
        if random_seed is not None:
            if self.hogwild:
                random_seed += os.getpid()
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

        if eval_metrics is None:
            eval_metrics = []
        self.eval_metrics = eval_metrics
        self.k = k

    def break_grads(self):
        for param in self.model.parameters():
            # Break gradient sharing
            if param.grad is not None:
                param.grad.data = param.grad.data.clone()

    def fit(self):
        for epoch in range(1, self.n_epochs + 1):

            if self.hogwild:
                self.model.share_memory()
                processes = []
                train_losses = []
                queue = mp.Queue()
                for rank in range(self.num_workers):
                    p = mp.Process(target=self._fit_epoch,
                                   kwargs={'epoch': epoch,
                                           'queue': queue})
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()

                while True:
                    is_alive = False
                    for p in processes:
                        if p.is_alive():
                            is_alive = True
                            break
                    if not is_alive and queue.empty():
                        break

                    while not queue.empty():
                        train_losses.append(queue.get())
                queue.close()
                train_loss = np.mean(train_losses)
            else:
                train_loss = self._fit_epoch(epoch)

            self.losses['train'].append(train_loss)
            row = 'Epoch: {0:^3}  train: {1:^10.5f}'.format(epoch, self.losses['train'][-1])
            if self.test is not None:
                self.losses['test'].append(self._validation_loss())
                row += 'val: {0:^10.5f}'.format(self.losses['test'][-1])
                for metric in self.eval_metrics:
                    func = getattr(metrics, metric)
                    res = func(self.model, self.test_loader.dataset.mat_csr,
                               num_workers=self.num_workers)
                    self.losses['eval-{}'.format(metric)].append(res)
                    row += 'eval-{0}: {1:^10.5f}'.format(metric, res)
            self.losses['epoch'].append(epoch)
            if self.verbose:
                print(row)

    def _fit_epoch(self, epoch=1, queue=None):
        if self.hogwild:
            self.break_grads()

        self.model.train()
        total_loss = torch.Tensor([0])
        pbar = tqdm(enumerate(self.train_loader),
                    total=len(self.train_loader),
                    desc='({0:^3})'.format(epoch))
        for batch_idx, ((row, col), val) in pbar:
            self.optimizer.zero_grad()

            row = row.long()
            # TODO: turn this into a collate_fn like the data_loader
            if isinstance(col, list):
                col = tuple(c.long() for c in col)
            else:
                col = col.long()
            val = val.float()

            preds = self.model(row, col)
            loss = self.loss_function(preds, val)
            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()
            batch_loss = loss.item() / row.size()[0]
            pbar.set_postfix(train_loss=batch_loss)
        total_loss /= self.train.nnz
        if queue is not None:
            queue.put(total_loss[0])
        else:
            return total_loss[0]

    def _validation_loss(self):
        self.model.eval()
        total_loss = torch.Tensor([0])
        for batch_idx, ((row, col), val) in enumerate(self.test_loader):
            row = row.long()
            if isinstance(col, list):
                col = tuple(c.long() for c in col)
            else:
                col = col.long()
            val = val.float()

            preds = self.model(row, col)
            loss = self.loss_function(preds, val)
            total_loss += loss.item()

        total_loss /= self.test.nnz
        return total_loss[0]
```

```python id="iKeUFiRNPsFY"
def explicit():
    train, test = get_movielens_train_test_split()
    pipeline = BasePipeline(train, test=test, model=BaseModule,
                            n_factors=10, batch_size=1024, dropout_p=0.02,
                            lr=0.02, weight_decay=0.1,
                            optimizer=torch.optim.Adam, n_epochs=40,
                            verbose=True, random_seed=2017)
    pipeline.fit()


def implicit():
    train, test = get_movielens_train_test_split(implicit=True)

    pipeline = BasePipeline(train, test=test, verbose=True,
                           batch_size=1024, num_workers=4,
                           n_factors=20, weight_decay=0,
                           dropout_p=0., lr=.2, sparse=True,
                           optimizer=torch.optim.SGD, n_epochs=40,
                           random_seed=2017, loss_function=bpr_loss,
                           model=BPRModule,
                           interaction_class=PairwiseInteractions,
                           eval_metrics=('auc', 'patk'))
    pipeline.fit()


def hogwild():
    train, test = get_movielens_train_test_split(implicit=True)

    pipeline = BasePipeline(train, test=test, verbose=True,
                            batch_size=1024, num_workers=4,
                            n_factors=20, weight_decay=0,
                            dropout_p=0., lr=.2, sparse=True,
                            optimizer=torch.optim.SGD, n_epochs=40,
                            random_seed=2017, loss_function=bpr_loss,
                            model=BPRModule, hogwild=True,
                            interaction_class=PairwiseInteractions,
                            eval_metrics=('auc', 'patk'))
    pipeline.fit()
```

```python colab={"base_uri": "https://localhost:8080/"} id="dPQaj1FjPsCo" outputId="42275f83-f4e9-43cc-a105-3ecde82efaa4"
explicit()
```

```python colab={"base_uri": "https://localhost:8080/"} id="mtI0DewsPr_0" outputId="6cc428e1-211f-4e7e-8264-e8332ad47e8b"
implicit()
```
