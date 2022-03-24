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

<!-- #region id="tbXR1BvPWXKO" -->
# Neural Graph Collaborative Filtering on MovieLens
> Applying NGCF PyTorch version on Movielens-100k

- toc: true
- badges: true
- comments: true
- categories: [Graph, PyTorch, Movie]
- author: "<a href='https://github.com/metahexane/ngcf_pytorch_g61'>Muhammed Imran Özyar</a>"
- image:
<!-- #endregion -->

<!-- #region id="sG-h_5yQQEvQ" -->
## Libraries
<!-- #endregion -->

```python id="3QJEhOlDwIR7"
!pip install -q git+https://github.com/sparsh-ai/recochef
```

```python id="08tq9wC8pdD1"
import os
import csv 
import argparse
import numpy as np
import pandas as pd
import random as rd
from time import time
from pathlib import Path
import scipy.sparse as sp
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F

from recochef.preprocessing.split import chrono_split
```

```python id="F_-NBXzHS0_o"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)
```

<!-- #region id="uuM8Q9GlP8RN" -->
## Data Loading

The MovieLens 100K data set consists of 100,000 ratings from 1000 users on 1700 movies as described on [their website](https://grouplens.org/datasets/movielens/100k/).
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Q-yPokpipXsY" outputId="04f04704-9844-43e3-abe9-4d36ec2294cd"
!wget http://files.grouplens.org/datasets/movielens/ml-100k.zip
```

```python colab={"base_uri": "https://localhost:8080/"} id="6CLWhSTXpbvW" outputId="e112db0d-7151-4049-ade2-7325f4d93e58"
!unzip ml-100k.zip
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="KS9OJY75pgwq" outputId="0c1ec91e-2ef6-4b5a-e613-fa4501e2737f"
df = pd.read_csv('ml-100k/u.data', sep='\t', header=None, names=['USERID','ITEMID','RATING','TIMESTAMP'])
df.head()
```

<!-- #region id="Mo6kTBolQYca" -->
## Train/Test Split

We split the data chronologically in 80:20 ratio. Validated the split for user 4.
<!-- #endregion -->

```python id="XTJpJj2uvpD2"
df_train, df_test = chrono_split(df, ratio=0.8)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="Dpeu1H5xxXcR" outputId="d94c244d-ad25-47c1-d084-e51f6b015645"
userid = 4

query = "USERID==@userid"
display(df.query(query))
display(df_train.query(query))
display(df_test.query(query))
```

<!-- #region id="qny2aVoWQknP" -->
## Preprocessing

1. Sort by User ID and Timestamp
2. Label encode user and item id - in this case, already label encoded starting from 1, so decreasing ids by 1 as a proxy for label encode
3. Remove Timestamp and Rating column. The reason is that we are training a recall-maximing model where the objective is to correctly retrieve the items that users can interact with. We can select a rating threshold also
4. Convert Item IDs into list format
5. Store as a space-seperated txt file
<!-- #endregion -->

```python id="1iSOiyCqpmYE"
def preprocess(data):
  data = data.copy()
  data = data.sort_values(by=['USERID','TIMESTAMP'])
  data['USERID'] = data['USERID'] - 1
  data['ITEMID'] = data['ITEMID'] - 1
  data.drop(['TIMESTAMP','RATING'], axis=1, inplace=True)
  data = data.groupby('USERID')['ITEMID'].apply(list).reset_index(name='ITEMID')
  return data
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="D7ZtrUPp22dO" outputId="56290e63-dcf3-448b-b3a5-ce60bd2c23db"
preprocess(df_train).head()
```

```python id="yDMAhrig1Lde"
def store(data, target_file='./data/movielens/train.txt'):
  Path(target_file).parent.mkdir(parents=True, exist_ok=True)
  with open(target_file, 'w+') as f:
    writer = csv.writer(f, delimiter=' ')
    for USERID, row in zip(data.USERID.values,data.ITEMID.values):
      row = [USERID] + row
      writer.writerow(row)
```

```python id="XUIFrKsavRzV"
store(preprocess(df_train), '/content/data/ml-100k/train.txt')
store(preprocess(df_test), '/content/data/ml-100k/test.txt')
```

```python colab={"base_uri": "https://localhost:8080/"} id="vq2IwCkJtUTy" outputId="bdd5df6b-213f-4e87-deae-b8f29e42ec87"
!head /content/data/ml-100k/train.txt
```

```python colab={"base_uri": "https://localhost:8080/"} id="E7YYuu2XuVQa" outputId="fd6fc81c-8ee7-4a34-cb7f-ae5c9ffb27d6"
!head /content/data/ml-100k/test.txt
```

```python id="C-f9mzEf4Ow6"
Path('/content/results').mkdir(parents=True, exist_ok=True)
```

```python id="Bxz7aV0Sws1S"
def parse_args():
    parser = argparse.ArgumentParser(description="Run NGCF.")
    parser.add_argument('--data_dir', type=str,
                        default='./data/',
                        help='Input data path.')
    parser.add_argument('--dataset', type=str, default='ml-100k',
                        help='Dataset name: Amazond-book, Gowella, ml-100k')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Store model to path.')
    parser.add_argument('--n_epochs', type=int, default=400,
                        help='Number of epoch.')
    parser.add_argument('--reg', type=float, default=1e-5,
                        help='l2 reg.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='number of embeddings.')
    parser.add_argument('--layers', type=str, default='[64,64]',
                        help='Output sizes of every layer')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size.')
    parser.add_argument('--node_dropout', type=float, default=0.,
                        help='Graph Node dropout.')
    parser.add_argument('--mess_dropout', type=float, default=0.1,
                        help='Message dropout.')
    parser.add_argument('--k', type=str, default=20,
                        help='k order of metric evaluation (e.g. NDCG@k)')
    parser.add_argument('--eval_N', type=int, default=5,
                        help='Evaluate every N epochs')
    parser.add_argument('--save_results', type=int, default=1,
                        help='Save model and results')

    return parser.parse_args(args={})
```

<!-- #region id="twi1ZIucR0ga" -->
## Helper Functions

- early_stopping()
- train()
- split_matrix()
- ndcg_k()
- eval_model
<!-- #endregion -->

<!-- #region id="aCShFbsCTPzw" -->
### Early Stopping
Premature stopping is applied if *recall@20* on the test set does not increase for 5 successive epochs.
<!-- #endregion -->

```python id="tHVTudWxTVZo"
def early_stopping(log_value, best_value, stopping_step, flag_step, expected_order='asc'):
    """
    Check if early_stopping is needed
    Function copied from original code
    """
    assert expected_order in ['asc', 'des']
    if (expected_order == 'asc' and log_value >= best_value) or (expected_order == 'des' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False

    return best_value, stopping_step, should_stop
```

```python id="6JEG5Jlpw3Nw"
def train(model, data_generator, optimizer):
    """
    Train the model PyTorch style
    Arguments:
    ---------
    model: PyTorch model
    data_generator: Data object
    optimizer: PyTorch optimizer
    """
    model.train()
    n_batch = data_generator.n_train // data_generator.batch_size + 1
    running_loss=0
    for _ in range(n_batch):
        u, i, j = data_generator.sample()
        optimizer.zero_grad()
        loss = model(u,i,j)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss

def split_matrix(X, n_splits=100):
    """
    Split a matrix/Tensor into n_folds (for the user embeddings and the R matrices)
    Arguments:
    ---------
    X: matrix to be split
    n_folds: number of folds
    Returns:
    -------
    splits: split matrices
    """
    splits = []
    chunk_size = X.shape[0] // n_splits
    for i in range(n_splits):
        start = i * chunk_size
        end = X.shape[0] if i == n_splits - 1 else (i + 1) * chunk_size
        splits.append(X[start:end])
    return splits

def compute_ndcg_k(pred_items, test_items, test_indices, k):
    """
    Compute NDCG@k
    
    Arguments:
    ---------
    pred_items: binary tensor with 1s in those locations corresponding to the predicted item interactions
    test_items: binary tensor with 1s in locations corresponding to the real test interactions
    test_indices: tensor with the location of the top-k predicted items
    k: k'th-order 
    Returns:
    -------
    NDCG@k
    """
    r = (test_items * pred_items).gather(1, test_indices)
    f = torch.from_numpy(np.log2(np.arange(2, k+2))).float().cuda()
    dcg = (r[:, :k]/f).sum(1)
    dcg_max = (torch.sort(r, dim=1, descending=True)[0][:, :k]/f).sum(1)
    ndcg = dcg/dcg_max
    ndcg[torch.isnan(ndcg)] = 0
    return ndcg
```

<!-- #region id="sx-Vzl2vTeWN" -->
### Eval Model

At every N epoch, the model is evaluated on the test set. From this evaluation, we compute the recall and normal discounted cumulative gain (ndcg) at the top-20 predictions. It is important to note that in order to evaluate the model on the test set we have to ‘unpack’ the sparse matrix (torch.sparse.todense()), and thus load a bunch of ‘zeros’ on memory. In order to prevent memory overload, we split the sparse matrices into 100 chunks, unpack the sparse chunks one by one, compute the metrics we need, and compute the mean value of all chunks.
<!-- #endregion -->

```python id="1dysqVKGTjm6"
def eval_model(u_emb, i_emb, Rtr, Rte, k):
    """
    Evaluate the model
    
    Arguments:
    ---------
    u_emb: User embeddings
    i_emb: Item embeddings
    Rtr: Sparse matrix with the training interactions
    Rte: Sparse matrix with the testing interactions
    k : kth-order for metrics
    
    Returns:
    --------
    result: Dictionary with lists correponding to the metrics at order k for k in Ks
    """
    # split matrices
    ue_splits = split_matrix(u_emb)
    tr_splits = split_matrix(Rtr)
    te_splits = split_matrix(Rte)

    recall_k, ndcg_k= [], []
    # compute results for split matrices
    for ue_f, tr_f, te_f in zip(ue_splits, tr_splits, te_splits):

        scores = torch.mm(ue_f, i_emb.t())

        test_items = torch.from_numpy(te_f.todense()).float().cuda()
        non_train_items = torch.from_numpy(1-(tr_f.todense())).float().cuda()
        scores = scores * non_train_items

        _, test_indices = torch.topk(scores, dim=1, k=k)

        # If you want to use a as the index in dim1 for t, this code should work:
        #t[torch.arange(t.size(0)), a]

        pred_items = torch.zeros_like(scores).float()
        # pred_items.scatter_(dim=1,index=test_indices,src=torch.tensor(1.0).cuda())
        pred_items.scatter_(dim=1,index=test_indices,src=torch.ones_like(test_indices, dtype=torch.float).cuda())

        topk_preds = torch.zeros_like(scores).float()
        # topk_preds.scatter_(dim=1,index=test_indices[:, :k],src=torch.tensor(1.0))
        _idx = test_indices[:, :k]
        topk_preds.scatter_(dim=1,index=_idx,src=torch.ones_like(_idx, dtype=torch.float))

        TP = (test_items * topk_preds).sum(1)
        rec = TP/test_items.sum(1)
        ndcg = compute_ndcg_k(pred_items, test_items, test_indices, k)

        recall_k.append(rec)
        ndcg_k.append(ndcg)

    return torch.cat(recall_k).mean(), torch.cat(ndcg_k).mean()
```

<!-- #region id="mvvKJOlpSLn4" -->
## Dataset Class
<!-- #endregion -->

<!-- #region id="AzoIqHHuUADD" -->
### Laplacian matrix

The components of the Laplacian matrix are as follows,

- **D**: a diagonal degree matrix, where D{t,t} is |N{t}|, which is the amount of first-hop neighbors for either item or user t,
- **R**: the user-item interaction matrix,
- **0**: an all-zero matrix,
- **A**: the adjacency matrix,

### Interaction and Adjacency Matrix

We create the sparse interaction matrix R, the adjacency matrix A, the degree matrix D, and the Laplacian matrix L, using the SciPy library. The adjacency matrix A is then transferred onto PyTorch tensor objects.
<!-- #endregion -->

```python id="s0w9GTdKw7Vj"
class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        #get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        self.exist_users = []

        # search train_file for max user_id/item_id
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    # first element is the user_id, rest are items
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    # item/user with highest number is number of items/users
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    # number of interactions
                    self.n_train += len(items)

        # search test_file for max item_id
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    if not items:
                        print("empyt test exists")
                        pass
                    else:
                        self.n_items = max(self.n_items, max(items))
                        self.n_test += len(items)
        # adjust counters: user_id/item_id starts at 0
        self.n_items += 1
        self.n_users += 1

        self.print_statistics()

        # create interactions/ratings matrix 'R' # dok = dictionary of keys
        print('Creating interaction matrices R_train and R_test...')
        t1 = time()
        self.R_train = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32) 
        self.R_test = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        self.train_items, self.test_set = {}, {}
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]
                    # enter 1 if user interacted with item
                    for i in train_items:
                        self.R_train[uid, i] = 1.
                    self.train_items[uid] = train_items

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue
                    uid, test_items = items[0], items[1:]
                    for i in test_items:
                        self.R_test[uid, i] = 1.0
                    self.test_set[uid] = test_items
        print('Complete. Interaction matrices R_train and R_test created in', time() - t1, 'sec')

    # if exist, get adjacency matrix
    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            print('Loaded adjacency-matrix (shape:', adj_mat.shape,') in', time() - t1, 'sec.')

        except Exception:
            print('Creating adjacency-matrix...')
            adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
        return adj_mat
    
    # create adjancency matrix
    def create_adj_mat(self):
        t1 = time()
        
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R_train.tolil() # to list of lists

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('Complete. Adjacency-matrix created in', adj_mat.shape, time() - t1, 'sec.')

        t2 = time()

        # normalize adjacency matrix
        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj).dot(d_mat_inv)
            return norm_adj.tocoo()

        print('Transforming adjacency-matrix to NGCF-adjacency matrix...')
        ngcf_adj_mat = normalized_adj_single(adj_mat) + sp.eye(adj_mat.shape[0])

        print('Complete. Transformed adjacency-matrix to NGCF-adjacency matrix in', time() - t2, 'sec.')
        return ngcf_adj_mat.tocsr()

    # create collections of N items that users never interacted with
    def negative_pool(self):
        t1 = time()
        for u in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            pools = [rd.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools
        print('refresh negative pools', time() - t1)

    # sample data for mini-batches
    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))
```

<!-- #region id="J2RxhIxmSYvl" -->
## NGCF Model
<!-- #endregion -->

<!-- #region id="P4vz1IOwTvED" -->
### Weight initialization

We then create tensors for the user embeddings and item embeddings with the proper dimensions. The weights are initialized using [Xavier uniform initialization](https://pytorch.org/docs/stable/nn.init.html).

For each layer, the weight matrices and corresponding biases are initialized using the same procedure.
<!-- #endregion -->

<!-- #region id="wo9vgNvJUWvR" -->
### Embedding Layer

The initial user and item embeddings are concatenated in an embedding lookup table as shown in the figure below. This embedding table is initialized using the user and item embeddings and will be optimized in an end-to-end fashion by the network.
<!-- #endregion -->

<!-- #region id="ntRUCJGMUeNf" -->
<!-- #endregion -->

<!-- #region id="HKqah7XFUhan" -->
### Embedding propagation

The embedding table is propagated through the network using the formula shown in the figure below.
<!-- #endregion -->

<!-- #region id="ZvrR6lvUUuIm" -->
<!-- #endregion -->

<!-- #region id="Co9D_oNXUlgj" -->
The components of the formula are as follows,

- **E⁽ˡ⁾**: the embedding table after l steps of embedding propagation, where E⁽⁰⁾ is the initial embedding table,
- **LeakyReLU**: the rectified linear unit used as activation function,
- **W**: the weights trained by the network,
- **I**: an identity matrix,
- **L**: the Laplacian matrix for the user-item graph, which is formulated as
<!-- #endregion -->

<!-- #region id="SLrIvZowUyyI" -->
<!-- #endregion -->

<!-- #region id="TcNXtDMFVLii" -->
### Architecture
<!-- #endregion -->

<!-- #region id="OBj_bkmxVNEf" -->
<!-- #endregion -->

```python id="4i1YYbJB4oGQ"
class NGCF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim, layers, reg, node_dropout, mess_dropout,
        adj_mtx):
        super().__init__()

        # initialize Class attributes
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.adj_mtx = adj_mtx
        self.laplacian = adj_mtx - sp.eye(adj_mtx.shape[0])
        self.reg = reg
        self.layers = layers
        self.n_layers = len(self.layers)
        self.node_dropout = node_dropout
        self.mess_dropout = mess_dropout

        #self.u_g_embeddings = nn.Parameter(torch.empty(n_users, emb_dim+np.sum(self.layers)))
        #self.i_g_embeddings = nn.Parameter(torch.empty(n_items, emb_dim+np.sum(self.layers)))

        # Initialize weights
        self.weight_dict = self._init_weights()
        print("Weights initialized.")

        # Create Matrix 'A', PyTorch sparse tensor of SP adjacency_mtx
        self.A = self._convert_sp_mat_to_sp_tensor(self.adj_mtx)
        self.L = self._convert_sp_mat_to_sp_tensor(self.laplacian)

    # initialize weights
    def _init_weights(self):
        print("Initializing weights...")
        weight_dict = nn.ParameterDict()

        initializer = torch.nn.init.xavier_uniform_
        
        weight_dict['user_embedding'] = nn.Parameter(initializer(torch.empty(self.n_users, self.emb_dim).to(device)))
        weight_dict['item_embedding'] = nn.Parameter(initializer(torch.empty(self.n_items, self.emb_dim).to(device)))

        weight_size_list = [self.emb_dim] + self.layers

        for k in range(self.n_layers):
            weight_dict['W_gc_%d' %k] = nn.Parameter(initializer(torch.empty(weight_size_list[k], weight_size_list[k+1]).to(device)))
            weight_dict['b_gc_%d' %k] = nn.Parameter(initializer(torch.empty(1, weight_size_list[k+1]).to(device)))
            
            weight_dict['W_bi_%d' %k] = nn.Parameter(initializer(torch.empty(weight_size_list[k], weight_size_list[k+1]).to(device)))
            weight_dict['b_bi_%d' %k] = nn.Parameter(initializer(torch.empty(1, weight_size_list[k+1]).to(device)))
           
        return weight_dict

    # convert sparse matrix into sparse PyTorch tensor
    def _convert_sp_mat_to_sp_tensor(self, X):
        """
        Convert scipy sparse matrix to PyTorch sparse matrix
        Arguments:
        ----------
        X = Adjacency matrix, scipy sparse matrix
        """
        coo = X.tocoo().astype(np.float32)
        i = torch.LongTensor(np.mat([coo.row, coo.col]))
        v = torch.FloatTensor(coo.data)
        res = torch.sparse.FloatTensor(i, v, coo.shape).to(device)
        return res

    # apply node_dropout
    def _droupout_sparse(self, X):
        """
        Drop individual locations in X
        
        Arguments:
        ---------
        X = adjacency matrix (PyTorch sparse tensor)
        dropout = fraction of nodes to drop
        noise_shape = number of non non-zero entries of X
        """
        
        node_dropout_mask = ((self.node_dropout) + torch.rand(X._nnz())).floor().bool().to(device)
        i = X.coalesce().indices()
        v = X.coalesce()._values()
        i[:,node_dropout_mask] = 0
        v[node_dropout_mask] = 0
        X_dropout = torch.sparse.FloatTensor(i, v, X.shape).to(X.device)

        return  X_dropout.mul(1/(1-self.node_dropout))

    def forward(self, u, i, j):
        """
        Computes the forward pass
        
        Arguments:
        ---------
        u = user
        i = positive item (user interacted with item)
        j = negative item (user did not interact with item)
        """
        # apply drop-out mask
        A_hat = self._droupout_sparse(self.A) if self.node_dropout > 0 else self.A
        L_hat = self._droupout_sparse(self.L) if self.node_dropout > 0 else self.L

        ego_embeddings = torch.cat([self.weight_dict['user_embedding'], self.weight_dict['item_embedding']], 0)

        all_embeddings = [ego_embeddings]

        # forward pass for 'n' propagation layers
        for k in range(self.n_layers):

            # weighted sum messages of neighbours
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            side_L_embeddings = torch.sparse.mm(L_hat, ego_embeddings)

            # transformed sum weighted sum messages of neighbours
            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) + self.weight_dict['b_gc_%d' % k]

            # bi messages of neighbours
            bi_embeddings = torch.mul(ego_embeddings, side_L_embeddings)
            # transformed bi messages of neighbours
            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) + self.weight_dict['b_bi_%d' % k]

            # non-linear activation 
            ego_embeddings = F.leaky_relu(sum_embeddings + bi_embeddings)
            # + message dropout
            mess_dropout_mask = nn.Dropout(self.mess_dropout)
            ego_embeddings = mess_dropout_mask(ego_embeddings)

            # normalize activation
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            all_embeddings.append(norm_embeddings)

        all_embeddings = torch.cat(all_embeddings, 1)
        
        # back to user/item dimension
        u_g_embeddings, i_g_embeddings = all_embeddings.split([self.n_users, self.n_items], 0)

        self.u_g_embeddings = nn.Parameter(u_g_embeddings)
        self.i_g_embeddings = nn.Parameter(i_g_embeddings)
        
        u_emb = u_g_embeddings[u] # user embeddings
        p_emb = i_g_embeddings[i] # positive item embeddings
        n_emb = i_g_embeddings[j] # negative item embeddings

        y_ui = torch.mul(u_emb, p_emb).sum(dim=1)
        y_uj = torch.mul(u_emb, n_emb).sum(dim=1)
        log_prob = (torch.log(torch.sigmoid(y_ui-y_uj))).mean()

        # compute bpr-loss
        bpr_loss = -log_prob
        if self.reg > 0.:
            l2norm = (torch.sum(u_emb**2)/2. + torch.sum(p_emb**2)/2. + torch.sum(n_emb**2)/2.) / u_emb.shape[0]
            l2reg  = self.reg*l2norm
            bpr_loss =  -log_prob + l2reg

        return bpr_loss
```

<!-- #region id="5xbcqHLUSowG" -->
## Training and Evaluation
<!-- #endregion -->

<!-- #region id="N6S7uZU3Tpht" -->
Training is done using the standard PyTorch method. If you are already familiar with PyTorch, the following code should look familiar.

One of the most useful functions of PyTorch is the torch.nn.Sequential() function, that takes existing and custom torch.nn modules. This makes it very easy to build and train complete networks. However, due to the nature of NCGF model structure, usage of torch.nn.Sequential() is not possible and the forward pass of the network has to be implemented ‘manually’. Using the Bayesian personalized ranking (BPR) pairwise loss, the forward pass is implemented as follows:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="LEMcstCz4vSm" outputId="4e06cd7c-8e69-4f6b-d4b8-054d373f01cb"
# read parsed arguments
args = parse_args()
data_dir = args.data_dir
dataset = args.dataset
batch_size = args.batch_size
layers = eval(args.layers)
emb_dim = args.emb_dim
lr = args.lr
reg = args.reg
mess_dropout = args.mess_dropout
node_dropout = args.node_dropout
k = args.k

# generate the NGCF-adjacency matrix
data_generator = Data(path=data_dir + dataset, batch_size=batch_size)
adj_mtx = data_generator.get_adj_mat()

# create model name and save
modelname =  "NGCF" + \
    "_bs_" + str(batch_size) + \
    "_nemb_" + str(emb_dim) + \
    "_layers_" + str(layers) + \
    "_nodedr_" + str(node_dropout) + \
    "_messdr_" + str(mess_dropout) + \
    "_reg_" + str(reg) + \
    "_lr_"  + str(lr)

# create NGCF model
model = NGCF(data_generator.n_users, 
              data_generator.n_items,
              emb_dim,
              layers,
              reg,
              node_dropout,
              mess_dropout,
              adj_mtx)
if use_cuda:
    model = model.cuda()

# current best metric
cur_best_metric = 0

# Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Set values for early stopping
cur_best_loss, stopping_step, should_stop = 1e3, 0, False
today = datetime.now()

print("Start at " + str(today))
print("Using " + str(device) + " for computations")
print("Params on CUDA: " + str(next(model.parameters()).is_cuda))

results = {"Epoch": [],
            "Loss": [],
            "Recall": [],
            "NDCG": [],
            "Training Time": []}

for epoch in range(args.n_epochs):

    t1 = time()
    loss = train(model, data_generator, optimizer)
    training_time = time()-t1
    print("Epoch: {}, Training time: {:.2f}s, Loss: {:.4f}".
        format(epoch, training_time, loss))

    # print test evaluation metrics every N epochs (provided by args.eval_N)
    if epoch % args.eval_N  == (args.eval_N - 1):
        with torch.no_grad():
            t2 = time()
            recall, ndcg = eval_model(model.u_g_embeddings.detach(),
                                      model.i_g_embeddings.detach(),
                                      data_generator.R_train,
                                      data_generator.R_test,
                                      k)
        print(
            "Evaluate current model:\n",
            "Epoch: {}, Validation time: {:.2f}s".format(epoch, time()-t2),"\n",
            "Loss: {:.4f}:".format(loss), "\n",
            "Recall@{}: {:.4f}".format(k, recall), "\n",
            "NDCG@{}: {:.4f}".format(k, ndcg)
            )

        cur_best_metric, stopping_step, should_stop = \
        early_stopping(recall, cur_best_metric, stopping_step, flag_step=5)

        # save results in dict
        results['Epoch'].append(epoch)
        results['Loss'].append(loss)
        results['Recall'].append(recall.item())
        results['NDCG'].append(ndcg.item())
        results['Training Time'].append(training_time)
    else:
        # save results in dict
        results['Epoch'].append(epoch)
        results['Loss'].append(loss)
        results['Recall'].append(None)
        results['NDCG'].append(None)
        results['Training Time'].append(training_time)

    if should_stop == True: break

# save
if args.save_results:
    date = today.strftime("%d%m%Y_%H%M")

    # save model as .pt file
    if os.path.isdir("./models"):
        torch.save(model.state_dict(), "./models/" + str(date) + "_" + modelname + "_" + dataset + ".pt")
    else:
        os.mkdir("./models")
        torch.save(model.state_dict(), "./models/" + str(date) + "_" + modelname + "_" + dataset + ".pt")

    # save results as pandas dataframe
    results_df = pd.DataFrame(results)
    results_df.set_index('Epoch', inplace=True)
    if os.path.isdir("./results"):
        results_df.to_csv("./results/" + str(date) + "_" + modelname + "_" + dataset + ".csv")
    else:
        os.mkdir("./results")
        results_df.to_csv("./results/" + str(date) + "_" + modelname + "_" + dataset + ".csv")
    # plot loss
    results_df['Loss'].plot(figsize=(12,8), title='Loss')
```

<!-- #region id="wBD0wM3JVXol" -->
## Appendix
<!-- #endregion -->

<!-- #region id="Iz0QI1P7VaDP" -->
### References
1. [https://medium.com/@yusufnoor_88274/implementing-neural-graph-collaborative-filtering-in-pytorch-4d021dff25f3](https://medium.com/@yusufnoor_88274/implementing-neural-graph-collaborative-filtering-in-pytorch-4d021dff25f3)
2. [https://github.com/xiangwang1223/neural_graph_collaborative_filtering](https://github.com/xiangwang1223/neural_graph_collaborative_filtering)
3. [https://arxiv.org/pdf/1905.08108.pdf](https://arxiv.org/pdf/1905.08108.pdf)
4. [https://github.com/metahexane/ngcf_pytorch_g61](https://github.com/metahexane/ngcf_pytorch_g61)
<!-- #endregion -->

<!-- #region id="_JywmGguVbNU" -->
### Next

Try out this notebook on the following datasets:
<!-- #endregion -->

<!-- #region id="MppKYNXJVswT" -->
Compare out the performance with these baselines:

1. MF: This is matrix factorization optimized by the Bayesian
personalized ranking (BPR) loss, which exploits the user-item
direct interactions only as the target value of interaction function.
2. NeuMF: The method is a state-of-the-art neural CF model
which uses multiple hidden layers above the element-wise and
concatenation of user and item embeddings to capture their nonlinear feature interactions. Especially, we employ two-layered
plain architecture, where the dimension of each hidden layer
keeps the same.
3. CMN: It is a state-of-the-art memory-based model, where
the user representation attentively combines the memory slots
of neighboring users via the memory layers. Note that the firstorder connections are used to find similar users who interacted
with the same items.
4. HOP-Rec: This is a state-of-the-art graph-based model,
where the high-order neighbors derived from random walks
are exploited to enrich the user-item interaction data.
5. PinSage: PinSage is designed to employ GraphSAGE
on item-item graph. In this work, we apply it on user-item interaction graph. Especially, we employ two graph convolution
layers, and the hidden dimension is set equal
to the embedding size.
6. GC-MC: This model adopts GCN encoder to generate
the representations for users and items, where only the first-order
neighbors are considered. Hence one graph convolution layer,
where the hidden dimension is set as the embedding size, is used.
<!-- #endregion -->
