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

<!-- #region id="5uDaQrFWMDif" -->
# Matrix Factorization based Movie Recommender built in PyTorch
> Simple PyTorch based Matrix Factorization models on movielens-100k dataset - implicit, explicit and hogwild variant

- toc: true
- badges: true
- comments: true
- categories: [PyTorch, Movie, MF, Factorization]
- author: "<a href='https://github.com/EthanRosenthal/torchmf'>Ethan Rosenthal</a>"
- image:
<!-- #endregion -->

<!-- #region id="2jqJNaQhK8ji" -->
## utils.py
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="vsby4vwWGlWJ" outputId="35f3b79f-4242-426d-9ace-10adc6f5fb69"
%%writefile utils.py

import os
import requests
import zipfile

import numpy as np
import pandas as pd
import scipy.sparse as sp

"""
Shamelessly stolen from
https://github.com/maciejkula/triplet_recommendations_keras
"""


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


def _get_data_path():
    """
    Get path to the movielens dataset file.
    """
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'data')
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


def get_movielens_train_test_split(implicit=False):
    interactions = get_movielens_interactions()
    if implicit:
        interactions = (interactions >= 4).astype(np.float32)
    train, test = train_test_split(interactions)
    train = sp.coo_matrix(train)
    test = sp.coo_matrix(test)
    return train, test
```

<!-- #region id="d9wghzkqLR2i" -->
## metrics.py
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="EWmOo0AqKiEN" outputId="4362643e-6e76-4913-9e5d-75afa8ec3753"
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

<!-- #region id="7N1Rl15-LJAj" -->
## torchmf.py
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="NUlsa6LeLKqu" outputId="1c82c38f-467e-4bfc-9280-4a053e3d924a"
%%writefile torchmf.py

import collections
import os

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
import torch.multiprocessing as mp
import torch.utils.data as data
from tqdm import tqdm

import metrics


# Models
# Interactions Dataset => Singular Iter => Singular Loss
# Pairwise Datasets => Pairwise Iter => Pairwise Loss
# Pairwise Iters
# Loss Functions
# Optimizers
# Metric callbacks

# Serve up users, items (and items could be pos_items, neg_items)
# In this case, the iteration remains the same. Pass both items into a model
# which is a concat of the base model. it handles the pos and neg_items
# accordingly. define the loss after.


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

<!-- #region id="5KpaDgMwLNI5" -->
## run.py
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="HxKI9a2dLDDy" outputId="d7b903c2-6adb-4afd-dcd3-f1e5d49af5f6"
%%writefile run.py

import argparse
import pickle

import torch

from torchmf import (BaseModule, BPRModule, BasePipeline,
                     bpr_loss, PairwiseInteractions)
import utils


def explicit():
    train, test = utils.get_movielens_train_test_split()
    pipeline = BasePipeline(train, test=test, model=BaseModule,
                            n_factors=10, batch_size=1024, dropout_p=0.02,
                            lr=0.02, weight_decay=0.1,
                            optimizer=torch.optim.Adam, n_epochs=40,
                            verbose=True, random_seed=2017)
    pipeline.fit()


def implicit():
    train, test = utils.get_movielens_train_test_split(implicit=True)

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
    train, test = utils.get_movielens_train_test_split(implicit=True)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='torchmf')
    parser.add_argument('--example',
                        help='explicit, implicit, or hogwild')
    args = parser.parse_args()
    if args.example == 'explicit':
        explicit()
    elif args.example == 'implicit':
        implicit()
    elif args.example == 'hogwild':
        hogwild()
    else:
        print('example must be explicit, implicit, or hogwild')
```

<!-- #region id="40lNybzWLtRP" -->
## explicit run
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="0i4BoW9HLHSb" outputId="e648ac4f-928a-4d50-ab8a-3d3015ba82a3"
!python run.py --example explicit
```

<!-- #region id="JxqWyDE4LxPb" -->
## implicit
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="IBXdHOUXLdPE" outputId="62bd667d-56d6-4969-e390-8cab4842353e"
!python run.py --example implicit
```
