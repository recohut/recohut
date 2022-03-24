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

<!-- #region id="STDCqK1APqNG" -->
# Training various Matrix Factorization models on ML-100k in PyTorch Lightning Framework
<!-- #endregion -->

<!-- #region id="KLntstJDLbA_" -->
**(Other) Tutorials**
1. MovieLens 1m Recommenders in PyTorch Lightning. [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jB5fczYDsF3qOJUXtBMgt-zt_7LyLGG_?usp=sharing)
2. BERT4Rec on ML-25m in PyTorch Lightning. [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/sparsh-ai/fb375c4a0d8c0c80b025a12c40dc2733/t595874-bert4rec-on-ml-25m-in-pytorch-lightning.ipynb)
3. Implicit Hybrid Movie Recommender using Collie Library. [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/sparsh-ai/290194a71f907f4544de1ca426a34246/t660394-implicit-hybrid-movie-recommender-using-collie-library.ipynb)
4. Training various Matrix Factorization models on ML-100k in PyTorch Lightning Framework. [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/recohut/notebooks/blob/main/nbs/P499733_Training_various_Matrix_Factorization_models_on_ML_100k_in_PyTorch_Lightning_Framework.ipynb)
<!-- #endregion -->

```python id="aB0dNVvVQ4HC"
!pip install -q pytorch_lightning
!pip install -q git+https://github.com/RecoHut-Projects/recohut.git
```

```python id="yBEuGDtpRIjc"
from abc import abstractmethod
from typing import Any, Iterable, List, Optional, Tuple, Union, Callable

from tqdm.notebook import tqdm
import sys
import os
from os import path as osp
from pathlib import Path
from collections import OrderedDict
import scipy.sparse as sparse
import collections
import random

import random
import numpy as np
import pandas as pd
from typing import Optional
from scipy.sparse import coo_matrix

import torch
from torch import nn
from torch.nn import Linear
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

from recohut.utils.common_utils import *
from recohut.datasets import base

import warnings
warnings.filterwarnings('ignore')
```

<!-- #region id="O80Wv6SaUKAm" -->
## Dataset
<!-- #endregion -->

```python cellView="code" id="Vu-9f1hzfWO3"
class ML1mDataset(torch.utils.data.Dataset, base.Dataset):
    url = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"

    def __init__(self, args, train=None):
        self.min_rating = args.min_rating
        self.min_uc = args.min_uc
        self.min_sc = args.min_sc
        self.num_negative_samples = args.num_negative_samples
        self.max_number_of_samples_to_consider = 200
        self.is_train = train

        super().__init__(args.data_dir)

        assert self.min_uc >= 2, 'Need at least 2 ratings per user for validation and test'

        self._process()

        if self.is_train is not None:
            self.load()

    @property
    def raw_file_names(self):
        return 'ratings.dat'

    @property
    def processed_file_names(self):
        return ['ml_1m_train.pt', 'ml_1m_test_pos.pt', 'ml_1m_test_neg.pt']

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        from shutil import move, rmtree
        move(osp.join(self.raw_dir, 'ml-1m', self.raw_file_names), self.raw_dir)
        rmtree(osp.join(self.raw_dir, 'ml-1m'))
        os.unlink(path)

    def make_implicit(self, df):
        "convert the explicit data to implicit by only keeping interactions with a rating >= min_rating"
        print('Turning into implicit ratings')
        df = df[df['rating'] >= self.min_rating].reset_index(drop=True)
        df['rating'] = 1
        return df

    def filter_triplets(self, df):
        print('Filtering triplets')
        if self.min_sc > 0 or self.min_uc > 0:
            item_sizes = df.groupby('sid').size()
            good_items = item_sizes.index[item_sizes >= self.min_sc]
            user_sizes = df.groupby('uid').size()
            good_users = user_sizes.index[user_sizes >= self.min_uc]
            while len(good_items) < len(item_sizes) or len(good_users) < len(user_sizes):
                if self.min_sc > 0:
                    item_sizes = df.groupby('sid').size()
                    good_items = item_sizes.index[item_sizes >= self.min_sc]
                    df = df[df['sid'].isin(good_items)]

                if self.min_uc > 0:
                    user_sizes = df.groupby('uid').size()
                    good_users = user_sizes.index[user_sizes >= self.min_uc]
                    df = df[df['uid'].isin(good_users)]

                item_sizes = df.groupby('sid').size()
                good_items = item_sizes.index[item_sizes >= self.min_sc]
                user_sizes = df.groupby('uid').size()
                good_users = user_sizes.index[user_sizes >= self.min_uc]
        return df

    def densify_index(self, df):
        print('Densifying index')
        umap = {u: i for i, u in enumerate(set(df['uid']))}
        smap = {s: i for i, s in enumerate(set(df['sid']))}
        df['uid'] = df['uid'].map(umap)
        df['sid'] = df['sid'].map(smap)
        return df, umap, smap

    def load_ratings_df(self):
        df = pd.read_csv(self.raw_paths[0], sep='::', header=None, engine='python')
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        # drop duplicate user-item pair records, keeping recent ratings only
        df.drop_duplicates(subset=['uid', 'sid'], keep='last', inplace=True)
        return df

    @staticmethod
    def _subset_interactions(mat, idxs):
        idxs = np.array(idxs)

        coo_mat = coo_matrix(
            (mat.data[idxs], (mat.row[idxs], mat.col[idxs])),
            shape=(mat.shape[0], mat.shape[1])
        )

        return coo_mat

    def random_split(self,
                     mat,
                     val_p = 0.0,
                     test_p = 0.2,
                     seed = 42):
        """Randomly split interactions into training, validation, and testing sets."""

        np.random.seed(seed)

        num_interactions = mat.nnz

        shuffle_indices = np.arange(num_interactions)
        np.random.shuffle(shuffle_indices)

        interactions = self._subset_interactions(mat=mat,
                                            idxs=shuffle_indices)

        validate_and_test_p = val_p + test_p
        validate_cutoff = int((1.0 - validate_and_test_p) * num_interactions)
        test_cutoff = int((1.0 - test_p) * num_interactions)

        train_idxs = np.arange(validate_cutoff)
        validate_idxs = np.arange(validate_cutoff, test_cutoff)
        test_idxs = np.arange(test_cutoff, num_interactions)

        train_interactions = self._subset_interactions(mat=mat,
                                                idxs=train_idxs)
        test_interactions = self._subset_interactions(mat=mat,
                                                idxs=test_idxs)

        if val_p > 0:
            validate_interactions = self._subset_interactions(mat=mat,
                                                        idxs=validate_idxs)

            return train_interactions, validate_interactions, test_interactions
        else:
            return train_interactions, test_interactions

    @staticmethod
    def _convert_to_torch_sparse(mat):
        values = mat.data
        indices = np.vstack((mat.row, mat.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = mat.shape

        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    def process(self):
        df = self.load_ratings_df()
        if self.min_rating:
            df = self.make_implicit(df)
        df = self.filter_triplets(df)
        df, umap, smap = self.densify_index(df)
        self.num_users = max(df.uid) + 1 # df.uid.nunique()
        self.num_items = max(df.sid) + 1 # df.sid.nunique()
        mat = coo_matrix((np.array(df.rating),
                          (np.array(df.uid), np.array(df.sid))),
                         shape=(self.num_users, self.num_items))
        
        self.positive_items = set(zip(mat.row, mat.col))

        mat_train, mat_test = self.random_split(mat)

        mat_train = self._convert_to_torch_sparse(mat_train)
        torch.save(mat_train, self.processed_paths[0])

        mat_test_pos = self._convert_to_torch_sparse(mat_test)._indices().T 
        _, indices = np.unique(mat_test_pos[:, 0], return_index=True)
        mat_test_pos = mat_test_pos[indices, :]
        torch.save(mat_test_pos, self.processed_paths[1])

        mat_test_neg = self._negative_sample(np.arange(mat_test.shape[0]))
        mat_test_neg = torch.tensor(mat_test_neg, dtype=torch.int)
        torch.save(mat_test_neg, self.processed_paths[2])

        return mat
        
    def todense(self) -> np.matrix:
        """Transforms sparse matrix to np.matrix, 2-d."""
        return self.mat.todense()

    def toarray(self) -> np.array:
        """Transforms sparse matrix to np.array, 2-d."""
        return self.mat.toarray()

    def head(self, n: int = 5) -> np.array:
        """Return the first ``n`` rows of the dense matrix as a np.array, 2-d."""
        n = self._prep_head_tail_n(n=n)
        return self.mat.tocsr()[range(n), :].toarray()

    def tail(self, n: int = 5) -> np.array:
        """Return the last ``n`` rows of the dense matrix as a np.array, 2-d."""
        n = self._prep_head_tail_n(n=n)
        return self.mat.tocsr()[range(-n, 0), :].toarray()

    def _prep_head_tail_n(self, n: int) -> int:
        """Ensure we don't run into an ``IndexError`` when using ``head`` or ``tail`` methods."""
        if n < 0:
            n = self.num_users + n
        if n > self.num_users:
            n = self.num_users
        return n

    def _negative_sample(self, user_id: Union[int, np.array]) -> np.array:
        """Generate negative samples for a ``user_id``."""
        if self.max_number_of_samples_to_consider > 0:
            # if we are here, we are doing true negative sampling
            negative_item_ids_list = list()

            if not isinstance(user_id, collections.abc.Iterable):
                user_id = [user_id]

            for specific_user_id in user_id:
                # generate true negative samples for the ``user_id``
                samples_checked = 0
                temp_negative_item_ids_list = list()

                while len(temp_negative_item_ids_list) < self.num_negative_samples:
                    negative_item_id = random.choice(range(self.num_items))
                    # we have a negative sample, make sure the user has not interacted with it
                    # before, else we resample and try again
                    while (
                        (specific_user_id, negative_item_id) in self.positive_items
                        or negative_item_id in temp_negative_item_ids_list
                    ):
                        if samples_checked >= self.max_number_of_samples_to_consider:
                            num_samples_left_to_generate = (
                                self.num_negative_samples - len(temp_negative_item_ids_list) - 1
                            )
                            temp_negative_item_ids_list += random.choices(
                                range(self.num_items), k=num_samples_left_to_generate
                            )
                            break

                        negative_item_id = random.choice(range(self.num_items))
                        samples_checked += 1

                    temp_negative_item_ids_list.append(negative_item_id)

                negative_item_ids_list += [np.array(temp_negative_item_ids_list)]

            if len(user_id) > 1:
                negative_item_ids_array = np.stack(negative_item_ids_list)
            else:
                negative_item_ids_array = negative_item_ids_list[0]
        else:
            # if we are here, we are doing approximate negative sampling
            if isinstance(user_id, collections.abc.Iterable):
                size = (len(user_id), self.num_negative_samples)
            else:
                size = (self.num_negative_samples,)

            negative_item_ids_array = np.random.randint(
                low=0,
                high=self.num_items,
                size=size,
            )

        return negative_item_ids_array

    def load(self):
        if self.is_train:
            self.train = torch.load(self.processed_paths[0])
            self.train_pos = self.train._indices().T
            self.n_users, self.n_items = self.train.size()

            self.score = torch.sparse.sum(self.train, dim=0).to_dense().repeat((self.n_users, 1))
            self.score[self.train_pos[:, 0], self.train_pos[:, 1]] = 0
        else:
            self.test_pos = torch.load(self.processed_paths[1])
            self.test_neg = torch.load(self.processed_paths[2])
            self.n_users = self.test_pos.shape[0]

            test_items = []
            for u in range(self.n_users):
                items = torch.cat((self.test_pos[u, 1].view(1), self.test_neg[u]))
                test_items.append(items)

            self.test_items = torch.vstack(test_items)
            self.test_labels = torch.zeros(self.test_items.shape)
            self.test_labels[:, 0] += 1


    def __len__(self):
            return self.n_users

    def __train__(self, index):
            return self.train_pos[index], self.score[self.train_pos[index][0]]

    def __test__(self, index):
            return self.test_pos[index], self.test_items[index], self.test_labels[index]

    def __getitem__(self, index):
            if self.is_train:
                return self.__train__(index)
            else:
                return self.__test__(index)
```

```python id="L93HCaIgxY3S"
class ML1mDataModule(LightningDataModule):

    def __init__(self, args) -> None:
        self.args = args
        super().__init__(args.data_dir)

        self.data_dir = args.data_dir if args.data_dir is not None else os.getcwd()
        self.val_split = args.val_split
        self.num_workers = args.num_workers
        self.normalize = args.normalize
        self.batch_size = args.batch_size
        self.seed = args.seed
        self.shuffle = args.shuffle
        self.pin_memory = args.pin_memory
        self.drop_last = args.drop_last

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        """Saves files to data_dir."""
        self.data = ML1mDataset(self.args)

    def setup(self, stage: Optional[str] = None) -> None:
        """Creates train, val, and test dataset."""
        if stage == "fit" or stage is None:
            dataset_train = ML1mDataset(self.args, train=True)
            dataset_val = ML1mDataset(self.args, train=True)

            # Split
            self.dataset_train = self._split_dataset(dataset_train)
            self.dataset_val = self._split_dataset(dataset_val, train=False)

        if stage == "test" or stage is None:
            self.dataset_test = ML1mDataset(self.args, train=False)

    def _split_dataset(self, dataset: Dataset, train: bool = True) -> Dataset:
        """Splits the dataset into train and validation set."""
        len_dataset = len(dataset)
        splits = self._get_splits(len_dataset)
        dataset_train, dataset_val = random_split(dataset, splits, generator=torch.Generator().manual_seed(self.seed))

        if train:
            return dataset_train
        return dataset_val

    def _get_splits(self, len_dataset: int) -> List[int]:
        """Computes split lengths for train and validation set."""
        if isinstance(self.val_split, int):
            train_len = len_dataset - self.val_split
            splits = [train_len, self.val_split]
        elif isinstance(self.val_split, float):
            val_len = int(self.val_split * len_dataset)
            train_len = len_dataset - val_len
            splits = [train_len, val_len]
        else:
            raise ValueError(f"Unsupported type {type(self.val_split)}")

        return splits

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """The train dataloader."""
        return self._data_loader(self.dataset_train, shuffle=self.shuffle)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """The val dataloader."""
        return self._data_loader(self.dataset_val)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """The test dataloader."""
        return self._data_loader(self.dataset_test)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
```

<!-- #region id="Z8AK22ObUH7C" -->
## Model
<!-- #endregion -->

```python id="yZDpRG2aSROg"
def get_ncdg(true, pred):
    match = pred.eq(true).nonzero(as_tuple=True)[1]
    ncdg = torch.log(torch.Tensor([2])).div(torch.log(match + 2))
    ncdg = ncdg.sum().div(pred.shape[0]).item()

    return ncdg


def get_apak(true, pred):
    k = pred.shape[1]
    apak = pred.eq(true).div(torch.arange(k) + 1)
    apak = apak.sum().div(pred.shape[0]).item()

    return apak


def get_hr(true, pred):
    hr = pred.eq(true).sum().div(pred.shape[0]).item()

    return hr


def get_eval_metrics(scores, true, k=10):
    test_items = [torch.LongTensor(list(item_scores.keys())) for item_scores in scores]
    test_scores = [torch.Tensor(list(item_scores.values())) for item_scores in scores]
    topk_indices = [s.topk(k).indices for s in test_scores]
    topk_items = [item[idx] for item, idx in zip(test_items, topk_indices)]
    pred = torch.vstack(topk_items)
    ncdg = get_ncdg(true, pred)
    apak = get_apak(true, pred)
    hr = get_hr(true, pred)

    return ncdg, apak, hr
```

```python id="ZLB8TQiUH0zQ"
class Model(pl.LightningModule):
    def __init__(self, n_neg=4, k=10):
        super().__init__()
        self.n_neg = n_neg
        self.k = k

    def forward(self, users, items):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        pos, score = batch
        users, pos_items = pos[:, 0], pos[:, 1]

        neg_items = torch.multinomial(score, self.n_neg)
        items = torch.cat((pos_items.view(-1, 1), neg_items), dim=1)

        labels = torch.zeros(items.shape)
        labels[:, 0] += 1
        users = users.view(-1, 1).repeat(1, items.shape[1])

        users = users.view(-1, 1).squeeze()
        items = items.view(-1, 1).squeeze()
        labels = labels.view(-1, 1).squeeze()

        logits = self(users, items)
        loss = self.loss_fn(logits, labels)

        return {
            "loss": loss,
            "logits": logits.detach(),
        }

    def training_epoch_end(self, outputs):
        # This function recevies as parameters the output from "training_step()"
        # Outputs is a list which contains a dictionary like:
        # [{'pred':x,'target':x,'loss':x}, {'pred':x,'target':x,'loss':x}, ...]
        pass

    def test_step(self, batch, batch_idx):
        pos, items, labels = batch
        n_items = items.shape[1]
        users = pos[:, 0].view(-1, 1).repeat(1, n_items)

        users = users.view(-1, 1).squeeze()
        items = items.view(-1, 1).squeeze()
        labels = labels.view(-1, 1).squeeze()

        logits = self(users, items)
        loss = self.loss_fn(logits, labels)

        items = items.view(-1, n_items)
        logits = logits.view(-1, n_items)
        item_true = pos[:, 1].view(-1, 1)
        item_scores = [dict(zip(item.tolist(), score.tolist())) for item, score in zip(items, logits)]
        ncdg, apak, hr = get_eval_metrics(item_scores, item_true, self.k)
        metrics = {
            'loss': loss.item(),
            'ncdg': ncdg,
            'apak': apak,
            'hr': hr,
        }
        self.log("Val Metrics", metrics, prog_bar=True)

        return {
            "loss": loss.item(),
            "logits": logits,
        }

    def test_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.005)
        return optimizer

    def loss_fn(self, logits, labels):
        return nn.BCEWithLogitsLoss()(logits, labels)
```

```python id="GP9RiW9tTOT0"
class MF(Model):
    """A matrix factorization model trained using SGD and negative sampling."""

    def __init__(self, n_users, n_items, embedding_dim):
        super().__init__()
        self.user_embedding = nn.Embedding(
            num_embeddings=n_users, embedding_dim=embedding_dim
        )
        self.item_embedding = nn.Embedding(
            num_embeddings=n_items, embedding_dim=embedding_dim
        )
        self.user_bias = nn.Parameter(torch.zeros((n_users)))
        self.item_bias = nn.Parameter(torch.zeros((n_items)))
        self.bias = nn.Parameter(torch.Tensor([0]))

    def forward(self, users, items):
        return (
                self.bias +
                self.user_bias[users] +
                self.item_bias[items] +
                (self.user_embedding(users).mul(self.item_embedding(items))).sum(dim=-1)
        )
```

```python id="kZEgvNlTJX2s"
class GMF(Model):
    def __init__(self, n_users, n_items, embedding_dim):
        super().__init__()

        self.user_embedding = nn.Embedding(
            num_embeddings=n_users, embedding_dim=embedding_dim
        )
        self.item_embedding = nn.Embedding(
            num_embeddings=n_items, embedding_dim=embedding_dim
        )
        self.fc = nn.Linear(embedding_dim, 1)
        
        # not using sigmoid layer because loss is BCEWithLogits in PairModel
        # self.logistic = nn.Sigmoid()

    def forward(self, users, items):
        user_embeddings = self.user_embedding(users)
        item_embeddings = self.item_embedding(items)
        embeddings = user_embeddings.mul(item_embeddings)
        output = self.fc(embeddings)

        # not using sigmoid layer because loss is BCEWithLogits in PairModel
        # rating = self.logistic(output)

        return output.squeeze()
```

```python id="GJcE3PLdJ2dq"
class MLP(Model):
    def __init__(self, n_users, n_items, embedding_dim, dropout=0.1):
        super().__init__()

        self.user_embedding = nn.Embedding(
            num_embeddings=n_users, embedding_dim=embedding_dim
        )
        self.item_embedding = nn.Embedding(
            num_embeddings=n_items, embedding_dim=embedding_dim
        )
        self.fc1 = nn.Linear(embedding_dim * 2, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, int(embedding_dim / 2))
        self.fc3 = nn.Linear(int(embedding_dim / 2), 1)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, users, items):
        user_embeddings = self.user_embedding(users)
        item_embeddings = self.item_embedding(items)
        embeddings = torch.cat([user_embeddings, item_embeddings], axis=1)
        output = nn.ReLU()(self.fc1(embeddings))
        output = self.dropout(output)
        output = nn.ReLU()(self.fc2(output))
        output = self.dropout(output)
        output = self.fc3(output)

        return output.squeeze()
```

```python id="SiYDp66AJ6cR"
class NeuMF(Model):
    def __init__(self, n_users, n_items, embedding_dim, dropout=0.1):
        super().__init__()

        self.user_embedding = nn.Embedding(
            num_embeddings=n_users, embedding_dim=embedding_dim
        )
        self.item_embedding = nn.Embedding(
            num_embeddings=n_items, embedding_dim=embedding_dim
        )

        self.user_embedding_gmf = nn.Embedding(
            num_embeddings=n_users, embedding_dim=embedding_dim
        )
        self.item_embedding_gmf = nn.Embedding(
            num_embeddings=n_items, embedding_dim=embedding_dim
        )

        self.gmf = nn.Linear(embedding_dim, int(embedding_dim / 2))

        self.fc1 = nn.Linear(embedding_dim * 2, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)
        self.fc3 = nn.Linear(embedding_dim, int(embedding_dim / 2))

        self.fc_final = nn.Linear(embedding_dim, 1)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, users, items):
        user_embeddings = self.user_embedding(users)
        item_embeddings = self.item_embedding(items)
        embeddings = torch.cat([user_embeddings, item_embeddings], dim=1)

        user_embeddings_gmf = self.user_embedding_gmf(users)
        item_embeddings_gmf = self.item_embedding_gmf(items)
        embeddings_gmf = user_embeddings_gmf.mul(item_embeddings_gmf)

        output_gmf = self.gmf(embeddings_gmf)
        output = nn.ReLU()(self.fc1(embeddings))
        output = self.dropout(output)
        output = nn.ReLU()(self.fc2(output))
        output = self.dropout(output)
        output = self.fc3(output)

        output = torch.cat([output, output_gmf], dim=1)
        output = self.fc_final(output)

        return output.squeeze()
```

<!-- #region id="xzdlvUKWUFwq" -->
## Trainer
<!-- #endregion -->

```python id="TcDhXSZYIbPP"
class Args:
    data_dir = '/content/data' # Where to save/load the data
    min_rating = 4
    num_negative_samples = 99
    min_uc = 5
    min_sc = 5

    log_dir = '/content/logs'
    model_dir = '/content/models'

    val_split = 0.2 # Percent (float) or number (int) of samples to use for the validation split
    num_workers = 2 # How many workers to use for loading data
    normalize = False # If true applies rating normalize
    batch_size = 32 # How many samples per batch to load
    seed = 42 # Random seed to be used for train/val/test splits
    shuffle = True # If true shuffles the train data every epoch
    pin_memory = True # If true, the data loader will copy Tensors into CUDA pinned memory before returning them
    drop_last = False # If true drops the last incomplete batch

    embedding_dim = 20
    max_epochs = 5

args = Args()
```

```python id="Vb64vRJF42DD"
ds = ML1mDataModule(args)

logger = TensorBoardLogger(
    save_dir=args.log_dir,
)

checkpoint_callback = ModelCheckpoint(
    monitor="valid_loss",
    mode="min",
    dirpath=args.model_dir,
    filename="recommender",
)

def pl_trainer(model, datamodule):

    trainer = pl.Trainer(
    max_epochs=args.max_epochs,
    logger=logger,
    check_val_every_n_epoch=10,
    callbacks=[checkpoint_callback],
    # enable_checkpointing=False,
    # num_sanity_val_steps=0,
    # gradient_clip_val=1,
    # gradient_clip_algorithm="norm",
    gpus=None
    )

    trainer.fit(model, datamodule=datamodule)
    test_result = trainer.test(model, datamodule=datamodule)
    return test_result
```

```python colab={"base_uri": "https://localhost:8080/"} id="gLErb2c0LpnK" executionInfo={"status": "ok", "timestamp": 1641738428619, "user_tz": -330, "elapsed": 8517, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e626687e-5ad8-422f-d01f-b31b86c4062a"
ds.prepare_data()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 498, "referenced_widgets": ["7272bee276b84864b4645bca01d95083", "e07224be774e454fbc2669722da56930", "de2f41506dc64afc8091704e35d04e22", "d7b72a83d51446d38cbed8241d793ed2", "72c15a978acc4227bdfed3ee9673c1d4", "f7ed387aec4d44e2a76d55a0173fc1d2", "79b7cf3ec64b4cf897961da48fb409fe", "beb8d249be914b36844894fadff89964", "2423e4dddf784e78948226fcee3c0289", "facc6961094b4ee488fe48a415268f91", "274dbeeea38a4b2195dff4ab219dcb93", "bbfef29cbe3e4de196b37a69e0d67572", "b0f148fe37394f29a98bedbbbeaa3633", "ddb1281659a24cd7b4c5335afa72ce66", "ce853f7a27214943bd71ccf0c72a7342", "e34643bbff844561b46d7686ada326ba", "b6d646364d0d40379722f6ee0a19af2c", "831536ed7211406a99081a5f6cc01f2d", "cf2dade378ea4d57a241eacac23c1671", "c5e086c24628403da5d91e45f6c76552", "9914b0df592744b59ad5b792ac722df1", "e2ce4119094e4c3d8cc0aa815a817682"]} id="ZqgvH4Ko3lGQ" executionInfo={"status": "ok", "timestamp": 1641738446025, "user_tz": -330, "elapsed": 12299, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="476275a0-207a-4b77-a220-b9655668fea0"
model = MF(n_items=ds.data.num_items, n_users=ds.data.num_users, embedding_dim=args.embedding_dim)

pl_trainer(model, ds)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 637, "referenced_widgets": ["95177e89f506463d913d455fa6907403", "e3fe24cefbc44e86babc83fe2d815e7e", "e69c6fd24ced4e648558d90f0fd45bfd", "4e8cb9ceebcd465a93b2dcb7e97d0afb", "86a76481a891493fabcc7397260ce489", "b937b6508e1246909a8b11fa1d27303d", "0b4c46ed9b3c4ccd8d94a769f41a1c2a", "69ea10f1fd8249e6addb3e524576cbc9", "d8601ff3555f4a22ae3dc8d5e6be0339", "57dc9f455a094ecfb8158860ffe83441", "4ff53c140e1e42f49100afc3ced6a7df", "1b345d23e1d9441eb98686345821f9de", "0bd828a5bc3445bf9e0389d69652a3b1", "0bef2a37b6624a2483a4ffc50ec89d01", "25367e77650b467e8003de5647f16cef", "9646b227abd241cb9b8c86361008f216", "a4b84028e8444f0098575d6d64de09a9", "9320ff24350247bcbbc307b122092a44", "80b642fb02d54e3d95b1b6459bb8424b", "7a202e9712ab4be297764255ead850b9", "bae2fa4228c5407f976e8a5b6f39cc31", "08bf152113e24e6b8215c4836e1dd7ec"]} id="shdwGCioIVfl" executionInfo={"status": "ok", "timestamp": 1641738493288, "user_tz": -330, "elapsed": 14407, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7f146a8f-ce37-43c8-9f42-c3d74b145dbc"
model = NeuMF(n_items=ds.data.num_items, n_users=ds.data.num_users, embedding_dim=args.embedding_dim)

pl_trainer(model, ds)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 515, "referenced_widgets": ["b76deecadc6643efaa93b8866dbde257", "25fa2034ca6441459e0057b65627ac48", "481aeed4e1624148b16af8d6cf77bb67", "5e7cca43416f4a2a834b55792e6d6f91", "63019df58b014431b44906017a612e7e", "951779694d2b44259350d7fdeb61613d", "25b0969c494c4d2690f0f1412814c1c6", "a78ab70d365145c1bc15aea0bd3e4e2b", "2e7472ee41454fde807e4abf0e0f00d6", "e2b35bdb74bb45429ba197d94c10174a", "f714f93f82c4440b905c28fa3a822a14", "abb8985c43204b3c90976e0f927292f6", "92ad1a9a236f4a8f93948622dd8a757f", "696a461e977746faa1c73d9749ba39e2", "1a3ca25d104e44f0872db68ee8860593", "f8426f12d10140b3a05d8ca539b26052", "b7ca7bc5860d44efa2c8ae3bc18e39ad", "170ec48b6ae94a9eb3b72cab78a7c3d0", "bf23584410e74452a753610646129f79", "6b8aa570f7d844729313820eb0b7b389", "f4339eb0c8364597932eb4f812ff0633", "882f508ad6a54c288035f87955a4ca4a"]} id="DfG--ovBKqjK" executionInfo={"status": "ok", "timestamp": 1641738505015, "user_tz": -330, "elapsed": 11746, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="023c8fec-4a46-4d00-f101-4bbce143ad72"
model = GMF(n_items=ds.data.num_items, n_users=ds.data.num_users, embedding_dim=args.embedding_dim)

pl_trainer(model, ds)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 567, "referenced_widgets": ["4700d7f58ce54492b1f00cc1ea5f86d2", "a43c9b8c698a4f3cb1de1a03222690ee", "322c0b432a78418ca67c30b3c1f201d8", "d76f2bbe0d744e47bf3c4ec618048e7e", "dc523ec1fd0045f891afaa4bc9f2bc4a", "59de47926db844ea8df84c332fc94b4b", "f7cceaf7d5a744149697592b4a7ab4a1", "259b85cb27ca4237bd50eef3c383e4d3", "a4d2f362546e4aba9ba9d6dbcd36d42c", "3c911327eeb84c1a847999c9bfada6ad", "019c94b8539a40fba54a60efdd0194db", "7649f75ce60e4cbbb69742d07cef4def", "8c5f14fbe633411c960e50077f27f20d", "43ef80d2f81b44a3ac27e7e81ea1506e", "36e09f2367714820a2471178e2133fa7", "9c11abec0a5145819f8eb32410c10dad", "4f6be0df67614231b640be743d7b4a30", "4240988a8a024cbfacc394c7ce109d3f", "5e3291b022af428288254bfb0b1fa15f", "e946d564be8b4ee89fc488e838484ed8", "b6d1919605154e21b8a6f06c541cdf25", "244535fe96994cde88566527b19f3b8a"]} id="pxx3bB0RKsdB" executionInfo={"status": "ok", "timestamp": 1641738519166, "user_tz": -330, "elapsed": 12198, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0c326196-2eb5-4cb6-dcca-b4a6b0234bc2"
model = MLP(n_items=ds.data.num_items, n_users=ds.data.num_users, embedding_dim=args.embedding_dim)

pl_trainer(model, ds)
```

<!-- #region id="efi1fMTlPYd_" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="WBiaCF6NPYeD" executionInfo={"status": "ok", "timestamp": 1641738527737, "user_tz": -330, "elapsed": 3590, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5700f7c2-2592-42a9-a20f-6429cffe6b2e"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="UF541iCjPYeF" -->
---
<!-- #endregion -->

<!-- #region id="HkJGqjDNPYeH" -->
**END**
<!-- #endregion -->
