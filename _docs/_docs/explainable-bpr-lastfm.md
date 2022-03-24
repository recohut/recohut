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

<!-- #region id="_tLSbR0PWx9j" -->
# Training Explainable BPR model on LastFM dataset
<!-- #endregion -->

<!-- #region id="LLaRmDxUWx50" -->
## Executive summary

| | |
| --- | --- |
| Problem | BPR is black-box model and vulnerable to Missing-not-at-random (MNaR) based exposure bias. |
| Solution | An explainable loss function and a corresponding Matrix Factorization-based model called Explainable Bayesian Personalized Ranking (EBPR) that generates recommendations along with item-based explanations. |
| Dataset | ML-100k, ML-1m, LastFM-200k, Yahoo R3. |
| Preprocessing | Interactions were converted into binary interactions, regardless of their values. Then we filtered out users with less than 10 interactions to ensure enough training and evaluation samples for every user and reduce the data sparsity. We follow the standard Leave-One-Out (LOO) procedure that consists of considering the latest interaction of each user as a test item and comparing it to 100 randomly sampled negative items. In the training, we sample, at every epoch, one negative item for every positive user-item interaction. |
| Metrics | NDCG@K, HR@K, Mean Explainability Precision (MEP@K), WMEP@K, Avg_Pop@K, EFD@K, and Div@K. |
| Hyperparams | NUM_CONFIGURATIONS, NUM_REPS, NUM_EPOCH, WEIGHT_DECAY, NEIGHBORHOOD, TOP_K, LR, OPTIMIZER, SGD_MOMENTUM, RMSPROP_ALPHA, RMSPROP_MOMENTUM, LOO_EVAL, TEST_RATE, USE_CUDA, DEVICE_ID, SAVE_MODELS, SAVE_RESULTS, INT_PER_ITEM. |
| Models | BPR, UBPR, EBPR, pUEBPR, UEBPR. |
| Cluster | Python 3.7+, PyTorch |
| Tags | `Fairness`, `Explainability`, `ExposureBias`, `ExplainableBPR` |
| Credits | Khalil Damak |
<!-- #endregion -->

<!-- #region id="CWxQF9gwMGmS" -->
## Process flow

![](https://github.com/RecoHut-Stanzas/S241566/raw/main/images/process_flow.svg)
<!-- #endregion -->

<!-- #region id="B_IXvLmOMGiw" -->
## Setup
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="cPAreB--NPMR" executionInfo={"status": "ok", "timestamp": 1639221891836, "user_tz": -330, "elapsed": 8796, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ffca50db-d08d-4fee-9164-de04b5a62959"
!pip install -q ml_metrics
!pip install -q pyprind
```

```python id="530fE0WPMCMX"
import torch
import random
import numpy as np
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

import math
from ml_metrics import mapk
from itertools import combinations
import sys
import pyprind
import argparse
```

```python id="X45wVmpOMJKG"
random.seed(1)
```

```python id="YhP7jqejMJHv"
!mkdir -p Output/checkpoints
```

<!-- #region id="dh1WUwE8MJEb" -->
## Dataset
<!-- #endregion -->

<!-- #region id="uLoOJoePMMYS" -->
Download LastFM raw dataset
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="oNxbyTO4_iC5" executionInfo={"status": "ok", "timestamp": 1639221898664, "user_tz": -330, "elapsed": 732, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8df28659-fb21-474e-f3ad-6c6872c55e26"
!mkdir -p Data/lastfm-2k
!git clone --branch v1 https://github.com/RecoHut-Datasets/lastfm.git Data/lastfm-2k
```

```python id="QLkKHjwOMKKk"
def read_data(dataset_name, int_per_item):
    """Read dataset"""
    dataset = pd.DataFrame()
    if dataset_name == 'ml-100k':
        # Load Movielens 100K Data
        data_dir = 'Data/ml-100k/u.data'
        dataset = pd.read_csv(data_dir, sep='\t', header=None, names=['uid', 'mid', 'rating', 'timestamp'],
                                  engine='python')
    elif dataset_name == 'ml-1m':
        # Load Movielens 1M Data
        data_dir = 'Data/ml-1m/ratings.dat'
        dataset = pd.read_csv(data_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')

    elif dataset_name == 'lastfm-2k':
        # Load Last.FM 2K Data
        data_dir = 'Data/lastfm-2k/user_artists.dat'
        dataset = pd.read_csv(data_dir, sep='\t', header=0, names=['uid', 'mid', 'rating'],  engine='python')
        dataset['timestamp'] = [1 for i in range(len(dataset))]
        # Filtering items with more than int_per_item interactions
        item_count = dataset[['uid', 'mid']].groupby('mid').count()['uid'].rename('count').reset_index()
        dataset = dataset.merge(item_count, how='left', on='mid')
        dataset = dataset.loc[dataset['count'] >= int_per_item][['uid', 'mid', 'rating', 'timestamp']]
        # Filtering users with more than 10 interactions
        user_count = dataset[['uid', 'mid']].groupby('uid').count()['mid'].rename('count').reset_index()
        dataset = dataset.merge(user_count, how='left', on='uid')
        dataset = dataset.loc[dataset['count'] >= 10][['uid', 'mid', 'rating', 'timestamp']]

    elif dataset_name == 'yahoo-r3':
        # Load Yahoo! R3 Data
        data_dir = 'Data/yahoo-r3/ydata-ymusic-rating-study-v1_0-train.txt'
        dataset = pd.read_csv(data_dir, sep='\t', header=None, names=['uid', 'mid', 'rating'],  engine='python')
        dataset['timestamp'] = [1 for i in range(len(dataset))]

    elif dataset_name == 'yahoo-r3-unbiased':
        # Load Yahoo! R3 Data
        data_dir = 'Data/yahoo-r3/ydata-ymusic-rating-study-v1_0-train.txt'
        test_data_dir = 'Data/yahoo-r3/ydata-ymusic-rating-study-v1_0-test.txt'
        dataset = pd.read_csv(data_dir, sep='\t', header=None, names=['uid', 'mid', 'rating'],  engine='python')
        dataset['test'] = [0 for i in range(len(dataset))]
        test_dataset = pd.read_csv(test_data_dir, sep='\t', header=None, names=['uid', 'mid', 'rating'],  engine='python')
        test_dataset['test'] = [1 for i in range(len(test_dataset))]
        dataset = pd.concat([dataset, test_dataset])
        dataset['timestamp'] = [1 for i in range(len(dataset))]
    # Reindex data
    user_id = dataset[['uid']].drop_duplicates().reindex()
    user_id['userId'] = np.arange(len(user_id))
    dataset = pd.merge(dataset, user_id, on=['uid'], how='left')
    item_id = dataset[['mid']].drop_duplicates()
    item_id['itemId'] = np.arange(len(item_id))
    dataset = pd.merge(dataset, item_id, on=['mid'], how='left')
    if 'test' in dataset:
        dataset = dataset[['userId', 'itemId', 'rating', 'timestamp', 'test']]
    else:
        dataset = dataset[['userId', 'itemId', 'rating', 'timestamp']]

    return dataset
```

```python id="zI-Bd881MKHr"
class data_loader(Dataset):
    """Convert user, item, negative and target Tensors into Pytorch Dataset"""

    def __init__(self, user_tensor, positive_item_tensor, negative_item_tensor, target_tensor):
        self.user_tensor = user_tensor
        self.positive_item_tensor = positive_item_tensor
        self.negative_item_tensor = negative_item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.positive_item_tensor[index], self.negative_item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)
```

```python id="m_uKfxmHMKEV"
class data_loader_implicit(Dataset):
    """Convert user and item Tensors into Pytorch Dataset"""

    def __init__(self, user_tensor, item_tensor):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)
```

```python id="PxzQ_HE7Mrkk"
class data_loader_test_explicit(Dataset):
    """Convert user, item and target Tensors into Pytorch Dataset"""

    def __init__(self, user_tensor, item_tensor, target_tensor):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)
```

```python id="A6GzzXxvMtbC"
class data_loader_negatives(Dataset):
    """Convert user and item negative Tensors into Pytorch Dataset"""

    def __init__(self, user_neg_tensor, item_neg_tensor):
        self.user_neg_tensor = user_neg_tensor
        self.item_neg_tensor = item_neg_tensor

    def __getitem__(self, index):
        return self.user_neg_tensor[index], self.item_neg_tensor[index]

    def __len__(self):
        return self.user_neg_tensor.size(0)
```

```python id="-9838B6oMvOu"
class SampleGenerator(object):
    """Construct dataset"""

    def __init__(self, ratings, config, split_val):
        """
        args:
            ratings: pd.DataFrame containing 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
            config: dictionary containing the configuration hyperparameters
            split_val: boolean that takes True if we are using a validation set
        """
        assert 'userId' in ratings.columns
        assert 'itemId' in ratings.columns
        assert 'rating' in ratings.columns

        self.config = config
        self.ratings = ratings
        self.split_val = split_val
        self.preprocess_ratings = self._binarize(ratings)
        self.user_pool = set(self.ratings['userId'].unique())
        self.item_pool = set(self.ratings['itemId'].unique())
        # create negative item samples
        self.negatives = self._sample_negative(ratings, self.split_val)
        if self.config['loo_eval']:
            if self.split_val:
                self.train_ratings, self.val_ratings = self._split_loo(self.preprocess_ratings, split_val=True)
            else:
                self.train_ratings, self.test_ratings = self._split_loo(self.preprocess_ratings, split_val=False)
        else:
            self.test_rate = self.config['test_rate']
            if self.split_val:
                self.train_ratings, self.val_ratings = self.train_test_split_random(self.ratings, split_val=True)
            else:
                self.train_ratings, self.test_ratings = self.train_test_split_random(self.ratings, split_val=False)

    def _binarize(self, ratings):
        """binarize into 0 or 1 for imlicit feedback"""
        ratings = deepcopy(ratings)
        ratings['rating'] = 1.0
        return ratings

    def train_test_split_random(self, ratings, split_val):
        """Random train/test split"""
        if 'test' in list(ratings):
            test = ratings[ratings['test'] == 1]
            train = ratings[ratings['test'] == 0]
        else:
            train, test = train_test_split(ratings, test_size=self.test_rate)
        if split_val:
            train, val = train_test_split(train, test_size=self.test_rate / (1 - self.test_rate))
            return train[['userId', 'itemId', 'rating']], val[['userId', 'itemId', 'rating']]
        else:
            return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]

    def _split_loo(self, ratings, split_val):
        """leave-one-out train/test split"""
        if 'test' in list(ratings):
            test = ratings[ratings['test'] == 1]
            ratings = ratings[ratings['test'] == 0]
            if split_val:
                ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
                val = ratings[ratings['rank_latest'] == 1]
                train = ratings[ratings['rank_latest'] > 1]
                return train[['userId', 'itemId', 'rating']], val[['userId', 'itemId', 'rating']]
            return ratings[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]
        ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        test = ratings[ratings['rank_latest'] == 1]
        if split_val:
            val = ratings[ratings['rank_latest'] == 2]
            train = ratings[ratings['rank_latest'] > 2]
            assert train['userId'].nunique() == test['userId'].nunique() == val['userId'].nunique()
            return train[['userId', 'itemId', 'rating']], val[['userId', 'itemId', 'rating']]
        train = ratings[ratings['rank_latest'] > 1]
        assert train['userId'].nunique() == test['userId'].nunique()
        return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]

    def _sample_negative(self, ratings, split_val):
        """return all negative items & 100 sampled negative test items & 100 sampled negative val items"""
        interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
            columns={'itemId': 'interacted_items'})
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        interact_status['test_negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, 100))
        interact_status['negative_items'] = interact_status.apply(lambda x: (x.negative_items - set(x.test_negative_samples)), axis=1)
        if split_val:
            interact_status['val_negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, 100))
            interact_status['negative_items'] = interact_status.apply(lambda x: (x.negative_items - set(x.val_negative_samples)), axis=1)
            return interact_status[['userId', 'negative_items', 'test_negative_samples', 'val_negative_samples']]
        else:
            return interact_status[['userId', 'negative_items', 'test_negative_samples']]

    def train_data_loader(self, batch_size):
        """instance train loader for one training epoch"""
        train_ratings = pd.merge(self.train_ratings, self.negatives[['userId', 'negative_items']], on='userId')
        users = [int(x) for x in train_ratings['userId']]
        items = [int(x) for x in train_ratings['itemId']]
        ratings = [float(x) for x in train_ratings['rating']]
        neg_items = [random.choice(list(neg_list)) for neg_list in train_ratings['negative_items']]
        dataset = data_loader(user_tensor=torch.LongTensor(users),
                              positive_item_tensor=torch.LongTensor(items),
                              negative_item_tensor=torch.LongTensor(neg_items),
                              target_tensor=torch.FloatTensor(ratings))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def test_data_loader(self, batch_size):
        """create evaluation data"""
        if self.config['loo_eval']:
            test_ratings = pd.merge(self.test_ratings, self.negatives[['userId', 'test_negative_samples']], on='userId')
            test_users, test_items, negative_users, negative_items = [], [], [], []
            for row in test_ratings.itertuples():
                test_users.append(int(row.userId))
                test_items.append(int(row.itemId))
                for i in range(len(row.test_negative_samples)):
                    negative_users.append(int(row.userId))
                    negative_items.append(int(row.test_negative_samples[i]))
            dataset = data_loader_implicit(user_tensor=torch.LongTensor(test_users),
                                           item_tensor=torch.LongTensor(test_items))
            dataset_negatives = data_loader_negatives(user_neg_tensor=torch.LongTensor(negative_users),
                                                      item_neg_tensor=torch.LongTensor(negative_items))
            return [DataLoader(dataset, batch_size=batch_size, shuffle=False), DataLoader(dataset_negatives, batch_size=batch_size, shuffle=False)]
        else:
            test_ratings = self.test_ratings
            test_users = [int(x) for x in test_ratings['userId']]
            test_items = [int(x) for x in test_ratings['itemId']]
            test_ratings = [float(x) for x in test_ratings['rating']]
            dataset = data_loader_test_explicit(user_tensor=torch.LongTensor(test_users),
                                                item_tensor=torch.LongTensor(test_items),
                                                target_tensor=torch.FloatTensor(test_ratings))
            return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def val_data_loader(self, batch_size):
        """create validation data"""
        if self.config['loo_eval']:
            val_ratings = pd.merge(self.val_ratings, self.negatives[['userId', 'val_negative_samples']], on='userId')
            val_users, val_items, negative_users, negative_items = [], [], [], []
            for row in val_ratings.itertuples():
                val_users.append(int(row.userId))
                val_items.append(int(row.itemId))
                for i in range(len(row.val_negative_samples)):
                    negative_users.append(int(row.userId))
                    negative_items.append(int(row.val_negative_samples[i]))
            dataset = data_loader_implicit(user_tensor=torch.LongTensor(val_users),
                                           item_tensor=torch.LongTensor(val_items))
            dataset_negatives = data_loader_negatives(user_neg_tensor=torch.LongTensor(negative_users),
                                                      item_neg_tensor=torch.LongTensor(negative_items))
            return [DataLoader(dataset, batch_size=batch_size, shuffle=False), DataLoader(dataset_negatives, batch_size=batch_size, shuffle=False)]
        else:
            val_ratings = self.val_ratings
            val_users = [int(x) for x in val_ratings['userId']]
            val_items = [int(x) for x in val_ratings['itemId']]
            val_ratings = [float(x) for x in val_ratings['rating']]
            dataset = data_loader_test_explicit(user_tensor=torch.LongTensor(val_users),
                                                item_tensor=torch.LongTensor(val_items),
                                                target_tensor=torch.FloatTensor(val_ratings))
            return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def create_explainability_matrix(self, include_test=False):
        """create explainability matrix"""
        if not include_test:
            print('Creating explainability matrix...')
            interaction_matrix = pd.crosstab(self.train_ratings.userId, self.train_ratings.itemId)
            missing_columns = list(set(range(self.config['num_items'])) - set(list(interaction_matrix)))
            missing_rows = list(set(range(self.config['num_users'])) - set(interaction_matrix.index))
            for missing_column in missing_columns:
                interaction_matrix[missing_column] = [0] * len(interaction_matrix)
            for missing_row in missing_rows:
                interaction_matrix.loc[missing_row] = [0] * self.config['num_items']
            interaction_matrix = np.array(interaction_matrix[list(range(self.config['num_items']))].sort_index())
        elif not self.split_val:
            print('Creating test explainability matrix...')
            interaction_matrix = np.array(pd.crosstab(self.preprocess_ratings.userId, self.preprocess_ratings.itemId)[
                                              list(range(self.config['num_items']))].sort_index())
        else:
            print('Creating val explainability matrix...')
            interaction_matrix = pd.crosstab(self.train_ratings.userId.append(self.val_ratings.userId), self.train_ratings.itemId.append(self.val_ratings.itemId))
            missing_columns = list(set(range(self.config['num_items'])) - set(list(interaction_matrix)))
            missing_rows = list(set(range(self.config['num_users'])) - set(interaction_matrix.index))
            for missing_column in missing_columns:
                interaction_matrix[missing_column] = [0] * len(interaction_matrix)
            for missing_row in missing_rows:
                interaction_matrix.loc[missing_row] = [0] * self.config['num_items']
            interaction_matrix = np.array(interaction_matrix[list(range(self.config['num_items']))].sort_index())
        #item_similarity_matrix = 1 - pairwise_distances(interaction_matrix.T, metric = "hamming")
        item_similarity_matrix = cosine_similarity(interaction_matrix.T)
        np.fill_diagonal(item_similarity_matrix, 0)
        neighborhood = [np.argpartition(row, - self.config['neighborhood'])[- self.config['neighborhood']:]
                        for row in item_similarity_matrix]
        explainability_matrix = np.array([[sum([interaction_matrix[user, neighbor] for neighbor in neighborhood[item]])
                                           for item in range(self.config['num_items'])] for user in
                                          range(self.config['num_users'])]) / self.config['neighborhood']
        #explainability_matrix[explainability_matrix < 0.1] = 0
        #explainability_matrix = explainability_matrix + self.config['epsilon']
        return explainability_matrix

    def create_popularity_vector(self, include_test=False):
        """create popularity vector"""
        if not include_test:
            print('Creating popularity vector...')
            interaction_matrix = pd.crosstab(self.train_ratings.userId, self.train_ratings.itemId)
            missing_columns = list(set(range(self.config['num_items'])) - set(list(interaction_matrix)))
            missing_rows = list(set(range(self.config['num_users'])) - set(interaction_matrix.index))
            for missing_column in missing_columns:
                interaction_matrix[missing_column] = [0] * len(interaction_matrix)
            for missing_row in missing_rows:
                interaction_matrix.loc[missing_row] = [0] * self.config['num_items']
            interaction_matrix = np.array(interaction_matrix[list(range(self.config['num_items']))].sort_index())
        elif not self.split_val:
            print('Creating test popularity vector...')
            interaction_matrix = np.array(pd.crosstab(self.preprocess_ratings.userId, self.preprocess_ratings.itemId)[
                                              list(range(self.config['num_items']))].sort_index())
        else:
            print('Creating val popularity vector...')
            interaction_matrix = pd.crosstab(self.train_ratings.userId.append(self.val_ratings.userId),
                                             self.train_ratings.itemId.append(self.val_ratings.itemId))
            missing_columns = list(set(range(self.config['num_items'])) - set(list(interaction_matrix)))
            missing_rows = list(set(range(self.config['num_users'])) - set(interaction_matrix.index))
            for missing_column in missing_columns:
                interaction_matrix[missing_column] = [0] * len(interaction_matrix)
            for missing_row in missing_rows:
                interaction_matrix.loc[missing_row] = [0] * self.config['num_items']
            interaction_matrix = np.array(interaction_matrix[list(range(self.config['num_items']))].sort_index())
        popularity_vector = np.sum(interaction_matrix, axis=0)
        popularity_vector = (popularity_vector / max(popularity_vector)) ** 0.5
        return popularity_vector

    def create_neighborhood(self, include_test=False):
        """Determine item neighbors"""
        if not include_test:
            print('Determining item neighborhoods...')
            interaction_matrix = pd.crosstab(self.train_ratings.userId, self.train_ratings.itemId)
            missing_columns = list(set(range(self.config['num_items'])) - set(list(interaction_matrix)))
            missing_rows = list(set(range(self.config['num_users'])) - set(interaction_matrix.index))
            for missing_column in missing_columns:
                interaction_matrix[missing_column] = [0] * len(interaction_matrix)
            for missing_row in missing_rows:
                interaction_matrix.loc[missing_row] = [0] * self.config['num_items']
            interaction_matrix = np.array(interaction_matrix[list(range(self.config['num_items']))].sort_index())
        elif not self.split_val:
            print('Determining test item neighborhoods...')
            interaction_matrix = np.array(pd.crosstab(self.preprocess_ratings.userId, self.preprocess_ratings.itemId)[
                                              list(range(self.config['num_items']))].sort_index())
        else:
            print('Determining val item neighborhoods...')
            interaction_matrix = pd.crosstab(self.train_ratings.userId.append(self.val_ratings.userId),
                                             self.train_ratings.itemId.append(self.val_ratings.itemId))
            missing_columns = list(set(range(self.config['num_items'])) - set(list(interaction_matrix)))
            missing_rows = list(set(range(self.config['num_users'])) - set(interaction_matrix.index))
            for missing_column in missing_columns:
                interaction_matrix[missing_column] = [0] * len(interaction_matrix)
            for missing_row in missing_rows:
                interaction_matrix.loc[missing_row] = [0] * self.config['num_items']
            interaction_matrix = np.array(interaction_matrix[list(range(self.config['num_items']))].sort_index())
        item_similarity_matrix = cosine_similarity(interaction_matrix.T)
        np.fill_diagonal(item_similarity_matrix, 0)
        neighborhood = np.array([np.argpartition(row, - self.config['neighborhood'])[- self.config['neighborhood']:]
                        for row in item_similarity_matrix])
        return neighborhood, item_similarity_matrix
```

<!-- #region id="SOVjqzQzM0dJ" -->
## Utils
<!-- #endregion -->

```python id="ZcBBxGO_NBJL"
# Checkpoints
def save_checkpoint(model, model_dir):
    torch.save(model.state_dict(), model_dir)


def resume_checkpoint(model, model_dir, device_id):
    state_dict = torch.load(model_dir,
                            map_location=lambda storage, loc: storage.cuda(device=device_id))  # ensure all storage are on gpu
    model.load_state_dict(state_dict)


# Hyper params
def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)


def use_optimizer(network, params):
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(),
                                    lr=params['lr'],
                                    momentum=params['sgd_momentum'],
                                    weight_decay=params['weight_decay'])
    elif params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(),
                                                          lr=params['lr'],
                                                          weight_decay=params['weight_decay'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(),
                                        lr=params['lr'],
                                        alpha=params['rmsprop_alpha'],
                                        momentum=params['rmsprop_momentum'])
    return optimizer
```

<!-- #region id="zkaQyeVjM-Pn" -->
## Metrics
<!-- #endregion -->

```python id="C6Igzex8NKvs"
class MetronAtK(object):
    def __init__(self, top_k, loo_eval):
        self._top_k = top_k
        self.loo_eval = loo_eval
        self._subjects = None  # Subjects which we ran evaluation on

    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, top_k):
        self._top_k = top_k

    @property
    def subjects(self):
        return self._subjects

    @subjects.setter
    def subjects(self, subjects):
        assert isinstance(subjects, list)
        if self.loo_eval == True:
            test_users, test_items, test_scores = subjects[0], subjects[1], subjects[2]
            neg_users, neg_items, neg_scores = subjects[3], subjects[4], subjects[5]
            # the golden set
            test = pd.DataFrame({'user': test_users,
                                 'test_item': test_items,
                                 'test_score': test_scores})
            # the full set
            full = pd.DataFrame({'user': neg_users + test_users,
                                 'item': neg_items + test_items,
                                 'score': neg_scores + test_scores})
            full = pd.merge(full, test, on=['user'], how='left')
            # rank the items according to the scores for each user
            full['rank'] = full.groupby('user')['score'].rank(method='first', ascending=False)
            full.sort_values(['user', 'rank'], inplace=True)
            self._subjects = full
        else:
            test_users, test_items, test_true, test_output = subjects[0], subjects[1], subjects[2], subjects[3]
            # the golden set
            full = pd.DataFrame({'user': test_users,
                                 'test_item': test_items,
                                 'test_true': test_true,
                                 'test_output': test_output})

            # rank the items according to the scores for each user
            full['rank'] = full.groupby('user')['test_output'].rank(method='first', ascending=False)
            full['rank_true'] = full.groupby('user')['test_true'].rank(method='first', ascending=False)
            full.sort_values(['user', 'rank'], inplace=True)
            self._subjects = full

    def cal_ndcg(self):
        """NDCG@K for explicit evaluation"""
        full, top_k = self._subjects, self._top_k
        topp_k = full[full['rank_true'] <= top_k].copy()
        topp_k['idcg_unit'] = topp_k['rank_true'].apply(
            lambda x: math.log(2) / math.log(1 + x))  # the rank starts from 1
        topp_k['idcg'] = topp_k.groupby(['user'])['idcg_unit'].transform('sum')

        test_in_top_k = topp_k[topp_k['rank'] <= top_k].copy()
        test_in_top_k['dcg_unit'] = test_in_top_k['rank'].apply(
            lambda x: math.log(2) / math.log(1 + x))  # the rank starts from 1
        test_in_top_k['dcg'] = test_in_top_k.groupby(['user'])['dcg_unit'].transform('sum')
        test_in_top_k['ndcg'] = test_in_top_k['dcg'] / topp_k['idcg']
        ndcg = np.sum(test_in_top_k.groupby(['user'])['ndcg'].max()) / len(full['user'].unique())
        del (topp_k, test_in_top_k)
        return ndcg

    def cal_map_at_k(self):
        """MAP@K for explicit evaluation"""
        full, top_k = self._subjects, self._top_k
        users = list(dict.fromkeys(list(full['user'])))
        actual = [list(full[(full['user'] == user) & (full['rank_true'] <= top_k)]['test_item']) for user in users]
        predicted = [list(full[(full['user'] == user) & (full['rank'] <= top_k)]['test_item']) for user in users]
        return mapk(actual, predicted, k=top_k)

    def cal_hit_ratio_loo(self):
        """HR@K for Leave-One-Out evaluation"""
        full, top_k = self._subjects, self._top_k
        top_k = full[full['rank'] <= top_k]
        test_in_top_k = top_k[top_k['test_item'] == top_k['item']]  # golden items hit in the top_K items
        return len(test_in_top_k) * 1.0 / full['user'].nunique()

    def cal_ndcg_loo(self):
        """NDCG@K for Leave-One-Out evaluation"""
        full, top_k = self._subjects, self._top_k
        top_k = full[full['rank'] <= top_k]
        test_in_top_k = top_k[top_k['test_item'] == top_k['item']]
        test_in_top_k['ndcg'] = test_in_top_k['rank'].apply(
            lambda x: math.log(2) / math.log(1 + x))  # the rank starts from 1
        return test_in_top_k['ndcg'].sum() * 1.0 / full['user'].nunique()

    def cal_mep(self, explainability_matrix, theta):
        """Mean Explainability Precision at cutoff top_k and threshold theta"""
        full, top_k = self._subjects, self._top_k
        if self.loo_eval == True:
            full['exp_score'] = full[['user', 'item']].apply(lambda x: explainability_matrix[x[0], x[1]].item(), axis=1)
        else:
            full['exp_score'] = full[['user', 'test_item']].apply(lambda x: explainability_matrix[x[0], x[1]].item(), axis=1)
        full['exp_and_rec'] = ((full['exp_score'] > theta) & (full['rank'] <= top_k)) * 1
        full['topN'] = (full['rank'] <= top_k) * 1
        return np.mean(full.groupby('user')['exp_and_rec'].sum() / full.groupby('user')['topN'].sum())

    def cal_weighted_mep(self, explainability_matrix, theta):
        """Weighted Mean Explainability Precision at cutoff top_k and threshold theta"""
        full, top_k = self._subjects, self._top_k
        if self.loo_eval == True:
            full['exp_score'] = full[['user', 'item']].apply(lambda x: explainability_matrix[x[0], x[1]].item(), axis=1)
        else:
            full['exp_score'] = full[['user', 'test_item']].apply(lambda x: explainability_matrix[x[0], x[1]].item(), axis=1)
        full['exp_and_rec'] = ((full['exp_score'] > theta) & (full['rank'] <= top_k)) * 1 * (full['exp_score'])
        full['topN'] = (full['rank'] <= top_k) * 1
        return np.mean(full.groupby('user')['exp_and_rec'].sum() / full.groupby('user')['topN'].sum())

    def avg_popularity(self, popularity_vector):
        """Average popularity of top_k recommended items"""
        full, top_k = self._subjects, self._top_k
        if self.loo_eval == True:
            recommended_items = list(full.loc[full['rank'] <= top_k]['item'])
        else:
            recommended_items = list(full.loc[full['rank'] <= top_k]['test_item'])
        return np.mean([popularity_vector[i] for i in recommended_items])

    def efd(self, popularity_vector):
        """Expected Free Discovery (EFD) in top_k recommended items"""
        full, top_k = self._subjects, self._top_k
        if self.loo_eval == True:
            recommended_items = list(full.loc[full['rank'] <= top_k]['item'])
        else:
            recommended_items = list(full.loc[full['rank'] <= top_k]['test_item'])
        return np.mean([- np.log2(popularity_vector[i] + sys.float_info.epsilon) for i in recommended_items])

    def avg_pairwise_similarity(self, item_similarity_matrix):
        """Average Pairwise Similarity of top_k recommended items"""
        full, top_k = self._subjects, self._top_k
        full = full.loc[full['rank'] <= top_k]
        users = list(dict.fromkeys(list(full['user'])))
        if self.loo_eval == True:
            rec_items_for_users = [list(full.loc[full['user'] == u]['item']) for u in users]
        else:
            rec_items_for_users = [list(full.loc[full['user'] == u]['test_item']) for u in users]
            rec_items_for_users = [x for x in rec_items_for_users if len(x) > 1]
        item_combinations = [set(combinations(rec_items_for_user, 2)) for rec_items_for_user in rec_items_for_users]
        return np.mean([np.mean([item_similarity_matrix[i, j] for (i, j) in item_combinations[k]]) for k in range(len(item_combinations))])
```

<!-- #region id="LJat1W4uM9mN" -->
## Model
<!-- #endregion -->

```python id="eKGBACSgNZmS"
class Engine(object):
    """Meta Engine for training & evaluating BPR"""

    def __init__(self, config):
        self.config = config
        self._metron = MetronAtK(top_k=config['top_k'], loo_eval=self.config['loo_eval'])
        self.opt = use_optimizer(self.model, config)

    def train_single_batch_EBPR(self, users, pos_items, neg_items, ratings, explainability_matrix, popularity_vector, neighborhood):
        assert hasattr(self, 'model'), 'Please specify the exact model!'
        assert self.config['model'] in ['BPR', 'UBPR', 'EBPR', 'pUEBPR', 'UEBPR'], 'Please specify the right model!'
        if self.config['use_cuda'] is True:
            users, pos_items, neg_items, ratings = users.cuda(), pos_items.cuda(), neg_items.cuda(), ratings.cuda()
        self.opt.zero_grad()
        pos_prediction, neg_prediction = self.model(users, pos_items, neg_items)
        if self.config['model'] == 'BPR':
            loss = - (pos_prediction - neg_prediction).sigmoid().log().sum()
        elif self.config['model'] == 'UBPR':
            loss = - ((pos_prediction - neg_prediction).sigmoid().log() / popularity_vector[pos_items]).sum()
        elif self.config['model'] == 'EBPR':
            loss = - ((pos_prediction - neg_prediction).sigmoid().log() * explainability_matrix[users, pos_items] * (
                        1 - explainability_matrix[users, neg_items])).sum()
        elif self.config['model'] == 'pUEBPR':
            loss = - ((pos_prediction - neg_prediction).sigmoid().log() / popularity_vector[pos_items] *
                      explainability_matrix[users, pos_items] * (1 - explainability_matrix[users, neg_items])).sum()
        elif self.config['model'] == 'UEBPR':
            loss = - ((pos_prediction - neg_prediction).sigmoid().log() / popularity_vector[pos_items] *
                      explainability_matrix[users, pos_items] / popularity_vector[
                          neighborhood[pos_items].flatten()].view(len(pos_items), self.config['neighborhood']).sum(
                        1) * (1 - explainability_matrix[users, neg_items] / popularity_vector[
                        neighborhood[neg_items].flatten()].view(len(neg_items), self.config['neighborhood']).sum(
                        1))).sum()
        if self.config['l2_regularization'] > 0:
            l2_reg = 0
            for param in self.model.parameters():
                l2_reg += torch.norm(param)
            loss += self.config['l2_regularization'] * l2_reg
        loss.backward()
        self.opt.step()
        loss = loss.item()
        return loss

    def train_an_epoch(self, train_loader, explainability_matrix, popularity_vector, neighborhood, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model!'
        self.model.train()
        if self.config['use_cuda'] is True:
            explainability_matrix = torch.from_numpy(explainability_matrix).float().cuda()
            popularity_vector = torch.from_numpy(popularity_vector).float().cuda()
            neighborhood = torch.from_numpy(neighborhood).cuda()
        total_loss = 0
        bar = pyprind.ProgBar(len(train_loader))
        for batch_id, batch in enumerate(train_loader):
            bar.update()
            assert isinstance(batch[0], torch.LongTensor)
            user, pos_item, neg_item, rating = batch[0], batch[1], batch[2], batch[3]
            loss = self.train_single_batch_EBPR(user, pos_item, neg_item, rating, explainability_matrix, popularity_vector, neighborhood)
            total_loss += loss

    def evaluate(self, evaluate_data, explainability_matrix, popularity_vector, item_similarity_matrix, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        if self.config['loo_eval']:
            test_users_eval, test_items_eval, test_scores_eval, negative_users_eval, negative_items_eval, negative_scores_eval = [], [], [], [], [], []
        else:
            test_users_eval, test_items_eval, test_scores_eval, test_output_eval = [], [], [], []
        self.model.eval()
        with torch.no_grad():
            if self.config['loo_eval']:
                for batch_id, batch in enumerate(evaluate_data[0]):
                    test_users, test_items = batch[0], batch[1]
                    if self.config['use_cuda'] is True:
                        test_users = test_users.cuda()
                        test_items = test_items.cuda()
                    test_scores, _ = self.model(test_users, test_items, test_items)
                    if self.config['use_cuda'] is True:
                        test_users_eval += test_users.cpu().data.view(-1).tolist()
                        test_items_eval += test_items.cpu().data.view(-1).tolist()
                        test_scores_eval += test_scores.cpu().data.view(-1).tolist()
                for batch_id, batch in enumerate(evaluate_data[1]):
                    negative_users, negative_items = batch[0], batch[1]
                    if self.config['use_cuda'] is True:
                        negative_users = negative_users.cuda()
                        negative_items = negative_items.cuda()
                    negative_scores, _ = self.model(negative_users, negative_items, negative_items)
                    if self.config['use_cuda'] is True:
                        negative_users_eval += negative_users.cpu().data.view(-1).tolist()
                        negative_items_eval += negative_items.cpu().data.view(-1).tolist()
                        negative_scores_eval += negative_scores.cpu().data.view(-1).tolist()
                self._metron.subjects = [test_users_eval, test_items_eval, test_scores_eval, negative_users_eval,
                                         negative_items_eval, negative_scores_eval]
                hr, ndcg, mep, wmep, avg_pop, efd, avg_pair_sim = self._metron.cal_hit_ratio_loo(), self._metron.cal_ndcg_loo(), self._metron.cal_mep(explainability_matrix, theta=0), self._metron.cal_weighted_mep(explainability_matrix, theta=0), self._metron.avg_popularity(popularity_vector), self._metron.efd(popularity_vector), self._metron.avg_pairwise_similarity(item_similarity_matrix)
                print('Evaluating Epoch {}: NDCG@{} = {:.4f}, HR@{} = {:.4f}, MEP@{} = {:.4f}, WMEP@{} = {:.4f}, Avg_Pop@{} = {:.4f}, EFD@{} = {:.4f}, Avg_Pair_Sim@{} = {:.4f}'.format(epoch_id, self.config['top_k'],
                                                                                     ndcg, self.config['top_k'], hr, self.config['top_k'], mep, self.config['top_k'], wmep, self.config['top_k'], avg_pop, self.config['top_k'], efd, self.config['top_k'], avg_pair_sim))
                return ndcg, hr, mep, wmep, avg_pop, efd, avg_pair_sim
            else:
                for batch_id, batch in enumerate(evaluate_data):
                    test_users, test_items, test_output = batch[0], batch[1], batch[2]
                    if self.config['use_cuda'] is True:
                        test_users = test_users.cuda()
                        test_items = test_items.cuda()
                        test_output = test_output.cuda()
                    test_scores, _ = self.model(test_users, test_items, test_items)
                    if self.config['use_cuda'] is True:
                        test_users_eval += test_users.cpu().data.view(-1).tolist()
                        test_items_eval += test_items.cpu().data.view(-1).tolist()
                        test_scores_eval += test_scores.cpu().data.view(-1).tolist()
                        test_output_eval += test_output.cpu().data.view(-1).tolist()
            self._metron.subjects = [test_users_eval, test_items_eval, test_output_eval, test_scores_eval]
            map, ndcg, mep, wmep, avg_pop, efd, avg_pair_sim = self._metron.cal_map_at_k(), self._metron.cal_ndcg(), self._metron.cal_mep(explainability_matrix, theta=0), self._metron.cal_weighted_mep(explainability_matrix, theta=0), self._metron.avg_popularity(popularity_vector), self._metron.efd(popularity_vector), self._metron.avg_pairwise_similarity(item_similarity_matrix)
            print('Evaluating Epoch {}: MAP@{} = {:.4f}, NDCG@{} = {:.4f}, MEP@{} = {:.4f}, WMEP@{} = {:.4f}, Avg_Pop@{} = {:.4f}, EFD@{} = {:.4f}, Avg_Pair_Sim@{} = {:.4f}'.format(epoch_id, self.config['top_k'], map, self.config['top_k'], ndcg, self.config['top_k'], mep, self.config['top_k'], wmep, self.config['top_k'], avg_pop, self.config['top_k'], efd, self.config['top_k'], avg_pair_sim))
            return map, ndcg, mep, wmep, avg_pop, efd, avg_pair_sim

    def save_explicit(self, epoch_id, map, ndcg, mep, wmep, avg_pop, efd, avg_pair_sim, num_epoch, best_model, best_performance, save_models):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        if ndcg > best_performance[1]:
            best_performance[0] = map
            best_performance[1] = ndcg
            best_performance[2] = mep
            best_performance[3] = wmep
            best_performance[4] = avg_pop
            best_performance[5] = efd
            best_performance[6] = avg_pair_sim
            best_performance[7] = epoch_id
            best_model = self.model
        if epoch_id == num_epoch - 1:
            alias = self.config['model'] + '_' + self.config['dataset'] + '_batchsize_' + str(self.config['batch_size']) + '_opt_' + str(self.config['optimizer']) + '_lr_' + str(self.config['lr']) + '_latent_' + str(self.config['num_latent']) + '_l2reg_' + str(self.config['l2_regularization'])
            model_dir = self.config['model_dir_explicit'].format(alias, best_performance[7], self.config['top_k'], best_performance[0], self.config['top_k'], best_performance[1], self.config['top_k'], best_performance[2], self.config['top_k'], best_performance[3], self.config['top_k'], best_performance[4], self.config['top_k'], best_performance[5], self.config['top_k'], best_performance[6])
            print('Best model: ' + model_dir)
            if save_models:
                save_checkpoint(best_model, model_dir)
        return best_model, best_performance

    def save_implicit(self, epoch_id, ndcg, hr, mep, wmep, avg_pop, efd, avg_pair_sim, num_epoch, best_model, best_performance, save_models):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        if ndcg > best_performance[0]:
            best_performance[0] = ndcg
            best_performance[1] = hr
            best_performance[2] = mep
            best_performance[3] = wmep
            best_performance[4] = avg_pop
            best_performance[5] = efd
            best_performance[6] = avg_pair_sim
            best_performance[7] = epoch_id
            best_model = self.model
        if epoch_id == num_epoch - 1:
            alias = self.config['model'] + '_' + self.config['dataset'] + '_batchsize_' + str(self.config['batch_size']) + '_opt_' + str(self.config['optimizer']) + '_lr_' + str(self.config['lr']) + '_latent_' + str(self.config['num_latent']) + '_l2reg_' + str(self.config['l2_regularization'])
            model_dir = self.config['model_dir_implicit'].format(alias, best_performance[7], self.config['top_k'], best_performance[0], self.config['top_k'], best_performance[1], self.config['top_k'], best_performance[2], self.config['top_k'], best_performance[3], self.config['top_k'], best_performance[4], self.config['top_k'], best_performance[5], self.config['top_k'], best_performance[6])
            print('Best model: ' + model_dir)
            if save_models:
                save_checkpoint(best_model, model_dir)
        return best_model, best_performance

    def load_model(self, test_model_path):
        resume_checkpoint(self.model, test_model_path, self.config['device_id'])
        return self.model
```

<!-- #region id="xyWX7moONrfE" -->
EBPR model
<!-- #endregion -->

```python id="x6EO1nzANygD"
class BPR(torch.nn.Module):
    """"BPR model definition"""

    def __init__(self, config):
        super(BPR, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.num_latent = config['num_latent']
        self.loo_eval = config['loo_eval']

        self.embed_user = torch.nn.Embedding(self.num_users, self.num_latent)
        self.embed_item = torch.nn.Embedding(self.num_items, self.num_latent)

        # torch.nn.init.xavier_uniform_(self.embed_user.weight)
        # torch.nn.init.xavier_uniform_(self.embed_item.weight)
        torch.nn.init.normal_(self.embed_user.weight, std=0.01)
        torch.nn.init.normal_(self.embed_item.weight, std=0.01)

    def forward(self, user_indices, pos_item_indices, neg_item_indices):

        user_latent = self.embed_user(user_indices)
        pos_item_latent = self.embed_item(pos_item_indices)
        neg_item_latent = self.embed_item(neg_item_indices)

        pos_prediction = (user_latent * pos_item_latent).sum(dim=-1)
        neg_prediction = (user_latent * neg_item_latent).sum(dim=-1)
        return pos_prediction, neg_prediction

    def init_weight(self):
        pass
```

```python id="fkKgipJ1Nzjv"
class BPREngine(Engine):
    """Engine for training & evaluating BPR"""
    def __init__(self, config):
        self.model = BPR(config)
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        super(BPREngine, self).__init__(config)
```

<!-- #region id="gSsRqIvUM-pb" -->
## Training and Evaluation
<!-- #endregion -->

```python id="K7Fi2n2-N323"
def main(args):
    # Read dataset
    dataset_name = args.dataset  # 'ml-100k' for Movielens 100K. 'ml-1m' for the Movielens 1M dataset. 'lastfm-2k' for the
    # Last.FM 2K dataset. 'yahoo-r3' for the Yahoo! R3 dataset.
    dataset = read_data(dataset_name, args.int_per_item)

    # Define hyperparameters
    config = {'model': args.model,  # Model to train: 'BPR', 'UBPR', 'EBPR', 'pUEBPR', 'UEBPR'.
              'dataset': dataset_name,
              'num_epoch': args.num_epoch,  # Number of training epochs.
              'batch_size': args.batch_size,  # Batch size.
              'lr': args.lr,  # Learning rate.
              #'optimizer': 'sgd',
              'sgd_momentum': args.sgd_momentum,
              #'optimizer': 'rmsprop',
              'rmsprop_alpha': args.rmsprop_alpha,
              'rmsprop_momentum': args.rmsprop_momentum,
              'optimizer': args.optimizer,
              'num_users': len(dataset['userId'].unique()),
              'num_items': len(dataset['itemId'].unique()),
              'test_rate': args.test_rate,  # Test rate for random train/val/test split. test_rate is the rate of test + validation. Used when 'loo_eval' is set to False.
              'num_latent': args.num_latent,  # Number of latent factors.
              'weight_decay': args.weight_decay,
              'l2_regularization': args.l2_regularization,
              'use_cuda': args.use_cuda,
              'device_id': args.device_id,
              'top_k': args.top_k,  # k in MAP@k, HR@k and NDCG@k.
              'loo_eval': args.loo_eval,  # True: LOO evaluation with HR@k and NDCG@k. False: Random train/test split
              # evaluation with MAP@k and NDCG@k.
              'neighborhood': args.neighborhood,  # Neighborhood size for explainability.
              'model_dir_explicit':'Output/checkpoints/{}_Epoch{}_MAP@{}_{:.4f}_NDCG@{}_{:.4f}_MEP@{}_{:.4f}_WMEP@{}_{:.4f}_Avg_Pop@{}_{:.4f}_EFD@{}_{:.4f}_Avg_Pair_Sim@{}_{:.4f}.model',
              'model_dir_implicit':'Output/checkpoints/{}_Epoch{}_NDCG@{}_{:.4f}_HR@{}_{:.4f}_MEP@{}_{:.4f}_WMEP@{}_{:.4f}_Avg_Pop@{}_{:.4f}_EFD@{}_{:.4f}_Avg_Pair_Sim@{}_{:.4f}.model'}

    # DataLoader
    sample_generator = SampleGenerator(dataset, config, split_val=False)
    test_data = sample_generator.test_data_loader(config['batch_size'])

    # Create explainability matrix
    explainability_matrix = sample_generator.create_explainability_matrix()
    test_explainability_matrix = sample_generator.create_explainability_matrix(include_test=True)

    # Create popularity vector
    popularity_vector = sample_generator.create_popularity_vector()
    test_popularity_vector = sample_generator.create_popularity_vector(include_test=True)

    #Create item neighborhood
    neighborhood, item_similarity_matrix = sample_generator.create_neighborhood()
    _, test_item_similarity_matrix = sample_generator.create_neighborhood(include_test=True)

    # Specify the exact model
    engine = BPREngine(config)

    # Initialize list of optimal results
    best_performance = [0] * 8
    best_ndcg = 0

    best_model = ''
    for epoch in range(config['num_epoch']):
        print('Training epoch {}'.format(epoch))
        train_loader = sample_generator.train_data_loader(config['batch_size'])
        engine.train_an_epoch(train_loader, explainability_matrix, popularity_vector, neighborhood, epoch_id=epoch)
        if config['loo_eval']:
            ndcg, hr, mep, wmep, avg_pop, efd, avg_pair_sim = engine.evaluate(test_data, test_explainability_matrix, test_popularity_vector, test_item_similarity_matrix, epoch_id=str(epoch) + ' on test data')
            print('-' * 80)
            best_model, best_performance = engine.save_implicit(epoch, ndcg, hr, mep, wmep, avg_pop, efd, avg_pair_sim, config['num_epoch'], best_model, best_performance, save_models = args.save_models)
        else:
            map, ndcg, mep, wmep, avg_pop, efd, avg_pair_sim = engine.evaluate(test_data, test_explainability_matrix, test_popularity_vector, test_item_similarity_matrix, epoch_id=str(epoch) + ' on test data')
            print('-' * 80)
            best_model, best_performance = engine.save_explicit(epoch, map, ndcg, mep, wmep, avg_pop, efd, avg_pair_sim, config['num_epoch'], best_model, best_performance, save_models = args.save_models)
```

```python colab={"base_uri": "https://localhost:8080/"} id="IpgFPihmOA7O" executionInfo={"status": "ok", "timestamp": 1639224302192, "user_tz": -330, "elapsed": 2400988, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3b2ea639-178f-4325-b6d7-08b6cb16eac8"
#collapse-hide
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script.")

    parser.add_argument("--model", type =str, default='EBPR', help="Model to train: 'BPR', 'UBPR', 'EBPR', 'pUEBPR', "
                                                                   "'UEBPR'.")
    parser.add_argument("--dataset", type =str, default='lastfm-2k', help="'ml-100k' for Movielens 100K. 'ml-1m' for "
                                                                        "the Movielens 1M dataset. 'lastfm-2k' for "
                                                                        "the Last.FM 2K dataset. 'yahoo-r3' for the "
                                                                        "Yahoo! R3 dataset.")
    parser.add_argument("--num_epoch", type =int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type =int, default=100, help="Batch size.")
    parser.add_argument("--num_latent", type=int, default=50, help="Number of latent features.")
    parser.add_argument("--l2_regularization", type=float, default=0.0, help="L2 regularization coefficient.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay coefficient.")
    parser.add_argument("--neighborhood", type=int, default=20, help="Neighborhood size for explainability.")
    parser.add_argument("--top_k", type=int, default=10, help="Cutoff k in MAP@k, HR@k and NDCG@k, etc.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--optimizer", type=str, default='adam', help="Optimizer: 'adam', 'sgd', 'rmsprop'.")
    parser.add_argument("--sgd_momentum", type =float, default=0.9, help="Momentum for SGD optimizer.")
    parser.add_argument("--rmsprop_alpha", type =float, default=0.9, help="alpha hyperparameter for RMSProp optimizer.")
    parser.add_argument("--rmsprop_momentum", type =float, default=0.0, help="Momentum for RMSProp optimizer.")
    parser.add_argument("--loo_eval", type=lambda x: (str(x).lower() == 'true'), default=True, help="True: LOO evaluation. False: Random "
                                                                            "train/test split")
    parser.add_argument("--test_rate", type=float, default=0.2, help="Test rate for random train/val/test "
                                                                            "split. test_rate is the rate of test + "
                                                                            "validation. Used when 'loo_eval' is set "
                                                                            "to False.")
    parser.add_argument("--use_cuda", type=lambda x: (str(x).lower() == 'true'), default=True, help="True is you want to use a CUDA device.")
    parser.add_argument("--device_id", type=int, default=0, help="ID of CUDA device if 'use_cuda' is True.")
    parser.add_argument("--save_models", type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="True if you want to save the best model(s).")
    parser.add_argument("--int_per_item", type =int, default=0, help="Minimum number of interactions per item for studying effect sparsity on the lastfm-2k dataset.")

    args = parser.parse_args([])
    main(args)
```

<!-- #region id="asGQoRLcZWwk" -->
---
<!-- #endregion -->

```python id="B7757SpaZWwl"
!apt-get -qq install tree
!rm -r sample_data
```

```python colab={"base_uri": "https://localhost:8080/"} id="2FhxHJlsZWwm" executionInfo={"status": "ok", "timestamp": 1639224572299, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="80b445d4-8d81-47f8-a169-e28fef30cda5"
!tree -h --du .
```

```python colab={"base_uri": "https://localhost:8080/"} id="OwoWkAq-ZWwn" executionInfo={"status": "ok", "timestamp": 1639224585066, "user_tz": -330, "elapsed": 3666, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1074c880-414b-4c28-df89-9954b9e35e54"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="mAkZcALhZWwo" -->
---
<!-- #endregion -->

<!-- #region id="RWCkGSF0ZWwo" -->
**END**
<!-- #endregion -->
