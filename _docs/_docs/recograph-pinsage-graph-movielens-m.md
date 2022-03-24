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

<!-- #region id="KGo-kjeeHYz2" -->
# PinSage graph model based recommender
> Applying pinsage model on movielens-1m dataset

- toc: false
- badges: true
- comments: true
- categories: [graph, movie]
- image:
<!-- #endregion -->

```python id="3cJyL6GKJbfy"
%reload_ext google.colab.data_table
```

```python id="D-GeM90JLOTw"
import warnings
warnings.filterwarnings('ignore')
```

```python id="_JkgOWMtcajy"
!pip install dgl
```

```python id="gkinTrU2clkZ"
# !wget https://s3.us-west-2.amazonaws.com/dgl-data/dataset/recsys/GATNE/example.zip && unzip example.zip
# !wget https://s3.us-west-2.amazonaws.com/dgl-data/dataset/recsys/GATNE/amazon.zip && unzip amazon.zip
# !wget https://s3.us-west-2.amazonaws.com/dgl-data/dataset/recsys/GATNE/youtube.zip && unzip youtube.zip
# !wget https://s3.us-west-2.amazonaws.com/dgl-data/dataset/recsys/GATNE/twitter.zip && unzip twitter.zip
```

```python id="BhZPU2nIB_uR"
!unzip example.zip
```

```python id="RLjy0-wdCB4Y"
# !wget http://files.grouplens.org/datasets/movielens/ml-1m.zip && unzip ml-1m
```

```python colab={"base_uri": "https://localhost:8080/"} id="m88Ip1w9H7Qn" executionInfo={"status": "ok", "timestamp": 1621234814397, "user_tz": -330, "elapsed": 1470, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1c1227ec-2959-4b8d-821c-589be230eb48"
%%writefile builder.py

"""Graph builder from pandas dataframes"""
from collections import namedtuple
from pandas.api.types import is_numeric_dtype, is_categorical_dtype, is_categorical
import dgl

__all__ = ['PandasGraphBuilder']

def _series_to_tensor(series):
    if is_categorical(series):
        return torch.LongTensor(series.cat.codes.values.astype('int64'))
    else:       # numeric
        return torch.FloatTensor(series.values)

class PandasGraphBuilder(object):
    """Creates a heterogeneous graph from multiple pandas dataframes.

    Examples
    --------
    Let's say we have the following three pandas dataframes:

    User table ``users``:

    ===========  ===========  =======
    ``user_id``  ``country``  ``age``
    ===========  ===========  =======
    XYZZY        U.S.         25
    FOO          China        24
    BAR          China        23
    ===========  ===========  =======

    Game table ``games``:

    ===========  =========  ==============  ==================
    ``game_id``  ``title``  ``is_sandbox``  ``is_multiplayer``
    ===========  =========  ==============  ==================
    1            Minecraft  True            True
    2            Tetris 99  False           True
    ===========  =========  ==============  ==================

    Play relationship table ``plays``:

    ===========  ===========  =========
    ``user_id``  ``game_id``  ``hours``
    ===========  ===========  =========
    XYZZY        1            24
    FOO          1            20
    FOO          2            16
    BAR          2            28
    ===========  ===========  =========

    One could then create a bidirectional bipartite graph as follows:
    >>> builder = PandasGraphBuilder()
    >>> builder.add_entities(users, 'user_id', 'user')
    >>> builder.add_entities(games, 'game_id', 'game')
    >>> builder.add_binary_relations(plays, 'user_id', 'game_id', 'plays')
    >>> builder.add_binary_relations(plays, 'game_id', 'user_id', 'played-by')
    >>> g = builder.build()
    >>> g.number_of_nodes('user')
    3
    >>> g.number_of_edges('plays')
    4
    """
    def __init__(self):
        self.entity_tables = {}
        self.relation_tables = {}

        self.entity_pk_to_name = {}     # mapping from primary key name to entity name
        self.entity_pk = {}             # mapping from entity name to primary key
        self.entity_key_map = {}        # mapping from entity names to primary key values
        self.num_nodes_per_type = {}
        self.edges_per_relation = {}
        self.relation_name_to_etype = {}
        self.relation_src_key = {}      # mapping from relation name to source key
        self.relation_dst_key = {}      # mapping from relation name to destination key

    def add_entities(self, entity_table, primary_key, name):
        entities = entity_table[primary_key].astype('category')
        if not (entities.value_counts() == 1).all():
            raise ValueError('Different entity with the same primary key detected.')
        # preserve the category order in the original entity table
        entities = entities.cat.reorder_categories(entity_table[primary_key].values)

        self.entity_pk_to_name[primary_key] = name
        self.entity_pk[name] = primary_key
        self.num_nodes_per_type[name] = entity_table.shape[0]
        self.entity_key_map[name] = entities
        self.entity_tables[name] = entity_table

    def add_binary_relations(self, relation_table, source_key, destination_key, name):
        src = relation_table[source_key].astype('category')
        src = src.cat.set_categories(
            self.entity_key_map[self.entity_pk_to_name[source_key]].cat.categories)
        dst = relation_table[destination_key].astype('category')
        dst = dst.cat.set_categories(
            self.entity_key_map[self.entity_pk_to_name[destination_key]].cat.categories)
        if src.isnull().any():
            raise ValueError(
                'Some source entities in relation %s do not exist in entity %s.' %
                (name, source_key))
        if dst.isnull().any():
            raise ValueError(
                'Some destination entities in relation %s do not exist in entity %s.' %
                (name, destination_key))

        srctype = self.entity_pk_to_name[source_key]
        dsttype = self.entity_pk_to_name[destination_key]
        etype = (srctype, name, dsttype)
        self.relation_name_to_etype[name] = etype
        self.edges_per_relation[etype] = (src.cat.codes.values.astype('int64'), dst.cat.codes.values.astype('int64'))
        self.relation_tables[name] = relation_table
        self.relation_src_key[name] = source_key
        self.relation_dst_key[name] = destination_key

    def build(self):
        # Create heterograph
        graph = dgl.heterograph(self.edges_per_relation, self.num_nodes_per_type)
        return graph

```

```python colab={"base_uri": "https://localhost:8080/"} id="rL9meK9zIF6x" executionInfo={"status": "ok", "timestamp": 1621234848112, "user_tz": -330, "elapsed": 1070, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e32ac217-2340-4528-e4a7-82207a4ec1d9"
%%writefile data_utils.py

import torch
import dgl
import numpy as np
import scipy.sparse as ssp
import tqdm
import dask.dataframe as dd

# This is the train-test split method most of the recommender system papers running on MovieLens
# takes.  It essentially follows the intuition of "training on the past and predict the future".
# One can also change the threshold to make validation and test set take larger proportions.
def train_test_split_by_time(df, timestamp, user):
    df['train_mask'] = np.ones((len(df),), dtype=np.bool)
    df['val_mask'] = np.zeros((len(df),), dtype=np.bool)
    df['test_mask'] = np.zeros((len(df),), dtype=np.bool)
    df = dd.from_pandas(df, npartitions=10)
    def train_test_split(df):
        df = df.sort_values([timestamp])
        if df.shape[0] > 1:
            df.iloc[-1, -3] = False
            df.iloc[-1, -1] = True
        if df.shape[0] > 2:
            df.iloc[-2, -3] = False
            df.iloc[-2, -2] = True
        return df
    df = df.groupby(user, group_keys=False).apply(train_test_split).compute(scheduler='processes').sort_index()
    print(df[df[user] == df[user].unique()[0]].sort_values(timestamp))
    return df['train_mask'].to_numpy().nonzero()[0], \
           df['val_mask'].to_numpy().nonzero()[0], \
           df['test_mask'].to_numpy().nonzero()[0]

def build_train_graph(g, train_indices, utype, itype, etype, etype_rev):
    train_g = g.edge_subgraph(
        {etype: train_indices, etype_rev: train_indices},
        preserve_nodes=True)
    # remove the induced node IDs - should be assigned by model instead
    del train_g.nodes[utype].data[dgl.NID]
    del train_g.nodes[itype].data[dgl.NID]

    # copy features
    for ntype in g.ntypes:
        for col, data in g.nodes[ntype].data.items():
            train_g.nodes[ntype].data[col] = data
    for etype in g.etypes:
        for col, data in g.edges[etype].data.items():
            train_g.edges[etype].data[col] = data[train_g.edges[etype].data[dgl.EID]]

    return train_g

def build_val_test_matrix(g, val_indices, test_indices, utype, itype, etype):
    n_users = g.number_of_nodes(utype)
    n_items = g.number_of_nodes(itype)
    val_src, val_dst = g.find_edges(val_indices, etype=etype)
    test_src, test_dst = g.find_edges(test_indices, etype=etype)
    val_src = val_src.numpy()
    val_dst = val_dst.numpy()
    test_src = test_src.numpy()
    test_dst = test_dst.numpy()
    val_matrix = ssp.coo_matrix((np.ones_like(val_src), (val_src, val_dst)), (n_users, n_items))
    test_matrix = ssp.coo_matrix((np.ones_like(test_src), (test_src, test_dst)), (n_users, n_items))

    return val_matrix, test_matrix

def linear_normalize(values):
    return (values - values.min(0, keepdims=True)) / \
        (values.max(0, keepdims=True) - values.min(0, keepdims=True))
```

```python id="r9YUPFDbIWjM"
!pip install dask[dataframe]
```

```python id="nK2226CYHmCC"
"""
Script that reads from raw MovieLens-1M data and dumps into a pickle
file the following:
* A heterogeneous graph with categorical features.
* A list with all the movie titles.  The movie titles correspond to
  the movie nodes in the heterogeneous graph.
This script exemplifies how to prepare tabular data with textual
features.  Since DGL graphs do not store variable-length features, we
instead put variable-length features into a more suitable container
(e.g. torchtext to handle list of texts)
"""
```

```python id="zKlLCpOlCanU"
import os
import re
import argparse
import pickle
import pandas as pd
import numpy as np
import scipy.sparse as ssp
import dgl
import torch
import torchtext
from builder import PandasGraphBuilder
from data_utils import *
```

```python id="8ZUqdOZtI6o2"
# parser = argparse.ArgumentParser()
# parser.add_argument('directory', type=str)
# parser.add_argument('output_path', type=str)
# args = parser.parse_args()
directory = './ml-1m'
output_path = './ml-graph-data.pkl'
```

```python id="vJSbnupmI6lw"
## Build heterogeneous graph
```

```python id="JRf3Cn7KI6i5"
# Load data
users = []
with open(os.path.join(directory, 'users.dat'), encoding='latin1') as f:
    for l in f:
        id_, gender, age, occupation, zip_ = l.strip().split('::')
        users.append({
            'user_id': int(id_),
            'gender': gender,
            'age': age,
            'occupation': occupation,
            'zip': zip_,
            })
users = pd.DataFrame(users).astype('category')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 194} id="CpgWv-wLJAhV" executionInfo={"status": "ok", "timestamp": 1621235210404, "user_tz": -330, "elapsed": 1120, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7a527419-c7fa-49ed-bcce-e42db9c19553"
users.head()
```

```python id="VsD6wICdJAch"
movies = []
with open(os.path.join(directory, 'movies.dat'), encoding='latin1') as f:
    for l in f:
        id_, title, genres = l.strip().split('::')
        genres_set = set(genres.split('|'))

        # extract year
        assert re.match(r'.*\([0-9]{4}\)$', title)
        year = title[-5:-1]
        title = title[:-6].strip()

        data = {'movie_id': int(id_), 'title': title, 'year': year}
        for g in genres_set:
            data[g] = True
        movies.append(data)
movies = pd.DataFrame(movies).astype({'year': 'category'})
```

```python colab={"base_uri": "https://localhost:8080/", "height": 194} id="8eFQ-9yEJJNl" executionInfo={"status": "ok", "timestamp": 1621235248289, "user_tz": -330, "elapsed": 1027, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9af3e74b-fc9c-451a-e963-852668d707ca"
movies.head().iloc[:,:10]
```

```python id="d4XufOxiJJLA"
ratings = []
with open(os.path.join(directory, 'ratings.dat'), encoding='latin1') as f:
    for l in f:
        user_id, movie_id, rating, timestamp = [int(_) for _ in l.split('::')]
        ratings.append({
            'user_id': user_id,
            'movie_id': movie_id,
            'rating': rating,
            'timestamp': timestamp,
            })
ratings = pd.DataFrame(ratings)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 194} id="EHxuDrxsJXPU" executionInfo={"status": "ok", "timestamp": 1621235270306, "user_tz": -330, "elapsed": 1194, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="80e99e5e-09e7-4a6f-dd44-0c24aefc3476"
ratings.head()
```

```python id="I85IXVwAJXPW"
# Filter the users and items that never appear in the rating table.
distinct_users_in_ratings = ratings['user_id'].unique()
distinct_movies_in_ratings = ratings['movie_id'].unique()
users = users[users['user_id'].isin(distinct_users_in_ratings)]
movies = movies[movies['movie_id'].isin(distinct_movies_in_ratings)]
```

```python id="4hTUdj4JJXPX"
# Group the movie features into genres (a vector), year (a category), title (a string)
genre_columns = movies.columns.drop(['movie_id', 'title', 'year'])
movies[genre_columns] = movies[genre_columns].fillna(False).astype('bool')
movies_categorical = movies.drop('title', axis=1)
```

```python id="R5xWZCItJXPY"
# Build graph
graph_builder = PandasGraphBuilder()
graph_builder.add_entities(users, 'user_id', 'user')
graph_builder.add_entities(movies_categorical, 'movie_id', 'movie')
graph_builder.add_binary_relations(ratings, 'user_id', 'movie_id', 'watched')
graph_builder.add_binary_relations(ratings, 'movie_id', 'user_id', 'watched-by')
g = graph_builder.build()
```

```python id="EXnIIM2WJJGT"
# Assign features.
# Note that variable-sized features such as texts or images are handled elsewhere.
g.nodes['user'].data['gender'] = torch.LongTensor(users['gender'].cat.codes.values)
g.nodes['user'].data['age'] = torch.LongTensor(users['age'].cat.codes.values)
g.nodes['user'].data['occupation'] = torch.LongTensor(users['occupation'].cat.codes.values)
g.nodes['user'].data['zip'] = torch.LongTensor(users['zip'].cat.codes.values)

g.nodes['movie'].data['year'] = torch.LongTensor(movies['year'].cat.codes.values)
g.nodes['movie'].data['genre'] = torch.FloatTensor(movies[genre_columns].values)

g.edges['watched'].data['rating'] = torch.LongTensor(ratings['rating'].values)
g.edges['watched'].data['timestamp'] = torch.LongTensor(ratings['timestamp'].values)
g.edges['watched-by'].data['rating'] = torch.LongTensor(ratings['rating'].values)
g.edges['watched-by'].data['timestamp'] = torch.LongTensor(ratings['timestamp'].values)
```

```python colab={"base_uri": "https://localhost:8080/"} id="6-78nSGXLCKO" executionInfo={"status": "ok", "timestamp": 1621235692031, "user_tz": -330, "elapsed": 11215, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e9ffbb28-7848-4211-a186-d533b63a61f4"
# Train-validation-test split
# This is a little bit tricky as we want to select the last interaction for test, and the
# second-to-last interaction for validation.
train_indices, val_indices, test_indices = train_test_split_by_time(ratings, 'timestamp', 'user_id')
```

```python id="7osDJmd0LeeY"
# Build the graph with training interactions only.
train_g = build_train_graph(g, train_indices, 'user', 'movie', 'watched', 'watched-by')
assert train_g.out_degrees(etype='watched').min() > 0
```

```python id="pxnbc5LbLh59"
# Build the user-item sparse matrix for validation and test set.
val_matrix, test_matrix = build_val_test_matrix(g, val_indices, test_indices, 'user', 'movie', 'watched')
```

```python id="OlbHm1BMLmQo"
## Build title set
movie_textual_dataset = {'title': movies['title'].values}
```

```python id="U32UKwxzLy11"
# The model should build their own vocabulary and process the texts.  Here is one example
# of using torchtext to pad and numericalize a batch of strings.
#     field = torchtext.data.Field(include_lengths=True, lower=True, batch_first=True)
#     examples = [torchtext.data.Example.fromlist([t], [('title', title_field)]) for t in texts]
#     titleset = torchtext.data.Dataset(examples, [('title', title_field)])
#     field.build_vocab(titleset.title, vectors='fasttext.simple.300d')
#     token_ids, lengths = field.process([examples[0].title, examples[1].title])
```

```python id="q79CfTh9IQdV"
## Dump the graph and the datasets
dataset = {
    'train-graph': train_g,
    'val-matrix': val_matrix,
    'test-matrix': test_matrix,
    'item-texts': movie_textual_dataset,
    'item-images': None,
    'user-type': 'user',
    'item-type': 'movie',
    'user-to-item-type': 'watched',
    'item-to-user-type': 'watched-by',
    'timestamp-edge-column': 'timestamp'}

with open(output_path, 'wb') as f:
    pickle.dump(dataset, f)
```

```python colab={"base_uri": "https://localhost:8080/"} id="AmkRQU-rIQZ6" executionInfo={"status": "ok", "timestamp": 1621236152369, "user_tz": -330, "elapsed": 1393, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="45eebeaf-c40e-4c56-c260-6f1ce0cd5be6"
%%writefile evaluation.py

import numpy as np
import torch
import pickle
import dgl
import argparse

def prec(recommendations, ground_truth):
    n_users, n_items = ground_truth.shape
    K = recommendations.shape[1]
    user_idx = np.repeat(np.arange(n_users), K)
    item_idx = recommendations.flatten()
    relevance = ground_truth[user_idx, item_idx].reshape((n_users, K))
    hit = relevance.any(axis=1).mean()
    return hit

class LatestNNRecommender(object):
    def __init__(self, user_ntype, item_ntype, user_to_item_etype, timestamp, batch_size):
        self.user_ntype = user_ntype
        self.item_ntype = item_ntype
        self.user_to_item_etype = user_to_item_etype
        self.batch_size = batch_size
        self.timestamp = timestamp

    def recommend(self, full_graph, K, h_user, h_item):
        """
        Return a (n_user, K) matrix of recommended items for each user
        """
        graph_slice = full_graph.edge_type_subgraph([self.user_to_item_etype])
        n_users = full_graph.number_of_nodes(self.user_ntype)
        latest_interactions = dgl.sampling.select_topk(graph_slice, 1, self.timestamp, edge_dir='out')
        user, latest_items = latest_interactions.all_edges(form='uv', order='srcdst')
        # each user should have at least one "latest" interaction
        assert torch.equal(user, torch.arange(n_users))

        recommended_batches = []
        user_batches = torch.arange(n_users).split(self.batch_size)
        for user_batch in user_batches:
            latest_item_batch = latest_items[user_batch].to(device=h_item.device)
            dist = h_item[latest_item_batch] @ h_item.t()
            # exclude items that are already interacted
            for i, u in enumerate(user_batch.tolist()):
                interacted_items = full_graph.successors(u, etype=self.user_to_item_etype)
                dist[i, interacted_items] = -np.inf
            recommended_batches.append(dist.topk(K, 1)[1])

        recommendations = torch.cat(recommended_batches, 0)
        return recommendations


def evaluate_nn(dataset, h_item, k, batch_size):
    g = dataset['train-graph']
    val_matrix = dataset['val-matrix'].tocsr()
    test_matrix = dataset['test-matrix'].tocsr()
    item_texts = dataset['item-texts']
    user_ntype = dataset['user-type']
    item_ntype = dataset['item-type']
    user_to_item_etype = dataset['user-to-item-type']
    timestamp = dataset['timestamp-edge-column']

    rec_engine = LatestNNRecommender(
        user_ntype, item_ntype, user_to_item_etype, timestamp, batch_size)

    recommendations = rec_engine.recommend(g, k, None, h_item).cpu().numpy()
    return prec(recommendations, val_matrix)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('item_embedding_path', type=str)
    parser.add_argument('-k', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    with open(args.dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    with open(args.item_embedding_path, 'rb') as f:
        emb = torch.FloatTensor(pickle.load(f))
    print(evaluate_nn(dataset, emb, args.k, args.batch_size))
```

```python colab={"base_uri": "https://localhost:8080/"} id="3xmC2B58IQWf" executionInfo={"status": "ok", "timestamp": 1621236173392, "user_tz": -330, "elapsed": 1163, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e5281891-163f-4a06-c08d-42861e49ff9f"
%%writefile layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
import dgl.function as fn

def disable_grad(module):
    for param in module.parameters():
        param.requires_grad = False

def _init_input_modules(g, ntype, textset, hidden_dims):
    # We initialize the linear projections of each input feature ``x`` as
    # follows:
    # * If ``x`` is a scalar integral feature, we assume that ``x`` is a categorical
    #   feature, and assume the range of ``x`` is 0..max(x).
    # * If ``x`` is a float one-dimensional feature, we assume that ``x`` is a
    #   numeric vector.
    # * If ``x`` is a field of a textset, we process it as bag of words.
    module_dict = nn.ModuleDict()

    for column, data in g.nodes[ntype].data.items():
        if column == dgl.NID:
            continue
        if data.dtype == torch.float32:
            assert data.ndim == 2
            m = nn.Linear(data.shape[1], hidden_dims)
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
            module_dict[column] = m
        elif data.dtype == torch.int64:
            assert data.ndim == 1
            m = nn.Embedding(
                data.max() + 2, hidden_dims, padding_idx=-1)
            nn.init.xavier_uniform_(m.weight)
            module_dict[column] = m

    if textset is not None:
        for column, field in textset.fields.items():
            if field.vocab.vectors:
                module_dict[column] = BagOfWordsPretrained(field, hidden_dims)
            else:
                module_dict[column] = BagOfWords(field, hidden_dims)

    return module_dict

class BagOfWordsPretrained(nn.Module):
    def __init__(self, field, hidden_dims):
        super().__init__()

        input_dims = field.vocab.vectors.shape[1]
        self.emb = nn.Embedding(
            len(field.vocab.itos), input_dims,
            padding_idx=field.vocab.stoi[field.pad_token])
        self.emb.weight[:] = field.vocab.vectors
        self.proj = nn.Linear(input_dims, hidden_dims)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0)

        disable_grad(self.emb)

    def forward(self, x, length):
        """
        x: (batch_size, max_length) LongTensor
        length: (batch_size,) LongTensor
        """
        x = self.emb(x).sum(1) / length.unsqueeze(1).float()
        return self.proj(x)

class BagOfWords(nn.Module):
    def __init__(self, field, hidden_dims):
        super().__init__()

        self.emb = nn.Embedding(
            len(field.vocab.itos), hidden_dims,
            padding_idx=field.vocab.stoi[field.pad_token])
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, x, length):
        return self.emb(x).sum(1) / length.unsqueeze(1).float()

class LinearProjector(nn.Module):
    """
    Projects each input feature of the graph linearly and sums them up
    """
    def __init__(self, full_graph, ntype, textset, hidden_dims):
        super().__init__()

        self.ntype = ntype
        self.inputs = _init_input_modules(full_graph, ntype, textset, hidden_dims)

    def forward(self, ndata):
        projections = []
        for feature, data in ndata.items():
            if feature == dgl.NID or feature.endswith('__len'):
                # This is an additional feature indicating the length of the ``feature``
                # column; we shouldn't process this.
                continue

            module = self.inputs[feature]
            if isinstance(module, (BagOfWords, BagOfWordsPretrained)):
                # Textual feature; find the length and pass it to the textual module.
                length = ndata[feature + '__len']
                result = module(data, length)
            else:
                result = module(data)
            projections.append(result)

        return torch.stack(projections, 1).sum(1)

class WeightedSAGEConv(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, act=F.relu):
        super().__init__()

        self.act = act
        self.Q = nn.Linear(input_dims, hidden_dims)
        self.W = nn.Linear(input_dims + hidden_dims, output_dims)
        self.reset_parameters()
        self.dropout = nn.Dropout(0.5)

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.Q.weight, gain=gain)
        nn.init.xavier_uniform_(self.W.weight, gain=gain)
        nn.init.constant_(self.Q.bias, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, g, h, weights):
        """
        g : graph
        h : node features
        weights : scalar edge weights
        """
        h_src, h_dst = h
        with g.local_scope():
            g.srcdata['n'] = self.act(self.Q(self.dropout(h_src)))
            g.edata['w'] = weights.float()
            g.update_all(fn.u_mul_e('n', 'w', 'm'), fn.sum('m', 'n'))
            g.update_all(fn.copy_e('w', 'm'), fn.sum('m', 'ws'))
            n = g.dstdata['n']
            ws = g.dstdata['ws'].unsqueeze(1).clamp(min=1)
            z = self.act(self.W(self.dropout(torch.cat([n / ws, h_dst], 1))))
            z_norm = z.norm(2, 1, keepdim=True)
            z_norm = torch.where(z_norm == 0, torch.tensor(1.).to(z_norm), z_norm)
            z = z / z_norm
            return z

class SAGENet(nn.Module):
    def __init__(self, hidden_dims, n_layers):
        """
        g : DGLHeteroGraph
            The user-item interaction graph.
            This is only for finding the range of categorical variables.
        item_textsets : torchtext.data.Dataset
            The textual features of each item node.
        """
        super().__init__()

        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(WeightedSAGEConv(hidden_dims, hidden_dims, hidden_dims))

    def forward(self, blocks, h):
        for layer, block in zip(self.convs, blocks):
            h_dst = h[:block.number_of_nodes('DST/' + block.ntypes[0])]
            h = layer(block, (h, h_dst), block.edata['weights'])
        return h

class ItemToItemScorer(nn.Module):
    def __init__(self, full_graph, ntype):
        super().__init__()

        n_nodes = full_graph.number_of_nodes(ntype)
        self.bias = nn.Parameter(torch.zeros(n_nodes))

    def _add_bias(self, edges):
        bias_src = self.bias[edges.src[dgl.NID]]
        bias_dst = self.bias[edges.dst[dgl.NID]]
        return {'s': edges.data['s'] + bias_src + bias_dst}

    def forward(self, item_item_graph, h):
        """
        item_item_graph : graph consists of edges connecting the pairs
        h : hidden state of every node
        """
        with item_item_graph.local_scope():
            item_item_graph.ndata['h'] = h
            item_item_graph.apply_edges(fn.u_dot_v('h', 'h', 's'))
            item_item_graph.apply_edges(self._add_bias)
            pair_score = item_item_graph.edata['s']
        return pair_score
```

```python colab={"base_uri": "https://localhost:8080/"} id="zkv9CEXiNMSR" executionInfo={"status": "ok", "timestamp": 1621236200548, "user_tz": -330, "elapsed": 1282, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="40a83e2e-cc88-485b-d697-8b1c569eed29"
%%writefile sampler.py

import numpy as np
import dgl
import torch
from torch.utils.data import IterableDataset, DataLoader

def compact_and_copy(frontier, seeds):
    block = dgl.to_block(frontier, seeds)
    for col, data in frontier.edata.items():
        if col == dgl.EID:
            continue
        block.edata[col] = data[block.edata[dgl.EID]]
    return block

class ItemToItemBatchSampler(IterableDataset):
    def __init__(self, g, user_type, item_type, batch_size):
        self.g = g
        self.user_type = user_type
        self.item_type = item_type
        self.user_to_item_etype = list(g.metagraph()[user_type][item_type])[0]
        self.item_to_user_etype = list(g.metagraph()[item_type][user_type])[0]
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            heads = torch.randint(0, self.g.number_of_nodes(self.item_type), (self.batch_size,))
            tails = dgl.sampling.random_walk(
                self.g,
                heads,
                metapath=[self.item_to_user_etype, self.user_to_item_etype])[0][:, 2]
            neg_tails = torch.randint(0, self.g.number_of_nodes(self.item_type), (self.batch_size,))

            mask = (tails != -1)
            yield heads[mask], tails[mask], neg_tails[mask]

class NeighborSampler(object):
    def __init__(self, g, user_type, item_type, random_walk_length, random_walk_restart_prob,
                 num_random_walks, num_neighbors, num_layers):
        self.g = g
        self.user_type = user_type
        self.item_type = item_type
        self.user_to_item_etype = list(g.metagraph()[user_type][item_type])[0]
        self.item_to_user_etype = list(g.metagraph()[item_type][user_type])[0]
        self.samplers = [
            dgl.sampling.PinSAGESampler(g, item_type, user_type, random_walk_length,
                random_walk_restart_prob, num_random_walks, num_neighbors)
            for _ in range(num_layers)]

    def sample_blocks(self, seeds, heads=None, tails=None, neg_tails=None):
        blocks = []
        for sampler in self.samplers:
            frontier = sampler(seeds)
            if heads is not None:
                eids = frontier.edge_ids(torch.cat([heads, heads]), torch.cat([tails, neg_tails]), return_uv=True)[2]
                if len(eids) > 0:
                    old_frontier = frontier
                    frontier = dgl.remove_edges(old_frontier, eids)
                    #print(old_frontier)
                    #print(frontier)
                    #print(frontier.edata['weights'])
                    #frontier.edata['weights'] = old_frontier.edata['weights'][frontier.edata[dgl.EID]]
            block = compact_and_copy(frontier, seeds)
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return blocks

    def sample_from_item_pairs(self, heads, tails, neg_tails):
        # Create a graph with positive connections only and another graph with negative
        # connections only.
        pos_graph = dgl.graph(
            (heads, tails),
            num_nodes=self.g.number_of_nodes(self.item_type))
        neg_graph = dgl.graph(
            (heads, neg_tails),
            num_nodes=self.g.number_of_nodes(self.item_type))
        pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])
        seeds = pos_graph.ndata[dgl.NID]

        blocks = self.sample_blocks(seeds, heads, tails, neg_tails)
        return pos_graph, neg_graph, blocks

def assign_simple_node_features(ndata, g, ntype, assign_id=False):
    """
    Copies data to the given block from the corresponding nodes in the original graph.
    """
    for col in g.nodes[ntype].data.keys():
        if not assign_id and col == dgl.NID:
            continue
        induced_nodes = ndata[dgl.NID]
        ndata[col] = g.nodes[ntype].data[col][induced_nodes]

def assign_textual_node_features(ndata, textset, ntype):
    """
    Assigns numericalized tokens from a torchtext dataset to given block.

    The numericalized tokens would be stored in the block as node features
    with the same name as ``field_name``.

    The length would be stored as another node feature with name
    ``field_name + '__len'``.

    block : DGLHeteroGraph
        First element of the compacted blocks, with "dgl.NID" as the
        corresponding node ID in the original graph, hence the index to the
        text dataset.

        The numericalized tokens (and lengths if available) would be stored
        onto the blocks as new node features.
    textset : torchtext.data.Dataset
        A torchtext dataset whose number of examples is the same as that
        of nodes in the original graph.
    """
    node_ids = ndata[dgl.NID].numpy()

    for field_name, field in textset.fields.items():
        examples = [getattr(textset[i], field_name) for i in node_ids]

        tokens, lengths = field.process(examples)

        if not field.batch_first:
            tokens = tokens.t()

        ndata[field_name] = tokens
        ndata[field_name + '__len'] = lengths

def assign_features_to_blocks(blocks, g, textset, ntype):
    # For the first block (which is closest to the input), copy the features from
    # the original graph as well as the texts.
    assign_simple_node_features(blocks[0].srcdata, g, ntype)
    assign_textual_node_features(blocks[0].srcdata, textset, ntype)
    assign_simple_node_features(blocks[-1].dstdata, g, ntype)
    assign_textual_node_features(blocks[-1].dstdata, textset, ntype)

class PinSAGECollator(object):
    def __init__(self, sampler, g, ntype, textset):
        self.sampler = sampler
        self.ntype = ntype
        self.g = g
        self.textset = textset

    def collate_train(self, batches):
        heads, tails, neg_tails = batches[0]
        # Construct multilayer neighborhood via PinSAGE...
        pos_graph, neg_graph, blocks = self.sampler.sample_from_item_pairs(heads, tails, neg_tails)
        assign_features_to_blocks(blocks, self.g, self.textset, self.ntype)

        return pos_graph, neg_graph, blocks

    def collate_test(self, samples):
        batch = torch.LongTensor(samples)
        blocks = self.sampler.sample_blocks(batch)
        assign_features_to_blocks(blocks, self.g, self.textset, self.ntype)
        return blocks
```

```python id="3n-wpT1VNWam"
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchtext
import dgl
import tqdm

import layers
import sampler as sampler_module
import evaluation
```

```python id="dBZNZjVzNYtH"
class PinSAGEModel(nn.Module):
    def __init__(self, full_graph, ntype, textsets, hidden_dims, n_layers):
        super().__init__()

        self.proj = layers.LinearProjector(full_graph, ntype, textsets, hidden_dims)
        self.sage = layers.SAGENet(hidden_dims, n_layers)
        self.scorer = layers.ItemToItemScorer(full_graph, ntype)

    def forward(self, pos_graph, neg_graph, blocks):
        h_item = self.get_repr(blocks)
        pos_score = self.scorer(pos_graph, h_item)
        neg_score = self.scorer(neg_graph, h_item)
        return (neg_score - pos_score + 1).clamp(min=0)

    def get_repr(self, blocks):
        h_item = self.proj(blocks[0].srcdata)
        h_item_dst = self.proj(blocks[-1].dstdata)
        return h_item_dst + self.sage(blocks, h_item)
```

```python id="GX-PImUXNuZZ"
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='./ml-graph-data.pkl')
parser.add_argument('--random-walk-length', type=int, default=2)
parser.add_argument('--random-walk-restart-prob', type=float, default=0.5)
parser.add_argument('--num-random-walks', type=int, default=10)
parser.add_argument('--num-neighbors', type=int, default=3)
parser.add_argument('--num-layers', type=int, default=2)
parser.add_argument('--hidden-dims', type=int, default=16)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--device', type=str, default='cpu')        # can also be "cuda:0"
parser.add_argument('--num-epochs', type=int, default=1)
parser.add_argument('--batches-per-epoch', type=int, default=20000)
parser.add_argument('--num-workers', type=int, default=0)
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('-k', type=int, default=10)
args, unknown = parser.parse_known_args()
```

```python id="Pt3zREy8Qtcl"
# Load dataset
with open(args.dataset_path, 'rb') as f:
    dataset = pickle.load(f)
```

```python id="RO7FR1jnQ2ok"
g = dataset['train-graph']
val_matrix = dataset['val-matrix'].tocsr()
test_matrix = dataset['test-matrix'].tocsr()
item_texts = dataset['item-texts']
user_ntype = dataset['user-type']
item_ntype = dataset['item-type']
user_to_item_etype = dataset['user-to-item-type']
timestamp = dataset['timestamp-edge-column']
```

```python id="uPUJQ48eQ4rf"
device = torch.device(args.device)
```

```python id="OXlNFiRcQ8N2"
# Assign user and movie IDs and use them as features (to learn an individual trainable
# embedding for each entity)
g.nodes[user_ntype].data['id'] = torch.arange(g.number_of_nodes(user_ntype))
g.nodes[item_ntype].data['id'] = torch.arange(g.number_of_nodes(item_ntype))
```

```python id="Ps7F5tajQ8JN"
# Prepare torchtext dataset and vocabulary
fields = {}
examples = []
for key, texts in item_texts.items():
    fields[key] = torchtext.legacy.data.Field(include_lengths=True, lower=True, batch_first=True)
for i in range(g.number_of_nodes(item_ntype)):
    example = torchtext.legacy.data.Example.fromlist(
        [item_texts[key][i] for key in item_texts.keys()],
        [(key, fields[key]) for key in item_texts.keys()])
    examples.append(example)
textset = torchtext.legacy.data.Dataset(examples, fields)
for key, field in fields.items():
    field.build_vocab(getattr(textset, key))
    #field.build_vocab(getattr(textset, key), vectors='fasttext.simple.300d')
```

```python id="2hSz7EwMNYoc"
# Sampler
batch_sampler = sampler_module.ItemToItemBatchSampler(g, user_ntype, item_ntype, args.batch_size)
neighbor_sampler = sampler_module.NeighborSampler(g, user_ntype, item_ntype, args.random_walk_length,
                                                  args.random_walk_restart_prob, args.num_random_walks,
                                                  args.num_neighbors, args.num_layers)
collator = sampler_module.PinSAGECollator(neighbor_sampler, g, item_ntype, textset)
dataloader = DataLoader(batch_sampler, collate_fn=collator.collate_train, num_workers=args.num_workers)
dataloader_test = DataLoader(torch.arange(g.number_of_nodes(item_ntype)), batch_size=args.batch_size,
                             collate_fn=collator.collate_test, num_workers=args.num_workers)
dataloader_it = iter(dataloader)
```

```python id="B4r309MtNYim"
# Model
model = PinSAGEModel(g, item_ntype, textset, args.hidden_dims, args.num_layers).to(device)
```

```python id="R7WgJFFsSI-R"
# Optimizer
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
```

```python colab={"base_uri": "https://localhost:8080/"} id="_NsYPL1bSseZ" executionInfo={"status": "ok", "timestamp": 1621238437494, "user_tz": -330, "elapsed": 114798, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="dc66e775-67e8-4a01-b305-85e1ec5a6bf9"
# For each batch of head-tail-negative triplets...
for epoch_id in range(args.num_epochs):
    model.train()
    for batch_id in tqdm.trange(args.batches_per_epoch):
        pos_graph, neg_graph, blocks = next(dataloader_it)
        # Copy to GPU
        for i in range(len(blocks)):
            blocks[i] = blocks[i].to(device)
        pos_graph = pos_graph.to(device)
        neg_graph = neg_graph.to(device)

        loss = model(pos_graph, neg_graph, blocks).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
```

```python colab={"base_uri": "https://localhost:8080/"} id="KSe9dRhQS0t_" executionInfo={"status": "ok", "timestamp": 1621238440227, "user_tz": -330, "elapsed": 7378, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="37fa6835-78d9-4c96-8bc4-b76156e4a060"
# Evaluate HIT@10
model.eval()
with torch.no_grad():
    item_batches = torch.arange(g.number_of_nodes(item_ntype)).split(args.batch_size)
    h_item_batches = []
    for blocks in dataloader_test:
        for i in range(len(blocks)):
            blocks[i] = blocks[i].to(device)

        h_item_batches.append(model.get_repr(blocks))
    h_item = torch.cat(h_item_batches, 0)

    print(evaluation.evaluate_nn(dataset, h_item, args.k, args.batch_size))
```

<!-- #region id="xE3cBBjIVev-" -->
https://github.com/dmlc/dgl/tree/master/examples/pytorch/pinsage
<!-- #endregion -->
