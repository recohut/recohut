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

```python id="3cJyL6GKJbfy"
%reload_ext google.colab.data_table
```

```python id="D-GeM90JLOTw"
import warnings
warnings.filterwarnings('ignore')
```

```python id="_JkgOWMtcajy" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1621242857176, "user_tz": -330, "elapsed": 6732, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="784d8161-05a9-4ef1-bae7-e5a675069bb7"
!pip install dgl
```

```python id="RLjy0-wdCB4Y" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1621243017301, "user_tz": -330, "elapsed": 154108, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="68e0bb9c-4235-4c8e-961e-4406a47a99fc"
!wget https://zenodo.org/record/3248543/files/nowplayingrs.zip?download=1
!unzip /content/nowplayingrs.zip?download=1 -d /content
```

```python colab={"base_uri": "https://localhost:8080/"} id="m88Ip1w9H7Qn" executionInfo={"status": "ok", "timestamp": 1621243017304, "user_tz": -330, "elapsed": 153111, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="58af2a4b-09d9-434d-ebdd-70a759e52972"
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

```python colab={"base_uri": "https://localhost:8080/"} id="rL9meK9zIF6x" executionInfo={"status": "ok", "timestamp": 1621243017306, "user_tz": -330, "elapsed": 152786, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6f6b070f-242f-49b5-bcce-d48e25be961a"
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

```python id="r9YUPFDbIWjM" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1621243023050, "user_tz": -330, "elapsed": 157861, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9438e18b-f222-4000-94b1-d7c687a5820f"
!pip install dask[dataframe]
```

```python id="nK2226CYHmCC" colab={"base_uri": "https://localhost:8080/", "height": 103} executionInfo={"status": "ok", "timestamp": 1621243023053, "user_tz": -330, "elapsed": 157381, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="424153d5-6e82-4e00-d887-f3649ec7bbb0"
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

```python id="zKlLCpOlCanU" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1621243027159, "user_tz": -330, "elapsed": 161284, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bacb223f-ccb8-49c7-a46a-e71c9cafe37f"
import os
import argparse
import pandas as pd
import scipy.sparse as ssp
import pickle
from builder import PandasGraphBuilder
from data_utils import *
```

```python id="8ZUqdOZtI6o2"
# parser = argparse.ArgumentParser()
# parser.add_argument('directory', type=str)
# parser.add_argument('output_path', type=str)
# args = parser.parse_args()
directory = './nowplaying_rs_dataset'
output_path = './nowplaying-graph-data.pkl'
```

```python id="vJSbnupmI6lw"
## Build heterogeneous graph
```

```python id="JRf3Cn7KI6i5"
data = pd.read_csv(os.path.join(directory, 'context_content_features.csv'))
track_feature_cols = list(data.columns[1:13])
data = data[['user_id', 'track_id', 'created_at'] + track_feature_cols].dropna()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 371} id="TDptSUWlaWVz" executionInfo={"status": "ok", "timestamp": 1621243062918, "user_tz": -330, "elapsed": 195848, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="70488742-fca9-4c29-ca71-4c72e0b624e2"
data.head()
```

```python id="VHJuPW2_achi"
users = data[['user_id']].drop_duplicates()
tracks = data[['track_id'] + track_feature_cols].drop_duplicates()
assert tracks['track_id'].value_counts().max() == 1
tracks = tracks.astype({'mode': 'int64', 'key': 'int64', 'artist_id': 'category'})
events = data[['user_id', 'track_id', 'created_at']]
events['created_at'] = events['created_at'].values.astype('datetime64[s]').astype('int64')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 194} id="0_2dzxwMadpu" executionInfo={"status": "ok", "timestamp": 1621243072439, "user_tz": -330, "elapsed": 203425, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7b2a17ba-a04e-40aa-a037-9ae4036d0b2b"
events.head()
```

```python id="p6cjVytial-L"
graph_builder = PandasGraphBuilder()
graph_builder.add_entities(users, 'user_id', 'user')
graph_builder.add_entities(tracks, 'track_id', 'track')
graph_builder.add_binary_relations(events, 'user_id', 'track_id', 'listened')
graph_builder.add_binary_relations(events, 'track_id', 'user_id', 'listened-by')

g = graph_builder.build()
```

```python id="q5GiN3Cwaoc_"
float_cols = []
for col in tracks.columns:
    if col == 'track_id':
        continue
    elif col == 'artist_id':
        g.nodes['track'].data[col] = torch.LongTensor(tracks[col].cat.codes.values)
    elif tracks.dtypes[col] == 'float64':
        float_cols.append(col)
    else:
        g.nodes['track'].data[col] = torch.LongTensor(tracks[col].values)
g.nodes['track'].data['song_features'] = torch.FloatTensor(linear_normalize(tracks[float_cols].values))
g.edges['listened'].data['created_at'] = torch.LongTensor(events['created_at'].values)
g.edges['listened-by'].data['created_at'] = torch.LongTensor(events['created_at'].values)
```

```python colab={"base_uri": "https://localhost:8080/"} id="One7WKGNaqON" executionInfo={"status": "ok", "timestamp": 1621243457360, "user_tz": -330, "elapsed": 587617, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7263f632-025e-4856-dae2-300f56c50041"
n_edges = g.number_of_edges('listened')
train_indices, val_indices, test_indices = train_test_split_by_time(events, 'created_at', 'user_id')
train_g = build_train_graph(g, train_indices, 'user', 'track', 'listened', 'listened-by')
assert train_g.out_degrees(etype='listened').min() > 0
val_matrix, test_matrix = build_val_test_matrix(
    g, val_indices, test_indices, 'user', 'track', 'listened')
```

```python id="n8UYEBV6aFh7"
dataset = {
    'train-graph': train_g,
    'val-matrix': val_matrix,
    'test-matrix': test_matrix,
    'item-texts': {},
    'item-images': None,
    'user-type': 'user',
    'item-type': 'track',
    'user-to-item-type': 'listened',
    'item-to-user-type': 'listened-by',
    'timestamp-edge-column': 'created_at'}

with open(output_path, 'wb') as f:
    pickle.dump(dataset, f)
```

```python colab={"base_uri": "https://localhost:8080/"} id="AmkRQU-rIQZ6" executionInfo={"status": "ok", "timestamp": 1621243460434, "user_tz": -330, "elapsed": 590372, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7e7c7f51-3627-4c5b-c69e-17f88d281f47"
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

```python colab={"base_uri": "https://localhost:8080/"} id="3xmC2B58IQWf" executionInfo={"status": "ok", "timestamp": 1621243460436, "user_tz": -330, "elapsed": 590254, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d574d01f-e5c2-43a6-b209-9707fc602f21"
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

```python colab={"base_uri": "https://localhost:8080/"} id="zkv9CEXiNMSR" executionInfo={"status": "ok", "timestamp": 1621243460439, "user_tz": -330, "elapsed": 589782, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2250b478-f498-43b4-95c8-9f678bd7dfd9"
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

```python id="IZFUHO4blk4C"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

```python id="GX-PImUXNuZZ"
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default=output_path)
parser.add_argument('--random-walk-length', type=int, default=2)
parser.add_argument('--random-walk-restart-prob', type=float, default=0.5)
parser.add_argument('--num-random-walks', type=int, default=10)
parser.add_argument('--num-neighbors', type=int, default=3)
parser.add_argument('--num-layers', type=int, default=2)
parser.add_argument('--hidden-dims', type=int, default=16)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--device', type=str, default=device)        # can also be "cuda:0"
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

```python colab={"base_uri": "https://localhost:8080/"} id="_NsYPL1bSseZ" executionInfo={"status": "ok", "timestamp": 1621247042077, "user_tz": -330, "elapsed": 4156985, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="77d12dc9-a3e7-44f0-9308-fdecb5b4e2d8"
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

```python colab={"base_uri": "https://localhost:8080/"} id="KSe9dRhQS0t_" executionInfo={"status": "ok", "timestamp": 1621247784222, "user_tz": -330, "elapsed": 624762, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c3be1fcb-fdca-497d-8dd1-7301b0a81c12"
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
