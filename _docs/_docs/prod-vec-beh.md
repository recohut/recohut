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

<!-- #region id="3rq9HpuMhtIR" -->
# Behavioral testing and evaluation of the Prod2vec model on Coveo dataset
<!-- #endregion -->

<!-- #region id="XQ37oedJhL1c" -->
Metrics like Hit Rate provide only a partial picture of the expected behavior of recommenders in the wild: two models with very similar accuracy can have very different behavior on, say, the long-tail, or model A can be better than model B overall, but at the expense of providing disastrous performance on a set of inputs that are particularly important in production. Metrics such as coverage, serendipity, and bias have been therefore proposed to capture other aspects of the behaviors of RSs, but they still fall short of what is needed to debug RSs in production, or provide any guarantee that a model will be reliable when released.

**Dataset** - This tutorial shows how easy it is to run behavioral tests on a target dataset, in this case a wrapper around a large e-commerce dataset (the [Coveo Data Challenge dataset](https://github.com/coveooss/SIGIR-ecom-data-challenge)).

**Use case** - Complementary items. We are targeting a "complementary items" use cases, such as for example a cart recommender. *if a shopper added item X to the cart, what is she likely to add next?*

**Model** - We train a simple, yet effective prod2vec baseline [model](https://arxiv.org/abs/2007.14906), re-using for convenience a "training embedding" function. Word2vec algorithm is used for embedding over product SKU sequences.
<!-- #endregion -->

<!-- #region id="3dhtPqgRhPLC" -->
### Flow diagram
![](data:image/png;base64,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)

*Sample workflow for behavioral tests. Starting with shopping data (left), the dataset split and model training mimic the usual training loop. We also create a latent space, which is used to measure the relationships between inputs, ground truths and predictions, such as how far misses are from ground truths. Since a session can be viewed as a sequence of items or features (brands), we can use the same method to create embeddings for different tests.*
<!-- #endregion -->

<!-- #region id="O8ZUJHMrho_A" -->
### Inputs and outputs

**Inputs:**

1. Model `RecModel`
2. Dataset `RecDataset`
    - Public datasets - Coveo, MSD, MovieLens
3. Use Case `RecTask`
    - Similar items - {GT:👞} → {Pred:👟} is ✅, but {GT:👞} → {Pred:🩳} is 🟥
    - Complementary items - {GT:💻} → {Pred:⌨️} is ✅, but {GT:⌨️} → {Pred:💻} is 🟥
    - Session-based recommendations
4. Behavioral Tests `RecTest`
5. List of behavioral tests `RecList`

**Outputs:**

- Results of behavioral tests.

### Modules

- **P2VRecModel**: is a very basic recommender that builds a space of products and use Knn to run predictions
- **CoveoDataset**: is the dataset we are going to play with
- **train_embeddings**: is a function that allows us to train prod2vec embeddings
- **CoveoCartRecList**: is the RecList! this is a pre-made RecList for you to evaluate your models (or our P2VRecModel, in this tutorial) on the Coveo Dataset.
<!-- #endregion -->

<!-- #region id="-NfELzGNIo2U" -->
## Setup
<!-- #endregion -->

```python id="7pTDbLiLPu_w"
!pip install gensim==4.0.1
!pip install jinja2==3.0.2
!pip install algoliasearch==2.6.0
!pip install appdirs==1.4.4
!pip install wget==3.2
!pip install pytest==6.2.5
!pip install requests==2.22.0
!pip install tqdm==4.62.3
!pip install matplotlib==3.4.3
!pip install numpy==1.21.2
!pip install pathos==0.2.8
!pip install flask==2.0.2
!pip install networkx==2.6.3
!pip install python-Levenshtein==0.12.2
```

```python id="4kCfrlxSIpU1"
import gensim
from abc import ABC, abstractmethod
import ast
from datetime import datetime
import inspect
import os
from abc import ABC, abstractmethod
from functools import wraps
from pathlib import Path
import time
import json
import tempfile
import zipfile
import random
import requests
from tqdm import tqdm
from enum import Enum
from typing import List
import itertools
import numpy as np
import collections
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from statistics import mean
import json
import networkx as nx
from networkx.algorithms.shortest_paths.generic import shortest_path
from collections import Counter, defaultdict
import math
```

<!-- #region id="fKz7N6DDKSAD" -->
## Dataset
<!-- #endregion -->

```python id="EXtGJMySIpXi"
COVEO_INTERACTION_DATASET_S3_URL = 'https://reclist-datasets-6d3c836d-6djh887d.s3.us-west-2.amazonaws.com/coveo_sigir.zip'
```

```python id="qo2bINDhKE2o"
def download_with_progress(url, destination):
    """
    Downloads a file with a progress bar
    :param url: url from which to download from
    :destination: file path for saving data
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise SystemExit(e)
    with tqdm.wrapattr(open(destination, "wb"), "write",
                       miniters=1, desc=url.split('/')[-1],
                       total=int(response.headers.get('content-length', 0))) as fout:
        for chunk in response.iter_content(chunk_size=4096):
            fout.write(chunk)


def get_cache_directory():
    """
    Returns the cache directory on the system
    """
    cache_dir = '/content/coveo_reclist'

    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    return cache_dir


class Dataset(Enum):
    COVEO = 'coveo'
    COVEO_INTERNAL = 'coveo-internal'
```

```python id="uteUWIghIpcg"
class RecDataset(ABC):
    """
    Implements an abstract class for the dataset
    """
    def __init__(self, force_download=False):
        """
        :param force_download: allows to force the download of the dataset in case it is needed.
        :type: force_download: bool, optional
        """
        self._x_train = None
        self._y_train = None
        self._x_test = None
        self._y_test = None
        self._catalog = None
        self.force_download = force_download
        self.load()

    @abstractmethod
    def load(self):
        """
        Abstract method that should implement dataset loading
        @return:
        """
        return

    @property
    def x_train(self):
        return self._x_train

    @property
    def y_train(self):
        return self._y_train

    @property
    def x_test(self):
        return self._x_test

    @property
    def y_test(self):
        return self._y_test

    @property
    def catalog(self):
        return self._catalog
```

```python id="iPzJOfxPIpg4"
class CoveoDataset(RecDataset):
    """
    Coveo SIGIR data challenge dataset
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self):
        cache_directory = get_cache_directory()
        filename = os.path.join(cache_directory, "coveo_sigir.zip")  # TODO: make var somewhere

        if not os.path.exists(filename) or self.force_download:
            download_with_progress(COVEO_INTERACTION_DATASET_S3_URL, filename)

        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            with open(os.path.join(temp_dir, 'dataset.json')) as f:
                data = json.load(f)

        self._x_train = data["x_train"]
        self._y_train = None
        self._x_test = data["x_test"]
        self._y_test = data["y_test"]
        self._catalog = data["catalog"]
```

<!-- #region id="yp337k5NKXr8" -->
## Model
<!-- #endregion -->

```python id="WxUtXx9YIpaF"
def train_embeddings(
    sessions: list,
    min_c: int = 3,
    size: int = 48,
    window: int = 5,
    iterations: int = 15,
    ns_exponent: float = 0.75,
    is_debug: bool = True):
    """
    Train CBOW to get product embeddings with sensible defaults (https://arxiv.org/abs/2007.14906).
    :param sessions: list of lists, as user sessions are list of interactions
    :param min_c: minimum frequency of an event for it to be calculated for product embeddings
    :param size: output dimension
    :param window: window parameter for gensim word2vec
    :param iterations: number of training iterations
    :param ns_exponent: ns_exponent parameter for gensim word2vec
    :param is_debug: if true, be more verbose when training
    :return: trained product embedding model
    """
    model = gensim.models.Word2Vec(sentences=sessions,
                                   min_count=min_c,
                                   vector_size=size,
                                   window=window,
                                   epochs=iterations,
                                   ns_exponent=ns_exponent)

    if is_debug:
        print("# products in the space: {}".format(len(model.wv.index_to_key)))

    return model.wv
```

```python id="DpKCy3pkJKdS"
class RecModel(ABC):
    """
    Abstract class for recommendation model
    """

    def __init__(self, model=None):
        """
        :param model: a model that can be used in the predict function
        """
        self._model = model

    @abstractmethod
    def predict(self, prediction_input: list, *args, **kwargs):
        """
        The predict function should implement the behaviour of the model at inference time.
        :param prediction_input: the input that is used to to do the prediction
        :param args:
        :param kwargs:
        :return:
        """
        return NotImplementedError

    @property
    def model(self):
        return self._model
```

```python id="2d4nsWPtK5_e"
class P2VRecModel(RecModel):
    """
    Implement of the prod2vec model through the standard RecModel interface.
    Since init is ok, we just need to overwrite the prediction methods to get predictions
    out of it.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    model_name = "prod2vec"
    def predict(self, prediction_input: list, *args, **kwargs):
        """
        Implement the abstract method, accepting a list of lists, each list being
        the content of a cart: the predictions returned by the model are the top K
        items suggested to complete the cart.
        """
        predictions = []
        for _x in prediction_input:
            # we assume here that every X is a list of one-element, the product already in the cart
            # i.e. our prediction_input list is [[sku_1], [sku_3], ...]
            key_item = _x[0]['product_sku']
            nn_products = self._model.most_similar(key_item, topn=10) if key_item in self._model else None
            if nn_products:
                predictions.append([{'product_sku':_[0]} for _ in nn_products])
            else:
                predictions.append([])

        return predictions

    def get_vector(self, product_sku):
        try:
            return list(self._model.get_vector(product_sku))
        except Exception as e:
            return []
```

<!-- #region id="1PdrG6g5LON2" -->
## Metrics
<!-- #endregion -->

<!-- #region id="a4jTjpDbVPxJ" -->
### Standard Metrics
<!-- #endregion -->

```python id="mY8Fb_W4LN9x"
def statistics(x_train, y_train, x_test, y_test, y_pred):
    train_size = len(x_train)
    test_size = len(x_test)
    # num non-zero preds
    num_preds = len([p for p in y_pred if p])
    return {
        'training_set__size': train_size,
        'test_set_size': test_size,
        'num_non_null_predictions': num_preds
    }


def sample_hits_at_k(y_preds, y_test, x_test=None, k=3, size=3):
    hits = []
    for idx, (_p, _y) in enumerate(zip(y_preds, y_test)):
        if _y[0] in _p[:k]:
            hit_info = {
                'Y_TEST': [_y[0]],
                'Y_PRED': _p[:k],
            }
            if x_test:
                hit_info['X_TEST'] = [x_test[idx][0]]
            hits.append(hit_info)

    if len(hits) < size or size == -1:
        return hits
    return random.sample(hits, k=size)


def sample_misses_at_k(y_preds, y_test, x_test=None, k=3, size=3):
    misses = []
    for idx, (_p, _y) in enumerate(zip(y_preds, y_test)):
        if _y[0] not in _p[:k]:
            miss_info =  {
                'Y_TEST': [_y[0]],
                'Y_PRED': _p[:k],
            }
            if x_test:
                miss_info['X_TEST'] = [x_test[idx][0]]
            misses.append(miss_info)

    if len(misses) < size or size == -1:
        return misses
    return random.sample(misses, k=size)


def hit_rate_at_k(y_preds, y_test, k=3):
    hits = 0
    for _p, _y in zip(y_preds, y_test):
        if _y[0] in _p[:k]:
            hits += 1
    return hits / len(y_test)


def mrr_at_k(y_preds, y_test, k=3):
    """
    Computes MRR
    :param y_preds: predictions, as lists of lists
    :param y_test: target data, as lists of lists (eventually [[sku1], [sku2],...]
    :param k: top-k
    """
    rr = []
    y_test = [k[0] for k in y_test]
    for _p, _y in zip(y_preds, y_test):
        if _y in _p[:k+1]:
            rank = _p[:k+1].index(_y) + 1
            rr.append(1/rank)
        else:
            rr.append(0)
    return np.mean(rr)


def coverage_at_k(y_preds, product_data, k=3):
    pred_skus = set(itertools.chain.from_iterable(y_preds[:k]))
    all_skus = set(product_data.keys())
    nb_overlap_skus = len(pred_skus.intersection(all_skus))

    return nb_overlap_skus / len(all_skus)


def popularity_bias_at_k(y_preds, x_train, k=3):
    # estimate popularity from training data
    pop_map = collections.defaultdict(lambda : 0)
    num_interactions = 0
    for session in x_train:
        for event in session:
            pop_map[event] += 1
            num_interactions += 1
    # normalize popularity
    pop_map = {k:v/num_interactions for k,v in pop_map.items()}
    all_popularity = []
    for p in y_preds:
        average_pop = sum(pop_map.get(_, 0.0) for _ in p[:k]) / len(p) if len(p) > 0 else 0
        all_popularity.append(average_pop)
    return sum(all_popularity) / len(y_preds)
```

<!-- #region id="Ydqnq1OIVUIF" -->
### Cosine Distance Metrics
<!-- #endregion -->

```python id="yaCxpR7bLlWf"
def error_by_cosine_distance(model, y_test, y_preds, k=3, bins=25, debug=False):
    if not(hasattr(model.__class__, 'get_vector') and callable(getattr(model.__class__, 'get_vector'))):
        error_msg = "Error : Model {} does not support retrieval of vector embeddings".format(model.__class__)
        print(error_msg)
        return error_msg
    misses = sample_misses_at_k(y_preds, y_test, k=k, size=-1)
    cos_distances = []
    for m in misses:
        if m['Y_PRED']:
            vector_test = model.get_vector(m['Y_TEST'][0])
            vector_pred = model.get_vector(m['Y_PRED'][0])
            if vector_pred and vector_test:
                cos_dist = cosine(vector_pred, vector_test)
                cos_distances.append(cos_dist)

    histogram = np.histogram(cos_distances, bins=bins, density=False)
    # cast to list
    histogram = (histogram[0].tolist(), histogram[1].tolist())
    # debug / viz
    if debug:
        plt.hist(cos_distances, bins=bins)
        plt.title('dist over cosine distance prod space')
        plt.show()

    return {'mean': np.mean(cos_distances), 'histogram': histogram}

def distance_to_query(model, x_test, y_test, y_preds, k=3, bins=25, debug=False):
    if not(hasattr(model.__class__, 'get_vector') and callable(getattr(model.__class__, 'get_vector'))):
        error_msg = "Error : Model {} does not support retrieval of vector embeddings".format(model.__class__)
        print(error_msg)
        return error_msg
    misses = sample_misses_at_k(y_preds, y_test, x_test=x_test, k=k, size=-1)
    x_to_y_cos = []
    x_to_p_cos = []
    for m in misses:
        if m['Y_PRED']:
            vector_x = model.get_vector(m['X_TEST'][0])
            vector_y = model.get_vector(m['Y_TEST'][0])
            vectors_p = [model.get_vector(_) for _ in m['Y_PRED']]
            c_dists =[]
            if not vector_x or not vector_y:
                continue
            x_to_y_cos.append(cosine(vector_x, vector_y))
            for v_p in vectors_p:
                if not v_p:
                    continue
                cos_dist = cosine(v_p, vector_x)
                if cos_dist:
                    c_dists.append(cos_dist)
            if c_dists:
                x_to_p_cos.append(mean(c_dists))

    h_xy = np.histogram(x_to_y_cos, bins=bins, density=False)
    h_xp = np.histogram(x_to_p_cos, bins=bins, density=False)

    h_xy = (h_xy[0].tolist(), h_xy[1].tolist())
    h_xp = (h_xp[0].tolist(), h_xp[1].tolist())

    # debug / viz
    if debug:
        plt.hist(x_to_y_cos, bins=bins, alpha=0.5)
        plt.hist(x_to_p_cos, bins=bins, alpha=0.5)
        plt.title('distribution of distance to input')
        plt.legend(['Distance from Input to Label',
                    'Distance from Input to Label'],
                   loc='upper right')
        plt.show()

    return {
        'histogram_x_to_y': h_xy,
        'histogram_x_to_p': h_xp,
        'raw_distances_x_to_y': x_to_y_cos,
        'raw_distances_x_to_p': x_to_p_cos,
    }
```

<!-- #region id="GhZatSAhVWq_" -->
### Generic Cosine Distance
<!-- #endregion -->

```python id="5DOZxRQyLqRk"
def generic_cosine_distance(embeddings: dict,
                            type_fn,
                            y_test,
                            y_preds,
                            k=10,
                            bins=25,
                            debug=False):

    misses = sample_misses_at_k(y_preds, y_test, k=k, size=-1)
    cos_distances = []
    for m in misses:
        if m['Y_TEST'] and m['Y_PRED'] and type_fn(m['Y_TEST'][0]) and type_fn(m['Y_PRED'][0]):
            vector_test = embeddings.get(type_fn(m['Y_TEST'][0]), None)
            vector_pred = embeddings.get(type_fn(m['Y_PRED'][0]), None)
            if vector_pred and vector_test:
                cos_dist = cosine(vector_pred, vector_test)
                cos_distances.append(cos_dist)

    # TODO: Maybe sample some examples from the bins
    histogram = np.histogram(cos_distances, bins=bins, density=False)
    # cast to list
    histogram = (histogram[0].tolist(), histogram[1].tolist())
    # debug / viz
    if debug:
        plt.hist(cos_distances, bins=bins)
        plt.title('cosine distance misses')
        plt.show()
    return {'mean': np.mean(cos_distances), 'histogram': histogram}
```

<!-- #region id="ujXEtRR9VddC" -->
### Graph Distance Test
<!-- #endregion -->

```python id="6evcPhDjLzBL"
def shortest_path_length():
    pass


get_nodes = lambda nodes, ancestors="": [] if not nodes else ['_'.join([ancestors, nodes[0]])] + \
                                                             get_nodes(nodes[1:], '_'.join([ancestors, nodes[0]]))


def graph_distance_test(y_test, y_preds, product_data, k=3):
    path_lengths = []
    misses = sample_misses_at_k(y_preds, y_test, k=k, size=-1)
    for _y, _y_p in zip([_['Y_TEST'] for _ in misses],
                        [_['Y_PRED'] for _ in misses]):
        if not _y_p:
            continue
        _y_sku = _y[0]
        _y_p_sku = _y_p[0]

        if _y_sku not in product_data or _y_p_sku not in product_data:
            continue
        if not product_data[_y_sku]['CATEGORIES'] or not product_data[_y_p_sku]['CATEGORIES']:
            continue
        # extract graph information
        catA = json.loads(product_data[_y_sku]['CATEGORIES'])
        catB = json.loads(product_data[_y_p_sku]['CATEGORIES'])
        catA_nodes = [get_nodes(c) for c in catA]
        catB_nodes = [get_nodes(c) for c in catB]
        all_nodes = list(set([n for c in catA_nodes + catB_nodes for n in c]))
        all_edges = [(n1, n2) for c in catA_nodes + catB_nodes for n1, n2 in zip(c[:-1], c[1:])]
        all_edges = list(set(all_edges))

        # build graph
        G = nx.Graph()
        G.add_nodes_from(all_nodes)
        G.add_edges_from(all_edges)

        # get leaves
        cat1_leaves = [c[-1] for c in catA_nodes]
        cat2_leaves = [c[-1] for c in catB_nodes]

        all_paths = [shortest_path(G, c1_l, c2_l) for c1_l in cat1_leaves for c2_l in cat2_leaves]
        s_path = min(all_paths, key=len)
        s_path_len = len(s_path) - 1
        path_lengths.append(s_path_len)

    histogram = np.histogram(path_lengths, bins=np.arange(0, max(path_lengths)))
    histogram = (histogram[0].tolist(), histogram[1].tolist())
    return {'mean': mean(path_lengths), 'hist': histogram}
```

<!-- #region id="ScJfIPh6VfsX" -->
### Hits Distribution
<!-- #endregion -->

```python id="jzUPryhgLy_E"
def roundup(x: int):
    div = 10.0 ** (len(str(x)))
    return int(math.ceil(x / div)) * div


def hits_distribution(x_train, x_test, y_test, y_preds, k=3, debug=False):
    # get product interaction frequency
    prod_interaction_cnt = Counter([_ for x in x_train for _ in x])
    hit_per_interaction_cnt = defaultdict(list)
    for _x, _y_test, _y_pred in zip(x_test, y_test, y_preds):
        _x_cnt = prod_interaction_cnt[_x[0]]
        # TODO: allow for generic metric
        hit_per_interaction_cnt[_x_cnt].append(hit_rate_at_k([_y_pred], [_y_test], k=k))
    # get max product frequency
    max_cnt = prod_interaction_cnt.most_common(1)[0][1]
    # round up to nearest place
    max_cnt = int(roundup(max_cnt))
    # generate log-bins
    indices = np.logspace(1, np.log10(max_cnt), num=int(np.log10(max_cnt))).astype(np.int64)
    indices = np.concatenate(([0], indices))
    counts_per_bin = [[_ for i in range(low, high) for _ in hit_per_interaction_cnt[i]]
                      for low, high in zip(indices[:-1], indices[1:])]

    histogram = [np.mean(counts) if counts else 0 for counts in counts_per_bin]
    count = [len(counts) for counts in counts_per_bin]

    if debug:
        # debug / visualization
        plt.bar(indices[1:], histogram, width=-np.diff(indices)/1.05, align='edge')
        plt.xscale('log', base=10)
        plt.title('HIT Distribution Across Product Frequency')
        plt.show()

    return {
             'histogram': {int(k): v for k, v in zip(indices[1:], histogram)},
             'counts':  {int(k): v for k, v in zip(indices[1:], count)}
           }
```

<!-- #region id="mKFUHI6rVj11" -->
### Hits Distribution by Data slice
<!-- #endregion -->

```python id="uXAIa7lkLy82"
def hits_distribution_by_slice(slice_fns: dict,
                               y_test,
                               y_preds,
                               product_data,
                               k=3,
                               sample_size=3,
                               debug=False):

    hit_rate_per_slice = defaultdict(dict)
    for slice_name, filter_fn in slice_fns.items():
        # get indices for slice
        slice_idx = [idx for idx,_y in enumerate(y_test) if _y[0] in product_data and filter_fn(product_data[_y[0]])]
        # get predictions for slice
        slice_y_preds = [y_preds[_] for _ in slice_idx]
        # get labels for slice
        slice_y_test = [y_test[_] for _ in slice_idx]
        # TODO: We may want to allow for generic metric to be used here
        slice_hr = hit_rate_at_k(slice_y_preds, slice_y_test,k=3)
        # store results
        hit_rate_per_slice[slice_name]['hit_rate'] = slice_hr
        hit_rate_per_slice[slice_name]['hits'] = sample_hits_at_k(slice_y_preds, slice_y_test, k=k, size=sample_size)
        hit_rate_per_slice[slice_name]['misses'] = sample_misses_at_k(slice_y_preds, slice_y_test, k=k, size=sample_size)

    # debug / visualization
    if debug:
        x_tick_names = list(hit_rate_per_slice.keys())
        x_tick_idx = list(range(len(x_tick_names)))
        plt.bar(x_tick_idx, hit_rate_per_slice.values(), align='center')
        plt.xticks(list(range(len(hit_rate_per_slice))), x_tick_names)
        plt.show()

    # cast to normal dict
    return dict(hit_rate_per_slice)
```

<!-- #region id="_DX1-DeLVqoJ" -->
### Perturbations
<!-- #endregion -->

```python id="55hBX3UkL_ZI"
# TODO: We might need to enforce some standardization of how CATEGORY is represented
def get_item_with_category(product_data: dict, category: set, to_ignore=None):
    to_ignore = [] if to_ignore is None else to_ignore
    skus = [_ for _ in product_data if product_data[_]['category_hash'] == category and _ not in to_ignore]
    if skus:
        return random.choice(skus)
    return []


def perturb_session(session, product_data):
    last_item = session[-1]
    if last_item not in product_data:
        return []
    last_item_category = product_data[last_item]['category_hash']
    similar_item = get_item_with_category(product_data, last_item_category, to_ignore=[last_item])
    if similar_item:
        new_session = session[:-1] + [similar_item]
        return new_session
    return []


def session_perturbation_test(model, x_test, y_preds, product_data):
    overlap_ratios = []
    # print(product_data)
    y_p = []
    s_perturbs = []

    # generate a batch of perturbations
    for idx, (s, _y_p) in enumerate(tqdm(zip(x_test,y_preds))):
        # perturb last item in session
        s = [ _.split('_')[0] for _ in s]
        s_perturb = perturb_session(s, product_data)
        if not s_perturb:
            continue

        s_perturb = ['_'.join([_,'add']) for _ in s_perturb]

        s_perturbs.append(s_perturb)
        y_p.append(_y_p)

    y_perturbs = model.predict(s_perturbs)

    for _y_p, _y_perturb in zip(y_p, y_perturbs):
        if _y_p and _y_perturb:
            # compute prediction intersection
            intersection = set(_y_perturb).intersection(_y_p)
            overlap_ratio = len(intersection)/len(_y_p)
            overlap_ratios.append(overlap_ratio)

    return np.mean(overlap_ratios)
```

<!-- #region id="_FM1vcOUVu4I" -->
### Price Homogeneity
<!-- #endregion -->

```python id="yg06OFvWL_Wk"
def price_homogeneity_test(y_test, y_preds, product_data, bins=25, debug=True, key='PRICE'):
    abs_log_price_diff = []
    for idx, (_y, _y_pred) in enumerate(zip(y_test, y_preds)):
        # need >=1 predictions
        if not _y_pred:
            continue
        # item must be in product data
        if _y[0] not in product_data or _y_pred[0] not in product_data:
            continue
        if product_data[_y[0]][key] and product_data[_y_pred[0]][key]:
            pred_item_price = float(product_data[_y_pred[0]][key])
            y_item_price = float(product_data[_y[0]][key])
            if pred_item_price and y_item_price:
                abs_log_price_diff.append(np.abs(np.log10(pred_item_price)-(np.log10(y_item_price))))

    histogram = np.histogram(abs_log_price_diff, bins=bins, density=False)
    histogram = (histogram[0].tolist(), histogram[1].tolist())
    if debug:
        # debug / viz
        plt.hist(abs_log_price_diff, bins=25)
        plt.show()
    return {
        'mean': np.mean(abs_log_price_diff),
        'histogram': histogram
    }
```

<!-- #region id="uG7H_aLoK0eA" -->
## RecList
<!-- #endregion -->

```python id="BcSsb7gEJJUO"
def rec_test(test_type: str):
    """
    Rec test decorator
    """
    def decorator(f):
        @wraps(f)
        def w(*args, **kwargs):
            return f(*args, **kwargs)

        # add attributes to f
        w.is_test = True
        w.test_type = test_type
        try:
            w.test_desc = f.__doc__.lstrip().rstrip()
        except:
            w.test_desc = ""
        try:
            # python 3
            w.name = w.__name__
        except:
            # python 2
            w.name = w.__func__.func_name
        return w

    return decorator
```

```python id="DrtqbN0XJGA8"
class RecList(ABC):
    
    META_DATA_FOLDER = '.reclist'

    def __init__(self, model: RecModel, dataset: RecDataset, y_preds: list = None):
        """
        :param model:
        :param dataset:
        :param y_preds:
        """
        self.name = self.__class__.__name__
        self._rec_tests = self.get_tests()
        self._x_train = dataset.x_train
        self._y_train = dataset.y_train
        self._x_test = dataset.x_test
        self._y_test = dataset.y_test
        self._y_preds = y_preds if y_preds else model.predict(dataset.x_test)
        self.rec_model = model
        self.product_data = dataset.catalog
        self._test_results = []
        self._test_data = {}
        self._dense_repr = {}

        assert len(self._y_test) == len(self._y_preds)

    def train_dense_repr(self, type_name: str, type_fn):
        """
        Train a dense representation over a type of meta-data & store into object
        """
        # type_fn: given a SKU returns some type i.e. brand
        x_train_transformed = [[type_fn(e) for e in session if type_fn(e)] for session in self._x_train]
        wv = train_embeddings(x_train_transformed)
        # store a dict
        self._dense_repr[type_name] = {word: list(wv.get_vector(word)) for word in wv.key_to_index}

    def get_tests(self):
        """
        Helper to extract methods decorated with rec_test
        """

        nodes = {}
        for _ in self.__dir__():
            if not hasattr(self,_):
                continue
            func = getattr(self, _)
            if hasattr(func, 'is_test'):
                nodes[func.name] = func

        return nodes

    def __call__(self, verbose=True, *args, **kwargs):
        run_epoch_time_ms = round(time.time() * 1000)
        # iterate through tests
        for test_func_name, test in self._rec_tests.items():
            test_result = test(*args, **kwargs)
            # we could store the results in the test function itself
            # test.__func__.test_result = test_result
            self._test_results.append({
                'test_name': test.test_type,
                'description': test.test_desc,
                'test_result': test_result}
            )
            if verbose:
                print("============= TEST RESULTS ===============")
                print("Test Type        : {}".format(test.test_type))
                print("Test Description : {}".format(test.test_desc))
                print("Test Result      : {}\n".format(test_result))
        # at the end, we dump it locally
        if verbose:
            print("Generating reports at {}".format(datetime.utcnow()))
        self.generate_report(run_epoch_time_ms)

    def generate_report(self, epoch_time_ms: int):
        # create path first: META_DATA_FOLDER / RecList / Model / Run Time
        report_path = os.path.join(
            self.META_DATA_FOLDER,
            self.name,
            self.rec_model.__class__.__name__,
            str(epoch_time_ms)
        )
        # now, dump results
        self.dump_results_to_json(self._test_results, report_path, epoch_time_ms)
        # now, store artifacts
        self.store_artifacts(report_path)

    def store_artifacts(self, report_path: str):
        target_path = os.path.join(report_path, 'artifacts')
        # make sure the folder is there, with all intermediate parents
        Path(target_path).mkdir(parents=True, exist_ok=True)
        # store predictions
        with open(os.path.join(target_path, 'model_predictions.json'), 'w') as f:
            json.dump({
                'x_test': self._x_test,
                'y_test': self._y_test,
                'y_preds': self._y_preds
            }, f)

    def dump_results_to_json(self, test_results: list, report_path: str, epoch_time_ms: int):
        target_path = os.path.join(report_path, 'results')
        # make sure the folder is there, with all intermediate parents
        Path(target_path).mkdir(parents=True, exist_ok=True)
        report = {
            'metadata': {
                'run_time': epoch_time_ms,
                'model_name': self.rec_model.__class__.__name__,
                'reclist': self.name,
                'tests': list(self._rec_tests.keys())
            },
            'data': test_results
        }
        with open(os.path.join(target_path, 'report.json'), 'w') as f:
            json.dump(report, f)

    @property
    def test_results(self):
        return self._test_results

    @property
    def test_data(self):
        return self._test_data

    @property
    def rec_tests(self):
        return self._rec_tests
```

```python id="lmIgV9TgIpjb"
class CoveoCartRecList(RecList):

    @rec_test(test_type='stats')
    def basic_stats(self):
        """
        Basic statistics on training, test and prediction data
        """
        return statistics(self._x_train,
                          self._y_train,
                          self._x_test,
                          self._y_test,
                          self._y_preds)

    @rec_test(test_type='Coverage@10')
    def coverage_at_k(self):
        """
        Coverage is the proportion of all possible products which the RS
        recommends based on a set of sessions
        """
        return coverage_at_k(self.sku_only(self._y_preds),
                             self.product_data,
                             k=10)

    @rec_test(test_type='HR@10')
    def hit_rate_at_k(self):
        """
        Compute the rate in which the top-k predictions contain the item to be predicted
        """
        return hit_rate_at_k(self.sku_only(self._y_preds),
                             self.sku_only(self._y_test),
                             k=10)

    @rec_test(test_type='hits_distribution')
    def hits_distribution(self):
        """
        Compute the distribution of hit-rate across product frequency in training data
        """
        return hits_distribution(self.sku_only(self._x_train),
                                 self.sku_only(self._x_test),
                                 self.sku_only(self._y_test),
                                 self.sku_only(self._y_preds),
                                 k=10,
                                 debug=True)

    @rec_test(test_type='distance_to_query')
    def dist_to_query(self):
        """
        Compute the distribution of distance from query to label and query to prediction
        """
        return distance_to_query(self.rec_model,
                                 self.sku_only(self._x_test),
                                 self.sku_only(self._y_test),
                                 self.sku_only(self._y_preds), k=10, bins=25, debug=True)

    def sku_only(self, l:List[List]):
        return [[e['product_sku'] for e in s] for s in l]
```

<!-- #region id="3InDare_Ipl_" -->
## Runs
<!-- #endregion -->

<!-- #region id="pYAZTgU3Ti5N" -->
Let's download the data, and load it. If dataset is already downloaded, it will automatically skip the downloading.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="oFAH7Bm-Ipoq" executionInfo={"status": "ok", "timestamp": 1637594039574, "user_tz": -330, "elapsed": 165506, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="070269cb-e712-474e-8736-638cac5732a3"
coveo_dataset = CoveoDataset()
```

<!-- #region id="PHUhFcAJT4p_" -->
Let's see what's inside the dataset, there is a lot of information here. We are seeing list of sessions; each element in the list is a user session, we know what the user was doing (e.g., "detail") and which product he/she/they were looking at.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="e-co-IK3Ipq-" executionInfo={"status": "ok", "timestamp": 1637594845203, "user_tz": -330, "elapsed": 20, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c27cb12c-fc83-47e2-df06-76461e8c5ee7"
coveo_dataset.x_train[0]
```

```python colab={"base_uri": "https://localhost:8080/"} id="HfUmzC40Ipte" executionInfo={"status": "ok", "timestamp": 1637595735164, "user_tz": -330, "elapsed": 637, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="82b62cd6-e874-4aad-ebd9-9fff49da81fd"
x_train_skus = [[e['product_sku'] for e in s] for s in coveo_dataset.x_train]
x_train_skus[0:3]
```

<!-- #region id="arJE3tvJUGKS" -->
Let's build a vanilla recommender with a KNN product based model!
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Mfl5t16eOUo0" executionInfo={"status": "ok", "timestamp": 1637594975286, "user_tz": -330, "elapsed": 115592, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="51f7a225-c139-4fda-e228-2055439e4f56"
embeddings = train_embeddings(sessions=x_train_skus)
```

```python id="mYJ6YEolPEBh"
model = P2VRecModel(model=embeddings)
```

<!-- #region id="vIkT1nQPUJHh" -->
Let's see if our model is working, we will try to do a simple prediction starting from a single product.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="mF1O-aF-Q2HU" executionInfo={"status": "ok", "timestamp": 1637595138472, "user_tz": -330, "elapsed": 643, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="621102b7-6f29-4940-be18-320c453d6efe"
test = [[{'event_type': 'event_product',
  'hashed_url': '803f6c2d4202e39d6d7fdb232d69366b86bc843869c809f1e1954465bfc6e17f',
  'product_action': 'detail',
  'product_sku': '624bc145579b67b608e6a7b0d0516cc75e0ec4cbe44ec42c6ac53cc83925bc3e',
  'server_timestamp_epoch_ms': '1547528580651',
  'session_id': '0f1416c8c68bb9209c1bbc4576386df5480e9757f55ce9cb0d4d4017cf14fc1c'}]]


model.predict(test)
```

<!-- #region id="CRQF4bYLUNDX" -->
Instantiate rec_list object, prepared with standard quantitative tests and sensible behavioral tests. Then, invoke rec_list to run tests.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="t_6fqcguSEUG" executionInfo={"status": "ok", "timestamp": 1637595329897, "user_tz": -330, "elapsed": 10310, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="31e53315-0364-4f41-8759-db41e6a4ecc1"
# instantiate rec_list object
rec_list = CoveoCartRecList(
    model=model,
    dataset=coveo_dataset
)
# invoke rec_list to run tests
rec_list(verbose=True)
```

<!-- #region id="kelwzaKBfxcg" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="olY--uGRV-uy" executionInfo={"status": "ok", "timestamp": 1637596476870, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9b61ad4c-ea52-4dd1-cf4b-530c15b29ecc"
!apt-get install tree
!tree -a --du -h . -L 3
```

```python colab={"base_uri": "https://localhost:8080/"} id="DvXBahfQfxch" executionInfo={"status": "ok", "timestamp": 1637598908502, "user_tz": -330, "elapsed": 2859, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="46b34c4c-3c15-4846-965c-9cec16e7f277"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="5XlMYZkxfxci" -->
---
<!-- #endregion -->

<!-- #region id="4JwYKmJ5fxci" -->
**END**
<!-- #endregion -->
