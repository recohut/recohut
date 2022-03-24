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

<!-- #region id="K-3LaItSDleH" -->
# LightFM Cold-start on ML-10m
<!-- #endregion -->

<!-- #region id="U-NnUhl3zfk5" -->
## Setup
<!-- #endregion -->

<!-- #region id="GL7U8z4wEmln" -->
### Installations
<!-- #endregion -->

```python id="_7xq83S8BMAN"
!pip install scikit-learn==0.19.2
!pip install lightfm
```

<!-- #region id="4YeMFsAYEn78" -->
### Datasets
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="r11m3EgHxjWZ" executionInfo={"status": "ok", "timestamp": 1635678419158, "user_tz": -330, "elapsed": 8023, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2bfdce82-a5a8-4854-810b-4cb591451c84"
!wget -q --show-progress http://files.grouplens.org/datasets/movielens/ml-10m.zip
!wget -q --show-progress http://files.grouplens.org/datasets/tag-genome/tag-genome.zip
!unzip ml-10m.zip
!unzip tag-genome.zip
```

<!-- #region id="mc5yuRfAEt9S" -->
### Imports
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="JeivkMQsxrJE" executionInfo={"status": "ok", "timestamp": 1635678603628, "user_tz": -330, "elapsed": 569, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0fbda86a-ea3c-4f03-c023-56d715c9decb"
import array
import collections
import numpy as np
import os
import re
import scipy.sparse as sp
import subprocess
import itertools

import logging
import logging.handlers
import logging.config

import json
from pprint import pformat
import sys

from lightfm import LightFM

# from sklearn.model_selection import ShuffleSplit
from sklearn.cross_validation import ShuffleSplit
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
sns.set_palette('Set1')
sns.set_style('white')
%matplotlib inline
```

```python id="HmgBVt4YyCpn"
SEPARATOR = '::'
DATA_DIR = 'ml-10M100K'
GENOME_DIR = 'tag-genome'
DIMS_RANGE = 10
```

```python id="vybo6QEbz7iO"
FONTSIZE = 7
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['legend.fontsize'] = FONTSIZE

DASHES = ['-', '--', '-.', ':']
MARKERS = ['.', '^', 'v', 'x', '+']

KEYS = ('LSI-LR',
        'LSI-UP',
        'LightFM (tags)',
        'LightFM (tags + ids)',
        'LightFM (tags + about)')

COLORS = ('#e41a1c',
          '#377eb8',
          '#4daf4a',
          '#984ea3',
          '#ff7f00')
```

```python id="lD2TGLsJzmwA"
logger = logging.getLogger(__name__)
```

<!-- #region id="3vZwLW0azdcU" -->
## Utils
<!-- #endregion -->

```python id="JQDtLaB8z-sX"
def dim_sensitivity_plot(x, Y, fname, show_legend=True):

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.figure(figsize=(3, 3))
    plt.xlabel('$d$', size=FONTSIZE)
    plt.ylabel('ROC AUC', size=FONTSIZE)

    plt.set_cmap('Set2')

    lines = []
    for i, label in enumerate(KEYS):
        line_data = Y.get(label)

        if line_data is None:
            continue
        
        line, = plt.plot(x, line_data, label=label, marker=MARKERS[i],
                         markersize=0.5 * FONTSIZE, color=COLORS[i])
        lines.append(line)



    if show_legend:
        plt.legend(handles=lines)
        plt.legend(loc='lower right')
    plt.xscale('log', basex=2)
    plt.xticks(x, [str(y) for y in x], size=FONTSIZE)
    plt.yticks(size=FONTSIZE)
    plt.tight_layout()

    plt.savefig(fname)
```

```python id="u45ikiiezvQs"
class StratifiedSplit(object):
    """
    Class responsible for producing train-test splits.
    """

    def __init__(self, user_ids, item_ids, n_iter=10, 
                 test_size=0.2, cold_start=False, random_seed=None):
        """
        Options:
        - test_size: the fraction of the dataset to be used as the test set.
        - cold_start: if True, test_size of items will be randomly selected to
                      be in the test set and removed from the training set. When
                      False, test_size of all training pairs are moved to the
                      test set.
        """

        self.user_ids = user_ids
        self.item_ids = item_ids
        self.no_interactions = len(self.user_ids)
        self.n_iter = n_iter
        self.test_size = test_size
        self.cold_start = cold_start

        self.shuffle_split = ShuffleSplit(self.no_interactions,
                                          n_iter=self.n_iter,
                                          test_size=self.test_size)

    def _cold_start_iterations(self):
        """
        Performs the cold-start splits.
        """

        for _ in range(self.n_iter):
            unique_item_ids = np.unique(self.item_ids)
            no_in_test = int(self.test_size * len(unique_item_ids))

            item_ids_in_test = set(np.random.choice(unique_item_ids, size=no_in_test))

            test_indices = array.array('i')
            train_indices = array.array('i')

            for i, item_id in enumerate(self.item_ids):
                if item_id in item_ids_in_test:
                    test_indices.append(i)
                else:
                    train_indices.append(i)

            train = np.frombuffer(train_indices, dtype=np.int32)
            test = np.frombuffer(test_indices, dtype=np.int32)

            # Shuffle data.
            np.random.shuffle(train)
            np.random.shuffle(test)

            yield train, test

    def __iter__(self):

        if self.cold_start:
            splits = self._cold_start_iterations()           
        else:
            splits = self.shuffle_split

        for train, test in splits:

            # Make sure that all the users in test
            # are represented in train.
            user_ids_in_train = collections.defaultdict(lambda: 0)
            item_ids_in_train = collections.defaultdict(lambda: 0)

            for uid in self.user_ids[train]:
                user_ids_in_train[uid] += 1

            for iid in self.item_ids[train]:
                item_ids_in_train[iid] += 1

            if self.cold_start:
                test = [x for x in test if self.user_ids[x] in user_ids_in_train]
            else:
                # For the non-cold start scenario, make sure that both users
                # and items are represented in the train set.
                test = [x for x in test if (self.user_ids[x] in user_ids_in_train
                                            and self.item_ids[x] in item_ids_in_train)]

            test = np.array(test)

            yield train, test


def stratified_roc_auc_score(y, yhat, user_indices):
    """
    Compute ROC AUC for each user individually, then average.
    """

    aucs = []

    y_dict = collections.defaultdict(lambda: array.array('d'))
    yhat_dict = collections.defaultdict(lambda: array.array('d'))

    for i, uid in enumerate(user_indices):
        y_dict[uid].append(y[i])
        yhat_dict[uid].append(yhat[i])

    for uid in y_dict:

        user_y = np.frombuffer(y_dict[uid], dtype=np.float64)
        user_yhat = np.frombuffer(yhat_dict[uid], dtype=np.float64)

        if len(user_y) and len(user_yhat) and len(np.unique(user_y)) == 2:
            aucs.append(roc_auc_score(user_y, user_yhat))

    logger.debug('%s users in stratified ROC AUC evaluation.', len(aucs))
    
    return np.mean(aucs)


def build_user_feature_matrix(user_ids):

    n = len(user_ids)

    return sp.coo_matrix((np.ones(n, dtype=np.int32), (np.arange(n), user_ids))).tocsr()


def fit_model(interactions, item_features_matrix,
              n_iter, epochs, modelfnc, test_size,
              cold_start, user_features_matrix=None):
    """
    Fits the model provided by modelfnc.
    """

    kf = StratifiedSplit(interactions.user_id, interactions.item_id,
                         n_iter=n_iter, test_size=test_size, cold_start=cold_start)

    logger.debug('Interaction density across all data: %s',
                 (float(len(interactions.data)) / (len(interactions.user_ids)
                                                   * len(interactions.item_ids))))
    logger.debug('Training model')

    # Store ROC AUC scores for all iterations.
    aucs = []

    # Iterate over train-test splits.
    for i, (train, test) in enumerate(kf):

        logger.debug('Split no %s', i)
        logger.debug('%s examples in training set, %s in test set. Interaction density: %s',
                    len(train), len(test), float(len(train)) / (len(interactions.user_ids)
                                                                * len(interactions.item_ids)))

        # For every split, get a new model instance.
        model = modelfnc()

        if isinstance(model, CFModel):
            logger.debug('Evaluating a CF model')
            test_auc, train_auc = evaluate_cf_model(model,
                                                    item_features_matrix,
                                                    interactions.user_id[train],
                                                    interactions.item_id[train],
                                                    interactions.data[train],
                                                    interactions.user_id[test],
                                                    interactions.item_id[test],
                                                    interactions.data[test])
            logger.debug('CF model test AUC %s, train AUC %s', test_auc, train_auc)
            aucs.append(test_auc)

        elif isinstance(model, LsiUpModel):
            logger.debug('Evaluating a LSI-UP model')

            # Prepare data.
            y = interactions.data
            no_users = np.max(interactions.user_id) + 1
            no_items = item_features_matrix.shape[0]

            train_user_ids = interactions.user_id[train]
            train_item_ids = interactions.item_id[train]

            user_features = sp.coo_matrix((interactions.data[train],
                                           (train_user_ids, train_item_ids)),
                                           shape=(no_users, no_items)).tocsr()
            user_feature_matrix = user_features * item_features_matrix

            # Fit model.
            model.fit(user_feature_matrix, item_features_matrix)
            
            # For larger datasets use incremental prediction. Slower, but
            # fits in far less memory.
            if len(train) or len(test) > 200000:
                train_predictions = model.predict(interactions.user_id[train],
                                                  interactions.item_id[train],
                                                  incremental=True)
                test_predictions = model.predict(interactions.user_id[test],
                                                 interactions.item_id[test],
                                                 incremental=True)
            else:
                train_predictions = model.predict(interactions.user_id[train],
                                                  interactions.item_id[train])
                test_predictions = model.predict(interactions.user_id[test],
                                                 interactions.item_id[test])

            # Compute mean ROC AUC scores on both test and train data.
            train_auc = stratified_roc_auc_score(y[train],
                                                 train_predictions,
                                                 interactions.user_id[train])
            test_auc = stratified_roc_auc_score(y[test],
                                                test_predictions,
                                                interactions.user_id[test])

            logger.debug('Test AUC %s, train AUC %s', test_auc, train_auc)

            aucs.append(test_auc)

        else:
            # LightFM and MF models using the LightFM implementation.
            if user_features_matrix is not None:
                user_features = user_features_matrix
            else:
                user_features = build_user_feature_matrix(interactions.user_id)

            item_features = item_features_matrix

            previous_auc = 0.0

            interactions.data[interactions.data == 0] = -1

            train_interactions = sp.coo_matrix((interactions.data[train],
                                                (interactions.user_id[train],
                                                 interactions.item_id[train])))

            # Run for a maximum of epochs epochs.
            # Stop if the test score starts falling, take the best result.
            for x in range(epochs):
                model.fit_partial(train_interactions,
                                  item_features=item_features,
                                  user_features=user_features,
                                  epochs=1, num_threads=1)

                train_predictions = model.predict(interactions.user_id[train],
                                                  interactions.item_id[train],
                                                  user_features=user_features,
                                                  item_features=item_features,
                                                  num_threads=4)
                test_predictions = model.predict(interactions.user_id[test],
                                                 interactions.item_id[test],
                                                 user_features=user_features,
                                                 item_features=item_features,
                                                 num_threads=4)

                train_auc = stratified_roc_auc_score(interactions.data[train],
                                                     train_predictions,
                                                     interactions.user_id[train])
                test_auc = stratified_roc_auc_score(interactions.data[test],
                                                    test_predictions,
                                                    interactions.user_id[test])

                logger.debug('Epoch %s, test AUC %s, train AUC %s', x, test_auc, train_auc)

                if previous_auc > test_auc:
                    break

                previous_auc = test_auc

            aucs.append(previous_auc)

    return model, np.mean(aucs)
```

```python id="dvhfH-CEyjsk"
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s",
            'datefmt': "%Y-%m-%d %H:%M:%S"
        },
        'simple': {
            'format': '%(levelname)s %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose'
        },
        'file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'verbose',
            'filename': 'model.log',
            'maxBytes': 10*10**6,
            'backupCount': 3
            }
    },
    'loggers': {
        '': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
        },
    }
}


logging.config.dictConfig(LOGGING)


def getLogger(name):

    return logging.getLogger(name)
```

```python id="hXXKI94Cx8LS"
class IncrementalCOOMatrix(object):

    def __init__(self, dtype):

        if dtype is np.int32:
            type_flag = 'i'
        elif dtype is np.int64:
            type_flag = 'l'
        elif dtype is np.float32:
            type_flag = 'f'
        elif dtype is np.float64:
            type_flag = 'd'
        else:
            raise Exception('Dtype not supported.')

        self.dtype = dtype
        self.shape = None

        self.rows = array.array('i')
        self.cols = array.array('i')
        self.data = array.array(type_flag)

    def append(self, i, j, v):

        self.rows.append(i)
        self.cols.append(j)
        self.data.append(v)

    def tocoo(self):

        rows = np.frombuffer(self.rows, dtype=np.int32)
        cols = np.frombuffer(self.cols, dtype=np.int32)
        data = np.frombuffer(self.data, dtype=self.dtype)

        self.shape = self.shape or (np.max(rows) + 1, np.max(cols) + 1)

        return sp.coo_matrix((data, (rows, cols)),
                             shape=self.shape)

    def __len__(self):

        return len(self.data)
```

```python id="pc4UX5OQx_CD"
class Features(object):

    def __init__(self):

        self.feature_ids = {}
        self.item_ids = {}
        self.title_mapping = {}

        self.mat = IncrementalCOOMatrix(np.int32)

    def add_item(self, item_id):

        iid = self.item_ids.setdefault(item_id, len(self.item_ids))
        
    def add_feature(self, item_id, feature):

        iid = self.item_ids.setdefault(item_id, len(self.item_ids))

        feature_id = self.feature_ids.setdefault(feature, len(self.feature_ids))

        self.mat.append(iid, feature_id, 1)

    def add_title(self, item_id, title):

        iid = self.item_ids.setdefault(item_id, len(self.item_ids))
        self.title_mapping[iid] = title

    def set_shape(self):

        self.mat.shape = len(self.item_ids), len(self.feature_ids)

    def add_latent_representations(self, latent_representations):

        dim = latent_representations.shape[1]
        lrepr = np.zeros((len(self.title_mapping), dim),
                         dtype=np.float32)

        for i, row in enumerate(self.mat.tocoo().tocsr()):
            lrepr[i] = np.sum(latent_representations[row.indices], axis=0)

        self.lrepr = lrepr
        self.inverse_title_mapping = {v: k for k, v in self.title_mapping.items()}

    def most_similar_movie(self, title, number=5):

        iid = self.inverse_title_mapping[title]

        vector = self.lrepr[iid]

        dst = (np.dot(self.lrepr, vector)
               / np.linalg.norm(self.lrepr, axis=1) / np.linalg.norm(vector))
        movie_ids = np.argsort(-dst)
        
        return [(self.title_mapping[x], dst[x]) for x in movie_ids[:number]
                if x in self.title_mapping]
```

```python id="uf6FRejKx9Gy"
class Interactions(object):

    def __init__(self, item_ids):

        self.item_ids = item_ids
        self.user_ids = {}

        self.user_data = collections.defaultdict(lambda: {1: array.array('i'),
                                                          0: array.array('i')})

        self.iids_sample_pool = np.array(item_ids.values())

        self._user_id = array.array('i')
        self._item_id = array.array('i')
        self._data = array.array('i')

    def add(self, user_id, item_id, value):

        iid = self.item_ids[item_id]
        user_id = self.user_ids.setdefault(user_id, len(self.user_ids))

        self.user_data[user_id][value].append(iid)

    def fit(self, min_positives=1, sampled_negatives_ratio=0, use_observed_negatives=True):
        """
        Constructs the training data set from raw interaction data.
        Parameters:
        - min_positives: users with fewer than min_positives interactions are excluded
                         from the training set
        - sampled_negatives_ratio: a ratio of 3 means that at most three negative examples
                         randomly sampled for the pids_sample_pool will be included.
        """

        for user_id, user_data in self.user_data.items():

            positives = user_data.get(1, [])
            raw_negatives = user_data.get(0, [])

            if len(positives) < min_positives:
                continue

            if use_observed_negatives:
                observed_negatives = list(set(raw_negatives) - set(positives))
            else:
                observed_negatives = []

            if sampled_negatives_ratio:
                sampled_negatives = np.random.choice(self.iids_sample_pool,
                                                     size=len(positives) * sampled_negatives_ratio)
                sampled_negatives = list(set(sampled_negatives) - set(positives))
            else:
                sampled_negatives = []

            for value, pids in zip((1, 0, 0), (positives, observed_negatives, sampled_negatives)):
                for pid in pids:
                    self._user_id.append(user_id)
                    self._item_id.append(pid)
                    self._data.append(value)

        self.user_id = np.frombuffer(self._user_id, dtype=np.int32)
        self.item_id = np.frombuffer(self._item_id, dtype=np.int32)
        self.data = np.frombuffer(self._data, dtype=np.int32)
```

```python id="92RB3N3WyUuJ"
def read_genome_tags(min_popularity=20):

    tag_dict = {}

    with open(os.path.join(GENOME_DIR, 'tags.dat'), 'r') as tagfile:
        for line in tagfile:

            tag_id, tag, popularity = line.split('\t')

            if int(popularity) >= min_popularity:
                tag_dict[int(tag_id)] = tag

    with open(os.path.join(GENOME_DIR, 'tag_relevance.dat'), 'r') as tagfile:
        for line in tagfile:

            iid, tag_id, relevance = line.split('\t')

            if int(tag_id) in tag_dict:
                yield iid, tag_dict[int(tag_id)], float(relevance)
```

```python id="-nXiZ2HtyS8l"
def _process_raw_tag(tag):

    tag = re.sub('[^a-zA-Z]+', ' ', tag.lower()).strip()

    return tag
```

```python id="_6VqecWOyRU5"
def read_tags():

    tag_dict = collections.defaultdict(lambda: 0)

    with open(os.path.join(DATA_DIR, 'tags.dat'), 'r') as tagfile:
        for line in tagfile:

            uid, iid, tag, timestamp = line.split(SEPARATOR)
            processed_tag = _process_raw_tag(tag)
            tag_dict[tag] += 1

    with open(os.path.join(DATA_DIR, 'tags.dat'), 'r') as tagfile:
        for line in tagfile:

            uid, iid, tag, timestamp = line.split(SEPARATOR)
            processed_tag = _process_raw_tag(tag)
            tag_count = tag_dict[processed_tag]

            yield iid, processed_tag, tag_count
```

```python id="j2N6l6zDyPF3"
def read_movie_features(titles=False, genres=False, genome_tag_threshold=1.0, tag_popularity_threshold=30):

    features = Features()

    with open(os.path.join(DATA_DIR, 'movies.dat'), 'r') as moviefile:
        for line in moviefile:
            (iid, title, genre_list) = line.split(SEPARATOR)
            genres_list = genre_list.split('|')

            features.add_item(iid)

            if genres:
                for genre in genres_list:
                    features.add_feature(iid, 'genre:' + genre.lower().replace('\n', ''))

            if titles:
                features.add_feature(iid, 'title:' + title.lower())

            features.add_title(iid, title)

    for iid, tag, relevance in read_genome_tags():
        # Do not include any tags for movies not in the 10M dataset
        if relevance >= genome_tag_threshold and iid in features.item_ids:
            features.add_feature(iid, 'genome:' + tag.lower())

    # Tags applied by users
    ## for iid, tag, count in read_tags():
    ##     if count >= tag_popularity_threshold and iid in features.item_ids:
    ##         features.add_feature(iid, 'tag:' + tag)

    features.set_shape()

    return features
```

```python id="2ZcNWAPWyJs7"
def read_interaction_data(item_id_mapping, positive_threshold=4.0):

    interactions = Interactions(item_id_mapping)

    with open(os.path.join(DATA_DIR, 'ratings.dat'), 'r') as ratingfile:
        for line in ratingfile:

            (uid, iid, rating, timestamp) = line.split(SEPARATOR)

            value = 1.0 if float(rating) >= positive_threshold else 0.0

            interactions.add(uid, iid, value)

    return interactions
```

<!-- #region id="vR3GV96ty4Lf" -->
## CF Model
<!-- #endregion -->

```python id="Cj5VHV_8y5TT"
class CFModel(object):
    """
    The LSI-LR model.
    """

    def __init__(self, dim=64):

        self.dim = dim
        self.model = None
        self.item_latent_features = None

    def fit_svd(self, mat):
        """
        Fit the feature latent factors.
        """

        model = TruncatedSVD(n_components=self.dim)
        model.fit(mat)

        self.model = model

    def fit_latent_features(self, feature_matrix):
        """
        Project items into the latent space.
        """

        self.item_latent_features = self.model.transform(feature_matrix)

    def fit_user(self, item_ids, y):
        """
        Fit a logistic regression model for a single user.
        """

        model = LogisticRegression()
        model.fit(self.item_latent_features[item_ids], y)

        return model

    def predict_user(self, model, item_ids):
        """
        Predict positive interaction probability for user represented by model.
        """

        return model.decision_function(self.item_latent_features[item_ids])



def evaluate_cf_model(model, feature_matrix, train_user_ids, train_item_ids, train_data,
                      test_user_ids, test_item_ids, test_data):
    """
    LSI-LR model: perform LSI (via truncated SVD on the item-feature matrix), then computer user models
    by fitting a logistic regression model to items represented as mixtures of LSI topics.
    """

    train_aucs = []
    test_aucs = []

    train_y_dict = collections.defaultdict(lambda: array.array('d'))
    train_iid_dict = collections.defaultdict(lambda: array.array('i'))

    test_y_dict = collections.defaultdict(lambda: array.array('d'))
    test_iid_dict = collections.defaultdict(lambda: array.array('i'))

    # Gather training data in user-sized chunks
    for i, (uid, iid, y) in enumerate(zip(train_user_ids, train_item_ids, train_data)):
        train_y_dict[uid].append(y)
        train_iid_dict[uid].append(iid)

    # Gather test data in user-sized chunks
    for i, (uid, iid, y) in enumerate(zip(test_user_ids, test_item_ids, test_data)):
        test_y_dict[uid].append(y)
        test_iid_dict[uid].append(iid)

    # Only use the items in the training set for LSI
    model.fit_svd(feature_matrix[np.unique(train_item_ids)])
    model.fit_latent_features(feature_matrix)

    # Fit models and generate predictions
    for uid in train_y_dict:
        train_iids = np.frombuffer(train_iid_dict[uid], dtype=np.int32)
        train_y = np.frombuffer(train_y_dict[uid], dtype=np.float64)

        test_iids = np.frombuffer(test_iid_dict[uid], dtype=np.int32)
        test_y = np.frombuffer(test_y_dict[uid], dtype=np.float64)

        if len(np.unique(test_y)) == 2 and len(np.unique(train_y)) == 2:
            user_model = model.fit_user(train_iids, train_y)
            train_yhat = model.predict_user(user_model, train_iids)
            test_yhat = model.predict_user(user_model, test_iids)
            
            train_aucs.append(roc_auc_score(train_y, train_yhat))
            test_aucs.append(roc_auc_score(test_y, test_yhat))

    return np.mean(test_aucs), np.mean(train_aucs)
```

<!-- #region id="pjvavJs8zMAu" -->
## LSI-UP Model
<!-- #endregion -->

```python id="AdqbqyMCzOlY"
class LsiUpModel(object):
    """
    The LSI-UP model.
    """

    def __init__(self, dim=64):

        self.dim = dim
        self.user_factors = None
        self.item_factors = None

    def fit(self, user_feature_matrix, product_feature_matrix):
        """
        Fit latent factors to the user-feature matrix through truncated SVD,
        then get item representations by projecting onto the latent feature
        space.
        """

        nrm = lambda x: normalize(x.astype(np.float64), norm='l2', axis=1)

        svd = TruncatedSVD(n_components=self.dim)
        svd.fit(nrm(user_feature_matrix))

        self.user_factors = svd.transform(nrm(user_feature_matrix))
        self.item_factors = svd.transform(nrm(product_feature_matrix))

    def predict(self, user_ids, product_ids, incremental=False):
        """
        Predict scores.
        """

        if not incremental:
            return np.inner(self.user_factors[user_ids],
                            self.item_factors[product_ids])
        else:
            result = array.array('f')
            
            for i in range(len(user_ids)):
                uid = user_ids[i]
                pid = product_ids[i]

                result.append(np.dot(self.user_factors[uid],
                                     self.item_factors[pid]))

            return np.frombuffer(result, dtype=np.float32)
```

<!-- #region id="JS2xbTvY0aX_" -->
## Main
<!-- #endregion -->

```python id="2VRPSChH0eGW"
def read_data(titles, genres,
              genome_tag_threshold,
              positive_threshold):

    logger.debug('Reading features')
    features = read_movie_features(titles=titles, genres=genres, genome_tag_threshold=genome_tag_threshold)
    item_features_matrix = features.mat.tocoo().tocsr()

    logger.debug('Reading interactions')
    interactions = read_interaction_data(features.item_ids,
                                         positive_threshold=positive_threshold)
    interactions.fit(min_positives=1, sampled_negatives_ratio=0, use_observed_negatives=True)

    logger.debug('%s users, %s items, %s interactions, %s item features in the dataset',
                len(interactions.user_ids), len(features.item_ids),
                len(interactions.data), len(features.feature_ids))

    return features, item_features_matrix, interactions
```

```python id="btIC7wB20f4o"
def run(features,
        item_features_matrix,
        interactions,
        cf_model,
        lsiup_model,
        n_iter,
        test_size,
        cold_start,
        learning_rate,
        no_components,
        a_alpha,
        b_alpha,
        epochs):

    logger.debug('Fitting the model with %s', locals())

    no_interactions = len(interactions.data)

    if cf_model:
        logger.info('Fitting the CF model')
        modelfnc = lambda: CFModel(dim=no_components)
    elif lsiup_model:
        logger.info('Fitting the LSI-UP model')
        modelfnc = lambda: LsiUpModel(dim=no_components)
    else:
        modelfnc = lambda: LightFM(learning_rate=learning_rate,
                                    no_components=no_components,
                                    item_alpha=a_alpha,
                                    user_alpha=b_alpha)

    model, auc = fit_model(interactions=interactions,
                           item_features_matrix=item_features_matrix, 
                           n_iter=n_iter,
                           epochs=epochs,
                           modelfnc=modelfnc,
                           test_size=test_size,
                           cold_start=cold_start)
    logger.debug('Average AUC: %s', auc)

    if not cf_model and not lsiup_model:
        model.add_item_feature_dictionary(features.feature_ids, check=False)
        features.add_latent_representations(model.item_features)

        titles = ('Lord of the Rings: The Two Towers, The (2002)',
                  'Toy Story (1995)',
                  'Terminator, The (1984)',
                  'Europa Europa (Hitlerjunge Salomon) (1990)')

        for title in titles:
            logger.debug('Most similar movies to %s: %s', title,
                        features.most_similar_movie(title, number=20))

            # Can only get similar tags if we have tag features
        test_features = ('genome:art house',
                         'genome:dystopia',
                         'genome:bond')

        for test_feature in test_features:
            try:
                logger.debug('Features most similar to %s: %s',
                             test_feature,
                             model.most_similar(test_feature, 'item', number=10))
            except KeyError:
                pass

    return auc
```

```python id="H6dEhvia1c5E"
class Args:
    ids = False
    tags = False
    split = 0.2
    cold = False
    lsi = False
    up = False
    dim = (64,)
    niter = 5
    plot = False
    table = False
```

```python id="7AZCdEP20iU9"
def main(args):

    logger.info('Running the MovieLens experiment.')
    logger.info('Configuration: %s', pformat(args))

    # A large tag threshold excludes all tags.
    tag_threshold = 0.8 if args.tags else 100.0
    features, item_features_matrix, interactions = read_data(titles=args.ids,
                                                             genres=False,
                                                             genome_tag_threshold=tag_threshold,
                                                             positive_threshold=4.0)

    results = {}
    
    for dim in args.dim:
        auc = run(features,
                  item_features_matrix,
                  interactions,
                  cf_model=args.lsi,
                  lsiup_model=args.up,
                  n_iter=args.niter,
                  test_size=args.split,
                  cold_start=args.cold,
                  learning_rate=0.05,
                  no_components=int(dim),
                  a_alpha=0.0,
                  b_alpha=0.0,
                  epochs=30)

        results[int(dim)] = auc
        logger.info('AUC %s for configuration %s', auc, pformat(args))

    sys.stdout.write(json.dumps(results))
```

```python colab={"base_uri": "https://localhost:8080/"} id="OYDCe2cw2gQS" executionInfo={"status": "ok", "timestamp": 1635680001640, "user_tz": -330, "elapsed": 1131125, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="754fbe9e-90a4-4b73-9ad3-cd7b4be136e5"
# run the CrossValidated experiment with 50-dimensional latent space, 
# using the LSI-LR model with both post tags and post ids
args = Args()
args.dim = (50,)
args.lsi = True
args.tags = True
args.ids = True
args.split = 0.2
main(args)
```

<!-- #region id="p_GdOBsQDhUS" -->
## Citations

Metadata Embeddings for User and Item Cold-start Recommendations. Maciej Kula. 2015. arXiv. [https://arxiv.org/abs/1507.08439](https://arxiv.org/abs/1507.08439)
<!-- #endregion -->
