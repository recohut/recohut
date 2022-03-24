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

```python executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1629805506511, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="d1o6gvXino6X"
import os
project_name = "reco-tut-csr"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3307, "status": "ok", "timestamp": 1629805510682, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="8MpUlXlWny29" outputId="4b971fe8-b952-4795-fd41-dee3db05d794"
if not os.path.exists(project_path):
    !cp /content/drive/MyDrive/mykeys.py /content
    import mykeys
    !rm /content/mykeys.py
    path = "/content/" + project_name; 
    !mkdir "{path}"
    %cd "{path}"
    import sys; sys.path.append(path)
    !git config --global user.email "recotut@recohut.com"
    !git config --global user.name  "reco-tut"
    !git init
    !git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout main
else:
    %cd "{project_path}"
```

```python id="ochl7m4B-XB3"
!git pull --rebase origin "{branch}"
```

```python id="jrUWb0jiny3G"
!git status
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 710, "status": "ok", "timestamp": 1629802744715, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="MpYWa13ony3I" outputId="67c37cbd-db3a-415f-94e3-fc90100c8203"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1629805621522, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="XlHj9DTvojCE"
!pip install -q dvc dvc[gdrive]
!dvc pull
```

```python id="yyixkXS6yuHZ"
!dvc commit && dvc push
```

<!-- #region id="z3GIu7oTzohV" -->
---
<!-- #endregion -->

<!-- #region id="oKotM2ku9r8C" -->
We adopt three different ranking evaluation metrics to evaluate model performance: Precision@k (P@k), Recall@k (R@k) and NDCG@k. We implement the proposed model by Tensorflow and Adam optimizer. For the hyper-parameters, we fix the CF latent factor dimension as 200, and set the learning rate as 0.005, the mini-batch size as 1024. Heater requires pretrained CF representations as input. Hence, we train a Bayesian Personalized Ranking (BPR) model with latent factors of 200 dimensions, L2 regularization weight 0.001, and learning rate as 0.005 and use the learned latent factors of BPR as P and Q.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 27, "status": "ok", "timestamp": 1629805621523, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="F8QX8zSHzoyX" outputId="16f840a0-87ce-4f34-f4b9-7ab7fc62340a"
%tensorflow_version 1.x
```

```python executionInfo={"elapsed": 5779, "status": "ok", "timestamp": 1629805632094, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="-PpBaO4s0O4U"
import tensorflow as tf
import numpy as np
import time
import datetime
import scipy
from sklearn import preprocessing as prep
import pandas as pd
import scipy.sparse
from sklearn import datasets
import scipy.sparse as sp
import argparse
from tqdm import tqdm
import pickle

np.random.seed(0)
tf.set_random_seed(0)
```

<!-- #region id="5JLwqkc70Vo5" -->
## Utils
<!-- #endregion -->

```python executionInfo={"elapsed": 654, "status": "ok", "timestamp": 1629805716073, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="qGvmxs1b9aDA"
class timer_class(object):
    def __init__(self, name='default'):
        """
        timer object to record running time of functions, not for micro-benchmarking
        :param name: label for the timer
        """
        self._start_time = None
        self._name = name
        self.tic()

    def tic(self):
        self._start_time = time.time()
        return self

    def toc(self, message):
        elapsed = time.time() - self._start_time
        message = '' if message is None else message
        print('[{0:s}] {1:s} elapsed [{2:s}]'.format(self._name, message, timer_class._format(elapsed)))
        return self

    def reset(self):
        self._start_time = None
        return self

    @staticmethod
    def _format(s):
        delta = datetime.timedelta(seconds=s)
        d = datetime.datetime(1, 1, 1) + delta
        s = ''
        if (d.day - 1) > 0:
            s = s + '{:d} days'.format(d.day - 1)
        if d.hour > 0:
            s = s + '{:d} hr'.format(d.hour)
        if d.minute > 0:
            s = s + '{:d} min'.format(d.minute)
        s = s + '{:d} s'.format(d.second)
        return s


def batch(iterable, _n=1, drop=True):
    """
    returns batched version of some iterable
    :param iterable: iterable object as input
    :param _n: batch size
    :param drop: if true, drop extra if batch size does not divide evenly,
        otherwise keep them (last batch might be shorter)
    :return: batched version of iterable
    """
    it_len = len(iterable)
    for ndx in range(0, it_len, _n):
        if ndx + _n < it_len:
            yield iterable[ndx:ndx + _n]
        elif drop is False:
            yield iterable[ndx:it_len]


def tfidf(x):
    """
    compute tfidf of numpy array x
    :param x: input array, document by terms
    :return: csr tfidf array
    """
    x_idf = np.log(x.shape[0] - 1) - np.log(1 + np.asarray(np.sum(x > 0, axis=0)).ravel())
    x_idf = np.asarray(x_idf)
    x_idf_diag = scipy.sparse.lil_matrix((len(x_idf), len(x_idf)))
    x_idf_diag.setdiag(x_idf)
    x_tf = x.tocsr()
    x_tf.data = np.log(x_tf.data + 1)
    x_tfidf = x_tf * x_idf_diag
    return x_tfidf


def standardize(x):
    """
    takes sparse input and compute standardized version
    Note:
        cap at 5 std
    :param x: 2D scipy sparse data array to standardize (column-wise), must support row indexing
    :return: the object to perform scale (stores mean/std) for inference, as well as the scaled x
    """
    x_nzrow = x.any(axis=1)
    scaler = prep.StandardScaler().fit(x[x_nzrow, :])
    x_scaled = np.copy(x)
    x_scaled[x_nzrow, :] = scaler.transform(x_scaled[x_nzrow, :])
    x_scaled[x_scaled > 5] = 5
    x_scaled[x_scaled < -5] = -5
    x_scaled[np.absolute(x_scaled) < 1e-5] = 0
    return scaler, x_scaled


def standardize_2(x):
    """
    takes sparse input and compute standardized version
    Note:
        cap at 5 std
    :param x: 2D scipy sparse data array to standardize (column-wise), must support row indexing
    :return: the object to perform scale (stores mean/std) for inference, as well as the scaled x
    """
    x_nzrow = x.any(axis=1)
    scaler = prep.StandardScaler().fit(x[x_nzrow, :])
    x_scaled = np.copy(x)
    x_scaled[x_nzrow, :] = scaler.transform(x_scaled[x_nzrow, :])
    x_scaled[x_scaled > 1] = 1
    x_scaled[x_scaled < -1] = -1
    x_scaled[np.absolute(x_scaled) < 1e-5] = 0
    return scaler, x_scaled


def standardize_3(x):
    """
    takes sparse input and compute standardized version
    Note:
        cap at 5 std
    :param x: 2D scipy sparse data array to standardize (column-wise), must support row indexing
    :return: the object to perform scale (stores mean/std) for inference, as well as the scaled x
    """
    x_nzrow = x.any(axis=1)
    scaler = prep.StandardScaler().fit(x[x_nzrow, :])
    x_scaled = np.copy(x)
    x_scaled[x_nzrow, :] = scaler.transform(x_scaled[x_nzrow, :])
    x_scaled[x_nzrow, :] /= 2.
    x_scaled[x_scaled > 1] = 1
    x_scaled[x_scaled < -1] = -1
    x_scaled[np.absolute(x_scaled) < 1e-5] = 0
    return scaler, x_scaled


def prep_standardize_dense(x):
    """
    takes dense input and compute standardized version
    Note:
        cap at 5 std
    :param x: 2D numpy data array to standardize (column-wise)
    :return: the object to perform scale (stores mean/std) for inference, as well as the scaled x
    """
    scaler = prep.StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
    x_scaled[x_scaled > 5] = 5
    x_scaled[x_scaled < -5] = -5
    x_scaled[np.absolute(x_scaled) < 1e-5] = 0
    return scaler, x_scaled


idcg_array = np.arange(100) + 1
idcg_array = 1 / np.log2(idcg_array + 1)
idcg_table = np.zeros(100)
for i in range(100):
    idcg_table[i] = np.sum(idcg_array[:(i + 1)])
```

```python executionInfo={"elapsed": 1195, "status": "ok", "timestamp": 1629805633274, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="I_sREzjM0XEt"
def batch_eval_recall(_sess, tf_eval, eval_feed_dict, recall_k, eval_data):
    """
    given EvalData and DropoutNet compute graph in TensorFlow, runs batch evaluation
    :param _sess: tf session
    :param tf_eval: the evaluate output symbol in tf
    :param eval_feed_dict: method to parse tf, pick from EvalData method
    :param recall_k: list of thresholds to compute recall at (information retrieval recall)
    :param eval_data: EvalData instance
    :return: recall array at thresholds matching recall_k
    """
    tf_eval_preds_batch = []
    for (batch, (eval_start, eval_stop)) in enumerate(eval_data.eval_batch):
        tf_eval_preds = _sess.run(tf_eval,
                                  feed_dict=eval_feed_dict(
                                      batch, eval_start, eval_stop, eval_data))
        tf_eval_preds_batch.append(tf_eval_preds)
    tf_eval_preds = np.concatenate(tf_eval_preds_batch)
    tf.local_variables_initializer().run()

    # filter non-zero targets
    y_nz = [len(x) > 0 for x in eval_data.R_test_inf.rows]
    y_nz = np.arange(len(eval_data.R_test_inf.rows))[y_nz]

    preds_all = tf_eval_preds[y_nz, :]

    recall = []
    precision = []
    ndcg = []
    for at_k in recall_k:
        preds_k = preds_all[:, :at_k]
        y = eval_data.R_test_inf[y_nz, :]

        x = scipy.sparse.lil_matrix(y.shape)
        x.rows = preds_k
        x.data = np.ones_like(preds_k)

        z = y.multiply(x)
        recall.append(np.mean(np.divide((np.sum(z, 1)), np.sum(y, 1))))
        precision.append(np.mean(np.sum(z, 1) / at_k))

        x_coo = x.tocoo()
        rows = x_coo.row
        cols = x_coo.col
        y_csr = y.tocsr()
        dcg_array = y_csr[(rows, cols)].A1.reshape((preds_k.shape[0], -1))
        dcg = np.sum(dcg_array * idcg_array[:at_k].reshape((1, -1)), axis=1)
        idcg = np.sum(y, axis=1) - 1
        idcg[np.where(idcg >= at_k)] = at_k-1
        idcg = idcg_table[idcg.astype(int)]
        ndcg.append(np.mean(dcg / idcg))

    return recall, precision, ndcg


def batch_eval_store(_sess, tf_eval, eval_feed_dict, eval_data, save_path='./data/pred_R.npy'):
    """
    given EvalData and DropoutNet compute graph in TensorFlow, runs batch evaluation
    :param _sess: tf session
    :param tf_eval: the evaluate output symbol in tf
    :param eval_feed_dict: method to parse tf, pick from EvalData method
    :param recall_k: list of thresholds to compute recall at (information retrieval recall)
    :param eval_data: EvalData instance
    :return: recall array at thresholds matching recall_k
    """
    tf_eval_preds_batch = []
    for (batch, (eval_start, eval_stop)) in enumerate(eval_data.eval_batch):
        tf_eval_preds = _sess.run(tf_eval,
                                  feed_dict=eval_feed_dict(
                                      batch, eval_start, eval_stop, eval_data))
        tf_eval_preds_batch.append(tf_eval_preds)
    tf_eval_preds = np.concatenate(tf_eval_preds_batch)
    tf.local_variables_initializer().run()

    np.save(save_path, tf_eval_preds)


def negative_sampling(pos_user_array, pos_item_array, neg, item_warm):
    neg = int(neg)
    user_pos = pos_user_array.reshape((-1))
    user_neg = np.tile(pos_user_array, neg).reshape((-1))
    pos = pos_item_array.reshape((-1))
    neg = np.random.choice(item_warm, size=(neg * pos_user_array.shape[0]), replace=True).reshape((-1))
    target_pos = np.ones_like(pos)
    target_neg = np.zeros_like(neg)
    return np.concatenate((user_pos, user_neg)), np.concatenate((pos, neg)), \
           np.concatenate((target_pos, target_neg))


idcg_array = np.arange(100) + 1
idcg_array = 1 / np.log2(idcg_array + 1)
idcg_table = np.zeros(100)
for i in range(100):
    idcg_table[i] = np.sum(idcg_array[:(i + 1)])


def evaluate(_sess, tf_eval, eval_feed_dict, eval_data, like, filters, recall_k, test_file, cold_user=False, test_item_ids=None):
    tf_eval_preds_batch = []
    for (batch, (eval_start, eval_stop)) in enumerate(eval_data.eval_batch):
        tf_eval_preds = _sess.run(tf_eval,
                                  feed_dict=eval_feed_dict(
                                      batch, eval_start, eval_stop, eval_data))
        tf_eval_preds_batch.append(tf_eval_preds)
    tf_eval_preds = np.concatenate(tf_eval_preds_batch)
    tf.local_variables_initializer().run()

    test = pd.read_csv(test_file, dtype=np.int32)

    if not cold_user:
        test_item_ids = list(set(test['iid'].values))

    test_data = test.values.ravel().view(dtype=[('uid', np.int32), ('iid', np.int32)])

    item_old2new_list = np.zeros(np.max(test_item_ids) + 1)
    test_item_ids_map = dict()
    for i, iid in enumerate(test_item_ids):
        test_item_ids_map[iid] = i
        item_old2new_list[iid] = i

    _test_ij_for_inf = [(t[0], t[1]) for t in test_data if t[1] in test_item_ids_map]
    test_user_ids = np.unique(test_data['uid'])

    user_old2new_list = np.zeros(np.max(test_user_ids) + 1)
    test_user_ids_map = dict()
    for i, uid in enumerate(test_user_ids):
        test_user_ids_map[uid] = i
        user_old2new_list[uid] = i

    _test_i_for_inf = [test_user_ids_map[_t[0]] for _t in _test_ij_for_inf]
    _test_j_for_inf = [test_item_ids_map[_t[1]] for _t in _test_ij_for_inf]
    R_test_inf = scipy.sparse.coo_matrix(
        (np.ones(len(_test_i_for_inf)),
         (_test_i_for_inf, _test_j_for_inf)),
        shape=[len(test_user_ids), len(test_item_ids)]
    ).tolil(copy=False)

    # filter non-zero targets
    y_nz = [len(x) > 0 for x in R_test_inf.rows]
    y_nz = np.arange(len(R_test_inf.rows))[y_nz]

    preds_all = tf_eval_preds[y_nz, :]

    recall = []
    precision = []
    ndcg = []
    for at_k in recall_k:
        preds_k = preds_all[:, :at_k]
        y = R_test_inf[y_nz, :]

        x = scipy.sparse.lil_matrix(y.shape)
        x.rows = preds_k
        x.data = np.ones_like(preds_k)

        z = y.multiply(x)
        recall.append(np.mean(np.divide((np.sum(z, 1)), np.sum(y, 1))))
        precision.append(np.mean(np.sum(z, 1) / at_k))

        x_coo = x.tocoo()
        rows = x_coo.row
        cols = x_coo.col
        y_csr = y.tocsr()
        dcg_array = y_csr[(rows, cols)].A1.reshape((preds_k.shape[0], -1))
        dcg = np.sum(dcg_array * idcg_array[:at_k].reshape((1, -1)), axis=1)
        idcg = np.sum(y, axis=1) - 1
        idcg[np.where(idcg >= at_k)] = at_k - 1
        idcg = idcg_table[idcg.astype(int)]
        ndcg.append(np.mean(dcg / idcg))

    f_measure_1 = 2 * (precision[0] * recall[0]) / (precision[0] + recall[0]) if not precision[0] + recall[
        0] == 0 else 0
    f_measure_5 = 2 * (precision[1] * recall[1]) / (precision[1] + recall[1]) if not precision[1] + recall[
        1] == 0 else 0
    f_measure_10 = 2 * (precision[2] * recall[2]) / (precision[2] + recall[2]) if not precision[2] + recall[
        2] == 0 else 0
    f_score = [f_measure_1, f_measure_5, f_measure_10]

    print('\t\t' + '\t '.join([('@' + str(i)).ljust(6) for i in recall_k]))
    print('recall\t\t%s' % (
        ' '.join(['%.6f' % i for i in recall]),
    ))
    print('precision\t%s' % (
        ' '.join(['%.6f' % i for i in precision]),
    ))
    print('F1 score\t%s' % (
        ' '.join(['%.6f' % i for i in f_score]),
    ))
    print('NDCG\t\t%s' % (
        ' '.join(['%.6f' % i for i in ndcg]),
    ))

    return precision, recall, f_score, ndcg
```

<!-- #region id="kuuxYkZD0heu" -->
## Data
<!-- #endregion -->

```python executionInfo={"elapsed": 24, "status": "ok", "timestamp": 1629805633276, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="SsyIisVQ0iT-"
"""
This module contains class and methods related to data used in Heater  
"""


def load_eval_data(test_file, cold_user=False, test_item_ids=None):
    timer = timer_class()
    test = pd.read_csv(test_file, dtype=np.int32)
    if not cold_user:
        test_item_ids = list(set(test['iid'].values))
    test_data = test.values.ravel().view(dtype=[('uid', np.int32), ('iid', np.int32)])

    timer.toc('read %s triplets' % test_data.shape[0]).tic()
    eval_data = EvalData(
        test_data,
        test_item_ids)
    print(eval_data.get_stats_string())
    return eval_data


class EvalData:
    """
    EvalData:
        EvalData packages test triplet (user, item, score) into appropriate formats for evaluation
        Compact Indices:
            Specifically, this builds compact indices and stores mapping between original and compact indices.
            Compact indices only contains:
                1) items in test set
                2) users who interacted with such test items
            These compact indices speed up testing significantly by ignoring irrelevant users or items
        Args:
            test_triplets(int triplets): user-item-interaction_value triplet to build the test data
            train(int triplets): user-item-interaction_value triplet from train data
        Attributes:
            is_cold(boolean): whether test data is used for cold start problem
            test_item_ids(list of int): maps compressed item ids to original item ids (via position)
            test_item_ids_map(dictionary of int->int): maps original item ids to compressed item ids
            test_user_ids(list of int): maps compressed user ids to original user ids (via position)
            test_user_ids_map(dictionary of int->int): maps original user ids to compressed user ids
            R_test_inf(scipy lil matrix): pre-built compressed test matrix
            R_train_inf(scipy lil matrix): pre-built compressed train matrix for testing
            other relevant input/output exposed from tensorflow graph
    """

    def __init__(self, test_triplets, test_item_ids):
        # build map both-ways between compact and original indices
        # compact indices only contains:
        #  1) items in test set
        #  2) users who interacted with such test items

        self.test_item_ids = test_item_ids
        # test_item_ids_map
        self.test_item_ids_map = {iid: i for i, iid in enumerate(self.test_item_ids)}

        _test_ij_for_inf = [(t[0], t[1]) for t in test_triplets if t[1] in self.test_item_ids_map]
        # test_user_ids
        self.test_user_ids = np.unique(test_triplets['uid'])
        # test_user_ids_map
        self.test_user_ids_map = {user_id: i for i, user_id in enumerate(self.test_user_ids)}

        _test_i_for_inf = [self.test_user_ids_map[_t[0]] for _t in _test_ij_for_inf]
        _test_j_for_inf = [self.test_item_ids_map[_t[1]] for _t in _test_ij_for_inf]
        self.R_test_inf = scipy.sparse.coo_matrix(
            (np.ones(len(_test_i_for_inf)),
             (_test_i_for_inf, _test_j_for_inf)),
            shape=[len(self.test_user_ids), len(self.test_item_ids)]
        ).tolil(copy=False)

        # allocate fields
        self.U_pref_test = None
        self.V_pref_test = None
        self.V_content_test = None
        self.U_content_test = None
        self.tf_eval_train = None
        self.tf_eval_test = None
        self.eval_batch = None

    def init_tf(self, user_factors, item_factors, user_content, item_content, eval_run_batchsize,
                cold_user=False, cold_item=False):
        self.U_pref_test = user_factors[self.test_user_ids, :]
        self.V_pref_test = item_factors[self.test_item_ids, :]
        if cold_user:
            self.U_content_test = user_content[self.test_user_ids, :]
            if scipy.sparse.issparse(self.U_content_test):
                self.U_content_test = self.U_content_test.todense()
        if cold_item:
            self.V_content_test = item_content[self.test_item_ids, :]
            if scipy.sparse.issparse(self.V_content_test):
                self.V_content_test = self.V_content_test.todense()
        eval_l = self.R_test_inf.shape[0]
        self.eval_batch = [(x, min(x + eval_run_batchsize, eval_l)) for x
                           in range(0, eval_l, eval_run_batchsize)]

        self.tf_eval_train = []
        self.tf_eval_test = []

    def get_stats_string(self):
        return ('\tn_test_users:[%d]\n\tn_test_items:[%d]' % (len(self.test_user_ids), len(self.test_item_ids))
                + '\n\tR_train_inf: %s' % (
                    'no R_train_inf for cold'
                )
                + '\n\tR_test_inf: shape=%s nnz=[%d]' % (
                    str(self.R_test_inf.shape), len(self.R_test_inf.nonzero()[0])
                ))
```

<!-- #region id="RIz0vSgZ0GSx" -->
## Model
<!-- #endregion -->

```python executionInfo={"elapsed": 24, "status": "ok", "timestamp": 1629805633278, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="9fPYLdF10Jcg"
def l2_norm(para):
    return tf.reduce_sum(tf.square(para))


def dense_batch_fc_tanh(x, units, is_training, scope, do_norm=False):
    with tf.variable_scope(scope):
        init = tf.truncated_normal_initializer(stddev=0.01)
        h1_w = tf.get_variable(scope + '_w',
                               shape=[x.get_shape().as_list()[1], units],
                               initializer=init)
        h1_b = tf.get_variable(scope + '_b',
                               shape=[1, units],
                               initializer=tf.zeros_initializer())
        h1 = tf.matmul(x, h1_w) + h1_b
        if do_norm:
            h2 = tf.contrib.layers.batch_norm(
                h1,
                decay=0.9,
                center=True,
                scale=True,
                is_training=is_training,
                scope=scope + '_bn')
            return tf.nn.tanh(h2, scope + '_tanh'), l2_norm(h1_w) + l2_norm(h1_b)
        else:
            return tf.nn.tanh(h1, scope + '_tanh'), l2_norm(h1_w) + l2_norm(h1_b)


def dense_fc(x, units, scope):
    with tf.variable_scope(scope):
        init = tf.truncated_normal_initializer(stddev=0.01)
        h1_w = tf.get_variable(scope + '_w',
                               shape=[x.get_shape().as_list()[1], units],
                               initializer=init)
        h1_b = tf.get_variable(scope + '_b',
                               shape=[1, units],
                               initializer=tf.zeros_initializer())
        h1 = tf.matmul(x, h1_w) + h1_b
        return h1, l2_norm(h1_w) + l2_norm(h1_b)


class Heater:
    def __init__(self, latent_rank_in, user_content_rank, item_content_rank,
                 model_select, rank_out, reg, alpha, dim):

        self.rank_in = latent_rank_in  # input embedding dimension
        self.phi_u_dim = user_content_rank  # user content dimension
        self.phi_v_dim = item_content_rank  # item content dimension
        self.model_select = model_select  # model architecture
        self.rank_out = rank_out  # output dimension
        self.reg = reg
        self.alpha = alpha
        self.dim = dim

        # inputs
        self.Uin = None  # input user embedding
        self.Vin = None  # input item embedding
        self.Ucontent = None  # input user content
        self.Vcontent = None  # input item content
        self.is_training = None
        self.target = None  # input training target

        self.eval_trainR = None  # input training rating matrix for evaluation
        self.U_pref_tf = None
        self.V_pref_tf = None
        self.rand_target_ui = None

        # outputs in the model
        self.preds = None  # output of the model, the predicted scores
        self.optimizer = None  # the optimizer
        self.loss = None

        self.U_embedding = None  # new user embedding
        self.V_embedding = None  # new item embedding

        self.lr_placeholder = None  # learning rate

        # predictor
        self.tf_topk_vals = None
        self.tf_topk_inds = None
        self.preds_random = None
        self.tf_latent_topk_cold = None
        self.tf_latent_topk_warm = None
        self.eval_preds_warm = None  # the top-k predicted indices for warm evaluation
        self.eval_preds_cold = None  # the top-k predicted indices for cold evaluation

    def build_model(self):
        self.lr_placeholder = tf.placeholder(tf.float32, shape=[], name='learn_rate')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.target = tf.placeholder(tf.float32, shape=[None], name='target')

        self.Uin = tf.placeholder(tf.float32, shape=[None, self.rank_in], name='U_in_raw')
        self.Vin = tf.placeholder(tf.float32, shape=[None, self.rank_in], name='V_in_raw')

        dim = self.dim
        self.reg_loss = 0.

        if self.phi_v_dim > 0:
            self.Vcontent = tf.placeholder(tf.float32, shape=[None, self.phi_v_dim], name='V_content')
            self.dropout_item_indicator = tf.placeholder(tf.float32, shape=[None, 1], name='dropout_item_indicator')

            vcontent_gate, vcontent_gate_reg = dense_fc(self.Vcontent, dim,
                                                        'vcontent_gate_layer')  # size: batch_size X dim
            vcontent_gate = tf.nn.tanh(vcontent_gate)

            self.reg_loss += vcontent_gate_reg

            vcontent_expert_list = []
            for i in range(dim):
                tmp_expert = self.Vcontent
                for ihid, hid in enumerate(self.model_select):
                    tmp_expert, tmp_reg = dense_fc(tmp_expert, hid, 'Vexpert_' + str(ihid) + '_' + str(i))
                    tmp_expert = tf.nn.tanh(tmp_expert)
                    self.reg_loss += tmp_reg
                vcontent_expert_list.append(tf.reshape(tmp_expert, [-1, 1, self.rank_out]))

            vcontent_expert_concat = tf.concat(vcontent_expert_list, 1)  # size: batch_size X dim X self.rank_out

            vcontent_expert_concat = tf.linalg.matmul(tf.reshape(vcontent_gate, [-1, 1, dim]),
                                                      vcontent_expert_concat)
            Vcontent_last = tf.reshape(tf.nn.tanh(vcontent_expert_concat), [-1, self.rank_out])  # size: batch_size X self.rank_out

            self.Vin_filter = 1 - self.dropout_item_indicator

            diff_item_loss = self.alpha \
                             * (tf.reduce_sum(tf.reduce_sum(tf.square(Vcontent_last - self.Vin),
                                                            axis=1, keepdims=True)))
            v_last = (self.Vin * self.Vin_filter + Vcontent_last * (1 - self.Vin_filter))
        else:
            v_last = self.Vin
            diff_item_loss = 0

        if self.phi_u_dim > 0:
            self.Ucontent = tf.placeholder(tf.float32, shape=[None, self.phi_u_dim], name='U_content')
            self.dropout_user_indicator = tf.placeholder(tf.float32, shape=[None, 1], name='dropout_user_indicator')

            ucontent_gate, ucontent_gate_reg = dense_fc(self.Ucontent, dim,
                                                        'ucontent_gate_layer')  # size: batch_size X dim
            ucontent_gate = tf.nn.tanh(ucontent_gate)

            self.reg_loss += ucontent_gate_reg

            ucontent_expert_list = []
            for i in range(dim):
                tmp_expert = self.Ucontent
                for ihid, hid in enumerate(self.model_select):
                    tmp_expert, tmp_reg = dense_fc(tmp_expert, hid, 'Uexpert_' + str(ihid) + '_' + str(i))
                    tmp_expert = tf.nn.tanh(tmp_expert)
                    self.reg_loss += tmp_reg
                ucontent_expert_list.append(tf.reshape(tmp_expert, [-1, 1, self.rank_out]))

            ucontent_expert_concat = tf.concat(ucontent_expert_list, 1)  # size: batch_size X dim X self.rank_out

            ucontent_expert_concat = tf.linalg.matmul(tf.reshape(ucontent_gate, [-1, 1, dim]),
                                                      ucontent_expert_concat)
            Ucontent_last = tf.reshape(tf.nn.tanh(ucontent_expert_concat), [-1, self.rank_out])  # size: batch_size X self.rank_out

            self.Uin_filter = 1 - self.dropout_user_indicator

            diff_user_loss = self.alpha \
                             * (tf.reduce_sum(tf.reduce_sum(tf.square(Ucontent_last - self.Uin),
                                                            axis=1, keepdims=True)))
            u_last = (self.Uin * self.Uin_filter + Ucontent_last * (1 - self.Uin_filter))
        else:
            u_last = self.Uin
            diff_user_loss = 0

        for ihid, hid in enumerate([self.rank_out]):
            u_last, u_reg = dense_batch_fc_tanh(u_last, hid, self.is_training, 'user_layer_%d'%ihid,
                                                do_norm=True)
            v_last, v_reg = dense_batch_fc_tanh(v_last, hid, self.is_training, 'item_layer_%d'%ihid,
                                                do_norm=True)
            self.reg_loss += u_reg
            self.reg_loss += v_reg

        with tf.variable_scope("U_embedding"):
            u_emb_w = tf.Variable(tf.truncated_normal([u_last.get_shape().as_list()[1], self.rank_out], stddev=0.01),
                                  name='u_emb_w')
            u_emb_b = tf.Variable(tf.zeros([1, self.rank_out]), name='u_emb_b')
            self.U_embedding = tf.matmul(u_last, u_emb_w) + u_emb_b

        with tf.variable_scope("V_embedding"):
            v_emb_w = tf.Variable(tf.truncated_normal([v_last.get_shape().as_list()[1], self.rank_out], stddev=0.01),
                                  name='v_emb_w')
            v_emb_b = tf.Variable(tf.zeros([1, self.rank_out]), name='v_emb_b')
            self.V_embedding = tf.matmul(v_last, v_emb_w) + v_emb_b

        self.reg_loss += (l2_norm(v_emb_w) + l2_norm(v_emb_b) + l2_norm(u_emb_w) + l2_norm(u_emb_b))
        self.reg_loss *= self.reg

        with tf.variable_scope("loss"):
            preds = tf.multiply(self.U_embedding, self.V_embedding)
            self.preds = tf.reduce_sum(preds, 1)  # output of the model, the predicted scores
            self.diff_loss = diff_item_loss + diff_user_loss
            self.loss = tf.reduce_mean(tf.squared_difference(self.preds, self.target)) + self.reg_loss + self.diff_loss

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            self.optimizer = tf.train.MomentumOptimizer(self.lr_placeholder, 0.9).minimize(self.loss)

    def build_predictor(self, recall_at):
        self.eval_trainR = tf.sparse_placeholder(
            dtype=tf.float32, shape=[None, None], name='trainR_sparse')

        with tf.variable_scope("eval"):
            embedding_prod_cold = tf.matmul(self.U_embedding, self.V_embedding, transpose_b=True, name='pred_all_items')
            embedding_prod_warm = tf.sparse_add(embedding_prod_cold, self.eval_trainR)
            _, self.eval_preds_cold = tf.nn.top_k(embedding_prod_cold, k=recall_at[-1], sorted=True,
                                                  name='topK_net_cold')
            _, self.eval_preds_warm = tf.nn.top_k(embedding_prod_warm, k=recall_at[-1], sorted=True,
                                                  name='topK_net_warm')

    def get_eval_dict(self, _i, _eval_start, _eval_finish, eval_data):
        _eval_dict = {
            self.Uin: eval_data.U_pref_test[_eval_start:_eval_finish, :],
            self.Vin: eval_data.V_pref_test,
            self.is_training: False
        }

        if self.phi_v_dim > 0:
            zero_index = np.where(np.sum(eval_data.V_pref_test, axis=1) == 0)[0]
            dropout_item_indicator = np.zeros((len(eval_data.test_item_ids), 1))
            dropout_item_indicator[zero_index] = 1
            _eval_dict[self.dropout_item_indicator] = dropout_item_indicator
            _eval_dict[self.Vcontent] = eval_data.V_content_test
        if self.phi_u_dim > 0:
            zero_index = np.where(np.sum(eval_data.U_pref_test[_eval_start:_eval_finish, :], axis=1) == 0)[0]
            dropout_user_indicator = np.zeros((_eval_finish - _eval_start, 1))
            dropout_user_indicator[zero_index] = 1
            _eval_dict[self.dropout_user_indicator] = dropout_user_indicator
            _eval_dict[self.Ucontent] = eval_data.U_content_test[_eval_start:_eval_finish, :]
        return _eval_dict

    def get_eval_dict_latent(self, _i, _eval_start, _eval_finish, eval_data, u_pref, v_pref):
        _eval_dict = {
            self.U_pref_tf: u_pref[eval_data.test_user_ids[_eval_start:_eval_finish], :],
            self.V_pref_tf: v_pref[eval_data.test_item_ids, :]
        }
        if not eval_data.is_cold:
            _eval_dict[self.eval_trainR] = eval_data.tf_eval_train[_i]
        return _eval_dict
```

<!-- #region id="lTwqzoe-3DwV" -->
## Main
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 25, "status": "ok", "timestamp": 1629805633280, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="RSzE_rbt3e6N" outputId="519ae0a8-e743-4386-e574-19ba5862e8a1"
parser = argparse.ArgumentParser(description="main_LastFM",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data', type=str, default='LastFM', help='path to eval in the downloaded folder')
parser.add_argument('--model-select', nargs='+', type=int,
                    default=[200],
                    help='specify the fully-connected architecture, starting from input,'
                            ' numbers indicate numbers of hidden units',
                    )
parser.add_argument('--rank', type=int, default=200, help='output rank of latent model')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--eval-every', type=int, default=1, help='evaluate every X user-batch')
parser.add_argument('--neg', type=float, default=5, help='negative sampling rate')
parser.add_argument('--lr', type=float, default=0.005, help='starting learning rate')
parser.add_argument('--alpha', type=float, default=0.0001, help='diff loss parameter')
parser.add_argument('--reg', type=float, default=0.001, help='regularization')
parser.add_argument('--dim', type=int, default=5, help='number of experts')

args = parser.parse_args(args={})
args, _ = parser.parse_known_args()

for key in vars(args):
    print(key + ":" + str(vars(args)[key]))
```

```python executionInfo={"elapsed": 20, "status": "ok", "timestamp": 1629805633282, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="JV7omyS33Ywm"
def load_data(data_name):
    timer = timer_class(name='main').tic()
    data_path_silver = './data/silver/lastfm'
    u_file = data_path_silver + '/U_BPR.npy'
    v_file = data_path_silver + '/V_BPR.npy'
    user_content_file = data_path_silver + '/user_content.npz'
    train_file = data_path_silver + '/train.csv'
    test_file = data_path_silver + '/test.csv'
    vali_file = data_path_silver + '/vali.csv'
    with open(data_path_silver + '/info.pkl', 'rb') as f:
        info = pickle.load(f)
        num_user = info['num_user']
        num_item = info['num_item']

    dat = {}
    # load preference data
    timer.tic()

    u_pref = np.load(u_file)
    v_pref = np.load(v_file)

    dat['u_pref'] = u_pref
    dat['v_pref'] = v_pref

    timer.toc('loaded U:%s,V:%s' % (str(u_pref.shape), str(v_pref.shape))).tic()

    # pre-process
    _, dat['u_pref'] = standardize_2(u_pref)
    _, dat['v_pref'] = standardize(v_pref)

    timer.toc('standardized U,V').tic()

    # load content data
    timer.tic()
    user_content = scipy.sparse.load_npz(user_content_file)
    dat['user_content'] = user_content.tolil(copy=False)
    timer.toc('loaded user feature sparse matrix: %s' % (str(user_content.shape))).tic()

    # load split
    timer.tic()
    train = pd.read_csv(train_file, dtype=np.int32)
    dat['user_list'] = train['uid'].values
    dat['item_list'] = train['iid'].values
    dat['warm_item'] = np.unique(train['iid'].values)
    timer.toc('read train triplets %s' % str(train.shape)).tic()

    dat['test_eval'] = load_eval_data(test_file, cold_user=True, test_item_ids=dat['warm_item'])
    dat['vali_eval'] = load_eval_data(vali_file, cold_user=True, test_item_ids=dat['warm_item'])
    return dat
```

```python executionInfo={"elapsed": 20, "status": "ok", "timestamp": 1629805633283, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="yOoWjmtH7DNq"
data_name = args.data
model_select = args.model_select
rank_out = args.rank
data_batch_size = 1024
dropout = args.dropout
recall_at = [20, 50, 100]
eval_batch_size = 5000  # the batch size when test
eval_every = args.eval_every
num_epoch = 100
neg = args.neg

_lr = args.lr
_decay_lr_every = 2
_lr_decay = 0.9
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 10, "status": "ok", "timestamp": 1629805720697, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="Wly5WVFD69sq" outputId="75d167f0-fa36-4c27-90ef-6553710caac3"
dat = load_data(data_name)
u_pref = dat['u_pref']
v_pref = dat['v_pref']
test_eval = dat['test_eval']
vali_eval = dat['vali_eval']
user_content = dat['user_content']
user_list = dat['user_list']
item_list = dat['item_list']
item_warm = np.unique(item_list)
```

```python colab={"background_save": true, "base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1224091, "status": "ok", "timestamp": 1629806949019, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="BOpn2IwT3EWq"
timer = timer_class(name='main').tic()

# prep eval
eval_batch_size = eval_batch_size
timer.tic()
test_eval.init_tf(u_pref, v_pref, user_content, None, eval_batch_size, cold_user=True)  # init data for evaluation
vali_eval.init_tf(u_pref, v_pref, user_content, None, eval_batch_size, cold_user=True)  # init data for evaluation
timer.toc('initialized eval data').tic()

heater = Heater(latent_rank_in=u_pref.shape[1],
                        user_content_rank=user_content.shape[1],
                        item_content_rank=0,
                        model_select=model_select,
                        rank_out=rank_out, reg=args.reg, alpha=args.alpha, dim=args.dim)

config = tf.ConfigProto(allow_soft_placement=True)

heater.build_model()
heater.build_predictor(recall_at)

with tf.Session(config=config) as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    timer.toc('initialized tf')

    n_step = 0
    best_recall = 0
    best_test_recall = 0
    best_step = 0
    tf.local_variables_initializer().run()
    for epoch in range(num_epoch):
        user_array, item_array, target_array = negative_sampling(user_list, item_list, neg, item_warm)
        random_idx = np.random.permutation(user_array.shape[0])
        n_targets = len(random_idx)
        data_batch = [(n, min(n + data_batch_size, n_targets)) for n in range(0, n_targets, data_batch_size)]
        loss_epoch = 0.
        reg_loss_epoch = 0.
        diff_loss_epoch = 0.
        expert_loss_epoch = 0.
        gen = data_batch
        gen = tqdm(gen)
        for (start, stop) in gen:
            n_step += 1

            batch_idx = random_idx[start:stop]
            batch_users = user_array[batch_idx]
            batch_items = item_array[batch_idx]
            batch_targets = target_array[batch_idx]

            # dropout
            if dropout != 0:
                n_to_drop = int(np.floor(dropout * len(batch_idx)))  # number of u-i pairs to be dropped
                zero_index = np.random.choice(np.arange(len(batch_idx)), n_to_drop, replace=False)
            else:
                zero_index = np.array([])

            user_content_batch = user_content[batch_users, :].todense()
            dropout_indicator = np.zeros_like(batch_targets).reshape((-1, 1))
            if len(zero_index) > 0:
                dropout_indicator[zero_index] = 1

            _, _, loss_out, reg_loss_out, diff_loss_out = sess.run(
                [heater.preds, heater.optimizer, heater.loss,
                    heater.reg_loss, heater.diff_loss],
                feed_dict={
                    heater.Uin: u_pref[batch_users, :],
                    heater.Vin: v_pref[batch_items, :],
                    heater.Ucontent: user_content_batch,
                    heater.dropout_user_indicator: dropout_indicator,
                    heater.target: batch_targets,
                    heater.lr_placeholder: _lr,
                    heater.is_training: True
                }
            )
            loss_epoch += loss_out
            reg_loss_epoch += reg_loss_out
            diff_loss_epoch += diff_loss_out
            if np.isnan(loss_epoch):
                raise Exception('f is nan')

        if (epoch + 1) % _decay_lr_every == 0:
            _lr = _lr_decay * _lr
            print('decayed lr:' + str(_lr))

        if epoch % eval_every == 0:
            recall, precision, ndcg = batch_eval_recall(sess, heater.eval_preds_cold,
                                                                eval_feed_dict=heater.get_eval_dict,
                                                                recall_k=recall_at, eval_data=vali_eval)

        # checkpoint
        if np.sum(recall) > np.sum(best_recall):
            best_recall = recall
            test_recall, test_precision, test_ndcg = batch_eval_recall(sess, heater.eval_preds_cold,
                                                                                eval_feed_dict=heater.get_eval_dict,
                                                                                recall_k=recall_at,
                                                                                eval_data=test_eval)
            best_test_recall = test_recall
            best_epoch = epoch

        timer.toc('%d [%d]b loss=%.4f reg_loss=%.4f diff_loss=%.4f expert_loss=%.4f best[%d]' % (
            epoch, len(data_batch), loss_epoch, reg_loss_epoch, diff_loss_epoch, expert_loss_epoch, best_step
        )).tic()
        print('\t\t\t' + '\t '.join([('@' + str(i)).ljust(6) for i in recall_at]))
        print('Current recall\t\t%s' % (
            ' '.join(['%.6f' % i for i in recall]),
        ))
        print('Current precision\t%s' % (
            ' '.join(['%.6f' % i for i in precision]),
        ))
        print('Current ndcg\t\t%s' % (
            ' '.join(['%.6f' % i for i in ndcg]),
        ))

        print('Current test recall\t\t%s' % (
            ' '.join(['%.6f' % i for i in test_recall]),
        ))
        print('Current test precision\t%s' % (
            ' '.join(['%.6f' % i for i in test_precision]),
        ))
        print('Current test ndcg\t\t%s' % (
            ' '.join(['%.6f' % i for i in test_ndcg]),
        ))

        print('best epoch[%d]\t vali recall: %s' % (
            best_epoch,
            ' '.join(['%.6f' % i for i in best_recall]),
        ))
        print('best epoch[%d]\t test recall: %s' % (
            best_epoch,
            ' '.join(['%.6f' % i for i in best_test_recall]),
        ))
```
