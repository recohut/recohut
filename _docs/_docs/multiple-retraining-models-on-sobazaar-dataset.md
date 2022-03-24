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

<!-- #region id="1QG8dU-gzPyr" -->
# Multiple Retraining Models on Sobazaar dataset
<!-- #endregion -->

<!-- #region id="CxiWmRiFzT2X" -->
## Setup
<!-- #endregion -->

<!-- #region id="zVtJ4JTGH353" -->
### Git
<!-- #endregion -->

```python id="Z3qjPp055tXf" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1636620657504, "user_tz": -330, "elapsed": 4282, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f5f01ff8-ff91-4be8-d164-3a9d79753b0d"
import os
project_name = "incremental-learning"; branch = "T967215"; account = "sparsh-ai"
project_path = os.path.join('/content', branch)

if not os.path.exists(project_path):
    !cp -r /content/drive/MyDrive/git_credentials/. ~
    !mkdir "{project_path}"
    %cd "{project_path}"
    !git init
    !git remote add origin https://github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout -b "{branch}"
else:
    %cd "{project_path}"
```

```python id="xoKGydGDIwSX" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1636620657506, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8489373e-9df2-40dd-b93e-5da8ba4ef2ad"
%cd /content
```

```python id="7lAKlgUD5tXi" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1636624928503, "user_tz": -330, "elapsed": 20142, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1c95a717-c06e-4e2b-80e3-5596ea0f3abb"
!cd /content/T967215 && git add .
!cd /content/T967215 && git commit -m 'commit'
```

```python id="IqqZ6Do-uswE" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1636625001663, "user_tz": -330, "elapsed": 68068, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="46fd5b99-f995-4a04-d4e2-158835d8a214"
!cd /content/T967215 && git pull --rebase origin "{branch}"
!cd /content/T967215 && git push origin "{branch}"
```

```python id="evKxFrICIpy_"
# !mv /content/ckpts .
# !mv /content/soba_4mth_2014_1neg_30seq_1.parquet.snappy .
```

<!-- #region id="BXJY8c9d4Xi5" -->
### Installations
<!-- #endregion -->

<!-- #region id="BK-ZCkf00xZt" -->
### Dataset
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="5eJSFty70xW-" executionInfo={"status": "ok", "timestamp": 1636620661898, "user_tz": -330, "elapsed": 4398, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="01c07006-458b-42dc-8d89-596871e2f0a4"
!wget -q --show-progress https://github.com/RecoHut-Datasets/sobazaar/raw/main/Data/Sobazaar-hashID.csv.gz
!wget -q --show-progress https://github.com/sparsh-ai/incremental-learning/raw/T644011/soba_4mth_2014_1neg_30seq_1.parquet.snappy
```

<!-- #region id="GB_yDppW3_Yt" -->
### Imports
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="3lFs8AyO1IWc" executionInfo={"status": "ok", "timestamp": 1636620662736, "user_tz": -330, "elapsed": 864, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d8699e86-0805-4952-cbaf-ad8549dfa689"
%tensorflow_version 1.x
```

```python id="vrEmNkAAsQlM" executionInfo={"status": "ok", "timestamp": 1636620666859, "user_tz": -330, "elapsed": 4126, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import numpy as np
from tqdm.notebook import tqdm
import sys
import os
import logging
import pandas as pd
from os import path as osp
from pathlib import Path
import random
import datetime
import time
import glob

import bz2
import pickle
import _pickle as cPickle

import tensorflow as tf
```

<!-- #region id="NyxCtlrJ3_Ta" -->
### Params
<!-- #endregion -->

```python id="MXBwnUCD3_RD" executionInfo={"status": "ok", "timestamp": 1636620666861, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class Args:
    path_bronze = '/content'
    path_silver = '/content'

args = Args()
```

```python id="K5cAMUaO2H8W" executionInfo={"status": "ok", "timestamp": 1636620667896, "user_tz": -330, "elapsed": 1043, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(123)
```

<!-- #region id="Q40X4lHf4JHw" -->
### Logger
<!-- #endregion -->

```python id="cibwpV5L4JFb" executionInfo={"status": "ok", "timestamp": 1636620667897, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
logging.basicConfig(stream=sys.stdout,
                    level = logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] : %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

logger = logging.getLogger('Logger')
```

<!-- #region id="2M0-cN2ZzWE-" -->
## Modules
<!-- #endregion -->

<!-- #region id="qY9Y0q2sz1MS" -->
### Utils
<!-- #endregion -->

```python id="tH7lmOJbAOIf" executionInfo={"status": "ok", "timestamp": 1636620667897, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def save_pickle(data, title):
 with bz2.BZ2File(title + '.pbz2', 'w') as f: 
    cPickle.dump(data, f)

def load_pickle(path):
    data = bz2.BZ2File(path+'.pbz2', 'rb')
    data = cPickle.load(data)
    return data
```

```python id="5lHX1pHU7fvN" executionInfo={"status": "ok", "timestamp": 1636620667898, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class BatchLoader:
    """
    batch data loader by batch size
    return: [[users], [items], np.array(item_seqs_matrix), [seq_lens], [labels]] in batch iterator
    """

    def __init__(self, data_df, batch_size):

        self.data_df = data_df.reset_index(drop=True)  # df ['userId', 'itemId', 'label']
        self.data_df['index'] = self.data_df.index
        self.data_df['batch'] = self.data_df['index'].apply(lambda x: int(x / batch_size) + 1)
        self.num_batches = self.data_df['batch'].max()

    def get_batch(self, batch_id):

        batch = self.data_df[self.data_df['batch'] == batch_id]
        users = batch['userId'].tolist()
        items = batch['itemId'].tolist()
        labels = batch['label'].tolist()
        seq_lens = batch['itemSeq'].apply(len).tolist()

        item_seqs_matrix = np.zeros([len(batch), 30], np.int32)

        i = 0
        for itemSeq in batch['itemSeq'].tolist():
            for j in range(len(itemSeq)):
                item_seqs_matrix[i][j] = itemSeq[j]  # convert list of itemSeq into a matrix with zero padding
            i += 1

        return [users, items, item_seqs_matrix, seq_lens, labels]


def cal_roc_auc(scores, labels):

    arr = sorted(zip(scores, labels), key=lambda d: d[0], reverse=True)
    pos, neg = 0., 0.
    for record in arr:
        if record[1] == 1.:
            pos += 1
        else:
            neg += 1

    if pos == 0 or neg == 0:
        return None

    fp, tp = 0., 0.
    xy_arr = []
    for record in arr:
        if record[1] == 1.:
            tp += 1
        else:
            fp += 1
        xy_arr.append([fp/neg, tp/pos])

    auc = 0.
    prev_x = 0.
    prev_y = 0.
    for x, y in xy_arr:
        auc += ((x - prev_x) * (y + prev_y) / 2.)
        prev_x = x
        prev_y = y
    return auc


def cal_roc_gauc(users, scores, labels):
    # weighted sum of individual auc
    df = pd.DataFrame({'user': users,
                       'score': scores,
                       'label': labels})

    df_gb = df.groupby('user').agg(lambda x: x.tolist())

    auc_ls = []  # collect auc for all users
    user_imp_ls = []

    for row in df_gb.itertuples():
        auc = cal_roc_auc(row.score, row.label)
        if auc is None:
            pass
        else:
            auc_ls.append(auc)
            user_imp = len(row.label)
            user_imp_ls.append(user_imp)

    total_imp = sum(user_imp_ls)
    weighted_auc_ls = [auc * user_imp / total_imp for auc, user_imp in zip(auc_ls, user_imp_ls)]

    return sum(weighted_auc_ls)
```

```python id="oVSMgDeANiLH" executionInfo={"status": "ok", "timestamp": 1636621467385, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class BatchLoaderYsoft:
    """
    batch data loader by batch size with y_soft
    return: [[users], [items], np.array(item_seqs_matrix), [seq_lens], [labels], [labels_soft]] in batch iterator
    """

    def __init__(self, data_df, batch_size):

        self.data_df = data_df.reset_index(drop=True)  # df ['userId', 'itemId', 'label']
        self.data_df['index'] = self.data_df.index
        self.data_df['batch'] = self.data_df['index'].apply(lambda x: int(x / batch_size) + 1)
        self.num_batches = self.data_df['batch'].max()

    def get_batch(self, batch_id):

        batch = self.data_df[self.data_df['batch'] == batch_id]
        users = batch['userId'].tolist()
        items = batch['itemId'].tolist()
        labels = batch['label'].tolist()
        labels_soft = batch['label_soft'].tolist()
        seq_lens = batch['itemSeq'].apply(len).tolist()

        item_seqs_matrix = np.zeros([len(batch), 30], np.int32)

        i = 0
        for itemSeq in batch['itemSeq'].tolist():
            for j in range(len(itemSeq)):
                item_seqs_matrix[i][j] = itemSeq[j]  # convert list of itemSeq into a matrix with zero padding
            i += 1

        return [users, items, item_seqs_matrix, seq_lens, labels, labels_soft]
```

```python id="SvNkS4mP7T7m" executionInfo={"status": "ok", "timestamp": 1636620667898, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def average_pooling(emb, seq_len):
    mask = tf.sequence_mask(seq_len, tf.shape(emb)[-2], dtype=tf.float32)  # [B, T]
    mask = tf.expand_dims(mask, -1)  # [B, T, 1]
    emb *= mask  # [B, T, H]
    sum_pool = tf.reduce_sum(emb, -2)  # [B, H]
    avg_pool = tf.div(sum_pool, tf.expand_dims(tf.cast(seq_len, tf.float32), -1) + 1e-8)  # [B, H]
    return avg_pool
```

```python id="WQjMHSPwCTgU" executionInfo={"status": "ok", "timestamp": 1636620667899, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def search_ckpt(search_alias, mode='last'):
    ckpt_ls = glob.glob(search_alias)

    if mode == 'best logloss':
        metrics_ls = [float(ckpt.split('.ckpt')[0].split('_')[-1].split('TestLOGLOSS')[-1]) for ckpt in ckpt_ls]  # logloss
        selected_metrics_pos_ls = [i for i, x in enumerate(metrics_ls) if x == min(metrics_ls)]  # find all positions of the selected ckpts
    elif mode == 'best auc':
        metrics_ls = [float(ckpt.split('.ckpt')[0].split('_')[-2].split('TestAUC')[-1]) for ckpt in ckpt_ls]  # auc
        selected_metrics_pos_ls = [i for i, x in enumerate(metrics_ls) if x == max(metrics_ls)]  # find all positions of the selected ckpts
    else:  # mode == 'last'
        metrics_ls = [float(ckpt.split('.ckpt')[0].split('_')[-3].split('Epoch')[-1]) for ckpt in ckpt_ls]  # epoch no.
        selected_metrics_pos_ls = [i for i, x in enumerate(metrics_ls) if x == max(metrics_ls)]  # find all positions of the selected ckpts
    ckpt = ckpt_ls[max(selected_metrics_pos_ls)]  # get the full path of the last selected ckpt

    ckpt = ckpt.split('.ckpt')[0]  # get the path name before .ckpt
    ckpt = ckpt + '.ckpt'  # get the path with .ckpt
    return ckpt
```

```python id="RASF-mMAHU4u" executionInfo={"status": "ok", "timestamp": 1636620667900, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def parquet_to_csv(path):
    save_path = path.split('.parquet')[0]+'.csv'
    pd.read_parquet(path).to_csv(save_path)
    logger.info('csv file saved at {}'.format(save_path))
```

<!-- #region id="PguTj6gN2oj8" -->
### Dataset
<!-- #endregion -->

```python id="XQbXoMDO26pN" executionInfo={"status": "ok", "timestamp": 1636620667900, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def _gen_neg(num_items, pos_ls, num_neg):
    neg_ls = []
    for n in range(num_neg):  # generate num_neg
        neg = pos_ls[0]
        while neg in pos_ls:
            neg = random.randint(0, num_items - 1)
        neg_ls.append(neg)
    return neg_ls
```

```python id="jnzORRSw2p_v" executionInfo={"status": "ok", "timestamp": 1636620667901, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def preprocess_sobazaar():
    # convert csv into pandas dataframe
    data_path = osp.join(args.path_bronze,'Sobazaar-hashID.csv.gz')
    save_path = osp.join(args.path_silver,'soba_4mth_2014_1neg_30seq_1.csv')

    if not osp.exists(save_path):
        df = pd.read_csv(data_path)
        
        # preprocess
        df['date'] = df['Timestamp'].apply(lambda x: int(''.join(c for c in x.split('T')[0] if c.isdigit())))  # extract date and convert to int
        df['timestamp'] = df['Timestamp'].apply(lambda x: int(datetime.datetime.strptime(x.split('.')[0], '%Y-%m-%dT%H:%M:%S').timestamp()))
        df = df.drop(['Action', 'Timestamp'], axis=1)  # drop useless
        df.columns = ['itemId', 'userId', 'date', 'timestamp']  # rename
        df = df[['userId', 'itemId', 'date', 'timestamp']]  # switch columns

        # remap id
        user_id = sorted(df['userId'].unique().tolist())  # sort column
        user_map = dict(zip(user_id, range(len(user_id))))  # create map, key is original id, value is mapped id starting from 0
        df['userId'] = df['userId'].map(lambda x: user_map[x])  # map key to value in df

        item_id = sorted(df['itemId'].unique().tolist())  # sort column
        item_map = dict(zip(item_id, range(len(item_id))))  # create map, key is original id, value is mapped id starting from 0
        df['itemId'] = df['itemId'].map(lambda x: item_map[x])  # map key to value in df

        logger.info('dataframe head - {}'.format(df.head(20)))
        logger.info('num_users: {}'.format(len(user_map)))  # 17126
        logger.info('num_items: {}'.format(len(item_map)))  # 24785
        logger.info('num_records: {}'.format(len(df)))  # 842660

        # collect user history
        df_gb = df.groupby(['userId'])
        neg_lss = []
        num_neg = 1
        item_seqs = []
        max_len = 30
        count = 0
        for row in tqdm(df.itertuples(), total=df.shape[0]):
            user_df = df_gb.get_group(row.userId)
            user_history_df = user_df[user_df['timestamp'] <= row.timestamp].sort_values(['timestamp'], ascending=False).reset_index(drop=True)
            userHist = user_history_df['itemId'].unique().tolist()
            neg_lss.append(_gen_neg(len(item_map), userHist, num_neg))

            user_history_df = user_history_df[user_history_df['timestamp'] < row.timestamp].sort_values(['timestamp'], ascending=False).reset_index(drop=True)
            item_seq_ls = user_history_df['itemId'][:max_len].tolist()
            itemSeq = '#'.join(str(i) for i in item_seq_ls)
            item_seqs.append(itemSeq)

            count += 1
            if count % 100000 == 0:
                logger.info('done row {}'.format(count))

        df['neg_itemId_ls'] = neg_lss
        df['itemSeq'] = item_seqs

        users, itemseqs, items, labels, dates, timestamps = [], [], [], [], [], []
        for row in tqdm(df.itertuples(), total=df.shape[0]):
            users.append(row.userId)
            itemseqs.append(row.itemSeq)
            items.append(row.itemId)
            labels.append(1)  # positive samples have label 1
            dates.append(row.date)
            timestamps.append(row.timestamp)
            for j in range(num_neg):
                users.append(row.userId)
                itemseqs.append(row.itemSeq)
                items.append(row.neg_itemId_ls[j])
                labels.append(0)  # negative samples have label 0
                dates.append(row.date)
                timestamps.append(row.timestamp)

        df = pd.DataFrame({'userId': users,
                        'itemSeq': itemseqs,
                        'itemId': items,
                        'label': labels,
                        'date': dates,
                        'timestamp': timestamps})

        logger.info('dataframe head - {}'.format(df.head(20)))
        logger.info(len(df))  # 1685320

        # save csv and pickle
        # ['userId', 'itemSeq', 'itemId', 'label', 'date', 'timestamp']
        df.to_csv(save_path, index=False)
        logger.info('processed data saved at {}'.format(save_path))
    else:
        logger.info('File already exists at {}'.format(save_path))
```

<!-- #region id="x5we-S657T-E" -->
### Pretraining
<!-- #endregion -->

```python id="nxC7vY0i-DtE" executionInfo={"status": "ok", "timestamp": 1636620667901, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class EmbMLPnocate(object):
    """
        Embedding&MLP base model without item category
    """
    def __init__(self, hyperparams, train_config=None):

        self.train_config = train_config

        # create placeholder
        self.u = tf.placeholder(tf.int32, [None])  # [B]
        self.i = tf.placeholder(tf.int32, [None])  # [B]
        self.hist_i = tf.placeholder(tf.int32, [None, None])  # [B, T]
        self.hist_len = tf.placeholder(tf.int32, [None])  # [B]
        self.y = tf.placeholder(tf.float32, [None])  # [B]
        self.base_lr = tf.placeholder(tf.float32, [], name='base_lr')  # scalar

        # -- create emb begin -------
        user_emb_w = tf.get_variable("user_emb_w", [hyperparams['num_users'], hyperparams['user_embed_dim']])
        item_emb_w = tf.get_variable("item_emb_w", [hyperparams['num_items'], hyperparams['item_embed_dim']])
        # -- create emb end -------

        # -- create mlp begin ---
        concat_dim = hyperparams['user_embed_dim'] + hyperparams['item_embed_dim'] * 2
        with tf.variable_scope('fcn1'):
            fcn1_kernel = tf.get_variable(name='kernel', shape=[concat_dim, hyperparams['layers'][1]])
            fcn1_bias = tf.get_variable(name='bias', shape=[hyperparams['layers'][1]])
        with tf.variable_scope('fcn2'):
            fcn2_kernel = tf.get_variable(name='kernel', shape=[hyperparams['layers'][1], hyperparams['layers'][2]])
            fcn2_bias = tf.get_variable(name='bias', shape=[hyperparams['layers'][2]])
        with tf.variable_scope('fcn3'):
            fcn3_kernel = tf.get_variable(name='kernel', shape=[hyperparams['layers'][2], 1])
            fcn3_bias = tf.get_variable(name='bias', shape=[1])
        # -- create mlp end ---

        # -- emb begin -------
        u_emb = tf.nn.embedding_lookup(user_emb_w, self.u)  # [B, H]
        i_emb = tf.nn.embedding_lookup(item_emb_w, self.i)  # [B, H]
        h_emb = tf.nn.embedding_lookup(item_emb_w, self.hist_i)  # [B, T, H]
        u_hist = average_pooling(h_emb, self.hist_len)  # [B, H]
        # -- emb end -------

        # -- mlp begin -------
        fcn = tf.concat([u_emb, u_hist, i_emb], axis=-1)  # [B, H x 3]
        fcn_layer_1 = tf.nn.relu(tf.matmul(fcn, fcn1_kernel) + fcn1_bias)  # [B, l1]
        fcn_layer_2 = tf.nn.relu(tf.matmul(fcn_layer_1, fcn2_kernel) + fcn2_bias)  # [B, l2]
        fcn_layer_3 = tf.matmul(fcn_layer_2, fcn3_kernel) + fcn3_bias  # [B, 1]
        # -- mlp end -------

        logits = tf.reshape(fcn_layer_3, [-1])  # [B]
        self.scores = tf.sigmoid(logits)  # [B]

        # return same dimension as input tensors, let x = logits, z = labels, z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        self.losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.y)
        self.loss = tf.reduce_mean(self.losses)

        # base_optimizer
        if train_config['base_optimizer'] == 'adam':
            base_optimizer = tf.train.AdamOptimizer(learning_rate=self.base_lr)
        elif train_config['base_optimizer'] == 'rmsprop':
            base_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.base_lr)
        else:
            base_optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.base_lr)

        trainable_params = tf.trainable_variables()

        # update base model
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            base_grads = tf.gradients(self.loss, trainable_params)  # return a list of gradients (A list of `sum(dy/dx)` for each x in `xs`)
            base_grads_tuples = zip(base_grads, trainable_params)
            self.train_base_op = base_optimizer.apply_gradients(base_grads_tuples)

    def train_base(self, sess, batch):
        loss, _ = sess.run([self.loss, self.train_base_op], feed_dict={
            self.u: batch[0],
            self.i: batch[1],
            self.hist_i: batch[2],
            self.hist_len: batch[3],
            self.y: batch[4],
            self.base_lr: self.train_config['base_lr'],
        })
        return loss

    def inference(self, sess, batch):
        scores, losses = sess.run([self.scores, self.losses], feed_dict={
            self.u: batch[0],
            self.i: batch[1],
            self.hist_i: batch[2],
            self.hist_len: batch[3],
            self.y: batch[4],
        })
        return scores, losses
```

```python id="rzeAyivB-DrL" executionInfo={"status": "ok", "timestamp": 1636620667902, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class Engine(object):
    """
    Training epoch and test
    """

    def __init__(self, sess, model):

        self.sess = sess
        self.model = model

    def base_train_an_epoch(self, epoch_id, cur_set, train_config):

        train_start_time = time.time()

        if train_config['shuffle']:
            cur_set = cur_set.sample(frac=1)

        cur_batch_loader = BatchLoader(cur_set, train_config['base_bs'])

        base_loss_cur_sum = 0

        for i in range(1, cur_batch_loader.num_batches + 1):

            cur_batch = cur_batch_loader.get_batch(batch_id=i)

            base_loss_cur = self.model.train_base(self.sess, cur_batch)  # sess.run

            if (i - 1) % 100 == 0:
                print('[Epoch {} Batch {}] base_loss_cur {:.4f}, time elapsed {}'.format(epoch_id,
                                                                                         i,
                                                                                         base_loss_cur,
                                                                                         time.strftime('%H:%M:%S',
                                                                                                       time.gmtime(
                                                                                                           time.time() - train_start_time))))

            base_loss_cur_sum += base_loss_cur

        # epoch done, compute average loss
        base_loss_cur_avg = base_loss_cur_sum / cur_batch_loader.num_batches

        return base_loss_cur_avg

    def test(self, test_set, train_config):

        test_batch_loader = BatchLoader(test_set, train_config['base_bs'])

        scores, losses, labels = [], [], []
        for i in range(1, test_batch_loader.num_batches + 1):
            test_batch = test_batch_loader.get_batch(batch_id=i)
            batch_scores, batch_losses = self.model.inference(self.sess, test_batch)  # sees.run
            scores.extend(batch_scores.tolist())
            losses.extend(batch_losses.tolist())
            labels.extend(test_batch[4])

        test_auc = cal_roc_auc(scores, labels)
        test_logloss = sum(losses) / len(losses)

        return test_auc, test_logloss
```

```python id="YECKrjGc-Dow" executionInfo={"status": "ok", "timestamp": 1636620667903, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def pretrain_sobazaar():
    # load data to df
    start_time = time.time()

    load_path = osp.join(args.path_silver,'soba_4mth_2014_1neg_30seq_1.csv')
    data_df = pd.read_csv(load_path)

    data_df['itemSeq'] = data_df['itemSeq'].fillna('')  # empty seq are NaN
    data_df['itemSeq'] = data_df['itemSeq'].apply(lambda x: [int(item) for item in x.split('#') if item != ''])

    logger.info('Done loading data! time elapsed: {}'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))))

    num_users = data_df['userId'].max() + 1
    num_items = data_df['itemId'].max() + 1

    train_config = {'method': 'pretrain',
                    'dir_name': 'pretrain_train1-10_test11_10epoch',  # edit train test period range, number of epochs
                    'start_date': 20140901,  # overall train start date
                    'end_date': 20141231,  # overall train end date
                    'num_periods': 31,  # number of periods divided into
                    'train_start_period': 1,
                    'train_end_period': 10,
                    'test_period': 11,
                    'train_set_size': None,
                    'test_set_size': None,

                    'base_optimizer': 'adam',  # base model optimizer: adam, rmsprop, sgd
                    'base_lr': None,  # base model learning rate
                    'base_bs': 256,  # base model batch size
                    'base_num_epochs': 10,  # base model number of epochs
                    'shuffle': True,  # whether to shuffle the dataset for each epoch
                    }

    EmbMLPnocate_hyperparams = {'num_users': num_users,
                                'num_items': num_items,
                                'user_embed_dim': 8,
                                'item_embed_dim': 8,
                                'layers': [24, 12, 6, 1]  # input dim is user_embed_dim + item_embed_dim x 2
                                }

    # sort train data into periods based on num_periods
    data_df = data_df[(data_df['date'] >= train_config['start_date']) & (data_df['date'] <= train_config['end_date'])]
    data_df = data_df.sort_values(['timestamp']).reset_index(drop=True)
    records_per_period = int(len(data_df) / train_config['num_periods'])
    data_df['index'] = data_df.index
    data_df['period'] = data_df['index'].apply(lambda x: int(x / records_per_period) + 1)
    data_df = data_df[data_df.period != train_config['num_periods'] + 1]  # delete last extra period
    period_df = data_df.groupby('period')['date'].agg(['count', 'min', 'max'])
    data_df = data_df.drop(['index', 'date', 'timestamp'], axis=1)

    # build base model computation graph
    base_model = EmbMLPnocate(EmbMLPnocate_hyperparams, train_config=train_config)

    # create session
    sess = tf.Session()

    # create saver
    saver = tf.train.Saver(max_to_keep=80)

    orig_dir_name = train_config['dir_name']

    for base_lr in [1e-3]:

        print('')
        print('base_lr', base_lr)

        train_config['base_lr'] = base_lr

        train_config['dir_name'] = orig_dir_name + '_' + str(base_lr)
        print('dir_name: ', train_config['dir_name'])

        # create current and next set
        train_set = data_df[(data_df['period'] >= train_config['train_start_period']) &
                            (data_df['period'] <= train_config['train_end_period'])]
        test_set = data_df[data_df['period'] == train_config['test_period']]
        train_config['train_set_size'] = len(train_set)
        train_config['test_set_size'] = len(test_set)
        print('train set size', len(train_set), 'test set size', len(test_set))

        # checkpoints directory
        checkpoints_dir = os.path.join('ckpts', train_config['dir_name'])
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        # write train_config to text file
        with open(os.path.join(checkpoints_dir, 'config.txt'), mode='w') as f:
            f.write('train_config: ' + str(train_config) + '\n')
            f.write('\n')
            f.write('EmbMLPnocate_hyperparams: ' + str(EmbMLPnocate_hyperparams) + '\n')

        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        # create an engine instance
        engine = Engine(sess, base_model)

        train_start_time = time.time()

        for epoch_id in range(1, train_config['base_num_epochs'] + 1):

            print('Training Base Model Epoch {} Start!'.format(epoch_id))

            base_loss_cur_avg = engine.base_train_an_epoch(epoch_id, train_set, train_config)
            print('Epoch {} Done! time elapsed: {}, base_loss_cur_avg {:.4f}'.format(
                epoch_id,
                time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start_time)),
                base_loss_cur_avg))

            test_auc, test_logloss = engine.test(test_set, train_config)
            print('test_auc {:.4f}, test_logloss {:.4f}'.format(
                test_auc,
                test_logloss))
            print('time elapsed {}'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start_time))))

            print('')

            # save checkpoint
            checkpoint_alias = 'Epoch{}_TestAUC{:.4f}_TestLOGLOSS{:.4f}.ckpt'.format(
                epoch_id,
                test_auc,
                test_logloss)
            checkpoint_path = os.path.join(checkpoints_dir, checkpoint_alias)
            saver.save(sess, checkpoint_path)
```

<!-- #region id="hr7X8V0h-DmW" -->
### Incremental Update
<!-- #endregion -->

```python id="l41GttlhB0Ry" executionInfo={"status": "ok", "timestamp": 1636620668793, "user_tz": -330, "elapsed": 903, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def iu_sobazaar():
    # load data to df
    start_time = time.time()

    load_path = osp.join(args.path_silver,'soba_4mth_2014_1neg_30seq_1.csv')
    data_df = pd.read_csv(load_path)

    data_df['itemSeq'] = data_df['itemSeq'].fillna('')  # empty seq are NaN
    data_df['itemSeq'] = data_df['itemSeq'].apply(lambda x: [int(item) for item in x.split('#') if item != ''])

    logger.info('Done loading data! time elapsed: {}'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))))

    num_users = data_df['userId'].max() + 1
    num_items = data_df['itemId'].max() + 1

    train_config = {'method': 'IU_by_period',
                    'dir_name': 'IU_train11-23_test24-30_1epoch',  # edit train test period, number of epochs
                    'pretrain_model': 'pretrain_train1-10_test11_10epoch_0.001',
                    'start_date': 20140901,  # overall train start date
                    'end_date': 20141231,  # overall train end date
                    'num_periods': 31,  # number of periods divided into
                    'train_start_period': 11,
                    'test_start_period': 24,
                    'cur_period': None,  # current incremental period
                    'next_period': None,  # next incremental period
                    'cur_set_size': None,  # current incremental dataset size
                    'next_set_size': None,  # next incremental dataset size
                    'period_alias': None,  # individual period directory alias to save ckpts
                    'restored_ckpt_mode': 'best auc',  # mode to search the checkpoint to restore, 'best auc', 'best gauc', 'last'
                    'restored_ckpt': None,  # configure in the for loop

                    'base_optimizer': 'adam',  # base model optimizer: adam, rmsprop, sgd
                    'base_lr': None,  # base model learning rate
                    'base_bs': 256,  # base model batch size
                    'base_num_epochs': 1,  # base model number of epochs
                    'shuffle': True,  # whether to shuffle the dataset for each epoch
                    }

    EmbMLPnocate_hyperparams = {'num_users': num_users,
                                'num_items': num_items,
                                'user_embed_dim': 8,
                                'item_embed_dim': 8,
                                'layers': [24, 12, 6, 1]  # input dim is user_embed_dim + item_embed_dim x 2
                                }

    # sort train data into periods based on num_periods
    data_df = data_df[(data_df['date'] >= train_config['start_date']) & (data_df['date'] <= train_config['end_date'])]
    data_df = data_df.sort_values(['timestamp']).reset_index(drop=True)
    records_per_period = int(len(data_df) / train_config['num_periods'])
    data_df['index'] = data_df.index
    data_df['period'] = data_df['index'].apply(lambda x: int(x / records_per_period) + 1)
    data_df = data_df[data_df.period != train_config['num_periods'] + 1]  # delete last extra period
    period_df = data_df.groupby('period')['date'].agg(['count', 'min', 'max'])
    data_df = data_df.drop(['index', 'date', 'timestamp'], axis=1)

    orig_dir_name = train_config['dir_name']

    for base_lr in [1e-3]:

        print('')
        print('base_lr', base_lr)

        train_config['base_lr'] = base_lr

        train_config['dir_name'] = orig_dir_name + '_' + str(base_lr)
        print('dir_name: ', train_config['dir_name'])

        test_aucs = []
        test_loglosses = []

        for i in range(train_config['train_start_period'], train_config['num_periods']):

            # configure cur_period, next_period
            train_config['cur_period'] = i
            train_config['next_period'] = i + 1
            print('')
            print('current period: {}, next period: {}'.format(
                train_config['cur_period'],
                train_config['next_period']))
            print('')

            # create current and next set
            cur_set = data_df[data_df['period'] == train_config['cur_period']]
            next_set = data_df[data_df['period'] == train_config['next_period']]
            train_config['cur_set_size'] = len(cur_set)
            train_config['next_set_size'] = len(next_set)
            print('current set size', len(cur_set), 'next set size', len(next_set))

            train_config['period_alias'] = 'period' + str(i)

            # checkpoints directory
            ckpts_dir = os.path.join('ckpts', train_config['dir_name'], train_config['period_alias'])
            if not os.path.exists(ckpts_dir):
                os.makedirs(ckpts_dir)

            if i == train_config['train_start_period']:
                search_alias = os.path.join('ckpts', train_config['pretrain_model'], 'Epoch*')
                train_config['restored_ckpt'] = search_ckpt(search_alias, mode=train_config['restored_ckpt_mode'])
            else:
                prev_period_alias = 'period' + str(i - 1)
                search_alias = os.path.join('ckpts', train_config['dir_name'], prev_period_alias, 'Epoch*')
                train_config['restored_ckpt'] = search_ckpt(search_alias, mode=train_config['restored_ckpt_mode'])
            print('restored checkpoint: {}'.format(train_config['restored_ckpt']))

            # write train_config to text file
            with open(os.path.join(ckpts_dir, 'config.txt'), mode='w') as f:
                f.write('train_config: ' + str(train_config) + '\n')
                f.write('\n')
                f.write('EmbMLPnocate_hyperparams: ' + str(EmbMLPnocate_hyperparams) + '\n')

            # build base model computation graph
            tf.reset_default_graph()
            base_model = EmbMLPnocate(EmbMLPnocate_hyperparams, train_config=train_config)

            # create session
            with tf.Session() as sess:
                
                saver = tf.train.Saver()
                saver.restore(sess, train_config['restored_ckpt'])
                # create an engine instance with base_model
                engine = Engine(sess, base_model)
                train_start_time = time.time()
                max_auc = 0
                best_logloss = 0

                for epoch_id in range(1, train_config['base_num_epochs'] + 1):
                    print('Training Base Model Epoch {} Start!'.format(epoch_id))
                    base_loss_cur_avg = engine.base_train_an_epoch(epoch_id, cur_set, train_config)
                    print('Epoch {} Done! time elapsed: {}, base_loss_cur_avg {:.4f}'.format(
                        epoch_id,
                        time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start_time)),
                        base_loss_cur_avg))
                    cur_auc, cur_logloss = engine.test(cur_set, train_config)
                    next_auc, next_logloss = engine.test(next_set, train_config)
                    print('cur_auc {:.4f}, cur_logloss {:.4f}, next_auc {:.4f}, next_logloss {:.4f}'.format(
                        cur_auc,
                        cur_logloss,
                        next_auc,
                        next_logloss))
                    print('time elapsed {}'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start_time))))
                    print('')
                    # save checkpoint
                    checkpoint_alias = 'Epoch{}_TestAUC{:.4f}_TestLOGLOSS{:.4f}.ckpt'.format(
                        epoch_id,
                        next_auc,
                        next_logloss)
                    checkpoint_path = os.path.join(ckpts_dir, checkpoint_alias)
                    saver.save(sess, checkpoint_path)
                    if next_auc > max_auc:
                        max_auc = next_auc
                        best_logloss = next_logloss

                if i >= train_config['test_start_period']:
                    test_aucs.append(max_auc)
                    test_loglosses.append(best_logloss)

            if i >= train_config['test_start_period']:
                average_auc = sum(test_aucs) / len(test_aucs)
                average_logloss = sum(test_loglosses) / len(test_loglosses)
                print('test aucs', test_aucs)
                print('average auc', average_auc)
                print('')
                print('test loglosses', test_loglosses)
                print('average logloss', average_logloss)

                # write metrics to text file
                with open(os.path.join(ckpts_dir, 'test_metrics.txt'), mode='w') as f:
                    f.write('test_aucs: ' + str(test_aucs) + '\n')
                    f.write('average_auc: ' + str(average_auc) + '\n')
                    f.write('test_loglosses: ' + str(test_loglosses) + '\n')
                    f.write('average_logloss: ' + str(average_logloss) + '\n')
```

<!-- #region id="sD75F5a3GbNR" -->
### Batch Update
<!-- #endregion -->

```python id="_q1TeLujGbKx" executionInfo={"status": "ok", "timestamp": 1636620668794, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def bu_sobazaar():
    # load data to df
    start_time = time.time()

    load_path = osp.join(args.path_silver,'soba_4mth_2014_1neg_30seq_1.csv')
    data_df = pd.read_csv(load_path)

    data_df['itemSeq'] = data_df['itemSeq'].fillna('')  # empty seq are NaN
    data_df['itemSeq'] = data_df['itemSeq'].apply(lambda x: [int(item) for item in x.split('#') if item != ''])

    logger.info('Done loading data! time elapsed: {}'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))))

    num_users = data_df['userId'].max() + 1
    num_items = data_df['itemId'].max() + 1

    train_config = {'method': 'BU_by_period',
                    'dir_name': 'BU_train11-23_test24-30_7_1epoch',  # edit train test period, window size, number of epochs
                    'pretrain_model': 'pretrain_train1-10_test11_10epoch_0.001',
                    'start_date': 20140901,  # overall train start date
                    'end_date': 20141231,  # overall train end date
                    'num_periods': 31,  # number of periods divided into
                    'train_start_period': 11,
                    'test_start_period': 24,
                    'window_size': 7,  # number of periods or 'full' for full retraining
                    'cur_periods': None,  # current batch periods
                    'next_period': None,  # next incremental period
                    'cur_set_size': None,  # current batch dataset size
                    'next_set_size': None,  # next incremental dataset size
                    'period_alias': None,  # individual period directory alias to save ckpts
                    'restored_ckpt_mode': 'best auc',  # mode to search the checkpoint to restore, 'best auc', 'best gauc', 'last'
                    'restored_ckpt': None,  # configure in the for loop

                    'base_optimizer': 'adam',  # base model optimizer: adam, rmsprop, sgd
                    'base_lr': None,  # base model learning rate
                    'base_bs': 256,  # base model batch size
                    'base_num_epochs': 1,  # base model number of epochs
                    'shuffle': True,  # whether to shuffle the dataset for each epoch
                    }

    EmbMLPnocate_hyperparams = {'num_users': num_users,
                                'num_items': num_items,
                                'user_embed_dim': 8,
                                'item_embed_dim': 8,
                                'layers': [24, 12, 6, 1]  # input dim is user_embed_dim + item_embed_dim x 2
                                }

    # sort train data into periods based on num_periods
    data_df = data_df[(data_df['date'] >= train_config['start_date']) & (data_df['date'] <= train_config['end_date'])]
    data_df = data_df.sort_values(['timestamp']).reset_index(drop=True)
    records_per_period = int(len(data_df) / train_config['num_periods'])
    data_df['index'] = data_df.index
    data_df['period'] = data_df['index'].apply(lambda x: int(x / records_per_period) + 1)
    data_df = data_df[data_df.period != train_config['num_periods'] + 1]  # delete last extra period
    period_df = data_df.groupby('period')['date'].agg(['count', 'min', 'max'])
    data_df = data_df.drop(['index', 'date', 'timestamp'], axis=1)

    orig_dir_name = train_config['dir_name']

    for base_lr in [1e-3]:

        print('')
        print('base_lr', base_lr)

        train_config['base_lr'] = base_lr

        train_config['dir_name'] = orig_dir_name + '_' + str(base_lr)
        print('dir_name: ', train_config['dir_name'])

        test_aucs = []
        test_loglosses = []

        for i in range(train_config['train_start_period'], train_config['num_periods']):

            # configure cur_periods, next_period
            if train_config['window_size'] == 'full':
                train_config['cur_periods'] = [i - prev_num for prev_num in reversed(range(i))]
            else:
                train_config['cur_periods'] = [i - prev_num for prev_num in reversed(range(train_config['window_size']))]
            train_config['next_period'] = i + 1
            print('')
            print('current periods: {}, next period: {}'.format(
                train_config['cur_periods'],
                train_config['next_period']))
            print('')

            # create current and next set
            cur_set = data_df[data_df['period'].isin(train_config['cur_periods'])]
            next_set = data_df[data_df['period'] == train_config['next_period']]
            train_config['cur_set_size'] = len(cur_set)
            train_config['next_set_size'] = len(next_set)
            print('current set size', len(cur_set), 'next set size', len(next_set))

            train_config['period_alias'] = 'period' + str(i)

            # checkpoints directory
            ckpts_dir = os.path.join('ckpts', train_config['dir_name'], train_config['period_alias'])
            if not os.path.exists(ckpts_dir):
                os.makedirs(ckpts_dir)

            if i == train_config['train_start_period']:
                search_alias = os.path.join('ckpts', train_config['pretrain_model'], 'Epoch*')
                train_config['restored_ckpt'] = search_ckpt(search_alias, mode=train_config['restored_ckpt_mode'])
            else:
                prev_period_alias = 'period' + str(i - 1)
                search_alias = os.path.join('ckpts', train_config['dir_name'], prev_period_alias, 'Epoch*')
                train_config['restored_ckpt'] = search_ckpt(search_alias, mode=train_config['restored_ckpt_mode'])
            print('restored checkpoint: {}'.format(train_config['restored_ckpt']))

            # write train_config to text file
            with open(os.path.join(ckpts_dir, 'config.txt'), mode='w') as f:
                f.write('train_config: ' + str(train_config) + '\n')
                f.write('\n')
                f.write('EmbMLPnocate_hyperparams: ' + str(EmbMLPnocate_hyperparams) + '\n')

            # build base model computation graph
            tf.reset_default_graph()
            base_model = EmbMLPnocate(EmbMLPnocate_hyperparams, train_config=train_config)

            # create session
            with tf.Session() as sess:
                saver = tf.train.Saver()
                saver.restore(sess, train_config['restored_ckpt'])

                # create an engine instance with base_model
                engine = Engine(sess, base_model)
                train_start_time = time.time()
                max_auc = 0
                best_logloss = 0

                for epoch_id in range(1, train_config['base_num_epochs'] + 1):
                    print('Training Base Model Epoch {} Start!'.format(epoch_id))
                    base_loss_cur_avg = engine.base_train_an_epoch(epoch_id, cur_set, train_config)
                    print('Epoch {} Done! time elapsed: {}, base_loss_cur_avg {:.4f}'.format(
                        epoch_id,
                        time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start_time)),
                        base_loss_cur_avg))
                    cur_auc, cur_logloss = engine.test(cur_set, train_config)
                    next_auc, next_logloss = engine.test(next_set, train_config)
                    print('cur_auc {:.4f}, cur_logloss {:.4f}, next_auc {:.4f}, next_logloss {:.4f}'.format(
                        cur_auc,
                        cur_logloss,
                        next_auc,
                        next_logloss))
                    print('time elapsed {}'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start_time))))
                    print('')
                    # save checkpoint
                    checkpoint_alias = 'Epoch{}_TestAUC{:.4f}_TestLOGLOSS{:.4f}.ckpt'.format(
                        epoch_id,
                        next_auc,
                        next_logloss)
                    checkpoint_path = os.path.join(ckpts_dir, checkpoint_alias)
                    saver.save(sess, checkpoint_path)

                    if next_auc > max_auc:
                        max_auc = next_auc
                        best_logloss = next_logloss

                if i >= train_config['test_start_period']:
                    test_aucs.append(max_auc)
                    test_loglosses.append(best_logloss)

            if i >= train_config['test_start_period']:
                average_auc = sum(test_aucs) / len(test_aucs)
                average_logloss = sum(test_loglosses) / len(test_loglosses)
                print('test aucs', test_aucs)
                print('average auc', average_auc)
                print('')
                print('test loglosses', test_loglosses)
                print('average logloss', average_logloss)

                # write metrics to text file
                with open(os.path.join(ckpts_dir, 'test_metrics.txt'), mode='w') as f:
                    f.write('test_aucs: ' + str(test_aucs) + '\n')
                    f.write('average_auc: ' + str(average_auc) + '\n')
                    f.write('test_loglosses: ' + str(test_loglosses) + '\n')
                    f.write('average_logloss: ' + str(average_logloss) + '\n')
```

<!-- #region id="QisHmVRqK7Bf" -->
### SPMF
<!-- #endregion -->

```python id="A8IzVQtPK6-2" executionInfo={"status": "ok", "timestamp": 1636622923604, "user_tz": -330, "elapsed": 1472, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def spmf_sobazaar():
    # load data to df
    start_time = time.time()

    load_path = osp.join(args.path_silver,'soba_4mth_2014_1neg_30seq_1.csv')
    data_df = pd.read_csv(load_path)

    data_df['itemSeq'] = data_df['itemSeq'].fillna('')  # empty seq are NaN
    data_df['itemSeq'] = data_df['itemSeq'].apply(lambda x: [int(item) for item in x.split('#') if item != ''])

    logger.info('Done loading data! time elapsed: {}'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))))

    num_users = data_df['userId'].max() + 1
    num_items = data_df['itemId'].max() + 1

    train_config = {'method': 'SPMF_by_period',
                    'dir_name': 'SPMF_2_train11-23_test24-30_1epoch',  # edit strategy type, train test period, number of epochs
                    'pretrain_model': 'pretrain_train1-10_test11_10epoch_0.001',
                    'start_date': 20140901,  # overall train start date
                    'end_date': 20141231,  # overall train end date
                    'num_periods': 31,  # number of periods divided into
                    'train_start_period': 11,
                    'test_start_period': 24,
                    'cur_period': None,  # current incremental period
                    'next_period': None,  # next incremental period
                    'cur_set_size': None,  # current incremental dataset size
                    'next_set_size': None,  # next incremental dataset size
                    'period_alias': None,  # individual period directory alias to save ckpts
                    'restored_ckpt_mode': 'best auc',  # mode to search the checkpoint to restore, 'best auc', 'best gauc', 'last'
                    'restored_ckpt': None,  # configure in the for loop

                    'strategy': 2,  # two different sampling strategies
                    'frac_of_pretrain_D': None,  # reservoir size as a fraction of pretrain dataset, less than or equal to 1
                    'res_cur_ratio': None,  # the ratio of reservoir sample to current set, only for strategy 2

                    'base_optimizer': 'adam',  # base model optimizer: adam, rmsprop, sgd
                    'base_lr': None,  # base model learning rate
                    'base_bs': 256,  # base model batch size
                    'base_num_epochs': 1,  # base model number of epochs
                    'shuffle': False,  # whether to shuffle the dataset for each epoch
                    }

    EmbMLPnocate_hyperparams = {'num_users': num_users,
                                'num_items': num_items,
                                'user_embed_dim': 8,
                                'item_embed_dim': 8,
                                'layers': [24, 12, 6, 1]  # input dim is user_embed_dim + item_embed_dim x 2
                                }

    # sort train data into periods based on num_periods
    data_df = data_df[(data_df['date'] >= train_config['start_date']) & (data_df['date'] <= train_config['end_date'])]
    data_df = data_df.sort_values(['timestamp']).reset_index(drop=True)
    records_per_period = int(len(data_df) / train_config['num_periods'])
    data_df['index'] = data_df.index
    data_df['period'] = data_df['index'].apply(lambda x: int(x / records_per_period) + 1)
    data_df = data_df[data_df.period != train_config['num_periods'] + 1]  # delete last extra period
    period_df = data_df.groupby('period')['date'].agg(['count', 'min', 'max'])
    data_df = data_df.drop(['index', 'date', 'timestamp'], axis=1)

    orig_dir_name = train_config['dir_name']

    for frac in [1]:

        for ratio in [0.1]:

            for base_lr in [1e-3]:

                print('')
                print('frac_of_pretrain_D', frac, 'res_cur_ratio', ratio, 'base_lr', base_lr)

                train_config['frac_of_pretrain_D'] = frac
                train_config['res_cur_ratio'] = ratio
                train_config['base_lr'] = base_lr

                train_config['dir_name'] = orig_dir_name + '_' + str(frac) + '_' + str(ratio) + '_' + str(base_lr)  # for strategy 2
                # train_config['dir_name'] = orig_dir_name + '_' + str(frac) + '_' + str(base_lr)  # for strategy 1
                print('dir_name: ', train_config['dir_name'])

                test_aucs = []
                test_loglosses = []

                for i in range(train_config['train_start_period'], train_config['num_periods']):

                    # configure cur_period, next_period
                    train_config['cur_period'] = i
                    train_config['next_period'] = i + 1
                    print('')
                    print('current period: {}, next period: {}'.format(
                        train_config['cur_period'],
                        train_config['next_period']))
                    print('')

                    # create current and next set
                    cur_set = data_df[data_df['period'] == train_config['cur_period']]
                    next_set = data_df[data_df['period'] == train_config['next_period']]
                    train_config['cur_set_size'] = len(cur_set)
                    train_config['next_set_size'] = len(next_set)
                    print('current set size', len(cur_set), 'next set size', len(next_set))

                    # create train
                    pos_cur_set = cur_set[cur_set['label'] == 1]
                    neg_cur_set = cur_set[cur_set['label'] == 0]

                    if i == train_config['train_start_period']:
                        pos_pretrain_set = data_df[(data_df['period'] < train_config['train_start_period']) & (data_df['label'] == 1)]
                        reservoir_size = int(len(pos_pretrain_set) * train_config['frac_of_pretrain_D'])
                        reservoir = pos_pretrain_set.sample(n=reservoir_size)

                        neg_pretrain_set = data_df[(data_df['period'] < train_config['train_start_period']) & (data_df['label'] == 0)]
                        neg_reservoir_size = int(len(neg_pretrain_set) * train_config['frac_of_pretrain_D'])
                        neg_reservoir = neg_pretrain_set.sample(n=neg_reservoir_size)

                    train_config['period_alias'] = 'period' + str(i)

                    # checkpoints directory
                    ckpts_dir = os.path.join('ckpts', train_config['dir_name'], train_config['period_alias'])
                    if not os.path.exists(ckpts_dir):
                        os.makedirs(ckpts_dir)

                    if i == train_config['train_start_period']:
                        search_alias = os.path.join('ckpts', train_config['pretrain_model'], 'Epoch*')
                        train_config['restored_ckpt'] = search_ckpt(search_alias, mode=train_config['restored_ckpt_mode'])
                    else:
                        prev_period_alias = 'period' + str(i - 1)
                        search_alias = os.path.join('ckpts', train_config['dir_name'], prev_period_alias, 'Epoch*')
                        train_config['restored_ckpt'] = search_ckpt(search_alias, mode=train_config['restored_ckpt_mode'])
                    print('restored checkpoint: {}'.format(train_config['restored_ckpt']))

                    # write train_config to text file
                    with open(os.path.join(ckpts_dir, 'config.txt'), mode='w') as f:
                        f.write('train_config: ' + str(train_config) + '\n')
                        f.write('\n')
                        f.write('EmbMLPnocate_hyperparams: ' + str(EmbMLPnocate_hyperparams) + '\n')

                    # build base model computation graph
                    tf.reset_default_graph()
                    base_model = EmbMLPnocate(EmbMLPnocate_hyperparams, train_config=train_config)

                    # create session
                    with tf.Session() as sess:
                        saver = tf.train.Saver()
                        saver.restore(sess, train_config['restored_ckpt'])

                        def compute_prob_and_gen_set_and_update_reservoir():

                            """
                            this strategy follows exactly the method from the paper "Streaming ranking based recommender systems"
                            train_set = samples of (current_set + reservoir)
                            """
                            compute_prob_start_time = time.time()

                            pos_train_set = pd.concat([reservoir, pos_cur_set], ignore_index=False)  # combine R and W
                            neg_train_set = pd.concat([neg_reservoir, neg_cur_set], ignore_index=False)  # combine R and W

                            # compute prob
                            pos_train_batch_loader = BatchLoader(pos_train_set, train_config['base_bs'])

                            scores = []
                            for i in range(1, pos_train_batch_loader.num_batches + 1):
                                pos_train_batch = pos_train_batch_loader.get_batch(batch_id=i)
                                batch_scores, batch_losses = base_model.inference(sess, pos_train_batch)  # sess.run
                                scores.extend(batch_scores)

                            ordered_pos_train_set = pos_train_set
                            ordered_pos_train_set['score'] = scores
                            ordered_pos_train_set = ordered_pos_train_set.sort_values(['score'], ascending=False).reset_index(drop=True)  # edit
                            ordered_pos_train_set['rank'] = np.arange(len(ordered_pos_train_set))
                            total_num = len(pos_train_set)
                            ordered_pos_train_set['weight'] = ordered_pos_train_set['rank'].apply(lambda x: np.exp(x / total_num))
                            total_weights = ordered_pos_train_set['weight'].sum()
                            ordered_pos_train_set['prob'] = ordered_pos_train_set['weight'].apply(lambda x: x / total_weights)
                            ordered_pos_train_set = ordered_pos_train_set.drop(['score', 'rank', 'weight'], axis=1)

                            # generate train set
                            sampled_pos_train_set = ordered_pos_train_set.sample(n=len(pos_cur_set), replace=False, weights='prob')
                            sampled_pos_train_set = sampled_pos_train_set.drop(['prob'], axis=1)
                            sampled_neg_train_set = neg_train_set.sample(n=len(neg_cur_set), replace=False)
                            sampled_train_set = pd.concat([sampled_pos_train_set, sampled_neg_train_set], ignore_index=False)
                            sampled_train_set = sampled_train_set.sort_values(['period']).reset_index(drop=True)

                            # update pos reservoir
                            t = len(data_df[(data_df['period'] < train_config['cur_period']) & (data_df['label'] == 1)])
                            probs_to_res = len(reservoir) / (t + np.arange(len(pos_cur_set)) + 1)
                            random_probs = np.random.rand(len(pos_cur_set))
                            selected_pos_cur_set = pos_cur_set[probs_to_res > random_probs]
                            num_left_in_res = len(reservoir) - len(selected_pos_cur_set)
                            updated_reservoir = pd.concat([reservoir.sample(n=num_left_in_res), selected_pos_cur_set], ignore_index=False)
                            print('selected_pos_cur_set size', len(selected_pos_cur_set))
                            # print('num_in_res', len(reservoir))
                            # print('num_left_in_res', num_left_in_res)
                            # print('num_in_updated_res', len(updated_reservoir))

                            # update neg reservoir
                            t = len(data_df[(data_df['period'] < train_config['cur_period']) & (data_df['label'] == 0)])
                            probs_to_res = len(neg_reservoir) / (t + np.arange(len(neg_cur_set)) + 1)
                            random_probs = np.random.rand(len(neg_cur_set))
                            selected_neg_cur_set = neg_cur_set[probs_to_res > random_probs]
                            num_left_in_res = len(neg_reservoir) - len(selected_neg_cur_set)
                            updated_neg_reservoir = pd.concat([neg_reservoir.sample(n=num_left_in_res), selected_neg_cur_set], ignore_index=False)
                            print('selected_neg_cur_set size', len(selected_neg_cur_set))
                            # print('num_in_neg_res', len(neg_reservoir))
                            # print('num_left_in_neg_res', num_left_in_res)
                            # print('num_in_updated_neg_res', len(updated_neg_reservoir))

                            print('compute prob and generate train set and update reservoir time elapsed: {}'.format(
                                time.strftime('%H:%M:%S', time.gmtime(time.time() - compute_prob_start_time))))

                            return sampled_train_set, updated_reservoir, updated_neg_reservoir


                        def compute_prob_and_gen_set_and_update_reservoir2():
                            """
                            this strategy modify slightly the method from paper "Streaming ranking based recommender systems"
                            train_set = current_set + samples of reservoir (need to set ratio of reservoir sample to current set)
                            """
                            compute_prob_start_time = time.time()

                            # compute prob
                            reservoir_batch_loader = BatchLoader(reservoir, train_config['base_bs'])

                            scores = []
                            for i in range(1, reservoir_batch_loader.num_batches + 1):
                                reservoir_batch = reservoir_batch_loader.get_batch(batch_id=i)
                                batch_scores, batch_losses = base_model.inference(sess, reservoir_batch)  # sess.run
                                scores.extend(batch_scores.tolist())

                            ordered_reservoir = reservoir
                            ordered_reservoir['score'] = scores
                            ordered_reservoir = ordered_reservoir.sort_values(['score'], ascending=False).reset_index(drop=True)  # edit
                            ordered_reservoir['rank'] = np.arange(len(ordered_reservoir))
                            total_num = len(reservoir)
                            ordered_reservoir['weight'] = ordered_reservoir['rank'].apply(lambda x: np.exp(x / total_num))
                            total_weights = ordered_reservoir['weight'].sum()
                            ordered_reservoir['prob'] = ordered_reservoir['weight'].apply(lambda x: x / total_weights)
                            ordered_reservoir = ordered_reservoir.drop(['score', 'rank', 'weight'], axis=1)

                            # generate train set
                            sampled_pos_reservoir = ordered_reservoir.sample(n=int(len(pos_cur_set) * train_config['res_cur_ratio']), replace=False, weights='prob')
                            sampled_pos_reservoir = sampled_pos_reservoir.drop(['prob'], axis=1)
                            sampled_neg_reservoir = neg_reservoir.sample(n=int(len(neg_cur_set) * train_config['res_cur_ratio']), replace=False)
                            sampled_reservoir = pd.concat([sampled_pos_reservoir, sampled_neg_reservoir], ignore_index=False)
                            sampled_train_set = pd.concat([sampled_reservoir, cur_set], ignore_index=False)
                            sampled_train_set = sampled_train_set.sort_values(['period']).reset_index(drop=True)
                            print('sampled_reservoir size', len(sampled_reservoir))
                            # print('sampled_train_set size', len(sampled_train_set))

                            # update reservoir
                            t = len(data_df[(data_df['period'] < train_config['cur_period']) & (data_df['label'] == 1)])
                            probs_to_res = len(reservoir) / (t + np.arange(len(pos_cur_set)) + 1)
                            random_probs = np.random.rand(len(pos_cur_set))
                            selected_pos_cur_set = pos_cur_set[probs_to_res > random_probs]
                            num_left_in_res = len(reservoir) - len(selected_pos_cur_set)
                            updated_reservoir = pd.concat([reservoir.sample(n=num_left_in_res), selected_pos_cur_set], ignore_index=False)
                            print('selected_pos_current_set size', len(selected_pos_cur_set))
                            # print('num_in_res', len(reservoir))
                            # print('num_left_in_res', num_left_in_res)
                            # print('num_in_updated_res', len(updated_reservoir))

                            # update neg reservoir
                            t = len(data_df[(data_df['period'] < train_config['cur_period']) & (data_df['label'] == 0)])
                            probs_to_res = len(neg_reservoir) / (t + np.arange(len(neg_cur_set)) + 1)
                            random_probs = np.random.rand(len(neg_cur_set))
                            selected_neg_cur_set = neg_cur_set[probs_to_res > random_probs]
                            num_left_in_res = len(neg_reservoir) - len(selected_neg_cur_set)
                            updated_neg_reservoir = pd.concat([neg_reservoir.sample(n=num_left_in_res), selected_neg_cur_set], ignore_index=False)
                            print('selected_neg_cur_set size', len(selected_neg_cur_set))
                            # print('num_in_neg_res', len(neg_reservoir))
                            # print('num_left_in_neg_res', num_left_in_res)
                            # print('num_in_updated_neg_res', len(updated_neg_reservoir))

                            print('compute prob and generate train set and update reservoir time elapsed: {}'.format(
                                time.strftime('%H:%M:%S', time.gmtime(time.time() - compute_prob_start_time))))

                            return sampled_train_set, updated_reservoir, updated_neg_reservoir

                        if train_config['strategy'] == 1:
                            cur_set, reservoir, neg_reservoir = compute_prob_and_gen_set_and_update_reservoir()
                        else:  # train_config['strategy'] == 2
                            cur_set, reservoir, neg_reservoir = compute_prob_and_gen_set_and_update_reservoir2()

                        # create an engine instance with base_model
                        engine = Engine(sess, base_model)

                        train_start_time = time.time()

                        max_auc = 0
                        best_logloss = 0

                        for epoch_id in range(1, train_config['base_num_epochs'] + 1):

                            print('Training Base Model Epoch {} Start!'.format(epoch_id))

                            base_loss_cur_avg = engine.base_train_an_epoch(epoch_id, cur_set, train_config)
                            print('Epoch {} Done! time elapsed: {}, base_loss_cur_avg {:.4f}'.format(
                                epoch_id,
                                time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start_time)),
                                base_loss_cur_avg))

                            cur_auc, cur_logloss = engine.test(cur_set, train_config)
                            next_auc, next_logloss = engine.test(next_set, train_config)
                            print('cur_auc {:.4f}, cur_logloss {:.4f}, next_auc {:.4f}, next_logloss {:.4f}'.format(
                                cur_auc,
                                cur_logloss,
                                next_auc,
                                next_logloss))
                            
                            print('time elapsed {}'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start_time))))
                            print('')

                            # save checkpoint
                            checkpoint_alias = 'Epoch{}_TestAUC{:.4f}_TestLOGLOSS{:.4f}.ckpt'.format(
                                epoch_id,
                                next_auc,
                                next_logloss)
                            checkpoint_path = os.path.join(ckpts_dir, checkpoint_alias)
                            saver.save(sess, checkpoint_path)

                            if next_auc > max_auc:
                                max_auc = next_auc
                                best_logloss = next_logloss

                        if i >= train_config['test_start_period']:
                            test_aucs.append(max_auc)
                            test_loglosses.append(best_logloss)

                    if i >= train_config['test_start_period']:
                        average_auc = sum(test_aucs) / len(test_aucs)
                        average_logloss = sum(test_loglosses) / len(test_loglosses)
                        print('test aucs', test_aucs)
                        print('average auc', average_auc)
                        print('')
                        print('test loglosses', test_loglosses)
                        print('average logloss', average_logloss)

                        # write metrics to text file
                        with open(os.path.join(ckpts_dir, 'test_metrics.txt'), mode='w') as f:
                            f.write('test_aucs: ' + str(test_aucs) + '\n')
                            f.write('average_auc: ' + str(average_auc) + '\n')
                            f.write('test_loglosses: ' + str(test_loglosses) + '\n')
                            f.write('average_logloss: ' + str(average_logloss) + '\n')
```

<!-- #region id="DAbNpqZ4NPq_" -->
### IncCTR
<!-- #endregion -->

```python id="LctucMACNPnE" executionInfo={"status": "ok", "timestamp": 1636623321604, "user_tz": -330, "elapsed": 801, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def incctr_sobazaar():
    # load data to df
    start_time = time.time()

    load_path = osp.join(args.path_silver,'soba_4mth_2014_1neg_30seq_1.csv')
    data_df = pd.read_csv(load_path)

    data_df['itemSeq'] = data_df['itemSeq'].fillna('')  # empty seq are NaN
    data_df['itemSeq'] = data_df['itemSeq'].apply(lambda x: [int(item) for item in x.split('#') if item != ''])

    logger.info('Done loading data! time elapsed: {}'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))))

    num_users = data_df['userId'].max() + 1
    num_items = data_df['itemId'].max() + 1

    train_config = {'method': 'IncCTR_by_period',
                    'dir_name': 'IncCTR_train11-23_test24-30_1epoch',  # edit train test period, number of epochs
                    'pretrain_model': 'pretrain_train1-10_test11_10epoch_0.001',
                    'start_date': 20140901,  # overall train start date
                    'end_date': 20141231,  # overall train end date
                    'num_periods': 31,  # number of periods divided into
                    'train_start_period': 11,
                    'test_start_period': 24,
                    'cur_period': None,  # current incremental period
                    'next_period': None,  # next incremental period
                    'cur_set_size': None,  # current incremental dataset size
                    'next_set_size': None,  # next incremental dataset size
                    'period_alias': None,  # individual period directory alias to save ckpts
                    'restored_ckpt_mode': 'best auc',  # mode to search the checkpoint to restore, 'best auc', 'best gauc', 'last'
                    'restored_ckpt': None,  # configure in the for loop

                    'lambda': None,  # weight assigned to knowledge distillation loss

                    'base_optimizer': 'adam',  # base model optimizer: adam, rmsprop, sgd
                    'base_lr': None,  # base model learning rate
                    'base_bs': 256,  # base model batch size
                    'base_num_epochs': 1,  # base model number of epochs
                    'shuffle': True,  # whether to shuffle the dataset for each epoch
                    }

    EmbMLPnocate_hyperparams = {'num_users': num_users,
                                'num_items': num_items,
                                'user_embed_dim': 8,
                                'item_embed_dim': 8,
                                'layers': [24, 12, 6, 1]  # input dim is user_embed_dim + item_embed_dim x 2
                                }

    # sort train data into periods based on num_periods
    data_df = data_df[(data_df['date'] >= train_config['start_date']) & (data_df['date'] <= train_config['end_date'])]
    data_df = data_df.sort_values(['timestamp']).reset_index(drop=True)
    records_per_period = int(len(data_df) / train_config['num_periods'])
    data_df['index'] = data_df.index
    data_df['period'] = data_df['index'].apply(lambda x: int(x / records_per_period) + 1)
    data_df = data_df[data_df.period != train_config['num_periods'] + 1]  # delete last extra period
    period_df = data_df.groupby('period')['date'].agg(['count', 'min', 'max'])
    data_df = data_df.drop(['index', 'date', 'timestamp'], axis=1)

    orig_dir_name = train_config['dir_name']

    for lam in [0.1]:

        for base_lr in [1e-3]:

            print('')
            print('lambda', lam, 'base_lr', base_lr)

            train_config['lambda'] = lam
            train_config['base_lr'] = base_lr

            train_config['dir_name'] = orig_dir_name + '_' + str(lam) + '_' + str(base_lr)
            print('dir_name: ', train_config['dir_name'])

            test_aucs = []
            test_loglosses = []

            for i in range(train_config['train_start_period'], train_config['num_periods']):

                # configure cur_period, next_period
                train_config['cur_period'] = i
                train_config['next_period'] = i + 1
                print('')
                print('current period: {}, next period: {}'.format(
                    train_config['cur_period'],
                    train_config['next_period']))
                print('')

                # create current and next set
                cur_set = data_df[data_df['period'] == train_config['cur_period']]
                next_set = data_df[data_df['period'] == train_config['next_period']]
                train_config['cur_set_size'] = len(cur_set)
                train_config['next_set_size'] = len(next_set)
                print('current set size', len(cur_set), 'next set size', len(next_set))

                train_config['period_alias'] = 'period' + str(i)

                # checkpoints directory
                ckpts_dir = os.path.join('ckpts', train_config['dir_name'], train_config['period_alias'])
                if not os.path.exists(ckpts_dir):
                    os.makedirs(ckpts_dir)

                if i == train_config['train_start_period']:
                    search_alias = os.path.join('ckpts', train_config['pretrain_model'], 'Epoch*')
                    train_config['restored_ckpt'] = search_ckpt(search_alias, mode=train_config['restored_ckpt_mode'])
                else:
                    prev_period_alias = 'period' + str(i - 1)
                    search_alias = os.path.join('ckpts', train_config['dir_name'], prev_period_alias, 'Epoch*')
                    train_config['restored_ckpt'] = search_ckpt(search_alias, mode=train_config['restored_ckpt_mode'])
                print('restored checkpoint: {}'.format(train_config['restored_ckpt']))

                # write train_config to text file
                with open(os.path.join(ckpts_dir, 'config.txt'), mode='w') as f:
                    f.write('train_config: ' + str(train_config) + '\n')
                    f.write('\n')
                    f.write('EmbMLPnocate_hyperparams: ' + str(EmbMLPnocate_hyperparams) + '\n')

                # build base model computation graph
                tf.reset_default_graph()
                base_model = EmbMLPnocate(EmbMLPnocate_hyperparams, train_config=train_config)

                # create session
                with tf.Session() as sess:
                    saver = tf.train.Saver()
                    saver.restore(sess, train_config['restored_ckpt'])

                    def infer_prev_base():
                        infer_start_time = time.time()
                        infer_batch_loader = BatchLoader(cur_set, train_config['base_bs'])  # load batch test set
                        scores = []
                        for i in range(1, infer_batch_loader.num_batches + 1):
                            infer_batch = infer_batch_loader.get_batch(batch_id=i)
                            batch_scores, batch_losses = base_model.inference(sess, infer_batch)  # sees.run
                            scores.extend(batch_scores.tolist())
                        print('Inference Done! time elapsed: {}'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - infer_start_time))))
                        return scores

                    cur_set['label_soft'] = infer_prev_base()
                    
                    # create an engine instance with base_model
                    engine = Engine(sess, base_model)

                    train_start_time = time.time()

                    max_auc = 0
                    best_logloss = 0

                    for epoch_id in range(1, train_config['base_num_epochs'] + 1):

                        print('Training Base Model Epoch {} Start!'.format(epoch_id))

                        base_loss_cur_avg = engine.base_train_an_epoch(epoch_id, cur_set, train_config)
                        print('Epoch {} Done! time elapsed: {}, base_loss_cur_avg {:.4f}'.format(
                            epoch_id,
                            time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start_time)),
                            base_loss_cur_avg))

                        cur_auc, cur_logloss = engine.test(cur_set, train_config)
                        next_auc, next_logloss = engine.test(next_set, train_config)
                        print('cur_auc {:.4f}, cur_logloss {:.4f}, next_auc {:.4f}, next_logloss {:.4f}'.format(
                            cur_auc,
                            cur_logloss,
                            next_auc,
                            next_logloss))
                        print('time elapsed {}'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start_time))))

                        print('')

                        # save checkpoint
                        checkpoint_alias = 'Epoch{}_TestAUC{:.4f}_TestLOGLOSS{:.4f}.ckpt'.format(
                            epoch_id,
                            next_auc,
                            next_logloss)
                        checkpoint_path = os.path.join(ckpts_dir, checkpoint_alias)
                        saver.save(sess, checkpoint_path)

                        if next_auc > max_auc:
                            max_auc = next_auc
                            best_logloss = next_logloss

                    if i >= train_config['test_start_period']:
                        test_aucs.append(max_auc)
                        test_loglosses.append(best_logloss)

                if i >= train_config['test_start_period']:
                    average_auc = sum(test_aucs) / len(test_aucs)
                    average_logloss = sum(test_loglosses) / len(test_loglosses)
                    print('test aucs', test_aucs)
                    print('average auc', average_auc)
                    print('')
                    print('test loglosses', test_loglosses)
                    print('average logloss', average_logloss)

                    # write metrics to text file
                    with open(os.path.join(ckpts_dir, 'test_metrics.txt'), mode='w') as f:
                        f.write('test_aucs: ' + str(test_aucs) + '\n')
                        f.write('average_auc: ' + str(average_auc) + '\n')
                        f.write('test_loglosses: ' + str(test_loglosses) + '\n')
                        f.write('average_logloss: ' + str(average_logloss) + '\n')
```

<!-- #region id="_vlw927jV7Bw" -->
### SML
<!-- #endregion -->

```python id="aV7fwSblWNtc" executionInfo={"status": "ok", "timestamp": 1636623752513, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class SMLEngine(object):
    """
    Training epoch and test
    """

    def __init__(self, sess, model):

        self.sess = sess
        self.model = model

    def base_train_an_epoch(self, epoch_id, cur_set, train_config):

        train_start_time = time.time()

        if train_config['shuffle']:
            cur_set = cur_set.sample(frac=1)

        cur_batch_loader = BatchLoader(cur_set, train_config['base_bs'])

        base_loss_cur_sum = 0

        for i in range(1, cur_batch_loader.num_batches + 1):

            cur_batch = cur_batch_loader.get_batch(batch_id=i)

            base_loss_cur = self.model.train_base(self.sess, cur_batch)  # sess.run

            if (i - 1) % 100 == 0:
                print('[Epoch {} Batch {}] base_loss_cur {:.4f}, time elapsed {}'.format(epoch_id,
                                                                                         i,
                                                                                         base_loss_cur,
                                                                                         time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start_time))))

            base_loss_cur_sum += base_loss_cur

        # epoch done, compute average loss
        base_loss_cur_avg = base_loss_cur_sum / cur_batch_loader.num_batches

        return base_loss_cur_avg

    def transfer_train_an_epoch(self, epoch_id, next_set, train_config):

        train_start_time = time.time()

        if train_config['shuffle']:
            next_set = next_set.sample(frac=1)

        next_batch_loader = BatchLoader(next_set, train_config['transfer_bs'])

        transfer_loss_next_sum = 0

        for i in range(1, next_batch_loader.num_batches + 1):

            next_batch = next_batch_loader.get_batch(batch_id=i)

            transfer_loss_next = self.model.train_transfer(self.sess, next_batch)  # sess.run

            if (i - 1) % 100 == 0:
                print('[Epoch {} Batch {}] transfer_loss_next {:.4f}, time elapsed {}'.format(epoch_id,
                                                                                              i,
                                                                                              transfer_loss_next,
                                                                                              time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start_time))))

                # test the performance of transferred model (can comment out if not needed)
                next_auc, next_logloss = self.test(next_set, train_config)
                print('next_auc {:.4f}, next_logloss {:.4f}'.format(
                    next_auc,
                    next_logloss))
                print('time elapsed {}'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start_time))))

                print('')

            transfer_loss_next_sum += transfer_loss_next

        # epoch done, compute average loss
        transfer_loss_next_avg = transfer_loss_next_sum / next_batch_loader.num_batches

        return transfer_loss_next_avg

    def test(self, test_set, train_config):

        test_batch_loader = BatchLoader(test_set, train_config['base_bs'])

        scores, losses, labels = [], [], []
        for i in range(1, test_batch_loader.num_batches + 1):
            test_batch = test_batch_loader.get_batch(batch_id=i)
            batch_scores, batch_losses = self.model.inference(self.sess, test_batch)  # sees.run
            scores.extend(batch_scores.tolist())
            losses.extend(batch_losses.tolist())
            labels.extend(test_batch[4])

        test_auc = cal_roc_auc(scores, labels)
        test_logloss = sum(losses) / len(losses)

        return test_auc, test_logloss
```

```python id="HZcrkhPbWUH-" executionInfo={"status": "ok", "timestamp": 1636623755395, "user_tz": -330, "elapsed": 1831, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def gelu(input_tensor):
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf


def transfer_emb(name, emb_prev, emb_upd, n1=10, n2=5, l1=20):

    with tf.variable_scope(name):
        embed_dim = emb_upd.get_shape().as_list()[-1]  # H
        embeds_norm = tf.sqrt(tf.reduce_sum(emb_prev * emb_prev, axis=-1))  # [num]
        embeds_dot = tf.div(emb_prev * emb_upd, tf.expand_dims(embeds_norm, -1) + tf.constant(1e-15))  # [num, H]
        stack_embeds = tf.stack([emb_prev, emb_upd, embeds_dot], axis=1)  # [num, 3, H]

        input1 = tf.expand_dims(stack_embeds, -1)  # [num, 3, H, 1]
        filter1 = tf.get_variable(name="cnn_filter1", shape=[3, 1, 1, n1])  # [3, 1, 1, n1]
        output1 = tf.nn.conv2d(input1, filter1, strides=[1, 1, 1, 1], padding='VALID')  # [num, 1, H, n1]
        output1 = gelu(output1)  # [num, 1, H, n1]

        input2 = tf.transpose(output1, perm=[0, 3, 2, 1])  # [num, n1, H, 1]
        filter2 = tf.get_variable(name="cnn_filter2", shape=[n1, 1, 1, n2])  # [n1, 1, 1, n2]
        output2 = tf.nn.conv2d(input2, filter2, strides=[1, 1, 1, 1], padding='VALID')  # [num, 1, H, n2]
        output2 = gelu(output2)  # [num, 1, H, n2]

        cnn_output = tf.transpose(output2, perm=[0, 3, 2, 1])  # [num, n2, H, 1]
        cnn_output = tf.reshape(cnn_output, shape=[-1, n2 * embed_dim])  # [num, n2 x H]

        with tf.variable_scope('fcn1'):
            fcn1_kernel = tf.get_variable(name='kernel', shape=[n2 * embed_dim, l1])  # [n2 x H, l1]
            fcn1_bias = tf.get_variable(name='bias', shape=[l1])  # [l1]
        with tf.variable_scope('fcn2'):
            fcn2_kernel = tf.get_variable(name='kernel', shape=[l1, embed_dim])  # [l1, H]
            fcn2_bias = tf.get_variable(name='bias', shape=[embed_dim])  # [H]

        fcn1 = gelu(tf.matmul(cnn_output, fcn1_kernel) + fcn1_bias)  # [num, l1]
        fcn2 = tf.matmul(fcn1, fcn2_kernel) + fcn2_bias  # [num, H]

    return fcn2


def transfer_mlp(name, param_prev, param_upd, param_shape, n1=5, n2=3, l1=40):

    with tf.variable_scope(name):
        param_prev = tf.reshape(param_prev, [-1])  # [dim]
        param_upd = tf.reshape(param_upd, [-1])  # [dim]
        param_dim = param_upd.get_shape().as_list()[-1]  # max_dim: 24 x 12 = 288
        param_norm = tf.sqrt(tf.reduce_sum(param_prev * param_prev))  # scalar
        param_dot = tf.div(param_prev * param_upd, param_norm + tf.constant(1e-15))  # [dim] / [] = [dim]
        stack_param = tf.stack([param_prev, param_upd, param_dot], axis=0)  # [3, dim]

        input1 = tf.expand_dims(tf.expand_dims(stack_param, -1), 0)  # [1, 3, dim, 1]
        filter1 = tf.get_variable(name="cnn_filter1", shape=[3, 1, 1, n1])  # [3, 1, 1, n1]
        output1 = tf.nn.conv2d(input1, filter1, strides=[1, 1, 1, 1], padding='VALID')  # [1, 1, dim, n1]
        output1 = gelu(output1)  # [1, 1, dim, n1]

        input2 = tf.transpose(output1, perm=[0, 3, 2, 1])  # [1, n1, dim, 1]
        filter2 = tf.get_variable(name="cnn_filter2", shape=[n1, 1, 1, n2])  # [n1, 1, 1, n2]
        output2 = tf.nn.conv2d(input2, filter2, strides=[1, 1, 1, 1], padding='VALID')  # [1, 1, dim, n2]
        output2 = gelu(output2)  # [1, 1, dim, n2]

        cnn_output = tf.transpose(output2, perm=[0, 3, 2, 1])  # [1, n2, dim, 1]
        cnn_output = tf.reshape(cnn_output, shape=[1, -1])  # [1, n2 x dim]

        with tf.variable_scope('fcn1'):
            fcn1_kernel = tf.get_variable(name='kernel', shape=[n2 * param_dim, l1])  # [n2 x dim, l1]
            fcn1_bias = tf.get_variable(name='bias', shape=[l1])  # [l1]
        with tf.variable_scope('fcn2'):
            fcn2_kernel = tf.get_variable(name='kernel', shape=[l1, param_dim])  # [l1, dim]
            fcn2_bias = tf.get_variable(name='bias', shape=[param_dim])  # [dim]

        fcn1 = gelu(tf.matmul(cnn_output, fcn1_kernel) + fcn1_bias)  # [1, l1]
        fcn2 = tf.matmul(fcn1, fcn2_kernel) + fcn2_bias  # [1, dim]
        output = tf.reshape(fcn2, shape=param_shape)  # [dim1, dim2, ...]

    return output


class SML(object):

    def __init__(self, hyperparams, prev_emb_dict, prev_mlp_dict, train_config=None):

        self.train_config = train_config

        # create placeholder
        self.u = tf.placeholder(tf.int32, [None])  # [B]
        self.i = tf.placeholder(tf.int32, [None])  # [B]
        self.hist_i = tf.placeholder(tf.int32, [None, None])  # [B, T]
        self.hist_len = tf.placeholder(tf.int32, [None])  # [B]
        self.y = tf.placeholder(tf.float32, [None])  # [B]
        self.base_lr = tf.placeholder(tf.float32, [], name='base_lr')  # scalar
        self.transfer_lr = tf.placeholder(tf.float32, [], name='transfer_lr')  # scalar

        if train_config['transfer_emb']:
            # -- create emb_w_upd begin -------
            user_emb_w_upd = tf.get_variable("user_emb_w", [hyperparams['num_users'], hyperparams['user_embed_dim']])
            item_emb_w_upd = tf.get_variable("item_emb_w", [hyperparams['num_items'], hyperparams['item_embed_dim']])
            # -- create emb_w_upd end -------

            # -- create emb_w_prev begin ----
            user_emb_w_prev = tf.convert_to_tensor(prev_emb_dict['user_emb_w'], tf.float32)
            item_emb_w_prev = tf.convert_to_tensor(prev_emb_dict['item_emb_w'], tf.float32)
            # -- create emb_w_prev end ----

            # -- transfer emb_w begin ----
            with tf.variable_scope('transfer_emb'):
                user_emb_w = transfer_emb(name='user_emb_w',
                                          emb_prev=user_emb_w_prev,
                                          emb_upd=user_emb_w_upd,
                                          n1=train_config['emb_n1'],
                                          n2=train_config['emb_n2'],
                                          l1=train_config['emb_l1'])
                item_emb_w = transfer_emb(name='item_emb_w',
                                          emb_prev=item_emb_w_prev,
                                          emb_upd=item_emb_w_upd,
                                          n1=train_config['emb_n1'],
                                          n2=train_config['emb_n2'],
                                          l1=train_config['emb_l1'])
            # -- transfer emb end ----

            # -- update op begin -------
            self.user_emb_w_upd_op = user_emb_w_upd.assign(user_emb_w)
            self.item_emb_w_upd_op = item_emb_w_upd.assign(item_emb_w)
            # -- update op end -------

        else:
            # -- create emb_w begin -------
            user_emb_w = tf.get_variable("user_emb_w", [hyperparams['num_users'], hyperparams['user_embed_dim']])
            item_emb_w = tf.get_variable("item_emb_w", [hyperparams['num_items'], hyperparams['item_embed_dim']])
            # -- create emb_w end -------

        if train_config['transfer_mlp']:
            # -- create mlp_upd begin ---
            concat_dim = hyperparams['user_embed_dim'] + hyperparams['item_embed_dim'] * 2
            with tf.variable_scope('fcn1'):
                fcn1_kernel_upd = tf.get_variable('kernel', [concat_dim, hyperparams['layers'][1]])
                fcn1_bias_upd = tf.get_variable('bias', [hyperparams['layers'][1]])
            with tf.variable_scope('fcn2'):
                fcn2_kernel_upd = tf.get_variable('kernel', [hyperparams['layers'][1], hyperparams['layers'][2]])
                fcn2_bias_upd = tf.get_variable('bias', [hyperparams['layers'][2]])
            with tf.variable_scope('fcn3'):
                fcn3_kernel_upd = tf.get_variable('kernel', [hyperparams['layers'][2], 1])
                fcn3_bias_upd = tf.get_variable('bias', [1])
            # -- create mlp_upd end ---

            # -- create mlp_prev begin ----
            fcn1_kernel_prev = tf.convert_to_tensor(prev_mlp_dict['fcn1/kernel'], tf.float32)
            fcn1_bias_prev = tf.convert_to_tensor(prev_mlp_dict['fcn1/bias'], tf.float32)
            fcn2_kernel_prev = tf.convert_to_tensor(prev_mlp_dict['fcn2/kernel'], tf.float32)
            fcn2_bias_prev = tf.convert_to_tensor(prev_mlp_dict['fcn2/bias'], tf.float32)
            fcn3_kernel_prev = tf.convert_to_tensor(prev_mlp_dict['fcn3/kernel'], tf.float32)
            fcn3_bias_prev = tf.convert_to_tensor(prev_mlp_dict['fcn3/bias'], tf.float32)
            # -- create mlp_prev end ----

            # -- transfer mlp begin ----
            with tf.variable_scope('transfer_mlp'):
                with tf.variable_scope('fcn1'):
                    fcn1_kernel = transfer_mlp(name='kernel',
                                               param_prev=fcn1_kernel_prev,
                                               param_upd=fcn1_kernel_upd,
                                               param_shape=[concat_dim, hyperparams['layers'][1]],
                                               n1=train_config['mlp_n1'],
                                               n2=train_config['mlp_n2'],
                                               l1=train_config['mlp_l1_dict']['fcn1/kernel'])
                    fcn1_bias = transfer_mlp(name='bias',
                                             param_prev=fcn1_bias_prev,
                                             param_upd=fcn1_bias_upd,
                                             param_shape=[hyperparams['layers'][1]],
                                             n1=train_config['mlp_n1'],
                                             n2=train_config['mlp_n2'],
                                             l1=train_config['mlp_l1_dict']['fcn1/bias'])
                with tf.variable_scope('fcn2'):
                    fcn2_kernel = transfer_mlp(name='kernel',
                                               param_prev=fcn2_kernel_prev,
                                               param_upd=fcn2_kernel_upd,
                                               param_shape=[hyperparams['layers'][1], hyperparams['layers'][2]],
                                               n1=train_config['mlp_n1'],
                                               n2=train_config['mlp_n2'],
                                               l1=train_config['mlp_l1_dict']['fcn2/kernel'])
                    fcn2_bias = transfer_mlp(name='bias',
                                             param_prev=fcn2_bias_prev,
                                             param_upd=fcn2_bias_upd,
                                             param_shape=[hyperparams['layers'][2]],
                                             n1=train_config['mlp_n1'],
                                             n2=train_config['mlp_n2'],
                                             l1=train_config['mlp_l1_dict']['fcn2/bias'])
                with tf.variable_scope('fcn3'):
                    fcn3_kernel = transfer_mlp(name='kernel',
                                               param_prev=fcn3_kernel_prev,
                                               param_upd=fcn3_kernel_upd,
                                               param_shape=[hyperparams['layers'][2], 1],
                                               n1=train_config['mlp_n1'],
                                               n2=train_config['mlp_n2'],
                                               l1=train_config['mlp_l1_dict']['fcn3/kernel'])
                    fcn3_bias = transfer_mlp(name='bias',
                                             param_prev=fcn3_bias_prev,
                                             param_upd=fcn3_bias_upd,
                                             param_shape=[1],
                                             n1=train_config['mlp_n1'],
                                             n2=train_config['mlp_n2'],
                                             l1=train_config['mlp_l1_dict']['fcn3/bias'])
            # -- transfer mlp end ----

            # -- update op begin -------
            self.fcn1_kernel_upd_op = fcn1_kernel_upd.assign(fcn1_kernel)
            self.fcn1_bias_upd_op = fcn1_bias_upd.assign(fcn1_bias)
            self.fcn2_kernel_upd_op = fcn2_kernel_upd.assign(fcn2_kernel)
            self.fcn2_bias_upd_op = fcn2_bias_upd.assign(fcn2_bias)
            self.fcn3_kernel_upd_op = fcn3_kernel_upd.assign(fcn3_kernel)
            self.fcn3_bias_upd_op = fcn3_bias_upd.assign(fcn3_bias)
            # -- update op end -------

        else:
            # -- create mlp begin ---
            concat_dim = hyperparams['user_embed_dim'] + hyperparams['item_embed_dim'] * 2
            with tf.variable_scope('fcn1'):
                fcn1_kernel = tf.get_variable('kernel', [concat_dim, hyperparams['layers'][1]])
                fcn1_bias = tf.get_variable('bias', [hyperparams['layers'][1]])
            with tf.variable_scope('fcn2'):
                fcn2_kernel = tf.get_variable('kernel', [hyperparams['layers'][1], hyperparams['layers'][2]])
                fcn2_bias = tf.get_variable('bias', [hyperparams['layers'][2]])
            with tf.variable_scope('fcn3'):
                fcn3_kernel = tf.get_variable('kernel', [hyperparams['layers'][2], 1])
                fcn3_bias = tf.get_variable('bias', [1])
            # -- create mlp end ---

        # -- emb begin -------
        u_emb = tf.nn.embedding_lookup(user_emb_w, self.u)  # [B, H]
        i_emb = tf.nn.embedding_lookup(item_emb_w, self.i)  # [B, H]
        h_emb = tf.nn.embedding_lookup(item_emb_w, self.hist_i)  # [B, T, H]
        u_hist = average_pooling(h_emb, self.hist_len)  # [B, H]
        # -- emb end -------

        # -- mlp begin -------
        fcn = tf.concat([u_emb, u_hist, i_emb], axis=-1)  # [B, H x 3]
        fcn_layer_1 = tf.nn.relu(tf.matmul(fcn, fcn1_kernel) + fcn1_bias)  # [B, l1]
        fcn_layer_2 = tf.nn.relu(tf.matmul(fcn_layer_1, fcn2_kernel) + fcn2_bias)  # [B, l2]
        fcn_layer_3 = tf.matmul(fcn_layer_2, fcn3_kernel) + fcn3_bias  # [B, 1]
        # -- mlp end -------

        logits = tf.reshape(fcn_layer_3, [-1])  # [B]
        self.scores = tf.sigmoid(logits)  # [B]

        # return same dimension as input tensors, let x = logits, z = labels, z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        self.losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.y)
        self.loss = tf.reduce_mean(self.losses)

        # base_optimizer
        if train_config['base_optimizer'] == 'adam':
            base_optimizer = tf.train.AdamOptimizer(learning_rate=self.base_lr)
        elif train_config['base_optimizer'] == 'rmsprop':
            base_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.base_lr)
        else:
            base_optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.base_lr)

        # transfer_optimizer
        if train_config['transfer_optimizer'] == 'adam':
            transfer_optimizer = tf.train.AdamOptimizer(learning_rate=self.transfer_lr)
        elif train_config['transfer_optimizer'] == 'rmsprop':
            transfer_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.transfer_lr)
        else:
            transfer_optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.transfer_lr)

        trainable_params = tf.trainable_variables()
        base_params = [v for v in trainable_params if 'transfer' not in v.name]
        transfer_params = [v for v in trainable_params if 'transfer' in v.name]

        # update base model and transfer module
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            base_grads = tf.gradients(self.loss, base_params)  # return a list of gradients (A list of `sum(dy/dx)` for each x in `xs`)
            base_grads_tuples = zip(base_grads, base_params)
            self.train_base_op = base_optimizer.apply_gradients(base_grads_tuples)

            transfer_grads = tf.gradients(self.loss, transfer_params)
            transfer_grads_tuples = zip(transfer_grads, transfer_params)
            with tf.variable_scope('transfer_opt'):
                self.train_transfer_op = transfer_optimizer.apply_gradients(transfer_grads_tuples)

    def train_base(self, sess, batch):
        loss, _ = sess.run([self.loss, self.train_base_op], feed_dict={
            self.u: batch[0],
            self.i: batch[1],
            self.hist_i: batch[2],
            self.hist_len: batch[3],
            self.y: batch[4],
            self.base_lr: self.train_config['base_lr'],
        })
        return loss

    def train_transfer(self, sess, batch):
        loss, _, = sess.run([self.loss, self.train_transfer_op], feed_dict={
            self.u: batch[0],
            self.i: batch[1],
            self.hist_i: batch[2],
            self.hist_len: batch[3],
            self.y: batch[4],
            self.transfer_lr: self.train_config['transfer_lr'],
        })
        return loss

    def update(self, sess):
        if self.train_config['transfer_emb']:
            sess.run([self.user_emb_w_upd_op,
                      self.item_emb_w_upd_op])
        if self.train_config['transfer_mlp']:
            sess.run([self.fcn1_kernel_upd_op,
                      self.fcn1_bias_upd_op,
                      self.fcn2_kernel_upd_op,
                      self.fcn2_bias_upd_op,
                      self.fcn3_kernel_upd_op,
                      self.fcn3_bias_upd_op])

    def inference(self, sess, batch):
        scores, losses = sess.run([self.scores, self.losses], feed_dict={
            self.u: batch[0],
            self.i: batch[1],
            self.hist_i: batch[2],
            self.hist_len: batch[3],
            self.y: batch[4],
        })
        return scores, losses
```

```python id="pd5O3RncV6_S" executionInfo={"status": "ok", "timestamp": 1636623985215, "user_tz": -330, "elapsed": 1390, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def sml_sobazaar():
    # load data to df
    start_time = time.time()

    load_path = osp.join(args.path_silver,'soba_4mth_2014_1neg_30seq_1.csv')
    data_df = pd.read_csv(load_path)

    data_df['itemSeq'] = data_df['itemSeq'].fillna('')  # empty seq are NaN
    data_df['itemSeq'] = data_df['itemSeq'].apply(lambda x: [int(item) for item in x.split('#') if item != ''])

    logger.info('Done loading data! time elapsed: {}'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))))

    num_users = data_df['userId'].max() + 1
    num_items = data_df['itemId'].max() + 1

    train_config = {'method': 'SML_by_period',
                    'dir_name': 'SML_emb&mlp_train11-23_test24-30_1epoch_1epoch',  # edit parameter to transfer, train test period, transfer number of epochs, base number of epochs
                    'pretrain_model': 'pretrain_train1-10_test11_10epoch_0.001',  # pretrained base model
                    'start_date': 20140901,  # overall train start date
                    'end_date': 20141231,  # overall train end date
                    'num_periods': 31,  # number of periods divided into
                    'train_start_period': 11,
                    'test_start_period': 24,
                    'cur_period': None,  # current incremental period
                    'next_period': None,  # next incremental period
                    'cur_set_size': None,  # current incremental dataset size
                    'next_set_size': None,  # next incremental dataset size
                    'period_alias': None,  # individual period directory alias to save ckpts
                    'restored_ckpt_mode': 'best auc',  # mode to search the ckpt to restore: 'best auc', 'best logloss', 'last'
                    'restored_ckpt': None,  # restored sml model checkpoint

                    'transfer_emb': True,
                    'emb_n1': 10,
                    'emb_n2': 5,
                    'emb_l1': 20,
                    'transfer_mlp': True,
                    'mlp_n1': 5,
                    'mlp_n2': 3,
                    'mlp_l1_dict': {'fcn1/kernel': 40,
                                    'fcn1/bias': 20,
                                    'fcn2/kernel': 20,
                                    'fcn2/bias': 10,
                                    'fcn3/kernel': 10,
                                    'fcn3/bias': 1},

                    'transfer_optimizer': 'adam',  # transfer module optimizer: adam, rmsprop, sgd
                    'transfer_lr': None,  # transfer module learning rate
                    'transfer_bs': 256,  # transfer module batch size
                    'transfer_num_epochs': 1,  # transfer module number of epochs
                    'test_stop_train': False,  # whether to stop updating transfer module during test periods

                    'base_optimizer': 'adam',  # base model optimizer: adam, rmsprop, sgd
                    'base_lr': None,  # base model learning rate
                    'base_bs': 256,  # base model batch size
                    'base_num_epochs': 1,  # base model number of epochs
                    'shuffle': True,  # whether to shuffle the dataset for each epoch
                    }

    EmbMLPnocate_hyperparams = {'num_users': num_users,
                                'num_items': num_items,
                                'user_embed_dim': 8,
                                'item_embed_dim': 8,
                                'layers': [24, 12, 6, 1]  # input dim is user_embed_dim + item_embed_dim x 2
                                }

    # sort train data into periods based on num_periods
    data_df = data_df[(data_df['date'] >= train_config['start_date']) & (data_df['date'] <= train_config['end_date'])]
    data_df = data_df.sort_values(['timestamp']).reset_index(drop=True)
    records_per_period = int(len(data_df) / train_config['num_periods'])
    data_df['index'] = data_df.index
    data_df['period'] = data_df['index'].apply(lambda x: int(x / records_per_period) + 1)
    data_df = data_df[data_df.period != train_config['num_periods'] + 1]  # delete last extra period
    period_df = data_df.groupby('period')['date'].agg(['count', 'min', 'max'])
    data_df = data_df.drop(['index', 'date', 'timestamp'], axis=1)

    orig_dir_name = train_config['dir_name']

    for transfer_lr in [1e-3]:

        for base_lr in [1e-2]:

            print('')
            print('transfer_lr', transfer_lr, 'base_lr', base_lr)

            train_config['transfer_lr'] = transfer_lr
            train_config['base_lr'] = base_lr

            train_config['dir_name'] = orig_dir_name + '_' + str(transfer_lr) + '_' + str(base_lr)
            print('dir_name: ', train_config['dir_name'])

            test_aucs = []
            test_loglosses = []

            for i in range(train_config['train_start_period'], train_config['num_periods']):

                # configure cur_period, next_period
                train_config['cur_period'] = i
                train_config['next_period'] = i + 1
                print('')
                print('current period: {}, next period: {}'.format(
                    train_config['cur_period'],
                    train_config['next_period']))
                print('')

                # create current and next set
                cur_set = data_df[data_df['period'] == train_config['cur_period']]
                next_set = data_df[data_df['period'] == train_config['next_period']]
                train_config['cur_set_size'] = len(cur_set)
                train_config['next_set_size'] = len(next_set)
                print('current set size', len(cur_set), 'next set size', len(next_set))

                train_config['period_alias'] = 'period' + str(i)

                # checkpoints directory
                ckpts_dir = os.path.join('ckpts', train_config['dir_name'], train_config['period_alias'])
                if not os.path.exists(ckpts_dir):
                    os.makedirs(ckpts_dir)

                if i == train_config['train_start_period']:
                    search_alias = os.path.join('ckpts', train_config['pretrain_model'], 'Epoch*')
                    train_config['restored_ckpt'] = search_ckpt(search_alias, mode=train_config['restored_ckpt_mode'])
                else:
                    prev_period_alias = 'period' + str(i - 1)
                    search_alias = os.path.join('ckpts', train_config['dir_name'], prev_period_alias, 'Epoch*')
                    train_config['restored_ckpt'] = search_ckpt(search_alias, mode=train_config['restored_ckpt_mode'])
                print('restored checkpoint: {}'.format(train_config['restored_ckpt']))

                # write train_config to text file
                with open(os.path.join(ckpts_dir, 'config.txt'), mode='w') as f:
                    f.write('train_config: ' + str(train_config) + '\n')
                    f.write('\n')
                    f.write('EmbMLPnocate_hyperparams: ' + str(EmbMLPnocate_hyperparams) + '\n')

                def collect_params():
                    """
                    collect previous period model parameters
                    :return: prev_emb_dict, prev_mlp_dict
                    """
                    collect_params_start_time = time.time()
                    emb_ls = ['user_emb_w', 'item_emb_w']
                    mlp_ls = ['fcn1/kernel', 'fcn2/kernel', 'fcn3/kernel', 'fcn3/bias', 'fcn1/bias', 'fcn2/bias']
                    prev_emb_dict_ = {name: tf.train.load_checkpoint(train_config['restored_ckpt']).get_tensor(name)
                                    for name, _ in tf.train.list_variables(train_config['restored_ckpt']) if name in emb_ls}
                    prev_mlp_dict_ = {name: tf.train.load_checkpoint(train_config['restored_ckpt']).get_tensor(name)
                                    for name, _ in tf.train.list_variables(train_config['restored_ckpt']) if name in mlp_ls}
                    print('collect params time elapsed: {}'.format(
                        time.strftime('%H:%M:%S', time.gmtime(time.time() - collect_params_start_time))))
                    return prev_emb_dict_, prev_mlp_dict_

                # collect previous period model parameters
                prev_emb_dict, prev_mlp_dict = collect_params()

                # build sml model computation graph
                tf.reset_default_graph()
                sml_model = SML(EmbMLPnocate_hyperparams, prev_emb_dict, prev_mlp_dict, train_config=train_config)

                # create session
                with tf.Session() as sess:

                    def train_base():
                        # create an engine instance with sml_model
                        engine = SMLEngine(sess, sml_model)
                        train_start_time = time.time()
                        max_auc = 0
                        best_logloss = 0
                        for epoch_id in range(1, train_config['base_num_epochs'] + 1):
                            print('Training Base Model Epoch {} Start!'.format(epoch_id))
                            base_loss_cur_avg = engine.base_train_an_epoch(epoch_id, cur_set, train_config)
                            print('Epoch {} Done! time elapsed: {}, base_loss_cur_avg {:.4f}'.format(
                                epoch_id,
                                time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start_time)),
                                base_loss_cur_avg))
                            cur_auc, cur_logloss = engine.test(cur_set, train_config)
                            next_auc, next_logloss = engine.test(next_set, train_config)
                            print('cur_auc {:.4f}, cur_logloss {:.4f}, next_auc {:.4f}, next_logloss {:.4f}'.format(
                                cur_auc,
                                cur_logloss,
                                next_auc,
                                next_logloss))
                            print('time elapsed {}'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start_time))))
                            print('')
                            # save checkpoint
                            if i >= train_config['test_start_period'] and train_config['test_stop_train']:
                                checkpoint_alias = 'Epoch{}_TestAUC{:.4f}_TestLOGLOSS{:.4f}.ckpt'.format(
                                    epoch_id,
                                    next_auc,
                                    next_logloss)
                                checkpoint_path = os.path.join(ckpts_dir, checkpoint_alias)
                                saver.save(sess, checkpoint_path)
                            if next_auc > max_auc:
                                max_auc = next_auc
                                best_logloss = next_logloss
                        if i >= train_config['test_start_period']:
                            test_aucs.append(max_auc)
                            test_loglosses.append(best_logloss)

                    def train_transfer():
                        # create an engine instance with sml_model
                        engine = SMLEngine(sess, sml_model)
                        train_start_time = time.time()
                        for epoch_id in range(1, train_config['transfer_num_epochs'] + 1):
                            print('Training Transfer Module Epoch {} Start!'.format(epoch_id))
                            transfer_loss_next_avg = engine.transfer_train_an_epoch(epoch_id, next_set, train_config)
                            print('Epoch {} Done! time elapsed: {}, transfer_loss_next_avg {:.4f}'.format(
                                epoch_id,
                                time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start_time)),
                                transfer_loss_next_avg))
                            cur_auc, cur_logloss = engine.test(cur_set, train_config)
                            next_auc, next_logloss = engine.test(next_set, train_config)
                            print('cur_auc {:.4f}, cur_logloss {:.4f}, next_auc {:.4f}, next_logloss {:.4f}'.format(
                                cur_auc,
                                cur_logloss,
                                next_auc,
                                next_logloss))
                            print('time elapsed {}'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start_time))))
                            print('')
                            # update transferred params
                            sml_model.update(sess)
                            # save checkpoint
                            checkpoint_alias = 'Epoch{}_TestAUC{:.4f}_TestLOGLOSS{:.4f}.ckpt'.format(
                                epoch_id,
                                next_auc,
                                next_logloss)
                            checkpoint_path = os.path.join(ckpts_dir, checkpoint_alias)
                            saver.save(sess, checkpoint_path)

                    # restore sml model (transfer module and base model)
                    if i == train_config['train_start_period']:
                        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])  # initialize transfer module
                        restorer = tf.train.Saver(var_list=[v for v in tf.global_variables() if 'transfer' not in v.name])  # restore base model
                        restorer.restore(sess, train_config['restored_ckpt'])
                    else:
                        restorer = tf.train.Saver()  # restore transfer module and base model
                        restorer.restore(sess, train_config['restored_ckpt'])
                    saver = tf.train.Saver()

                    # test transfer module by training base model with it
                    train_base()

                    # train transfer module
                    if i < train_config['test_start_period'] or not train_config['test_stop_train']:
                        train_transfer()

                if i >= train_config['test_start_period']:
                    average_auc = sum(test_aucs) / len(test_aucs)
                    average_logloss = sum(test_loglosses) / len(test_loglosses)
                    print('test aucs', test_aucs)
                    print('average auc', average_auc)
                    print('')
                    print('test loglosses', test_loglosses)
                    print('average logloss', average_logloss)

                    # write metrics to text file
                    with open(os.path.join(ckpts_dir, 'test_metrics.txt'), mode='w') as f:
                        f.write('test_aucs: ' + str(test_aucs) + '\n')
                        f.write('average_auc: ' + str(average_auc) + '\n')
                        f.write('test_loglosses: ' + str(test_loglosses) + '\n')
                        f.write('average_logloss: ' + str(average_logloss) + '\n')
```

<!-- #region id="P0nc1kyjzX9T" -->
## Jobs
<!-- #endregion -->

```python id="2y8mdDjds6dr" colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["83c9f2d6bc6a42109250d55a39b4b3e7", "6af9c2d14cd344408ebeb165aecd5812", "14a3c2eab70348fd8336ae11dbbfcffd", "1f7c997368b34aedb5ecafc1ad6b1121", "712fb93cec584db29a18f17f48ce85dc", "6149942fe6ad4e0ea119545de5a8d242", "f4349bb599ab413c9c087714eb7ba11f", "7b90eb42b83641b4890847c92356a8f6", "8d3fb8b1ab5742ec983d41c822319729", "27e8cfe72e7a4a93b1f2e1f705a551a5", "74ce21ef75514a30a12822ebc5165352", "30c470ccdef34a2999ab74aad3503ba7", "7ef1f03aed684fdbbc6602ec174cd2cd", "45c6daf562174f79ad42975093c84210", "d3fc9033cc8f45c9a9bc114a6c590fd9", "53e9c4c383124944a89e833d7f0a56df", "e69ef6e531314efe9a9ee6dc58e5998b", "813103a929a84c78aa67e72419563cb7", "0fa28afc080a4e98a0f5ccd7d083e3b3", "60fe4254494e4bc589c971c5e287ec88", "bb7538a3fc0045c28b51edc7f7cbbc00", "d30f88908fe54f779659b41bd1e5ba6f"]} executionInfo={"status": "ok", "timestamp": 1636550822729, "user_tz": -330, "elapsed": 2141725, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="af2586b2-0874-4753-f5f5-7da803ca06aa"
# logger.info('JOB START: PREPROCESS_DATASET')
# preprocess_sobazaar()
# logger.info('JOB END: PREPROCESS_DATASET')
```

```python colab={"base_uri": "https://localhost:8080/"} id="0A3Bwm7DHyc4" executionInfo={"status": "ok", "timestamp": 1636620685985, "user_tz": -330, "elapsed": 11366, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="549d6d2a-20b2-482c-fb65-6379ccb8a12e"
logger.info('JOB START: CONVERT_PARQUET_TO_CSV')
parquet_to_csv('/content/soba_4mth_2014_1neg_30seq_1.parquet.snappy')
logger.info('JOB END: CONVERT_PARQUET_TO_CSV')
```

```python id="3ig3tPpB2Fx-" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1636620920481, "user_tz": -330, "elapsed": 231988, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3a50d93c-353a-489c-8a68-682a72717f36"
logger.info('JOB START: EMBEDMLP_PRETRAINING')
pretrain_sobazaar()
logger.info('JOB END: EMBEDMLP_PRETRAINING')
```

```python colab={"base_uri": "https://localhost:8080/"} id="Y3quOeuqFHwF" executionInfo={"status": "ok", "timestamp": 1636621464860, "user_tz": -330, "elapsed": 544424, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f4a6c221-0e6e-499c-e78a-066abf8da6bb"
logger.info('JOB START: BATCH_UPDATE')
bu_sobazaar()
logger.info('JOB END: BATCH_UPDATE')
```

```python colab={"base_uri": "https://localhost:8080/"} id="ZvFN3zuYJq-U" executionInfo={"status": "ok", "timestamp": 1636623218955, "user_tz": -330, "elapsed": 291436, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="72fa099b-f84c-42c0-bf93-c6ee4b83c8ad"
logger.info('JOB START: SPMF_MODEL_TRAINING')
spmf_sobazaar()
logger.info('JOB END: SPMF_MODEL_TRAINING')
```

```python colab={"base_uri": "https://localhost:8080/"} id="G7LFV1L2SHQ8" executionInfo={"status": "ok", "timestamp": 1636623488608, "user_tz": -330, "elapsed": 159072, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="916d9322-47f5-48a4-c541-dbe02f663b3c"
logger.info('JOB START: INCCTR_MODEL_TRAINING')
incctr_sobazaar()
logger.info('JOB END: INCCTR_MODEL_TRAINING')
```

```python colab={"base_uri": "https://localhost:8080/"} id="S5HsNVFXYOjE" executionInfo={"status": "ok", "timestamp": 1636624812859, "user_tz": -330, "elapsed": 823665, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ee691545-f06e-4e9f-e9d2-44b4a0413802"
logger.info('JOB START: SML_MODEL_TRAINING')
sml_sobazaar()
logger.info('JOB END: SML_MODEL_TRAINING')
```

```python id="JHHTsxBsYRfF" executionInfo={"status": "ok", "timestamp": 1636624882399, "user_tz": -330, "elapsed": 1614, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
!cp -r /content/ckpts /content/T967215
```
