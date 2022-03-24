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
# Incremental Update Retraining on Sobazaar dataset
<!-- #endregion -->

<!-- #region id="CxiWmRiFzT2X" -->
## Setup
<!-- #endregion -->

<!-- #region id="zVtJ4JTGH353" -->
### Git
<!-- #endregion -->

```python id="Z3qjPp055tXf" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1636552445257, "user_tz": -330, "elapsed": 1642, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c950f0d4-0b6b-40dc-e267-25e09b0bf937"
import os
project_name = "incremental-learning"; branch = "T644011"; account = "sparsh-ai"
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

```python id="xoKGydGDIwSX"
%cd /content
```

```python id="7lAKlgUD5tXi"
!cd /content/T644011 && git add .
!cd /content/T644011 && git commit -m 'commit'
```

```python id="IqqZ6Do-uswE"
!cd /content/T644011 && git pull --rebase origin "{branch}"
!cd /content/T644011 && git push origin "{branch}"
```

```python id="evKxFrICIpy_"
# !mv /content/ckpts .
# !mv /content/soba_4mth_2014_1neg_30seq_1.parquet.snappy .
```

<!-- #region id="BXJY8c9d4Xi5" -->
### Installations
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="DctyNOSdx-7h" executionInfo={"status": "ok", "timestamp": 1636547390860, "user_tz": -330, "elapsed": 5352, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="86a0cc18-1db8-443e-9f08-779d567b1414"
!pip install -q wget
```

<!-- #region id="BK-ZCkf00xZt" -->
### Dataset
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="5eJSFty70xW-" executionInfo={"status": "ok", "timestamp": 1636547571751, "user_tz": -330, "elapsed": 1335, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0bf7afa6-fa7d-4bec-989c-721d461fa242"
!wget -q --show-progress https://github.com/RecoHut-Datasets/sobazaar/raw/main/Data/Sobazaar-hashID.csv.gz
```

<!-- #region id="GB_yDppW3_Yt" -->
### Imports
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="3lFs8AyO1IWc" executionInfo={"status": "ok", "timestamp": 1636548677008, "user_tz": -330, "elapsed": 1010, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b2992811-db3f-4f5c-c9a9-abcbfbd6bb4c"
%tensorflow_version 1.x
```

```python id="vrEmNkAAsQlM"
import numpy as np
from tqdm.notebook import tqdm
import sys
import wget
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

```python id="MXBwnUCD3_RD"
class Args:
    path_bronze = '/content'
    path_silver = '/content'

args = Args()
```

```python id="K5cAMUaO2H8W"
random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(123)
```

<!-- #region id="Q40X4lHf4JHw" -->
### Logger
<!-- #endregion -->

```python id="cibwpV5L4JFb"
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

```python id="tH7lmOJbAOIf"
def save_pickle(data, title):
 with bz2.BZ2File(title + '.pbz2', 'w') as f: 
    cPickle.dump(data, f)

def load_pickle(path):
    data = bz2.BZ2File(path+'.pbz2', 'rb')
    data = cPickle.load(data)
    return data
```

```python id="5lHX1pHU7fvN"
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

```python id="SvNkS4mP7T7m"
def average_pooling(emb, seq_len):
    mask = tf.sequence_mask(seq_len, tf.shape(emb)[-2], dtype=tf.float32)  # [B, T]
    mask = tf.expand_dims(mask, -1)  # [B, T, 1]
    emb *= mask  # [B, T, H]
    sum_pool = tf.reduce_sum(emb, -2)  # [B, H]
    avg_pool = tf.div(sum_pool, tf.expand_dims(tf.cast(seq_len, tf.float32), -1) + 1e-8)  # [B, H]
    return avg_pool
```

```python id="WQjMHSPwCTgU"
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

<!-- #region id="PguTj6gN2oj8" -->
### Dataset
<!-- #endregion -->

```python id="XQbXoMDO26pN"
def _gen_neg(num_items, pos_ls, num_neg):
    neg_ls = []
    for n in range(num_neg):  # generate num_neg
        neg = pos_ls[0]
        while neg in pos_ls:
            neg = random.randint(0, num_items - 1)
        neg_ls.append(neg)
    return neg_ls
```

```python id="jnzORRSw2p_v"
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

```python id="nxC7vY0i-DtE"
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

```python id="rzeAyivB-DrL"
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

```python id="YECKrjGc-Dow"
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

```python id="l41GttlhB0Ry"
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

<!-- #region id="P0nc1kyjzX9T" -->
## Jobs
<!-- #endregion -->

```python id="2y8mdDjds6dr" colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["83c9f2d6bc6a42109250d55a39b4b3e7", "6af9c2d14cd344408ebeb165aecd5812", "14a3c2eab70348fd8336ae11dbbfcffd", "1f7c997368b34aedb5ecafc1ad6b1121", "712fb93cec584db29a18f17f48ce85dc", "6149942fe6ad4e0ea119545de5a8d242", "f4349bb599ab413c9c087714eb7ba11f", "7b90eb42b83641b4890847c92356a8f6", "8d3fb8b1ab5742ec983d41c822319729", "27e8cfe72e7a4a93b1f2e1f705a551a5", "74ce21ef75514a30a12822ebc5165352", "30c470ccdef34a2999ab74aad3503ba7", "7ef1f03aed684fdbbc6602ec174cd2cd", "45c6daf562174f79ad42975093c84210", "d3fc9033cc8f45c9a9bc114a6c590fd9", "53e9c4c383124944a89e833d7f0a56df", "e69ef6e531314efe9a9ee6dc58e5998b", "813103a929a84c78aa67e72419563cb7", "0fa28afc080a4e98a0f5ccd7d083e3b3", "60fe4254494e4bc589c971c5e287ec88", "bb7538a3fc0045c28b51edc7f7cbbc00", "d30f88908fe54f779659b41bd1e5ba6f"]} executionInfo={"status": "ok", "timestamp": 1636550822729, "user_tz": -330, "elapsed": 2141725, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="af2586b2-0874-4753-f5f5-7da803ca06aa"
logger.info('JOB START: PREPROCESS_DATASET')
preprocess_sobazaar()
logger.info('JOB END: PREPROCESS_DATASET')
```

```python id="3ig3tPpB2Fx-" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1636551549040, "user_tz": -330, "elapsed": 294427, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="59104a12-2943-4b0d-da80-edb832a4ff2b"
logger.info('JOB START: EMBEDMLP_PRETRAINING')
pretrain_sobazaar()
logger.info('JOB END: EMBEDMLP_PRETRAINING')
```

```python colab={"base_uri": "https://localhost:8080/"} id="Y3quOeuqFHwF" executionInfo={"status": "ok", "timestamp": 1636552312547, "user_tz": -330, "elapsed": 146736, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="81947f97-65ce-4aa0-e9de-1e12d5e5645a"
logger.info('JOB START: INCREMENTAL_UPDATE')
iu_sobazaar()
logger.info('JOB END: INCREMENTAL_UPDATE')
```
