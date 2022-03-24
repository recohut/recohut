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

```python colab={"base_uri": "https://localhost:8080/"} id="qLK7OOF3cyjk" executionInfo={"status": "ok", "timestamp": 1621241408113, "user_tz": -330, "elapsed": 1917, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="364434f4-9b0b-47c2-d057-ec146dc73bdd"
!wget https://s3.us-west-2.amazonaws.com/dgl-data/dataset/gowalla.zip
!unzip gowalla.zip
```

```python colab={"base_uri": "https://localhost:8080/"} id="h8mBkow3dF7o" executionInfo={"status": "ok", "timestamp": 1621241413829, "user_tz": -330, "elapsed": 7595, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c7c73d05-0b5e-4ac8-9efe-7861a384e6a5"
!pip install -q dgl
```

```python id="ws05UY1zeoXL" executionInfo={"status": "ok", "timestamp": 1621241413832, "user_tz": -330, "elapsed": 7577, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
!mkdir -p utility
```

```python colab={"base_uri": "https://localhost:8080/"} id="Xja7qUtLeyGn" executionInfo={"status": "ok", "timestamp": 1621241413833, "user_tz": -330, "elapsed": 7574, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fa32d3d7-91e2-426e-e98f-36cb7a80f65d"
%%writefile utility/load_data.py

# This file is based on the NGCF author's implementation
# <https://github.com/xiangwang1223/neural_graph_collaborative_filtering/blob/master/NGCF/utility/load_data.py>.
# It implements the data processing and graph construction.
import numpy as np
import random as rd
import dgl

class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        #get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.exist_users = []

        user_item_src = []
        user_item_dst = []

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)
                    for i in l[1:]:
                        user_item_src.append(uid)
                        user_item_dst.append(int(i))

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        self.n_items += 1
        self.n_users += 1

        self.print_statistics()

        #training positive items corresponding to each user; testing positive items corresponding to each user
        self.train_items, self.test_set = {}, {}
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0:
                        break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]
                    self.train_items[uid] = train_items

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue

                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items
        
        #construct graph from the train data and add self-loops
        user_selfs = [ i for i in range(self.n_users)]
        item_selfs = [ i for i in range(self.n_items)]
        
        data_dict = {
                ('user', 'user_self', 'user') : (user_selfs, user_selfs),
                ('item', 'item_self', 'item') : (item_selfs, item_selfs),
                ('user', 'ui', 'item') : (user_item_src, user_item_dst),
                ('item', 'iu', 'user') : (user_item_dst, user_item_src)
                }
        num_dict = {
            'user': self.n_users, 'item': self.n_items
        }

        self.g = dgl.heterograph(data_dict, num_nodes_dict=num_dict)


    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            # sample num pos items for u-th user
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            # sample num neg items for u-th user
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items


        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))
```

```python colab={"base_uri": "https://localhost:8080/"} id="48eSz3I9exya" executionInfo={"status": "ok", "timestamp": 1621241413835, "user_tz": -330, "elapsed": 7570, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4140d1d0-cdaa-45c6-ba50-e5056b282a1f"
%%writefile utility/metrics.py

# This file is copied from the NGCF author's implementation
# <https://github.com/xiangwang1223/neural_graph_collaborative_filtering/blob/master/NGCF/utility/metrics.py>.
# It implements the metrics.
'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''

import numpy as np
from sklearn.metrics import roc_auc_score

def recall(rank, ground_truth, N):
    return len(set(rank[:N]) & set(ground_truth)) / float(len(set(ground_truth)))


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)


def average_precision(r,cut):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    Returns:
        Average precision
    """
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(cut) if r[k]]
    if not out:
        return 0.
    return np.sum(out)/float(min(cut, np.sum(r)))


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num


def hit_at_k(r, k):
    r = np.array(r)[:k]
    if np.sum(r) > 0:
        return 1.
    else:
        return 0.

def F1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.

def auc(ground_truth, prediction):
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.
    return res
```

```python colab={"base_uri": "https://localhost:8080/"} id="-VWXIW6BgAkc" executionInfo={"status": "ok", "timestamp": 1621241413836, "user_tz": -330, "elapsed": 7566, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5ee53cda-dc57-4dd2-855b-eb1c8a2726c2"
%%writefile utility/helper.py

# This file is copied from the NGCF author's implementation
# <https://github.com/xiangwang1223/neural_graph_collaborative_filtering/blob/master/NGCF/utility/helper.py>.
# It implements the helper functions.
'''
Created on Aug 19, 2016
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
__author__ = "xiangwang"
import os
import re

def txt2list(file_src):
    orig_file = open(file_src, "r")
    lines = orig_file.readlines()
    return lines

def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)

def uni2str(unicode_str):
    return str(unicode_str.encode('ascii', 'ignore')).replace('\n', '').strip()

def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

def delMultiChar(inputString, chars):
    for ch in chars:
        inputString = inputString.replace(ch, '')
    return inputString

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop
```

```python id="GRsRMAjQjj7R" executionInfo={"status": "ok", "timestamp": 1621244667266, "user_tz": -330, "elapsed": 1138, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# This file is based on the NGCF author's implementation
# <https://github.com/xiangwang1223/neural_graph_collaborative_filtering/blob/master/NGCF/utility/parser.py>.

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run NGCF.")
    parser.add_argument('--weights_path', nargs='?', default='./',
                        help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default='./',
                        help='Input data path.')
    parser.add_argument('--model_name', type=str, default='NGCF.pkl',
                        help='Saved model name.')


    parser.add_argument('--dataset', nargs='?', default='gowalla',
                        help='Choose a dataset from {gowalla, yelp2018, amazon-book}')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=400,
                        help='Number of epoch.')

    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[64,64,64]',
                        help='Output sizes of every layer')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')

    parser.add_argument('--regs', nargs='?', default='[1e-5]',
                        help='Regularizations.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')


    parser.add_argument('--gpu', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')

    parser.add_argument('--mess_dropout', nargs='?', default='[0.1,0.1,0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--Ks', nargs='?', default='[20, 40]',
                        help='Output sizes of every layer')

    parser.add_argument('--save_flag', type=int, default=1,
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')
    return parser.parse_known_args()
```

```python colab={"base_uri": "https://localhost:8080/"} id="4gCHaVRfdrIJ" executionInfo={"status": "ok", "timestamp": 1621243364490, "user_tz": -330, "elapsed": 1283, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8659c80d-80d0-46be-dfd7-98b1047e5993"
%%writefile model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

class NGCFLayer(nn.Module):
    def __init__(self, in_size, out_size, norm_dict, dropout):
        super(NGCFLayer, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

        #weights for different types of messages
        self.W1 = nn.Linear(in_size, out_size, bias = True)
        self.W2 = nn.Linear(in_size, out_size, bias = True)

        #leaky relu
        self.leaky_relu = nn.LeakyReLU(0.2)

        #dropout layer
        self.dropout = nn.Dropout(dropout)

        #initialization
        torch.nn.init.xavier_uniform_(self.W1.weight)
        torch.nn.init.constant_(self.W1.bias, 0)
        torch.nn.init.xavier_uniform_(self.W2.weight)
        torch.nn.init.constant_(self.W2.bias, 0)

        #norm
        self.norm_dict = norm_dict

    def forward(self, g, feat_dict):

        funcs = {} #message and reduce functions dict
        #for each type of edges, compute messages and reduce them all
        for srctype, etype, dsttype in g.canonical_etypes:
            if srctype == dsttype: #for self loops
                messages = self.W1(feat_dict[srctype])
                g.nodes[srctype].data[etype] = messages   #store in ndata
                funcs[(srctype, etype, dsttype)] = (fn.copy_u(etype, 'm'), fn.sum('m', 'h'))  #define message and reduce functions
            else:
                src, dst = g.edges(etype=(srctype, etype, dsttype))
                norm = self.norm_dict[(srctype, etype, dsttype)]
                messages = norm * (self.W1(feat_dict[srctype][src]) + self.W2(feat_dict[srctype][src]*feat_dict[dsttype][dst])) #compute messages
                g.edges[(srctype, etype, dsttype)].data[etype] = messages  #store in edata
                funcs[(srctype, etype, dsttype)] = (fn.copy_e(etype, 'm'), fn.sum('m', 'h'))  #define message and reduce functions

        g.multi_update_all(funcs, 'sum') #update all, reduce by first type-wisely then across different types
        feature_dict={}
        for ntype in g.ntypes:
            h = self.leaky_relu(g.nodes[ntype].data['h']) #leaky relu
            h = self.dropout(h) #dropout
            h = F.normalize(h,dim=1,p=2) #l2 normalize
            feature_dict[ntype] = h
        return feature_dict

class NGCF(nn.Module):
    def __init__(self, g, in_size, layer_size, dropout, lmbd=1e-5):
        super(NGCF, self).__init__()
        self.lmbd = lmbd
        self.norm_dict = dict()
        for srctype, etype, dsttype in g.canonical_etypes:
            src, dst = g.edges(etype=(srctype, etype, dsttype))
            dst_degree = g.in_degrees(dst, etype=(srctype, etype, dsttype)).float() #obtain degrees
            src_degree = g.out_degrees(src, etype=(srctype, etype, dsttype)).float()
            norm = torch.pow(src_degree * dst_degree, -0.5).unsqueeze(1) #compute norm
            self.norm_dict[(srctype, etype, dsttype)] = norm

        self.layers = nn.ModuleList()
        self.layers.append(
            NGCFLayer(in_size, layer_size[0], self.norm_dict, dropout[0])
        )
        self.num_layers = len(layer_size)
        for i in range(self.num_layers-1):
            self.layers.append(
                NGCFLayer(layer_size[i], layer_size[i+1], self.norm_dict, dropout[i+1])
            )
        self.initializer = nn.init.xavier_uniform_

        #embeddings for different types of nodes
        self.feature_dict = nn.ParameterDict({
            ntype: nn.Parameter(self.initializer(torch.empty(g.num_nodes(ntype), in_size))) for ntype in g.ntypes
        })

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = (users * pos_items).sum(1)
        neg_scores = (users * neg_items).sum(1)

        mf_loss = nn.LogSigmoid()(pos_scores - neg_scores).mean()
        mf_loss = -1 * mf_loss

        regularizer = (torch.norm(users) ** 2 + torch.norm(pos_items) ** 2 + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.lmbd * regularizer / users.shape[0]

        return mf_loss + emb_loss, mf_loss, emb_loss

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    def forward(self, g,user_key, item_key, users, pos_items, neg_items):
        h_dict = {ntype : self.feature_dict[ntype] for ntype in g.ntypes}
        #obtain features of each layer and concatenate them all
        user_embeds = []
        item_embeds = []
        user_embeds.append(h_dict[user_key])
        item_embeds.append(h_dict[item_key])
        for layer in self.layers:
            h_dict = layer(g, h_dict)
            user_embeds.append(h_dict[user_key])
            item_embeds.append(h_dict[item_key])
        user_embd = torch.cat(user_embeds, 1)
        item_embd = torch.cat(item_embeds, 1)

        u_g_embeddings = user_embd[users, :]
        pos_i_g_embeddings = item_embd[pos_items, :]
        neg_i_g_embeddings = item_embd[neg_items, :]

        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
```

```python id="UrPPVUb9ktx3" executionInfo={"status": "ok", "timestamp": 1621244712262, "user_tz": -330, "elapsed": 918, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# This file is based on the NGCF author's implementation
# <https://github.com/xiangwang1223/neural_graph_collaborative_filtering/blob/master/NGCF/utility/batch_test.py>.
# It implements the batch test.

import utility.metrics as metrics
from utility.load_data import *
import multiprocessing
import heapq
```

```python id="mUvhoJ86kwDj" executionInfo={"status": "ok", "timestamp": 1621244679290, "user_tz": -330, "elapsed": 5929, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
cores = multiprocessing.cpu_count()
```

```python id="oAl-MEIKkzoI" executionInfo={"status": "ok", "timestamp": 1621244679292, "user_tz": -330, "elapsed": 5736, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
args, unknown = parse_args()
Ks = eval(args.Ks)
```

```python colab={"base_uri": "https://localhost:8080/"} id="3jVnlhvsk16t" executionInfo={"status": "ok", "timestamp": 1621244748440, "user_tz": -330, "elapsed": 2737, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1343e0c0-9b33-41fa-b222-741b73b2d1bf"
data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = args.batch_size
```

```python id="dSpkZGjMk13H" executionInfo={"status": "ok", "timestamp": 1621244755168, "user_tz": -330, "elapsed": 1434, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc
```

```python id="qNJI8Yu7lBI6" executionInfo={"status": "ok", "timestamp": 1621244756255, "user_tz": -330, "elapsed": 2361, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc
```

```python id="2rsJSP1XlDgs" executionInfo={"status": "ok", "timestamp": 1621244756258, "user_tz": -330, "elapsed": 2219, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc
```

```python id="uLYLa2eXlE-9" executionInfo={"status": "ok", "timestamp": 1621244756260, "user_tz": -330, "elapsed": 2087, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}
```

```python id="BIPxCEThlHg3" executionInfo={"status": "ok", "timestamp": 1621244756996, "user_tz": -330, "elapsed": 1094, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    #uid
    u = x[1]
    #user u's items in the training set
    try:
        training_items = data_generator.train_items[u]
    except Exception:
        training_items = []
    #user u's items in the test set
    user_pos_test = data_generator.test_set[u]

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)
```

```python id="49cf5Y8LeE1-" executionInfo={"status": "ok", "timestamp": 1621244756997, "user_tz": -330, "elapsed": 855, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def test(model, g, users_to_test, batch_test_flag=False):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

    pool = multiprocessing.Pool(cores)

    u_batch_size = 5000
    i_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]

        if batch_test_flag:
            # batch-item test
            n_item_batchs = ITEM_NUM // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)

                item_batch = range(i_start, i_end)

                u_g_embeddings, pos_i_g_embeddings, _ = model(g, 'user', 'item',user_batch, item_batch, [])
                i_rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()

                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == ITEM_NUM

        else:
            # all-item test
            item_batch = range(ITEM_NUM)
            u_g_embeddings, pos_i_g_embeddings, _ = model(g, 'user', 'item',user_batch, item_batch, [])
            rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()

        user_batch_rating_uid = zip(rate_batch.numpy(), user_batch)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['auc'] += re['auc']/n_test_users


    assert count == n_test_users
    pool.close()
    return result
```

```python id="IOsb7AqbgvXy" executionInfo={"status": "ok", "timestamp": 1621244759331, "user_tz": -330, "elapsed": 827, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import torch
import torch.optim as optim
from model import NGCF
from utility.helper import early_stopping
from time import time
import os
```

```python id="BtnwxwbwnLws" executionInfo={"status": "ok", "timestamp": 1621244761149, "user_tz": -330, "elapsed": 1580, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Step 1: Prepare graph data and device ================================================================= #
if args.gpu >= 0 and torch.cuda.is_available():
    device = 'cuda:{}'.format(args.gpu)
else:
    device = 'cpu'
g=data_generator.g
g=g.to(device)
```

```python id="ENA17frtnrYk" executionInfo={"status": "ok", "timestamp": 1621244764165, "user_tz": -330, "elapsed": 1196, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
if not os.path.exists(args.weights_path):
    os.makedirs(args.weights_path)
args.mess_dropout = eval(args.mess_dropout)
args.layer_size = eval(args.layer_size)
args.regs = eval(args.regs)
```

```python colab={"base_uri": "https://localhost:8080/"} id="BatinFK7rhIO" executionInfo={"status": "ok", "timestamp": 1621244764781, "user_tz": -330, "elapsed": 1641, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7dcbb385-9711-4078-8b61-dcb36ced448d"
print(args)
```

```python id="2RlATJDZnT8f" executionInfo={"status": "ok", "timestamp": 1621244764784, "user_tz": -330, "elapsed": 1505, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Step 2: Create model and training components=========================================================== #
model = NGCF(g, args.embed_size, args.layer_size, args.mess_dropout, args.regs[0]).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
```

```python colab={"base_uri": "https://localhost:8080/"} id="PZ0Km0PCrzQH" executionInfo={"status": "ok", "timestamp": 1621244766172, "user_tz": -330, "elapsed": 1149, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="490cae6e-46ba-4900-9961-19a0ed2f1cc8"
[len(data_generator.sample()[i]) for i in range(len(data_generator.sample()))]
```

```python colab={"base_uri": "https://localhost:8080/"} id="IVg5nsofdwfN" outputId="c960e760-1e7b-44ba-d29b-bc22d2fbaf76"
# Step 3: training epoches ============================================================================== #
n_batch = data_generator.n_train // args.batch_size + 1
t0 = time()
cur_best_pre_0, stopping_step = 0, 0
loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
for epoch in range(args.epoch):
    t1 = time()
    loss, mf_loss, emb_loss = 0., 0., 0.
    for idx in range(n_batch):
        users, pos_items, neg_items = data_generator.sample()
        u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(g, 'user', 'item', users,
                                                                        pos_items,
                                                                        neg_items)

        batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings,
                                                                          pos_i_g_embeddings,
                                                                          neg_i_g_embeddings)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss
        mf_loss += batch_mf_loss
        emb_loss += batch_emb_loss
        

    if (epoch + 1) % 10 != 0:
        if args.verbose > 0 and epoch % args.verbose == 0:
            perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                epoch, time() - t1, loss, mf_loss, emb_loss)
            print(perf_str)
        continue #end the current epoch and move to the next epoch, let the following evaluation run every 10 epoches

    #evaluate the model every 10 epoches
    t2 = time()
    users_to_test = list(data_generator.test_set.keys())
    ret = test(model, g, users_to_test)
    t3 = time()

    loss_loger.append(loss)
    rec_loger.append(ret['recall'])
    pre_loger.append(ret['precision'])
    ndcg_loger.append(ret['ndcg'])
    hit_loger.append(ret['hit_ratio'])

    if args.verbose > 0:
        perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
                    'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                    (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, ret['recall'][0], ret['recall'][-1],
                    ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                    ret['ndcg'][0], ret['ndcg'][-1])
        print(perf_str)

    cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                stopping_step, expected_order='acc', flag_step=5)

    # early stop
    if should_stop == True:
        break

    if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
        torch.save(model.state_dict(), args.weights_path + args.model_name)
        print('save the weights in path: ', args.weights_path + args.model_name)

recs = np.array(rec_loger)
pres = np.array(pre_loger)
ndcgs = np.array(ndcg_loger)
hit = np.array(hit_loger)

best_rec_0 = max(recs[:, 0])
idx = list(recs[:, 0]).index(best_rec_0)

final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
              (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
              '\t'.join(['%.5f' % r for r in pres[idx]]),
              '\t'.join(['%.5f' % r for r in hit[idx]]),
              '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
print(final_perf)
```

```python id="nCrV8a6it_0Z"

```
