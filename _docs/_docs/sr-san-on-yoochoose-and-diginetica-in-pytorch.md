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

<!-- #region id="P1vXdMVJy82x" -->
# SR-SAN on Yoochoose and Diginetica in PyTorch
<!-- #endregion -->

<!-- #region id="4O2XCIFPx9hu" -->
Session-based recommendation aims to predict user's next behavior from current session and previous anonymous sessions. Capturing long-range dependencies between items is a vital challenge in session-based recommendation. A novel approach is proposed for session-based recommendation with self-attention networks (SR-SAN) as a remedy. The self-attention networks (SAN) allow SR-SAN capture the global dependencies among all items of a session regardless of their distance. In SR-SAN, a single item latent vector is used to capture both current interest and global interest instead of session embedding which is composed of current interest embedding and global interest embedding. Some experiments have been performed on some open benchmark datasets. Experimental results show that the proposed method outperforms some state-of-the-arts by comparisons.

Firstly, a self-attention based model which captures and reserves the full dependencies among all items regardless of their distance is proposed without using RNNs or GNNs. Secondly, to generate session-based recommendations, the proposed method use a single item latent vector which jointly represents current interest and global interest instead of session embedding which is composed of current interest embedding and global interest embedding. In RNNs or GNNs based methods, the global interest embedding usually obtained by aggregating all items in the session with attention mechanism which is based on current interest embedding. However, this is redundant in SR-SAN which last item embedding is aggregating all items in the session with self-attention mechanism. In this way, the last item embedding in session can jointly represent current interest and global interest.

It utilizes the self-attention to learn global item dependencies. The multi-head attention mechanism is adopted to allow SR-SAN focus on different important part of the session. The latent vector of the last item in the session is used to jointly represents current interest and global interest with prediction layer.

<img src='_images/T701627_i1.png'>

Session-based recommender system makes prediction based upon current user sessions data without accessing to the long-term preference profile. Let $V = \{v_1, v_2, . . ., v_{|V|}\}$ denote the set consisting of all unique items involved in all the sessions. An anonymous session sequence $S$ can be represented by a list $S = [s_1, s_2, . . ., s_n]$, where $s_i ∈ V$ represents a clicked item of the user within the session $S$. The task of session-based recommendation is to predict the next click $s_{n+1}$ for session $S$. Our models are constructed and trained as a classifier that learns to generate a score for each of the candidates in $V$. Let $\hat{y} = \{\hat{y}_1, \hat{y}_2, . . ., \hat{y}_{|V|}\}$ denote the output score vector, where $\hat{y}_i$ corresponds to the score of item $v_i$. The items with top-K values in $\hat{y}$ will be the candidate items for recommendation.

The proposed model is made up of two parts. The first part is obtaining item latent vectors with self-attention networks, the second part of the proposed model is making recommendation with prediction layer.

**Comparisons with Some Baseline Methods**

<img src='_images/T701627_i2.png' width=50%>
<!-- #endregion -->

<!-- #region id="-HT9tL9qOxOA" -->
## Setup
<!-- #endregion -->

<!-- #region id="M6aJ98EmOyMm" -->
### Installations
<!-- #endregion -->

```python id="pVaRENuTLoKT"
!pip install torch==1.2
```

<!-- #region id="tYx9mLxs1z3v" -->
### Imports
<!-- #endregion -->

```python id="F2D3QNhe5XYn"
import argparse
import time
import csv
import pickle
import operator
import datetime
import os
import numpy as np
import math

import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.nn import TransformerEncoder
from torch.nn import TransformerEncoderLayer
```

<!-- #region id="aGz_V3mX12rJ" -->
## Datasets
<!-- #endregion -->

<!-- #region id="FIEoKj3k16q1" -->
### Yoochoose
<!-- #endregion -->

```python id="6avuKGrH3m72"
!wget https://s3-eu-west-1.amazonaws.com/yc-rdata/yoochoose-data.7z
!7z x yoochoose-data.7z -o./yoochoose-data
```

```python id="jHlbN_a-19Bv"
yoochoose_data_path = '/content/yoochoose-data'
```

```python id="JiCBCRl84aZF"
class YoochooseDataset:
    def __init__(self, path='./'):
        self.path = path

    def preprocess(self):
        dataset = os.path.join(self.path, 'yoochoose-clicks.dat')
        !sed -i '1i session_id,timestamp,item_id,category' $dataset
        print("-- Starting @ %ss" % datetime.datetime.now())
        with open(dataset, "r") as f:
            reader = csv.DictReader(f, delimiter=',')
            sess_clicks = {}
            sess_date = {}
            ctr = 0
            curid = -1
            curdate = None
            for data in reader:
                sessid = data['session_id']
                if curdate and not curid == sessid:
                    date = ''
                    date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
                    sess_date[curid] = date
                curid = sessid
                item = data['item_id']
                curdate = ''
                curdate = data['timestamp']
                if sessid in sess_clicks:
                    sess_clicks[sessid] += [item]
                else:
                    sess_clicks[sessid] = [item]
                ctr += 1
            date = ''
            date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
            sess_date[curid] = date

        print("-- Reading data @ %ss" % datetime.datetime.now())

        # Filter out length 1 sessions
        for s in list(sess_clicks):
            if len(sess_clicks[s]) == 1:
                del sess_clicks[s]
                del sess_date[s]

        # Count number of times each item appears
        iid_counts = {}
        for s in sess_clicks:
            seq = sess_clicks[s]
            for iid in seq:
                if iid in iid_counts:
                    iid_counts[iid] += 1
                else:
                    iid_counts[iid] = 1

        sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

        length = len(sess_clicks)
        for s in list(sess_clicks):
            curseq = sess_clicks[s]
            filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
            if len(filseq) < 2:
                del sess_clicks[s]
                del sess_date[s]
            else:
                sess_clicks[s] = filseq

        # Split out test set based on dates
        dates = list(sess_date.items())
        maxdate = dates[0][1]

        for _, date in dates:
            if maxdate < date:
                maxdate = date

        # 7 days for test
        splitdate = 0
        splitdate = maxdate - 86400 * 1  # the number of seconds for a day：86400

        print('Splitting date', splitdate)      # Yoochoose: ('Split date', 1411930799.0)
        tra_sess = filter(lambda x: x[1] < splitdate, dates)
        tes_sess = filter(lambda x: x[1] > splitdate, dates)

        # Sort sessions by date
        tra_sess = sorted(tra_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
        tes_sess = sorted(tes_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
        print(len(tra_sess))    # 186670    # 7966257
        print(len(tes_sess))    # 15979     # 15324
        print(tra_sess[:3])
        print(tes_sess[:3])
        
        print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())

        # Choosing item count >=5 gives approximately the same number of items as reported in paper
        item_dict = {}
        # Convert training sessions to sequences and renumber items to start from 1
        def obtian_tra():
            train_ids = []
            train_seqs = []
            train_dates = []
            item_ctr = 1
            for s, date in tra_sess:
                seq = sess_clicks[s]
                outseq = []
                for i in seq:
                    if i in item_dict:
                        outseq += [item_dict[i]]
                    else:
                        outseq += [item_ctr]
                        item_dict[i] = item_ctr
                        item_ctr += 1
                if len(outseq) < 2:  # Doesn't occur
                    continue
                train_ids += [s]
                train_dates += [date]
                train_seqs += [outseq]
            print(item_ctr)     # 43098, 37484
            return train_ids, train_dates, train_seqs


        # Convert test sessions to sequences, ignoring items that do not appear in training set
        def obtian_tes():
            test_ids = []
            test_seqs = []
            test_dates = []
            for s, date in tes_sess:
                seq = sess_clicks[s]
                outseq = []
                for i in seq:
                    if i in item_dict:
                        outseq += [item_dict[i]]
                if len(outseq) < 2:
                    continue
                test_ids += [s]
                test_dates += [date]
                test_seqs += [outseq]
            return test_ids, test_dates, test_seqs

        tra_ids, tra_dates, tra_seqs = obtian_tra()
        tes_ids, tes_dates, tes_seqs = obtian_tes()

        def process_seqs(iseqs, idates):
            out_seqs = []
            out_dates = []
            labs = []
            ids = []
            for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
                for i in range(1, len(seq)):
                    tar = seq[-i]
                    labs += [tar]
                    out_seqs += [seq[:-i]]
                    out_dates += [date]
                    ids += [id]
            return out_seqs, out_dates, labs, ids

        tr_seqs, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates)
        te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates)
        tra = (tr_seqs, tr_labs)
        tes = (te_seqs, te_labs)
        print(len(tr_seqs))
        print(len(te_seqs))
        print(tr_seqs[:3], tr_dates[:3], tr_labs[:3])
        print(te_seqs[:3], te_dates[:3], te_labs[:3])
        all = 0

        for seq in tra_seqs:
            all += len(seq)
        for seq in tes_seqs:
            all += len(seq)
        print('avg length: ', all/(len(tra_seqs) + len(tes_seqs) * 1.0))

        if not os.path.exists('yoochoose1_64'):
            os.makedirs('yoochoose1_64')
        pickle.dump(tes, open('yoochoose1_64/test.txt', 'wb'))
        split64 = int(len(tr_seqs) / 64)
        print(len(tr_seqs[-split64:]))

        tra64 = (tr_seqs[-split64:], tr_labs[-split64:])
        seq64 = tra_seqs[tr_ids[-split64]:]

        pickle.dump(tra64, open('yoochoose1_64/train.txt', 'wb'))
        pickle.dump(seq64, open('yoochoose1_64/all_train_seq.txt', 'wb'))

        print('Done.')
```

```python colab={"base_uri": "https://localhost:8080/"} id="hQ0FWjgt7gP4" executionInfo={"status": "ok", "timestamp": 1633848858187, "user_tz": -330, "elapsed": 577936, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ad0618db-e1a2-47c4-e0f9-9c15262fa48e"
yc_data = YoochooseDataset(path=yoochoose_data_path)
yc_data.preprocess()
```

<!-- #region id="XDyp3JBgCr-0" -->
### Diginetica
<!-- #endregion -->

```python id="umn5D6v-CtOC"
!pip install -U -q PyDrive dvc dvc[gdrive]
!dvc get https://github.com/recohut/recodata diginetica/train-item-views.parquet.snappy
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="Ugi7XgwtC2TN" executionInfo={"status": "ok", "timestamp": 1633850030417, "user_tz": -330, "elapsed": 1985, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="82ff6f13-b153-4994-82cb-4f4c1c4cd7b1"
import pandas as pd

df = pd.read_parquet('/content/train-item-views.parquet.snappy')
df.head()
```

```python id="ue4HY2SIDZLU"
df.columns = ['session_id', 'user_id', 'item_id', 'timeframe', 'eventdate']
df.to_csv('/content/train-item-views.csv', index=False)
```

```python id="uhtw0alvDPcD"
class DigineticaDataset:
    def __init__(self, path='./'):
        self.path = path

    def preprocess(self):
        dataset = os.path.join(self.path, 'train-item-views.csv')
        print("-- Starting @ %ss" % datetime.datetime.now())
        with open(dataset, "r") as f:
            reader = csv.DictReader(f, delimiter=',')
            sess_clicks = {}
            sess_date = {}
            ctr = 0
            curid = -1
            curdate = None
            for data in reader:
                sessid = data['session_id']
                if curdate and not curid == sessid:
                    date = ''
                    date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
                    sess_date[curid] = date
                curid = sessid
                item = data['item_id'], int(data['timeframe'])
                curdate = ''
                curdate = data['eventdate']
                if sessid in sess_clicks:
                    sess_clicks[sessid] += [item]
                else:
                    sess_clicks[sessid] = [item]
                ctr += 1
            date = ''
            date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
            for i in list(sess_clicks):
                sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
                sess_clicks[i] = [c[0] for c in sorted_clicks]
            sess_date[curid] = date

        print("-- Reading data @ %ss" % datetime.datetime.now())

        # Filter out length 1 sessions
        for s in list(sess_clicks):
            if len(sess_clicks[s]) == 1:
                del sess_clicks[s]
                del sess_date[s]

        # Count number of times each item appears
        iid_counts = {}
        for s in sess_clicks:
            seq = sess_clicks[s]
            for iid in seq:
                if iid in iid_counts:
                    iid_counts[iid] += 1
                else:
                    iid_counts[iid] = 1

        sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

        length = len(sess_clicks)
        for s in list(sess_clicks):
            curseq = sess_clicks[s]
            filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
            if len(filseq) < 2:
                del sess_clicks[s]
                del sess_date[s]
            else:
                sess_clicks[s] = filseq

        # Split out test set based on dates
        dates = list(sess_date.items())
        maxdate = dates[0][1]

        for _, date in dates:
            if maxdate < date:
                maxdate = date

        # 7 days for test
        splitdate = 0
        splitdate = maxdate - 86400 * 7

        print('Splitting date', splitdate)      # Yoochoose: ('Split date', 1411930799.0)
        tra_sess = filter(lambda x: x[1] < splitdate, dates)
        tes_sess = filter(lambda x: x[1] > splitdate, dates)

        # Sort sessions by date
        tra_sess = sorted(tra_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
        tes_sess = sorted(tes_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
        print(len(tra_sess))    # 186670    # 7966257
        print(len(tes_sess))    # 15979     # 15324
        print(tra_sess[:3])
        print(tes_sess[:3])
        
        print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())

        # Choosing item count >=5 gives approximately the same number of items as reported in paper
        item_dict = {}
        # Convert training sessions to sequences and renumber items to start from 1
        def obtian_tra():
            train_ids = []
            train_seqs = []
            train_dates = []
            item_ctr = 1
            for s, date in tra_sess:
                seq = sess_clicks[s]
                outseq = []
                for i in seq:
                    if i in item_dict:
                        outseq += [item_dict[i]]
                    else:
                        outseq += [item_ctr]
                        item_dict[i] = item_ctr
                        item_ctr += 1
                if len(outseq) < 2:  # Doesn't occur
                    continue
                train_ids += [s]
                train_dates += [date]
                train_seqs += [outseq]
            print(item_ctr)     # 43098, 37484
            return train_ids, train_dates, train_seqs


        # Convert test sessions to sequences, ignoring items that do not appear in training set
        def obtian_tes():
            test_ids = []
            test_seqs = []
            test_dates = []
            for s, date in tes_sess:
                seq = sess_clicks[s]
                outseq = []
                for i in seq:
                    if i in item_dict:
                        outseq += [item_dict[i]]
                if len(outseq) < 2:
                    continue
                test_ids += [s]
                test_dates += [date]
                test_seqs += [outseq]
            return test_ids, test_dates, test_seqs

        tra_ids, tra_dates, tra_seqs = obtian_tra()
        tes_ids, tes_dates, tes_seqs = obtian_tes()

        def process_seqs(iseqs, idates):
            out_seqs = []
            out_dates = []
            labs = []
            ids = []
            for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
                for i in range(1, len(seq)):
                    tar = seq[-i]
                    labs += [tar]
                    out_seqs += [seq[:-i]]
                    out_dates += [date]
                    ids += [id]
            return out_seqs, out_dates, labs, ids

        tr_seqs, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates)
        te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates)
        tra = (tr_seqs, tr_labs)
        tes = (te_seqs, te_labs)
        print(len(tr_seqs))
        print(len(te_seqs))
        print(tr_seqs[:3], tr_dates[:3], tr_labs[:3])
        print(te_seqs[:3], te_dates[:3], te_labs[:3])
        all = 0

        for seq in tra_seqs:
            all += len(seq)
        for seq in tes_seqs:
            all += len(seq)
        print('avg length: ', all/(len(tra_seqs) + len(tes_seqs) * 1.0))

        if not os.path.exists('diginetica'):
            os.makedirs('diginetica')
        pickle.dump(tra, open('diginetica/train.txt', 'wb'))
        pickle.dump(tes, open('diginetica/test.txt', 'wb'))
        pickle.dump(tra_seqs, open('diginetica/all_train_seq.txt', 'wb'))

        print('Done.')
```

```python colab={"base_uri": "https://localhost:8080/"} id="SHhyliplElJk" executionInfo={"status": "ok", "timestamp": 1633851227433, "user_tz": -330, "elapsed": 21388, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="952303f5-4b15-4a8b-b24b-87c33eb6f3ae"
yc_data = DigineticaDataset(path='/content')
yc_data.preprocess()
```

<!-- #region id="LMTOnf-H81gC" -->
## Utils
<!-- #endregion -->

```python id="vO2GDZgJApAK"
def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, data, shuffle=False, graph=None):
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i):
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)
        for u_input in inputs:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        return alias_inputs, A, items, mask, targets
```

<!-- #region id="9yRZQEHDAwQC" -->
## Model
<!-- #endregion -->

```python id="3SlFIw0VA6g_"
class SelfAttentionNetwork(Module):
    def __init__(self, opt, n_node):
        super(SelfAttentionNetwork, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.transformerEncoderLayer = TransformerEncoderLayer(d_model=self.hidden_size, nhead=opt.nhead,dim_feedforward=self.hidden_size * opt.feedforward)
        self.transformerEncoder = TransformerEncoder(self.transformerEncoderLayer, opt.layer)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(ht, b.transpose(1, 0))
        return scores

    def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden = hidden.transpose(0,1).contiguous()
        hidden = self.transformerEncoder(hidden)
        hidden = hidden.transpose(0,1).contiguous()
        return hidden


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    hidden = model(items, A)
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    return targets, model.compute_scores(seq_hidden, mask)


def train_test(model, train_data, test_data):    
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores = forward(model, i, test_data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    model.scheduler.step()
    return hit, mrr
```

<!-- #region id="KyDKrrXQBMNi" -->
## Run
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="0uiAZTj_A7C4" executionInfo={"status": "ok", "timestamp": 1633852518310, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="65607ce6-ef44-4c48-911c-03a44fa3bc01"
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='dataset name: diginetica/yoochoose1_64')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=96, help='hidden state size')
parser.add_argument('--nhead', type=int, default=2, help='the number of heads of multi-head attention')
parser.add_argument('--layer', type=int, default=1, help='number of SAN layers')
parser.add_argument('--feedforward', type=int, default=4, help='the multipler of hidden state size')
parser.add_argument('--epoch', type=int, default=12, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
opt = parser.parse_args(args={})
print(opt)
```

```python colab={"base_uri": "https://localhost:8080/"} id="DZJqCsxOBLc1" executionInfo={"status": "ok", "timestamp": 1633860699299, "user_tz": -330, "elapsed": 5984747, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e4ab275e-b67a-4508-cb30-572211f2d0e9"
def main():
    train_data = pickle.load(open(opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open(opt.dataset + '/test.txt', 'rb'))

    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)
    
    # n_node = 37484
    n_node = 43098
    # if opt.dataset == 'diginetica':
    #     n_node = 43098        

    model = trans_to_cuda(SelfAttentionNetwork(opt, n_node))

    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit, mrr = train_test(model, train_data, test_data)
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
```
