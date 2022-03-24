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

<!-- #region id="I0vYn_6Ks2-h" -->
# MetaTL for Cold-start users on Amazon Electronics dataset
<!-- #endregion -->

<!-- #region id="HGt5M9GHvm_F" -->
## Introduction
<!-- #endregion -->

<!-- #region id="MEjeY5K_voyF" -->
A fundamental challenge for sequential recommenders is to capture the sequential patterns of users toward modeling how users transit among items. In many practical scenarios, however, there are a great number of cold-start users with only minimal logged interactions. As a result, existing sequential recommendation models will lose their predictive power due to the difficulties in learning sequential patterns over users with only limited interactions. In this work, we aim to improve sequential recommendation for cold-start users with a novel framework named MetaTL, which learns to model the transition patterns of users through meta-learning.

Specifically, the proposed MetaTL:

1. formulates sequential recommendation for cold-start users as a few-shot learning problem;
2. extracts the dynamic transition patterns among users with a translation-based architecture; and
3. adopts meta transitional learning to enable fast learning for cold-start users with only limited interactions, leading to accurate inference of sequential interactions.
<!-- #endregion -->

<!-- #region id="yiZ8MmmCvhkG" -->
## Background
<!-- #endregion -->

<!-- #region id="W-phAZTyvsW-" -->
### Sequential Recommenders

One of the first approaches for sequential recommendation is the use of Markov Chains to model the transitions of users among items. More recently, TransRec embeds items in a “transition space” and learns a translation vector for each user. With the advance in neural networks, many different neural structures including Recurrent Neural Networks, Convolutional Neural Networks, Transformers and Graph Neural Networks, have been adopted to model the dynamic preferences of users over their behavior sequences. While these methods aim to improve the overall performance via representation learning for sequences, they suffer from weak prediction power for cold-start users with short behavior sequences.
<!-- #endregion -->

<!-- #region id="-Yy2uAfGwYBs" -->
### Meta Learning

This line of research aims to learn a model which can adapt and generalize to new tasks and new environments with a few training samples. To achieve the goal of “learning-to-learn”, there are three types of different approaches. Metric-based methods are based on a similar idea to the nearest neighbors algorithm with a well-designed metric or distance function, prototypical networks or Siamese Neural Network. Model-based methods usually perform a rapid parameter update with an internal architecture or are controlled by another meta-learner model. As for the optimization-based approaches, by adjusting the optimization algorithm, the models can be efficiently updated with a few examples.
<!-- #endregion -->

<!-- #region id="2Da77sgAwZl4" -->
### Cold-Start Meta Recommenders

MetaRec proposes a meta-learning strategy to learn user-specific logistic regression. There are also methods including MetaCF, Warm-up and MeLU, adopting Model-Agnostic Meta-Learning (MAML) methods to learn a model to achieve fast adaptation for cold-start users.
<!-- #endregion -->

<!-- #region id="y6fHnRMKwbZq" -->
### Cold-Start Meta Sequential Recommenders

cold-start sequential recommendation targets a setting where no additional auxiliary knowledge can be accessed due to privacy issues, and more importantly, the user-item interactions are sequentially dependent. A user’s preferences and tastes may change over time and such dynamics are of great significance in sequential recommendation. Hence, it is necessary to develop a new sequential recommendation framework that can distill short-range item transitional dynamics, and make fast adaptation to those cold-start users with limited user-item interactions.
<!-- #endregion -->

<!-- #region id="f8jhVM9EvvT9" -->
## Problem Statement

Let $I = \{𝑖_1,𝑖_2, \dots,𝑖_𝑃\}$ and $U = \{u_1,u_2, \dots,u_G\}$ represent the item set and user set in the platform respectively. Each item is mapped to a trainable embedding associated with its ID. There is no auxiliary information for users or items. In sequential recommendation, given the sequence of items ${𝑆𝑒𝑞}_𝑢 = (𝑖_{𝑢,1},𝑖_{𝑢,2}, \dots,𝑖_{𝑢,𝑛})$ that user 𝑢 has interacted with in chronological order, the model aims to infer the next interesting item $𝑖_{𝑢,𝑛+1}$. That is to say, we need to predict the preference score for each candidate item based on ${𝑆𝑒𝑞}_𝑢$ and thus recommend the top-N items with the highest scores.

In our task, we train the model on $U_{𝑡𝑟𝑎𝑖𝑛}$, which contains users with various numbers of logged interactions. Then given 𝑢 in a separate test set $U_{𝑡𝑒𝑠𝑡},\ U_{𝑡𝑟𝑎𝑖𝑛} ∩ U_{𝑡𝑒𝑠𝑡} = \phi$, the model can quickly learn user transition patterns according to the 𝐾 initial interactions and thus infer the sequential interactions. Note that the size of a user’s initial interactions (i.e., 𝐾) is assumed to be a small number (e.g., 2, 3 or 4) considering the cold-start scenario.
<!-- #endregion -->

<!-- #region id="zc6NiwbDXm5i" -->
## Setup
<!-- #endregion -->

<!-- #region id="baVt4qbEl5yQ" -->
### Imports
<!-- #endregion -->

```python id="8JV0JbNymEEe"
import os
import sys
import copy
import json
import random
import shutil
import logging
import numpy as np
from collections import defaultdict, Counter, OrderedDict
from multiprocessing import Process, Queue

import torch
import torch.nn as nn
from torch.nn import functional as F
```

<!-- #region id="t9Dgt2BOl_18" -->
### Params
<!-- #endregion -->

```python id="H6PGFIYtmA4s"
class Args:
    dataset = "electronics"
    seed = None
    K = 3 #NUMBER OF SHOT
    embed_dim = 100
    batch_size = 1024
    learning_rate = 0.001
    epoch = 1000
    print_epoch = 100
    eval_epoch = 100
    beta = 5
    margin = 1
    dropout_p = 0.5
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
```

```python colab={"base_uri": "https://localhost:8080/"} id="e5MVLDKMrG2b" executionInfo={"elapsed": 8, "status": "ok", "timestamp": 1635840881817, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} outputId="4d27e948-c7d6-424d-ab4c-3f1c8a8cee8e"
params = dict(Args.__dict__)
params
```

```python id="1zh6XmmwnGFD"
if params['seed'] is not None:
    SEED = params['seed']
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    np.random.seed(SEED)
    random.seed(SEED)
```

<!-- #region id="21UavFIVl65C" -->
## Dataset
<!-- #endregion -->

<!-- #region id="3_iy2vErwCW5" -->
***Electronics*** is adopted from the public Amazon review dataset, which includes reviews ranging from May 1996 to July 2014 on Amazon products belonging to the “Electronics” category.

We filter out items with fewer than 10 interactions. We split each dataset with a corresponding cutting timestamp 𝑇, such that we construct $U_{𝑡𝑟𝑎𝑖𝑛}$ with users who have interactions before 𝑇 and construct $U_{𝑡𝑒𝑠𝑡}$ with users who start their first interactions after 𝑇.

When evaluating few-shot sequential recommendation for a choice of 𝐾 (i.e., the number of initial interactions), we keep 𝐾 interactions as initialization for each user in $U_{𝑡𝑒𝑠𝑡}$ and predict for the user’s next interactions.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="K_2jZQEYr6n2" executionInfo={"elapsed": 1389, "status": "ok", "timestamp": 1635840884082, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} outputId="d35e3a9a-00a2-4ff3-e4e6-ea064d4c3064"
!wget -q --show-progress https://github.com/sparsh-ai/coldstart-recsys/raw/main/data/electronics/electronics_train.csv
!wget -q --show-progress https://github.com/sparsh-ai/coldstart-recsys/raw/main/data/electronics/electronics_test_new_user.csv
```

```python id="yLznX0HIl9wH"
# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t
```

```python id="2aewgHVUpa84"
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
```

```python id="m1pBm8_9pWGj"
# train/val/test data generation
def data_load(fname, num_sample):
    usernum = 0
    itemnum = 0
    user_train = defaultdict(list)

    # assume user/item index starting from 1
    f = open('%s_train.csv' % (fname), 'r')
    for line in f:
        u, i, t = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        user_train[u].append(i)
    f.close()

    # read in new users for testing
    user_input_test = {}
    user_input_valid = {}
    user_valid = {}
    user_test = {}


    User_test_new = defaultdict(list)
    f = open('%s_test_new_user.csv' % (fname), 'r')
    for line in f:
        u, i, t = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        User_test_new[u].append(i)
    f.close()

    for user in User_test_new:
        if len(User_test_new[user]) > num_sample:
            if random.random()<0.3:
                user_input_valid[user] = User_test_new[user][:num_sample]
                user_valid[user] = []
                user_valid[user].append(User_test_new[user][num_sample])
            else:
                user_input_test[user] = User_test_new[user][:num_sample]
                user_test[user] = []
                user_test[user].append(User_test_new[user][num_sample])
    

    return [user_train, usernum, itemnum, user_input_test, user_test, user_input_valid, user_valid]
```

```python id="Nilscd6kpTek"
class DataLoader(object):
    def __init__(self, user_train, user_test, itemnum, parameter):
        self.curr_rel_idx = 0
        
        self.bs = parameter['batch_size']
        self.maxlen = parameter['K']

        self.valid_user = []
        for u in user_train:
            if len(user_train[u]) < self.maxlen or len(user_test[u]) < 1: continue
            self.valid_user.append(u)
        
        self.num_tris = len(self.valid_user)

        self.train = user_train
        self.test = user_test
        
        self.itemnum = itemnum

    def next_one_on_eval(self):
        if self.curr_tri_idx == self.num_tris:
            return "EOT", "EOT"

        u = self.valid_user[self.curr_tri_idx]

        self.curr_tri_idx += 1
        
        seq = np.zeros([self.maxlen], dtype=np.int32)
        pos = np.zeros([self.maxlen - 1], dtype=np.int32)
        neg = np.zeros([self.maxlen - 1], dtype=np.int32)
        
        idx = self.maxlen - 1

        ts = set(self.train[u])
        for i in reversed(self.train[u]):
            seq[idx] = i
            if idx > 0:
                pos[idx - 1] = i
                if i != 0: neg[idx - 1] = random_neq(1, self.itemnum + 1, ts)
            idx -= 1
            if idx == -1: break

        curr_rel = u
        support_triples, support_negative_triples, query_triples, negative_triples = [], [], [], []
        for idx in range(self.maxlen-1):
            support_triples.append([seq[idx],curr_rel,pos[idx]])
            support_negative_triples.append([seq[idx],curr_rel,neg[idx]])

        rated = ts
        rated.add(0)
        query_triples.append([seq[-1],curr_rel,self.test[u][0]])
        for _ in range(100):
            t = np.random.randint(1, self.itemnum + 1)
            while t in rated: t = np.random.randint(1, self.itemnum + 1)
            negative_triples.append([seq[-1],curr_rel,t])

        support_triples = [support_triples]
        support_negative_triples = [support_negative_triples]
        query_triples = [query_triples]
        negative_triples = [negative_triples]

        return [support_triples, support_negative_triples, query_triples, negative_triples], curr_rel
```

<!-- #region id="suxbalu2nPuW" -->
## Sampling
<!-- #endregion -->

```python id="lzXq4p-YnPrk"
def sample_function_mixed(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        if random.random()<0.5:
            user = np.random.randint(1, usernum + 1)
            while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

            
            seq = np.zeros([maxlen], dtype=np.int32)
            pos = np.zeros([maxlen], dtype=np.int32)
            neg = np.zeros([maxlen], dtype=np.int32)

            if len(user_train[user]) < maxlen:
                nxt_idx = len(user_train[user]) - 1
            else:
                nxt_idx = np.random.randint(maxlen,len(user_train[user]))

            nxt = user_train[user][nxt_idx]
            idx = maxlen - 1

            ts = set(user_train[user])
            for i in reversed(user_train[user][min(0, nxt_idx - 1 - maxlen) : nxt_idx - 1]):
                seq[idx] = i
                pos[idx] = nxt
                if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
                nxt = i
                idx -= 1
                if idx == -1: break

            curr_rel = user
            support_triples, support_negative_triples, query_triples, negative_triples = [], [], [], []
            for idx in range(maxlen-1):
                support_triples.append([seq[idx],curr_rel,pos[idx]])
                support_negative_triples.append([seq[idx],curr_rel,neg[idx]])
            query_triples.append([seq[-1],curr_rel,pos[-1]])
            negative_triples.append([seq[-1],curr_rel,neg[-1]])

            return support_triples, support_negative_triples, query_triples, negative_triples, curr_rel

        else:
            user = np.random.randint(1, usernum + 1)
            while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

            seq = np.zeros([maxlen], dtype=np.int32)
            pos = np.zeros([maxlen], dtype=np.int32)
            neg = np.zeros([maxlen], dtype=np.int32)

            list_idx = random.sample([i for i in range(len(user_train[user]))], maxlen + 1)
            list_item = [user_train[user][i] for i in sorted(list_idx)]

            nxt = list_item[-1]
            idx = maxlen - 1

            ts = set(user_train[user])
            for i in reversed(list_item[:-1]):
                seq[idx] = i
                pos[idx] = nxt
                if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
                nxt = i
                idx -= 1
                if idx == -1: break

            curr_rel = user
            support_triples, support_negative_triples, query_triples, negative_triples = [], [], [], []
            for idx in range(maxlen-1):
                support_triples.append([seq[idx],curr_rel,pos[idx]])
                support_negative_triples.append([seq[idx],curr_rel,neg[idx]])
            query_triples.append([seq[-1],curr_rel,pos[-1]])
            negative_triples.append([seq[-1],curr_rel,neg[-1]])

            return support_triples, support_negative_triples, query_triples, negative_triples, curr_rel

    np.random.seed(SEED)
    
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        support, support_negative, query, negative, curr_rel = zip(*one_batch)

        result_queue.put(([support, support_negative, query, negative], curr_rel))
```

```python id="dyBxylgHpvSr"
class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function_mixed, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
```

<!-- #region id="fkhFoNOunPo1" -->
## Model Definition
<!-- #endregion -->

<!-- #region id="x0Vg_yHM0Q3m" -->
<p><center><img src='_images/T714933_1.png'></center></p>
<!-- #endregion -->

```python id="ax73L49BnPl7"
class Embedding(nn.Module):
    def __init__(self, num_ent, parameter):
        super(Embedding, self).__init__()
        self.device = parameter['device']
        self.es = parameter['embed_dim']
        
        self.embedding = nn.Embedding(num_ent + 1, self.es)
        nn.init.xavier_uniform_(self.embedding.weight)


    def forward(self, triples):
        idx = [[[t[0], t[2]] for t in batch] for batch in triples]
        idx = torch.LongTensor(idx).to(self.device)
        return self.embedding(idx)
```

```python id="vQP6N9Eqp_Lr"
class MetaLearner(nn.Module):
    def __init__(self, K, embed_size=100, num_hidden1=500, num_hidden2=200, out_size=100, dropout_p=0.5):
        super(MetaLearner, self).__init__()
        self.embed_size = embed_size
        self.K = K
        self.out_size = out_size
        self.rel_fc1 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(2*embed_size, num_hidden1)),
            ('bn',   nn.BatchNorm1d(K)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc2 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(num_hidden1, num_hidden2)),
            ('bn',   nn.BatchNorm1d(K)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc3 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(num_hidden2, out_size)),
            ('bn', nn.BatchNorm1d(K)),
        ]))
        nn.init.xavier_normal_(self.rel_fc1.fc.weight)
        nn.init.xavier_normal_(self.rel_fc2.fc.weight)
        nn.init.xavier_normal_(self.rel_fc3.fc.weight)

    def forward(self, inputs):
        size = inputs.shape
        x = inputs.contiguous().view(size[0], size[1], -1)
        x = self.rel_fc1(x)
        x = self.rel_fc2(x)
        x = self.rel_fc3(x)
        x = torch.mean(x, 1)

        return x.view(size[0], 1, 1, self.out_size)
```

```python id="DZ4h-PY8p-AK"
class EmbeddingLearner(nn.Module):
    def __init__(self):
        super(EmbeddingLearner, self).__init__()

    def forward(self, h, t, r, pos_num):
        score = -torch.norm(h + r - t, 2, -1).squeeze(2)
        p_score = score[:, :pos_num]
        n_score = score[:, pos_num:]
        return p_score, n_score
```

```python id="mdPC2RHrp8uQ"
class MetaTL(nn.Module):
    def __init__(self, itemnum, parameter):
        super(MetaTL, self).__init__()
        self.device = parameter['device']
        self.beta = parameter['beta']
        self.dropout_p = parameter['dropout_p']
        self.embed_dim = parameter['embed_dim']
        self.margin = parameter['margin']
        self.embedding = Embedding(itemnum, parameter)

        self.relation_learner = MetaLearner(parameter['K'] - 1, embed_size=100, num_hidden1=500,
                                                    num_hidden2=200, out_size=100, dropout_p=self.dropout_p)

        self.embedding_learner = EmbeddingLearner()
        self.loss_func = nn.MarginRankingLoss(self.margin)
        self.rel_q_sharing = dict()

    def split_concat(self, positive, negative):
        pos_neg_e1 = torch.cat([positive[:, :, 0, :],
                                negative[:, :, 0, :]], 1).unsqueeze(2)
        pos_neg_e2 = torch.cat([positive[:, :, 1, :],
                                negative[:, :, 1, :]], 1).unsqueeze(2)
        return pos_neg_e1, pos_neg_e2

    def forward(self, task, iseval=False, curr_rel=''):
        # transfer task string into embedding
        support, support_negative, query, negative = [self.embedding(t) for t in task]

        K = support.shape[1]              # num of K
        num_sn = support_negative.shape[1]  # num of support negative
        num_q = query.shape[1]              # num of query
        num_n = negative.shape[1]           # num of query negative

        rel = self.relation_learner(support)
        rel.retain_grad()

        rel_s = rel.expand(-1, K+num_sn, -1, -1)

        if iseval and curr_rel != '' and curr_rel in self.rel_q_sharing.keys():
            rel_q = self.rel_q_sharing[curr_rel]
        else:
            sup_neg_e1, sup_neg_e2 = self.split_concat(support, support_negative)

            p_score, n_score = self.embedding_learner(sup_neg_e1, sup_neg_e2, rel_s, K)

            y = torch.Tensor([1]).to(self.device)
            self.zero_grad()
            loss = self.loss_func(p_score, n_score, y)
            loss.backward(retain_graph=True)

            grad_meta = rel.grad
            rel_q = rel - self.beta*grad_meta


            self.rel_q_sharing[curr_rel] = rel_q

        rel_q = rel_q.expand(-1, num_q + num_n, -1, -1)

        que_neg_e1, que_neg_e2 = self.split_concat(query, negative)  
        p_score, n_score = self.embedding_learner(que_neg_e1, que_neg_e2, rel_q, num_q)

        return p_score, n_score
```

<!-- #region id="cW-_AcNcmHc9" -->
## Training and Inference
<!-- #endregion -->

<!-- #region id="kImSZhfqv3HE" -->
Meta-learning aims to learn a model which can adapt to new tasks (i.e., new users) with a few training samples. To enable meta-learning in sequential recommendation for cold-start users, we formulate training a sequential recommender as solving a new few-shot learning problem (i.e., meta-testing task) by training on many sampled similar tasks (i.e., the meta-training tasks). Each task includes a 𝑠𝑢𝑝𝑝𝑜𝑟𝑡 set S and a 𝑞𝑢𝑒𝑟𝑦 set Q, which can be regarded as the “training” set and “testing” set of the task. For example, while constructing a task $T_𝑛$, given user $𝑢_𝑗$ with initial interactions in sequence (e.g., $𝑖_𝐴 \rightarrow_{u_j} i_B \rightarrow_{u_j} i_C$), we will have the a set of transition pairs $\{ 𝑖_𝐴 \rightarrow_{u_j} i_B, i_B \rightarrow_{u_j} i_C \}$ as support and predict for the query $i_C \rightarrow_{u_j} ?$.

When testing on a new user $𝑢_{𝑡𝑒𝑠𝑡}$, we will firstly construct the support set $S_{𝑡𝑒𝑠𝑡}$ based on the user’s initial interactions. The model $𝑓_\theta$ is fine-tuned with all the transition pairs in $S_{𝑡𝑒𝑠𝑡}$ and updated to $𝑓_{\theta_{𝑡𝑒𝑠𝑡}'}$ , which can be used to generate the updated $tr_{𝑡𝑒𝑠𝑡}$. Given the test query $𝑖_𝑜 \rightarrow_{u_{test}}?$, the preference score for item $𝑖_𝑝$ (as the next interaction) is calculated as −$∥i_𝑜 + tr_{𝑡𝑒𝑠𝑡} − i_𝑝 ∥^2$.
<!-- #endregion -->

```python id="etxT15BbqMOp"
class Trainer:
    def __init__(self, data_loaders, itemnum, parameter):
        self.parameter = parameter
        # data loader
        self.train_data_loader = data_loaders[0]
        self.dev_data_loader = data_loaders[1]
        self.test_data_loader = data_loaders[2]
        # parameters
        self.batch_size = parameter['batch_size']
        self.learning_rate = parameter['learning_rate']
        self.epoch = parameter['epoch']
        self.print_epoch = parameter['print_epoch']
        self.eval_epoch = parameter['eval_epoch']
        self.device = parameter['device']

        self.MetaTL = MetaTL(itemnum, parameter)
        self.MetaTL.to(self.device)

        self.optimizer = torch.optim.Adam(self.MetaTL.parameters(), self.learning_rate)

            
    def rank_predict(self, data, x, ranks):
        # query_idx is the idx of positive score
        query_idx = x.shape[0] - 1
        # sort all scores with descending, because more plausible triple has higher score
        _, idx = torch.sort(x, descending=True)
        rank = list(idx.cpu().numpy()).index(query_idx) + 1
        ranks.append(rank)
        # update data
        if rank <= 10:
            data['Hits@10'] += 1
            data['NDCG@10'] += 1 / np.log2(rank + 1)
        if rank <= 5:
            data['Hits@5'] += 1
            data['NDCG@5'] += 1 / np.log2(rank + 1)
        if rank == 1:
            data['Hits@1'] += 1
            data['NDCG@1'] += 1 / np.log2(rank + 1)
        data['MRR'] += 1.0 / rank

    def do_one_step(self, task, iseval=False, curr_rel=''):
        loss, p_score, n_score = 0, 0, 0
        if not iseval:
            self.optimizer.zero_grad()
            p_score, n_score = self.MetaTL(task, iseval, curr_rel)
            y = torch.Tensor([1]).to(self.device)
            loss = self.MetaTL.loss_func(p_score, n_score, y)
            loss.backward()
            self.optimizer.step()
        elif curr_rel != '':
            p_score, n_score = self.MetaTL(task, iseval, curr_rel)
            y = torch.Tensor([1]).to(self.device)
            loss = self.MetaTL.loss_func(p_score, n_score, y)
        return loss, p_score, n_score

    def train(self):
        # initialization
        best_epoch = 0
        best_value = 0
        bad_counts = 0

        # training by epoch
        for e in range(self.epoch):
            # sample one batch from data_loader
            train_task, curr_rel = self.train_data_loader.next_batch()
            loss, _, _ = self.do_one_step(train_task, iseval=False, curr_rel=curr_rel)
            # print the loss on specific epoch
            if e % self.print_epoch == 0:
                loss_num = loss.item()
                print("Epoch: {}\tLoss: {:.4f}".format(e, loss_num))
            # do evaluation on specific epoch
            if e % self.eval_epoch == 0 and e != 0:
                print('Epoch  {} Validating...'.format(e))
                valid_data = self.eval(istest=False, epoch=e)

                print('Epoch  {} Testing...'.format(e))
                test_data = self.eval(istest=True, epoch=e)
                
        print('Finish')

    def eval(self, istest=False, epoch=None):
        self.MetaTL.eval()
        
        self.MetaTL.rel_q_sharing = dict()

        if istest:
            data_loader = self.test_data_loader
        else:
            data_loader = self.dev_data_loader
        data_loader.curr_tri_idx = 0

        # initial return data of validation
        data = {'MRR': 0, 'Hits@1': 0, 'Hits@5': 0, 'Hits@10': 0, 'NDCG@1': 0, 'NDCG@5': 0, 'NDCG@10': 0}
        ranks = []

        t = 0
        temp = dict()
        while True:
            # sample all the eval tasks
            eval_task, curr_rel = data_loader.next_one_on_eval()
            # at the end of sample tasks, a symbol 'EOT' will return
            if eval_task == 'EOT':
                break
            t += 1

            _, p_score, n_score = self.do_one_step(eval_task, iseval=True, curr_rel=curr_rel)

            x = torch.cat([n_score, p_score], 1).squeeze()

            self.rank_predict(data, x, ranks)

            # print current temp data dynamically
            for k in data.keys():
                temp[k] = data[k] / t
            sys.stdout.write("{}\tMRR: {:.3f}\tNDCG@10: {:.3f}\tNDCG@5: {:.3f}\tNDCG@1: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
                t, temp['MRR'], temp['NDCG@10'], temp['NDCG@5'], temp['NDCG@1'], temp['Hits@10'], temp['Hits@5'], temp['Hits@1']))
            sys.stdout.flush()

        # print overall evaluation result and return it
        for k in data.keys():
            data[k] = round(data[k] / t, 3)

        
        if istest:
            print("TEST: \tMRR: {:.3f}\tNDCG@10: {:.3f}\tNDCG@5: {:.3f}\tNDCG@1: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
                    temp['MRR'], temp['NDCG@10'], temp['NDCG@5'], temp['NDCG@1'], temp['Hits@10'], temp['Hits@5'], temp['Hits@1']))
        else:
            print("VALID: \tMRR: {:.3f}\tNDCG@10: {:.3f}\tNDCG@5: {:.3f}\tNDCG@1: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
                    temp['MRR'], temp['NDCG@10'], temp['NDCG@5'], temp['NDCG@1'], temp['Hits@10'], temp['Hits@5'], temp['Hits@1']))

        return data
```

```python colab={"background_save": true, "base_uri": "https://localhost:8080/"} id="HVJ1u8VnmIup" executionInfo={"elapsed": 665242, "status": "ok", "timestamp": 1635841573128, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} outputId="b844e740-b9dd-410c-8df4-e19737dd6a40"
user_train, usernum_train, itemnum, user_input_test, user_test, user_input_valid, user_valid = data_load(params['dataset'], params['K'])    

sampler = WarpSampler(user_train, usernum_train, itemnum, batch_size=params['batch_size'], maxlen=params['K'], n_workers=2)
sampler_test = DataLoader(user_input_test, user_test, itemnum, params)
sampler_valid = DataLoader(user_input_valid, user_valid, itemnum, params)

trainer = Trainer([sampler, sampler_valid, sampler_test], itemnum, params)
trainer.train()

sampler.close()
```

<!-- #region id="Zqr-aRykwP-V" -->
## Citations

Sequential Recommendation for Cold-start Users with Meta Transitional Learning. Jianling Wang, Kaize Ding, James Caverlee. 2021. SIGIR. [https://arxiv.org/abs/2107.06427](https://arxiv.org/abs/2107.06427)
<!-- #endregion -->
