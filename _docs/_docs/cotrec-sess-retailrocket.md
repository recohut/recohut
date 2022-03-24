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

<!-- #region id="F7Ac-gx2fLdU" -->
# Training COTREC Session-based recommendation model on Diginetica, Tmall, and RetailRocket datasets
<!-- #endregion -->

<!-- #region id="LnuDlaMJfO7-" -->
## Executive summary

| | |
| --- | --- |
| Problem | Session-based recommendation targets next-item prediction by exploiting user behaviors within a short time period. Compared with other recommendation paradigms, session-based recommendation suffers more from the problem of data sparsity due to the very limited short-term interactions. Self-supervised learning, which can discover ground-truth samples from the raw data, holds vast potentials to tackle this problem. However, existing self-supervised recommendation models mainly rely on item/segment dropout to augment data, which are not fit for session-based recommendation because the dropout leads to sparser data, creating unserviceable self-supervision signals. |
| Solution | For informative session-based data augmentation, we combine self-supervised learning with co-training, and then develop a framework to enhance session-based recommendation. Technically, we first exploit the session-based graph to augment two views that exhibit the internal and external connectivities of sessions, and then we build two distinct graph encoders over the two views, which recursively leverage the different connectivity information to generate ground-truth samples to supervise each other by contrastive learning. In contrast to the dropout strategy, the proposed self-supervised graph co-training preserves the complete session information and fulfills genuine data augmentation. |
| Dataset | Diginetica, Tmall, and RetailRocket. |
| Preprocessing | We filter out all sessions whose length is 1 and items appearing less than 5 times. Latest data (such as, the data of last week) is set to be test set and previous data is used as training set. Then, we augment and label the training and test datasets by using a sequence splitting method, which generates multiple labeled sequences with the corresponding labels ( [𝑖𝑠,1],𝑖𝑠,2), ( [𝑖𝑠,1,𝑖𝑠,2],𝑖𝑠,3), ..., ( [𝑖𝑠,1,𝑖𝑠,2, ...,𝑖𝑠,𝑚−1],𝑖𝑠,𝑚) for every session 𝑠 = [𝑖𝑠,1,𝑖𝑠,2,𝑖𝑠,3, ...,𝑖𝑠,𝑚]. Note that the label of each sequence is the last click item in it. |
| Metrics | Precision, MRR. |
| Hyperparams | We set the embedding size to 100, the batch size for mini-batch to 100, and the 𝐿2 regularization to 10−5 . All parameters are initialized with the Gaussian Distribution N (0, 0.1). We use Adam with the learning rate of 0.001 to optimize our model. For the number of layers of graph convolution on the three datasets, a three-layer setting achieves the best performance. |
| Models | CO-Training framework for session-based RECommendation (COTREC) |
| Cluster | Python 3.6+, PyTorch, CUDA GPU. |
| Tags | `SessionRecommender`, `SelfSupervisedLearning`, `GraphNeuralNetwork`, `CoTraining` |
| Credits | Xin Xia |
<!-- #endregion -->

<!-- #region id="bKUmaoaae_81" -->
## Process flow

![](https://github.com/RecoHut-Stanzas/S969915/raw/main/images/process_flow.svg)
<!-- #endregion -->

<!-- #region id="LNRmdbRPFLnI" -->
## Codebase
<!-- #endregion -->

<!-- #region id="ofJF8YIrFU5k" -->
### Imports
<!-- #endregion -->

```python id="BL8LJStJETvq"
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from operator import itemgetter
import random
import datetime
import math
import torch
from torch import nn, backends
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.sparse
import time
from numba import jit
import time
import os
import argparse
import pickle
```

<!-- #region id="MeBacCorFWw9" -->
### Dataset
<!-- #endregion -->

```python id="1NBSGWstETqF"
def data_masks(all_sessions, n_node):
    adj = dict()
    for sess in all_sessions:
        for i, item in enumerate(sess):
            if i == len(sess)-1:
                break
            else:
                if sess[i] - 1 not in adj.keys():
                    adj[sess[i]-1] = dict()
                    adj[sess[i]-1][sess[i]-1] = 1
                    adj[sess[i]-1][sess[i+1]-1] = 1
                else:
                    if sess[i+1]-1 not in adj[sess[i]-1].keys():
                        adj[sess[i] - 1][sess[i + 1] - 1] = 1
                    else:
                        adj[sess[i]-1][sess[i+1]-1] += 1
    row, col, data = [], [], []
    for i in adj.keys():
        item = adj[i]
        for j in item.keys():
            row.append(i)
            col.append(j)
            data.append(adj[i][j])
    coo = coo_matrix((data, (row, col)), shape=(n_node, n_node))
    return coo
```

```python id="_9aSNmwTEYi_"
class Data():
    def __init__(self, data, all_train, shuffle=False, n_node=None):
        self.raw = np.asarray(data[0])
        adj = data_masks(all_train, n_node)
        # # print(adj.sum(axis=0))
        self.adjacency = adj.multiply(1.0/adj.sum(axis=0).reshape(1, -1))
        self.n_node = n_node
        self.targets = np.asarray(data[1])
        self.length = len(self.raw)
        self.shuffle = shuffle

    def get_overlap(self, sessions):
        matrix = np.zeros((len(sessions), len(sessions)))
        for i in range(len(sessions)):
            seq_a = set(sessions[i])
            seq_a.discard(0)
            for j in range(i+1, len(sessions)):
                seq_b = set(sessions[j])
                seq_b.discard(0)
                overlap = seq_a.intersection(seq_b)
                ab_set = seq_a | seq_b
                matrix[i][j] = float(len(overlap))/float(len(ab_set))
                matrix[j][i] = matrix[i][j]
        # matrix = self.dropout(matrix, 0.2)
        matrix = matrix + np.diag([1.0]*len(sessions))
        degree = np.sum(np.array(matrix), 1)
        degree = np.diag(1.0/degree)
        return matrix, degree

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.raw = self.raw[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length-batch_size, self.length)
        return slices

    def get_slice(self, index):
        items, num_node = [], []
        inp = self.raw[index]
        for session in inp:
            num_node.append(len(np.nonzero(session)[0]))
        max_n_node = np.max(num_node)
        session_len = []
        reversed_sess_item = []
        mask = []
        # item_set = set()
        for session in inp:
            nonzero_elems = np.nonzero(session)[0]
            # item_set.update(set([t-1 for t in session]))
            session_len.append([len(nonzero_elems)])
            items.append(session + (max_n_node - len(nonzero_elems)) * [0])
            mask.append([1]*len(nonzero_elems) + (max_n_node - len(nonzero_elems)) * [0])
            reversed_sess_item.append(list(reversed(session)) + (max_n_node - len(nonzero_elems)) * [0])
        # item_set = list(item_set)
        # index_list = [item_set.index(a) for a in self.targets[index]-1]
        diff_mask = np.ones(shape=[100, self.n_node]) * (1/(self.n_node - 1))
        for count, value in enumerate(self.targets[index]-1):
            diff_mask[count][value] = 1
        return self.targets[index]-1, session_len,items, reversed_sess_item, mask, diff_mask
```

<!-- #region id="fB8DWdW8FaJp" -->
### Utils
<!-- #endregion -->

```python id="mwdd7VldETkB"
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

<!-- #region id="_LFNK75KFbpn" -->
### Model
<!-- #endregion -->

```python id="2o29iEygEz5g"
class ItemConv(Module):
    def __init__(self, layers, emb_size=100):
        super(ItemConv, self).__init__()
        self.emb_size = emb_size
        self.layers = layers[0]
        self.w_item = {}
        for i in range(self.layers):
            self.w_item['weight_item%d' % (i)] = nn.Linear(self.emb_size, self.emb_size, bias=False)

    def forward(self, adjacency, embedding):
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adjacency.shape
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        item_embeddings = embedding
        item_embedding_layer0 = item_embeddings
        final = [item_embedding_layer0]
        for i in range(self.layers):
            item_embeddings = trans_to_cuda(self.w_item['weight_item%d' % (i)])(item_embeddings)
            item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), item_embeddings)
            final.append(F.normalize(item_embeddings, dim=-1, p=2))
        item_embeddings = np.sum(final, 0)/(self.layers+1)
        return item_embeddings
```

```python id="ragoLFq2EyyE"
class SessConv(Module):
    def __init__(self, layers, batch_size, emb_size=100):
        super(SessConv, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.layers = layers[0]
        self.w_sess = {}
        for i in range(self.layers):
            self.w_sess['weight_sess%d' % (i)] = nn.Linear(self.emb_size, self.emb_size, bias=False)

    def forward(self, item_embedding, D, A, session_item, session_len):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        # zeros = torch.zeros([1,self.emb_size])
        item_embedding = torch.cat([zeros, item_embedding], 0)
        seq_h = []
        for i in torch.arange(len(session_item)):
            seq_h.append(torch.index_select(item_embedding, 0, session_item[i]))
        seq_h1 = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in seq_h]))
        session_emb = torch.div(torch.sum(seq_h1, 1), session_len)
        session = [session_emb]
        DA = torch.mm(D, A).float()
        for i in range(self.layers):
            session_emb = trans_to_cuda(self.w_sess['weight_sess%d' % (i)])(session_emb)
            session_emb = torch.mm(DA, session_emb)
            session.append(F.normalize(session_emb, p=2, dim=-1))
        sess = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in session]))
        session_emb = torch.sum(sess, 0)/(self.layers+1)
        return session_emb
```

```python id="lV-0ePwdExw5"
class COTREC(Module):
    def __init__(self, adjacency, n_node, lr, layers, l2, beta,lam,eps, dataset, emb_size=100, batch_size=100):
        super(COTREC, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.n_node = n_node
        self.dataset = dataset
        self.L2 = l2
        self.lr = lr
        self.layers = layers
        self.beta = beta
        self.lam = lam
        self.eps = eps
        self.K = 10
        self.w_k = 10
        self.num = 5000
        self.adjacency = adjacency
        self.embedding = nn.Embedding(self.n_node, self.emb_size)
        self.pos_len = 200
        if self.dataset == 'retailrocket':
            self.pos_len = 300
        self.pos_embedding = nn.Embedding(self.pos_len, self.emb_size)
        self.ItemGraph = ItemConv(self.layers)
        self.SessGraph = SessConv(self.layers, self.batch_size)
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.emb_size, self.emb_size))
        self.w_2 = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.w_i = nn.Linear(self.emb_size, self.emb_size)
        self.w_s = nn.Linear(self.emb_size, self.emb_size)
        self.glu1 = nn.Linear(self.emb_size, self.emb_size)
        self.glu2 = nn.Linear(self.emb_size, self.emb_size, bias=False)

        self.adv_item = torch.cuda.FloatTensor(self.n_node, self.emb_size).fill_(0).requires_grad_(True)
        self.adv_sess = torch.cuda.FloatTensor(self.n_node, self.emb_size).fill_(0).requires_grad_(True)
        # self.adv_item = torch.zeros(self.n_node, self.emb_size).requires_grad_(True)
        # self.adv_sess = torch.zeros(self.n_node, self.emb_size).requires_grad_(True)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def generate_sess_emb(self, item_embedding, session_item, session_len, reversed_sess_item, mask):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        # zeros = torch.zeros(1, self.emb_size)
        item_embedding = torch.cat([zeros, item_embedding], 0)
        get = lambda i: item_embedding[reversed_sess_item[i]]
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size).fill_(0)
        # seq_h = torch.zeros(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size)
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)
        hs = torch.div(torch.sum(seq_h, 1), session_len)
        mask = mask.float().unsqueeze(-1)
        len = seq_h.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(self.batch_size, 1, 1)

        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, seq_h], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * seq_h, 1)
        return select

    def generate_sess_emb_npos(self, item_embedding, session_item, session_len, reversed_sess_item, mask):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        # zeros = torch.zeros(1, self.emb_size)
        item_embedding = torch.cat([zeros, item_embedding], 0)
        get = lambda i: item_embedding[reversed_sess_item[i]]
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size).fill_(0)
        # seq_h = torch.zeros(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size)
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)
        hs = torch.div(torch.sum(seq_h, 1), session_len)
        mask = mask.float().unsqueeze(-1)
        len = seq_h.shape[1]

        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.sigmoid(self.glu1(seq_h) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * seq_h, 1)
        return select

    def example_predicting(self, item_emb, sess_emb):
        x_u = torch.matmul(item_emb, sess_emb)
        pos = torch.softmax(x_u, 0)
        return pos

    def adversarial_item(self, item_emb, tar,sess_emb):
        adv_item_emb = item_emb + self.adv_item
        score = torch.mm(sess_emb, torch.transpose(adv_item_emb, 1, 0))
        loss = self.loss_function(score, tar)
        grad = torch.autograd.grad(loss, self.adv_item,retain_graph=True)[0]
        adv = grad.detach()
        self.adv_item = (F.normalize(adv, p=2,dim=1) * self.eps).requires_grad_(True)

    def adversarial_sess(self, item_emb, tar,sess_emb):
        adv_item_emb = item_emb + self.adv_sess
        score = torch.mm(sess_emb, torch.transpose(adv_item_emb, 1, 0))
        loss = self.loss_function(score, tar)
        grad = torch.autograd.grad(loss, self.adv_sess,retain_graph=True)[0]
        adv = grad.detach()
        self.adv_sess = (F.normalize(adv, p=2,dim=1) * self.eps).requires_grad_(True)

    def diff(self, score_item, score_sess, score_adv2, score_adv1, diff_mask):
        # compute KL(score_item, score_adv2), KL(score_sess, score_adv1)
        score_item = F.softmax(score_item, dim=1)
        score_sess = F.softmax(score_sess, dim=1)
        score_adv2 = F.softmax(score_adv2, dim=1)
        score_adv1 = F.softmax(score_adv1, dim=1)
        score_item = torch.mul(score_item, diff_mask)
        score_sess = torch.mul(score_sess, diff_mask)
        score_adv1 = torch.mul(score_adv1, diff_mask)
        score_adv2 = torch.mul(score_adv2, diff_mask)

        h1 = torch.sum(torch.mul(score_item, torch.log(1e-8 + ((score_item + 1e-8)/(score_adv2 + 1e-8)))))
        h2 = torch.sum(torch.mul(score_sess, torch.log(1e-8 + ((score_sess + 1e-8)/(score_adv1 + 1e-8)))))

        return h1+h2

    def SSL_topk(self, anchor, sess_emb, pos, neg):
        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 2)

        anchor = F.normalize(anchor + sess_emb, p=2, dim=-1)
        pos = torch.reshape(pos, (self.batch_size, self.K, self.emb_size)) + sess_emb.unsqueeze(1).repeat(1, self.K, 1)
        neg = torch.reshape(neg, (self.batch_size, self.K, self.emb_size)) + sess_emb.unsqueeze(1).repeat(1, self.K, 1)
        pos_score = score(anchor.unsqueeze(1).repeat(1, self.K, 1), F.normalize(pos, p=2, dim=-1))
        neg_score = score(anchor.unsqueeze(1).repeat(1, self.K, 1), F.normalize(neg, p=2, dim=-1))
        pos_score = torch.sum(torch.exp(pos_score / 0.2), 1)
        neg_score = torch.sum(torch.exp(neg_score / 0.2), 1)
        con_loss = -torch.sum(torch.log(pos_score / (pos_score + neg_score)))
        return con_loss

    def topk_func_random(self, score1,score2, item_emb_I, item_emb_S):
        values, pos_ind_I = score1.topk(self.num, dim=0, largest=True, sorted=True)
        values, pos_ind_S = score2.topk(self.num, dim=0, largest=True, sorted=True)
        pos_emb_I = torch.cuda.FloatTensor(self.K, self.batch_size, self.emb_size).fill_(0)
        pos_emb_S = torch.cuda.FloatTensor(self.K, self.batch_size, self.emb_size).fill_(0)
        neg_emb_I = torch.cuda.FloatTensor(self.K, self.batch_size, self.emb_size).fill_(0)
        neg_emb_S = torch.cuda.FloatTensor(self.K, self.batch_size, self.emb_size).fill_(0)
        for i in torch.arange(self.K):
            pos_emb_S[i] = item_emb_S[pos_ind_I[i]]
            pos_emb_I[i] = item_emb_I[pos_ind_S[i]]
        random_slices = torch.randint(self.K, self.num, (self.K,))  # choose negative items
        for i in torch.arange(self.K):
            neg_emb_S[i] = item_emb_S[pos_ind_I[random_slices[i]]]
            neg_emb_I[i] = item_emb_I[pos_ind_S[random_slices[i]]]
        return pos_emb_I, neg_emb_I, pos_emb_S, neg_emb_S

    def forward(self, session_item, session_len, D, A, reversed_sess_item, mask, epoch, tar, train, diff_mask):
        if train:
            item_embeddings_i = self.ItemGraph(self.adjacency, self.embedding.weight)
            if self.dataset == 'Tmall':
                # for Tmall dataset, we do not use position embedding to learn temporal order
                sess_emb_i = self.generate_sess_emb_npos(item_embeddings_i, session_item, session_len,reversed_sess_item, mask)
            else:
                sess_emb_i = self.generate_sess_emb(item_embeddings_i, session_item, session_len, reversed_sess_item, mask)
            sess_emb_i = self.w_k * F.normalize(sess_emb_i, dim=-1, p=2)
            item_embeddings_i = F.normalize(item_embeddings_i, dim=-1, p=2)
            scores_item = torch.mm(sess_emb_i, torch.transpose(item_embeddings_i, 1, 0))
            loss_item = self.loss_function(scores_item, tar)

            sess_emb_s = self.SessGraph(self.embedding.weight, D, A, session_item, session_len)
            scores_sess = torch.mm(sess_emb_s, torch.transpose(item_embeddings_i, 1, 0))
            # compute probability of items to be positive examples
            pos_prob_I = self.example_predicting(item_embeddings_i, sess_emb_i)
            pos_prob_S = self.example_predicting(self.embedding.weight, sess_emb_s)

            # choose top-10 items as positive samples and randomly choose 10 items as negative and get their embedding
            pos_emb_I, neg_emb_I, pos_emb_S, neg_emb_S = self.topk_func_random(pos_prob_I,pos_prob_S, item_embeddings_i, self.embedding.weight)

            last_item = torch.squeeze(reversed_sess_item[:, 0])
            last_item = last_item - 1
            last = item_embeddings_i.index_select(0, last_item)
            con_loss = self.SSL_topk(last, sess_emb_i, pos_emb_I, neg_emb_I)
            last = self.embedding(last_item)
            con_loss += self.SSL_topk(last, sess_emb_s, pos_emb_S, neg_emb_S)

            # compute and update adversarial examples
            self.adversarial_item(item_embeddings_i, tar, sess_emb_i)
            self.adversarial_sess(item_embeddings_i, tar, sess_emb_s)

            adv_emb_item = item_embeddings_i + self.adv_item
            adv_emb_sess = item_embeddings_i + self.adv_sess

            score_adv1 = torch.mm(sess_emb_s, torch.transpose(adv_emb_item, 1, 0))
            score_adv2 = torch.mm(sess_emb_i, torch.transpose(adv_emb_sess, 1, 0))
            # add difference constraint
            loss_diff = self.diff(scores_item, scores_sess, score_adv2, score_adv1, diff_mask)
        else:
            item_embeddings_i = self.ItemGraph(self.adjacency, self.embedding.weight)
            if self.dataset == 'Tmall':
                sess_emb_i = self.generate_sess_emb_npos(item_embeddings_i, session_item, session_len, reversed_sess_item, mask)
            else:
                sess_emb_i = self.generate_sess_emb(item_embeddings_i, session_item, session_len, reversed_sess_item, mask)
            sess_emb_i = self.w_k * F.normalize(sess_emb_i, dim=-1, p=2)
            item_embeddings_i = F.normalize(item_embeddings_i, dim=-1, p=2)
            scores_item = torch.mm(sess_emb_i, torch.transpose(item_embeddings_i, 1, 0))
            loss_item = self.loss_function(scores_item, tar)
            loss_diff = 0
            con_loss = 0
        return self.beta * con_loss, loss_item, scores_item, loss_diff*self.lam
```

```python id="F4RYYnkVEvjH"
def forward(model, i, data, epoch, train):
    tar, session_len, session_item, reversed_sess_item, mask, diff_mask = data.get_slice(i)
    diff_mask = trans_to_cuda(torch.Tensor(diff_mask).long())
    A_hat, D_hat = data.get_overlap(session_item)
    session_item = trans_to_cuda(torch.Tensor(session_item).long())
    session_len = trans_to_cuda(torch.Tensor(session_len).long())
    A_hat = trans_to_cuda(torch.Tensor(A_hat))
    D_hat = trans_to_cuda(torch.Tensor(D_hat))
    tar = trans_to_cuda(torch.Tensor(tar).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    reversed_sess_item = trans_to_cuda(torch.Tensor(reversed_sess_item).long())
    con_loss, loss_item, scores_item, loss_diff = model(session_item, session_len, D_hat, A_hat, reversed_sess_item, mask, epoch,tar, train, diff_mask)
    return tar, scores_item, con_loss, loss_item, loss_diff
```

```python id="aR_vVZ6tEuoL"
@jit(nopython=True)
def find_k_largest(K, candidates):
    n_candidates = []
    for iid,score in enumerate(candidates[:K]):
        n_candidates.append((iid, score))
    n_candidates.sort(key=lambda d: d[1], reverse=True)
    k_largest_scores = [item[1] for item in n_candidates]
    ids = [item[0] for item in n_candidates]
    # find the N biggest scores
    for iid,score in enumerate(candidates):
        ind = K
        l = 0
        r = K - 1
        if k_largest_scores[r] < score:
            while r >= l:
                mid = int((r - l) / 2) + l
                if k_largest_scores[mid] >= score:
                    l = mid + 1
                elif k_largest_scores[mid] < score:
                    r = mid - 1
                if r < l:
                    ind = r
                    break
        # move the items backwards
        if ind < K - 2:
            k_largest_scores[ind + 2:] = k_largest_scores[ind + 1:-1]
            ids[ind + 2:] = ids[ind + 1:-1]
        if ind < K - 1:
            k_largest_scores[ind + 1] = score
            ids[ind + 1] = iid
    return ids#,k_largest_scores
```

```python id="rGN4s_UzEtKp"
def train_test(model, train_data, test_data, epoch):
    print('start training: ', datetime.datetime.now())
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i in slices:
        model.zero_grad()
        tar, scores_item, con_loss, loss_item, loss_diff = forward(model, i, train_data, epoch, train=True)
        loss = loss_item + con_loss + loss_diff
        loss.backward()
        model.optimizer.step()
        total_loss += loss.item()
    print('\tLoss:\t%.3f' % total_loss)
    top_K = [5, 10, 20]
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
    print('start predicting: ', datetime.datetime.now())

    model.eval()
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        tar,scores_item, con_loss, loss_item, loss_diff = forward(model, i, test_data, epoch, train=False)
        scores = trans_to_cpu(scores_item).detach().numpy()
        index = []
        for idd in range(model.batch_size):
            index.append(find_k_largest(20, scores[idd]))
        index = np.array(index)
        tar = trans_to_cpu(tar).detach().numpy()
        for K in top_K:
            for prediction, target in zip(index[:, :K], tar):
                metrics['hit%d' % K].append(np.isin(target, prediction))
                if len(np.where(prediction == target)[0]) == 0:
                    metrics['mrr%d' % K].append(0)
                else:
                    metrics['mrr%d' % K].append(1 / (np.where(prediction == target)[0][0] + 1))
    return metrics, total_loss
```

<!-- #region id="DQIMDzHsFQ4y" -->
## Experiment
<!-- #endregion -->

<!-- #region id="lFs2goSZFTK3" -->
### Config
<!-- #endregion -->

```python id="N4K7VTjpFkyy"
class Args:
    def __init__(self, dataset='tmall'):
        self.dataset = dataset
        self.epoch = 30
        self.batchSize = 100
        self.embSize = 100
        self.l2 = 1e-5 # l2 penalty
        self.lr = 0.001 # learning rate
        self.layer = 2, # he number of layer used
        self.beta = 0.005 # ssl task maginitude
        self.lam = 0.005 # diff task maginitude
        self.eps = 0.2 # eps
        if self.dataset == 'diginetica':
            self.n_node = 43097
            self.eps = 0.5
        elif self.dataset == 'tmall':
            self.n_node = 40727
        elif self.dataset == 'retailrocket':
            self.n_node = 36968
```

```python colab={"base_uri": "https://localhost:8080/"} id="jO3LB3xrE6VD" executionInfo={"status": "ok", "timestamp": 1639321562209, "user_tz": -330, "elapsed": 6489, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="eded1e1f-70c3-4ba1-8ba7-e847daca009d"
opt = Args()

opt.dataset = input('Dataset options=tmall/diginetica/retail_rocket, default=tmall:') or 'tmall'
opt.epoch = int(input('Training epochs type=int, default=30:') or '30')
```

<!-- #region id="dZOWaIHMHlaE" -->
### Data acquisition
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="pXIZ-LNXE6SM" executionInfo={"status": "ok", "timestamp": 1639321568933, "user_tz": -330, "elapsed": 1889, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f7c431f7-aa68-4549-c1b7-a166d7170bf7"
if opt.dataset == 'tmall':
    !git clone --branch v1 https://github.com/RecoHut-Datasets/tmall.git
elif opt.dataset == 'diginetica':
    !git clone --branch v2 https://github.com/RecoHut-Datasets/diginetica.git
elif opt.dataset == 'retail_rocket':
    !git clone --branch v1 https://github.com/RecoHut-Datasets/retail_rocket.git
else:
    raise ValueError()
```

<!-- #region id="F4OC5AzQHjhZ" -->
### Main
<!-- #endregion -->

```python id="AQ_fUnI7E6Pd"
def main():
    train_data = pickle.load(open(opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open(opt.dataset + '/test.txt', 'rb'))
    all_train = pickle.load(open(opt.dataset + '/all_train_seq.txt', 'rb'))

    train_data = Data(train_data,all_train, shuffle=True, n_node=opt.n_node)
    test_data = Data(test_data,all_train, shuffle=True, n_node=opt.n_node)
    model = trans_to_cuda(COTREC(adjacency=train_data.adjacency,n_node=opt.n_node,lr=opt.lr, l2=opt.l2, beta=opt.beta,lam= opt.lam,eps=opt.eps,layers=opt.layer,emb_size=opt.embSize, batch_size=opt.batchSize,dataset=opt.dataset))
    top_K = [5, 10, 20]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0]
        best_results['metric%d' % K] = [0, 0]

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        metrics, total_loss = train_test(model, train_data, test_data, epoch)
        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
        print(metrics)
        for K in top_K:
            print('train_loss:\t%.4f\tRecall@%d: %.4f\tMRR%d: %.4f\tEpoch: %d,  %d' %
                  (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],
                   best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))
```

<!-- #region id="sedn_4vAHiQn" -->
### Run
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="kCwmqwv1E6MW" executionInfo={"status": "ok", "timestamp": 1639325397177, "user_tz": -330, "elapsed": 3484549, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8897f524-16ef-444f-89fa-3e6748769765"
if __name__ == '__main__':
    main()
```

<!-- #region id="mVXR1idQaHda" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="jf-FC4UBaKyL" executionInfo={"status": "ok", "timestamp": 1639325418944, "user_tz": -330, "elapsed": 719, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f2ea18f0-8a53-4bec-c06a-a4a22097d65a"
!nvidia-smi
```

```python id="sl-MH_X4aHdb" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1639325433456, "user_tz": -330, "elapsed": 3935, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fba3eb4b-8b63-4c81-dd18-8b07b0089d4e"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="YfczFAHcaHdc" -->
---
<!-- #endregion -->

<!-- #region id="LN23OxNzaHdc" -->
**END**
<!-- #endregion -->
