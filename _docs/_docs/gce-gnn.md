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

<!-- #region id="dY--55OUYQ-7" -->
# GCE-GNN Session Recommender on NowPlaying in PyTorch
<!-- #endregion -->

<!-- #region id="vhEnZ48N2FPy" -->
## Setup
<!-- #endregion -->

```python id="sn50Ffkb2Esp"
import numpy as np
import datetime
import math
from tqdm.notebook import tqdm
import pickle
import time

import torch
from torch.utils.data import Dataset
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
```

```python id="p26c4bNm3LOt"
class Args:
    def __init__(self, dataset = 'diginetica'):
        self.dataset = dataset
        self.sample_num = 12
        if dataset == 'diginetica':
            self.num = 43098
            self.n_iter = 2
            self.dropout_gcn = 0.2 # Dropout rate
            self.dropout_local = 0.0 # Dropout rate
        elif dataset == 'tmall':
            self.num = 40728
            self.n_iter = 1
            self.dropout_gcn = 0.6
            self.dropout_local = 0.5
        elif dataset == 'nowplaying':
            self.num = 60417
            self.n_iter = 1
            self.dropout_gcn = 0.0
            self.dropout_local = 0.0
        self.hiddenSize = 100
        self.epoch = 2
        self.activate = 'relu'
        self.n_sample_all = 12
        self.n_sample = 12
        self.batch_size = 100
        self.lr = 0.001 # learning rate
        self.lr_dc = 0.1 # learning rate decay
        self.lr_dc_step = 3 # the number of steps after which the learning rate decay
        self.l2 = 1e-5 # l2 penalty
        self.dropout_global = 0.5 # Dropout rate
        self.validation = True
        self.valid_portion = 0.1 # split the portion
        self.alpha = 0.2 # Alpha for the leaky_relu.
        self.patience = 3
```

```python id="JE9xMI9u3Q9D"
opt = Args(dataset = 'nowplaying')
```

<!-- #region id="hy3oGaMP2H8G" -->
## Dataset
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="hT2umfIz4gOQ" executionInfo={"status": "ok", "timestamp": 1638277603910, "user_tz": -330, "elapsed": 6994, "user": {"displayName": "sparsh agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "00322518567794762549"}} outputId="8e516d15-959d-449e-8789-964ee5f2e342"
!wget -q --show-progress https://github.com/RecoHut-Datasets/nowplaying/raw/v1/all_train_seq.txt
!wget -q --show-progress https://github.com/RecoHut-Datasets/nowplaying/raw/v1/train.txt
!wget -q --show-progress https://github.com/RecoHut-Datasets/nowplaying/raw/v1/test.txt
```

<!-- #region id="t8YR0Rts3AKg" -->
### Build Graph
<!-- #endregion -->

```python id="BClobEbS3AD3"
dataset = opt.dataset
sample_num = opt.sample_num
num = opt.num

seq = pickle.load(open('all_train_seq.txt', 'rb'))

relation = []
neighbor = [] * num

all_test = set()

adj1 = [dict() for _ in range(num)]
adj = [[] for _ in range(num)]

for i in range(len(seq)):
    data = seq[i]
    for k in range(1, 4):
        for j in range(len(data)-k):
            relation.append([data[j], data[j+k]])
            relation.append([data[j+k], data[j]])

for tup in relation:
    if tup[1] in adj1[tup[0]].keys():
        adj1[tup[0]][tup[1]] += 1
    else:
        adj1[tup[0]][tup[1]] = 1

weight = [[] for _ in range(num)]

for t in range(num):
    x = [v for v in sorted(adj1[t].items(), reverse=True, key=lambda x: x[1])]
    adj[t] = [v[0] for v in x]
    weight[t] = [v[1] for v in x]

for i in range(num):
    adj[i] = adj[i][:sample_num]
    weight[i] = weight[i][:sample_num]

pickle.dump(adj, open('adj_' + str(sample_num) + '.pkl', 'wb'))
pickle.dump(weight, open('num_' + str(sample_num) + '.pkl', 'wb'))
```

```python id="fkFzOJHT5Pof"
def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

```python id="R2AvOltD2LXN"
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
```

```python id="XTbsyqEV2PCt"
def handle_data(inputData, train_len=None):
    len_data = [len(nowData) for nowData in inputData]
    if train_len is None:
        max_len = max(len_data)
    else:
        max_len = train_len
    # reverse the sequence
    us_pois = [list(reversed(upois)) + [0] * (max_len - le) if le < max_len else list(reversed(upois[-max_len:]))
               for upois, le in zip(inputData, len_data)]
    us_msks = [[1] * le + [0] * (max_len - le) if le < max_len else [1] * max_len
               for le in len_data]
    return us_pois, us_msks, max_len
```

```python id="MbL8mMda2Qhp"
def handle_adj(adj_dict, n_entity, sample_num, num_dict=None):
    adj_entity = np.zeros([n_entity, sample_num], dtype=np.int64)
    num_entity = np.zeros([n_entity, sample_num], dtype=np.int64)
    for entity in range(1, n_entity):
        neighbor = list(adj_dict[entity])
        neighbor_weight = list(num_dict[entity])
        n_neighbor = len(neighbor)
        if n_neighbor == 0:
            continue
        if n_neighbor >= sample_num:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=True)
        adj_entity[entity] = np.array([neighbor[i] for i in sampled_indices])
        num_entity[entity] = np.array([neighbor_weight[i] for i in sampled_indices])
    return adj_entity, num_entity
```

```python id="FWYw5eoo2USj"
class Data(Dataset):
    def __init__(self, data, train_len=None):
        inputs, mask, max_len = handle_data(data[0], train_len)
        self.inputs = np.asarray(inputs)
        self.targets = np.asarray(data[1])
        self.mask = np.asarray(mask)
        self.length = len(data[0])
        self.max_len = max_len

    def __getitem__(self, index):
        u_input, mask, target = self.inputs[index], self.mask[index], self.targets[index]
        max_n_node = self.max_len
        node = np.unique(u_input)
        items = node.tolist() + (max_n_node - len(node)) * [0]
        adj = np.zeros((max_n_node, max_n_node))
        for i in np.arange(len(u_input) - 1):
            u = np.where(node == u_input[i])[0][0]
            adj[u][u] = 1
            if u_input[i + 1] == 0:
                break
            v = np.where(node == u_input[i + 1])[0][0]
            if u == v or adj[u][v] == 4:
                continue
            adj[v][v] = 1
            if adj[v][u] == 2:
                adj[u][v] = 4
                adj[v][u] = 4
            else:
                adj[u][v] = 2
                adj[v][u] = 3
        alias_inputs = [np.where(node == i)[0][0] for i in u_input]
        
        return [torch.tensor(alias_inputs), torch.tensor(adj), torch.tensor(items),
                torch.tensor(mask), torch.tensor(target), torch.tensor(u_input)]

    def __len__(self):
        return self.length
```

<!-- #region id="Iig4FRMX4t2e" -->
## Aggregator
<!-- #endregion -->

```python id="c-GdN3UO4ty1"
class Aggregator(nn.Module):
    def __init__(self, batch_size, dim, dropout, act, name=None):
        super(Aggregator, self).__init__()
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim

    def forward(self):
        pass
```

```python id="9ZzMV7ov42_o"
class LocalAggregator(nn.Module):
    def __init__(self, dim, alpha, dropout=0., name=None):
        super(LocalAggregator, self).__init__()
        self.dim = dim
        self.dropout = dropout

        self.a_0 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_1 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_3 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.bias = nn.Parameter(torch.Tensor(self.dim))

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, hidden, adj, mask_item=None):
        h = hidden
        batch_size = h.shape[0]
        N = h.shape[1]

        a_input = (h.repeat(1, 1, N).view(batch_size, N * N, self.dim)
                   * h.repeat(1, N, 1)).view(batch_size, N, N, self.dim)

        e_0 = torch.matmul(a_input, self.a_0)
        e_1 = torch.matmul(a_input, self.a_1)
        e_2 = torch.matmul(a_input, self.a_2)
        e_3 = torch.matmul(a_input, self.a_3)

        e_0 = self.leakyrelu(e_0).squeeze(-1).view(batch_size, N, N)
        e_1 = self.leakyrelu(e_1).squeeze(-1).view(batch_size, N, N)
        e_2 = self.leakyrelu(e_2).squeeze(-1).view(batch_size, N, N)
        e_3 = self.leakyrelu(e_3).squeeze(-1).view(batch_size, N, N)

        mask = -9e15 * torch.ones_like(e_0)
        alpha = torch.where(adj.eq(1), e_0, mask)
        alpha = torch.where(adj.eq(2), e_1, alpha)
        alpha = torch.where(adj.eq(3), e_2, alpha)
        alpha = torch.where(adj.eq(4), e_3, alpha)
        alpha = torch.softmax(alpha, dim=-1)

        output = torch.matmul(alpha, h)
        return output
```

```python id="4dLm1v7Y46Vm"
class GlobalAggregator(nn.Module):
    def __init__(self, dim, dropout, act=torch.relu, name=None):
        super(GlobalAggregator, self).__init__()
        self.dropout = dropout
        self.act = act
        self.dim = dim

        self.w_1 = nn.Parameter(torch.Tensor(self.dim + 1, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.w_3 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.bias = nn.Parameter(torch.Tensor(self.dim))

    def forward(self, self_vectors, neighbor_vector, batch_size, masks, neighbor_weight, extra_vector=None):
        if extra_vector is not None:
            alpha = torch.matmul(torch.cat([extra_vector.unsqueeze(2).repeat(1, 1, neighbor_vector.shape[2], 1)*neighbor_vector, neighbor_weight.unsqueeze(-1)], -1), self.w_1).squeeze(-1)
            alpha = F.leaky_relu(alpha, negative_slope=0.2)
            alpha = torch.matmul(alpha, self.w_2).squeeze(-1)
            alpha = torch.softmax(alpha, -1).unsqueeze(-1)
            neighbor_vector = torch.sum(alpha * neighbor_vector, dim=-2)
        else:
            neighbor_vector = torch.mean(neighbor_vector, dim=2)
        # self_vectors = F.dropout(self_vectors, 0.5, training=self.training)
        output = torch.cat([self_vectors, neighbor_vector], -1)
        output = F.dropout(output, self.dropout, training=self.training)
        output = torch.matmul(output, self.w_3)
        output = output.view(batch_size, -1, self.dim)
        output = self.act(output)
        return output
```

<!-- #region id="xAkDD17B2aV5" -->
## Model
<!-- #endregion -->

```python id="0xKI49N-23aa"
class CombineGraph(Module):
    def __init__(self, opt, num_node, adj_all, num):
        super(CombineGraph, self).__init__()
        self.opt = opt

        self.batch_size = opt.batch_size
        self.num_node = num_node
        self.dim = opt.hiddenSize
        self.dropout_local = opt.dropout_local
        self.dropout_global = opt.dropout_global
        self.hop = opt.n_iter
        self.sample_num = opt.n_sample
        self.adj_all = trans_to_cuda(torch.Tensor(adj_all)).long()
        self.num = trans_to_cuda(torch.Tensor(num)).float()

        # Aggregator
        self.local_agg = LocalAggregator(self.dim, self.opt.alpha, dropout=0.0)
        self.global_agg = []
        for i in range(self.hop):
            if opt.activate == 'relu':
                agg = GlobalAggregator(self.dim, opt.dropout_gcn, act=torch.relu)
            else:
                agg = GlobalAggregator(self.dim, opt.dropout_gcn, act=torch.tanh)
            self.add_module('agg_gcn_{}'.format(i), agg)
            self.global_agg.append(agg)

        # Item representation & Position representation
        self.embedding = nn.Embedding(num_node, self.dim)
        self.pos_embedding = nn.Embedding(200, self.dim)

        # Parameters
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.glu1 = nn.Linear(self.dim, self.dim)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)
        self.linear_transform = nn.Linear(self.dim, self.dim, bias=False)

        self.leakyrelu = nn.LeakyReLU(opt.alpha)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def sample(self, target, n_sample):
        # neighbor = self.adj_all[target.view(-1)]
        # index = np.arange(neighbor.shape[1])
        # np.random.shuffle(index)
        # index = index[:n_sample]
        # return self.adj_all[target.view(-1)][:, index], self.num[target.view(-1)][:, index]
        return self.adj_all[target.view(-1)], self.num[target.view(-1)]

    def compute_scores(self, hidden, mask):
        mask = mask.float().unsqueeze(-1)

        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1)

        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(select, b.transpose(1, 0))
        return scores

    def forward(self, inputs, adj, mask_item, item):
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]
        h = self.embedding(inputs)

        # local
        h_local = self.local_agg(h, adj, mask_item)

        # global
        item_neighbors = [inputs]
        weight_neighbors = []
        support_size = seqs_len

        for i in range(1, self.hop + 1):
            item_sample_i, weight_sample_i = self.sample(item_neighbors[-1], self.sample_num)
            support_size *= self.sample_num
            item_neighbors.append(item_sample_i.view(batch_size, support_size))
            weight_neighbors.append(weight_sample_i.view(batch_size, support_size))

        entity_vectors = [self.embedding(i) for i in item_neighbors]
        weight_vectors = weight_neighbors

        session_info = []
        item_emb = self.embedding(item) * mask_item.float().unsqueeze(-1)
        
        # mean 
        sum_item_emb = torch.sum(item_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1)
        
        # sum
        # sum_item_emb = torch.sum(item_emb, 1)
        
        sum_item_emb = sum_item_emb.unsqueeze(-2)
        for i in range(self.hop):
            session_info.append(sum_item_emb.repeat(1, entity_vectors[i].shape[1], 1))

        for n_hop in range(self.hop):
            entity_vectors_next_iter = []
            shape = [batch_size, -1, self.sample_num, self.dim]
            for hop in range(self.hop - n_hop):
                aggregator = self.global_agg[n_hop]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vector=entity_vectors[hop+1].view(shape),
                                    masks=None,
                                    batch_size=batch_size,
                                    neighbor_weight=weight_vectors[hop].view(batch_size, -1, self.sample_num),
                                    extra_vector=session_info[hop])
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        h_global = entity_vectors[0].view(batch_size, seqs_len, self.dim)

        # combine
        h_local = F.dropout(h_local, self.dropout_local, training=self.training)
        h_global = F.dropout(h_global, self.dropout_global, training=self.training)
        output = h_local + h_global

        return output
```

```python id="5RVP050T27xr"
def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable
```

```python id="MuXi-8D626u3"
def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable
```

```python id="5E53SVJQ25xm"
def forward(model, data):
    alias_inputs, adj, items, mask, targets, inputs = data
    alias_inputs = trans_to_cuda(alias_inputs).long()
    items = trans_to_cuda(items).long()
    adj = trans_to_cuda(adj).float()
    mask = trans_to_cuda(mask).long()
    inputs = trans_to_cuda(inputs).long()

    hidden = model(items, adj, mask, inputs)
    get = lambda index: hidden[index][alias_inputs[index]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    return targets, model.compute_scores(seq_hidden, mask)
```

```python id="tajDb_Wy24pf"
def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=2, batch_size=model.batch_size,
                                               shuffle=True, pin_memory=True)
    for data in tqdm(train_loader):
        model.optimizer.zero_grad()
        targets, scores = forward(model, data)
        targets = trans_to_cuda(targets).long()
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=2, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)
    result = []
    hit, mrr = [], []
    for data in test_loader:
        targets, scores = forward(model, data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        targets = targets.numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))

    result.append(np.mean(hit) * 100)
    result.append(np.mean(mrr) * 100)

    return result
```

<!-- #region id="H9RRpEwj23XD" -->
## Main
<!-- #endregion -->

```python id="4hIJPbsv23Uc"
init_seed(2020)
```

```python id="fQADBzqW66dD"
train_data = pickle.load(open('train.txt', 'rb'))
if opt.validation:
    train_data, valid_data = split_validation(train_data, opt.valid_portion)
    test_data = valid_data
else:
    test_data = pickle.load(open('test.txt', 'rb'))

adj = pickle.load(open('adj_' + str(opt.n_sample_all) + '.pkl', 'rb'))
num = pickle.load(open('num_' + str(opt.n_sample_all) + '.pkl', 'rb'))

train_data = Data(train_data)
test_data = Data(test_data)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 446, "referenced_widgets": ["e26c6c12f93840bc886812d4cb1a99af", "06f85218263c43469527fc00d49e6843", "1c436fe4c4d04ae79fe616adba623638", "a61b1a6def6c43f2b1febc0dc127cb31", "914607e9897a44ab808ee9e8f5017d36", "c002ce1ada9d49bbba0ff6c80ead0c53", "624dfb1fd79c439f87f3f1323e250d18", "9f1505b46a224d90a95538492faa6f79", "ee1ae38a2a6342f08b5f60f0e4b5e734", "2dce44bdd2f74c46a1f4ab5f35863f72", "8a110f09c87b490292f67df2f4949df2", "0561f6f1d60e4cc39674f3f6d7df9fb5", "5ee075b8912c447bae2d65ea8feabbf9", "670559f3fc9c4c0b93d9ad8ed498e503", "7a8b2970f29844f994f1bea946f6282a", "9234f7fc0bb14c208325130847e2fedd", "0ce61317d8aa47099408390ed3a1dd43", "178ae75750614b4899adbfbe48491288", "79ed5b5c1f034223838be07691f91db3", "764a8e1c1019416eadabcd1fcda32db7", "b12fa09ce5b54ad8a967ffd7114f7d1a", "0ef8a196c2d04344a4659cdc891c11c7"]} id="HveobxF67avY" executionInfo={"status": "ok", "timestamp": 1638278874762, "user_tz": -330, "elapsed": 1221246, "user": {"displayName": "sparsh agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "00322518567794762549"}} outputId="e6c1da35-cd58-4341-b326-c0a0229b0d59"
adj, num = handle_adj(adj, opt.num, opt.n_sample_all, num)
model = trans_to_cuda(CombineGraph(opt, opt.num, adj, num))

print(opt)

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
    print('Current Result:')
    print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f' % (hit, mrr))
    print('Best Result:')
    print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d' % (
        best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
    bad_counter += 1 - flag
    if bad_counter >= opt.patience:
        break
print('-------------------------------------------------------')
end = time.time()
print("Run time: %f s" % (end - start))
```

<!-- #region id="CbbpODHA8dr8" -->
---
<!-- #endregion -->

```python id="SsNqDkWE8dr_" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1638279047169, "user_tz": -330, "elapsed": 6514, "user": {"displayName": "sparsh agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "00322518567794762549"}} outputId="c0938d6f-af93-4193-8a90-bfacdd1afa4f"
!apt-get -qq install tree
!rm -r sample_data
```

```python id="luP0BjVC8dr_" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1638279048053, "user_tz": -330, "elapsed": 893, "user": {"displayName": "sparsh agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "00322518567794762549"}} outputId="c50fcaa9-9405-4142-819e-58abaf37cf55"
!tree -h --du .
```

```python id="_S3rQXBz8dsA" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1638279493625, "user_tz": -330, "elapsed": 3497, "user": {"displayName": "sparsh agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "00322518567794762549"}} outputId="7fc388ce-7737-44d2-bb99-965e782021ab"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="I0TYBU4w8dsB" -->
---
<!-- #endregion -->

<!-- #region id="SYzSMRCX8dsC" -->
**END**
<!-- #endregion -->
