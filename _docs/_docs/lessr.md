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

<!-- #region id="f2oOCVn5Hq2k" -->
# LESSR Session-based Recommendations on Sample data in PyTorch
<!-- #endregion -->

<!-- #region id="Ha4vDCQEHYnL" -->
## Setup
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="131W9rq_QwwD" executionInfo={"status": "ok", "timestamp": 1638014333875, "user_tz": -330, "elapsed": 647, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d111077c-a689-4b2d-f0c1-c54e1d979899"
import torch
torch.__version__
```

```python id="bxJibf5NKaZ0"
!pip install dgl-cu111
```

```python id="5WWtr35DK48_"
class Args:
    dataset_dir = '.'
    embedding_dim = 32
    num_layers = 3
    feat_drop = 0.2
    lr = 1e-3
    batch_size = 512
    epochs = 30
    weight_decay = 1e-4
    Ks = '10,20' # the values of K in evaluation metrics, separated by commas
    patience = 2
    num_workers = 2
    valid_split = None
    log_interval = 100
```

<!-- #region id="FHkHxp1VKmtS" -->
## Dataset
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="_6R5Toa6KOEi" executionInfo={"status": "ok", "timestamp": 1638014176142, "user_tz": -330, "elapsed": 1496, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8ad293d6-d60c-425b-b8c1-2378a937cd44"
!wget -q --show-progress https://github.com/sparsh-ai/stanza/raw/S969796/datasets/sample/num_items.txt
!wget -q --show-progress https://github.com/sparsh-ai/stanza/raw/S969796/datasets/sample/train.txt
!wget -q --show-progress https://github.com/sparsh-ai/stanza/raw/S969796/datasets/sample/test.txt
```

<!-- #region id="3SDfoUOvKnyN" -->
### Collate
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="HgDdcSIHKdjz" executionInfo={"status": "ok", "timestamp": 1638014376657, "user_tz": -330, "elapsed": 717, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a4eac9c3-dee3-4202-a831-da9bafbb34cc"
from collections import Counter
import numpy as np
import torch as th
import dgl


def label_last(g, last_nid):
    is_last = th.zeros(g.number_of_nodes(), dtype=th.int32)
    is_last[last_nid] = 1
    g.ndata['last'] = is_last
    return g


def seq_to_eop_multigraph(seq):
    items = np.unique(seq)
    iid2nid = {iid: i for i, iid in enumerate(items)}
    num_nodes = len(items)

    if len(seq) > 1:
        seq_nid = [iid2nid[iid] for iid in seq]
        src = seq_nid[:-1]
        dst = seq_nid[1:]
    else:
        src = th.LongTensor([])
        dst = th.LongTensor([])
    g = dgl.graph((src, dst), num_nodes=num_nodes)
    g.ndata['iid'] = th.from_numpy(items)
    label_last(g, iid2nid[seq[-1]])
    return g


def seq_to_shortcut_graph(seq):
    items = np.unique(seq)
    iid2nid = {iid: i for i, iid in enumerate(items)}
    num_nodes = len(items)

    seq_nid = [iid2nid[iid] for iid in seq]
    counter = Counter(
        [(seq_nid[i], seq_nid[j]) for i in range(len(seq)) for j in range(i, len(seq))]
    )
    edges = counter.keys()
    src, dst = zip(*edges)

    g = dgl.graph((src, dst), num_nodes=num_nodes)
    return g


def collate_fn_factory(*seq_to_graph_fns):
    def collate_fn(samples):
        seqs, labels = zip(*samples)
        inputs = []
        for seq_to_graph in seq_to_graph_fns:
            graphs = list(map(seq_to_graph, seqs))
            bg = dgl.batch(graphs)
            inputs.append(bg)
        labels = th.LongTensor(labels)
        return inputs, labels

    return collate_fn
```

<!-- #region id="lWg59bP_Kv-1" -->
### Augment
<!-- #endregion -->

```python id="TXYtRBkpMUhE"
import itertools
import numpy as np
import pandas as pd


def create_index(sessions):
    lens = np.fromiter(map(len, sessions), dtype=np.long)
    session_idx = np.repeat(np.arange(len(sessions)), lens - 1)
    label_idx = map(lambda l: range(1, l), lens)
    label_idx = itertools.chain.from_iterable(label_idx)
    label_idx = np.fromiter(label_idx, dtype=np.long)
    idx = np.column_stack((session_idx, label_idx))
    return idx


def read_sessions(filepath):
    sessions = pd.read_csv(filepath, sep='\t', header=None, squeeze=True)
    sessions = sessions.apply(lambda x: list(map(int, x.split(',')))).values
    return sessions


def read_dataset(dataset_dir):
    train_sessions = read_sessions(dataset_dir / 'train.txt')
    test_sessions = read_sessions(dataset_dir / 'test.txt')
    with open(dataset_dir / 'num_items.txt', 'r') as f:
        num_items = int(f.readline())
    return train_sessions, test_sessions, num_items


class AugmentedDataset:
    def __init__(self, sessions, sort_by_length=True):
        self.sessions = sessions
        index = create_index(sessions)  # columns: sessionId, labelIndex
        if sort_by_length:
            # sort by labelIndex in descending order
            ind = np.argsort(index[:, 1])[::-1]
            index = index[ind]
        self.index = index

    def __getitem__(self, idx):
        sid, lidx = self.index[idx]
        seq = self.sessions[sid][:lidx]
        label = self.sessions[sid][lidx]
        return seq, label

    def __len__(self):
        return len(self.index)
```

<!-- #region id="874vbgVdMWL6" -->
## Utils
<!-- #endregion -->

```python id="5LXZkAU2Mbpj"
import time
from collections import defaultdict

import torch as th
from torch import nn, optim


# ignore weight decay for parameters in bias, batch norm and activation
def fix_weight_decay(model):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(map(lambda x: x in name, ['bias', 'batch_norm', 'activation'])):
            no_decay.append(param)
        else:
            decay.append(param)
    params = [{'params': decay}, {'params': no_decay, 'weight_decay': 0}]
    return params


def prepare_batch(batch, device):
    inputs, labels = batch
    inputs_gpu = [x.to(device) for x in inputs]
    labels_gpu = labels.to(device)
    return inputs_gpu, labels_gpu


def evaluate(model, data_loader, device, Ks=[20]):
    model.eval()
    num_samples = 0
    max_K = max(Ks)
    results = defaultdict(float)
    with th.no_grad():
        for batch in data_loader:
            inputs, labels = prepare_batch(batch, device)
            logits = model(*inputs)
            batch_size = logits.size(0)
            num_samples += batch_size
            topk = th.topk(logits, k=max_K, sorted=True)[1]
            labels = labels.unsqueeze(-1)
            for K in Ks:
                hit_ranks = th.where(topk[:, :K] == labels)[1] + 1
                hit_ranks = hit_ranks.float().cpu()
                results[f'HR@{K}'] += hit_ranks.numel()
                results[f'MRR@{K}'] += hit_ranks.reciprocal().sum().item()
                results[f'NDCG@{K}'] += th.log2(1 + hit_ranks).reciprocal().sum().item()
    for metric in results:
        results[metric] /= num_samples
    return results


def print_results(results, epochs=None):
    print('Metric\t' + '\t'.join(results.keys()))
    print(
        'Value\t' +
        '\t'.join([f'{round(val * 100, 2):.2f}' for val in results.values()])
    )
    if epochs is not None:
        print('Epoch\t' + '\t'.join([str(epochs[metric]) for metric in results]))


class TrainRunner:
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        device,
        lr=1e-3,
        weight_decay=0,
        patience=3,
        Ks=[20],
    ):
        self.model = model
        if weight_decay > 0:
            params = fix_weight_decay(model)
        else:
            params = model.parameters()
        self.optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epoch = 0
        self.batch = 0
        self.patience = patience
        self.Ks = Ks

    def train(self, epochs, log_interval=100):
        max_results = defaultdict(float)
        max_epochs = defaultdict(int)
        bad_counter = 0
        t = time.time()
        mean_loss = 0
        for epoch in range(epochs):
            self.model.train()
            for batch in self.train_loader:
                inputs, labels = prepare_batch(batch, self.device)
                self.optimizer.zero_grad()
                logits = self.model(*inputs)
                loss = nn.functional.cross_entropy(logits, labels)
                loss.backward()
                self.optimizer.step()
                mean_loss += loss.item() / log_interval
                if self.batch > 0 and self.batch % log_interval == 0:
                    print(
                        f'Batch {self.batch}: Loss = {mean_loss:.4f}, Time Elapsed = {time.time() - t:.2f}s'
                    )
                    t = time.time()
                    mean_loss = 0
                self.batch += 1

            curr_results = evaluate(
                self.model, self.test_loader, self.device, Ks=self.Ks
            )

            print(f'\nEpoch {self.epoch}:')
            print_results(curr_results)

            any_better_result = False
            for metric in curr_results:
                if curr_results[metric] > max_results[metric]:
                    max_results[metric] = curr_results[metric]
                    max_epochs[metric] = self.epoch
                    any_better_result = True

            if any_better_result:
                bad_counter = 0
            else:
                bad_counter += 1
                if bad_counter == self.patience:
                    break

            self.epoch += 1
        print('\nBest results')
        print_results(max_results, max_epochs)
        return max_results
```

<!-- #region id="FVTTbG64MtwT" -->
## Model
<!-- #endregion -->

```python id="F_AJMToXMtts"
import torch as th
from torch import nn
import dgl
import dgl.ops as F
import dgl.function as fn


class EOPA(nn.Module):
    def __init__(
        self, input_dim, output_dim, batch_norm=True, feat_drop=0.0, activation=None
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.gru = nn.GRU(input_dim, input_dim, batch_first=True)
        self.fc_self = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neigh = nn.Linear(input_dim, output_dim, bias=False)
        self.activation = activation

    def reducer(self, nodes):
        m = nodes.mailbox['m']  # (num_nodes, deg, d)
        # m[i]: the messages passed to the i-th node with in-degree equal to 'deg'
        # the order of messages follows the order of incoming edges
        # since the edges are sorted by occurrence time when the EOP multigraph is built
        # the messages are in the order required by EOPA
        _, hn = self.gru(m)  # hn: (1, num_nodes, d)
        return {'neigh': hn.squeeze(0)}

    def forward(self, mg, feat):
        with mg.local_scope():
            if self.batch_norm is not None:
                feat = self.batch_norm(feat)
            mg.ndata['ft'] = self.feat_drop(feat)
            if mg.number_of_edges() > 0:
                mg.update_all(fn.copy_u('ft', 'm'), self.reducer)
                neigh = mg.ndata['neigh']
                rst = self.fc_self(feat) + self.fc_neigh(neigh)
            else:
                rst = self.fc_self(feat)
            if self.activation is not None:
                rst = self.activation(rst)
            return rst


class SGAT(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        batch_norm=True,
        feat_drop=0.0,
        activation=None,
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_q = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc_k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_v = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_e = nn.Linear(hidden_dim, 1, bias=False)
        self.activation = activation

    def forward(self, sg, feat):
        if self.batch_norm is not None:
            feat = self.batch_norm(feat)
        feat = self.feat_drop(feat)
        q = self.fc_q(feat)
        k = self.fc_k(feat)
        v = self.fc_v(feat)
        e = F.u_add_v(sg, q, k)
        e = self.fc_e(th.sigmoid(e))
        a = F.edge_softmax(sg, e)
        rst = F.u_mul_e_sum(sg, v, a)
        if self.activation is not None:
            rst = self.activation(rst)
        return rst


class AttnReadout(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        batch_norm=True,
        feat_drop=0.0,
        activation=None,
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_u = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_v = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc_e = nn.Linear(hidden_dim, 1, bias=False)
        self.fc_out = (
            nn.Linear(input_dim, output_dim, bias=False)
            if output_dim != input_dim else None
        )
        self.activation = activation

    def forward(self, g, feat, last_nodes):
        if self.batch_norm is not None:
            feat = self.batch_norm(feat)
        feat = self.feat_drop(feat)
        feat_u = self.fc_u(feat)
        feat_v = self.fc_v(feat[last_nodes])
        feat_v = dgl.broadcast_nodes(g, feat_v)
        e = self.fc_e(th.sigmoid(feat_u + feat_v))
        alpha = F.segment.segment_softmax(g.batch_num_nodes(), e)
        feat_norm = feat * alpha
        rst = F.segment.segment_reduce(g.batch_num_nodes(), feat_norm, 'sum')
        if self.fc_out is not None:
            rst = self.fc_out(rst)
        if self.activation is not None:
            rst = self.activation(rst)
        return rst


class LESSR(nn.Module):
    def __init__(
        self, num_items, embedding_dim, num_layers, batch_norm=True, feat_drop=0.0
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_items, embedding_dim, max_norm=1)
        self.indices = nn.Parameter(
            th.arange(num_items, dtype=th.long), requires_grad=False
        )
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        input_dim = embedding_dim
        for i in range(num_layers):
            if i % 2 == 0:
                layer = EOPA(
                    input_dim,
                    embedding_dim,
                    batch_norm=batch_norm,
                    feat_drop=feat_drop,
                    activation=nn.PReLU(embedding_dim),
                )
            else:
                layer = SGAT(
                    input_dim,
                    embedding_dim,
                    embedding_dim,
                    batch_norm=batch_norm,
                    feat_drop=feat_drop,
                    activation=nn.PReLU(embedding_dim),
                )
            input_dim += embedding_dim
            self.layers.append(layer)
        self.readout = AttnReadout(
            input_dim,
            embedding_dim,
            embedding_dim,
            batch_norm=batch_norm,
            feat_drop=feat_drop,
            activation=nn.PReLU(embedding_dim),
        )
        input_dim += embedding_dim
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_sr = nn.Linear(input_dim, embedding_dim, bias=False)

    def forward(self, mg, sg=None):
        iid = mg.ndata['iid']
        feat = self.embedding(iid)
        for i, layer in enumerate(self.layers):
            if i % 2 == 0:
                out = layer(mg, feat)
            else:
                out = layer(sg, feat)
            feat = th.cat([out, feat], dim=1)
        last_nodes = mg.filter_nodes(lambda nodes: nodes.data['last'] == 1)
        sr_g = self.readout(mg, feat, last_nodes)
        sr_l = feat[last_nodes]
        sr = th.cat([sr_l, sr_g], dim=1)
        if self.batch_norm is not None:
            sr = self.batch_norm(sr)
        sr = self.fc_sr(self.feat_drop(sr))
        logits = sr @ self.embedding(self.indices).t()
        return logits
```

<!-- #region id="Udy2JJZtMbnP" -->
## Main
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="krAmCUY7M7Jc" executionInfo={"status": "ok", "timestamp": 1638014955678, "user_tz": -330, "elapsed": 577644, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4bfa0dc4-2a72-413d-f91e-350737bcb252"
from pathlib import Path
import torch as th
from torch.utils.data import DataLoader


args = Args()


dataset_dir = Path(args.dataset_dir)
args.Ks = [int(K) for K in args.Ks.split(',')]
print('reading dataset')
train_sessions, test_sessions, num_items = read_dataset(dataset_dir)


if args.valid_split is not None:
    num_valid = int(len(train_sessions) * args.valid_split)
    test_sessions = train_sessions[-num_valid:]
    train_sessions = train_sessions[:-num_valid]


train_set = AugmentedDataset(train_sessions)
test_set = AugmentedDataset(test_sessions)


if args.num_layers > 1:
    collate_fn = collate_fn_factory(seq_to_eop_multigraph, seq_to_shortcut_graph)
else:
    collate_fn = collate_fn_factory(seq_to_eop_multigraph)


train_loader = DataLoader(
    train_set,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=args.num_workers,
    collate_fn=collate_fn,
)

test_loader = DataLoader(
    test_set,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    collate_fn=collate_fn,
)


model = LESSR(num_items, args.embedding_dim, args.num_layers, feat_drop=args.feat_drop)
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
model = model.to(device)
print(model)


runner = TrainRunner(
    model,
    train_loader,
    test_loader,
    device=device,
    lr=args.lr,
    weight_decay=args.weight_decay,
    patience=args.patience,
    Ks=args.Ks,
)


print('start training')
runner.train(args.epochs, args.log_interval)
```

<!-- #region id="68pdexKKQNuS" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="C10k8bo_QNuT" executionInfo={"status": "ok", "timestamp": 1638014967203, "user_tz": -330, "elapsed": 5705, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a4afb425-6bf4-4edf-860d-20fdec83b7c2"
!apt-get -qq install tree
!rm -r sample_data
```

```python colab={"base_uri": "https://localhost:8080/"} id="TZRs_d-2QNuU" executionInfo={"status": "ok", "timestamp": 1638014967204, "user_tz": -330, "elapsed": 20, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="eac15d60-3331-4c84-959a-bd98e353f3a2"
!tree -h --du .
```

```python colab={"base_uri": "https://localhost:8080/"} id="1CfCN3IZQNuU" executionInfo={"status": "ok", "timestamp": 1638014977422, "user_tz": -330, "elapsed": 3650, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="16693ead-fdbf-49dd-d62a-37e9126bb4dd"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="yspsGY6iQNuV" -->
---
<!-- #endregion -->

<!-- #region id="1RSSPS8eQNuV" -->
**END**
<!-- #endregion -->
