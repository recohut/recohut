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

<!-- #region id="cRez1G9Reb0i" -->
# FGNN for Session Recommendation on Sample Data
<!-- #endregion -->

<!-- #region id="rKne4VhzjmJD" -->
## Setup
<!-- #endregion -->

```python id="scBUqujvfEPO"
!pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
!pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
!pip install torch-geometric
```

```python id="ITI1SE8ye9Wh"
import pickle
import collections
import numpy as np
import logging
import math
import os
import argparse
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, GATConv, GatedGraphConv, SAGEConv
from torch_geometric.data import InMemoryDataset, Data
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros
```

<!-- #region id="ZIwOhkJ2jiNR" -->
## Dataset
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="IG0uROICgcV0" executionInfo={"status": "ok", "timestamp": 1638288064931, "user_tz": -330, "elapsed": 10708, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fb1341f5-6684-4a89-b301-5ad9e2fc3bbc"
!mkdir -p raw
%cd raw
!wget -q --show-progress -O sample_data_preprocessing.ipynb https://gist.githubusercontent.com/sparsh-ai/12d1f5ca07add606f27b0f841b550a82/raw/4e6521f0633e3bb20f601e04eb9fbc87390231b8/t182546-preprocessing-of-sample-data-for-session-based-recommendations.ipynb
%run sample_data_preprocessing.ipynb
%cd /content
```

```python id="hlX_QPG9fSRa"
class MultiSessionsGraph(InMemoryDataset):
    """Every session is a graph."""
    def __init__(self, root, phrase, transform=None, pre_transform=None):
        assert phrase in ['train', 'test']
        self.phrase = phrase
        super(MultiSessionsGraph, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
     
    @property
    def raw_file_names(self):
        return [self.phrase + '.txt']
    
    @property
    def processed_file_names(self):
        return [self.phrase + '.pt']
    
    def download(self):
        pass
    
    def process(self):
        data = pickle.load(open(self.raw_dir + '/' + self.raw_file_names[0], 'rb'))
        data_list = []
        
        for sequence, y in zip(data[0], data[1]):
            # sequence = [1, 3, 2, 2, 1, 3, 4]
            i = 0
            nodes = {}    # dict{15: 0, 16: 1, 18: 2, ...}
            senders = []
            x = []
            for node in sequence:
                if node not in nodes:
                    nodes[node] = i
                    x.append([node])
                    i += 1
                senders.append(nodes[node])
            receivers = senders[:]
            
            if len(senders) != 1:
                del senders[-1]    # the last item is a receiver
                del receivers[0]    # the first item is a sender

            # undirected
            # senders, receivers = senders + receivers, receivers + senders

            pair = {}
            sur_senders = senders[:]
            sur_receivers = receivers[:]
            i = 0
            for sender, receiver in zip(sur_senders, sur_receivers):
                if str(sender)+'-'+str(receiver) in pair:
                    pair[str(sender)+'-'+str(receiver)] += 1
                    del senders[i]
                    del receivers[i]
                else:
                    pair[str(sender)+'-'+str(receiver)] = 1
                    i += 1

            count = collections.Counter(senders)
            out_degree_inv = torch.tensor([1/count[i] for i in senders], dtype=torch.float)

            count = collections.Counter(receivers)
            in_degree_inv = torch.tensor([1 / count[i] for i in receivers], dtype=torch.float)
            
            edge_attr = torch.tensor([pair[str(senders[i])+'-'+str(receivers[i])] for i in range(len(senders))],
                                     dtype=torch.float)

            edge_index = torch.tensor([senders, receivers], dtype=torch.long)
            x = torch.tensor(x, dtype=torch.long)
            y = torch.tensor([y], dtype=torch.long)
            sequence = torch.tensor(sequence, dtype=torch.long)
            sequence_len = torch.tensor([len(sequence)], dtype=torch.long)
            session_graph = Data(x=x, y=y,
                                 edge_index=edge_index, edge_attr=edge_attr,
                                 sequence=sequence, sequence_len=sequence_len,
                                 out_degree_inv=out_degree_inv, in_degree_inv=in_degree_inv)
            data_list.append(session_graph)
            
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
```

<!-- #region id="YwzLkwhKj0ZQ" -->
## Modules
<!-- #endregion -->

```python id="cFwq1_GdfcIk"
class GRUSet2Set(torch.nn.Module):
    r"""The global pooling operator based on iterative content-based attention
    from the `"Order Matters: Sequence to sequence for sets"
    <https://arxiv.org/abs/1511.06391>`_ paper
    .. math::
        \mathbf{q}_t &= \mathrm{LSTM}(\mathbf{q}^{*}_{t-1})
        \alpha_{i,t} &= \mathrm{softmax}(\mathbf{x}_i \cdot \mathbf{q}_t)
        \mathbf{r}_t &= \sum_{i=1}^N \alpha_{i,t} \mathbf{x}_i
        \mathbf{q}^{*}_t &= \mathbf{q}_t \, \Vert \, \mathbf{r}_t,
    where :math:`\mathbf{q}^{*}_T` defines the output of the layer with twice
    the dimensionality as the input.
    Args:
        in_channels (int): Size of each input sample.
        processing_steps (int): Number of iterations :math:`T`.
        num_layers (int, optional): Number of recurrent layers, *.e.g*, setting
            :obj:`num_layers=2` would mean stacking two LSTMs together to form
            a stacked LSTM, with the second LSTM taking in outputs of the first
            LSTM and computing the final results. (default: :obj:`1`)
    """

    def __init__(self, in_channels, processing_steps, num_layers=1):
        super(GRUSet2Set, self).__init__()

        self.in_channels = in_channels
        self.out_channels = 2 * in_channels
        self.processing_steps = processing_steps
        self.num_layers = num_layers

        self.rnn = nn.GRU(self.out_channels, self.in_channels,
                          num_layers)
        self.linear = nn.Linear(in_channels * 3, in_channels)

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.rnn.parameters():
            if len(weight.size()) > 1:
                torch.nn.init.orthogonal_(weight.data)

    def forward(self, x, batch):
        """"""
        batch_size = batch.max().item() + 1

        h = (x.new_zeros((self.num_layers, batch_size, self.in_channels)),
             x.new_zeros((self.num_layers, batch_size, self.in_channels)))
        q_star = x.new_zeros(batch_size, self.out_channels)

        sections = torch.bincount(batch)
        v_i = torch.split(x, tuple(sections.cpu().numpy()))  # split whole x back into graphs G_i
        v_n_repeat = tuple(nodes[-1].view(1, -1).repeat(nodes.shape[0], 1) for nodes in
                           v_i)  # repeat |V|_i times for the last node embedding
        
        # x = x * v_n_repeat
        
        for i in range(self.processing_steps):
            if i == 0:
                q, h = self.rnn(q_star.unsqueeze(0))
            else:
                q, h = self.rnn(q_star.unsqueeze(0), h)
            q = q.view(batch_size, self.in_channels)

            # e = self.linear(torch.cat((x, q[batch], torch.cat(v_n_repeat, dim=0)), dim=-1)).sum(dim=-1, keepdim=True)
            e = (x * q[batch]).sum(dim=-1, keepdim=True)
            a = softmax(e, batch, num_nodes=batch_size)
            r = scatter_add(a * x, batch, dim=0, dim_size=batch_size)
            q_star = torch.cat([q, r], dim=-1)

        return q_star

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
```

```python id="6AhbGQJJfKz5"
class WeightedGATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},
    where the attention coefficients :math:`\alpha_{i,j}` are computed as
    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
        attentions are averaged instead of concatenated. (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True,
                 weighted=True):
        super(WeightedGATConv, self).__init__('add')

        self.weighted = weighted
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(
            torch.Tensor(in_channels, heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels + 1))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr=None):
        """"""
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr)

        x = torch.mm(x, self.weight).view(-1, self.heads, self.out_channels)
        return self.propagate(edge_index, x=x, num_nodes=edge_index.max() + 1, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_index, num_nodes, edge_attr):
        # Compute attention coefficients.
        if edge_attr is not None:
            # alpha = ((torch.cat([x_i, x_j], dim=-1) * self.att) * edge_attr.view(-1, 1, 1)).sum(dim=-1)
            alpha = (torch.cat([x_i, x_j, edge_attr.view(-1, 1).repeat(1, x_i.shape[1]).view(-1, x_i.shape[1], 1)],
                               dim=-1) * self.att).sum(dim=-1)
        else:
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0], num_nodes)

        # Sample attention coefficients stochastically.
        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
```

```python id="xTHP0D0dfsWe"
class Embedding2Score(nn.Module):
    def __init__(self, hidden_size):
        super(Embedding2Score, self).__init__()
        self.hidden_size = hidden_size
        self.W_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.q = nn.Linear(self.hidden_size, 1)
        self.W_3 = nn.Linear(2 * self.hidden_size, self.hidden_size)

    def forward(self, session_embedding, all_item_embedding, batch):
        sections = torch.bincount(batch)
        v_i = torch.split(session_embedding, tuple(sections.cpu().numpy()))    # split whole x back into graphs G_i
        v_n_repeat = tuple(nodes[-1].view(1, -1).repeat(nodes.shape[0], 1) for nodes in v_i)    # repeat |V|_i times for the last node embedding

        # Eq(6)
        alpha = self.q(torch.sigmoid(self.W_1(torch.cat(v_n_repeat, dim=0)) + self.W_2(session_embedding)))    # |V|_i * 1
        s_g_whole = alpha * session_embedding    # |V|_i * hidden_size
        s_g_split = torch.split(s_g_whole, tuple(sections.cpu().numpy()))    # split whole s_g into graphs G_i
        s_g = tuple(torch.sum(embeddings, dim=0).view(1, -1) for embeddings in s_g_split)
        
        # Eq(7)
        v_n = tuple(nodes[-1].view(1, -1) for nodes in v_i)
        s_h = self.W_3(torch.cat((torch.cat(v_n, dim=0), torch.cat(s_g, dim=0)), dim=1))
        
        # Eq(8)
        z_i_hat = torch.mm(s_h, all_item_embedding.weight.transpose(1, 0))
        
        return z_i_hat


class GNNModel(nn.Module):
    """
    Args:
        hidden_size: the number of units in a hidden layer.
        n_node: the number of items in the whole item set for embedding layer.
    """
    def __init__(self, hidden_size, n_node):
        super(GNNModel, self).__init__()
        self.hidden_size, self.n_node = hidden_size, n_node
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gat1 = GATConv(self.hidden_size, self.hidden_size, heads=8, negative_slope=0.2)
        self.gat2 = GATConv(8 * self.hidden_size, self.hidden_size, heads=1, negative_slope=0.2)
        self.sage1 = SAGEConv(self.hidden_size, self.hidden_size)
        self.sage2 = SAGEConv(self.hidden_size, self.hidden_size)
        self.gated = GatedGraphConv(self.hidden_size, num_layers=2)
        self.e2s = Embedding2Score(self.hidden_size)
        self.loss_function = nn.CrossEntropyLoss()
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x - 1, data.edge_index, data.batch, data.edge_attr

        embedding = self.embedding(x).squeeze()
        hidden = self.gated(embedding, edge_index)#, edge_attr.float())
        #
        # hidden = F.relu(self.gat1(embedding, edge_index))
        # hidden = self.gat2(hidden, edge_index)
        
        # hidden1 = F.relu(self.sage1(embedding, edge_index))
        # hidden2 = F.relu(self.sage2(hidden1, edge_index))
        return self.e2s(hidden, self.embedding, batch)
```

```python id="AkknSFG8fhf8"
def forward(model, loader, device, writer, epoch, top_k=20, optimizer=None, train_flag=True):
    if train_flag:
        model.train()
    else:
        model.eval()
        hit, mrr = [], []

    mean_loss = 0.0
    updates_per_epoch = len(loader)

    for i, batch in enumerate(loader):
        if train_flag:
            optimizer.zero_grad()
        scores = model(batch.to(device))
        targets = batch.y - 1
        loss = model.loss_function(scores, targets)

        if train_flag:
            loss.backward()
            # nn.utils.clip_grad_norm(model.parameters(), 0.5)
            optimizer.step()
            writer.add_scalar('loss/train_batch_loss', loss.item(), epoch * updates_per_epoch + i)
        else:
            sub_scores = scores.topk(top_k)[1]    # batch * top_k
            for score, target in zip(sub_scores.detach().cpu().numpy(), targets.detach().cpu().numpy()):
                hit.append(np.isin(target, score))
                if len(np.where(score == target)[0]) == 0:
                    mrr.append(0)
                else:
                    mrr.append(1 / (np.where(score == target)[0][0] + 1))

        mean_loss += loss / batch.num_graphs

    if train_flag:
        writer.add_scalar('loss/train_loss', mean_loss.item(), epoch)
    else:
        writer.add_scalar('loss/test_loss', mean_loss.item(), epoch)
        hit = np.mean(hit) * 100
        mrr = np.mean(mrr) * 100
        writer.add_scalar('index/hit', hit, epoch)
        writer.add_scalar('index/mrr', mrr, epoch)
```

<!-- #region id="JpiwWBZXjgX4" -->
## Main
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 34} id="iVIoR-5Jf1zX" executionInfo={"status": "ok", "timestamp": 1638288075473, "user_tz": -330, "elapsed": 1296, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="812a24e9-25d3-4be9-f579-7e54086a246f"
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name')
parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=10, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--top_k', type=int, default=20, help='top K indicator for evaluation')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
opt = parser.parse_args(args={})
logging.warning(opt)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 471} id="rkgJxnqif695" executionInfo={"status": "ok", "timestamp": 1638288102032, "user_tz": -330, "elapsed": 7364, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1156f2d3-8cb2-489c-8dc6-5394b28598d1"
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cur_dir = os.getcwd()
    train_dataset = MultiSessionsGraph('.', phrase='train')
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_dataset = MultiSessionsGraph('.', phrase='test')
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

    log_dir = './logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.warning('logging to {}'.format(log_dir))
    writer = SummaryWriter(log_dir)

    n_node = 309

    model = GNNModel(hidden_size=opt.hidden_size, n_node=n_node).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    # optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, weight_decay=opt.l2, momentum=opt.momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

    logging.warning(model)
    
    for epoch in tqdm(range(opt.epoch)):
        scheduler.step()
        forward(model, train_loader, device, writer, epoch, top_k=opt.top_k, optimizer=optimizer, train_flag=True)
        with torch.no_grad():
            forward(model, test_loader, device, writer, epoch, top_k=opt.top_k, train_flag=False)


if __name__ == '__main__':
    main()
```

<!-- #region id="ow2QwwhjlP-8" -->
---
<!-- #endregion -->

```python id="cgQpjxIrlP-8"
!apt-get -qq install tree
!rm -r sample_data
```

```python colab={"base_uri": "https://localhost:8080/"} id="jXMEsvGFlP-9" executionInfo={"status": "ok", "timestamp": 1638288415068, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b4b66849-c7d6-4fca-c17f-99ae4de69abf"
!tree -h --du .
```

```python colab={"base_uri": "https://localhost:8080/"} id="YefCxtvJlP-9" executionInfo={"status": "ok", "timestamp": 1638288437035, "user_tz": -330, "elapsed": 3530, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8cc2d129-9cab-49b7-b955-6601c0376c05"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="qER4Raj6lP--" -->
---
<!-- #endregion -->

<!-- #region id="Nolfb57llP_A" -->
**END**
<!-- #endregion -->
