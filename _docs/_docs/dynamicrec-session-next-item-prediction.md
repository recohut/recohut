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

<!-- #region id="McmC9DgmRZmb" -->
# Next-item Prediction using DynamicRec
> Improve accuracy of in-session recommenders using dynamic convolutional layers to train next-item recommender model

- toc: true
- badges: true
- comments: true
- categories: [Convolution, Session, Sequential, PyTorch, LastFM]
- image:
<!-- #endregion -->

<!-- #region id="tS7CPBiVNt9P" -->
### Introduction
<!-- #endregion -->

<!-- #region id="St7OFOMYNwyu" -->
| |  |
| :-: | -:|
| Vision | Improve accuracy of in-session recommenders |
| Mission | Use dynamic convolutional layers to train next-item recommender model  |
| Scope | Model training, Offline evaluation, Single pre-processed dataset |
| Task | Next-item Prediction |
| Data | LastFM, YooChoose, NowPlaying, Diginetica |
| Tool | PyTorch, Colab, Python |
| Technique | DynamicRec: A Dynamic Convolutional Network for Next Item Recommendation |
| Process | 1) Load pre-processed dataset, 2) Prepare Utils and Model layers, 3) Train the model, 4) Evaluate |
| Takeaway | Convolution-based NN models are lightweight and estimate the sequence very well. |
| Credit | [Mehrab Tanjim](https://github.com/Mehrab-Tanjim) |
| Link | [link1](https://github.com/Mehrab-Tanjim/DynamicRec), [link2](http://cseweb.ucsd.edu/~gary/pubs/mehrab-cikm-2020.pdf) |
<!-- #endregion -->

<!-- #region id="w5evpwmoKlNx" -->
### Libraries
<!-- #endregion -->

```python id="JfuKmHl2rlnD"
import os
import csv
import math
import pickle
import random
import argparse
import numpy as np
import pandas as pd
from random import shuffle
from torch.utils import data
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')
```

<!-- #region id="oq7ofxqBKjfI" -->
### Datasets

There are 4 datasets, already preprocessed:
<!-- #endregion -->

<!-- #region id="4kERU80hG-az" -->
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="snEV0RHWG9cD" outputId="57bf2ae0-f919-47aa-82a3-fbf648e114a5"
#hide-output
!wget https://github.com/recohut/reco-data/raw/master/yoochoose/v3/yoochoose.csv
!wget https://github.com/recohut/reco-data/raw/master/lastfm/v3/last_fm.csv
!wget https://github.com/recohut/reco-data/raw/master/nowplaying/v3/nowplaying.csv
!wget https://github.com/recohut/reco-data/raw/master/diginetica/v3/diginetica.csv
```

<!-- #region id="l69JlXTf-_sz" -->
### PyTorch Dataset
<!-- #endregion -->

```python id="_dzgeo-l4w1z"
class Dataset(data.Dataset):

    'Characterizes a dataset for PyTorch'
    def __init__(self, data, args, itemnum, train):
            'Initialization'
            self.data = data
            self.args = args
            self.itemnum = itemnum
            self.train = train

    def __len__(self):
            'Denotes the total number of samples'            
            return len(self.data)

    def __train__(self, index):
            
            session = np.asarray(self.data[index], dtype=np.int64)
    
            if len(session) > self.args.maxlen:
                session = session[-self.args.maxlen:]
            else:
                session = np.pad(session, (self.args.maxlen-len(session), 0), 'constant', constant_values=0)

            curr_seq = session[:-1]
            curr_pos = session[1:]

            return curr_seq, curr_pos
    
    def __test__(self, index):

            session = self.data[index]

            seq = np.zeros([self.args.maxlen], dtype=np.int64)
            idx = self.args.maxlen - 1

            for i in reversed(session[:-1]): #everything except the last one
                seq[idx] = i
                idx -= 1
                if idx == -1: break

            return seq, session[-1]-1 #index of the item in the list of all items

    def __getitem__(self, index):
            'Generates one sample of data'
            # Select sample

            if self.train:
                return self.__train__(index)
            else:
                return self.__test__(index)
```

<!-- #region id="5k2tChVz_CT_" -->
### Train/Val/Test Split
<!-- #endregion -->

```python id="iKHA9LQA96fP"
def data_partition(fname, percentage=[0.1, 0.2]):
    itemnum = 0

    sessions = defaultdict(list)
    session_train = []
    session_valid = []
    session_test = []
    # assume user/item index starting from 1
    session_id = 0
    f = open(fname, 'r')
    total_length = 0
    max_length = 0
    for line in f:

        items = [int(l) for l in line.rstrip().split(',')]

        if len(items) < 5: continue
        total_length += len(items)

        if max_length< len(items):
            max_length = len(items)
        
        itemnum = max(max(items), itemnum)
        sessions[session_id].append(items)
        session_id += 1

    print("Avg length:", total_length/session_id)
    print("Maximum length:", max_length)

    valid_perc = percentage[0]
    test_perc = percentage[1]

    total_sessions = session_id
    
    shuffle_indices = np.random.permutation(range(total_sessions)) #
    
    train_index = int(total_sessions*(1 - valid_perc - test_perc))
    valid_index = int(total_sessions*(1 - test_perc))

    if (train_index == valid_index): valid_index += 1 #break the tie
    
    train_indices = shuffle_indices[:train_index]
    valid_indices = shuffle_indices[train_index:valid_index]
    test_indices = shuffle_indices[valid_index:]

    for i in train_indices:
        session_train.extend(sessions[i])
    for i in valid_indices:
        session_valid.extend(sessions[i])
    for i in test_indices:
        session_test.extend(sessions[i])
    
    return [np.asarray(session_train), np.asarray(session_valid), np.asarray(session_test), itemnum]
```

<!-- #region id="B9vi0-Lw_F-B" -->
### Next Item Format
<!-- #endregion -->

```python id="l1CfzG-V-A0q"
def saveAsNextItNetFormat(fname, maxlen):
        
    sessions = []

    # assume user/item index starting from 1
    f = open(fname, 'r')

    for line in f:

        items = [int(l) for l in line.rstrip().split(',')]

        if len(items) < 5: continue
        
        seq = np.zeros([maxlen], dtype=np.int32)
        
        idx = maxlen - 1

        for i in reversed(items):
            seq[idx] = i
            idx -= 1
            if idx == -1: break        
        
        sessions.append(seq)
        
    print("number of session:", len(sessions))

    with open(fname+'_nextitnet_format.csv',"w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(sessions)
```

<!-- #region id="uBuGE875_IeC" -->
### GRU Format
<!-- #endregion -->

```python id="zI051wme_USU"
def saveAsGRUFormat(fname, user_train, user_valid, user_test):
    
    session_id = 0
    train = []
    for session in user_train:
        for item in session:
            train.append([session_id, item, 0])
        session_id += 1

    valid = []
    for session in user_valid:
        for item in session:
            valid.append([session_id, item, 0])
        session_id += 1

    test = []
    for session in user_test:
        for item in session:
            test.append([session_id, item, 0])
        session_id += 1

    train_data = pd.DataFrame(train, columns= ['SessionId', 'ItemId', 'Time'])
    valid_data = pd.DataFrame(valid, columns= ['SessionId', 'ItemId', 'Time'])
    test_data = pd.DataFrame(test, columns= ['SessionId', 'ItemId', 'Time'])

    train_data.to_csv(fname+'_grurec_train_data.csv',  sep=' ', index=None)
    valid_data.to_csv(fname+'_grurec_valid_data.csv',  sep=' ', index=None)
    test_data.to_csv(fname+'_grurec_test_data.csv',  sep=' ', index=None)
```

<!-- #region id="1nd-5uvt_YzI" -->
### PyTorch Evaluation Function
<!-- #endregion -->

```python id="kbEwiy6b4npz"
def evaluate(model, test_sessions, itemnum, args, num_workers=4):
    #set the environment
    model.eval()
    
    MRR = 0.0
    NDCG = 0.0
    HT = 0.0

    MRR_plus_10 = 0.0
    NDCG_plus_10 = 0.0
    HT_plus_10 = 0.0

    valid_sessions = 0.0


    all_items = np.array(range(1, itemnum+1))
    all_items_tensor = torch.LongTensor(all_items).to(args.computing_device, non_blocking=True)

    dataset = Dataset(test_sessions, args, itemnum, False)

    sampler = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=num_workers, pin_memory=True)

    with torch.no_grad():
            
        for step, (seq, grouth_truth) in tqdm(enumerate(sampler), total=len(sampler)): 

            seq = torch.LongTensor(seq).to(args.computing_device, non_blocking=True)
            
            _, rank_20 = model.forward(seq, test_item = all_items_tensor)

            rank_20 = rank_20.cpu().detach().numpy()
            grouth_truth = grouth_truth.view(-1, 1).numpy()
            

            try:
                ranks = np.where(rank_20 == grouth_truth)

                try:
                    ranks = ranks[1]
                except:
                    ranks = ranks[0]

                for rank in ranks:

                    if rank < args.top_k:
                        
                        MRR += 1.0/(rank + 1)
                        NDCG += 1 / np.log2(rank + 2)
                        HT += 1

                    if rank < args.top_k + 10:

                        MRR_plus_10 += 1.0/(rank + 1)
                        NDCG_plus_10 += 1 / np.log2(rank + 2)
                        HT_plus_10 += 1
                
            except:
                continue #where rank returns none
                    
        valid_sessions = len(dataset)

    return MRR / valid_sessions, NDCG / valid_sessions, HT / valid_sessions, MRR_plus_10 / valid_sessions, NDCG_plus_10 / valid_sessions, HT_plus_10 / valid_sessions
```

<!-- #region id="_NARw7gevBbP" -->
### Embeddings

Know more - https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
<!-- #endregion -->

```python id="qe8K8E7br4VJ"
def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m
```

```python colab={"base_uri": "https://localhost:8080/"} id="KWUs2SaBr5I0" outputId="b179b2a1-26b2-4c77-eb06-27eebb798e0b"
# initialize embeddings of length 3 for 5 objects
_x = Embedding(num_embeddings=5, embedding_dim=3, padding_idx=0)

# and show these embeddings
input = torch.LongTensor([[0,1,2,3,4]])

_x(input)
```

<!-- #region id="BaADNl1-vdUP" -->
### Layer Normalization

Know more - https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
<!-- #endregion -->

```python id="kkdmS6DStAyE"
def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):
    if torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm
            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)
```

```python colab={"base_uri": "https://localhost:8080/"} id="AHFrSIiBvkQ9" outputId="6d92891c-489c-47f1-b36d-274967f1a0d0"
input = torch.randn(2, 2, 3)
input
```

```python colab={"base_uri": "https://localhost:8080/"} id="zEUKkdUsxOKh" outputId="5873e3b1-934e-43f9-9fc2-1873dc642e1f"
_x = LayerNorm(normalized_shape=[2, 3], eps=1e-5, elementwise_affine=True)

_x(input)
```

<!-- #region id="429P3_7oxwzd" -->
### Linear Layer
<!-- #endregion -->

```python id="rqyelLR_xt9m"
def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m
```

```python colab={"base_uri": "https://localhost:8080/"} id="sDedJnnLyPRA" outputId="3f781c7e-05af-4aa3-eeb8-ef8ffab3e88a"
input = torch.randn(6, 5)
input
```

```python colab={"base_uri": "https://localhost:8080/"} id="56uptF5fxziz" outputId="0f52b753-c10e-47a0-f44e-cdcf2bd79303"
_x = Linear(in_features=5, out_features=2, bias=True)

_x(input)
```

<!-- #region id="4BvJn6quzjxI" -->
### Tensor Unfolding
<!-- #endregion -->

```python id="E_GEpynxyvW3"
def unfold1d(x, kernel_size, padding_l, pad_value=0):
    '''unfold T x B x C to T x B x C x K'''
    if kernel_size > 1:
        T, B, C = x.size()
        x = F.pad(x, (0, 0, 0, 0, padding_l, kernel_size - 1 - padding_l), value=pad_value)
        x = x.as_strided((T, B, C, kernel_size), (B*C, C, 1, B*C))
    else:
        x = x.unsqueeze(3)
    return x
```

```python colab={"base_uri": "https://localhost:8080/"} id="qaiy6emsyv5a" outputId="d51716c0-e5a2-4cac-a555-60bd7d5cffcc"
input = torch.randn(3, 2, 2)
input
```

```python colab={"base_uri": "https://localhost:8080/"} id="LZAeli7_zE03" outputId="bc6158ed-52c4-4f20-8a17-20e9041f5ad2"
unfold1d(input, kernel_size=2, padding_l=0, pad_value=0).shape
```

<!-- #region id="SSshuNcKz339" -->
### Dynamic Convolution

Know more - https://fairseq.readthedocs.io/en/latest/modules.html?highlight=DynamicConv1dTBC#fairseq.modules.DynamicConv1dTBC

Other references:-
- https://arxiv.org/abs/1912.03458
- https://youtu.be/FNkY7I2R_zM
- [Medium blog post](https://medium.com/visionwizard/dynamic-convolution-an-exciting-innovation-over-convolution-kernels-from-microsoft-research-f3cb433cd780)
<!-- #endregion -->

```python id="0j54wgV4rkBW"
class DynamicConv1dTBC(nn.Module):

    '''
    # Copyright (c) Facebook, Inc. and its affiliates.
    #
    # This source code is licensed under the MIT license found in the
    # LICENSE file in the root directory of this source tree: https://github.com/pytorch/fairseq
    # 
    # Dynamic lightweight convolution taking T x B x C inputs
    Args:
        input_size: # of channels of the input
        kernel_size: convolution channels
        padding_l: padding to the left when using "same" padding
        num_heads: number of heads used. The weight is of shape (num_heads, 1, kernel_size)
        weight_dropout: the drop rate of the DropConnect to drop the weight
        weight_softmax: normalize the weight with softmax before the convolution
        renorm_padding: re-normalize the filters to ignore the padded part (only the non-padding parts sum up to 1)
        bias: use bias
        conv_bias: bias of the convolution
        query_size: specified when feeding a different input as the query
        in_proj: project the input and generate the filter together
    Shape:
        Input: TxBxC, i.e. (timesteps, batch_size, input_size)
        Output: TxBxC, i.e. (timesteps, batch_size, input_size)
    Attributes:
        weight: the learnable weights of the module of shape
            `(num_heads, 1, kernel_size)`
        bias:   the learnable bias of the module of shape `(input_size)`
    '''
    def __init__(self, input_size, kernel_size=1, padding_l=None, num_heads=1, unfold=False,
                 weight_dropout=0., weight_softmax=False,
                 renorm_padding=False, bias=False, conv_bias=False,
                 query_size=None):
        super().__init__()
        self.input_size = input_size
        self.query_size = input_size if query_size is None else query_size
        self.kernel_size = kernel_size
        self.padding_l = padding_l
        self.num_heads = num_heads
        self.weight_dropout = weight_dropout
        self.weight_softmax = weight_softmax
        self.renorm_padding = renorm_padding
        self.unfold = unfold
        self.weight_linear = Linear(self.query_size, num_heads * kernel_size * 1, bias=bias) 

        if conv_bias:
            self.conv_bias = nn.Parameter(torch.Tensor(input_size))
        else:
            self.conv_bias = None
        self.reset_parameters()

    @property
    def in_proj(self):
        return self.weight_linear.out_features == self.input_size + self.num_heads * self.kernel_size

    def reset_parameters(self):
        self.weight_linear.reset_parameters()
        if self.conv_bias is not None:
            nn.init.constant_(self.conv_bias, 0.)

    def forward(self, x, incremental_state=None, query=None):
        '''Assuming the input, x, of the shape T x B x C and producing an output in the shape T x B x C
        args:
            x: Input of shape T x B x C, i.e. (timesteps, batch_size, input_size)
            incremental_state: A dict to keep the state
            unfold: unfold the input or not. If not, we use the matrix trick instead
            query: use the specified query to predict the conv filters
        '''
    
        query = x

        output = self._forward_unfolded(x, incremental_state, query)
        

        if self.conv_bias is not None:
            output = output + self.conv_bias.view(1, 1, -1)

        return output

    def _forward_unfolded(self, x, incremental_state, query):
        '''The conventional implementation of convolutions.
        Unfolding the input by having a window shifting to the right.'''
        T, B, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H   
        
        assert R * H == C == self.input_size

        
        weight = self.weight_linear(query).view(T*B*H, -1) 
        
        weight = F.dropout(weight, self.weight_dropout, training=self.training, inplace=False)


        padding_l = self.padding_l
        if K > T and padding_l == K-1:
            weight = weight.narrow(1, K-T, T)
            K, padding_l = T, T-1
        # unfold the input: T x B x C --> T' x B x C x K
        x_unfold = unfold1d(x, K, padding_l, 0)
        x_unfold = x_unfold.view(T*B*H, R, K)

        if self.weight_softmax:
            weight = F.softmax(weight, dim=1)
            
        weight = weight.narrow(1, 0, K)
      
        output = torch.bmm(x_unfold, weight.unsqueeze(2)) # T*B*H x R x 1
        output = output.view(T, B, C)
        return output
```

```python colab={"base_uri": "https://localhost:8080/"} id="VnvHsz3G10-F" outputId="75908af9-e35a-4151-b709-c315bee2167d"
# Input: TxBxC, i.e. (timesteps, batch_size, input_size)
input = torch.randn(3, 2, 3)
input
```

```python id="IdB0IVw62KTk"
_x = DynamicConv1dTBC(input_size=3)
```

```python colab={"base_uri": "https://localhost:8080/"} id="qP6NdWdr2V8m" outputId="37c68d21-3fb5-4ecf-f915-dc8f766a98c6"
# Output: TxBxC, i.e. (timesteps, batch_size, input_size)
_x.forward(input)
```

<!-- #region id="VzKdqIu93VyJ" -->
### ConvRec Layer
<!-- #endregion -->

```python id="_df6sjcK2inA"
class ConvRecLayer(nn.Module):
   

    def __init__(self, args,  kernel_size=0):
        super().__init__()
        self.embed_dim = args.embed_dim
        
        self.conv = DynamicConv1dTBC(args.embed_dim, kernel_size, padding_l=kernel_size-1,
                                         weight_softmax=args.weight_softmax,
                                         num_heads=args.heads,
                                         unfold = None, 
                                         weight_dropout=args.weight_dropout)
        
        self.dropout = args.dropout
        self.layer_norm = LayerNorm(self.embed_dim)

        self.fc1 = Linear(self.embed_dim, args.ffn_embed_dim)
        self.fc2 = Linear(args.ffn_embed_dim, self.embed_dim)


    def forward(self, x, conv_mask=None,
                conv_padding_mask=None):
        
        T, B, C = x.size()
        
        x = self.conv(x)
        x = self.layer_norm(x)  
        attn = None
        residual = x
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x += residual
        return x, attn
```

<!-- #region id="ChdJAWOf3rXJ" -->
### ConvRec Model
<!-- #endregion -->

<!-- #region id="9UrdRPjPLZdd" -->
**Abstract**

*Recently convolutional networks have shown significant promise for modeling sequential user interactions for recommendations. Critically, such networks rely on fixed convolutional kernels to capture sequential behavior. In this paper, we argue that all the dynamics of the item-to-item transition in session-based settings may not be observable at training time. Hence we propose DynamicRec, which uses dynamic convolutions to compute the convolutional kernels on the fly based on the current input. We show through experiments that this approach significantly outperforms existing convolutional models on real datasets in session-based settings.*
<!-- #endregion -->

<!-- #region id="u9EGx0IvLQIO" -->
<!-- #endregion -->

```python id="GZubaj6d3TXQ"
class ConvRec(nn.Module):
    
    def __init__(self, args, itemnum):
        super(ConvRec, self).__init__()
        
        add_args(args)

        self.args = args
        self.dropout = args.dropout
        self.maxlen = args.maxlen
        self.itemnum = itemnum

        self.item_embedding = Embedding(itemnum + 1, args.embed_dim, 0)
        self.embed_scale = math.sqrt(args.embed_dim)  
        self.position_encoding = Embedding(args.maxlen, args.embed_dim, 0)
        
        self.layers = nn.ModuleList([])
        self.layers.extend([
            ConvRecLayer(args, kernel_size=args.decoder_kernel_size_list[i])
            for i in range(args.layers)
        ])
        
        self.layer_norm = LayerNorm(args.embed_dim)

    def forward(self, seq, pos=None, neg=None, test_item = None):
        
 
        x =  self.item_embedding(seq)
        

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        
        attn = None

        inner_states = [x]

        # decoder layers
        for layer in self.layers:
            x = self.layer_norm(x)
            x, attn = layer(x)
            inner_states.append(x)

        # if self.normalize:
        x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        
        seq_emb = x.contiguous().view(-1, x.size(-1)) # reshaping it to [arg.batch_size x args.maxlen * args.hidden_units]
        pos_logits = None 
        neg_logits = None 
        rank_20 = None 
        istarget = None
        loss = None

        if pos is not None:
            pos = torch.reshape(pos, (-1,))

            nnz = torch.ne(pos, 0).nonzero().squeeze(-1)
            neg = torch.randint(1,self.itemnum+1, (self.args.num_neg_samples, nnz.size(0)), device=self.args.computing_device)

            pos_emb = self.item_embedding(pos[nnz])
            neg_emb = self.item_embedding(neg)
            seq_emb = seq_emb[nnz]

            #sequential context
            pos_logits = torch.sum(pos_emb * seq_emb, -1)
            neg_logits = torch.sum(neg_emb * seq_emb, -1)
            negative_scores = torch.sum((1 - torch.sigmoid(neg_logits) + 1e-24).log(), axis = 0)

            loss = torch.sum(-(torch.sigmoid(pos_logits) + 1e-24).log() - negative_scores)/nnz.size(0)

                
        if test_item is not None:        

            test_item_emb = self.item_embedding(test_item)
            seq_emb = seq_emb.view(seq.size(0), seq.size(1), -1)
            seq_emb = seq_emb[:, -1, :]
            seq_emb = seq_emb.contiguous().view(-1, seq_emb.size(-1))
            test_logits = torch.mm(seq_emb, test_item_emb.t()) #check

            test_logits_indices = torch.argsort(-test_logits)
            rank_20 = test_logits_indices[:, :20]

        return loss, rank_20
```

```python id="1TnzynLt4O4C"
def add_args(args):

    if len(args.decoder_kernel_size_list) == 1: # For safety in case kernel size list does not match with # of convolution layers
        args.decoder_kernel_size_list = args.decoder_kernel_size_list * args.layers

    args.weight_softmax = True

    print("Model arguments", args)
```

<!-- #region id="KIoVC3u4H80i" -->
### Argument Parsing
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="cRpgBeDWIGab" outputId="0220839a-a636-4d21-9975-9edb3da1d0da"
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='last_fm.csv')
parser.add_argument('--top_k', default=10, type=int)

parser.add_argument('--train_dir', default='default')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)

parser.add_argument('--maxlen', default=30, type=int)

parser.add_argument('--embed_dim', default=200, type=int) 
parser.add_argument('--ffn_embed_dim', default=200, type=int)
parser.add_argument('--dropout', default=0.2, type=float) 
parser.add_argument('--weight_dropout', default=0.2, type=float)


parser.add_argument('--layers', default=2, type=int) 
parser.add_argument('--heads', default=1, type=int) 

parser.add_argument('--decoder_kernel_size_list', default = [5, 5]) #depends on the number of layer

parser.add_argument('--num_epochs', default=30, type=int)
parser.add_argument('--num_neg_samples', default = 400, type=int) #Note: 100 is sufficient
parser.add_argument('--eval_epoch', default = 5, type=int)
```

```python colab={"base_uri": "https://localhost:8080/"} id="iFf72706IPwJ" outputId="e75ed772-af18-4c3a-e7e1-9c06751a2b89"
# Check if your system supports CUDA
use_cuda = torch.cuda.is_available()

# Setup GPU optimization if CUDA is supported
if use_cuda:
    computing_device = torch.device("cuda")
    extras = {"num_workers": 1, "pin_memory": True}
    print("CUDA is supported")
else:  # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    extras = False
    print("CUDA NOT supported")

parser.add_argument('--computing_device', default=computing_device)

# # Get the arguments
try:
    #if running from command line
    args = parser.parse_args(args={})
except:
    #if running in IDEs
    args = parser.parse_known_args(args={})[0] 
```

```python id="2cVPDFJ7IbXc"
result_path = 'results/'+args.dataset + '_' + args.train_dir

if not os.path.isdir(result_path):
    os.makedirs(result_path)
with open(os.path.join(result_path, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()
```

<!-- #region id="AIrSkKlLIdsL" -->
### Loading the Data
<!-- #endregion -->

```python id="mO9paKbzIn3W"
!mkdir -p ./data && mv ./*.csv ./data
```

```python colab={"base_uri": "https://localhost:8080/"} id="PATB3-q0ImSN" outputId="ff21c6f9-1cea-47aa-9d1c-82a17cb6db51"
if os.path.exists("data/"+args.dataset + '.pkl'):
    pickle_in = open("data/"+args.dataset+".pkl","rb")
    dataset = pickle.load(pickle_in)
else:
    dataset = data_partition("data/"+args.dataset)
    pickle_out = open("data/"+args.dataset+".pkl","wb")
    pickle.dump(dataset, pickle_out)
    pickle_out.close()
```

```python colab={"base_uri": "https://localhost:8080/"} id="cxZsko3qJeuz" outputId="6dcbdbe3-6911-482e-d2da-e0355706c9bf"
[train, valid, test, itemnum] = dataset

print("Number of sessions:",len(train)+len(valid)+len(test))
print("Number of items:", itemnum)

action = 0
for i in train:
    action += np.count_nonzero(i)

for i in valid:
    action += np.count_nonzero(i)


for i in test:
    action += np.count_nonzero(i)

print("Number of actions:", action)

print("Average length of sessions:", action/(len(train)+len(valid)+len(test)))


num_batch = len(train) // args.batch_size
print("The batch size is:", num_batch)
```

<!-- #region id="FyaoYvpFKGwl" -->
### Loading the Pre-trained Model if exist
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="oJKyvQQHJn_d" outputId="ab336048-d276-4c04-a94b-1ac3c5d70182"
f = open(os.path.join(result_path, 'log.txt'), 'w')

conv_model = ConvRec(args, itemnum)
conv_model = conv_model.to(args.computing_device, non_blocking=True)

# Note: testing a pretrained model
if os.path.exists(result_path+"pretrained_model.pth"):
    conv_model.load_state_dict(torch.load(result_path+"pretrained_model.pth"))       
    t_test = evaluate(conv_model, test, itemnum, args, num_workers=4)
    model_performance = "Model performance on test: "+str(t_test)
    print(model_performance)


optimizer = optim.Adam(conv_model.parameters(), lr = args.lr, betas=(0.9, 0.98), weight_decay = 0.0)

f.write(str(args)+'\n')
f.flush()
```

<!-- #region id="uogNIBk5KAdg" -->
### Training the Model
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="_hbHw3_AJwlJ" outputId="2dd0c2bc-de2f-489f-ae6d-0637f336d531"
best_val_loss = 1e6
train_losses = []
val_losses = []

best_ndcg = 0
best_hit = 0
model_performance = None

stop_count = 0
total_epochs = 1    

dataset = Dataset(train, args, itemnum, train=True)

  
sampler = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)

for epoch in range(1, args.num_epochs + 1):  
    conv_model.train()

    epoch_losses = []

    
    for step, (seq, pos) in tqdm(enumerate(sampler), total=len(sampler)):  
            
        optimizer.zero_grad()


        seq = torch.LongTensor(seq).to(args.computing_device, non_blocking=True)
        pos = torch.LongTensor(pos).to(args.computing_device, non_blocking=True)

        loss, _  = conv_model.forward(seq, pos=pos)

        epoch_losses.append(loss.item())

        # Compute gradients
        loss.backward()

        # Update the parameters
        optimizer.step()
        
    
    if total_epochs % args.eval_epoch == 0:

        t_valid = evaluate(conv_model, valid, itemnum, args, num_workers=4)
        
        print ('\nnum of steps:%d, valid (MRR@%d: %.4f, NDCG@%d: %.4f, HR@%d: %.4f), valid (MRR@%d: %.4f, NDCG@%d: %.4f, HR@%d: %.4f)' % (total_epochs, args.top_k, t_valid[0], args.top_k, t_valid[1], args.top_k, t_valid[2],
        args.top_k+10, t_valid[3], args.top_k+10, t_valid[4], args.top_k+10, t_valid[5]))

        f.write(str(t_valid) + '\n')
        f.flush()

        
        if t_valid[0]>best_ndcg:
            best_ndcg = t_valid[0]
            torch.save(conv_model.state_dict(), result_path+"pretrained_model.pth")
            stop_count = 1
        else:
            stop_count += 1

        if stop_count == 3: #model did not improve 3 consequetive times
            break
            
    total_epochs += 1

    train_loss = np.mean(epoch_losses)
    print(str(epoch) + "epoch loss", train_loss)
```

<!-- #region id="CFBnYNhCJ30c" -->
### Loading and Evaluating the Trained model
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="SioKI-E-G5XU" outputId="bbe6839a-7f7b-4eef-8f95-7ae1b9cd82bf"
conv_model = ConvRec(args, itemnum)
conv_model.load_state_dict(torch.load(result_path+"pretrained_model.pth"))

conv_model = conv_model.to(args.computing_device)
    
t_test = evaluate(conv_model, test, itemnum, args, num_workers=4)

model_performance = "Model performance on test: "+str(t_test)
print(model_performance)

f.write(model_performance+'\n')
f.flush()
f.close()

print("Done")
```

<!-- #region id="MQVdrmmQL7tr" -->
### Performance
<!-- #endregion -->

<!-- #region id="QJmRRZNiL-QY" -->
<!-- #endregion -->

<!-- #region id="Q0LiaC0uLFan" -->
### References
1. https://github.com/Mehrab-Tanjim/DynamicRec
2. http://cseweb.ucsd.edu/~gary/pubs/mehrab-cikm-2020.pdf
<!-- #endregion -->
