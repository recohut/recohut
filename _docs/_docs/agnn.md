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

<!-- #region id="gcKmx_vMzTTs" -->
# AGGN Cold-start Recommendation on ML-100k
<!-- #endregion -->

<!-- #region id="aqBXKpEN1VZh" -->
![](https://github.com/recohut/coldstart-recsys/raw/da72950ca514faee94f010a2cb6e99a373044ec1/docs/_images/T290734_1.png)
<!-- #endregion -->

<!-- #region id="DIWL_ukaz3-A" -->
## Introduction
<!-- #endregion -->

<!-- #region id="k6EyiwszzpyN" -->
### Input Layer

We first design an input layer to build the user (item) attribute graph $\mathcal{A}_u$ ($\mathcal{A}_i$). We calculate two kinds of proximity scores between the nodes - preference proximity and attribute proximity (can be calculated with cosine similarity).

- The preference proximity measures the historical preference similarity between two nodes. If two users have similar rating record list (or two items have similar rated record list), they will have a high preference proximity. Note we cannot calculate preference proximity for the cold start nodes as they do not have the historical ratings.
- The attribute proximity measures the similarity between the attributes of two nodes. If two users have similar user profiles, e.g., gender, occupation (or two items have similar properties, e.g., category), they will have a high attribute proximity.

After calculating the overall proximity between two nodes, it becomes a natural choice to build a k-NN graph as adopted in (Monti, Bronstein, and Bresson 2017). Such a method will keep a fixed number of neighbors once the graph is constructed.

### Attribute Interaction Layer

In the constructed attribute graph $\mathcal{A}_u$ and $\mathcal{A}_i$, each nodes has an attached multi-hot attribute encoding and a unique one-hot representation denoting its identity. Due to the huge number of users and items in the web-scale recommender systems, the dimensionality of nodes’ one-hot representation is extremely high. Moreover, the multi-hot attribute representation simply combines multiple types of attributes into one long vector without considering their interactive relations. The goal of interaction layer is to reduce the dimensionality for one-hot identity representation and learn the high-order attribute interactions for multi-hot attribute representation. To this end, we first set up a lookup table to transform a node’s one-hot representation into the low-dimensional dense vector. The lookup layers correspond to two parameter matrices $M \in \mathbb{R}^{M×D}$ and $N \in \mathbb{R}^{ N×D}$. Each entry $m_u \in \mathbb{R}^D$ and $n_i \in \mathbb{R}^D$ encodes the user $u$’s preference and the item $i$’s property, respectively. Note that $m_u$ and $n_i$ for cold start nodes are meaningless, since no interaction is observed to train their preference embedding. Inspired by (He and Chua 2017), we capture the high-order attribute interactions with a ***Bi-Interactive pooling operation***, in addition to the linear combination operation.

### Gated GNN Layer

Intuitively, different neighbors have different relations to a node. Furthermore, one neighbor usually has multiple attributes. For example, in a social network, a user’s neighborhood may consist of classmates, family members, colleagues, and so on, and each neighbor may have several attributes such as age, gender, and occupation. Since all these attributes (along with the preferences) are now encoded in the node’s embedding, it is necessary to pay different attentions to different dimensions of the neighbor node’s embedding. However, existing GCN (Kipf and Welling 2017) or GAT (Veliˇckovi´c et al. 2018) structures cannot do this because they are at the coarse granularity. GCN treats all neighbors equally and GAT differentiates the importance of neighbors at the node level. To solve this problem, we design a gated-GNN structure to aggregate the fine-grained neighbor information.

### Prediction Layer

Given a user $u$’s final representation $\tilde{p}_u$ and an item $i$’s final representation $\tilde{q}_i$ after the gated-GNN layer, we model the predicted rating of the user $u$ to the item $i$ as:

$$\hat{R}_{u,i} = MLP([\tilde{p}_u; \tilde{q}_i]) + \tilde{p}_u\tilde{q}_i^T + b_u + b_i + \mu,$$

where the MLP function is the multilayer perceptron implemented with one hidden layer, and $b_u$, $b_i$ , and $\mu$ denotes user bias, item bias, and global bias, respectively. The second term is inner product interaction function (Koren, Bell, and Volinsky 2009), and we add the first term to capture the complicated nonlinear interaction between the user and the item.
<!-- #endregion -->

<!-- #region id="lq1voif5zuYJ" -->
### Solution to Cold Start Problem

The cold start problem is caused by the lack of historical interactions for cold start nodes. We view this as a missing preference problem, and solve it by employing the variational autoencoder structure to reconstruct the preference from the attribute distribution.
<!-- #endregion -->

<!-- #region id="XmTZBwpH1YRW" -->
![](https://github.com/recohut/coldstart-recsys/raw/da72950ca514faee94f010a2cb6e99a373044ec1/docs/_images/T290734_2.png)
<!-- #endregion -->

<!-- #region id="HYSIQmIozybK" -->
### Loss Functions

For the rating prediction loss, we employ the square loss as the objective function:

$$L_{pred} = \sum_{u,i \in \mathcal{T}} (\hat{R}_{u,i} - R_{u,i})^2,$$

where $\mathcal{T}$ denotes the set of instances for training, i.e., $\mathcal{T} = {(u, i, r_{u,i}, a_u, a_i)}$, $R_{u,i}$ is ground truth rating in the training set $\mathcal{T}$ , and $\hat{R}_{u,i}$ is the predicted rating.

The reconstruction loss function in eVAE is defined as follows:

$$L_{recon} = − KL(q_\phi(z_u|x_u)||p(z_u)) + \mathbb{E}_{q_\phi}(z_u|x_u)[\log p_θ(x'_u|z_u)] + ||x'_u − m_u||_2,$$

where the first two terms are same as those in standard VAE, and the last one is our extension for the approximation part.

The overall loss function then becomes:

$$L = L_{pred} + L_{recon},$$

where $L_{pred}$ is the task-specific rating prediction loss, and $L_{recon}$ is the reconstruction loss.
<!-- #endregion -->

<!-- #region id="7NIAMKRYuhPq" -->
## Setup
<!-- #endregion -->

<!-- #region id="eNDV6EZpvmOe" -->
### Imports
<!-- #endregion -->

```python id="iwoXoXHwvnrH"
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

import os
import json
import time
import pickle
import argparse
from collections import OrderedDict
```

```python id="09-uWbsAyRKD"
import warnings
warnings.filterwarnings('ignore')
```

<!-- #region id="zLMkK0q7wvmG" -->
### Params
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="_aDgk0kGwwjl" executionInfo={"status": "ok", "timestamp": 1635858986458, "user_tz": -330, "elapsed": 18, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="099ab43a-2cb2-43a0-bcb5-60ec79e4c9f7"
parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=0.0005, type=float,
					help="learning rate.")
parser.add_argument("--dropout", default=0.5, type=float,
					help="dropout rate.")
parser.add_argument("--batch_size", default=128, type=int,
					help="batch size when training.")
parser.add_argument("--gpu", default="0", type=str,
					help="gpu card ID.")
parser.add_argument("--epochs", default=20, type=str,
					help="training epoches.")
parser.add_argument("--clip_norm", default=5.0, type=float,
					help="clip norm for preventing gradient exploding.")
parser.add_argument("--embed_size", default=30, type=int, help="embedding size for users and items.")
parser.add_argument("--attention_size", default=50, type=int, help="embedding size for users and items.")
parser.add_argument("--item_layer1_nei_num", default=10, type=int)
parser.add_argument("--user_layer1_nei_num", default=10, type=int)
parser.add_argument("--vae_lambda", default=1, type=int)
```

<!-- #region id="MSfLDlCQw9VD" -->
## Dataset
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="utY5GhUavnjt" executionInfo={"status": "ok", "timestamp": 1635858988645, "user_tz": -330, "elapsed": 2191, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2459cbd5-5d6d-48a3-aacd-3c1eb30ed840"
!wget -q --show-progress https://github.com/sparsh-ai/coldstart-recsys/raw/main/data/AGNN/ml100k.zip
!unzip ml100k.zip
```

```python id="-f34Bwvow54T"
def get_data_list(ftrain, batch_size):
    f = open(ftrain, 'r')
    train_list = []
    for eachline in f:
        eachline = eachline.strip().split('\t')
        u, i, l = int(eachline[0]), int(eachline[1]), float(eachline[2])
        train_list.append([u, i, l])
    num_batches_per_epoch = int((len(train_list) - 1) / batch_size) + 1
    return num_batches_per_epoch, train_list
```

```python id="rRWBAr5IxClg"
def get_batch_instances(train_list, user_feature_dict, item_feature_dict, item_director_dict, item_writer_dict, item_star_dict, item_country_dict, batch_size, user_nei_dict, item_nei_dict, shuffle=True):
    num_batches_per_epoch = int((len(train_list) - 1) / batch_size) + 1
    def data_generator(train_list):
        data_size = len(train_list)
        user_feature_arr = np.array(list(user_feature_dict.values()))
        max_user_cate_size = user_feature_arr.shape[1]

        item_genre_arr = np.array(list(item_feature_dict.values())) #len=6 ,0
        item_director_arr = np.array(list(item_director_dict.values())) #len=3 ,6
        item_writer_arr = np.array(list(item_writer_dict.values())) #len=3, 9
        item_star_arr = np.array(list(item_star_dict.values())) #len=3, 12
        item_country_arr = np.array(list(item_country_dict.values()))   #len=8, 15

        item_feature_arr = np.concatenate([item_genre_arr, item_director_arr, item_writer_arr, item_star_arr, item_country_arr], axis=1)
        max_item_cate_size = item_feature_arr.shape[1]

        item_layer1_nei_num = FLAGS.item_layer1_nei_num
        user_layer1_nei_num = FLAGS.user_layer1_nei_num

        if shuffle == True:
            np.random.shuffle(train_list)
        train_list = np.array(train_list)

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            current_batch_size = end_index - start_index

            u = train_list[start_index: end_index][:, 0].astype(np.int)
            i = train_list[start_index: end_index][:, 1].astype(np.int)
            l = train_list[start_index: end_index][:, 2]

            i_self_cate = np.zeros([current_batch_size, max_item_cate_size], dtype=np.int)
            i_onehop_id = np.zeros([current_batch_size, item_layer1_nei_num], dtype=np.int)
            i_onehop_cate = np.zeros([current_batch_size, item_layer1_nei_num, max_item_cate_size], dtype=np.int)

            u_self_cate = np.zeros([current_batch_size, max_user_cate_size], dtype=np.int)
            u_onehop_id = np.zeros([current_batch_size, user_layer1_nei_num], dtype=np.int)
            u_onehop_cate = np.zeros([current_batch_size, user_layer1_nei_num, max_user_cate_size], dtype=np.int)

            for index, each_i in enumerate(i):
                i_self_cate[index] = item_feature_arr[each_i]    #item_self_cate

                tmp_one_nei = item_nei_dict[each_i][0]
                tmp_prob = item_nei_dict[each_i][1]
                if len(tmp_one_nei) > item_layer1_nei_num:  #re-sampling
                    tmp_one_nei = np.random.choice(tmp_one_nei, item_layer1_nei_num, replace=False, p=tmp_prob)
                elif len(tmp_one_nei) < item_layer1_nei_num:
                    tmp_one_nei = np.random.choice(tmp_one_nei, item_layer1_nei_num, replace=True, p=tmp_prob)
                tmp_one_nei[-1] = each_i

                i_onehop_id[index] = tmp_one_nei    #item_1_neigh
                i_onehop_cate[index] = item_feature_arr[tmp_one_nei]  #item_1_neigh_cate

            for index, each_u in enumerate(u):
                u_self_cate[index] = user_feature_dict[each_u]  # item_self_cate

                tmp_one_nei = user_nei_dict[each_u][0]
                tmp_prob = user_nei_dict[each_u][1]
                if len(tmp_one_nei) > user_layer1_nei_num:  # re-sampling
                    tmp_one_nei = np.random.choice(tmp_one_nei, user_layer1_nei_num, replace=False, p=tmp_prob)
                elif len(tmp_one_nei) < user_layer1_nei_num:
                    tmp_one_nei = np.random.choice(tmp_one_nei, user_layer1_nei_num, replace=True, p=tmp_prob)
                tmp_one_nei[-1] = each_u

                u_onehop_id[index] = tmp_one_nei  # user_1_neigh
                u_onehop_cate[index] = user_feature_arr[tmp_one_nei]  # user_1_neigh_cate

            yield ([u, i, l, u_self_cate, u_onehop_id, u_onehop_cate, i_self_cate, i_onehop_id, i_onehop_cate])
    return data_generator(train_list)
```

<!-- #region id="EGEcbwIxvng5" -->
## Model
<!-- #endregion -->

```python id="PAev2rRAvndh"
class VAE(nn.Module):

    def __init__(self, embed_size):
        super(VAE, self).__init__()

        Z_dim = X_dim = h_dim = embed_size
        self.Z_dim = Z_dim
        self.X_dim= X_dim
        self.h_dim = h_dim
        self.embed_size= embed_size

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)

        # =============================== Q(z|X) ======================================
        self.dense_xh = nn.Linear(X_dim, h_dim)
        init_weights(self.dense_xh)

        self.dense_hz_mu = nn.Linear(h_dim, Z_dim)
        init_weights(self.dense_hz_mu)

        self.dense_hz_var = nn.Linear(h_dim, Z_dim)
        init_weights(self.dense_hz_var)

        # =============================== P(X|z) ======================================
        self.dense_zh = nn.Linear(Z_dim, h_dim)
        init_weights(self.dense_zh)

        self.dense_hx = nn.Linear(h_dim, X_dim)
        init_weights(self.dense_hx)

    def Q(self, X):
        h = nn.ReLU()(self.dense_xh(X))
        z_mu = self.dense_hz_mu(h)
        z_var = self.dense_hz_var(h)
        return z_mu, z_var

    def sample_z(self, mu, log_var):
        mb_size = mu.shape[0]
        eps = Variable(torch.randn(mb_size, self.Z_dim)).cuda()
        return mu + torch.exp(log_var / 2) * eps

    def P(self, z):
        h = nn.ReLU()(self.dense_zh(z))
        X = self.dense_hx(h)
        return X
```

```python id="MA3ueUySwMW1"
class AGNN(torch.nn.Module):
    def __init__(self, user_size, item_size, gender_size, age_size, occupation_size, genre_size, director_size, writer_size, star_size, country_size, embed_size, attention_size, dropout):
        super(AGNN, self).__init__()
        self.user_size = user_size
        self.item_size = item_size
        self.gender_size = gender_size
        self.age_size = age_size
        self.occupation_size = occupation_size
        self.genre_size = genre_size
        self.director_size = director_size
        self.writer_size = writer_size
        self.star_size = star_size
        self.country_size = country_size
        self.embed_size = embed_size
        self.dropout = dropout
        self.attention_size = attention_size

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)

        self.user_embed = torch.nn.Embedding(self.user_size, self.embed_size)
        self.item_embed = torch.nn.Embedding(self.item_size, self.embed_size)
        nn.init.xavier_uniform(self.user_embed.weight)
        nn.init.xavier_uniform(self.item_embed.weight)

        self.user_bias = torch.nn.Embedding(self.user_size, 1)
        self.item_bias = torch.nn.Embedding(self.item_size, 1)
        nn.init.constant(self.user_bias.weight, 0)
        nn.init.constant(self.item_bias.weight, 0)

        self.miu = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

        self.gender_embed = torch.nn.Embedding(self.gender_size, self.embed_size)
        self.gender_embed.weight.data.normal_(0, 0.05)
        self.age_embed = torch.nn.Embedding(self.age_size, self.embed_size)
        self.age_embed.weight.data.normal_(0, 0.05)
        self.occupation_embed = torch.nn.Embedding(self.occupation_size, self.embed_size)
        self.occupation_embed.weight.data.normal_(0, 0.05)

        self.genre_embed = torch.nn.Embedding(self.genre_size, self.embed_size)
        self.genre_embed.weight.data.normal_(0, 0.05)
        self.director_embed = torch.nn.Embedding(self.director_size, self.embed_size)
        self.director_embed.weight.data.normal_(0, 0.05)
        self.writer_embed = torch.nn.Embedding(self.writer_size, self.embed_size)
        self.writer_embed.weight.data.normal_(0, 0.05)
        self.star_embed = torch.nn.Embedding(self.star_size, self.embed_size)
        self.star_embed.weight.data.normal_(0, 0.05)
        self.country_embed = torch.nn.Embedding(self.country_size, self.embed_size)
        self.country_embed.weight.data.normal_(0, 0.05)


        #--------------------------------------------------
        self.dense_item_self_biinter = nn.Linear(self.embed_size, self.embed_size)
        self.dense_item_self_siinter = nn.Linear(self.embed_size, self.embed_size)
        self.dense_item_onehop_biinter = nn.Linear(self.embed_size, self.embed_size)
        self.dense_item_onehop_siinter = nn.Linear(self.embed_size, self.embed_size)
        self.dense_user_self_biinter = nn.Linear(self.embed_size, self.embed_size)
        self.dense_user_self_siinter = nn.Linear(self.embed_size, self.embed_size)
        self.dense_user_onehop_biinter = nn.Linear(self.embed_size, self.embed_size)
        self.dense_user_onehop_siinter = nn.Linear(self.embed_size, self.embed_size)
        init_weights(self.dense_item_self_biinter)
        init_weights(self.dense_item_self_siinter)
        init_weights(self.dense_item_onehop_biinter)
        init_weights(self.dense_item_onehop_siinter)
        init_weights(self.dense_user_self_biinter)
        init_weights(self.dense_user_self_siinter)
        init_weights(self.dense_user_onehop_biinter)
        init_weights(self.dense_user_onehop_siinter)

        self.dense_item_cate_self = nn.Linear(2 * self.embed_size, self.embed_size)
        self.dense_item_cate_hop1 = nn.Linear(2 * self.embed_size, self.embed_size)
        self.dense_user_cate_self = nn.Linear(2 * self.embed_size, self.embed_size)
        self.dense_user_cate_hop1 = nn.Linear(2 * self.embed_size, self.embed_size)
        init_weights(self.dense_item_cate_self)
        init_weights(self.dense_item_cate_hop1)
        init_weights(self.dense_user_cate_self)
        init_weights(self.dense_user_cate_hop1)

        self.dense_item_addgate = nn.Linear(self.embed_size * 2, self.embed_size)
        init_weights(self.dense_item_addgate)
        self.dense_item_erasegate = nn.Linear(self.embed_size * 2, self.embed_size)
        init_weights(self.dense_item_erasegate)
        self.dense_user_addgate = nn.Linear(self.embed_size * 2, self.embed_size)
        init_weights(self.dense_user_addgate)
        self.dense_user_erasegate = nn.Linear(self.embed_size * 2, self.embed_size)

        self.user_vae = VAE(embed_size)
        self.item_vae = VAE(embed_size)

        #----------------------------------------------------
        #concat, mlp
        self.FC_pre = nn.Linear(2 * embed_size, 1)
        init_weights(self.FC_pre)

        """# dot
        self.user_bias = nn.Embedding(self.user_size, 1)
        self.item_bias = nn.Embedding(self.item_size, 1)
        self.user_bias.weight.data.normal_(0, 0.01)
        self.item_bias.weight.data.normal_(0, 0.01)
        self.bias = torch.nn.Parameter(torch.rand(1), requires_grad=True)
        self.bias.data.uniform_(0, 0.1)"""

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.2)

    def feat_interaction(self, feature_embedding, fun_bi, fun_si, dimension):
        summed_features_emb_square = (torch.sum(feature_embedding, dim=dimension)).pow(2)
        squared_sum_features_emb = torch.sum(feature_embedding.pow(2), dim=dimension)
        deep_fm = 0.5 * (summed_features_emb_square - squared_sum_features_emb)
        deep_fm = self.leakyrelu(fun_bi(deep_fm))
        bias_fm = self.leakyrelu(fun_si(feature_embedding.sum(dim=dimension)))
        nfm = deep_fm + bias_fm
        return nfm

    def forward(self, user, item, user_self_cate, user_onehop_id, user_onehop_cate, item_self_cate, item_self_director, item_self_writer, item_self_star, item_self_country, item_onehop_id, item_onehop_cate, item_onehop_director, item_onehop_writer, item_onehop_star, item_onehop_country, mode='train'):

        uids_list = user.cuda()
        sids_list = item.cuda()
        if mode == 'train' or mode == 'warm':
            user_embedding = self.user_embed(torch.autograd.Variable(uids_list))
            item_embedding = self.item_embed(torch.autograd.Variable(sids_list))
        if mode == 'ics':
            user_embedding = self.user_embed(torch.autograd.Variable(uids_list))
        if mode == 'ucs':
            item_embedding = self.item_embed(torch.autograd.Variable(sids_list))

        batch_size = item_self_cate.shape[0]
        cate_size = item_self_cate.shape[1]
        director_size = item_self_director.shape[1]
        writer_size = item_self_writer.shape[1]
        star_size = item_self_star.shape[1]
        country_size = item_self_country.shape[1]
        user_onehop_size = user_onehop_id.shape[1]
        item_onehop_size = item_onehop_id.shape[1]

        #------------------------------------------------------GCN-item
        # K=2
        item_onehop_id = self.item_embed(Variable(item_onehop_id))

        item_onehop_cate = self.genre_embed(Variable(item_onehop_cate).view(-1, cate_size)).view(batch_size,item_onehop_size,cate_size, -1)
        item_onehop_director = self.director_embed(Variable(item_onehop_director).view(-1, director_size)).view(batch_size, item_onehop_size, director_size, -1)
        item_onehop_writer = self.writer_embed(Variable(item_onehop_writer).view(-1, writer_size)).view(batch_size, item_onehop_size, writer_size, -1)
        item_onehop_star = self.star_embed(Variable(item_onehop_star).view(-1, star_size)).view(batch_size, item_onehop_size, star_size, -1)
        item_onehop_country = self.country_embed(Variable(item_onehop_country).view(-1, country_size)).view(batch_size, item_onehop_size, country_size, -1)

        item_onehop_feature = torch.cat([item_onehop_cate, item_onehop_director, item_onehop_writer, item_onehop_star, item_onehop_country], dim=2)
        item_onehop_embed = self.dense_item_cate_hop1(torch.cat([self.feat_interaction(item_onehop_feature, self.dense_item_onehop_biinter,  self.dense_item_onehop_siinter, dimension=2), item_onehop_id], dim=-1))

        # K=1
        item_self_cate = self.genre_embed(Variable(item_self_cate))
        item_self_director = self.director_embed(Variable(item_self_director))
        item_self_writer = self.writer_embed(Variable(item_self_writer))
        item_self_star = self.star_embed(Variable(item_self_star))
        item_self_country = self.country_embed(Variable(item_self_country))

        item_self_feature = torch.cat([item_self_cate, item_self_director, item_self_writer, item_self_star, item_self_country], dim=1)
        item_self_feature = self.feat_interaction(item_self_feature, self.dense_item_self_biinter, self.dense_item_self_siinter, dimension=1)

        if mode == 'ics':
            item_mu, item_var = self.item_vae.Q(item_self_feature)
            item_z = self.item_vae.sample_z(item_mu, item_var)
            item_embedding = self.item_vae.P(item_z)
        item_self_embed = self.dense_item_cate_self(torch.cat([item_self_feature, item_embedding], dim=-1))

        item_addgate = self.sigmoid(self.dense_item_addgate(torch.cat([item_self_embed.unsqueeze(1).repeat(1, item_onehop_size, 1), item_onehop_embed], dim=-1)))  # 商品的邻居门，控制邻居信息多少作为输入
        item_erasegate = self.sigmoid(self.dense_item_erasegate(torch.cat([item_self_embed, item_onehop_embed.mean(dim=1)], dim=-1)))
        item_onehop_embed_final = (item_onehop_embed * item_addgate).mean(1)
        item_self_embed = (1 - item_erasegate) * item_self_embed

        item_gcn_embed = self.leakyrelu(item_self_embed + item_onehop_embed_final)  # [batch, embed]

        #----------------------------------------------------------GCN-user
        # K=2
        user_onehop_id = self.user_embed(Variable(user_onehop_id))

        user_onehop_gender_emb = self.gender_embed(Variable(user_onehop_cate[:, :, 0]))
        user_onehop_age_emb = self.age_embed(Variable(user_onehop_cate[:, :, 1]))
        user_onehop_occupation_emb = self.occupation_embed(Variable(user_onehop_cate[:, :, 2]))

        user_onehop_feat = torch.cat([user_onehop_gender_emb.unsqueeze(2), user_onehop_age_emb.unsqueeze(2), user_onehop_occupation_emb.unsqueeze(2)], dim=2)
        user_onehop_embed = self.dense_user_cate_hop1(torch.cat([self.feat_interaction(user_onehop_feat, self.dense_user_onehop_biinter, self.dense_user_onehop_siinter, dimension=2), user_onehop_id], dim=-1))

        # K=1
        user_gender_emb = self.gender_embed(Variable(user_self_cate[:, 0]))
        user_age_emb = self.age_embed(Variable(user_self_cate[:, 1]))
        user_occupation_emb = self.occupation_embed(Variable(user_self_cate[:, 2]))

        user_self_feature = torch.cat([user_gender_emb.unsqueeze(1), user_age_emb.unsqueeze(1), user_occupation_emb.unsqueeze(1)], dim=1)
        user_self_feature = self.feat_interaction(user_self_feature, self.dense_user_self_biinter,  self.dense_user_onehop_siinter, dimension=1)

        if mode == 'ucs':
            user_mu, user_var = self.user_vae.Q(user_self_feature)
            user_z = self.user_vae.sample_z(user_mu, user_var)
            user_embedding = self.user_vae.P(user_z)
        user_self_embed = self.dense_user_cate_self(torch.cat([user_self_feature, user_embedding], dim=-1))

        user_addgate = self.sigmoid(self.dense_user_addgate(torch.cat([user_self_embed.unsqueeze(1).repeat(1, user_onehop_size, 1), user_onehop_embed],dim=-1)))
        user_erasegate = self.sigmoid(self.dense_user_erasegate(torch.cat([user_self_embed, user_onehop_embed.mean(dim=1)], dim=-1)))
        user_onehop_embed_final = (user_onehop_embed * user_addgate).mean(dim=1)
        user_self_embed = (1 - user_erasegate) * user_self_embed

        user_gcn_embed = self.leakyrelu(user_self_embed + user_onehop_embed_final)

        #--------------------------------------------------norm
        item_mu, item_var = self.item_vae.Q(item_self_feature)
        item_z = self.item_vae.sample_z(item_mu, item_var)
        item_preference_sample = self.item_vae.P(item_z)

        user_mu, user_var = self.user_vae.Q(user_self_feature)
        user_z = self.user_vae.sample_z(user_mu, user_var)
        user_preference_sample = self.user_vae.P(user_z)

        recon_loss = torch.norm(item_preference_sample - item_embedding) + torch.norm(user_preference_sample - user_embedding)
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(item_z) + item_mu ** 2 - 1. - item_var, 1)) + \
                  torch.mean(0.5 * torch.sum(torch.exp(user_z) + user_mu ** 2 - 1. - user_var, 1))

        ####################################prediction#####################################################

        #concat -> mlp
        bu = self.user_bias(Variable(uids_list))
        bi = self.item_bias(Variable(sids_list))
        #pred = (user_gcn_embed * item_gcn_embed).sum(1, keepdim=True) + bu + bi + (self.miu).repeat(batch_size, 1)
        tmp = torch.cat([user_gcn_embed, item_gcn_embed], dim=1)
        pred = self.FC_pre(tmp) + (user_gcn_embed * item_gcn_embed).sum(1, keepdim=True) + bu + bi + (self.miu).repeat(batch_size, 1)

        return pred.squeeze(), recon_loss, kl_loss
```

<!-- #region id="SdtWr805wMR0" -->
## Metrics and Evaluation
<!-- #endregion -->

```python id="dmvZXQfdwMLT"
def metrics(model, test_dataloader):
    label_lst, pred_lst = [], []
    rmse, mse, mae = 0,0,0
    count = 0
    for batch_data in test_dataloader:
        user = torch.LongTensor(batch_data[0]).cuda()
        item = torch.LongTensor(batch_data[1]).cuda()
        label = torch.FloatTensor(batch_data[2]).cuda()
        user_self_cate = torch.LongTensor(batch_data[3]).cuda()
        user_onehop_id = torch.LongTensor(batch_data[4]).cuda()
        user_onehop_cate = torch.LongTensor(batch_data[5]).cuda()
        item_self_cate, item_self_director, item_self_writer, item_self_star, item_self_country = torch.LongTensor(
            batch_data[6])[:, 0:6].cuda(), torch.LongTensor(batch_data[6])[:, 6:9].cuda(), torch.LongTensor(
            batch_data[6])[:, 9:12].cuda(), torch.LongTensor(batch_data[6])[:, 12:15].cuda(), torch.LongTensor(
            batch_data[6])[:, 15:].cuda()
        item_onehop_id = torch.LongTensor(batch_data[7]).cuda()
        item_onehop_cate, item_onehop_director, item_onehop_writer, item_onehop_star, item_onehop_country = torch.LongTensor(
            batch_data[8])[:, :, 0:6].cuda(), torch.LongTensor(batch_data[8])[:, :, 6:9].cuda(), torch.LongTensor(
            batch_data[8])[:, :, 9:12].cuda(), torch.LongTensor(batch_data[8])[:, :, 12:15].cuda(), torch.LongTensor(
            batch_data[8])[:, :, 15:].cuda()

        prediction, recon_loss, kl_loss = model(user, item, user_self_cate, user_onehop_id, user_onehop_cate, item_self_cate,
                           item_self_director, item_self_writer, item_self_star, item_self_country, item_onehop_id,
                           item_onehop_cate, item_onehop_director, item_onehop_writer, item_onehop_star,
                           item_onehop_country, mode = mode)
        prediction = prediction.cpu().data.numpy()
        prediction = prediction.reshape(prediction.shape[0])
        label = label.cpu().numpy()
        my_rmse = np.sum((prediction - label) ** 2)
        my_mse = np.sum((prediction - label) ** 2)
        my_mae = np.sum(np.abs(prediction - label))
        # my_rmse = torch.sqrt(torch.sum((prediction - label) ** 2) / FLAGS.batch_size)
        rmse+=my_rmse
        mse+=my_mse
        mae+=my_mae
        count += len(user)
        label_lst.extend(list([float(l) for l in label]))
        pred_lst.extend(list([float(l) for l in prediction]))

    my_mse = mse/count
    my_rmse = np.sqrt(rmse/count)
    my_mae = mae/count
    return my_rmse, my_mse, my_mae, label_lst, pred_lst
```

```python id="w4sXP-MRxRVM" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1635860051781, "user_tz": -330, "elapsed": 993538, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e3f51476-ec76-404f-a86e-5eacbc1e5291"
if __name__ == '__main__':
    #item cold start
    f_info = 'ml100k/uiinfo.pkl'
    f_neighbor = 'ml100k/neighbor_aspect_extension_2_zscore_ics_uuii_0.20.pkl'
    f_train = 'ml100k/ics_train.dat'
    f_test = 'ml100k/ics_val.dat'
    f_model = 'ml100k/agnn_ics_'
    mode = 'ics'

    """# user cold start
    f_info = 'ml100k/uiinfo.pkl'
    f_neighbor = 'ml100k/neighbor_aspect_extension_2_zscore_ucs_uuii.pkl'
    f_train = 'ml100k/ucs_train.dat'
    f_test = 'ml100k/ucs_val.dat'
    f_model = 'ml100k/agnn_ucs_'
    mode = 'ucs'"""

    """# warm start
    f_info = 'ml100k/uiinfo.pkl'
    f_neighbor = 'ml100k/neighbor_aspect_extension_2_zscore_warm_uuii.pkl'
    f_train = 'ml100k/warm_train.dat'
    f_test = 'ml100k/warm_val.dat'
    f_model = 'ml100k/agnn_warm_'
    mode = 'warm'"""


    FLAGS = parser.parse_args(args={})
    print("\nParameters:")
    print(FLAGS.__dict__)

    with open(f_neighbor, 'rb') as f:
        neighbor_dict = pickle.load(f)
    user_nei_dict = neighbor_dict['user_nei_dict']
    item_nei_dict = neighbor_dict['item_nei_dict']
    director_num = neighbor_dict['director_num']
    writer_num = neighbor_dict['writer_num']
    star_num = neighbor_dict['star_num']
    country_num = neighbor_dict['country_num']

    item_director_dict = neighbor_dict['item_director_dict']    #dict[i]=[x,x,x]
    item_writer_dict = neighbor_dict['item_writer_dict']        #dict[i]=[x,x,x]
    item_star_dict = neighbor_dict['item_star_dict']            #dict[i]=[x,x,x]
    item_country_dict = neighbor_dict['item_country_dict']      #dict[i]=[x,x,x,x,x,x,x,x]

    with open(f_info, 'rb') as f:
        item_info = pickle.load(f)
    user_num = item_info['user_num']
    item_num = item_info['item_num']
    gender_num = item_info['gender_num']
    age_num = item_info['age_num']
    occupation_num = item_info['occupation_num']
    genre_num = item_info['genre_num']
    user_feature_dict = item_info['user_feature_dict']  #gender, age, occupation    dict[u]=[x,x,x]
    item_feature_dict = item_info['item_feature_dict']  #genre                      dict[i]=[x,x,x,x,x,x]

    print("user_num {}, item_num {}, gender_num {}, age_num {}, occupation_num {}, genre_num {}, director_num {}, writer_num {}, star_num {}, country_num {}, mode {} ".format(user_num, item_num, gender_num, age_num, occupation_num, genre_num, director_num, writer_num, star_num, country_num, mode))

    train_steps, train_list = get_data_list(f_train, batch_size=FLAGS.batch_size)
    test_steps, test_list = get_data_list(f_test, batch_size=FLAGS.batch_size)

    model = AGNN(user_num, item_num, gender_num, age_num, occupation_num, genre_num, director_num, writer_num, star_num, country_num, FLAGS.embed_size, FLAGS.attention_size, FLAGS.dropout)
    model.cuda()

    loss_function = torch.nn.MSELoss(size_average=False)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=FLAGS.lr, weight_decay=0.001)

    writer = SummaryWriter()  # For visualization
    #f_loss_curve = open('tmp_loss_curve.txt', 'w')
    best_rmse = 5

    count = 0
    for epoch in range(FLAGS.epochs):
        #tmp_main_loss, tmp_vae_loss = [], []
        model.train()  # Enable dropout (if have).
        start_time = time.time()
        train_dataloader = get_batch_instances(train_list, user_feature_dict, item_feature_dict, item_director_dict, item_writer_dict, item_star_dict, item_country_dict,  batch_size=FLAGS.batch_size, user_nei_dict=user_nei_dict, item_nei_dict=item_nei_dict, shuffle=True)

        for idx, batch_data in enumerate(train_dataloader): #u, i, l, u_self_cate, u_onehop_id, u_onehop_rating, u_onehop_cate, i_self_cate, i_onehop_id, i_onehop_cate
            user = torch.LongTensor(batch_data[0]).cuda()
            item = torch.LongTensor(batch_data[1]).cuda()
            label = torch.FloatTensor(batch_data[2]).cuda()
            user_self_cate = torch.LongTensor(batch_data[3]).cuda()
            user_onehop_id = torch.LongTensor(batch_data[4]).cuda()
            user_onehop_cate = torch.LongTensor(batch_data[5]).cuda()
            item_self_cate, item_self_director, item_self_writer, item_self_star, item_self_country = torch.LongTensor(batch_data[6])[:, 0:6].cuda(), torch.LongTensor(batch_data[6])[:, 6:9].cuda(), torch.LongTensor(batch_data[6])[:, 9:12].cuda(), torch.LongTensor(batch_data[6])[:, 12:15].cuda(), torch.LongTensor(batch_data[6])[:, 15:].cuda()
            item_onehop_id = torch.LongTensor(batch_data[7]).cuda()
            item_onehop_cate, item_onehop_director, item_onehop_writer, item_onehop_star, item_onehop_country = torch.LongTensor(batch_data[8])[:, :, 0:6].cuda(), torch.LongTensor(batch_data[8])[:, :, 6:9].cuda(), torch.LongTensor(batch_data[8])[:, :, 9:12].cuda(), torch.LongTensor(batch_data[8])[:, :, 12:15].cuda(), torch.LongTensor(batch_data[8])[:, :, 15:].cuda()

            model.zero_grad()
            prediction, recon_loss, kl_loss = model(user, item, user_self_cate, user_onehop_id, user_onehop_cate, item_self_cate, item_self_director, item_self_writer, item_self_star, item_self_country, item_onehop_id, item_onehop_cate, item_onehop_director, item_onehop_writer, item_onehop_star, item_onehop_country, mode='train')

            label = Variable(label)

            main_loss = loss_function(prediction, label)
            loss = main_loss + FLAGS.vae_lambda * (recon_loss + kl_loss)

            loss.backward()
            # nn.utils.clip_grad_norm(model.parameters(), FLAGS.clip_norm)
            optimizer.step()
            writer.add_scalar('data/loss', loss.data, count)
            count += 1

        tmploss = torch.sqrt(loss / FLAGS.batch_size)
        print(50 * '#')
        print('epoch: ', epoch, '     ', tmploss.detach())

        model.eval()
        print('time = ', time.time() - start_time)
        test_dataloader = get_batch_instances(test_list, user_feature_dict, item_feature_dict, item_director_dict, item_writer_dict, item_star_dict, item_country_dict, batch_size=FLAGS.batch_size, user_nei_dict=user_nei_dict, item_nei_dict=item_nei_dict, shuffle=False)
        rmse, mse, mae, label_lst, pred_lst = metrics(model, test_dataloader)
        print('test rmse,mse,mae: ', rmse,mse,mae)

        """if (rmse < best_rmse):
            best_rmse = rmse
            f_name = f_model + str(best_rmse)[:7] + '.dat' #f_model + str(best_rmse)[:7] + '.dat'
            #torch.save(model, f_name)
            f = open(f_name, 'w')
            res_dict = {}
            res_dict['label'] = label_lst
            res_dict['pred'] = pred_lst
            json.dump(res_dict, f)
            f.close()
            print('save result ok')"""
```

<!-- #region id="Uo_PSXGbxxil" -->
**END**
<!-- #endregion -->
