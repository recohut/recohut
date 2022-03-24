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

<!-- #region id="_zVKXBtcSwT4" -->
# Training GroupIM model on Weeplaces dataset
<!-- #endregion -->

<!-- #region id="uD-PM-u9QCGx" -->
## Executive summary
<!-- #endregion -->

<!-- #region id="7DDul7W4PxNH" -->
| | |
| --- | --- |
| Problem | Group interactions are sparse in nature which makes it difficult to provide relevant recommendation to the group. |
| Solution | Regularize the user-group latent space to overcome group interaction sparsity by: maximizing mutual information between representations of groups and group members; and dynamically prioritizing the preferences of highly informative members through contextual preference weighting. |
| Dataset | Weeplaces |
| Preprocessing | We extract check-ins on POIs over all major cities in the United States, across various categories including Food, Nightlife, Outdoors, Entertainment and Travel. We randomly split the set of all groups into training (70%), validation (10%), and test (20%) sets, while utilizing the individual interactions of all users for training. Note that each group appears only in one of the three sets. The test set contains strict ephemeral groups (i.e., a specific combination of users) that do not occur in the training set. Thus, we train on ephemeral groups and test on strict ephemeral groups. |
| Metrics | NDCG, Recall |
| Hyperparams | We tune the latent dimension in the range {32, 64, 128} and other baseline hyper-parameters in ranges centered at author-provided values. In GroupIM, we use two fully connected layers of size 64 each in fenc(·) and tune λ in the range {$2^{−4}$,$2^{−3}$,$\dots$, $2^{6}$}. We use 5 negatives for each true user-group pair to train the discriminator. |
| Models | GroupIM along with Encoder, 3 types of Aggregators to choose from, and a discriminator module. |
| Platform | PyTorch, preferable GPU for faster computation. |
| Links | [Paper](https://arxiv.org/abs/2006.03736), [Code](https://github.com/RecoHut-Stanzas/S168471) |
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="CQ_O1rPQ1geg" executionInfo={"status": "ok", "timestamp": 1640781863463, "user_tz": -330, "elapsed": 4272, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="79d0080c-5eaf-4ea2-d649-29986fe021b9"
!pip install -q recohut
```

<!-- #region id="8Wdr7RzCyw68" -->
## Datasets
<!-- #endregion -->

```python id="84M64rqgyzz6"
import os
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from torch.utils import data

from recohut.datasets import base
from recohut.utils.common_utils import download_url, extract_zip
```

```python id="KVLWlKBV1AOh"
class WeeplacesDataset(base.Dataset, data.Dataset):
    url = "https://github.com/RecoHut-Datasets/weeplaces/raw/v2/data.zip"

    def __init__(self, root, datatype='train', is_group=False, n_items=None, negs_per_group=None, padding_idx=None, verbose=True):
        super().__init__(root)
        self.datatype = datatype
        self.n_items = n_items
        self.negs_per_group = negs_per_group
        self.is_group = is_group
        self.padding_idx = padding_idx
        if is_group:
            if datatype=='train':
                self.user_data = self.load_user_data_train()
                self.group_data, self.group_users = self.load_group_data_train()
                self.group_inputs = [self.user_data[self.group_users[g]] for g in self.groups_list]
            else:
                self.eval_groups_list = []
                self.user_data = self.load_user_data_tr_te(datatype)
                self.eval_group_data, self.eval_group_users = self.load_group_data_tr_te(datatype)
        else:
            if datatype=='train':
                self.train_data_ui = self.load_ui_train()
                self.user_list = list(range(self.n_users))
            else:
                self.data_tr, self.data_te = self.load_ui_tr_te(datatype)
    
    def __len__(self):
        if self.is_group:
            if self.datatype=='train':
                return len(self.groups_list)
            return len(self.eval_groups_list)
        return len(self.user_list)

    def __train__(self, index):
        """ load user_id, binary vector over items """
        user = self.user_list[index]
        user_items = torch.from_numpy(self.train_data_ui[user, :].toarray()).squeeze()  # [I]
        return torch.from_numpy(np.array([user], dtype=np.int32)), user_items

    def __test__(self, index):
        """ load user_id, fold-in items, held-out items """
        user = self.user_list[index]
        fold_in, held_out = self.data_tr[user, :].toarray(), self.data_te[user, :].toarray()  # [I], [I]
        return user, torch.from_numpy(fold_in).squeeze(), held_out.squeeze()  # user, fold-in items, fold-out items.

    def __train_group__(self, index):
        """ load group_id, padded group users, mask, group items, group member items, negative user items """
        group = self.groups_list[index]
        user_ids = torch.from_numpy(np.array(self.group_users[group], np.int32))  # [G] group member ids
        group_items = torch.from_numpy(self.group_data[group].toarray().squeeze())  # [I] items per group

        corrupted_group = self.get_corrupted_users(group)  # [# negs]
        corrupted_user_items = torch.from_numpy(self.user_data[corrupted_group].toarray().squeeze())  # [# negs, I]

        # group mask to create fixed-size padded groups.
        group_length = self.max_group_size - list(user_ids).count(self.padding_idx)
        group_mask = torch.from_numpy(np.concatenate([np.zeros(group_length, dtype=np.float32), (-1) * np.inf *
                                                      np.ones(self.max_group_size - group_length,
                                                              dtype=np.float32)]))  # [G]

        user_items = torch.from_numpy(self.group_inputs[group].toarray())  # [G, |I|] group member items

        return torch.tensor([group]), user_ids, group_mask, group_items, user_items, corrupted_user_items

    def __test_group__(self, index):
        """ load group_id, padded group users, mask, group items, group member items """
        group = self.eval_groups_list[index]
        user_ids = self.eval_group_users[group]  # [G]
        length = self.max_gsize - list(user_ids).count(self.padding_idx)
        mask = torch.from_numpy(np.concatenate([np.zeros(length, dtype=np.float32), (-1) * np.inf *
                                                np.ones(self.max_gsize - length, dtype=np.float32)]))  # [G]
        group_items = torch.from_numpy(self.eval_group_data[group].toarray().squeeze())  # [I]
        user_items = torch.from_numpy(self.user_data[user_ids].toarray().squeeze())  # [G, I]

        return torch.tensor([group]), torch.tensor(user_ids), mask, group_items, user_items

    def __getitem__(self, index):
        if self.is_group:
            if self.datatype=='train':
                return self.__train_group__(index)
            return self.__test_group__(index)
        else:
            if self.datatype=='train':
                return self.__train__(index)
            return self.__test__(index)

    @property
    def raw_file_names(self) -> str:
        return ['train_ui.csv',
                'val_ui_te.csv',
                'group_users.csv',
                'data.zip',
                'train_gi.csv',
                'test_ui_tr.csv',
                'val_ui_tr.csv',
                'test_ui_te.csv',
                'val_gi.csv',
                'test_gi.csv']

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)

    @property
    def processed_file_names(self) -> str:
        pass

    def process(self):
        pass

    def load_ui_train(self):
        """ load training user-item interactions as a sparse matrix """
        path_ui = [p for p in self.raw_paths if "train_ui" in p][0]
        df_ui = pd.read_csv(path_ui)
        self.n_users, self.n_items = df_ui['user'].max() + 1, df_ui['item'].max() + 1
        rows_ui, cols_ui = df_ui['user'], df_ui['item']
        data_ui = sp.csr_matrix((np.ones_like(rows_ui), (rows_ui, cols_ui)), dtype='float32',
                                shape=(self.n_users, self.n_items))  # [# train users, I] sparse matrix
        print("# train users", self.n_users, "# items", self.n_items)
        return data_ui

    def load_ui_tr_te(self, datatype='val'):
        """ load user-item interactions of val/test user sets as two sparse matrices of fold-in and held-out items """
        ui_tr_path = [p for p in self.raw_paths if '{}_ui_tr.csv'.format(datatype) in p][0]

        ui_te_path = [p for p in self.raw_paths if '{}_ui_te.csv'.format(datatype) in p][0]

        ui_df_tr, ui_df_te = pd.read_csv(ui_tr_path), pd.read_csv(ui_te_path)

        start_idx = min(ui_df_tr['user'].min(), ui_df_te['user'].min())
        end_idx = max(ui_df_tr['user'].max(), ui_df_te['user'].max())

        rows_tr, cols_tr = ui_df_tr['user'] - start_idx, ui_df_tr['item']
        rows_te, cols_te = ui_df_te['user'] - start_idx, ui_df_te['item']
        self.user_list = list(range(0, end_idx - start_idx + 1))

        ui_data_tr = sp.csr_matrix((np.ones_like(rows_tr), (rows_tr, cols_tr)), dtype='float32',
                                   shape=(end_idx - start_idx + 1, self.n_items))  # [# eval users, I] sparse matrix
        ui_data_te = sp.csr_matrix((np.ones_like(rows_te), (rows_te, cols_te)), dtype='float32',
                                   shape=(end_idx - start_idx + 1, self.n_items))  # [# eval users, I] sparse matrix
        return ui_data_tr, ui_data_te

    def get_corrupted_users(self, group):
        """ negative user sampling per group (eta balances item-biased and random sampling) """
        eta = 0.5
        p = np.ones(self.n_users + 1)
        p[self.group_users[group]] = 0
        p = normalize([p], norm='l1')[0]
        item_biased = normalize(self.user_data[:, self.group_data[group].indices].sum(1).squeeze(), norm='l1')[0]
        p = eta * item_biased + (1 - eta) * p
        negative_users = torch.multinomial(torch.from_numpy(p), self.negs_per_group)
        return negative_users

    def load_user_data_train(self):
        """ load user-item interactions of all users that appear in training groups, as a sparse matrix """
        df_ui = pd.DataFrame()
        train_path_ui = [p for p in self.raw_paths if 'train_ui.csv' in p][0]
        df_train_ui = pd.read_csv(train_path_ui)
        df_ui = df_ui.append(df_train_ui)

        # include users from the (fold-in item set) of validation and test sets of user-item data.
        val_path_ui = [p for p in self.raw_paths if 'val_ui_tr.csv' in p][0]
        df_val_ui = pd.read_csv(val_path_ui)
        df_ui = df_ui.append(df_val_ui)

        test_path_ui = [p for p in self.raw_paths if 'test_ui_tr.csv' in p][0]
        df_test_ui = pd.read_csv(test_path_ui)
        df_ui = df_ui.append(df_test_ui)

        self.n_users = df_ui['user'].max() + 1
        self.padding_idx = self.n_users  # padding idx for user when creating groups of fixed size.
        assert self.n_items == df_ui['item'].max() + 1
        rows_ui, cols_ui = df_ui['user'], df_ui['item']

        data_ui = sp.csr_matrix((np.ones_like(rows_ui), (rows_ui, cols_ui)), dtype='float32',
                                shape=(self.n_users + 1, self.n_items))  # [U, I] sparse matrix
        return data_ui

    def load_user_data_tr_te(self, datatype):
        """ load all user-item interactions of users that occur in val/test groups, as a sparse matrix """
        df_ui = pd.DataFrame()
        train_path_ui = [p for p in self.raw_paths if 'train_ui.csv' in p][0]
        df_train_ui = pd.read_csv(train_path_ui)
        df_ui = df_ui.append(df_train_ui)

        val_path_ui = [p for p in self.raw_paths if 'val_ui_tr.csv' in p][0]
        df_val_ui = pd.read_csv(val_path_ui)
        df_ui = df_ui.append(df_val_ui)

        if datatype == 'val' or datatype == 'test':
            # include eval user set (tr) items (since they might occur in evaluation set)
            test_path_ui = [p for p in self.raw_paths if 'test_ui_tr.csv' in p][0]
            df_test_ui = pd.read_csv(test_path_ui)
            df_ui = df_ui.append(df_test_ui)

        n_users = df_ui['user'].max() + 1
        assert self.n_items == df_ui['item'].max() + 1
        rows_ui, cols_ui = df_ui['user'], df_ui['item']
        data_ui = sp.csr_matrix((np.ones_like(rows_ui), (rows_ui, cols_ui)), dtype='float32',
                                shape=(n_users + 1, self.n_items))  # [# users, I] sparse matrix
        return data_ui

    def load_group_data_train(self):
        """ load training group-item interactions as a sparse matrix and user-group memberships """
        path_ug = [p for p in self.raw_paths if 'group_users.csv' in p][0]
        path_gi = [p for p in self.raw_paths if 'train_gi.csv' in p][0]

        df_gi = pd.read_csv(path_gi)  # load training group-item interactions.
        start_idx, end_idx = df_gi['group'].min(), df_gi['group'].max()
        self.n_groups = end_idx - start_idx + 1
        rows_gi, cols_gi = df_gi['group'] - start_idx, df_gi['item']

        data_gi = sp.csr_matrix((np.ones_like(rows_gi), (rows_gi, cols_gi)), dtype='float32',
                                shape=(self.n_groups, self.n_items))  # [# groups,  I] sparse matrix.

        df_ug = pd.read_csv(path_ug).astype(int)  # load user-group memberships.
        df_ug_train = df_ug[df_ug.group.isin(range(start_idx, end_idx + 1))]
        df_ug_train = df_ug_train.sort_values('group')  # sort in ascending order of group ids.
        self.max_group_size = df_ug_train.groupby('group').size().max()  # max group size denoted by G

        g_u_list_train = df_ug_train.groupby('group')['user'].apply(list).reset_index()
        g_u_list_train['user'] = list(map(lambda x: x + [self.padding_idx] * (self.max_group_size - len(x)),
                                          g_u_list_train.user))
        data_gu = np.squeeze(np.array(g_u_list_train[['user']].values.tolist()))  # [# groups, G] with padding.
        self.groups_list = list(range(0, end_idx - start_idx + 1))

        assert len(df_ug_train['group'].unique()) == self.n_groups
        print("# training groups: {}, # max train group size: {}".format(self.n_groups, self.max_group_size))

        return data_gi, data_gu

    def load_group_data_tr_te(self, datatype):
        """ load val/test group-item interactions as a sparse matrix and user-group memberships """
        path_ug = [p for p in self.raw_paths if 'group_users.csv' in p][0]
        path_gi = [p for p in self.raw_paths if '{}_gi.csv'.format(datatype) in p][0]

        df_gi = pd.read_csv(path_gi)  # load group-item interactions
        start_idx, end_idx = df_gi['group'].min(), df_gi['group'].max()
        self.n_groups = end_idx - start_idx + 1
        rows_gi, cols_gi = df_gi['group'] - start_idx, df_gi['item']
        data_gi = sp.csr_matrix((np.ones_like(rows_gi), (rows_gi, cols_gi)), dtype='float32',
                                shape=(self.n_groups, self.n_items))  # [# eval groups, I] sparse matrix

        df_ug = pd.read_csv(path_ug)  # load user-group memberships
        df_ug_eval = df_ug[df_ug.group.isin(range(start_idx, end_idx + 1))]
        df_ug_eval = df_ug_eval.sort_values('group')  # sort in ascending order of group ids
        self.max_gsize = df_ug_eval.groupby('group').size().max()  # max group size denoted by G
        g_u_list_eval = df_ug_eval.groupby('group')['user'].apply(list).reset_index()
        g_u_list_eval['user'] = list(map(lambda x: x + [self.padding_idx] * (self.max_gsize - len(x)),
                                         g_u_list_eval.user))
        data_gu = np.squeeze(np.array(g_u_list_eval[['user']].values.tolist(), dtype=np.int32))  # [# groups, G]
        self.eval_groups_list = list(range(0, end_idx - start_idx + 1))
        return data_gi, data_gu
```

<!-- #region id="OBjX6QsQyzww" -->
## Models
<!-- #endregion -->

```python id="oYz8jin7FPMV"
import torch
import torch.nn as nn
import torch.nn.functional as F
```

<!-- #region id="clxmhOVTFQXE" -->
### Encoder
<!-- #endregion -->

```python id="4aPpiMu-yzt8"
class Encoder(nn.Module):
    """ User Preference Encoder implemented as fully connected layers over binary bag-of-words vector
    (over item set) per user """

    def __init__(self, n_items, user_layers, embedding_dim, drop_ratio):
        super(Encoder, self).__init__()
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.drop = nn.Dropout(drop_ratio)
        self.user_preference_encoder = torch.nn.ModuleList()  # user individual preference encoder layers.

        for idx, (in_size, out_size) in enumerate(zip([self.n_items] + user_layers[:-1], user_layers)):
            layer = torch.nn.Linear(in_size, out_size, bias=True)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.user_preference_encoder.append(layer)

        self.transform_layer = nn.Linear(self.embedding_dim, self.embedding_dim)
        nn.init.xavier_uniform_(self.transform_layer.weight)
        nn.init.zeros_(self.transform_layer.bias)

        self.user_predictor = nn.Linear(self.embedding_dim, self.n_items, bias=False)  # item embedding for pre-training
        nn.init.xavier_uniform_(self.user_predictor.weight)

    def pre_train_forward(self, user_items):
        """ user individual preference encoder (excluding final layer) for user-item pre-training
            :param user_items: [B, G, I] or [B, I]
        """
        user_items_norm = F.normalize(user_items)  # [B, G, I] or [B, I]
        user_pref_embedding = self.drop(user_items_norm)
        for idx, _ in enumerate(range(len(self.user_preference_encoder))):
            user_pref_embedding = self.user_preference_encoder[idx](user_pref_embedding)  # [B, G, D] or [B, D]
            user_pref_embedding = torch.tanh(user_pref_embedding)  # [B, G, D] or [B, D]

        logits = self.user_predictor(user_pref_embedding)  # [B, G, D] or [B, D]
        return logits, user_pref_embedding

    def forward(self, user_items):
        """ user individual preference encoder
            :param user_items: [B, G, I]
        """
        _, user_embeds = self.pre_train_forward(user_items)  # [B, G, D]
        user_embeds = torch.tanh(self.transform_layer(user_embeds))  # [B, G, D]
        return user_embeds
```

<!-- #region id="IhLIxDOVFSkt" -->
### Aggregator
<!-- #endregion -->

```python id="Zf3EFFfrFShx"
class MaxPoolAggregator(nn.Module):
    """ Group Preference Aggregator implemented as max pooling over group member embeddings """

    def __init__(self, input_dim, output_dim, drop_ratio=0):
        super(MaxPoolAggregator, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(drop_ratio)
        )
        nn.init.xavier_uniform_(self.mlp[0].weight)
        if self.mlp[0].bias is not None:
            self.mlp[0].bias.data.fill_(0.0)

    def forward(self, x, mask, mlp=False):
        """ max pooling aggregator:
            :param x: [B, G, D]  group member embeddings
            :param mask: [B, G]  -inf/0 for absent/present
            :param mlp: flag to add a linear layer before max pooling
        """
        if mlp:
            h = torch.tanh(self.mlp(x))
        else:
            h = x

        if mask is None:
            return torch.max(h, dim=1)
        else:
            res = torch.max(h + mask.unsqueeze(2), dim=1)
            return res.values


# mask:  -inf/0 for absent/present.
class MeanPoolAggregator(nn.Module):
    """ Group Preference Aggregator implemented as mean pooling over group member embeddings """

    def __init__(self, input_dim, output_dim, drop_ratio=0):
        super(MeanPoolAggregator, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(drop_ratio)
        )
        nn.init.xavier_uniform_(self.mlp[0].weight)
        if self.mlp[0].bias is not None:
            self.mlp[0].bias.data.fill_(0.0)

    def forward(self, x, mask, mlp=False):
        """ mean pooling aggregator:
            :param x: [B, G, D]  group member embeddings
            :param mask: [B, G]  -inf/0 for absent/present
            :param mlp: flag to add a linear layer before mean pooling
        """
        if mlp:
            h = torch.tanh(self.mlp(x))
        else:
            h = x
        if mask is None:
            return torch.mean(h, dim=1)
        else:
            mask = torch.exp(mask)
            res = torch.sum(h * mask.unsqueeze(2), dim=1) / mask.sum(1).unsqueeze(1)
            return res


class AttentionAggregator(nn.Module):
    """ Group Preference Aggregator implemented as attention over group member embeddings """

    def __init__(self, input_dim, output_dim, drop_ratio=0):
        super(AttentionAggregator, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(drop_ratio)
        )

        self.attention = nn.Linear(output_dim, 1)
        self.drop = nn.Dropout(drop_ratio)
        nn.init.xavier_uniform_(self.mlp[0].weight)
        if self.mlp[0].bias is not None:
            self.mlp[0].bias.data.fill_(0.0)

    def forward(self, x, mask, mlp=False):
        """ attentive aggregator:
            :param x: [B, G, D]  group member embeddings
            :param mask: [B, G]  -inf/0 for absent/present
            :param mlp: flag to add a linear layer before attention
        """
        if mlp:
            h = torch.tanh(self.mlp(x))
        else:
            h = x

        attention_out = torch.tanh(self.attention(h))
        if mask is None:
            weight = torch.softmax(attention_out, dim=1)
        else:
            weight = torch.softmax(attention_out + mask.unsqueeze(2), dim=1)
        ret = torch.matmul(h.transpose(2, 1), weight).squeeze(2)
        return ret
```

<!-- #region id="ZDAiDRqFFSfk" -->
### Discriminator
<!-- #endregion -->

```python id="p8tP8cr3FSdk"
class Discriminator(nn.Module):
    """ Discriminator for Mutual Information Estimation and Maximization, implemented with bilinear layers and
    binary cross-entropy loss training """

    def __init__(self, embedding_dim=64):
        super(Discriminator, self).__init__()
        self.embedding_dim = embedding_dim

        self.fc_layer = torch.nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        nn.init.xavier_uniform_(self.fc_layer.weight)
        nn.init.zeros_(self.fc_layer.bias)

        self.bilinear_layer = nn.Bilinear(self.embedding_dim, self.embedding_dim, 1)  # output_dim = 1 => single score.
        nn.init.zeros_(self.bilinear_layer.weight)
        nn.init.zeros_(self.bilinear_layer.bias)

        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, group_inputs, user_inputs, group_mask):
        """ bilinear discriminator:
            :param group_inputs: [B, I]
            :param user_inputs: [B, n_samples, I] where n_samples is either G or # negs
            :param group_mask: [B, G]
        """
        # FC + activation.
        group_encoded = self.fc_layer(group_inputs)  # [B, D]
        group_embed = torch.tanh(group_encoded)  # [B, D]

        # FC + activation.
        user_pref_embedding = self.fc_layer(user_inputs)
        user_embed = torch.tanh(user_pref_embedding)  # [B, n_samples, D]

        return self.bilinear_layer(user_embed, group_embed.unsqueeze(1).repeat(1, user_inputs.shape[1], 1))

    def mi_loss(self, scores_group, group_mask, scores_corrupted, device='cpu'):
        """ binary cross-entropy loss over (group, user) pairs for discriminator training
            :param scores_group: [B, G]
            :param group_mask: [B, G]
            :param scores_corrupted: [B, N]
            :param device (cpu/gpu)
         """
        batch_size = scores_group.shape[0]
        pos_size, neg_size = scores_group.shape[1], scores_corrupted.shape[1]

        one_labels = torch.ones(batch_size, pos_size).to(device)  # [B, G]
        zero_labels = torch.zeros(batch_size, neg_size).to(device)  # [B, N]

        labels = torch.cat((one_labels, zero_labels), 1)  # [B, G+N]
        logits = torch.cat((scores_group, scores_corrupted), 1).squeeze(2)  # [B, G + N]

        mask = torch.cat((torch.exp(group_mask), torch.ones([batch_size, neg_size]).to(device)),
                         1)  # torch.exp(.) to binarize since original mask has -inf.

        mi_loss = self.bce_loss(logits * mask, labels * mask) * (batch_size * (pos_size + neg_size)) \
                  / (torch.exp(group_mask).sum() + batch_size * neg_size)

        return mi_loss
```

<!-- #region id="DUqPcXSjFhUh" -->
### GroupIM Model
<!-- #endregion -->

```python id="RrPrpdiQFhR2"
class GroupIM(nn.Module):
    """
    GroupIM framework for Group Recommendation:
    (a) User Preference encoding: user_preference_encoder
    (b) Group Aggregator: preference_aggregator
    (c) InfoMax Discriminator: discriminator
    """

    def __init__(self, n_items, user_layers, lambda_mi=0.1, drop_ratio=0.4, aggregator_type='attention'):
        super(GroupIM, self).__init__()
        self.n_items = n_items
        self.lambda_mi = lambda_mi
        self.drop = nn.Dropout(drop_ratio)
        self.embedding_dim = user_layers[-1]
        self.aggregator_type = aggregator_type

        self.user_preference_encoder = Encoder(self.n_items, user_layers, self.embedding_dim, drop_ratio)

        if self.aggregator_type == 'maxpool':
            self.preference_aggregator = MaxPoolAggregator(self.embedding_dim, self.embedding_dim)
        elif self.aggregator_type == 'meanpool':
            self.preference_aggregator = MeanPoolAggregator(self.embedding_dim, self.embedding_dim)
        elif self.aggregator_type == 'attention':
            self.preference_aggregator = AttentionAggregator(self.embedding_dim, self.embedding_dim)
        else:
            raise NotImplementedError("Aggregator type {} not implemented ".format(self.aggregator_type))

        self.group_predictor = nn.Linear(self.embedding_dim, self.n_items, bias=False)
        nn.init.xavier_uniform_(self.group_predictor.weight)

        self.discriminator = Discriminator(embedding_dim=self.embedding_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, group, group_users, group_mask, user_items):
        """ compute group embeddings and item recommendations by user preference encoding, group aggregation and
        item prediction
        :param group: [B] group id
        :param group_users: [B, G] group user ids with padding
        :param group_mask: [B, G] -inf/0 for absent/present user
        :param user_items: [B, G, I] individual item interactions of group members
        """
        user_pref_embeds = self.user_preference_encoder(user_items)
        group_embed = self.preference_aggregator(user_pref_embeds, group_mask, mlp=False)  # [B, D]
        group_logits = self.group_predictor(group_embed)  # [B, I]

        if self.train:
            obs_user_embeds = self.user_preference_encoder(user_items)  # [B, G, D]
            scores_ug = self.discriminator(group_embed, obs_user_embeds, group_mask).detach()  # [B, G]
            return group_logits, group_embed, scores_ug
        else:
            return group_logits, group_embed

    def multinomial_loss(self, logits, items):
        """ multinomial likelihood with softmax over item set """
        return -torch.mean(torch.sum(F.log_softmax(logits, 1) * items, -1))

    def user_loss(self, user_logits, user_items):
        return self.multinomial_loss(user_logits, user_items)

    def infomax_group_loss(self, group_logits, group_embeds, scores_ug, group_mask, group_items, user_items,
                           corrupted_user_items, device='cpu'):
        """ loss function with three terms: L_G, L_UG, L_MI
            :param group_logits: [B, G, I] group item predictions
            :param group_embeds: [B, D] group embedding
            :param scores_ug: [B, G] discriminator scores for group members
            :param group_mask: [B, G] -inf/0 for absent/present user
            :param group_items: [B, I] item interactions of group
            :param user_items: [B, G, I] individual item interactions of group members
            :param corrupted_user_items: [B, N, I] individual item interactions of negative user samples
            :param device: cpu/gpu
        """

        group_user_embeds = self.user_preference_encoder(user_items)  # [B, G, D]
        corrupt_user_embeds = self.user_preference_encoder(corrupted_user_items)  # [B, N, D]

        scores_observed = self.discriminator(group_embeds, group_user_embeds, group_mask)  # [B, G]
        scores_corrupted = self.discriminator(group_embeds, corrupt_user_embeds, group_mask)  # [B, N]

        mi_loss = self.discriminator.mi_loss(scores_observed, group_mask, scores_corrupted, device=device)

        ui_sum = user_items.sum(2, keepdim=True)  # [B, G]
        user_items_norm = user_items / torch.max(torch.ones_like(ui_sum), ui_sum)  # [B, G, I]
        gi_sum = group_items.sum(1, keepdim=True)
        group_items_norm = group_items / torch.max(torch.ones_like(gi_sum), gi_sum)  # [B, I]
        assert scores_ug.requires_grad is False

        group_mask_zeros = torch.exp(group_mask).unsqueeze(2)  # [B, G, 1]
        scores_ug = torch.sigmoid(scores_ug)  # [B, G, 1]

        user_items_norm = torch.sum(user_items_norm * scores_ug * group_mask_zeros, dim=1) / group_mask_zeros.sum(1)
        user_group_loss = self.multinomial_loss(group_logits, user_items_norm)
        group_loss = self.multinomial_loss(group_logits, group_items_norm)

        return mi_loss, user_group_loss, group_loss

    def loss(self, group_logits, summary_embeds, scores_ug, group_mask, group_items, user_items, corrupted_user_items,
             device='cpu'):
        """ L_G + lambda L_UG + L_MI """
        mi_loss, user_group_loss, group_loss = self.infomax_group_loss(group_logits, summary_embeds, scores_ug,
                                                                       group_mask, group_items, user_items,
                                                                       corrupted_user_items, device)

        return group_loss + mi_loss + self.lambda_mi * user_group_loss
```

<!-- #region id="r2oN9j_xy2xo" -->
## Trainers
<!-- #endregion -->

```python id="yW_3qhU2J2Xe"
import torch
import numpy as np
import gc
```

```python id="8FzcPqcWy20h"
def ndcg_binary_at_k_batch_torch(X_pred, heldout_batch, k=100, device='cpu'):
    """
    Normalized Discounted Cumulative Gain@k for for predictions [B, I] and ground-truth [B, I], with binary relevance.
    ASSUMPTIONS: all the 0's in heldout_batch indicate 0 relevance.
    """

    batch_users = X_pred.shape[0]  # batch_size
    _, idx_topk = torch.topk(X_pred, k, dim=1, sorted=True)
    tp = 1. / torch.log2(torch.arange(2, k + 2, device=device).float())
    heldout_batch_nonzero = (heldout_batch > 0).float()
    DCG = (heldout_batch_nonzero[torch.arange(batch_users, device=device).unsqueeze(1), idx_topk] * tp).sum(dim=1)
    heldout_nonzero = (heldout_batch > 0).sum(dim=1)  # num. of non-zero items per batch. [B]
    IDCG = torch.tensor([(tp[:min(n, k)]).sum() for n in heldout_nonzero]).to(device)
    return DCG / IDCG


def recall_at_k_batch_torch(X_pred, heldout_batch, k=100):
    """
    Recall@k for predictions [B, I] and ground-truth [B, I].
    """
    batch_users = X_pred.shape[0]
    _, topk_indices = torch.topk(X_pred, k, dim=1, sorted=False)  # [B, K]
    X_pred_binary = torch.zeros_like(X_pred)
    if torch.cuda.is_available():
        X_pred_binary = X_pred_binary.cuda()
    X_pred_binary[torch.arange(batch_users).unsqueeze(1), topk_indices] = 1
    X_true_binary = (heldout_batch > 0).float()  # .toarray() #  [B, I]
    k_tensor = torch.tensor([k], dtype=torch.float32)
    if torch.cuda.is_available():
        X_true_binary = X_true_binary.cuda()
        k_tensor = k_tensor.cuda()
    tmp = (X_true_binary * X_pred_binary).sum(dim=1).float()
    recall = tmp / torch.min(k_tensor, X_true_binary.sum(dim=1).float())
    return recall
```

```python id="mrgOyG7sKRMi"
def evaluate_user(model, eval_loader, device, mode='pretrain'):
    """ evaluate model on recommending items to users (primarily during pre-training step) """
    model.eval()
    eval_loss = 0.0
    n100_list, r20_list, r50_list = [], [], []
    eval_preds = []
    with torch.no_grad():
        for batch_index, eval_data in enumerate(eval_loader):
            eval_data = [x.to(device, non_blocking=True) for x in eval_data]
            (users, fold_in_items, held_out_items) = eval_data
            fold_in_items = fold_in_items.to(device)
            if mode == 'pretrain':
                recon_batch, emb = model.user_preference_encoder.pre_train_forward(fold_in_items)
            else:
                recon_batch = model.group_predictor(model.user_preference_encoder(fold_in_items))

            loss = model.multinomial_loss(recon_batch, held_out_items)
            eval_loss += loss.item()
            fold_in_items = fold_in_items.cpu().numpy()
            recon_batch = torch.softmax(recon_batch, 1)  # softmax over the item set to get normalized scores.
            recon_batch[fold_in_items.nonzero()] = -np.inf

            n100 = ndcg_binary_at_k_batch_torch(recon_batch, held_out_items, 100, device=device)
            r20 = recall_at_k_batch_torch(recon_batch, held_out_items, 20)
            r50 = recall_at_k_batch_torch(recon_batch, held_out_items, 50)

            n100_list.append(n100)
            r20_list.append(r20)
            r50_list.append(r50)

            eval_preds.append(recon_batch.cpu().numpy())
            del users, fold_in_items, held_out_items, recon_batch
    gc.collect()
    num_batches = max(1, len(eval_loader.dataset) / eval_loader.batch_size)
    eval_loss /= num_batches
    n100_list = torch.cat(n100_list)
    r20_list = torch.cat(r20_list)
    r50_list = torch.cat(r50_list)
    return eval_loss, torch.mean(n100_list), torch.mean(r20_list), torch.mean(r50_list), np.array(eval_preds)


def evaluate_group(model, eval_group_loader, device):
    """ evaluate model on recommending items to groups """
    model.eval()
    eval_loss = 0.0
    n100_list, r20_list, r50_list = [], [], []
    eval_preds = []

    with torch.no_grad():
        for batch_idx, data in enumerate(eval_group_loader):
            data = [x.to(device, non_blocking=True) for x in data]
            group, group_users, group_mask, group_items, user_items = data
            recon_batch, _, _ = model(group, group_users, group_mask, user_items)

            loss = model.multinomial_loss(recon_batch, group_items)
            eval_loss += loss.item()
            result = recon_batch.softmax(1)  # softmax over the item set to get normalized scores.
            heldout_data = group_items

            r20 = recall_at_k_batch_torch(result, heldout_data, 20)
            r50 = recall_at_k_batch_torch(result, heldout_data, 50)
            n100 = ndcg_binary_at_k_batch_torch(result, heldout_data, 100, device=device)

            n100_list.append(n100)
            r20_list.append(r20)
            r50_list.append(r50)

            eval_preds.append(recon_batch.cpu().numpy())
            del group, group_users, group_mask, group_items, user_items
    gc.collect()

    n100_list = torch.cat(n100_list)
    r20_list = torch.cat(r20_list)
    r50_list = torch.cat(r50_list)
    return eval_loss, torch.mean(n100_list), torch.mean(r20_list), torch.mean(r50_list), np.array(eval_preds)
```

<!-- #region id="kl-g_0UuFzqC" -->
## Experiments
<!-- #endregion -->

```python id="soNLg-ekF2Y9"
import argparse
import time
import gc
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
```

```python id="ZLKloOcQF3UQ"
if torch.cuda.is_available():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    gpu_id = int(np.argmax(memory_available))
    torch.cuda.set_device(gpu_id)
```

```python id="hwBY4kJkGXm3"
class Args:

    # Dataset
    dataset = 'weeplaces'
    root = '/content/data'

    # Training settings
    lr = 5e-3 # initial learning rate
    wd = 0.00 # weight decay coefficient
    lambda_mi = 1.0 # MI lambda hyper param
    drop_ratio = 0.4 # Dropout ratio
    batch_size = 256 # batch size
    epochs = 20 # maximum # training epochs
    eval_freq = 5 # frequency to evaluate performance on validation set

    # Model settings
    emb_size = 64 # layer size
    aggregator = 'attention' # choice of group preference aggregator', choices=['maxpool', 'meanpool', 'attention']
    negs_per_group = 5 # negative users sampled per group

    # Pre-training settings
    pretrain_user = True # Pre-train user encoder on user-item interactions
    pretrain_mi = True # Pre-train MI estimator for a few epochs
    pretrain_epochs = 10 # pre-train epochs for user encoder layer

    cuda = True # use CUDA
    seed = 1111 # random seed for reproducibility

    # Model save file parameters
    save = 'model_user.pt' # path to save the final model
    save_group = 'model_group.pt' # path to save the final model

args = Args()
```

```python id="bbyhnoy2GLFR"
torch.manual_seed(args.seed)  # Set the random seed manually for reproducibility.

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
```

```python colab={"base_uri": "https://localhost:8080/"} id="DVrUjYsSH0n1" executionInfo={"status": "ok", "timestamp": 1640782074661, "user_tz": -330, "elapsed": 5215, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5723723d-7f41-4de1-f6cd-5edf255fd60b"
###############################################################################
# Load data
###############################################################################

train_params = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': 2, 'pin_memory': True}
eval_params = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': 2, 'pin_memory': True}
device = torch.device("cuda" if args.cuda else "cpu")

# Define train/val/test datasets on user interactions.
train_dataset = WeeplacesDataset(args.root, is_group=False, datatype='train')  # train dataset for user-item interactions.
n_users, n_items = train_dataset.n_users, train_dataset.n_items
val_dataset = WeeplacesDataset(args.root, is_group=False, datatype='val', n_items=n_items)
test_dataset = WeeplacesDataset(args.root, is_group=False, datatype='test', n_items=n_items)

# Define train/val/test datasets on group and user interactions.
train_group_dataset = WeeplacesDataset(args.root, is_group=True, datatype='train', negs_per_group=args.negs_per_group, n_items=n_items)
padding_idx = train_group_dataset.padding_idx
val_group_dataset = WeeplacesDataset(args.root, is_group=True, datatype='val', n_items=n_items, padding_idx=padding_idx)
test_group_dataset = WeeplacesDataset(args.root, is_group=True, datatype='test', n_items=n_items, padding_idx=padding_idx)

# Define data loaders on user interactions.
train_loader = DataLoader(train_dataset, **train_params)
val_loader = DataLoader(val_dataset, **eval_params)
test_loader = DataLoader(test_dataset, **eval_params)

# Define data loaders on group interactions.
train_group_loader = DataLoader(train_group_dataset, **train_params)
val_group_loader = DataLoader(val_group_dataset, **eval_params)
test_group_loader = DataLoader(test_group_dataset, **eval_params)
```

```python id="mZnxV1iJIU2a"
###############################################################################
# Build the model
###############################################################################

user_layers = [args.emb_size]  # user encoder layer configuration is tunable.

model = GroupIM(n_items, user_layers, drop_ratio=args.drop_ratio, aggregator_type=args.aggregator,
                lambda_mi=args.lambda_mi).to(device)
optimizer_gr = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

best_user_n100, best_group_n100 = -np.inf, -np.inf
```

```python id="bYZBkv7NORbK"
import warnings
warnings.filterwarnings('ignore')
```

```python colab={"base_uri": "https://localhost:8080/"} id="IZZUqXQBIzYm" outputId="acd2ab8f-cf51-440b-f3f9-68f6f9b8e031" executionInfo={"status": "ok", "timestamp": 1640785403536, "user_tz": -330, "elapsed": 3328901, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
if args.pretrain_user:
    optimizer_ur = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=args.wd)
    print("Pre-training model on user-item interactions")
    for epoch in range(0, args.pretrain_epochs):
        epoch_start_time = time.time()
        model.train()
        train_user_loss = 0.0
        start_time = time.time()

        for batch_index, data in enumerate(train_loader):
            optimizer_ur.zero_grad()
            data = [x.to(device, non_blocking=True) for x in data]
            (train_users, train_items) = data
            user_logits, user_embeds = model.user_preference_encoder.pre_train_forward(train_items)
            user_loss = model.user_loss(user_logits, train_items)
            user_loss.backward()
            train_user_loss += user_loss.item()
            optimizer_ur.step()
            del train_users, train_items, user_logits, user_embeds
        elapsed = time.time() - start_time
        print('| epoch {:3d} |  time {:4.2f} | loss {:4.2f}'.format(epoch + 1, elapsed,
                                                                    train_user_loss / len(train_loader)))
        if epoch % args.eval_freq == 0:
            val_loss, n100, r20, r50, _ = evaluate_user(model, val_loader, device, mode='pretrain')

            if n100 > best_user_n100:
                torch.save(model.state_dict(), args.save)
                best_user_n100 = n100

    print("Load best pre-trained user encoder")
    model.load_state_dict(torch.load(args.save))
    model = model.to(device)

    val_loss, n100, r20, r50, _ = evaluate_user(model, val_loader, device, mode='pretrain')
    print('=' * 89)
    print('| User evaluation | val loss {:4.4f} | n100 {:4.4f} | r20 {:4.4f} | '
            'r50 {:4.4f}'.format(val_loss, n100, r20, r50))
    print("Initializing group recommender with pre-train user encoder")
    # Initialize the group predictor (item embedding) weight based on the pre-trained user predictor.
    model.group_predictor.weight.data = model.user_preference_encoder.user_predictor.weight.data

if args.pretrain_mi:
    # pre-train MI estimator.
    for epoch in range(0, 10):
        model.train()
        t = time.time()
        mi_epoch_loss = 0.0
        for batch_index, data in enumerate(train_group_loader):
            data = [x.to(device, non_blocking=True) for x in data]
            group, group_users, group_mask, group_items, user_items, corrupted_user_items = data
            optimizer_gr.zero_grad()
            model.zero_grad()
            model.train()
            _, group_embeds, _ = model(group, group_users, group_mask, user_items)
            obs_user_embed = model.user_preference_encoder(user_items).detach()  # [B, G, D]
            corrupted_user_embed = model.user_preference_encoder(corrupted_user_items).detach()  # [B, # negs, D]

            scores_observed = model.discriminator(group_embeds, obs_user_embed, group_mask)  # [B, G]
            scores_corrupted = model.discriminator(group_embeds, corrupted_user_embed, group_mask)  # [B, # negs]

            mi_loss = model.discriminator.mi_loss(scores_observed, group_mask, scores_corrupted, device=device)
            mi_loss.backward()
            optimizer_gr.step()
            mi_epoch_loss += mi_loss
            del group, group_users, group_mask, group_items, user_items, corrupted_user_items, \
                obs_user_embed, corrupted_user_embed
        gc.collect()
        print("MI loss: {}".format(float(mi_epoch_loss) / len(train_group_loader)))

optimizer_gr = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

for epoch in range(0, args.epochs):
    epoch_start_time = time.time()
    model.train()
    train_group_epoch_loss = 0.0
    for batch_index, data in enumerate(train_group_loader):
        data = [x.to(device, non_blocking=True) for x in data]
        group, group_users, group_mask, group_items, user_items, corrupted_user_items = data
        optimizer_gr.zero_grad()
        model.zero_grad()
        group_logits, group_embeds, scores_ug = model(group.squeeze(), group_users, group_mask, user_items)
        group_loss = model.loss(group_logits, group_embeds, scores_ug, group_mask, group_items, user_items,
                                corrupted_user_items, device=device)
        group_loss.backward()
        train_group_epoch_loss += group_loss.item()
        optimizer_gr.step()
        del group, group_users, group_mask, group_items, user_items, corrupted_user_items, \
            group_logits, group_embeds, scores_ug

    gc.collect()

    print("Train loss: {}".format(float(train_group_epoch_loss) / len(train_group_loader)))

    if epoch % args.eval_freq == 0:
        # Group evaluation.
        val_loss_group, n100_group, r20_group, r50_group, _ = evaluate_group(model, val_group_loader, device)

        print('-' * 89)
        print('| end of epoch {:3d} | time: {:4.2f}s | n100 (group) {:5.4f} | r20 (group) {:5.4f} | r50 (group) '
                '{:5.4f}'.format(epoch + 1, time.time() - epoch_start_time, n100_group, r20_group, r50_group))
        print('-' * 89)

        # Save the model if the n100 is the best we've seen so far.
        if n100_group > best_group_n100:
            with open(args.save_group, 'wb') as f:
                torch.save(model, f)
            best_group_n100 = n100_group
```

```python id="z7MeW-auK822" colab={"base_uri": "https://localhost:8080/"} outputId="2382dde5-3852-4e53-f4c4-b7320761dba2"
# Load the best saved model.
with open(args.save_group, 'rb') as f:
    model = torch.load(f, map_location='cuda')
    model = model.to(device)

# Best validation evaluation
val_loss, n100, r20, r50, _ = evaluate_user(model, val_loader, device, mode='group')
print('=' * 89)
print('| User evaluation | val loss {:4.4f} | n100 {:4.4f} | r20 {:4.4f} | r50 {:4.4f}'
      .format(val_loss, n100, r20, r50))

# Test evaluation
test_loss, n100, r20, r50, _ = evaluate_user(model, test_loader, device, mode='group')
print('=' * 89)
print('| User evaluation | test loss {:4.4f} | n100 {:4.4f} | r20 {:4.4f} | r50 {:4.4f}'
      .format(test_loss, n100, r20, r50))

print('=' * 89)
_, n100_group, r20_group, r50_group, _ = evaluate_group(model, val_group_loader, device)
print('| Group evaluation (val) | n100 (group) {:4.4f} | r20 (group) {:4.4f} | r50 (group) {:4.4f}'
      .format(n100_group, r20_group, r50_group))

print('=' * 89)
_, n100_group, r20_group, r50_group, _ = evaluate_group(model, test_group_loader, device)
print('| Group evaluation (test) | n100 (group) {:4.4f} | r20 (group) {:4.4f} | r50 (group) {:4.4f}'
      .format(n100_group, r20_group, r50_group))
```

<!-- #region id="1y5VtjP9cDjn" -->
---
<!-- #endregion -->

```python id="ErtS_8LncDjp"
!apt-get -qq install tree
```

```python colab={"base_uri": "https://localhost:8080/"} id="wPdV96gQcDjq" executionInfo={"status": "ok", "timestamp": 1640785622982, "user_tz": -330, "elapsed": 536, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="cbde2ebb-8ae3-4099-ec9f-5c1beaa4bdb1"
!tree -h --du -C .
```

```python colab={"base_uri": "https://localhost:8080/"} id="DD0Xu8BocDjs" executionInfo={"status": "ok", "timestamp": 1640785647137, "user_tz": -330, "elapsed": 3387, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="805706f3-e262-4530-88ff-2a8923441c1c"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="6ws-JR3acDjt" -->
---
<!-- #endregion -->

<!-- #region id="dYtXGUdocDju" -->
**END**
<!-- #endregion -->
