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

<!-- #region id="P8_N8CXrApV9" -->
# MKR
<!-- #endregion -->

<!-- #region id="1_H4_cJqA7Y_" -->
## Imports
<!-- #endregion -->

```python id="rjFG6wmlA7Vs"
import numpy as np
import os
from tqdm import tqdm
from abc import abstractmethod

import sys
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
# from trace_grad import plot_grad_flow

import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
```

<!-- #region id="NorXOGuyBKOM" -->
## Preprocessing
<!-- #endregion -->

```python id="5eApT8YjChHH"
# !git clone https://github.com/hsientzucheng/MKR.PyTorch.git
# !mv MKR.PyTorch/data/* /content
```

```python id="esqzN_l2f0io"
!git clone https://github.com/sparsh-ai/multiobjective-optimizations.git
!mv multiobjective-optimizations/data/book .
!mv multiobjective-optimizations/data/movie .
!mv multiobjective-optimizations/data/music .
```

```python id="ppNWjSvKBWH4"
class Args:
    datapath = '/content/'
    DATASET = 'book'
    RATING_FILE_NAME = dict({'movie': 'ratings.dat',
                         'book': 'BX-Book-Ratings.csv',
                         'music': 'user_artists.dat',
                         'news': 'ratings.txt'})
    SEP = dict({'movie': '::', 'book': ';', 'music': '\t', 'news': '\t'})
    THRESHOLD = dict({'movie': 4, 'book': 0, 'music': 0, 'news': 0})


args = Args()
```

```python colab={"base_uri": "https://localhost:8080/"} id="4vSyQUDTBKKH" executionInfo={"status": "ok", "timestamp": 1634925327270, "user_tz": -330, "elapsed": 31841, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3ab638f0-6800-43d3-ad14-89614b0e64e2"
def read_item_index_to_entity_id_file():
    file = args.datapath + args.DATASET + '/item_index2entity_id.txt'
    print('reading item index to entity id file: ' + file + ' ...')
    i = 0
    for line in open(file, encoding='utf-8').readlines():
        item_index = line.strip().split('\t')[0]
        satori_id = line.strip().split('\t')[1]
        item_index_old2new[item_index] = i
        entity_id2index[satori_id] = i
        i += 1


def convert_rating():
    file = args.datapath + args.DATASET + '/' + args.RATING_FILE_NAME[args.DATASET]

    print('reading rating file ...')
    item_set = set(item_index_old2new.values())
    user_pos_ratings = dict()
    user_neg_ratings = dict()

    for line in open(file, encoding='utf-8').readlines()[1:]:
        array = line.strip().split(args.SEP[args.DATASET])

        # remove prefix and suffix quotation marks for BX dataset
        if args.DATASET == 'book':
            array = list(map(lambda x: x[1:-1], array))

        item_index_old = array[1]
        if item_index_old not in item_index_old2new:  # the item is not in the final item set
            continue
        item_index = item_index_old2new[item_index_old]

        user_index_old = int(array[0])

        rating = float(array[2])
        if rating >= args.THRESHOLD[args.DATASET]:
            if user_index_old not in user_pos_ratings:
                user_pos_ratings[user_index_old] = set()
            user_pos_ratings[user_index_old].add(item_index)
        else:
            if user_index_old not in user_neg_ratings:
                user_neg_ratings[user_index_old] = set()
            user_neg_ratings[user_index_old].add(item_index)

    print('converting rating file ...')
    writer = open(args.datapath + args.DATASET + '/ratings_final.txt', 'w', encoding='utf-8')
    user_cnt = 0
    user_index_old2new = dict()
    for user_index_old, pos_item_set in user_pos_ratings.items():
        if user_index_old not in user_index_old2new:
            user_index_old2new[user_index_old] = user_cnt
            user_cnt += 1
        user_index = user_index_old2new[user_index_old]

        for item in pos_item_set:
            writer.write('%d\t%d\t1\n' % (user_index, item))
        unwatched_set = item_set - pos_item_set
        if user_index_old in user_neg_ratings:
            unwatched_set -= user_neg_ratings[user_index_old]
        for item in np.random.choice(list(unwatched_set), size=len(pos_item_set), replace=False):
            writer.write('%d\t%d\t0\n' % (user_index, item))
    writer.close()
    print('number of users: %d' % user_cnt)
    print('number of items: %d' % len(item_set))


def convert_kg():
    print('converting kg.txt file ...')
    entity_cnt = len(entity_id2index)
    relation_cnt = 0

    writer = open(args.datapath + args.DATASET + '/kg_final.txt', 'w', encoding='utf-8')
    file = open(args.datapath + args.DATASET + '/kg.txt', encoding='utf-8')

    for line in file:
        array = line.strip().split('\t')
        head_old = array[0]
        relation_old = array[1]
        tail_old = array[2]

        if head_old not in entity_id2index:
            continue
        head = entity_id2index[head_old]

        if tail_old not in entity_id2index:
            entity_id2index[tail_old] = entity_cnt
            entity_cnt += 1
        tail = entity_id2index[tail_old]

        if relation_old not in relation_id2index:
            relation_id2index[relation_old] = relation_cnt
            relation_cnt += 1
        relation = relation_id2index[relation_old]

        writer.write('%d\t%d\t%d\n' % (head, relation, tail))

    writer.close()
    print('number of entities (containing items): %d' % entity_cnt)
    print('number of relations: %d' % relation_cnt)


if __name__ == '__main__':
    np.random.seed(555)

    entity_id2index = dict()
    relation_id2index = dict()
    item_index_old2new = dict()

    read_item_index_to_entity_id_file()
    convert_rating()
    convert_kg()

    print('done')
```

<!-- #region id="paBHi79_GtyH" -->
## Dataloader
<!-- #endregion -->

```python id="r9hyrJaQGwAU"
class RSDataset:
    def __init__(self, args):
        self.args = args
        self.n_user, self.n_item, self.raw_data, self.data, self.indices = self._load_rating()

    def __getitem__(self, index):
        return self.raw_data[index]

    def _load_rating(self):
        print('Reading rating file')

        rating_file = os.path.join(self.args.dataset, 'ratings_final')
        if os.path.exists(rating_file + '.npy'):
            rating_np = np.load(rating_file + '.npy')
        else:
            rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int32)
            np.save(rating_file + '.npy', rating_np)

        n_user = len(set(rating_np[:, 0]))
        n_item = len(set(rating_np[:, 1]))
        raw_data, data, indices = self._dataset_split(rating_np)

        return n_user, n_item, raw_data, data, indices


    def _dataset_split(self, rating_np):
        print('Splitting dataset')

        # train:eval:test = 6:2:2
        eval_ratio = 0.2
        test_ratio = 0.2
        n_ratings = rating_np.shape[0]

        eval_indices = np.random.choice(list(range(n_ratings)), size=int(n_ratings * eval_ratio), replace=False)
        left = set(range(n_ratings)) - set(eval_indices)
        test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
        train_indices = list(left - set(test_indices))

        train_data = rating_np[train_indices]
        eval_data = rating_np[eval_indices]
        test_data = rating_np[test_indices]

        return rating_np, [train_data, eval_data, test_data], [train_indices, eval_indices, test_indices]

class KGDataset:
    def __init__(self, args):
        self.args = args
        self.n_entity, self.n_relation, self.kg = self._load_kg()

    def __getitem__(self, index):
        return self.kg[index]

    def __len__(self):
        return len(self.kg)

    def _load_kg(self):
        print('Reading KG file')

        kg_file = os.path.join(self.args.dataset, 'kg_final')
        if os.path.exists(kg_file + '.npy'):
            kg = np.load(kg_file + '.npy')
        else:
            kg = np.loadtxt(kg_file + '.txt', dtype=np.int32)
            np.save(kg_file + '.npy', kg)

        n_entity = len(set(kg[:, 0]) | set(kg[:, 2]))
        n_relation = len(set(kg[:, 1]))

        return n_entity, n_relation, kg
```

<!-- #region id="SrsWQL25GiWj" -->
## Layers
<!-- #endregion -->

```python id="o-4MmZLXGjrW"
LAYER_IDS = {}

class Dense(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.0, chnl=8):
        super(Dense, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = nn.ReLU()
        self.drop_layer = nn.Dropout(p=self.dropout) # Pytorch drop: ratio to zeroed
        self.fc = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, inputs):
        x = self.drop_layer(inputs)
        output = self.fc(x)
        return self.act(output)

class Bias(nn.Module):
    def __init__(self, dim):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        x = x + self.bias
        return x


class CrossCompressUnit(nn.Module):
    def __init__(self, dim):
        super(CrossCompressUnit, self).__init__()
        self.dim = dim
        self.fc_vv = nn.Linear(dim, 1, bias=False)
        self.fc_ev = nn.Linear(dim, 1, bias=False)
        self.fc_ve = nn.Linear(dim, 1, bias=False)
        self.fc_ee = nn.Linear(dim, 1, bias=False)

        self.bias_v = Bias(dim)
        self.bias_e = Bias(dim)

        # self.fc_v = nn.Linear(dim, dim)
        # self.fc_e = nn.Linear(dim, dim)

    def forward(self, inputs):
        v, e = inputs

        # [batch_size, dim, 1], [batch_size, 1, dim]
        v = torch.unsqueeze(v, 2)
        e = torch.unsqueeze(e, 1)

        # [batch_size, dim, dim]
        c_matrix = torch.matmul(v, e)
        c_matrix_transpose = c_matrix.permute(0,2,1)

        # [batch_size * dim, dim]
        c_matrix = c_matrix.view(-1, self.dim)
        c_matrix_transpose = c_matrix_transpose.contiguous().view(-1, self.dim)

        # [batch_size, dim]
        v_intermediate = self.fc_vv(c_matrix) + self.fc_ev(c_matrix_transpose)
        e_intermediate = self.fc_ve(c_matrix) + self.fc_ee(c_matrix_transpose)
        v_intermediate = v_intermediate.view(-1, self.dim)
        e_intermediate = e_intermediate.view(-1, self.dim)

        v_output = self.bias_v(v_intermediate)
        e_output = self.bias_e(e_intermediate)

        # v_output = self.fc_v(v_intermediate)
        # e_output = self.fc_e(e_intermediate)


        return v_output, e_output
```

<!-- #region id="Ai3xfyXnA7S8" -->
## Model
<!-- #endregion -->

```python id="wBl8B-3PA7QI"
class MKR_model(nn.Module):
    def __init__(self, args, n_user, n_item, n_entity, n_relation, use_inner_product=True):
        super(MKR_model, self).__init__()

        # <Lower Model>
        self.args = args
        self.n_user = n_user
        self.n_item = n_item
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.use_inner_product = use_inner_product

        # Init embeddings
        self.user_embeddings_lookup = nn.Embedding(self.n_user, self.args.dim)
        self.item_embeddings_lookup = nn.Embedding(self.n_item, self.args.dim)
        self.entity_embeddings_lookup = nn.Embedding(self.n_entity, self.args.dim)
        self.relation_embeddings_lookup = nn.Embedding(self.n_relation, self.args.dim)

        self.user_mlp = nn.Sequential()
        self.tail_mlp = nn.Sequential()
        self.cc_unit = nn.Sequential()
        for i_cnt in range(self.args.L):
            self.user_mlp.add_module('user_mlp{}'.format(i_cnt),
                                     Dense(self.args.dim, self.args.dim))
            self.tail_mlp.add_module('tail_mlp{}'.format(i_cnt),
                                     Dense(self.args.dim, self.args.dim))
            self.cc_unit.add_module('cc_unit{}'.format(i_cnt),
                                     CrossCompressUnit(self.args.dim))
        # <Higher Model>
        self.kge_pred_mlp = Dense(self.args.dim * 2, self.args.dim)
        self.kge_mlp = nn.Sequential()
        for i_cnt in range(self.args.H - 1):
            self.kge_mlp.add_module('kge_mlp{}'.format(i_cnt),
                                    Dense(self.args.dim * 2, self.args.dim * 2))
        if self.use_inner_product==False:
            self.rs_pred_mlp = Dense(self.args.dim * 2, 1)
            self.rs_mlp = nn.Sequential()
            for i_cnt in range(self.args.H - 1):
                self.rs_mlp.add_module('rs_mlp{}'.format(i_cnt),
                                       Dense(self.args.dim * 2, self.args.dim * 2))

    def forward(self, user_indices=None, item_indices=None, head_indices=None,
            relation_indices=None, tail_indices=None):

        # <Lower Model>

        if user_indices is not None:
            self.user_indices = user_indices
        if item_indices is not None:
            self.item_indices = item_indices
        if head_indices is not None:
            self.head_indices = head_indices
        if relation_indices is not None:
            self.relation_indices = relation_indices
        if tail_indices is not None:
            self.tail_indices = tail_indices

        # Embeddings
        self.item_embeddings = self.item_embeddings_lookup(self.item_indices)
        self.head_embeddings = self.entity_embeddings_lookup(self.head_indices)
        self.item_embeddings, self.head_embeddings = self.cc_unit([self.item_embeddings, self.head_embeddings])

        # <Higher Model>
        if user_indices is not None:
            # RS
            self.user_embeddings = self.user_embeddings_lookup(self.user_indices)
            self.user_embeddings = self.user_mlp(self.user_embeddings)
            if self.use_inner_product:
                # [batch_size]
                self.scores = torch.sum(self.user_embeddings * self.item_embeddings, 1)
            else:
                # [batch_size, dim * 2]
                self.user_item_concat = torch.cat([self.user_embeddings, self.item_embeddings], 1)
                self.user_item_concat = self.rs_mlp(self.user_item_concat)
                # [batch_size]
                self.scores = torch.squeeze(self.rs_pred_mlp(self.user_item_concat))
            self.scores_normalized = torch.sigmoid(self.scores)
            outputs = [self.user_embeddings, self.item_embeddings, self.scores, self.scores_normalized]
        if relation_indices is not None:
            # KGE
            self.tail_embeddings = self.entity_embeddings_lookup(self.tail_indices)
            self.relation_embeddings = self.relation_embeddings_lookup(self.relation_indices)
            self.tail_embeddings = self.tail_mlp(self.tail_embeddings)
            # [batch_size, dim * 2]
            self.head_relation_concat = torch.cat([self.head_embeddings, self.relation_embeddings], 1)
            self.head_relation_concat = self.kge_mlp(self.head_relation_concat)
            # [batch_size, 1]
            self.tail_pred = self.kge_pred_mlp(self.head_relation_concat)
            self.tail_pred = torch.sigmoid(self.tail_pred)
            self.scores_kge = torch.sigmoid(torch.sum(self.tail_embeddings * self.tail_pred, 1))
            self.rmse = torch.mean(
                torch.sqrt(torch.sum(torch.pow(self.tail_embeddings -
                           self.tail_pred, 2), 1) / self.args.dim))
            outputs = [self.head_embeddings, self.tail_embeddings, self.scores_kge, self.rmse]

        return outputs


class MKR(object):
    def __init__(self, args, n_users, n_items, n_entities,
                 n_relations):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._parse_args(n_users, n_items, n_entities, n_relations)
        self._build_model()
        self._build_loss()
        self._build_ops()

    def _parse_args(self, n_users, n_items, n_entities, n_relations):
        self.n_user = n_users
        self.n_item = n_items
        self.n_entity = n_entities
        self.n_relation = n_relations

    def _build_model(self):
        print("Build models")
        self.MKR_model = MKR_model(self.args, self.n_user, self.n_item, self.n_entity, self.n_relation)
        self.MKR_model = self.MKR_model.to(self.device, non_blocking=True)
        for m in self.MKR_model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding):
                torch.nn.init.xavier_uniform_(m.weight)
        # for param in self.MKR_model.parameters():
        #     param.requires_grad = True

    def _build_loss(self):
        self.sigmoid_BCE = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

    def _build_ops(self):
        self.optimizer_rs = torch.optim.Adam(self.MKR_model.parameters(),
                                             lr=self.args.lr_rs)
        self.optimizer_kge = torch.optim.Adam(self.MKR_model.parameters(),
                                              lr=self.args.lr_kge)

    def _inference_rs(self, inputs):
        # Inputs
        self.user_indices = inputs[:, 0].long().to(self.device,
                non_blocking=True)
        self.item_indices = inputs[:, 1].long().to(self.device,
                non_blocking=True)
        labels = inputs[:, 2].float().to(self.device)
        self.head_indices = inputs[:, 1].long().to(self.device,
                non_blocking=True)

        # Inference
        outputs = self.MKR_model(user_indices=self.user_indices,
                                 item_indices=self.item_indices,
                                 head_indices=self.head_indices,
                                 relation_indices=None,
                                 tail_indices=None)

        user_embeddings, item_embeddings, scores, scores_normalized = outputs
        return user_embeddings, item_embeddings, scores, scores_normalized, labels

    def _inference_kge(self, inputs):
        # Inputs
        self.item_indices = inputs[:, 0].long().to(self.device,
                non_blocking=True)
        self.head_indices = inputs[:, 0].long().to(self.device,
                non_blocking=True)
        self.relation_indices = inputs[:, 1].long().to(self.device,
                non_blocking=True)
        self.tail_indices = inputs[:, 2].long().to(self.device,
                non_blocking=True)

        # Inference
        outputs = self.MKR_model(user_indices=None,
                                 item_indices=self.item_indices,
                                 head_indices=self.head_indices,
                                 relation_indices=self.relation_indices,
                                 tail_indices=self.tail_indices)

        head_embeddings, tail_embeddings, scores_kge, rmse = outputs
        return head_embeddings, tail_embeddings, scores_kge, rmse

    def l2_loss(self, inputs):
        return torch.sum(inputs ** 2) / 2

    def loss_rs(self, user_embeddings, item_embeddings, scores, labels):
        # scores_for_signll = torch.cat([1-self.sigmoid(scores).unsqueeze(1),
        #                                self.sigmoid(scores).unsqueeze(1)], 1)
        # base_loss_rs = torch.mean(self.nll_loss(scores_for_signll, labels))
        base_loss_rs = torch.mean(self.sigmoid_BCE(scores, labels))
        l2_loss_rs = self.l2_loss(user_embeddings) + self.l2_loss(item_embeddings)
        for name, param in self.MKR_model.named_parameters():
            if param.requires_grad and ('embeddings_lookup' not in name) \
                    and (('rs' in name) or ('cc_unit' in name) or ('user' in name)) \
                    and ('weight' in name):
                l2_loss_rs = l2_loss_rs + self.l2_loss(param)
        loss_rs = base_loss_rs + l2_loss_rs * self.args.l2_weight
        return loss_rs, base_loss_rs, l2_loss_rs

    def loss_kge(self, scores_kge, head_embeddings, tail_embeddings):
        base_loss_kge = -scores_kge
        l2_loss_kge = self.l2_loss(head_embeddings) + self.l2_loss(tail_embeddings)
        for name, param in self.MKR_model.named_parameters():
            if param.requires_grad and ('embeddings_lookup' not in name) \
                    and (('kge' in name) or ('tail' in name) or ('cc_unit' in name)) \
                    and ('weight' in name):
                l2_loss_kge = l2_loss_kge + self.l2_loss(param)
        # Note: L2 regularization will be done by weight_decay of pytorch optimizer
        loss_kge = base_loss_kge + l2_loss_kge * self.args.l2_weight
        return loss_kge, base_loss_kge, l2_loss_kge

    def train_rs(self, inputs, show_grad=False, glob_step=None):
        self.MKR_model.train()
        user_embeddings, item_embeddings, scores, _, labels= self._inference_rs(inputs)
        loss_rs, base_loss_rs, l2_loss_rs = self.loss_rs(user_embeddings, item_embeddings, scores, labels)

        self.optimizer_rs.zero_grad()
        loss_rs.backward()
        # if show_grad:
        #     plot_grad_flow(self.MKR_model.named_parameters(),
        #                    "grad_plot/rs_grad_step{}".format(glob_step))

        self.optimizer_rs.step()
        loss_rs.detach()
        user_embeddings.detach()
        item_embeddings.detach()
        scores.detach()
        labels.detach()

        return loss_rs, base_loss_rs, l2_loss_rs

    def train_kge(self, inputs, show_grad=False, glob_step=None):
        self.MKR_model.train()
        head_embeddings, tail_embeddings, scores_kge, rmse = self._inference_kge(inputs)
        loss_kge, base_loss_kge, l2_loss_kge = self.loss_kge(scores_kge, head_embeddings, tail_embeddings)

        self.optimizer_kge.zero_grad()
        loss_kge.sum().backward()
        # if show_grad:
        #     plot_grad_flow(self.MKR_model.named_parameters(),
        #                    "grad_plot/kge_grad_step{}".format(glob_step))

        self.optimizer_kge.step()
        loss_kge.detach()
        head_embeddings.detach()
        tail_embeddings.detach()
        scores_kge.detach()
        rmse.detach()
        return rmse, loss_kge.sum(), base_loss_kge.sum(), l2_loss_kge

    def eval(self, inputs):
        self.MKR_model.eval()
        inputs = torch.from_numpy(inputs)
        user_embeddings, item_embeddings, _, scores, labels = self._inference_rs(inputs)
        labels = labels.to("cpu").detach().numpy()
        scores = scores.to("cpu").detach().numpy()
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, labels))

        return auc, acc

    def topk_eval(self, user_list, train_record, test_record, item_set, k_list):
        print("Eval TopK")
        precision_list = {k: [] for k in k_list}
        recall_list = {k: [] for k in k_list}
        for user in tqdm(user_list):
            test_item_list = list(item_set - train_record[user])
            item_score_map = dict()
            scores = self._get_scores(np.array([user]*len(test_item_list)),
                                      np.array(test_item_list),
                                      np.array(test_item_list))
            items = np.array(test_item_list)
            for item, score in zip(items, scores):
                item_score_map[item] = score
            item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
            item_sorted = [i[0] for i in item_score_pair_sorted]
            for k in k_list:
                hit_num = len(set(item_sorted[:k]) & test_record[user])
                precision_list[k].append(hit_num / k)
                recall_list[k].append(hit_num / len(test_record[user]))
        precision = [np.mean(precision_list[k]) for k in k_list]
        recall = [np.mean(recall_list[k]) for k in k_list]
        f1 = [2 / (1 / precision[i] + 1 / recall[i]) for i in range(len(k_list))]

        return precision, recall, f1

    def _get_scores(self, user, item_list, head_list):
        # Inputs
        user = torch.from_numpy(user)
        item_list = torch.from_numpy(item_list)
        head_list = torch.from_numpy(head_list)
        self.user_indices = user.long().to(self.device)
        self.item_indices = item_list.long().to(self.device)
        self.head_indices = head_list.long().to(self.device)

        self.MKR_model.eval()
        outputs = self.MKR_model(self.user_indices, self.item_indices,
                                 self.head_indices, self.relation_indices,
                                 self.tail_indices)
        user_embeddings, item_embeddings, _, scores = outputs
        return scores
```

<!-- #region id="h5LBo3T6BDo_" -->
## Training
<!-- #endregion -->

```python id="WqVtFhtPA7Mi"
def train(args, rs_dataset, kg_dataset):

    show_loss = args.show_loss
    show_topk = args.show_topk

    # Get RS data
    n_user = rs_dataset.n_user
    n_item = rs_dataset.n_item
    train_data, eval_data, test_data = rs_dataset.data
    train_indices, eval_indices, test_indices = rs_dataset.indices

    # Get KG data
    n_entity = kg_dataset.n_entity
    n_relation = kg_dataset.n_relation
    kg = kg_dataset.kg

    # Init train sampler
    train_sampler = SubsetRandomSampler(train_indices)

    # Init MKR model
    model = MKR(args, n_user, n_item, n_entity, n_relation)

    # Init Sumwriter
    writer = SummaryWriter(args.summary_path)

    # Top-K evaluation settings
    user_num = 100
    k_list = [1, 2, 5, 10, 20, 50, 100]
    train_record = get_user_record(train_data, True)
    test_record = get_user_record(test_data, False)
    user_list = list(set(train_record.keys()) & set(test_record.keys()))
    if len(user_list) > user_num:
        user_list = np.random.choice(user_list, size=user_num, replace=False)
    item_set = set(list(range(n_item)))
    step = 0
    for epoch in range(args.n_epochs):
        print("Train RS")
        train_loader = DataLoader(rs_dataset, batch_size=args.batch_size,
                                  num_workers=args.workers, sampler=train_sampler)
        for i, rs_batch_data in enumerate(train_loader):
            loss, base_loss_rs, l2_loss_rs = model.train_rs(rs_batch_data)
            writer.add_scalar("rs_loss", loss.cpu().detach().numpy(), global_step=step)
            writer.add_scalar("rs_base_loss", base_loss_rs.cpu().detach().numpy(), global_step=step)
            writer.add_scalar("rs_l2_loss", l2_loss_rs.cpu().detach().numpy(), global_step=step)
            step += 1
            if show_loss:
                print(loss)

        if epoch % args.kge_interval == 0:
            print("Train KGE")
            kg_train_loader = DataLoader(kg_dataset, batch_size=args.batch_size,
                                         num_workers=args.workers, shuffle=True)
            for i, kg_batch_data in enumerate(kg_train_loader):
                rmse, loss_kge, base_loss_kge, l2_loss_kge = model.train_kge(kg_batch_data)
                writer.add_scalar("kge_rmse_loss", rmse.cpu().detach().numpy(), global_step=step)
                writer.add_scalar("kge_loss", loss_kge.cpu().detach().numpy(), global_step=step)
                writer.add_scalar("kge_base_loss", base_loss_kge.cpu().detach().numpy(), global_step=step)
                writer.add_scalar("kge_l2_loss", l2_loss_kge.cpu().detach().numpy(), global_step=step)
                step += 1
                if show_loss:
                    print(rmse)


        # CTR evaluation
        train_auc, train_acc = model.eval(train_data)
        eval_auc, eval_acc = model.eval(eval_data)
        test_auc, test_acc = model.eval(test_data)

        print('epoch %d    train auc: %.4f  acc: %.4f    eval auc: %.4f  acc: %.4f    test auc: %.4f  acc: %.4f'
              % (epoch, train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc))

        # top-K evaluation
        if show_topk:
            precision, recall, f1 = model.topk_eval(user_list, train_record, test_record, item_set, k_list)
            print('precision: ', end='')
            for i in precision:
                print('%.4f\t' % i, end='')
            print()
            print('recall: ', end='')
            for i in recall:
                print('%.4f\t' % i, end='')
            print()
            print('f1: ', end='')
            for i in f1:
                print('%.4f\t' % i, end='')
            print('\n')

def get_user_record(data, is_train):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict
```

```python colab={"base_uri": "https://localhost:8080/"} id="UbZIyHrtA7IK" executionInfo={"status": "ok", "timestamp": 1634925619317, "user_tz": -330, "elapsed": 291346, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7c45643d-2c23-480d-a63f-b6ad64206f79"
import argparse

parser = argparse.ArgumentParser()

'''
# movie
parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=20, help='the number of epochs')
parser.add_argument('--dim', type=int, default=8, help='dimension of user and entity embeddings')
parser.add_argument('--L', type=int, default=1, help='number of low layers')
parser.add_argument('--H', type=int, default=1, help='number of high layers')
parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-6, help='weight of l2 regularization')
parser.add_argument('--lr_rs', type=float, default=0.02, help='learning rate of RS task')
parser.add_argument('--lr_kge', type=float, default=0.01, help='learning rate of KGE task')
parser.add_argument('--kge_interval', type=int, default=3, help='training interval of KGE task')
'''

# '''
# book
parser.add_argument('--dataset', type=str, default='book', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
parser.add_argument('--dim', type=int, default=8, help='dimension of user and entity embeddings')
parser.add_argument('--L', type=int, default=1, help='number of low layers')
parser.add_argument('--H', type=int, default=1, help='number of high layers')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-6, help='weight of l2 regularization')
parser.add_argument('--lr_rs', type=float, default=2e-4, help='learning rate of RS task')
parser.add_argument('--lr_kge', type=float, default=2e-5, help='learning rate of KGE task')
parser.add_argument('--kge_interval', type=int, default=2, help='training interval of KGE task')
# '''

'''
# music
parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
parser.add_argument('--dim', type=int, default=4, help='dimension of user and entity embeddings')
parser.add_argument('--L', type=int, default=2, help='number of low layers')
parser.add_argument('--H', type=int, default=1, help='number of high layers')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-6, help='weight of l2 regularization')
parser.add_argument('--lr_rs', type=float, default=1e-3, help='learning rate of RS task')
parser.add_argument('--lr_kge', type=float, default=2e-4, help='learning rate of KGE task')
parser.add_argument('--kge_interval', type=int, default=2, help='training interval of KGE task')
'''

parser.add_argument('--workers', type=int, default=2,
                    help='number of data loading workers')
parser.add_argument('-sl', '--show_loss', action='store_true',
                    help='show loss or not')
parser.add_argument('-st', '--show_topk', action='store_true',
                    help='show topK or not')
parser.add_argument('-sum', '--summary_path', type=str, default='./summary',
                    help='path to store training summary')
args = parser.parse_args(args={})


rs_dataset = RSDataset(args)
kg_dataset = KGDataset(args)
train(args, rs_dataset, kg_dataset)
```
