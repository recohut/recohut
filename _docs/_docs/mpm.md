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

<!-- #region id="tb_GkOpT86gH" -->
# Training MPM Recommendation Model on ML-1m in PyTorch
<!-- #endregion -->

<!-- #region id="9bdc_KAbmRRM" -->
| | |
| --- | --- |
| Problem | In the implicit feedback recommendation, incorporating short-term preference into recommender systems has attracted increasing attention in recent years. However, unexpected behaviors in historical interactions like clicking some items by accident don’t well reflect users’ inherent preferences. Existing studies fail to model the effects of unexpected behaviors thus achieve inferior recommendation performance |
| Solution | Multi-Preferences Model (MPM) tries to eliminate the effects of unexpected behaviors by first extracting the users’ instant preferences from their recent historical interactions by a fine-grained preferences module. Then an unexpected-behaviors detector is trained to judge whether these instant preferences are biased by unexpected behaviors. we also integrate user’s general preference in MPM. Finally, an output module is performed to eliminates the effects of unexpected behaviors and integrates all the information to make a final recommendation. |
| Dataset | ML-1m |
| Preprocessing | We evaluate the performance of our proposed model by the leave-one-out evaluation. For each dataset, we hold out the last one item that each user has interacted with and sample 99 items that unobserved interactions to form the test set, a validation set is also created like the test set and remaining data as a training set. For each positive user-item interaction pair in the training set, we conducted the negative sampling strategy to pair it with four negative items. |
| Metrics | HR@10, NDCG@10 |
| Models | MPM (Multi-Preferences Model) |
| Platform | PyTorch 1.10.0+cpu, Ubuntu 18.0 Google Colab instnce (VM) |
| Links | [Paper](https://arxiv.org/pdf/2112.11023v1.pdf), [Code](https://github.com/chenjie04/MPM) |
<!-- #endregion -->

<!-- #region id="42lNJbTq89Jh" -->
## Setup
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4442, "status": "ok", "timestamp": 1640861410731, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="0TQOKHhszdpq" outputId="77728d1c-266b-4bf7-8571-943d0ea40153"
!pip install mlperf_compliance
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 843, "status": "ok", "timestamp": 1640861412178, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="3gil-2exz-Sn" outputId="e15ab3a9-2007-4248-d846-687fed313b20"
!mkdir /content/data
%cd /content/data
!wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
!unzip ml-1m.zip
%cd /content
```

<!-- #region id="RcpXGxRS8-vS" -->
## Datasets
<!-- #endregion -->

```python id="0G2kNVoGzYNO"
from collections import namedtuple

import pandas as pd


RatingData = namedtuple('RatingData',
                        ['items', 'users', 'ratings', 'min_date', 'max_date'])


def describe_ratings(ratings):
    info = RatingData(items=len(ratings['item_id'].unique()),
                      users=len(ratings['user_id'].unique()),
                      ratings=len(ratings),
                      min_date=ratings['timestamp'].min(),
                      max_date=ratings['timestamp'].max())
    print("{ratings} ratings on {items} items from {users} users"
          " from {min_date} to {max_date}"
          .format(**(info._asdict())))
    return info


def process_movielens(ratings, sort=True):
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    if sort:
        ratings.sort_values(by='timestamp', inplace=True)
    describe_ratings(ratings)
    return ratings

def process_taobao(ratings,sort=True):
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'],unit='s')
    if sort:
        ratings.sort_values(by='timestamp', inplace=True)
    describe_ratings(ratings)
    return ratings


def load_ml_100k(filename, sort=True):
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings = pd.read_csv(filename, sep='\t', names=names)
    return process_movielens(ratings, sort=sort)


def load_ml_1m(filename, sort=True):
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings = pd.read_csv(filename, sep='::', names=names, engine='python')
    return process_movielens(ratings, sort=sort)


def load_ml_10m(filename, sort=True):
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings = pd.read_csv(filename, sep='::', names=names, engine='python')
    return process_movielens(ratings, sort=sort)


def load_ml_20m(filename, sort=True):
    ratings = pd.read_csv(filename)
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    names = {'userId': 'user_id', 'movieId': 'item_id'}
    ratings.rename(columns=names, inplace=True)
    return process_movielens(ratings, sort=sort)



def load_taobao(filename,sort=True):
    names = ['user_id','item_id','category_id','behavior_type','timestamp']
    ratings = pd.read_csv(filename, names=names)
    return process_taobao(ratings,sort=sort)



DATASETS = [k.replace('load_', '') for k in locals().keys() if "load_" in k]


def get_dataset_name(filename):
    for dataset in DATASETS:
        if dataset in filename.replace('-', '_').lower():
            return dataset
    raise NotImplementedError


def implicit_load(filename, sort=True):

    func = globals()["load_" + get_dataset_name(filename)]
    return func(filename, sort=sort)
```

```python id="LI6gcxXOzCz6"
import os
from argparse import ArgumentParser
from collections import defaultdict
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from collections import namedtuple


from mlperf_compliance import mlperf_log


MIN_RATINGS = 20


USER_COLUMN = 'user_id'
ITEM_COLUMN = 'item_id'


TRAIN_RATINGS_FILENAME = 'train_ratings.csv'
TEST_RATINGS_FILENAME = 'test_ratings.csv'
TEST_NEG_FILENAME = 'test_negative.csv'
DATA_SUMMARY_FILENAME = "data_summary.csv"

# PATH = 'data/taobao-1m'
# OUTPUT = 'data/taobao-1m'
PATH = 'data/ml-1m'
OUTPUT = 'data/ml-1m'
NEGATIVES = 99
HISTORY_SIZE = 9
RANDOM_SEED = 0

def parse_args():
    parser = ArgumentParser()

    # parser.add_argument('--file',type=str,default=(os.path.join(PATH,'UserBehavior01.csv')),
    #                     help='Path to reviews CSV file from dataset')
    parser.add_argument('--file',type=str,default=(os.path.join(PATH,'ratings.dat')),
                        help='Path to reviews CSV file from dataset')
    parser.add_argument('--output', type=str, default=OUTPUT,
                        help='Output directory for train and test CSV files')
    parser.add_argument('-n', '--negatives', type=int, default=NEGATIVES,
                        help='Number of negative samples for each positive'
                             'test example')
    parser.add_argument('--history_size',type=int,default=HISTORY_SIZE,
                        help='The size of history')
    parser.add_argument('-s', '--seed', type=int, default=RANDOM_SEED,
                        help='Random seed to reproduce same negative samples')
    return parser.parse_args({})


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print("Loading raw data from {}".format(args.file))
    #-------------- MovieLens dataset ------------------------------
    df = implicit_load(args.file, sort=False)
    #---------------------------------------------------------------

    #------ retailrocket-recommender-system-dataset --------------------
    # df = pd.read_csv(args.file, sep=',', header=0)
    # df.columns = ['timestamp', 'user_id', 'event', 'item_id', 'transaction_id']
    # df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    #
    #
    # RatingData = namedtuple('RatingData',
    #                         ['items', 'users', 'ratings', 'min_date', 'max_date'])
    # info = RatingData(items=len(df['item_id'].unique()),
    #                   users=len(df['user_id'].unique()),
    #                   ratings=len(df),
    #                   min_date=df['timestamp'].min(),
    #                   max_date=df['timestamp'].max())
    # print("{ratings} ratings on {items} items from {users} users"
    #           " from {min_date} to {max_date}"
    #           .format(**(info._asdict())))
    # #--------------------------------------------------------------------

    #-------------------amazon dataset------------------------
    # df = pd.read_csv(args.file, sep=',', header=None)
    # df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    # df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    #
    # RatingData = namedtuple('RatingData',
    #                         ['items', 'users', 'ratings', 'min_date', 'max_date'])
    # info = RatingData(items=len(df['item_id'].unique()),
    #                   users=len(df['user_id'].unique()),
    #                   ratings=len(df),
    #                   min_date=df['timestamp'].min(),
    #                   max_date=df['timestamp'].max())
    # print("{ratings} ratings on {items} items from {users} users"
    #           " from {min_date} to {max_date}"
    #           .format(**(info._asdict())))


    #-------------------------------------------------------------------------

    #------------------- hetrec2011 dataset------------------------
    # df = pd.read_csv(args.file, sep='\t', header=0)
    # df.columns = ['user_id', 'item_id', 'tag_id', 'timestamp']
    # df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    #
    # RatingData = namedtuple('RatingData',
    #                         ['items', 'users', 'ratings', 'min_date', 'max_date'])
    # info = RatingData(items=len(df['item_id'].unique()),
    #                   users=len(df['user_id'].unique()),
    #                   ratings=len(df),
    #                   min_date=df['timestamp'].min(),
    #                   max_date=df['timestamp'].max())
    # print("{ratings} ratings on {items} items from {users} users"
    #           " from {min_date} to {max_date}"
    #           .format(**(info._asdict())))
    #

    #-------------------------------------------------------------------------

    #------------------- taobao UserBehavior dataset------------------------
    # df = pd.read_csv(args.file, sep=',', header=None)
    # df.columns = ['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp']
    # df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    # RatingData = namedtuple('RatingData',
    #                         ['items', 'users', 'ratings', 'min_date', 'max_date'])
    # info = RatingData(items=len(df['item_id'].unique()),
    #                   users=len(df['user_id'].unique()),
    #                   ratings=len(df),
    #                   min_date=df['timestamp'].min(),
    #                   max_date=df['timestamp'].max())
    # print("{ratings} ratings on {items} items from {users} users"
    #           " from {min_date} to {max_date}"
    #           .format(**(info._asdict())))


    #-------------------------------------------------------------------------

    print("Filtering out users with less than {} ratings".format(MIN_RATINGS))
    grouped = df.groupby(USER_COLUMN)
    mlperf_log.ncf_print(key=mlperf_log.PREPROC_HP_MIN_RATINGS, value=MIN_RATINGS)
    df = grouped.filter(lambda x: len(x) >= MIN_RATINGS)

    print("Mapping original user and item IDs to new sequential IDs")
    original_users = df[USER_COLUMN].unique()
    original_items = df[ITEM_COLUMN].unique()

    nb_users = len(original_users)
    nb_items = len(original_items)

    user_map = {user: index for index, user in enumerate(original_users)}
    item_map = {item: index for index, item in enumerate(original_items)}

    df[USER_COLUMN] = df[USER_COLUMN].apply(lambda user: user_map[user])
    df[ITEM_COLUMN] = df[ITEM_COLUMN].apply(lambda item: item_map[item])

    # print(df)


    assert df[USER_COLUMN].max() == len(original_users) - 1
    assert df[ITEM_COLUMN].max() == len(original_items) - 1

    print("Creating list of items for each user")
    # Need to sort before popping to get last item
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.sort_values(by='timestamp', inplace=True)
    all_ratings = set(zip(df[USER_COLUMN], df[ITEM_COLUMN]))
    user_to_items = defaultdict(list)
    for row in tqdm(df.itertuples(), desc='Ratings', total=len(df)):
        user_to_items[getattr(row, USER_COLUMN)].append(getattr(row, ITEM_COLUMN))  # noqa: E501

    print(len(user_to_items[0]))
    print(user_to_items[0])
    print(user_to_items[0][-args.history_size:])



    print("Generating {} negative samples for each user and creating training set"
          .format(args.negatives))
    mlperf_log.ncf_print(key=mlperf_log.PREPROC_HP_NUM_EVAL, value=args.negatives)

    train_ratings = []
    test_ratings = []
    test_negs = []
    all_items = set(range(len(original_items)))

    for key, value in tqdm(user_to_items.items(), total=len(user_to_items)):
        all_negs = all_items - set(value)
        all_negs = sorted(list(all_negs))
        negs = random.sample(all_negs, args.negatives)

        test_item = value.pop()

        tmp = [key, test_item]
        tmp.extend(negs)
        test_negs.append(tmp)

        tmp = [key, test_item]
        tmp.extend(value[-args.history_size:])
        test_ratings.append(tmp)

        while len(value) > args.history_size:
            tgItem = value.pop()
            tmp = [key,tgItem]
            tmp.extend(value[-args.history_size:])
            train_ratings.append(tmp)



    print("\nSaving train and test CSV files to {}".format(args.output))



    df_train_ratings = pd.DataFrame(list(train_ratings))
    df_test_ratings = pd.DataFrame(list(test_ratings))
    df_test_negs = pd.DataFrame(list(test_negs))


    print('Saving data description ...')
    data_summary = pd.DataFrame(
        {'users': nb_users, 'items': nb_items, 'history_size': HISTORY_SIZE, 'train_entries': len(df_train_ratings), 'test': len(df_test_ratings)},
        index=[0])
    data_summary.to_csv(os.path.join(args.output, DATA_SUMMARY_FILENAME), header=True, index=False, sep=',')

    df_train_ratings['fake_rating'] = 1
    df_train_ratings.to_csv(os.path.join(args.output, TRAIN_RATINGS_FILENAME),
                            index=False, header=False, sep='\t')

    mlperf_log.ncf_print(key=mlperf_log.INPUT_SIZE, value=len(df_train_ratings))


    df_test_ratings['fake_rating'] = 1
    df_test_ratings.to_csv(os.path.join(args.output, TEST_RATINGS_FILENAME),
                           index=False, header=False, sep='\t')


    df_test_negs.to_csv(os.path.join(args.output, TEST_NEG_FILENAME),
                        index=False, header=False, sep='\t')


# if __name__ == '__main__':
    # main()
```

```python id="hK-ITnsw0saV"
import numpy as np
import scipy
import scipy.sparse
import torch
import torch.utils.data
import pandas as pd

from mlperf_compliance import mlperf_log


class CFTrainDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, train_fname, data_summary_fname, nb_neg):
        data_summary = pd.read_csv(data_summary_fname, sep=',', header=0)
        self.nb_users = data_summary.loc[0,'users']
        self.nb_items = data_summary.loc[0,'items']
        self._load_train_matrix(train_fname)
        self.nb_neg = nb_neg

        mlperf_log.ncf_print(key=mlperf_log.INPUT_STEP_TRAIN_NEG_GEN, value=nb_neg)
        mlperf_log.ncf_print(key=mlperf_log.INPUT_HP_SAMPLE_TRAIN_REPLACEMENT)

    def _load_train_matrix(self, train_fname):
        def process_line(line):
            line = line.strip().split('\t')
            tmp = []
            tmp.extend(np.array(line[0:-1]).astype(int))
            tmp.extend([float(line[-1]) > 0])

            return tmp

        with open(train_fname, 'r') as file:
            data = list(map(process_line, file))
        # self.nb_users = max(data, key=lambda x: x[0])[0] + 1
        # self.nb_items = max(data, key=lambda x: x[1])[1] + 1

        length = len(data)

        self.data = list(filter(lambda x: x[-1], data))
        self.mat = scipy.sparse.dok_matrix(
                (self.nb_users, self.nb_items), dtype=np.float32)
        for i in range(length):
            user = self.data[i][0]
            item = self.data[i][1]
            self.mat[user, item] = 1.

    def __len__(self):
        return (self.nb_neg + 1) * len(self.data)

    def __getitem__(self, idx):
        if idx % (self.nb_neg + 1) == 0:
            idx = idx // (self.nb_neg + 1)
            return self.data[idx][0], self.data[idx][1], torch.LongTensor(self.data[idx][2:-1]), np.ones(1, dtype=np.float32)  # noqa: E501
        else:
            idx = idx // (self.nb_neg + 1)
            u = self.data[idx][0]
            j = torch.LongTensor(1).random_(0, int(self.nb_items)).item()
            while (u, j) in self.mat:
                j = torch.LongTensor(1).random_(0, int(self.nb_items)).item()
            return u, j, torch.LongTensor(self.data[idx][2:-1]), np.zeros(1, dtype=np.float32)


def load_test_ratings(fname):
    def process_line(line):
        tmp = map(int, line.strip().split('\t')[:-1])
        return list(tmp)
    ratings = map(process_line, open(fname, 'r'))
    return list(ratings)


def load_test_negs(fname):
    def process_line(line):
        tmp = map(int, line.strip().split('\t')[2:])
        return list(tmp)
    negs = map(process_line, open(fname, 'r'))
    return list(negs)
```

<!-- #region id="OGfBzM1P9Eyr" -->
## Utils
<!-- #endregion -->

```python id="rQXI4k4r01Gs"
import os
import json
from functools import reduce


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_parameters(model):
    c = map(lambda p: reduce(lambda x, y: x * y, p.size()), model.parameters())
    return sum(c)


def save_config(config, run_dir):
    path = os.path.join(run_dir, "config_{}.json".format(config['timestamp']))
    with open(path, 'w') as config_file:
        json.dump(config, config_file)
        config_file.write('\n')


def save_result(result, path):
    write_heading = not os.path.exists(path)
    with open(path, mode='a') as out:
        if write_heading:
            out.write(",".join([str(k) for k, v in result.items()]) + '\n')
        out.write(",".join([str(v) for k, v in result.items()]) + '\n')
```

<!-- #region id="o6BXm5oj9GPw" -->
## Layers
<!-- #endregion -->

```python id="YbTWj0x30-oa"
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
```

<!-- #region id="kagIn0Ws9IHd" -->
## MPM Model
<!-- #endregion -->

```python id="MB7CKp_w1B7h"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Multi_Preference_Model(nn.Module):
    def __init__(self, nb_users, nb_items, embed_dim, history_size):
        super(Multi_Preference_Model, self).__init__()

        self.nb_users = nb_users
        self.nb_items = nb_items
        self.embed_dim = embed_dim
        self.history_size = history_size

        #user and item embedding
        self.user_embed = nn.Embedding(self.nb_users, self.embed_dim)
        self.item_embed = nn.Embedding(self.nb_items, self.embed_dim)
        self.user_embed.weight.data.normal_(0., 0.01)
        self.item_embed.weight.data.normal_(0., 0.01)

        #TCN
        nhid = self.embed_dim
        level = 5
        num_channels = [nhid] * (level - 1) + [embed_dim]
        self.tcn = TemporalConvNet(num_inputs=self.embed_dim, num_channels=num_channels, kernel_size=3, dropout=0.25)

        #MLP
        mlp_layer_sizes = [self.embed_dim * 2, 128, 64, 32]
        nb_mlp_layers = len(mlp_layer_sizes)
        self.mlp = nn.ModuleList()
        for i in range(1, nb_mlp_layers):
            self.mlp.extend([nn.Linear(mlp_layer_sizes[i-1], mlp_layer_sizes[i])])

        #Output Module
        self.output_1 = nn.Linear(mlp_layer_sizes[-1] * (self.history_size + 1),128,bias=True)
        self.output_2 = nn.Linear(128,64,bias=True)
        self.output_3 = nn.Linear(64,32,bias=True)
        self.output_4 = nn.Linear(32,1,bias=True)

        def golorot_uniform(layer):
            fan_in, fan_out = layer.in_features, layer.out_features
            limit = np.sqrt(6. / (fan_in + fan_out))
            layer.weight.data.uniform_(-limit, limit)

        def lecunn_uniform(layer):
            fan_in, fan_out = layer.in_features, layer.out_features  # noqa: F841, E501
            limit = np.sqrt(3. / fan_in)
            layer.weight.data.uniform_(-limit, limit)

        for layer in self.mlp:
            if type(layer) != nn.Linear:
                continue
            golorot_uniform(layer)

        lecunn_uniform(self.output_1)
        lecunn_uniform(self.output_2)
        lecunn_uniform(self.output_3)
        lecunn_uniform(self.output_4)

    def forward(self, user, item, history,sigmoid=False):

        item = self.item_embed(item)

        #multi granularity preference module
        xhistory = self.item_embed(history)

        output_TCN = self.tcn(xhistory.transpose(1,2)).transpose(1,2)

        predict_vectors = list()

        for i in range(self.history_size):
            preference = output_TCN[:, i, :]
            output_mlp = torch.cat((preference,item),dim=1)
            for j, layer in enumerate(self.mlp):
                output_mlp = layer(output_mlp)
                output_mlp = F.relu(output_mlp)

            output_mlp = output_mlp.view(-1, 1, output_mlp.size()[-1])
            predict_vectors.append(output_mlp)

        predict_vectors_sum = torch.cat(predict_vectors, dim=1)

        # general preference module
        user = self.user_embed(user)
        xmlp = torch.cat((user, item), dim=1)
        for i, layer in enumerate(self.mlp):
            xmlp = layer(xmlp)
            xmlp = F.relu(xmlp)

        #output module
        xmlp = xmlp.view(-1,1,xmlp.size()[-1])
        x = torch.cat((predict_vectors_sum,xmlp),dim=1)
        x = x.view(x.size()[0],-1)
        x = self.output_1(x)
        x = F.relu(x)
        x = self.output_2(x)
        x = F.relu(x)
        x = self.output_3(x)
        x = F.relu(x)
        x = self.output_4(x)

        if sigmoid:
            x = torch.sigmoid(x)
        return x
```

<!-- #region id="HZ9Egjd-9KO4" -->
## Training and Evaluation
<!-- #endregion -->

```python colab={"background_save": true, "base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1441496, "status": "ok", "timestamp": 1640864863496, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="b1JUvaOc1GKa" outputId="e7c32729-76b1-4c40-aa60-e306d4099c81"
import heapq
import math
import time
from functools import partial
from datetime import datetime
from collections import OrderedDict
from argparse import ArgumentParser

import random
import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import multiprocessing as mp

from mlperf_compliance import mlperf_log

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def parse_args():
    parser = ArgumentParser(description="Train a Nerual Collaborative"
                                        " Filtering model")
    parser.add_argument('--data', type=str, default='data/ml-1m',
                        help='path to test and training data files')
    parser.add_argument('-e', '--epochs', type=int, default=2,
                        help='number of epochs for training')
    parser.add_argument('-b', '--batch-size', type=int, default=2048,
                        help='number of examples for each iteration')
    parser.add_argument('-n', '--negative-samples', type=int, default=4,
                        help='number of negative examples per interaction')
    parser.add_argument('-l', '--learning-rate', type=float, default=0.001,
                        help='learning rate for optimizer')
    parser.add_argument('-k', '--topk', type=int, default=10,
                        help='rank for test examples to be considered a hit')
    parser.add_argument('--no-cuda', action='store_true',default=False,
                        help='use available GPUs')
    parser.add_argument('--seed', '-s', type=int,default=3,
                        help='manually set random seed for torch')
    parser.add_argument('--processes', '-p', type=int, default=1,
                        help='Number of processes for evaluating model')
    parser.add_argument('--workers', '-w', type=int, default=4,
                        help='Number of workers for training DataLoader')
    parser.add_argument('--resume', '-r',action='store_true', default=False,
                        help='resume from checkpoint')
    return parser.parse_args({})


def predict(model, users, items, history, batch_size=1024, use_cuda=True):
    batches = [(users[i:i + batch_size], items[i:i + batch_size],history[i:i + batch_size])
               for i in range(0, len(users), batch_size)]
    preds = []
    for user, item, _history in batches:
        def proc(x):
            x = np.array(x,dtype=int)
            x = torch.from_numpy(x)
            if use_cuda:
                x = x.cuda()
            return torch.autograd.Variable(x)

        # outp, _ = model(proc(user), proc(item), proc(_history), sigmoid=True)
        outp = model(proc(user), proc(item), proc(_history), sigmoid=True)

        outp = outp.data.cpu().numpy()
        preds += list(outp.flatten())
    return preds


def _calculate_hit(ranked, test_item):
    return int(test_item in ranked)


def _calculate_ndcg(ranked, test_item):
    for i, item in enumerate(ranked):
        if item == test_item:
            return math.log(2) / math.log(i + 2)
    return 0.


def eval_one(rating, items, model, K, use_cuda=True):

    user = rating[0]
    test_item = rating[1]
    items.append(test_item)
    users = [user] * len(items)
    history = []
    _history = rating[2:]
    for i in range(len(items)):
        history.append(_history)

    assert len(users) == len(items) == len(history)

    predictions = predict(model, users, items, history, use_cuda=use_cuda)

    map_item_score = {item: pred for item, pred in zip(items, predictions)}
    ranked = heapq.nlargest(K, map_item_score, key=map_item_score.get)

    hit = _calculate_hit(ranked, test_item)
    ndcg = _calculate_ndcg(ranked, test_item)
    return hit, ndcg, len(predictions)


def val_epoch(model, ratings, negs, K, use_cuda=True, output=None, epoch=None,
              processes=1):
    if epoch is None:
        print("Initial evaluation")
    else:
        print("Epoch {} evaluation".format(epoch))

    mlperf_log.ncf_print(key=mlperf_log.EVAL_START, value=epoch)
    start = datetime.now()
    model.eval()
    if processes > 1:
        context = mp.get_context('spawn')
        _eval_one = partial(eval_one, model=model, K=K, use_cuda=use_cuda)
        with context.Pool(processes=processes) as workers:
            hits_ndcg_numpred = workers.starmap(_eval_one, zip(ratings, negs))
        hits, ndcgs, num_preds = zip(*hits_ndcg_numpred)
    else:
        hits, ndcgs, num_preds = [], [], []
        for rating, items in zip(ratings, negs):
            hit, ndcg, num_pred = eval_one(rating, items, model, K, use_cuda=use_cuda)
            hits.append(hit)
            ndcgs.append(ndcg)
            num_preds.append(num_pred)

    hits = np.array(hits, dtype=np.float32)
    ndcgs = np.array(ndcgs, dtype=np.float32)

    assert len(set(num_preds)) == 1
    num_neg = num_preds[0] - 1  # one true positive, many negatives
    mlperf_log.ncf_print(key=mlperf_log.EVAL_SIZE, value={"epoch": epoch, "value": len(hits) * (1 + num_neg)})
    mlperf_log.ncf_print(key=mlperf_log.EVAL_HP_NUM_USERS, value=len(hits))
    mlperf_log.ncf_print(key=mlperf_log.EVAL_HP_NUM_NEG, value=num_neg)

    end = datetime.now()
    if output is not None:
        result = OrderedDict()
        result['timestamp'] = datetime.now()
        result['duration'] = end - start
        result['epoch'] = epoch
        result['K'] = K
        result['hit_rate'] = np.mean(hits)
        result['NDCG'] = np.mean(ndcgs)
        save_result(result, output)

    return hits, ndcgs


def main():
    # Note: The run start is in data_preprocess.py

    args = parse_args()
    if args.seed is not None:
        print("Using seed = {}".format(args.seed))
        torch.manual_seed(args.seed)
        np.random.seed(seed=args.seed)

    # Save configuration to file
    config = {k: v for k, v in args.__dict__.items()}
    config['timestamp'] = "{:.0f}".format(datetime.utcnow().timestamp())
    config['local_timestamp'] = str(datetime.now())
    run_dir = "./run/MGPM/{}/{}".format(os.path.basename(os.path.normpath(args.data)),config['timestamp'])
    print("Saving config and results to {}".format(run_dir))
    if not os.path.exists(run_dir) and run_dir != '':
        os.makedirs(run_dir)
    save_config(config, run_dir)

    # Check that GPUs are actually available
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if use_cuda:
        print("Using cuda ...")
    else:
        print("Using CPU ...")

    t1 = time.time()

    best_hit, best_ndcg = 0., 0.
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Load Data
    print('Loading data')
    print(os.path.join(args.data, TRAIN_RATINGS_FILENAME))
    train_dataset = CFTrainDataset(
        os.path.join(args.data, TRAIN_RATINGS_FILENAME),os.path.join(args.data, DATA_SUMMARY_FILENAME), args.negative_samples)

    mlperf_log.ncf_print(key=mlperf_log.INPUT_BATCH_SIZE, value=args.batch_size)
    mlperf_log.ncf_print(key=mlperf_log.INPUT_ORDER)  # set shuffle=True in DataLoader
    train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
    test_ratings = load_test_ratings(os.path.join(args.data, TEST_RATINGS_FILENAME))  # noqa: E501
    test_negs = load_test_negs(os.path.join(args.data, TEST_NEG_FILENAME))
    nb_users, nb_items = train_dataset.nb_users, train_dataset.nb_items
    print('Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d'
          % (time.time()-t1, nb_users, nb_items, train_dataset.mat.nnz,
             len(test_ratings)))

    # Create model
    model = Multi_Preference_Model(nb_users=nb_users, nb_items=nb_items,
                      embed_dim=32,history_size=9)
    print(model)
    print("{} parameters".format(count_parameters(model)))

    # Save model text description
    with open(os.path.join(run_dir, 'model.txt'), 'w') as file:
        file.write(str(model))

    # Add optimizer and loss to graph
    mlperf_log.ncf_print(key=mlperf_log.OPT_LR, value=args.learning_rate)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    mlperf_log.ncf_print(key=mlperf_log.OPT_NAME, value="Adam")
    mlperf_log.ncf_print(key=mlperf_log.OPT_HP_ADAM_BETA1, value=beta1)
    mlperf_log.ncf_print(key=mlperf_log.OPT_HP_ADAM_BETA2, value=beta2)
    mlperf_log.ncf_print(key=mlperf_log.OPT_HP_ADAM_EPSILON, value=epsilon)
    optimizer = torch.optim.Adam(model.parameters(), betas=(beta1, beta2),
                                 lr=args.learning_rate, eps=epsilon)

    mlperf_log.ncf_print(key=mlperf_log.MODEL_HP_LOSS_FN, value=mlperf_log.BCE)
    # optimizer = torch.optim.SGD(model.parameters(),lr=args.learning_rate,momentum=0.9)
    criterion = nn.BCEWithLogitsLoss()


    if use_cuda:
        # Move model and loss to GPU
        model = model.cuda()
        criterion = criterion.cuda()

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/' + model._get_name() + '.pd')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_hit = checkpoint['hit']
        best_ndcg = checkpoint['ndcg']


    # Create files for tracking training
    valid_results_file = os.path.join(run_dir, 'valid_results.csv')

    # Calculate initial Hit Ratio and NDCG
    if start_epoch == 0:
        hits, ndcgs = val_epoch(model, test_ratings, test_negs, args.topk,
                                use_cuda=use_cuda, processes=args.processes)
        print('Initial HR@{K} = {hit_rate:.4f}, NDCG@{K} = {ndcg:.4f}'
              .format(K=args.topk, hit_rate=np.mean(hits), ndcg=np.mean(ndcgs)))

    mlperf_log.ncf_print(key=mlperf_log.TRAIN_LOOP)
    for epoch in range(start_epoch,args.epochs):
        mlperf_log.ncf_print(key=mlperf_log.TRAIN_EPOCH, value=epoch)
        model.train()
        losses = AverageMeter()

        mlperf_log.ncf_print(key=mlperf_log.INPUT_HP_NUM_NEG, value=train_dataset.nb_neg)
        mlperf_log.ncf_print(key=mlperf_log.INPUT_STEP_TRAIN_NEG_GEN)
        begin = time.time()
        loader = tqdm.tqdm(train_dataloader)
        for batch_index, (user, item, history, label) in enumerate(loader):
            user = torch.autograd.Variable(user, requires_grad=False)
            item = torch.autograd.Variable(item, requires_grad=False)
            history = torch.autograd.Variable(history, requires_grad=False)
            label = torch.autograd.Variable(label, requires_grad=False)
            if use_cuda:
                user = user.cuda()
                item = item.cuda()
                history = history.cuda()
                label = label.cuda()

            # outputs, _ = model(user, item,history)
            outputs = model(user, item, history)
            loss = criterion(outputs, label)
            losses.update(loss.data.item(), user.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Save stats to file
            description = ('Epoch {} Loss {loss.val:.4f} ({loss.avg:.4f})'
                           .format(epoch, loss=losses))
            loader.set_description(description)

        train_time = time.time() - begin
        begin = time.time()
        hits, ndcgs = val_epoch(model, test_ratings, test_negs, args.topk,
                                use_cuda=use_cuda, output=valid_results_file,
                                epoch=epoch, processes=args.processes)
        mlperf_log.ncf_print(key=mlperf_log.EVAL_ACCURACY, value={"epoch": epoch, "value": float(np.mean(hits))})
        mlperf_log.ncf_print(key=mlperf_log.EVAL_STOP)
        val_time = time.time() - begin
        print('Epoch {epoch}: HR@{K} = {hit_rate:.4f}, NDCG@{K} = {ndcg:.4f},'
              ' train_time = {train_time:.2f}, val_time = {val_time:.2f}'
              .format(epoch=epoch, K=args.topk, hit_rate=np.mean(hits),
                      ndcg=np.mean(ndcgs), train_time=train_time,
                      val_time=val_time))
        if np.mean(hits) >= best_hit or np.mean(ndcgs) >= best_ndcg:
            best_hit = np.mean(hits)
            best_ndcg = np.mean(ndcgs)
            # Save checkpoint.
            print('Saving checkpoint..')
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'hit':best_hit,
                'ndcg':best_ndcg,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/' + model._get_name()  + '.pd')

    print("Best hit: ",best_hit)
    print("Best_ndcg: ", best_ndcg)

    mlperf_log.ncf_print(key=mlperf_log.RUN_STOP)
    mlperf_log.ncf_print(key=mlperf_log.RUN_FINAL)


if __name__ == '__main__':
    main()
```

```python id="EIdOGHLM1mRe"

```
