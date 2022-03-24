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

```python id="05rycybwtrgI" executionInfo={"status": "ok", "timestamp": 1638283291233, "user_tz": -330, "elapsed": 537, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import os
import pickle
import time
import math
```

```python colab={"base_uri": "https://localhost:8080/"} id="hHGT6El6SsEt" executionInfo={"status": "ok", "timestamp": 1638283275052, "user_tz": -330, "elapsed": 656, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e6a17915-e3a2-4b5e-c54f-157795fb6d9a"
!wget -q --show-progress https://github.com/RecoHut-Projects/US969796/raw/main/datasets/sample_train-item-views.csv
```

```python colab={"base_uri": "https://localhost:8080/"} id="GIFIDP9GcmJK" executionInfo={"status": "ok", "timestamp": 1638285868468, "user_tz": -330, "elapsed": 472, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6dae6a5c-c4c6-44b9-c5df-94d86243889e"
!head sample_train-item-views.csv
```

<!-- #region id="B7ZtDP_ObNIM" -->
## Preprocessing
<!-- #endregion -->

```python id="TfaZ1Uw7twYw" executionInfo={"status": "ok", "timestamp": 1638283293383, "user_tz": -330, "elapsed": 3, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
raw_path = 'sample_train-item-views'
save_path = 'processed'
```

<!-- #region id="ZKNqm1XWbPZ1" -->
### Unaugmented
<!-- #endregion -->

```python id="NSVO6oQHt_FD" executionInfo={"status": "ok", "timestamp": 1638285935298, "user_tz": -330, "elapsed": 446, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def load_data(file):
    print("Start load_data")
    # load csv
    data = pd.read_csv(file+'.csv', sep=';', header=0, usecols=[0, 2, 4], dtype={0: np.int32, 1: np.int64, 3: str})
    # specify header names
    data.columns = ['SessionId', 'ItemId', 'Eventdate']
    # convert time string to timestamp and remove the original column
    data['Time'] = data.Eventdate.apply(lambda x: datetime.strptime(x, '%Y-%m-%d').timestamp())
    print(data['Time'].min())
    print(data['Time'].max())
    del(data['Eventdate'])

    # output
    data_start = datetime.fromtimestamp(data.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)

    print('Loaded data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(data), data.SessionId.nunique(), data.ItemId.nunique(),
                 data_start.date().isoformat(), data_end.date().isoformat()))
    return data


def filter_data(data, min_item_support=5, min_session_length=2):
    print("Start filter_data")

    # y?
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[session_lengths > 1].index)]

    # filter item support
    item_supports = data.groupby('ItemId').size()
    data = data[np.in1d(data.ItemId, item_supports[item_supports >= min_item_support].index)]

    # filter session length
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[session_lengths >= min_session_length].index)]
    print(data['Time'].min())
    print(data['Time'].max())
    # output
    data_start = datetime.fromtimestamp(data.Time.astype(np.int64).min(), timezone.utc)
    data_end = datetime.fromtimestamp(data.Time.astype(np.int64).max(), timezone.utc)

    print('Filtered data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(data), data.SessionId.nunique(), data.ItemId.nunique(),
                 data_start.date().isoformat(), data_end.date().isoformat()))
    return data


def split_train_test(data):
    print("Start split_train_test")
    tmax = data.Time.max()
    session_max_times = data.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < tmax-7*86400].index
    session_test = session_max_times[session_max_times >= tmax-7*86400].index
    train = data[np.in1d(data.SessionId, session_train)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength >= 2].index)]

    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(), train.ItemId.nunique()))
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(), test.ItemId.nunique()))

    return train, test


def get_dict(data):
    print("Start get_dict")
    item2idx = {}
    pop_scores = data.groupby('ItemId').size().sort_values(ascending=False)
    pop_scores = pop_scores / pop_scores[:1].values[0]
    items = pop_scores.index
    for idx, item in enumerate(items):
        item2idx[item] = idx+1
    return item2idx


def process_seqs(seqs, shift):
    start = time.time()
    labs = []
    index = shift
    for count, seq in enumerate(seqs):
        index += (len(seq) - 1)
        labs += [index]
        end = time.time()
        print("\rprocess_seqs: [%d/%d], %.2f, usetime: %fs, " % (count, len(seqs), count/len(seqs) * 100, end - start),
              end='', flush=True)
    print("\n")
    return seqs, labs


def get_sequence(data, item2idx, shift=-1):
    start = time.time()
    sess_ids = data.drop_duplicates('SessionId', 'first')
    print(sess_ids)
    sess_ids.sort_values(['Time'], inplace=True)
    sess_ids = sess_ids['SessionId'].unique()
    seqs = []
    for count, sess_id in enumerate(sess_ids):
        seq = data[data['SessionId'].isin([sess_id])] 
        # seq = data[data['SessionId'].isin([sess_id])].sort_values(['Timeframe'])
        seq = seq['ItemId'].values
        outseq = []
        for i in seq:
            if i in item2idx:
                outseq += [item2idx[i]]
        seqs += [outseq]
        end = time.time()
        print("\rGet_sequence: [%d/%d], %.2f , usetime: %fs" % (count, len(sess_ids), count/len(sess_ids) * 100, end - start),
              end='', flush=True)

    print("\n")
    # print(seqs)
    out_seqs, labs = process_seqs(seqs, shift)
    # print(out_seqs)
    # print(labs)
    print(len(out_seqs), len(labs))
    return out_seqs, labs


def preprocess(train, test, path=save_path):
    print("--------------")
    print("Start preprocess cikm16")
    # print("Start preprocess sample")
    item2idx = get_dict(train)
    train_seqs, train_labs = get_sequence(train, item2idx)
    test_seqs, test_labs = get_sequence(test, item2idx, train_labs[-1])
    train = (train_seqs, train_labs)
    test = (test_seqs, test_labs)
    if not os.path.exists(path):
        os.makedirs(path)

    pickle.dump(test, open(path+'/unaug_test.txt', 'wb'))
    pickle.dump(train, open(path+'/unaug_train.txt', 'wb'))
    print("finished")
```

```python id="tl9tQIl0uhF_" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1638285941796, "user_tz": -330, "elapsed": 3657, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fdce7982-7d5a-4753-cbec-70d74eb21400"
data = load_data(raw_path)
data = filter_data(data)
train, test = split_train_test(data)
preprocess(train, test)
```

<!-- #region id="UvhADxQ2bdUj" -->
### Augmented
<!-- #endregion -->

```python id="GLhfWYTvb4yB" executionInfo={"status": "ok", "timestamp": 1638286015021, "user_tz": -330, "elapsed": 641, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def load_data(file):
    print("Start load_data")
    # load csv
    data = pd.read_csv(file+'.csv', sep=';', header=0, usecols=[0, 2, 3, 4], dtype={0: np.int32, 1: np.int64, 2: str, 3: str})
    # specify header names
    data.columns = ['SessionId', 'ItemId', 'Timeframe', 'Eventdate']
    # convert time string to timestamp and remove the original column
    data['Time'] = data.Eventdate.apply(lambda x: datetime.strptime(x, '%Y-%m-%d').timestamp())
    print(data['Time'].max())
    del(data['Eventdate'])

    # output
    data_start = datetime.fromtimestamp(data.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)

    print('Loaded data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(data), data.SessionId.nunique(), data.ItemId.nunique(),
                 data_start.date().isoformat(), data_end.date().isoformat()))
    return data


def filter_data(data, min_item_support=5, min_session_length=2):
    print("Start filter_data")

    # y?
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[session_lengths > 1].index)]

    # filter item support
    item_supports = data.groupby('ItemId').size()
    data = data[np.in1d(data.ItemId, item_supports[item_supports >= min_item_support].index)]

    # filter session length
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[session_lengths >= min_session_length].index)]
    print(data['Time'].min())
    print(data['Time'].max())
    # output
    data_start = datetime.fromtimestamp(data.Time.astype(np.int64).min(), timezone.utc)
    data_end = datetime.fromtimestamp(data.Time.astype(np.int64).max(), timezone.utc)

    print('Filtered data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(data), data.SessionId.nunique(), data.ItemId.nunique(),
                 data_start.date().isoformat(), data_end.date().isoformat()))
    return data


def split_train_test(data):
    print("Start split_train_test")
    tmax = data.Time.max()
    session_max_times = data.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < tmax-7*86400].index
    session_test = session_max_times[session_max_times >= tmax-7*86400].index
    train = data[np.in1d(data.SessionId, session_train)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength >= 2].index)]

    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(), train.ItemId.nunique()))
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(), test.ItemId.nunique()))

    return train, test


def get_dict(data):
    print("Start get_dict")
    item2idx = {}
    pop_scores = data.groupby('ItemId').size().sort_values(ascending=False)
    pop_scores = pop_scores / pop_scores[:1].values[0]
    items = pop_scores.index
    for idx, item in enumerate(items):
        item2idx[item] = idx+1

    return item2idx


def process_seqs(seqs):
    start = time.time()
    out_seqs = []
    labs = []
    for count, seq in enumerate(seqs):
        for i in range(1, len(seq)):
            tar = seq[i]
            labs += [tar]
            out_seqs += [seq[:i]]
        end = time.time()
        print("\rprocess_seqs: [%d/%d], %.2f, usetime: %fs, " % (count, len(seqs), count/len(seqs) * 100, end - start),
              end='', flush=True)
    print("\n")
    return out_seqs, labs


def get_sequence(data, item2idx):
    start = time.time()
    sess_ids = data.drop_duplicates('SessionId', 'first')
    print(sess_ids)
    sess_ids.sort_values(['Time'], inplace=True)
    sess_ids = sess_ids['SessionId'].unique()
    seqs = []
    for count, sess_id in enumerate(sess_ids):
        seq = data[data['SessionId'].isin([sess_id])].sort_values(['Timeframe'])
        seq = seq['ItemId'].values
        outseq = []
        for i in seq:
            if i in item2idx:
                outseq += [item2idx[i]]
        seqs += [outseq]
        end = time.time()
        print("\rGet_sequence: [%d/%d], %.2f , usetime: %fs" % (count, len(sess_ids), count/len(sess_ids) * 100, end - start),
              end='', flush=True)
    print("\n")
    out_seqs, labs = process_seqs(seqs)
    print(len(out_seqs), len(labs))
    return out_seqs, labs


def preprocess(train, test, path=save_path):
    print("--------------")
    print("Start preprocess cikm16")
    item2idx = get_dict(train)
    train_seqs, train_labs = get_sequence(train, item2idx)
    test_seqs, test_labs = get_sequence(test, item2idx)
    train = (train_seqs, train_labs)
    test = (test_seqs, test_labs)
    if not os.path.exists(path):
        os.makedirs(path)
    print("Start Save data")

    pickle.dump(test, open(path+'/test.txt', 'wb'))
    pickle.dump(train, open(path+'/train.txt', 'wb'))
    print("finished")
```

```python id="9XXa7eK-br--" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1638286021175, "user_tz": -330, "elapsed": 4096, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="eff6254c-cbbd-487e-c44b-0eab4668c2bb"
data = load_data(raw_path)
data = filter_data(data)
train, test = split_train_test(data)
preprocess(train, test)
```

<!-- #region id="zB8v252abXBo" -->
## Neighborhood Retrieval
<!-- #endregion -->

```python id="CiE8LV3MaoHS" executionInfo={"status": "ok", "timestamp": 1638286104087, "user_tz": -330, "elapsed": 417, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class KNN:
    def __init__(self, k, all_sess, unaug_data, unaug_index, threshold=0.5, samples=10):
        self.k = k
        self.all_sess = all_sess
        self.threshold = threshold
        self.samples = samples
        self.item_sess_map = self.get_item_sess_map(unaug_index, unaug_data)
        self.no_pro_data = unaug_data
        self.no_pro_index = unaug_index


    def get_item_sess_map(self, unaug_index, unaug_data):
        item_sess_map = {}
        for index, sess in zip(unaug_index, unaug_data):
            items = np.unique(sess[:-1])
            for item in items:
                if item not in item_sess_map.keys():
                    item_sess_map[item] = []
                item_sess_map[item].append(index)
        print("get_item_sess_map over")
        return item_sess_map

    def jaccard(self, first, second):

        intersection = len(set(first).intersection(set(second)))
        union = len(set(first).union(set(second)))
        res = intersection / union

        return res

    def cosine(self, first, second):

        li = len(set(first).intersection(set(second)))
        la = len(first)
        lb = len(second)
        result = li / (math.sqrt(la) * math.sqrt(lb))

        return result

    def vec(self, first, second, pos_map):
        a = set(first).intersection(set(second))
        sum = 0
        for i in a:
            sum += pos_map[i]

        result = sum / len(pos_map)

        return result

    def find_sess(self, sess, item_sess_map):
        items = np.unique(sess)
        sess_index = []
        for item in items:
            sess_index += item_sess_map[item]
        return sess_index

    def calc_similarity(self, target_session, all_data, sess_index):
        neighbors = []
        session_items = np.unique(target_session)

        possible_sess_index = self.find_sess(session_items, self.item_sess_map)
        possible_sess_index = [p_index for p_index in possible_sess_index if p_index < sess_index]
        possible_sess_index = sorted(np.unique(possible_sess_index))[-self.samples:]
        possible_sess_index = sorted(np.unique(possible_sess_index))

        pos_map = {}
        length = len(target_session)

        count = 1
        for item in target_session:
            pos_map[item] = count / length
            count += 1

        for index in possible_sess_index:
            session = all_data[index]
            session_items_test = np.unique(session)
            similarity = np.around(self.cosine(session_items_test, session_items), 4)
            if similarity >= self.threshold:
                neighbors.append([index, similarity])

        return neighbors

    def get_neigh_sess(self, index):
        all_sess_neigh = []
        start = time.time()
        all_sess = self.all_sess[index:]
        for sess in all_sess:
            possible_neighbors = self.calc_similarity(sess, self.all_sess, index)
            possible_neighbors = sorted(possible_neighbors, reverse=True, key=lambda x: x[1])

            if len(possible_neighbors) > 0:
                possible_neighbors = list(np.asarray(possible_neighbors)[:, 0])
            if len(possible_neighbors) > self.k:
                all_sess_neigh.append(possible_neighbors[:self.k])
            elif len(possible_neighbors) > 0:
                all_sess_neigh.append(possible_neighbors)
            else:
                all_sess_neigh.append(0)
            index += 1
            end = time.time()

            if index % (len(self.all_sess) // 100) == 0:
                print("\rProcess_seqs: [%d/%d], %.2f, usetime: %fs, " % (index, len(self.all_sess), index/len(self.all_sess) * 100, end - start),
              end='', flush=True)

        return all_sess_neigh
```

```python id="QpwTWoZsdR0z"
org_test_data = pickle.load(open(save_path + '/test.txt', 'rb'))
org_train_data = pickle.load(open(save_path + '/train.txt', 'rb'))
unaug_test_data = pickle.load(open(save_path + '/unaug_test.txt', 'rb'))
unaug_train_data = pickle.load(open(save_path + '/unaug_train.txt', 'rb'))

test_data = org_test_data[0]
train_data = org_train_data[0]
all_data = np.concatenate((train_data, test_data), axis=0)

unaug_data = np.concatenate((unaug_train_data[0], unaug_test_data[0]), axis=0)
unaug_index = np.concatenate((unaug_train_data[1], unaug_test_data[1]), axis=0)

del org_test_data, org_train_data
del test_data, train_data
del unaug_train_data, unaug_test_data

k_num = [20,40,60,100,140, 160, 180, 200]

for k in k_num:
    knn = KNN(k, all_data, unaug_data, unaug_index)
    all_sess_neigh = knn.get_neigh_sess(0)
    pickle.dump(all_sess_neigh, open(save_path+"/neigh_data_"+str(k)+".txt", "wb"))
    lens = 0
    for i in all_sess_neigh:
        if i != 0:
            lens += len(i)
    print(lens / len(all_sess_neigh))
```

```python id="p7pFAf1Qhh_Y"
def print_txt(base_path, args, results, epochs, top_k, note=None, save_config=True):
    path = base_path + "\Best_result_top-"+str(top_k)+".txt"
    outfile = open(path, 'w')
    if note is not None:
        outfile.write("Note:\n"+note+"\n")
    if save_config:
        outfile.write("Configs:\n")
        for attr, value in sorted(args.__dict__.items()):
            outfile.write("{} = {}\n".format(attr, value))

    outfile.write('\nBest results:\n')
    outfile.write("Mrr@{}:\t{}\tEpoch: {}\n".format(top_k, results[1], epochs[1]))
    outfile.write("Recall@{}:\t{}\tEpoch: {}\n".format(top_k, results[0], epochs[0]))
    outfile.close()
```

<!-- #region id="G6TKOLdRhl7_" -->
## Model
<!-- #endregion -->

```python id="ToOgW5Crhl1a"
import torch
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.data import InMemoryDataset, Data, Dataset

from torch import Tensor
from torch.nn import Parameter as Param
from torch_geometric.nn.inits import uniform

import torch.nn as nn
from torch_geometric.nn import GATConv, SGConv, GCNConv, GatedGraphConv

import math
import collections
```

```python id="TaKUuC68jRIj"
class MultiSessionsGraph(InMemoryDataset):
    """Every session is a graph."""
    def __init__(self, root, phrase, knn_phrase, transform=None, pre_transform=None):
        """
        Args:
            root: 'sample', 'yoochoose1_4', 'yoochoose1_64' or 'diginetica'
            phrase: 'train' or 'test'
        """
        assert phrase in ['train', 'test']
        self.phrase = phrase
        self.knn_phrase = knn_phrase
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

    def find_neighs(self, index, knn_data):
        sess_neighs = knn_data[index]
        if sess_neighs == 0:
            return []
        else:
            return list(np.asarray(sess_neighs).astype(np.int32))

    def multi_process(self, train_data, knn_data, sess_index, y):
        # find neigh
        neigh_index = self.find_neighs(sess_index, knn_data)
        # neigh_index = []
        neigh_index.append(sess_index)
        temp_neighs = train_data[neigh_index]
        neighs = []

        # append y
        for neigh, idx in zip(temp_neighs, neigh_index):
            if idx != sess_index:
                neigh.append(y[idx])
            neighs.append(neigh)

        nodes = {}    # dict{15: 0, 16: 1, 18: 2, ...}
        all_senders = []
        all_receivers = []
        x = []
        i = 0
        for sess in neighs:
            senders = []
            for node in sess:
                if node not in nodes:
                    nodes[node] = i
                    x.append([node])
                    i += 1
                senders.append(nodes[node])
            receivers = senders[:]

            if len(senders) != 1:
                del senders[-1]  # the last item is a receiver
                del receivers[0]  # the first item is a sender
            all_senders += senders
            all_receivers += receivers

        sess = train_data[sess_index]
        sess_item_index = [nodes[item] for item in sess]
        # num_count = [count[i[0]] for i in x]

        sess_masks = np.zeros(len(nodes))
        sess_masks[sess_item_index] = 1

        pair = {}
        sur_senders = all_senders[:]
        sur_receivers = all_receivers[:]
        i = 0
        for sender, receiver in zip(sur_senders, sur_receivers):
            if str(sender) + '-' + str(receiver) in pair:
                pair[str(sender) + '-' + str(receiver)] += 1
                del all_senders[i]
                del all_receivers[i]
            else:
                pair[str(sender) + '-' + str(receiver)] = 1
                i += 1

        node_num = len(x)

        # num_count = torch.tensor(num_count, dtype=torch.float)
        edge_index = torch.tensor([all_senders, all_receivers], dtype=torch.long)
        x = torch.tensor(x, dtype=torch.long)
        node_num = torch.tensor([node_num], dtype=torch.long)
        sess_item_idx = torch.tensor(sess_item_index, dtype=torch.long)
        sess_masks = torch.tensor(sess_masks, dtype=torch.long)

        return x, edge_index, node_num, sess_item_idx, sess_masks

    def single_process(self, sequence, y):
        # sequence = [1, 2, 3, 2, 4]
        count = collections.Counter(sequence)
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
        num_count = [count[i[0]] for i in x]

        sess_item_index = [nodes[item] for item in sequence]

        if len(senders) != 1:
            del senders[-1]  # the last item is a receiver
            del receivers[0]  # the first item is a sender

        pair = {}
        sur_senders = senders[:]
        sur_receivers = receivers[:]
        i = 0
        for sender, receiver in zip(sur_senders, sur_receivers):
            if str(sender) + '-' + str(receiver) in pair:
                pair[str(sender) + '-' + str(receiver)] += 1
                del senders[i]
                del receivers[i]
            else:
                pair[str(sender) + '-' + str(receiver)] = 1
                i += 1

        count = collections.Counter(senders)
        out_degree_inv = [1 / count[i] for i in senders]

        count = collections.Counter(receivers)
        in_degree_inv = [1 / count[i] for i in receivers]

        in_degree_inv = torch.tensor(in_degree_inv, dtype=torch.float)
        out_degree_inv = torch.tensor(out_degree_inv, dtype=torch.float)

        edge_count = [pair[str(senders[i]) + '-' + str(receivers[i])] for i in range(len(senders))]
        edge_count = torch.tensor(edge_count, dtype=torch.float)

        # senders, receivers = senders + receivers, receivers + senders

        edge_index = torch.tensor([senders, receivers], dtype=torch.long)
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor([y], dtype=torch.long)
        num_count = torch.tensor(num_count, dtype=torch.float)
        sequence = torch.tensor(sequence, dtype=torch.long)
        sequence_len = torch.tensor([len(sequence)], dtype=torch.long)
        sess_item_idx = torch.tensor(sess_item_index, dtype=torch.long)


        return x, y, num_count, edge_index, edge_count, sess_item_idx, sequence_len, in_degree_inv, out_degree_inv

    def process(self):
        start = time.time()
        train_data = pickle.load(open(self.raw_dir + '/' + 'train.txt', 'rb'))
        test_data = pickle.load(open(self.raw_dir + '/' + 'test.txt', 'rb'))
        # knn_data = np.load(self.raw_dir + '/' + self.knn_phrase + '.npy')
        knn_data = pickle.load(open(self.raw_dir + '/' + self.knn_phrase + '.txt', "rb"))
        data_list = []
        if self.phrase == "train":
            sess_index = 0
            data = train_data
            total_data = np.asarray(train_data[0])
            total_label = np.asarray(train_data[1])
        else:
            sess_index = len(train_data[0])
            data = test_data
            total_data = np.concatenate((train_data[0], test_data[0]), axis=0)
            total_label = np.concatenate((train_data[1], test_data[1]), axis=0)

        for sequence, y in zip(data[0], data[1]):

            mt_x, mt_edge_index, mt_node_num, mt_sess_item_idx, sess_masks = \
                self.multi_process(total_data, knn_data, sess_index, total_label)

            x, y, num_count, edge_index, edge_count, sess_item_idx, sequence_len, in_degree_inv, out_degree_inv = \
                self.single_process(sequence, y)

            session_graph = Data(x=x, y=y, num_count=num_count, sess_item_idx=sess_item_idx,
                                    edge_index=edge_index, edge_count=edge_count, sequence_len=sequence_len,
                                    in_degree_inv=in_degree_inv, out_degree_inv=out_degree_inv,
                                    mt_x=mt_x, mt_edge_index=mt_edge_index, mt_node_num=mt_node_num,
                                    mt_sess_item_idx=mt_sess_item_idx, sess_masks=sess_masks)

            data_list.append(session_graph)
            sess_index += 1

            end = time.time()
            if sess_index % (len(data[0]) // 1000) == 0:
                print("\rProcess_seqs: [%d/%d], %.2f, usetime: %fs, " % (sess_index, len(data[0]), sess_index/len(data[0]) * 100, end - start),
              end='', flush=True)
        print('\nStart collate')
        data, slices = self.collate(data_list)
        print('\nStart save')
        torch.save((data, slices), self.processed_paths[0])
```

```python id="W4yM19tPh6o7"
def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


def kaiming_uniform(tensor, fan, a):
    if tensor is not None:
        bound = math.sqrt(6 / ((1 + a**2) * fan))
        tensor.data.uniform_(-bound, bound)


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(1)


def normal(tensor, mean, std):
    if tensor is not None:
        tensor.data.normal_(mean, std)


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

class InOutGATConv(MessagePassing):
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
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=8,
                 concat=False,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True,
                 middle_layer=False,
                 **kwargs):
        super(InOutGATConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.middle_layer = middle_layer
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight1 = Parameter(
            torch.Tensor(2, in_channels, heads * out_channels))
        self.weight2 = Parameter(
            torch.Tensor(2, in_channels, heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        if concat and not middle_layer:
            self.rnn = torch.nn.GRUCell(2 * out_channels * heads, in_channels * heads, bias=bias)
        elif middle_layer:
            self.rnn = torch.nn.GRUCell(2 * out_channels * heads, in_channels, bias=bias)
        else:
            self.rnn = torch.nn.GRUCell(2 * out_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight1)
        glorot(self.weight2)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, sess_masks):
        """"""
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        sess_masks = sess_masks.view(sess_masks.shape[0], 1).float()
        xs = x * sess_masks
        xns = x * (1 - sess_masks)

        # self.flow = 'source_to_target'
        # x1 = torch.mm(x, self.weight[0]).view(-1, self.heads, self.out_channels)
        # m1 = self.propagate(edge_index, x=x1, num_nodes=x.size(0))
        # self.flow = 'target_to_source'
        # x2 = torch.mm(x, self.weight[1]).view(-1, self.heads, self.out_channels)
        # m2 = self.propagate(edge_index, x=x2, num_nodes=x.size(0))

        self.flow = 'source_to_target'
        x1s = torch.mm(xs, self.weight1[0]).view(-1, self.heads, self.out_channels)
        print(x1s.shape())
        x1ns = torch.mm(xns, self.weight2[0]).view(-1, self.heads, self.out_channels)
        print(x1ns.shape())
        x1 = x1s + x1ns
        m1 = self.propagate(edge_index, x=x1, num_nodes=x.size(0))
        self.flow = 'target_to_source'
        x2s = torch.mm(xs, self.weight1[1]).view(-1, self.heads, self.out_channels)
        x2ns = torch.mm(xns, self.weight2[1]).view(-1, self.heads, self.out_channels)
        x2 = x2s + x2ns
        m2 = self.propagate(edge_index, x=x2, num_nodes=x.size(0))

        if not self.middle_layer:
            if self.concat:
                x = x.repeat(1, self.heads)
            else:
                x = x.view(-1, self.heads, self.out_channels).mean(dim=1)

        # x = self.rnn(torch.cat((m1, m2), dim=-1), x)
        x = m1 + m2
        # x = m1
        return x

    def message(self, edge_index_i, x_i, x_j, num_nodes):
        # Compute attention coefficients.
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

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


class InOutGATConv_intra(MessagePassing):
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
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=8,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True,
                 middle_layer=False,
                 **kwargs):
        super(InOutGATConv_intra, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.middle_layer = middle_layer
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(
            torch.Tensor(2, in_channels, heads * out_channels))
        self.weight1 = Parameter(
            torch.Tensor(2, in_channels, heads * out_channels))
        self.weight2 = Parameter(
            torch.Tensor(2, in_channels, heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        if concat and not middle_layer:
            self.rnn = torch.nn.GRUCell(2 * out_channels * heads, in_channels * heads, bias=bias)
        elif middle_layer:
            self.rnn = torch.nn.GRUCell(2 * out_channels * heads, in_channels, bias=bias)
        else:
            self.rnn = torch.nn.GRUCell(2 * out_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight1)
        glorot(self.weight2)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, sess_masks):
        """"""
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # sess_masks = sess_masks.view(sess_masks.shape[0], 1).float()
        # xs = x * sess_masks
        # xns = x * (1 - sess_masks)

        self.flow = 'source_to_target'
        x1 = torch.mm(x, self.weight[0]).view(-1, self.heads, self.out_channels)
        m1 = self.propagate(edge_index, x=x1, num_nodes=x.size(0))
        self.flow = 'target_to_source'
        x2 = torch.mm(x, self.weight[1]).view(-1, self.heads, self.out_channels)
        m2 = self.propagate(edge_index, x=x2, num_nodes=x.size(0))

        # self.flow = 'source_to_target'
        # x1s = torch.mm(xs, self.weight1[0]).view(-1, self.heads, self.out_channels)
        # x1ns = torch.mm(xns, self.weight2[0]).view(-1, self.heads, self.out_channels)
        # x1 = x1s + x1ns
        # m1 = self.propagate(edge_index, x=x1, num_nodes=x.size(0))
        # self.flow = 'target_to_source'
        # x2s = torch.mm(xs, self.weight1[1]).view(-1, self.heads, self.out_channels)
        # x2ns = torch.mm(xns, self.weight2[1]).view(-1, self.heads, self.out_channels)
        # x2 = x2s + x2ns
        # m2 = self.propagate(edge_index, x=x2, num_nodes=x.size(0))

        if not self.middle_layer:
            if self.concat:
                x = x.repeat(1, self.heads)
            else:
                x = x.view(-1, self.heads, self.out_channels).mean(dim=1)

        # x = self.rnn(torch.cat((m1, m2), dim=-1), x)
        x = m1 + m2
        return x

    def message(self, edge_index_i, x_i, x_j, num_nodes):
        # Compute attention coefficients.
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

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

```python id="hG8oOf2wh6mv"
class InOutGGNN(MessagePassing):
    r"""The gated graph convolution operator from the `"Gated Graph Sequence
    Neural Networks" <https://arxiv.org/abs/1511.05493>`_ paper
    .. math::
        \mathbf{h}_i^{(0)} &= \mathbf{x}_i \, \Vert \, \mathbf{0}
        \mathbf{m}_i^{(l+1)} &= \sum_{j \in \mathcal{N}(i)} \mathbf{\Theta}
        \cdot \mathbf{h}_j^{(l)}
        \mathbf{h}_i^{(l+1)} &= \textrm{GRU} (\mathbf{m}_i^{(l+1)},
        \mathbf{h}_i^{(l)})
    up to representation :math:`\mathbf{h}_i^{(L)}`.
    The number of input channels of :math:`\mathbf{x}_i` needs to be less or
    equal than :obj:`out_channels`.
    Args:
        out_channels (int): Size of each input sample.
        num_layers (int): The sequence length :math:`L`.
        aggr (string): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, out_channels, num_layers, aggr='add', bias=True):
        super(InOutGGNN, self).__init__(aggr)

        self.out_channels = out_channels
        self.num_layers = num_layers

        self.weight = Param(Tensor(num_layers, 2, out_channels, out_channels))
        self.rnn = torch.nn.GRUCell(2 * out_channels, out_channels, bias=bias)
        self.bias_in = Param(Tensor(self.out_channels))
        self.bias_out = Param(Tensor(self.out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        size = self.out_channels
        uniform(size, self.weight)
        self.rnn.reset_parameters()

    def forward(self, x, edge_index, edge_weight=[None, None]):
        #print(edge_weight[0].size(), edge_weight[1].size)

        """"""
        h = x if x.dim() == 2 else x.unsqueeze(-1)
        if h.size(1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')

        if h.size(1) < self.out_channels:
            zero = h.new_zeros(h.size(0), self.out_channels - h.size(1))
            h = torch.cat([h, zero], dim=1)

        for i in range(self.num_layers):
            self.flow = 'source_to_target'
            h1 = torch.matmul(h, self.weight[i, 0])
            m1 = self.propagate(edge_index, x=h1, edge_weight=edge_weight[0], bias=self.bias_in)
            self.flow = 'target_to_source'
            h2 = torch.matmul(h, self.weight[i, 1])
            m2 = self.propagate(edge_index, x=h2, edge_weight=edge_weight[1], bias=self.bias_out)
            h = self.rnn(torch.cat((m1, m2), dim=-1), h)

        return h

    def message(self, x_j, edge_weight):
        if edge_weight is not None:
            return edge_weight.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out, bias):
        if bias is not None:
            return aggr_out + bias
        else:
            return aggr_out

    def __repr__(self):
        return '{}({}, num_layers={})'.format(
            self.__class__.__name__, self.out_channels, self.num_layers)
```

```python id="M4EFTsBkh6iI"
class SRGNN(nn.Module):
    """
    Args:
        hidden_size: the number of units in a hidden layer.
        n_node: the number of items in the whole item set for embedding layer.
    """
    def __init__(self, hidden_size, n_node, dropout=0.5, negative_slope=0.2, heads=8, item_fusing=False):
        super(SRGNN, self).__init__()
        self.hidden_size, self.n_node = hidden_size, n_node
        self.item_fusing = item_fusing
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        # self.gated = InOutGGNN(self.hidden_size, num_layers=1)

        self.gcn = GCNConv(in_channels=hidden_size, out_channels=hidden_size)
        self.gcn2 = GCNConv(in_channels=hidden_size, out_channels=hidden_size)

        self.gated = SGConv(in_channels=hidden_size, out_channels=hidden_size, K=2)
        # self.gated = InOutGATConv_intra(in_channels=hidden_size, out_channels=hidden_size, dropout=dropout,
        #                           negative_slope=negative_slope, heads=heads, concat=True)
        # self.gated2 = InOutGATConv(in_channels=hidden_size * heads, out_channels=hidden_size, dropout=dropout,
        #                            negative_slope=negative_slope, heads=heads, concat=True, middle_layer=True)
        # self.gated3 = InOutGATConv(in_channels=hidden_size * heads, out_channels=hidden_size, dropout=dropout,
        #                            negative_slope=negative_slope, heads=heads, concat=False)

        self.W_1 = nn.Linear(self.hidden_size * 8, self.hidden_size)
        self.W_2 = nn.Linear(self.hidden_size * 8, self.hidden_size)
        self.q = nn.Linear(self.hidden_size, 1)
        self.W_3 = nn.Linear(16 * self.hidden_size, self.hidden_size)

        self.loss_function = nn.CrossEntropyLoss()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def rebuilt_sess(self, session_embedding, batchs, sess_item_index, seq_lens):
        sections = torch.bincount(batchs)
        split_embs = torch.split(session_embedding, tuple(sections.cpu().numpy()))
        sess_item_index = torch.split(sess_item_index, tuple(seq_lens.cpu().numpy()))

        rebuilt_sess = []
        for embs, index in zip(split_embs, sess_item_index):
            sess = tuple(embs[i].view(1, -1) for i in index)
            sess = torch.cat(sess, dim=0)
            rebuilt_sess.append(sess)
        return tuple(rebuilt_sess)


    def get_h_s(self, hidden, seq_len):
        # split whole x back into graphs G_i
        v_n = tuple(nodes[-1].view(1, -1) for nodes in hidden)
        v_n_repeat = tuple(nodes[-1].view(1, -1).repeat(nodes.shape[0], 1) for nodes in hidden)
        v_n_repeat = torch.cat(v_n_repeat, dim=0)
        hidden = torch.cat(hidden, dim=0)

        # Eq(6)
        # print("v_n_repeat", v_n_repeat.size())
        # print("hidden", hidden.size())
        alpha = self.q(torch.sigmoid(self.W_1(v_n_repeat) + self.W_2(hidden)))    # |V|_i * 1

        s_g_whole = alpha * hidden    # |V|_i * hidden_size
        s_g_split = torch.split(s_g_whole, tuple(seq_len.cpu().numpy()))    # split whole s_g into graphs G_i
        s_g = tuple(torch.sum(embeddings, dim=0).view(1, -1) for embeddings in s_g_split)

        # Eq(7)
        # print("torch.cat((torch.cat(v_n, dim=0), torch.cat(s_g, dim=0)), dim=1)", torch.cat((torch.cat(v_n, dim=0), torch.cat(s_g, dim=0)), dim=1).size())
        h_s = self.W_3(torch.cat((torch.cat(v_n, dim=0), torch.cat(s_g, dim=0)), dim=1))
        # h_s = torch.cat((torch.cat(v_n, dim=0), torch.cat(s_g, dim=0)), dim=1)
        return h_s

    def forward(self, data, hidden):
        edge_index, batch, edge_count, in_degree_inv, out_degree_inv, num_count, sess_item_index, seq_len = \
            data.edge_index, data.batch, data.edge_count, data.in_degree_inv, data.out_degree_inv,\
            data.num_count, data.sess_item_idx, data.sequence_len

        hidden = self.gated.forward(hidden, edge_index)
        # hidden = self.gcn.forward(hidden, edge_index)
        # hidden = self.gcn2.forward(hidden, edge_index)
        sess_embs = self.rebuilt_sess(hidden, batch, sess_item_index, seq_len)
        if self.item_fusing:
            return sess_embs
        else:
            return self.get_h_s(sess_embs, seq_len)
```

```python id="jO_DQcFgizkN"
class GroupGraph(Module):
    def __init__(self, hidden_size, dropout=0.5, negative_slope=0.2, heads=8, item_fusing=False):
        super(GroupGraph, self).__init__()
        self.hidden_size = hidden_size
        self.item_fusing = item_fusing

        self.W_1 = nn.Linear(8 * self.hidden_size, self.hidden_size)
        self.W_2 = nn.Linear(8 * self.hidden_size, self.hidden_size)
        self.q = nn.Linear(self.hidden_size, 1)
        self.W_3 = nn.Linear(16 * self.hidden_size, self.hidden_size)

        # self.gat = GATConv(in_channels=hidden_size, out_channels=hidden_size, dropout=dropout, negative_slope=negative_slope, heads=heads, concat=True)
        # self.gat2 = GATConv(in_channels=hidden_size*heads, out_channels=hidden_size*heads, dropout=dropout, negative_slope=negative_slope, heads=heads, concat=False)
        # self.gat3 = GATConv(in_channels=hidden_size*heads, out_channels=hidden_size, dropout=dropout, negative_slope=negative_slope, heads=heads, concat=True)
        # self.gat_out = GATConv(in_channels=hidden_size*heads, out_channels=hidden_size, dropout=dropout, negative_slope=negative_slope, heads=heads, concat=False)
        # self.gated = InOutGGNN(self.hidden_size, num_layers=2)
        self.gcn = GCNConv(in_channels=hidden_size, out_channels=hidden_size)
        self.gcn2 = GCNConv(in_channels=hidden_size, out_channels=hidden_size)

        self.sgcn = SGConv(in_channels=hidden_size, out_channels=hidden_size, K=2)
        # self.gat = InOutGATConv(in_channels=hidden_size, out_channels=hidden_size, dropout=dropout,
        #                           negative_slope=negative_slope, heads=heads, concat=True)
        # self.gat2 = InOutGATConv(in_channels=hidden_size * heads, out_channels=hidden_size, dropout=dropout,
        #                            negative_slope=negative_slope, heads=heads, concat=False)
        #

    def group_att_old(self, session_embedding, node_num, batch_h_s):  # hs: # batch_size x latent_size
        v_i = torch.split(session_embedding, tuple(node_num))    # split whole x back into graphs G_i
        h_s_repeat = tuple(h_s.view(1, -1).repeat(nodes.shape[0], 1) for h_s, nodes in zip(batch_h_s, v_i))    # repeat |V|_i times for the last node embedding

        alpha = self.q(torch.sigmoid(self.W_1(torch.cat(h_s_repeat, dim=0)) + self.W_2(session_embedding)))    # |V|_i * 1
        s_g_whole = alpha * session_embedding    # |V|_i * hidden_size
        s_g_split = torch.split(s_g_whole, tuple(node_num.cpu().numpy()))    # split whole s_g into graphs G_i
        s_g = tuple(torch.sum(embeddings, dim=0).view(1, -1) for embeddings in s_g_split)

        return torch.cat(s_g, dim=0)

    def group_att(self, session_embedding, hidden, node_num, num_count):  # hs: # batch_size x latent_size
        v_i = torch.split(session_embedding, tuple(node_num))    # split whole x back into graphs G_i
        v_n = tuple(nodes[-1].view(1, -1) for nodes in hidden)
        v_n_repeat = tuple(sess_nodes[-1].view(1, -1).repeat(nodes.shape[0], 1) for sess_nodes, nodes in zip(hidden, v_i))    # repeat |V|_i times for the last node embedding

        alpha = self.q(torch.sigmoid(self.W_1(torch.cat(v_n_repeat, dim=0)) + self.W_2(session_embedding)))    # |V|_i * 1
        s_g_whole = num_count.view(-1, 1) * alpha * session_embedding    # |V|_i * hidden_size
        s_g_split = torch.split(s_g_whole, tuple(node_num.cpu().numpy()))    # split whole s_g into graphs G_i
        s_g = tuple(torch.sum(embeddings, dim=0).view(1, -1) for embeddings in s_g_split)

        h_s = self.W_3(torch.cat((torch.cat(v_n, dim=0), torch.cat(s_g, dim=0)), dim=1))

        return h_s


    def rebuilt_sess(self, session_embedding, node_num, sess_item_index, seq_lens):
        split_embs = torch.split(session_embedding, tuple(node_num))
        sess_item_index = torch.split(sess_item_index, tuple(seq_lens.cpu().numpy()))

        rebuilt_sess = []
        for embs, index in zip(split_embs, sess_item_index):
            sess = tuple(embs[i].view(1, -1) for i in index)
            sess = torch.cat(sess, dim=0)
            rebuilt_sess.append(sess)
        return tuple(rebuilt_sess)

    def get_h_group(self, hidden, seq_len):
        # split whole x back into graphs G_i
        v_n = tuple(nodes[-1].view(1, -1) for nodes in hidden)
        v_n_repeat = tuple(nodes[-1].view(1, -1).repeat(nodes.shape[0], 1) for nodes in hidden)
        v_n_repeat = torch.cat(v_n_repeat, dim=0)
        hidden = torch.cat(hidden, dim=0)

        # Eq(5)
        alpha = self.q(torch.sigmoid(self.W_1(v_n_repeat) + self.W_2(hidden)))    # |V|_i * 1
        s_g_whole = alpha * hidden    # |V|_i * hidden_size
        # s_g_whole = hidden
        s_g_split = torch.split(s_g_whole, tuple(seq_len.cpu().numpy()))    # split whole s_g into graphs G_i
        s_g = tuple(torch.sum(embeddings, dim=0).view(1, -1) for embeddings in s_g_split)
        # s_g = tuple(torch.mean(embeddings, dim=0).view(1, -1) for embeddings in s_g_split)

        h_s = self.W_3(torch.cat((torch.cat(v_n, dim=0), torch.cat(s_g, dim=0)), dim=1))
        # h_s = torch.cat((torch.cat(v_n, dim=0), torch.cat(s_g, dim=0)), dim=1)
        return h_s

    def h_mean(self, hidden, node_num):
        split_embs = torch.split(hidden, tuple(node_num))
        means = []
        for embs in split_embs:
            mean = torch.mean(embs, dim=0)
            means.append(mean)

        means = torch.cat(tuple(means), dim=0).view(len(split_embs), -1)

        return means

    def forward(self, hidden, data):
        # edge_index, node_num, batch, sess_item_index, seq_lens, sess_masks = \
        #     data.mt_edge_index, data.mt_node_num, data.batch, data.mt_sess_item_idx, data.sequence_len, data.sess_masks
        edge_index, node_num, batch, sess_item_index, seq_lens = \
            data.mt_edge_index, data.mt_node_num, data.batch, data.mt_sess_item_idx, data.sequence_len


        # edge_count, in_degree_inv, out_degree_inv = data.mt_edge_count, data.mt_in_degree_inv, data.mt_out_degree_inv
        # hidden = self.gat.forward(hidden, edge_index, sess_masks)
        # hidden = self.gat2.forward(hidden, edge_index)
        # hidden = self.gat3.forward(hidden, edge_index)

        # hidden = self.gat.forward(hidden, edge_index, sess_masks)

        hidden - self.sgcn(hidden, edge_index)
        # hidden = self.gcn.forward(hidden, edge_index)
        # hidden = self.gcn2.forward(hidden, edge_index)

        # hidden = self.gat.forward(hidden, edge_index)
        # hidden = self.gated.forward(hidden, edge_index, [edge_count * in_degree_inv, edge_count * out_degree_inv])
        # hidden = self.gated.forward(hidden, edge_index)

        # hidden = self.gat1.forward(hidden, edge_index)

        sess_hidden = self.rebuilt_sess(hidden, node_num, sess_item_index, seq_lens)

        if self.item_fusing:
            return sess_hidden
        else:
            return self.get_h_group(sess_hidden, seq_lens)
```

```python id="2POOFBLKi4r0"
class Embedding2Score(nn.Module):
    def __init__(self, hidden_size, n_node, using_represent, item_fusing):
        super(Embedding2Score, self).__init__()
        self.hidden_size = hidden_size
        self.n_node = n_node
        self.using_represent = using_represent
        self.item_fusing = item_fusing


        self.W_1 = nn.Linear(self.hidden_size, self.hidden_size * 2)
        self.W_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_3 = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, h_s, h_group, final_s, item_embedding_table):
        emb = item_embedding_table.weight.transpose(1, 0)
        if self.item_fusing:
            z_i_hat = torch.mm(final_s, emb)
        else:
            gate = F.sigmoid(self.W_2(h_s) + self.W_3(h_group))
            sess_rep = h_s * gate + h_group * (1 - gate)
            if self.using_represent == 'comb':
                z_i_hat = torch.mm(sess_rep, emb)
            elif self.using_represent == 'h_s':
                z_i_hat = torch.mm(h_s, emb)
            elif self.using_represent == 'h_group':
                z_i_hat = torch.mm(h_group, emb)
            else:
                raise NotImplementedError

        return z_i_hat,


class ItemFusing(nn.Module):
    def __init__(self, hidden_size):
        super(ItemFusing, self).__init__()
        self.hidden_size = hidden_size
        self.use_rnn = True
        self.Wf1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.Wf2 = nn.Linear(self.hidden_size, self.hidden_size)

        self.W_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.q = nn.Linear(self.hidden_size, 1)
        self.W_3 = nn.Linear(2 * self.hidden_size, self.hidden_size)

        self.rnn = torch.nn.GRUCell(hidden_size, hidden_size, bias=True)

    def forward(self, intra_item_emb, inter_item_emb, seq_len):
        final_emb = self.item_fusing(intra_item_emb, inter_item_emb)
        # final_emb = self.avg_fusing(intra_item_emb, inter_item_emb)
        final_s = self.get_final_s(final_emb, seq_len)
        return final_s

    def item_fusing(self, local_emb, global_emb):
        local_emb = torch.cat(local_emb, dim=0)
        global_emb = torch.cat(global_emb, dim=0)
        if self.use_rnn:
            final_emb = self.rnn(local_emb, global_emb)
        else:
            gate = F.sigmoid(self.Wf1(local_emb) + self.Wf2(global_emb))
            final_emb = local_emb * gate + global_emb * (1 - gate)

        return final_emb

    def cnn_fusing(self, local_emb, global_emb):
        local_emb = torch.cat(local_emb, dim=0)
        global_emb = torch.cat(global_emb, dim=0)
        embedding = torch.stack([local_emb, global_emb], dim=2)
        embedding = embedding.permute(0, 2, 1)
        embedding = self.conv(embedding).permute(0, 2, 1)
        embedding = self.W_c(embedding).squeeze()
        return embedding

    def max_fusing(self, local_emb, global_emb):
        local_emb = torch.cat(local_emb, dim=0)
        global_emb = torch.cat(global_emb, dim=0)
        embedding = torch.stack([local_emb, global_emb], dim=2)
        embedding = torch.max(embedding, dim=2)[0]
        return embedding

    def avg_fusing(self, local_emb, global_emb):
        local_emb = torch.cat(local_emb, dim=0)
        global_emb = torch.cat(global_emb, dim=0)
        embedding = (local_emb + global_emb) / 2
        return embedding

    def concat_fusing(self, local_emb, global_emb):
        local_emb = torch.cat(local_emb, dim=0)
        global_emb = torch.cat(global_emb, dim=0)
        embedding = torch.cat([local_emb, global_emb], dim=1)
        embedding = self.W_4(embedding)
        return embedding
    def get_final_s(self, hidden, seq_len):
        hidden = torch.split(hidden, tuple(seq_len.cpu().numpy()))
        v_n = tuple(nodes[-1].view(1, -1) for nodes in hidden)
        v_n_repeat = tuple(nodes[-1].view(1, -1).repeat(nodes.shape[0], 1) for nodes in hidden)
        v_n_repeat = torch.cat(v_n_repeat, dim=0)
        hidden = torch.cat(hidden, dim=0)

        # Eq(6)
        alpha = self.q(torch.sigmoid(self.W_1(v_n_repeat) + self.W_2(hidden)))    # |V|_i * 1
        s_g_whole = alpha * hidden    # |V|_i * hidden_size
        s_g_split = torch.split(s_g_whole, tuple(seq_len.cpu().numpy()))    # split whole s_g into graphs G_i
        s_g = tuple(torch.sum(embeddings, dim=0).view(1, -1) for embeddings in s_g_split)

        # Eq(7)
        h_s = self.W_3(torch.cat((torch.cat(v_n, dim=0), torch.cat(s_g, dim=0)), dim=1))
        # h_s = torch.cat((torch.cat(v_n, dim=0), torch.cat(s_g, dim=0)), dim=1)
        return h_s


class NARM(nn.Module):
    def __init__(self, opt):
        super(NARM, self).__init__()
        self.hidden_size = opt.hidden_size
        self.gru = nn.GRU(self.hidden_size * 2, self.hidden_size, batch_first=True)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)

    def sess_att(self, hidden, ht, mask):
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        hs = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        # hs = torch.sum(alpha * hidden, 1)
        return hs

    def padding(self, intra_item_embs, inter_item_embs, seq_lens):
        inter_padded, intra_padded = [], []
        max_len = max(seq_lens).detach().cpu().numpy()
        for intra_item_emb, inter_item_emb, seq_len in zip(intra_item_embs, inter_item_embs, seq_lens):
            if intra_item_emb.size(0) < max_len:
                pad_vec = torch.zeros(max_len - intra_item_emb.size(0), self.hidden_size)
                pad_vec = pad_vec.to('cuda')
                intra_item_emb = torch.cat((intra_item_emb, pad_vec), dim=0)
                inter_item_emb = torch.cat((inter_item_emb, pad_vec), dim=0)
            inter_padded.append(inter_item_emb.unsqueeze(dim=0))
            intra_padded.append(intra_item_emb.unsqueeze(dim=0))
        inter_padded = torch.cat(tuple(inter_padded), dim=0)
        intra_padded = torch.cat(tuple(intra_padded), dim=0)
        item_embs = torch.cat((inter_padded, intra_padded), dim=-1)
        return item_embs

    def get_h_s(self, padded, seq_lens, masks):
        outputs, _ = self.gru(padded)
        output_last = outputs[torch.arange(seq_lens.shape[0]).long(), seq_lens - 1]
        hs = self.sess_att(outputs, output_last, masks)
        return hs

    def forward(self, intra_item_embs, inter_item_embs, seq_lens):
        max_len = max(seq_lens).detach().cpu().numpy()
        masks = [[1] * le + [0] * (max_len - le) for le in seq_lens.detach().cpu().numpy()]
        masks = torch.tensor(masks).to('cuda')
        item_embs = self.padding(intra_item_embs, inter_item_embs, seq_lens)
        return self.get_h_s(item_embs, seq_lens, masks)


class CNNFusing(nn.Module):
    def __init__(self, hidden_size, num_filters):
        super(CNNFusing, self).__init__()
        self.hidden_size = hidden_size
        self.num_filters = num_filters

        self.Wf1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.Wf2 = nn.Linear(self.hidden_size, self.hidden_size)

        self.W_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.q = nn.Linear(self.hidden_size, 1)
        self.W_3 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.W_4 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)

        # self.conv = torch.nn.Conv2d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=(1, 2))
        self.conv = torch.nn.Conv1d(in_channels=2, out_channels=self.num_filters, kernel_size=1)
        self.W_c = nn.Linear(self.num_filters, 1)
    # def forward(self, inter_item_emb, intra_item_emb, seq_len):
    #     final_emb = self.cnn_fusing(inter_item_emb, intra_item_emb)
    #     final_s = self.get_final_s(final_emb, seq_len)
    #     return final_s
    def forward(self, intra_item_emb, inter_item_emb, seq_len):
        # final_emb = self.cnn_fusing(intra_item_emb, inter_item_emb)
        # final_emb = self.concat_fusing(intra_item_emb, inter_item_emb)
        # final_emb = self.avg_fusing(intra_item_emb, inter_item_emb)
        final_emb = self.max_fusing(intra_item_emb, inter_item_emb)
        # final_emb = intra_item_emb
        final_s = self.get_final_s(final_emb, seq_len)
        return final_s

    def cnn_fusing(self, local_emb, global_emb):
        local_emb = torch.cat(local_emb, dim=0)
        global_emb = torch.cat(global_emb, dim=0)
        embedding = torch.stack([local_emb, global_emb], dim=2)
        embedding = embedding.permute(0, 2, 1)
        embedding = self.conv(embedding).permute(0, 2, 1)
        embedding = self.W_c(embedding).squeeze()
        return embedding

    def max_fusing(self, local_emb, global_emb):
        local_emb = torch.cat(local_emb, dim=0)
        global_emb = torch.cat(global_emb, dim=0)
        embedding = torch.stack([local_emb, global_emb], dim=2)
        embedding = torch.max(embedding, dim=2)[0]
        return embedding

    def avg_fusing(self, local_emb, global_emb):
        local_emb = torch.cat(local_emb, dim=0)
        global_emb = torch.cat(global_emb, dim=0)
        embedding = (local_emb + global_emb) / 2
        return embedding

    def concat_fusing(self, local_emb, global_emb):
        local_emb = torch.cat(local_emb, dim=0)
        global_emb = torch.cat(global_emb, dim=0)
        embedding = torch.cat([local_emb, global_emb], dim=1)
        embedding = self.W_4(embedding)
        return embedding

    def get_final_s(self, hidden, seq_len):
        hidden = torch.split(hidden, tuple(seq_len.cpu().numpy()))
        v_n = tuple(nodes[-1].view(1, -1) for nodes in hidden)
        v_n_repeat = tuple(nodes[-1].view(1, -1).repeat(nodes.shape[0], 1) for nodes in hidden)
        v_n_repeat = torch.cat(v_n_repeat, dim=0)
        hidden = torch.cat(hidden, dim=0)

        # Eq(6)
        alpha = self.q(torch.sigmoid(self.W_1(v_n_repeat) + self.W_2(hidden)))  # |V|_i * 1
        s_g_whole = alpha * hidden  # |V|_i * hidden_size
        s_g_split = torch.split(s_g_whole, tuple(seq_len.cpu().numpy()))  # split whole s_g into graphs G_i
        s_g = tuple(torch.sum(embeddings, dim=0).view(1, -1) for embeddings in s_g_split)

        # Eq(7)
        h_s = self.W_3(torch.cat((torch.cat(v_n, dim=0), torch.cat(s_g, dim=0)), dim=1))
        # h_s = torch.cat((torch.cat(v_n, dim=0), torch.cat(s_g, dim=0)), dim=1)
        return h_s



class GraphModel(nn.Module):
    def __init__(self, opt, n_node):
        super(GraphModel, self).__init__()
        self.hidden_size, self.n_node = opt.hidden_size, n_node
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.dropout = opt.gat_dropout
        self.negative_slope = opt.negative_slope
        self.heads = opt.heads
        self.item_fusing = opt.item_fusing
        self.num_filters = opt.num_filters

        self.srgnn = SRGNN(self.hidden_size, n_node=n_node, item_fusing=opt.item_fusing)
        self.group_graph = GroupGraph(self.hidden_size, dropout=self.dropout, negative_slope=self.negative_slope,
                                      heads=self.heads, item_fusing=opt.item_fusing)
        self.fuse_model = ItemFusing(self.hidden_size)
        self.narm = NARM(opt)
        self.cnn_fusing = CNNFusing(self.hidden_size, self.num_filters)
        self.e2s = Embedding2Score(self.hidden_size, n_node, opt.using_represent, opt.item_fusing)

        self.loss_function = nn.CrossEntropyLoss()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, data):
        if self.item_fusing:
            x = data.x - 1
            embedding = self.embedding(x)
            embedding = embedding.squeeze()
            intra_item_emb = self.srgnn(data, embedding)
            num_filters = self.num_filters

            mt_x = data.mt_x - 1

            embedding = self.embedding(mt_x)
            embedding = embedding.squeeze()

            inter_item_emb = self.group_graph.forward(embedding, data)

            # final_s = self.fuse_model.forward(intra_item_emb, inter_item_emb, data.sequence_len)
            # final_s = self.narm.forward(intra_item_emb, inter_item_emb, data.sequence_len)
            final_s = self.cnn_fusing.forward(intra_item_emb, inter_item_emb, data.sequence_len)

            scores = self.e2s(h_s=None, h_group=None, final_s=final_s, item_embedding_table=self.embedding)

        else:
            x = data.x - 1
            embedding = self.embedding(x)
            embedding = embedding.squeeze()
            h_s = self.srgnn(data, embedding)

            mt_x = data.mt_x - 1

            embedding = self.embedding(mt_x)
            embedding = embedding.squeeze()

            h_group = self.group_graph.forward(embedding, data)
            scores = self.e2s(h_s=h_s, h_group=h_group, final_s=None, item_embedding_table=self.embedding)

        return scores[0]
```

<!-- #region id="WrO1uT0dhlyt" -->
## Trainer
<!-- #endregion -->

```python id="UPqeYRD1h1ZM"
import numpy as np
import logging
import time



def forward(model, loader, device, writer, epoch, top_k=20, optimizer=None, train_flag=True):
    start = time.time()
    if train_flag:
        model.train()
    else:
        model.eval()
        hit10, mrr10 = [], []
        hit5, mrr5 = [], []
        hit20, mrr20 = [], []

    mean_loss = 0.0
    updates_per_epoch = len(loader)
    test_dict = {}
    for i, batch in enumerate(loader):
        if train_flag:
            optimizer.zero_grad()
        scores = model(batch.to(device))
        targets = batch.y - 1
        loss = model.loss_function(scores, targets)

        if train_flag:
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss/train_batch_loss', loss.item(), epoch * updates_per_epoch + i)
        else:
            sub_scores = scores.topk(20)[1]    # batch * top_k
            for score, target in zip(sub_scores.detach().cpu().numpy(), targets.detach().cpu().numpy()):
                hit20.append(np.isin(target, score))
                if len(np.where(score == target)[0]) == 0:
                    mrr20.append(0)
                else:
                    mrr20.append(1 / (np.where(score == target)[0][0] + 1))

            sub_scores = scores.topk(top_k)[1]    # batch * top_k
            for score, target in zip(sub_scores.detach().cpu().numpy(), targets.detach().cpu().numpy()):
                hit10.append(np.isin(target, score))
                if len(np.where(score == target)[0]) == 0:
                    mrr10.append(0)
                else:
                    mrr10.append(1 / (np.where(score == target)[0][0] + 1))

            sub_scores = scores.topk(5)[1]    # batch * top_k
            for score, target in zip(sub_scores.detach().cpu().numpy(), targets.detach().cpu().numpy()):
                hit5.append(np.isin(target, score))
                if len(np.where(score == target)[0]) == 0:
                    mrr5.append(0)
                else:
                    mrr5.append(1 / (np.where(score == target)[0][0] + 1))


        mean_loss += loss / batch.num_graphs
        end = time.time()
        print("\rProcess: [%d/%d]   %.2f   usetime: %fs" % (i, updates_per_epoch, i/updates_per_epoch * 100, end - start),
              end='', flush=True)
    print('\n')

    if train_flag:
        writer.add_scalar('loss/train_loss', mean_loss.item(), epoch)
        print("Train_loss: ", mean_loss.item())
    else:
        writer.add_scalar('loss/test_loss', mean_loss.item(), epoch)
        hit20 = np.mean(hit20) * 100
        mrr20 = np.mean(mrr20) * 100

        hit10 = np.mean(hit10) * 100
        mrr10 = np.mean(mrr10) * 100

        hit5 = np.mean(hit5) * 100
        mrr5 = np.mean(mrr5) * 100
        # writer.add_scalar('index/hit', hit, epoch)
        # writer.add_scalar('index/mrr', mrr, epoch)
        print("Result:")
        print("\tMrr@", 20, ": ", mrr20)
        print("\tRecall@", 20, ": ", hit20)

        print("\tMrr@", top_k, ": ", mrr10)
        print("\tRecall@", top_k, ": ", hit10)

        print("\tMrr@", 5, ": ", mrr5)
        print("\tRecall@", 5, ": ", hit5)
        # for seq_len in range(1, 31):
        #     sub_hit = test_dict[seq_len][0]
        #     sub_mrr = test_dict[seq_len][1]
        #     print("Len ", seq_len, ": Recall@", top_k, ": ", np.mean(sub_hit) * 100, "Mrr@", top_k, ": ", np.mean(sub_mrr) * 100)

        return mrr20, hit20, mrr10, hit10, mrr5, hit5


def case_study(model, loader, device, n_node):
    model.eval()
    for i, batch in enumerate(loader):
        sc, ss, sg, mg, alpha_s, alpha_g = model(batch.to(device))
        targets = batch.y - 1
        scs = sc.topk(n_node)[1].detach().cpu().numpy()
        sss = ss.topk(n_node)[1].detach().cpu().numpy()
        sgs = sg.topk(n_node)[1].detach().cpu().numpy()
        mgs = mg.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        # batch * top_k
        for sc, ss, sg, ms, a_s, a_g, target in zip(scs, sss, sgs, mgs, alpha_s, alpha_g, targets):
            rc = np.where(sc == target)[0][0] + 1
            rs = np.where(ss == target)[0][0] + 1
            rg = np.where(sg == target)[0][0] + 1
            print("rank c:", rc, "rank s:", rs, "rank g:", rg, "gate:", ms)
            print("att s:", a_s, "att g:", a_g)
```

<!-- #region id="OWHZhL0fh1W2" -->
## Main
<!-- #endregion -->

```python id="b-j_jzmgh1Ut"
import os
import argparse
import logging
from tqdm.notebook import tqdm
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
```

```python id="SSGHP_PEjlKP"
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yoochoose1_64', help='dataset name: diginetica/yoochoose1_64/sample')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=15, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.5, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=4, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--top_k', type=int, default=20, help='top K indicator for evaluation')
parser.add_argument('--negative_slope', type=float, default=0.2, help='negative_slope')
parser.add_argument('--gat_dropout', type=float, default=0.6, help='dropout rate in gat')
parser.add_argument('--heads', type=int, default=8, help='gat heads number')
parser.add_argument('--num_filters', type=int, default=2, help='gat heads number')
parser.add_argument('--using_represent', type=str, default='comb', help='comb, h_s, h_group')
parser.add_argument('--predict', type=bool, default=False, help='gat heads number')
parser.add_argument('--item_fusing', type=bool, default=True, help='gat heads number')
parser.add_argument('--random_seed', type=int, default=24, help='input batch size')
parser.add_argument('--id', type=int, default=120, help='id')
opt = parser.parse_args(args={})


def main():

    torch.manual_seed(opt.random_seed)
    torch.cuda.manual_seed(opt.random_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    cur_dir = os.getcwd()
    train_dataset = MultiSessionsGraph(cur_dir + '/datasets/' + opt.dataset, phrase='train', knn_phrase='neigh_data_'+str(opt.id))
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_dataset = MultiSessionsGraph(cur_dir + '/datasets/' + opt.dataset, phrase='test', knn_phrase='neigh_data_'+str(opt.id))
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

    log_dir = cur_dir + '/log/' + str(opt.dataset) + '/' + time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime())
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)

    if opt.dataset == 'cikm16':
        n_node = 43097
    elif opt.dataset == 'yoochoose1_64':
        n_node = 17400
    else:
        n_node = 309

    model = GraphModel(opt, n_node=n_node).to(device)

    multigraph_parameters = list(map(id, model.group_graph.parameters()))
    srgnn_parameters = (p for p in model.parameters() if id(p) not in multigraph_parameters)
    parameters = [{"params": model.group_graph.parameters(), "lr": 0.001}, {"params": srgnn_parameters}]

    # best 0.1
    lambda1 = lambda epoch: 0.1 ** (epoch // 3)
    lambda2 = lambda epoch: 0.1 ** (epoch // 3)

    optimizer = torch.optim.Adam(parameters, lr=opt.lr, weight_decay=opt.l2)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])

    if not opt.predict:
        best_result20 = [0, 0]
        best_epoch20 = [0, 0]

        best_result10 = [0, 0]
        best_epoch10 = [0, 0]

        best_result5 = [0, 0]
        best_epoch5 = [0, 0]
        for epoch in range(opt.epoch):
            scheduler.step(epoch)
            print("Epoch ", epoch)
            forward(model, train_loader, device, writer, epoch, top_k=opt.top_k, optimizer=optimizer, train_flag=True)
            with torch.no_grad():
                mrr20, hit20, mrr10, hit10, mrr5, hit5 = forward(model, test_loader, device, writer, epoch, top_k=opt.top_k, train_flag=False)

            if hit20 >= best_result20[0]:
                best_result20[0] = hit20
                best_epoch20[0] = epoch
                # torch.save(model.state_dict(), log_dir+'/best_recall_params.pkl')
            if mrr20 >= best_result20[1]:
                best_result20[1] = mrr20
                best_epoch20[1] = epoch

            if hit10 >= best_result10[0]:
                best_result10[0] = hit10
                best_epoch10[0] = epoch
                # torch.save(model.state_dict(), log_dir+'/best_recall_params.pkl')
            if mrr10 >= best_result10[1]:
                best_result10[1] = mrr10
                best_epoch10[1] = epoch
                # torch.save(model.state_dict(), log_dir+'/best_mrr_params.pkl')

            if hit5 >= best_result5[0]:
                best_result5[0] = hit5
                best_epoch5[0] = epoch
                # torch.save(model.state_dict(), log_dir+'/best_recall_params.pkl')
            if mrr5 >= best_result5[1]:
                best_result5[1] = mrr5
                best_epoch5[1] = epoch

            print('Best Result:')
            print('\tMrr@%d:\t%.4f\tEpoch:\t%d' % (20, best_result20[1], best_epoch20[1]))
            print('\tRecall@%d:\t%.4f\tEpoch:\t%d\n' % (20, best_result20[0], best_epoch20[0]))
            print('\tMrr@%d:\t%.4f\tEpoch:\t%d' % (opt.top_k, best_result10[1], best_epoch10[1]))
            print('\tRecall@%d:\t%.4f\tEpoch:\t%d\n' % (opt.top_k, best_result10[0], best_epoch10[0]))
            print('\tMrr@%d:\t%.4f\tEpoch:\t%d' % (5, best_result5[1], best_epoch5[1]))
            print('\tRecall@%d:\t%.4f\tEpoch:\t%d' % (5, best_result5[0], best_epoch5[0]))
            print("-"*20)
        # print_txt(log_dir, opt, best_result, best_epoch, opt.top_k, note, save_config=True)
    else:
        log_dir = 'log/cikm16/2019-08-19 14:27:33'
        model.load_state_dict(torch.load(log_dir+'/best_mrr_params.pkl'))
        mrr, hit = forward(model, test_loader, device, writer, 0, top_k=opt.top_k, train_flag=False)
        best_result = [hit, mrr]
        best_epoch = [0, 0]
        # print_txt(log_dir, opt, best_result, best_epoch, opt.top_k, save_config=False)

if __name__ == '__main__':
    main()
```
