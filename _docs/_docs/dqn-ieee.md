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

<!-- #region id="cpAvJVlFATkd" -->
# DQN RL Model on IEEE 2021 RecSys dataset
<!-- #endregion -->

<!-- #region id="3GofkbEjwQCI" -->
## Setup
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="BaXpEKLdXNPa" executionInfo={"status": "ok", "timestamp": 1636279874196, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="81c0ec32-7610-4cf3-e46f-12f54bdfc85e"
import os
project_name = "ieee21cup-recsys"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', branch)

if not os.path.exists(project_path):
    !cp -r /content/drive/MyDrive/git_credentials/. ~
    !mkdir "{project_path}"
    %cd "{project_path}"
    !git init
    !git remote add origin https://github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout -b "{branch}"
else:
    %cd "{project_path}"
```

```python colab={"base_uri": "https://localhost:8080/"} id="MZvPHRyMXdlS" executionInfo={"status": "ok", "timestamp": 1636279881076, "user_tz": -330, "elapsed": 1014, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4987f64f-cf0e-40d2-dafa-14b6d3a5e0c6"
%cd /content
```

```python id="2eRcpGL6XfDs"
!cd /content/main && git add . && git commit -m 'commit' && git push origin main
```

```python id="DctyNOSdx-7h" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1636277206359, "user_tz": -330, "elapsed": 4288, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="28b72c68-ee59-4a95-b7ae-c1ea87509527"
!pip install -q wget
```

```python id="vrEmNkAAsQlM"
import io
import copy
import sys
import wget
import os
import random
import logging
import pandas as pd
from os import path as osp
import numpy as np
from tqdm.notebook import tqdm
from pathlib import Path
import math
from copy import deepcopy
from collections import OrderedDict

import multiprocessing as mp
import functools
from sklearn.preprocessing import MinMaxScaler
import pdb

from prettytable import PrettyTable

import bz2
import pickle
import _pickle as cPickle

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
%matplotlib inline
```

```python id="M4swQxyAsQnj"
class Args:

    # Paths
    datapath_bronze = '/content/main/data/bronze'
    datapath_silver = '/content/main/data/silver/T304746'
    datapath_gold = '/content/main/data/gold/T304746'

    filename_trainset = 'train.csv'
    filename_iteminfo = 'item_info.csv'
    filename_track1_testset = 'track1_testset.csv'
    filename_track2_testset = 'track2_testset.csv'

    data_sep = ' '


args = Args()
```

```python id="wIDRSKqOtEdb"
logging.basicConfig(stream=sys.stdout,
                    level = logging.INFO,
                    format='%(asctime)s [%(levelname)s] : %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

logger = logging.getLogger('IEEE21 Logger')
```

<!-- #region id="N1bmqnvQv27E" -->
## Utilities
<!-- #endregion -->

```python id="tH7lmOJbAOIf"
def save_pickle(data, title):
 with bz2.BZ2File(title, 'w') as f: 
    cPickle.dump(data, f)

def load_pickle(path):
    data = bz2.BZ2File(path, 'rb')
    data = cPickle.load(data)
    return data
```

```python id="AGTVUdmtwWgZ"
def download_dataset():
    # create bronze folder if not exist
    Path(args.datapath_bronze).mkdir(parents=True, exist_ok=True)
    # also creating silver and gold folder for later use
    Path(args.datapath_silver).mkdir(parents=True, exist_ok=True)
    Path(args.datapath_gold).mkdir(parents=True, exist_ok=True)
    # for each of the file, download if not exist
    datasets = ['train.parquet.snappy', 'item_info.parquet.snappy',
                'track1_testset.parquet.snappy', 'track2_testset.parquet.snappy']
    for filename in datasets:
        file_savepath = osp.join(args.datapath_bronze,filename)
        if not osp.exists(file_savepath):
            logger.info('Downloading {}'.format(filename))
            wget.download(url='https://github.com/sparsh-ai/ieee21cup-recsys/raw/main/data/bronze/{}'.format(filename),
                          out=file_savepath)
        else:
            logger.info('{} file already exists, skipping!'.format(filename))
```

```python id="vk93jRMwtEWP"
def parquet_to_csv(path):
    savepath = osp.join(str(Path(path).parent),str(Path(path).name).split('.')[0]+'.csv')
    pd.read_parquet(path).to_csv(savepath, index=False, sep=args.data_sep)
```

```python id="_F4vRpFCzYsf"
def convert_dataset():
    # for each of the file, convert into csv, if csv not exist
    datasets = ['train.parquet.snappy', 'item_info.parquet.snappy',
                'track1_testset.parquet.snappy', 'track2_testset.parquet.snappy']
    datasets = {x:str(Path(x).name).split('.')[0]+'.csv' for x in datasets}
    for sfilename, tfilename in datasets.items():
        file_loadpath = osp.join(args.datapath_bronze,sfilename)
        file_savepath = osp.join(args.datapath_bronze,tfilename)
        if not osp.exists(file_savepath):
            logger.info('Converting {} to {}'.format(sfilename, tfilename))
            parquet_to_csv(file_loadpath)
        else:
            logger.info('{} file already exists, skipping!'.format(tfilename))
```

```python id="O_uT85xbufVi"
def normalize(array, axis=0):
    _min = array.min(axis=axis, keepdims=True)
    _max = array.max(axis=axis, keepdims=True)
    factor = _max - _min
    return (array - _min) / np.where(factor != 0, factor, 1)


def parse_click_history(history_list):
    clicks = list(map(lambda user_click: list(map(lambda item: item.split(':')[0],
                                                  user_click.split(','))),
                      history_list))
    _max_len = max(len(items) for items in clicks)
    clicks = [items + [0] * (_max_len - len(items)) for items in clicks]
    clicks = torch.tensor(np.array(clicks, dtype=np.long)) - 1
    return clicks


def parse_user_protrait(protrait_list):
    return torch.tensor(normalize(np.array(list(map(lambda x: x.split(','),
                                                    protrait_list)),
                                           dtype=np.float32)))
```

```python id="d2RIZ1rnugVW"
def process_item():
    readfilepath = osp.join(args.datapath_bronze,args.filename_iteminfo)
    outfilepath = osp.join(args.datapath_silver,'items_info.pt')

    if not osp.exists(outfilepath):
        logger.info('processing items ...')
        item_info = pd.read_csv(readfilepath, sep=args.data_sep)
        item2id = np.array(item_info['item_id']) - 1
        item2loc = torch.tensor(np.array(item_info['location'], dtype=np.float32)[item2id])
        item2price = torch.tensor(normalize(np.array(item_info['price'], dtype=np.float32)[item2id]) * 10, dtype=torch.float32)
        item2feature = torch.tensor(normalize(np.array(list(map(lambda x: x.split(','),
                                                item_info['item_vec'])),
                                        dtype=np.float32)[item2id]))
        item2info = torch.cat([item2feature, item2price[:, None], item2loc[:, None]], dim=-1)
        torch.save([item2info, item2price, item2loc], outfilepath)
        logger.info('processed data saved at {}'.format(outfilepath))
    else:
        logger.info('{} already exists, skipping ...'.format(outfilepath))
```

```python id="ngXkd__hvcRN"
def process_data(readfilepath, outfilepath):
    if not osp.exists(outfilepath):
        logger.info('processing data ...')
        logger.info('loading raw file {} ...'.format(readfilepath))
        dataset = pd.read_csv(readfilepath, sep=args.data_sep)
        click_items = parse_click_history(dataset['user_click_history'])
        user_protrait = parse_user_protrait(dataset['user_protrait'])
        exposed_items = None
        if 'exposed_items' in dataset.columns:
            exposed_items = torch.tensor(np.array(list(map(lambda x: x.split(','),
                                                        dataset['exposed_items'])),
                                                dtype=np.long) - 1)
        torch.save([user_protrait, click_items, exposed_items], outfilepath)
        logger.info('processed data saved at {}'.format(outfilepath))
    else:
        logger.info('{} already exists, skipping ...'.format(outfilepath))
```

```python id="i7G5jruLvGDH"
def process_data_wrapper():
    ds = {
        args.filename_trainset:'train.pt',
        args.filename_track1_testset:'dev.pt',
        args.filename_track2_testset:'test.pt',
    }
    process_item()
    for k,v in ds.items():
        readfilepath = osp.join(args.datapath_bronze,k)
        outfilepath = osp.join(args.datapath_silver,v)
        process_data(readfilepath, outfilepath)
```

```python id="KB8J9nl1FzVm"
class Dataset:
    def __init__(self, filename, batch_size=1024):
        self.user_protrait, self.click_items, self.exposed_items \
                = torch.load(filename)
        self.click_mask = self.click_items != -1
        self.click_items[self.click_items == -1] = 0

        self.all_indexs = list(range(len(self.user_protrait)))
        self.cur_index = 0
        self.bz = batch_size

    def reset(self):
        random.shuffle(self.all_indexs)
        self.cur_index = 0

    def __len__(self):
        return len(self.all_indexs) // self.bz + int(bool(len(self.all_indexs) % self.bz))

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur_index >= len(self.all_indexs):
            raise StopIteration
    
        i = self.all_indexs[self.cur_index:self.cur_index + self.bz]
        user, click_items, click_mask = \
                self.user_protrait[i], self.click_items[i], self.click_mask[i]
        exposed_items = self.exposed_items[i] if self.exposed_items is not None else None

        self.cur_index += self.bz

        return user, click_items, click_mask, exposed_items if exposed_items is not None else None
```

```python id="Y1J1rPHpGBsm"
class Env:
    def __init__(self, value, K=3):
        self.K = K - 1
        self.value = np.asarray(value)

    def done(self, obs):
        return obs[3] is not None and obs[3].size(1) == self.K

    def __recall(self, s, t):
        return sum(i in t for i in s) / len(t)

    def __reward(self, s, t):
        return self.__recall(s, t) * self.value[s].sum()

    def new_obs(self, batch_obs, batch_actions):
        batch_users, batch_click_items, batch_click_mask, \
                batch_exposed_items, batch_exposed_mask = batch_obs
        
        batch_new_exposed_items = torch.cat(
            [batch_exposed_items, batch_actions.unsqueeze(1)], dim=1
        ) if batch_exposed_items is not None else batch_actions.unsqueeze(1)
        
        _add_mask = torch.tensor([[True]]).expand(batch_users.size(0), -1)
        batch_new_exposed_mask = torch.cat(
            [batch_exposed_mask, _add_mask], dim=1
        ) if batch_exposed_mask is not None else _add_mask
        
        batch_new_obs = (batch_users, batch_click_items, batch_click_mask,
                         batch_new_exposed_items, batch_new_exposed_mask)
        return batch_new_obs

    def step(self, batch_obs, batch_actions, batch_target_bundles, time):
        batch_rews = torch.tensor([self.__reward(action, bundle) \
                                   for action, bundle in zip(batch_actions, batch_target_bundles[:, time])],
                                  dtype=torch.float32)
        batch_users, batch_click_items, batch_click_mask, \
                batch_exposed_items, batch_exposed_mask = batch_obs
        done = batch_exposed_mask is not None and batch_exposed_mask[0].sum() == self.K
        if done:
            batch_new_obs = [None] * batch_users.size(0)
        else:
            batch_new_obs = self.new_obs(batch_obs, batch_actions)
        
        return batch_new_obs, batch_rews, torch.tensor([done] * batch_actions.size(0))
```

```python id="fKXTFUjOGJ8T"
def table_format(data, field_names=None, title=None):
    tb = PrettyTable()
    if field_names is not None:
        tb.field_names = field_names
        for i, name in enumerate(field_names):
            tb.align[name] = 'r' if i else 'l'
    if title is not None:
        tb.title = title
    tb.add_rows(data)
    return tb.get_string()


def recall(batch_pred_bundles, batch_target_bundles):
    rec, rec1, rec2, rec3 = [], [], [], []
    for pred_bundle, target_bundle in zip(batch_pred_bundles, batch_target_bundles):
        recs = []
        for bundle_a, bundle_b in zip(pred_bundle, target_bundle):
            recs.append(len(set(bundle_a.tolist()) & set(bundle_b.tolist())) / len(bundle_b))
        rec1.append(recs[0])
        rec2.append(recs[1])
        rec3.append(recs[2])
        rec.append((rec1[-1] + rec2[-1] + rec3[-1]) / 3)
    return np.mean(rec), np.mean(rec1), np.mean(rec2), np.mean(rec3)


def nan2num(tensor, num=0):
    tensor[tensor != tensor] = num


def inf2num(tensor, num=0):
    tensor[tensor == float('-inf')] = num
    tensor[tensor == float('inf')] = num


def tensor2device(tensors, device):
    return [tensor.to(device) if tensor is not None else None \
            for tensor in tensors]
```

```python id="TMZwihM7Gdzg"
def _calc_q_value(obs, net, act_mask, device):
    batch_users, batch_encoder_item_ids, encoder_mask, \
            batch_decoder_item_ids, decoder_mask = tensor2device(obs, device)
    return net(batch_users,
               batch_encoder_item_ids,
               encoder_mask,
               batch_decoder_item_ids,
               decoder_mask,
               act_mask.unsqueeze(0).expand(batch_users.size(0), -1) if act_mask is not None else None)


def build_train(q_net,
                optimizer,
                grad_norm_clipping,
                act_mask,
                gamma=0.99,
                is_gpu=False):
    device = torch.device('cuda') if is_gpu else torch.device('cpu')
    q_net.to(device)

    t_net = deepcopy(q_net)
    t_net.eval()
    t_net.to(device)
    optim = optimizer(q_net.parameters())

    act_mask = act_mask.to(device)

    if is_gpu and torch.cuda.device_count() > 1:
        q_net = torch.nn.DataParallel(q_net)
        t_net = torch.nn.DataParallel(t_net)

    def save_model(filename,
                   epoch,
                   episode_rewards,
                   saved_mean_reward):
        torch.save({
            'epoch': epoch,
            'episode_rewards': episode_rewards,
            'saved_mean_reward': saved_mean_reward,
            'model': q_net.state_dict(),
            'optim': optim.state_dict()
        }, filename)

    def load_model(filename):
        checkpoint = torch.load(filename,
                                map_location=torch.device('cpu'))
        #q_net.load_state_dict(checkpoint['model'])
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model'].items():
            if k.find('module.') != -1:
                k = k[7:]
            new_state_dict[k] = v
        q_net.load_state_dict(new_state_dict)
        optim.load_state_dict(checkpoint['optim'])
        return checkpoint['epoch'], checkpoint['episode_rewards'], checkpoint['saved_mean_reward']

    def train(obs,
              act,
              rew,
              next_obs,
              isweights,
              done_mask,
              topk=3):
        act, rew, isweights = act.to(device), rew.to(device), isweights.to(device)
        # q value at t+1 in double q
        with torch.no_grad():
            q_net.eval()
            next_q_val = _calc_q_value(next_obs, q_net, act_mask, device).detach()
            q_net.train()
            
            _next_mask = next_obs[4].to(device).sum(dim=1, keepdim=True) + 1 == act_mask.unsqueeze(0)
            assert next_q_val.size() == _next_mask.size()

            next_q_val[_next_mask == False] = float('-inf')

            next_action_max = next_q_val.argsort(dim=1, descending=True)[:, :topk]
            next_q_val_max = _calc_q_value(next_obs, t_net, act_mask, device) \
                                   .detach() \
                                   .gather(dim=1, index=next_action_max) \
                                   .sum(dim=1)

            _next_q_val_max = next_q_val_max.new_zeros(done_mask.size())
            _next_q_val_max[done_mask == False] = next_q_val_max
        # q value at t
        q_val = _calc_q_value(obs, q_net, act_mask, device)
        q_val_t = q_val.gather(dim=1, index=act.to(device)).sum(dim=1)
        assert q_val_t.size() == _next_q_val_max.size()
        #print('done')
        # Huber Loss
        loss = F.smooth_l1_loss(q_val_t,
                                rew + gamma * _next_q_val_max,
                                reduction='none')
        assert loss.size() == isweights.size()
        #wloss = (loss * isweights).mean()
        wloss = loss.mean()
        wloss.backward()
        torch.nn.utils.clip_grad_norm_(q_net.parameters(), grad_norm_clipping)
        optim.step()
        q_net.zero_grad()

        return wloss.detach().data.item(), (loss.detach().mean().data.item()), loss.cpu().detach().abs()

    def act(obs,
            eps_greedy,
            topk=3,
            is_greedy=False):
        return build_act(obs, act_mask, q_net, eps_greedy, topk,
                         is_greedy=is_greedy, device=device)

    def update_target():
        for target_param, local_param in zip(t_net.parameters(), q_net.parameters()):
            target_param.data.copy_(local_param.data)

    return q_net, act, train, update_target, save_model, load_model


def build_act(obs,
              act_mask,
              net,
              eps_greedy,
              topk=3,
              is_greedy=False,
              device=None):
    devcie = torch.device('cpu') if device is None else device
    act_mask = act_mask.to(device)
    def _epsilon_greedy(size):
        return torch.rand(size).to(device) < eps_greedy
    def _gen_act_mask():
        #if obs[3] is not None:
        if obs[4] is not None:
            #length = torch.tensor([len(o) + 1 if o is not None else 1 for o in obs[3]],
            #                      dtype=torch.float).view(-1, 1).to(device)
            length = obs[4].to(device).sum(dim=1, keepdim=True) + 1
        else:
            length = act_mask.new_ones((1,)).view(-1, 1)
        return act_mask.unsqueeze(0) == length
    net.eval()
    with torch.no_grad():
        q_val = _calc_q_value(obs, net, act_mask, device).detach()
        _act_mask = _gen_act_mask()
        if q_val.size() != _act_mask.size():
            assert _act_mask.size(0) == 1
            _act_mask = _act_mask.expand(q_val.size(0), -1)
        q_val[_act_mask == False] = float('-inf')
        _deterministic_acts = q_val.argsort(dim=1, descending=True)[:, :topk]
        if not is_greedy:
            _stochastic_acts = _deterministic_acts.new_empty(_deterministic_acts.size())
            chose_random = _epsilon_greedy(_stochastic_acts.size(0))
            _tmp = torch.arange(0, _act_mask.size(1), dtype=_deterministic_acts.dtype)
            for i in range(_act_mask.size(0)):
                _available_acts = _act_mask[i].nonzero().view(-1)
                _stochastic_acts[i] = _available_acts[torch.randperm(_available_acts.size(0))[:topk]]
            #if chose_random.sum() != len(chose_random):
            #    pdb.set_trace()
            _acts = torch.where(chose_random.unsqueeze(1).expand(-1, _stochastic_acts.size(1)),
                                _stochastic_acts,
                                _deterministic_acts)
            # TODO 去重       
        else:
            _acts = _deterministic_acts
            eps_greedy = 0.
    net.train()
    
    return _acts, eps_greedy
```

<!-- #region id="koFQxtgos6gE" -->
## Jobs
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="2y8mdDjds6dr" executionInfo={"status": "ok", "timestamp": 1636280333092, "user_tz": -330, "elapsed": 578, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d31698d0-83b0-4d1a-dae7-0697787c5c5f"
logger.info('JOB START: DOWNLOAD_RAW_DATASET')
download_dataset()
logger.info('JOB END: DOWNLOAD_RAW_DATASET')
```

```python colab={"base_uri": "https://localhost:8080/"} id="3ig3tPpB2Fx-" executionInfo={"status": "ok", "timestamp": 1636277232354, "user_tz": -330, "elapsed": 14778, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1ba6042e-b8f6-4084-a1c3-69b7e0cefeaa"
logger.info('JOB START: DATASET_CONVERSION_PARQUET_TO_CSV')
convert_dataset()
logger.info('JOB END: DATASET_CONVERSION_PARQUET_TO_CSV')
```

```python colab={"base_uri": "https://localhost:8080/"} id="kj4VvaPE7s4P" executionInfo={"status": "ok", "timestamp": 1636283043925, "user_tz": -330, "elapsed": 659, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3f057f8b-db97-4196-90f5-77e39351c32a"
logger.info('JOB START: DATASET_PREPROCESSING')
process_data_wrapper()
logger.info('JOB END: DATASET_PREPROCESSING')
```

```python id="_8QpqVrsEcr5"

```
