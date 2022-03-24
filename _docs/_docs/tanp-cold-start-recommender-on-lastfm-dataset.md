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

<!-- #region id="le6ItuZ9aCvt" -->
# TaNP Cold-start Recommender on LastFM dataset
<!-- #endregion -->

<!-- #region id="x47nzfcCZ8cJ" -->
Recent studies seek to address this challenge from the perspective of meta learning, and most of them follow a manner of parameter initialization, where the model parameters can be learned by a few steps of gradient updates. While these gradient-based meta-learning models achieve promising performances to some extent, a fundamental problem of them is how to adapt the global knowledge learned from previous tasks for the recommendations of cold-start users more effectively.

TaNP directly maps the observed interactions of each user to a predictive distribution, sidestepping some training issues in gradient-based meta-learning models. More importantly, to balance the trade-off between model capacity and adaptation reliability, TaNP uses a novel task-adaptive mechanism. It enables this model to learn the relevance of different tasks and customize the global knowledge to the task-related decoder parameters for estimating user preferences.
<!-- #endregion -->

<!-- #region id="cRWoUWlraKKb" -->
Inspired by the significant improvements of meta learning, the pioneering work of [Vartak et. al.](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46346.pdf) provides a meta-learning strategy to solve cold-start problems. It uses a task-dependent way to generate the varying biases of decision layers for different tasks, but it is prone to underfitting and is not flexible enough to handle various recommendation scenarios. [MeLU](https://arxiv.org/abs/1908.00413) adopts the framework of MAML. Specifically, it divides the model parameters into two groups, i.e., the personalized parameter and the embedding parameter. The personalized parameter is characterized as a fully-connected DNN to estimate user preferences. The embedding parameter is referred as the embeddings of users and items learned from side-information. An inner-outer loop is used to update these two groups of parameters. In the inner loop, the personalized parameter is locally updated via the prediction loss of support set in current task. In the outer loop, these parameters are globally updated according to the prediction loss of query sets in multiple tasks. Through the fashion of local-global update, MeLU can provide a shared initialization for different tasks. The later work [MetaCS](https://www.semanticscholar.org/paper/Meta-Learning-for-User-Cold-Start-Recommendation-Bharadhwaj/f3135b553f592dc42d4202c90739c99486103fc3) is much similar to MeLU, and the main difference is that the local-global update involves all parameters from input embedding to model prediction. To generalize well for different tasks, [MetaHIN](https://www.kdd.org/kdd2020/accepted-papers/view/meta-learning-on-heterogeneous-information-networks-for-cold-start-recommen) and [MAMO](https://arxiv.org/abs/2007.03183) propose different task-specific adaptation strategies. In particular, MetaHIN incorporates heterogeneous information networks (HINs) into MAML to capture rich semantics of meta-paths. MAMO introduces two memory matrices based on user profiles: a feature-specific memory that provides a specific bias term for the shared parameter initialization; a task-specific memory that guides the model for predictions. However, these two gradient-based meta-learning models may still suffer from potential training issues in MAML, and the model-level innovations of them are closely related with side-information, which limits their application scenarios.
<!-- #endregion -->

<!-- #region id="2LroOGYgZ9Zg" -->
## Background

### Meta Learning in Recommenders

Inspired by the huge progress on few-shot learning and meta learning, there emerge some promising works on solving cold-start problems from the perspective of meta learning, where making recommendations for one user is regarded as a single task.

In the training phase, they try to derive the global knowledge across different tasks as a strong generalization prior. When a cold-start user comes in the test phase, the personalized recommendation for her/him can be predicted with only a few interacted items are available, but does so by using the global knowledge already learned.

Most meta-learning recommenders are built upon the well-known framework of model-agnostic meta learning (MAML), aiming to learn a parameter initialization where a few steps of gradient updates will lead to good performances on the new tasks. A typical assumption here is the recommendations of different users are highly relevant. However, this assumption does not necessarily hold in actual scenarios. When the users exhibit different purchase intentions, the task relevance among them is actually very weak, which makes it problematic to find a shared parameter initialization optimal for all users. As shown in the image below:
<!-- #endregion -->

<!-- #region id="MHKAnNDj0uDM" -->
<p><center><img src='_images/T722684_1.png'></center></p>
<!-- #endregion -->

<!-- #region id="Kh2GBWwXTSsP" -->
## Setup
<!-- #endregion -->

<!-- #region id="ej09KyMmTSo-" -->
### Imports
<!-- #endregion -->

```python id="fRVNWsrtSoad"
import os
import json
import math
import random
import pickle
import codecs
import re
import time
import datetime
from tqdm.notebook import tqdm
from datetime import datetime

from random import randint
from copy import deepcopy
from collections import OrderedDict

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch import optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.optimizer import Optimizer
```

<!-- #region id="ltEYBEhdWwdF" -->
### Data
<!-- #endregion -->

```python id="NP_OQPAbWxcC"
!wget -q --show-progress https://github.com/sparsh-ai/coldstart-recsys/raw/main/data/TaNP/data.zip
!unzip data.zip
```

<!-- #region id="UFr79XRqVTlz" -->
### Params
<!-- #endregion -->

```python id="19RU9DZyVXmx"
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data/lastfm_20')
parser.add_argument('--model_save_dir', type=str, default='save_model_dir')
parser.add_argument('--id', type=str, default='1', help='used for save hyper-parameters.')

parser.add_argument('--first_embedding_dim', type=int, default=32, help='Embedding dimension for item and user.')
parser.add_argument('--second_embedding_dim', type=int, default=16, help='Embedding dimension for item and user.')

parser.add_argument('--z1_dim', type=int, default=32, help='The dimension of z1 in latent path.')
parser.add_argument('--z2_dim', type=int, default=32, help='The dimension of z2 in latent path.')
parser.add_argument('--z_dim', type=int, default=32, help='The dimension of z in latent path.')

parser.add_argument('--enc_h1_dim', type=int, default=64, help='The hidden first dimension of encoder.')
parser.add_argument('--enc_h2_dim', type=int, default=64, help='The hidden second dimension of encoder.')

parser.add_argument('--taskenc_h1_dim', type=int, default=128, help='The hidden first dimension of task encoder.')
parser.add_argument('--taskenc_h2_dim', type=int, default=64, help='The hidden second dimension of task encoder.')
parser.add_argument('--taskenc_final_dim', type=int, default=64, help='The hidden second dimension of task encoder.')

parser.add_argument('--clusters_k', type=int, default=7, help='Cluster numbers of tasks.')
parser.add_argument('--temperature', type=float, default=1.0, help='used for student-t distribution.')
parser.add_argument('--lambda', type=float, default=0.1, help='used to balance the clustering loss and NP loss.')

parser.add_argument('--dec_h1_dim', type=int, default=128, help='The hidden first dimension of encoder.')
parser.add_argument('--dec_h2_dim', type=int, default=128, help='The hidden second dimension of encoder.')
parser.add_argument('--dec_h3_dim', type=int, default=128, help='The hidden third dimension of encoder.')

# # used for movie datasets
# parser.add_argument('--num_gender', type=int, default=2, help='User information.')
# parser.add_argument('--num_age', type=int, default=7, help='User information.')
# parser.add_argument('--num_occupation', type=int, default=21, help='User information.')
# parser.add_argument('--num_zipcode', type=int, default=3402, help='User information.')
# parser.add_argument('--num_rate', type=int, default=6, help='Item information.')
# parser.add_argument('--num_genre', type=int, default=25, help='Item information.')
# parser.add_argument('--num_director', type=int, default=2186, help='Item information.')
# parser.add_argument('--num_actor', type=int, default=8030, help='Item information.')

parser.add_argument('--dropout_rate', type=float, default=0, help='used in encoder and decoder.')
parser.add_argument('--lr', type=float, default=1e-4, help='Applies to SGD and Adagrad.')
parser.add_argument('--optim', type=str, default='adam', help='sgd, adagrad, adam or adamax.')
parser.add_argument('--num_epoch', type=int, default=150)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--train_ratio', type=float, default=0.7, help='Warm user ratio for training.')
parser.add_argument('--valid_ratio', type=float, default=0.1, help='Cold user ratio for validation.')
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--use_cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
parser.add_argument('--support_size', type=int, default=20)
parser.add_argument('--query_size', type=int, default=10)
parser.add_argument('--max_len', type=int, default=200, help='The max length of interactions for each user.')
parser.add_argument('--context_min', type=int, default=20, help='Minimum size of context range.')
args = parser.parse_args(args={})
```

```python id="IP7Eu0gRV4mQ"
def seed_everything(seed=1023):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = args.seed
seed_everything(seed)

if args.cpu:
    args.use_cuda = False
elif args.use_cuda:
    torch.cuda.manual_seed(args.seed)

opt = vars(args)
```

<!-- #region id="vxvdxVEvTLqJ" -->
## Utils
<!-- #endregion -->

```python id="oCQDsZmCTpeI"
#convert userids to userdict key-id(int), val:onehot_vector(tensor)
#element in list is str type.
def to_onehot_dict(list):
    dict={}
    length = len(list)
    for index, element in enumerate(list):
        vector = torch.zeros(1, length).long()
        element = int(element)
        vector[:, element] = 1.0
        dict[element] = vector
    return dict

def load_list(fname):
    list_ = []
    with open(fname, encoding="utf-8") as f:
        for line in f.readlines():
            list_.append(line.strip())
    return list_

# used for merge dictionaries.
def merge_key(dict1, dict2):
    res = {**dict1, **dict2}
    return res

def merge_value(dict1, dict2): # merge and item_cold
    for key, value in dict2.items():
        if key in dict1.keys():
            # if list(set(dict1[key]+value)) the final number of movies-1m is 1000205
            new_value = dict1[key]+value
            dict1[key] = new_value
        else:
            print('Unexpected key.')

def count_values(dict):
    count_val = 0
    for key, value in dict.items():
        count_val += len(value)
    return count_val

def construct_dictionary(user_list, total_dict):
    dict = {}
    for i in range(len(user_list)):
        dict[str(user_list[i])] = total_dict[str(user_list[i])]
    return dict
```

```python id="xrF-BoerTZBw"
### IO
def check_dir(d):
    if not os.path.exists(d):
        print("Directory {} does not exist. Exit.".format(d))
        exit(1)


def check_files(files):
    for f in files:
        if f is not None and not os.path.exists(f):
            print("File {} does not exist. Exit.".format(f))
            exit(1)


def ensure_dir(d, verbose=True):
    if not os.path.exists(d):
        if verbose:
            print("Directory {} do not exist; creating...".format(d))
        os.makedirs(d)


def save_config(config, path, verbose=True):
    with open(path, 'w') as outfile:
        json.dump(config, outfile, indent=2)
    if verbose:
        print("Config saved to file {}".format(path))
    return config


def load_config(path, verbose=True):
    with open(path) as f:
        config = json.load(f)
    if verbose:
        print("Config loaded from file {}".format(path))
    return config


def print_config(config):
    info = "Running with the following configs:\n"
    for k, v in config.items():
        info += "\t{} : {}\n".format(k, str(v))
    print("\n" + info + "\n")
    return


class FileLogger(object):
    """
    A file logger that opens the file periodically and write to it.
    """

    def __init__(self, filename, header=None):
        self.filename = filename
        if os.path.exists(filename):
            # remove the old file
            os.remove(filename)
        if header is not None:
            with open(filename, 'w') as out:
                print(header, file=out)

    def log(self, message):
        with open(self.filename, 'a') as out:
            print(message)
            print(message, file=out)
```

```python id="2N-_QNTRTEAB"
class MyAdagrad(Optimizer):
    """My modification of the Adagrad optimizer that allows to specify an initial
    accumulater value. This mimics the behavior of the default Adagrad implementation
    in Tensorflow. The default PyTorch Adagrad uses 0 for initial acculmulator value.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        init_accu_value (float, optional): initial accumulater value.
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=1e-2, lr_decay=0, init_accu_value=0.1, weight_decay=0):
        defaults = dict(lr=lr, lr_decay=lr_decay, init_accu_value=init_accu_value, \
                weight_decay=weight_decay)
        super(MyAdagrad, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['sum'] = torch.ones(p.data.size()).type_as(p.data) *\
                        init_accu_value

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['sum'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                state['step'] += 1

                if group['weight_decay'] != 0:
                    if p.grad.data.is_sparse:
                        raise RuntimeError("weight_decay option is not compatible with sparse gradients ")
                    grad = grad.add(group['weight_decay'], p.data)

                clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])

                if p.grad.data.is_sparse:
                    grad = grad.coalesce()  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = torch.Size([x for x in grad.size()])

                    def make_sparse(values):
                        constructor = type(p.grad.data)
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor()
                        return constructor(grad_indices, values, size)
                    state['sum'].add_(make_sparse(grad_values.pow(2)))
                    std = state['sum']._sparse_mask(grad)
                    std_values = std._values().sqrt_().add_(1e-10)
                    p.data.add_(-clr, make_sparse(grad_values / std_values))
                else:
                    state['sum'].addcmul_(1, grad, grad)
                    std = state['sum'].sqrt().add_(1e-10)
                    p.data.addcdiv_(-clr, grad, std)

        return loss
```

```python id="654V_X3gTGwu"
### torch specific functions
def get_optimizer(name, parameters, lr, l2=0):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=l2)
    elif name in ['adagrad', 'myadagrad']:
        # use my own adagrad to allow for init accumulator value
        return MyAdagrad(parameters, lr=lr, init_accu_value=0.1, weight_decay=l2)
    elif name == 'adam':
        return torch.optim.Adam(parameters, weight_decay=l2) # use default lr
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, weight_decay=l2) # use default lr
    elif name == 'adadelta':
        return torch.optim.Adadelta(parameters, lr=lr, weight_decay=l2)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))
```

```python id="dVrMNCx6TGuP"
def change_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def flatten_indices(seq_lens, width):
    flat = []
    for i, l in enumerate(seq_lens):
        for j in range(l):
            flat.append(i * width + j)
    return flat


def set_cuda(var, cuda):
    if cuda:
        return var.cuda()
    return var


def keep_partial_grad(grad, topk):
    """
    Keep only the topk rows of grads.
    """
    assert topk < grad.size(0)
    grad.data[topk:].zero_()
    return grad

### model IO
def save(model, optimizer, opt, filename):
    params = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': opt
    }
    try:
        torch.save(params, filename)
    except BaseException:
        print("[ Warning: model saving failed. ]")


def load(model, optimizer, filename):
    try:
        dump = torch.load(filename)
    except BaseException:
        print("[ Fail: model loading failed. ]")
    if model is not None:
        model.load_state_dict(dump['model'])
    if optimizer is not None:
        optimizer.load_state_dict(dump['optimizer'])
    opt = dump['config']
    return model, optimizer, opt


def load_config(filename):
    try:
        dump = torch.load(filename)
    except BaseException:
        print("[ Fail: model loading failed. ]")
    return dump['config']
```

<!-- #region id="hJAQunbsTGrP" -->
## Dataset
<!-- #endregion -->

```python id="o_c54cnQTGok"
class Preprocess(object):
    """
    Preprocess the training, validation and test data.
    Generate the episode-style data.
    """

    def __init__(self, opt):
        self.batch_size = opt["batch_size"]
        self.opt = opt
        # warm data ratio
        self.train_ratio = opt['train_ratio']
        self.valid_ratio = opt['valid_ratio']
        self.test_ratio = 1 - self.train_ratio - self.valid_ratio
        self.dataset_path = opt["data_dir"]
        self.support_size = opt['support_size']
        self.query_size = opt['query_size']
        self.max_len = opt['max_len']
        # save one-hot dimension length
        uf_dim, if_dim = self.preprocess(self.dataset_path)
        self.uf_dim = uf_dim
        self.if_dim = if_dim

    def preprocess(self, dataset_path):
        """ Preprocess the data and convert to ids. """
        #Create training-validation-test datasets
        print('Create training, validation and test data from scratch!')
        with open('./{}/interaction_dict_x.json'.format(dataset_path), 'r', encoding='utf-8') as f:
            inter_dict_x = json.loads(f.read())
        with open('./{}/interaction_dict_y.json'.format(dataset_path), 'r', encoding='utf-8') as f:
            inter_dict_y = json.loads(f.read())
        print('The size of total interactions is %d.' % (count_values(inter_dict_x)))  # 42346
        assert count_values(inter_dict_x) == count_values(inter_dict_y)

        with open('./{}/user_list.json'.format(dataset_path), 'r', encoding='utf-8') as f:
            userids = json.loads(f.read())

        with open('./{}/item_list.json'.format(dataset_path), 'r', encoding='utf-8') as f:
            itemids = json.loads(f.read())

        #userids = list(inter_dict_x.keys())
        random.shuffle(userids)
        warm_user_size = int(len(userids) * self.train_ratio)
        valid_user_size = int(len(userids) * self.valid_ratio)
        warm_users = userids[:warm_user_size]
        valid_users = userids[warm_user_size:warm_user_size+valid_user_size]
        cold_users = userids[warm_user_size+valid_user_size:]
        assert len(userids) == len(warm_users)+len(valid_users)+len(cold_users)


            # Construct the training data dict
        training_dict_x = construct_dictionary(warm_users, inter_dict_x)
        training_dict_y = construct_dictionary(warm_users, inter_dict_y)

            #Avoid the new items shown in test data in the case of cold user.
        item_set = set()
        for i in training_dict_x.values():
            i = set(i)
            item_set = item_set.union(i)

        # Construct one-hot dictionary
        user_dict = to_onehot_dict(userids)
        # only items contained in all data are encoded.
        item_dict = to_onehot_dict(itemids)

        # This part of data is not used, so we do not process it temporally.
        valid_dict_x = construct_dictionary(valid_users, inter_dict_x)
        valid_dict_y = construct_dictionary(valid_users, inter_dict_y)
        assert count_values(valid_dict_x) == count_values(valid_dict_y)

        test_dict_x = construct_dictionary(cold_users, inter_dict_x)
        test_dict_y = construct_dictionary(cold_users, inter_dict_y)
        assert count_values(test_dict_x) == count_values(test_dict_y)

        print('Before delete new items in test data, test data has %d interactions.' % (count_values(test_dict_x)))

        #Delete the new items in test data.
        unseen_count = 0
        for key, value in test_dict_x.items():
            assert len(value) == len(test_dict_y[key])
            unseen_item_index = [index for index, i in enumerate(value) if i not in item_set]
            unseen_count+=len(unseen_item_index)
            if len(unseen_item_index) == 0:
                continue
            else:
                new_value_x = [element for index, element in enumerate(value) if index not in unseen_item_index]
                new_value_y = [test_dict_y[key][index] for index, element in enumerate(value) if index not in unseen_item_index]
                test_dict_x[key] = new_value_x
                test_dict_y[key] = new_value_y
        print('After delete new items in test data, test data has %d interactions.' % (count_values(test_dict_x)))
        assert count_values(test_dict_x) == count_values(test_dict_y)
        print('The number of total unseen interactions is %d.' % (unseen_count))

        pickle.dump(training_dict_x, open("{}/training_dict_x_{:2f}.pkl".format(dataset_path, self.train_ratio), "wb"))
        pickle.dump(training_dict_y, open("{}/training_dict_y_{:2f}.pkl".format(dataset_path, self.train_ratio), "wb"))
        pickle.dump(valid_dict_x, open("{}/valid_dict_x_{:2f}.pkl".format(dataset_path, self.valid_ratio), "wb"))
        pickle.dump(valid_dict_y, open("{}/valid_dict_y_{:2f}.pkl".format(dataset_path, self.valid_ratio), "wb"))
        pickle.dump(test_dict_x, open("{}/test_dict_x_{:2f}.pkl".format(dataset_path, self.test_ratio), "wb"))
        pickle.dump(test_dict_y, open("{}/test_dict_y_{:2f}.pkl".format(dataset_path, self.test_ratio), "wb"))

        def generate_episodes(dict_x, dict_y, category, support_size, query_size, max_len, dir="log"):
            idx = 0
            if not os.path.exists("{}/{}/{}".format(dataset_path, category, dir)):
                os.makedirs("{}/{}/{}".format(dataset_path, category, dir))
                os.makedirs("{}/{}/{}".format(dataset_path, category, "evidence"))
                for _, user_id in enumerate(dict_x.keys()):
                    u_id = int(user_id)
                    seen_music_len = len(dict_x[str(u_id)])
                    indices = list(range(seen_music_len))
                    # filter some users with their interactions, i.e., tasks
                    if seen_music_len < (support_size + query_size) or seen_music_len > max_len:
                        continue
                    random.shuffle(indices)
                    tmp_x = np.array(dict_x[str(u_id)])
                    tmp_y = np.array(dict_y[str(u_id)])

                    support_x_app = None
                    for m_id in tmp_x[indices[:support_size]]:
                        m_id = int(m_id)
                        tmp_x_converted = torch.cat((item_dict[m_id], user_dict[u_id]), 1)
                        try:
                            support_x_app = torch.cat((support_x_app, tmp_x_converted), 0)
                        except:
                            support_x_app = tmp_x_converted

                    query_x_app = None
                    for m_id in tmp_x[indices[support_size:]]:
                        m_id = int(m_id)
                        u_id = int(user_id)
                        tmp_x_converted = torch.cat((item_dict[m_id], user_dict[u_id]), 1)
                        try:
                            query_x_app = torch.cat((query_x_app, tmp_x_converted), 0)
                        except:
                            query_x_app = tmp_x_converted

                    support_y_app = torch.FloatTensor(tmp_y[indices[:support_size]])
                    query_y_app = torch.FloatTensor(tmp_y[indices[support_size:]])

                    pickle.dump(support_x_app, open("{}/{}/{}/supp_x_{}.pkl".format(dataset_path, category, dir, idx), "wb"))
                    pickle.dump(support_y_app, open("{}/{}/{}/supp_y_{}.pkl".format(dataset_path, category, dir, idx), "wb"))
                    pickle.dump(query_x_app, open("{}/{}/{}/query_x_{}.pkl".format(dataset_path, category, dir, idx), "wb"))
                    pickle.dump(query_y_app, open("{}/{}/{}/query_y_{}.pkl".format(dataset_path, category, dir, idx), "wb"))
                    # used for evidence candidate selection
                    with open("{}/{}/{}/supp_x_{}_u_m_ids.txt".format(dataset_path, category, "evidence", idx), "w") as f:
                        for m_id in tmp_x[indices[:support_size]]:
                            f.write("{}\t{}\n".format(u_id, m_id))
                    with open("{}/{}/{}/query_x_{}_u_m_ids.txt".format(dataset_path, category, "evidence", idx), "w") as f:
                        for m_id in tmp_x[indices[support_size:]]:
                            f.write("{}\t{}\n".format(u_id, m_id))
                    idx+=1

        print("Generate eposide data for training.")
        generate_episodes(training_dict_x, training_dict_y, "training", self.support_size, self.query_size, self.max_len)
        print("Generate eposide data for validation.")
        generate_episodes(valid_dict_x, valid_dict_y, "validation", self.support_size, self.query_size, self.max_len)
        print("Generate eposide data for testing.")
        generate_episodes(test_dict_x, test_dict_y, "testing", self.support_size, self.query_size, self.max_len)

        return len(userids), len(itemids)
```

<!-- #region id="keJ-Uj4jUHO9" -->
## Model Definition
<!-- #endregion -->

<!-- #region id="26Yb8kiSadxb" -->
TaNP includes the encoder $ℎ_\theta$, the customization module (task identity network $𝑚_\phi$ and global pool 𝑨) and the adaptive decoder $𝑔_{\omega𝑖}$. Both of $𝑆_𝑖$ and $\tau_𝑖$ are encoded by $ℎ_\theta$ to generate the variational prior and posterior, respectively. The final task embedding $𝒐_𝑖$ learned from the customized module is used to modulate the model parameters of $𝑔_{\omega𝑖} \cdot 𝒛_𝑖$ sampled from $𝑞(𝒛_𝑖|\tau_𝑖)$ is concatenated with $𝒙_{𝑖,𝑗}$ to predict $\hat{𝑦}_{𝑖,𝑗}$ via $𝑔_{\omega𝑖}$.
<!-- #endregion -->

<!-- #region id="uYP-fzbwUVvL" -->
### Embeddings
<!-- #endregion -->

```python id="93NCqJnOUIjZ"
class Item(torch.nn.Module):
    def __init__(self, config):
        super(Item, self).__init__()
        self.feature_dim = config['if_dim']
        self.first_embedding_dim = config['first_embedding_dim']
        self.second_embedding_dim = config['second_embedding_dim']

        self.first_embedding_layer = torch.nn.Linear(
            in_features=self.feature_dim,
            out_features=self.first_embedding_dim,
            bias=True
        )

        self.second_embedding_layer = torch.nn.Linear(
            in_features=self.first_embedding_dim,
            out_features=self.second_embedding_dim,
            bias=True
        )

    def forward(self, x, vars=None):
        first_hidden = self.first_embedding_layer(x)
        first_hidden = F.relu(first_hidden)
        sec_hidden = self.second_embedding_layer(first_hidden)
        return F.relu(sec_hidden)

class Movie_item(torch.nn.Module):
    def __init__(self, config):
        super(Moive_item, self).__init__()
        self.num_rate = config['num_rate']
        self.num_genre = config['num_genre']
        self.num_director = config['num_director']
        self.num_actor = config['num_actor']
        self.embedding_dim = config['embedding_dim']

        self.embedding_rate = torch.nn.Embedding(
            num_embeddings=self.num_rate, 
            embedding_dim=self.embedding_dim
        )
        
        self.embedding_genre = torch.nn.Linear(
            in_features=self.num_genre,
            out_features=self.embedding_dim,
            bias=False
        )
        
        self.embedding_director = torch.nn.Linear(
            in_features=self.num_director,
            out_features=self.embedding_dim,
            bias=False
        )
        
        self.embedding_actor = torch.nn.Linear(
            in_features=self.num_actor,
            out_features=self.embedding_dim,
            bias=False
        )

    def forward(self, rate_idx, genre_idx, director_idx, actors_idx, vars=None):
        rate_emb = self.embedding_rate(rate_idx)
        genre_emb = self.embedding_genre(genre_idx.float()) / torch.sum(genre_idx.float(), 1).view(-1, 1)
        director_emb = self.embedding_director(director_idx.float()) / torch.sum(director_idx.float(), 1).view(-1, 1)
        actors_emb = self.embedding_actor(actors_idx.float()) / torch.sum(actors_idx.float(), 1).view(-1, 1)
        return torch.cat((rate_emb, genre_emb, director_emb, actors_emb), 1)
```

```python id="Ox55_5OgUR6q"
class User(torch.nn.Module):
    def __init__(self, config):
        super(User, self).__init__()
        self.feature_dim = config['uf_dim']
        self.first_embedding_dim = config['first_embedding_dim']
        self.second_embedding_dim = config['second_embedding_dim']

        self.first_embedding_layer = torch.nn.Linear(
            in_features=self.feature_dim,
            out_features=self.first_embedding_dim,
            bias=True
        )

        self.second_embedding_layer = torch.nn.Linear(
            in_features=self.first_embedding_dim,
            out_features=self.second_embedding_dim,
            bias=True
        )

    def forward(self, x, vars=None):
        first_hidden = self.first_embedding_layer(x)
        first_hidden = F.relu(first_hidden)
        sec_hidden = self.second_embedding_layer(first_hidden)
        return F.relu(sec_hidden)
```

```python id="XT5VoHl0UQh9"
class Movie_user(torch.nn.Module):
    def __init__(self, config):
        super(Movie_user, self).__init__()
        self.num_gender = config['num_gender']
        self.num_age = config['num_age']
        self.num_occupation = config['num_occupation']
        self.num_zipcode = config['num_zipcode']
        self.embedding_dim = config['embedding_dim']

        self.embedding_gender = torch.nn.Embedding(
            num_embeddings=self.num_gender,
            embedding_dim=self.embedding_dim
        )

        self.embedding_age = torch.nn.Embedding(
            num_embeddings=self.num_age,
            embedding_dim=self.embedding_dim
        )

        self.embedding_occupation = torch.nn.Embedding(
            num_embeddings=self.num_occupation,
            embedding_dim=self.embedding_dim
        )

        self.embedding_area = torch.nn.Embedding(
            num_embeddings=self.num_zipcode,
            embedding_dim=self.embedding_dim
        )

    def forward(self, gender_idx, age_idx, occupation_idx, area_idx):
        gender_emb = self.embedding_gender(gender_idx)
        age_emb = self.embedding_age(age_idx)
        occupation_emb = self.embedding_occupation(occupation_idx)
        area_emb = self.embedding_area(area_idx)
        return torch.cat((gender_emb, age_emb, occupation_emb, area_emb), 1)
```

```python id="BG3qnUnjUPTF"
class Encoder(nn.Module):
    #Maps an (x_i, y_i) pair to a representation r_i.
    # Add the dropout into encoder ---03.31
    def __init__(self, x_dim, y_dim, h1_dim, h2_dim, z1_dim, dropout_rate):
        super(Encoder, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        self.z1_dim = z1_dim
        self.dropout_rate = dropout_rate

        layers = [nn.Linear(self.x_dim + self.y_dim, self.h1_dim),
                  torch.nn.Dropout(self.dropout_rate),
                  nn.ReLU(inplace=True),
                  nn.Linear(self.h1_dim, self.h2_dim),
                  torch.nn.Dropout(self.dropout_rate),
                  nn.ReLU(inplace=True),
                  nn.Linear(self.h2_dim, self.z1_dim)]

        self.input_to_hidden = nn.Sequential(*layers)

    def forward(self, x, y):
        y = y.view(-1, 1)
        input_pairs = torch.cat((x, y), dim=1)
        return self.input_to_hidden(input_pairs)
```

```python id="2OBDuceVUOFr"
class MuSigmaEncoder(nn.Module):
    def __init__(self, z1_dim, z2_dim, z_dim):
        super(MuSigmaEncoder, self).__init__()

        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.z_dim = z_dim
        self.z_to_hidden = nn.Linear(self.z1_dim, self.z2_dim)
        self.hidden_to_mu = nn.Linear(self.z2_dim, z_dim)
        self.hidden_to_logsigma = nn.Linear(self.z2_dim, z_dim)

    def forward(self, z_input):
        hidden = torch.relu(self.z_to_hidden(z_input))
        mu = self.hidden_to_mu(hidden)
        log_sigma = self.hidden_to_logsigma(hidden)
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        return mu, log_sigma, z
```

```python id="y4Gt1yo0UMxL"
class TaskEncoder(nn.Module):
    def __init__(self, x_dim, y_dim, h1_dim, h2_dim, final_dim, dropout_rate):
        super(TaskEncoder, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        self.final_dim = final_dim
        self.dropout_rate = dropout_rate
        layers = [nn.Linear(self.x_dim + self.y_dim, self.h1_dim),
                  torch.nn.Dropout(self.dropout_rate),
                  nn.ReLU(inplace=True),
                  nn.Linear(self.h1_dim, self.h2_dim),
                  torch.nn.Dropout(self.dropout_rate),
                  nn.ReLU(inplace=True),
                  nn.Linear(self.h2_dim, self.final_dim)]

        self.input_to_hidden = nn.Sequential(*layers)

    def forward(self, x, y):
        y = y.view(-1, 1)
        input_pairs = torch.cat((x, y), dim=1)
        return self.input_to_hidden(input_pairs)

class MemoryUnit(nn.Module):
    # clusters_k is k keys
    def __init__(self, clusters_k, emb_size, temperature):
        super(MemoryUnit, self).__init__()
        self.clusters_k = clusters_k
        self.embed_size = emb_size
        self.temperature = temperature
        self.array = nn.Parameter(init.xavier_uniform_(torch.FloatTensor(self.clusters_k, self.embed_size)))

    def forward(self, task_embed):
        res = torch.norm(task_embed-self.array, p=2, dim=1, keepdim=True)
        res = torch.pow((res / self.temperature) + 1, (self.temperature + 1) / -2)
        # 1*k
        C = torch.transpose(res / res.sum(), 0, 1)
        # 1*k, k*d, 1*d
        value = torch.mm(C, self.array)
        # simple add operation
        new_task_embed = value + task_embed
        # calculate target distribution
        return C, new_task_embed
```

```python id="x7nG1TReULk4"
class Decoder(nn.Module):
    """
    Maps target input x_target and z, r to predictions y_target.
    """
    def __init__(self, x_dim, z_dim, task_dim, h1_dim, h2_dim, h3_dim, y_dim, dropout_rate):
        super(Decoder, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.task_dim = task_dim
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        self.h3_dim = h3_dim
        self.y_dim = y_dim
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)

        self.hidden_layer_1 = nn.Linear(self.x_dim + self.z_dim, self.h1_dim)
        self.hidden_layer_2 = nn.Linear(self.h1_dim, self.h2_dim)
        self.hidden_layer_3 = nn.Linear(self.h2_dim, self.h3_dim)

        self.film_layer_1_beta = nn.Linear(self.task_dim, self.h1_dim, bias=False)
        self.film_layer_1_gamma = nn.Linear(self.task_dim, self.h1_dim, bias=False)
        self.film_layer_2_beta = nn.Linear(self.task_dim, self.h2_dim, bias=False)
        self.film_layer_2_gamma = nn.Linear(self.task_dim, self.h2_dim, bias=False)
        self.film_layer_3_beta = nn.Linear(self.task_dim, self.h3_dim, bias=False)
        self.film_layer_3_gamma = nn.Linear(self.task_dim, self.h3_dim, bias=False)

        self.final_projection = nn.Linear(self.h3_dim, self.y_dim)

    def forward(self, x, z, task):
        interaction_size, _ = x.size()
        z = z.unsqueeze(0).repeat(interaction_size, 1)
        # Input is concatenation of z with every row of x
        inputs = torch.cat((x, z), dim=1)
        hidden_1 = self.hidden_layer_1(inputs)
        beta_1 = torch.tanh(self.film_layer_1_beta(task))
        gamma_1 = torch.tanh(self.film_layer_1_gamma(task))
        hidden_1 = torch.mul(hidden_1, gamma_1) + beta_1
        hidden_1 = self.dropout(hidden_1)
        hidden_2 = F.relu(hidden_1)

        hidden_2 = self.hidden_layer_2(hidden_2)
        beta_2 = torch.tanh(self.film_layer_2_beta(task))
        gamma_2 = torch.tanh(self.film_layer_2_gamma(task))
        hidden_2 = torch.mul(hidden_2, gamma_2) + beta_2
        hidden_2 = self.dropout(hidden_2)
        hidden_3 = F.relu(hidden_2)

        hidden_3 = self.hidden_layer_3(hidden_3)
        beta_3 = torch.tanh(self.film_layer_3_beta(task))
        gamma_3 = torch.tanh(self.film_layer_3_gamma(task))
        hidden_final = torch.mul(hidden_3, gamma_3) + beta_3
        hidden_final = self.dropout(hidden_final)
        hidden_final = F.relu(hidden_final)

        y_pred = self.final_projection(hidden_final)
        return y_pred
```

```python id="edivGiDDUJ_V"
class Gating_Decoder(nn.Module):

    def __init__(self, x_dim, z_dim, task_dim, h1_dim, h2_dim, h3_dim, y_dim, dropout_rate):
        super(Gating_Decoder, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.task_dim = task_dim
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        self.h3_dim = h3_dim
        self.y_dim = y_dim
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)

        self.hidden_layer_1 = nn.Linear(self.x_dim + self.z_dim, self.h1_dim)
        self.hidden_layer_2 = nn.Linear(self.h1_dim, self.h2_dim)
        self.hidden_layer_3 = nn.Linear(self.h2_dim, self.h3_dim)

        self.film_layer_1_beta = nn.Linear(self.task_dim, self.h1_dim, bias=False)
        self.film_layer_1_gamma = nn.Linear(self.task_dim, self.h1_dim, bias=False)
        self.film_layer_1_eta = nn.Linear(self.task_dim, self.h1_dim, bias=False)
        self.film_layer_1_delta = nn.Linear(self.task_dim, self.h1_dim, bias=False)

        self.film_layer_2_beta = nn.Linear(self.task_dim, self.h2_dim, bias=False)
        self.film_layer_2_gamma = nn.Linear(self.task_dim, self.h2_dim, bias=False)
        self.film_layer_2_eta = nn.Linear(self.task_dim, self.h2_dim, bias=False)
        self.film_layer_2_delta = nn.Linear(self.task_dim, self.h2_dim, bias=False)


        self.film_layer_3_beta = nn.Linear(self.task_dim, self.h3_dim, bias=False)
        self.film_layer_3_gamma = nn.Linear(self.task_dim, self.h3_dim, bias=False)
        self.film_layer_3_eta = nn.Linear(self.task_dim, self.h3_dim, bias=False)
        self.film_layer_3_delta = nn.Linear(self.task_dim, self.h3_dim, bias=False)


        self.final_projection = nn.Linear(self.h3_dim, self.y_dim)

    def forward(self, x, z, task):
        interaction_size, _ = x.size()
        z = z.unsqueeze(0).repeat(interaction_size, 1)
        # Input is concatenation of z with every row of x
        inputs = torch.cat((x, z), dim=1)
        hidden_1 = self.hidden_layer_1(inputs)
        beta_1 = torch.tanh(self.film_layer_1_beta(task))
        gamma_1 = torch.tanh(self.film_layer_1_gamma(task))
        eta_1 = torch.tanh(self.film_layer_1_eta(task))
        delta_1 = torch.sigmoid(self.film_layer_1_delta(task))

        gamma_1 = gamma_1 * delta_1 + eta_1 * (1-delta_1)
        beta_1 = beta_1 * delta_1 + eta_1 * (1-delta_1)

        hidden_1 = torch.mul(hidden_1, gamma_1) + beta_1
        hidden_1 = self.dropout(hidden_1)
        hidden_2 = F.relu(hidden_1)

        hidden_2 = self.hidden_layer_2(hidden_2)
        beta_2 = torch.tanh(self.film_layer_2_beta(task))
        gamma_2 = torch.tanh(self.film_layer_2_gamma(task))
        eta_2 = torch.tanh(self.film_layer_2_eta(task))
        delta_2 = torch.sigmoid(self.film_layer_2_delta(task))

        gamma_2 = gamma_2 * delta_2 + eta_2 * (1 - delta_2)
        beta_2 = beta_2 * delta_2 + eta_2 * (1 - delta_2)


        hidden_2 = torch.mul(hidden_2, gamma_2) + beta_2
        hidden_2 = self.dropout(hidden_2)
        hidden_3 = F.relu(hidden_2)
        hidden_3 = self.hidden_layer_3(hidden_3)
        beta_3 = torch.tanh(self.film_layer_3_beta(task))
        gamma_3 = torch.tanh(self.film_layer_3_gamma(task))
        eta_3 = torch.tanh(self.film_layer_3_eta(task))
        delta_3 = torch.sigmoid(self.film_layer_3_delta(task))

        gamma_3 = gamma_3 * delta_3 + eta_3 * (1 - delta_3)
        beta_3 = beta_3 * delta_3 + eta_3 * (1 - delta_3)

        hidden_final = torch.mul(hidden_3, gamma_3) + beta_3
        hidden_final = self.dropout(hidden_final)
        hidden_final = F.relu(hidden_final)

        y_pred = self.final_projection(hidden_final)
        return y_pred
```

<!-- #region id="1G27QCrDVCO1" -->
### TaNP
<!-- #endregion -->

<!-- #region id="QpkQ2ri90z_5" -->
<p><center><img src='_images/T722684_2.png'></center></p>
<!-- #endregion -->

```python id="cg1xj6X_VEVH"
class NP(nn.Module):
    def __init__(self, config):
        super(NP, self).__init__()
        self.x_dim = config['second_embedding_dim'] * 2
        # use one-hot or not?
        self.y_dim = 1
        self.z1_dim = config['z1_dim']
        self.z2_dim = config['z2_dim']
        # z is the dimension size of mu and sigma.
        self.z_dim = config['z_dim']
        # the dimension size of rc.
        self.enc_h1_dim = config['enc_h1_dim']
        self.enc_h2_dim = config['enc_h2_dim']

        self.dec_h1_dim = config['dec_h1_dim']
        self.dec_h2_dim = config['dec_h2_dim']
        self.dec_h3_dim = config['dec_h3_dim']

        self.taskenc_h1_dim = config['taskenc_h1_dim']
        self.taskenc_h2_dim = config['taskenc_h2_dim']
        self.taskenc_final_dim = config['taskenc_final_dim']

        self.clusters_k = config['clusters_k']
        self.temperture = config['temperature']
        self.dropout_rate = config['dropout_rate']

        # Initialize networks
        self.item_emb = Item(config)
        self.user_emb = User(config)
        # This encoder is used to generated z actually, it is a latent encoder in ANP.
        self.xy_to_z = Encoder(self.x_dim, self.y_dim, self.enc_h1_dim, self.enc_h2_dim, self.z1_dim, self.dropout_rate)
        self.z_to_mu_sigma = MuSigmaEncoder(self.z1_dim, self.z2_dim, self.z_dim)
        # This encoder is used to generated r actually, it is a deterministic encoder in ANP.
        self.xy_to_task = TaskEncoder(self.x_dim, self.y_dim, self.taskenc_h1_dim, self.taskenc_h2_dim, self.taskenc_final_dim,
                                      self.dropout_rate)
        self.memoryunit = MemoryUnit(self.clusters_k, self.taskenc_final_dim, self.temperture)
        #self.xz_to_y = Gating_Decoder(self.x_dim, self.z_dim, self.taskenc_final_dim, self.dec_h1_dim, self.dec_h2_dim, self.dec_h3_dim, self.y_dim, self.dropout_rate)
        self.xz_to_y = Decoder(self.x_dim, self.z_dim, self.taskenc_final_dim, self.dec_h1_dim, self.dec_h2_dim, self.dec_h3_dim, self.y_dim, self.dropout_rate)

    def aggregate(self, z_i):
        return torch.mean(z_i, dim=0)

    def xy_to_mu_sigma(self, x, y):
        # Encode each point into a representation r_i
        z_i = self.xy_to_z(x, y)
        # Aggregate representations r_i into a single representation r
        z = self.aggregate(z_i)
        # Return parameters of distribution
        return self.z_to_mu_sigma(z)

    # embedding each (item, user) as the x for np
    def embedding(self, x):
        if_dim = self.item_emb.feature_dim
        item_x = Variable(x[:, 0:if_dim], requires_grad=False).float()
        user_x = Variable(x[:, if_dim:], requires_grad=False).float()
        item_emb = self.item_emb(item_x)
        user_emb = self.user_emb(user_x)
        x = torch.cat((item_emb, user_emb), 1)
        return x

    def forward(self, x_context, y_context, x_target, y_target):
        x_context_embed = self.embedding(x_context)
        x_target_embed = self.embedding(x_target)

        if self.training:
            # sigma is log_sigma actually
            mu_target, sigma_target, z_target = self.xy_to_mu_sigma(x_target_embed, y_target)
            mu_context, sigma_context, z_context = self.xy_to_mu_sigma(x_context_embed, y_context)
            task = self.xy_to_task(x_context_embed, y_context)
            mean_task = self.aggregate(task)
            C_distribution, new_task_embed = self.memoryunit(mean_task)
            p_y_pred = self.xz_to_y(x_target_embed, z_target, new_task_embed)
            return p_y_pred, mu_target, sigma_target, mu_context, sigma_context, C_distribution
        else:
            mu_context, sigma_context, z_context = self.xy_to_mu_sigma(x_context_embed, y_context)
            task = self.xy_to_task(x_context_embed, y_context)
            mean_task = self.aggregate(task)
            C_distribution, new_task_embed = self.memoryunit(mean_task)
            p_y_pred = self.xz_to_y(x_target_embed, z_context, new_task_embed)
            return p_y_pred


class Trainer(torch.nn.Module):
    def __init__(self, config):
        self.opt = config
        super(Trainer, self).__init__()
        self.use_cuda = config['use_cuda']
        self.np = NP(self.opt)
        self._lambda = config['lambda']
        self.optimizer = torch.optim.Adam(self.np.parameters(), lr=config['lr'])

    # our kl divergence
    def kl_div(self, mu_target, logsigma_target, mu_context, logsigma_context):
        target_sigma = torch.exp(logsigma_target)
        context_sigma = torch.exp(logsigma_context)
        kl_div = (logsigma_context - logsigma_target) - 0.5 + (((target_sigma ** 2) + (mu_target - mu_context) ** 2) / 2 * context_sigma ** 2)
        #kl_div = (t.exp(posterior_var) + (posterior_mu-prior_mu) ** 2) / t.exp(prior_var) - 1. + (prior_var - posterior_var)
        #kl_div = 0.5 * kl_div.sum()
        kl_div = kl_div.sum()
        return kl_div

    # new kl divergence -- kl(st|sc)
    def new_kl_div(self, prior_mu, prior_var, posterior_mu, posterior_var):
        kl_div = (torch.exp(posterior_var) + (posterior_mu-prior_mu) ** 2) / torch.exp(prior_var) - 1. + (prior_var - posterior_var)
        kl_div = 0.5 * kl_div.sum()
        return kl_div

    def loss(self, p_y_pred, y_target, mu_target, sigma_target, mu_context, sigma_context):
        #print('p_y_pred size is ', p_y_pred.size())
        regression_loss = F.mse_loss(p_y_pred, y_target.view(-1, 1))
        #print('regession loss size is ', regression_loss.size())
        # kl divergence between target and context
        #print('regession_loss is ', regression_loss.item())
        kl = self.new_kl_div(mu_context, sigma_context, mu_target, sigma_target)
        #print('KL_loss is ', kl.item())
        return regression_loss+kl

    def context_target_split(self, support_set_x, support_set_y, query_set_x, query_set_y):
        total_x = torch.cat((support_set_x, query_set_x), 0)
        total_y = torch.cat((support_set_y, query_set_y), 0)
        total_size = total_x.size(0)
        context_min = self.opt['context_min']
        context_max = self.opt['context_max']
        extra_tar_min = self.opt['target_extra_min']
        #here we simply use the total_size as the maximum of target size.
        num_context = randint(context_min, context_max)
        num_target = randint(extra_tar_min, total_size - num_context)
        sampled = np.random.choice(total_size, num_context+num_target, replace=False)
        x_context = total_x[sampled[:num_context], :]
        y_context = total_y[sampled[:num_context]]
        x_target = total_x[sampled, :]
        y_target = total_y[sampled]
        return x_context, y_context, x_target, y_target

    def new_context_target_split(self, support_set_x, support_set_y, query_set_x, query_set_y):
        total_x = torch.cat((support_set_x, query_set_x), 0)
        total_y = torch.cat((support_set_y, query_set_y), 0)
        total_size = total_x.size(0)
        context_min = self.opt['context_min']
        num_context = np.random.randint(context_min, total_size)
        num_target = np.random.randint(0, total_size - num_context)
        sampled = np.random.choice(total_size, num_context+num_target, replace=False)
        x_context = total_x[sampled[:num_context], :]
        y_context = total_y[sampled[:num_context]]
        x_target = total_x[sampled, :]
        y_target = total_y[sampled]
        return x_context, y_context, x_target, y_target

    def global_update(self, support_set_xs, support_set_ys, query_set_xs, query_set_ys):
        batch_sz = len(support_set_xs)
        losses = []
        C_distribs = []
        if self.use_cuda:
            for i in range(batch_sz):
                support_set_xs[i] = support_set_xs[i].cuda()
                support_set_ys[i] = support_set_ys[i].cuda()
                query_set_xs[i] = query_set_xs[i].cuda()
                query_set_ys[i] = query_set_ys[i].cuda()
        for i in range(batch_sz):
            x_context, y_context, x_target, y_target = self.new_context_target_split(support_set_xs[i], support_set_ys[i],
                                                                                 query_set_xs[i], query_set_ys[i])
            p_y_pred, mu_target, sigma_target, mu_context, sigma_context, C_distribution = self.np(x_context, y_context, x_target,
                                                                                  y_target)
            C_distribs.append(C_distribution)
            loss = self.loss(p_y_pred, y_target, mu_target, sigma_target, mu_context, sigma_context)
            #print('Each task has loss: ', loss)
            losses.append(loss)
        # calculate target distribution for clustering in batch manner.
        # batchsize * k
        C_distribs = torch.stack(C_distribs)
        # batchsize * k
        C_distribs_sq = torch.pow(C_distribs, 2)
        # 1*k
        C_distribs_sum = torch.sum(C_distribs, dim=0, keepdim=True)
        # batchsize * k
        temp = C_distribs_sq / C_distribs_sum
        # batchsize * 1
        temp_sum = torch.sum(temp, dim=1, keepdim=True)
        target_distribs = temp / temp_sum
        # calculate the kl loss
        clustering_loss = self._lambda * F.kl_div(C_distribs.log(), target_distribs, reduction='batchmean')
        #print('The clustering loss is %.6f' % (clustering_loss.item()))
        np_losses_mean = torch.stack(losses).mean(0)
        total_loss = np_losses_mean + clustering_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item(), C_distribs.cpu().detach().numpy()

    def query_rec(self, support_set_xs, support_set_ys, query_set_xs, query_set_ys):
        batch_sz = 1
        # used for calculating the rmse.
        losses_q = []
        if self.use_cuda:
            for i in range(batch_sz):
                support_set_xs[i] = support_set_xs[i].cuda()
                support_set_ys[i] = support_set_ys[i].cuda()
                query_set_xs[i] = query_set_xs[i].cuda()
                query_set_ys[i] = query_set_ys[i].cuda()
        for i in range(batch_sz):
            #query_set_y_pred = self.forward(support_set_xs[i], support_set_ys[i], query_set_xs[i], num_local_update)
            query_set_y_pred = self.np(support_set_xs[i], support_set_ys[i], query_set_xs[i], query_set_ys[i])
            # obtain the mean of gaussian distribution
            #(interation_size, y_dim)
            #query_set_y_pred = query_set_y_pred.loc.detach()
            #print('test_y_pred size is ', query_set_y_pred.size())
            loss_q = F.mse_loss(query_set_y_pred, query_set_ys[i].view(-1, 1))
            losses_q.append(loss_q)
        losses_q = torch.stack(losses_q).mean(0)
        output_list, recommendation_list = query_set_y_pred.view(-1).sort(descending=True)
        return losses_q.item(), recommendation_list
```

<!-- #region id="Ee3MqBcWUf1j" -->
## Evaluation modules
<!-- #endregion -->

<!-- #region id="R5hFfpM3T4t3" -->
### Metrics
<!-- #endregion -->

```python id="DcoNb06IT5fW"
def AP(ranked_list, ground_truth, topn):
    hits, sum_precs = 0, 0.0
    t = [a for a in ground_truth]
    t.sort(reverse=True)
    t=t[:topn]
    for i in range(topn):
        id = ranked_list[i]
        if ground_truth[id] in t:
            hits += 1
            sum_precs += hits / (i+1.0)
            t.remove(ground_truth[id])
    if hits > 0:
        return sum_precs / topn
    else:
        return 0.0

def RR(ranked_list, ground_truth,topn):
    t = [a for a in ground_truth]
    t.sort(reverse=True)
    t = t[:topn]
    for i in range(topn):
        id = ranked_list[i]
        if ground_truth[id] in t:
            return 1 / (i + 1.0)
    return 0

def precision(ranked_list,ground_truth,topn):
    t = [a for a in ground_truth]
    t.sort(reverse=True)
    t = t[:topn]
    hits = 0
    for i in range(topn):
        id = ranked_list[i]
        if ground_truth[id] in t:
            t.remove(ground_truth[id])
            hits += 1
    pre = hits/topn
    return pre


def nDCG(ranked_list, ground_truth, topn):
    dcg = 0
    idcg = IDCG(ground_truth, topn)
    # print(ranked_list)
    # input()
    for i in range(topn):
        id = ranked_list[i]
        dcg += ((2 ** ground_truth[id]) -1)/ math.log(i+2, 2)
    # print('dcg is ', dcg, " n is ", topn)
    # print('idcg is ', idcg, " n is ", topn)
    return dcg / idcg

def IDCG(ground_truth,topn):
    t = [a for a in ground_truth]
    t.sort(reverse=True)
    idcg = 0
    for i in range(topn):
        idcg += ((2**t[i]) - 1) / math.log(i+2, 2)
    return idcg

def add_metric(recommend_list, ALL_group_list, precision_list, ap_list, ndcg_list, topn):
    ndcg = nDCG(recommend_list, ALL_group_list, topn)
    ap = AP(recommend_list, ALL_group_list, topn)
    pre = precision(recommend_list, ALL_group_list, topn)
    precision_list.append(pre)
    ap_list.append(ap)
    ndcg_list.append(ndcg)



def cal_metric(precision_list,ap_list,ndcg_list):
    mpre = sum(precision_list) / len(precision_list)
    map = sum(ap_list) / len(ap_list)
    mndcg = sum(ndcg_list) / len(ndcg_list)
    return mpre, mndcg, map
```

<!-- #region id="lwmoOPmMUi4L" -->
### Testing
<!-- #endregion -->

```python id="gueZYV8CUh4F"
def testing(trainer, opt, test_dataset):
    test_dataset_len = len(test_dataset)
    #batch_size = opt["batch_size"]
    minibatch_size = 1
    a, b, c, d = zip(*test_dataset)
    trainer.eval()
    all_loss = 0
    pre5 = []
    ap5 = []
    ndcg5 = []
    pre7 = []
    ap7 = []
    ndcg7 = []
    pre10 = []
    ap10 = []
    ndcg10 = []
    for i in range(test_dataset_len):
        try:
            supp_xs = list(a[minibatch_size * i:minibatch_size * (i + 1)])
            supp_ys = list(b[minibatch_size * i:minibatch_size * (i + 1)])
            query_xs = list(c[minibatch_size * i:minibatch_size * (i + 1)])
            query_ys = list(d[minibatch_size * i:minibatch_size * (i + 1)])
        except IndexError:
            continue
        test_loss, recommendation_list = trainer.query_rec(supp_xs, supp_ys, query_xs, query_ys)
        all_loss += test_loss

        add_metric(recommendation_list, query_ys[0].cpu().detach().numpy(), pre5, ap5, ndcg5, 5)
        add_metric(recommendation_list, query_ys[0].cpu().detach().numpy(), pre7, ap7, ndcg7, 7)
        add_metric(recommendation_list, query_ys[0].cpu().detach().numpy(), pre10, ap10, ndcg10, 10)

    mpre5, mndcg5, map5 = cal_metric(pre5, ap5, ndcg5)
    mpre7, mndcg7, map7 = cal_metric(pre7, ap7, ndcg7)
    mpre10, mndcg10, map10 = cal_metric(pre10, ap10, ndcg10)

    return mpre5, mndcg5, map5, mpre7, mndcg7, map7, mpre10, mndcg10, map10
```

<!-- #region id="zRhfeDkuUpjQ" -->
## Training and Evaluation
<!-- #endregion -->

```python id="DcrOaTdSUq7h"
def training(trainer, opt, train_dataset, test_dataset, batch_size, num_epoch, model_save=True, model_filename=None, logger=None):
    training_set_size = len(train_dataset)
    for epoch in range(num_epoch):
        random.shuffle(train_dataset)
        num_batch = int(training_set_size / batch_size)
        a, b, c, d = zip(*train_dataset)
        trainer.train()
        all_C_distribs = []
        for i in range(num_batch):
            try:
                supp_xs = list(a[batch_size*i:batch_size*(i+1)])
                supp_ys = list(b[batch_size*i:batch_size*(i+1)])
                query_xs = list(c[batch_size*i:batch_size*(i+1)])
                query_ys = list(d[batch_size*i:batch_size*(i+1)])
            except IndexError:
                continue
            train_loss, batch_C_distribs = trainer.global_update(supp_xs, supp_ys, query_xs, query_ys)
            all_C_distribs.append(batch_C_distribs)

        P5, NDCG5, MAP5, P7, NDCG7, MAP7, P10, NDCG10, MAP10 = testing(trainer, opt, test_dataset)
        logger.log(
            "{}\t{:.6f}\t TOP-5 {:.4f}\t{:.4f}\t{:.4f}\t TOP-7: {:.4f}\t{:.4f}\t{:.4f}"
            "\t TOP-10: {:.4f}\t{:.4f}\t{:.4f}".
                format(epoch, train_loss, P5, NDCG5, MAP5, P7, NDCG7, MAP7, P10, NDCG10, MAP10))
        if epoch == (num_epoch-1):
            with open('output_att', 'wb') as fp:
                pickle.dump(all_C_distribs, fp)

    if model_save:
        torch.save(trainer.state_dict(), model_filename)
```

```python colab={"base_uri": "https://localhost:8080/"} id="tofDHS2uWG_w" executionInfo={"status": "ok", "timestamp": 1635851932664, "user_tz": -330, "elapsed": 1023, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3cf1dc5d-f881-4957-80b8-a7406d253183"
# print model info
print_config(opt)
ensure_dir(opt["model_save_dir"], verbose=True)

# save model config
save_config(opt, opt["model_save_dir"] + "/" +opt["id"] + '.config', verbose=True)

# record training log
file_logger = FileLogger(opt["model_save_dir"] + '/' + opt['id'] + ".log",
                                header="# epoch\ttrain_loss\tprecision5\tNDCG5\tMAP5\tprecision7"
                                       "\tNDCG7\tMAP7\tprecision10\tNDCG10\tMAP10")

preprocess = Preprocess(opt)
print("Preprocess is done.")
```

```python colab={"base_uri": "https://localhost:8080/"} id="0LcVm-jbX2DY" executionInfo={"status": "ok", "timestamp": 1635852001756, "user_tz": -330, "elapsed": 629, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="41cf96af-8acd-48b3-fe86-f97f4e285df1"
print("Create model TaNP...")

opt['uf_dim'] = preprocess.uf_dim
opt['if_dim'] = preprocess.if_dim

trainer = Trainer(opt)

if opt['use_cuda']:
    trainer.cuda()

model_filename = "{}/{}.pt".format(opt['model_save_dir'], opt["id"])
```

```python colab={"base_uri": "https://localhost:8080/"} id="akwldBc9YI6G" executionInfo={"status": "ok", "timestamp": 1635852007273, "user_tz": -330, "elapsed": 1811, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="653f6042-f6f8-4647-87d5-0c24fd303157"
# /4 since sup_x, sup_y, query_x, query_y
training_set_size = int(len(os.listdir("{}/{}/{}".format(opt["data_dir"], "training", "log"))) / 4)
supp_xs_s = []
supp_ys_s = []
query_xs_s = []
query_ys_s = []

for idx in range(training_set_size):
    supp_xs_s.append(pickle.load(open("{}/{}/{}/supp_x_{}.pkl".format(opt["data_dir"], "training", "log", idx), "rb")))
    supp_ys_s.append(pickle.load(open("{}/{}/{}/supp_y_{}.pkl".format(opt["data_dir"], "training", "log", idx), "rb")))
    query_xs_s.append(pickle.load(open("{}/{}/{}/query_x_{}.pkl".format(opt["data_dir"], "training", "log", idx), "rb")))
    query_ys_s.append(pickle.load(open("{}/{}/{}/query_y_{}.pkl".format(opt["data_dir"], "training", "log", idx), "rb")))

train_dataset = list(zip(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s))

del (supp_xs_s, supp_ys_s, query_xs_s, query_ys_s)

testing_set_size = int(len(os.listdir("{}/{}/{}".format(opt["data_dir"], "testing", "log"))) / 4)
supp_xs_s = []
supp_ys_s = []
query_xs_s = []
query_ys_s = []

for idx in range(testing_set_size):
    supp_xs_s.append(
        pickle.load(open("{}/{}/{}/supp_x_{}.pkl".format(opt["data_dir"], "testing", "log", idx), "rb")))
    supp_ys_s.append(
        pickle.load(open("{}/{}/{}/supp_y_{}.pkl".format(opt["data_dir"], "testing", "log", idx), "rb")))
    query_xs_s.append(
        pickle.load(open("{}/{}/{}/query_x_{}.pkl".format(opt["data_dir"], "testing", "log", idx), "rb")))
    query_ys_s.append(
        pickle.load(open("{}/{}/{}/query_y_{}.pkl".format(opt["data_dir"], "testing", "log", idx), "rb")))
    
test_dataset = list(zip(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s))

del (supp_xs_s, supp_ys_s, query_xs_s, query_ys_s)

print("# epoch\ttrain_loss\tprecision5\tNDCG5\tMAP5\tprecision7\tNDCG7\tMAP7\tprecision10\tNDCG10\tMAP10")
```

```python colab={"base_uri": "https://localhost:8080/"} id="B250JVLpYHZg" executionInfo={"status": "ok", "timestamp": 1635852355637, "user_tz": -330, "elapsed": 335740, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="df0b37ad-6ec8-4fcd-97f6-28bbf010c763"
if not os.path.exists(model_filename):
    print("Start training...")
    training(trainer, opt, train_dataset, test_dataset, batch_size=opt['batch_size'], num_epoch=opt['num_epoch'],
            model_save=opt["save"], model_filename=model_filename, logger=file_logger)

else:
    print("Load pre-trained model...")
    opt = helper.load_config(model_filename[:-2]+"config")
    helper.print_config(opt)
    trained_state_dict = torch.load(model_filename)
    trainer.load_state_dict(trained_state_dict)
```

<!-- #region id="iXB7KEkLYOXQ" -->
**END**
<!-- #endregion -->
