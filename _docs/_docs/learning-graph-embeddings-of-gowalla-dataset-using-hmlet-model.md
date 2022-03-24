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

<!-- #region id="NtgN996tgfc6" -->
# Learning Graph Embeddings of Gowalla Dataset using HMLET Model
<!-- #endregion -->

<!-- #region id="BXJY8c9d4Xi5" -->
## Data Ingestion
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="DctyNOSdx-7h" executionInfo={"status": "ok", "timestamp": 1637227997776, "user_tz": -330, "elapsed": 4696, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="69b1c651-a356-4e5d-d1b7-087a806ed1f1"
!mkdir -p /content/data/gowalla
%cd /content/data/gowalla
!wget -q --show-progress https://github.com/RecoHut-Datasets/gowalla/raw/main/silver/v1/s_pre_adj_mat_train.npz
!wget -q --show-progress https://github.com/RecoHut-Datasets/gowalla/raw/main/silver/v1/train.txt
!wget -q --show-progress https://github.com/RecoHut-Datasets/gowalla/raw/main/silver/v1/test.txt
!wget -q --show-progress https://github.com/RecoHut-Datasets/gowalla/raw/main/silver/v1/val.txt
%cd /content
```

<!-- #region id="GB_yDppW3_Yt" -->
## Imports
<!-- #endregion -->

```python id="vrEmNkAAsQlM"
import numpy as np
from tqdm.notebook import tqdm
import sys
import os
import math
import logging
import pandas as pd
from pathlib import Path
from os.path import join, dirname
import multiprocessing
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from time import time
import time as tm
import random
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch import log
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
```

<!-- #region id="NyxCtlrJ3_Ta" -->
## Params
<!-- #endregion -->

```python id="MXBwnUCD3_RD"
class Args:

    # Model
    model = 'HMLET_End' # "model type", choices={HMLET_End", "HMLET_Middle", "HMLET_Front", "HMLET_All"}
    embedding_dim = 512 # the embedding size
    non_linear_acti = 'elu' # activation function to use in non-linear aggregation, choices={"relu", "leaky-relu", "elu"}
    dropout = 1 # using the dropout or not
    keepprob = 0.6 # dropout node keeping probability
                            
    # Dataset
    dataset = 'gowalla' # dataset, choices={"gowalla", "yelp2018", "amazon-book"
    bpr_batch = 2048 # the batch size for bpr loss training procedure

    # Gumbel-Softmax
    ori_temp = 0.7 # start temperature
    min_temp = 0.01 # min temperature
    gum_temp_decay = 0.005 # value of temperature decay
    epoch_temp_decay = 1 # epoch to apply temperature decay
    division_noise = 3 # division number of noise
                            
    # Train
    # epochs = 1000 # train epochs
    epochs = 4 # train epochs
    lr = 0.001 # the learning rate
    decay = 1e-4 # the weight decay for l2 normalizaton
                                    
    # Test
    topks = "[10,20,30,40,50]" # at-k test list
    testbatch = 100 # the batch size of users for testing
    a_split = 0 # split large adj matrix or not
    a_n_fold = 100 # the fold num used to split large adj matrix
                            
    # Util
    root_path = '/content'
    pretrain = 0 # using pretrained weight or not
    pretrained_checkpoint_name = '' # file name of pretrained model
    load_epoch = 1 # epoch of pretrained model
    seed = 2020 # random seed
    multicore = 1 # help using multiprocessing or not
    gpu_num = 0 # gpu number
    save_checkpoints_path = "checkpoints" # path to save weights
    save_excel_path = "excel" # path to save eval files
    tensorboard = 1 # enable tensorboard

args = Args()
```

```python colab={"base_uri": "https://localhost:8080/"} id="kaJFxf6d1JDf" executionInfo={"status": "ok", "timestamp": 1637229242415, "user_tz": -330, "elapsed": 676, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="82e8415d-a10e-41c9-ce9d-003c8da12279"
dataset = args.dataset
model_name = args.model

# Model & Train Param
EPOCHS = args.epochs
SEED = args.seed
pretrain = True if args.pretrain else False
load_epoch = args.load_epoch
topks = eval(args.topks)
a_n_fold = args.a_n_fold
tensorboard = True if args.tensorboard else False

bpr_batch_size = args.bpr_batch
test_u_batch_size = args.testbatch
lr = args.lr
decay = args.decay

config = {}
config['embedding_dim'] = args.embedding_dim
config['activation_function'] = args.non_linear_acti
config['dropout'] = args.dropout
config['keep_prob']  = args.keepprob
config['a_split'] = args.a_split
config['gating_mlp_dims'] = [128, 2]

print('='*30)
print('Model:', model_name)
print('Model config:', config)
print('Dataset:', dataset)
print("EPOCHS:", EPOCHS)
print("Pretrain:", pretrain)
print("BPR batch size:", bpr_batch_size)
print("Test batch size:", test_u_batch_size)
print("Test topks:", topks)
print("N fold:", a_n_fold)
print("Tensorboard:", tensorboard)
print('='*30)

# Gumbel-Softmax Param
ori_temp = args.ori_temp
min_temp = args.min_temp
gum_temp_decay = args.gum_temp_decay
epoch_temp_decay = args.epoch_temp_decay
config['division_noise'] = args.division_noise
train_hard = False
test_hard = True


# PATH
ROOT_PATH = args.root_path
DATA_PATH = join(ROOT_PATH, 'data', args.dataset)
SAVE_FILE_PATH = join(ROOT_PATH, args.save_checkpoints_path, model_name, dataset)
LOAD_FILE_PATH = join(ROOT_PATH, args.save_checkpoints_path, model_name, dataset, args.pretrained_checkpoint_name)
EXCEL_PATH = join(ROOT_PATH, args.save_excel_path)
BOARD_PATH = join(ROOT_PATH, 'tensorboard')

print('='*30)
print('DATA PATH:', DATA_PATH)
print('SAVE FILE PATH:', SAVE_FILE_PATH)
print('LOAD FILE PATH:', LOAD_FILE_PATH)
print('EXCEL PATH:', EXCEL_PATH)
print('BOARD PATH:', BOARD_PATH)
print('='*30)

# Making folder
os.makedirs(SAVE_FILE_PATH, exist_ok=True)
os.makedirs(EXCEL_PATH, exist_ok=True)
os.makedirs(BOARD_PATH, exist_ok=True)
   
# GPU
print('='*30)
print('Cuda:', torch.cuda.is_available())
GPU_NUM = args.gpu_num
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
	torch.cuda.set_device(device)
	print('CUDA device:', torch.cuda.current_device())
print('='*30)

# Multi-processing 
multicore = args.multicore
CORES = multiprocessing.cpu_count() // 2
print('='*30)
print("Multicore:", multicore)
print("CORES:", CORES)
print('='*30)

# Excel results dict
excel_results_valid = {}
excel_results_valid['Model'] = []
excel_results_valid['Dataset'] = []
excel_results_valid['Epochs'] = []
excel_results_valid['Precision'] = []
excel_results_valid['Recall(HR)'] = []
excel_results_valid['Ndcg'] = []

excel_results_test = {}
excel_results_test['Model'] = []
excel_results_test['Dataset'] = []
excel_results_test['Epochs'] = []
excel_results_test['Precision'] = []
excel_results_test['Recall(HR)'] = []
excel_results_test['Ndcg'] = []
```

<!-- #region id="qY9Y0q2sz1MS" -->
## Utils
<!-- #endregion -->

```python id="oCeclRK663Of"
def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
```

```python id="wlLXKc2f65Uf"
def getFileName():
    if model_name == 'mf':
        file = f"mf-{dataset}-{config['latent_dim_rec']}.pth.tar"
    elif model_name == 'lgn':
        file = f"lgn-{dataset}-{config['lightGCN_n_layers']}-{config['latent_dim_rec']}.pth.tar"
    return os.path.join(FILE_PATH,file)
```

```python id="4qORYRxz69VI"
def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', bpr_batch_size)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)
```

```python id="ErZPo_bw69Sd"
def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result
```

```python id="HizlUpCV69Qa"
class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                #TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)
```

<!-- #region id="nhSnKpry7POv" -->
## Metrics
<!-- #endregion -->

```python id="MjPJkxKQ7QUM"
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1./np.arange(1, k+1))
    pred_data = pred_data/scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)

def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def AUC(all_item_scores, dataset, test_data):
    """
        design for a single user
    """
    dataset : BasicDataset
    r_all = np.zeros((dataset.m_items, ))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')
```

<!-- #region id="6nPwMgRj4dSn" -->
## Sampling
<!-- #endregion -->

```python id="Itnc1ZWA4Zyz"
sample_ext = False
```

```python id="c4JjyZdt6vcw"
class BPRLoss:
    def __init__(self,
                 model):
        self.model = model
        self.weight_decay = decay
        self.lr = lr
        self.opt = optim.Adam(model.parameters(), lr=self.lr)

    def stageOne(self, users, pos, neg, gum_temp, hard):
        loss, reg_loss, gating_dist, embs = self.model.bpr_loss(users, pos, neg, gum_temp, hard)
        reg_loss = reg_loss*self.weight_decay
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item(), gating_dist, embs

def UniformSample_original(dataset, neg_ratio = 1):
    dataset : BasicDataset
    allPos = dataset.allPos
    start = time()
    if sample_ext:
        S = sampling.sample_negative(dataset.n_users, dataset.m_items,
                                     dataset.trainDataSize, allPos, neg_ratio)
    else:
        S = UniformSample_original_python(dataset)
    return S

def UniformSample_original_python(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    total_start = time()
    dataset : BasicDataset
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        end = time()
        sample_time1 += end - start
    total = time() - total_start
    return np.array(S)
```

<!-- #region id="aWWfMIfz14aY" -->
## Dataloader
<!-- #endregion -->

```python id="aFd-IA_f14MG"
class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError
    
    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError
```

```python id="UEcHRac02JTx"
class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    def __init__(self, path):
        print('Loading', path)
        self.split = config['a_split']
        self.folds = a_n_fold
        self.mode_dict = {'train & valid': 0, "test": 1}
        self.mode = self.mode_dict['train & valid']
        self.n_user = 0
        self.m_item = 0
        train_file = path + '/train.txt'
        valid_file = path + '/val.txt'
        test_file = path + '/test.txt'
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        validUniqueUsers, validItem, validUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0
        self.validDataSize = 0
        self.testDataSize = 0

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)
		
        with open(valid_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    try:
                      items = [int(i) for i in l[1:]]
                    except Exception:
                      continue
                    uid = int(l[0])
                    validUniqueUsers.append(uid)
                    validUser.extend([uid] * len(items))
                    validItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.validDataSize += len(items)
        self.validUniqueUsers = np.array(validUniqueUsers)
        self.validUser = np.array(validUser)
        self.validItem = np.array(validItem)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    try:
                      items = [int(i) for i in l[1:]]
                    except Exception:
                      continue
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)
        self.m_item += 1
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        
        self.Graph = None
        print('='*30)
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{dataset} Sparsity : {(self.traindataSize + self.validDataSize + self.testDataSize) / self.n_users / self.m_items}")
        print('='*30)

        # (users,items), bipartite graph (train)
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.

        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__trainDict = self.__build_train()
        self.__validDict = self.__build_valid()
        self.__testDict = self.__build_test()
        print(f"{dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def trainDict(self):
        return self.__trainDict

    @property
    def validDict(self):
        return self.__validDict
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat_train.npz')
                print("successfully train loaded...")
                norm_adj_train = pre_adj_mat
            except :
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                norm_adj_train = d_mat.dot(adj_mat)
                norm_adj_train = norm_adj_train.dot(d_mat)
                norm_adj_train = norm_adj_train.tocsr()
                end = time()
                print(f"costing {end-s}s, saved train norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat_train.npz', norm_adj_train)

            if self.split:
                self.Graph = self._split_A_hat(norm_adj_train)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj_train)
                self.Graph = self.Graph.coalesce().to(device)
                print("don't split the matrix")
        return self.Graph

    def __build_train(self):
        """
        return:
            dict: {user: [items]}
        """
        train_data = {}
        for i, item in enumerate(self.trainItem):
            user = self.trainUser[i]
            if train_data.get(user):
                train_data[user].append(item)
            else:
                train_data[user] = [item]
        return train_data
        
    def __build_valid(self):
        """
        return:
            dict: {user: [items]}
        """
        valid_data = {}
        for i, item in enumerate(self.validItem):
            user = self.validUser[i]
            if valid_data.get(user):
                valid_data[user].append(item)
            else:
                valid_data[user] = [item]
        return valid_data

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
```

<!-- #region id="1n_jTcJTx0LA" -->
## Gating Network
<!-- #endregion -->

```python id="XvBM9e8sx8Wl"
class Gating_Net(nn.Module):

    def __init__(self, embedding_dim, mlp_dims):
        super(Gating_Net, self).__init__()
        self.embedding_dim = embedding_dim
        self.softmax =  nn.LogSoftmax(dim=1)
        fc_layers = []
        for i in range(len(mlp_dims)):
            if i == 0:
                fc_layers.append(nn.Linear(embedding_dim*2, mlp_dims[i]))
            else:
                fc_layers.append(nn.Linear(mlp_dims[i-1], mlp_dims[i]))	
            if i != len(mlp_dims) - 1:
                fc_layers.append(nn.BatchNorm1d(mlp_dims[i]))
                fc_layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*fc_layers)

    def gumbel_softmax(self, logits, temperature, division_noise, hard):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
          logits: [batch_size, n_class] unnormalized log-probs
          temperature: non-negative scalar
          hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
          [batch_size, n_class] sample from the Gumbel-Softmax distribution.
          If hard=True, then the returned sample will be one-hot, otherwise it will
          be a probabilitiy distribution that sums to 1 across classes
        """
        y = self.gumbel_softmax_sample(logits, temperature, division_noise) ## (0.6, 0.2, 0.1,..., 0.11)
        if hard:
            k = logits.size(1) # k is numb of classes
            # y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)  ## (1, 0, 0, ..., 0)
            y_hard = torch.eq(y, torch.max(y, dim=1, keepdim=True)[0]).type_as(y)
            y = (y_hard - y).detach() + y
        return y

    def gumbel_softmax_sample(self, logits, temperature, division_noise):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        noise = self.sample_gumbel(logits)
        y = (logits + (noise/division_noise)) / temperature
        return F.softmax(y)

    def sample_gumbel(self, logits):
        """Sample from Gumbel(0, 1)"""
        noise = torch.rand(logits.size())
        eps = 1e-20
        noise.add_(eps).log_().neg_()
        noise.add_(eps).log_().neg_()
        return Variable(noise.float()).cuda()

    def forward(self, feature, temperature, hard, division_noise): #z= batch x z_dim // #feature =  batch x num_gen x 256*8*8
        x = self.mlp(feature)
        out = self.gumbel_softmax(x, temperature, division_noise, hard)
        out_value = out.unsqueeze(2)
        out = out_value.repeat(1, 1, self.embedding_dim)
                
        return out, torch.sum(out_value[:,0]), torch.sum(out_value[:,1])
```

<!-- #region id="OuDsPV1mxvQ0" -->
## HMLET (End) Model
<!-- #endregion -->

```python id="nAqs6mNixvN1"
class BasicModel(nn.Module):    
	def __init__(self):
		super(BasicModel, self).__init__()

	def getUsersRating(self, users):
		raise NotImplementedError
```

```python id="In00VlPU3apv"
class HMLET_End(nn.Module):
	def __init__(self, 
					config:dict, 
					dataset:BasicDataset):
		super(HMLET_End, self).__init__()
		self.config = config
		self.dataset : dataloader.BasicDataset = dataset
		self.__init_model()

	def __init_model(self):
		self.num_users = self.dataset.n_users
		self.num_items = self.dataset.m_items
		self.embedding_dim = self.config['embedding_dim']
   
		self.n_layers = 4
		self.dropout = self.config['dropout']
		self.keep_prob = self.config['keep_prob']
		self.A_split = self.config['a_split']

		# Embedding
		self.embedding_user = torch.nn.Embedding(
			num_embeddings=self.num_users, embedding_dim=self.embedding_dim)
		self.embedding_item = torch.nn.Embedding(
			num_embeddings=self.num_items, embedding_dim=self.embedding_dim)
		
		# Normal distribution initilizer
		nn.init.normal_(self.embedding_user.weight, std=0.1)
		nn.init.normal_(self.embedding_item.weight, std=0.1)      
		
		# Activation function
		selected_activation_function = self.config['activation_function']
		
		if selected_activation_function == 'relu':
			self.r = nn.ReLU()
			self.activation_function = self.r
		if selected_activation_function == 'leaky-relu':
			self.leaky = nn.LeakyReLU(0.1)
			self.activation_function = self.leaky
		elif selected_activation_function == 'elu':
			self.elu = nn.ELU()
			self.activation_function = self.elu
		print('activation_function:',self.activation_function)
		
		self.g_train = self.dataset.getSparseGraph()

		# Gating Net with Gumbel-Softmax
		self.gating_network_list = []
		for i in range(2):
			self.gating_network_list.append(Gating_Net(embedding_dim=self.embedding_dim, mlp_dims=self.config['gating_mlp_dims']).to(device))

	def __choosing_one(self, features, gumbel_out):
		feature = torch.sum(torch.mul(features, gumbel_out), dim=1)  # batch x embedding_dim (or batch x embedding_dim x layer_num)
		return feature

	def __dropout_x(self, x, keep_prob):
		size = x.size()
		index = x.indices().t()
		values = x.values()
		random_index = torch.rand(len(values)) + keep_prob
		random_index = random_index.int().bool()
		index = index[random_index]
		values = values[random_index]/keep_prob
		g = torch.sparse.FloatTensor(index.t(), values, size)
		return g

	def __dropout(self, keep_prob):
		if self.A_split:   
			graph = []
			for g in self.Graph:
				graph.append(self.__dropout_x(g, keep_prob))
		else:
			graph = self.__dropout_x(self.Graph, keep_prob)
		return graph

	def computer(self, gum_temp, hard):     
		
		self.Graph = self.g_train   
		if self.dropout:
			if self.training:
				g_droped = self.__dropout(self.keep_prob)
			else:
				g_droped = self.Graph        
		else:
			g_droped = self.Graph
    
    
		# Init users & items embeddings  
		users_emb = self.embedding_user.weight
		items_emb = self.embedding_item.weight
      
      
		## Layer 0
		all_emb_0 = torch.cat([users_emb, items_emb])
		
		# Residual embeddings
		embs = [all_emb_0]
		
   
		## Layer 1
		all_emb_lin_1 = torch.sparse.mm(g_droped, all_emb_0)
		
		# Residual embeddings	
		embs.append(all_emb_lin_1)
		
   
		## layer 2
		all_emb_lin_2 = torch.sparse.mm(g_droped, all_emb_lin_1)
		
		# Residual embeddings
		embs.append(all_emb_lin_2)
		
   
		## layer 3
		all_emb_lin_3 = torch.sparse.mm(g_droped, all_emb_lin_2)
		all_emb_non_1 = self.activation_function(torch.sparse.mm(g_droped, all_emb_0))
		
		# Gating
		stack_embedding_1 = torch.stack([all_emb_lin_3, all_emb_non_1],dim=1)
		concat_embeddings_1 = torch.cat((all_emb_lin_3, all_emb_non_1),-1)

		gumbel_out_1, lin_count_3, non_count_3 = self.gating_network_list[0](concat_embeddings_1, gum_temp, hard, self.config['division_noise'])
		embedding_1 = self.__choosing_one(stack_embedding_1, gumbel_out_1)

		# Residual embeddings
		embs.append(embedding_1)
	
  	
		# layer 4
		all_emb_lin_4 = torch.sparse.mm(g_droped, embedding_1)
		all_emb_non_2 = self.activation_function(torch.sparse.mm(g_droped, embedding_1))
		
		# Gating
		stack_embedding_2 = torch.stack([all_emb_lin_4, all_emb_non_2],dim=1)
		concat_embeddings_2 = torch.cat((all_emb_lin_4, all_emb_non_2),-1)

		gumbel_out_2, lin_count_4, non_count_4 = self.gating_network_list[1](concat_embeddings_2, gum_temp, hard, self.config['division_noise'])
		embedding_2 = self.__choosing_one(stack_embedding_2, gumbel_out_2)

		# Residual embeddings  		
		embs.append(embedding_2)


		## Stack & mean residual embeddings
		embs = torch.stack(embs, dim=1)
		light_out = torch.mean(embs, dim=1)
   
		users, items = torch.split(light_out, [self.num_users, self.num_items])
		
		return users, items, [lin_count_3, non_count_3, lin_count_4, non_count_4], embs

	def getUsersRating(self, users, gum_temp, hard):
		all_users, all_items, gating_dist, embs = self.computer(gum_temp, hard)
		
		users_emb = all_users[users.long()]
		items_emb = all_items

		rating = self.activation_function(torch.matmul(users_emb, items_emb.t()))

		return rating, gating_dist, embs

	def getEmbedding(self, users, pos_items, neg_items, gum_temp, hard):
		all_users, all_items, gating_dist, embs = self.computer(gum_temp, hard)
		
		users_emb = all_users[users]
		pos_emb = all_items[pos_items]
		neg_emb = all_items[neg_items]

		users_emb_ego = self.embedding_user(users)
		pos_emb_ego = self.embedding_item(pos_items)
		neg_emb_ego = self.embedding_item(neg_items)

		return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego, gating_dist, embs

	def bpr_loss(self, users, pos, neg, gum_temp, hard):
		(users_emb, pos_emb, neg_emb, 
		userEmb0,  posEmb0, negEmb0, gating_dist, embs) = self.getEmbedding(users.long(), pos.long(), neg.long(), gum_temp, hard)
		
		reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
							posEmb0.norm(2).pow(2)  +
							negEmb0.norm(2).pow(2))/float(len(users))
		
		pos_scores = torch.mul(users_emb, pos_emb)
		pos_scores = torch.sum(pos_scores, dim=1)
		neg_scores = torch.mul(users_emb, neg_emb)
		neg_scores = torch.sum(neg_scores, dim=1)
		
		loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
		
		return loss, reg_loss, gating_dist, embs
		
	def forward(self, users, items, gum_temp, hard):
		# compute embedding
		all_users, all_items, gating_dist, embs = self.computer(gum_temp, hard)

		users_emb = all_users[users]
		items_emb = all_items[items]

		inner_pro = torch.mul(users_emb, items_emb)
		gamma     = torch.sum(inner_pro, dim=1)

		return gamma, gating_dist, embs
```

```python id="pNkhPss9xvLC"
MODELS = {
  "HMLET_End": HMLET_End
}
```

<!-- #region id="a2HIfFRR7fP1" -->
## Procedures
<!-- #endregion -->

```python id="HuIhwj6g7hRJ"
def BPR_train_original(dataset, recommend_model, loss_class, epoch, gum_temp, hard, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: BPRLoss = loss_class
    
    with timer(name="Sample"):
        S = UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(device)
    posItems = posItems.to(device)
    negItems = negItems.to(device)
    users, posItems, negItems = shuffle(users, posItems, negItems)
    total_batch = len(users) // bpr_batch_size + 1
    aver_loss = 0.
    
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(tqdm(minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=bpr_batch_size), total=396)):
        cri, gating_dist, embs = bpr.stageOne(batch_users, batch_pos, batch_neg, gum_temp, hard)
        aver_loss += cri
        if tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / bpr_batch_size) + batch_i)
    aver_loss = aver_loss / total_batch

    return f"loss{aver_loss:.3f}"
    
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in topks:
        ret = RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}
        
def Test(dataset, Recmodel, epoch, gum_temp, hard, mode, w=None, multicore=0):

    u_batch_size = test_u_batch_size
    dataset: BasicDataset
    
    # Mode
    if mode == 'valid':
      print('valid mode')
      testDict: dict = dataset.validDict
      excel_results = excel_results_valid
    elif mode == 'test':
      print('test mode')
      testDict: dict = dataset.testDict
      excel_results = excel_results_test
    
    Recmodel = Recmodel.eval()
    max_K = max(topks)
    
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    
    # Results
    results = {'precision': np.zeros(len(topks)),
               'recall': np.zeros(len(topks)),
               'ndcg': np.zeros(len(topks))}
               
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
            
        users_list = []
        rating_list = []
        groundTrue_list = []
        #gating_dist_list = []
        #embs_list = []
        
        total_batch = len(users) // u_batch_size + 1
        
        for batch_users in minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(device)

            rating, gating_dist, embs = Recmodel.getUsersRating(batch_users_gpu, gum_temp, hard)
            #gating_dist_list.append(gating_dist)
            #embs_list.append(embs)
            
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()

            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
            
        assert total_batch == len(users_list)
        
        X = zip(rating_list, groundTrue_list)
        
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
                
        scale = float(u_batch_size/len(users))
        
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
            
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        
        if tensorboard:
            w.add_scalars(f'Test/Recall@{topks}',
                          {str(topks[i]): results['recall'][i] for i in range(len(topks))}, epoch)
            w.add_scalars(f'Test/Precision@{topks}',
                          {str(topks[i]): results['precision'][i] for i in range(len(topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{topks}',
                          {str(topks[i]): results['ndcg'][i] for i in range(len(topks))}, epoch)
        
        if multicore == 1:
            pool.close()

        excel_results['Model'].append(model_name)
        excel_results['Dataset'].append(dataset)
        excel_results['Epochs'].append(epoch)
        excel_results['Precision'].append(results['precision'])
        excel_results['Recall(HR)'].append(results['recall'])
        excel_results['Ndcg'].append(results['ndcg'])
          
        excel_data = pd.DataFrame(excel_results)
            
        print(results)
        return results, excel_data
```

```python colab={"base_uri": "https://localhost:8080/", "height": 839} id="tjq8a9oLaQoy" executionInfo={"status": "ok", "timestamp": 1637236508340, "user_tz": -330, "elapsed": 442, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2df3061a-7759-4861-f514-3f39589971a3"
%reload_ext tensorboard
%tensorboard --logdir tensorboard
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["912c1c27df2645d39444fb31d580b6e9", "8345b2752ac848d8bd2eeb5fa87452db", "14fca280eba6402494cdc2bd805bad85", "9a5c8c1781ae4affb72f4aed8ecbaa9b", "95a586c260424696957c8428ef46f81a", "e26fbc49e7c3437b8d6a2946ffa446d5", "a36ec965a3df4d16a950869c120a4120", "34f38bddc8d4465da785b875bb4178fa", "8bf54d0dfabd4aea849ce964cfa89a28", "19758a2fd7d54c939132c08eaba49427", "b51a8eab1ea84735b435a6f786ed2008", "d5c0fc6dcfba42ff91ec853655866888", "a070050fb1cd45b99fcc10b05f280f86", "7c4f106b503d41e5adc491fab41e2689", "1d7b32e3299c4f069fcef39433f381d1", "b034587a2c0741b4ba7cb9275986d767", "8c6bf77ea39d48138f5b57a9188cdbe8", "92689a3815bc4ff5972c792d5ae927ce", "45ecfbe2d4194683962fdea8840ec448", "ca3519631d5a4a82a11c7692e3f1cd04", "a56ccd7e2a6b45839ae5847e24aa611f", "bfbc8a93e6cd436abf843b0965b63a93", "0ef4fced1ee84b2eabea4d7f5bf9e40f", "bfcf80694dee450d81b6b593a1a83a94", "ecdc27d9ad454b43990aa87333843b6a", "271897dd6cb6449e8d1d6d8354d6327a", "ed234409a0864d589f727482c7c1b89d", "b10e93e426b64d0ba208fec34851f30c", "18a56d0cef044affbf5315e983d46977", "90f18d27ec964b11b2ed0dc6cdbc072b", "e97daa5463fd4da59442e17ffc02c27a", "c18d6b1b980342669d25a89951aa25db", "b39dcd7c2add4224980b51b662a4e994", "7e323e37c96145f3a6c95eab137b15c0", "b6d78f8875ff40b08341e66bf3c0ddc0", "2dc4426c29ad4a6da8ad32028b715fa5", "c51ebb88de2a4767a1c90019dd701c68", "b71bff74949343e992d23878966be450", "1cf296ab5a564ab4b4d2d8abcbff9310", "c6ba6b3191a54b0fae0c785655402aed", "aef511f34d534675be13090e4cb2c484", "5138422f413c4acf8cd9cb763fff7b41", "6888aa3b35e14033848e8c63d1edd740", "2c1e50c7114f4a6697f6eddddcdc9146"]} id="3Xd6J7Cd8Af9" executionInfo={"status": "ok", "timestamp": 1637234734790, "user_tz": -330, "elapsed": 3829956, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4a3c033d-d150-4b1c-b7a4-1552e94e6d79"
set_seed(SEED)
dataset = Loader(DATA_PATH)
model = MODELS[model_name](config, dataset)
model = model.to(device)
bpr = BPRLoss(model)

# Pretrain
if pretrain:
    try:
        pretrained_file = LOAD_FILE_PATH
        model.load_state_dict(torch.load(pretrained_file))
        print(f"loaded model weights from {pretrained_file}")
    except FileNotFoundError:
        print(f"{pretrained_file} not exists, start from beginning")

# Tensorboard
if tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(BOARD_PATH, tm.strftime("%m-%d-%Hh%Mm%Ss-"))
                                    )
else:
    w = None
    print("not enable tensorflowboard")

try:
    start_epoch = load_epoch
    gum_temp = ori_temp
    for epoch in range(start_epoch, EPOCHS+1):
        start = tm.time()
        
        print('Train', epoch, '='*30)
        print('gum_temp:', gum_temp)
        output_information = BPR_train_original(dataset, model, bpr, EPOCHS, gum_temp, hard=train_hard, w=w)
        print(f'EPOCH[{epoch}/{EPOCHS}] {output_information}')
        
        end = tm.time()
        print('train time:', end-start)
        
        if epoch % epoch_temp_decay == 0:
            # Temp decay
            gum_temp = ori_temp * math.exp(-gum_temp_decay*epoch)
            gum_temp = max(gum_temp, min_temp)
            print('decay gum_temp:', gum_temp)
        
        # if epoch % 10 == 0:
        if epoch % 1 == 0:
            print("model save...")
            torch.save(model.state_dict(), SAVE_FILE_PATH+'/'+str(model_name)+'_'+str(ori_temp)+'_'+str(gum_temp_decay)+'_'+str(min_temp)+'_'+str(epoch_temp_decay)+'_'+str(config['division_noise'])+'_'+str(epoch)+".pth.tar")
            
            print('Valid', '='*50)
            valid_results, valid_excel_data = Test(dataset, model, epoch, gum_temp, hard=test_hard, mode='valid', w=w, multicore=multicore)

            xlxs_dir = EXCEL_PATH + '/valid_'+str(model_name)+'_'+str(config['embedding_dim'])+'_'+str(ori_temp)+'_'+str(gum_temp_decay)+'_'+str(min_temp)+'_'+str(epoch_temp_decay)+'_'+str(config['division_noise'])+'_'+str(config['dropout'])+'_'+str(config['keep_prob'])+'_'+str(topks)+'.xlsx'
        
            with pd.ExcelWriter(xlxs_dir) as writer:
                valid_excel_data.to_excel(writer, sheet_name = 'result')            
            
            print('Test', '='*50)
            test_results, test_excel_data = Test(dataset, model, epoch, gum_temp, hard=test_hard, mode='test', w=w, multicore=multicore)
            
            xlxs_dir = EXCEL_PATH + '/test_'+str(model_name)+'_'+str(config['embedding_dim'])+'_'+str(ori_temp)+'_'+str(gum_temp_decay)+'_'+str(min_temp)+'_'+str(epoch_temp_decay)+'_'+str(config['division_noise'])+'_'+str(config['dropout'])+'_'+str(config['keep_prob'])+'_'+str(topks)+'.xlsx'
        
            with pd.ExcelWriter(xlxs_dir) as writer:
                test_excel_data.to_excel(writer, sheet_name = 'result')
            
finally:
    if tensorboard:
        w.close()
```

```python id="DenNZlSA8uT1" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637236296468, "user_tz": -330, "elapsed": 5939, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="68bed208-5905-4572-da22-397d6292dca6"
!apt-get -qq install tree
```

```python colab={"base_uri": "https://localhost:8080/"} id="Hvjv05JE4zEf" executionInfo={"status": "ok", "timestamp": 1637236363050, "user_tz": -330, "elapsed": 431, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d5e3bb9b-531d-4aef-c653-65d01502f337"
!tree --du -h -L 3 .
```

```python id="JF6Z1Z2F415w"

```
