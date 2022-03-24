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

```python id="2mKh3mM6wllC"
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--bpr_batch', type=int,default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int,default=2,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int,default=0,
                        help="using the dropout or not")
    parser.add_argument('--keep_prob', type=float,default=0.9,
                        help="the batch size for bpr loss training procedure")
    # parser.add_argument('--maskfea', type=float,default=0.0,
    #                     help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='yelp2018',
                        help="available datasets: [gowalla, yelp2018, amazon-book]")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[20]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int,default=1,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str,default="lgn")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--epochs', type=int,default=100)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='lgn', help='rec-model, support [mf, lgn]')
    parser.add_argument('--info', type=str, default='' )
    parser.add_argument('--temperature', type=float, default=0.1, help='temperature')
    parser.add_argument('--methods', type=str,default='LightGCN', )
    return parser.parse_args(args={})
```

```python id="sZsUudt8v9SX"
import os
from os.path import join
import torch
from enum import Enum
import multiprocessing

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

ROOT_PATH = "./LightGCNMultiCL/"
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs3')
FILE_PATH = join(CODE_PATH, 'checkpoints')
import sys
sys.path.append(join(CODE_PATH, 'sources'))


if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)


config = {}
all_dataset = ['gowalla', 'yelp2018', 'amazon-book', 'pinterest', 'steam','ifashion']
all_models  = ['mf', 'lgn']
# config['batch_size'] = 4096
config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers']= args.layer
config['dropout'] = args.dropout  
config['keep_prob']=args.keep_prob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['pretrain'] = args.pretrain
config['A_split'] = False
config['bigdata'] = False
config['dataset'] = args.dataset
config['info'] = args.info
config['temperature'] = args.temperature
config['methods'] = args.methods

GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2
seed = args.seed

dataset = args.dataset
model_name = args.model
if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")


TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
topks = eval(args.topks)
tensorboard = args.tensorboard
comment = args.comment
# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)

def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")
```

```python id="rbInFFI9w6u-"
import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from time import time
```

```python id="aXem_nB9xM-S"
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

class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    def __init__(self,config = config,path="."):
        # train or test
        cprint(f'loading [{path}]')
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0
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

        with open(test_file) as f:
            for l in f.readlines():
                l=l.rstrip() 
                l = l.strip('\n').split(' ')
                if len(l) > 1:
                    
                    items = [int(i) for i in l[1:]]
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
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        self.Gui=None
        self.uu=True
        self.ii=True
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
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()

                #adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
                
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(device)
                print("don't split the matrix")
        return self.Graph

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
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
```

```python id="XYFBBLgoxd3M"
import torch
from torch import nn
import numpy as np

from torch.nn import Module
import torch.nn.functional as F

class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        

    def bpr_loss_gcl_Kpos(self,users_emb0,pos_emb0,users,pos_items,alpha):
       # users_emb0,pos_emb0 =  self.computer(nd,fd)   #几次？

        users_emb=   users_emb0[users] 
        loss=self.bpr_loss_gcl_unit(users_emb,pos_emb0[pos_items[:,0]],alpha)
        for i in range(pos_items.size()[1])[1:]:
            pos_emb=   pos_emb0[pos_items[:,i]]  
            loss=loss+self.bpr_loss_gcl_unit(users_emb,pos_emb,alpha)
        
       # losii=self.iiloss(pos_emb0,pos_items)
               
        return  loss.mean() ,0
  
    def bpr_loss_gcl_unit(self,users_emb,pos_emb,alpha): 
        T=self.T 
        sim_batch=torch.exp(torch.mm(users_emb,pos_emb.t() ) /T )  
        posself=sim_batch.diag() 
        neg=  sim_batch.sum(dim=1)  
       # lossRS=-torch.log(( posself+0.00001) /(neg+0.00001) ).mean()

        lossRS= -alpha*torch.log( posself+0.00001)+(1-alpha)*torch.log(neg+0.00001) 
        lossRS=lossRS.mean()
 
        return  lossRS 

    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)
 


class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    # def getSparseEye(self,num):
    #     i = torch.LongTensor([[k for k in range(0,num)],[j for j in range(0,num)]])
    #     val = torch.FloatTensor([1]*num)
    #     return torch.sparse.FloatTensor(i,val)
    


    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.T = self.config['temperature']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings = self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings = self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        self.alphapara = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alphapara.data.fill_(0.5)
       
         # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]#/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
        

    def drop_feature(self,x, drop_prob):
        drop_mask = torch.empty(
            (x.size(1), ),
            dtype=torch.float32,
            device=x.device).uniform_(0, 1) < drop_prob
        x = x.clone()
        x[:, drop_mask] = 0 
        return x


    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
     

    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
 
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]

        if self.config['dropout']:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph     
        else:
            g_droped = self.Graph 

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)

        light_out=all_emb

                # #mean
                # embs = torch.stack(embs, dim=1) 
                # light_out = torch.mean(embs, dim=1)
                #sg
        light_out= embs[-1] 
        users, items = torch.split(light_out, [self.num_users, self.num_items])

        return users, items 


    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
   
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss

#fei qita zhegnli
############################################################################################################################################
    def bpr_loss_gcl_Kpos(self,users_emb0,pos_emb0,users,pos_items,alpha):

        alpha=self.alphapara
        users_emb=   users_emb0[users] 
        loss=self.bpr_loss_gcl_unit(users_emb,pos_emb0[pos_items[:,0]],alpha)
        for i in range(pos_items.size()[1])[1:]:
            pos_emb=   pos_emb0[pos_items[:,i]]  
            loss=loss+self.bpr_loss_gcl_unit(users_emb,pos_emb,alpha)

        return  loss.mean() 
    
 
 
 
    def bpr_loss_gcl_unit(self,users_emb,pos_emb,alpha): 
        T=self.T 
 
        ##ori
        sim_batch=torch.exp(torch.mm(users_emb,pos_emb.t() ) /T )  
 
        posself=sim_batch.diag() 
        neg=  sim_batch.sum(dim=1)    ########################################
        
        lossRS= -alpha*torch.log( posself+0.00001)+(1-alpha)*torch.log(neg+0.00001) 
        lossRS=lossRS.mean()
 
        return  lossRS 
 
####################################################################################
#
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma
```

```python colab={"base_uri": "https://localhost:8080/"} id="t0jJwvYJyQx7" executionInfo={"status": "ok", "timestamp": 1639464776731, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0e8aeaed-3400-453c-ae3a-ebdace0fa9a0"
import torch
from torch import nn, optim
import numpy as np
from torch import log
from time import time
from sklearn.metrics import roc_auc_score
import random
import os

import torch.nn.functional as F

try:
    from cppimport import imp_from_filepath
    from os.path import join, dirname
    path = join(dirname(__file__), "sources/sampling.cpp")
    sampling = imp_from_filepath(path)
    sampling.seed(seed)
    sample_ext = True
    cprint("Cpp extension loaded")
except:
    cprint("Cpp extension not loaded !")
    sample_ext = False


class BPRLoss:
    def __init__(self,
                 recmodel : PairWiseModel,
                 config : dict):
        self.model = recmodel
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.T=config['temperature']
        # self.dropedge = config['dropedge']        
        # self.maskfea = config['maskfea']

    def stageOne(self, users, pos,info="",epoch=5 ): 
        
        if True:
            users_emb0,pos_emb0 =  self.model.computer() 
            users_emb0=F.normalize(users_emb0)
            pos_emb0=F.normalize(pos_emb0)
            


            if "alpha" in info:
                alpha=float(info.split("alpha")[1])#.split("_")[0])
            else:
                alpha=0.5  
         
            loss = self.model.bpr_loss_gcl_Kpos(users_emb0,pos_emb0,users, pos,alpha)# bpr_loss_gcl_Kpos
                
            
            
        self.opt.zero_grad()
        loss.backward()
        self.opt.step() 
        return loss.cpu().item() 
 

def UniformSample_original(dataset, neg_ratio = 1):
    dataset : BasicDataset
    allPos = dataset.allPos
    start = time()
    #S = UniformSample_original_python(dataset,neg_ratio)

    if sample_ext:
        S = sampling.sample_negative(dataset.n_users, dataset.m_items,
                                     dataset.trainDataSize, allPos, neg_ratio)
    else:
        S = UniformSample_original_python(dataset)
    return S

def UniformSample_original_python(dataset,neg_ratio=10):
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
        posindex = np.random.randint(0, len(posForUser), size=neg_ratio)
        positem = posForUser[posindex]
        # while True:
        #     negitem = np.random.randint(0, dataset.m_items)
        #     if negitem in posForUser:
        #         continue
        #     else:
        #         break
        S.append([user, positem])
        end = time()
        sample_time1 += end - start
    total = time() - total_start

    return np.array(S)

# ===================end samplers==========================
# =====================utils====================================

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def getFileName():
    if model_name == 'mf':
        file = f"mf-{dataset}-{config['latent_dim_rec']}.pth.tar"
    elif model_name == 'lgn':
        file = f"lgn-{dataset}-{config['lightGCN_n_layers']}-{config['latent_dim_rec']}.pth.tar"
    return os.path.join(FILE_PATH,file)

def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', config['bpr_batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


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


# ====================Metrics==============================
# =========================================================
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

```python colab={"base_uri": "https://localhost:8080/"} id="H7GlVt5ByzMv" executionInfo={"status": "ok", "timestamp": 1639464779617, "user_tz": -330, "elapsed": 2892, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="72b276cd-8b0c-4f64-b746-1738ed95f0df"
!wget -q --show-progress https://github.com/RecoHut-Datasets/yelp/raw/v2/s_pre_adj_mat.npz
!wget -q --show-progress https://github.com/RecoHut-Datasets/yelp/raw/v2/train.txt
!wget -q --show-progress https://github.com/RecoHut-Datasets/yelp/raw/v2/test.txt
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="IJhhlXtYulph" executionInfo={"status": "ok", "timestamp": 1639465031494, "user_tz": -330, "elapsed": 26992, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e519cfa6-f826-41b5-d6ef-dad3fef2d7ee"
!pip install -r ../requirements.txt
```

```python colab={"base_uri": "https://localhost:8080/"} id="CkjipO8Ot5DN" executionInfo={"status": "ok", "timestamp": 1639465138769, "user_tz": -330, "elapsed": 19151, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="cf055f74-246f-4a6d-9b14-1643738ab610"
# !git clone https://github.com/haotangxjtu/MSCL.git
%cd MSCL/code
!python main.py --layer 2 --dataset="yelp2018" --temperature 0.2 --info sgk15_alpha0.45
```

```python colab={"base_uri": "https://localhost:8080/"} id="HwieAgTUxsoy" executionInfo={"status": "ok", "timestamp": 1639464787230, "user_tz": -330, "elapsed": 7623, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="aa876d6b-c535-4735-87eb-ce91d17a1e29"
from pprint import pprint

dataset = Loader(path='.')

print('===========config================')
pprint(config)
print("cores for test:", CORES)
print("comment:", comment)
print("tensorboard:", tensorboard)
print("LOAD:", LOAD)
print("Weight path:", PATH)
print("Test Topks:", topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'mf': PureMF,
    'lgn': LightGCN
}
```

```python id="dFJns77Qyo77"
import numpy as np
import torch
from pprint import pprint
from time import time
from tqdm import tqdm
import multiprocessing
from sklearn.metrics import roc_auc_score
 

CORES = multiprocessing.cpu_count() // 2


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: BPRLoss = loss_class
    
    # u i+ i+  
    with timer(name="Sample"):
        kk=config["info"].split("k")[1]
        if "_" in kk:
            kk=int(kk.split("_")[0] )
        else:
            kk=int(kk)

        S = UniformSample_original(dataset,kk)
     
    S=S.astype('int32') 
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1:]).long() 
  
    users = users.to(device)
    posItems = posItems.to(device)
   
  
    users, posItems = shuffle(users, posItems)
    total_batch = len(users) // config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos, )) in enumerate(minibatch(users,
                                                   posItems,
                                                   batch_size=config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos,config["info"],epoch)
        aver_loss += cri
        if tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"
    
    
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
        
            
def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = config['test_u_batch_size']
    dataset: BasicDataset
    testDict: dict = dataset.testDict
    Recmodel: LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
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
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [ 
            #         AUC(rating[i],
            #                   dataset, 
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
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
        # results['auc'] = np.mean(auc_record)
        if tensorboard:
            w.add_scalars(f'Test/Recall@{topks}',
                          {str(topks[i]): results['recall'][i] for i in range(len(topks))}, epoch)
            w.add_scalars(f'Test/Precision@{topks}',
                          {str(topks[i]): results['precision'][i] for i in range(len(topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{topks}',
                          {str(topks[i]): results['ndcg'][i] for i in range(len(topks))}, epoch)
        if multicore == 1:
            pool.close()
        print(results)
        return results
```

```python colab={"base_uri": "https://localhost:8080/", "height": 495} id="lghPWvaczUm1" executionInfo={"status": "error", "timestamp": 1639464792311, "user_tz": -330, "elapsed": 5089, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="22a97172-cbe2-4244-f46c-3ac5be9f213f"
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
from os.path import join
import os
# ==============================
set_seed(seed)
print(">>SEED:", seed)
# ==============================

Recmodel = MODELS[model_name](config, dataset)
Recmodel = Recmodel.to(device)
bpr = BPRLoss(Recmodel, config)

weight_file = getFileName()
print(f"load and save to {weight_file}")
if LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

def early_stopping(log_value, best_value, stopping_step, expected_order='recall', flag_step=100):
    # early stopping strategy:
    #assert expected_order in ['recall', 'ndcg']

    if (expected_order == 'recall' and log_value >= best_value)   :
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop

 
#####
stopping_step = 0
should_stop = False
cur_best_pre_0=0.0
# init tensorboard
if tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + comment)
                                    )
else:
    w = None
    cprint("not enable tensorflowboard")

save_path = join(BOARD_PATH,config['dataset'] +config['methods'] + config['info']+"-T" + str(config['temperature'])+"L"+str(config['lightGCN_n_layers'])+time.strftime("%m-%d-%Hh%Mm%Ss-") +".txt") 

d = os.path.dirname(save_path)
if not os.path.exists(d):
    os.makedirs(d) 
lossinfo=[]
epochtime=[]
try:
    for epoch in range(TRAIN_epochs):
        f = open(save_path, 'a') 
        start = time.time()

        if epoch ==0  : 
            f.write("Info :   ")
            for k,v in config.items():
                f.write( str(k)+","+str(v)+"\n")
            print("start")  


 
            
        output_information = BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        epochtime.append(time.time()-start)
       # print(f'EPOCH[{epoch+1}/{TRAIN_epochs}] {output_information}',config['dataset'] + config['info']+str(np.mean(epochtime)))
        
        if epoch > 0:
            cprint("[TEST]")
            outs=Test(dataset, Recmodel, epoch, w, config['multicore'])

            f.write("\n"+str(epoch)+":   ")
            for k,v in outs.items():
                f.write( str(k)+","+str(v))  
            cur_best_pre_0, stopping_step, should_stop = early_stopping(outs['recall'][0], cur_best_pre_0,
                                                                                stopping_step, expected_order='recall', flag_step=20)


        lossinfo.append(output_information)
        torch.save(Recmodel.state_dict(), weight_file)
        if epoch == range(TRAIN_epochs)[-1]:
            f.write( 'finish ,best recall '+str(cur_best_pre_0) )
            f.close()  

        if should_stop == True:
            f.write( 'early stop ,best recall '+str(cur_best_pre_0)+str(epoch) )
            f.close()
            break        
 
    f = open(save_path, 'a')
    for i, val in enumerate(lossinfo):
        f.write("\n"+ str(i+1)+","+str(val)) 
    f.close()

    print(f'last-time',config['dataset'] + config['info']+str(np.mean(epochtime)))

finally: 
    if tensorboard:
        w.close()
```

```python id="Hp6-GM--zrOF"

```
