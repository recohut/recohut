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

<!-- #region id="IMsv845dXuFO" -->
# Sign-Aware Recommendation Using Graph Neural Networks on Yelp Dataset in PyTorch
<!-- #endregion -->

<!-- #region id="dfEu7_3GMP6Z" -->
## Setup
<!-- #endregion -->

```python id="3X8DNyvx6-06"
!pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
!pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
!pip install torch-geometric
```

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="-8cfttRjGXjg" executionInfo={"status": "ok", "timestamp": 1637761110365, "user_tz": -330, "elapsed": 5264, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="cfef03c8-4508-494c-fd9c-2f80aa77faa3"
import torch
torch.__version__
```

```python id="EaxoGLQO7D5S"
from torch import nn
from torch import Tensor
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

import pandas as pd
import numpy as np
import time
from tqdm.notebook import tqdm

import os
import pickle
```

```python id="jTHP9X3uLtwe"
import warnings
warnings.filterwarnings('ignore')
```

```python colab={"base_uri": "https://localhost:8080/"} id="FkWDKmmUBFJW" executionInfo={"status": "ok", "timestamp": 1637763356486, "user_tz": -330, "elapsed": 474, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="75a5ecb0-e822-4804-b883-88d3d09de39f"
class Args:
    dataset = 'yelp' # Dataset
    version = 1 # Dataset version
    batch_size = 1024 # Batch size
    dim = 64 # Dimension
    lr = 5e-3 # Learning rate
    offset = 3.5 # Criterion of likes/dislikes
    K = 40 # The number of negative samples
    num_layers = 4 # The number of layers of a GNN model for the graph with positive edges
    MLP_layers = 2 # The number of layers of MLP for the graph with negative edges
    epoch = 4 # The number of epochs
    reg = 0.05 # Regularization coefficient

args = Args()
Args.__dict__
```

<!-- #region id="qpVGR7NVMUa-" -->
## Dataset
<!-- #endregion -->

```python id="eKhodi5h-aNm"
!git clone -q https://github.com/RecoHut-Datasets/yelp.git
```

```python id="xw2tVzO2-HcI"
class Data_loader():
    def __init__(self,dataset,version):
        self.path_for_whole='/content/yelp/bronze/v1/YELP_encoded.csv'
        self.path_for_train='/content/yelp/bronze/v1/train_yelp%s.dat'%(version)
        self.path_for_test='/content/yelp/bronze/v1/test_yelp%s.dat'%(version)
        self.num_u=41772; self.num_v=30037;
        
    def data_load(self):
        self.whole_=pd.read_csv(self.path_for_whole,index_col=0).sample(frac=1,replace=False);
        self.train_set=pd.read_csv(self.path_for_train,index_col=0)
        self.test_set=pd.read_csv(self.path_for_test,index_col=0)          
        return self.train_set, self.test_set
```

```python id="K26T4NSC-HZl"
class bipartite_dataset(Dataset): 
    def __init__(self, train,neg_dist,offset,num_u,num_v,K): 
        self.edge_1 = torch.tensor(train['userId'].values-1)
        self.edge_2 = torch.tensor(train['movieId'].values-1) +num_u
        self.edge_3 = torch.tensor(train['rating'].values) - offset
        self.neg_dist = neg_dist
        self.K = K;
        self.num_u = num_u
        self.num_v = num_v
        self.tot = np.arange(num_v)
        self.train = train
        
    def negs_gen_(self):
        print('negative sampling...'); st=time.time()
        self.edge_4 = torch.empty((len(self.edge_1),self.K),dtype=torch.long)
        prog = tqdm(desc='negative sampling for each epoch...',total=len(set(self.train['userId'].values)),position=0)
        for j in set(self.train['userId'].values):
            pos=self.train[self.train['userId']==j]['movieId'].values-1
            neg = np.setdiff1d(self.tot,pos)
            temp = (torch.tensor(np.random.choice(neg,len(pos)*self.K,replace=True,p=self.neg_dist[neg]/self.neg_dist[neg].sum()))+self.num_u).long()
            self.edge_4[self.edge_1==j-1]=temp.view(int(len(temp)/self.K),self.K)
            prog.update(1)
        prog.close()
        self.edge_4 = torch.tensor(self.edge_4).long()
        print('complete ! %s'%(time.time()-st))
        
    def negs_gen_EP(self,epoch):
        print('negative sampling for next epochs...'); st=time.time()
        self.edge_4_tot = torch.empty((len(self.edge_1),self.K,epoch),dtype=torch.long)
        prog = tqdm(desc='negative sampling for next epochs...',total=len(set(self.train['userId'].values)),position=0)
        for j in set(self.train['userId'].values):
            pos=self.train[self.train['userId']==j]['movieId'].values-1
            neg = np.setdiff1d(self.tot,pos)
            temp = (torch.tensor(np.random.choice(neg,len(pos)*self.K*epoch,replace=True,p=self.neg_dist[neg]/self.neg_dist[neg].sum()))+self.num_u).long()
            self.edge_4_tot[self.edge_1==j-1]=temp.view(int(len(temp)/self.K/epoch),self.K,epoch)
            prog.update(1)
        prog.close()
        self.edge_4_tot = torch.tensor(self.edge_4_tot).long()
        print('complete ! %s'%(time.time()-st))

    def __len__(self):
        return len(self.edge_1)

    def __getitem__(self,idx):
        u = self.edge_1[idx]
        v = self.edge_2[idx]
        w = self.edge_3[idx]
        negs = self.edge_4[idx]
        return u,v,w,negs
```

```python id="iuji2aOB-HXH"
def deg_dist(train, num_v):
    uni, cou = np.unique(train['movieId'].values-1,return_counts=True)
    cou = cou**(0.75)
    deg = np.zeros(num_v)
    deg[uni] = cou
    return torch.tensor(deg)
```

<!-- #region id="9bXdW8gMMXF-" -->
## Modules
<!-- #endregion -->

```python id="ylnEwRr5AcDM"
def gen_top_K(data_class,emb,train,directory_):
    no_items = np.array(list(set(np.arange(1,data_class.num_v+1))-set(train['movieId'])))
    total_users = set(np.arange(1,data_class.num_u+1))
    reco = dict()
    pbar = tqdm(desc = 'top-k recommendation...',total=len(total_users),position=0)
    for j in total_users:
        pos = train[train['userId']==j]['movieId'].values-1
        embedding_ = emb[j-1].view(1,len(emb[0])).mm(emb[data_class.num_u:].t()).detach();
        embedding_[0][no_items-1]=-np.inf;
        embedding_[0][pos]=-np.inf;
        reco[j]=torch.topk(embedding_[0],300).indices.cpu().numpy()+1
        pbar.update(1)
    pbar.close()
    return reco
```

```python id="wSZobaqg77bu"
class LightGConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')
        
    def forward(self,x,edge_index):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, norm=norm)
    
    def message(self,x_j,norm):
        return norm.view(-1,1) * x_j
        
    def update(self,inputs: Tensor) -> Tensor:
        return inputs
```

```python id="GQ8cVF6V9Xx8"
class LRGCCF(MessagePassing):
    def __init__(self, in_channels,out_channels):
        super(LRGCCF,self).__init__(aggr='mean')
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self,x,edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0));
        return self.lin(self.propagate(edge_index,x=x))

    def message(self,x_j):
        return x_j
        
    def update(self,inputs: Tensor) -> Tensor:
        return inputs
```

<!-- #region id="P5g2bFwbMeOh" -->
## Model
<!-- #endregion -->

```python id="4sKS8Pd2_Pq3"
class SiReN(nn.Module):
    def __init__(self,train,num_u,num_v,offset,num_layers = 2,MLP_layers=2,dim = 64,reg=1e-4
                 ,device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super(SiReN,self).__init__()
        self.M = num_u; self.N = num_v;
        self.num_layers = num_layers
        self.MLP_layers = MLP_layers
        self.device = device
        self.reg = reg
        self.embed_dim = dim
        edge_user = torch.tensor(train[train['rating']>offset]['userId'].values-1)
        edge_item = torch.tensor(train[train['rating']>offset]['movieId'].values-1)+self.M
        edge_ = torch.stack((torch.cat((edge_user,edge_item),0),torch.cat((edge_item,edge_user),0)),0)
        self.data_p=Data(edge_index=edge_)
        # For the graph with positive edges
        self.E = nn.Parameter(torch.empty(self.M + self.N, dim))
        nn.init.xavier_normal_(self.E.data)
        self.convs = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for _ in range(num_layers):
            # self.convs.append(LRGCCF(dim,dim)) 
            self.convs.append(LightGConv()) 
        # For the graph with negative edges
        self.E2 = nn.Parameter(torch.empty(self.M + self.N, dim))
        nn.init.xavier_normal_(self.E2.data)
        for _ in range(MLP_layers):
            self.mlps.append(nn.Linear(dim,dim,bias=True))
            nn.init.xavier_normal_(self.mlps[-1].weight.data)
        # Attntion model
        self.attn = nn.Linear(dim,dim,bias=True)
        self.q = nn.Linear(dim,1,bias=False)
        self.attn_softmax = nn.Softmax(dim=1)
        
    def aggregate(self):
        # Generate embeddings z_p
        B=[]; B.append(self.E)
        x = self.convs[0](self.E,self.data_p.edge_index)
        B.append(x)
        for i in range(1,self.num_layers):
            x = self.convs[i](x,self.data_p.edge_index)
            B.append(x)
        z_p = sum(B)/len(B) 
        # Generate embeddings z_n
        C = []; C.append(self.E2)
        x = F.dropout(F.relu(self.mlps[0](self.E2)),p=0.5,training=self.training)
        for i in range(1,self.MLP_layers):
            x = self.mlps[i](x);
            x = F.relu(x)
            x = F.dropout(x,p=0.5,training=self.training)
            C.append(x)
        z_n = C[-1]
        # Attntion for final embeddings Z
        w_p = self.q(F.dropout(torch.tanh((self.attn(z_p))),p=0.5,training=self.training))
        w_n = self.q(F.dropout(torch.tanh((self.attn(z_n))),p=0.5,training=self.training))
        alpha_ = self.attn_softmax(torch.cat([w_p,w_n],dim=1))
        Z = alpha_[:,0].view(len(z_p),1) * z_p + alpha_[:,1].view(len(z_p),1) * z_n
        return Z
    
    def forward(self,u,v,w,n,device):
        emb = self.aggregate()
        u_ = emb[u].to(device);
        v_ = emb[v].to(device);
        n_ = emb[n].to(device);
        w_ = w.to(device)
        positivebatch = torch.mul(u_ , v_ ); 
        negativebatch = torch.mul(u_.view(len(u_),1,self.embed_dim),n_)  
        sBPR_loss =  F.logsigmoid((torch.sign(w_).view(len(u_),1) * (positivebatch.sum(dim=1).view(len(u_),1))) - negativebatch.sum(dim=2)).sum(dim=1)
        reg_loss = u_.norm(dim=1).pow(2).sum() + v_.norm(dim=1).pow(2).sum() + n_.norm(dim=2).pow(2).sum();
        return -torch.sum(sBPR_loss) + self.reg * reg_loss
```

<!-- #region id="8pVwjmftMcku" -->
## Evaluator
<!-- #endregion -->

```python id="qZFsrwih-9P7"
class evaluate():
    def __init__(self,reco,train,test,threshold,num_u,num_v,N=[5,10,15,20,25],ratings=[20,50]):
        '''
        train : training set
        test : test set
        threshold : To generate ground truth set from test set
        '''
        self.reco = reco
        self.num_u = num_u;
        self.num_v = num_v;
        self.N=N
        self.p=[]
        self.r=[]
        self.NDCG=[]
        self.p_c1=[]; self.p_c2=[]; self.p_c3=[]
        self.r_c1=[]; self.r_c2=[]; self.r_c3=[]
        self.NDCG_c1=[]; self.NDCG_c2=[]; self.NDCG_c3=[]
        self.tr = train; self.te = test;
        self.threshold = threshold;
        self.gen_ground_truth_set()
        self.ratings = ratings
        self.partition_into_groups_(self.ratings)
        print('\nevaluating recommendation accuracy....')
        self.precision_and_recall_G(self.group1,1)
        self.precision_and_recall_G(self.group2,2)
        self.precision_and_recall_G(self.group3,3)
        self.Normalized_DCG_G(self.group1,1)
        self.Normalized_DCG_G(self.group2,2)
        self.Normalized_DCG_G(self.group3,3)
        self.metric_total()

    def gen_ground_truth_set(self):
        result = dict()
        GT = self.te[self.te['rating']>=self.threshold];
        U = set(GT['userId'])
        for i in U:
            result[i] = list(set([j for j in GT[GT['userId']==i]['movieId']]))#-set(self.TOP))
            if len(result[i])==0:
                del(result[i])
        self.GT = result

    def precision_and_recall(self):
        user_in_GT=[j for j in self.GT];
        for n in self.N:
            p=0; r=0;
            for i in user_in_GT:
                topn=self.reco[i][:n]
                num_hit=len(set(topn).intersection(set(self.GT[i])));
                p+=num_hit/n; r+=num_hit/len(self.GT[i]);
            self.p.append(p/len(user_in_GT)); self.r.append(r/len(user_in_GT));
                
    def Normalized_DCG(self):
        maxn=max(self.N);
        user_in_GT=[j for j in self.GT];
        ndcg=np.zeros(maxn);
        for i in user_in_GT:
            idcg_len = min(len(self.GT[i]), maxn)
            temp_idcg = np.cumsum(1.0 / np.log2(np.arange(2, maxn + 2)))
            temp_idcg[idcg_len:] = temp_idcg[idcg_len-1]
            temp_dcg=np.cumsum([1.0/np.log2(idx+2) if item in self.GT[i] else 0.0 for idx, item in enumerate(self.reco[i][:maxn])])
            ndcg+=temp_dcg/temp_idcg;
        ndcg/=len(user_in_GT);
        for n in self.N:
            self.NDCG.append(ndcg[n-1])
            
    def metric_total(self):
        self.p = self.len1 * np.array(self.p_c1) + self.len2 * np.array(self.p_c2) + self.len3 * np.array(self.p_c3);
        self.p/= self.len1 + self.len2 + self.len3
        self.p = list(self.p)
        self.r = self.len1 * np.array(self.r_c1) + self.len2 * np.array(self.r_c2) + self.len3 * np.array(self.r_c3);
        self.r/= self.len1 + self.len2 + self.len3
        self.r = list(self.r)
        self.NDCG = self.len1 * np.array(self.NDCG_c1) + self.len2 * np.array(self.NDCG_c2) + self.len3 * np.array(self.NDCG_c3);
        self.NDCG/= self.len1 + self.len2 + self.len3
        self.NDCG = list(self.NDCG)

    def partition_into_groups_(self,ratings=[20,50]):
        unique_u, counts_u = np.unique(self.tr['userId'].values,return_counts=True)
        self.group1 = unique_u[np.argwhere(counts_u<ratings[0])]
        temp = unique_u[np.argwhere(counts_u<ratings[1])]
        self.group2 = np.setdiff1d(temp,self.group1)
        self.group3 = np.setdiff1d(unique_u,temp)
        self.cold_groups = ratings
        self.group1 = list(self.group1.reshape(-1))
        self.group2 = list(self.group2.reshape(-1))
        self.group3 = list(self.group3.reshape(-1))
    
    def precision_and_recall_G(self,group,gn):
        user_in_GT=[j for j in self.GT];
        leng = 0 ; maxn = max(self.N) ; p = np.zeros(maxn); r = np.zeros(maxn);
        for i in user_in_GT:
            if i in group:
                leng+=1
                hit_ = np.cumsum([1.0 if item in self.GT[i] else 0.0 for idx, item in enumerate(self.reco[i][:maxn])])
                p+=hit_ / np.arange(1,maxn+1); r+=hit_/len(self.GT[i])
        p/= leng; r/=leng;
        for n in self.N:
            if gn == 1 :
                self.p_c1.append(p[n-1])
                self.r_c1.append(r[n-1])
                self.len1 = leng;
            elif gn == 2 :
                self.p_c2.append(p[n-1])
                self.r_c2.append(r[n-1])
                self.len2 = leng;
            elif gn == 3 :
                self.p_c3.append(p[n-1])
                self.r_c3.append(r[n-1])
                self.len3 = leng;
            
    def Normalized_DCG_G(self,group,gn):
        maxn=max(self.N);
        user_in_GT=[j for j in self.GT];
        ndcg=np.zeros(maxn);
        leng = 0
        for i in user_in_GT:
            if i in group:
                leng+=1
                idcg_len = min(len(self.GT[i]), maxn)
                temp_idcg = np.cumsum(1.0 / np.log2(np.arange(2, maxn + 2)))
                temp_idcg[idcg_len:] = temp_idcg[idcg_len-1]
                temp_dcg=np.cumsum([1.0/np.log2(idx+2) if item in self.GT[i] else 0.0 for idx, item in enumerate(self.reco[i][:maxn])])
                ndcg+=temp_dcg/temp_idcg;
        ndcg/=leng
        for n in self.N:
            if gn == 1 :
                self.NDCG_c1.append(ndcg[n-1])
            elif gn == 2 :
                self.NDCG_c2.append(ndcg[n-1])
            elif gn == 3 :
                self.NDCG_c3.append(ndcg[n-1])
```

<!-- #region id="TEY2dcSCMgDo" -->
## Main
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="RgXHcZo8Au28" executionInfo={"status": "ok", "timestamp": 1637763396868, "user_tz": -330, "elapsed": 1648, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="096d0cd3-09fc-49c3-90ea-3a403e491fc8"
data_class=Data_loader(args.dataset,args.version)
threshold = round(args.offset) # To generate ground truth set 

print('data loading...'); st=time.time()
train,test = data_class.data_load();
train = train.astype({'userId':'int64', 'movieId':'int64'})
print('loading complete! time :: %s'%(time.time()-st))

print('generate negative candidates...'); st=time.time()
neg_dist = deg_dist(train,data_class.num_v)
print('complete ! time : %s'%(time.time()-st))    
```

```python colab={"base_uri": "https://localhost:8080/", "height": 707, "referenced_widgets": ["6b05545cf8584d6c801943952502a502", "090f14dc32c84efaa8aefaa81c6045bc", "84ceb9c9e96c474aac7ff993dd4fdc0c", "001fd3d370734623a20f12f5c56e0cc1", "2a94d0bf53b84c8cb1651a01cb4282de", "269fed7761924035908a88496b17cca5", "6db2bef3b1c94eab8b2e9641eaec926c", "025857c072ba415ab6bc6fba090a5c6b", "d695c6eba0224a5ba9e3d140ceba2dec", "1b34b7a8c9954515a605f0e1cfb3895b", "94acb950211d4139abbd579100195876", "5fcb735075bd4706b68e3fc6e7e3a03e", "daa2dcc06a0b4e3498dc155a1714ce6c", "dcf7a29706984c1eadcb23fc62bc2fa6", "024020eda04040ddb2dd1c25290e7251", "0812a34902134098802c3ebc31252c61", "85b631676945440cb8f51fea0c67f93a", "a591efdee4344099b24c780f8125835d", "bada41b99dd549ec83f1ef5dbb76ad41", "7d986f4df919480fa4326cf57353c7d8", "7ff6be3935b14cc78ea9f26dc5e4fcaa", "905286f8c3f4416abbe6b736265bcf64", "fe0a7a3ecee94f919027a89314b3c92e", "5ef79342f4054e90bdcccaeb6c24d2ff", "c2feaad8667447d4a06b83b47af9070b", "6855ba890b834aca82bf682648dc546a", "0355c423072e403992bcc5458deae85a", "8b83c85fb68b47e3b205a40660e1db86", "d2c9cadd5a4c46169945a68c1ce4959f", "9c5b734f0e0642bfbceafe75d55b1265", "22bf95920f1e4b16a72ea26a12807735", "083632443c5c477fb1e204347b07f416", "4fa997e8815c46ceafad19f232de48d5", "4c2d45a170b948e7adf7272de3011e26", "33f328c1cb5d447bbe9cb65e19ef0292", "fe0ef63265ae4351919ffcf526fdf3f5", "ba54ae92f1f84e9ba1b4764c58109ebd", "7658fed0eb9d48f0ab4f34beb4eedbac", "f9501e7af7e6467fbf62c8b055957479", "cc3d6adba3a54f18bbb2035f38bcd3d1", "c0acb27f7e054068ab043e0f18040214", "a531aa57763e4aad9b13cb2a1b19e59a", "8584444a386f4d10af86821ec8df5b5a", "e95a76e944344a2e9178b205ee86280c", "52c03fe15ba14e2aa6fa3331d7faf0e8", "22cb0955175646c9b7fb53421a1aa872", "c2b9cf743f724e5688021203b3755470", "601153c213fb40a199b2d15676dc0bc6", "4fb3ff7952bd493e97094a113466e255", "15cb391228d7480bb16db24afadbd1a6", "a3d7b8e0001d405591c09a6b94c5cc30", "7ed0e2d2fc554dc18bad07668f08b7f1", "8d469b761ac84470b182461d4345a155", "57117757e82a4663bbc9b64269ec0965", "eaf1ebaf78424511a5722d4d7c1f95ae", "2ed6e8b82c1243f98f1e1c072e653f6f", "db2d9c59fb194a1a9258f9c28255f08c", "d86b6500c57642e0ace5a692e35f3a10", "c0e3927d489449dc9a51db67eb87d494", "e7393fe33c0a42baac336d4788f70c70", "3075ee2be7a1443aab8ad8f80accb57a", "8b7880c58e72480fb9d6cee40f948cf8", "0f04f271f12143df906add35dfa99b1f", "b69e21ae7c7348028a7e60a50022dc3a", "865b5267622547c29a4300963eb4c9f9", "7b74e2d14cc74f0ba47fd148a6518f2c", "1c67af5a89614de4942d60c387698101", "c822104303fa441fafb61207a5063892", "43a389e38511481cbea9004b30e814fb", "603c26fb29b548c3a127bdbaeb58d604", "055a9712672346319ac96b41872a749f", "fc0afc5a542c4c22bef31bbdc1787fd1", "44863b0cb01a45699d7df15b4d80b683", "2e4b43b8113a42e9a5de1cca90a77728", "b4e0868e70a2470c94d4d32f04b72bad", "215e69bf90484f628b5072b1d23b6de6", "2cd766c2df4841f19b6aa57110401b14", "356010d3da8e463495d3cdb983b5e3b5", "71280423f7984edabf64eaa3cdab7b51", "5ceeefeb70674d448a1f6e81f771a2f4", "8cd5412e926b4e5b857bca1dcac3d48e", "12895981e07043b79e25a5845d944931", "a8c4ab53149f4230945455096ade21c5", "ca2e4b0f26ed4e3bb1c8773c0e512685", "8955a9c236354a8296a219ddee7696b8", "363b66fba0a54e2b9c1bd90bf11d8c68", "f13d93cf4ed64abdb6803eaf05afef47", "d5c1f1a1496447f8a863ea9d5946791e"]} id="93VSXEfkH9CG" executionInfo={"status": "ok", "timestamp": 1637768616952, "user_tz": -330, "elapsed": 5186919, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ee64dc2e-e04a-4df5-9258-4a5317c1d825"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model= SiReN(train, data_class.num_u,data_class.num_v,offset=args.offset,num_layers = args.num_layers,MLP_layers=args.MLP_layers,dim=args.dim,device=device,reg=args.reg)#.to(device);
model.data_p.to(device)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr = args.lr)

print("\nTraining on {}...\n".format(device))
model.train()
training_dataset=bipartite_dataset(train,neg_dist,args.offset,data_class.num_u,data_class.num_v,args.K);

for EPOCH in range(1,args.epoch+1):
    if EPOCH%2-1==0:training_dataset.negs_gen_EP(2)
    LOSS=0
    training_dataset.edge_4 = training_dataset.edge_4_tot[:,:,EPOCH%2-1]
    ds = DataLoader(training_dataset,batch_size=args.batch_size,shuffle=True)
    q=0
    pbar = tqdm(desc = 'Version : {} Epoch {}/{}'.format(args.version,EPOCH,args.epoch),total=len(ds),position=0)
    for u,v,w,negs in ds:
        q+=len(u)
        st=time.time()
        optimizer.zero_grad()
        loss = model(u,v,w,negs,device) # original
        loss.backward()                
        optimizer.step()
        LOSS+=loss.item() * len(ds)
        pbar.update(1);
        pbar.set_postfix({'loss':loss.item()})
    pbar.close()

    if EPOCH%2==0 :
        directory = os.getcwd() + '/results/%s/SiReN/epoch%s_batch%s_dim%s_lr%s_offset%s_K%s_num_layers%s_MLP_layers%s_threshold%s_reg%s/'%(args.dataset,EPOCH,args.batch_size,args.dim,args.lr,args.offset,args.K,args.num_layers,args.MLP_layers,threshold,args.reg)
        if not os.path.exists(directory):
            os.makedirs(directory)
        model.eval()
        emb = model.aggregate();
        top_k_list = gen_top_K(data_class,emb,train,directory+'r%s_reco.pickle'%(args.version)) 
        eval_ = evaluate(top_k_list,train,test,threshold,data_class.num_u,data_class.num_v,N=[10,15,20],ratings=[20,50])
        print("\n***************************************************************************************")
        print(" /* Recommendation Accuracy */")
        print('Precision at [10, 15, 20] :: ',eval_.p)
        print('Recall at [10, 15, 20] :: ',eval_.r)
        print('NDCG at [10, 15, 20] :: ',eval_.NDCG)
        print("***************************************************************************************")
        directory_ = directory+'r%s_reco.pickle'%(args.version)
        with open(directory_,'wb') as fw:
            pickle.dump(eval_,fw)
        model.train()
```

<!-- #region id="N1SNkv69OkWM" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="QqAlsR9BOkWN" executionInfo={"status": "ok", "timestamp": 1637762417561, "user_tz": -330, "elapsed": 3668, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="68ec4782-32d3-4fc0-bc2b-32eb6f7deea9"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d -p torch_geometric
```

<!-- #region id="_UTXo_AeOkWO" -->
---
<!-- #endregion -->

<!-- #region id="ARVn9tFAOkWO" -->
**END**
<!-- #endregion -->
