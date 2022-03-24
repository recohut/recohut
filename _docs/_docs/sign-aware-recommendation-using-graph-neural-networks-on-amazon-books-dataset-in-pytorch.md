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

<!-- #region id="dfEu7_3GMP6Z" -->
## Setup
<!-- #endregion -->

```python id="3X8DNyvx6-06"
!pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
!pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
!pip install torch-geometric
```

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="-8cfttRjGXjg" executionInfo={"status": "ok", "timestamp": 1637764407701, "user_tz": -330, "elapsed": 5492, "user": {"displayName": "sparsh agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "00322518567794762549"}} outputId="edabca3f-b8d3-4b08-96fc-6532de7c16c9"
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

```python colab={"base_uri": "https://localhost:8080/"} id="FkWDKmmUBFJW" executionInfo={"status": "ok", "timestamp": 1637764571878, "user_tz": -330, "elapsed": 420, "user": {"displayName": "sparsh agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "00322518567794762549"}} outputId="febf08ca-1e5b-4a94-a8d7-aa39175524f9"
class Args:
    dataset = 'amazon' # Dataset
    version = 1 # Dataset version
    batch_size = 1024 # Batch size
    dim = 64 # Dimension
    lr = 5e-3 # Learning rate
    offset = 3.5 # Criterion of likes/dislikes
    K = 40 # The number of negative samples
    num_layers = 4 # The number of layers of a GNN model for the graph with positive edges
    MLP_layers = 2 # The number of layers of MLP for the graph with negative edges
    epoch = 2 # The number of epochs
    reg = 0.05 # Regularization coefficient

args = Args()
Args.__dict__
```

<!-- #region id="qpVGR7NVMUa-" -->
## Dataset
<!-- #endregion -->

```python id="eKhodi5h-aNm"
!git clone -q https://github.com/RecoHut-Datasets/amazon_reviews_books.git
```

```python id="xw2tVzO2-HcI"
class Data_loader():
    def __init__(self,dataset,version):
        self.path_for_whole='/content/amazon_reviews_books/bronze/v1/amazon-books-enc.csv'
        self.path_for_train='/content/amazon_reviews_books/bronze/v1/train_amazon%s.dat'%(version)
        self.path_for_test='/content/amazon_reviews_books/bronze/v1/test_amazon%s.dat'%(version)
        self.num_u=35736; self.num_v=38121;
        
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

```python colab={"base_uri": "https://localhost:8080/"} id="RgXHcZo8Au28" executionInfo={"status": "ok", "timestamp": 1637764545901, "user_tz": -330, "elapsed": 1407, "user": {"displayName": "sparsh agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "00322518567794762549"}} outputId="784fae71-96a3-43a1-dd3b-65de243c6517"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 388, "referenced_widgets": ["84a4d33f8ade489e9acd24370c97da50", "4a52d2c73f1549289313460a93672483", "fc5d29fb7a1743a5bff305235e69128d", "da97ea309076432cb180680d2eb140a5", "96db3603ca794058b16d6ea5e35b94fc", "05f8c91e1e2a482ab419c7ced83d7608", "b4656f4af19a48caa41c312a08d3057b", "0c5b3199300f48c88bcda2bf1ef1f3c3", "90c14217d84844c78c3fda6cde0d78d5", "cba8e42b49ed47b3ba99491216b1ebe7", "3a845662d7b647f294c18844be4243dc", "67c9ef4ee92b4daab4bbb3ce16c01abd", "835e2d4cff0342c1840c68a9a12fe4b7", "723210d65d0448ee8c0c784a8b4524af", "d79ef7b7343943afbc60fdac0e61e363", "631e20e21f794bc084b4420134a533a6", "3c065cd60961400f9f7a098a9246205c", "c15f49c4d7fe4f89b52a9066a085310e", "abfe280210344af8a50a8048cb1f90f0", "ecbbcc0a6d93497d8769915ceba39af8", "8a2c42de69014875b32e6961d3bbcacb", "d26caaf5787949da83837649648c2cf1", "c5424f75ed85459888a6c9ae925e43c6", "7e0413d157d84751bdd10fc49868a32d", "29a9465288db46aebaea9162e48f05ff", "74f4282fad5c4c24b54df251cff7dd0d", "6aa32bcf4e1c4a459d7d6f17a96b9b88", "63462286b5cd43cab62f507044882322", "39fc926575674dc19628a7b38e90cb79", "d29c9aa56bae4a3ab4150c7c2bc0fe85", "bdecdd11f422439b9287ac86b82898b4", "0bd0710169904786b2d524c78d10dc03", "3189ba622e47472399afa56030fbbd20", "2c2cec53b39d49209df734576d1cdfcc", "08c36aced6924a13931ecf6dd65d1b15", "dc19bc675e684054808db32fe4c08466", "00ef9a6a22c94ce599ecc91836136b71", "faff2a0614cd4acaae886a6c8a1bf72e", "7b63b67cf11d4a2f8dd3fcc7bb48a50e", "c67ce14e7d9f4319b77bf7740676e5c4", "a9ac77f0e3f64341beb2cabdd7b18651", "dbf4a444252c477195350a9af301c118", "36ff91464b5846b29f8c74f23b680aa4", "c4398654e7264c7eaff1dbe31643dada"]} id="93VSXEfkH9CG" executionInfo={"status": "ok", "timestamp": 1637767222736, "user_tz": -330, "elapsed": 2642004, "user": {"displayName": "sparsh agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "00322518567794762549"}} outputId="cfd0195d-ba02-40ea-b2c8-d1987785828d"
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
