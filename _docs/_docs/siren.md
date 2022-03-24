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

<!-- #region id="7VlmnB3gSjE7" -->
# Sign-Aware Recommendation Using Graph Neural Networks on ML-1m Dataset in PyTorch
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

```python colab={"base_uri": "https://localhost:8080/"} id="FkWDKmmUBFJW" executionInfo={"status": "ok", "timestamp": 1637761329872, "user_tz": -330, "elapsed": 392, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="81d350ac-c184-48a0-84ee-0f55a849e9fe"
class Args:
    dataset = 'ML-1M' # Dataset
    version = 1 # Dataset version
    batch_size = 1024 # Batch size
    dim = 64 # Dimension
    lr = 5e-3 # Learning rate
    offset = 3.5 # Criterion of likes/dislikes
    K = 40 # The number of negative samples
    num_layers = 4 # The number of layers of a GNN model for the graph with positive edges
    MLP_layers = 2 # The number of layers of MLP for the graph with negative edges
    epoch = 5 # The number of epochs
    reg = 0.05 # Regularization coefficient

args = Args()
Args.__dict__
```

<!-- #region id="qpVGR7NVMUa-" -->
## Dataset
<!-- #endregion -->

```python id="eKhodi5h-aNm"
!git clone -q --branch v2 https://github.com/RecoHut-Datasets/movielens_1m.git
```

```python id="xw2tVzO2-HcI"
class Data_loader():
    def __init__(self,dataset,version):
        self.dataset=dataset; self.version=version
        self.sep='::'
        self.names=['userId','movieId','rating','timestemp'];
        self.path_for_whole='./movielens_1m/ratings.dat'
        self.path_for_train='./movielens_1m/train_1m%s.dat'%(version)
        self.path_for_test='./movielens_1m/test_1m%s.dat'%(version)
        self.num_u=6040; self.num_v=3952;
        
    def data_load(self):
        self.whole_=pd.read_csv(self.path_for_whole, names = self.names, sep=self.sep, engine='python').drop('timestemp',axis=1).sample(frac=1,replace=False,random_state=self.version)
        self.train_set = pd.read_csv(self.path_for_train,engine='python',names=self.names).drop('timestemp',axis=1)
        self.test_set = pd.read_csv(self.path_for_test,engine='python',names=self.names).drop('timestemp',axis=1)            
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

```python colab={"base_uri": "https://localhost:8080/"} id="RgXHcZo8Au28" executionInfo={"status": "ok", "timestamp": 1637761195212, "user_tz": -330, "elapsed": 9968, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="dbad62b2-20cd-4f17-a7c3-2329269c94b6"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 806, "referenced_widgets": ["ead6047367b944f79636469f7f71015b", "d7019f566c05455c8b4b62b1344e21cd", "107f6502e41b46a1a696ac059ca4f754", "9a39640fcf074a78b10217ed37fcd31d", "0a809c7dd9ce4c9eb27cda7468391b55", "f4f7559bd3714a989420f91381cd3223", "972176e5a0914db68c2f273f320ca69f", "4f738267f6fe48529bc7642a8cfe21da", "11962373c90c4ab4a6894a3ac61c56f9", "a3c2d027e0d84f21baa8c68656e7010a", "e99793784ef44ddf95284d2bdfeb76eb", "014cbdfba6af4da5a74fda8f0f0e0b0c", "cb5a582617af4acfbaa5ae5dd786c858", "02ee86c7461b40598583b436015fe847", "63441a8c0d0d4faa9aa02537952007f0", "ce2c4a7eb9b14cbfb1d28d256505a142", "7333789cccd54827bfd3e295923ca976", "17ec08c591dc41e189925ea3c05b1fd2", "0d65d5610232404187802c3028054db9", "38af2612ebc3439bb8b72696d58bc363", "e3e4a7c24a33428eb5aceaa8ed9c7e78", "c1d59aced543427d88ac758f690122c2", "0ffd362a9c824471aca634a17ff25235", "c20636ca4bc741f8a58d866b70ab4ab8", "d96d6f64aa9f4959825dd18e160c02eb", "0e92d8d315674c0e9be12a78d457f0cb", "c370695bcb1940d29ecee5073d1bb864", "8922da611bdf4c8892327edcc8266e7a", "c40339c6839c464fbbf87e5e86e03974", "947f4b5bd9714813806cb5abe910eafe", "a596a3a3b4834269bf6ed0d6bf154a59", "0a48f710dd7b424aa22abc343305ba32", "1cd92555ca474c15881a04208fb1917a", "096e5187287141f2a139151142fe6acd", "e5d933ff65294037876a0b3f2ee0e049", "ff4945d9a1a043e5a7230cda054d381f", "87db259f866c4241abba5bb5058c068e", "263404e08b614993ad6f9e959f0cd5cd", "f4927693b7bc4cf690b9e86125a49f95", "a2de5d902df24b849811bcaf7a9a4ee3", "a53299e5b8094edc9dd8169b42c7dd22", "632774c513ba487c930e04dbbc330d7e", "e29cb7026f58446f8583809fefb161ed", "2b8ced7ca75448e7accf3abb3dc73675", "29cbaccc00b94ddc99f3b0d92482ad6a", "7525cca057714ef691c20080d89e323e", "171813ed80f8407586da544dccff22b5", "ddcab42cb8db40eea6f847c8da8d8285", "50b1849639104386a0fbfa4bdd3cdd19", "680b514c97b64c51ab285356e1cf5bec", "01ffcc2fc2934165a97bed2ed9d6f2b4", "03557b71b8ab489c85e3b04e21210900", "a72ca43431d04813935a3c48111a6565", "1f19f24bebf34d20bcd403525584bd40", "c163af3b4316498e9990e53155a4abc0", "ede4338eeb62493fa76f937d603a16a5", "75d2e07780df4c10bf06939086b772f0", "93e02bae7f4342929a7b05058fe5774b", "b009734c5bee44fd8fbf2e0ac49094b5", "178d6724c12a4046af8e91e9224b8068", "58fc54ccf1214cf0ba48fe75ea115a86", "c0a01a0fb8f44031bcb368a2bd1276d6", "f50e24eaaab345b98bc209a4a8bf3a73", "37ef3244203c4f818de6af92da151a64", "3d0ade9c2ef74f5baa35a227e2eaedd4", "e04cfbd2a7cc4296804a20d1e33a8a2b", "aa1851c2dd8f461489c9059da5470f42", "9e87c375e1ad4984a9c61af36ba91081", "228f4b3a33924ce69c5f90fcef4fdd2f", "b63c999a1d164727ae814d4304854c19", "875db3e6a8e144c5b66e421547919c88", "79b2d990eff04acba57e6820bdf33e45", "37c294403d7b49519dc747054eabc8b3", "5795943776ed4abd947e503bb44f7a7a", "c3f10f2baf6e4ef6a4a3ac911046eae5", "f628f2edeb7349d5b478554bfc94393b", "66a8015fdb9146dfbbee0fe8b504fd3f", "b481673d99a746d3820e5098f2b62f83", "35b1d4537ccf4f30b9285a94cada0cbd", "2e26bc06be844e9e8d7102dabba5dbb3", "231f6b3a34e4449b92a039ce2d17b7c0", "00c14c7d772a4557b946999a2cb7165d", "fc4913f1d4e24a6bb02781c8f0a6e05a", "636d9b01fbdb485182e5ae62672d8dce", "44e2cacb8eb04d3c90f9fae407986a82", "31b5bc83ba0b434abf9badec077d32dd", "d6e32a2d33694a888c5de660ce577223", "fc872aab29b94dde8aa9950cbeb37613", "e83e4c84d0a948e2b7e5da1c5b642bd2", "0a860f75211c466f8509832ee95f91af", "01a0ba7bde004e5282f53473a69d209f", "b96c17c668e440288264cd3a929f1551", "25a0a94d88a84ed9bf5b94a551ada82e", "51a374e955ed4786b75473e485411bd3", "fb9fe648cc5044edb3a931ca25d56433", "bd083664967e4c94b130763e63484916", "75c6a649a14f4834a984f95af808cbea", "79a8c652316048df93621c205b0d2c9e", "8ab9f40f74ef4ebeb94937ded992516e", "e145f46ecd36420284bcbde614f6f768", "8f0a81812e8e443a9db4e3180e8ca2b3", "766b730d130b4b4590dcbd57cf01a085", "d1f4619fb6ec4f1cbc2e24a84928fbc5", "e9060be8b2504d458a78e171359eceb8", "24c41df87d8541a9b94251d0c1b6f0bb", "ded72cecb54d448d987b5e0e9659578d", "cdd7970076f447a097af679ecd5a6344", "2b8cd80de36f409a8542c53fdd28adb0", "145a81cf42064d6ca67483796af406da", "df131514b9d6434eb2969da766ba3065"]} id="93VSXEfkH9CG" executionInfo={"status": "ok", "timestamp": 1637762281358, "user_tz": -330, "elapsed": 924484, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2ac89b0a-7791-4e03-e658-5fcde15e2451"
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
