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

<!-- #region id="IAOQyj4UgNXC" -->
# SiReN on ML-1m in PyTorch
<!-- #endregion -->

<!-- #region id="GqM8GSs3gNXF" -->
## **Step 1 - Setup the environment**
<!-- #endregion -->

<!-- #region id="FSaTwnTpgNXG" -->
### **1.1 Install libraries**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="CpfvzIdsgiAC" executionInfo={"status": "ok", "timestamp": 1639981557502, "user_tz": -330, "elapsed": 32761, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5bbee329-9eb5-4fb2-e626-225aaac8bbb2"
# torch geometric
try: 
    import torch_geometric
except ModuleNotFoundError:
    # Installing torch geometric packages with specific CUDA+PyTorch version. 
    # See https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html for details 
    import torch
    TORCH = torch.__version__.split('+')[0]
    CUDA = 'cu' + torch.version.cuda.replace('.','')

    !pip install torch-scatter     -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
    !pip install torch-sparse      -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
    !pip install torch-cluster     -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
    !pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
    !pip install torch-geometric 
    import torch_geometric
import torch_geometric.nn as geom_nn
import torch_geometric.data as geom_data
```

```python id="zS2gINO3gNXG" outputId="2b07839e-2eba-443b-efc4-9a8f9c604692" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1639981563873, "user_tz": -330, "elapsed": 6391, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
!pip install -q -U git+https://github.com/RecoHut-Projects/recohut.git -b v0.0.4
```

<!-- #region id="gTMly1omgNXJ" -->
### **1.2 Download datasets**
<!-- #endregion -->

```python id="PZO3etVugNXK" executionInfo={"status": "ok", "timestamp": 1639981598539, "user_tz": -330, "elapsed": 8407, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
!git clone -q --branch v2 https://github.com/RecoHut-Datasets/movielens_1m.git
```

<!-- #region id="EcrfHgL-gNXL" -->
### **1.3 Import libraries**
<!-- #endregion -->

```python id="IkpnKkKFgNXM" executionInfo={"status": "ok", "timestamp": 1639981606929, "user_tz": -330, "elapsed": 513, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import torch
from torch import optim
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import time
from tqdm.notebook import tqdm

import os
import pickle

import warnings
warnings.filterwarnings('ignore')
```

```python id="ghLHoTimgNXP" executionInfo={"status": "ok", "timestamp": 1639981624575, "user_tz": -330, "elapsed": 639, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# layers
from recohut.layers.message_passing import LightGConv, LRGCCF

# models
from recohut.models.siren import SiReN

# transforms
from recohut.transforms.bipartite import BipartiteDataset
```

<!-- #region id="QaZRU55ogNXR" -->
### **1.4 Set params**
<!-- #endregion -->

```python id="8n6VamnDgNXS" executionInfo={"status": "ok", "timestamp": 1639981636885, "user_tz": -330, "elapsed": 399, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
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
    epoch = 4 # The number of epochs
    reg = 0.05 # Regularization coefficient
```

<!-- #region id="8fFrHdTigNXT" -->
## **Step 2 - Data preparation**
<!-- #endregion -->

```python id="GG9lThzagNXU" executionInfo={"status": "ok", "timestamp": 1639981649941, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
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

```python id="XH5Der9BhkgE" executionInfo={"status": "ok", "timestamp": 1639981672242, "user_tz": -330, "elapsed": 4, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def deg_dist(train, num_v):
    uni, cou = np.unique(train['movieId'].values-1,return_counts=True)
    cou = cou**(0.75)
    deg = np.zeros(num_v)
    deg[uni] = cou
    return torch.tensor(deg)
```

```python id="o1t4IeJ4hlDC" executionInfo={"status": "ok", "timestamp": 1639981672903, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
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

```python colab={"base_uri": "https://localhost:8080/"} id="ixFEggVphr4T" executionInfo={"status": "ok", "timestamp": 1639981721238, "user_tz": -330, "elapsed": 9797, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2b131484-0cca-4b3e-f647-ccf7a3cfcadc"
args = Args()
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

<!-- #region id="vmUIRtkwgNXe" -->
## **Step 3 - Training & Evaluation**
<!-- #endregion -->

```python id="5l_3d_KwgNXj" executionInfo={"status": "ok", "timestamp": 1639981685757, "user_tz": -330, "elapsed": 873, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
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

```python id="RW0u2_auh2hu" executionInfo={"status": "ok", "timestamp": 1639981778365, "user_tz": -330, "elapsed": 9463, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args.user_col = 'userId'
args.item_col = 'movieId'
args.feedback_col = 'rating'

model= SiReN(train,
             data_class.num_u,
             data_class.num_v,
             offset=args.offset,
             num_layers=args.num_layers,
             MLP_layers=args.MLP_layers,
             dim=args.dim,
             device=device,
             reg=args.reg,
            graph_enc = 'lightgcn',
            user_col = args.user_col,
            item_col = args.item_col,
            rating_col = args.feedback_col)

model.data_p.to(device)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr = args.lr)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 707, "referenced_widgets": ["db4ec0a687d04355a3ddc2c266b4fb02", "1dabe35ff89d42b095467b8a28967c40", "51068d426f6743ab823613fe54c8f92b", "ad284b3dbc564b34918e627f6a6160ed", "80bd36bece0c4df1b6240976a76744ab", "a8ec4992696642648817df6e25b19234", "aeb3a63adca3430c99676a313bd78d21", "e987e70d549f44758b88c6889efb935e", "8b7bf527a43348d998b112bed2ba1173", "a8eee4056fbc442a86b6b4489f58b199", "37b731c37498406199d4cdc50f3cf347", "da00ae1a6d844879920d15450bd70087", "ee846d746be2423daceee1d4a3d701c6", "32263760a6924b35b422d8ad50b3c137", "7adbad24ab3044c38614579ef9e0e363", "31cbbc433b0247afa7f2649af1872db0", "64d719fbfb434bbaa1ba901e674c5a56", "188e681ad70b4c989b722f547d5c7e86", "017673f173004669be005d648d7bec8d", "ed53b06d9440416e85e4cd5a75784720", "d5068168d46d40c18953f1429006f6a7", "8717200518d44aa98c116e4911f20e20", "c4cc38955e754b56935444f7ea1683df", "11cc9eb00fb348a2902384929da679c9", "1744f75803e342abaa7ce8c76ffcaa5d", "4fab249144a945bfa41233f76d1d5753", "21964d022b7e49498fac70fcf51fa863", "b13f32c5343d4bd6bb7e08103ae5ffdc", "6aee9f245c404a83a2f9d05ee8648e95", "691dbf2829334cbabcb3ef618caab41a", "cf197ebe4cbe4656abe3d41270c10ed7", "36cfbde84f2a4e1eb94e135a5b3a1414", "087219652f8049c3b8b992cb94da9d0e", "83eaf9d1b71d4dc08c4df7e2dbe623b5", "985b29ded0c547df80b1297a4b33a097", "8d108db4b56441a6a89fc771fd6003a6", "a1a6542b1d974085910cf310b4a971a7", "e7ad8d022ea6472699fc1f3e87b72caa", "d25ce4aa87934dfbbff8d43264aaa2ab", "72af0997ce44459b812fabc8395e60ef", "9f284e166b224cc7a269217638a7ce6e", "19c0c721cb78440a9118dbbd8ddf2475", "e562220dbae24138a691763141eeb517", "be31756d491e41f79f0a48faa781a746", "72f93c8f098e436d991b864139676f10", "3777e8b419f54846a4517f4c6f4b9ce1", "e58bee7a14a54f7da65db2476f430afa", "df4ccc48b83440cf8add2f6589f4bbf1", "8285321ec95e4e43b3832b31fd4e2737", "bc9d29ad38b14ae8a14f93bf29ab3e50", "f5273092fd874b9e807deb5c31dc9bfa", "b04abef361ca4cd895dbbea55c7870f9", "054228b3539d4057a0bdb78bd2833b77", "74f32b0c83784f268581a6618ea0e912", "5307ffddffc842ff8dc3d928cbc95514", "f4d5f46afaec44f396088c241b8014de", "949850a7ecc44ba39b5abb91b9935024", "d25983e69a3148fb9733d95101e943f1", "18e3f3be0d3c4b16ad0021f9dffe9fe4", "7fe65d2eef084c999ea0a5186dff3544", "e7e0030576e2401da8f999e55526d6ed", "36ea35177933450eb06f04f11ca8f553", "09d4239d85db4dde8e8ccc4f990fa6fb", "ba247b0f524444eaa59fffa01e6e137b", "3b2522a650e647129339cf3869b3d40f", "2d3fe56d31cb455d9e3ae8652ba9400b", "0f6d8285f502433eb842ea01c78f7de8", "7d8fe07f64ac40cf9f9a390971257706", "ea0f6c47030d433f83aef96e31540e9b", "7a4eb798b4e34e50b4563969e34ca563", "e81710937d2b4336909aa7b9f91008e4", "aab4be0f60a341f3af4d4a661a19893d", "3ed4aaea00c2488bab3fa52c00c15176", "2813295b1bb84a6e8c2e8dae725e155d", "d0dd60f5c86a444a8089446cb04483a1", "602102f486174e1cae29c81b3a53f4c1", "b37564fbdc8d414f9a2f757c51265234", "3377ec386afb4a4a82e8fe17730bafc6", "355c900e87994598b72c22eb46f4beb9", "fb85334c68944fc7a51fd9e69192c302", "7e805330ee3d478c9a4b42f48363086a", "1adf6a66c5234c65bf9ad88732644fcc", "f7c3996471624a529c1c91e02a4cb7d2", "c5c66a9a3b51481fb25ff569e1c3b4e7", "043eb90abcef4e5aa1ef84fec993131c", "4e548a7575154192be257d0228dfaa62", "b00a15dcd36440388d087d37f5e32828", "c0938cab98c243fa9856c095006e0cac"]} id="93VSXEfkH9CG" outputId="1f8e7ac8-b308-45fe-9b53-a0d576624c07" executionInfo={"status": "ok", "timestamp": 1639981419159, "user_tz": -330, "elapsed": 689791, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
print("\nTraining on {}...\n".format(device))
model.train()
training_dataset = BipartiteDataset(args, train, neg_dist, args.offset, 
                                    data_class.num_u, data_class.num_v, args.K)

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

<!-- #region id="7jgvQq8jgNXm" -->
## **Closure**
<!-- #endregion -->

<!-- #region id="Wy8hB0bKgNXn" -->
For more details, you can refer to https://github.com/RecoHut-Stanzas/S138006.
<!-- #endregion -->

<!-- #region id="7K9x02G0gNXn" -->
<a href="https://github.com/RecoHut-Stanzas/S138006/blob/main/reports/S138006_Report.ipynb" alt="S138006_Report"> <img src="https://img.shields.io/static/v1?label=report&message=active&color=green" /></a> <a href="https://github.com/RecoHut-Stanzas/S138006" alt="S138006"> <img src="https://img.shields.io/static/v1?label=code&message=github&color=blue" /></a>
<!-- #endregion -->

```python id="S9xtIm3cgNXo" outputId="fc33050c-7274-4dbb-f1ca-a70de241b6e2" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1639981816750, "user_tz": -330, "elapsed": 5291, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d -p recohut
```

<!-- #region id="gZTIo-_KgNXo" -->
---
<!-- #endregion -->

<!-- #region id="7Ea7LAPQgNXo" -->
**END**
<!-- #endregion -->
