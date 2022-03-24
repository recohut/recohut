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

<!-- #region id="mVyRGyhdtRFw" -->
# LR-GCCF on Gowalla
<!-- #endregion -->

<!-- #region id="Y3oNohENVHAH" -->
## Executive summary
<!-- #endregion -->

<!-- #region id="DK7RWly8VKvU" -->
| | |
| --- | --- |
| Problem | GCNs suffer from training difficulty due to non-linear activations, and over-smoothing problem. |
| Hypothesis | removing non-linearities would enhance recommendation performance.  |
| Solution | Linear model with residual network structure |
| Dataset | Gowalla |
| Preprocessing | we remove users (items) that have less than 10 interaction records. After that, we randomly select 80% of the records for training, 10% for validation and the remaining 10% for test. |
| Metrics | HR, NDCG |
| Hyperparams | There are two important parameters: the dimension D of the user and item embedding matrix E, and the regularization parameter λ in the objective function. The embedding size is fixed to 64. We try the regularization parameter λ in the range [0.0001, 0.001, 0.01, 0.1], and find λ = 0.01 reaches the best performance. |
| Models | LR-GCCF |
| Cluster | PyTorch with GPU |
<!-- #endregion -->

<!-- #region id="s4weKl5kcU3v" -->
## Process flow

![](https://github.com/RecoHut-Stanzas/S794944/raw/main/images/process_flow.svg)
<!-- #endregion -->

<!-- #region id="paRynetXLa6r" -->
## Setup
<!-- #endregion -->

```python id="zfsN6pxILcQG"
import random
import torch 
import time
import pdb
import math
import os
import sys
from shutil import copyfile
from collections import defaultdict
import numpy as np
import pandas as pd 
import scipy.sparse as sp 

import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import torch.utils.data as data
```

```python id="CkWg0kI9LjgC"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
```

<!-- #region id="VhlaApRdJJDh" -->
## Data
<!-- #endregion -->

```python id="V5NS5Z5ULdnO"
# download
dataset = 'gowalla'
!git clone --branch v1 https://github.com/RecoHut-Datasets/gowalla.git
!wget -q --show-progress -O gowalla/val.txt https://github.com/RecoHut-Datasets/gowalla/raw/main/silver/v1/val.txt

# set paths
training_path='./gowalla/train.txt'
testing_path='./gowalla/test.txt'
val_path='./gowalla/val.txt'

# meta
user_num=29858
item_num=40981 
factor_num=64
batch_size=2048*512
top_k=20 
num_negative_test_val=-1##all

#testing
start_i_test=3
end_i_test=4
setp=1

path_save_base = './datanpy'
if not os.path.exists(path_save_base):
    os.makedirs(path_save_base) 

run_id='0'
path_save_log_base='./log/'+dataset+'/newloss'+run_id
if not os.path.exists(path_save_log_base):
    os.makedirs(path_save_log_base)  

result_file=open(path_save_log_base+'/results.txt','w+')

path_save_model_base='./newlossModel/'+dataset+'/s'+run_id
if not os.path.exists(path_save_model_base):
    os.makedirs(path_save_model_base)
```

<!-- #region id="iYvFa6MOwiF0" -->
data2npy
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="1v2H-ZUPwjUi" executionInfo={"status": "ok", "timestamp": 1639038539256, "user_tz": -330, "elapsed": 4316, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6f97fdcd-fb87-481f-d314-dee370d6ae73"
train_data_user = defaultdict(set)
train_data_item = defaultdict(set) 
links_file = open(training_path)
num_u=0
num_u_i=0
for _, line in enumerate(links_file):
    line=line.strip('\n')
    tmp = line.split(' ')
    num_u_i+=len(tmp)-1
    num_u+=1
    u_id=int(tmp[0])
    for i_id in tmp[1:]: 
        train_data_user[u_id].add(int(i_id))
        train_data_item[int(i_id)].add(u_id)
np.save(os.path.join(path_save_base,'training_set.npy'),[train_data_user,train_data_item,num_u_i]) 
print(num_u,num_u_i)
 
test_data_user = defaultdict(set)
test_data_item = defaultdict(set) 
links_file = open(testing_path)
num_u=0
num_u_i=0
for _, line in enumerate(links_file):
    line=line.strip('\n')
    tmp = line.split(' ')
    num_u_i+=len(tmp)-1
    num_u+=1
    u_id=int(tmp[0])
    for i_id in tmp[1:]: 
        test_data_user[u_id].add(int(i_id))
        test_data_item[int(i_id)].add(u_id)
np.save(os.path.join(path_save_base,'testing_set.npy'),[test_data_user,test_data_item,num_u_i]) 
print(num_u,num_u_i)


val_data_user = defaultdict(set)
val_data_item = defaultdict(set) 
links_file = open(val_path)
num_u=0
num_u_i=0
for _, line in enumerate(links_file):
    line=line.strip('\n')
    tmp = line.split(' ')
    num_u_i+=len(tmp)-1
    num_u+=1
    u_id=int(tmp[0])
    for i_id in tmp[1:]: 
        val_data_user[u_id].add(int(i_id))
        val_data_item[int(i_id)].add(u_id)
np.save(os.path.join(path_save_base,'val_set.npy'),[val_data_user,val_data_item,num_u_i]) 
print(num_u,num_u_i)


user_rating_set_all = defaultdict(set)
for u in range(num_u):
    train_tmp = set()
    test_tmp = set() 
    val_tmp = set() 
    if u in train_data_user:
        train_tmp = train_data_user[u]
    if u in test_data_user:
        test_tmp = test_data_user[u] 
    if u in val_data_user:
        val_tmp = val_data_user[u] 
    user_rating_set_all[u]=train_tmp|test_tmp|val_tmp
np.save(os.path.join(path_save_base,'user_rating_set_all.npy'),user_rating_set_all) 
```

<!-- #region id="0iTyipe9wzfU" -->
## Dataset
<!-- #endregion -->

```python id="LBJTWOBOxph7"
class BPRData(data.Dataset):
    def __init__(self,train_dict=None,num_item=0, num_ng=1, is_training=None, data_set_count=0,all_rating=None):
        super(BPRData, self).__init__()

        self.num_item = num_item
        self.train_dict = train_dict
        self.num_ng = num_ng
        self.is_training = is_training
        self.data_set_count = data_set_count
        self.all_rating=all_rating
        self.set_all_item=set(range(num_item))  

    def ng_sample(self):
        # assert self.is_training, 'no need to sampling when testing'
        # print('ng_sample----is----call-----') 
        self.features_fill = []
        for user_id in self.train_dict:
            positive_list=self.train_dict[user_id]#self.train_dict[user_id]
            all_positive_list=self.all_rating[user_id]
            #item_i: positive item ,,item_j:negative item   
            # temp_neg=list(self.set_all_item-all_positive_list)
            # random.shuffle(temp_neg)
            # count=0
            # for item_i in positive_list:
            #     for t in range(self.num_ng):   
            #         self.features_fill.append([user_id,item_i,temp_neg[count]])
            #         count+=1  
            for item_i in positive_list:   
                for t in range(self.num_ng):
                    item_j=np.random.randint(self.num_item)
                    while item_j in all_positive_list:
                        item_j=np.random.randint(self.num_item)
                    self.features_fill.append([user_id,item_i,item_j]) 
      
    def __len__(self):  
        return self.num_ng*self.data_set_count#return self.num_ng*len(self.train_dict)
         

    def __getitem__(self, idx):
        features = self.features_fill  
        
        user = features[idx][0]
        item_i = features[idx][1]
        item_j = features[idx][2] 
        return user, item_i, item_j 
```

```python id="c3O3h1d8J6Yx"
class resData(data.Dataset):
    def __init__(self,train_dict=None,batch_size=0,num_item=0,all_pos=None):
        super(resData, self).__init__() 
      
        self.train_dict = train_dict 
        self.batch_size = batch_size
        self.all_pos_train=all_pos 

        self.features_fill = []
        for user_id in self.train_dict:
            self.features_fill.append(user_id)
        self.set_all=set(range(num_item))
   
    def __len__(self):  
        return math.ceil(len(self.train_dict)*1.0/self.batch_size)#self.data_set_count==batch_size
         

    def __getitem__(self, idx): 
        
        user_test=[]
        item_test=[]
        split_test=[]
        for i in range(self.batch_size):#self.data_set_count==batch_size 
            index_my=self.batch_size*idx+i 
            if index_my == len(self.train_dict):
                break   
            user = self.features_fill[index_my]
            item_i_list = list(self.train_dict[user])
            item_j_list = list(self.set_all-self.all_pos_train[user])
            # pdb.set_trace() 
            u_i=[user]*(len(item_i_list)+len(item_j_list))
            user_test.extend(u_i)
            item_test.extend(item_i_list)
            item_test.extend(item_j_list)  
            split_test.append([(len(item_i_list)+len(item_j_list)),len(item_j_list)]) 
           
        return torch.from_numpy(np.array(user_test)), torch.from_numpy(np.array(item_test)), split_test
```

<!-- #region id="acpOF47Ix24z" -->
## Evaluate
<!-- #endregion -->

```python id="7p77Vhgxx9D9"
def metrics_loss(model, test_val_loader_loss, batch_size): 
    start_time = time.time() 
    loss_sum=[]
    loss_sum2=[]
    for user, item_i, item_j in test_val_loader_loss:
        user = user.cuda()
        item_i = item_i.cuda()
        item_j = item_j.cuda() 
     
        prediction_i, prediction_j,loss,loss2 = model(user, item_i, item_j) 
        loss_sum.append(loss.item())  
        loss_sum2.append(loss2.item())

        # if np.isnan(loss2.item()).any():
        #     pdb.set_trace()
    # pdb.set_trace()
    elapsed_time = time.time() - start_time
    test_val_loss1=round(np.mean(loss_sum),4)
    test_val_loss=round(np.mean(loss_sum2),4)
    str_print_val_loss=' val loss:'+str(test_val_loss)
    # print(round(elapsed_time,3))
    # print(test_val_loss1,test_val_loss)
    return test_val_loss
```

```python id="QgxclVVvKBY0"
def hr_ndcg(indices_sort_top,index_end_i,top_k): 
    hr_topK=0
    ndcg_topK=0

    ndcg_max=[0]*top_k
    temp_max_ndcg=0
    for i_topK in range(top_k):
        temp_max_ndcg+=1.0/math.log(i_topK+2)
        ndcg_max[i_topK]=temp_max_ndcg

    max_hr=top_k
    max_ndcg=ndcg_max[top_k-1]
    if index_end_i<top_k:
        max_hr=(index_end_i)*1.0
        max_ndcg=ndcg_max[index_end_i-1] 
    count=0
    for item_id in indices_sort_top:
        if item_id < index_end_i:
            hr_topK+=1.0
            ndcg_topK+=1.0/math.log(count+2) 
        count+=1
        if count==top_k:
            break

    hr_t=hr_topK/max_hr
    ndcg_t=ndcg_topK/max_ndcg  
    # hr_t,ndcg_t,index_end_i,indices_sort_top
    # pdb.set_trace() 
    return hr_t,ndcg_t
```

```python id="SPI2Ho8DKDzY"
def metrics(model, test_val_loader, top_k, num_negative_test_val, batch_size):
    HR, NDCG = [], [] 
    test_loss_sum=[]
    # pdb.set_trace()  
 
    test_start_time = time.time()
    for user, item_i, item_j in test_val_loader:  
        # start_time = time.time()
        # pdb.set_trace()
        user = user.cuda()
        item_i = item_i.cuda()
        item_j = item_j #index to split

        prediction_i, prediction_j,loss_test,loss2_test = model(user, item_i, torch.cuda.LongTensor([0])) 
        test_loss_sum.append(loss2_test.item())  
        # pdb.set_trace()   
        elapsed_time = time.time() - test_start_time
        print('time:'+str(round(elapsed_time,2)))
        courrent_index=0
        courrent_user_index=0
        for len_i,len_j in item_j:
            index_end_i=(len_i-len_j).item()  
            #pre_error=(prediction_i[0][courrent_index:(courrent_index+index_end_i)]- prediction_i[0][(courrent_index+index_end_i):(courrent_index+index_end_j)])#.sum() 
            #loss_test=nn.MSELoss((pre_error).sum())#-(prediction_i[0][courrent_index:(courrent_index+index_end_i)]- prediction_i[0][(courrent_index+index_end_i):(courrent_index+index_end_j)]).sigmoid().log()#.sum()   
            _, indices = torch.topk(prediction_i[0][courrent_index:(courrent_index+len_i)], top_k)   
            hr_t,ndcg_t=hr_ndcg(indices.tolist(),index_end_i,top_k)  
            # print(hr_t,ndcg_t,indices,index_end_i)
            # pdb.set_trace()
            HR.append(hr_t)
            NDCG.append(ndcg_t) 
            courrent_index+=len_i 
            courrent_user_index+=1 
    test_loss=round(np.mean(test_loss_sum[:-1]),4)  
    return test_loss,round(np.mean(HR),4) , round(np.mean(NDCG),4) 
```

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="UlSlchBlUAaI" executionInfo={"status": "ok", "timestamp": 1639038587411, "user_tz": -330, "elapsed": 550, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ccfccbfe-4b2d-4285-92f4-1f22cbe31ded"
os.path.join(path_save_base,'/training_set.npy')
```

```python id="ih-TuDTPyY6d"
training_user_set,training_item_set,training_set_count = np.load(os.path.join(path_save_base,'training_set.npy'),allow_pickle=True)
testing_user_set,testing_item_set,testing_set_count = np.load(os.path.join(path_save_base,'testing_set.npy'),allow_pickle=True)  
val_user_set,val_item_set,val_set_count = np.load(os.path.join(path_save_base,'val_set.npy'),allow_pickle=True)    
user_rating_set_all = np.load(os.path.join(path_save_base,'user_rating_set_all.npy'),allow_pickle=True).item()
```

```python id="fwB0JUa-yWRo"
def readD(set_matrix,num_):
    user_d=[] 
    for i in range(num_):
        len_set=1.0/(len(set_matrix[i])+1)  
        user_d.append(len_set)
    return user_d
```

```python id="rFv-2iRkKs6V"
u_d=readD(training_user_set,user_num)
i_d=readD(training_item_set,item_num)
#1/(d_i+1)
d_i_train=u_d
d_j_train=i_d
#1/sqrt((d_i+1)(d_j+1)) 
# d_i_j_train=np.sqrt(u_d*i_d) 
```

```python id="iyUc-TfcySm-"
#user-item  to user-item matrix and item-user matrix
def readTrainSparseMatrix(set_matrix,is_user):
    user_items_matrix_i=[]
    user_items_matrix_v=[] 
    if is_user:
        d_i=u_d
        d_j=i_d
    else:
        d_i=i_d
        d_j=u_d
    for i in set_matrix:
        len_set=len(set_matrix[i])  
        for j in set_matrix[i]:
            user_items_matrix_i.append([i,j])
            d_i_j=np.sqrt(d_i[i]*d_j[j])
            #1/sqrt((d_i+1)(d_j+1)) 
            user_items_matrix_v.append(d_i_j)#(1./len_set) 
    user_items_matrix_i=torch.cuda.LongTensor(user_items_matrix_i)
    user_items_matrix_v=torch.cuda.FloatTensor(user_items_matrix_v)
    return torch.sparse.FloatTensor(user_items_matrix_i.t(), user_items_matrix_v)
```

```python id="MSoboFoQyRHv"
sparse_u_i=readTrainSparseMatrix(training_user_set,True)
sparse_i_u=readTrainSparseMatrix(training_item_set,False)
#user-item  to user-item matrix and item-user matrix
# pdb.set_trace()
```

<!-- #region id="W3fawtyxMs3O" -->
## Model
<!-- #endregion -->

<!-- #region id="7OFu8P4HU2Dy" -->
![](https://github.com/RecoHut-Stanzas/S794944/raw/main/images/Overall_framework.jpg)
<!-- #endregion -->

```python id="KpquhpXeyKp3"
class BPR(nn.Module):
    def __init__(self, user_num, item_num, factor_num,user_item_matrix,item_user_matrix,d_i_train,d_j_train):
        super(BPR, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        """     
        self.user_item_matrix = user_item_matrix
        self.item_user_matrix = item_user_matrix
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num) 

        for i in range(len(d_i_train)):
            d_i_train[i]=[d_i_train[i]]
        for i in range(len(d_j_train)):
            d_j_train[i]=[d_j_train[i]]

        self.d_i_train=torch.cuda.FloatTensor(d_i_train)
        self.d_j_train=torch.cuda.FloatTensor(d_j_train)
        self.d_i_train=self.d_i_train.expand(-1,factor_num)
        self.d_j_train=self.d_j_train.expand(-1,factor_num)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)  

    def forward(self, user, item_i, item_j):    

        users_embedding=self.embed_user.weight
        items_embedding=self.embed_item.weight  

        gcn1_users_embedding = (torch.sparse.mm(self.user_item_matrix, items_embedding) + users_embedding.mul(self.d_i_train))#*2. #+ users_embedding
        gcn1_items_embedding = (torch.sparse.mm(self.item_user_matrix, users_embedding) + items_embedding.mul(self.d_j_train))#*2. #+ items_embedding
   
        gcn2_users_embedding = (torch.sparse.mm(self.user_item_matrix, gcn1_items_embedding) + gcn1_users_embedding.mul(self.d_i_train))#*2. + users_embedding
        gcn2_items_embedding = (torch.sparse.mm(self.item_user_matrix, gcn1_users_embedding) + gcn1_items_embedding.mul(self.d_j_train))#*2. + items_embedding
          
        gcn3_users_embedding = (torch.sparse.mm(self.user_item_matrix, gcn2_items_embedding) + gcn2_users_embedding.mul(self.d_i_train))#*2. + gcn1_users_embedding
        gcn3_items_embedding = (torch.sparse.mm(self.item_user_matrix, gcn2_users_embedding) + gcn2_items_embedding.mul(self.d_j_train))#*2. + gcn1_items_embedding
        
        # gcn4_users_embedding = (torch.sparse.mm(self.user_item_matrix, gcn3_items_embedding) + gcn3_users_embedding.mul(self.d_i_train))#*2. + gcn1_users_embedding
        # gcn4_items_embedding = (torch.sparse.mm(self.item_user_matrix, gcn3_users_embedding) + gcn3_items_embedding.mul(self.d_j_train))#*2. + gcn1_items_embedding
        
        gcn_users_embedding= torch.cat((users_embedding,gcn1_users_embedding,gcn2_users_embedding,gcn3_users_embedding),-1)#+gcn4_users_embedding
        gcn_items_embedding= torch.cat((items_embedding,gcn1_items_embedding,gcn2_items_embedding,gcn3_items_embedding),-1)#+gcn4_items_embedding#
      
        
        user = F.embedding(user,gcn_users_embedding)
        item_i = F.embedding(item_i,gcn_items_embedding)
        item_j = F.embedding(item_j,gcn_items_embedding)  
        # # pdb.set_trace() 
        prediction_i = (user * item_i).sum(dim=-1)
        prediction_j = (user * item_j).sum(dim=-1) 
        # loss=-((rediction_i-prediction_j).sigmoid())**2#self.loss(prediction_i,prediction_j)#.sum()
        l2_regulization = 0.01*(user**2+item_i**2+item_j**2).sum(dim=-1)
        # l2_regulization = 0.01*((gcn1_users_embedding**2).sum(dim=-1).mean()+(gcn1_items_embedding**2).sum(dim=-1).mean())
      
        loss2= -((prediction_i - prediction_j).sigmoid().log().mean())
        # loss= loss2 + l2_regulization
        loss= -((prediction_i - prediction_j)).sigmoid().log().mean() +l2_regulization.mean()
        # pdb.set_trace()
        return prediction_i, prediction_j,loss,loss2
```

```python id="wAwTA05SyNpP"
train_dataset = BPRData(
        train_dict=training_user_set, num_item=item_num, num_ng=5, is_training=True,\
        data_set_count=training_set_count,all_rating=user_rating_set_all)
train_loader = DataLoader(train_dataset,
        batch_size=batch_size, shuffle=True, num_workers=2)
  
testing_dataset_loss = BPRData(
        train_dict=testing_user_set, num_item=item_num, num_ng=5, is_training=True,\
        data_set_count=testing_set_count,all_rating=user_rating_set_all)
testing_loader_loss = DataLoader(testing_dataset_loss,
        batch_size=batch_size, shuffle=False, num_workers=0)

val_dataset_loss = BPRData(
        train_dict=val_user_set, num_item=item_num, num_ng=5, is_training=True,\
        data_set_count=val_set_count,all_rating=user_rating_set_all)
val_loader_loss = DataLoader(val_dataset_loss,
        batch_size=batch_size, shuffle=False, num_workers=0)
   
   
model = BPR(user_num, item_num, factor_num,sparse_u_i,sparse_i_u,d_i_train,d_j_train)
model=model.to('cuda') 

optimizer_bpr = torch.optim.Adam(model.parameters(), lr=0.005)#, betas=(0.5, 0.99))
```

<!-- #region id="SO9SZNWfMuh5" -->
## Training
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="D9ksZjDmyMBg" executionInfo={"status": "ok", "timestamp": 1639038912797, "user_tz": -330, "elapsed": 237974, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7cbe674b-26d3-407f-ed5a-890bf00428c9"
########################### TRAINING #####################################
 
# testing_loader_loss.dataset.ng_sample() 

print('--------training processing-------')
count, best_hr = 0, 0
for epoch in range(5):
    model.train() 
    start_time = time.time()
    train_loader.dataset.ng_sample()
    # pdb.set_trace()
    print('train data of ng_sample is  end')
    # elapsed_time = time.time() - start_time
    # print(' time:'+str(round(elapsed_time,1)))
    # start_time = time.time()
    
    train_loss_sum=[]
    train_loss_sum2=[]
    for user, item_i, item_j in train_loader:
        user = user.cuda()
        item_i = item_i.cuda()
        item_j = item_j.cuda() 

        model.zero_grad()
        prediction_i, prediction_j,loss,loss2 = model(user, item_i, item_j) 
        loss.backward()
        optimizer_bpr.step() 
        count += 1  
        train_loss_sum.append(loss.item())  
        train_loss_sum2.append(loss2.item())  
        # print(count)

    elapsed_time = time.time() - start_time
    train_loss=round(np.mean(train_loss_sum[:-1]),4)
    train_loss2=round(np.mean(train_loss_sum2[:-1]),4)
    str_print_train="epoch:"+str(epoch)+' time:'+str(round(elapsed_time,1))+'\t train loss:'+str(train_loss)+"="+str(train_loss2)+"+" 
    print('--train--',elapsed_time)

    PATH_model=path_save_model_base+'/epoch'+str(epoch)+'.pt'
    torch.save(model.state_dict(), PATH_model)
    
    model.eval()   
    # ######test and val###########   
    val_loader_loss.dataset.ng_sample() 
    val_loss=metrics_loss(model,val_loader_loss,batch_size)  
    # str_print_train+=' val loss:'+str(val_loss)

    testing_loader_loss.dataset.ng_sample() 
    test_loss=metrics_loss(model,testing_loader_loss,batch_size) 
    print(str_print_train+' val loss:'+str(val_loss)+' test loss:'+str(test_loss)) 
    result_file.write(str_print_train+' val loss:'+str(val_loss)+' test loss:'+str(test_loss)) 
    result_file.write('\n') 
    result_file.flush() 
```

<!-- #region id="jxYefqWOzUSX" -->
## Testing
<!-- #endregion -->

```python id="kWWePLOTVxSg"
def readD(set_matrix,num_):
    user_d=[] 
    for i in range(num_):
        len_set=1.0/(len(set_matrix[i])+1)  
        user_d.append(len_set)
    return user_d
u_d=readD(training_user_set,user_num)
i_d=readD(training_item_set,item_num)
d_i_train=u_d
d_j_train=i_d
```

```python id="ONukrcRw2ngS"
#user-item  to user-item matrix and item-user matrix
def readTrainSparseMatrix(set_matrix,is_user):
    user_items_matrix_i=[]
    user_items_matrix_v=[] 
    if is_user:
        d_i=u_d
        d_j=i_d
    else:
        d_i=i_d
        d_j=u_d
    for i in set_matrix:
        len_set=len(set_matrix[i])  
        for j in set_matrix[i]:
            user_items_matrix_i.append([i,j])
            d_i_j=np.sqrt(d_i[i]*d_j[j])
            #1/sqrt((d_i+1)(d_j+1)) 
            user_items_matrix_v.append(d_i_j)#(1./len_set) 
    user_items_matrix_i=torch.cuda.LongTensor(user_items_matrix_i)
    user_items_matrix_v=torch.cuda.FloatTensor(user_items_matrix_v)
    return torch.sparse.FloatTensor(user_items_matrix_i.t(), user_items_matrix_v)

sparse_u_i=readTrainSparseMatrix(training_user_set,True)
sparse_i_u=readTrainSparseMatrix(training_item_set,False)

#user-item  to user-item matrix and item-user matrix
# pdb.set_trace()
```

```python id="37odx_9d2xCS"
class BPR(nn.Module):
    def __init__(self, user_num, item_num, factor_num,user_item_matrix,item_user_matrix,d_i_train,d_j_train):
        super(BPR, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        """     
        self.user_item_matrix = user_item_matrix
        self.item_user_matrix = item_user_matrix
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num) 

        for i in range(len(d_i_train)):
            d_i_train[i]=[d_i_train[i]]
        for i in range(len(d_j_train)):
            d_j_train[i]=[d_j_train[i]]

        self.d_i_train=torch.cuda.FloatTensor(d_i_train)
        self.d_j_train=torch.cuda.FloatTensor(d_j_train)
        self.d_i_train=self.d_i_train.expand(-1,factor_num)
        self.d_j_train=self.d_j_train.expand(-1,factor_num)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)  

    def forward(self, user, item_i, item_j):    

        users_embedding=self.embed_user.weight
        items_embedding=self.embed_item.weight  

        gcn1_users_embedding = (torch.sparse.mm(self.user_item_matrix, items_embedding) + users_embedding.mul(self.d_i_train))#*2. #+ users_embedding
        gcn1_items_embedding = (torch.sparse.mm(self.item_user_matrix, users_embedding) + items_embedding.mul(self.d_j_train))#*2. #+ items_embedding
   
        gcn2_users_embedding = (torch.sparse.mm(self.user_item_matrix, gcn1_items_embedding) + gcn1_users_embedding.mul(self.d_i_train))#*2. + users_embedding
        gcn2_items_embedding = (torch.sparse.mm(self.item_user_matrix, gcn1_users_embedding) + gcn1_items_embedding.mul(self.d_j_train))#*2. + items_embedding
          
        gcn3_users_embedding = (torch.sparse.mm(self.user_item_matrix, gcn2_items_embedding) + gcn2_users_embedding.mul(self.d_i_train))#*2. + gcn1_users_embedding
        gcn3_items_embedding = (torch.sparse.mm(self.item_user_matrix, gcn2_users_embedding) + gcn2_items_embedding.mul(self.d_j_train))#*2. + gcn1_items_embedding
       
        gcn_users_embedding= torch.cat((users_embedding,gcn1_users_embedding,gcn2_users_embedding,gcn3_users_embedding),-1)#+gcn4_users_embedding
        gcn_items_embedding= torch.cat((items_embedding,gcn1_items_embedding,gcn2_items_embedding,gcn3_items_embedding),-1)#+gcn4_items_embedding#
        
        
        g0_mean=torch.mean(users_embedding)
        g0_var=torch.var(users_embedding)
        g1_mean=torch.mean(gcn1_users_embedding)
        g1_var=torch.var(gcn1_users_embedding) 
        g2_mean=torch.mean(gcn2_users_embedding)
        g2_var=torch.var(gcn2_users_embedding)
        g3_mean=torch.mean(gcn3_users_embedding)
        g3_var=torch.var(gcn3_users_embedding)
        # g4_mean=torch.mean(gcn4_users_embedding)
        # g4_var=torch.var(gcn4_users_embedding)
        # g5_mean=torch.mean(gcn5_users_embedding)
        # g5_var=torch.var(gcn5_users_embedding)
        # g6_mean=torch.mean(gcn6_users_embedding)
        # g6_var=torch.var(gcn6_users_embedding)
        g_mean=torch.mean(gcn_users_embedding)
        g_var=torch.var(gcn_users_embedding)

        i0_mean=torch.mean(items_embedding)
        i0_var=torch.var(items_embedding)
        i1_mean=torch.mean(gcn1_items_embedding)
        i1_var=torch.var(gcn1_items_embedding)
        i2_mean=torch.mean(gcn2_items_embedding)
        i2_var=torch.var(gcn2_items_embedding)
        i3_mean=torch.mean(gcn3_items_embedding)
        i3_var=torch.var(gcn3_items_embedding)
        # i4_mean=torch.mean(gcn4_items_embedding)
        # i4_var=torch.var(gcn4_items_embedding) 
        # i5_mean=torch.mean(gcn5_items_embedding)
        # i5_var=torch.var(gcn5_items_embedding)
        # i6_mean=torch.mean(gcn6_items_embedding)
        # i6_var=torch.var(gcn6_items_embedding)
        i_mean=torch.mean(gcn_items_embedding)
        i_var=torch.var(gcn_items_embedding)

        # pdb.set_trace() 

        str_user=str(round(g0_mean.item(),7))+' '
        str_user+=str(round(g0_var.item(),7))+' '
        str_user+=str(round(g1_mean.item(),7))+' '
        str_user+=str(round(g1_var.item(),7))+' '
        str_user+=str(round(g2_mean.item(),7))+' '
        str_user+=str(round(g2_var.item(),7))+' '
        str_user+=str(round(g3_mean.item(),7))+' '
        str_user+=str(round(g3_var.item(),7))+' '
        # str_user+=str(round(g4_mean.item(),7))+' '
        # str_user+=str(round(g4_var.item(),7))+' '
        # str_user+=str(round(g5_mean.item(),7))+' '
        # str_user+=str(round(g5_var.item(),7))+' '
        # str_user+=str(round(g6_mean.item(),7))+' '
        # str_user+=str(round(g6_var.item(),7))+' '
        str_user+=str(round(g_mean.item(),7))+' '
        str_user+=str(round(g_var.item(),7))+' '

        str_item=str(round(i0_mean.item(),7))+' '
        str_item+=str(round(i0_var.item(),7))+' '
        str_item+=str(round(i1_mean.item(),7))+' '
        str_item+=str(round(i1_var.item(),7))+' '
        str_item+=str(round(i2_mean.item(),7))+' '
        str_item+=str(round(i2_var.item(),7))+' '
        str_item+=str(round(i3_mean.item(),7))+' '
        str_item+=str(round(i3_var.item(),7))+' '
        # str_item+=str(round(i4_mean.item(),7))+' '
        # str_item+=str(round(i4_var.item(),7))+' '
        # str_item+=str(round(i5_mean.item(),7))+' '
        # str_item+=str(round(i5_var.item(),7))+' '
        # str_item+=str(round(i6_mean.item(),7))+' '
        # str_item+=str(round(i6_var.item(),7))+' '
        str_item+=str(round(i_mean.item(),7))+' '
        str_item+=str(round(i_var.item(),7))+' '
        print(str_user)
        print(str_item)
        return gcn_users_embedding, gcn_items_embedding,str_user,str_item 

test_batch=52#int(batch_size/32) 
testing_dataset = resData(train_dict=testing_user_set, batch_size=test_batch,num_item=item_num,all_pos=training_user_set)
testing_loader = DataLoader(testing_dataset,batch_size=1, shuffle=False, num_workers=0) 
 
model = BPR(user_num, item_num, factor_num,sparse_u_i,sparse_i_u,d_i_train,d_j_train)
model=model.to('cuda')
   
optimizer_bpr = torch.optim.Adam(model.parameters(), lr=0.001)#, betas=(0.5, 0.99))
```

```python colab={"base_uri": "https://localhost:8080/"} id="LbN03HqY2scu" executionInfo={"status": "ok", "timestamp": 1639039232397, "user_tz": -330, "elapsed": 164076, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c20f2291-2639-45ea-a4a9-ad4f148f724a"
########################### TESTING ##################################### 
# testing_loader_loss.dataset.ng_sample() 

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

print('--------test processing-------')
count, best_hr = 0, 0
for epoch in range(start_i_test,end_i_test,setp):
    model.train()   

    PATH_model=path_save_model_base+'/epoch'+str(epoch)+'.pt'
    #torch.save(model.state_dict(), PATH_model) 
    model.load_state_dict(torch.load(PATH_model)) 
    model.eval()     
    # ######test and val###########    
    gcn_users_embedding, gcn_items_embedding,gcn_user_emb,gcn_item_emb= model(torch.cuda.LongTensor([0]), torch.cuda.LongTensor([0]), torch.cuda.LongTensor([0])) 
    user_e=gcn_users_embedding.cpu().detach().numpy()
    item_e=gcn_items_embedding.cpu().detach().numpy()
    all_pre=np.matmul(user_e,item_e.T) 
    HR, NDCG = [], [] 
    set_all=set(range(item_num))  
    #spend 461s 
    test_start_time = time.time()
    for u_i in testing_user_set: 
        item_i_list = list(testing_user_set[u_i])
        index_end_i=len(item_i_list)
        item_j_list = list(set_all-training_user_set[u_i]-testing_user_set[u_i])
        item_i_list.extend(item_j_list) 

        pre_one=all_pre[u_i][item_i_list] 
        indices=largest_indices(pre_one, top_k)
        indices=list(indices[0])   

        hr_t,ndcg_t=hr_ndcg(indices,index_end_i,top_k) 
        elapsed_time = time.time() - test_start_time 
        HR.append(hr_t)
        NDCG.append(ndcg_t)    
    hr_test=round(np.mean(HR),4)
    ndcg_test=round(np.mean(NDCG),4)    
        
    # test_loss,hr_test,ndcg_test = metrics(model,testing_loader,top_k,num_negative_test_val,batch_size)  
    str_print_evl="epoch:"+str(epoch)+'time:'+str(round(elapsed_time,2))+"\t test"+" hit:"+str(hr_test)+' ndcg:'+str(ndcg_test) 
    print(str_print_evl)   
    result_file.write(gcn_user_emb)
    result_file.write('\n')
    result_file.write(gcn_item_emb)
    result_file.write('\n')  

    result_file.write(str_print_evl)
    result_file.write('\n')
    result_file.flush()
```

<!-- #region id="TLHw8weh7OQB" -->
---
<!-- #endregion -->

```python id="ZYGZlyeY7OQD"
!apt-get -qq install tree
!rm -r sample_data
```

```python colab={"base_uri": "https://localhost:8080/"} id="oVruV5yt7OQD" executionInfo={"status": "ok", "timestamp": 1638797225389, "user_tz": -330, "elapsed": 24, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8cc165b8-6167-4329-c3f1-5cd2ae01d647"
!tree -h --du .
```

```python colab={"base_uri": "https://localhost:8080/"} id="MrbNlTOW7OQE" executionInfo={"status": "ok", "timestamp": 1638797236830, "user_tz": -330, "elapsed": 3692, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b4f06826-9df0-476d-ecb4-ff603183437b"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

```python id="jzgrKrkC7Y-N" executionInfo={"status": "ok", "timestamp": 1638797255976, "user_tz": -330, "elapsed": 448, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="672fccb0-588b-49ca-8a77-5ab549796055" colab={"base_uri": "https://localhost:8080/"}
!nvidia-smi
```

<!-- #region id="9Kw61_pe7OQE" -->
---
<!-- #endregion -->

<!-- #region id="OCasCymq7OQG" -->
**END**
<!-- #endregion -->
