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

<!-- #region id="pSyOXuQv7RGx" -->
# Building models from scratch
> Applying various MF and Deep learning recommender models on movielens 100k

- toc: true
- badges: true
- comments: true
- categories: [Movie, AutoEncoder, Tensorflow]
- image:
<!-- #endregion -->

<!-- #region id="WCAZqdXHp6P1" -->
## Setup
<!-- #endregion -->

<!-- #region id="nyQlzit4qwZg" -->
### Imports
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="PwfX0h11vXPt" outputId="dce7beb1-b779-4c1f-cbf4-bbab130e4ed0"
%tensorflow_version 1.x
```

```python id="7kMaXO6ypymg"
import math
import numpy as np
import pandas as pd

from pylab import *
import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf

%matplotlib inline
```

<!-- #region id="4xYEBXL2qdp4" -->
### Data utils
<!-- #endregion -->

```python id="l9Tbu8YbqcCZ"
def load_data(file_dir):
    # output: 
    # N: the number of user;
    # M: the number of item
    # data: the list of rating information
    user_ids_dict, rated_item_ids_dict = {},{}
    N, M, u_idx, i_idx = 0,0,0,0 
    data = []
    f = open(file_dir)
    for line in f.readlines():
        if '::' in line:
            u, i, r, _ = line.split('::')
        else:
            u, i, r, _ = line.split()
    
        if int(u) not in user_ids_dict:
            user_ids_dict[int(u)]=u_idx
            u_idx+=1
        if int(i) not in rated_item_ids_dict:
            rated_item_ids_dict[int(i)]=i_idx
            i_idx+=1
        data.append([user_ids_dict[int(u)],rated_item_ids_dict[int(i)],float(r)])
    
    f.close()
    N = u_idx
    M = i_idx

    return N, M, data, rated_item_ids_dict
	

def sequence2mat(sequence, N, M):
    # input:
    # sequence: the list of rating information
    # N: row number, i.e. the number of users
    # M: column number, i.e. the number of items
    # output:
    # mat: user-item rating matrix
    records_array = np.array(sequence)
    mat = np.zeros([N,M])
    row = records_array[:,0].astype(int)
    col = records_array[:,1].astype(int)
    values = records_array[:,2].astype(np.float32)
    mat[row,col]=values
    
    return mat
```

<!-- #region id="Knvcmsydq2bU" -->
### Data download
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="4Uw0eStMq3s1" outputId="262a7b15-fb4a-482e-9131-8c660a605590"
!wget http://files.grouplens.org/datasets/movielens/ml-100k.zip
!unzip ml-100k.zip
```

<!-- #region id="sYcCrA4fuWlT" -->
### Data loading
<!-- #endregion -->

```python id="esRh9cRgw9ud"
data_col = ['user_id','item_id','rating','timestamp']
 
item_col = ['movie_id','movie_title','release_date','video_release_date','IMDb_URL','unknown','Action',
           'Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy',
           'Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
 
data_dir = 'ml-100k/u.data' 
data = pd.read_table(data_dir,header=None, names=data_col, parse_dates=['timestamp'])
 
item_dir = 'ml-100k/u.item'
item = pd.read_table(item_dir, header=None, names=item_col, parse_dates=['release_date','video_release_date'], encoding='ISO-8859-1', sep='|')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 360} id="iTkDPpl5w9pt" outputId="467a633e-50dd-4ff7-fc0c-241e76e24e1e"
item.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="LrgynatixM1-" outputId="a87217ad-dafe-4e91-9316-c4b07e0a4a6b"
data.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="8FRK6CykuV1a" outputId="01ace7ca-0912-4aee-92ba-c8415df28b7f"
data_dir = 'ml-100k/u.data' 
N, M, data_list, _ = load_data(file_dir=data_dir)
print(' data length: %d \n user number: %d \n item number: %d' %(len(data_list),N,M))
```

<!-- #region id="WA4hLCNHugGn" -->
### Train test split
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="AJsuK4squcX8" outputId="2ea11ef0-df76-4d69-c482-16f24b4129c9"
train_list, test_list = train_test_split(data_list,test_size=0.2)
print ('train length: %d \n test length: %d' %(len(train_list),len(test_list)))
```

```python colab={"base_uri": "https://localhost:8080/"} id="S4EXPPHJucUT" outputId="1dc97e2d-1c8b-4f5b-ff72-5a72dafbbfb5"
train_mat = sequence2mat(sequence = train_list, N = N, M = M)
test_mat = sequence2mat(sequence = test_list, N = N, M = M)
train_mat[0,:100]
```

<!-- #region id="VVkAtg8OqVM3" -->
### Evaluation policy
<!-- #endregion -->

```python id="y9AcNHGGqKjs"
def get_topn(r_pred, train_mat, n=10):
    unrated_items = r_pred * (train_mat==0)
    idx = np.argsort(-unrated_items)
    return idx[:,:n]


def recall_precision(topn, test_mat):
    n,m = test_mat.shape
    hits,total_pred,total_true = 0.,0.,0.
    for u in range(n):
        hits += len([i for i in topn[u,:] if test_mat[u,i]>0])
        size_pred = len(topn[u,:])
        size_true = np.sum(test_mat[u,:]>0,axis=0)
        total_pred += size_pred
        total_true += size_true

    recall = hits/total_true
    precision = hits/total_pred
    return recall, precision	
	
	
def mae_rmse(r_pred, test_mat):
    y_pred = r_pred[test_mat>0]
    y_true = test_mat[test_mat>0]
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse 


def evaluation(pred_mat, train_mat, test_mat):
    topn = get_topn(pred_mat, train_mat, n=10)
    mae, rmse = mae_rmse(pred_mat, test_mat)
    recall, precision = recall_precision(topn, test_mat)
    return mae, rmse, recall, precision
	
	
def get_hit(ranklist,rated_item):
    result = 0
    for item in ranklist:
        if item==rated_item:
            result = 1
    return result
    
    
def get_ndcg(ranklist,rated_item):
    result = 0
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item==rated_item:
            result = math.log(2)/math.log(i+2)
    return result


def hit_ndcg(test_sequence, ranklist):
    length = len(test_sequence)
    hits,ndcgs=[],[]
    for idx in range(length):
        user = test_sequence[idx,0].astype(np.int32)
        rated_item = test_sequence[idx,1].astype(np.int32)
        hr = get_hit(ranklist[user],rated_item)
        ndcg = get_ndcg(ranklist[user],rated_item)
        hits.append(hr)
        ndcgs.append(ndcg)
    #hr,ndcg = np.array(hits).mean(),np.array(ndcgs).mean()
    return hits,ndcgs	
```

```python id="mlC5tvsGvNwh"
def plot_figure(values_list, name=''):
    fig=plt.figure(name)
    x = range(len(values_list))
    plot(x, values_list, color='g',linewidth=3)
    plt.title(name + ' curve')
    plt.xlabel('Iterations')
    plt.ylabel(name)
    show()
```

<!-- #region id="FGkcdMF5xsRv" -->
## KNN
<!-- #endregion -->

<!-- #region id="vw652KuS4SxL" -->
<!-- #endregion -->

```python id="flScGexcxuQS"
def calculate_similarity(a, b, model='pearson', minimum_common_items=5):
    assert a.shape==b.shape
    dim = len(a.shape) #向量维度
    common_items = a*b>0 # 共同评分的项
    common_size = np.sum(common_items,axis=dim-1)
    
    if model=='pearson':
        mean_a = np.sum(a,axis=dim-1)/np.sum(a>0,axis=dim-1)
        mean_b = np.sum(b,axis=dim-1)/np.sum(b>0,axis=dim-1)
        if dim ==1:
            aa = (a - mean_a)*common_items
            bb = (b - mean_b)*common_items
        else:
            aa = (a - np.reshape(mean_a, (-1,1)))*common_items
            bb = (b - np.reshape(mean_b, (-1,1)))*common_items
    else: #consine
        mean_u = np.sum(b,axis=0)/np.sum(b>0,axis=0)
        aa = (a - mean_u)*common_items
        bb = (b - mean_u)*common_items
        
    sim = np.sum(aa*bb, axis=dim-1)/(np.sqrt(np.sum(aa**2, axis=dim-1))*np.sqrt(np.sum(bb**2, axis=dim-1)) + 1e-10)
    least_common_items = common_size>minimum_common_items
    return sim*least_common_items
```

```python id="0XO5F1q9xuMz"
def similarity_matrix(mat, model='pearson', minimum_common_items=5):
    n,m = mat.shape
    sim_list=[]
    for u in range(n):
        a = np.tile(mat[u,:], (n,1))
        b = mat
        if model=='pearson':
            sim = calculate_similarity(a, b, model='pearson', minimum_common_items=minimum_common_items)
        else: # consine
            sim = calculate_similarity(a, b, model='consine', minimum_common_items=minimum_common_items)
        sim_list.append(sim)
        if u % 100 ==0:
            print(u)
    return np.array(sim_list)
```

```python colab={"base_uri": "https://localhost:8080/"} id="we_GmLgdxuHu" outputId="df187034-0314-4ee5-9aa7-7977ae4cee53"
sim_mat = similarity_matrix(mat=train_mat, model='pearson')
neighbors = np.argsort(-np.array(sim_mat))
sim_sort = -1*np.sort(-np.array(sim_mat))
```

```python colab={"base_uri": "https://localhost:8080/"} id="JD0QuQLaxuDT" outputId="36ff178e-6d98-45bf-e7b4-9a2b836355db"
np.set_printoptions(precision=4, suppress=True)
print('user 0:')
print('neighbors:') # 用户0的近邻
print(neighbors[0,:10])
print('sim:\n') # 用户0 的近邻相似度
print(sim_sort[0,:10])
print('similarity_mat:') # 用户之间的相似度矩阵
print(sim_mat[:6,:6])
```

```python id="8Yg5OGiqxt_a"
def get_K(sim_mat, min_similarity=0.5):
    num = np.sum(sim_mat[:,1:]>min_similarity, axis=1)
    plt.rcParams['font.sans-serif'] = [u'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.hist(num, bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.show()
    
    num_sort = np.sort(-num)
    line = int(0.8*len(sim_mat))
    K = -1*num_sort[line]
    return K
```

```python colab={"base_uri": "https://localhost:8080/", "height": 282} id="ravLCJALxt6g" outputId="801955ef-2ff3-4a51-d29d-7fa4daa33963"
min_similarity=0.8
K = get_K(sim_mat, min_similarity=min_similarity)
print('min_similarity:',min_similarity,'K:',K)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 282} id="y9pasQ6lxt2L" outputId="d8aad146-3962-42d9-c64c-f1dc548461c5"
min_similarity=0.5
K = get_K(sim_mat, min_similarity=min_similarity)
print('min_similarity:',min_similarity,'K:',K)
```

```python id="HhPgdztQyReq"
def prediction(train_mat, sim_mat, K=1, model='user_based'):
    assert len(train_mat.shape)>1
    n,m = train_mat.shape
    
    if  model=='user_based':
        sim_sort = -1*np.sort(-np.array(sim_mat))[:,1:K+1] # 除去最相似的自己
        neighbors = np.argsort(-np.array(sim_mat))[:,1:K+1]
        common_items = train_mat[neighbors]>0 
        mean_user = np.reshape(np.sum(train_mat,axis=1)/np.sum(train_mat>0,axis=1), (-1,1))
        mat_m = train_mat - mean_user
        aa = np.sum(sim_sort[:,:,np.newaxis]*mat_m[neighbors]*common_items,axis=1)
        bb = np.sum(sim_sort[:,:,np.newaxis]*common_items,axis=1)+1e-10 # 1e-10保证分母不为０
        r_pred = mean_user + aa/bb
        return r_pred
    else: # 'item_based'
        r_pred=[]
        for u in range(n):
            u_mat = np.tile(train_mat[u],(m,1)) # m份用户u的记录,m*m
            rated_items_sim = (u_mat>0)*sim_mat # 保留有评分记录的相似度 m*m
            sim_sort = -1*np.sort(-np.array(rated_items_sim))[:,:K] # m*K
            neighbors = np.argsort(-np.array(rated_items_sim))[:,:K] # m*K
            neighbor_ratings = np.array([u_mat[i,neighbors[i]] for i in range(m)])# m*K
            aa = np.sum(sim_sort*neighbor_ratings,axis=1) # m*1
            bb = np.sum(sim_sort,axis=1)+1e-10 # 1e-10保证分母不为０ m*1
            r_pred.append(aa/bb)
        
        return np.array(r_pred)
```

```python id="pNNaRn04yRag"
r_pred = prediction(train_mat=train_mat, sim_mat=sim_mat, K=K, model='user_based')
```

```python colab={"base_uri": "https://localhost:8080/"} id="aOWmxpWGyRWz" outputId="2ae7bb61-22e3-497f-e096-5506a27e6e71"
n = 10
topn = get_topn(r_pred=r_pred, train_mat=train_mat, n=n)
print('user 0:')
print('top-n list:',topn[0])
```

```python colab={"base_uri": "https://localhost:8080/"} id="4Fr1dzohyRR7" outputId="690e03a9-4c93-433c-c7f9-fe248dcd6220"
mae, rmse = mae_rmse(r_pred=r_pred, test_mat=test_mat)
print('mae:%.4f; rmse:%.4f'%(mae,rmse))
recall, precision = recall_precision(topn=topn, test_mat=test_mat)
print('recall:%.4f; precision:%.4f'%(recall,precision))
```

```python colab={"base_uri": "https://localhost:8080/"} id="a38qGqSKyaGc" outputId="9e7cc10c-0545-44ac-b99c-ed4db9349082"
sim_mat = similarity_matrix(mat=train_mat.T, model='consine', minimum_common_items=3)
neighbors = np.argsort(-np.array(sim_mat))
sim_sort = -1*np.sort(-np.array(sim_mat))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 282} id="7n1GsgusyaCF" outputId="24368bde-3a0e-48db-f279-2d38c3b1144d"
min_similarity = 0.5
K = get_K(sim_mat, min_similarity=min_similarity)
print('min_similarity:',min_similarity,'K:',K)
```

```python colab={"base_uri": "https://localhost:8080/"} id="4m_lc4feyjqX" outputId="b95b1f82-87a8-4752-dca6-cb8ea7d4e5bb"
r_pred = prediction(train_mat=train_mat, sim_mat=sim_mat, K=5, model='item_based')

n = 10
topn = get_topn(r_pred=r_pred, train_mat=train_mat, n=n)

mae, rmse = mae_rmse(r_pred=r_pred, test_mat=test_mat)
print('mae:%.4f; rmse:%.4f'%(mae,rmse))
recall, precision = recall_precision(topn=topn, test_mat=test_mat)
print('recall:%.4f; precision:%.4f'%(recall,precision))
```

<!-- #region id="Y3sZ87x-1HgB" -->
## MF
<!-- #endregion -->

<!-- #region id="dNM4DxgN4tDC" -->
<!-- #endregion -->

```python id="Eaapp4qL1KOR"
class mf():
    def __init__(self, 
                 train_list,            # train_list: train data 
                 test_list,             # test_list: test data
                 N,                     # N:the number of user
                 M,                     # M:the number of item
                 K=10,                  # K: the number of latent factor
                 learning_rate=0.001,   # learning_rate: the learning rata
                 lamda_regularizer=0.1, # lamda_regularizer: regularization parameters
                 max_iteration=50       # max_iteration: the max iteration
                ):
        self.train_list = train_list
        self.test_list = test_list
        self.N = N
        self.M = M
        self.K = K
        self.learning_rate = learning_rate
        self.lamda_regularizer = lamda_regularizer
        self.max_iteration = max_iteration
    
    
    def train(self):
        P = np.random.normal(0, 0.1, (self.N, self.K))
        Q = np.random.normal(0, 0.1, (self.M, self.K))

        train_mat = sequence2mat(sequence = self.train_list, N = self.N, M = self.M)
        test_mat = sequence2mat(sequence = self.test_list, N = self.N, M = self.M)

        records_list = []
        for step in range(self.max_iteration):
            los=0.0
            for data in self.train_list:
                u,i,r = data
                P[u],Q[i],ls = self.update(P[u], Q[i], r=r, 
                                           learning_rate=self.learning_rate, 
                                           lamda_regularizer=self.lamda_regularizer)
                los += ls
            pred_mat = self.prediction(P,Q)
            mae, rmse, recall, precision = evaluation(pred_mat, train_mat, test_mat)
            records_list.append(np.array([los, mae, rmse, recall, precision]))

            if step % 10 ==0:
                print(' step:%d \n loss:%.4f,mae:%.4f,rmse:%.4f,recall:%.4f,precision:%.4f'
                      %(step,los,mae,rmse,recall,precision))

        print(' end. \n loss:%.4f,mae:%.4f,rmse:%.4f,recall:%.4f,precision:%.4f'
              %(records_list[-1][0],records_list[-1][1],records_list[-1][2],records_list[-1][3],records_list[-1][4]))
        return P, Q, np.array(records_list)


    def update(self, p, q, r, learning_rate=0.001, lamda_regularizer=0.1):
        error = r - np.dot(p, q.T)            
        p = p + learning_rate*(error*q - lamda_regularizer*p)
        q = q + learning_rate*(error*p - lamda_regularizer*q)
        loss = 0.5 * (error**2 + lamda_regularizer*(np.square(p).sum() + np.square(q).sum()))
        return p, q, loss


    def prediction(self, P, Q):
        N,K = P.shape
        M,K = Q.shape

        rating_list=[]
        for u in range(N):
            u_rating = np.sum(P[u,:]*Q, axis=1)
            rating_list.append(u_rating)
        r_pred = np.array(rating_list)
        return r_pred
```

```python colab={"base_uri": "https://localhost:8080/"} id="Q39mNlGc1KJ6" outputId="d32b6bb7-c56d-4c3f-ac42-6ade462c3b3a"
model = mf(train_list=train_list, 
           test_list=test_list, 
           N=N, 
           M=M,
           K=K,
           learning_rate=learning_rate, 
           lamda_regularizer=lamda_regularizer, 
           max_iteration=max_iteration)
P, Q, records_array = model.train()
print('MAE:%.4f;RMSE:%.4f;Recall:%.4f;Precision:%.4f'
      %(records_array[:,1][-1],records_array[:,2][-1],records_array[:,3][-1],records_array[:,4][-1]))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 295} id="Kh12Jqwx1KGW" outputId="c8e754fc-bc08-4a17-a9c9-88da926c39f7"
plot_figure(values_list=records_array[:,0],name='loss')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 295} id="iah43b0s1QRL" outputId="a42f0cea-4bae-4745-d954-9316f47d9a73"
plot_figure(values_list=records_array[:,1],name='MAE')
```

<!-- #region id="fViwyvbAzSsS" -->
## NMF
<!-- #endregion -->

<!-- #region id="2eWIFoQ55B9s" -->
<!-- #endregion -->

```python id="MjWHQv0Bze3q"
class nmf_sgd():
    def __init__(self, 
                 train_list,            # train_list: train data 
                 test_list,             # test_list: test data
                 N,                     # N:the number of user
                 M,                     # M:the number of item
                 K=10,                  # K: the number of latent factor
                 learning_rate=0.001,   # learning_rate: the learning rata
                 max_iteration=50       # max_iteration: the max iteration
                ):
        self.train_list = train_list
        self.test_list = test_list
        self.N = N
        self.M = M
        self.K = K
        self.learning_rate = learning_rate
        self.max_iteration = max_iteration
    
    
    def train(self):
        P = np.random.normal(0, 0.1, (self.N, self.K))
        Q = np.random.normal(0, 0.1, (self.M, self.K))

        train_mat = sequence2mat(sequence = self.train_list, N = self.N, M = self.M)
        test_mat = sequence2mat(sequence = self.test_list, N = self.N, M = self.M)

        records_list = []
        for step in range(self.max_iteration):
            los=0.0
            for data in self.train_list:
                u,i,r = data
                P[u],Q[i],ls = self.update(P[u], Q[i], r=r, learning_rate=self.learning_rate)
                los += ls
            pred_mat = self.prediction(P,Q)
            mae, rmse, recall, precision = evaluation(pred_mat, train_mat, test_mat)
            records_list.append(np.array([los, mae, rmse, recall, precision]))

            if step % 10 ==0:
                print(' step:%d \n loss:%.4f,mae:%.4f,rmse:%.4f,recall:%.4f,precision:%.4f'
                      %(step,los,mae,rmse,recall,precision))

        print(' end. \n loss:%.4f,mae:%.4f,rmse:%.4f,recall:%.4f,precision:%.4f'
              %(records_list[-1][0],records_list[-1][1],records_list[-1][2],records_list[-1][3],records_list[-1][4]))
        return P,Q,np.array(records_list)


    def update(self, p, q, r, learning_rate=0.001):
        error = r - np.dot(p, q.T)            
        p = p + learning_rate*error*q
        q = q + learning_rate*error*p
        loss = 0.5 * error**2 
        return p, q, loss


    def prediction(self, P, Q):
        N,K = P.shape
        M,K = Q.shape

        rating_list=[]
        for u in range(N):
            u_rating = np.sum(P[u,:]*Q, axis=1)
            rating_list.append(u_rating)
        r_pred = np.array(rating_list)
        return r_pred
		
		
class nmf_mult():
    def __init__(self, 
                 train_list,            # train_list: train data 
                 test_list,             # test_list: test data
                 N,                     # N:the number of user
                 M,                     # M:the number of item
                 K=10,                  # K: the number of latent factor
                 max_iteration=50       # max_iteration: the max iteration
                ):
        self.train_list = train_list
        self.test_list = test_list
        self.N = N
        self.M = M
        self.K = K
        self.max_iteration = max_iteration
    
    
    def train(self):
        train_mat = sequence2mat(sequence = self.train_list, N = self.N, M = self.M)
        test_mat = sequence2mat(sequence = self.test_list, N = self.N, M = self.M)

        avg = np.sqrt(train_mat.mean() / self.K)
        P = avg*np.random.normal(0, 1., (self.N, self.K))
        Q = avg*np.random.normal(0, 1., (self.M, self.K))

        records_list = []
        for step in range(self.max_iteration):
            P,Q = self.update(P, Q, R=train_mat)
            user = np.array(self.train_list)[:,0].astype(np.int16)
            item = np.array(self.train_list)[:,1].astype(np.int16)
            rating_true = np.array(self.train_list)[:,2]
            rating_pred = np.sum(P[user,:]*Q[item,:],axis=1)
            los = np.sum((rating_true-rating_pred)**2)
            pred_mat = self.prediction(P,Q)
            mae, rmse, recall, precision = evaluation(pred_mat, train_mat, test_mat)
            records_list.append(np.array([los, mae, rmse, recall, precision]))

            if step % 10 ==0:
                print(' step:%d \n loss:%.4f,mae:%.4f,rmse:%.4f,recall:%.4f,precision:%.4f'
                      %(step,los,mae,rmse,recall,precision))

        print(' end. \n loss:%.4f,mae:%.4f,rmse:%.4f,recall:%.4f,precision:%.4f'
              %(records_list[-1][0],records_list[-1][1],records_list[-1][2],records_list[-1][3],records_list[-1][4]))
        return P,Q,np.array(records_list)


    def update(self, P, Q, R ,eps=1e-6):            
        P = P * (np.dot(R+eps,Q)/(np.dot(P,np.dot(Q.T,Q)))+eps)
        Q = Q * (np.dot(R.T+eps,P)/(np.dot(Q,np.dot(P.T,P)))+eps)
        return P, Q
    
    
    def prediction(self, P, Q):
        N,K = P.shape
        M,K = Q.shape

        rating_list=[]
        for u in range(N):
            u_rating = np.sum(P[u,:]*Q, axis=1)
            rating_list.append(u_rating)
        r_pred = np.array(rating_list)
        return r_pred
```

```python id="8TAv8QK8zUgI"
learning_rate = 0.005
lamda_regularizer = 0.1
max_iteration = 100
K = 10
```

```python colab={"base_uri": "https://localhost:8080/"} id="c-kpzaKZzUkm" outputId="8da534ae-a191-494e-edeb-29b73042e1ea"
model = nmf_sgd(train_list=train_list, 
                test_list=test_list, 
                N=N, 
                M=M,
                K=K,
                learning_rate=learning_rate,
                max_iteration=max_iteration)
P, Q, records_array = model.train()
print('MAE:%.4f;RMSE:%.4f;Recall:%.4f;Precision:%.4f'
      %(records_array[:,1][-1],records_array[:,2][-1],records_array[:,3][-1],records_array[:,4][-1]))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 312} id="YJ0unBdwzUbz" outputId="7d4f1605-544a-4e6a-cf40-f3701409e4b5"
plot_figure(values_list=records_array[:,0],name='loss')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 295} id="xlGXdrMsza0i" outputId="88151d28-b048-4607-96c2-6755df2ee97c"
plot_figure(values_list=records_array[:,1],name='MAE')
```

```python colab={"base_uri": "https://localhost:8080/"} id="4OJD6IPdznYu" outputId="2723c4a4-bf2a-4b2f-fad9-bed9a06c33af"
model = nmf_mult(train_list=train_list, 
                 test_list=test_list, 
                 N=N, 
                 M=M,
                 K=K,
                 max_iteration=max_iteration)
P, Q, records_array = model.train()
print('MAE:%.4f;RMSE:%.4f;Recall:%.4f;Precision:%.4f'
      %(records_array[:,1][-1],records_array[:,2][-1],records_array[:,3][-1],records_array[:,4][-1]))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 295} id="hs7z96hIz0zT" outputId="9ab16fa8-37de-486b-8c22-cceed04a0302"
plot_figure(values_list=records_array[:,0],name='loss')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 295} id="CIW3Awamz0vA" outputId="568a0fe9-a158-41c0-9a66-4e51cce72a36"
plot_figure(values_list=records_array[:,1],name='MAE')
```

```python colab={"base_uri": "https://localhost:8080/"} id="O-J9Tnblz477" outputId="7253b768-b3f1-4ca3-fdd0-50594e5b8c95"
from sklearn.decomposition import NMF
model = NMF(n_components=10, init='random', random_state=0)
train_mat = sequence2mat(sequence = train_list, N = N, M = M)
W = model.fit_transform(train_mat)
H = model.components_

def prediction(P, Q):
    N,K = P.shape
    M,K = Q.shape

    rating_list=[]
    for u in range(N):
        u_rating = np.sum(P[u,:]*Q, axis=1)
        rating_list.append(u_rating)
    r_pred = np.array(rating_list)
    return r_pred

# 预测评分
user = np.array(train_list)[:,0].astype(np.int16)
item = np.array(train_list)[:,1].astype(np.int16)
rating_true = np.array(train_list)[:,2]
rating_pred = np.sum(W[user,:]*H.T[item,:],axis=1)
loss = np.sum((rating_true-rating_pred)**2)
pred_mat = prediction(W, H.T)

# 评估算法
mae, rmse, recall, precision = evaluation(pred_mat, train_mat, test_mat)
print(loss, mae, rmse, recall, precision)
```

<!-- #region id="k4yOC6JE1URM" -->
## PMF
<!-- #endregion -->

<!-- #region id="cKWwC9UE5ip3" -->
<!-- #endregion -->

```python id="FnAsea7O1bQV"
class pmf():
    def __init__(self, 
                 train_list,            # train_list: train data 
                 test_list,             # test_list: test data
                 N,                     # N:the number of user
                 M,                     # M:the number of item
                 K=10,                  # K: the number of latent factor
                 learning_rate=0.001,   # learning_rate: the learning rata
                 lamda_regularizer=0.1, # lamda_regularizer: regularization parameters
                 max_iteration=50       # max_iteration: the max iteration
                ):
        self.train_list = train_list
        self.test_list = test_list
        self.N = N
        self.M = M
        self.K = K
        self.learning_rate = learning_rate
        self.lamda_regularizer = lamda_regularizer
        self.max_iteration = max_iteration
    
    
    def train(self):
        P = np.random.normal(0, 0.1, (self.N, self.K))
        Q = np.random.normal(0, 0.1, (self.M, self.K))

        train_mat = sequence2mat(sequence = self.train_list, N = self.N, M = self.M)
        test_mat = sequence2mat(sequence = self.test_list, N = self.N, M = self.M)

        records_list = []
        for step in range(self.max_iteration):
            los=0.0
            for data in self.train_list:
                u,i,r = data
                P[u],Q[i],ls = self.update(P[u], Q[i], r=r, 
                                           learning_rate=self.learning_rate, 
                                           lamda_regularizer=self.lamda_regularizer)
                los += ls
            pred_mat = self.prediction(P,Q)
            mae, rmse, recall, precision = evaluation(pred_mat, train_mat, test_mat)
            records_list.append(np.array([los, mae, rmse, recall, precision]))

            if step % 10 ==0:
                print(' step:%d \n loss:%.4f,mae:%.4f,rmse:%.4f,recall:%.4f,precision:%.4f'
                      %(step,los,mae,rmse,recall,precision))

        print(' end. \n loss:%.4f,mae:%.4f,rmse:%.4f,recall:%.4f,precision:%.4f'
              %(records_list[-1][0],records_list[-1][1],records_list[-1][2],records_list[-1][3],records_list[-1][4]))
        return P, Q, np.array(records_list)


    def update(self, p, q, r, learning_rate=0.001, lamda_regularizer=0.1):
        error = r - np.dot(p, q.T)            
        p = p + learning_rate*(error*q - lamda_regularizer*p)
        q = q + learning_rate*(error*p - lamda_regularizer*q)
        loss = 0.5 * (error**2 + lamda_regularizer*(np.square(p).sum() + np.square(q).sum()))
        return p, q, loss


    def prediction(self, P, Q):
        N,K = P.shape
        M,K = Q.shape

        rating_list=[]
        for u in range(N):
            u_rating = np.sum(P[u,:]*Q, axis=1)
            rating_list.append(u_rating)
        r_pred = np.array(rating_list)
        return r_pred
```

```python colab={"base_uri": "https://localhost:8080/"} id="JBChoVIA1W4r" outputId="3335756d-cdc4-48eb-891c-b53b4201906b"
model = pmf(train_list=train_list, 
            test_list=test_list, 
            N=N, 
            M=M,
            K=K,
            learning_rate=learning_rate, 
            lamda_regularizer=lamda_regularizer, 
            max_iteration=max_iteration)
P, Q, records_array = model.train()
print('MAE:%.4f;RMSE:%.4f;Recall:%.4f;Precision:%.4f'
      %(records_array[:,1][-1],records_array[:,2][-1],records_array[:,3][-1],records_array[:,4][-1]))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 295} id="xbljd6Kk1eR8" outputId="a25cb940-ddf1-4220-edd5-37385a91d75a"
plot_figure(values_list=records_array[:,0],name='loss')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 295} id="SvSgjhFw1eNl" outputId="e7b1eaa0-2fd8-41bc-a6b0-3bb288e253a0"
plot_figure(values_list=records_array[:,1],name='MAE')
```

<!-- #region id="_H8cNZeI1jFc" -->
## WMF
<!-- #endregion -->

<!-- #region id="tcDGxz1x52kt" -->
<!-- #endregion -->

```python id="aBWRkwfp1wNX"
class wmf():
    def __init__(self, 
                 train_list,            # train_list: train data 
                 test_list,             # test_list: test data
                 N,                     # N:the number of user
                 M,                     # M:the number of item
                 K=10,                  # K: the number of latent factor
                 alpha=40,              # alpha: the confidence of negtive samplers
                 lamda_regularizer=0.1, # lamda_regularizer: regularization parameters
                 max_iteration=50       # max_iteration: the max iteration
                ):
        self.train_list = train_list
        self.test_list = test_list
        self.N = N
        self.M = M
        self.K = K
        self.alpha = alpha
        self.lamda_regularizer = lamda_regularizer
        self.max_iteration = max_iteration
    
    
    def train(self):
        P = np.random.normal(0, 0.1, (self.N, self.K))
        Q = np.random.normal(0, 0.1, (self.M, self.K))

        train_mat = sequence2mat(sequence = self.train_list, N = self.N, M = self.M)
        test_mat = sequence2mat(sequence = self.test_list, N = self.N, M = self.M)

        records_list = []
        for step in range(self.max_iteration):
            for u in range(self.N):
                Ru = train_mat[u,:]
                P[u,:] = self.update(Q, Ru, lamda_regularizer=self.lamda_regularizer, alpha=self.alpha)

            for i in range(self.M):
                Ri = train_mat[:,i]
                Q[i,:] = self.update(P, Ri.T, lamda_regularizer=self.lamda_regularizer, alpha=self.alpha)

            pred_mat = self.prediction(P, Q)
            mae, rmse, recall, precision = evaluation(pred_mat, train_mat, test_mat)
            records_list.append(np.array([mae, rmse, recall, precision]))

            print(' step:%d \n mae:%.4f,rmse:%.4f,recall:%.4f,precision:%.4f'%(step,mae,rmse,recall,precision))

        print(' end. \n mae:%.4f,rmse:%.4f,recall:%.4f,precision:%.4f'
              %(records_list[-1][0],records_list[-1][1],records_list[-1][2],records_list[-1][3]))
        return P, Q, np.array(records_list)


    def update(self, P, Ru, lamda_regularizer=0.1, alpha=40):
        # P: N/M *K
        # Ru: N/M *1
        N, K = P.shape
        c_ui = 1 + alpha*Ru
        Cu = c_ui* np.eye(N)   

        YtCY_I = P.T.dot(Cu).dot(P) + lamda_regularizer*np.eye(K)
        YtCRu = P.T.dot(Cu).dot(Ru)
        p = np.linalg.inv(YtCY_I).dot(YtCRu)
        return p.T


    def prediction(self, P, Q):
        N,K = P.shape
        M,K = Q.shape

        rating_list=[]
        for u in range(N):
            u_rating = np.sum(P[u,:]*Q, axis=1)
            rating_list.append(u_rating)
        r_pred = np.array(rating_list)
        return r_pred
```

```python id="PvX8cFdH1nRr"
max_iteration = 10
alpha=40
```

```python colab={"base_uri": "https://localhost:8080/"} id="5ErZrClB1lMA" outputId="1da25038-253a-4f09-e774-e8acda3f4c1b"
model = wmf(train_list=train_list, 
            test_list=test_list, 
            N=N, 
            M=M,
            K=K,
            alpha=alpha,
            lamda_regularizer=lamda_regularizer, 
            max_iteration=max_iteration)
P, Q, records_array = model.train()
print('MAE:%.4f;RMSE:%.4f;Recall:%.4f;Precision:%.4f'
      %(records_array[:,0][-1],records_array[:,1][-1],records_array[:,2][-1],records_array[:,3][-1]))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 295} id="H8-xojut1o8Y" outputId="62633f90-12da-4d33-dfe8-34673c1764a6"
plot_figure(values_list=records_array[:,2],name='Recall')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 295} id="uHyI8bSa1poa" outputId="6af19b66-0d96-44ad-b243-fdd0769016c6"
plot_figure(values_list=records_array[:,3],name='Precision')
```

<!-- #region id="TjeeTO8n2afs" -->
## SVD
<!-- #endregion -->

<!-- #region id="EOyvJ1bg6S60" -->
<!-- #endregion -->

```python id="LjNmCWDi2dAR"
from sklearn.utils.extmath import randomized_svd
def puresvd(R = None, # train mat
            k=150, # the number of latent factor
            ):
    P, sigma, QT = randomized_svd(R, k)
    sigma = scipy.sparse.diags(sigma, 0)
    P = P * sigma
    Q = QT.T   
    # R_= np.dot(P, QT)
    R_ = np.dot(R, np.dot(Q, QT)) #
    return R_
```

```python id="vPs7Bvat2c7N"
R_score = puresvd(R=train_mat, k=K)
mae, rmse, recall, precision = evaluation(pred_mat=R_score, train_mat=train_mat, test_mat=test_mat)
print('mae:%.4f,rmse:%.4f,recall:%.4f,precision:%.4f'%(mae,rmse,recall,precision))
```

<!-- #region id="DWaLjZ5k11hs" -->
## BiasSVD
<!-- #endregion -->

```python id="xGcRvmnG13f_"
class biassvd():
    def __init__(self, 
                 train_list,            # train_list: train data 
                 test_list,             # test_list: test data
                 N,                     # N:the number of user
                 M,                     # M:the number of item
                 K=10,                  # K: the number of latent factor
                 learning_rate=0.001,   # learning_rate: the learning rata
                 lamda_regularizer=0.1, # lamda_regularizer: regularization parameters
                 max_iteration=50       # max_iteration: the max iteration
                ):
        self.train_list = train_list
        self.test_list = test_list
        self.N = N
        self.M = M
        self.K = K
        self.learning_rate = learning_rate
        self.lamda_regularizer = lamda_regularizer
        self.max_iteration = max_iteration
        
        
    def train(self):
        P = np.random.normal(0, 0.1, (self.N, self.K))
        Q = np.random.normal(0, 0.1, (self.M, self.K))
        bu = np.zeros([self.N])
        bi = np.zeros([self.M])

        train_mat = sequence2mat(sequence = self.train_list, N = self.N, M = self.M)
        test_mat = sequence2mat(sequence = self.test_list, N = self.N, M = self.M)

        aveg_rating = np.mean(train_mat[train_mat>0])

        records_list = []
        for step in range(self.max_iteration):
            los=0.0
            for data in self.train_list:
                u,i,r = data
                P[u],Q[i],bu[u],bi[i],ls = self.update(P[u], Q[i], bu=bu[u], bi=bi[i], 
                                                       aveg_rating=aveg_rating, r=r,
                                                       learning_rate=self.learning_rate, 
                                                       lamda_regularizer=self.lamda_regularizer)
                los += ls
            pred_mat = self.prediction(P, Q, bu, bi, aveg_rating)
            mae, rmse, recall, precision = evaluation(pred_mat, train_mat, test_mat)
            records_list.append(np.array([los, mae, rmse, recall, precision]))

            if step % 10 ==0:
                print(' step:%d \n loss:%.4f,mae:%.4f,rmse:%.4f,recall:%.4f,precision:%.4f'
                      %(step,los,mae,rmse,recall,precision))

        print(' end. \n loss:%.4f,mae:%.4f,rmse:%.4f,recall:%.4f,precision:%.4f'
              %(records_list[-1][0],records_list[-1][1],records_list[-1][2],records_list[-1][3],records_list[-1][4]))
        return P, Q, bu, bi, np.array(records_list)


    def prediction(self, P, Q, bu, bi, aveg_rating):
        N,K = P.shape
        M,K = Q.shape

        rating_list=[]
        for u in range(N):
            u_rating = aveg_rating + bu[u] + bi + np.sum(P[u,:]*Q, axis=1)
            rating_list.append(u_rating)
        r_pred = np.array(rating_list)
        return r_pred


    def update(self, p, q, bu, bi, aveg_rating, r, learning_rate=0.001, lamda_regularizer=0.1):
        error = r - (aveg_rating + bu + bi + np.dot(p, q.T))            
        p = p + learning_rate*(error*q - lamda_regularizer*p)
        q = q + learning_rate*(error*p - lamda_regularizer*q)
        bu = bu + learning_rate*(error - lamda_regularizer*bu)
        bi = bi + learning_rate*(error - lamda_regularizer*bi)
        loss = 0.5 * (error**2 + lamda_regularizer*(np.square(p).sum() + np.square(q).sum()) + bu**2 + bi**2)
        return p, q, bu, bi, loss
```

```python colab={"base_uri": "https://localhost:8080/"} id="I7nYXhr913bk" outputId="203a00c0-2d9c-405b-90d8-cc510ce306ec"
max_iteration = 100
model = biassvd(train_list=train_list, 
                test_list=test_list, 
                N=N, 
                M=M,
                K=K,
                learning_rate=learning_rate, 
                lamda_regularizer=lamda_regularizer, 
                max_iteration=max_iteration)
P, Q, bu, bi, records_array = model.train()
print('MAE:%.4f;RMSE:%.4f;Recall:%.4f;Precision:%.4f'
      %(records_array[:,1][-1],records_array[:,2][-1],records_array[:,3][-1],records_array[:,4][-1]))
```

```python id="13YRenYq13MW" colab={"base_uri": "https://localhost:8080/", "height": 295} outputId="0ee4f30c-e849-4dcc-b0bf-3b20b838a0b5"
plot_figure(values_list=records_array[:,0],name='loss')
```

```python id="s84OHQQQ13G9" colab={"base_uri": "https://localhost:8080/", "height": 295} outputId="3207c491-98ea-4c75-f24d-1146c74d78d1"
plot_figure(values_list=records_array[:,1],name='MAE')
```

<!-- #region id="ydmWSYbJ2NP2" -->
## SVD++
<!-- #endregion -->

```python id="8PE0Le8c2O6e"
class svdplus():
    def __init__(self, 
                 train_list,            # train_list: train data 
                 test_list,             # test_list: test data
                 N,                     # N:the number of user
                 M,                     # M:the number of item
                 K=10,                  # K: the number of latent factor
                 learning_rate=0.001,   # learning_rate: the learning rata
                 lamda_regularizer=0.1, # lamda_regularizer: regularization parameters
                 max_iteration=50       # max_iteration: the max iteration
                ):
        self.train_list = train_list
        self.test_list = test_list
        self.N = N
        self.M = M
        self.K = K
        self.learning_rate = learning_rate
        self.lamda_regularizer = lamda_regularizer
        self.max_iteration = max_iteration
    
    
    def train(self): 
        P = np.random.normal(0, 0.1, (self.N, self.K))
        Q = np.random.normal(0, 0.1, (self.M, self.K))
        Y = np.random.normal(0, 0.1, (self.M, self.K))
        bu = np.zeros([self.N])
        bi = np.zeros([self.M])

        train_mat = sequence2mat(sequence = self.train_list, N = self.N, M = self.M)
        test_mat = sequence2mat(sequence = self.test_list, N = self.N, M = self.M)

        aveg_rating = np.mean(train_mat[train_mat>0])

        records_list = []
        for step in range(self.max_iteration):
            los=0.0
            for data in self.train_list:
                u,i,r = data
                P[u],Q[i],bu[u],bi[i],Y, ls = self.update(p=P[u], q=Q[i], bu=bu[u], bi=bi[i], Y=Y, 
                                                          aveg_rating=aveg_rating, r=r,Ru = train_mat[u], 
                                                          learning_rate=self.learning_rate, 
                                                          lamda_regularizer=self.lamda_regularizer)
                los += ls
            pred_mat = self.prediction(P, Q, Y, bu, bi, aveg_rating, train_mat)
            mae, rmse, recall, precision = evaluation(pred_mat, train_mat, test_mat)
            records_list.append(np.array([los, mae, rmse, recall, precision]))

            if step % 10 ==0:
                print(' step:%d \n loss:%.4f,mae:%.4f,rmse:%.4f,recall:%.4f,precision:%.4f'
                      %(step,los,mae,rmse,recall,precision))

        print(' end. \n loss:%.4f,mae:%.4f,rmse:%.4f,recall:%.4f,precision:%.4f'
              %(records_list[-1][0],records_list[-1][1],records_list[-1][2],records_list[-1][3],records_list[-1][4]))
        return P, Q, Y, bu, bi, np.array(records_list)


    def update(self, p, q, bu, bi, Y, aveg_rating, r, Ru, learning_rate=0.001, lamda_regularizer=0.1):
        Iu = np.sum(Ru>0)
        y_sum = np.sum(Y[np.where(Ru>0)], axis=0)
        error = r - (aveg_rating + bu + bi + np.dot(p+Iu**(-0.5)*y_sum, q.T))            
        p = p + learning_rate*(error*q - lamda_regularizer*p)
        q = q + learning_rate*(error*(p + Iu**(-0.5)*y_sum) - lamda_regularizer*q)
        bu = bu + learning_rate*(error - lamda_regularizer*bu)
        bi = bi + learning_rate*(error - lamda_regularizer*bi)

        l = 0
        for j in np.where(Ru>0):
            Y[j] = Y[j] + learning_rate*(error*Iu**(-0.5)*q - lamda_regularizer*Y[j])
            l = l + np.square(Y[j]).sum()

        loss = 0.5 * (error**2 + lamda_regularizer*(np.square(p).sum() + np.square(q).sum()) + bu**2 + bi**2 + l)
        return p, q, bu, bi, Y, loss


    def prediction(self, P, Q, Y, bu, bi, aveg_rating, R):
        N,K = P.shape
        M,K = Q.shape

        rating_list=[]
        for u in range(N):
            Ru = R[u]
            Iu = np.sum(Ru>0)
            y_sum = np.sum(Y[np.where(Ru>0)],axis=0)
            u_rating = aveg_rating + bu[u]+ bi + np.sum((P[u,:]+Iu**(-0.5)*y_sum)*Q,axis=1)
            rating_list.append(u_rating)
        r_pred = np.array(rating_list)
        return r_pred
```

```python id="p925itar2O1f"
model = svdplus(train_list=train_list, 
                test_list=test_list, 
                N=N, 
                M=M,
                K=K,
                learning_rate=learning_rate, 
                lamda_regularizer=lamda_regularizer, 
                max_iteration=max_iteration)
P, Q, Y, bu, bi, records_array = model.train()
print('MAE:%.4f;RMSE:%.4f;Recall:%.4f;Precision:%.4f'
      %(records_array[:,1][-1],records_array[:,2][-1],records_array[:,3][-1],records_array[:,4][-1]))
```

```python id="ws9gfAFQ2OwU"
plot_figure(values_list=records_array[:,0],name='loss')
```

```python id="5mLW292z2OrI"
plot_figure(values_list=records_array[:,1],name='MAE')
```

<!-- #region id="jugX3EBkuvfU" -->
## MLP
<!-- #endregion -->

<!-- #region id="H75OtwC_62N7" -->
<!-- #endregion -->

```python id="tAr7HlhRuwJa"
class mlp():
    def __init__(self,               
                 users_num = None,
                 items_num = None,
                 embedding_size = 16,
                 hidden_sizes = [16,8],
                 learning_rate = 1e-3,
                 lamda_regularizer=1e-3,
                 batch_size = 256
                ):
        self.users_num = users_num
        self.items_num = items_num
        self.embedding_size = embedding_size
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.lamda_regularizer = lamda_regularizer
        self.batch_size = batch_size

        # loss records
        self.train_loss_records = []  
        self.build_graph()   

        
    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():       
            # _________ input data _________
            self.users_inputs = tf.compat.v1.placeholder(tf.int32, shape = [None], name='users_inputs')
            self.items_inputs = tf.compat.v1.placeholder(tf.int32, shape = [None], name='items_inputs')
            self.train_labels = tf.compat.v1.placeholder(tf.float32, shape = [None], name='train_labels') 
            
            # _________ variables _________
            self.weights = self._initialize_weights()
            
            # _________ train _____________
            self.y_ = self.inference(users_inputs=self.users_inputs, items_inputs=self.items_inputs)
            self.loss_train = self.loss_function(true_labels=self.train_labels, 
                                                 predicted_labels=tf.reshape(self.y_,shape=[-1]),
                                                 lamda_regularizer=self.lamda_regularizer)
            self.train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(self.loss_train) 

            # _________ prediction _____________
            self.predictions = self.inference(users_inputs=self.users_inputs, items_inputs=self.items_inputs)
        
            #变量初始化 init 
            init = tf.compat.v1.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)
    
    
    def _init_session(self):
        # adaptively growing memory
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.compat.v1.Session(config=config)
    
    
    def _initialize_weights(self):
        all_weights = dict()

        # -----embedding layer------
        all_weights['embedding_users'] = tf.Variable(tf.random.normal([self.users_num, self.embedding_size],0, 0.1),name='embedding_users')
        all_weights['embedding_items'] = tf.Variable(tf.random.normal([self.items_num, self.embedding_size],
                                                                      0, 0.1),name='embedding_items') 
        
        # ------hidden layer------
        all_weights['weight_0'] = tf.Variable(tf.random.normal([self.embedding_size * 2,self.hidden_sizes[0]], 0.0, 0.1),name='weight_0')
        all_weights['bias_0'] = tf.Variable(tf.zeros([self.hidden_sizes[0]]), name='bias_0')
        all_weights['weight_1'] = tf.Variable(tf.random.normal([self.hidden_sizes[0],self.hidden_sizes[1]], 0.0, 0.1), name='weight_1')
        all_weights['bias_1'] = tf.Variable(tf.zeros([self.hidden_sizes[1]]), name='bias_1')
        
        # ------output layer-----
        all_weights['weight_n'] = tf.Variable(tf.random.normal([self.hidden_sizes[-1], 1], 0, 0.1), name='weight_n')
        all_weights['bias_n'] = tf.Variable(tf.zeros([1]), name='bias_n')
        return all_weights
        
    
    def train(self, data_sequence):
        train_size = len(data_sequence)
        batch_size = self.batch_size
        total_batch = math.ceil(train_size/batch_size)

        for batch in range(total_batch):
            start = (batch*batch_size)% train_size
            end = min(start+batch_size, train_size)
            data_array = np.array(data_sequence[start:end])
            X = data_array[:,:2] # u,i
            y = data_array[:,-1] # label

            feed_dict = {self.users_inputs: X[:,0], self.items_inputs: X[:,1], self.train_labels:y}  
            loss, opt = self.sess.run([self.loss_train,self.train_op], feed_dict=feed_dict)
            self.train_loss_records.append(loss)
            
        return self.train_loss_records


    def inference(self, users_inputs, items_inputs):
        embed_users = tf.reshape(tf.nn.embedding_lookup(self.weights['embedding_users'], users_inputs),
                                 shape=[-1, self.embedding_size])
        embed_items = tf.reshape(tf.nn.embedding_lookup(self.weights['embedding_items'], items_inputs),
                                 shape=[-1, self.embedding_size])
            
        layer0 = tf.nn.relu(tf.matmul(tf.concat([embed_items,embed_users],1), self.weights['weight_0']) + self.weights['bias_0'])
        layer1 = tf.nn.relu(tf.matmul(layer0, self.weights['weight_1']) + self.weights['bias_1'])       
        y_ = tf.matmul(layer1,self.weights['weight_n']) + self.weights['bias_n']
        return y_         
        
        
    def loss_function(self, true_labels, predicted_labels,lamda_regularizer=1e-3):   
        loss = tf.compat.v1.losses.mean_squared_error(true_labels, predicted_labels)
        cost = loss
        if lamda_regularizer>0:
            regularizer_1 = tf.contrib.layers.l2_regularizer(lamda_regularizer)
            regularization = regularizer_1(
                self.weights['embedding_users']) + regularizer_1(
                self.weights['embedding_items'])+ regularizer_1(
                self.weights['weight_0']) + regularizer_1(
                self.weights['weight_1']) + regularizer_1(
                self.weights['weight_n'])
            cost = loss + regularization

        return cost   
    
    
    def predict_ratings(self, data_sequence):
        pred_mat = np.zeros([self.users_num, self.items_num])
        
        instances_size = len(data_sequence)
        data_array = np.array(data_sequence)
        items_id = np.array([i for i in range(self.items_num)])
        for u in range(self.users_num):
            users_id = u*np.ones_like(items_id)
            feed_dict = {self.users_inputs:users_id, 
                         self.items_inputs:items_id}  
            out = self.sess.run([self.predictions], feed_dict=feed_dict)
            pred_mat[u] = np.reshape(out,(-1))

        return pred_mat
```

```python id="YPWGOfXIy880"
users_num = N
items_num = M
hidden_size = 500
batch_size = 256
lamda_regularizer = 1e-3
learning_rate = 1e-4
epoches  = 100
embedding_size = 8
```

```python colab={"base_uri": "https://localhost:8080/"} id="oy-_STKYvCt-" outputId="f2483d82-31fd-4ebb-edc2-abd637de1f91"
learning_rate = 1e-4
hidden_sizes = [embedding_size,int(embedding_size/2)]
model = mlp(users_num = users_num,
            items_num = items_num,
            embedding_size = embedding_size,
            hidden_sizes = hidden_sizes,
            learning_rate = learning_rate,
            lamda_regularizer = lamda_regularizer,
            batch_size = batch_size)

records_list = []
input_data = train_list
for epoch in range(epoches):
    data_mat = np.random.permutation(input_data) 
    loss = model.train(data_sequence=data_mat)
    pred_mat = model.predict_ratings(data_sequence=test_list)
    mae, rmse, recall, precision = evaluation(pred_mat, train_mat, test_mat)
    records_list.append([loss[-1],mae, rmse, recall, precision])
    if epoch % 10==0:
        topn = get_topn(pred_mat, train_mat, n=10)
        hit_list, ndcg_list = hit_ndcg(test_sequence=np.array(test_list), ranklist=topn)
        hit, ndcg = np.array(hit_list).mean(),np.array(ndcg_list).mean()
        print('epoch:%d  loss=%.4f; \n MAE=%.4f; RMSE=%.4f; Recall=%.4f; Precision=%.4f; Hit=%.4f; NDCG=%.4f'
              %(epoch, loss[-1], mae, rmse, recall, precision, hit, ndcg))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 295} id="XAWk0KZ7wUoB" outputId="c0dc08f4-e6b8-4243-fdb2-e8efb6cdb1c6"
plot_figure(values_list=np.array(records_list)[:,0],name='loss')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 295} id="AQUTWyIPwT4Z" outputId="f6351872-8848-430e-ecf4-bace43b11dcb"
plot_figure(values_list=np.array(records_list)[:,2],name='RMSE')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 295} id="-eiw9viUwT0y" outputId="a6b267c0-62f8-4982-dda8-7ba994b07002"
plot_figure(values_list=np.array(records_list)[:,-1],name='Precision')
```

<!-- #region id="6c_9EgLXvqPP" -->
## AutoRec
<!-- #endregion -->

<!-- #region id="o5N19fa969Ov" -->
<!-- #endregion -->

```python id="TNB74RhTvQhZ"
class autorec():
    def __init__(self,
                 users_num = None,         #用户数
                 items_num = None,         #商品数
                 hidden_size = 500,        #隐层节点数目，即用户的嵌入空间维度
                 batch_size = 256,         #batch大小
                 learning_rate = 1e-3,     #学习率
                 lamda_regularizer = 1e-3, #正则项系数
                ):
        self.users_num = users_num
        self.items_num = items_num
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lamda_regularizer = lamda_regularizer
        
        self.train_loss_records = []  
        self.build_graph()   

        
    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():      
            # _________ input data _________
            self.rating_inputs = tf.compat.v1.placeholder(tf.float32, shape = [None, self.items_num], name='rating_inputs')
            
            # _________ variables _________
            self.weights = self._initialize_weights()
            
            # _________ train _____________
            self.y_ = self.inference(rating_inputs=self.rating_inputs)
            self.loss_train = self.loss_function(true_r=self.rating_inputs, predicted_r=self.y_, lamda_regularizer=self.lamda_regularizer)
            self.train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(self.loss_train) 
        
            # _________ prediction _____________
            self.predictions = self.inference(rating_inputs=self.rating_inputs)
            
            #变量初始化 init 
            init = tf.compat.v1.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)
    
    
    def _init_session(self):
        # adaptively growing memory
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.compat.v1.Session(config=config)
    
    
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['V'] = tf.Variable(tf.random.normal([self.items_num, self.hidden_size], 0.0, 0.1), name='V')
        all_weights['mu'] = tf.Variable(tf.zeros([self.hidden_size]), name='mu')
        all_weights['W'] = tf.Variable(tf.random.normal([self.hidden_size, self.items_num], 0.0, 0.1), name='W')
        all_weights['b'] = tf.Variable(tf.zeros([self.items_num]), name='b')
        return all_weights
    
    
    def train(self, data_mat):
        instances_size = len(data_mat)
        batch_size = self.batch_size
        total_batch = math.ceil(instances_size/batch_size)
        for batch in range(total_batch):
            start = (batch*batch_size)% instances_size
            end = min(start+batch_size, instances_size)
            feed_dict = {self.rating_inputs: data_mat[start:end]}  
            loss, opt = self.sess.run([self.loss_train, self.train_op], feed_dict=feed_dict)
            self.train_loss_records.append(loss)
            
        return self.train_loss_records

        
    # 网络的前向传播
    def inference(self, rating_inputs):
        encoder = tf.nn.sigmoid(tf.matmul(rating_inputs, self.weights['V']) + self.weights['mu'])
        decoder = tf.identity(tf.matmul(encoder, self.weights['W']) + self.weights['b'])
        return decoder         
        
        
    def loss_function(self, true_r, predicted_r, lamda_regularizer=1e-3):
        idx = tf.where(true_r>0)
        true_y = tf.gather_nd(true_r, idx)
        predicted_y = tf.gather_nd(predicted_r, idx)
        mse = tf.compat.v1.losses.mean_squared_error(true_y, predicted_y)
        regularizer = tf.contrib.layers.l2_regularizer(lamda_regularizer)
        regularization = regularizer(self.weights['V']) + regularizer(self.weights['W'])
        cost = mse + regularization
        return cost 
    
    
    def predict_ratings(self, data_mat):
        pred_mat = np.zeros([self.users_num, self.items_num])
        
        instances_size = len(data_mat)
        batch_size = self.batch_size
        total_batch = math.ceil(instances_size/batch_size)
        for batch in range(total_batch):
            start = (batch*batch_size)% instances_size
            end = min(start+batch_size, instances_size)
            feed_dict = {self.rating_inputs: data_mat[start:end]}  
            out = self.sess.run([self.predictions], feed_dict=feed_dict)
            pred_mat[start:end,:] = np.reshape(out,(-1,self.items_num))

        return pred_mat
```

```python id="lpoYqqy0zDDK"
users_num = N
items_num = M
hidden_size = 500
batch_size = 256
lamda_regularizer = 1e-3
learning_rate = 1e-3
epoches  = 100
embedding_size = 8
```

```python colab={"base_uri": "https://localhost:8080/"} id="OHor8fAsv2jA" outputId="4e93ff39-1470-4de8-a19d-c53643ce16c0"
model = autorec(users_num = users_num,
                items_num = items_num,
                hidden_size = hidden_size,
                batch_size = batch_size,
                learning_rate = learning_rate,
                lamda_regularizer = lamda_regularizer)

records_list = []
for epoch in range(epoches):
    data_mat = np.random.permutation(train_mat) 
    loss = model.train(data_mat=data_mat)
    pred_mat = model.predict_ratings(data_mat=train_mat)
    mae, rmse, recall, precision = evaluation(pred_mat, train_mat, test_mat)
    records_list.append([loss[-1],mae, rmse, recall, precision])
    if epoch % 10==0:
        print('epoch:%d  loss=%.4f; \n MAE=%.4f; RMSE=%.4f; Recall=%.4f; Precision=%.4f'
              %(epoch, loss[-1], mae, rmse, recall, precision))
```

```python id="O9JlBOzdv7O4"
plot_figure(values_list=np.array(records_list)[:,0],name='loss')
```

```python id="UPgCJTrDwe1X"
plot_figure(values_list=np.array(records_list)[:,2],name='RMSE')
```

```python id="xpTg-5JLweyW"
plot_figure(values_list=np.array(records_list)[:,-1],name='Precision')
```

<!-- #region id="TjQhYy6Qwh51" -->
## CDAE
<!-- #endregion -->

<!-- #region id="vDusov7V7IJk" -->
<!-- #endregion -->

```python id="60VqlGw9weuM"
class cdae():
    def __init__(self,
                 users_num = None,         #用户数
                 items_num = None,         #商品数
                 hidden_size = 500,        #隐层节点数目，即用户的嵌入空间维度
                 batch_size = 256,         #batch大小
                 learning_rate = 1e-3,     #学习率
                 lamda_regularizer = 1e-3, #正则项系数
                 dropout_rate = 0.5,       # dropout rate
                 noise_level = 1e-3
                ):
        self.users_num = users_num
        self.items_num = items_num
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lamda_regularizer = lamda_regularizer
        self.dropout_rate = dropout_rate
        self.noise_level = noise_level
        
        self.train_loss_records = []  
        self.build_graph()   

        
    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():      
            # _________ input data _________
            self.rating_inputs = tf.compat.v1.placeholder(tf.float32, shape = [None, self.items_num], name='rating_inputs')
            self.user_inputs = tf.compat.v1.placeholder(tf.int32, shape = [None, 1], name='user_inputs')
            self.dropout_prob = tf.compat.v1.placeholder(tf.float32, name = "dropout_prob")

            # _________ variables _________
            self.weights = self._initialize_weights()
            
            # _________ train _____________
            self.y_ = self.inference(rating_inputs=self.rating_inputs, user_inputs=self.user_inputs)
            self.loss_train = self.loss_function(true_r=self.corrupted_inputs, predicted_r=self.y_, lamda_regularizer=self.lamda_regularizer)
            self.train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(self.loss_train) 
        
            # _________ prediction _____________
            self.predictions = self.inference(rating_inputs=self.rating_inputs, user_inputs=self.user_inputs)
            
            #变量初始化 init 
            init = tf.compat.v1.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)
    
    
    def _init_session(self):
        # adaptively growing memory
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.compat.v1.Session(config=config)
    
    
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['W1'] = tf.Variable(tf.random.normal([self.items_num, self.hidden_size], 0.0, 0.1), name='W1')
        all_weights['b1'] = tf.Variable(tf.zeros([self.hidden_size]), name='b1')
        all_weights['W2'] = tf.Variable(tf.random.normal([self.hidden_size, self.items_num], 0.0, 0.1), name='W2')
        all_weights['b2'] = tf.Variable(tf.zeros([self.items_num]), name='b2')
        all_weights['V'] = tf.Variable(tf.zeros([self.users_num, self.hidden_size]), name='V')
        return all_weights
    
    
    def train(self, data_mat):
        instances_size = len(data_mat)
        batch_size = self.batch_size
        total_batch = math.ceil(instances_size/batch_size)
        for batch in range(total_batch):
            start = (batch*batch_size)% instances_size
            end = min(start+batch_size, instances_size)
            feed_dict = {self.user_inputs:np.reshape(data_mat[start:end,0],(-1,1)), 
                         self.rating_inputs:data_mat[start:end,1:], 
                         self.dropout_prob:self.dropout_rate}  
            loss, opt = self.sess.run([self.loss_train, self.train_op], feed_dict=feed_dict)
            self.train_loss_records.append(loss)
            
        return self.train_loss_records

        
    # 网络的前向传播
    def inference(self, rating_inputs, user_inputs):
        inputs_noisy = rating_inputs + self.noise_level * tf.random_normal(tf.shape(rating_inputs))
        self.corrupted_inputs = tf.nn.dropout(inputs_noisy, rate=self.dropout_prob)
        Vu = tf.reshape(tf.nn.embedding_lookup(self.weights['V'], user_inputs),(-1, self.hidden_size))
        encoder = tf.nn.sigmoid(tf.matmul(self.corrupted_inputs, self.weights['W1']) + Vu + self.weights['b1'])
        decoder = tf.identity(tf.matmul(encoder, self.weights['W2']) + self.weights['b2'])
        return decoder         
        
        
    def loss_function(self, true_r, predicted_r, lamda_regularizer=1e-3, loss_type='square'):
        idx = tf.where(true_r>0)
        true_y = tf.gather_nd(true_r, idx)
        predicted_y = tf.gather_nd(predicted_r, idx)
       
        if loss_type=='square':
            loss = tf.compat.v1.losses.mean_squared_error(true_y, predicted_y)
        elif loss_type=='cross_entropy':
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=true_y, logits=predicted_y)
        
        regularizer = tf.contrib.layers.l2_regularizer(lamda_regularizer)
        regularization = regularizer(self.weights['V']) + regularizer(self.weights['W1']) + regularizer(
            self.weights['W2']) + regularizer(self.weights['b1']) + regularizer(self.weights['b2'])
        cost = loss + regularization
        return cost 
    
    
    def predict_ratings(self, data_mat):
        pred_mat = np.zeros([self.users_num, self.items_num])
        
        instances_size = len(data_mat)
        batch_size = self.batch_size
        total_batch = math.ceil(instances_size/batch_size)
        for batch in range(total_batch):
            start = (batch*batch_size)% instances_size
            end = min(start+batch_size, instances_size)
            feed_dict = {self.user_inputs:np.reshape(data_mat[start:end,0],(-1,1)), 
                         self.rating_inputs:data_mat[start:end,1:],
                         self.dropout_prob:0.}  
            out = self.sess.run([self.predictions], feed_dict=feed_dict)
            pred_mat[start:end,:] = np.reshape(out,(-1,self.items_num))

        return pred_mat
```

```python id="qKpBLXlizEwU"
users_num = N
items_num = M
hidden_size = 50
batch_size = 256
lamda_regularizer = 1e-2
learning_rate = 0.01
epoches  = 100
embedding_size = 8
hidden_size = 50
dropout_rate = 0.5
```

```python colab={"base_uri": "https://localhost:8080/"} id="DCXwXaEbweq3" outputId="8d4462ac-8d2a-4f10-da9e-f31d50ae1b66"
model = cdae(users_num = users_num,
             items_num = items_num,
             hidden_size = hidden_size,
             batch_size = batch_size,
             learning_rate = learning_rate,
             lamda_regularizer = lamda_regularizer,
             dropout_rate = dropout_rate)

user_array = np.array([u for u in range(users_num)])
input_data = np.c_[user_array, train_mat]
records_list = []
for epoch in range(epoches):
    data_mat = np.random.permutation(input_data) 
    loss = model.train(data_mat=data_mat)
    pred_mat = model.predict_ratings(data_mat=np.c_[user_array, train_mat])
    mae, rmse, recall, precision = evaluation(pred_mat, train_mat, test_mat)
    records_list.append([loss[-1],mae, rmse, recall, precision])
    if epoch % 10==0:
        print('epoch:%d  loss=%.4f; \n MAE=%.4f; RMSE=%.4f; Recall=%.4f; Precision=%.4f'
              %(epoch, loss[-1], mae, rmse, recall, precision))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 295} id="ebwEonEQws-N" outputId="5ca5a221-1a33-4dde-9f62-14c0a464283c"
plot_figure(values_list=np.array(records_list)[:,0],name='loss')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 295} id="iM3W5Jjhws7A" outputId="e964b3aa-efc1-49fe-fb50-778fc39b3843"
plot_figure(values_list=np.array(records_list)[:,2],name='RMSE')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 295} id="dBmfXHKVws38" outputId="16e7d880-4582-4ebb-e255-55233b929143"
plot_figure(values_list=np.array(records_list)[:,-1],name='Precision')
```
