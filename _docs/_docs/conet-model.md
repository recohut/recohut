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

<!-- #region id="Wvyb-8J6gj9E" -->
# CoNet model
> Applying Co-occurrence Neural Networks for Recommendation on MovieLens dataset

- toc: true
- badges: true
- comments: true
- author: "<a href='https://github.com/XiuzeZhou/conet'>Xiuze Zhou</a>"
- categories: [Movie, Pytorch, AttentionMechanism]
- image:
<!-- #endregion -->

<!-- #region id="79GiSwT8T2jf" -->
### Introduction
<!-- #endregion -->

<!-- #region id="0Ci0tBUaOhAY" -->
CoNet stands for *Co-occurrence Neural Networks for Recommendation Chinese explanation*. At present, most recommendation algorithms assume that users-users and commodities-commodities are independent and identically distributed, so they only model the interaction between users-commodities, and ignore the relationship between commodities and commodities. CoNet assumes that commodities always appear in pairs, that is, commodities co-occur.
<!-- #endregion -->

<!-- #region id="qNFTljPoPs-d" -->
<!-- #endregion -->

<!-- #region id="dABgGo43PEZz" -->
1. Give a small example, as shown in the figure above. For example, "Harry Potter 1" and "Harry Potter 2" are always watched by users who like magic at the same time. This is the co-occurrence model of commodities. In order to learn this model, we need to model the user-commodity and the product-commodity at the same time.

2. At the same time, CoNet assume that the more two commodities appear together, the more similar they are. For example, in the movie viewing record, "Harry Potter 1" and "Harry Potter 2" co-occur more than "Harry Potter 1" and "Robot Walle", we think that "Harry Potter 1" and "Harry Potter 2" are more similar.

3. Further, CoNet use the attention mechanism to learn the user's comparative preferences. When rated the two products separately, authors gave the same score. For example, when they watched "Harry Potter 1" and "Harry Potter 2", they found them to look good, and gave them 5 points. However, when compare the two of them, it always felt that one of them might be better. Therefore, authors used the attention mechanism to model and learn this psychological preference.
<!-- #endregion -->

<!-- #region id="Ay2bwrc1T4vU" -->
### Setup
<!-- #endregion -->

```python id="4pj9k-BsT6bM"
import random
import math
import os
import numpy as np
import scipy.sparse as sp
import heapq # for retrieval topK
import multiprocessing
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
```

```python id="FUavP4FHT6Xo"
def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
```

<!-- #region id="eIOItkgqTxp9" -->
### Dataset
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="zZJ6UOC5T6TZ" outputId="194c81df-2980-42af-9a50-c48649ae0b58"
!wget https://github.com/sparsh-ai/reco-data/raw/master/ml-conet.zip
!unzip ml-conet.zip
```

```python colab={"base_uri": "https://localhost:8080/"} id="Hy-nFOu1VeWi" outputId="92e6d7be-de50-464c-de67-52872cd85d19"
!wget http://files.grouplens.org/datasets/movielens/ml-100k.zip
!unzip ml-100k.zip
```

```python id="pSVrsL5tOgX_"
class Dataset(object):

    def __init__(self, path):
        '''
        Constructor
        '''
        self.trainMatrix = self.load_rating_file_as_matrix(path + ".train.rating")
        self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
        self.testNegatives = self.load_negative_file(path + ".test.negative")
        assert len(self.testRatings) == len(self.testNegatives)
        
        self.num_users, self.num_items = self.trainMatrix.shape
        
    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList
    
    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList
    
    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()    
        return mat
```

```python id="wslT-mijUkMB"
def generate_instances(train_mat, positive_size=2, negative_time=8, is_sparse=False):
    data = []
    users_num,items_num = train_mat.shape
    
    if is_sparse:
        indptr = train_mat.indptr
        indices = train_mat.indices
    for u in range(users_num):
        if is_sparse:
            rated_items = indices[indptr[u]:indptr[u+1]] #用户u中有评分项的id
        else:
            rated_items = np.where(train_mat[u,:]>0)[0]
        
        for item0 in rated_items:
            for item1 in np.random.choice(rated_items, size=positive_size):
                data.append([u,item0,item1,1.])
            for _ in range(positive_size*negative_time):
                item1 = np.random.randint(items_num) # no matter item1 is positive or negtive
                item2 = np.random.randint(items_num)
                while item2 in rated_items:
                    item2 = np.random.randint(items_num)
                data.append([u,item2,item1,0.])
    return data
```

<!-- #region id="oHWD1KM3S8dS" -->
### Architecture
<!-- #endregion -->

<!-- #region id="ikktWL7yTBCu" -->
CoNet consists of 7 parts: input layer, embedding layer, attention module, co-occurrence layer, interaction layer, hidden layer and prediction layer (output layer).
<!-- #endregion -->

<!-- #region id="softP2MIS-nB" -->
<!-- #endregion -->

```python id="m1k1exS1U6W5"
class CoNet(nn.Module):
    def __init__(self, users_num, items_num, embedding_size_users=64, embedding_size_items = 64, 
                 hidden_size = [64,32,16,8], is_attention = False):
        super(CoNet, self).__init__()
        self.embedding_size_users, self.embedding_size_items= embedding_size_users, embedding_size_items 
        self.items_num, self.users_num = items_num, users_num
        self.hidden_size, self.is_attention = hidden_size, is_attention
        self.embedding_user  = nn.Embedding(self.users_num, self.embedding_size_users)
        self.embedding_item = nn.Embedding(self.items_num, self.embedding_size_items)
        self.layer1 = nn.Linear(self.embedding_size_users + self.embedding_size_items, self.hidden_size[0])
        self.layers = [nn.Sequential(nn.Linear(self.hidden_size[i], self.hidden_size[i+1]), nn.ReLU()) for i in range(len(self.hidden_size) - 1)]
        self.linear = nn.Linear(self.hidden_size[-1], 1)
 
    def forward(self, x):
        embed_users = self.embedding_user(x[:,0])
        embed_items0 = self.embedding_item(x[:,1])
        embed_items1 = self.embedding_item(x[:,2])
        
        embed_items = (embed_items0 + embed_items1)/2.
        if self.is_attention:
            score0 = torch.reshape(torch.sum(embed_users * embed_items0, 1), shape=[-1,1])
            score1 = torch.reshape(torch.sum(embed_users * embed_items1, 1), shape=[-1,1])
            alpha = torch.sigmoid(score0 - score1)
            embed_items = alpha * embed_items0 + (1. - alpha) * embed_items1
            
        out = torch.cat([embed_users, embed_items],1)
        out = self.layer1(out)
        for layer in self.layers:
            out = layer(out)
        out = self.linear(out) 
        return out
    
    def predict(self, pairs, batch_size, verbose):
        """Computes predictions for a given set of user-item pairs.
        Args:
          pairs: A pair of lists (users, items) of the same length.
          batch_size: unused.
          verbose: unused.
        Returns:
          predictions: A list of the same length as users and items, such that
          predictions[i] is the models prediction for (users[i], items[i]).
        """
        del batch_size, verbose
        num_examples = len(pairs[0])
        assert num_examples == len(pairs[1])
        predictions = np.empty(num_examples)
        pairs = np.array(pairs, dtype=np.int16)
        for i in range(num_examples):
            x = np.c_[pairs[0][i],pairs[1][i],pairs[1][i]]
            x = torch.from_numpy(x).long()
            out = self.forward(x)
            predictions[i] = out.reshape(-1).data.numpy()
        return predictions
    
    def get_embeddings(self):
        idx = torch.LongTensor([i for i in range(self.items_num)])
        embeddings = self.embedding_item(idx)
        return embeddings
```

<!-- #region id="50d_gmEjUF1x" -->
### Evaluation method
<!-- #endregion -->

```python id="559LI-shUAM_"
# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None

def evaluate_model(model, testRatings, testNegatives, K, num_thread):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K
        
    hits, ndcgs = [],[]
    if(num_thread > 1): # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return (hits, ndcgs)
    # Single thread
    for idx in range(len(_testRatings)):
        (hr,ndcg) = eval_one_rating(idx)
        hits.append(hr)
        ndcgs.append(ndcg)      
    return (hits, ndcgs)

def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype = 'int32')
    predictions = _model.predict([users, np.array(items)], 
                                 batch_size=100, verbose=0)
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()
    
    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0
```

```python id="5dTeXurSYIy0"
def evaluate(model, test_ratings, test_negatives, K=10):
    """Helper that calls evaluate from the NCF libraries."""
    (hits, ndcgs) = evaluate_model(model, test_ratings, test_negatives, K=K, num_thread=1)
    return np.array(hits).mean(), np.array(ndcgs).mean()
```

```python id="s7sgokhqUrFK"
def get_similar_items(item_mat, idx, topk=5):
    m,k = item_mat.shape
    target_item = item_mat[idx,:]
    target_mat = np.reshape(np.tile(target_item,m),(-1,k))
    sim = [np.dot(target_mat[i], item_mat[i])/(np.linalg.norm(target_mat[i])*np.linalg.norm(item_mat[i])) 
           for i in range(m)] 
    sorted_items = np.argsort(-np.array(sim))
    return sorted_items[:topk+1] # the most similar is itself

def get_key(item_dict, value):
    key = -1
    for (k, v) in item_dict.items():
        if v == value:
            key = k
    return key


# read original records
def get_item_dict(file_dir):
    # output: 
    # N: the number of user;
    # M: the number of item
    # data: the list of rating information
    user_ids_dict, rated_item_ids_dict = {},{}
    N, M, u_idx, i_idx = 0,0,0,0 
    data_rating = []
    data_time = []
    f = open(file_dir)
    for line in f.readlines():
        if '::' in line:
            u, i, r = line.split('::')[:3]
        elif ',' in line:
            u, i, r = line.split(',')[:3]
        else:
            u, i, r = line.split()[:3]
    
        if u not in user_ids_dict:
            user_ids_dict[u]=u_idx
            u_idx+=1
        if i not in rated_item_ids_dict:
            rated_item_ids_dict[i]=i_idx
            i_idx+=1
        data_rating.append([user_ids_dict[u],rated_item_ids_dict[i],float(r)])
    
    f.close()
    N = u_idx
    M = i_idx

    return rated_item_ids_dict

# read id and its name
def id_name(file_dir):
    id_name_dict = {}
    f = open(file_dir, 'r', encoding='latin-1')
    for line in f.readlines():
        movie_id, movie_name = line.split('|')[:2]
        id_name_dict[int(movie_id)] = movie_name
        
    return id_name_dict
```

<!-- #region id="ipZIayxlU-Kp" -->
### Model training
<!-- #endregion -->

```python id="NCs4D7oxU_Wt"
def train(model, train_mat, test_ratings, test_negatives, users_num, items_num, train_list=None, test_list=None,
          learning_rate = 1e-2, weight_decay=0.01, positive_size=1, negative_time=4, epochs=64, 
          batch_size=1024, topK=10, mode='hr'):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    if train_list!=None:
        train_mat= sequence2mat(sequence=train_list, N=users_num, M=items_num) # train data : user-item matrix
        is_sparse = False
    
    hr_list=[]
    ndcg_list=[]
    hr, ndcg = evaluate(model, test_ratings, test_negatives, K=topK)
    embeddings = model.get_embeddings()
    hr_list.append(hr)
    ndcg_list.append(ndcg)
    print('Init: HR = %.4f, NDCG = %.4f' %(hr, ndcg))
    best_hr, best_ndcg = hr, ndcg
    for epoch in range(epochs):
        data_sequence = generate_instances(train_mat, positive_size=positive_size, negative_time=negative_time, is_sparse=True)
        #data_sequence = read_list("output/" + str(epoch) + ".txt")
        
        train_size = len(data_sequence)
        np.random.shuffle(data_sequence)
        batch_size = batch_size
        total_batch = math.ceil(train_size/batch_size)

        for batch in range(total_batch):
            start = (batch*batch_size)% train_size
            end = min(start + batch_size, train_size)
            data_array = np.array(data_sequence[start:end])
            x = torch.from_numpy(data_array[:,:3]).long()
            y = torch.from_numpy(data_array[:,-1]).reshape(-1,1)
            y_ = model(x)
            loss = criterion(y_.float(), y.float())
            optimizer.zero_grad()              # clear gradients for this training step
            loss.backward()                    # backpropagation, compute gradients
            optimizer.step()                   # apply gradients
            
        # Evaluation
        hr, ndcg = evaluate(model, test_ratings, test_negatives, K=topK)
        hr_list.append(hr)
        ndcg_list.append(ndcg)
        print('epoch=%d, loss=%.4f, HR=%.4f, NDCG=%.4f' %(epoch, loss, hr, ndcg))
        
        mlist = hr_list
        if mode == 'ndcg':
            mlist = ndcg_list
        if (len(mlist) > 20) and (mlist[-2] < mlist[-3] > mlist[-1]):
            best_hr, best_ndcg = hr_list[-3], ndcg_list[-3]
            embeddings = model.get_embeddings()
            break
        best_hr, best_ndcg = hr, ndcg
        embeddings = model.get_embeddings()
            
    print("End. Best HR = %.4f, NDCG = %.4f. " %(best_hr, best_ndcg))
    return embeddings
```

```python colab={"base_uri": "https://localhost:8080/"} id="WjM_mEC1VFLr" outputId="5d768b6f-2687-410f-c39f-3a745f55a1d4"
dataset_path = '/content/100k'

# Load the dataset
dataset = Dataset(dataset_path)
train_mat, test_ratings, test_negatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
print('Dataset: #user=%d, #item=%d, #train_pairs=%d, #test_pairs=%d' 
      % (dataset.num_users, dataset.num_items, train_mat.nnz, len(test_ratings)))

embedding_size_users = 64
embedding_size_items = 64
hidden_size = [64,32,16]
is_attention = True
learning_rate = 1e-3
weight_decay = 1e-5

positive_size = 2
negative_time = 8
epochs = 10
batch_size = 1024
topK = 10
mode = 'hr'
seed = 18

setup_seed(seed)
# Initialize the model
model = CoNet(users_num=dataset.num_users, items_num=dataset.num_items, embedding_size_users=embedding_size_users, 
              embedding_size_items=embedding_size_items, hidden_size=hidden_size, is_attention=is_attention)

if torch.cuda.is_available():
    model = model.cuda()

# Train and evaluate model
embeddings = train(model=model, 
                  train_mat=train_mat.tocsr(), 
                  test_ratings=test_ratings, 
                  test_negatives=test_negatives, 
                  users_num=dataset.num_users, 
                  items_num=dataset.num_items,  
                  learning_rate=learning_rate,
                  weight_decay=weight_decay,
                  positive_size=positive_size,
                  negative_time=negative_time,
                  epochs=epochs,
                  batch_size=batch_size,
                  topK=topK,
                  mode=mode)
print('----------------------------------------------------------')
```

<!-- #region id="SVkrbwidTNr5" -->
### Experiment results
<!-- #endregion -->

<!-- #region id="6QJ8Er_YTPqR" -->
<!-- #endregion -->

<!-- #region id="qGsE0hnlTgAW" -->
### Effect of attention
<!-- #endregion -->

<!-- #region id="7GrqNgxlTlGG" -->
The red curve is the attention mechanism, and the blue is the result of not paying attention.

In fact, it can be understood that the attention mechanism provides more parameters to fit the user's comparative preferences to achieve better results.
<!-- #endregion -->

<!-- #region id="AttiewHGThvt" -->
<!-- #endregion -->

<!-- #region id="swqo10z2YiqE" -->
### Inference
<!-- #endregion -->

```python id="0guTc2D3VQOg"
file_dir = '/content/ml-100k/u.item'
id_name_dict = id_name(file_dir) # original id : movie name

file_dir = '/content/ml-100k/u.data'
item_dict = get_item_dict(file_dir) # original id : new id
```

```python colab={"base_uri": "https://localhost:8080/"} id="RZiL3xIuVR-_" outputId="78a992dd-4736-438f-c1a9-c31fa9c079dc"
movieid_list = [113, 347, 537]
    
for movieid in movieid_list:
    print('MovieID:', movieid, '; MovieName:', id_name_dict[movieid])
    original_id = str(movieid)
    target_item = item_dict[original_id]

    top5 = get_similar_items(embeddings.data.numpy(), idx=target_item)
    movie_list = [get_key(item_dict=item_dict, value=i) for i in top5]
    rec_list = [id_name_dict[int(movie_id)] for movie_id in movie_list[1:]]
    for i in range(len(rec_list)):
        print('\n{0}: {1}'.format(i+1, rec_list[i]))
    print('------------------------------------------------------------------')
```
