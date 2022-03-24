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

<!-- #region id="SlSi8kPSEhMP" -->
# Data Poisoning Attacks on Factorization-Based Collaborative Filtering
<!-- #endregion -->

<!-- #region id="uuZ91pC1Em6L" -->
## Dataset
<!-- #endregion -->

```python id="5WeFh6cvEvI_"
import numpy as np
import scipy.sparse as sparse
from itertools import islice


def load_movielens_ratings(ratings_file):
    with open(ratings_file) as f:
        ratings = []
        for line in f:
        # for line in islice(f, 1, None):  
            line = line.split("\t")[:3]
            line = [int(float(l)) for l in line]
            ratings.append(line)
        ratings = np.array(ratings)
    return ratings

def build_user_item_matrix(n_user, n_item, ratings):
    """Build user-item matrix
    Return
    ------
        sparse matrix with shape (n_user, n_item)
    """
    data = ratings[:, 2]
    row_ind = ratings[:, 0]
    col_ind = ratings[:, 1]
    shape = (n_user, n_item)
    return sparse.csr_matrix((data, (row_ind, col_ind)), shape=shape)
```

<!-- #region id="itGzXbdIEwP9" -->
## Evaluation
<!-- #endregion -->

```python id="FPuyxKchE1wg"
import numpy as np

def predict(data, user_features_, item_features_, mean_rating_, max_rating = 1, min_rating = -1):
    data = data.astype(int)
    u_features = user_features_.take(data.take(0, axis=1), axis=0) 
    i_features = item_features_.take(data.take(1, axis=1), axis=0)
    preds = np.sum(u_features * i_features, 1) + mean_rating_
    if max_rating:
        preds[preds > max_rating] = max_rating
    if min_rating:
        preds[preds < min_rating] = min_rating
    return preds

def RMSE(estimation, truth):
    """Root Mean Square Error"""
    estimation = np.float64(estimation)
    truth = np.float64(truth)
    num_sample = estimation.shape[0]
    
    # sum square error
    sse = np.sum(np.square(truth - estimation))
    return np.sqrt(np.divide(sse, num_sample - 1))
```

<!-- #region id="61rp8QBME2Hn" -->
## Compute gradients
<!-- #endregion -->

```python id="pwRI02VrE9XM"
import random
import numpy as np
from six.moves import xrange
from numpy.linalg import inv


#compute the gradient of the hyrid utility function
def compute_utility_grad(n_user, n_item, train, user_features_, item_features_,user_features_origin_, item_features_origin_, \
    w_j0 = 0.8, u1 = 0.5, u2 = 0.5):
    ratings_csr_ = build_user_item_matrix(n_user, n_item, train)
    grad_av = 2 * (np.dot(user_features_, item_features_.T) - np.dot(user_features_origin_, item_features_origin_.T))
    for i in xrange(n_user):
        _, item_idx = ratings_csr_[i, :].nonzero()
        grad_av[i, item_idx] = 0
    avg_rating = np.mean(np.dot(user_features_, item_features_.T), axis = 0)
    perfer_index = np.where(avg_rating > 0.03)
    J0 = random.sample(list(perfer_index[0]), 1)
    grad_in = np.zeros([n_user, n_item])
    grad_in[:, J0] = w_j0 
    grad_hy = u1 * grad_av + u2 * grad_in
    return grad_hy

def compute_grad(n_user, n_item, mal_user, mal_ratings, train, user_features_, mal_user_features_, \
    item_features_, lamda_v, n_feature, user_features_origin_, item_features_origin_):
    '''
    A : inv(lamda_v * Ik + sum(u_i* u_i))   (for u_i of item j)  k * k
    u_i : 1 * k
    grad_model: d(u_i * v_j.T)/d(M_ij) = u_i * A * u_i.T
    '''
    grad_R = compute_utility_grad(n_user, n_item, train, user_features_, \
            item_features_, user_features_origin_, item_features_origin_)
    ratings_csr_ = build_user_item_matrix(n_user, n_item, train)
    ratings_csc_ = ratings_csr_.tocsc()
    mal_ratings_csr_ = build_user_item_matrix(mal_user, n_item, mal_ratings)
    mal_ratings_csc_ = mal_ratings_csr_.tocsc()
    grad_total = np.zeros([mal_user, n_item])
    for i in xrange(mal_user):
        for j in xrange(n_item):
            if j % 100 == 0:
                print('Computing the %dth malicious user, the %d item(total users: %d, total items: %d)' % (i, j, n_user, n_item))
            user_idx, _ = ratings_csc_[:, j].nonzero()
            mal_user_idx, _ = mal_ratings_csc_[:, j].nonzero()
            user_features = user_features_.take(user_idx, axis=0)
            mal_user_features = mal_user_features_.take(mal_user_idx, axis=0)
            U = np.vstack((user_features, mal_user_features))  
            u_i = user_features_.take(i, axis = 0)
            A = np.dot(U.T, U) + lamda_v * np.eye(n_feature)  
            A_u = np.dot(A, u_i.T)
            grad_model = np.zeros([n_user, n_item])
            for m in xrange(n_user):
                u_m = user_features_.take(i, axis = 0)
                grad_model[m, j] = np.dot(u_m, np.dot(inv(A), u_i.T))
            grad_total[i, j] = sum(sum(grad_model * grad_R))
    return grad_total
```

<!-- #region id="yV3L3c0yFAxg" -->
## ALS Optimize
<!-- #endregion -->

```python id="8Ex3G7aqFI9a"
import numpy as np
from six.moves import xrange
from numpy.linalg import inv


def _update_user_feature(n_user, ratings_csr_, n_feature, lamda_u, mean_rating_, user_features_, item_features_):
    '''
    n_u : number of rating items of user i
    item_features: n_u * n_feature (108 * 8)  
    A_i = v_j' * u_i + lamda_u * I(n_feature)
    V_i = sum(M_ij * v_j)
    '''
    for i in xrange(n_user):
        _, item_idx = ratings_csr_[i, :].nonzero()
        n_u = item_idx.shape[0]
        if n_u == 0:
            continue
        item_features = item_features_.take(item_idx, axis=0) 

        ratings = ratings_csr_[i, :].data - mean_rating_ 
        A_i = (np.dot(item_features.T, item_features) +
                   lamda_u * n_u * np.eye(n_feature))
        V_i = np.dot(item_features.T, ratings)
        user_features_[i, :] = np.dot(inv(A_i), V_i)


def _update_item_feature(n_item, ratings_csc_, n_feature, lamda_v, mean_rating_, user_features_, item_features_):
    '''
    n_i : number of rating items of item j
    '''
    for j in xrange(n_item):
        user_idx, _ = ratings_csc_[:, j].nonzero()
        n_i = user_idx.shape[0]
        if n_i == 0:
            continue
        user_features = user_features_.take(user_idx, axis=0)
        ratings = ratings_csc_[:, j].data - mean_rating_
    
        A_j = (np.dot(user_features.T, user_features)  + lamda_v * n_i * np.eye(n_feature))
        V_j = np.dot(user_features.T, ratings)
        item_features_[j, :] = np.dot(inv(A_j), V_j)

def ALS_origin(n_user, n_item, n_feature, ratings, mean_rating_, lamda_u, lamda_v, user_features_, item_features_):
    ratings_csr_ = build_user_item_matrix(n_user, n_item, ratings)
    ratings_csc_ = ratings_csr_.tocsc()
    _update_user_feature(n_user, ratings_csr_, n_feature, lamda_u, mean_rating_, user_features_, item_features_)
    _update_item_feature(n_item, ratings_csc_, n_feature, lamda_v, mean_rating_, user_features_, item_features_)
```

```python id="VoAtbSQ4FLx5"
def _update_user_feature(n_user, ratings_csr_, n_feature, lamda_u, mean_rating_, user_features_, item_features_):
    '''
    n_u : number of rating items of user i
    item_features: n_u * n_feature (108 * 8)  
    A_i = v_j' * u_i + lamda_u * I(n_feature)
    V_i = sum(M_ij * v_j)
    '''
    for i in xrange(n_user):
        _, item_idx = ratings_csr_[i, :].nonzero()
        n_u = item_idx.shape[0]
        if n_u == 0:
            continue
        item_features = item_features_.take(item_idx, axis=0) 

        ratings = ratings_csr_[i, :].data - mean_rating_ 
        A_i = (np.dot(item_features.T, item_features) +
                   lamda_u * n_u * np.eye(n_feature))
        V_i = np.dot(item_features.T, ratings)
        user_features_[i, :] = np.dot(inv(A_i), V_i)

def _update_mal_feature(mal_user, mal_ratings_csr_, n_feature, lamda_u, mal_mean_rating_, mal_user_features_, item_features_):
    for m in xrange(mal_user):
        _, item_idx = mal_ratings_csr_[m, :].nonzero()
        n_m = item_idx.shape[0]
        if n_m == 0:
            continue
        item_features = item_features_.take(item_idx, axis=0) 
        
        ratings = mal_ratings_csr_[m, :].data - mal_mean_rating_ 
        A_i = (np.dot(item_features.T, item_features) +
                   lamda_u * n_m * np.eye(n_feature))
        V_i = np.dot(item_features.T, ratings)
        mal_user_features_[m, :] = np.dot(inv(A_i), V_i)

def _update_item_feature(n_item, ratings_csc_, mal_ratings_csc_, n_feature, lamda_v, mean_rating_, \
    mal_mean_rating_, user_features_, mal_user_features_, item_features_):
    '''
    n_i : number of rating items of item j
    '''
    for j in xrange(n_item):
        user_idx, _ = ratings_csc_[:, j].nonzero()
        n_i = user_idx.shape[0]
        if n_i == 0:
            continue
        user_features = user_features_.take(user_idx, axis=0)
        ratings = ratings_csc_[:, j].data - mean_rating_
        
        mal_user_idx, _ = mal_ratings_csc_[:, j].nonzero()
        m_i = mal_user_idx.shape[0]
        if m_i == 0:
            continue
        mal_user_features = mal_user_features_.take(mal_user_idx, axis=0)
        mal_ratings = mal_ratings_csc_[:, j].data - mal_mean_rating_

        A_j = (np.dot(user_features.T, user_features) + np.dot(mal_user_features.T, mal_user_features) \
                + lamda_v * (n_i + m_i) * np.eye(n_feature))
        V_j = np.dot(user_features.T, ratings) + np.dot(mal_user_features.T, mal_ratings)
        item_features_[j, :] = np.dot(inv(A_j), V_j)

def ALS(n_user, n_item, n_feature, mal_user, ratings, mean_rating_, mal_mean_rating_, mal_ratings, lamda_u, lamda_v, \
    user_features_, mal_user_features_, item_features_):
    ratings_csr_ = build_user_item_matrix(n_user, n_item, ratings)
    ratings_csc_ = ratings_csr_.tocsc()
    mal_ratings_csr_ = build_user_item_matrix(mal_user, n_item, mal_ratings)
    mal_ratings_csc_ = mal_ratings_csr_.tocsc()

    _update_user_feature(n_user, ratings_csr_, n_feature, lamda_u, mean_rating_, user_features_, item_features_)
    _update_mal_feature(mal_user, mal_ratings_csr_, n_feature, lamda_u, mal_mean_rating_, mal_user_features_, item_features_)
    _update_item_feature(n_item, ratings_csc_, mal_ratings_csc_, n_feature, lamda_v, mean_rating_, \
    mal_mean_rating_, user_features_, mal_user_features_, item_features_)
```

<!-- #region id="5wK8dTZ0FOmK" -->
## Main
<!-- #endregion -->

```python id="JvZtrsCsFsBw"
# !wget -q --show-progress https://files.grouplens.org/datasets/movielens/ml-20m.zip
# !unzip ml-20m.zip
```

```python colab={"base_uri": "https://localhost:8080/"} id="tR_HhE5lM-Tw" executionInfo={"status": "ok", "timestamp": 1633263578262, "user_tz": -330, "elapsed": 33, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ac1f5751-e31b-484d-8a62-16248d6dbca0"
!wget -q --show-progress https://files.grouplens.org/datasets/movielens/ml-100k.zip
!unzip ml-100k.zip
```

```python colab={"base_uri": "https://localhost:8080/"} id="5wfY_ozPFdlY" executionInfo={"status": "ok", "timestamp": 1633267397004, "user_tz": -330, "elapsed": 2550885, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1bfb986e-9cef-4bcc-9cb6-f034c3b71ab6"
#collapse-hide
import random
import time

from six.moves import xrange
import numpy as np
from numpy.random import RandomState
from numpy.linalg import inv


# ratings_file = 'ml-20m/ratings.csv'
ratings_file = 'ml-100k/u.data'
ratings = load_movielens_ratings(ratings_file)
rand_state = RandomState(0)

max_rating = max(ratings[:, 2])
min_rating = min(ratings[:, 2])
'''
parameters:
lamda_u: the regularization parameter of user
lamda_v: the regularization parameter of item
alpha: the proportion of malicious users
B: the items of malicious users rating
n_iter: number of iteration
converge: the least RMSE between two iterations
train_pct: the proportion of train dataset
'''
lamda_u = 5e-2
lamda_v = 5e-2
alpha = 0.005
B = 25
n_iters = 10
n_feature = 8
seed = None
last_rmse = None
converge = 1e-5
mal_item = B
# split data to training & testing
train_pct = 0.9
rand_state.shuffle(ratings)
train_size = int(train_pct * ratings.shape[0])
train = ratings[:train_size]
validation = ratings[train_size:]


n_user = max(train[:, 0]) + 1
n_item = max(train[:, 1]) + 1 
mal_user = int(alpha * n_user) 


# add malicious users data
mal_ratings = []
for u in xrange(mal_user):
    mal_user_idx = u
    mal_item_idx = random.sample(range(n_item), mal_item)
    for i in xrange(mal_item):
        mal_movie_idx = mal_item_idx[i]
        mal_rating = 2 * (RandomState(seed).rand() > 0.5) - 1
        mal_ratings.append([mal_user_idx, mal_movie_idx, mal_rating])
        
mal_ratings = np.array(mal_ratings)
#initialize the matrix U U~ and V 
user_features_ = 0.1 * RandomState(seed).rand(n_user, n_feature)
mal_user_features_ = 0.1 * RandomState(seed).rand(mal_user, n_feature)
item_features_ = 0.1 * RandomState(seed).rand(n_item, n_feature)
mean_rating_ = np.mean(train.take(2, axis=1))
mal_mean_rating_ = np.mean(mal_ratings.take(2, axis=1))
user_features_origin_ = 0.1 * RandomState(seed).rand(n_user, n_feature)
item_features_origin_ = 0.1 * RandomState(seed).rand(n_item, n_feature)


#train origin model
def optimize_model_origin():
    print("Start training model without data poisoning attacks!")
    last_rmse = None
    for iteration in xrange(n_iters):
        t1 = time.time()
        ALS_origin(n_user, n_item, n_feature, train, mean_rating_, lamda_u, lamda_v, user_features_origin_, item_features_origin_)
        train_preds = predict(train.take([0, 1], axis=1), user_features_origin_, item_features_origin_, mean_rating_)
        train_rmse = RMSE(train_preds, train.take(2, axis=1))
        t2 = time.time()
        print("The %d th iteration \t time: %ds \t RMSE: %f " % (iteration + 1, t2 - t1, train_rmse))
        # stop when converge
        if last_rmse and abs(train_rmse - last_rmse) < converge:
            break
        else:
            last_rmse = train_rmse
    return last_rmse


#train added attack data model
def optimize_model():
    print("Start training model with data poisoning attacks!")
    last_rmse = None
    for iteration in xrange(n_iters):
        t1 = time.time()
        ALS(n_user, n_item, n_feature, mal_user, train, mean_rating_, mal_mean_rating_, mal_ratings, lamda_u, lamda_v, \
        user_features_, mal_user_features_, item_features_)
        train_preds = predict(train.take([0, 1], axis=1), user_features_, item_features_, mean_rating_)
        train_rmse = RMSE(train_preds, train.take(2, axis=1))
        t2 = time.time()
        print("The %d th iteration \t time: %ds \t RMSE: %f " % (iteration + 1, t2 - t1, train_rmse))
        # stop when converge
        if last_rmse and abs(train_rmse - last_rmse) < converge:
            break
        else:
            last_rmse = train_rmse
    return last_rmse


#using the algorithm of PGA to optimize the utility function
'''
m_iters: number of iteration in PGA
s_t: step size 
Lamda: the contraint of vector
'''
m_iters = 10
s_t = 0.2 * np.ones([m_iters])
converge = 1e-5
Lamda = 1
last_rmse = None

#optimize_model_origin()
for t in xrange(m_iters):
    t1 = time.time()
    #optimize_model()
    grad_total = compute_grad(n_user, n_item, mal_user, mal_ratings, train, user_features_, mal_user_features_, \
                        item_features_, lamda_v, n_feature, user_features_origin_, item_features_origin_)
    mal_data = np.dot(mal_user_features_, item_features_.T)
    temp = mal_data
    mal_data +=  grad_total * s_t[t]
    mal_data[mal_data > Lamda] = Lamda
    mal_data[mal_data < - Lamda] = - Lamda
    rmse = RMSE(mal_data, temp)
    t2 = time.time()
    print("The %d th iteration \t time: %ds \t RMSE: %f " % (t + 1, t2 - t1, rmse))
    if last_rmse and abs(rmse - last_rmse) < converge:
        break
    else:
        last_rmse = rmse
```
