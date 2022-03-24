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

<!-- #region id="w2Y4fFh5MlG-" -->
# Incremental Stochastic Gradient Descent
<!-- #endregion -->

<!-- #region id="7NDmk2h8NZZ0" -->
## Setup
<!-- #endregion -->

```python id="gqLbUBEwNZ_z" executionInfo={"status": "ok", "timestamp": 1635161607410, "user_tz": -330, "elapsed": 666, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import numpy as np
import sys

import matplotlib.pyplot as plt
```

```python id="MGbk-SC4Obq-" executionInfo={"status": "ok", "timestamp": 1635161608110, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
%matplotlib inline
```

<!-- #region id="ksIFFATiNbs8" -->
## Model
<!-- #endregion -->

```python id="NBw6cf_1NcNn" executionInfo={"status": "ok", "timestamp": 1635161609840, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class ISGD:
    def __init__(self, n_user, n_item, k, l2_reg=0.01, learn_rate=0.05):
        self.k = k
        self.l2_reg = l2_reg
        self.learn_rate = learn_rate
        self.known_users = np.array([])
        self.known_items = np.array([])
        self.n_user = n_user
        self.n_item = n_item
        self.A = np.random.normal(0., 0.1, (n_user, self.k))
        self.B = np.random.normal(0., 0.1, (n_item, self.k))

    def update(self, u_index, i_index):
        if u_index not in self.known_users: self.known_users = np.append(self.known_users, u_index)
        u_vec = self.A[u_index]

        if i_index not in self.known_items: self.known_items = np.append(self.known_items, i_index)
        i_vec = self.B[i_index]

        err = 1. - np.inner(u_vec, i_vec)
        self.A[u_index] = u_vec + self.learn_rate * (err * i_vec - self.l2_reg * u_vec)
        self.B[i_index] = i_vec + self.learn_rate * (err * u_vec - self.l2_reg * i_vec)

    def recommend(self, u_index, N, history_vec):
        """
        Recommend Top-N items for the user u
        """

        if u_index not in self.known_users: raise ValueError('Error: the user is not known.')

        recos = []
        scores = np.abs(1. - np.dot(np.array([self.A[u_index]]), self.B.T)).reshape(self.B.shape[0])

        cnt = 0
        for i_index in np.argsort(scores):
            if history_vec[i_index] == 1: continue
            recos.append(i_index)
            cnt += 1
            if cnt == N: break

        return recos
```

<!-- #region id="eCK3lTm9NlJB" -->
## Dataset
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="8Xvnsut9NmRL" executionInfo={"status": "ok", "timestamp": 1635161378610, "user_tz": -330, "elapsed": 3026, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a6eca3c0-d45e-496b-89ba-0659a9c55220"
!wget -q --show-progress http://files.grouplens.org/datasets/movielens/ml-1m.zip
!unzip ml-1m.zip
```

```python colab={"base_uri": "https://localhost:8080/"} id="5FtYxISONn9y" executionInfo={"status": "ok", "timestamp": 1635161943005, "user_tz": -330, "elapsed": 3826, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="957469b5-56c4-48a1-ff20-572341a41bc8"
ratings = []

with open('ml-1m/ratings.dat') as f:
    lines = list(map(lambda l: list(map(int, l.rstrip().split('::'))), f.readlines()))
    for l in lines:
        # Since we consider positive-only feedback setting, ratings < 5 will be excluded.
        if l[2] == 5: ratings.append(l)

ratings = np.asarray(ratings)
ratings.shape
```

<!-- #region id="M5iYJ5R3O17t" -->
sorted by timestamp
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="SylXf9EJP1tr" executionInfo={"status": "ok", "timestamp": 1635161966125, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="26d28037-85a4-43c4-92fe-c4ed43bcd83e"
ratings = ratings[np.argsort(ratings[:, 3])]
ratings[:10]
```

```python colab={"base_uri": "https://localhost:8080/"} id="cnn8enF2P3cO" executionInfo={"status": "ok", "timestamp": 1635161976973, "user_tz": -330, "elapsed": 588, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="96d6a502-d7b8-4bed-821b-7dbee47da7ee"
users = set([])
items = set([])

for u, i, r, t in ratings:
    users.add(u)
    items.add(i)

users = list(users)
items = list(items)

n_user = len(users)
n_item = len(items)

n_user, n_item
```

<!-- #region id="ZSAq2_lHOpoL" -->
## Training

Simple Moving Average (SMA) with window size n=5000
<!-- #endregion -->

```python id="I6T01H5TOt07" executionInfo={"status": "ok", "timestamp": 1635162171618, "user_tz": -330, "elapsed": 179260, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
window_size = 5000 # for SMA
N = 10 # recall@10

history_mat = np.zeros((n_user, n_item))

# initialize
isgd = ISGD(n_user, n_item, k=100)

# to avoid cold-start, train initial 20% samples
n_ratings = ratings.shape[0]
n_train = int(n_ratings * 0.2)

for ri in range(n_train):
    u, i, rating, timestamp = ratings[ri]
    u_index = users.index(u)
    i_index = items.index(i)
    isgd.update(u_index, i_index)
    history_mat[u_index, i_index] = 1

avgs = []
sma = []

for ri in range(n_train, n_ratings):
    u, i, rating, timestamp = ratings[ri]
    u_index = users.index(u)
    i_index = items.index(i)
    
    # 1.
    if u_index in isgd.known_users:
        # If u is a known user, use the current model to recommend N items,
        recos = isgd.recommend(u_index, N, history_mat[u_index])
        
        # 2. Score the recommendation list given the true observed item i
        recall = 1 if (i_index in recos) else 0
        
        sma.append(recall)
        n = len(sma)
        if n > window_size: 
            del sma[0]
            n -= 1
        avgs.append(sum(sma) / float(n))
    
    # 3. update the model with the observed event
    isgd.update(u_index, i_index)
    history_mat[u_index, i_index] = 1
```

<!-- #region id="PSVnjSMsQAIT" -->
## Evaluation
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 229} id="SL8MZZbCOltw" executionInfo={"status": "ok", "timestamp": 1635162171624, "user_tz": -330, "elapsed": 55, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9bc1d881-080e-47fd-e63b-b1e2611e1e8a"
fig, ax = plt.subplots()
ax.plot(range(len(avgs)), avgs)
ax.set_xlabel('index')
ax.set_ylabel('recall@10')
ax.grid(True)
ax.set_xticks([0, 50000, 100000, 150000])
ax.set_yticks([0.00, 0.05, 0.10, 0.15])
fig.set_size_inches((5.5,3))
fig.patch.set_alpha(0.0)
plt.show()
```
