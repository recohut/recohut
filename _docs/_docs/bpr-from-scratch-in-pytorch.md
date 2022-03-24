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

<!-- #region id="OAR8iFNhmhlu" -->
# BPR from scratch in PyTorch
<!-- #endregion -->

<!-- #region id="rfKNHLT9nCbX" -->
### Setup
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="JKJh70rrmjXm" executionInfo={"status": "ok", "timestamp": 1633087583427, "user_tz": -330, "elapsed": 1493, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8e3d127d-097c-4e9a-c497-ab68d60da33f"
!wget -q --show-progress https://github.com/leafinity/gradient_dscent_svd/raw/master/the-movies-dataset/numpy/users.npy
!wget -q --show-progress https://github.com/leafinity/gradient_dscent_svd/raw/master/the-movies-dataset/numpy/movies.npy
!wget -q --show-progress https://github.com/leafinity/gradient_dscent_svd/raw/master/the-movies-dataset/numpy/small_ratings.npy
```

```python colab={"base_uri": "https://localhost:8080/"} id="6xiIBYK3oRub" executionInfo={"status": "ok", "timestamp": 1633088025327, "user_tz": -330, "elapsed": 1570, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3cf490c6-7481-4f20-9b6c-d566431bd0e8"
# this link is temporary, you can generate a new one by visiting kaggle data site https://www.kaggle.com/rounakbanik/the-movies-dataset
!wget -O meta.zip -q --show-progress "https://storage.googleapis.com/kaggle-data-sets/3405/6663/compressed/movies_metadata.csv.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20211001%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20211001T113251Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=3144a3fa20ed4c96a11e2c8a91ea021b3ecbdd0d8b7452ede9395d43bd93b0a808cf1ec71d08416ff2c12e7d8861c8154fb4936b7544f78de4668b1dc21926cc7848b44447eeae139fd5e31fe43c8ae7abe8c39e1d8c5fc61e88a6d7d10805a67740fefd53d38908d73e34f902a5afdc0008dbecfa3873a7bb55740ef127e90e6f95056bfe7d7dc91e8f1306153918dc8bcc3f22224d13ea4a4e639fb09abf5a0470d7ad0f1c8b320192be2f2b04b317d95c11de6c5e618ef95d7fe99745d1200ed83f65c4ae534342e97bd638fb50ea0e27b05a2a1c358fa4529a3f442ec8c9d95e75780a3ede5849fefa66a9b1767de4cce3a49815c117806f4dbae0553b47"
!unzip meta.zip
```

```python id="sfpfsxzDmutX"
import torch
import pandas as pd
import numpy as np
```

```python id="rli5q1XAmxfV"
dtype = torch.float
device = torch.device('cpu')
```

<!-- #region id="0mYBOX7vnEKj" -->
### Loading
<!-- #endregion -->

```python id="q14G9s3amx03"
rating = np.load('small_ratings.npy')
users = np.load('users.npy')[:100]
movies = np.load('movies.npy')[:100]
```

<!-- #region id="K11EONpSnGny" -->
### Preparation
<!-- #endregion -->

```python id="SRyg0iyfm0Vh"
users_num, movies_num, k = len(users), len(movies), 5
rating_len = len(rating)
```

```python colab={"base_uri": "https://localhost:8080/"} id="-luFbm-Sm0x9" executionInfo={"status": "ok", "timestamp": 1633087622766, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="def9e623-d672-409b-8736-30e86a788927"
print(rating[:, 2].max())
print(rating[:, 0].max())
print(len(users)-1)
print(rating_len)
```

```python colab={"base_uri": "https://localhost:8080/"} id="-JNAUN2Hm4eB" executionInfo={"status": "ok", "timestamp": 1633087631938, "user_tz": -330, "elapsed": 489, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="eee41e11-b41c-4f4e-ec6f-89a5bff14852"
# normalization
rating[:, 2] -= 2.5
rating[:, 2] /= 2.5

# thita
W = torch.randn(users_num, k, device=device, dtype=dtype) / 10
H = torch.randn(movies_num, k, device=device, dtype=dtype) / 10

print(rating[:, 2].max())
```

```python colab={"base_uri": "https://localhost:8080/"} id="m9Nt7TsAm60n" executionInfo={"status": "ok", "timestamp": 1633087647535, "user_tz": -330, "elapsed": 1336, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f9579ee6-3a74-4488-e6ed-e8f7c75e0b8c"
ds = []
for ui, i, ri in rating:
    for uj, j, rj in rating:
        if ui != uj or i == j:
            continue
        if ri > rj:
            ds.append((int(ui), int(i), int(j)))

ds = list(set(ds))

ds[np.random.randint(len(ds))]
```

<!-- #region id="IgD6s1q_m-bN" -->
### Training
<!-- #endregion -->

```python id="txDA7UBcnJJs"
def predict(u, i):
    Wu = W[u].view(1, W[u].size()[0])
    return torch.mv(Wu, H[i])
```

```python colab={"base_uri": "https://localhost:8080/"} id="SXXIM2AenNs7" executionInfo={"status": "ok", "timestamp": 1633087714096, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0a14b47c-ecd6-4bdb-d28b-09e0c2b45ece"
def predict_diff(u, i, j):
    return  (predict(u, i) - predict(u, j))[0]

print(predict_diff(0, 0, 1))
```

```python id="MESdIow3nOxB"
def partial_BPR(x_uij, partial_x):
    exp_x = np.exp(-x_uij)
    return exp_x / (1 + exp_x) * partial_x
```

```python colab={"base_uri": "https://localhost:8080/"} id="llgnrfYMnQo_" executionInfo={"status": "ok", "timestamp": 1633087808714, "user_tz": -330, "elapsed": 81870, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7aee9156-30a3-452c-90ca-067d8495bec0"
iteration = 100000
lr = 1e-4

def train(W, H, lr=1e-3, rr=0.02):
    for itr in range(iteration):
        u, i, j = ds[np.random.randint(len(ds))]
        x_uij = predict_diff(u, i, j)

        for f in range(k):
            W[u][f] -= lr * (partial_BPR(x_uij, H[i][f] - H[j][f]) + rr * W[u][f])
            H[i][f] -= lr * (partial_BPR(x_uij, W[u][f]) + rr * H[i][f])
            H[j][f] -= lr * (partial_BPR(x_uij, -W[u][f]) + rr * H[f][f])
        
        if itr % 10000 == 0:
            print(W[18])
            print(H[0])
            
    return W, H

W, H = train(W, H, lr)
```

```python colab={"base_uri": "https://localhost:8080/"} id="pWmYk73KnSHP" executionInfo={"status": "ok", "timestamp": 1633087808715, "user_tz": -330, "elapsed": 36, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="59469368-6cc6-49cb-df9f-0a434aaa91f2"
print(W[18])
print(H[0])
```

```python colab={"base_uri": "https://localhost:8080/"} id="Bs5Xs4YhnTvx" executionInfo={"status": "ok", "timestamp": 1633087808716, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9f5af9bf-8629-43b6-d880-48a5bf3a037e"
print(W[18])
print(H[0])
```

<!-- #region id="PBxUPPmGnVBr" -->
### Evaluation
<!-- #endregion -->

```python id="MQD_Ykd9nWov"
user_id = 18
```

```python id="66iXyRrKnYeb"
Wu = W[user_id].view(1, W[user_id].size()[0])
prediction = list(zip(list(range(movies_num)), torch.mm(Wu, H.t()).tolist()[0]))
prediction.sort(key=lambda x: x[1], reverse=True)
```

```python id="aorziDEBnYbm"
movie_rates = []
movie_predict_rates = []

for u, i, r in rating:
    if u == user_id:
        movie_rates.append((int(i), r))
```

```python id="5p1XW6pOnYZT"
import json
movie_data = []
df = pd.read_csv('movies_metadata.csv')

for index, row in df.iloc[:, [3, 8]].iterrows():
    movie_data += [{'title': row['original_title'], 'genres': [x['name'] for x in json.loads(row['genres'].replace('\'', '"'))]}]
```

```python colab={"base_uri": "https://localhost:8080/"} id="9GhIiDp7nYU4" executionInfo={"status": "ok", "timestamp": 1633088102422, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="49295d29-38c8-4a50-c728-6004062c2148"
print('User ', users[user_id])
print('from rating, he/she likes:')
print('%s %25s %43s' % ('movie_id', 'movie_title', 'movie_genres'))
for m, r in movie_rates:
    if r > 0.5:
        mid = movies[m]-1
        print('%8s %25s %43s' % (mid, movie_data[mid]['title'][:24], movie_data[mid]['genres'][:4]))

print('')
print('from rating, he/she might like:')
print('%s %25s %43s' % ('movie_id', 'movie_title', 'movie_genres'))
for m, r in prediction[:5]:
    mid = movies[m]-1
    r = r * 2.5 + 2.5
    print('%8s %25s %43s' % (mid, movie_data[mid]['title'][:24], movie_data[mid]['genres'][:4]))
```
