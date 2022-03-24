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

```python id="UV_mis-jdwLd" executionInfo={"status": "ok", "timestamp": 1628676759690, "user_tz": -330, "elapsed": 905, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-mlh"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python id="KRGLEjqMd3dV" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628676763112, "user_tz": -330, "elapsed": 3429, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="29ab6abd-12bc-41ef-e1fd-ef3e7e0f0675"
if not os.path.exists(project_path):
    !cp /content/drive/MyDrive/mykeys.py /content
    import mykeys
    !rm /content/mykeys.py
    path = "/content/" + project_name; 
    !mkdir "{path}"
    %cd "{path}"
    import sys; sys.path.append(path)
    !git config --global user.email "recotut@recohut.com"
    !git config --global user.name  "reco-tut"
    !git init
    !git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout main
else:
    %cd "{project_path}"
```

```python id="HWliEWwod3dX" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628677574362, "user_tz": -330, "elapsed": 776, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6b0b8876-8be9-43d1-944a-bb869cb88c72"
!git status
```

```python id="dGCJpyjLd3dY" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628677579588, "user_tz": -330, "elapsed": 1324, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d321e789-c3a3-4802-c754-081960e6c7b8"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python id="ClFYXwQcqWub" executionInfo={"status": "ok", "timestamp": 1628676808721, "user_tz": -330, "elapsed": 465, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import sys
sys.path.insert(0, './code')
```

<!-- #region id="t0QPGVexfcaM" -->
---
<!-- #endregion -->

<!-- #region id="BEudW5MKLC8E" -->
# Collaborative Filtering Comparison

In this notebook we compare different recommendation systems starting with the state-of-the-art LightGCN and going back to the winning algorithm for 2009's Netflix Prize competition, SVD++.

Models include in order are LightGCN, NGCF, SVAE, SVD++, and SVD. Each model has their own individual notebooks where we go more indepth, especially LightGCN and NGCF, where we implemented them from scratch in Tensorflow. 

The last cell compares the performance of the different models using ranking metrics:


*   Precision@k
*   Recall@k
*   Mean Average Precision (MAP)
*   Normalized Discounted Cumulative Gain (NDCG)

where $k=10$


<!-- #endregion -->

<!-- #region id="eESDthIVHdOY" -->
# Imports
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="sKYWteSRsyl0" executionInfo={"status": "ok", "timestamp": 1628676795829, "user_tz": -330, "elapsed": 26139, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0394f9a4-1641-4959-ce68-2770d748e50b"
!pip install -q surprise
```

```python id="m10O1R5R8IUO" executionInfo={"status": "ok", "timestamp": 1628676846212, "user_tz": -330, "elapsed": 1151, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import math
import numpy as np
import os
import pandas as pd
import random
import requests
import scipy.sparse as sp
import surprise
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.python.framework.ops import disable_eager_execution
from tqdm import tqdm

from utils import stratified_split, numpy_stratified_split
import build_features
import metrics
from models import SVAE
from models.GCN import LightGCN, NGCF
```

<!-- #region id="_c_Xsn7eHgpU" -->
# Prepare data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 376} id="BTTB4DsY1_Ch" executionInfo={"status": "ok", "timestamp": 1628676850862, "user_tz": -330, "elapsed": 717, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="60391cfa-406b-401e-cc9f-2e93241f7ea3"
fp = os.path.join('./data/bronze', 'u.data')
raw_data = pd.read_csv(fp, sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'])
print(f'Shape: {raw_data.shape}')
raw_data.sample(10, random_state=123)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 376} id="e9qxAgkc2mAc" executionInfo={"status": "ok", "timestamp": 1628676860325, "user_tz": -330, "elapsed": 1065, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cdcc1ca9-ee67-40ed-95d7-979a9cedcac0"
# Load movie titles.
fp = os.path.join('./data/bronze', 'u.item')
movie_titles = pd.read_csv(fp, sep='|', names=['movieId', 'title'], usecols = range(2), encoding='iso-8859-1')
print(f'Shape: {movie_titles.shape}')
movie_titles.sample(10, random_state=123)
```

```python colab={"base_uri": "https://localhost:8080/"} id="MhdGfZOW24rT" executionInfo={"status": "ok", "timestamp": 1628676880673, "user_tz": -330, "elapsed": 2196, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1757f467-67e1-4716-b513-dadc57c731a8"
train_size = 0.75
train, test = stratified_split(raw_data, 'userId', train_size)

print(f'Train Shape: {train.shape}')
print(f'Test Shape: {test.shape}')
print(f'Do they have the same users?: {set(train.userId) == set(test.userId)}')
```

```python colab={"base_uri": "https://localhost:8080/"} id="kV-Q7up827yC" executionInfo={"status": "ok", "timestamp": 1628676880675, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6f633f98-4bbe-4d8f-abdf-9f0f2150fed6"
combined = train.append(test)

n_users = combined['userId'].nunique()
print('Number of users:', n_users)

n_movies = combined['movieId'].nunique()
print('Number of movies:', n_movies)
```

```python id="RRzUqVW027v1" executionInfo={"status": "ok", "timestamp": 1628676880676, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Create DataFrame with reset index of 0-n_movies.
movie_new = combined[['movieId']].drop_duplicates()
movie_new['movieId_new'] = np.arange(len(movie_new))

train_reindex = pd.merge(train, movie_new, on='movieId', how='left')
# Reset index to 0-n_users.
train_reindex['userId_new'] = train_reindex['userId'] - 1  
train_reindex = train_reindex[['userId_new', 'movieId_new', 'rating']]

test_reindex = pd.merge(test, movie_new, on='movieId', how='left')
# Reset index to 0-n_users.
test_reindex['userId_new'] = test_reindex['userId'] - 1
test_reindex = test_reindex[['userId_new', 'movieId_new', 'rating']]

# Create dictionaries so we can convert to and from indexes
item2id = dict(zip(movie_new['movieId'], movie_new['movieId_new']))
id2item = dict(zip(movie_new['movieId_new'], movie_new['movieId']))
user2id = dict(zip(train['userId'], train_reindex['userId_new']))
id2user = dict(zip(train_reindex['userId_new'], train['userId']))
```

```python colab={"base_uri": "https://localhost:8080/"} id="pjwETtvX27rK" executionInfo={"status": "ok", "timestamp": 1628676895186, "user_tz": -330, "elapsed": 11964, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="aab6cdd0-6bc0-458b-c854-a55673833452"
# Create user-item graph (sparse matix where users are rows and movies are columns.
# 1 if a user reviewed that movie, 0 if they didn't).
R = sp.dok_matrix((n_users, n_movies), dtype=np.float32)
R[train_reindex['userId_new'], train_reindex['movieId_new']] = 1

# Create the adjaceny matrix with the user-item graph.
adj_mat = sp.dok_matrix((n_users + n_movies, n_users + n_movies), dtype=np.float32)

# List of lists.
adj_mat.tolil()
R = R.tolil()

# Put together adjacency matrix. Movies and users are nodes/vertices.
# 1 if the movie and user are connected.
adj_mat[:n_users, n_users:] = R
adj_mat[n_users:, :n_users] = R.T

adj_mat
```

```python id="-hzlq37t27mv" executionInfo={"status": "ok", "timestamp": 1628676895188, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Calculate degree matrix D (for every row count the number of nonzero entries)
D_values = np.array(adj_mat.sum(1))

# Square root and inverse.
D_inv_values = np.power(D_values  + 1e-9, -0.5).flatten()
D_inv_values[np.isinf(D_inv_values)] = 0.0

 # Create sparse matrix with the values of D^(-0.5) are the diagonals.
D_inv_sq_root = sp.diags(D_inv_values)

# Eval (D^-0.5 * A * D^-0.5).
norm_adj_mat = D_inv_sq_root.dot(adj_mat).dot(D_inv_sq_root)
```

```python colab={"base_uri": "https://localhost:8080/"} id="TjPnmm4l3GDK" executionInfo={"status": "ok", "timestamp": 1628676899585, "user_tz": -330, "elapsed": 4412, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4eab03b6-5471-4b16-abdb-ece3821c236c"
# to COOrdinate format first ((row, column), data)
coo = norm_adj_mat.tocoo().astype(np.float32)

# create an index that will tell SparseTensor where the non-zero points are
indices = np.mat([coo.row, coo.col]).transpose()

# covert to sparse tensor
A_tilde = tf.SparseTensor(indices, coo.data, coo.shape)
A_tilde
```

<!-- #region id="YuRBZNznJpD6" -->
# Train models
<!-- #endregion -->

<!-- #region id="KshSoBWYI0rC" -->
## Graph Convoultional Networks (GCNs)
<!-- #endregion -->

<!-- #region id="kFRUrT-4HnW1" -->
### Light Graph Convolution Network (LightGCN)
<!-- #endregion -->

```python id="p7-joA-u3GBB" executionInfo={"status": "ok", "timestamp": 1628676899586, "user_tz": -330, "elapsed": 32, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
light_model = LightGCN(A_tilde,
                 n_users = n_users,
                 n_items = n_movies,
                 n_layers = 3)
```

```python colab={"base_uri": "https://localhost:8080/"} id="7wUU0odC3F8f" executionInfo={"status": "ok", "timestamp": 1628676953665, "user_tz": -330, "elapsed": 54108, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0ca4667d-445d-4af6-b5f4-05de045a2e6f"
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
light_model.fit(epochs=25, batch_size=1024, optimizer=optimizer)
```

<!-- #region id="oRp6NESrHqPf" -->
### Neural Graph Collaborative Filtering (NGCF)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="z0q0dyg1zTQP" executionInfo={"status": "ok", "timestamp": 1628677002247, "user_tz": -330, "elapsed": 48603, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8732edb2-198c-44a0-d2d1-1e151c10b995"
ngcf_model = NGCF(A_tilde,
                  n_users = n_users,
                  n_items = n_movies,
                  n_layers = 3
                  )

ngcf_model.fit(epochs=25, batch_size=1024, optimizer=optimizer)
```

<!-- #region id="Vsq8nRDbIqUm" -->
### Recommend with LightGCN and NGCF
<!-- #endregion -->

```python id="n_V5VkktAwMh" executionInfo={"status": "ok", "timestamp": 1628677003039, "user_tz": -330, "elapsed": 818, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Convert test user ids to the new ids
users = np.array([user2id[x] for x in test['userId'].unique()])

recs = []
for model in [light_model, ngcf_model]:
    recommendations = model.recommend(users, k=10)
    recommendations = recommendations.replace({'userId': id2user, 'movieId': id2item})
    recommendations = recommendations.merge(movie_titles,
                                                    how='left',
                                                    on='movieId'
                                                    )[['userId', 'movieId', 'title', 'prediction']]

    # Create column with the predicted movie's rank for each user 
    top_k = recommendations.copy()
    top_k['rank'] = recommendations.groupby('userId', sort=False).cumcount() + 1  # For each user, only include movies recommendations that are also in the test set

    recs.append(top_k)
```

<!-- #region id="VI56qeyiIAQE" -->
## Standard Variational Autoencoder (SVAE)
<!-- #endregion -->

```python id="t7VC52PGVFlD" executionInfo={"status": "ok", "timestamp": 1628677022498, "user_tz": -330, "elapsed": 3948, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Binarize the data (only keep ratings >= 4)
df_preferred = raw_data[raw_data['rating'] > 3.5]
df_low_rating = raw_data[raw_data['rating'] <= 3.5]

df = df_preferred.groupby('userId').filter(lambda x: len(x) >= 5)
df = df.groupby('movieId').filter(lambda x: len(x) >= 1)

# Obtain both usercount and itemcount after filtering
usercount = df[['userId']].groupby('userId', as_index = False).size()
itemcount = df[['movieId']].groupby('movieId', as_index = False).size()

unique_users =sorted(df.userId.unique())
np.random.seed(123)
unique_users = np.random.permutation(unique_users)

HELDOUT_USERS = 200

# Create train/validation/test users
n_users = len(unique_users)
train_users = unique_users[:(n_users - HELDOUT_USERS * 2)]
val_users = unique_users[(n_users - HELDOUT_USERS * 2) : (n_users - HELDOUT_USERS)]
test_users = unique_users[(n_users - HELDOUT_USERS):]

train_set = df.loc[df['userId'].isin(train_users)]
val_set = df.loc[df['userId'].isin(val_users)]
test_set = df.loc[df['userId'].isin(test_users)]
unique_train_items = pd.unique(train_set['movieId'])
val_set = val_set.loc[val_set['movieId'].isin(unique_train_items)]
test_set = test_set.loc[test_set['movieId'].isin(unique_train_items)]

# Instantiate the sparse matrix generation for train, validation and test sets
# use list of unique items from training set for all sets
am_train = build_features.AffinityMatrix(df=train_set, items_list=unique_train_items)
am_val = build_features.AffinityMatrix(df=val_set, items_list=unique_train_items)
am_test = build_features.AffinityMatrix(df=test_set, items_list=unique_train_items)

# Obtain the sparse matrix for train, validation and test sets
train_data, _, _ = am_train.gen_affinity_matrix()
val_data, val_map_users, val_map_items = am_val.gen_affinity_matrix()
test_data, test_map_users, test_map_items = am_test.gen_affinity_matrix()

# Split validation and test data into training and testing parts
val_data_tr, val_data_te = numpy_stratified_split(val_data, ratio=0.75, seed=123)
test_data_tr, test_data_te = numpy_stratified_split(test_data, ratio=0.75, seed=123)

# Binarize train, validation and test data
train_data = np.where(train_data > 3.5, 1.0, 0.0)
val_data = np.where(val_data > 3.5, 1.0, 0.0)
test_data = np.where(test_data > 3.5, 1.0, 0.0)

# Binarize validation data
val_data_tr = np.where(val_data_tr > 3.5, 1.0, 0.0)
val_data_te_ratings = val_data_te.copy()
val_data_te = np.where(val_data_te > 3.5, 1.0, 0.0)

# Binarize test data: training part 
test_data_tr = np.where(test_data_tr > 3.5, 1.0, 0.0)

# Binarize test data: testing part (save non-binary version in the separate object, will be used for calculating NDCG)
test_data_te_ratings = test_data_te.copy()
test_data_te = np.where(test_data_te > 3.5, 1.0, 0.0)

# retrieve real ratings from initial dataset 
test_data_te_ratings=pd.DataFrame(test_data_te_ratings)
val_data_te_ratings=pd.DataFrame(val_data_te_ratings)

for index,i in df_low_rating.iterrows():
    user_old= i['userId'] # old value 
    item_old=i['movieId'] # old value 

    if (test_map_users.get(user_old) is not None)  and (test_map_items.get(item_old) is not None) :
        user_new=test_map_users.get(user_old) # new value 
        item_new=test_map_items.get(item_old) # new value 
        rating=i['rating'] 
        test_data_te_ratings.at[user_new,item_new]= rating   

    if (val_map_users.get(user_old) is not None)  and (val_map_items.get(item_old) is not None) :
        user_new=val_map_users.get(user_old) # new value 
        item_new=val_map_items.get(item_old) # new value 
        rating=i['rating'] 
        val_data_te_ratings.at[user_new,item_new]= rating   


val_data_te_ratings=val_data_te_ratings.to_numpy()    
test_data_te_ratings=test_data_te_ratings.to_numpy()    
```

```python colab={"base_uri": "https://localhost:8080/"} id="rM8THbGWVFik" executionInfo={"status": "ok", "timestamp": 1628677182024, "user_tz": -330, "elapsed": 155241, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="73eeb9a6-c72e-4cf1-9f7a-4c613a2bdd26"
disable_eager_execution()
svae_model = SVAE.StandardVAE(n_users=train_data.shape[0],
                                   original_dim=train_data.shape[1], 
                                   intermediate_dim=200, 
                                   latent_dim=64, 
                                   n_epochs=400, 
                                   batch_size=100, 
                                   k=10,
                                   verbose=0,
                                   seed=123,
                                   drop_encoder=0.5,
                                   drop_decoder=0.5,
                                   annealing=False,
                                   beta=1.0
                                   )

svae_model.fit(x_train=train_data,
          x_valid=val_data,
          x_val_tr=val_data_tr,
          x_val_te=val_data_te_ratings,
          mapper=am_val
          )
```

<!-- #region id="LK6CVrxqI_Ve" -->
### Recommend with SVAE
<!-- #endregion -->

```python id="N9jsBPW-XQIn" executionInfo={"status": "ok", "timestamp": 1628677182026, "user_tz": -330, "elapsed": 30, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Model prediction on the training part of test set 
top_k =  svae_model.recommend_k_items(x=test_data_tr,k=10,remove_seen=True)

# Convert sparse matrix back to df
recommendations = am_test.map_back_sparse(top_k, kind='prediction')
test_df = am_test.map_back_sparse(test_data_te_ratings, kind='ratings') # use test_data_te_, with the original ratings

# Create column with the predicted movie's rank for each user 
top_k = recommendations.copy()
top_k['rank'] = recommendations.groupby('userId', sort=False).cumcount() + 1  # For each user, only include movies recommendations that are also in the test set

recs.append(top_k)
```

<!-- #region id="WJllF7bdJHHN" -->
## Singular Value Decomposition (SVD)
<!-- #endregion -->

<!-- #region id="uzwll5SnITUq" -->
### SVD++
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="qbRc2KJ_8S2W" executionInfo={"status": "ok", "timestamp": 1628677324150, "user_tz": -330, "elapsed": 142147, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="14c6711d-aada-4eff-dc57-70e560ee3c12"
surprise_train = surprise.Dataset.load_from_df(train.drop('timestamp', axis=1), reader=surprise.Reader('ml-100k')).build_full_trainset()
svdpp = surprise.SVDpp(random_state=0, n_factors=64, n_epochs=10, verbose=True)
svdpp.fit(surprise_train)
```

<!-- #region id="od_CZ4pCJQef" -->
### SVD
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="C52AB0rs-kOu" executionInfo={"status": "ok", "timestamp": 1628677326290, "user_tz": -330, "elapsed": 2169, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4f95672f-1643-471c-9672-d1af4b7d0a47"
svd = surprise.SVD(random_state=0, n_factors=64, n_epochs=10, verbose=True)
svd.fit(surprise_train)
```

<!-- #region id="XkGR-_YyJSz6" -->
### Recommend with SVD++ and SVD
<!-- #endregion -->

```python id="kLUfa3Xf9vaJ" executionInfo={"status": "ok", "timestamp": 1628677532805, "user_tz": -330, "elapsed": 206521, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
for model in [svdpp, svd]:
    predictions = []
    users = train['userId'].unique()
    items = train['movieId'].unique()

    for user in users:
            for item in items:
                predictions.append([user, item, model.predict(user, item).est])

    predictions = pd.DataFrame(predictions, columns=['userId', 'movieId', 'prediction'])

    # Remove movies already seen by users
    # Create column of all 1s
    temp = train[['userId', 'movieId']].copy()
    temp['seen'] = 1

    # Outer join and remove movies that have alread been seen (seen=1)
    merged = pd.merge(temp, predictions, on=['userId', 'movieId'], how="outer")
    merged = merged[merged['seen'].isnull()].drop('seen', axis=1)

    # Create filter for users that appear in both the train and test set
    common_users = set(test['userId']).intersection(set(predictions['userId']))

    # Filter the test and predictions so they have the same users between them
    test_common = test[test['userId'].isin(common_users)]
    svd_pred_common = merged[merged['userId'].isin(common_users)]

    if len(set(merged['userId'])) != len(set(test['userId'])):
        print('Number of users in train and test are NOT equal')
        print(f"# of users in train and test respectively: {len(set(merged['userId']))}, {len(set(test['userId']))}")
        print(f"# of users in BOTH train and test: {len(set(svd_pred_common['userId']))}")
        continue
        
    # From the predictions, we want only the top k for each user,
    # not all the recommendations.
    # Extract the top k recommendations from the predictions
    top_movies = svd_pred_common.groupby('userId', as_index=False).apply(lambda x: x.nlargest(10, 'prediction')).reset_index(drop=True)
    top_movies['rank'] = top_movies.groupby('userId', sort=False).cumcount() + 1
    
    top_k = top_movies.copy()
    top_k['rank'] = top_movies.groupby('userId', sort=False).cumcount() + 1  # For each user, only include movies recommendations that are also in the test set
    
    recs.append(top_k)
```

<!-- #region id="VybicxKFJZXP" -->
# Compare performance
<!-- #endregion -->

<!-- #region id="8R5oKqnp_4Q4" -->
Looking at all 5 of our models, we can see that the state-of-the-art model LightGCN vastly outperforms all other models. When compared to SVD++, a widely used algorithm during the Netflix Prize competition, LightGCN achieves an increase in **Percision@k by 29%, Recall@k by 18%, MAP by 12%, and NDCG by 35%**.

NGCF is the older sister model to LightGCN, but only by a single year. We can see how LightGCN improves in ranking metrics compared to NGCF by simply removing unnecessary operations. 

In conclusion, this demonstrates how far recommendation systems have advanced since 2009, and how new model architectures with notable performance increases can be developed in the span of just 1-2 years.
<!-- #endregion -->

```python id="ZA8JdPhf555b" executionInfo={"status": "ok", "timestamp": 1628677532806, "user_tz": -330, "elapsed": 28, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
model_names = ['LightGCN', 'NGCF', 'SVAE', 'SVD++', 'SVD']
comparison = pd.DataFrame(columns=['Algorithm', 'Precision@k', 'Recall@k', 'MAP', 'NDCG'])

# Convert test user ids to the new ids
users = np.array([user2id[x] for x in test['userId'].unique()])

for rec, name in zip(recs, model_names):
    tester = test_df if name == 'SVAE' else test

    pak = metrics.precision_at_k(rec, tester, 'userId', 'movieId', 'rank')
    rak = metrics.recall_at_k(rec, tester, 'userId', 'movieId', 'rank')
    map = metrics.mean_average_precision(rec, tester, 'userId', 'movieId', 'rank')
    ndcg = metrics.ndcg(rec, tester, 'userId', 'movieId', 'rank')

    comparison.loc[len(comparison)] = [name, pak, rak, map, ndcg]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="ECyE_Qls7KkQ" executionInfo={"status": "ok", "timestamp": 1628677532808, "user_tz": -330, "elapsed": 26, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="32c8dfd6-a4a1-46b6-bd44-c2e05ae479d5"
comparison
```

<!-- #region id="9DUTuMtTLFcA" -->
# References:

1.   Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang & Meng Wang, LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation, 2020, https://arxiv.org/abs/2002.02126
2.   Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, & Tata-Seng Chua, Neural Graph Collaorative Filtering, 2019, https://arxiv.org/abs/1905.08108
3.   Microsoft SVAE implementation: https://github.com/microsoft/recommenders/blob/main/examples/02_model_collaborative_filtering/standard_vae_deep_dive.ipynb
4. Simon Gower, Netflix Prize and SVD, 2014, https://www.semanticscholar.org/paper/Netflix-Prize-and-SVD-Gower/ce7b81b46939d7852dbb30538a7796e69fdd407c

<!-- #endregion -->
