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

```python
import os
project_name = "reco-tut-mlh"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python
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

```python
!git status
```

```python
!git pull --rebase origin main
```

```python
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python
import sys
sys.path.insert(0,'./code')
```

---


# Standard Variational Autoencoder (SVAE)

The Standard Variational Autoencoder (SVAE), SVAE uses an autoencoder to generate a salient feature representation of users, learning a latent vector for each user. The decoder then takes this latent representation and outputs a probability distribution over all items; we get probabilities of all the movies being watched by each user.


# Imports

```python
import numpy as np
import os
import pandas as pd

from utils import numpy_stratified_split
import build_features
import metrics
from models import SVAE
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()
```

# Prepare Data

```python
fp = os.path.join('./data/bronze', 'u.data')
raw_data = pd.read_csv(fp, sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'])
print(f'Shape: {raw_data.shape}')
raw_data.sample(5, random_state=123)
```

```python
# Binarize the data (only keep ratings >= 4)
df_preferred = raw_data[raw_data['rating'] > 3.5]
print (df_preferred.shape)
df_low_rating = raw_data[raw_data['rating'] <= 3.5]

df_preferred.head(10)
```

```python
# Keep users who clicked on at least 5 movies
df = df_preferred.groupby('userId').filter(lambda x: len(x) >= 5)

# Keep movies that were clicked on by at least on 1 user
df = df.groupby('movieId').filter(lambda x: len(x) >= 1)

print(df.shape)
```

```python
# Obtain both usercount and itemcount after filtering
usercount = df[['userId']].groupby('userId', as_index = False).size()
itemcount = df[['movieId']].groupby('movieId', as_index = False).size()

# Compute sparsity after filtering
sparsity = 1. * raw_data.shape[0] / (usercount.shape[0] * itemcount.shape[0])

print("After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)" % 
      (raw_data.shape[0], usercount.shape[0], itemcount.shape[0], sparsity * 100))
```

## Split

```python
unique_users =sorted(df.userId.unique())
np.random.seed(123)
unique_users = np.random.permutation(unique_users)
```

```python
HELDOUT_USERS = 200

# Create train/validation/test users
n_users = len(unique_users)
print("Number of unique users:", n_users)

train_users = unique_users[:(n_users - HELDOUT_USERS * 2)]
print("\nNumber of training users:", len(train_users))

val_users = unique_users[(n_users - HELDOUT_USERS * 2) : (n_users - HELDOUT_USERS)]
print("\nNumber of validation users:", len(val_users))

test_users = unique_users[(n_users - HELDOUT_USERS):]
print("\nNumber of test users:", len(test_users))
```

```python
# For training set keep only users that are in train_users list
train_set = df.loc[df['userId'].isin(train_users)]
print("Number of training observations: ", train_set.shape[0])

# For validation set keep only users that are in val_users list
val_set = df.loc[df['userId'].isin(val_users)]
print("\nNumber of validation observations: ", val_set.shape[0])

# For test set keep only users that are in test_users list
test_set = df.loc[df['userId'].isin(test_users)]
print("\nNumber of test observations: ", test_set.shape[0])

# train_set/val_set/test_set contain user - movie interactions with rating 4 or 5
```

```python
# Obtain list of unique movies used in training set
unique_train_items = pd.unique(train_set['movieId'])
print("Number of unique movies that rated in training set", unique_train_items.size)
```

```python
# For validation set keep only movies that used in training set
val_set = val_set.loc[val_set['movieId'].isin(unique_train_items)]
print("Number of validation observations after filtering: ", val_set.shape[0])

# For test set keep only movies that used in training set
test_set = test_set.loc[test_set['movieId'].isin(unique_train_items)]
print("\nNumber of test observations after filtering: ", test_set.shape[0])
```

```python
# Instantiate the sparse matrix generation for train, validation and test sets
# use list of unique items from training set for all sets
am_train = build_features.AffinityMatrix(df=train_set, items_list=unique_train_items)

am_val = build_features.AffinityMatrix(df=val_set, items_list=unique_train_items)

am_test = build_features.AffinityMatrix(df=test_set, items_list=unique_train_items)
```

```python
# Obtain the sparse matrix for train, validation and test sets
train_data, _, _ = am_train.gen_affinity_matrix()
print(train_data.shape)

val_data, val_map_users, val_map_items = am_val.gen_affinity_matrix()
print(val_data.shape)

test_data, test_map_users, test_map_items = am_test.gen_affinity_matrix()
print(test_data.shape)
```

```python
# Split validation and test data into training and testing parts
val_data_tr, val_data_te = numpy_stratified_split(val_data, ratio=0.75, seed=123)
test_data_tr, test_data_te = numpy_stratified_split(test_data, ratio=0.75, seed=123)
```

```python
# Binarize train, validation and test data
train_data = np.where(train_data > 3.5, 1.0, 0.0)
val_data = np.where(val_data > 3.5, 1.0, 0.0)
test_data = np.where(test_data > 3.5, 1.0, 0.0)

# Binarize validation data: training part  
val_data_tr = np.where(val_data_tr > 3.5, 1.0, 0.0)
# Binarize validation data: testing part (save non-binary version in the separate object, will be used for calculating NDCG)
val_data_te_ratings = val_data_te.copy()
val_data_te = np.where(val_data_te > 3.5, 1.0, 0.0)

# Binarize test data: training part 
test_data_tr = np.where(test_data_tr > 3.5, 1.0, 0.0)

# Binarize test data: testing part (save non-binary version in the separate object, will be used for calculating NDCG)
test_data_te_ratings = test_data_te.copy()
test_data_te = np.where(test_data_te > 3.5, 1.0, 0.0)
```

```python
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

# SVAE

```python
INTERMEDIATE_DIM = 200
LATENT_DIM = 64
EPOCHS = 400
BATCH_SIZE = 100
```

```python
model = SVAE.StandardVAE(n_users=train_data.shape[0], # Number of unique users in the training set
                                   original_dim=train_data.shape[1], # Number of unique items in the training set
                                   intermediate_dim=INTERMEDIATE_DIM, 
                                   latent_dim=LATENT_DIM, 
                                   n_epochs=EPOCHS, 
                                   batch_size=BATCH_SIZE, 
                                   k=10,
                                   verbose=0,
                                   seed=123,
                                   drop_encoder=0.5,
                                   drop_decoder=0.5,
                                   annealing=False,
                                   beta=1.0
                                   )
```

```python
%%time
model.fit(x_train=train_data,
          x_valid=val_data,
          x_val_tr=val_data_tr,
          x_val_te=val_data_te_ratings, # with the original ratings
          mapper=am_val
          )
```

# Recommend

```python
# Model prediction on the training part of test set 
top_k =  model.recommend_k_items(x=test_data_tr,k=10,remove_seen=True)

# Convert sparse matrix back to df
recommendations = am_test.map_back_sparse(top_k, kind='prediction')
test_df = am_test.map_back_sparse(test_data_te_ratings, kind='ratings') # use test_data_te_, with the original ratings
```

## Evaluation metrics

```python
# Create column with the predicted movie's rank for each user 
top_k = recommendations.copy()
top_k['rank'] = recommendations.groupby('userId', sort=False).cumcount() + 1  # For each user, only include movies recommendations that are also in the test set
```

```python
precision_at_k = metrics.precision_at_k(top_k, test_df, 'userId', 'movieId', 'rank')
recall_at_k = metrics.recall_at_k(top_k, test_df, 'userId', 'movieId', 'rank')
mean_average_precision = metrics.mean_average_precision(top_k, test_df, 'userId', 'movieId', 'rank')
ndcg = metrics.ndcg(top_k, test_df, 'userId', 'movieId', 'rank')
```

```python
print(f'Precision: {precision_at_k:.6f}',
      f'Recall: {recall_at_k:.6f}',
      f'MAP: {mean_average_precision:.6f} ',
      f'NDCG: {ndcg:.6f}', sep='\n')
```

<!-- #region -->
# References


1.   Kilol Gupta, Mukunds Y. Raghuprasad, Pankhuri Kumar, A Hybrid Variational Autoencoder for Collaborative Filtering, 2018, https://arxiv.org/pdf/1808.01006.pdf

2.   Microsoft SVAE implementation: https://github.com/microsoft/recommenders/blob/main/examples/02_model_collaborative_filtering/standard_vae_deep_dive.ipynb

<!-- #endregion -->
