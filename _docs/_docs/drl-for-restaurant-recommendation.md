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

```python colab={"base_uri": "https://localhost:8080/"} id="pCTg3xVIukbF" executionInfo={"status": "ok", "timestamp": 1634802571824, "user_tz": -330, "elapsed": 30580, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="72331c9f-2ba5-4b0f-fe2c-0eb663627618"
# https://www.kaggle.com/yelp-dataset/yelp-dataset/version/6
# these are temp links, go to data url and generate fresh ones
!wget -q --show-progress -O user.csv.zip "https://storage.googleapis.com/kaggle-data-sets/10100/16731/compressed/yelp_user.csv.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20211021%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20211021T074650Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=459f7d432a97832d9a1610cb71d22fa684cc720c5417c53b749bb1d51fa9d949734f56ad19a63ad3c5008d25003d07e4b2c51a872da1d6cb9d8fcb96ff9acdd829b3e098458a49cdde22fb26d012e9501b6f0474ec0715c7822a1a545a1ac16e325620fc9abddb197c70e2035cefb9592d14fdf807f4abd915e3e3efbe67c35efe494f26282654c568cb469afbcbc11b57f8fefcacebfbef987a96a7d9e55cf92f00613b88c9c6e91c7fa804bb8f2a38917e73491f6c491bde8ba17989702c7ea6fbe5a17430e4d79c0c8bce5090c27e2e7fb9fbbada2cc1856ad733a542365f553e05675e8de563e23c5d9e9afb917c02a72fb77e15581ea5a5d5a1fc576a84"
!wget -q --show-progress -O review.csv.zip "https://storage.googleapis.com/kaggle-data-sets/10100/16731/compressed/yelp_review.csv.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20211021%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20211021T074659Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=1ad98bf7c7f5ef70a4dc84ec4c47ea5decb32e80dcfcf0684fd19b646734642e343569251e8118880caf3e9eb2bfcc286fee002b571de0de619763bac9d011191e9c77c47709aed879c61f82e516443554e6725ef45413b5e45ea7248660c471129364e79ea02b3c811ceb8221448a1683215b0817f3384a4f666bd6abb235db1269003b81d3397106d0991b4b281a52c97fb5020be66e6b3d7a55211ffdcf88dfd98ccc9dff2d65ef091b9d4315ec0668a94496d1dd5d988a612dc97a23013fe17529c55edbf10abfe503bc4a77d5e5b5258a10a71db10f403ef6ac04cee035722998e822abd504fec15daed2c0c378d52c3ed81182e59e505960c2a045fdae"
!wget -q --show-progress -O business.csv.zip "https://storage.googleapis.com/kaggle-data-sets/10100/16731/compressed/yelp_business.csv.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20211021%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20211021T074709Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=5f46b4972fea3753a9809830699ed4e51fb676c41df168cf7273b478c4340e19b5290ff24643847bcdd0c90388aeff081e01bec686bbcf0a6e7f0550b9c3f2ebf7863b336450ddff03fd6444c1013e36f470fd1ed28ff9318b4cba2c6fb00ff687c55f5522b979dcf0a62530c05aaea88c99540bf39ea32e50df4d5ab7f7b49572cf36ce1cc995b8f38147f007a972db8ea9540190d3e3145b4fc6172a3455e9e1df9559d9c88b084f20daf5f3f05c98cd484bac3f2b1b29321638b99d404165015c9f8d904a2fb1c54042dd1bd0c2bf9fa1497da400ad4d659adacb5036ffe7c88f81eea0de80a1ad776dbacab8c8d66769d6cb0d6a65f96ff5239e6f3cb9ee"
```

```python colab={"base_uri": "https://localhost:8080/"} id="KsI5k6kY1Atz" executionInfo={"status": "ok", "timestamp": 1634802719014, "user_tz": -330, "elapsed": 99789, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7584519c-0088-4217-a17b-13e34254cadd"
!unzip user.csv.zip
!unzip review.csv.zip
!unzip business.csv.zip
```

```python id="SafQwxCUwK3A" executionInfo={"status": "ok", "timestamp": 1634804183900, "user_tz": -330, "elapsed": 861, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import pandas as pd
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timedelta
import ast
import gc
```

```python id="E-_4IVM6wK0_" executionInfo={"status": "ok", "timestamp": 1634804184708, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# unix datetime
base = pd.Timestamp("1970-01-01")
CHUNK_SIZE = 1000000
REVIEW_DROP = 0
RESTAURANTS_PATH = 'yelp_business.csv'
REVIEWS_PATH = 'yelp_review.csv'
USERS_PATH = 'yelp_user.csv'


# https://www.kaggle.com/zolboo/recommender-systems-knn-svd-nn-keras
# Function that extract keys from the nested dictionary
def extract_keys(attr, key):
    if attr == None:
        return "{}"
    if key in attr:
        return attr.pop(key)


# convert string to dictionary
def str_to_dict(attr):
    if attr != None:
        return ast.literal_eval(attr)
    else:
        return ast.literal_eval("{}")


def sub_timestamp(element):
    element = element[0]
    a, b = element.split('-')
    a = datetime.strptime(a, "%H:%M")
    b = datetime.strptime(b, "%H:%M")
    return timedelta.total_seconds(b - a)


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device


def df_to_tensor(df):
    device = get_device()
    return torch.from_numpy(df.values).long().to(device)

def df_to_tensor_cpu(df):
    return torch.from_numpy(df.values).long()

def process_data_chunk(reviews, users, restaurants):
    reviews = pd.merge(reviews, users, how='inner', on='user_id')
    reviews = reviews.drop(columns='user_id')
    reviews = pd.merge(reviews, restaurants, how='inner', on='business_id')
    reviews = reviews.drop(columns='business_id')
    print("REVIEWS.HEAD() -------------------------------------------------------------------")
    print(reviews.head())
    reviews = reviews.drop(columns=reviews.columns[0], axis=1)
    print("REVIEWS.DROP() -------------------------------------------------------------------")
    print(reviews.head())
    return df_to_tensor(reviews)
```

```python colab={"base_uri": "https://localhost:8080/"} id="4d3zke6k6alN" executionInfo={"status": "ok", "timestamp": 1634804456254, "user_tz": -330, "elapsed": 230727, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0c5b6087-c3da-4cc9-cf9f-cd82c9549f3d"
train_percent, val_percent, test_percent  = 0.6, 0.3, 0.1

print("Reading users")
users = pd.read_csv(USERS_PATH)
users = users[users['review_count'] > REVIEW_DROP]
users['user_id'] = users['user_id'].astype('category')
users['user_id_num'] = users['user_id'].cat.codes
users = users[['user_id', 'user_id_num', 'review_count']]
user_id_to_num = dict(zip(users['user_id'], users['user_id_num']))

print("Reading businesses")
restaurants = pd.read_csv(RESTAURANTS_PATH)
restaurants['business_id'] = restaurants['business_id'].astype('category')
restaurants['business_id_num'] = restaurants['business_id'].cat.codes
restaurants = restaurants[['business_id', 'business_id_num']]
rest_id_to_num = dict(zip(restaurants['business_id'], restaurants['business_id_num']))

print("Reading reviews")
reviews = pd.read_csv(REVIEWS_PATH)

reviews = pd.merge(reviews, users, how='inner', on='user_id')
reviews = reviews.drop(columns='user_id')
reviews = pd.merge(reviews, restaurants, how='inner', on='business_id')
reviews = reviews.drop(columns='business_id')
print("REVIEWS.HEAD() -------------------------------------------------------------------")
print(reviews.head())
reviews = reviews.drop(columns=reviews.columns[0], axis=1)
print("REVIEWS.DROP() -------------------------------------------------------------------")
print(reviews.head())

pickle.dump(user_id_to_num, open('user_id_to_num.pkl', 'wb'))
pickle.dump(rest_id_to_num, open('rest_id_to_num.pkl', 'wb'))

# np.save('data.npy', reviews.values)

training = reviews.sample(frac=train_percent)

left = reviews.drop(training.index)
validation = left.sample(frac=val_percent / (val_percent + test_percent))

test = left.drop(validation.index)

print("loaded")
```

```python colab={"base_uri": "https://localhost:8080/"} id="Gq9Sdga5Ca-o" executionInfo={"status": "ok", "timestamp": 1634806286218, "user_tz": -330, "elapsed": 5391, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f0da5893-7831-4e2c-a882-afd93bffdd41"
gc.collect()
```

```python id="zs1TrEXx7MmC"
train, val, test, user, rest = df_to_tensor_cpu(training), df_to_tensor_cpu(validation), df_to_tensor_cpu(test), user_id_to_num, rest_id_to_num
```

```python id="RKWahI4Y7N6L"
print("TRAIN ----------------------------------------------")
print(train.shape)
print("VAL ----------------------------------------------")
print(val.shape)
print("TEST ----------------------------------------------")
print(test.shape)
```

```python id="528oEpspwKyP"

```

<!-- #region id="UdWhMcdgwKwO" -->
## Model
<!-- #endregion -->

```python id="RYb_-bExwKuJ" executionInfo={"status": "ok", "timestamp": 1634801408439, "user_tz": -330, "elapsed": 27596, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.random import RandomState


class DRRAveStateRepresentation(nn.Module):
    def __init__(self, n_items=5, item_features=100, user_features=100):
        super(DRRAveStateRepresentation, self).__init__()
        self.n_items = n_items
        self.random_state = RandomState(1)
        self.item_features = item_features
        self.user_features = user_features

        self.attention_weights = nn.Parameter(torch.from_numpy(0.1 * self.random_state.rand(self.n_items)).float())

    def forward(self, user, items):
        '''
        DRR-AVE State Representation
        :param items: (torch tensor) shape = (n_items x item_features),
                Matrix of items in history buffer
        :param user: (torch tensor) shape = (1 x user_features),
                User embedding
        :return: output: (torch tensor) shape = (3 * item_features)
        '''
        right = items.t() @ self.attention_weights
        middle = user * right
        output = torch.cat((user, middle, right), 0).flatten()
        return output


class Actor(nn.Module):
    def __init__(self, in_features=100, out_features=18):
        super(Actor, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.linear1 = nn.Linear(self.in_features, self.in_features)
        self.linear2 = nn.Linear(self.in_features, self.in_features)
        self.linear3 = nn.Linear(self.in_features, self.out_features)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = F.tanh(self.linear3(output))
        return output


class Critic(nn.Module):
    def __init__(self, action_size=20, in_features=128, out_features=18):
        super(Critic, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.combo_features = in_features + action_size
        self.action_size = action_size

        self.linear1 = nn.Linear(self.in_features, self.in_features)
        self.linear2 = nn.Linear(self.combo_features, self.combo_features)
        self.linear3 = nn.Linear(self.combo_features, self.combo_features)
        self.output_layer = nn.Linear(self.combo_features, self.out_features)

    def forward(self, state, action):
        output = F.relu(self.linear1(state))
        output = torch.cat((action, output), dim=1)
        output = F.relu(self.linear2(output))
        output = F.relu(self.linear3(output))
        output = self.output_layer(output)
        return output


class PMF(nn.Module):
    def __init__(self, n_users, n_items, n_factors=20, is_sparse=False, no_cuda=None):
        super(PMF, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.no_cuda = no_cuda
        self.random_state = RandomState(1)

        self.user_embeddings = nn.Embedding(n_users, n_factors, sparse=is_sparse)
        self.user_embeddings.weight.data = torch.from_numpy(0.1 * self.random_state.rand(n_users, n_factors)).float()

        self.item_embeddings = nn.Embedding(n_items, n_factors, sparse=is_sparse)
        self.item_embeddings.weight.data = torch.from_numpy(0.1 * self.random_state.rand(n_items, n_factors)).float()

        self.ub = nn.Embedding(n_users, 1)
        self.ib = nn.Embedding(n_items, 1)
        self.ub.weight.data.uniform_(-.01, .01)
        self.ib.weight.data.uniform_(-.01, .01)

    def forward(self, users_index, items_index):
        user_h1 = self.user_embeddings(users_index)
        item_h1 = self.item_embeddings(items_index)
        R_h = (user_h1 * item_h1).sum(dim=1 if len(user_h1.shape) > 1 else 0) + self.ub(users_index).squeeze() + self.ib(items_index).squeeze()
        return R_h

    def __call__(self, *args):
        return self.forward(*args)

    def predict(self, users_index, items_index):
        preds = self.forward(users_index, items_index)
        return preds
```

<!-- #region id="hf1fNuJNwKr8" -->
## Evaluation
<!-- #endregion -->

```python id="lJVq2LgawSg_" executionInfo={"status": "ok", "timestamp": 1634801377341, "user_tz": -330, "elapsed": 594, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import numpy as np

def RMSE(preds, truth):
    return np.sqrt(np.mean(np.square(preds-truth)))
```

```python id="2sbXxJfBwSfW"
from __future__ import print_function
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data
import matplotlib.pyplot as plt
```

```python id="PfgcCA5qwSc-"

```

```python id="6aoXoCLHwSap"

```
