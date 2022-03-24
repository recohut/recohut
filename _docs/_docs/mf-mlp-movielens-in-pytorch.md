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

<!-- #region id="thC-jHYLJKkz" -->
# Training neural factorization model on movielens dataset
> Training MF, MF+bias, and MLP model on movielens-100k dataset in PyTorch

- toc: false
- badges: true
- comments: true
- categories: [Pytorch, Movie, MF, MLP, RecoChef]
- author: "<a href='https://github.com/yanneta/pytorch-tutorials'>Yannet</a>"
- image:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="U9XYsONJClRh" outputId="f32b3306-07df-47de-ae54-92e2c2c2333c"
!pip install -q git+https://github.com/sparsh-ai/recochef.git
```

```python id="2LS69WtgCuxJ"
import torch
import torch.nn.functional as F

from recochef.datasets.synthetic import Synthetic
from recochef.datasets.movielens import MovieLens
from recochef.preprocessing.split import chrono_split
from recochef.preprocessing.encode import label_encode as le
from recochef.models.factorization import MF, MF_bias
from recochef.models.dnn import CollabFNet
```

```python id="X7-2sy7dDJte"
# # generate synthetic implicit data
# synt = Synthetic()
# df = synt.implicit()

movielens = MovieLens()
df = movielens.load_interactions()

# changing rating colname to event following implicit naming conventions
df = df.rename(columns={'RATING': 'EVENT'})
```

```python colab={"base_uri": "https://localhost:8080/"} id="EGLNfBJBCw38" outputId="06429212-3b1c-4a95-df70-927b8e8a3e43"
# drop duplicates
df = df.drop_duplicates()

# chronological split
df_train, df_valid = chrono_split(df, ratio=0.8, min_rating=10)
print(f"Train set:\n\n{df_train}\n{'='*100}\n")
print(f"Validation set:\n\n{df_valid}\n{'='*100}\n")
```

```python colab={"base_uri": "https://localhost:8080/"} id="68zLUPlvC5LK" outputId="46c0f8b6-dd84-4c54-8d55-8eb61fb3fc47"
# label encoding
df_train, uid_maps = le(df_train, col='USERID')
df_train, iid_maps = le(df_train, col='ITEMID')
df_valid = le(df_valid, col='USERID', maps=uid_maps)
df_valid = le(df_valid, col='ITEMID', maps=iid_maps)

# # event implicit to rating conversion
# event_weights = {'click':1, 'add':2, 'purchase':4}
# event_maps = dict({'EVENT_TO_IDX':event_weights})
# df_train = le(df_train, col='EVENT', maps=event_maps)
# df_valid = le(df_valid, col='EVENT', maps=event_maps)

print(f"Processed Train set:\n\n{df_train}\n{'='*100}\n")
print(f"Processed Validation set:\n\n{df_valid}\n{'='*100}\n")
```

```python colab={"base_uri": "https://localhost:8080/"} id="VnhEaj5QC8j1" outputId="f15ef434-6b1d-4f51-f11c-4bfbe4af649b"
# get number of unique users and items
num_users = len(df_train.USERID.unique())
num_items = len(df_train.ITEMID.unique())

num_users_t = len(df_valid.USERID.unique())
num_items_t = len(df_valid.ITEMID.unique())

print(f"There are {num_users} users and {num_items} items in the train set.\n{'='*100}\n")
print(f"There are {num_users_t} users and {num_items_t} items in the validation set.\n{'='*100}\n")
```

```python id="xTiGbb5UCpwM"
# training and testing related helper functions
def train_epocs(model, epochs=10, lr=0.01, wd=0.0, unsqueeze=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model.train()
    for i in range(epochs):
        users = torch.LongTensor(df_train.USERID.values) # .cuda()
        items = torch.LongTensor(df_train.ITEMID.values) #.cuda()
        ratings = torch.FloatTensor(df_train.EVENT.values) #.cuda()
        if unsqueeze:
            ratings = ratings.unsqueeze(1)
        y_hat = model(users, items)
        loss = F.mse_loss(y_hat, ratings)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item()) 
    test_loss(model, unsqueeze)

def test_loss(model, unsqueeze=False):
    model.eval()
    users = torch.LongTensor(df_valid.USERID.values) #.cuda()
    items = torch.LongTensor(df_valid.ITEMID.values) #.cuda()
    ratings = torch.FloatTensor(df_valid.EVENT.values) #.cuda()
    if unsqueeze:
        ratings = ratings.unsqueeze(1)
    y_hat = model(users, items)
    loss = F.mse_loss(y_hat, ratings)
    print("test loss %.3f " % loss.item())
```

```python colab={"base_uri": "https://localhost:8080/"} id="LxhbI4ECC_Jb" outputId="fa326841-1e15-4900-c0e4-fc7790beb762"
# training MF model
model = MF(num_users, num_items, emb_size=100) # .cuda() if you have a GPU
print(f"Training MF model:\n")
train_epocs(model, epochs=10, lr=0.1)
print(f"\n{'='*100}\n")
```

```python colab={"base_uri": "https://localhost:8080/"} id="fnbkknGIDAs6" outputId="e8466582-7078-49ab-dda7-eeffaa65c8de"
# training MF with bias model
model = MF_bias(num_users, num_items, emb_size=100) #.cuda()
print(f"Training MF+bias model:\n")
train_epocs(model, epochs=10, lr=0.05, wd=1e-5)
print(f"\n{'='*100}\n")
```

```python colab={"base_uri": "https://localhost:8080/"} id="N9ltu-ISDCUY" outputId="06c01140-05a4-4b06-9819-546a7ecdba66"
# training MLP model
model = CollabFNet(num_users, num_items, emb_size=100) #.cuda()
print(f"Training MLP model:\n")
train_epocs(model, epochs=15, lr=0.05, wd=1e-6, unsqueeze=True)
print(f"\n{'='*100}\n")
```
