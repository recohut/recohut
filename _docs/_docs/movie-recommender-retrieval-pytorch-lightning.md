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

<!-- #region id="mesNcoILdln4" -->
# Movie Candidate Retrieval with PyTorch Lightning
> In this notebook, we will build a simple yet accurate model using movielens-25m dataset and pytorch lightning library. This will be a retrieval model where the objective is to maximize recall over precision.

- toc: true
- badges: true
- comments: true
- categories: [PyTorch, MovieLens, Retrieval]
- author: "<a href='https://towardsdatascience.com/deep-learning-based-recommender-systems-3d120201db7e'>James Loy</a>"
- image:
<!-- #endregion -->

<!-- #region id="c5dwnLIupPaB" -->
## Setup
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="2SbNhrnyfIcg" outputId="410e27d1-6b78-4de5-8f0c-1ea56d6ea2f9"
#hide-output
!pip install -q pytorch-lightning
```

```python id="Pc2bntgJezOR"
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

np.random.seed(123)
```

<!-- #region id="7jIQm_WLpR2P" -->
## Data Loading
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="F_ShWD1yfAOb" outputId="acb871c0-1157-439a-f3e1-377676ba1c32"
#hide-output
!wget https://files.grouplens.org/datasets/movielens/ml-25m.zip && unzip ml-25m.zip
```

```python id="InLSsx4hfeBm"
ratings = pd.read_csv('ml-25m/ratings.csv', infer_datetime_format=True)
```

<!-- #region id="i_27d9mBpUl-" -->
## Subset
<!-- #endregion -->

<!-- #region id="h-uvTYoAgJSW" -->
In order to keep memory usage manageable within Kaggle's kernel, we will only use data from 30% of the users in this dataset. Let's randomly select 30% of the users and only use data from the selected users.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="v5kxycTAgCUF" outputId="2f2f6e1b-950e-4f84-d017-b0c6e11b4c18"
rand_userIds = np.random.choice(ratings['userId'].unique(), 
                                size=int(len(ratings['userId'].unique())*0.3), 
                                replace=False)

ratings = ratings.loc[ratings['userId'].isin(rand_userIds)]

print('There are {} rows of data from {} users'.format(len(ratings), len(rand_userIds)))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="amSm_bU1gO2C" outputId="ddbee782-ced5-4fc4-c333-d1faa04e33c9"
ratings.sample(5)
```

<!-- #region id="w7OS6UqDpXdi" -->
## Train/Test Split
**Chronological Leave-One-Out Split**
<!-- #endregion -->

<!-- #region id="ZaqAMrH-gn_i" -->
Along with the rating, there is also a timestamp column that shows the date and time the review was submitted. Using the timestamp column, we will implement our train-test split strategy using the leave-one-out methodology. For each user, the most recent review is used as the test set (i.e. leave one out), while the rest will be used as training data .
<!-- #endregion -->

<!-- #region id="A4COa9yVguUO" -->
> Note: Doing a random split would not be fair, as we could potentially be using a user's recent reviews for training and earlier reviews for testing. This introduces data leakage with a look-ahead bias, and the performance of the trained model would not be generalizable to real-world performance.
<!-- #endregion -->

```python id="WtdtS0FMgTez"
ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'] \
                                .rank(method='first', ascending=False)

train_ratings = ratings[ratings['rank_latest'] != 1]
test_ratings = ratings[ratings['rank_latest'] == 1]

# drop columns that we no longer need
train_ratings = train_ratings[['userId', 'movieId', 'rating']]
test_ratings = test_ratings[['userId', 'movieId', 'rating']]
```

<!-- #region id="XFoYAbhqpnPD" -->
## Implicit Conversion
<!-- #endregion -->

<!-- #region id="HwYvXqJ6hz2u" -->
We will train a recommender system using implicit feedback. However, the MovieLens dataset that we're using is based on explicit feedback. To convert this dataset into an implicit feedback dataset, we'll simply binarize the ratings such that they are are '1' (i.e. positive class). The value of '1' represents that the user has interacted with the item.

> Note: Using implicit feedback reframes the problem that our recommender is trying to solve. Instead of trying to predict movie ratings (when using explicit feedback), we are trying to predict whether the user will interact (i.e. click/buy/watch) with each movie, with the aim of presenting to users the movies with the highest interaction likelihood.

> Tip: This setting is suitable at retrieval stage where the objective is to maximize recall by identifying items that user will at least interact with.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="sQYuW1Otg_Cg" outputId="5d785fa4-6263-4708-d218-4055a1c86dc8"
train_ratings.loc[:, 'rating'] = 1

train_ratings.sample(5)
```

<!-- #region id="uhwZiaBPpsQl" -->
## Negative Sampling
<!-- #endregion -->

<!-- #region id="ngZdyoMjizlw" -->
We do have a problem now though. After binarizing our dataset, we see that every sample in the dataset now belongs to the positive class. However we also require negative samples to train our models, to indicate movies that the user has not interacted with. We assume that such movies are those that the user are not interested in - even though this is a sweeping assumption that may not be true, it usually works out rather well in practice.

The code below generates 4 negative samples for each row of data. In other words, the ratio of negative to positive samples is 4:1. This ratio is chosen arbitrarily but I found that it works rather well (feel free to find the best ratio yourself!)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 66, "referenced_widgets": ["fa8a0718bba846af8863604811c539d5", "24f33c9a9ef54658bb5bd8b623b4cc37", "e21c4e39bec54e9b8b26c0c31bb8edc8", "82d38bf1a9c343f8a32ff44e4e7fcf8b", "e932cfbc755544feaf9cb3918c24b4f3", "8a1da7d330704c0ea800d4c0a30f56b9", "1607718f934446f796d454654b09a150", "662066d58d8446f09979c0de782bf6e8"]} id="4T0_UVhTizVn" outputId="8b92d6ca-e2cb-4cfc-a53a-b4f90a0d600d"
# Get a list of all movie IDs
all_movieIds = ratings['movieId'].unique()

# Placeholders that will hold the training data
users, items, labels = [], [], []

# This is the set of items that each user has interaction with
user_item_set = set(zip(train_ratings['userId'], train_ratings['movieId']))

# 4:1 ratio of negative to positive samples
num_negatives = 4

for (u, i) in tqdm(user_item_set):
    users.append(u)
    items.append(i)
    labels.append(1) # items that the user has interacted with are positive
    for _ in range(num_negatives):
        # randomly select an item
        negative_item = np.random.choice(all_movieIds) 
        # check that the user has not interacted with this item
        while (u, negative_item) in user_item_set:
            negative_item = np.random.choice(all_movieIds)
        users.append(u)
        items.append(negative_item)
        labels.append(0) # items not interacted with are negative
```

<!-- #region id="9brxnZqlpvXD" -->
## PyTorch Dataset
<!-- #endregion -->

<!-- #region id="Br2u5nn5jAy1" -->
Great! We now have the data in the format required by our model. Before we move on, let's define a PyTorch Dataset to facilitate training. The class below simply encapsulates the code we have written above into a PyTorch Dataset class.
<!-- #endregion -->

```python id="pCn0M346i6Z8"
class MovieLensTrainDataset(Dataset):
    """MovieLens PyTorch Dataset for Training
    
    Args:
        ratings (pd.DataFrame): Dataframe containing the movie ratings
        all_movieIds (list): List containing all movieIds
    
    """

    def __init__(self, ratings, all_movieIds):
        self.users, self.items, self.labels = self.get_dataset(ratings, all_movieIds)

    def __len__(self):
        return len(self.users)
  
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

    def get_dataset(self, ratings, all_movieIds):
        users, items, labels = [], [], []
        user_item_set = set(zip(ratings['userId'], ratings['movieId']))

        num_negatives = 4
        for u, i in user_item_set:
            users.append(u)
            items.append(i)
            labels.append(1)
            for _ in range(num_negatives):
                negative_item = np.random.choice(all_movieIds)
                while (u, negative_item) in user_item_set:
                    negative_item = np.random.choice(all_movieIds)
                users.append(u)
                items.append(negative_item)
                labels.append(0)

        return torch.tensor(users), torch.tensor(items), torch.tensor(labels)
```

<!-- #region id="SqMDcEYRjOZN" -->
## Model - Neural Collaborative Filtering (NCF)

While there are many deep learning based architecture for recommendation systems, I find that the framework proposed by He et al. is the most straightforward and it is simple enough to be implemented in a tutorial such as this.
<!-- #endregion -->

```python id="xwlBJpqljJvS"
class NCF(pl.LightningModule):
    """ Neural Collaborative Filtering (NCF)
    
        Args:
            num_users (int): Number of unique users
            num_items (int): Number of unique items
            ratings (pd.DataFrame): Dataframe containing the movie ratings for training
            all_movieIds (list): List containing all movieIds (train + test)
    """
    
    def __init__(self, num_users, num_items, ratings, all_movieIds):
        super().__init__()
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=8)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=8)
        self.fc1 = nn.Linear(in_features=16, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.output = nn.Linear(in_features=32, out_features=1)
        self.ratings = ratings
        self.all_movieIds = all_movieIds
        
    def forward(self, user_input, item_input):
        
        # Pass through embedding layers
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)

        # Concat the two embedding layers
        vector = torch.cat([user_embedded, item_embedded], dim=-1)

        # Pass through dense layer
        vector = nn.ReLU()(self.fc1(vector))
        vector = nn.ReLU()(self.fc2(vector))

        # Output layer
        pred = nn.Sigmoid()(self.output(vector))

        return pred
    
    def training_step(self, batch, batch_idx):
        user_input, item_input, labels = batch
        predicted_labels = self(user_input, item_input)
        loss = nn.BCELoss()(predicted_labels, labels.view(-1, 1).float())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return DataLoader(MovieLensTrainDataset(self.ratings, self.all_movieIds),
                          batch_size=512, num_workers=4)
```

<!-- #region id="I8wT1WK9jzeJ" -->
We instantiate the NCF model using the class that we have defined above.
<!-- #endregion -->

```python id="F3Bh9dorjww7"
num_users = ratings['userId'].max()+1
num_items = ratings['movieId'].max()+1

all_movieIds = ratings['movieId'].unique()

model = NCF(num_users, num_items, train_ratings, all_movieIds)
```

<!-- #region id="K4Mw8CdVp5lF" -->
## Model Training
<!-- #endregion -->

<!-- #region id="xnYHNWe3kRRD" -->
> Note: One advantage of PyTorch Lightning over vanilla PyTorch is that you don't need to write your own boiler plate training code. Notice how the Trainer class allows us to train our model with just a few lines of code.

Let's train our NCF model for 5 epochs using the GPU. 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 375, "referenced_widgets": ["68bcd7bfc32f4d9ebaba5c08437bca28", "2805d9154e374318a467cad92c87d889", "e68d0f42e5284ee5ab6fd8545b789097", "857ee1a3d8274f87a2681bbe41a61413", "b54f4d631bc643f9830098f40e3221fd", "934559e820bf4c7793372990727b52b2", "2e5512ac9bc54aa59955ed68bb0bdf4a", "3bfc383196bc4c929c1b6fa503f93cad"]} id="0JganCIMj2EW" outputId="6fa64b89-c835-4f39-d6ad-bac6f2369c64"
trainer = pl.Trainer(max_epochs=5, gpus=1, reload_dataloaders_every_epoch=True,
                     progress_bar_refresh_rate=50, logger=False, checkpoint_callback=False)

trainer.fit(model)
```

<!-- #region id="R6V1Tiw1kIxk" -->
> Note: We are using the argument reload_dataloaders_every_epoch=True. This creates a new randomly chosen set of negative samples for each epoch, which ensures that our model is not biased by the selection of negative samples.
<!-- #endregion -->

<!-- #region id="I3BXx1YzlAUq" -->
## Evaluating our Recommender System

Now that our model is trained, we are ready to evaluate it using the test data. In traditional Machine Learning projects, we evaluate our models using metrics such as Accuracy (for classification problems) and RMSE (for regression problems). However, such metrics are too simplistic for evaluating recommender systems.

The key here is that we don't need the user to interact on every single item in the list of recommendations. Instead, we just need the user to interact with at least one item on the list - as long as the user does that, the recommendations have worked.

To simulate this, let's run the following evaluation protocol to generate a list of 10 recommended items for each user.
- For each user, randomly select 99 items that the user has not interacted with
- Combine these 99 items with the test item (the actual item that the user interacted with). We now have 100 items.
- Run the model on these 100 items, and rank them according to their predicted probabilities
- Select the top 10 items from the list of 100 items. If the test item is present within the top 10 items, then we say that this is a hit.
- Repeat the process for all users. The Hit Ratio is then the average hits.
<!-- #endregion -->

<!-- #region id="B2PVVpUflN34" -->
> Note: This evaluation protocol is known as Hit Ratio @ 10, and it is commonly used to evaluate recommender systems.
<!-- #endregion -->

```python id="uSLTYZuhlNEV"
# User-item pairs for testing
test_user_item_set = set(zip(test_ratings['userId'], test_ratings['movieId']))

# Dict of all items that are interacted with by each user
user_interacted_items = ratings.groupby('userId')['movieId'].apply(list).to_dict()

hits = []
for (u,i) in tqdm(test_user_item_set):
    interacted_items = user_interacted_items[u]
    not_interacted_items = set(all_movieIds) - set(interacted_items)
    selected_not_interacted = list(np.random.choice(list(not_interacted_items), 99))
    test_items = selected_not_interacted + [i]
    
    predicted_labels = np.squeeze(model(torch.tensor([u]*100), 
                                        torch.tensor(test_items)).detach().numpy())
    
    top10_items = [test_items[i] for i in np.argsort(predicted_labels)[::-1][0:10].tolist()]
    
    if i in top10_items:
        hits.append(1)
    else:
        hits.append(0)
        
print("The Hit Ratio @ 10 is {:.2f}".format(np.average(hits)))
```

<!-- #region id="s1XtzBFsllfN" -->
We got a pretty good Hit Ratio @ 10 score! To put this into context, what this means is that 86% of the users were recommended the actual item (among a list of 10 items) that they eventually interacted with. Not bad!
<!-- #endregion -->
