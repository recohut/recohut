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

<!-- #region id="iVkUysixCyk6" -->
# Neural Collaborative Filtering Recommenders
<!-- #endregion -->

```python id="-xzeGn4mvtvd"
!pip install -q pytorch-lightning
```

```python id="xNgD2QXOkq4g"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset
from tqdm.notebook import tqdm
import pytorch_lightning as pl

np.random.seed(123)
```

<!-- #region id="8fqqbAgmrxoT" -->
## NCF with PyTorch Lightning on ML-25m

In this section, we will build a simple yet accurate model using movielens-25m dataset and pytorch lightning library. This will be a retrieval model where the objective is to maximize recall over precision.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="MrlkpJITv8Rw" outputId="4fd97487-1f04-4563-8adf-5aa306c13d2b"
!wget -q --show-progress https://files.grouplens.org/datasets/movielens/ml-25m.zip
!unzip ml-25m.zip
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="5-juuyKOwCmL" outputId="93ff3c63-a959-4e2d-fdff-c5e72a475160"
ratings = pd.read_csv('ml-25m/ratings.csv', infer_datetime_format=True)
ratings.head()
```

<!-- #region id="A2bKYi9WwGyP" -->
### Subset

In order to keep memory usage manageable, we will only use data from 20% of the users in this dataset. Let's randomly select 30% of the users and only use data from the selected users.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="HsYAMEXqwZoH" outputId="434b4209-8c53-4474-edd8-6c5d109c3184"
rand_userIds = np.random.choice(ratings['userId'].unique(), 
                                size=int(len(ratings['userId'].unique())*0.2), 
                                replace=False)

ratings = ratings.loc[ratings['userId'].isin(rand_userIds)]

print('There are {} rows of data from {} users'.format(len(ratings), len(rand_userIds)))
```

<!-- #region id="w7OS6UqDpXdi" -->
### Train/Test Split
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
### Implicit Conversion
<!-- #endregion -->

<!-- #region id="HwYvXqJ6hz2u" -->
We will train a recommender system using implicit feedback. However, the MovieLens dataset that we're using is based on explicit feedback. To convert this dataset into an implicit feedback dataset, we'll simply binarize the ratings such that they are are '1' (i.e. positive class). The value of '1' represents that the user has interacted with the item.

> Note: Using implicit feedback reframes the problem that our recommender is trying to solve. Instead of trying to predict movie ratings (when using explicit feedback), we are trying to predict whether the user will interact (i.e. click/buy/watch) with each movie, with the aim of presenting to users the movies with the highest interaction likelihood.

> Tip: This setting is suitable at retrieval stage where the objective is to maximize recall by identifying items that user will at least interact with.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="sQYuW1Otg_Cg" outputId="096aec13-d21e-4690-e88d-258fe9a681a0"
train_ratings.loc[:, 'rating'] = 1

train_ratings.sample(5)
```

<!-- #region id="uhwZiaBPpsQl" -->
### Negative Sampling
<!-- #endregion -->

<!-- #region id="ngZdyoMjizlw" -->
We do have a problem now though. After binarizing our dataset, we see that every sample in the dataset now belongs to the positive class. However we also require negative samples to train our models, to indicate movies that the user has not interacted with. We assume that such movies are those that the user are not interested in - even though this is a sweeping assumption that may not be true, it usually works out rather well in practice.

The code below generates 4 negative samples for each row of data. In other words, the ratio of negative to positive samples is 4:1. This ratio is chosen arbitrarily but I found that it works rather well (feel free to find the best ratio yourself!)
<!-- #endregion -->

```python id="4T0_UVhTizVn"
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
### PyTorch Dataset
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
### Model

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
                          batch_size=512, num_workers=2)
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
### Model Training
<!-- #endregion -->

<!-- #region id="xnYHNWe3kRRD" -->
> Note: One advantage of PyTorch Lightning over vanilla PyTorch is that you don't need to write your own boiler plate training code. Notice how the Trainer class allows us to train our model with just a few lines of code.

Let's train our NCF model for 5 epochs using the GPU. 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 375, "referenced_widgets": ["68bcd7bfc32f4d9ebaba5c08437bca28"]} id="0JganCIMj2EW" outputId="6fa64b89-c835-4f39-d6ad-bac6f2369c64"
trainer = pl.Trainer(max_epochs=5, gpus=1, reload_dataloaders_every_epoch=True,
                     progress_bar_refresh_rate=50, logger=False, checkpoint_callback=False)

trainer.fit(model)
```

<!-- #region id="R6V1Tiw1kIxk" -->
> Note: We are using the argument reload_dataloaders_every_epoch=True. This creates a new randomly chosen set of negative samples for each epoch, which ensures that our model is not biased by the selection of negative samples.
<!-- #endregion -->

<!-- #region id="I3BXx1YzlAUq" -->
### Evaluating our Recommender System

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

<!-- #region id="_xofdqRI29zl" -->
## NMF with PyTorch on ML-1m
<!-- #endregion -->

```python id="3wo7aehx3AyG"
import os
import time
import random
import argparse
import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
```

```python colab={"base_uri": "https://localhost:8080/"} id="y1iNipgl3JhO" outputId="363cf5f9-b062-4930-b122-d7573d824ab0"
DATA_URL = "https://raw.githubusercontent.com/sparsh-ai/rec-data-public/master/ml-1m-dat/ratings.dat"
MAIN_PATH = '/content/'
DATA_PATH = MAIN_PATH + 'ratings.dat'
MODEL_PATH = MAIN_PATH + 'models/'
MODEL = 'ml-1m_Neu_MF'

!wget -q --show-progress https://raw.githubusercontent.com/sparsh-ai/rec-data-public/master/ml-1m-dat/ratings.dat
```

```python id="EXFnsMFy3YTE"
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
```

<!-- #region id="KvTX81Z23bFs" -->
### Dataset
<!-- #endregion -->

```python id="NN5GjJCf3rI8"
class Rating_Datset(torch.utils.data.Dataset):
	def __init__(self, user_list, item_list, rating_list):
		super(Rating_Datset, self).__init__()
		self.user_list = user_list
		self.item_list = item_list
		self.rating_list = rating_list

	def __len__(self):
		return len(self.user_list)

	def __getitem__(self, idx):
		user = self.user_list[idx]
		item = self.item_list[idx]
		rating = self.rating_list[idx]
		
		return (
			torch.tensor(user, dtype=torch.long),
			torch.tensor(item, dtype=torch.long),
			torch.tensor(rating, dtype=torch.float)
			)
```

<!-- #region id="d4xgxyBsfoJM" -->
- *_reindex*: process dataset to reindex userID and itemID, also set rating as binary feedback
- *_leave_one_out*: leave-one-out evaluation protocol in paper https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf
- *negative_sampling*: randomly selects n negative examples for each positive one
<!-- #endregion -->

```python id="HggfgX_8Oqmq"
class NCF_Data(object):
	"""
	Construct Dataset for NCF
	"""
	def __init__(self, args, ratings):
		self.ratings = ratings
		self.num_ng = args.num_ng
		self.num_ng_test = args.num_ng_test
		self.batch_size = args.batch_size

		self.preprocess_ratings = self._reindex(self.ratings)

		self.user_pool = set(self.ratings['user_id'].unique())
		self.item_pool = set(self.ratings['item_id'].unique())

		self.train_ratings, self.test_ratings = self._leave_one_out(self.preprocess_ratings)
		self.negatives = self._negative_sampling(self.preprocess_ratings)
		random.seed(args.seed)
	
	def _reindex(self, ratings):
		"""
		Process dataset to reindex userID and itemID, also set rating as binary feedback
		"""
		user_list = list(ratings['user_id'].drop_duplicates())
		user2id = {w: i for i, w in enumerate(user_list)}

		item_list = list(ratings['item_id'].drop_duplicates())
		item2id = {w: i for i, w in enumerate(item_list)}

		ratings['user_id'] = ratings['user_id'].apply(lambda x: user2id[x])
		ratings['item_id'] = ratings['item_id'].apply(lambda x: item2id[x])
		ratings['rating'] = ratings['rating'].apply(lambda x: float(x > 0))
		return ratings

	def _leave_one_out(self, ratings):
		"""
		leave-one-out evaluation protocol in paper https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf
		"""
		ratings['rank_latest'] = ratings.groupby(['user_id'])['timestamp'].rank(method='first', ascending=False)
		test = ratings.loc[ratings['rank_latest'] == 1]
		train = ratings.loc[ratings['rank_latest'] > 1]
		assert train['user_id'].nunique()==test['user_id'].nunique(), 'Not Match Train User with Test User'
		return train[['user_id', 'item_id', 'rating']], test[['user_id', 'item_id', 'rating']]

	def _negative_sampling(self, ratings):
		interact_status = (
			ratings.groupby('user_id')['item_id']
			.apply(set)
			.reset_index()
			.rename(columns={'item_id': 'interacted_items'}))
		interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
		interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, self.num_ng_test))
		return interact_status[['user_id', 'negative_items', 'negative_samples']]

	def get_train_instance(self):
		users, items, ratings = [], [], []
		train_ratings = pd.merge(self.train_ratings, self.negatives[['user_id', 'negative_items']], on='user_id')
		train_ratings['negatives'] = train_ratings['negative_items'].apply(lambda x: random.sample(x, self.num_ng))
		for row in train_ratings.itertuples():
			users.append(int(row.user_id))
			items.append(int(row.item_id))
			ratings.append(float(row.rating))
			for i in range(self.num_ng):
				users.append(int(row.user_id))
				items.append(int(row.negatives[i]))
				ratings.append(float(0))  # negative samples get 0 rating
		dataset = Rating_Datset(
			user_list=users,
			item_list=items,
			rating_list=ratings)
		return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)

	def get_test_instance(self):
		users, items, ratings = [], [], []
		test_ratings = pd.merge(self.test_ratings, self.negatives[['user_id', 'negative_samples']], on='user_id')
		for row in test_ratings.itertuples():
			users.append(int(row.user_id))
			items.append(int(row.item_id))
			ratings.append(float(row.rating))
			for i in getattr(row, 'negative_samples'):
				users.append(int(row.user_id))
				items.append(int(i))
				ratings.append(float(0))
		dataset = Rating_Datset(
			user_list=users,
			item_list=items,
			rating_list=ratings)
		return torch.utils.data.DataLoader(dataset, batch_size=self.num_ng_test+1, shuffle=False, num_workers=2)
```

<!-- #region id="hfXyLsgVOsBs" -->
### Metrics
Using Hit Rate and NDCG as our evaluation metrics
<!-- #endregion -->

```python id="KM4B7r12OvnS"
def hit(ng_item, pred_items):
	if ng_item in pred_items:
		return 1
	return 0


def ndcg(ng_item, pred_items):
	if ng_item in pred_items:
		index = pred_items.index(ng_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def metrics(model, test_loader, top_k, device):
	HR, NDCG = [], []

	for user, item, label in test_loader:
		user = user.to(device)
		item = item.to(device)

		predictions = model(user, item)
		_, indices = torch.topk(predictions, top_k)
		recommends = torch.take(
				item, indices).cpu().numpy().tolist()

		ng_item = item[0].item() # leave one-out evaluation has only one item per user
		HR.append(hit(ng_item, recommends))
		NDCG.append(ndcg(ng_item, recommends))

	return np.mean(HR), np.mean(NDCG)
```

<!-- #region id="HWyCD7pLOxjq" -->
### Models
- Generalized Matrix Factorization
- Multi Layer Perceptron
- Neural Matrix Factorization
<!-- #endregion -->

```python id="aTQaitu7d1R3"
class Generalized_Matrix_Factorization(nn.Module):
    def __init__(self, args, num_users, num_items):
        super(Generalized_Matrix_Factorization, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.factor_num = args.factor_num

        self.embedding_user = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.factor_num)
        self.embedding_item = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.factor_num)

        self.affine_output = nn.Linear(in_features=self.factor_num, out_features=1)
        self.logistic = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        element_product = torch.mul(user_embedding, item_embedding)
        logits = self.affine_output(element_product)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass
```

```python id="7kSFzPlNd50f"
class Multi_Layer_Perceptron(nn.Module):
    def __init__(self, args, num_users, num_items):
        super(Multi_Layer_Perceptron, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.factor_num = args.factor_num
        self.layers = args.layers

        self.embedding_user = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.factor_num)
        self.embedding_item = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.factor_num)

        self.fc_layers = nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            self.fc_layers.append(nn.Linear(in_size, out_size))

        self.affine_output = nn.Linear(in_features=self.layers[-1], out_features=1)
        self.logistic = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        vector = torch.cat([user_embedding, item_embedding], dim=-1)  # the concat latent vector
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = nn.ReLU()(vector)
            # vector = nn.BatchNorm1d()(vector)
            # vector = nn.Dropout(p=0.5)(vector)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass
```

```python id="7DQpVuaV9cF0"
class NeuMF(nn.Module):
    def __init__(self, args, num_users, num_items):
        super(NeuMF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.factor_num_mf = args.factor_num
        self.factor_num_mlp =  int(args.layers[0]/2)
        self.layers = args.layers
        self.dropout = args.dropout

        self.embedding_user_mlp = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.factor_num_mlp)
        self.embedding_item_mlp = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.factor_num_mlp)

        self.embedding_user_mf = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.factor_num_mf)
        self.embedding_item_mf = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.factor_num_mf)

        self.fc_layers = nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(args.layers[:-1], args.layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
            self.fc_layers.append(nn.ReLU())

        self.affine_output = nn.Linear(in_features=args.layers[-1] + self.factor_num_mf, out_features=1)
        self.logistic = nn.Sigmoid()
        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.embedding_user_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_item_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_user_mf.weight, std=0.01)
        nn.init.normal_(self.embedding_item_mf.weight, std=0.01)
        
        for m in self.fc_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                
        nn.init.xavier_uniform_(self.affine_output.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, user_indices, item_indices):
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)

        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
        mf_vector =torch.mul(user_embedding_mf, item_embedding_mf)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating.squeeze()
```

<!-- #region id="tpBX6rqNfSc9" -->
### Setting Arguments

Here is the brief description of important ones:
- Learning rate is 0.001
- Dropout rate is 0.2
- Running for 10 epochs
- HitRate@10 and NDCG@10
- 4 negative samples for each positive one
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Bc5Vg1Ik_gnF" outputId="072e970d-c6d2-413c-d6f4-2f25e13ee4bf"
parser = argparse.ArgumentParser()
parser.add_argument("--seed", 
	type=int, 
	default=42, 
	help="Seed")
parser.add_argument("--lr", 
	type=float, 
	default=0.001, 
	help="learning rate")
parser.add_argument("--dropout", 
	type=float,
	default=0.2,  
	help="dropout rate")
parser.add_argument("--batch_size", 
	type=int, 
	default=256, 
	help="batch size for training")
parser.add_argument("--epochs", 
	type=int,
	default=10,  
	help="training epoches")
parser.add_argument("--top_k", 
	type=int, 
	default=10, 
	help="compute metrics@top_k")
parser.add_argument("--factor_num", 
	type=int,
	default=32, 
	help="predictive factors numbers in the model")
parser.add_argument("--layers",
    nargs='+', 
    default=[64,32,16,8],
    help="MLP layers. Note that the first layer is the concatenation of user \
    and item embeddings. So layers[0]/2 is the embedding size.")
parser.add_argument("--num_ng", 
	type=int,
	default=4, 
	help="Number of negative samples for training set")
parser.add_argument("--num_ng_test", 
	type=int,
	default=100, 
	help="Number of negative samples for test set")
parser.add_argument("--out", 
	default=True,
	help="save model or not")
```

<!-- #region id="RnaRWy2gg_Nw" -->
### Training
<!-- #endregion -->

```python colab={"background_save": true, "base_uri": "https://localhost:8080/"} id="VyWquJG893CV" outputId="61938a61-f7f5-4885-85d1-1e3e2d2a06f6"
# set device and parameters
args = parser.parse_args(args={})
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()

# seed for Reproducibility
seed_everything(args.seed)

# load data
ml_1m = pd.read_csv(
	DATA_PATH, 
	sep="::", 
	names = ['user_id', 'item_id', 'rating', 'timestamp'], 
	engine='python')

# set the num_users, items
num_users = ml_1m['user_id'].nunique()+1
num_items = ml_1m['item_id'].nunique()+1

# construct the train and test datasets
data = NCF_Data(args, ml_1m)
train_loader = data.get_train_instance()
test_loader = data.get_test_instance()

# set model and loss, optimizer
model = NeuMF(args, num_users, num_items)
model = model.to(device)
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# train, evaluation
best_hr = 0
for epoch in range(1, args.epochs+1):
	model.train() # Enable dropout (if have).
	start_time = time.time()

	for user, item, label in train_loader:
		user = user.to(device)
		item = item.to(device)
		label = label.to(device)

		optimizer.zero_grad()
		prediction = model(user, item)
		loss = loss_function(prediction, label)
		loss.backward()
		optimizer.step()
		writer.add_scalar('loss/Train_loss', loss.item(), epoch)

	model.eval()
	HR, NDCG = metrics(model, test_loader, args.top_k, device)
	writer.add_scalar('Perfomance/HR@10', HR, epoch)
	writer.add_scalar('Perfomance/NDCG@10', NDCG, epoch)

	elapsed_time = time.time() - start_time
	print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
			time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
	print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

	if HR > best_hr:
		best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
		if args.out:
			if not os.path.exists(MODEL_PATH):
				os.mkdir(MODEL_PATH)
			torch.save(model, 
				'{}{}.pth'.format(MODEL_PATH, MODEL))

writer.close()
```

```python colab={"background_save": true, "base_uri": "https://localhost:8080/"} id="fkiRJWeD_trR" outputId="d3efcab5-fa0b-4938-d5ff-7967f38dab4d"
print("Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(
									best_epoch, best_hr, best_ndcg))
```

<!-- #region id="WOTaSMGnPoAG" -->
## MF with PyTorch on ML-100k

Training Pytorch MLP model on movielens-100k dataset and visualizing factors by decomposing using PCA
<!-- #endregion -->

```python id="I2f_R0Yo6BUp"
!pip install -U -q git+https://github.com/sparsh-ai/recochef.git
```

```python id="KXT07lHDBzAQ"
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F 
from torch.optim.lr_scheduler import _LRScheduler

from recochef.datasets.movielens import MovieLens
from recochef.preprocessing.encode import label_encode
from recochef.utils.iterators import batch_generator
from recochef.models.embedding import EmbeddingNet

import math
import copy
import pickle
import numpy as np
import pandas as pd
from textwrap import wrap
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
plt.style.use('ggplot')
```

<!-- #region id="NINhOhYAxt5n" -->
### Data loading and preprocessing
<!-- #endregion -->

```python id="3Z4R3bXNjaNP"
data = MovieLens()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="A2Xgw-sXk7Ac" outputId="399404ea-51bd-476f-f74e-0dc4807ebaa9"
ratings_df = data.load_interactions()
ratings_df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 343} id="wIUc-Ba_6xBK" outputId="3f1382b8-d132-48b8-f85f-fae21140922d"
movies_df = data.load_items()
movies_df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="X-IcaBzgmOrN" outputId="7a6afd50-b93a-498b-f9ea-96d7db363795"
ratings_df, umap = label_encode(ratings_df, 'USERID')
ratings_df, imap = label_encode(ratings_df, 'ITEMID')
ratings_df.head()
```

```python id="36dsiSqWwNRz"
X = ratings_df[['USERID','ITEMID']]
y = ratings_df[['RATING']]
```

```python colab={"base_uri": "https://localhost:8080/"} id="Mp-OqA2jyNAS" outputId="969eca89-0f0e-464d-d98e-70aeb215b99b"
for _x_batch, _y_batch in batch_generator(X, y, bs=4):
    print(_x_batch)
    print(_y_batch)
    break
```

```python colab={"base_uri": "https://localhost:8080/"} id="oQlnTmST0cx6" outputId="96db7cc7-9614-4215-f0e8-be69eb17e4e0"
_x_batch[:, 1]
```

<!-- #region id="mVavggU2_WUY" -->
### Embedding Net
<!-- #endregion -->

<!-- #region id="D39WDxT5_f3l" -->
The PyTorch is a framework that allows to build various computational graphs (not only neural networks) and run them on GPU. The conception of tensors, neural networks, and computational graphs is outside the scope of this article but briefly speaking, one could treat the library as a set of tools to create highly computationally efficient and flexible machine learning models. In our case, we want to create a neural network that could help us to infer the similarities between users and predict their ratings based on available data.
<!-- #endregion -->

<!-- #region id="9Gve8w7f_l8i" -->
The picture above schematically shows the model we're going to build. At the very beginning, we put our embeddings matrices, or look-ups, which convert integer IDs into arrays of floating-point numbers. Next, we put a bunch of fully-connected layers with dropouts. Finally, we need to return a list of predicted ratings. For this purpose, we use a layer with sigmoid activation function and rescale it to the original range of values (in case of MovieLens dataset, it is usually from 1 to 5).
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="9zDzhH2c0Cv-" outputId="134739d6-e9ff-4cd7-a0b5-0aac999500fa"
netx = EmbeddingNet(
    n_users=50, n_items=20, 
    n_factors=10, hidden=[500], 
    embedding_dropout=0.05, dropouts=[0.5])
netx
```

<!-- #region id="o4vQFZ6iwiyM" -->
### Cyclical Learning Rate (CLR)
<!-- #endregion -->

<!-- #region id="6RIaav5rwk66" -->
One of the `fastai` library features is the cyclical learning rate scheduler. We can implement something similar inheriting the `_LRScheduler` class from the `torch` library. Following the [original paper's](https://arxiv.org/abs/1506.01186) pseudocode, this [CLR Keras callback implementation](https://github.com/bckenstler/CLR), and making a couple of adjustments to support [cosine annealing](https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.CosineAnnealingLR) with restarts, let's create our own CLR scheduler.

The implementation of this idea is quite simple. The [base PyTorch scheduler class](https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html) has the `get_lr()` method that is invoked each time when we call the `step()` method. The method should return a list of learning rates depending on the current training epoch. In our case, we have the same learning rate for all of the layers, and therefore, we return a list with a single value. 

The next cell defines a `CyclicLR` class that expectes a single callback function. This function should accept the current training epoch and the base value of learning rate, and return a new learning rate value.
<!-- #endregion -->

```python id="eYQh4ZCmmgW9"
class CyclicLR(_LRScheduler):
    
    def __init__(self, optimizer, schedule, last_epoch=-1):
        assert callable(schedule)
        self.schedule = schedule
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.schedule(self.last_epoch, lr) for lr in self.base_lrs]
```

<!-- #region id="1bpK5hOvw7Hg" -->
Our scheduler is very similar to [LambdaLR](https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.LambdaLR) one but expects a bit different callback signature. 

So now we only need to define appropriate scheduling functions. We're createing a couple of functions that accept scheduling parameters and return a _new function_ with the appropriate signature:
<!-- #endregion -->

```python id="I6st2zPctj1T"
def triangular(step_size, max_lr, method='triangular', gamma=0.99):
    
    def scheduler(epoch, base_lr):
        period = 2 * step_size
        cycle = math.floor(1 + epoch/period)
        x = abs(epoch/step_size - 2*cycle + 1)
        delta = (max_lr - base_lr)*max(0, (1 - x))

        if method == 'triangular':
            pass  # we've already done
        elif method == 'triangular2':
            delta /= float(2 ** (cycle - 1))
        elif method == 'exp_range':
            delta *= (gamma**epoch)
        else:
            raise ValueError('unexpected method: %s' % method)
            
        return base_lr + delta
        
    return scheduler
```

```python id="k-CjYin0toWa"
def cosine(t_max, eta_min=0):
    
    def scheduler(epoch, base_lr):
        t = epoch % t_max
        return eta_min + (base_lr - eta_min)*(1 + math.cos(math.pi*t/t_max))/2
    
    return scheduler
```

<!-- #region id="oB_zTLW-wwdM" -->
To understand how the created functions work, and to check the correctness of our implementation, let's create a couple of plots visualizing learning rates changes depending on the number of epoch:
<!-- #endregion -->

```python id="Dl-TWx4OwwdN"
def plot_lr(schedule):
    ts = list(range(1000))
    y = [schedule(t, 0.001) for t in ts]
    plt.plot(ts, y)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 265} id="wCfhKAoMwwdN" outputId="52bd67ed-e05a-4a07-9747-c8d68b1eae5a"
plot_lr(triangular(250, 0.005))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 265} id="UPXizw35wwdO" outputId="674c0c79-c01c-41d7-a10d-466fb3dbefd1"
plot_lr(triangular(250, 0.005, 'triangular2'))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 265} id="2MMOua0WwwdO" outputId="5b3a0bc1-b1f8-4157-b292-4f81795f459c"
plot_lr(triangular(250, 0.005, 'exp_range', gamma=0.999))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 265} id="gdwJu6i3wwdP" outputId="3501bcf8-9895-44e1-e6bd-c6b2051e69f7"
plot_lr(cosine(t_max=500, eta_min=0.0005))
```

<!-- #region id="agK_98cWwwdQ" -->
### Training Loop

Now we're ready to start the training process. First of all, let's split the original dataset using [train_test_split](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) function from the `scikit-learn` library. (Though you can use anything else instead, like, [get_cv_idxs](https://github.com/fastai/fastai/blob/921777feb46f215ed2b5f5dcfcf3e6edd299ea92/fastai/dataset.py#L6-L22) from `fastai`).
<!-- #endregion -->

```python id="216J9e39wwdR"
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
datasets = {'train': (X_train, y_train), 'val': (X_valid, y_valid)}
dataset_sizes = {'train': len(X_train), 'val': len(X_valid)}
```

```python colab={"base_uri": "https://localhost:8080/"} id="fFgH5tuLwwdR" outputId="4e64cc2c-8377-46a1-d220-96fbee9a59bb"
minmax = ratings_df.RATING.astype(float).min(), ratings_df.RATING.astype(float).max()
minmax
```

```python colab={"base_uri": "https://localhost:8080/"} id="WlAmBo9Jys0m" outputId="59234538-cd74-426f-8543-72cac11d7ee0"
n_users = ratings_df.USERID.nunique()
n_movies = ratings_df.ITEMID.nunique()
n_users, n_movies
```

```python id="KHeaux79wwdS"
net = EmbeddingNet(
    n_users=n_users, n_items=n_movies, 
    n_factors=150, hidden=[500, 500, 500], 
    embedding_dropout=0.05, dropouts=[0.5, 0.5, 0.25])
```

<!-- #region id="onmMNylKwwdS" -->
The next cell is preparing and running the training loop with cyclical learning rate, validation and early stopping. We use `Adam` optimizer with cosine-annealing learnign rate. The rate is decreased on each batch during `2` epochs, and then is reset to the original value.

Note that our loop has two phases. One of them is called `train`. During this phase, we update our network's weights and change the learning rate. The another one is called `val` and is used to check the model's performence. When the loss value decreases, we save model parameters to restore them later. If there is no improvements after `10` sequential training epochs, we exit from the loop.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="4mRf0N9kwwdT" outputId="50f21317-af76-4dd8-a8f0-5ab7b61cc726"
lr = 1e-3
wd = 1e-5
bs = 50
n_epochs = 100
patience = 10
no_improvements = 0
best_loss = np.inf
best_weights = None
history = []
lr_history = []

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net.to(device)
criterion = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
iterations_per_epoch = int(math.ceil(dataset_sizes['train'] // bs))
scheduler = CyclicLR(optimizer, cosine(t_max=iterations_per_epoch * 2, eta_min=lr/10))

for epoch in range(n_epochs):
    stats = {'epoch': epoch + 1, 'total': n_epochs}
    
    for phase in ('train', 'val'):
        training = phase == 'train'
        running_loss = 0.0
        n_batches = 0
        
        for batch in batch_generator(*datasets[phase], shuffle=training, bs=bs):
            x_batch, y_batch = [b.to(device) for b in batch]
            optimizer.zero_grad()
        
            # compute gradients only during 'train' phase
            with torch.set_grad_enabled(training):
                outputs = net(x_batch[:, 0], x_batch[:, 1], minmax)
                loss = criterion(outputs, y_batch)
                
                # don't update weights and rates when in 'val' phase
                if training:
                    scheduler.step()
                    loss.backward()
                    optimizer.step()
                    lr_history.extend(scheduler.get_lr())
                    
            running_loss += loss.item()
            
        epoch_loss = running_loss / dataset_sizes[phase]
        stats[phase] = epoch_loss
        
        # early stopping: save weights of the best model so far
        if phase == 'val':
            if epoch_loss < best_loss:
                print('loss improvement on epoch: %d' % (epoch + 1))
                best_loss = epoch_loss
                best_weights = copy.deepcopy(net.state_dict())
                no_improvements = 0
            else:
                no_improvements += 1
                
    history.append(stats)
    print('[{epoch:03d}/{total:03d}] train: {train:.4f} - val: {val:.4f}'.format(**stats))
    if no_improvements >= patience:
        print('early stopping after epoch {epoch:03d}'.format(**stats))
        break
```

<!-- #region id="EGrWJQGnwwdT" -->
### Metrics

To visualize the training process and to check the correctness of the learning rate scheduling, let's create a couple of plots using collected stats:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 282} id="y3vAtcy9wwdU" outputId="612b10fe-440e-4422-90ca-cef19cc5ce36"
ax = pd.DataFrame(history).drop(columns='total').plot(x='epoch')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 265} id="8SNXU2CKwwdU" outputId="c4ba4191-0a03-4cff-8a2c-afc478a1658e"
_ = plt.plot(lr_history[:2*iterations_per_epoch])
```

<!-- #region id="Yy8lkzpGwwdV" -->
As expected, the learning rate is updated in accordance with cosine annealing schedule.
<!-- #endregion -->

<!-- #region id="t_4cZUXgwwdV" -->
The training process was terminated after _16 epochs_. Now we're going to restore the best weights saved during training, and apply the model to the validation subset of the data to see the final model's performance:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="b9WGkwZ7wwdV" outputId="90510a76-d643-4682-a7b3-aaf17de15494"
net.load_state_dict(best_weights)
```

```python id="lI_DqmEIwwda"
# groud_truth, predictions = [], []

# with torch.no_grad():
#     for batch in batch_generator(*datasets['val'], shuffle=False, bs=bs):
#         x_batch, y_batch = [b.to(device) for b in batch]
#         outputs = net(x_batch[:, 1], x_batch[:, 0], minmax)
#         groud_truth.extend(y_batch.tolist())
#         predictions.extend(outputs.tolist())

# groud_truth = np.asarray(groud_truth).ravel()
# predictions = np.asarray(predictions).ravel()
```

```python id="YdXslUMBwwda"
# final_loss = np.sqrt(np.mean((predictions - groud_truth)**2))
# print(f'Final RMSE: {final_loss:.4f}')
```

```python id="hB9t-ARGwwdb"
with open('best.weights', 'wb') as file:
    pickle.dump(best_weights, file)
```

<!-- #region id="8Qcj-GoNwwdb" -->
### Embeddings Visualization

Finally, we can create a couple of visualizations to show how various movies are encoded in embeddings space. Again, we're repeting the approach shown in the original post and apply the [Principal Components Analysis](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) to reduce the dimentionality of embeddings and show some of them with bar plots.
<!-- #endregion -->

<!-- #region id="OqwMsqWHwwdc" -->
Loading previously saved weights:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="JKlP4S1zwwdc" outputId="ed020588-98cd-4d38-9ba8-4fee5dcd31b1"
with open('best.weights', 'rb') as file:
    best_weights = pickle.load(file)
net.load_state_dict(best_weights)
```

```python id="omN9TxzFwwdd"
def to_numpy(tensor):
    return tensor.cpu().numpy()
```

<!-- #region id="_8x4DFcbwwdd" -->
Creating the mappings between original users's and movies's IDs, and new contiguous values:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="6TOmrBbbg9WA" outputId="6b34ad31-b1a9-4dbe-b12f-040dcb3a2c2b"
maps.keys()
```

```python id="qdHe7Nekwwdd"
user_id_map = umap['USERID_TO_IDX']
movie_id_map = imap['ITEMID_TO_IDX']
embed_to_original = imap['IDX_TO_ITEMID']

popular_movies = ratings_df.groupby('ITEMID').ITEMID.count().sort_values(ascending=False).values[:1000]
```

<!-- #region id="L7cJFbDXwwde" -->
Reducing the dimensionality of movie embeddings vectors:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="A0My7Epuwwde" outputId="44f167aa-656f-4fbc-ee20-898672fd5ee4"
embed = to_numpy(net.m.weight.data)
pca = PCA(n_components=5)
components = pca.fit(embed[popular_movies].T).components_
components.shape
```

<!-- #region id="MeHZJsE3wwdf" -->
Finally, creating a joined data frame with projected embeddings and movies they represent:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="g4_xKa2p7bIO" outputId="1f7539cf-8346-4f83-9a33-9d9abcd739e2"
movies = movies_df[['ITEMID','TITLE']].dropna()
movies.shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="YkGwvIUqwwdf" outputId="c99192b1-7bd9-41da-efff-dd2e5a379e5a"
components_df = pd.DataFrame(components.T, columns=[f'fc{i}' for i in range(pca.n_components_)])
components_df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="chcYgtkw8ij6" outputId="4e996939-45d5-4031-ca55-5cef20b87a6a"
components_df.shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 173} id="lTu7ZbLL8NaO" outputId="820a8b21-2a52-4c36-983b-cc0c9123efff"
movie_ids = [embed_to_original[idx] for idx in components_df.index]
meta = movies.set_index('ITEMID')
components_df['ITEMID'] = movie_ids
components_df['TITLE'] = meta.reindex(movie_ids).TITLE.values
components_df.sample(4)
```

```python id="rW-l-0Izwwdg"
def plot_components(components, component, ascending=False):
    fig, ax = plt.subplots(figsize=(18, 12))
    
    subset = components.sort_values(by=component, ascending=ascending).iloc[:12]
    columns = components_df.columns
    features = columns[columns.str.startswith('fc')].tolist()
    
    fc = subset[features]
    labels = ['\n'.join(wrap(t, width=10)) for t in subset.TITLE]
    
    fc.plot(ax=ax, kind='bar')
    y_ticks = [f'{t:2.2f}' for t in ax.get_yticks()]
    ax.set_xticklabels(labels, rotation=0, fontsize=14)
    ax.set_yticklabels(y_ticks, fontsize=14)
    ax.legend(loc='best', fontsize=14)
    
    plot_title = f"Movies with {['highest', 'lowest'][ascending]} '{component}' component values" 
    ax.set_title(plot_title, fontsize=20)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 704} id="WpJkrTX9-Asp" outputId="6724cabe-23d6-4ffe-8a21-1c3ba6cb3066"
plot_components(components_df, 'fc0', ascending=False)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 690} id="VCnP_uOT-ApN" outputId="af46a636-923f-4a9b-81e1-418a70319878"
plot_components(components_df, 'fc0', ascending=True)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 676} id="Z8sdZC7-wwdh" outputId="2288028a-5149-4e5b-9380-9c12e72378ad"
plot_components(components_df, 'fc1', ascending=False)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 732} id="8xp79OV8wwdh" outputId="b8a17815-aa7f-4e6e-d818-4f9d3b13ad12"
plot_components(components_df, 'fc1', ascending=True)
```

<!-- #region id="6zcqiJUEwwdQ" -->
Note that cosine annealing scheduler is a bit different from other schedules as soon as it starts with `base_lr` and gradually decreases it to the minimal value while triangle schedulers increase the original rate.
<!-- #endregion -->

<!-- #region id="27Y_3GcPgTfS" -->
## MF with PyTorch on ML-100k
<!-- #endregion -->

<!-- #region id="2jqJNaQhK8ji" -->
### Utils
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="vsby4vwWGlWJ" outputId="52308560-91b9-4db7-daea-3dc502d34bb5"
%%writefile utils.py

import os
import requests
import zipfile

import numpy as np
import pandas as pd
import scipy.sparse as sp

"""
Shamelessly stolen from
https://github.com/maciejkula/triplet_recommendations_keras
"""


def train_test_split(interactions, n=10):
    """
    Split an interactions matrix into training and test sets.
    Parameters
    ----------
    interactions : np.ndarray
    n : int (default=10)
        Number of items to select / row to place into test.

    Returns
    -------
    train : np.ndarray
    test : np.ndarray
    """
    test = np.zeros(interactions.shape)
    train = interactions.copy()
    for user in range(interactions.shape[0]):
        if interactions[user, :].nonzero()[0].shape[0] > n:
            test_interactions = np.random.choice(interactions[user, :].nonzero()[0],
                                                 size=n,
                                                 replace=False)
            train[user, test_interactions] = 0.
            test[user, test_interactions] = interactions[user, test_interactions]

    # Test and training are truly disjoint
    assert(np.all((train * test) == 0))
    return train, test


def _get_data_path():
    """
    Get path to the movielens dataset file.
    """
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'data')
    if not os.path.exists(data_path):
        print('Making data path')
        os.mkdir(data_path)
    return data_path


def _download_movielens(dest_path):
    """
    Download the dataset.
    """

    url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
    req = requests.get(url, stream=True)

    print('Downloading MovieLens data')

    with open(os.path.join(dest_path, 'ml-100k.zip'), 'wb') as fd:
        for chunk in req.iter_content(chunk_size=None):
            fd.write(chunk)

    with zipfile.ZipFile(os.path.join(dest_path, 'ml-100k.zip'), 'r') as z:
        z.extractall(dest_path)


def read_movielens_df():
    path = _get_data_path()
    zipfile = os.path.join(path, 'ml-100k.zip')
    if not os.path.isfile(zipfile):
        _download_movielens(path)
    fname = os.path.join(path, 'ml-100k', 'u.data')
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(fname, sep='\t', names=names)
    return df


def get_movielens_interactions():
    df = read_movielens_df()

    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]

    interactions = np.zeros((n_users, n_items))
    for row in df.itertuples():
        interactions[row[1] - 1, row[2] - 1] = row[3]
    return interactions


def get_movielens_train_test_split(implicit=False):
    interactions = get_movielens_interactions()
    if implicit:
        interactions = (interactions >= 4).astype(np.float32)
    train, test = train_test_split(interactions)
    train = sp.coo_matrix(train)
    test = sp.coo_matrix(test)
    return train, test
```

<!-- #region id="d9wghzkqLR2i" -->
### Metrics
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="EWmOo0AqKiEN" outputId="5b4b9d36-218e-42ed-da99-f5af0b4b93f9"
%%writefile metrics.py

import numpy as np
from sklearn.metrics import roc_auc_score
from torch import multiprocessing as mp
import torch


def get_row_indices(row, interactions):
    start = interactions.indptr[row]
    end = interactions.indptr[row + 1]
    return interactions.indices[start:end]


def auc(model, interactions, num_workers=1):
    aucs = []
    processes = []
    n_users = interactions.shape[0]
    mp_batch = int(np.ceil(n_users / num_workers))

    queue = mp.Queue()
    rows = np.arange(n_users)
    np.random.shuffle(rows)
    for rank in range(num_workers):
        start = rank * mp_batch
        end = np.min((start + mp_batch,  n_users))
        p = mp.Process(target=batch_auc,
                       args=(queue, rows[start:end], interactions, model))
        p.start()
        processes.append(p)

    while True:
        is_alive = False
        for p in processes:
            if p.is_alive():
                is_alive = True
                break
        if not is_alive and queue.empty():
            break

        while not queue.empty():
            aucs.append(queue.get())

    queue.close()
    for p in processes:
        p.join()
    return np.mean(aucs)


def batch_auc(queue, rows, interactions, model):
    n_items = interactions.shape[1]
    items = torch.arange(0, n_items).long()
    users_init = torch.ones(n_items).long()
    for row in rows:
        row = int(row)
        users = users_init.fill_(row)

        preds = model.predict(users, items)
        actuals = get_row_indices(row, interactions)

        if len(actuals) == 0:
            continue
        y_test = np.zeros(n_items)
        y_test[actuals] = 1
        queue.put(roc_auc_score(y_test, preds.data.numpy()))


def patk(model, interactions, num_workers=1, k=5):
    patks = []
    processes = []
    n_users = interactions.shape[0]
    mp_batch = int(np.ceil(n_users / num_workers))

    queue = mp.Queue()
    rows = np.arange(n_users)
    np.random.shuffle(rows)
    for rank in range(num_workers):
        start = rank * mp_batch
        end = np.min((start + mp_batch, n_users))
        p = mp.Process(target=batch_patk,
                       args=(queue, rows[start:end], interactions, model),
                       kwargs={'k': k})
        p.start()
        processes.append(p)

    while True:
        is_alive = False
        for p in processes:
            if p.is_alive():
                is_alive = True
                break
        if not is_alive and queue.empty():
            break

        while not queue.empty():
            patks.append(queue.get())

    queue.close()
    for p in processes:
        p.join()
    return np.mean(patks)


def batch_patk(queue, rows, interactions, model, k=5):
    n_items = interactions.shape[1]

    items = torch.arange(0, n_items).long()
    users_init = torch.ones(n_items).long()
    for row in rows:
        row = int(row)
        users = users_init.fill_(row)

        preds = model.predict(users, items)
        actuals = get_row_indices(row, interactions)

        if len(actuals) == 0:
            continue

        top_k = np.argpartition(-np.squeeze(preds.data.numpy()), k)
        top_k = set(top_k[:k])
        true_pids = set(actuals)
        if true_pids:
            queue.put(len(top_k & true_pids) / float(k))
```

<!-- #region id="7N1Rl15-LJAj" -->
### Model
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="NUlsa6LeLKqu" outputId="6326b89c-3659-4008-8801-f8d3b991b6fc"
%%writefile torchmf.py

import collections
import os

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
import torch.multiprocessing as mp
import torch.utils.data as data
from tqdm import tqdm

import metrics


# Models
# Interactions Dataset => Singular Iter => Singular Loss
# Pairwise Datasets => Pairwise Iter => Pairwise Loss
# Pairwise Iters
# Loss Functions
# Optimizers
# Metric callbacks

# Serve up users, items (and items could be pos_items, neg_items)
# In this case, the iteration remains the same. Pass both items into a model
# which is a concat of the base model. it handles the pos and neg_items
# accordingly. define the loss after.


class Interactions(data.Dataset):
    """
    Hold data in the form of an interactions matrix.
    Typical use-case is like a ratings matrix:
    - Users are the rows
    - Items are the columns
    - Elements of the matrix are the ratings given by a user for an item.
    """

    def __init__(self, mat):
        self.mat = mat.astype(np.float32).tocoo()
        self.n_users = self.mat.shape[0]
        self.n_items = self.mat.shape[1]

    def __getitem__(self, index):
        row = self.mat.row[index]
        col = self.mat.col[index]
        val = self.mat.data[index]
        return (row, col), val

    def __len__(self):
        return self.mat.nnz


class PairwiseInteractions(data.Dataset):
    """
    Sample data from an interactions matrix in a pairwise fashion. The row is
    treated as the main dimension, and the columns are sampled pairwise.
    """

    def __init__(self, mat):
        self.mat = mat.astype(np.float32).tocoo()

        self.n_users = self.mat.shape[0]
        self.n_items = self.mat.shape[1]

        self.mat_csr = self.mat.tocsr()
        if not self.mat_csr.has_sorted_indices:
            self.mat_csr.sort_indices()

    def __getitem__(self, index):
        row = self.mat.row[index]
        found = False

        while not found:
            neg_col = np.random.randint(self.n_items)
            if self.not_rated(row, neg_col, self.mat_csr.indptr,
                              self.mat_csr.indices):
                found = True

        pos_col = self.mat.col[index]
        val = self.mat.data[index]

        return (row, (pos_col, neg_col)), val

    def __len__(self):
        return self.mat.nnz

    @staticmethod
    def not_rated(row, col, indptr, indices):
        # similar to use of bsearch in lightfm
        start = indptr[row]
        end = indptr[row + 1]
        searched = np.searchsorted(indices[start:end], col, 'right')
        if searched >= (end - start):
            # After the array
            return False
        return col != indices[searched]  # Not found

    def get_row_indices(self, row):
        start = self.mat_csr.indptr[row]
        end = self.mat_csr.indptr[row + 1]
        return self.mat_csr.indices[start:end]


class BaseModule(nn.Module):
    """
    Base module for explicit matrix factorization.
    """
    
    def __init__(self,
                 n_users,
                 n_items,
                 n_factors=40,
                 dropout_p=0,
                 sparse=False):
        """

        Parameters
        ----------
        n_users : int
            Number of users
        n_items : int
            Number of items
        n_factors : int
            Number of latent factors (or embeddings or whatever you want to
            call it).
        dropout_p : float
            p in nn.Dropout module. Probability of dropout.
        sparse : bool
            Whether or not to treat embeddings as sparse. NOTE: cannot use
            weight decay on the optimizer if sparse=True. Also, can only use
            Adagrad.
        """
        super(BaseModule, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.user_biases = nn.Embedding(n_users, 1, sparse=sparse)
        self.item_biases = nn.Embedding(n_items, 1, sparse=sparse)
        self.user_embeddings = nn.Embedding(n_users, n_factors, sparse=sparse)
        self.item_embeddings = nn.Embedding(n_items, n_factors, sparse=sparse)
        
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=self.dropout_p)

        self.sparse = sparse
        
    def forward(self, users, items):
        """
        Forward pass through the model. For a single user and item, this
        looks like:

        user_bias + item_bias + user_embeddings.dot(item_embeddings)

        Parameters
        ----------
        users : np.ndarray
            Array of user indices
        items : np.ndarray
            Array of item indices

        Returns
        -------
        preds : np.ndarray
            Predicted ratings.

        """
        ues = self.user_embeddings(users)
        uis = self.item_embeddings(items)

        preds = self.user_biases(users)
        preds += self.item_biases(items)
        preds += (self.dropout(ues) * self.dropout(uis)).sum(dim=1, keepdim=True)

        return preds.squeeze()
    
    def __call__(self, *args):
        return self.forward(*args)

    def predict(self, users, items):
        return self.forward(users, items)


def bpr_loss(preds, vals):
    sig = nn.Sigmoid()
    return (1.0 - sig(preds)).pow(2).sum()


class BPRModule(nn.Module):
    
    def __init__(self,
                 n_users,
                 n_items,
                 n_factors=40,
                 dropout_p=0,
                 sparse=False,
                 model=BaseModule):
        super(BPRModule, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.dropout_p = dropout_p
        self.sparse = sparse
        self.pred_model = model(
            self.n_users,
            self.n_items,
            n_factors=n_factors,
            dropout_p=dropout_p,
            sparse=sparse
        )

    def forward(self, users, items):
        assert isinstance(items, tuple), \
            'Must pass in items as (pos_items, neg_items)'
        # Unpack
        (pos_items, neg_items) = items
        pos_preds = self.pred_model(users, pos_items)
        neg_preds = self.pred_model(users, neg_items)
        return pos_preds - neg_preds

    def predict(self, users, items):
        return self.pred_model(users, items)


class BasePipeline:
    """
    Class defining a training pipeline. Instantiates data loaders, model,
    and optimizer. Handles training for multiple epochs and keeping track of
    train and test loss.
    """

    def __init__(self,
                 train,
                 test=None,
                 model=BaseModule,
                 n_factors=40,
                 batch_size=32,
                 dropout_p=0.02,
                 sparse=False,
                 lr=0.01,
                 weight_decay=0.,
                 optimizer=torch.optim.Adam,
                 loss_function=nn.MSELoss(reduction='sum'),
                 n_epochs=10,
                 verbose=False,
                 random_seed=None,
                 interaction_class=Interactions,
                 hogwild=False,
                 num_workers=0,
                 eval_metrics=None,
                 k=5):
        self.train = train
        self.test = test

        if hogwild:
            num_loader_workers = 0
        else:
            num_loader_workers = num_workers
        self.train_loader = data.DataLoader(
            interaction_class(train), batch_size=batch_size, shuffle=True,
            num_workers=num_loader_workers)
        if self.test is not None:
            self.test_loader = data.DataLoader(
                interaction_class(test), batch_size=batch_size, shuffle=True,
                num_workers=num_loader_workers)
        self.num_workers = num_workers
        self.n_users = self.train.shape[0]
        self.n_items = self.train.shape[1]
        self.n_factors = n_factors
        self.batch_size = batch_size
        self.dropout_p = dropout_p
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_function = loss_function
        self.n_epochs = n_epochs
        if sparse:
            assert weight_decay == 0.0
        self.model = model(self.n_users,
                           self.n_items,
                           n_factors=self.n_factors,
                           dropout_p=self.dropout_p,
                           sparse=sparse)
        self.optimizer = optimizer(self.model.parameters(),
                                   lr=self.lr,
                                   weight_decay=self.weight_decay)
        self.warm_start = False
        self.losses = collections.defaultdict(list)
        self.verbose = verbose
        self.hogwild = hogwild
        if random_seed is not None:
            if self.hogwild:
                random_seed += os.getpid()
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

        if eval_metrics is None:
            eval_metrics = []
        self.eval_metrics = eval_metrics
        self.k = k

    def break_grads(self):
        for param in self.model.parameters():
            # Break gradient sharing
            if param.grad is not None:
                param.grad.data = param.grad.data.clone()

    def fit(self):
        for epoch in range(1, self.n_epochs + 1):

            if self.hogwild:
                self.model.share_memory()
                processes = []
                train_losses = []
                queue = mp.Queue()
                for rank in range(self.num_workers):
                    p = mp.Process(target=self._fit_epoch,
                                   kwargs={'epoch': epoch,
                                           'queue': queue})
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()

                while True:
                    is_alive = False
                    for p in processes:
                        if p.is_alive():
                            is_alive = True
                            break
                    if not is_alive and queue.empty():
                        break

                    while not queue.empty():
                        train_losses.append(queue.get())
                queue.close()
                train_loss = np.mean(train_losses)
            else:
                train_loss = self._fit_epoch(epoch)

            self.losses['train'].append(train_loss)
            row = 'Epoch: {0:^3}  train: {1:^10.5f}'.format(epoch, self.losses['train'][-1])
            if self.test is not None:
                self.losses['test'].append(self._validation_loss())
                row += 'val: {0:^10.5f}'.format(self.losses['test'][-1])
                for metric in self.eval_metrics:
                    func = getattr(metrics, metric)
                    res = func(self.model, self.test_loader.dataset.mat_csr,
                               num_workers=self.num_workers)
                    self.losses['eval-{}'.format(metric)].append(res)
                    row += 'eval-{0}: {1:^10.5f}'.format(metric, res)
            self.losses['epoch'].append(epoch)
            if self.verbose:
                print(row)

    def _fit_epoch(self, epoch=1, queue=None):
        if self.hogwild:
            self.break_grads()

        self.model.train()
        total_loss = torch.Tensor([0])
        pbar = tqdm(enumerate(self.train_loader),
                    total=len(self.train_loader),
                    desc='({0:^3})'.format(epoch))
        for batch_idx, ((row, col), val) in pbar:
            self.optimizer.zero_grad()

            row = row.long()
            # TODO: turn this into a collate_fn like the data_loader
            if isinstance(col, list):
                col = tuple(c.long() for c in col)
            else:
                col = col.long()
            val = val.float()

            preds = self.model(row, col)
            loss = self.loss_function(preds, val)
            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()
            batch_loss = loss.item() / row.size()[0]
            pbar.set_postfix(train_loss=batch_loss)
        total_loss /= self.train.nnz
        if queue is not None:
            queue.put(total_loss[0])
        else:
            return total_loss[0]

    def _validation_loss(self):
        self.model.eval()
        total_loss = torch.Tensor([0])
        for batch_idx, ((row, col), val) in enumerate(self.test_loader):
            row = row.long()
            if isinstance(col, list):
                col = tuple(c.long() for c in col)
            else:
                col = col.long()
            val = val.float()

            preds = self.model(row, col)
            loss = self.loss_function(preds, val)
            total_loss += loss.item()

        total_loss /= self.test.nnz
        return total_loss[0]
```

<!-- #region id="5KpaDgMwLNI5" -->
### Trainer
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="HxKI9a2dLDDy" outputId="538238ee-aa22-49e9-aa44-68eb30d5e92a"
%%writefile run.py

import argparse
import pickle

import torch

from torchmf import (BaseModule, BPRModule, BasePipeline,
                     bpr_loss, PairwiseInteractions)
import utils


def explicit():
    train, test = utils.get_movielens_train_test_split()
    pipeline = BasePipeline(train, test=test, model=BaseModule,
                            n_factors=10, batch_size=1024, dropout_p=0.02,
                            lr=0.02, weight_decay=0.1,
                            optimizer=torch.optim.Adam, n_epochs=40,
                            verbose=True, random_seed=2017)
    pipeline.fit()


def implicit():
    train, test = utils.get_movielens_train_test_split(implicit=True)

    pipeline = BasePipeline(train, test=test, verbose=True,
                           batch_size=1024, num_workers=4,
                           n_factors=20, weight_decay=0,
                           dropout_p=0., lr=.2, sparse=True,
                           optimizer=torch.optim.SGD, n_epochs=40,
                           random_seed=2017, loss_function=bpr_loss,
                           model=BPRModule,
                           interaction_class=PairwiseInteractions,
                           eval_metrics=('auc', 'patk'))
    pipeline.fit()


def hogwild():
    train, test = utils.get_movielens_train_test_split(implicit=True)

    pipeline = BasePipeline(train, test=test, verbose=True,
                            batch_size=1024, num_workers=4,
                            n_factors=20, weight_decay=0,
                            dropout_p=0., lr=.2, sparse=True,
                            optimizer=torch.optim.SGD, n_epochs=40,
                            random_seed=2017, loss_function=bpr_loss,
                            model=BPRModule, hogwild=True,
                            interaction_class=PairwiseInteractions,
                            eval_metrics=('auc', 'patk'))
    pipeline.fit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='torchmf')
    parser.add_argument('--example',
                        help='explicit, implicit, or hogwild')
    args = parser.parse_args()
    if args.example == 'explicit':
        explicit()
    elif args.example == 'implicit':
        implicit()
    elif args.example == 'hogwild':
        hogwild()
    else:
        print('example must be explicit, implicit, or hogwild')
```

<!-- #region id="40lNybzWLtRP" -->
### Explicit Model Training
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="0i4BoW9HLHSb" outputId="b2a2a2bc-c5c8-4c6e-f576-6a68326d5a30"
!python run.py --example explicit
```

<!-- #region id="JxqWyDE4LxPb" -->
### Implicit Model Training
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="IBXdHOUXLdPE" outputId="62bd667d-56d6-4969-e390-8cab4842353e"
!python run.py --example implicit
```

<!-- #region id="Y2yIPrmu98bL" -->
## Hybrid Model with PyTorch on ML-100k
<!-- #endregion -->

<!-- #region id="kImzuHXJ9_kV" -->
Testing out the features of Collie Recs library on MovieLens-100K. Training Factorization and Hybrid models with Pytorch Lightning.
<!-- #endregion -->

```python id="P43fS4H27gCt"
!pip install -q collie_recs
!pip install -q git+https://github.com/sparsh-ai/recochef.git
```

```python id="PD0n8kefAb67"
import os
import joblib
import numpy as np
import pandas as pd

from collie_recs.interactions import Interactions
from collie_recs.interactions import ApproximateNegativeSamplingInteractionsDataLoader
from collie_recs.cross_validation import stratified_split
from collie_recs.metrics import auc, evaluate_in_batches, mapk, mrr
from collie_recs.model import CollieTrainer, MatrixFactorizationModel, HybridPretrainedModel
from collie_recs.movielens import get_recommendation_visualizations

import torch
from pytorch_lightning.utilities.seed import seed_everything

from recochef.datasets.movielens import MovieLens
from recochef.preprocessing.encode import label_encode as le

from IPython.display import HTML
```

```python colab={"base_uri": "https://localhost:8080/"} id="ClWobbREN4VU" outputId="3f4c1125-a726-444e-d6f5-bc231df2a167"
# this handy PyTorch Lightning function fixes random seeds across all the libraries used here
seed_everything(22)
```

<!-- #region id="T0N-kmIrcDZr" -->
### Data Loading
<!-- #endregion -->

```python id="LEuHGsYgGv-e"
data_object = MovieLens()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="g3lLxQE1G120" outputId="2c6b310f-b237-4f1f-a69e-bee2ac458172"
df = data_object.load_interactions()
df.head()
```

<!-- #region id="3X8-bE9YcGBg" -->
### Preprocessing
<!-- #endregion -->

```python id="tJvifZdrLsGK"
# drop duplicate user-item pair records, keeping recent ratings only
df.drop_duplicates(subset=['USERID','ITEMID'], keep='last', inplace=True)
```

```python id="eq7GY0lEINgA"
# convert the explicit data to implicit by only keeping interactions with a rating ``>= 4``
df = df[df.RATING>=4].reset_index(drop=True)
df['RATING'] = 1
```

```python id="Jo4FnRzSHPzs"
# label encode
df, umap = le(df, col='USERID')
df, imap = le(df, col='ITEMID')

df = df.astype('int64')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="LA9BtiDrJ_wf" outputId="9044fa8b-53c4-421f-cbfa-8ce6b374d666"
df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="W3hUWib-MyjB" outputId="148d3ca2-17fb-4622-db52-122f045d1563"
user_counts = df.groupby(by='USERID')['ITEMID'].count()
user_list = user_counts[user_counts>=3].index.tolist()
df = df[df.USERID.isin(user_list)]

df.head()
```

<!-- #region id="37Y7qEEJNiE4" -->
### Interactions
While we have chosen to represent the data as a ``pandas.DataFrame`` for easy viewing now, Collie uses a custom ``torch.utils.data.Dataset`` called ``Interactions``. This class stores a sparse representation of the data and offers some handy benefits, including: 

* The ability to index the data with a ``__getitem__`` method 
* The ability to sample many negative items (we will get to this later!) 
* Nice quality checks to ensure data is free of errors before model training 

Instantiating the object is simple! 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="dLcqt57TI-6l" outputId="b399e7e8-fec0-460a-8048-e7a1e9395106"
interactions = Interactions(
    users=df['USERID'],
    items=df['ITEMID'],
    ratings=df['RATING'],
    allow_missing_ids=True,
)

interactions
```

<!-- #region id="4TaWM_OFNZzn" -->
### Data Splits 
With an ``Interactions`` dataset, Collie supports two types of data splits. 

1. **Random split**: This code randomly assigns an interaction to a ``train``, ``validation``, or ``test`` dataset. While this is significantly faster to perform than a stratified split, it does not guarantee any balance, meaning a scenario where a user will have no interactions in the ``train`` dataset and all in the ``test`` dataset is possible. 
2. **Stratified split**: While this code runs slower than a random split, this guarantees that each user will be represented in the ``train``, ``validation``, and ``test`` dataset. This is by far the most fair way to train and evaluate a recommendation model. 

Since this is a small dataset and we have time, we will go ahead and use ``stratified_split``. If you're short on time, a ``random_split`` can easily be swapped in, since both functions share the same API! 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="U_4IPy2aNLVE" outputId="5db1d12d-5d17-46f5-c4d1-05cfe4d458e4"
train_interactions, val_interactions = stratified_split(interactions, test_p=0.1, seed=42)
train_interactions, val_interactions
```

<!-- #region id="ZO5rLYzH-imu" -->
### Model Architecture 
With our data ready-to-go, we can now start training a recommendation model. While Collie has several model architectures built-in, the simplest by far is the ``MatrixFactorizationModel``, which use ``torch.nn.Embedding`` layers and a dot-product operation to perform matrix factorization via collaborative filtering.
<!-- #endregion -->

<!-- #region id="ZWi4VUjeghUA" -->
Digging through the code of [``collie_recs.model.MatrixFactorizationModel``](../collie_recs/model.py) shows the architecture is as simple as we might think. For simplicity, we will include relevant portions below so we know exactly what we are building: 

````python
def _setup_model(self, **kwargs) -> None:
    self.user_biases = ZeroEmbedding(num_embeddings=self.hparams.num_users,
                                     embedding_dim=1,
                                     sparse=self.hparams.sparse)
    self.item_biases = ZeroEmbedding(num_embeddings=self.hparams.num_items,
                                     embedding_dim=1,
                                     sparse=self.hparams.sparse)
    self.user_embeddings = ScaledEmbedding(num_embeddings=self.hparams.num_users,
                                           embedding_dim=self.hparams.embedding_dim,
                                           sparse=self.hparams.sparse)
    self.item_embeddings = ScaledEmbedding(num_embeddings=self.hparams.num_items,
                                           embedding_dim=self.hparams.embedding_dim,
                                           sparse=self.hparams.sparse)

        
def forward(self, users: torch.tensor, items: torch.tensor) -> torch.tensor:
    user_embeddings = self.user_embeddings(users)
    item_embeddings = self.item_embeddings(items)

    preds = (
        torch.mul(user_embeddings, item_embeddings).sum(axis=1)
        + self.user_biases(users).squeeze(1)
        + self.item_biases(items).squeeze(1)
    )

    if self.hparams.y_range is not None:
        preds = (
            torch.sigmoid(preds)
            * (self.hparams.y_range[1] - self.hparams.y_range[0])
            + self.hparams.y_range[0]
        )

    return preds
````

Let's go ahead and instantiate the model and start training! Note that even if you are running this model on a CPU instead of a GPU, this will still be relatively quick to fully train. 
<!-- #endregion -->

<!-- #region id="L7o3t9vNN-Lt" -->
Collie is built with PyTorch Lightning, so all the model classes and the ``CollieTrainer`` class accept all the training options available in PyTorch Lightning. Here, we're going to set the embedding dimension and learning rate differently, and go with the defaults for everything else
<!-- #endregion -->

```python id="LNfxzlruN1xx"
model = MatrixFactorizationModel(
    train=train_interactions,
    val=val_interactions,
    embedding_dim=10,
    lr=1e-2,
)
```

```python id="TKxeyMsMN1vg"
trainer = CollieTrainer(model, max_epochs=10, deterministic=True)

trainer.fit(model)
```

<!-- #region id="dx8EoEhC_Cjh" -->
```text
GPU available: False, used: False
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs

  | Name            | Type            | Params
----------------------------------------------------
0 | user_biases     | ZeroEmbedding   | 942   
1 | item_biases     | ZeroEmbedding   | 1.4 K 
2 | user_embeddings | ScaledEmbedding | 9.4 K 
3 | item_embeddings | ScaledEmbedding | 14.5 K
4 | dropout         | Dropout         | 0     
----------------------------------------------------
26.3 K    Trainable params
0         Non-trainable params
26.3 K    Total params
0.105     Total estimated model params size (MB)
Validation sanity check: 0%
0/2 [00:00<?, ?it/s]
Global seed set to 22
Epoch 9: 100%
55/55 [00:05<00:00, 10.37it/s, loss=1.74, v_num=0]
Epoch     3: reducing learning rate of group 0 to 1.0000e-03.
Epoch     3: reducing learning rate of group 0 to 1.0000e-03.
Epoch     5: reducing learning rate of group 0 to 1.0000e-04.
Epoch     5: reducing learning rate of group 0 to 1.0000e-04.
Epoch     7: reducing learning rate of group 0 to 1.0000e-05.
Epoch     7: reducing learning rate of group 0 to 1.0000e-05.
Epoch     9: reducing learning rate of group 0 to 1.0000e-06.
Epoch     9: reducing learning rate of group 0 to 1.0000e-06.
```
<!-- #endregion -->

<!-- #region id="b1PhE5nDghUC" -->
### Evaluate the Model 
We have a model! Now, we need to figure out how well we did. Evaluating implicit recommendation models is a bit tricky, but Collie offers the following metrics that are built into the library. They use vectorized operations that can run on the GPU in a single pass for speed-ups. 

* [``Mean Average Precision at K (MAP@K)``](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision) 
* [``Mean Reciprocal Rank``](https://en.wikipedia.org/wiki/Mean_reciprocal_rank) 
* [``Area Under the Curve (AUC)``](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve) 

We'll go ahead and evaluate all of these at once below. 
<!-- #endregion -->

```python id="fUFAraDsOHtL"
model.eval()  # set model to inference mode
mapk_score, mrr_score, auc_score = evaluate_in_batches([mapk, mrr, auc], val_interactions, model)

print(f'MAP@10 Score: {mapk_score}')
print(f'MRR Score:    {mrr_score}')
print(f'AUC Score:    {auc_score}')
```

<!-- #region id="CFcwh56u-9bM" -->
```text
MAP@10 Score: 0.04792124467542778
MRR Score:    0.1670155641949101
AUC Score:    0.8854083361899018
```
<!-- #endregion -->

<!-- #region id="zlqMa7eWOMGl" -->
### Inference
<!-- #endregion -->

<!-- #region id="z7y-J7KUghUD" -->
We can also look at particular users to get a sense of what the recs look like. 
<!-- #endregion -->

```python id="sIQsXedgghUD" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="5496ece7-9dd4-47c9-b804-a2fb283cf8dd"
# select a random user ID to look at recommendations for
user_id = np.random.randint(10, train_interactions.num_users)

display(
    HTML(
        get_recommendation_visualizations(
            model=model,
            user_id=user_id,
            filter_films=True,
            shuffle=True,
            detailed=True,
        )
    )
)
```

<!-- #region id="DbZ9ufGvghUE" -->
### Save and Load a Standard Model 
<!-- #endregion -->

```python id="WqJbXHXgghUG"
# we can save the model with...
os.makedirs('models', exist_ok=True)
model.save_model('models/matrix_factorization_model.pth')
```

```python id="Dz_8miLPghUG" colab={"base_uri": "https://localhost:8080/"} outputId="1f2f8c07-be29-4fb9-ca8c-168ac1515f16"
# ... and if we wanted to load that model back in, we can do that easily...
model_loaded_in = MatrixFactorizationModel(load_model_path='models/matrix_factorization_model.pth')

model_loaded_in
```

<!-- #region id="cQpWlCuughUH" -->
Now that we've built our first model and gotten some baseline metrics, we now will be looking at some more advanced features in Collie's ``MatrixFactorizationModel``. 
<!-- #endregion -->

<!-- #region id="Q3Ne8ETsgzLe" -->
### Faster Data Loading Through Approximate Negative Sampling 
<!-- #endregion -->

<!-- #region id="8nBu6PZhgzLe" -->
With sufficiently large enough data, verifying that each negative sample is one a user has *not* interacted with becomes expensive. With many items, this can soon become a bottleneck in the training process. 

Yet, when we have many items, the chances a user has interacted with most is increasingly rare. Say we have ``1,000,000`` items and we want to sample ``10`` negative items for a user that has positively interacted with ``200`` items. The chance that we accidentally select a positive item in a random sample of ``10`` items is just ``0.2%``. At that point, it might be worth it to forgo the expensive check to assert our negative sample is true, and instead just randomly sample negative items with the hope that most of the time, they will happen to be negative. 

This is the theory behind the ``ApproximateNegativeSamplingInteractionsDataLoader``, an alternate DataLoader built into Collie. Let's train a model with this below, noting how similar this procedure looks to that in the previous tutorial. 
<!-- #endregion -->

```python id="MgSijz04gzLf"
train_loader = ApproximateNegativeSamplingInteractionsDataLoader(train_interactions, batch_size=1024, shuffle=True)
val_loader = ApproximateNegativeSamplingInteractionsDataLoader(val_interactions, batch_size=1024, shuffle=False)
```

```python id="n9wQLd9-gzLf"
model = MatrixFactorizationModel(
    train=train_loader,
    val=val_loader,
    embedding_dim=10,
    lr=1e-2,
)
```

```python id="AbYirCSNgzLg"
trainer = CollieTrainer(model, max_epochs=10, deterministic=True)

trainer.fit(model)
model.eval()
```

<!-- #region id="uvykQiW2_Oof" -->
```text
GPU available: True, used: True
TPU available: False, using: 0 TPU cores
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name            | Type            | Params
----------------------------------------------------
0 | user_biases     | ZeroEmbedding   | 941   
1 | item_biases     | ZeroEmbedding   | 1.4 K 
2 | user_embeddings | ScaledEmbedding | 9.4 K 
3 | item_embeddings | ScaledEmbedding | 14.5 K
4 | dropout         | Dropout         | 0     
----------------------------------------------------
26.3 K    Trainable params
0         Non-trainable params
26.3 K    Total params
0.105     Total estimated model params size (MB)
Detected GPU. Setting ``gpus`` to 1.
Global seed set to 22

MatrixFactorizationModel(
  (user_biases): ZeroEmbedding(941, 1)
  (item_biases): ZeroEmbedding(1447, 1)
  (user_embeddings): ScaledEmbedding(941, 10)
  (item_embeddings): ScaledEmbedding(1447, 10)
  (dropout): Dropout(p=0.0, inplace=False)
)
```
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 117, "referenced_widgets": ["d7c34c7be74246acb1a7da702b490029", "8ca15bf2e3be4ba88b7dbbf4ed26820d", "3e63ae601740455e81c35d4fbab2db6e", "b289ad3cdd3f4a25a9e92ed22c37aeed", "3d7d55e46c7d4e2caa6680229fe73b4e", "e841d95562314124b242c2e4225afbdb", "7b9d13b53df04e2082d4e236f50b807d", "9a4137c369cc4f26817ed3ab568a5ef7"]} id="xLKDSq-hgzLg" outputId="44183d74-a567-418d-a85e-946bf1443713"
mapk_score, mrr_score, auc_score = evaluate_in_batches([mapk, mrr, auc], val_interactions, model)

print(f'MAP@10 Score: {mapk_score}')
print(f'MRR Score:    {mrr_score}')
print(f'AUC Score:    {auc_score}')
```

<!-- #region id="qipOCQzxgzLh" -->
We're seeing a small hit on performance and only a marginal improvement in training time compared to the standard ``MatrixFactorizationModel`` model because MovieLens 100K has so few items. ``ApproximateNegativeSamplingInteractionsDataLoader`` is especially recommended for when we have more items in our data and training times need to be optimized. 

For more details on this and other DataLoaders in Collie (including those for out-of-memory datasets), check out the [docs](https://collie.readthedocs.io/en/latest/index.html)! 
<!-- #endregion -->

<!-- #region id="iGgM-FaegzLi" -->
### Multiple Optimizers 
<!-- #endregion -->

<!-- #region id="4xcFspsbgzLi" -->
Training recommendation models at ShopRunner, we have encountered something we call "the curse of popularity." 

This is best thought of in the viewpoint of a model optimizer - say we have a user, a positive item, and several negative items that we hope have recommendation scores that score lower than the positive item. As an optimizer, you can either optimize every single embedding dimension (hundreds of parameters) to achieve this, or instead choose to score a quick win by optimizing the bias terms for the items (just add a positive constant to the positive item and a negative constant to each negative item). 

While we clearly want to have varied embedding layers that reflect each user and item's taste profiles, some models learn to settle for popularity as a recommendation score proxy by over-optimizing the bias terms, essentially just returning the same set of recommendations for every user. Worst of all, since popular items are... well, popular, **the loss of this model will actually be decent, solidifying the model getting stuck in a local loss minima**. 

To counteract this, Collie supports multiple optimizers in a ``MatrixFactorizationModel``. With this, we can have a faster optimizer work to optimize the embedding layers for users and items, and a slower optimizer work to optimize the bias terms. With this, we impel the model to do the work actually coming up with varied, personalized recommendations for users while still taking into account the necessity of the bias (popularity) terms on recommendations. 

At ShopRunner, we have seen significantly better metrics and results from this type of model. With Collie, this is simple to do, as shown below. 
<!-- #endregion -->

```python id="GxUMsr61gzLj"
model = MatrixFactorizationModel(
    train=train_interactions,
    val=val_interactions,
    embedding_dim=10,
    lr=1e-2,
    bias_lr=1e-1,
    optimizer='adam',
    bias_optimizer='sgd',
)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 557, "referenced_widgets": ["49f97e2e5f55454bab313b1570b4b9c4", "b3c7b293fe8b44e0920b060f00382401", "80b4510b6a7e4d77879516c59662180c", "f1e818c6ef934704ae0ef1f80eaa7b5b", "695c6402e0a1475eb2965bee589c2bf4", "06f180111dfe4f57a12723bccb253f98", "316eaa17f9cd4558b8c7f6951b98d6f0", "75fc6c2f339b4e47bb22790d37e7ebab", "8d714e45f0d64e85b637961f4a25f3d0", "b5a4c2dd499c4d5c8d3ab2e890b44642", "d1dd8920ddfc4c97beffe89ece6b982e", "fc7d1f4bea4145af93d9059177a477fd", "ef66bb3e20eb4d25a6dc3af0fccf2e2a", "9275275798974477a5f5858880d1b429", "2bcffbf7f05b41688936d51d6c3ffcac", "28198fbab991465583818c47d8ddbb6d", "83658275a6c346d7822fd1368345e09a", "3389371180f64e088e7c6549a25b3ee1", "54d47349fff54cd0bca2e4903b06e5d8", "ec37483292aa4715a97c6f0de133cb24", "78845118227b40adac2503e57e1f38a7", "ce20fc80baf0494f976d7ebb54303833", "b0606438031a4d0fbe1b3b1bc4d49e9e", "38cb1183eb294363af2d2762d8a49a13", "b9a2d6afed7641d995fba91c5a94f8ec", "6f42c45ff32f46deb83569801ab976c8", "58d6593d8810421bb8bd39099e82feb9", "7cdc87e9c27f48e2b081bdbf2d6228e8", "45b86b0dc3394da4b2f87191d23dfed7", "585898b88c044e2e8d6fd0c46c3b575d", "a0259f8bb2264993a6d27acd097b6c05", "5b0ed212d376472987768ea06e314f9d", "a59ea6641fca44bbaa553a1624deab51", "671b80df47dc444d817c51101c7adf49", "d56ba50fc09c4502a1055ff9a3199062", "1bd1fa4ccb564778be34c5fe9036e731", "1b3ff709422c4864acff0ca42ec89e07", "5b6f8c0d04be4ec19c55259caf622c02", "b8903c665b7c4a3695101aa7d6e8f929", "518b4b044b4d411fb43dac5fc993c02c", "36d600f8e9be46a6b05ec580025aac20", "f14194bdbf724d20bdd7d5b572374608", "b62d3801dd3d4f45b53de6d6c25e26bd", "066ce24dcd2e46acb54f1fd437963b85", "bd80f8f37cb247c69f969d96b146ea8c", "4100968fe609430f8cd1f73c48a356f2", "9e6c6c48df824c548aa0425888adac9b", "27866404913f4820bb46d1641f63dc1f", "d75b9faa61b242279706a77458e55276", "ba792412f171494d937051219e01607d", "f43bba18ba134b269006e100f8de55d1", "63c6fb5bfa3147a89962641d1a7d215b", "6d7a19d43c4846d99d3d470fd5d7d66e", "90af7f79f905435f8c81ca21e59cab59", "fad0632c95f74aca92b6e72751da0632", "7eb49fb403564170b47e0c6f867f81e1", "57322a343ddd4ad2bc408a840eca0e98", "18b3142ac5c24401941cd4f79bac783b", "6d730313b939447bbd396526cdee3fad", "73cccc23b004406e8521b0cc26401c9c", "7520241c3b2c4591bc0532d20dafccda", "5dd8ebbab9ce418cae4eaf9f568ab5c9", "0afd2fe607b04590813dc297c3bfc4da", "c14405b6297a435eaab8032dbbb9966f", "0351e40f8c0f46708aeea7233565800a", "d7e152362cd54b4380a6cd3e506e7b86", "73dff796a59d444a96016b2578f277d7", "89a5fbcb3a4e4732a72c8cee79a346b6", "bcde36e5444444568f7a6fa34d85ccaa", "53e3249bbef446208f54a6216420062f", "235ef9871d5443a9ab3b31f1c231f53d", "b329742fdb3e44e2974d3a31700072a3", "93fbdf1f9df1464888cdecc6285ef575", "7ef0973e658740528843af596067fe8e", "dad5c7f76e20437ca651d4d39ac68660", "3a61c8ddd8b74af78c6ceffb7001da02", "fec250de7d6f45688eb48b51d837b53a", "54d415a20e204d53bbe9baddd4598e2f", "136520e97ebe4b04b9b46a44e46297fc", "c4d1f1810116465b9dcb9282eed0f113", "2c8df37c614447d5b5064dbbaa837081", "dc9b62bd3eb94dc8afd9c21491faff0b", "2416bc3ad43149ff9e8d10572693f115", "93054dc57a344b9daae9e6e36e14b594", "4d9cf1e48cb74afb91410b46b2d601b5", "897727bf519648e0bc3f204771ded957", "795429863893411cab25dfe5771a2e4a", "93a527dd2045487393fb8280ff3945b1", "3d3ca53ac0cb41d1bbb58aa754a79619", "600e766c4f4c4136875db902954f9ac7", "2cf5bd88c0c74729a89a70f6ba9dd02e", "a247748059184bbaac041b8fffd99e3f", "010c2bf6e2af470d950542394f8ceef5", "5bb0a358fcd64106acdac950f82d12a7", "16202361258644f6ab7b168990f73136", "c2fffba8678b48f9957b9a581aae9e2e"]} id="dQ_tTRfOgzLj" outputId="4d3af437-c62d-4b9b-ddb5-eaf3731556da"
trainer = CollieTrainer(model, max_epochs=10, deterministic=True)

trainer.fit(model)
model.eval()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 117, "referenced_widgets": ["5c2218b13d6345ff91c4d8980b2b2ca0", "309d24f05b1b4bc5b7d0148795fd5cf4", "ccfb5dc3eef14f1f8324bc48b3dc0e7f", "a84e5372c5c24417a08487dd3efcebe8", "c28c07424b7f45088e73ac25f2bb712a", "5d01bda6f6354f25bb6dcb15ef4596a0", "758cba83e9414f60bf87799a71b3cd7f", "48d71d4847be4a25a237568371ae4493"]} id="ENZSOd1DgzLk" outputId="7b1844fb-b81a-43cb-8a98-5206b87b4506"
mapk_score, mrr_score, auc_score = evaluate_in_batches([mapk, mrr, auc], val_interactions, model)

print(f'MAP@10 Score: {mapk_score}')
print(f'MRR Score:    {mrr_score}')
print(f'AUC Score:    {auc_score}')
```

<!-- #region id="blXcVpg3gzLk" -->
Again, we're not seeing as much performance increase here compared to the standard model because MovieLens 100K has so few items. For a more dramatic difference, try training this model on a larger dataset, such as MovieLens 10M, adjusting the architecture-specific hyperparameters, or train longer. 
<!-- #endregion -->

<!-- #region id="1Dt0IsWJgzLk" -->
### Item-Item Similarity 
<!-- #endregion -->

<!-- #region id="sqQJpyKVgzLl" -->
While we've trained every model thus far to work for member-item recommendations (given a *member*, recommend *items* - think of this best as "Personalized recommendations for you"), we also have access to item-item recommendations for free (given a seed *item*, recommend similar *items* - think of this more like "People who interacted with this item also interacted with..."). 

With Collie, accessing this is simple! 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 343} id="RRMouFweUOhw" outputId="02e6281a-f64e-43c4-a151-84601d91ab50"
df_item = data_object.load_items()
df_item.head()
```

```python id="Ycl8PcLIgzLl" colab={"base_uri": "https://localhost:8080/", "height": 343} outputId="27f2b6aa-9b6f-4fe4-9d8f-3637cadd4bbe"
df_item = le(df_item, col='ITEMID', maps=imap)
df_item.head()
```

```python id="TvtbOrbtgzLl" colab={"base_uri": "https://localhost:8080/", "height": 117} outputId="67ef757a-91e9-4b32-c2ac-5f43c4742634"
df_item.loc[df_item['TITLE'] == 'GoldenEye (1995)']
```

```python id="Uom6EVG8gzLm" colab={"base_uri": "https://localhost:8080/"} outputId="a4f6c8a4-c472-48c2-f240-14763b5d1d8e"
# let's start by finding movies similar to GoldenEye (1995)
item_similarities = model.item_item_similarity(item_id=160)

item_similarities
```

```python id="MW741iOcgzLm" colab={"base_uri": "https://localhost:8080/", "height": 428} outputId="ad4b1617-bc85-4698-ebf6-65b70161e7ce"
df_item.iloc[item_similarities.index][:5]
```

<!-- #region id="_py39oQ8gzLm" -->
Unfortunately, not seen these movies. Can't say if these are relevant.

``item_item_similarity`` method is available in all Collie models, not just ``MatrixFactorizationModel``! 

Next, we will incorporate item metadata into recommendations for even better results.
<!-- #endregion -->

<!-- #region id="8hkoWyfVg9AK" -->
### Partial Credit Loss
Most of the time, we don't *only* have user-item interactions, but also side-data about our items that we are recommending. These next two notebooks will focus on incorporating this into the model training process. 

In this notebook, we're going to add a new component to our loss function - "partial credit". Specifically, we're going to use the genre information to give our model "partial credit" for predicting that a user would like a movie that they haven't interacted with, but is in the same genre as one that they liked. The goal is to help our model learn faster from these similarities. 
<!-- #endregion -->

<!-- #region id="4iFhjr7eg9AK" -->
### Read in Data
<!-- #endregion -->

<!-- #region id="bK4bGSUEWe9F" -->
To do the partial credit calculation, we need this data in a slightly different form. Instead of the one-hot-encoded version above, we're going to make a ``1 x n_items`` tensor with a number representing the first genre associated with the film, for simplicity. Note that with Collie, we could instead make a metadata tensor for each genre
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="mi7IZRHbWwsc" outputId="20906809-3682-4bee-eee3-1d98fe9b03ca"
df_item.columns[5:]
```

```python id="bWBxAUUXYvZ3"
metadata_df = df_item[df_item.columns[5:]]
```

```python id="LSy_-Jsxg9AL" colab={"base_uri": "https://localhost:8080/"} outputId="a1d9aa32-d10c-454a-dc7b-9cfcda410071"
genres = (
    torch.tensor(metadata_df.values)
    .topk(1)
    .indices
    .view(-1)
)

genres
```

<!-- #region id="LhaQLQQig9AM" -->
### Train a model with our new loss
<!-- #endregion -->

<!-- #region id="5NrYwTFVXCco" -->
now, we will pass in ``metadata_for_loss`` and ``metadata_for_loss_weights`` into the model ``metadata_for_loss`` should have a tensor containing the integer representations for metadata we created above for every item ID in our dataset ``metadata_for_loss_weights`` should have the weights for each of the keys in ``metadata_for_loss``
<!-- #endregion -->

```python id="Sysr04kSg9AN"
model = MatrixFactorizationModel(
    train=train_interactions,
    val=val_interactions,
    embedding_dim=10,
    lr=1e-2,
    metadata_for_loss={'genre': genres},
    metadata_for_loss_weights={'genre': 0.4},
)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 472, "referenced_widgets": ["7f9d29d6c1f04d4b9adcda6292a698fb", "4d95aa15fb8e45168224c12f033db89e", "567edd5b84854cc3ba9d8b34702716ce", "3ccc6ebee63d473aba28adc868b5b5c0", "5db3013f6fca47428b7f05d89d611441", "650154b034e843648fdd31198e6b5c8b", "b85c146244634db5a8ec5bb8c37a6a3f", "71e7704f60a44170988aed05dc8a4788", "71fb17475171409ab7b30f0915d3c3f9", "ec82867f02de401faf33bdeb69a1bf2e", "4ffc6e6cdbbb47239b7da5a07f16b5fb", "cce5557f779742e3932bb8ef2016a17c", "668011e6db324325974aa6a6b1fdc782", "ebcdd1df171847a38f7655716f2d6061", "9d11e361b37a4849a83531e249455615", "30fefc2060004b2bb028ed54ea9383b6", "85b96bc221984d08815cf3c09fde7c9f", "1abf2c736a91486eb8621d1cd679f9b2", "89b238d17aad4ef389c20de16ed1e55c", "70d5a3c3cf41478b83813ec40298cd22", "bdf376ea2f8e4b9e9d495a1fe44b2b59", "c6213cfdba3d4228aad387361e062fc7", "a28aa408fe444c179080e2ef9433a877", "6e3313408a184dcca85da1f7514d8bfe", "b6f3004c83574b379bc3520a410b5df5", "57e0d47b70b744898e4e548c28d27095", "d5f9ad23ba314382a2ffa29ecc311b5c", "75bcb22fb4514a288fdd43cdb2084b11", "d7bf1b026d10490fa70221af127986e5", "bef2ed39c203462a80a1f217adfc301b", "1038bcbdb10a4da08c7b2ffbc08c7e81", "2a435432b9914e3faf2d49a0645a9fd5", "127ad96af11c41519ded336bbc246c42", "b5fe20560cc64837af9e13936fd702d3", "2260ec6d57724ecca381bf65a151e97f", "add9b2d1bfeb4b3baaa97d39ca4eb8e3", "2baf39d33c2a4f28bfd557ba7b9f8c78", "bdf0ea6c2c8042d68c50cd5e19b1c15b", "143ecbd32aa14efeb1d2eae6873609ce", "b4eee80b60f748b4886ede6073b37a7c", "1997f9f7a4d04e5ebec4b8907e32b797", "21380c3a63df4a7592b0e580535603df", "0bd610aa3a55452dae1e3fbc721cb87a", "07e2308d6aff45c5bd0ad77e246e9981", "2354778e0ce6427385e2e107ea30332b", "717568c66bc444ddb60b415c5169588c", "13a5b3039ac04a2dbd6049992f822c89", "96773fc1d61d4b9faf8f8e7f0d0c3786", "2bb338780a1a40318d91fc689692c9ae", "869cd72009244cd997fc6b8bca19cd53", "8f341ddecce34a2e9967972e901123b6", "4f53235cc013444c8a71141df07ccc4b", "d76f878252a54a79bd98b1e093285964", "69b2d3fe2f5d4f73ba34142ccea3dbf8", "818427165e3c40d6b9c60f5fc26ea0db", "a62b8b75b5bb4b1a9a35a98b9f87edc0", "b066cece2abe4a74b1aed6c238cd82b3", "5a5b267987ed4ad0a143744fe30f27b1", "2666a159d7f1406887a6bff3efdd1dd6", "7b3b88973bc74728a3ea7eb04a122371", "7e684d1ba34a49e1b108909220e0a487", "a96a161717124b54ba9473f5d4476177", "2137633428de4deb8056902144e7551d", "2454cc03bb5342eca1ab02501d3ac39e", "2da2765da24841709884a7cca16f1107", "9d9f37ef8c9844a9ad5f4022ce904af3", "b7febc3da8714283b798aafaab3a8d8f", "f44ea695c0b2485faff83a6bbe12407e", "1e1888b6dc3e4c47aadf843a62f03233", "d75a669ab0a147ddb4185d6950c8267e", "a1eab9ad9a6147c2a3a13ed1d837f94b", "dd40e90d90a34827bb07d9d86c4234ce", "2391e70d0de74d71ab5372f39a4fa49b", "05fa6eea049e4b41a40604eed4c798d0", "e271494364c24f4e901c42b64e464056", "69b72f3ef4ae40a0a7b060be0849a354", "3e756777c21b4c56a06b537ca232ffc3", "0d4e9b5e9a3241dd920146b8be70d1ce", "d6e7bf43dc2346c0a2278d6b19078c90", "852593a171a048bc81b0cd3466366bd0", "13c2754aca2a42c2ac06d3640062b6d4", "1ac7aa00d4b24eccaf9b0cb622a65b05", "50691a81b511451aa9f78cbd0803a652", "43a743d7e9f14787a606cea5d0176931", "8561cb62512f47d9929c3abd557ef739", "52ac26d1722c40b68b18e0d92c6ac994", "cbf2e61a7dea4b91a8f86774bcfc8ff7", "8e58b5a80dca4cca8b03635c8d529072", "9adf92080dd2458a942dbd76a285ce48", "9a6a243411e84f049dfaeb99cccbc562", "5458638e910744509182e8c4ace883ec", "e692df9e6fb949c6a345dbaf9235d195", "7409891326ad4a259842211e306e980a", "0f57dd28b6954df693a160125075b506", "ae151ed064b547b6a8bca5181668f491", "5e8575da45094823bce0d16b9fe01f09"]} id="ZAk1C815g9AN" outputId="0c5600e3-d81d-4f6e-d621-c7882a767fb1"
trainer = CollieTrainer(model=model, max_epochs=10, deterministic=True)

trainer.fit(model)
```

<!-- #region id="NQdGTCPvg9AN" -->
### Evaluate the Model 
<!-- #endregion -->

<!-- #region id="ISQzTUnVg9AO" -->
Again, we'll evaluate the model and look at some particular users' recommendations to get a sense of what these recommendations look like using a partial credit loss function during model training. 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 117, "referenced_widgets": ["b1c126acba99422c9aad4bc9b1b0293f", "3e7789b77e3342acabd132d924906d5e", "a9db4a7b06d34e2eaee4933580e8e28e", "2e28af584886415c869f079215546e5e", "e078e83f8216456997e974acfc7f4816", "12b2182ba29547a8845685618988224e", "6d1261287d6042b486877c29f1056305", "6715a50cf1074e98b4e8e8f9aca4932e"]} id="6STWH4Ozg9AO" outputId="861755ec-75be-40fe-975b-17accf21477f"
mapk_score, mrr_score, auc_score = evaluate_in_batches([mapk, mrr, auc], val_interactions, model)

print(f'MAP@10 Score: {mapk_score}')
print(f'MRR Score:    {mrr_score}')
print(f'AUC Score:    {auc_score}')
```

<!-- #region id="Bk75mVQWg9AP" -->
Broken record alert: we're not seeing as much performance increase here compared to the standard model because MovieLens 100K has so few items. For a more dramatic difference, try training this model on a larger dataset, such as MovieLens 10M, adjusting the architecture-specific hyperparameters, or train longer. 
<!-- #endregion -->

<!-- #region id="9X25yfucbbKx" -->
### Inference
<!-- #endregion -->

```python id="dB6eeXWfg9AP" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="805ffd4f-38fd-4f4b-d9d1-b9ddfbdda60c"
user_id = np.random.randint(10, train_interactions.num_users)

display(
    HTML(
        get_recommendation_visualizations(
            model=model,
            user_id=user_id,
            filter_films=True,
            shuffle=True,
            detailed=True,
        )
    )
)
```

<!-- #region id="ZoIh0oHfg9AQ" -->
Partial credit loss is useful when we want an easy way to boost performance of any implicit model architecture, hybrid or not. When tuned properly, partial credit loss more fairly penalizes the model for more egregious mistakes and relaxes the loss applied when items are more similar. 

Of course, the loss function isn't the only place we can incorporate this metadata - we can also directly use this in the model (and even use a hybrid model combined with partial credit loss). Next, we will train a hybrid Collie model! 
<!-- #endregion -->

<!-- #region id="Laxa0vh1hE3o" -->
### Train a ``MatrixFactorizationModel`` 
<!-- #endregion -->

<!-- #region id="Fj3tJg-1hE3o" -->
The first step towards training a Collie Hybrid model is to train a regular ``MatrixFactorizationModel`` to generate rich user and item embeddings. We'll use these embeddings in a ``HybridPretrainedModel`` a bit later. 
<!-- #endregion -->

```python id="m75xWkQLhE3o"
model = MatrixFactorizationModel(
    train=train_interactions,
    val=val_interactions,
    embedding_dim=30,
    lr=1e-2,
)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 438, "referenced_widgets": ["62986af6e0394e969bb8942274ba6784", "4e51bc72f9fe41ca82096d01e26a5c23", "a3b4a6f432f7406b97c59bdd31eec848", "d6b7dc0b58874a93927b9182fe92ae63", "2528199682404110adb35e51bf1c39ff", "16dc9e31d8bb483faa575fd6db3915d7", "d9b41c7acb1741d98f5edf4e942e3b77", "7aabb759b56a4fc295687e0a24b6cb62", "b1ccac573c5c4c79b4357c14f99a08b6", "1c45f04d9cdc4876969a8a85d9087f18", "84065182993c4cdcbc3e5a37c769447b", "13ef2e644cae4a48ad8ab142e8b6435e", "191a266c1f034c9fa6e7b4138805d60c", "b8f64ba5faab4995a8b6f676b5a8e5d5", "22cb3e6197d34688914565f07d26ecef", "cc2768fdc75942c4a894bcf6006ad11f", "952b7b171ddb4f4eb96cc91f1257ca9a", "660daa560af241f2911156e2b7187c7e", "847fd69ae9924b03929b1a0e59e80f87", "e1cfc5099d1b4cc7bedcc7a53da34453", "9b62c6c326784d799f790398e7fa22e9", "65115b3c64504cc09ed3459b778995bf", "069b286ddde94f77a425f288e6e14d09", "f051d817821744d78a30577f69178e1e", "c0632ad487cf400e906da9a52efa239a", "12183c2f615f435a9cd920100f2820d4", "839eb1025574493ca1cf19f5a2f2010e", "ca5087b11f79495e8d32cfe8cca64f7b", "949ae8d78ed644cebf9b8fd86a6be7fc", "38f2d3b8e5ad41d1af054ef4c2ea47be", "195cc632420d4e85aadf55741dafe166", "e6ee962e992e44cca46b92e93b9f2c37", "b06e9ed8cacb428ca888b8c7e0558e54", "77d14fa2a785407181e4a9da9d40a3d0", "fd4cec7630304a159031e309fdbd715c", "cffb8b1a826b429488fc7d6af390df80", "c58a9370d7484bf8bf50c8e1fef3da12", "8d8c48b368c943aca716058d8a6c7485", "4ff6be2f8fe8422fa6ed67aa4e4bb2cc", "741722e693c344cca086aacee15eb874", "d227738e76a3420ba0a79a6684de0dcf", "0057a0667a434d1e8ac207b121049b2c", "c6316fd570b741628ea906f553c52679", "845c79c672984d0bab35df090dc632e5", "52d654208b6b4001a56c287a822354d3", "c47d7500fff14be39a2184e2d30b3795", "f6273966027146e9b345c02f935e65be", "72d633d88bd6413cad4cd3d2717dc7ca", "b423c474bb5d4ba98c14502c6b02d95d", "75241cec097b4fba877099eede78e3e7", "2cef863fdeb0413a8a3ce9bc68d1f985", "be5f89ff1dbc41b2a8263817d4a59f1f", "9459414718e44c6a9b88760f1fb1b46a", "1c8e2a3fde7942e697e111b07aefe26b", "637168904e464ac9a6782ada14784eb1", "a069be3b5a6d46e083524d14053bcb9f", "71907183c5584961bfd232d68e685d3d", "9e35548cd1714a0c9bf3ccc52d268fb1", "5a131f872f8d4fcb96b713b60e56656b", "7e72190f3fd44d1489badb8d28b6bb29", "82dc472697914b1aaf5866e1573c662c", "dd1af81809254587aa2a3d0deacff0a0", "b61d4544c6a54060a2dc120eee5362f3", "3b096679a00843148f3b874b9c70e901", "2e2eb1edad7748ee92249d555840d612", "8b2752894a8b4a69af0ad6be380c8c23", "09a7ea71de884bcc94aa330697ab728b", "29c1f761f5634be2bf0ff25736fd7d61", "be4983b296534b4ba8991262f81d6c0a", "058111735ae7416fab7f7adb8c981d14", "1e7ed32472134c508325266efe272c40", "d46475c0add24876b714a409bc9933b2", "51757d5291844c6981c0258a245ad5be", "36090e9a0cc94f56a377ff368eca2b64", "cb77a4c364704cdca658427246c1a3fb", "921e0906b18841d4b68b0b72dc911121", "86439582332e43cea3d1eea6062510f2", "edb0b8167858489c8f94c95337647d9f", "92b45c3467f44cdfb4b12fc7d9f25f5a", "1b13814df36f42969c8795c40b351ccf", "d1a11fbd7c3a4dadbdee14b744937c30", "248753435c404995812b93422ca8c062", "2b3ae44914934e31ab2229c3b7d1b9a7", "08c6b30a70024da28e546ff11dce4f94", "ce736cb44cec467e8d6db28528e30780", "d0ea236e247c46c78d6ee489433b76cf", "33e822baba3b4f0bbec237b48deba7f4", "fb8e91ab0228428487907e96827a91b9", "c7fec0dc1e8648ca869c65caefa05751", "acaa2bdd5eef41a49a99c02f384a8173", "2a62b154d02740988eaf6df9ce39cd93", "369e4fbc8a1b4c6d91f502a7fe5b18f4", "f2ff83d1d99d40c7bb9d563ad623e164", "e359213b91d841e3a28cdc293668a104", "86a3bee8f0604c9a922c94e4161fc24b", "f260359ad48e4f4c99e7aec1fc5cd09f"]} id="bjh0jE0yhE3p" outputId="34e09142-f66f-4d32-9052-5e41d3e7d166"
trainer = CollieTrainer(model=model, max_epochs=10, deterministic=True)

trainer.fit(model)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 117, "referenced_widgets": ["9b69ca9af02e4dcead166064746dfcb2", "1748fb86e5824e739a39350e4777a4eb", "27157458047e4a769ae205d23b8e6221", "2c7ac2735ed649cfafc52ca48eb38d1a", "ed61762550634ce6a906b6da4550f4c7", "b1ab812f3fae427db2b8cef8fe572f83", "08103e370a6047068d6e7d54653c1f3a", "1d603bf9ea684001b97a63be7f446e3b"]} id="6tvE66cfhE3p" outputId="1424a753-c468-48c2-dfc9-3b2118195955"
mapk_score, mrr_score, auc_score = evaluate_in_batches([mapk, mrr, auc], val_interactions, model)

print(f'Standard MAP@10 Score: {mapk_score}')
print(f'Standard MRR Score:    {mrr_score}')
print(f'Standard AUC Score:    {auc_score}')
```

<!-- #region id="CRq29RVfhE3q" -->
### Train a ``HybridPretrainedModel`` 
<!-- #endregion -->

<!-- #region id="lFh1LEcChE3q" -->
With our trained ``model`` above, we can now use these embeddings and additional side data directly in a hybrid model. The architecture essentially takes our user embedding, item embedding, and item metadata for each user-item interaction, concatenates them, and sends it through a simple feedforward network to output a recommendation score. 

We can initially freeze the user and item embeddings from our previously-trained ``model``, train for a few epochs only optimizing our newly-added linear layers, and then train a model with everything unfrozen at a lower learning rate. We will show this process below. 
<!-- #endregion -->

```python id="RPgUTdR1hE3r"
# we will apply a linear layer to the metadata with ``metadata_layers_dims`` and
# a linear layer to the combined embeddings and metadata data with ``combined_layers_dims``
hybrid_model = HybridPretrainedModel(
    train=train_interactions,
    val=val_interactions,
    item_metadata=metadata_df,
    trained_model=model,
    metadata_layers_dims=[8],
    combined_layers_dims=[16],
    lr=1e-2,
    freeze_embeddings=True,
)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 438, "referenced_widgets": ["1b0f046d63ec46909920000d416a956b", "2010b6b4500e4b139e93077008383a3c", "3a31cc9116d7465bbeea6471c6e93968", "560c2c1f4ca94b73a47aeb344ee81ab1", "cf1581e1b6614caa92b7845465c9de39", "346d17d614ee426fbe2be794a34737f3", "52271d9aa0d64d0dacb8d2d0d827adc0", "eb62d6f5333a4cad90fe00ecab4f4de3", "73d516b143a94e0a9ff2115a975dfe0b", "cabcae1934154c03a2ddbfa47b2b4501", "726b226d57474815ac786d93f4039902", "1a2825490a2043aca84c13d316d11ed2", "293790e04a5a44bb9f14edf7ac0ab8cc", "5ccacb85b1214ee1a0e5665364c235b7", "fb3d481dfea04ea3b6d1f8d57682826c", "044c05d2a4824890a595a2f196d64459", "ad5390af70b14da0aa0d3a0b5d20a90a", "0f93d0dc68984a279b6382fbedda6630", "e20d6099034e4567a726f6a7b038395c", "76f0fc24f91d40e4869be6e1b5b4b0d3", "a0666c429914487faf1fac59402e7364", "d6c7474eb9114cde9ff9adde036f3bc2", "83e1559c03c543c88418b270c16c88f6", "01c61537bfe842649a60c70fde9a51a3", "32b3c6d930f94d56b9f6e652303cc1ca", "27b7e22b72b94b9da9d22190c338ac84", "e39a8caba8854f5696d3ff03277bfaa0", "ad367116739e4f4c8717c5b5b741b2ee", "916dfdca43f74c058da5b45dff1eb621", "564e25c0cc9d4f8bb9a00e8f480eaf7b", "0cfb4e7dc348480d9cae85167df7a73e", "24f32e83468d4a4eb03b91130a526df5", "cf3dcff653a3443184d0942defe61230", "7f75f89935e84173bf5d914bc8f50150", "8c86d4c10827436b90b7f62885c57ef6", "3971fc6570fc4f3ca85df75030b1a984", "b5be05a8a7c045c1bce4eda3852a9256", "5802083d9af4473684eadd22aa17d5c0", "00a30c0b9eac4a38b0bd3be2cc05476b", "dc0817dca939450caa82a751377451d4", "aa23f91202f7432ab937a60fd0cc69c2", "fa12cd0edffc4459aadb255b2f85309c", "cfa25ee4254c479498e117189651d8c4", "7e6c87609e254706b7e6ee6ac7bf78b4", "a50062a20b2a475fb760c5ddf6266197", "82c055fff9914a1d8a6c838062046b99", "c16130a0a9624f848af1b5553264b78d", "1f2cb0a8875946439822315ad3a7e1e5", "22fbff68d90c43d6960fea7ddd420810", "6f02c441db1a45488e41b29c656de74e", "1408cb730c964bd39fd32f56916b9701", "095d3de7d1e1464eb81d927e6edc3b0c", "e5379796148347bc8f6ab506e1f5b1cd", "bb09a545620c4dd1af632339d6fdd04a", "08ee77ef97744d6d92a86674b9247b25", "ef2fb16bb26f4a489fe456a33a9aaabf", "c4588fb79e384aeabc04a175fe8d6548", "d519f414b012498bafdd6ebd8503bbf0", "a571a202b33d4a0dae7428d244d64cf6", "f8841a4feab24ea1ace015833bbf0891", "e0eb2bb0242e44aba710350121762fad", "2cbdcb710a39477494596ee8e941152a", "7ef288048daf401699f47fcf24ae2b2a", "7927e1af2f644102b87c7c13ad22546a", "c560a29789ca48129d338f88feddfeee", "a2dc9ff9d3b6413489e37645e5a81969", "e23fffe8236f4f9ab8b76858b371ed8c", "7a1cdaf584e44e3392d963cd39f65cc0", "0303b557637e45078eb791c8e6091f61", "7025cdb08e7646169e13713fc442b5e3", "c3216b05839d47d781220cc4dc762e46", "41b780dbe1394396a24bb180a7945c3d", "c58dc7499eef43f39a406d873aa9bf2f", "3a7fe9e14ff94e14a3ea9be6a3685299", "65d8dc2103ba4794a94ddbf4e1ef9825", "16a6b8dc6f36458da47ce830fac15d26", "5be8f16d753c47a6a68248c561a9605e", "c85c795ddd734662b1277c3b4dd4bb6f", "37aacb2f901c46c4bc7b9c580890a3c6", "f837d58642d247cf9ffb657002d83ae6", "109fb55f40334538966bbdb2961cfdf9", "1a1fa94f37524d4cb93f5a2bfde53b12", "7978b82dcd874c2782f4821d859ef918", "e4f3bb71ec4645649a3e81bfd8b3e310", "0374e23a55c249bb955a605926214583", "14acab5f63754fd0970ca0a6e204d47e", "720887ecd42045539a4f03a57b8a2d5e", "94f0ec5f8e5e43de8397102e899dde43", "fcbf9f9aecf54c1f97c3e2e3915c36c2", "2b71910b6bf74406a468ef78e914db9f", "7bfe5a6f9f2f4787acb15eedd6e23341", "365f4e59afcb443f8c3cad99137dcd40", "d9068f85980247bc89628635dd221f3c", "c98453288d7e416c8ae6067db3220b5f", "6657854d04e446339437228998a63325", "27a1d1d028474aab8ee3cdc1c00166ad"]} id="vyyUg5ilhE3r" outputId="77ffdf06-0964-4a7d-f491-a83d53d3e210"
hybrid_trainer = CollieTrainer(model=hybrid_model, max_epochs=10, deterministic=True)

hybrid_trainer.fit(hybrid_model)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 117, "referenced_widgets": ["f3842767c3b9491ca0e39c7d83c3bba7", "42e789d0121c43df87d03eeb4e58d3fe", "bdebaa528fe345f58ed0e85d0015187b", "08455bd7b312416a8828902e10a0a6de", "4a1d091431614ff2a9a94baa79f4ebdd", "78f224d19e054aa3a6787b9c3aa536eb", "b6cff5f6b6004d6999a277ef1ed64fe1", "9c1fd029256144a68b8c996e201c3e08"]} id="I8eEYwcfhE3s" outputId="9babb089-060c-4636-f3dd-ebbf91d939e6"
mapk_score, mrr_score, auc_score = evaluate_in_batches([mapk, mrr, auc], val_interactions, hybrid_model)

print(f'Hybrid MAP@10 Score: {mapk_score}')
print(f'Hybrid MRR Score:    {mrr_score}')
print(f'Hybrid AUC Score:    {auc_score}')
```

```python id="EEw83cTUhE3s"
hybrid_model_unfrozen = HybridPretrainedModel(
    train=train_interactions,
    val=val_interactions,
    item_metadata=metadata_df,
    trained_model=model,
    metadata_layers_dims=[8],
    combined_layers_dims=[16],
    lr=1e-4,
    freeze_embeddings=False,
)

hybrid_model.unfreeze_embeddings()
hybrid_model_unfrozen.load_from_hybrid_model(hybrid_model)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 472, "referenced_widgets": ["b776627648254a919b1a18b4025fbcb3", "2b4a275a432241ec91b5fe2a67fff267", "b9fed3cbac434c5e8e8eccc37fdc7909", "518844d0b3cc4deabe8b0ebb657eac8f", "20a1726ffbf9408b8a00ccd5c44b81a8", "f9e3ccff7f41406eb5ec4f9658b24718", "d354c4d6b73d412f87a41f8cbb7bb9eb", "01f11efae5cc4bc48726f6554c346655", "bede57139cc14b599167088250548b43", "512af7cae0c84552b736431802ed4402", "2a5f13e1cce34787ba032800b8a59ef6", "2dd137e524324334b5f7c56e7d4d8877", "e7f0ad3f8d204c1a9c611a897655abbe", "dbaaabf67d1640d5b988ca6e095577c2", "9814e7c9d49245e7802828d385800a6d", "7efc774340274141a6a6d4c3e5bfcf12", "fe7f069fe1f94fe3a98c44c3cf5c3ed1", "82de05cc4d4f4b32bf7127007ba17bef", "3e5ca461bd1f4d6ba6e3f5d2a5c708b4", "4d565f9c62d84ba487c79a1e2053770d", "7beb2d7bb4b8421fa6a3fdc5a0a578b1", "b56b52a8d05c4c2496e18f32df63ee5b", "55a7ecf02b1f492e82b95e5460e04f22", "7d93dfd6d22843108d76c8a2416bee21", "3b0d7093adde4f64bfddffdb4444e1f2", "aa9c6996aa3c40d18f0b3f0b8cd1705d", "7dc6a6399f604fef8dda1b5a1d7b2920", "c1ab4cda390d4a99b92162421e86718a", "c894ed9774704d88bff3e9d1ad542900", "4968d6ba2303488c9256042f3a7f8206", "27da441d4d55492ebc526ca00dd7d01e", "ad048e4075c847f1b911080e51548cdf", "84cd0ec866694431a1bc3f6eb7686107", "8b4924ef271d4097abd6e57303794327", "200eca6c62424921ab682d2ab8a0785d", "80117a933a694fd9914f88917466a00f", "47c26c18cccf44f4b6a5caf3ccdd4e83", "4fdf8e0a36244ff1bdf34e9060e3f035", "7478d08325084ef28fa9f1c5a6ab18f6", "1b0c6cff674c4930b18748d9fa4f9090", "73232374fc7f4880a87dec552959d3a4", "03dad58a3f004920aff305a2357ad121", "fe02f58536de4d49b132a067fc065671", "7805d39a07414d199a012bad80e90acb", "094a99863ecc425dae15237f990ffb3e", "b91817bbe07449b2a9a8d1b6be2ce378", "82c6bf962cd04ee8bf02d24b03952b28", "3e392b5c457a4ef2b7c883982f60c36b", "5ebee37d75004546b316d10399b69431", "b7fbdb82ca614754a12635170518e0cb", "6cb3f43666734c869fcc960645e129e9", "ad4885e1e6ab41f48ac113c30aa13b39", "87093d6b3c924e6087e9d2b78ba0c6eb", "0a4702427f524e768b8d1b2379e65499", "3c9b57c8d7e24cb181e50d3b5e9a89d4", "1f6a03090e2f4bd5a08b973a2e31a48e", "43ff46cfff674d37a04bf926feca9048", "7fb80bf0ab9549989de36323648126cd", "34c2e6b19152468e8e8bbdcbf1e7d87e", "521b635587644d588e28f0efba61aea2", "6408b9869970482d949d6a794700716b", "61a742fb4fde49fb9e4026e330cf2159", "05bdc0f323064d359a32a0b8d345dd78", "424e11f8cff442ee8a7643e56ffb36af", "2ac6cf2e1f304f06bdc354d04507fecd", "44ac79c4dfec4caa9bd2e4f987aeca9d", "ebe6a36807834fb38ff46654e27075c1", "ac257e025a9545d5b924d2946b422735", "4606964ad6b5447db1d7498178cb5a78", "e823a52deb434d5f81deb90ae34adc5e", "9dc20f131e8642ddae5a90537309d835", "44eea8a841af4ddb92d93bfe06c4c6fc", "5a147c6e4b59428ebd7e2e3412cc52fc", "49bf717484f84a25a7ac27b01b2606a3", "c91bb1aa56074b398362c4326c9b9b13", "1aad34c1efeb4af6a9ec6809a12a3569", "7214617bec6645ba891a2071f6bc6442", "ab827148c2884c728fe50dd09ce55912", "d48633090c514ddd9e9fc2baa4bf347d", "6315ef0a9eb14469a4de8063b9e745ca", "afe817a0babe466cbbf8e4b802ef3360", "60f26b29709e462381c221393c45e76b", "cd7d3ce3534447b98fafe4d5aab0194b", "f44f6872eb08434ea56d62708770cd25", "cb38f538344b43928074a1184cd05997", "286f2daf07f745d891110f78c116823c", "c456a6bba5804e8e911a89abf34e5670", "b4b847b559944ce2ba7a4ce34c472120", "975f803a94994f91891c3fb7f187a135", "47d4f8168a3049b09471968aca76fa08", "0af18a7ade304878bdcb8dbca2ef1074", "a521fbb49c5946b1afca37cbc4052b52", "dac2cbf4f3f2477ab18adce6db8e77a8", "5e9949db3b97478a905b23c0a437dd45", "544a63b30c714580be37d69eb8669328", "33fbfb7b8c2d4a5bb0e979c626862b13"]} id="yiA-EylqhE3t" outputId="aefcb665-c88d-43ab-eb7a-322cbc58a262"
hybrid_trainer_unfrozen = CollieTrainer(model=hybrid_model_unfrozen, max_epochs=10, deterministic=True)

hybrid_trainer_unfrozen.fit(hybrid_model_unfrozen)
```

<!-- #region id="2Txbqf3fbqvD" -->
### Evaluate the Model
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 117, "referenced_widgets": ["0d9f2c4528634f63a26527a755b33773", "6d196fda01ee4b63a7731b169c2f36b7", "20d9caea7c744c09a8efd05793d2c6db", "1760eeca2a084b1c8e57a9989139cdf6", "8ccaa38836fd4fa588723ba2516f635b", "113a8ae2c462494abf3ca97f65e51d06", "c0282a49c81f4dce83322f9734eb0efd", "9848dcb0b0e048c88f9c6243dc5ede33"]} id="sof4rqMbhE3u" outputId="a344f1bb-0627-4e46-968e-f89bc81a5224"
mapk_score, mrr_score, auc_score = evaluate_in_batches([mapk, mrr, auc],
                                                       val_interactions,
                                                       hybrid_model_unfrozen)

print(f'Hybrid Unfrozen MAP@10 Score: {mapk_score}')
print(f'Hybrid Unfrozen MRR Score:    {mrr_score}')
print(f'Hybrid Unfrozen AUC Score:    {auc_score}')
```

<!-- #region id="0FzTQc6WbtJA" -->
### Inference
<!-- #endregion -->

```python id="EwM1pkf_hE3v" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="650e4d07-8566-484c-c7b7-543e7c7428a9"
user_id = np.random.randint(10, train_interactions.num_users)

display(
    HTML(
        get_recommendation_visualizations(
            model=hybrid_model_unfrozen,
            user_id=user_id,
            filter_films=True,
            shuffle=True,
            detailed=True,
        )
    )
)
```

<!-- #region id="fNwj-u-AhE3w" -->
The metrics and results look great, and we should only see a larger difference compared to a standard model as our data becomes more nuanced and complex (such as with MovieLens 10M data). 

If we're happy with this model, we can go ahead and save it for later! 
<!-- #endregion -->

<!-- #region id="xYmFQZEhhE3w" -->
### Save and Load a Hybrid Model 
<!-- #endregion -->

```python id="2ZDlfmAVhE3w"
# we can save the model with...
os.makedirs('models', exist_ok=True)
hybrid_model_unfrozen.save_model('models/hybrid_model_unfrozen')
```

```python id="qW3kPpenhE3x" colab={"base_uri": "https://localhost:8080/"} outputId="da7c6d72-d7ca-4913-98fc-e85691a771f4"
# ... and if we wanted to load that model back in, we can do that easily...
hybrid_model_loaded_in = HybridPretrainedModel(load_model_path='models/hybrid_model_unfrozen')


hybrid_model_loaded_in
```

<!-- #region id="qKn2XvzuSaqU" -->
## Yet another Movie Recommender from scratch
> Building and training Item-popularity and MLP model on movielens dataset in pure pytorch.
<!-- #endregion -->

<!-- #region id="GUoQMgjCPmkp" -->
### Setup
<!-- #endregion -->

```python id="Q6wvep55K6of"
import math
import torch
import heapq
import pickle
import argparse
import numpy as np
import pandas as pd
from torch import nn
import seaborn as sns
from time import time
import scipy.sparse as sp
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
```

```python id="jz4ocKNnLN4P"
np.random.seed(7)
torch.manual_seed(0)

_model = None
_testRatings = None
_testNegatives = None
_topk = None

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
```

<!-- #region id="srnZMdMoPh9V" -->
### Data Loading
<!-- #endregion -->

```python id="J52BdmTvKvUv"
!wget https://github.com/HarshdeepGupta/recommender_pytorch/raw/master/Data/movielens.train.rating
!wget https://github.com/HarshdeepGupta/recommender_pytorch/raw/master/Data/movielens.test.rating
!wget https://github.com/HarshdeepGupta/recommender_pytorch/raw/master/Data/u.data
```

<!-- #region id="csjr7o5wPd7n" -->
### Eval Methods
<!-- #endregion -->

```python id="et-6h-pkLLMk"
def evaluate_model(model, full_dataset: MovieLensDataset, topK: int):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _topk
    _model = model
    _testRatings = full_dataset.testRatings
    _testNegatives = full_dataset.testNegatives
    _topk = topK

    hits, ndcgs = [], []
    for idx in range(len(_testRatings)):
        (hr, ndcg) = eval_one_rating(idx, full_dataset)
        hits.append(hr)
        ndcgs.append(ndcg)
    return (hits, ndcgs)


def eval_one_rating(idx, full_dataset: MovieLensDataset):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]

    gtItem = rating[1]
    items.append(gtItem)
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype='int32')

    feed_dict = {
        'user_id': users,
        'item_id': np.array(items),
    }
    predictions = _model.predict(feed_dict)
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]

    # Evaluate top rank list
    ranklist = heapq.nlargest(_topk, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)
```

<!-- #region id="SvpXLWBpPYKk" -->
### Eval Metrics
<!-- #endregion -->

```python id="vvU46669Mnmz"
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

<!-- #region id="Rv_-b2rnPQXI" -->
### Pytorch Dataset
<!-- #endregion -->

```python id="grg5RywRK1H8"
class MovieLensDataset(Dataset):
    'Characterizes the dataset for PyTorch, and feeds the (user,item) pairs for training'

    def __init__(self, file_name, num_negatives_train=5, num_negatives_test=100):
        'Load the datasets from disk, and store them in appropriate structures'

        self.trainMatrix = self.load_rating_file_as_matrix(
            file_name + ".train.rating")
        self.num_users, self.num_items = self.trainMatrix.shape
        # make training set with negative sampling
        self.user_input, self.item_input, self.ratings = self.get_train_instances(
            self.trainMatrix, num_negatives_train)
        # make testing set with negative sampling
        self.testRatings = self.load_rating_file_as_list(
            file_name + ".test.rating")
        self.testNegatives = self.create_negative_file(
            num_samples=num_negatives_test)
        assert len(self.testRatings) == len(self.testNegatives)

    def __len__(self):
        'Denotes the total number of rating in test set'
        return len(self.user_input)

    def __getitem__(self, index):
        'Generates one sample of data'

        # get the train data
        user_id = self.user_input[index]
        item_id = self.item_input[index]
        rating = self.ratings[index]

        return {'user_id': user_id,
                'item_id': item_id,
                'rating': rating}

    def get_train_instances(self, train, num_negatives):
        user_input, item_input, ratings = [], [], []
        num_users, num_items = train.shape
        for (u, i) in train.keys():
            # positive instance
            user_input.append(u)
            item_input.append(i)
            ratings.append(1)
            # negative instances
            for _ in range(num_negatives):
                j = np.random.randint(1, num_items)
                # while train.has_key((u, j)):
                while (u, j) in train:
                    j = np.random.randint(1, num_items)
                user_input.append(u)
                item_input.append(j)
                ratings.append(0)
        return user_input, item_input, ratings

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

    def create_negative_file(self, num_samples=100):
        negativeList = []
        for user_item_pair in self.testRatings:
            user = user_item_pair[0]
            item = user_item_pair[1]
            negatives = []
            for t in range(num_samples):
                j = np.random.randint(1, self.num_items)
                while (user, j) in self.trainMatrix or j == item:
                    j = np.random.randint(1, self.num_items)
                negatives.append(j)
            negativeList.append(negatives)
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

<!-- #region id="M_CJ2wlKPS-w" -->
### Utils
<!-- #endregion -->

```python id="YhQs5tfvK9Pf"
def train_one_epoch(model, data_loader, loss_fn, optimizer, epoch_no, device, verbose = 1):
    'trains the model for one epoch and returns the loss'
    print("Epoch = {}".format(epoch_no))
    # Training
    # get user, item and rating data
    t1 = time()
    epoch_loss = []
    # put the model in train mode before training
    model.train()
    # transfer the data to GPU
    for feed_dict in data_loader:
        for key in feed_dict:
            if type(feed_dict[key]) != type(None):
                feed_dict[key] = feed_dict[key].to(dtype = torch.long, device = device)
        # get the predictions
        prediction = model(feed_dict)
        # print(prediction.shape)
        # get the actual targets
        rating = feed_dict['rating']
        
      
        # convert to float and change dim from [batch_size] to [batch_size,1]
        rating = rating.float().view(prediction.size())  
        loss = loss_fn(prediction, rating)
        # clear the gradients
        optimizer.zero_grad()
        # backpropagate
        loss.backward()
        # update weights
        optimizer.step()
        # accumulate the loss for monitoring
        epoch_loss.append(loss.item())
    epoch_loss = np.mean(epoch_loss)
    if verbose:
        print("Epoch completed {:.1f} s".format(time() - t1))
        print("Train Loss: {}".format(epoch_loss))
    return epoch_loss
        

def test(model, full_dataset : MovieLensDataset, topK):
    'Test the HR and NDCG for the model @topK'
    # put the model in eval mode before testing
    if hasattr(model,'eval'):
        # print("Putting the model in eval mode")
        model.eval()
    t1 = time()
    (hits, ndcgs) = evaluate_model(model, full_dataset, topK)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Eval: HR = %.4f, NDCG = %.4f [%.1f s]' % (hr, ndcg, time()-t1))
    return hr, ndcg
    

def plot_statistics(hr_list, ndcg_list, loss_list, model_alias, path):
    'plots and saves the figures to a local directory'
    plt.figure()
    hr = np.vstack([np.arange(len(hr_list)),np.array(hr_list)]).T
    ndcg = np.vstack([np.arange(len(ndcg_list)),np.array(ndcg_list)]).T
    loss = np.vstack([np.arange(len(loss_list)),np.array(loss_list)]).T
    plt.plot(hr[:,0], hr[:,1],linestyle='-', marker='o', label = "HR")
    plt.plot(ndcg[:,0], ndcg[:,1],linestyle='-', marker='v', label = "NDCG")
    plt.plot(loss[:,0], loss[:,1],linestyle='-', marker='s', label = "Loss")

    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(path+model_alias+".jpg")
    return


def get_items_interacted(user_id, interaction_df):
    # returns a set of items the user has interacted with
    userid_mask = interaction_df['userid'] == user_id
    interacted_items = interaction_df.loc[userid_mask].courseid
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])


def save_to_csv(df,path, header = False, index = False, sep = '\t', verbose = False):
    if verbose:
        print("Saving df to path: {}".format(path))
        print("Columns in df are: {}".format(df.columns.tolist()))

    df.to_csv(path, header = header, index = index, sep = sep)
```

<!-- #region id="qEl945qyM1FB" -->
### Item Popularity Model
<!-- #endregion -->

```python id="dzjxYN-mM3uv"
def parse_args():
    parser = argparse.ArgumentParser(description="Run ItemPop")
    parser.add_argument('--path', nargs='?', default='/content/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='movielens',
                        help='Choose a dataset.')
    parser.add_argument('--num_neg_test', type=int, default=100,
                        help='Number of negative instances to pair with a positive instance while testing')
    
    return parser.parse_args(args={})
```

```python id="xAr3XZ4sM73x"
class ItemPop():
    def __init__(self, train_interaction_matrix: sp.dok_matrix):
        """
        Simple popularity based recommender system
        """
        self.__alias__ = "Item Popularity without metadata"
        # Sum the occurences of each item to get is popularity, convert to array and 
        # lose the extra dimension
        self.item_ratings = np.array(train_interaction_matrix.sum(axis=0, dtype=int)).flatten()

    def forward(self):
        pass

    def predict(self, feeddict) -> np.array:
        # returns the prediction score for each (user,item) pair in the input
        items = feeddict['item_id']
        output_scores = [self.item_ratings[itemid] for itemid in items]
        return np.array(output_scores)

    def get_alias(self):
        return self.__alias__
```

```python colab={"base_uri": "https://localhost:8080/"} id="0EDavlooLWT0" outputId="a1f13e86-030e-4530-e2d9-34b0d676570a"
args = parse_args()
path = args.path
dataset = args.dataset
num_negatives_test = args.num_neg_test
print("Model arguments: %s " %(args))

topK = 10

# Load data

t1 = time()
full_dataset = MovieLensDataset(path + dataset, num_negatives_test=num_negatives_test)
train, testRatings, testNegatives = full_dataset.trainMatrix, full_dataset.testRatings, full_dataset.testNegatives
num_users, num_items = train.shape
print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
      % (time()-t1, num_users, num_items, train.nnz, len(testRatings)))

model = ItemPop(train)
test(model, full_dataset, topK)
```

<!-- #region id="IaJQ8h8kNWmr" -->
### MLP Model
<!-- #endregion -->

```python id="F_GYre42NhDX"
def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--path', nargs='?', default='/content/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='movielens',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[16,32,16,8]',
                        help="Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        help="Regularization for each layer")
    parser.add_argument('--num_neg_train', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance while training')
    parser.add_argument('--num_neg_test', type=int, default=100,
                        help='Number of negative instances to pair with a positive instance while testing')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--dropout', type=float, default=0,
                        help='Add dropout layer after each dense layer, with p = dropout_prob')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args(args={})
```

```python id="BJVmkqWGNoXM"
class MLP(nn.Module):

    def __init__(self, n_users, n_items, layers=[16, 8], dropout=False):
        """
        Simple Feedforward network with Embeddings for users and items
        """
        super().__init__()
        assert (layers[0] % 2 == 0), "layers[0] must be an even number"
        self.__alias__ = "MLP {}".format(layers)
        self.__dropout__ = dropout

        # user and item embedding layers
        embedding_dim = int(layers[0]/2)
        self.user_embedding = torch.nn.Embedding(n_users, embedding_dim)
        self.item_embedding = torch.nn.Embedding(n_items, embedding_dim)

        # list of weight matrices
        self.fc_layers = torch.nn.ModuleList()
        # hidden dense layers
        for _, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
        # final prediction layer
        self.output_layer = torch.nn.Linear(layers[-1], 1)

    def forward(self, feed_dict):
        users = feed_dict['user_id']
        items = feed_dict['item_id']
        user_embedding = self.user_embedding(users)
        item_embedding = self.item_embedding(items)
        # concatenate user and item embeddings to form input
        x = torch.cat([user_embedding, item_embedding], 1)
        for idx, _ in enumerate(range(len(self.fc_layers))):
            x = self.fc_layers[idx](x)
            x = F.relu(x)
            x = F.dropout(x,  p=self.__dropout__, training=self.training)
        logit = self.output_layer(x)
        rating = torch.sigmoid(logit)
        return rating

    def predict(self, feed_dict):
        # return the score, inputs and outputs are numpy arrays
        for key in feed_dict:
            if type(feed_dict[key]) != type(None):
                feed_dict[key] = torch.from_numpy(
                    feed_dict[key]).to(dtype=torch.long, device=device)
        output_scores = self.forward(feed_dict)
        return output_scores.cpu().detach().numpy()

    def get_alias(self):
        return self.__alias__
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="lA9s7rv0LspV" outputId="57d64091-be5b-493d-d20a-5f8d991e69ef"
print("Device available: {}".format(device))

args = parse_args()
path = args.path
dataset = args.dataset
layers = eval(args.layers)
weight_decay = args.weight_decay
num_negatives_train = args.num_neg_train
num_negatives_test = args.num_neg_test
dropout = args.dropout
learner = args.learner
learning_rate = args.lr
batch_size = args.batch_size
epochs = args.epochs
verbose = args.verbose

topK = 10
print("MLP arguments: %s " % (args))
model_out_file = '%s_MLP_%s_%d.h5' %(args.dataset, args.layers, time())

# Load data

t1 = time()
full_dataset = MovieLensDataset(
    path + dataset, num_negatives_train=num_negatives_train, num_negatives_test=num_negatives_test)
train, testRatings, testNegatives = full_dataset.trainMatrix, full_dataset.testRatings, full_dataset.testNegatives
num_users, num_items = train.shape
print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
      % (time()-t1, num_users, num_items, train.nnz, len(testRatings)))

training_data_generator = DataLoader(
    full_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Build model
model = MLP(num_users, num_items, layers=layers, dropout=dropout)
# Transfer the model to GPU, if one is available
model.to(device)
if verbose:
    print(model)

loss_fn = torch.nn.BCELoss()
# Use Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay)

# Record performance
hr_list = []
ndcg_list = []
BCE_loss_list = []

# Check Init performance
hr, ndcg = test(model, full_dataset, topK)
hr_list.append(hr)
ndcg_list.append(ndcg)
BCE_loss_list.append(1)

# do the epochs now

for epoch in range(epochs):
    epoch_loss = train_one_epoch( model, training_data_generator, loss_fn, optimizer, epoch, device)

    if epoch % verbose == 0:
        hr, ndcg = test(model, full_dataset, topK)
        hr_list.append(hr)
        ndcg_list.append(ndcg)
        BCE_loss_list.append(epoch_loss)
        if hr > max(hr_list):
            if args.out > 0:
                model.save(model_out_file, overwrite=True)

print("hr for epochs: ", hr_list)
print("ndcg for epochs: ", ndcg_list)
print("loss for epochs: ", BCE_loss_list)
plot_statistics(hr_list, ndcg_list, BCE_loss_list, model.get_alias(), "/content")
with open("metrics", 'wb') as fp:
    pickle.dump(hr_list, fp)
    pickle.dump(ndcg_list, fp)

best_iter = np.argmax(np.array(hr_list))
best_hr = hr_list[best_iter]
best_ndcg = ndcg_list[best_iter]
print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %
      (best_iter, best_hr, best_ndcg))
if args.out > 0:
    print("The best MLP model is saved to %s" %(model_out_file))
```

<!-- #region id="1rDDCMkehE3x" -->
Thus far, we keep our focus only on the implicit feedback based matrix factorization model on small movielens dataset. In future, we will be expanding this MVP in the following directions:
1. Large scale industrial datasets - Yoochoose, Trivago
2. Other available models in [this](https://github.com/ShopRunner/collie_recs/tree/main/collie_recs/model) repo
3. Really liked the poster carousel. Put it in dash/streamlit app.
<!-- #endregion -->

<!-- #region id="thC-jHYLJKkz" -->
## Training neural factorization model on movielens dataset
> Training MF, MF+bias, and MLP model on movielens-100k dataset in PyTorch.
<!-- #endregion -->

```python id="U9XYsONJClRh"
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

<!-- #region id="UxWgUAp7vnIM" -->
## Neural Matrix Factorization on Movielens
> Experiments with different variations of Neural matrix factorization model in PyTorch on movielens dataset.
<!-- #endregion -->

```python id="hfac3W-Z4yEs"
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="74OaJCfn5H4Z" outputId="9ec26cd4-a004-49cf-aefb-9a24c670bf11"
data = pd.read_csv("https://raw.githubusercontent.com/sparsh-ai/reco-data/master/MovieLens_LatestSmall_ratings.csv")
data.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="F0Irmpk2oUna" outputId="082a3deb-3f36-4d93-8917-77c27db5fc55"
data.shape
```

<!-- #region id="AsSMbrTG6LQr" -->
Data encoding
<!-- #endregion -->

```python id="m_wEgrHx5U93"
np.random.seed(3)
msk = np.random.rand(len(data)) < 0.8
train = data[msk].copy()
valid = data[~msk].copy()
```

```python id="fmeQCQXP6Vtv"
# here is a handy function modified from fast.ai
def proc_col(col, train_col=None):
    """Encodes a pandas column with continous ids. 
    """
    if train_col is not None:
        uniq = train_col.unique()
    else:
        uniq = col.unique()
    name2idx = {o:i for i,o in enumerate(uniq)}
    return name2idx, np.array([name2idx.get(x, -1) for x in col]), len(uniq)
```

```python id="OUfpCvFJ6W72"
def encode_data(df, train=None):
    """ Encodes rating data with continous user and movie ids. 
    If train is provided, encodes df with the same encoding as train.
    """
    df = df.copy()
    for col_name in ["userId", "movieId"]:
        train_col = None
        if train is not None:
            train_col = train[col_name]
        _,col,_ = proc_col(df[col_name], train_col)
        df[col_name] = col
        df = df[df[col_name] >= 0]
    return df
```

```python id="TZDUY2rt6Z9B"
# encoding the train and validation data
df_train = encode_data(train)
df_valid = encode_data(valid, train)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="c9VIAfR6otTe" outputId="dc5ad891-2004-4fda-acfa-22d04525df3b"
df_train.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="iy77qSeCo2WY" outputId="c4ce1748-6c39-44f6-c094-476873135fdb"
df_train.shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="sOSEDMQPo2Tq" outputId="4327e4a5-b01b-4d03-c829-0220e7c8b36c"
df_valid.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="2uiymy6po2Lo" outputId="f1ebfdf2-a139-46d4-d8e7-850701cb57a2"
df_valid.shape
```

<!-- #region id="y3QSDyZj61Iy" -->
Matrix factorization model
<!-- #endregion -->

```python id="HBPnUZl-6z1g"
class MF(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100):
        super(MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.user_emb.weight.data.uniform_(0, 0.05)
        self.item_emb.weight.data.uniform_(0, 0.05)
        
    def forward(self, u, v):
        u = self.user_emb(u)
        v = self.item_emb(v)
        return (u*v).sum(1)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="dBQhfy2l7AAn" outputId="d667d3a3-2baa-467b-e8ab-905c3282780f"
# unit testing the architecture
sample = encode_data(train.sample(5))
display(sample)
```

```python id="9tmmtDTuqnIB"
num_users = 5
num_items = 5
emb_size = 3

user_emb = nn.Embedding(num_users, emb_size)
item_emb = nn.Embedding(num_items, emb_size)

users = torch.LongTensor(sample.userId.values)
items = torch.LongTensor(sample.movieId.values)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 102} id="H557JwqSqimK" outputId="b45ad07e-0cf7-4cb0-f9ad-5de2f8a86c08"
U = user_emb(users)
display(U)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 102} id="IsdtlmtBq3cj" outputId="418f529d-1481-475e-ad2c-39578baa4cdf"
V = item_emb(items)
display(V)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 102} id="TKtMvkNfq0q2" outputId="c7bb19a7-3a7b-45db-f4c1-b95f0994750f"
display(U*V) # element wise multiplication
```

```python colab={"base_uri": "https://localhost:8080/", "height": 34} id="5_AN_dhQq0nE" outputId="7cc9d053-c9f3-49d0-e2a6-f0ea809ecd8f"
display((U*V).sum(1))
```

<!-- #region id="W01e58dr86WY" -->
Model training
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="VC5vARcP7QAc" outputId="b4cea803-bbac-4301-bb1c-12683689948d"
num_users = len(df_train.userId.unique())
num_items = len(df_train.movieId.unique())
print("{} users and {} items in the training set".format(num_users, num_items))
```

```python id="yRi5sy-K8-fr"
model = MF(num_users, num_items, emb_size=100) # .cuda() if you have a GPU
```

```python id="4nAGJ4l08_83"
def train_epocs(model, epochs=10, lr=0.01, wd=0.0, unsqueeze=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model.train()
    for i in range(epochs):
        users = torch.LongTensor(df_train.userId.values) # .cuda()
        items = torch.LongTensor(df_train.movieId.values) #.cuda()
        ratings = torch.FloatTensor(df_train.rating.values) #.cuda()
        if unsqueeze:
            ratings = ratings.unsqueeze(1)
        y_hat = model(users, items)
        loss = F.mse_loss(y_hat, ratings)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item()) 
    test_loss(model, unsqueeze)
```

```python id="7l_3G5gn9GH3"
def test_loss(model, unsqueeze=False):
    model.eval()
    users = torch.LongTensor(df_valid.userId.values) #.cuda()
    items = torch.LongTensor(df_valid.movieId.values) #.cuda()
    ratings = torch.FloatTensor(df_valid.rating.values) #.cuda()
    if unsqueeze:
        ratings = ratings.unsqueeze(1)
    y_hat = model(users, items)
    loss = F.mse_loss(y_hat, ratings)
    print("test loss %.3f " % loss.item())
```

```python colab={"base_uri": "https://localhost:8080/"} id="EztQtZKl9M53" outputId="25713f3c-edff-4333-e45b-ffb2c79375fb"
train_epocs(model, epochs=10, lr=0.1)
```

```python colab={"base_uri": "https://localhost:8080/"} id="AoSgUhWV9O1q" outputId="48e887fa-b55c-4465-e2d0-05789b5a7419"
train_epocs(model, epochs=10, lr=0.01)
```

```python colab={"base_uri": "https://localhost:8080/"} id="erRnsApY9Q7e" outputId="1e6a68f3-13b1-4963-b187-5d1b1bc3e438"
train_epocs(model, epochs=10, lr=0.01)
```

<!-- #region id="HAwA9Rts9UI1" -->
MF with bias
<!-- #endregion -->

```python id="Dur1n3lo9S3C"
class MF_bias(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100):
        super(MF_bias, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.item_bias = nn.Embedding(num_items, 1)
        self.user_emb.weight.data.uniform_(0,0.05)
        self.item_emb.weight.data.uniform_(0,0.05)
        self.user_bias.weight.data.uniform_(-0.01,0.01)
        self.item_bias.weight.data.uniform_(-0.01,0.01)
        
    def forward(self, u, v):
        U = self.user_emb(u)
        V = self.item_emb(v)
        b_u = self.user_bias(u).squeeze()
        b_v = self.item_bias(v).squeeze()
        return (U*V).sum(1) +  b_u  + b_v
```

```python id="WAyaietL9ZAq"
model = MF_bias(num_users, num_items, emb_size=100) #.cuda()
```

```python colab={"base_uri": "https://localhost:8080/"} id="5nEO-IVp9acn" outputId="773ee576-ea2e-40ac-97c7-7d78606595e1"
train_epocs(model, epochs=10, lr=0.05, wd=1e-5)
```

```python colab={"base_uri": "https://localhost:8080/"} id="nD2fD5A59cK4" outputId="03df230c-70ff-4767-e088-928d54f6cd37"
train_epocs(model, epochs=10, lr=0.01, wd=1e-5)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Os53hZxr9e_T" outputId="d9bf93d3-8411-4535-dcc7-ebcf4a3fbc48"
train_epocs(model, epochs=10, lr=0.001, wd=1e-5)
```

<!-- #region id="-XhFy6bU9h48" -->
Note that these models are susceptible to weight initialization, optimization algorithm and regularization.


<!-- #endregion -->

<!-- #region id="NugoowzF9kCk" -->
### Neural Network Model
Note here there is no matrix multiplication, we could potentially make the embeddings of different sizes. Here we could get better results by keep playing with regularization.
<!-- #endregion -->

```python id="qLVWHOxQ9fVX"
class CollabFNet(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100, n_hidden=10):
        super(CollabFNet, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.lin1 = nn.Linear(emb_size*2, n_hidden)
        self.lin2 = nn.Linear(n_hidden, 1)
        self.drop1 = nn.Dropout(0.1)
        
    def forward(self, u, v):
        U = self.user_emb(u)
        V = self.item_emb(v)
        x = F.relu(torch.cat([U, V], dim=1))
        x = self.drop1(x)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x
```

```python id="ljjju7Yy9x7b"
model = CollabFNet(num_users, num_items, emb_size=100) #.cuda()
```

```python colab={"base_uri": "https://localhost:8080/"} id="YuG2Hz5e9yyl" outputId="67b716ed-84e4-4dcf-eaf0-f06609542d0d"
train_epocs(model, epochs=15, lr=0.05, wd=1e-6, unsqueeze=True)
```

```python colab={"base_uri": "https://localhost:8080/"} id="JYyXb1qO90vb" outputId="b3bcc463-51a8-4625-f547-38ebbf105a7f"
train_epocs(model, epochs=10, lr=0.001, wd=1e-6, unsqueeze=True)
```

```python colab={"base_uri": "https://localhost:8080/"} id="o0d-WRvW92h-" outputId="20f81203-7fbb-44db-b421-c6c20239cd22"
train_epocs(model, epochs=10, lr=0.001, wd=1e-6, unsqueeze=True)
```

<!-- #region id="6fpmHCaYAn2d" -->
### Neural network model - different approach
> youtube: https://youtu.be/MVB1cbe923A
<!-- #endregion -->

<!-- #region id="SMuCwuPWPfGz" -->
### Ethan Rosenthal

Ref - https://github.com/EthanRosenthal/torchmf
<!-- #endregion -->

```python id="J31f-camBorB"
import os
import requests
import zipfile
import collections

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

import torch
from torch import nn
import torch.multiprocessing as mp
import torch.utils.data as data
from tqdm import tqdm
```

```python id="ahu8EWCGQJkI"
def _get_data_path():
    """
    Get path to the movielens dataset file.
    """
    data_path = '/content/data'
    if not os.path.exists(data_path):
        print('Making data path')
        os.mkdir(data_path)
    return data_path


def _download_movielens(dest_path):
    """
    Download the dataset.
    """

    url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
    req = requests.get(url, stream=True)

    print('Downloading MovieLens data')

    with open(os.path.join(dest_path, 'ml-100k.zip'), 'wb') as fd:
        for chunk in req.iter_content(chunk_size=None):
            fd.write(chunk)

    with zipfile.ZipFile(os.path.join(dest_path, 'ml-100k.zip'), 'r') as z:
        z.extractall(dest_path)
```

```python id="DouK7x1nPsNb"
def read_movielens_df():
    path = _get_data_path()
    zipfile = os.path.join(path, 'ml-100k.zip')
    if not os.path.isfile(zipfile):
        _download_movielens(path)
    fname = os.path.join(path, 'ml-100k', 'u.data')
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(fname, sep='\t', names=names)
    return df


def get_movielens_interactions():
    df = read_movielens_df()

    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]

    interactions = np.zeros((n_users, n_items))
    for row in df.itertuples():
        interactions[row[1] - 1, row[2] - 1] = row[3]
    return interactions


def train_test_split(interactions, n=10):
    """
    Split an interactions matrix into training and test sets.
    Parameters
    ----------
    interactions : np.ndarray
    n : int (default=10)
        Number of items to select / row to place into test.

    Returns
    -------
    train : np.ndarray
    test : np.ndarray
    """
    test = np.zeros(interactions.shape)
    train = interactions.copy()
    for user in range(interactions.shape[0]):
        if interactions[user, :].nonzero()[0].shape[0] > n:
            test_interactions = np.random.choice(interactions[user, :].nonzero()[0],
                                                 size=n,
                                                 replace=False)
            train[user, test_interactions] = 0.
            test[user, test_interactions] = interactions[user, test_interactions]

    # Test and training are truly disjoint
    assert(np.all((train * test) == 0))
    return train, test


def get_movielens_train_test_split(implicit=False):
    interactions = get_movielens_interactions()
    if implicit:
        interactions = (interactions >= 4).astype(np.float32)
    train, test = train_test_split(interactions)
    train = sp.coo_matrix(train)
    test = sp.coo_matrix(test)
    return train, test
```

```python colab={"base_uri": "https://localhost:8080/"} id="0x9xMyW6PsK6" outputId="00096a84-7542-4d07-920d-48830410244e"
%%writefile metrics.py

import numpy as np
from sklearn.metrics import roc_auc_score
from torch import multiprocessing as mp
import torch

def get_row_indices(row, interactions):
    start = interactions.indptr[row]
    end = interactions.indptr[row + 1]
    return interactions.indices[start:end]


def auc(model, interactions, num_workers=1):
    aucs = []
    processes = []
    n_users = interactions.shape[0]
    mp_batch = int(np.ceil(n_users / num_workers))

    queue = mp.Queue()
    rows = np.arange(n_users)
    np.random.shuffle(rows)
    for rank in range(num_workers):
        start = rank * mp_batch
        end = np.min((start + mp_batch,  n_users))
        p = mp.Process(target=batch_auc,
                       args=(queue, rows[start:end], interactions, model))
        p.start()
        processes.append(p)

    while True:
        is_alive = False
        for p in processes:
            if p.is_alive():
                is_alive = True
                break
        if not is_alive and queue.empty():
            break

        while not queue.empty():
            aucs.append(queue.get())

    queue.close()
    for p in processes:
        p.join()
    return np.mean(aucs)


def batch_auc(queue, rows, interactions, model):
    n_items = interactions.shape[1]
    items = torch.arange(0, n_items).long()
    users_init = torch.ones(n_items).long()
    for row in rows:
        row = int(row)
        users = users_init.fill_(row)

        preds = model.predict(users, items)
        actuals = get_row_indices(row, interactions)

        if len(actuals) == 0:
            continue
        y_test = np.zeros(n_items)
        y_test[actuals] = 1
        queue.put(roc_auc_score(y_test, preds.data.numpy()))


def patk(model, interactions, num_workers=1, k=5):
    patks = []
    processes = []
    n_users = interactions.shape[0]
    mp_batch = int(np.ceil(n_users / num_workers))

    queue = mp.Queue()
    rows = np.arange(n_users)
    np.random.shuffle(rows)
    for rank in range(num_workers):
        start = rank * mp_batch
        end = np.min((start + mp_batch, n_users))
        p = mp.Process(target=batch_patk,
                       args=(queue, rows[start:end], interactions, model),
                       kwargs={'k': k})
        p.start()
        processes.append(p)

    while True:
        is_alive = False
        for p in processes:
            if p.is_alive():
                is_alive = True
                break
        if not is_alive and queue.empty():
            break

        while not queue.empty():
            patks.append(queue.get())

    queue.close()
    for p in processes:
        p.join()
    return np.mean(patks)


def batch_patk(queue, rows, interactions, model, k=5):
    n_items = interactions.shape[1]

    items = torch.arange(0, n_items).long()
    users_init = torch.ones(n_items).long()
    for row in rows:
        row = int(row)
        users = users_init.fill_(row)

        preds = model.predict(users, items)
        actuals = get_row_indices(row, interactions)

        if len(actuals) == 0:
            continue

        top_k = np.argpartition(-np.squeeze(preds.data.numpy()), k)
        top_k = set(top_k[:k])
        true_pids = set(actuals)
        if true_pids:
            queue.put(len(top_k & true_pids) / float(k))
```

```python colab={"base_uri": "https://localhost:8080/"} id="I6IqWKWFhAQh" outputId="3bef0869-34b5-488c-ede7-bf9be08c115f"
import metrics
import importlib
importlib.reload(metrics)
```

```python id="UEhEE_GhPsH2"
class Interactions(data.Dataset):
    """
    Hold data in the form of an interactions matrix.
    Typical use-case is like a ratings matrix:
    - Users are the rows
    - Items are the columns
    - Elements of the matrix are the ratings given by a user for an item.
    """

    def __init__(self, mat):
        self.mat = mat.astype(np.float32).tocoo()
        self.n_users = self.mat.shape[0]
        self.n_items = self.mat.shape[1]

    def __getitem__(self, index):
        row = self.mat.row[index]
        col = self.mat.col[index]
        val = self.mat.data[index]
        return (row, col), val

    def __len__(self):
        return self.mat.nnz


class PairwiseInteractions(data.Dataset):
    """
    Sample data from an interactions matrix in a pairwise fashion. The row is
    treated as the main dimension, and the columns are sampled pairwise.
    """

    def __init__(self, mat):
        self.mat = mat.astype(np.float32).tocoo()

        self.n_users = self.mat.shape[0]
        self.n_items = self.mat.shape[1]

        self.mat_csr = self.mat.tocsr()
        if not self.mat_csr.has_sorted_indices:
            self.mat_csr.sort_indices()

    def __getitem__(self, index):
        row = self.mat.row[index]
        found = False

        while not found:
            neg_col = np.random.randint(self.n_items)
            if self.not_rated(row, neg_col, self.mat_csr.indptr,
                              self.mat_csr.indices):
                found = True

        pos_col = self.mat.col[index]
        val = self.mat.data[index]

        return (row, (pos_col, neg_col)), val

    def __len__(self):
        return self.mat.nnz

    @staticmethod
    def not_rated(row, col, indptr, indices):
        # similar to use of bsearch in lightfm
        start = indptr[row]
        end = indptr[row + 1]
        searched = np.searchsorted(indices[start:end], col, 'right')
        if searched >= (end - start):
            # After the array
            return False
        return col != indices[searched]  # Not found

    def get_row_indices(self, row):
        start = self.mat_csr.indptr[row]
        end = self.mat_csr.indptr[row + 1]
        return self.mat_csr.indices[start:end]


class BaseModule(nn.Module):
    """
    Base module for explicit matrix factorization.
    """
    
    def __init__(self,
                 n_users,
                 n_items,
                 n_factors=40,
                 dropout_p=0,
                 sparse=False):
        """

        Parameters
        ----------
        n_users : int
            Number of users
        n_items : int
            Number of items
        n_factors : int
            Number of latent factors (or embeddings or whatever you want to
            call it).
        dropout_p : float
            p in nn.Dropout module. Probability of dropout.
        sparse : bool
            Whether or not to treat embeddings as sparse. NOTE: cannot use
            weight decay on the optimizer if sparse=True. Also, can only use
            Adagrad.
        """
        super(BaseModule, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.user_biases = nn.Embedding(n_users, 1, sparse=sparse)
        self.item_biases = nn.Embedding(n_items, 1, sparse=sparse)
        self.user_embeddings = nn.Embedding(n_users, n_factors, sparse=sparse)
        self.item_embeddings = nn.Embedding(n_items, n_factors, sparse=sparse)
        
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=self.dropout_p)

        self.sparse = sparse
        
    def forward(self, users, items):
        """
        Forward pass through the model. For a single user and item, this
        looks like:

        user_bias + item_bias + user_embeddings.dot(item_embeddings)

        Parameters
        ----------
        users : np.ndarray
            Array of user indices
        items : np.ndarray
            Array of item indices

        Returns
        -------
        preds : np.ndarray
            Predicted ratings.

        """
        ues = self.user_embeddings(users)
        uis = self.item_embeddings(items)

        preds = self.user_biases(users)
        preds += self.item_biases(items)
        preds += (self.dropout(ues) * self.dropout(uis)).sum(dim=1, keepdim=True)

        return preds.squeeze()
    
    def __call__(self, *args):
        return self.forward(*args)

    def predict(self, users, items):
        return self.forward(users, items)


def bpr_loss(preds, vals):
    sig = nn.Sigmoid()
    return (1.0 - sig(preds)).pow(2).sum()


class BPRModule(nn.Module):
    
    def __init__(self,
                 n_users,
                 n_items,
                 n_factors=40,
                 dropout_p=0,
                 sparse=False,
                 model=BaseModule):
        super(BPRModule, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.dropout_p = dropout_p
        self.sparse = sparse
        self.pred_model = model(
            self.n_users,
            self.n_items,
            n_factors=n_factors,
            dropout_p=dropout_p,
            sparse=sparse
        )

    def forward(self, users, items):
        assert isinstance(items, tuple), \
            'Must pass in items as (pos_items, neg_items)'
        # Unpack
        (pos_items, neg_items) = items
        pos_preds = self.pred_model(users, pos_items)
        neg_preds = self.pred_model(users, neg_items)
        return pos_preds - neg_preds

    def predict(self, users, items):
        return self.pred_model(users, items)


class BasePipeline:
    """
    Class defining a training pipeline. Instantiates data loaders, model,
    and optimizer. Handles training for multiple epochs and keeping track of
    train and test loss.
    """

    def __init__(self,
                 train,
                 test=None,
                 model=BaseModule,
                 n_factors=40,
                 batch_size=32,
                 dropout_p=0.02,
                 sparse=False,
                 lr=0.01,
                 weight_decay=0.,
                 optimizer=torch.optim.Adam,
                 loss_function=nn.MSELoss(reduction='sum'),
                 n_epochs=10,
                 verbose=False,
                 random_seed=None,
                 interaction_class=Interactions,
                 hogwild=False,
                 num_workers=0,
                 eval_metrics=None,
                 k=5):
        self.train = train
        self.test = test

        if hogwild:
            num_loader_workers = 0
        else:
            num_loader_workers = num_workers
        self.train_loader = data.DataLoader(
            interaction_class(train), batch_size=batch_size, shuffle=True,
            num_workers=num_loader_workers)
        if self.test is not None:
            self.test_loader = data.DataLoader(
                interaction_class(test), batch_size=batch_size, shuffle=True,
                num_workers=num_loader_workers)
        self.num_workers = num_workers
        self.n_users = self.train.shape[0]
        self.n_items = self.train.shape[1]
        self.n_factors = n_factors
        self.batch_size = batch_size
        self.dropout_p = dropout_p
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_function = loss_function
        self.n_epochs = n_epochs
        if sparse:
            assert weight_decay == 0.0
        self.model = model(self.n_users,
                           self.n_items,
                           n_factors=self.n_factors,
                           dropout_p=self.dropout_p,
                           sparse=sparse)
        self.optimizer = optimizer(self.model.parameters(),
                                   lr=self.lr,
                                   weight_decay=self.weight_decay)
        self.warm_start = False
        self.losses = collections.defaultdict(list)
        self.verbose = verbose
        self.hogwild = hogwild
        if random_seed is not None:
            if self.hogwild:
                random_seed += os.getpid()
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

        if eval_metrics is None:
            eval_metrics = []
        self.eval_metrics = eval_metrics
        self.k = k

    def break_grads(self):
        for param in self.model.parameters():
            # Break gradient sharing
            if param.grad is not None:
                param.grad.data = param.grad.data.clone()

    def fit(self):
        for epoch in range(1, self.n_epochs + 1):

            if self.hogwild:
                self.model.share_memory()
                processes = []
                train_losses = []
                queue = mp.Queue()
                for rank in range(self.num_workers):
                    p = mp.Process(target=self._fit_epoch,
                                   kwargs={'epoch': epoch,
                                           'queue': queue})
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()

                while True:
                    is_alive = False
                    for p in processes:
                        if p.is_alive():
                            is_alive = True
                            break
                    if not is_alive and queue.empty():
                        break

                    while not queue.empty():
                        train_losses.append(queue.get())
                queue.close()
                train_loss = np.mean(train_losses)
            else:
                train_loss = self._fit_epoch(epoch)

            self.losses['train'].append(train_loss)
            row = 'Epoch: {0:^3}  train: {1:^10.5f}'.format(epoch, self.losses['train'][-1])
            if self.test is not None:
                self.losses['test'].append(self._validation_loss())
                row += 'val: {0:^10.5f}'.format(self.losses['test'][-1])
                for metric in self.eval_metrics:
                    func = getattr(metrics, metric)
                    res = func(self.model, self.test_loader.dataset.mat_csr,
                               num_workers=self.num_workers)
                    self.losses['eval-{}'.format(metric)].append(res)
                    row += 'eval-{0}: {1:^10.5f}'.format(metric, res)
            self.losses['epoch'].append(epoch)
            if self.verbose:
                print(row)

    def _fit_epoch(self, epoch=1, queue=None):
        if self.hogwild:
            self.break_grads()

        self.model.train()
        total_loss = torch.Tensor([0])
        pbar = tqdm(enumerate(self.train_loader),
                    total=len(self.train_loader),
                    desc='({0:^3})'.format(epoch))
        for batch_idx, ((row, col), val) in pbar:
            self.optimizer.zero_grad()

            row = row.long()
            # TODO: turn this into a collate_fn like the data_loader
            if isinstance(col, list):
                col = tuple(c.long() for c in col)
            else:
                col = col.long()
            val = val.float()

            preds = self.model(row, col)
            loss = self.loss_function(preds, val)
            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()
            batch_loss = loss.item() / row.size()[0]
            pbar.set_postfix(train_loss=batch_loss)
        total_loss /= self.train.nnz
        if queue is not None:
            queue.put(total_loss[0])
        else:
            return total_loss[0]

    def _validation_loss(self):
        self.model.eval()
        total_loss = torch.Tensor([0])
        for batch_idx, ((row, col), val) in enumerate(self.test_loader):
            row = row.long()
            if isinstance(col, list):
                col = tuple(c.long() for c in col)
            else:
                col = col.long()
            val = val.float()

            preds = self.model(row, col)
            loss = self.loss_function(preds, val)
            total_loss += loss.item()

        total_loss /= self.test.nnz
        return total_loss[0]
```

```python id="iKeUFiRNPsFY"
def explicit():
    train, test = get_movielens_train_test_split()
    pipeline = BasePipeline(train, test=test, model=BaseModule,
                            n_factors=10, batch_size=1024, dropout_p=0.02,
                            lr=0.02, weight_decay=0.1,
                            optimizer=torch.optim.Adam, n_epochs=40,
                            verbose=True, random_seed=2017)
    pipeline.fit()


def implicit():
    train, test = get_movielens_train_test_split(implicit=True)

    pipeline = BasePipeline(train, test=test, verbose=True,
                           batch_size=1024, num_workers=4,
                           n_factors=20, weight_decay=0,
                           dropout_p=0., lr=.2, sparse=True,
                           optimizer=torch.optim.SGD, n_epochs=40,
                           random_seed=2017, loss_function=bpr_loss,
                           model=BPRModule,
                           interaction_class=PairwiseInteractions,
                           eval_metrics=('auc', 'patk'))
    pipeline.fit()


def hogwild():
    train, test = get_movielens_train_test_split(implicit=True)

    pipeline = BasePipeline(train, test=test, verbose=True,
                            batch_size=1024, num_workers=4,
                            n_factors=20, weight_decay=0,
                            dropout_p=0., lr=.2, sparse=True,
                            optimizer=torch.optim.SGD, n_epochs=40,
                            random_seed=2017, loss_function=bpr_loss,
                            model=BPRModule, hogwild=True,
                            interaction_class=PairwiseInteractions,
                            eval_metrics=('auc', 'patk'))
    pipeline.fit()
```

```python colab={"base_uri": "https://localhost:8080/"} id="dPQaj1FjPsCo" outputId="42275f83-f4e9-43cc-a105-3ecde82efaa4"
explicit()
```

```python colab={"base_uri": "https://localhost:8080/"} id="mtI0DewsPr_0" outputId="6cc428e1-211f-4e7e-8264-e8332ad47e8b"
implicit()
```

<!-- #region id="tbXR1BvPWXKO" -->
## Neural Graph Collaborative Filtering on MovieLens
> Applying NGCF PyTorch version on Movielens-100k.
<!-- #endregion -->

<!-- #region id="sG-h_5yQQEvQ" -->
### Libraries
<!-- #endregion -->

```python id="3QJEhOlDwIR7"
!pip install -q git+https://github.com/sparsh-ai/recochef
```

```python id="08tq9wC8pdD1"
import os
import csv 
import argparse
import numpy as np
import pandas as pd
import random as rd
from time import time
from pathlib import Path
import scipy.sparse as sp
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F

from recochef.preprocessing.split import chrono_split
```

```python id="F_-NBXzHS0_o"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)
```

<!-- #region id="uuM8Q9GlP8RN" -->
### Data Loading

The MovieLens 100K data set consists of 100,000 ratings from 1000 users on 1700 movies as described on [their website](https://grouplens.org/datasets/movielens/100k/).
<!-- #endregion -->

```python id="Q-yPokpipXsY"
!wget http://files.grouplens.org/datasets/movielens/ml-100k.zip
```

```python id="6CLWhSTXpbvW"
!unzip ml-100k.zip
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="KS9OJY75pgwq" outputId="0c1ec91e-2ef6-4b5a-e613-fa4501e2737f"
df = pd.read_csv('ml-100k/u.data', sep='\t', header=None, names=['USERID','ITEMID','RATING','TIMESTAMP'])
df.head()
```

<!-- #region id="Mo6kTBolQYca" -->
### Train/Test Split

We split the data chronologically in 80:20 ratio. Validated the split for user 4.
<!-- #endregion -->

```python id="XTJpJj2uvpD2"
df_train, df_test = chrono_split(df, ratio=0.8)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="Dpeu1H5xxXcR" outputId="d94c244d-ad25-47c1-d084-e51f6b015645"
userid = 4

query = "USERID==@userid"
display(df.query(query))
display(df_train.query(query))
display(df_test.query(query))
```

<!-- #region id="qny2aVoWQknP" -->
### Preprocessing

1. Sort by User ID and Timestamp
2. Label encode user and item id - in this case, already label encoded starting from 1, so decreasing ids by 1 as a proxy for label encode
3. Remove Timestamp and Rating column. The reason is that we are training a recall-maximing model where the objective is to correctly retrieve the items that users can interact with. We can select a rating threshold also
4. Convert Item IDs into list format
5. Store as a space-seperated txt file
<!-- #endregion -->

```python id="1iSOiyCqpmYE"
def preprocess(data):
  data = data.copy()
  data = data.sort_values(by=['USERID','TIMESTAMP'])
  data['USERID'] = data['USERID'] - 1
  data['ITEMID'] = data['ITEMID'] - 1
  data.drop(['TIMESTAMP','RATING'], axis=1, inplace=True)
  data = data.groupby('USERID')['ITEMID'].apply(list).reset_index(name='ITEMID')
  return data
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="D7ZtrUPp22dO" outputId="56290e63-dcf3-448b-b3a5-ce60bd2c23db"
preprocess(df_train).head()
```

```python id="yDMAhrig1Lde"
def store(data, target_file='./data/movielens/train.txt'):
  Path(target_file).parent.mkdir(parents=True, exist_ok=True)
  with open(target_file, 'w+') as f:
    writer = csv.writer(f, delimiter=' ')
    for USERID, row in zip(data.USERID.values,data.ITEMID.values):
      row = [USERID] + row
      writer.writerow(row)
```

```python id="XUIFrKsavRzV"
store(preprocess(df_train), '/content/data/ml-100k/train.txt')
store(preprocess(df_test), '/content/data/ml-100k/test.txt')
```

```python colab={"base_uri": "https://localhost:8080/"} id="vq2IwCkJtUTy" outputId="bdd5df6b-213f-4e87-deae-b8f29e42ec87"
!head /content/data/ml-100k/train.txt
```

```python colab={"base_uri": "https://localhost:8080/"} id="E7YYuu2XuVQa" outputId="fd6fc81c-8ee7-4a34-cb7f-ae5c9ffb27d6"
!head /content/data/ml-100k/test.txt
```

```python id="C-f9mzEf4Ow6"
Path('/content/results').mkdir(parents=True, exist_ok=True)
```

```python id="Bxz7aV0Sws1S"
def parse_args():
    parser = argparse.ArgumentParser(description="Run NGCF.")
    parser.add_argument('--data_dir', type=str,
                        default='./data/',
                        help='Input data path.')
    parser.add_argument('--dataset', type=str, default='ml-100k',
                        help='Dataset name: Amazond-book, Gowella, ml-100k')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Store model to path.')
    parser.add_argument('--n_epochs', type=int, default=400,
                        help='Number of epoch.')
    parser.add_argument('--reg', type=float, default=1e-5,
                        help='l2 reg.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='number of embeddings.')
    parser.add_argument('--layers', type=str, default='[64,64]',
                        help='Output sizes of every layer')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size.')
    parser.add_argument('--node_dropout', type=float, default=0.,
                        help='Graph Node dropout.')
    parser.add_argument('--mess_dropout', type=float, default=0.1,
                        help='Message dropout.')
    parser.add_argument('--k', type=str, default=20,
                        help='k order of metric evaluation (e.g. NDCG@k)')
    parser.add_argument('--eval_N', type=int, default=5,
                        help='Evaluate every N epochs')
    parser.add_argument('--save_results', type=int, default=1,
                        help='Save model and results')

    return parser.parse_args(args={})
```

<!-- #region id="twi1ZIucR0ga" -->
### Helper Functions

- early_stopping()
- train()
- split_matrix()
- ndcg_k()
- eval_model
<!-- #endregion -->

<!-- #region id="aCShFbsCTPzw" -->
#### Early Stopping
Premature stopping is applied if *recall@20* on the test set does not increase for 5 successive epochs.
<!-- #endregion -->

```python id="tHVTudWxTVZo"
def early_stopping(log_value, best_value, stopping_step, flag_step, expected_order='asc'):
    """
    Check if early_stopping is needed
    Function copied from original code
    """
    assert expected_order in ['asc', 'des']
    if (expected_order == 'asc' and log_value >= best_value) or (expected_order == 'des' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False

    return best_value, stopping_step, should_stop
```

```python id="6JEG5Jlpw3Nw"
def train(model, data_generator, optimizer):
    """
    Train the model PyTorch style
    Arguments:
    ---------
    model: PyTorch model
    data_generator: Data object
    optimizer: PyTorch optimizer
    """
    model.train()
    n_batch = data_generator.n_train // data_generator.batch_size + 1
    running_loss=0
    for _ in range(n_batch):
        u, i, j = data_generator.sample()
        optimizer.zero_grad()
        loss = model(u,i,j)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss

def split_matrix(X, n_splits=100):
    """
    Split a matrix/Tensor into n_folds (for the user embeddings and the R matrices)
    Arguments:
    ---------
    X: matrix to be split
    n_folds: number of folds
    Returns:
    -------
    splits: split matrices
    """
    splits = []
    chunk_size = X.shape[0] // n_splits
    for i in range(n_splits):
        start = i * chunk_size
        end = X.shape[0] if i == n_splits - 1 else (i + 1) * chunk_size
        splits.append(X[start:end])
    return splits

def compute_ndcg_k(pred_items, test_items, test_indices, k):
    """
    Compute NDCG@k
    
    Arguments:
    ---------
    pred_items: binary tensor with 1s in those locations corresponding to the predicted item interactions
    test_items: binary tensor with 1s in locations corresponding to the real test interactions
    test_indices: tensor with the location of the top-k predicted items
    k: k'th-order 
    Returns:
    -------
    NDCG@k
    """
    r = (test_items * pred_items).gather(1, test_indices)
    f = torch.from_numpy(np.log2(np.arange(2, k+2))).float().cuda()
    dcg = (r[:, :k]/f).sum(1)
    dcg_max = (torch.sort(r, dim=1, descending=True)[0][:, :k]/f).sum(1)
    ndcg = dcg/dcg_max
    ndcg[torch.isnan(ndcg)] = 0
    return ndcg
```

<!-- #region id="sx-Vzl2vTeWN" -->
#### Eval Model

At every N epoch, the model is evaluated on the test set. From this evaluation, we compute the recall and normal discounted cumulative gain (ndcg) at the top-20 predictions. It is important to note that in order to evaluate the model on the test set we have to ‘unpack’ the sparse matrix (torch.sparse.todense()), and thus load a bunch of ‘zeros’ on memory. In order to prevent memory overload, we split the sparse matrices into 100 chunks, unpack the sparse chunks one by one, compute the metrics we need, and compute the mean value of all chunks.
<!-- #endregion -->

```python id="1dysqVKGTjm6"
def eval_model(u_emb, i_emb, Rtr, Rte, k):
    """
    Evaluate the model
    
    Arguments:
    ---------
    u_emb: User embeddings
    i_emb: Item embeddings
    Rtr: Sparse matrix with the training interactions
    Rte: Sparse matrix with the testing interactions
    k : kth-order for metrics
    
    Returns:
    --------
    result: Dictionary with lists correponding to the metrics at order k for k in Ks
    """
    # split matrices
    ue_splits = split_matrix(u_emb)
    tr_splits = split_matrix(Rtr)
    te_splits = split_matrix(Rte)

    recall_k, ndcg_k= [], []
    # compute results for split matrices
    for ue_f, tr_f, te_f in zip(ue_splits, tr_splits, te_splits):

        scores = torch.mm(ue_f, i_emb.t())

        test_items = torch.from_numpy(te_f.todense()).float().cuda()
        non_train_items = torch.from_numpy(1-(tr_f.todense())).float().cuda()
        scores = scores * non_train_items

        _, test_indices = torch.topk(scores, dim=1, k=k)

        # If you want to use a as the index in dim1 for t, this code should work:
        #t[torch.arange(t.size(0)), a]

        pred_items = torch.zeros_like(scores).float()
        # pred_items.scatter_(dim=1,index=test_indices,src=torch.tensor(1.0).cuda())
        pred_items.scatter_(dim=1,index=test_indices,src=torch.ones_like(test_indices, dtype=torch.float).cuda())

        topk_preds = torch.zeros_like(scores).float()
        # topk_preds.scatter_(dim=1,index=test_indices[:, :k],src=torch.tensor(1.0))
        _idx = test_indices[:, :k]
        topk_preds.scatter_(dim=1,index=_idx,src=torch.ones_like(_idx, dtype=torch.float))

        TP = (test_items * topk_preds).sum(1)
        rec = TP/test_items.sum(1)
        ndcg = compute_ndcg_k(pred_items, test_items, test_indices, k)

        recall_k.append(rec)
        ndcg_k.append(ndcg)

    return torch.cat(recall_k).mean(), torch.cat(ndcg_k).mean()
```

<!-- #region id="mvvKJOlpSLn4" -->
### Dataset Class
<!-- #endregion -->

<!-- #region id="AzoIqHHuUADD" -->
#### Laplacian matrix

The components of the Laplacian matrix are as follows,

- **D**: a diagonal degree matrix, where D{t,t} is |N{t}|, which is the amount of first-hop neighbors for either item or user t,
- **R**: the user-item interaction matrix,
- **0**: an all-zero matrix,
- **A**: the adjacency matrix,

#### Interaction and Adjacency Matrix

We create the sparse interaction matrix R, the adjacency matrix A, the degree matrix D, and the Laplacian matrix L, using the SciPy library. The adjacency matrix A is then transferred onto PyTorch tensor objects.
<!-- #endregion -->

```python id="s0w9GTdKw7Vj"
class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        #get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        self.exist_users = []

        # search train_file for max user_id/item_id
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    # first element is the user_id, rest are items
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    # item/user with highest number is number of items/users
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    # number of interactions
                    self.n_train += len(items)

        # search test_file for max item_id
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    if not items:
                        print("empyt test exists")
                        pass
                    else:
                        self.n_items = max(self.n_items, max(items))
                        self.n_test += len(items)
        # adjust counters: user_id/item_id starts at 0
        self.n_items += 1
        self.n_users += 1

        self.print_statistics()

        # create interactions/ratings matrix 'R' # dok = dictionary of keys
        print('Creating interaction matrices R_train and R_test...')
        t1 = time()
        self.R_train = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32) 
        self.R_test = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        self.train_items, self.test_set = {}, {}
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]
                    # enter 1 if user interacted with item
                    for i in train_items:
                        self.R_train[uid, i] = 1.
                    self.train_items[uid] = train_items

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue
                    uid, test_items = items[0], items[1:]
                    for i in test_items:
                        self.R_test[uid, i] = 1.0
                    self.test_set[uid] = test_items
        print('Complete. Interaction matrices R_train and R_test created in', time() - t1, 'sec')

    # if exist, get adjacency matrix
    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            print('Loaded adjacency-matrix (shape:', adj_mat.shape,') in', time() - t1, 'sec.')

        except Exception:
            print('Creating adjacency-matrix...')
            adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
        return adj_mat
    
    # create adjancency matrix
    def create_adj_mat(self):
        t1 = time()
        
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R_train.tolil() # to list of lists

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('Complete. Adjacency-matrix created in', adj_mat.shape, time() - t1, 'sec.')

        t2 = time()

        # normalize adjacency matrix
        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj).dot(d_mat_inv)
            return norm_adj.tocoo()

        print('Transforming adjacency-matrix to NGCF-adjacency matrix...')
        ngcf_adj_mat = normalized_adj_single(adj_mat) + sp.eye(adj_mat.shape[0])

        print('Complete. Transformed adjacency-matrix to NGCF-adjacency matrix in', time() - t2, 'sec.')
        return ngcf_adj_mat.tocsr()

    # create collections of N items that users never interacted with
    def negative_pool(self):
        t1 = time()
        for u in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            pools = [rd.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools
        print('refresh negative pools', time() - t1)

    # sample data for mini-batches
    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))
```

<!-- #region id="J2RxhIxmSYvl" -->
### NGCF Model
<!-- #endregion -->

<!-- #region id="P4vz1IOwTvED" -->
#### Weight initialization

We then create tensors for the user embeddings and item embeddings with the proper dimensions. The weights are initialized using [Xavier uniform initialization](https://pytorch.org/docs/stable/nn.init.html).

For each layer, the weight matrices and corresponding biases are initialized using the same procedure.
<!-- #endregion -->

<!-- #region id="wo9vgNvJUWvR" -->
#### Embedding Layer

The initial user and item embeddings are concatenated in an embedding lookup table as shown in the figure below. This embedding table is initialized using the user and item embeddings and will be optimized in an end-to-end fashion by the network.
<!-- #endregion -->

<!-- #region id="ntRUCJGMUeNf" -->
<!-- #endregion -->

<!-- #region id="HKqah7XFUhan" -->
#### Embedding propagation

The embedding table is propagated through the network using the formula shown in the figure below.
<!-- #endregion -->

<!-- #region id="ZvrR6lvUUuIm" -->
<!-- #endregion -->

<!-- #region id="Co9D_oNXUlgj" -->
The components of the formula are as follows,

- **E⁽ˡ⁾**: the embedding table after l steps of embedding propagation, where E⁽⁰⁾ is the initial embedding table,
- **LeakyReLU**: the rectified linear unit used as activation function,
- **W**: the weights trained by the network,
- **I**: an identity matrix,
- **L**: the Laplacian matrix for the user-item graph, which is formulated as
<!-- #endregion -->

<!-- #region id="SLrIvZowUyyI" -->
<!-- #endregion -->

<!-- #region id="TcNXtDMFVLii" -->
#### Architecture
<!-- #endregion -->

<!-- #region id="kyKDfxNfBWl-" -->
![](https://github.com/recohut/reco-static/raw/master/media/images/120222_ncf.png)
<!-- #endregion -->

```python id="4i1YYbJB4oGQ"
class NGCF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim, layers, reg, node_dropout, mess_dropout,
        adj_mtx):
        super().__init__()

        # initialize Class attributes
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.adj_mtx = adj_mtx
        self.laplacian = adj_mtx - sp.eye(adj_mtx.shape[0])
        self.reg = reg
        self.layers = layers
        self.n_layers = len(self.layers)
        self.node_dropout = node_dropout
        self.mess_dropout = mess_dropout

        #self.u_g_embeddings = nn.Parameter(torch.empty(n_users, emb_dim+np.sum(self.layers)))
        #self.i_g_embeddings = nn.Parameter(torch.empty(n_items, emb_dim+np.sum(self.layers)))

        # Initialize weights
        self.weight_dict = self._init_weights()
        print("Weights initialized.")

        # Create Matrix 'A', PyTorch sparse tensor of SP adjacency_mtx
        self.A = self._convert_sp_mat_to_sp_tensor(self.adj_mtx)
        self.L = self._convert_sp_mat_to_sp_tensor(self.laplacian)

    # initialize weights
    def _init_weights(self):
        print("Initializing weights...")
        weight_dict = nn.ParameterDict()

        initializer = torch.nn.init.xavier_uniform_
        
        weight_dict['user_embedding'] = nn.Parameter(initializer(torch.empty(self.n_users, self.emb_dim).to(device)))
        weight_dict['item_embedding'] = nn.Parameter(initializer(torch.empty(self.n_items, self.emb_dim).to(device)))

        weight_size_list = [self.emb_dim] + self.layers

        for k in range(self.n_layers):
            weight_dict['W_gc_%d' %k] = nn.Parameter(initializer(torch.empty(weight_size_list[k], weight_size_list[k+1]).to(device)))
            weight_dict['b_gc_%d' %k] = nn.Parameter(initializer(torch.empty(1, weight_size_list[k+1]).to(device)))
            
            weight_dict['W_bi_%d' %k] = nn.Parameter(initializer(torch.empty(weight_size_list[k], weight_size_list[k+1]).to(device)))
            weight_dict['b_bi_%d' %k] = nn.Parameter(initializer(torch.empty(1, weight_size_list[k+1]).to(device)))
           
        return weight_dict

    # convert sparse matrix into sparse PyTorch tensor
    def _convert_sp_mat_to_sp_tensor(self, X):
        """
        Convert scipy sparse matrix to PyTorch sparse matrix
        Arguments:
        ----------
        X = Adjacency matrix, scipy sparse matrix
        """
        coo = X.tocoo().astype(np.float32)
        i = torch.LongTensor(np.mat([coo.row, coo.col]))
        v = torch.FloatTensor(coo.data)
        res = torch.sparse.FloatTensor(i, v, coo.shape).to(device)
        return res

    # apply node_dropout
    def _droupout_sparse(self, X):
        """
        Drop individual locations in X
        
        Arguments:
        ---------
        X = adjacency matrix (PyTorch sparse tensor)
        dropout = fraction of nodes to drop
        noise_shape = number of non non-zero entries of X
        """
        
        node_dropout_mask = ((self.node_dropout) + torch.rand(X._nnz())).floor().bool().to(device)
        i = X.coalesce().indices()
        v = X.coalesce()._values()
        i[:,node_dropout_mask] = 0
        v[node_dropout_mask] = 0
        X_dropout = torch.sparse.FloatTensor(i, v, X.shape).to(X.device)

        return  X_dropout.mul(1/(1-self.node_dropout))

    def forward(self, u, i, j):
        """
        Computes the forward pass
        
        Arguments:
        ---------
        u = user
        i = positive item (user interacted with item)
        j = negative item (user did not interact with item)
        """
        # apply drop-out mask
        A_hat = self._droupout_sparse(self.A) if self.node_dropout > 0 else self.A
        L_hat = self._droupout_sparse(self.L) if self.node_dropout > 0 else self.L

        ego_embeddings = torch.cat([self.weight_dict['user_embedding'], self.weight_dict['item_embedding']], 0)

        all_embeddings = [ego_embeddings]

        # forward pass for 'n' propagation layers
        for k in range(self.n_layers):

            # weighted sum messages of neighbours
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            side_L_embeddings = torch.sparse.mm(L_hat, ego_embeddings)

            # transformed sum weighted sum messages of neighbours
            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) + self.weight_dict['b_gc_%d' % k]

            # bi messages of neighbours
            bi_embeddings = torch.mul(ego_embeddings, side_L_embeddings)
            # transformed bi messages of neighbours
            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) + self.weight_dict['b_bi_%d' % k]

            # non-linear activation 
            ego_embeddings = F.leaky_relu(sum_embeddings + bi_embeddings)
            # + message dropout
            mess_dropout_mask = nn.Dropout(self.mess_dropout)
            ego_embeddings = mess_dropout_mask(ego_embeddings)

            # normalize activation
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            all_embeddings.append(norm_embeddings)

        all_embeddings = torch.cat(all_embeddings, 1)
        
        # back to user/item dimension
        u_g_embeddings, i_g_embeddings = all_embeddings.split([self.n_users, self.n_items], 0)

        self.u_g_embeddings = nn.Parameter(u_g_embeddings)
        self.i_g_embeddings = nn.Parameter(i_g_embeddings)
        
        u_emb = u_g_embeddings[u] # user embeddings
        p_emb = i_g_embeddings[i] # positive item embeddings
        n_emb = i_g_embeddings[j] # negative item embeddings

        y_ui = torch.mul(u_emb, p_emb).sum(dim=1)
        y_uj = torch.mul(u_emb, n_emb).sum(dim=1)
        log_prob = (torch.log(torch.sigmoid(y_ui-y_uj))).mean()

        # compute bpr-loss
        bpr_loss = -log_prob
        if self.reg > 0.:
            l2norm = (torch.sum(u_emb**2)/2. + torch.sum(p_emb**2)/2. + torch.sum(n_emb**2)/2.) / u_emb.shape[0]
            l2reg  = self.reg*l2norm
            bpr_loss =  -log_prob + l2reg

        return bpr_loss
```

<!-- #region id="5xbcqHLUSowG" -->
### Training and Evaluation
<!-- #endregion -->

<!-- #region id="N6S7uZU3Tpht" -->
Training is done using the standard PyTorch method. If you are already familiar with PyTorch, the following code should look familiar.

One of the most useful functions of PyTorch is the torch.nn.Sequential() function, that takes existing and custom torch.nn modules. This makes it very easy to build and train complete networks. However, due to the nature of NCGF model structure, usage of torch.nn.Sequential() is not possible and the forward pass of the network has to be implemented ‘manually’. Using the Bayesian personalized ranking (BPR) pairwise loss, the forward pass is implemented as follows:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="LEMcstCz4vSm" outputId="4e06cd7c-8e69-4f6b-d4b8-054d373f01cb"
# read parsed arguments
args = parse_args()
data_dir = args.data_dir
dataset = args.dataset
batch_size = args.batch_size
layers = eval(args.layers)
emb_dim = args.emb_dim
lr = args.lr
reg = args.reg
mess_dropout = args.mess_dropout
node_dropout = args.node_dropout
k = args.k

# generate the NGCF-adjacency matrix
data_generator = Data(path=data_dir + dataset, batch_size=batch_size)
adj_mtx = data_generator.get_adj_mat()

# create model name and save
modelname =  "NGCF" + \
    "_bs_" + str(batch_size) + \
    "_nemb_" + str(emb_dim) + \
    "_layers_" + str(layers) + \
    "_nodedr_" + str(node_dropout) + \
    "_messdr_" + str(mess_dropout) + \
    "_reg_" + str(reg) + \
    "_lr_"  + str(lr)

# create NGCF model
model = NGCF(data_generator.n_users, 
              data_generator.n_items,
              emb_dim,
              layers,
              reg,
              node_dropout,
              mess_dropout,
              adj_mtx)
if use_cuda:
    model = model.cuda()

# current best metric
cur_best_metric = 0

# Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Set values for early stopping
cur_best_loss, stopping_step, should_stop = 1e3, 0, False
today = datetime.now()

print("Start at " + str(today))
print("Using " + str(device) + " for computations")
print("Params on CUDA: " + str(next(model.parameters()).is_cuda))

results = {"Epoch": [],
            "Loss": [],
            "Recall": [],
            "NDCG": [],
            "Training Time": []}

for epoch in range(args.n_epochs):

    t1 = time()
    loss = train(model, data_generator, optimizer)
    training_time = time()-t1
    print("Epoch: {}, Training time: {:.2f}s, Loss: {:.4f}".
        format(epoch, training_time, loss))

    # print test evaluation metrics every N epochs (provided by args.eval_N)
    if epoch % args.eval_N  == (args.eval_N - 1):
        with torch.no_grad():
            t2 = time()
            recall, ndcg = eval_model(model.u_g_embeddings.detach(),
                                      model.i_g_embeddings.detach(),
                                      data_generator.R_train,
                                      data_generator.R_test,
                                      k)
        print(
            "Evaluate current model:\n",
            "Epoch: {}, Validation time: {:.2f}s".format(epoch, time()-t2),"\n",
            "Loss: {:.4f}:".format(loss), "\n",
            "Recall@{}: {:.4f}".format(k, recall), "\n",
            "NDCG@{}: {:.4f}".format(k, ndcg)
            )

        cur_best_metric, stopping_step, should_stop = \
        early_stopping(recall, cur_best_metric, stopping_step, flag_step=5)

        # save results in dict
        results['Epoch'].append(epoch)
        results['Loss'].append(loss)
        results['Recall'].append(recall.item())
        results['NDCG'].append(ndcg.item())
        results['Training Time'].append(training_time)
    else:
        # save results in dict
        results['Epoch'].append(epoch)
        results['Loss'].append(loss)
        results['Recall'].append(None)
        results['NDCG'].append(None)
        results['Training Time'].append(training_time)

    if should_stop == True: break

# save
if args.save_results:
    date = today.strftime("%d%m%Y_%H%M")

    # save model as .pt file
    if os.path.isdir("./models"):
        torch.save(model.state_dict(), "./models/" + str(date) + "_" + modelname + "_" + dataset + ".pt")
    else:
        os.mkdir("./models")
        torch.save(model.state_dict(), "./models/" + str(date) + "_" + modelname + "_" + dataset + ".pt")

    # save results as pandas dataframe
    results_df = pd.DataFrame(results)
    results_df.set_index('Epoch', inplace=True)
    if os.path.isdir("./results"):
        results_df.to_csv("./results/" + str(date) + "_" + modelname + "_" + dataset + ".csv")
    else:
        os.mkdir("./results")
        results_df.to_csv("./results/" + str(date) + "_" + modelname + "_" + dataset + ".csv")
    # plot loss
    results_df['Loss'].plot(figsize=(12,8), title='Loss')
```

<!-- #region id="wBD0wM3JVXol" -->
### Appendix
<!-- #endregion -->

<!-- #region id="Iz0QI1P7VaDP" -->
#### References
1. [https://medium.com/@yusufnoor_88274/implementing-neural-graph-collaborative-filtering-in-pytorch-4d021dff25f3](https://medium.com/@yusufnoor_88274/implementing-neural-graph-collaborative-filtering-in-pytorch-4d021dff25f3)
2. [https://github.com/xiangwang1223/neural_graph_collaborative_filtering](https://github.com/xiangwang1223/neural_graph_collaborative_filtering)
3. [https://arxiv.org/pdf/1905.08108.pdf](https://arxiv.org/pdf/1905.08108.pdf)
4. [https://github.com/metahexane/ngcf_pytorch_g61](https://github.com/metahexane/ngcf_pytorch_g61)
<!-- #endregion -->

<!-- #region id="_JywmGguVbNU" -->
#### Next

Try out this notebook on the following datasets:

![](https://github.com/recohut/reco-static/raw/master/media/images/120222_data.png)
<!-- #endregion -->

<!-- #region id="MppKYNXJVswT" -->
Compare out the performance with these baselines:

1. MF: This is matrix factorization optimized by the Bayesian
personalized ranking (BPR) loss, which exploits the user-item
direct interactions only as the target value of interaction function.
2. NeuMF: The method is a state-of-the-art neural CF model
which uses multiple hidden layers above the element-wise and
concatenation of user and item embeddings to capture their nonlinear feature interactions. Especially, we employ two-layered
plain architecture, where the dimension of each hidden layer
keeps the same.
3. CMN: It is a state-of-the-art memory-based model, where
the user representation attentively combines the memory slots
of neighboring users via the memory layers. Note that the firstorder connections are used to find similar users who interacted
with the same items.
4. HOP-Rec: This is a state-of-the-art graph-based model,
where the high-order neighbors derived from random walks
are exploited to enrich the user-item interaction data.
5. PinSage: PinSage is designed to employ GraphSAGE
on item-item graph. In this work, we apply it on user-item interaction graph. Especially, we employ two graph convolution
layers, and the hidden dimension is set equal
to the embedding size.
6. GC-MC: This model adopts GCN encoder to generate
the representations for users and items, where only the first-order
neighbors are considered. Hence one graph convolution layer,
where the hidden dimension is set as the embedding size, is used.
<!-- #endregion -->

<!-- #region id="e7RRc2UQBuc9" -->
## A simple recommender with tensorflow
> A tutorial on how to build a simple deep learning based movie recommender using tensorflow library.
<!-- #endregion -->

```python id="hLtJPt_5idKN"
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

tf.random.set_seed(343)
```

```python id="DNLlAwKUihC1"
# Clean up the logdir if it exists
import shutil
shutil.rmtree('logs', ignore_errors=True)

# Load TensorBoard extension for notebooks
%load_ext tensorboard
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="8IRTF0EVjQuX" outputId="932eaa43-725c-4fb8-e9d4-dca92ced4cf0"
movielens_ratings_file = 'https://github.com/sparsh-ai/reco-data/blob/master/MovieLens_100K_ratings.csv?raw=true'
df_raw = pd.read_csv(movielens_ratings_file)
df_raw.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="El1C8OwWjhxk" outputId="f8ed06e7-8554-45f2-982d-987f153a5cc7"
df = df_raw.copy()
df.columns = ['userId', 'movieId', 'rating', 'timestamp']
user_ids = df['userId'].unique()
user_encoding = {x: i for i, x in enumerate(user_ids)}   # {user_id: index}
movie_ids = df['movieId'].unique()
movie_encoding = {x: i for i, x in enumerate(movie_ids)} # {movie_id: index}

df['user'] = df['userId'].map(user_encoding)    # Map from IDs to indices
df['movie'] = df['movieId'].map(movie_encoding)

n_users = len(user_ids)
n_movies = len(movie_ids)

min_rating = min(df['rating'])
max_rating = max(df['rating'])

print(f'Number of users: {n_users}\nNumber of movies: {n_movies}\nMin rating: {min_rating}\nMax rating: {max_rating}')

# Shuffle the data
df = df.sample(frac=1, random_state=42)
```

<!-- #region id="1W5V8T-C8Gpv" -->
### Scheme of the model

![](https://github.com/recohut/reco-static/raw/master/media/images/120222_scheme.png)
<!-- #endregion -->

```python id="G-iv9rijkaBf"
class MatrixFactorization(models.Model):
    def __init__(self, n_users, n_movies, n_factors, **kwargs):
        super(MatrixFactorization, self).__init__(**kwargs)
        self.n_users = n_users
        self.n_movies = n_movies
        self.n_factors = n_factors
        
        # We specify the size of the matrix,
        # the initializer (truncated normal distribution)
        # and the regularization type and strength (L2 with lambda = 1e-6)
        self.user_emb = layers.Embedding(n_users, 
                                         n_factors, 
                                         embeddings_initializer='he_normal',
                                         embeddings_regularizer=keras.regularizers.l2(1e-6),
                                         name='user_embedding')
        self.movie_emb = layers.Embedding(n_movies, 
                                          n_factors, 
                                          embeddings_initializer='he_normal',
                                          embeddings_regularizer=keras.regularizers.l2(1e-6),
                                          name='movie_embedding')
        
        # Embedding returns a 3D tensor with one dimension = 1, so we reshape it to a 2D tensor
        self.reshape = layers.Reshape((self.n_factors,))
        
        # Dot product of the latent vectors
        self.dot = layers.Dot(axes=1)

    def call(self, inputs):
        # Two inputs
        user, movie = inputs
        u = self.user_emb(user)
        u = self.reshape(u)
    
        m = self.movie_emb(movie)
        m = self.reshape(m)
        
        return self.dot([u, m])

n_factors = 50
model = MatrixFactorization(n_users, n_movies, n_factors)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.MeanSquaredError()
)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Bac1w7u49Ddx" outputId="bb733033-9aba-446b-a56d-f971897221d0"
try:
    model.summary()
except ValueError as e:
    print(e, type(e))
```

<!-- #region id="o-JSFnJA-1dz" -->
This is why building models via subclassing is a bit annoying - you can run into errors such as this. We'll fix it by calling the model with some fake data so it knows the shapes of the inputs.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="7wkIhqmO92Ca" outputId="6825be87-3d5e-4d25-e276-5043ff3a3bb9"
_ = model([np.array([1, 2, 3]), np.array([2, 88, 5])])
model.summary()
```

<!-- #region id="9Nxdrz7b_HOq" -->
We're going to expand our toolbox by introducing callbacks. Callbacks can be used to monitor our training progress, decay the learning rate, periodically save the weights or even stop early in case of detected overfitting. In Keras, they are really easy to use: you just create a list of desired callbacks and pass it to the model.fit method. It's also really easy to define your own by subclassing the Callback class. You can also specify when they will be triggered - the default is at the end of every epoch.

We'll use two: an early stopping callback which will monitor our loss and stop the training early if needed and TensorBoard, a utility for visualizing models, monitoring the training progress and much more.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="6N_Y7u5o-QpY" outputId="b4f299bc-25e2-4e38-b349-dc07184fb488"
callbacks = [
    keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor='val_loss',
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-2,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=2,
        verbose=1,
    ),
    keras.callbacks.TensorBoard(log_dir='logs')
]

history = model.fit(
    x=(df['user'].values, df['movie'].values),  # The model has two inputs!
    y=df['rating'],
    batch_size=128,
    epochs=20,
    verbose=1,
    validation_split=0.1,
    callbacks=callbacks
)
```

<!-- #region id="5QUGLmtw_eWA" -->
We see that we stopped early because the validation loss was not improving. Now, we'll open TensorBoard (it's a separate program called via command-line) to read the written logs and visualize the loss over all epochs. We will also look at how to visualize the model as a computational graph.
<!-- #endregion -->

```python id="_J-v9Hua_SV8"
# Run TensorBoard and specify the log dir
%tensorboard --logdir logs
```

<!-- #region id="Ldq0DwgI_lWC" -->
We've seen how easy it is to implement a recommender system with Keras and use a few utilities to make it easier to experiment. Note that this model is still quite basic and we could easily improve it: we could try adding a bias for each user and movie or adding non-linearity by using a sigmoid function and then rescaling the output. It could also be extended to use other features of a user or movie.
<!-- #endregion -->

<!-- #region id="-dpCn5hm_nUM" -->
Next, we'll try a bigger, more state-of-the-art model: a deep autoencoder.
<!-- #endregion -->

<!-- #region id="zTGdZ0b4_4rl" -->
We'll apply a more advanced algorithm to the same dataset as before, taking a different approach. We'll use a deep autoencoder network, which attempts to reconstruct its input and with that gives us ratings for unseen user / movie pairs.
<!-- #endregion -->

<!-- #region id="yIf926SkCOEp" -->
![](https://github.com/recohut/reco-static/raw/master/media/images/120222_algo.png)
<!-- #endregion -->

<!-- #region id="rSdY5NKQAYfI" -->
Preprocessing will be a bit different due to the difference in our model. Our autoencoder will take a vector of all ratings for a movie and attempt to reconstruct it. However, our input vector will have a lot of zeroes due to the sparsity of our data. We'll modify our loss so our model won't predict zeroes for those combinations - it will actually predict unseen ratings.

To facilitate this, we'll use the sparse tensor that TF supports. Note: to make training easier, we'll transform it to dense form, which would not work in larger datasets - we would have to preprocess the data in a different way or stream it into the model.
<!-- #endregion -->

<!-- #region id="HBcKm55rCVkk" -->
### Sparse representation and autoencoder reconstruction

![](https://github.com/recohut/reco-static/raw/master/media/images/120222_ae.png)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="OWybs9LyE8bB" outputId="1e108c11-a007-4942-bf0d-50409e4bbb1d"
df_raw.head()
```

```python id="M9jASOsh_gvU"
# Create a sparse tensor: at each user, movie location, we have a value, the rest is 0
sparse_x = tf.sparse.SparseTensor(indices=df[['movie', 'user']].values, values=df['rating'], dense_shape=(n_movies, n_users))

# Transform it to dense form and to float32 (good enough precision)
dense_x = tf.cast(tf.sparse.to_dense(tf.sparse.reorder(sparse_x)), tf.float32)

# Shuffle the data
x = tf.random.shuffle(dense_x, seed=42)
```

<!-- #region id="1j2-lFANEp8t" -->
Now, let's create the model. We'll have to specify the input shape. Because we have 9724 movies and only 610 users, we'll prefer to predict ratings for movies instead of users - this way, our dataset is larger.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="9s4qXdbuEpuX" outputId="45ac7925-b8f3-44dd-8e35-53fa7192b538"
class Encoder(layers.Layer):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.dense1 = layers.Dense(28, activation='selu', kernel_initializer='glorot_uniform')
        self.dense2 = layers.Dense(56, activation='selu', kernel_initializer='glorot_uniform')
        self.dense3 = layers.Dense(56, activation='selu', kernel_initializer='glorot_uniform')
        self.dropout = layers.Dropout(0.3)
        
    def call(self, x):
        d1 = self.dense1(x)
        d2 = self.dense2(d1)
        d3 = self.dense3(d2)
        return self.dropout(d3)
        
        
class Decoder(layers.Layer):
    def __init__(self, n, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.dense1 = layers.Dense(56, activation='selu', kernel_initializer='glorot_uniform')
        self.dense2 = layers.Dense(28, activation='selu', kernel_initializer='glorot_uniform')
        self.dense3 = layers.Dense(n, activation='selu', kernel_initializer='glorot_uniform')

    def call(self, x):
        d1 = self.dense1(x)
        d2 = self.dense2(d1)
        return self.dense3(d2)

n = n_users
inputs = layers.Input(shape=(n,))

encoder = Encoder()
decoder = Decoder(n)

enc1 = encoder(inputs)
dec1 = decoder(enc1)
enc2 = encoder(dec1)
dec2 = decoder(enc2)

model = models.Model(inputs=inputs, outputs=dec2, name='DeepAutoencoder')
model.summary()
```

<!-- #region id="aqXWA_TQGMDa" -->
Because our inputs are sparse, we'll need to create a modified mean squared error function. We have to look at which ratings are zero in the ground truth and remove them from our loss calculation (if we didn't, our model would quickly learn to predict zeros almost everywhere). We'll use masking - first get a boolean mask of non-zero values and then extract them from the result.
<!-- #endregion -->

```python id="G7AyGH8IFXAj"
def masked_mse(y_true, y_pred):
    mask = tf.not_equal(y_true, 0)
    se = tf.boolean_mask(tf.square(y_true - y_pred), mask)
    return tf.reduce_mean(se)

model.compile(
    loss=masked_mse,
    optimizer=keras.optimizers.Adam()
)
```

<!-- #region id="6O-Hqm_FGTmz" -->
The model training will be similar as before - we'll use early stopping and TensorBoard. Our batch size will be smaller due to the lower number of examples. Note that we are passing the same array for both x and y, because the autoencoder reconstructs its input.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="OHoZ3IuJGSrL" outputId="7923e0b0-7bc6-42ba-b3e6-c2bfe025291d"
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=1e-2,
        patience=5,
        verbose=1,
    ),
    keras.callbacks.TensorBoard(log_dir='logs')
]

model.fit(
    x, 
    x, 
    batch_size=16, 
    epochs=100, 
    validation_split=0.1,
    callbacks=callbacks
)
```

<!-- #region id="kkkhIjHhGhP5" -->
Let's visualize our loss and the model itself with TensorBoard.
<!-- #endregion -->

```python id="MMVp_HbwGdGQ"
%tensorboard --logdir logs
```

<!-- #region id="eSBFppW8Gkih" -->
That's it! We've seen how to use TensorFlow to implement recommender systems in a few different ways. I hope this short introduction has been informative and has prepared you to use TF on new problems. Thank you for your attention!
<!-- #endregion -->
