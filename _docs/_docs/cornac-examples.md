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

<!-- #region id="xyM7qLhOvaBn" -->
# Cornac Examples
> Multiple examples showcasing the features of cornac library and recommender concepts

- toc: true
- badges: true
- comments: true
- categories: [Cornac]
- author: "<a href='https://nbviewer.jupyter.org/github/PreferredAI/tutorials/tree/master/recommender-systems/'>Cornac</a>"
- image:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="2Rb7rRglxsAC" outputId="a666ec2f-ded8-4cc8-a31e-76cc03145461"
!pip install -q -U cornac
```

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="HBScIcOWx_IC" outputId="6e1d5e1e-3771-41a7-fe60-03f562b67ac3"
import cornac
cornac.__version__
```

```python id="N8w-V8wpClul"
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, sparse

from cornac.data import Reader
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit
from cornac.models import UserKNN, ItemKNN
from cornac.models import MF, NMF, BaselineOnly
from cornac.models import BPR, WMF
from cornac.models import GMF, MLP, NeuMF, VAECF, WMF

%tensorflow_version 1.x
```

<!-- #region id="T3B_RqHdCgJH" -->
## EDA
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="MVvs4_AqBtlN" outputId="94e578f3-02b4-455f-f16c-ed08e993e741"
df_100k3 = pd.read_csv('./data/ml-100k/ratings.csv',
                 usecols=["UserId","MovieId","Rating"])
df_100k3.columns = ["user_id", "item_id", "rating"]
df_100k3.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="9jsrpWuNWUEl" outputId="9f8eed9f-389f-4b39-bdad-fc13c04aeaec"
df_user_100k = pd.read_csv('./data/ml-100k/users.csv').set_index("UserID")
df_user_100k.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="p64287nKYYpa" outputId="be08472f-377f-4ab4-88f3-2b915e56cfbe"
df_item_100k = pd.read_csv('./data/ml-100k/items.csv').set_index("ItemID").drop(columns=["Video Release Date", "IMDb URL", "unknown"])
df_item_100k.head()
```

<!-- #region id="kWs2vodx_bU-" -->
Rating distribution
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 351} id="LmOzwRWPmSiK" outputId="333733cb-9d03-4c09-98f8-268083ebc225"
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
sns.countplot(x="rating", data=df, palette="ch:.25", ax=axes[0])
sns.boxplot(x="rating", data=df, palette="ch:.25", ax=axes[1])
```

<!-- #region id="Myv9wawVn0Ez" -->
Sparsity
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="92tt9AMlnRpV" outputId="f4931224-9e20-4750-df64-2dd3062de4dc"
def print_sparsity(df):
  n_users = df.user_id.nunique()
  n_items = df.item_id.nunique()
  n_ratings = len(df)
  rating_matrix_size = n_users * n_items
  sparsity = 1 - n_ratings / rating_matrix_size

  print(f"Number of users: {n_users}")
  print(f"Number of items: {n_items}")
  print(f"Number of available ratings: {n_ratings}")
  print(f"Number of all possible ratings: {rating_matrix_size}")
  print("-" * 40)
  print(f"SPARSITY: {sparsity * 100.0:.2f}%")

print_sparsity(df_100k3)
```

<!-- #region id="t2vfDKIhDXLJ" -->
For this MovieLens dataset, the data has been prepared in such a way that each user has at least 20 ratings. As a result, it's relatively dense as compared to many other recommendation datasets that are usually much sparser (often 99% or more).
<!-- #endregion -->

<!-- #region id="TJDj2NvWn26D" -->
Power Law Distribution
<!-- #endregion -->

```python id="tGIE5hqEDH0H"
item_rate_count = df_100k3.groupby("item_id")["user_id"].nunique().sort_values(ascending=False)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 350} id="W-npg0h7nkbK" outputId="680a2175-a119-4bd6-a8b2-86e818d7b4f8"
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
axes[0].bar(x=item_rate_count.index, height=item_rate_count.values, width=1.0, align="edge")
axes[0].set_xticks([])
axes[0].set(title="long tail of rating frequency", 
            xlabel="item ordered by decreasing frequency", 
            ylabel="#ratings")

count = item_rate_count.value_counts()
sns.scatterplot(x=np.log(count.index), y=np.log(count.values), ax=axes[1])
axes[1].set(title="log-log plot", xlabel="#ratings (log scale)", ylabel="#items (log scale)");
```

<!-- #region id="8vfKryGTZO-n" -->
User profiling
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="7KQyjHkIeA1N" outputId="78dffbbe-8760-46ce-82a0-3a78924e4747"
dataset = cornac.data.Dataset.from_uir(df.itertuples(index=False))
uknn_pearson = UserKNN(k=50, similarity="pearson", name="UserKNN-Pearson", verbose=False).fit(dataset)

uknn_pearson.train_set.uid_map[1]
```

```python colab={"base_uri": "https://localhost:8080/"} id="JIOlW8t3ZQp6" outputId="441114a9-b41c-4de7-8431-4327cd9be077"
def user_profiling(UID, df, user_df, item_df, TOPK=5):
  dataset = cornac.data.Dataset.from_uir(df.itertuples(index=False))
  uknn_pearson = UserKNN(k=50, similarity="pearson", name="UserKNN-Pearson", verbose=False).fit(dataset)

  rating_mat = uknn_pearson.train_set.matrix
  user_id2idx = uknn_pearson.train_set.uid_map
  user_idx2id = list(uknn_pearson.train_set.user_ids)
  item_id2idx = uknn_pearson.train_set.iid_map
  item_idx2id = list(uknn_pearson.train_set.item_ids)

  UIDX = uknn_pearson.train_set.uid_map[UID]

  print(f"UserID = {UID}")
  print("-" * 25)
  print(user_df.loc[UID])

  rating_arr = rating_mat[UIDX].A.ravel()
  top_rated_items = np.argsort(rating_arr)[-TOPK:]
  print(f"\nTOP {TOPK} RATED ITEMS BY USER {UID}:")
  print(item_df.loc[[int(item_idx2id[i]) for i in top_rated_items]])

user_profiling(2, df_100k3, df_user_100k, df_item_100k)
```

<!-- #region id="cmdaVm6-oRAm" -->
## Recomendations based on item popularity

Since some items are much more popular than the rest, intuitively many users may prefer these popular items. From that observation, it inspires a simple approach for providing recommendations based on popularity (i.e., number of ratings) of the items.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="G-ER0E9FpXlz" outputId="a86fe054-63ff-4d2d-e1c2-5bcc6f43e92a"
'''Let's do a simple experiment with the popularity approach. In this 
experiment, we will split the rating data into 5 folds for cross-validation. 
For each run, 4 folds will be used for training and the remaining fold will 
be used for evaluation. We measure the recommendation performance using 
Recall@20 metric.'''

def itempop_cornac(df):
  df = df.astype({'user_id':object, 'item_id':object})
  records = df.to_records(index=False)
  result = list(records)
  eval_method = cornac.eval_methods.CrossValidation(result, n_folds=5, seed=42) # 5-fold cross validation
  most_pop = cornac.models.MostPop() # recommender system based on item popularity
  rec_20 = cornac.metrics.Recall(k=20) # recall@20 metric
  cornac.Experiment(eval_method=eval_method, models=[most_pop], metrics=[rec_20]).run() # put everything together into an experiment

itempop_cornac(df_100k3)
```

<!-- #region id="9RFUnlYoLapQ" -->
## User-based collaborative filtering
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="cT44vKJ7ou6W" outputId="7ae77520-cb32-4e4a-83c0-4d27b70445fb"
def userknn_cornac(df):
  df = df.astype({'user_id':object, 'item_id':object})
  records = df.to_records(index=False)
  result = list(records)
  K = 50  # number of nearest neighbors
  VERBOSE = False
  SEED = 42
  uknn_cosine = UserKNN(k=K, similarity="cosine", name="UserKNN-Cosine", verbose=VERBOSE)
  uknn_cosine_mc = UserKNN(k=K, similarity="cosine", mean_centered=True, name="UserKNN-Cosine-MC", verbose=VERBOSE)
  uknn_pearson = UserKNN(k=K, similarity="pearson", name="UserKNN-Pearson", verbose=VERBOSE)
  uknn_pearson_mc = UserKNN(k=K, similarity="pearson", mean_centered=True, name="UserKNN-Pearson-MC", verbose=VERBOSE)
  ratio_split = RatioSplit(result, test_size=0.1, seed=SEED, verbose=VERBOSE)
  cornac.Experiment(eval_method=ratio_split,
                    models=[uknn_cosine, uknn_cosine_mc, uknn_pearson, uknn_pearson_mc],
                    metrics=[cornac.metrics.RMSE()],
                    ).run()

userknn_cornac(df_100k3)
```

<!-- #region id="pGdylRKeQJWI" -->
## Item-based collaborative filtering
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="VD24_0vMNZat" outputId="e1281edd-2b6b-471d-d82c-de4d08a6ef8e"
def itemknn_cornac(df):
  df = df.astype({'user_id':object, 'item_id':object})
  records = df.to_records(index=False)
  result = list(records)
  K = 50  # number of nearest neighbors
  VERBOSE = False
  SEED = 42
  iknn_cosine = ItemKNN(k=K, similarity="cosine", name="ItemKNN-Cosine", verbose=VERBOSE)
  iknn_cosine_mc = ItemKNN(k=K, similarity="cosine", mean_centered=True, name="ItemKNN-AdjustedCosine", verbose=VERBOSE)
  iknn_pearson = ItemKNN(k=K, similarity="pearson", name="ItemKNN-Pearson", verbose=VERBOSE)
  iknn_pearson_mc = ItemKNN(k=K, similarity="pearson", mean_centered=True, name="ItemKNN-Pearson-MC", verbose=VERBOSE)
  ratio_split = RatioSplit(result, test_size=0.1, seed=SEED, verbose=VERBOSE)
  cornac.Experiment(eval_method=ratio_split,
                    models=[iknn_cosine, iknn_cosine_mc, iknn_pearson, iknn_pearson_mc],
                    metrics=[cornac.metrics.RMSE()],
                    ).run()

itemknn_cornac(df_100k3)
```

<!-- #region id="gtZAznTeSKlK" -->
## Matrix factorization
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="HiB8C9xDQs7l" outputId="6f371b68-4601-4c10-f292-b797ad3ae85a"
def mf_cornac(df):
  df = df.astype({'user_id':object, 'item_id':object})
  records = df.to_records(index=False)
  result = list(records)
  K = 100
  VERBOSE = False
  SEED = 42
  lbd = 0.01
  baseline = BaselineOnly(max_iter=20, learning_rate=0.01, lambda_reg=lbd, verbose=VERBOSE)
  mf1 = MF(k=K, max_iter=20, learning_rate=0.01, lambda_reg=0.0, use_bias=False, verbose=VERBOSE, seed=SEED, name=f"MF(K={K})")
  mf2 = MF(k=K, max_iter=20, learning_rate=0.01, lambda_reg=lbd, use_bias=False, verbose=VERBOSE, seed=SEED, name=f"MF(K={K},lambda={lbd:.4f})")
  mf3 = MF(k=K, max_iter=20, learning_rate=0.01, lambda_reg=lbd, use_bias=True, verbose=VERBOSE, seed=SEED, name=f"MF(K={K},bias)")
  nmf = NMF(k=K, max_iter=200, learning_rate=0.01, use_bias=False, verbose=VERBOSE, seed=SEED, name=f"NMF(K={K})")
  ratio_split = RatioSplit(result, test_size=0.1, seed=SEED, verbose=VERBOSE)
  cornac.Experiment(eval_method=ratio_split,
                    models=[baseline, mf1, mf2, mf3, nmf],
                    metrics=[cornac.metrics.RMSE()],
                    ).run()

mf_cornac(df_100k3)
```

<!-- #region id="4BvNeQLFyjmQ" -->
## Implicit matrix factorization
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="K_oMK3l1t_YD" outputId="21a2563d-5e42-4528-c90c-14a5430f9942"
def implicitmf_cornac(df):
  df = df.astype({'user_id':object, 'item_id':object})
  records = df.to_records(index=False)
  result = list(records)

  K = 50
  VERBOSE = False
  SEED = 42
  mf = MF(k=K, max_iter=20, learning_rate=0.01, lambda_reg=0.01, verbose=VERBOSE, seed=SEED, name=f"MF(K={K})")
  wmf = WMF(k=K, max_iter=100, a=1.0, b=0.01, learning_rate=0.001, lambda_u=0.01, lambda_v=0.01, verbose=VERBOSE, seed=SEED, name=f"WMF(K={K})")
  bpr = BPR(k=K, max_iter=200, learning_rate=0.01, lambda_reg=0.001, verbose=VERBOSE, seed=SEED, name=f"BPR(K={K})")

  eval_metrics = [
  cornac.metrics.RMSE(), 
  cornac.metrics.AUC(),
  cornac.metrics.Precision(k=10),
  cornac.metrics.Recall(k=10),
  cornac.metrics.FMeasure(k=10),
  cornac.metrics.NDCG(k=[10, 20, 30]),
  cornac.metrics.MRR(),
  cornac.metrics.MAP()
  ]

  rs = RatioSplit(result, test_size=0.2, seed=SEED, verbose=VERBOSE)

  cornac.Experiment(eval_method=rs,
                    models=[mf, wmf, bpr],
                    metrics=eval_metrics,
                    ).run()

implicitmf_cornac(df_100k3)
```

<!-- #region id="ZQDxtSJN04_J" -->
As we can observe, the strength of the MF model is the ability to predict ratings well (lower RMSE). However, WMF model is designed to rank items, by fitting binary adoptions, thus it outperforms MF across all the listed ranking metrics.

BPR only tries to preserve the ordinal constraints without learning to predict the rating values. Thus, RMSE is not the right metric to evaluate BPR model. Minimizing the loss function of BPR is analogous to maximizing AUC, therefore, we expect BPR to do well on that metric.

Both BPR and WMF models are designed to obtain good performances in terms of ranking metrics. With reasonable efforts for hyper-parameter tuning, we should see comparable performance between the two models.
<!-- #endregion -->

<!-- #region id="lT5-e3QiNch9" -->
## Neural Collaborative Filtering
Neural collaborative filtering consists of a family of models developed based on neural networks to tackle the problem of collaborative filtering based on implicit feedback. The final model NeuMF is a combination of two components, namely Generalized Matrix Factorization (GMF) and Multi-Layer Perceptrons (MLP), which are also independent models respectively.
<!-- #endregion -->

```python id="wdZLRcBb4vuR"
GMF_FACTORS = 8  # @param
MLP_LAYERS = [32, 16, 8]  # @param
ACTIVATION = "tanh"  # @param ["tanh", "sigmoid", "relu", "leaky_relu"]
NEG_SAMPLES = 3  # @param
NUM_EPOCHS = 10  # @param 
BATCH_SIZE = 256  # @param
LEARNING_RATE = 0.001  # @param
```

```python colab={"base_uri": "https://localhost:8080/", "height": 287, "referenced_widgets": ["1ceb3d202e4b47e7ba222da7a0bf9299", "c57a2bcceaed403398e799e6d969671b", "87e562b9864b44e7acac1fdbe09557ce", "93fb5b2685894caeba3f7858df79b05f", "91452e4acc6c4dd98d40d415d91f352c", "d4dae8de13c64080a11b760ad649060c", "b52dec35097643f3affeb1059adf1f55", "c19b9124719a454eb77e6dba1dcce690"]} id="8UVTvr4FOgS_" outputId="8b02e1ff-77fb-4fe9-f65c-4376784576a0"
gmf = GMF(num_factors=GMF_FACTORS, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, 
          num_neg=NEG_SAMPLES, lr=LEARNING_RATE, seed=SEED, verbose=VERBOSE)
mlp = MLP(layers=MLP_LAYERS, act_fn=ACTIVATION, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, 
          num_neg=NEG_SAMPLES, lr=LEARNING_RATE, seed=SEED, verbose=VERBOSE)
neumf = NeuMF(num_factors=GMF_FACTORS, layers=MLP_LAYERS, act_fn=ACTIVATION, num_epochs=NUM_EPOCHS,
              num_neg=NEG_SAMPLES, batch_size=BATCH_SIZE, lr=LEARNING_RATE, seed=SEED, verbose=VERBOSE)

wmf = WMF(k=GMF_FACTORS, max_iter=200, learning_rate=0.001, seed=SEED, verbose=VERBOSE)

ml_100k = movielens.load_feedback(variant="100K", reader=Reader(bin_threshold=4.0))
ratio_split = RatioSplit(
  data=ml_100k, test_size=0.2, exclude_unknowns=True, seed=SEED, verbose=VERBOSE
)
ndcg_50 = cornac.metrics.NDCG(k=50)
rec_50 = cornac.metrics.Recall(k=50)

cornac.Experiment(
  eval_method=ratio_split,
  models=[gmf, mlp, neumf, wmf],
  metrics=[ndcg_50, rec_50],
).run()
```

<!-- #region id="PK0C0xs3Pm0I" -->
## Variational Autoencoder for Collaborative Filtering (VAECF)
Variational Autoencoders (VAE) is a type of autoencoders which is a neural network with auto-associative mapping of inputs. Normal autoencoders learns a determinisic latent representation of an input, while VAE learns a distribution of that representation. VAECF model extends VAE for the collaborative filtering problem with implicit feedback data.
<!-- #endregion -->

```python id="VGDQstOZPIY4"
NUM_FACTORS = 25  # @param
AE_LAYERS = [100, 50]  # @param
ACTIVATION = "tanh"  # @param ["tanh", "sigmoid", "relu", "leaky_relu"]
LIKELIHOOD = "bern"  # @param ["bern", "mult", "gaus", "pois"]
NUM_EPOCHS = 600  # @param 
BATCH_SIZE = 256  # @param
LEARNING_RATE = 0.001  # @param
```

```python colab={"base_uri": "https://localhost:8080/"} id="3pgddYbhPr3N" outputId="6b6c174d-ec1a-43e3-836b-0ae0090b379b"
vaecf = VAECF(k=NUM_FACTORS, autoencoder_structure=AE_LAYERS, act_fn=ACTIVATION,
              likelihood=LIKELIHOOD, n_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
              learning_rate=LEARNING_RATE, seed=SEED, verbose=VERBOSE, use_gpu=True)

wmf = WMF(k=NUM_FACTORS, max_iter=200, learning_rate=0.001, seed=SEED, verbose=VERBOSE)

ml_100k = movielens.load_feedback(variant="100K", reader=Reader(bin_threshold=4.0))
ratio_split = RatioSplit(
  data=ml_100k, test_size=0.2, exclude_unknowns=True, seed=SEED, verbose=VERBOSE
)

cornac.Experiment(
  eval_method=ratio_split, models=[vaecf, wmf], metrics=[rec_50, ndcg_50],
).run()
```

<!-- #region id="vcy5ObIUxm2X" -->
## Running Experiment
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="JaastgN_i3ry" outputId="e298df05-0bc1-49cc-d94c-216e43ad15db"
import cornac
from cornac.eval_methods import RatioSplit
from cornac.models import MF, PMF, BPR
from cornac.metrics import MAE, RMSE, Precision, Recall, NDCG, AUC

# load the built-in MovieLens 100K and split the data based on ratio
ml_100k = cornac.datasets.movielens.load_feedback()
rs = RatioSplit(data=ml_100k, test_size=0.2, rating_threshold=4.0, seed=123)

# initialize models, here we are comparing: Biased MF, PMF, and BPR
models = [
    MF(k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02, use_bias=True, seed=123),
    PMF(k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=123),
    BPR(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123),
]

# define metrics to evaluate the models
metrics = [MAE(), RMSE(), Precision(k=10), Recall(k=10), NDCG(k=10), AUC()]

# put it together in an experiment, voilà!
cornac.Experiment(eval_method=rs, models=models, metrics=metrics, user_based=True).run()
```

<!-- #region id="TolEnvXox4Wh" -->
## PMF ratio
Example to run Probabilistic Matrix Factorization (PMF) model with Ratio Split evaluation strategy
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ljDp8ol7xtqy" outputId="f5cb41da-a871-47d4-e0c5-080f1514d6e4"
import cornac
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit
from cornac.models import PMF


# Load the MovieLens 100K dataset
ml_100k = movielens.load_feedback()

# Instantiate an evaluation method.
ratio_split = RatioSplit(
    data=ml_100k, test_size=0.2, rating_threshold=4.0, exclude_unknowns=False
)

# Instantiate a PMF recommender model.
pmf = PMF(k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001)

# Instantiate evaluation metrics.
mae = cornac.metrics.MAE()
rmse = cornac.metrics.RMSE()
rec_20 = cornac.metrics.Recall(k=20)
pre_20 = cornac.metrics.Precision(k=20)

# Instantiate and then run an experiment.
cornac.Experiment(
    eval_method=ratio_split,
    models=[pmf],
    metrics=[mae, rmse, rec_20, pre_20],
    user_based=True,
).run()
```

<!-- #region id="mVH9z_KCyOks" -->
## Given data
Example on how to train and evaluate a model with provided train and test sets
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="J0uUBh-Px8h_" outputId="f594968e-c079-4cf9-d0e0-9eeab70740c9"
from cornac.data import Reader
from cornac.eval_methods import BaseMethod
from cornac.models import MF
from cornac.metrics import MAE, RMSE
from cornac.utils import cache


# Download MovieLens 100K provided train and test sets
reader = Reader()
train_data = reader.read(
    cache(url="http://files.grouplens.org/datasets/movielens/ml-100k/u1.base")
)
test_data = reader.read(
    cache(url="http://files.grouplens.org/datasets/movielens/ml-100k/u1.test")
)

# Instantiate a Base evaluation method using the provided train and test sets
eval_method = BaseMethod.from_splits(
    train_data=train_data, test_data=test_data, exclude_unknowns=False, verbose=True
)

# Instantiate the MF model
mf = MF(
    k=10,
    max_iter=25,
    learning_rate=0.01,
    lambda_reg=0.02,
    use_bias=True,
    early_stop=True,
    verbose=True,
)

# Evaluation
test_result, val_result = eval_method.evaluate(
    model=mf, metrics=[MAE(), RMSE()], user_based=True
)
print(test_result)
```

<!-- #region id="FrscRmQiyZEv" -->
## C2PF
<!-- #endregion -->

<!-- #region id="w8lgdJSDzzTn" -->
Collaborative Context Poisson Factorization (C2PF) with Amazon Office dataset.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="y4hzd9qCySw6" outputId="136a548e-9373-4cb3-bb74-926a6c30eafe"
"""Fit to and evaluate C2PF [1] on the Office Amazon dataset.
[1] Salah, Aghiles, and Hady W. Lauw. A Bayesian Latent Variable Model of User Preferences with Item Context. \
    In IJCAI, pp. 2667-2674. 2018.
"""

from cornac.data import GraphModality
from cornac.eval_methods import RatioSplit
from cornac.experiment import Experiment
from cornac import metrics
from cornac.models import C2PF
from cornac.datasets import amazon_office as office


# In addition to user-item feedback, C2PF integrates item-to-item contextual relationships
# The necessary data can be loaded as follows
ratings = office.load_feedback()
contexts = office.load_graph()

# Instantiate a GraphModality, it makes it convenient to work with graph (network) auxiliary information
# For more details, please refer to the tutorial on how to work with auxiliary data
item_graph_modality = GraphModality(data=contexts)

# Define an evaluation method to split feedback into train and test sets
ratio_split = RatioSplit(
    data=ratings,
    test_size=0.2,
    rating_threshold=3.5,
    exclude_unknowns=True,
    verbose=True,
    item_graph=item_graph_modality,
)

# Instantiate C2PF
c2pf = C2PF(k=100, max_iter=80, variant="c2pf")

# Evaluation metrics
ndcg = metrics.NDCG(k=-1)
mrr = metrics.MRR()
rec = metrics.Recall(k=20)
pre = metrics.Precision(k=20)

# Put everything together into an experiment and run it
Experiment(eval_method=ratio_split, models=[c2pf], metrics=[ndcg, mrr, rec, pre]).run()
```

<!-- #region id="OM2G66bhzibY" -->
## Hyperparameters
<!-- #endregion -->

```python id="VwIvk9C6zfM-" colab={"base_uri": "https://localhost:8080/"} outputId="25ba9fb9-cf12-4764-b469-f6d7eca8d23d"
"""Example for hyper-parameter searching with Matrix Factorization"""

import numpy as np
import cornac
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit
from cornac.hyperopt import Discrete, Continuous
from cornac.hyperopt import GridSearch, RandomSearch


# Load MovieLens 100K ratings
ml_100k = movielens.load_feedback(variant="100K")

# Define an evaluation method to split feedback into train, validation and test sets
ratio_split = RatioSplit(data=ml_100k, test_size=0.1, val_size=0.1, verbose=True)

# Instantiate MAE and RMSE for evaluation
mae = cornac.metrics.MAE()
rmse = cornac.metrics.RMSE()

# Define a base MF model with fixed hyper-parameters
mf = cornac.models.MF(max_iter=20, learning_rate=0.01, early_stop=True, verbose=True)

# Wrap MF model inside GridSearch along with the searching space
gs_mf = GridSearch(
    model=mf,
    space=[
        Discrete("k", [10, 30, 50]),
        Discrete("use_bias", [True, False]),
        Discrete("lambda_reg", [1e-1, 1e-2, 1e-3, 1e-4]),
    ],
    metric=rmse,
    eval_method=ratio_split,
)

# Wrap MF model inside RandomSearch along with the searching space, try 30 times
rs_mf = RandomSearch(
    model=mf,
    space=[
        Discrete("k", [10, 30, 50]),
        Discrete("use_bias", [True, False]),
        Continuous("lambda_reg", low=1e-4, high=1e-1),
    ],
    metric=rmse,
    eval_method=ratio_split,
    n_trails=30,
)

# Put everything together into an experiment and run it
cornac.Experiment(
    eval_method=ratio_split,
    models=[gs_mf, rs_mf],
    metrics=[mae, rmse],
    user_based=False,
).run()
```

<!-- #region id="5X_TsyJF0c7s" -->
## Social BPR
Example for Social Bayesian Personalized Ranking (SBPR) with Epinions dataset
<!-- #endregion -->

```python id="MI4gRJCg0bio" colab={"base_uri": "https://localhost:8080/"} outputId="8030df8c-5266-419d-df07-2932ec76fdab"
import cornac
from cornac.data import Reader, GraphModality
from cornac.datasets import epinions
from cornac.eval_methods import RatioSplit


# SBPR integrates user social network into Bayesian Personalized Ranking.
# The necessary data can be loaded as follows
feedback = epinions.load_feedback(
    Reader(bin_threshold=4.0)
)  # feedback is binarised (turned into implicit) using Reader.
trust = epinions.load_trust()

# Instantiate a GraphModality, it makes it convenient to work with graph (network) auxiliary information
# For more details, please refer to the tutorial on how to work with auxiliary data
user_graph_modality = GraphModality(data=trust)

# Define an evaluation method to split feedback into train and test sets
ratio_split = RatioSplit(
    data=feedback,
    test_size=0.1,
    rating_threshold=0.5,
    exclude_unknowns=True,
    verbose=True,
    user_graph=user_graph_modality,
)

# Instantiate SBPR model
sbpr = cornac.models.SBPR(
    k=10,
    max_iter=50,
    learning_rate=0.001,
    lambda_u=0.015,
    lambda_v=0.025,
    lambda_b=0.01,
    verbose=True,
)

# Use Recall@10 for evaluation
rec_10 = cornac.metrics.Recall(k=10)

# Put everything together into an experiment and run it
cornac.Experiment(eval_method=ratio_split, models=[sbpr], metrics=[rec_10]).run()
```
