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

<!-- #region id="NzKpIG2P7rCx" -->
# Tensorflow Recommenders - Implementation on Retail Data
> TFRS Retrieval, ranking, time and text feature embeddings, and multi-task modeling on Olist retail dataset

- toc: true
- badges: true
- comments: true
- categories: [TFRS, Tensorflow, Retail, Olist, MultiTask]
- author: "<a href='https://github.com/fickaz/TFRS-on-Retail-Data/tree/main'>Taufik Azri</a>"
- image:
<!-- #endregion -->

<!-- #region id="ldsMmwkkbFfj" -->
## Introduction
<!-- #endregion -->

<!-- #region id="4qUeXNabhENS" -->
<!-- #endregion -->

<!-- #region id="73ngMKyL8ehG" -->
### Overview
- Objective: To demonstrate TensorFlow 2.0 TFRS recommenders library to build a recommendation system on a customer retail data.
- Data source: https://www.kaggle.com/olistbr/brazilian-ecommerce/home/
- Benefit: Flexible model, ability to add different features and specify and adjust model complexity easily.

### Theory
Two types of recommendation model-- Retrieval and Ranking.
- Retrieval: The retrieval stage is responsible for selecting an initial set of hundreds of candidates from all possible candidates. The main objective of this model is to efficiently weed out all candidates that the user is not interested in. Because the retrieval model may be dealing with millions of candidates, it has to be computationally efficient. Retrieval can be computationally more efficient because it only returns smaller set of items a user would strongly interested.
- Ranking: The ranking stage takes the outputs of the retrieval model and fine-tunes them to select the best possible handful of recommendations. Its task is to narrow down the set of items the user may be interested in to a shortlist of likely candidates.
<!-- #endregion -->

<!-- #region id="p1rpU624hKri" -->
Built with TensorFlow 2.x, TFRS makes it possible to:

- Build and evaluate flexible **[candidate nomination models](https://research.google/pubs/pub48840/)**;
- Freely incorporate item, user, and context **[information](https://tensorflow.org/recommenders/examples/featurization)** into recommendation models;
- Train **[multi-task models](https://tensorflow.org/recommenders/examples/multitask)** that jointly optimize multiple recommendation objectives;
- Efficiently serve the resulting models using **[TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)**.
- [Multi-task learning](https://research.google/pubs/pub47842/), [feature cross modeling](https://arxiv.org/abs/1708.05123), [self-supervised learning](https://arxiv.org/abs/2007.12865), and state-of-the-art efficient [approximate nearest neighbours computation](https://ai.googleblog.com/2020/07/announcing-scann-efficient-vector.html).
<!-- #endregion -->

<!-- #region id="DOxZmD-VaEdu" -->
### Outline
1. Retrieval model
2. Ranking model
3. Adding text and timestamp embedding
4. Multitask recommendation, combining retrieval and ranking
5. Add more features using Cross Network.
<!-- #endregion -->

<!-- #region id="Ge2APNZwbHF1" -->
## Setup
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="hOL4L1OCyjTj" outputId="062f6651-7741-403b-baf8-e2a8a6762ec1"
!pip install -q tensorflow-recommenders
```

```python id="AOf0GG14ZoDX"
import os
import pprint
import tempfile
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from typing import Dict, Text

import tensorflow as tf
import tensorflow_recommenders as tfrs
```

<!-- #region id="UosmyIAPbIS1" -->
## Data Loading
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="uORj0Lar5hrG" outputId="f1556b32-7fd2-4ff9-f63b-522e594dad31"
!pip install -q -U kaggle
!pip install --upgrade --force-reinstall --no-deps kaggle
!mkdir ~/.kaggle
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d olistbr/brazilian-ecommerce
```

```python colab={"base_uri": "https://localhost:8080/"} id="Za4XrT-6bJWY" outputId="f0ce65b3-df58-4a0c-c37e-55e5199441f1"
!unzip brazilian-ecommerce.zip
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="4kzXioHdedga" outputId="834681a1-3a64-49e8-a9b0-8225acbac7dc"
import glob
files = sorted(glob.glob('/content/*.csv'))
dfs = [pd.read_csv(data) for data in files]
for i, x in enumerate(dfs):
  print(f"\n\ndfs {i}: {files[i].split('/')[-1].split('.')[0]}\n")
  display(x.head())
```

<!-- #region id="uRci8wzMNOPT" -->
## Data Preparation
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="_GCUFv9reddm" outputId="ceb6209b-ecca-4c28-939f-1616188d6293"
df11 = pd.merge(dfs[2], dfs[5], how='outer', on='order_id')[['product_id','customer_id','order_purchase_timestamp']]
df12 = pd.merge(dfs[6], dfs[8], how='left', on='product_category_name')[['product_id','product_category_name_english']]
df1 = pd.merge(df11, df12, how='left', on='product_id')
df1['sku'] = df1.groupby(['product_category_name_english']).cumcount().astype('str')
df1['product_id'] = df1['product_category_name_english'] + '_' + df1['sku']
df1['implicit_interaction_weight'] = 1
df1['order_purchase_timestamp'] = df1['order_purchase_timestamp'].map(lambda x: pd.to_datetime(x).timestamp())
df1 = df1[['customer_id', 'product_id', 'implicit_interaction_weight', 'order_purchase_timestamp']]
df1.columns = ['USERID','ITEMID','RATING','TIMESTAMP']
df1.dropna(inplace=True)
df1
```

```python colab={"base_uri": "https://localhost:8080/"} id="_MtcW265zt0r" outputId="7a29b4d0-b747-439a-e198-52e15a3dbef4"
interactions_dict = df1.groupby(['USERID', 'ITEMID'])['RATING'].count().reset_index()
interactions_dict = {name: np.array(value) for name, value in interactions_dict.items()}
interactions = tf.data.Dataset.from_tensor_slices(interactions_dict)

for x in interactions.take(5): print(x)

interactions = interactions.map(lambda x: {
    'USERID' : x['USERID'], 
    'ITEMID' : x['ITEMID'], 
    'RATING' : float(x['RATING']),
})
```

```python id="xaDMgvUWeYwm"
items_dict = df1[['ITEMID']].drop_duplicates()
items_dict = {name: np.array(value) for name, value in items_dict.items()}
items = tf.data.Dataset.from_tensor_slices(items_dict)

items = items.map(lambda x: x['ITEMID'])
```

```python id="oKxYOXCzrea5"
### get unique item and user id's as a lookup table
unique_item_titles = np.unique(np.concatenate(list(items.batch(1000))))
unique_user_ids = np.unique(np.concatenate(list(interactions.batch(1_000).map(lambda x: x["USERID"]))))

# Randomly shuffle data and split between train and test.
tf.random.set_seed(42)
shuffled = interactions.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(60_000)
test = shuffled.skip(60_000).take(20_000)
```

<!-- #region id="44S48b-H3ewY" -->
## Retrieval Model
<!-- #endregion -->

<!-- #region id="a8ESTThw7Kse" -->
There are five important component of the query and candicate tower: candidate model (item_model), querty model (user_model), metrics, task, and compute loss.
<!-- #endregion -->

```python id="qfcgl8Vj3dTm"
class RetailModel(tfrs.Model):

    def __init__(self, user_model, item_model):
      super().__init__()
      
      ### Candidate model (item)
      ### This is Keras preprocessing layers to first convert user ids to integers, 
      ### and then convert those to user embeddings via an Embedding layer. 
      ### We use the list of unique user ids we computed earlier as a vocabulary:
      item_model = tf.keras.Sequential([
                                      tf.keras.layers.experimental.preprocessing.StringLookup(
                                      vocabulary=unique_item_titles, mask_token=None),
                                      tf.keras.layers.Embedding(len(unique_item_titles) + 1, embedding_dimension)
                                      ])
      ### we pass the embedding layer into item model
      self.item_model: tf.keras.Model = item_model
          
      ### Query model (users)    
      user_model = tf.keras.Sequential([
                                      tf.keras.layers.experimental.preprocessing.StringLookup(
                                      vocabulary=unique_user_ids, mask_token=None),
                                      # We add an additional embedding to account for unknown tokens.
                                      tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
                                      ])
      self.user_model: tf.keras.Model = user_model
      
      ### for retrieval model. we take top-k accuracy as metrics
      metrics = tfrs.metrics.FactorizedTopK(candidates=items.batch(128).map(item_model))
      
      # define the task, which is retrieval
      task = tfrs.tasks.Retrieval(metrics=metrics)
      
      self.task: tf.keras.layers.Layer = task

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
      # We pick out the user features and pass them into the user model.
      user_embeddings = self.user_model(features["USERID"])
      # And pick out the movie features and pass them into the movie model,
      # getting embeddings back.
      positive_movie_embeddings = self.item_model(features["ITEMID"])

      # The task computes the loss and the metrics.
      return self.task(user_embeddings, positive_movie_embeddings)
```

```python colab={"base_uri": "https://localhost:8080/"} id="qcAXUSfT39IG" outputId="0a519872-eb30-4613-bfbc-9fc003fdcbcd"
### Fitting and evaluating

### we choose the dimensionality of the query and candicate representation.
embedding_dimension = 32

## we pass the model, which is the same model we created in the query and candidate tower, into the model
item_model = tf.keras.Sequential([
                                tf.keras.layers.experimental.preprocessing.StringLookup(
                                vocabulary=unique_item_titles, mask_token=None),
                                tf.keras.layers.Embedding(len(unique_item_titles) + 1, embedding_dimension)
                                ])

user_model = tf.keras.Sequential([
                                tf.keras.layers.experimental.preprocessing.StringLookup(
                                vocabulary=unique_user_ids, mask_token=None),
                                # We add an additional embedding to account for unknown tokens.
                                tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
                                ])

model = RetailModel(user_model, item_model)

# a smaller learning rate may make the model move slower and prone to overfitting, so we stick to 0.1
# other optimizers, such as SGD and Adam, are listed here https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()

## fit the model with ten epochs
model_hist = model.fit(cached_train, epochs=10)

#evaluate the model
model.evaluate(cached_test, return_dict=True)
```

```python colab={"base_uri": "https://localhost:8080/"} id="3yvfAWTa5OJ6" outputId="cb2ef58c-7ae9-41d8-c553-8ef9f19bc140"
model.evaluate(cached_test, return_dict=True)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 312} id="T48UBb5A6Zv2" outputId="2e231d6d-711e-45a3-c9ee-112b45bd080b"
# num_validation_runs = len(one_layer_history.history["val_factorized_top_k/top_100_categorical_accuracy"])
epochs = [i for i in range(10)]

plt.plot(epochs, model_hist.history["factorized_top_k/top_100_categorical_accuracy"], label="accuracy")
plt.title("Accuracy vs epoch")
plt.xlabel("epoch")
plt.ylabel("Top-100 accuracy");
plt.legend()
```

```python colab={"base_uri": "https://localhost:8080/"} id="TKCDYWVO6dnG" outputId="d5161d99-f90b-460f-d7eb-4f44ddc7601d"
# Create a model that takes in raw query features, and
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
# recommends items out of the entire dataset.
index.index(items.batch(100).map(model.item_model), items)

# Get recommendations.
j = str(40)
_, items = index(tf.constant([j]))
print(f"Recommendations for user %s: {items[0]}" %(j))
```

<!-- #region id="Y09nBvYJ7CLQ" -->
There you are, our first simple yet effective recommendation engine using retrieval task. But what about ranking? can we rank all the items for best to worst, only then run retrieval task to retrieve selected items from the short list? Now we can explore another type of recommendation task: ranking.
<!-- #endregion -->

<!-- #region id="C_IBmLuxC-x2" -->
## Ranking Model
<!-- #endregion -->

<!-- #region id="XMDlO4aKDFSL" -->
Ranking model is able to assist retrieval by ranking all the items from highest to lowest, predcting a probablity that a user may or may not like it. Ranking model is useful to filter out items that are not relevant for the user before retrieval task, making retrieval task much more accurate and efficient.

Here, many embedding layers works similarly with retrieval model, with addition of multiple hidden layers under Sequential latyers, where we can stack multiple dense layers. We split the query and candidate tower separately, and call them later into the model.
<!-- #endregion -->

```python id="nqsYVhXZDAm2"
class RankingModel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        embedding_dimension = 32

        # Compute embeddings for users.
        self.user_embeddings = tf.keras.Sequential([
          tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=unique_user_ids, mask_token=None),
          tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
        ])

        # Compute embeddings for movies.
        self.item_embeddings = tf.keras.Sequential([
          tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=unique_item_titles, mask_token=None),
          tf.keras.layers.Embedding(len(unique_item_titles) + 1, embedding_dimension)
        ])

        # Compute predictions.
        self.ratings = tf.keras.Sequential([
          # Learn multiple dense layers.
          tf.keras.layers.Dense(256, activation="relu"),
          tf.keras.layers.Dense(64, activation="relu"),
          # Make rating predictions in the final layer.
          tf.keras.layers.Dense(1)
  ])

    def call(self, inputs):

        user_id, item_id = inputs

        user_embedding = self.user_embeddings(user_id)
        item_embedding = self.item_embeddings(item_id)

        return self.ratings(tf.concat([user_embedding, item_embedding], axis=1))
```

<!-- #region id="wQ6uPSZ0DpvA" -->
This model takes user ids and item ids, and outputs a predicted rating, for example:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="-HkN_XVoDqPE" outputId="b5cabe5b-8dff-4600-cdd1-07de58bb1132"
RankingModel()((["f6dd3ec061db4e3987629fe6b26e5cce"], ["pet_shop_0"]))
```

```python id="HUQ1uAWHEJve"
class RetailModel(tfrs.models.Model):

    def __init__(self):
        super().__init__()
        self.ranking_model: tf.keras.Model = RankingModel()
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
          loss = tf.keras.losses.MeanSquaredError(),
          metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        rating_predictions = self.ranking_model(
            (features["USERID"], features["ITEMID"]))

        # The task computes the loss and the metrics.
        return self.task(labels=features["RATING"], predictions=rating_predictions)
```

```python colab={"base_uri": "https://localhost:8080/"} id="InDnIB3JEUdV" outputId="3415ad4e-662c-41a0-a137-46a36e3d38c7"
model = RetailModel()

model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.5))

cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()

model.fit(cached_train, epochs=100)
```

```python colab={"base_uri": "https://localhost:8080/"} id="R2FSXa6fEUZk" outputId="fb0c87aa-9652-4545-e1a4-abc14dc86b23"
model.evaluate(cached_test, return_dict=True)
```

<!-- #region id="hvjpk_oyEcfF" -->
The RMSE is not very good, which we shall see how we can improve it by adding more features and combining ranking and retrieval model together.
<!-- #endregion -->

<!-- #region id="iOl88rdWMqtd" -->
## Adding Context
Adding Timestamp and Text embeddings
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="_93rwbdCEUVL" outputId="9853264f-70e8-4d87-e31a-a729d4723292"
interactions_dict = df1.groupby(['USERID', 'ITEMID', 'TIMESTAMP'])['RATING'].count().reset_index()
interactions_dict = {name: np.array(value) for name, value in interactions_dict.items()}
interactions = tf.data.Dataset.from_tensor_slices(interactions_dict)

for x in interactions.take(5): print(x)

interactions = interactions.map(lambda x: {
    'USERID' : x['USERID'], 
    'ITEMID' : x['ITEMID'], 
    'RATING' : float(x['RATING']),
    'TIMESTAMP' : x['TIMESTAMP'],
})
```

```python id="KbfzbgIbNVtB"
items_dict = df1[['ITEMID']].drop_duplicates()
items_dict = {name: np.array(value) for name, value in items_dict.items()}
items = tf.data.Dataset.from_tensor_slices(items_dict)

items = items.map(lambda x: x['ITEMID'])
```

<!-- #region id="XHRL37wANiXV" -->
timestamp is an exmaple of continuous features, which needs to be rescaled, or otherwise it will be too large for the model. there are other methods to reduce the size of the timestamp, ,such as standardization and normalization here we use discretization, which puts them into buckets of categorical features,
<!-- #endregion -->

```python id="32XGSiMdNbWS"
timestamps = np.concatenate(list(interactions.map(lambda x: x["TIMESTAMP"]).batch(100)))
max_timestamp = timestamps.max()
min_timestamp = timestamps.min()
timestamp_buckets = np.linspace(
    min_timestamp, max_timestamp, num=1000,)

item_ids = interactions.batch(10_000).map(lambda x: x["ITEMID"])
user_ids = interactions.batch(10_000).map(lambda x: x["USERID"])

unique_item_ids = np.unique(np.concatenate(list(item_ids)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))
```

```python id="m5qEXLMUN9R9"
tf.random.set_seed(42)
shuffled = interactions.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(60_000)
test = shuffled.skip(60_000).take(20_000)

cached_train = train.shuffle(100_000).batch(2048)
cached_test = test.batch(4096).cache()
```

<!-- #region id="enlr1TvoPw3-" -->
We split the query and candidate model separately to allow more stacked embedding layers before we pass it into the model. In the user model (query model), in addition to user embedding, we also add timestamp embedding.
<!-- #endregion -->

```python id="EzLslJVrPp5a"
### user model

class UserModel(tf.keras.Model):

    def __init__(self, use_timestamps):
        super().__init__()

        self._use_timestamps = use_timestamps

        ## embed user id from unique_user_ids
        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, 32),
        ])

        ## embed timestamp
        if use_timestamps:
            self.timestamp_embedding = tf.keras.Sequential([
              tf.keras.layers.experimental.preprocessing.Discretization(timestamp_buckets.tolist()),
              tf.keras.layers.Embedding(len(timestamp_buckets) + 1, 32),
            ])
            self.normalized_timestamp = tf.keras.layers.experimental.preprocessing.Normalization()

            self.normalized_timestamp.adapt(timestamps)

    def call(self, inputs):
        if not self._use_timestamps:
              return self.user_embedding(inputs["USERID"])

        ## all features here
        return tf.concat([
            self.user_embedding(inputs["USERID"]),
            self.timestamp_embedding(inputs["TIMESTAMP"]),
            self.normalized_timestamp(inputs["TIMESTAMP"]),
        ], axis=1)
```

<!-- #region id="Z5wg7SVAQGKZ" -->
For the candidate model, we want the model to learn from the text features too by processing the text features that is able to learn words that are similar to each other. It can also identify OOV (out of Vocabulary) word, so if we are predicing a new item, the model can calculate them appropriately.

Below, the item name will be transformated by tokenization (splitting into constituent words or word-pieces), followed by vocabulary learning, then followed by an embedding.


<!-- #endregion -->

```python id="ADFkvoNtPp15"
### candidate model

class ItemModel(tf.keras.Model):

    def __init__(self):
        super().__init__()

        max_tokens = 10_000

        ## embed title from unique_item_ids
        self.title_embedding = tf.keras.Sequential([
          tf.keras.layers.experimental.preprocessing.StringLookup(
              vocabulary=unique_item_ids, mask_token=None),
          tf.keras.layers.Embedding(len(unique_item_ids) + 1, 32)
        ])

        ## processing text features: item title vectorizer (see self.title_vectorizer)
        self.title_vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
            max_tokens=max_tokens)

        ## we apply title vectorizer to items
        self.title_text_embedding = tf.keras.Sequential([
          self.title_vectorizer,
          tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
          tf.keras.layers.GlobalAveragePooling1D(),
        ])

        self.title_vectorizer.adapt(items)

    def call(self, titles):
        return tf.concat([
            self.title_embedding(titles),
            self.title_text_embedding(titles),
        ], axis=1)
```

<!-- #region id="TN7GQh_FQpbM" -->
With both UserModel and ItemModel defined, we can put together a combined model and implement our loss and metrics logic.

Note that we also need to make sure that the query model and candidate model output embeddings of compatible size. Because we'll be varying their sizes by adding more features, the easiest way to accomplish this is to use a dense projection layer after each model:
<!-- #endregion -->

```python id="52mV0P7vPpyN"
class RetailModel(tfrs.models.Model):

    def __init__(self, use_timestamps):
        super().__init__()
        
        ## query model is user model
        self.query_model = tf.keras.Sequential([
          UserModel(use_timestamps),
          tf.keras.layers.Dense(32)
        ])
        
        ## candidate model is the item model
        self.candidate_model = tf.keras.Sequential([
          ItemModel(),
          tf.keras.layers.Dense(32)
        ])
        
        ## retrieval task, choose metrics
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=items.batch(128).map(self.candidate_model),
            ),
        )

    def compute_loss(self, features, training=False):
        # We only pass the user id and timestamp features into the query model. This
        # is to ensure that the training inputs would have the same keys as the
        # query inputs. Otherwise the discrepancy in input structure would cause an
        # error when loading the query model after saving it.
        
        query_embeddings = self.query_model({
            "USERID": features["USERID"],
            "TIMESTAMP": features["TIMESTAMP"],
        })
        
        item_embeddings = self.candidate_model(features["ITEMID"])

        return self.task(query_embeddings, item_embeddings)
```

<!-- #region id="_c_3EzxbQ6ne" -->
> Note: Baseline is with no timestamp feature.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="bBhWoMJtQ4gc" outputId="a2c73265-8f61-4c34-c30d-11ebcfc24c54"
model = RetailModel(use_timestamps=False)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
model.fit(cached_train, epochs=3)
```

```python colab={"base_uri": "https://localhost:8080/"} id="3_LGm5ZJRXsB" outputId="b081514d-ee96-4562-c431-18aedcde18c3"
model.evaluate(cached_test, return_dict=True)
```

<!-- #region id="CL-lraRjRPki" -->
Including time into the model:

Do the result change if we add time features?
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="hP5JGQXRRMx7" outputId="8c5f421b-1da1-49cf-f83b-424cb9525c46"
model =x RetailModel(use_timestamps=True)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
model.fit(cached_train, epochs=3)
```

```python colab={"base_uri": "https://localhost:8080/"} id="wb9rJdWMRaQY" outputId="19a3faeb-56b5-4401-f717-cf44107b603a"
model.evaluate(cached_test, return_dict=True)
```

<!-- #region id="MmPAVEhVRexT" -->
> Note: Eventhough we only run it at three epochs, we can see accuracy increase as we add time into the model.
<!-- #endregion -->

<!-- #region id="hNd03e3Qc_P2" -->
## Multi-Task Model with ReLU-based DNN
<!-- #endregion -->

<!-- #region id="cTkDkTyWdKnN" -->
The new component here is that - since we have two tasks and two losses - we need to decide on how important each loss is. We can do this by giving each of the losses a weight, and treating these weights as hyperparameters. If we assign a large loss weight to the rating task, our model is going to focus on predicting ratings (but still use some information from the retrieval task); if we assign a large loss weight to the retrieval task, it will focus on retrieval instead.
<!-- #endregion -->

```python id="XcipDXldRg2a"
class Model(tfrs.models.Model):

    def __init__(self,
                 rating_weight: float, retrieval_weight: float) -> None:
        # We take the loss weights in the constructor: this allows us to instantiate
        # several model objects with different loss weights.

        super().__init__()

        embedding_dimension = 32

        # item models.
        self.item_model: tf.keras.layers.Layer = tf.keras.Sequential([
          tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=unique_item_ids, mask_token=None),
          tf.keras.layers.Embedding(len(unique_item_ids) + 1, embedding_dimension)
        ])
            
        ## user model    
        self.user_model: tf.keras.layers.Layer = tf.keras.Sequential([
          tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=unique_user_ids, mask_token=None),
          tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
        ])

        # A small model to take in user and item embeddings and predict ratings.
        # We can make this as complicated as we want as long as we output a scalar
        # as our prediction.
        
        ## this is Relu-Based DNN
        self.rating_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1),
        ])

        # rating and retrieval task.
        self.rating_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )
            
        self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=items.batch(128).map(self.item_model)
            )
        )

        # The loss weights.
        self.rating_weight = rating_weight
        self.retrieval_weight = retrieval_weight

    def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.user_model(features["USERID"])
        
        # And pick out the item features and pass them into the item model.
        item_embeddings = self.item_model(features["ITEMID"])

        return (
            user_embeddings,
            item_embeddings,
            # We apply the multi-layered rating model to a concatentation of
            # user and item embeddings.
            self.rating_model(
                tf.concat([user_embeddings, item_embeddings], axis=1)
            ),
        )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:

        ## ratings go here as a method to compute loss
        ratings = features.pop("RATING")

        user_embeddings, item_embeddings, rating_predictions = self(features)

        # We compute the loss for each task.
        rating_loss = self.rating_task(
            labels=ratings,
            predictions=rating_predictions,
        )
        retrieval_loss = self.retrieval_task(user_embeddings, item_embeddings)

        # And combine them using the loss weights.
        return (self.rating_weight * rating_loss
                + self.retrieval_weight * retrieval_loss)
```

<!-- #region id="mFPyxPmYdtio" -->
### Rating-specialized model
Depending on the weights we assign, the model will encode a different balance of the tasks. Let's start with a model that only considers ratings.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="9yRty3OOdqEh" outputId="d800cc38-0313-4c34-e438-bc09b56599d9"
model = Model(rating_weight=1.0, retrieval_weight=0.0)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()

model.fit(cached_train, epochs=3)
metrics = model.evaluate(cached_test, return_dict=True)

print(f"Retrieval top-100 accuracy: {metrics['factorized_top_k/top_100_categorical_accuracy']:.3f}.")
print(f"Ranking RMSE: {metrics['root_mean_squared_error']:.3f}.")
```

<!-- #region id="RS-P_DJLd-UE" -->
### Retrieval-specialized model
Let's now try a model that focuses on retrieval only.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="VazdjAAudzpX" outputId="8bbf4321-2cfe-46c8-fe20-1ea305d08042"
model = Model(rating_weight=0.0, retrieval_weight=1.0)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

model.fit(cached_train, epochs=3)
metrics = model.evaluate(cached_test, return_dict=True)

print(f"Retrieval top-100 accuracy: {metrics['factorized_top_k/top_100_categorical_accuracy']:.3f}.")
print(f"Ranking RMSE: {metrics['root_mean_squared_error']:.3f}.")
```

<!-- #region id="dT9KY_YreDZh" -->
### Joint model
Let's now train a model that assigns positive weights to both tasks.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="cAy8T18LeE9R" outputId="3cea02f2-6f91-4dc4-aa93-f6f76f32e27c"
model = Model(rating_weight=0.5, retrieval_weight=0.5)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

model.fit(cached_train, epochs=3)
metrics = model.evaluate(cached_test, return_dict=True)

print(f"Retrieval top-100 accuracy: {metrics['factorized_top_k/top_100_categorical_accuracy']:.3f}.")
print(f"Ranking RMSE: {metrics['root_mean_squared_error']:.3f}.")
```

<!-- #region id="oEQF2KeJeKha" -->
We can see that accuracy is highest and RMSE is lowest when we combine both ranking and retrieval together.
<!-- #endregion -->
