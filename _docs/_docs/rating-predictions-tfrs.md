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

<!-- #region id="nhQsfM2bI76g" -->
# Predicting Ratings using TFRS
> Training a top-k retrieval model using Tensorflow Recommenders library on Amazon Electronics dataset

- toc: true
- badges: true
- comments: true
- categories: [TFRS, Tensorflow, Surprise, Retail, AmazonDataset]
- image:
<!-- #endregion -->

<!-- #region id="3DYJx4IBIZ0G" -->
## Setup
<!-- #endregion -->

```python id="bFRnZKtwFhHI"
!pip install tensorflow-recommenders
!pip install surprise
```

```python id="Hfbvk4LCFj1B"
import os
import pprint
import tempfile
from typing import Dict, Text
import seaborn as sns

import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
import matplotlib.pyplot as plt
import pandas as pd

from surprise import Reader
from surprise.model_selection import train_test_split
from surprise import Dataset
```

```python id="VurySgqbI172"
!pip install -q watermark
%reload_ext watermark
%watermark -m -iv -u -t -d
```

<!-- #region id="xHFcbs93Ibn7" -->
## Loading dataset
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="2tKKNN1ZGth-" outputId="64e75ec8-d53a-4d08-ee39-be36ee1dd5a2"
!wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Electronics.csv
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="kiU_1VtgFoA3" outputId="b1fef80c-0ab4-4757-8ec0-0b623f780ec4"
df = pd.read_csv('ratings_Electronics.csv', names=['userId', 'productId','Rating','timestamp'])
df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="R7-U3n53G3kW" outputId="125e871c-13b5-436f-c94a-fc01657c77d8"
df.dropna(inplace=True)
df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="DFW0EfKzG5k3" outputId="38f0a9dd-c125-414f-c7ba-6f1b1d471045"
df.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 297} id="mWj0ytKBG5iq" outputId="5c07f50c-3bab-48ec-deae-7fd8dd92331d"
df.describe()
```

```python colab={"base_uri": "https://localhost:8080/"} id="OsElU9JCG5gV" outputId="a65c1959-2420-4ab7-b0f8-d141833624ff"
print('Minimum rating is: %d' %(df.Rating.min()))
print('Maximum rating is: %d' %(df.Rating.max()))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 383} id="hSwbskUZHDLR" outputId="a721303d-a117-42a1-c78c-8e0601b6cc3b"
g = sns.catplot(x="Rating", data=df, aspect=2.0, kind='count')
g.set_ylabels("Total number of ratings")
plt.show()
```

<!-- #region id="W16zuRoYIfll" -->
## Preprocessing
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="vKceR-XIHjlU" outputId="00d10597-a68a-4b04-cf56-3e6c5f83aaa4"
df.drop(['timestamp'], axis=1, inplace=True)

num_of_rated_products_per_user = df.groupby(by='userId')['Rating'].count().sort_values(ascending=False)
num_of_rated_products_per_user.head()
```

```python id="nMVmHLo1Hknj"
new_df = df.sample(frac=0.1).groupby("productId").filter(lambda x:x['Rating'].count() >=50)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 390} id="J47-2t6jRQFB" outputId="4e948f93-a806-456f-96a0-532bcb9f27d1"
new_df.describe(include='all')
```

<!-- #region id="vAmhuypKIozj" -->
## Train/test split
<!-- #endregion -->

```python id="gKDLT9yrSXCd"
df = new_df.copy()
```

```python id="hihYnbpuIBEi"
interactions_dict = df.drop(['Rating'],axis=1)
interactions_dict = {name: np.array(value) for name, value in interactions_dict.items()}
interactions = tf.data.Dataset.from_tensor_slices(interactions_dict)

items_dict = df[['productId']].drop_duplicates()
items_dict = {name: np.array(value) for name, value in items_dict.items()}
items = tf.data.Dataset.from_tensor_slices(items_dict)

interactions = interactions.map(lambda x: {
    'userId' : x['userId'], 
    'productId' : x['productId'], 
})
```

```python id="gQlSkHERIEWR"
items = items.map(lambda x: x['productId'])

### get unique item and user id's as a lookup table
unique_item_titles = np.unique(np.concatenate(list(items.batch(1000))))
unique_user_ids = np.unique(np.concatenate(list(interactions.batch(1_000).map(lambda x: x["userId"]))))

# Randomly shuffle data and split between train and test.
tf.random.set_seed(42)
shuffled = interactions.shuffle(200_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(180_000)
test = shuffled.skip(180_000).take(20_000)
```

<!-- #region id="cOe-GBonIkpj" -->
## Defining model architecture
<!-- #endregion -->

```python id="afoYwurkIHl4"
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
        metrics = tfrs.metrics.FactorizedTopK(
                                            candidates=items.batch(128).map(item_model))
        
        # define the task, which is retrieval                                    )    
        task = tfrs.tasks.Retrieval(
                                    metrics=metrics
                                    )
       
        self.task: tf.keras.layers.Layer = task

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.user_model(features["userId"])
        # And pick out the movie features and pass them into the movie model,
        # getting embeddings back.
        positive_movie_embeddings = self.item_model(features["productId"])

        # The task computes the loss and the metrics.
        return self.task(user_embeddings, positive_movie_embeddings)
```

<!-- #region id="_12w_KC0IJeW" -->
## Fitting and evaluating
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="tIrcnNihFfMP" outputId="475f821a-d268-442c-e807-877c1d5ade60"
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

<!-- #region id="5sNzoCTJIuQM" -->
## Plotting results
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 312} id="VdzOCAFGIUMq" outputId="3703cdcc-0d2f-4700-f437-01b45a669ff7"
# num_validation_runs = len(one_layer_history.history["val_factorized_top_k/top_100_categorical_accuracy"])
epochs = [i for i in range(10)]

plt.plot(epochs, model_hist.history["factorized_top_k/top_100_categorical_accuracy"], label="accuracy")
plt.title("Accuracy vs epoch")
plt.xlabel("epoch")
plt.ylabel("Top-100 accuracy");
plt.legend()
```

<!-- #region id="9TL2dqVkIvrF" -->
## Recommendations
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="UjvhIlbXIUoC" outputId="8aacb2f3-263e-40f0-ac36-95927bb9c652"
# Create a model that takes in raw query features, and
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)

# recommends products out of the entire product dataset.
index.index(items.batch(100).map(model.item_model), items)

# Get recommendations.
j = str(20)
_, titles = index(tf.constant([j]))
print(f"Recommendations for user %s: {titles[0]}" %(j))
```
