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

<!-- #region id="63EN8Vw-x65q" -->
# Retrieval model using TFRS on Amazon Electronics
<!-- #endregion -->

<!-- #region id="FUyr2TnRx5-g" -->
## Setup
<!-- #endregion -->

```python id="_evs_5ymrdF8"
!pip install tensorflow-recommenders
```

```python id="TXi-2EESrgy3"
import os
import pprint
import tempfile
from typing import Dict, Text
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
```

<!-- #region id="zUcE7t5Kr22-" -->
## Data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="7Sshfbo_rweB" executionInfo={"status": "ok", "timestamp": 1639380399563, "user_tz": -330, "elapsed": 25720, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="08cf5a70-c0b3-4da0-dc14-8fd29144783c"
!wget -q --show-progress http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Electronics.csv
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="fGpSpZGprw1M" executionInfo={"status": "ok", "timestamp": 1639380410073, "user_tz": -330, "elapsed": 7057, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="10d507f3-21d6-4627-ea44-9e1f0f8225ef"
df = pd.read_csv('ratings_Electronics.csv', names=['userId', 'productId','Rating','timestamp'])
df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="qbHyuqkwr383" executionInfo={"status": "ok", "timestamp": 1639380411312, "user_tz": -330, "elapsed": 1246, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6e262b95-b68f-406e-bc9a-26afe37e80f4"
df.dropna(inplace=True)
df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="S4gLHzlar7UP" executionInfo={"status": "ok", "timestamp": 1639380411313, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3e6194f1-8639-4cb0-8e04-4bcfcaf03256"
df.info()
```

```python colab={"base_uri": "https://localhost:8080/"} id="QIdEhL9Hr8La" executionInfo={"status": "ok", "timestamp": 1639380411313, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="48f5633f-529a-4429-96ab-fcf68bd8de77"
print('Minimum rating is: %d' %(df.Rating.min()))
print('Maximum rating is: %d' %(df.Rating.max()))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 369} id="zkSCYBXdr9rk" executionInfo={"status": "ok", "timestamp": 1639380423375, "user_tz": -330, "elapsed": 3825, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6e35cc2c-8148-406b-c17b-f35873efb13d"
g = sns.catplot(x="Rating", data=df, aspect=2.0, kind='count')
g.set_ylabels("Total number of ratings")
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="zglRF93RsAEP" executionInfo={"status": "ok", "timestamp": 1639380442271, "user_tz": -330, "elapsed": 16639, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="eef9ed0f-f83b-4707-cb46-34cc7cebf302"
df.drop(['timestamp'], axis=1, inplace=True)

num_of_rated_products_per_user = df.groupby(by='userId')['Rating'].count().sort_values(ascending=False)
num_of_rated_products_per_user.head()
```

```python id="ZbLS0UorsBjB"
new_df = df.sample(frac=0.1).groupby("productId").filter(lambda x:x['Rating'].count() >=50)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 394} id="twDkCX6usCdZ" executionInfo={"status": "ok", "timestamp": 1639380567651, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4f90dcd6-afb3-40a2-8b93-229d7d4ed1ac"
new_df.describe(include='all')
```

```python id="XwYj5X7QsDka"
df = new_df.copy()
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

<!-- #region id="EQ5J6EelsGX7" -->
## Model
<!-- #endregion -->

```python id="nw5pRXpTsG-R"
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

<!-- #region id="E7j2gUfVsRYU" -->
## Training and Evaluation
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="-4g-bfEjsLHp" executionInfo={"status": "ok", "timestamp": 1639381333149, "user_tz": -330, "elapsed": 758342, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2df56cc6-1dc8-42a8-e166-18685a60127a"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 312} id="Pm3tWZP_sTni" executionInfo={"status": "ok", "timestamp": 1639381333151, "user_tz": -330, "elapsed": 18, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a714c932-1ed4-462e-b1c4-60a6902f0e4b"
# num_validation_runs = len(one_layer_history.history["val_factorized_top_k/top_100_categorical_accuracy"])
epochs = [i for i in range(10)]

plt.plot(epochs, model_hist.history["factorized_top_k/top_100_categorical_accuracy"], label="accuracy")
plt.title("Accuracy vs epoch")
plt.xlabel("epoch")
plt.ylabel("Top-100 accuracy");
plt.legend()
```

<!-- #region id="X7eOVKSxsVXW" -->
## Inference
<!-- #endregion -->

```python id="0s8KR10-sWbP"
# Create a model that takes in raw query features, and
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)

# recommends products out of the entire product dataset.
index.index(items.batch(100).map(model.item_model), items)

# Get recommendations.
j = str(20)
_, titles = index(tf.constant([j]))
print(f"Recommendations for user %s: {titles[0]}" %(j))
```

<!-- #region id="J3Z3489FvvyJ" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="3yg-TSopvvyK" executionInfo={"status": "ok", "timestamp": 1639381959500, "user_tz": -330, "elapsed": 2561, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="20aacec0-432a-4c34-ab73-39212d1c76b1"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="YwV3XyTmvvyK" -->
---
<!-- #endregion -->

<!-- #region id="6gz9cHB3vvyK" -->
**END**
<!-- #endregion -->
