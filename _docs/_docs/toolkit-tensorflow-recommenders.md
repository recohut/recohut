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

```python colab={"base_uri": "https://localhost:8080/"} id="nCVkgf9bJM1x" executionInfo={"status": "ok", "timestamp": 1609876971098, "user_tz": -330, "elapsed": 4502, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f121bd71-413b-4a19-cd40-f23af1e565f3"
!git clone https://github.com/leoncvlt/loconotion.git
```

```python id="vP4EJloazo0N"
!pip install -q tensorflow-recommenders
!pip install -q --upgrade tensorflow-datasets
!pip install -q scann
```

```python id="roFVSopw1ds_" executionInfo={"status": "ok", "timestamp": 1609839764621, "user_tz": -330, "elapsed": 1312, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
```

```python id="LkOJW5UfBgiD" executionInfo={"status": "ok", "timestamp": 1609839912590, "user_tz": -330, "elapsed": 1459, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Ratings data.
ratings = tfds.load('movielens/100k-ratings', split="train")
# Features of all the available movies.
movies = tfds.load('movielens/100k-movies', split="train")
```

```python colab={"base_uri": "https://localhost:8080/"} id="EiAGBDiT7Rko" executionInfo={"status": "ok", "timestamp": 1609839913210, "user_tz": -330, "elapsed": 1714, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ffff3e94-ac80-481d-e285-4ccb5cdf5346"
for x in ratings.take(1).as_numpy_iterator():
  pprint.pprint(x)
```

```python id="zXFqomlC76dr" executionInfo={"status": "ok", "timestamp": 1609839974129, "user_tz": -330, "elapsed": 1567, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Select the basic features.
ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"]
})
movies = movies.map(lambda x: x["movie_title"])
```

```python colab={"base_uri": "https://localhost:8080/"} id="EAXXgj2s8pLi" executionInfo={"status": "ok", "timestamp": 1609840134003, "user_tz": -330, "elapsed": 9720, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0581fdec-a5c4-4ada-8f2e-682c275494fd"
movie_titles = movies.batch(1_000)
user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

unique_movie_titles[:10]
```

```python id="lm6qGu1U8Jkz" executionInfo={"status": "ok", "timestamp": 1609840104108, "user_tz": -330, "elapsed": 1343, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)
```

```python id="ND35SHcWzgMc" executionInfo={"status": "ok", "timestamp": 1609838619857, "user_tz": -330, "elapsed": 20029, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Build vocabularies to convert user ids and movie titles into integer indices for embedding layers
user_ids_vocabulary = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
user_ids_vocabulary.adapt(ratings.map(lambda x: x["user_id"]))

movie_titles_vocabulary = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
movie_titles_vocabulary.adapt(movies)
```

<!-- #region id="sqVvPkem3UJX" -->
Build the model
<!-- #endregion -->

```python id="zx_5Nw-J2539" executionInfo={"status": "ok", "timestamp": 1609839030440, "user_tz": -330, "elapsed": 2951, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
class MovieLensModel(tfrs.Model):
  # We derive from a custom base class to help reduce boilerplate. Under the hood,
  # these are still plain Keras Models.

  def __init__(
      self,
      user_model: tf.keras.Model,
      movie_model: tf.keras.Model,
      task: tfrs.tasks.Retrieval):
    super().__init__()

    # Set up user and movie representations.
    self.user_model = user_model
    self.movie_model = movie_model

    # Set up a retrieval task.
    self.task = task

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    # Define how the loss is computed.

    user_embeddings = self.user_model(features["user_id"])
    movie_embeddings = self.movie_model(features["movie_title"])

    return self.task(user_embeddings, movie_embeddings)
```

```python id="Gl0lzyuu4S64" executionInfo={"status": "ok", "timestamp": 1609839031519, "user_tz": -330, "elapsed": 4023, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Define user and movie models.
user_model = tf.keras.Sequential([
    user_ids_vocabulary,
    tf.keras.layers.Embedding(user_ids_vocabulary.vocab_size(), 64)
])
movie_model = tf.keras.Sequential([
    movie_titles_vocabulary,
    tf.keras.layers.Embedding(movie_titles_vocabulary.vocab_size(), 64)
])

# Define your objectives.
task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
    movies.batch(128).map(movie_model)
  )
)
```

<!-- #region id="LpuyyysI38h8" -->
Fit and evaluate
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="kBpHOnjl390M" executionInfo={"status": "ok", "timestamp": 1609839238221, "user_tz": -330, "elapsed": 91214, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="dbb00b6a-e231-4338-de99-de09f0295a00"
# Create a retrieval model.
model = MovieLensModel(user_model, movie_model, task)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))

# Train for 3 epochs.
model.fit(ratings.batch(4096), epochs=3)

# Use brute-force search to set up retrieval using the trained representations.
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
index.index(movies.batch(100).map(model.movie_model), movies)

# Get some recommendations.
_, titles = index(np.array(["42"]))
print(f"Top 3 recommendations for user 42: {titles[0, :3]}")
```

```python id="t8bygRrt4Mvx"

```
