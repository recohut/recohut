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

<!-- #region id="EcE-fVyLnlaR" -->
# TFRS Two-tower Retrieval Model on ML-1m
<!-- #endregion -->

<!-- #region id="b50uwaHSPw-C" -->
## Setup
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="oUcRd9PPKihx" executionInfo={"status": "ok", "timestamp": 1636018107771, "user_tz": -330, "elapsed": 4567, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e829c62c-7db8-4cf9-e146-b8f6d7c51dab"
!pip install -q tensorflow_recommenders
```

```python id="zy82mzsUKdwv"
import os
import pprint
import tempfile
import matplotlib.pyplot as plt
from typing import Dict, Text

import numpy as np
import tensorflow as tf
from typing import Dict, Text
import pandas as pd
import numpy as np

import tensorflow as tf

import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

plt.style.use('ggplot')
```

<!-- #region id="Oo9aZsTfPyUe" -->
## Data Loading and Processing
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 497, "referenced_widgets": ["6f659f29afcd45c88e58b85cd7589510", "00b3d8c2963b4bcaa63289460c2a29a7", "c91d3454b3d945d2a4f7a839653f5f9e", "90620c1f337144538797b724cb989541", "7ceb905e45da4498872ec3c176f9925e", "45131df318e042809a1c803a6a8ea915", "723538beb5004d0da0b6cc09a39ee53e", "9e699982db094ab19a8497c250ff7800", "982f1678089741c0b4e8f953fc4c644f", "a4cf1074c4084d05bdc8e4052ffccce3", "83833ed236924204b967c22db0bf09e6", "890e1a63a9ce4333a73b949300835306", "1437ebc9209d4669a2048535030a45d8", "d6908d60b08d42fc96609570070e86ab", "7cdab38d07434ebdbd23d91bfc260974", "066387f9c71645089a372da18e954f49", "6e77e96b53e64627ba94bc335a5d0989", "6cf311e0d0254353a1113421d3cea621", "86f3a2527e6e4e2ba88a0a4869b84a7a", "7313ada0386f428c812846de27786e64", "7929ea9cc93f4009a2165c888215fdc4", "c4f39b92db4d4efea5bb2417fc174626", "08ae082590e14602a5390c4b1579050b", "f0bda57e9b1849ef9818d841d7eff301", "7115b065bcf84b74b2bcc5ec793bf938", "5966267b1bea47cfba70787850f530a6", "ab6f803d2e6e4facb206a27017f191ee", "344d4405dea3482684b6a72f6066f2cd", "476fdd686f8f46f0ba876301e5db1a43", "3bb0a3af0252436e90357197719fb522", "8646826de0134b78bff98794a1e1e254", "b1c246c735cd4631a69dae51f96afb22", "30b0cd873514424a9cef14e9392a5106", "fecb782718fb4bea869c3b71c6d0458b", "3dd51c39d5dd44d29bb24e9b068d016e", "6c2781a64ea44761bf546683b4584684", "a243d94a27f24504a8f339b0c365a141", "ddf61d3c70844c2487e8cf2a0dd78ae4", "5a117b5b00c34939b080ca1c5ed5366e", "ee4854d0d895413693f9d7be753d495a", "b63e43df68934fdb86563942b5199a13", "b5bb151def9b455cb83c758f09d3c7a5", "15c9b93815a141a59d3f0e0fbc8e2e71", "385c8c38a595477099210b80ee4d3c0a", "3bd6f957b63842d386d4bd1aaee5d4e8", "ad0ae077e31c4387a5f217d0d2c49876", "23c94bed3e2b4d5b96bd5bb80af4c7a6", "0c2f15aa6553430c957903714b09acaf", "b6e11a3ccc3a46ca81e0e5567372db0b", "17534eeb9b8948c2a4fb81bf68e2a094", "c4226d0a5cc046d181836d32eb2ef643", "668faa607dc8437e91a628c3c6d838d7", "6a775e4fb3f843528993116b95d2f969", "63ebd9d5b3e84ebc932856ad8384a509", "38f0ad9615eb4516b2bbd35cab7ccee8", "c394520c795f48d0a079d9e14047e512", "ebd129626ff142099b951d98d6e0f1e5", "685ff51d2bd84b92921c30d15b690d8d", "d8db6cd2b97e43b6bbe528c1cf78efb6", "f0d6e96ede0e47c0ad4c0632acfb98a3", "0a9dcfa143e3497297a792b053ff382b", "73cf3a33ca424b359e5d710307af4390", "c36852f3b4734def85d070c122543832", "8cdb28d8803049f59ea5f82c92da418d", "132f46a804424e74885baabfe7c773d4", "91d328ee8cb540ca97b48d0aa684d21d", "0bc9953451c44707bd33fada7ceb9112", "2dd7226bc73c4721a0b98b2a547d370f", "4cad39ee52cb4cb5b229faf83b9ffbef", "41b6f84bb8d74ad893d25dfc99fd519c", "507f7b8b10584f4ca2b90a9c4dfe29b7", "bd8d78b143714e9f802b5c4bbf40c232", "803dd352a7a04bc4894201e38bf51eb2", "3800494bcccc46428e1f20c446fe28bd", "51f309a9cb4a4bffa61f27e314d22878", "cfb64d85f93c403e8476a0725017df93", "7ee751784d804a70a3d25b170a7b49a6", "c7cd890b95164102afbdb6e6388eb4c3", "bef38e8beaa74a089900757017be5809", "7a7853a145304ba5a6b2a7ad392237a3", "91e0eec763f3489691f091f2ea0b7eb9", "28d0d2b439ca4824b003294c26af03f4", "9008ce61d6c14194843a5231c2b934ff", "8ad8cef750a144728512b8825d2ae2f4", "7ba9ff85a8f343f08cd63a047c89cd82", "ad72eab6ae214df18f9019fa41cc8c63", "8943a5af11164f89b72660f419010fcd", "8e9c50ab283547c28f9cca43bbe9b6f8", "4568c707075d412892c85c4dbf5ad93b", "39bacdd2f02241e585a1a4de1223554f", "bfbed280ed794e3fa74ef7aac7f0151d", "7f2f806290cc4f45930c8c9771a11e9b", "f563333bf05149c6aa8f63da775eed47", "bee34a54733d4db085d84a715fda691c", "ae6d23ee3bdd4030871b394dde281ae0", "c357c79a96bd41cabcaf47e3f87d7993", "d681df83c3be4c838497fa631051a7fc", "83fd258db2f144bcb506454dad8df744", "096b27dfa13e446fa8eb764163360aab", "583dc08479954008959d001ced8b25bc", "641a8493c78e4e56acc90d94b9ea1703", "eee86b97f62e48aa8b7b72f4e2306c7f", "1bcfd65aba2341269f3aebcb208a76b8", "de1a81082638418399bf57890ad49638", "2aaf5f5d0b12498f9add61c8085d4073", "5d682ece209d4c2092d85ed859dae02f", "d3d2b222828e4e48a4d63f07c6f251bd", "2479e66fdd23473b9cc332ebd156bf17", "fae53a890b384edf92e521e8d1efb099", "dfd3b730e8014ccab1bdb53e5a44595b"]} id="KuRJ9EHuKxIA" executionInfo={"status": "ok", "timestamp": 1636019437486, "user_tz": -330, "elapsed": 1326826, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3c4f3ebf-f7b0-4164-8919-235b732daaad"
# Ratings data.
ratings = tfds.load("movielens/1m-ratings", split="train", shuffle_files=True)
# Features of all the available movies.
movies = tfds.load("movielens/1m-movies", split="train", shuffle_files=True)
```

```python colab={"base_uri": "https://localhost:8080/"} id="9aL1dnCULANK" executionInfo={"status": "ok", "timestamp": 1636019439728, "user_tz": -330, "elapsed": 738, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="af3e3b13-7f39-4474-889d-b02cf2e61957"
#The ratings dataset returns a dictionary of movie id, user id, the assigned rating, timestamp, movie information, and user information:
#View the data from ratings dataset:
for x in ratings.take(1).as_numpy_iterator():
    pprint.pprint(x)
```

```python colab={"base_uri": "https://localhost:8080/"} id="MbBj-IHEL1zI" executionInfo={"status": "ok", "timestamp": 1636019440462, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5cf3aadd-3cbb-4179-fe6a-b4d5980ff96a"
#The movies dataset contains the movie id, movie title, and data on what genres it belongs to. Note that the genres are encoded with integer labels:
#View the data from movies dataset:
for x in movies.take(1).as_numpy_iterator():
    pprint.pprint(x)
```

```python colab={"base_uri": "https://localhost:8080/"} id="TNI9_ie4L2lD" executionInfo={"status": "ok", "timestamp": 1636019441154, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0d2734ac-7f80-4774-e897-1ba0de473e73"
type(ratings)
```

<!-- #region id="Z69RMmvHKag6" -->
We're only going to extract the movie title and the user id. So, we're actually not going to extract the rating itself and the reason why is that we're treating these as implicit recommendations in this case because that's easier to do a large scale like we mentioned earlier we want to try to keep things simple in the retrieval stage because it's operating over potentially a massive amount of data so to that end we're just going to assume that any movie that a user rated is one that they were really interested in if they took the time to watch it then it expresses some level of interest:
<!-- #endregion -->

```python id="GwQevswBKv4y"
#Let's select the necessary attributes:

ratings = ratings.map(lambda x: {
                                 "movie_title": x["movie_title"],
                                 "user_id": x["user_id"],
                                })

movies = movies.map(lambda x: x["movie_title"])
```

```python id="gsY0gEkOLCcn"
# let's use a random split, putting 75% of the ratings in the train set, and 25% in the test set:
# Assign a seed=42 for consistency of results and reproducibility:
seed = 42
l = len(ratings)

tf.random.set_seed(seed)
shuffled = ratings.shuffle(l, seed=seed, reshuffle_each_iteration=False)

#Save 75% of the data for training and 25% for testing:
train_ = int(0.75 * l)
test_ = int(0.25 * l)

train = shuffled.take(train_)
test = shuffled.skip(train_).take(test_)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Ixswf8SGPpSR" executionInfo={"status": "ok", "timestamp": 1636019532262, "user_tz": -330, "elapsed": 90606, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e46133c9-0945-4290-eb13-73370293b08d"
# Now, let's find out how many uniques users/movies:
movie_titles = movies.batch(l)
user_ids = ratings.batch(l).map(lambda x: x["user_id"])

#Movies uniques:
unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))

#users unique
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

# take a look at the movies:
unique_movie_titles[:10]
```

```python colab={"base_uri": "https://localhost:8080/"} id="pD_X4XH8Pqy3" executionInfo={"status": "ok", "timestamp": 1636019532264, "user_tz": -330, "elapsed": 61, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="20389e8d-6a74-4b6d-ebd0-8c1f0dc9df1d"
#Movies uniques
len_films = len(unique_movie_titles)
print(len_films) 
```

```python colab={"base_uri": "https://localhost:8080/"} id="Avhdjoq5PsFM" executionInfo={"status": "ok", "timestamp": 1636019532266, "user_tz": -330, "elapsed": 51, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="33c952c0-8cca-4126-ca9f-31670687cd92"
#users unique
len_users = len(unique_user_ids)
print(len_users) 
```

<!-- #region id="qkLbhNsgPtM8" -->
## Implementing a Retrieval Model

Choosing the architecture of our model is a key part of modelling.

Because we are building a two-tower retrieval model, we can build each tower separately and then combine them in the final model.
<!-- #endregion -->

<!-- #region id="IC9RB-ARQcrn" -->
### The query tower

Let's start with the query tower.

The first step is to decide on the dimensionality of the query and candidate representations:
<!-- #endregion -->

```python id="fXHCBU0-QYmg"
#Higher values will correspond to models that may be more accurate, but will also be slower to fit and more prone to overfitting:
embedding_dimension = 32

#We define the embedding on the user side, we must transform the user ids into a vector representation:
#we're going to use Keras preprocessing layers to first convert user ids to integers, and then convert those to user embeddings via an Embedding layer:
user_model = tf.keras.Sequential([
                                  tf.keras.layers.experimental.preprocessing.StringLookup(
                                  #User_ids vocabulary: list of unique integers that represents each user_id
                                  vocabulary=unique_user_ids, mask_token=None),
                                  # We add an additional embedding to account for unknown tokens.
                                  tf.keras.layers.Embedding(len(unique_user_ids) + 1, 
                                  embedding_dimension)#embedding layer with a vector size of 32
                                ])
```

<!-- #region id="SpU9JdhZQiLo" -->
So, basically we want an embedding layer for however many user ids we have where each one is represented by a vector of 32 floating point values that  basically represents in that 64-dimensional space how similar users 
<!-- #endregion -->

<!-- #region id="3cM3WtUVQfRv" -->
### The candidate tower
<!-- #endregion -->

```python id="i3C1Or8LQmPq"
# We now define the embedding of the movie portion 
movie_model = tf.keras.Sequential([
                                  tf.keras.layers.experimental.preprocessing.StringLookup(
                                  vocabulary=unique_movie_titles, mask_token=None),
                                  tf.keras.layers.Embedding(len(unique_movie_titles) + 1,
                                  embedding_dimension) #embedding layer with a vector size of 32
                                 ])
```

<!-- #region id="FuJD-hJ9QpG8" -->
### Metrics

In our training data we have positive (user, movie) pairs. To figure out how good our model is, we need to compare the affinity score that the model calculates for this pair to the scores of all the other possible candidates: if the score for the positive pair is higher than for all other candidates, our model is highly accurate.

To do this, we can use the tfrs.metrics.FactorizedTopK metric. The metric has one required argument: the dataset of candidates that are used as implicit negatives for evaluation.

In our case, that's the movies dataset, converted into embeddings via our movie model:
<!-- #endregion -->

```python id="UnWXonnIQrTV"
#We define the desired metrics : FactorizedTopK
metrics = tfrs.metrics.FactorizedTopK(
                                     candidates=movies.batch(128).map(movie_model)
                                     )
#The Retrieval task is defined according to the FactorizedTopK metrics:
task = tfrs.tasks.Retrieval(
                            metrics=metrics
                           )
```

<!-- #region id="lXOv5ZSHQtb_" -->
### The Full Model

We can now put it all together into a model. TFRS exposes a base model class (tfrs.models.Model) which streamlines building models: all we need to do is to set up the components in the init method, and implement the compute_loss method, taking in the raw features and returning a loss value.

The base model will then take care of creating the appropriate training loop to fit our model.
<!-- #endregion -->

```python id="UMExxYDiQvoK"
class MovielensModel(tfrs.Model):
    
    def __init__(self, user_model, movie_model):
        super().__init__()
        #The Two Towers: Movie and user Models:
        self.movie_model: tf.keras.Model = movie_model
        self.user_model: tf.keras.Model = user_model
        self.task: tf.keras.layers.Layer = task

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.user_model(features["user_id"])
        # And pick out the movie features and pass them into the movie model,
        # getting embeddings back.
        positive_movie_embeddings = self.movie_model(features["movie_title"])

        # The task computes the loss and the metrics.
        return self.task(user_embeddings, positive_movie_embeddings)
```

<!-- #region id="sVQjc_dmQxfB" -->
The tfrs.Model base class is a simply convenience class: it allows us to compute both training and test losses using the same method.

Under the hood, it's still a plain Keras model. You could achieve the same functionality by inheriting from tf.keras.Model and overriding the train_step and test_step functions (see the guide for details):
<!-- #endregion -->

```python id="bCYXQFiaQylA"
class NoBaseClassMovielensModel(tf.keras.Model):

    def __init__(self, user_model, movie_model):
        super().__init__()
        self.movie_model: tf.keras.Model = movie_model
        self.user_model: tf.keras.Model = user_model
        self.task: tf.keras.layers.Layer = task

    def train_step(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:

        # Set up a gradient tape to record gradients.
        with tf.GradientTape() as tape:

            # Loss computation.
            user_embeddings = self.user_model(features["user_id"])
            positive_movie_embeddings = self.movie_model(features["movie_title"])
            loss = self.task(user_embeddings, positive_movie_embeddings)

            # Handle regularization losses as well.
            regularization_loss = sum(self.losses)

            total_loss = loss + regularization_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics

    def test_step(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:

        # Loss computation.
        user_embeddings = self.user_model(features["user_id"])
        positive_movie_embeddings = self.movie_model(features["movie_title"])
        loss = self.task(user_embeddings, positive_movie_embeddings)

        # Handle regularization losses as well.
        regularization_loss = sum(self.losses)

        total_loss = loss + regularization_loss

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics
```

<!-- #region id="ssFiVAo9QQ-H" -->
### Fitting and evaluating
After defining the model, we can use standard Keras fitting and evaluation routines to fit and evaluate the model.

Adagrad is an algorithm for gradient-based optimization that does just this: It adapts the learning rate to the parameters, performing smaller updates (i.e. low learning rates) for parameters associated with frequently occurring features, and larger updates (i.e. high learning rates) for parameters associated with infrequent features. For this reason, it is well-suited for dealing with sparse data.
<!-- #endregion -->

```python id="1qdMav8uQOTs"
#Let's first instantiate the model.
model = MovielensModel(user_model, movie_model)

model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1)) #defauly learning_rate=0.001
```

```python id="5yPHHWPqQM6b"
# Then shuffle, batch, and cache the training and evaluation data:
# Segment the batches so that the model runs 13 training batches (2^13) and 11 test batches (2^11) per epoch, 
# while having a batch size which is a multiple of 2^n.
cached_train = train.shuffle(l).batch(8192).cache()
cached_test = test.batch(2048).cache()
```

```python colab={"base_uri": "https://localhost:8080/"} id="pATY5TMuQLqD" executionInfo={"status": "ok", "timestamp": 1636021194232, "user_tz": -330, "elapsed": 1662006, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="258f6411-cd01-4bff-fdaf-3d7947291305"
# Then, let's train the model:
history_train = model.fit(cached_train, validation_data = cached_test, epochs=5)
```

<!-- #region id="9mpbRzG_Q6wH" -->
As the model trains, the loss is falling and a set of top-k retrieval metrics is updated. These tell us whether the true positive is in the top-k retrieved items from the entire candidate set. For example, a top-5 categorical accuracy metric of 0.2 would tell us that, on average, the true positive is in the top 5 retrieved items 20% of the time.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="V26excQlRA_T" executionInfo={"status": "ok", "timestamp": 1636021338006, "user_tz": -330, "elapsed": 142391, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="188d6f10-adcf-4199-b48c-c7b5c68edef5"
#Evaluate the model
model.evaluate(cached_test, return_dict=True)
```

<!-- #region id="UbQcs6agQ7DC" -->
### Visualization: Total loss and Accuracy over epochs
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 400} id="icvrXIJHQ9IR" executionInfo={"status": "ok", "timestamp": 1636021338008, "user_tz": -330, "elapsed": 35, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="baba63a7-487b-4692-b43b-bec84883a715"
plt.subplots(figsize = (16,6))
plt.plot(history_train.history['total_loss'] )
plt.title("Total Loss over epochs", fontsize=14)
plt.ylabel('Loss Total')
plt.xlabel('epochs')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 404} id="YGOKUvo1Q_Z1" executionInfo={"status": "ok", "timestamp": 1636021339128, "user_tz": -330, "elapsed": 1131, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="28a3d13d-1df7-431f-dc6b-b468fe08d0ca"
plt.subplots(figsize = (16,6))
plt.plot(history_train.history['factorized_top_k/top_100_categorical_accuracy'],color='green', alpha=0.8, label='Train' )
plt.plot(history_train.history['val_factorized_top_k/top_100_categorical_accuracy'],color='red', alpha=0.8, label='Test' )
plt.title("Accuracy over epochs", fontsize=14)
plt.ylabel('Accuracy')
plt.xlabel('epochs')
plt.legend(loc='upper left')
plt.show()
```

<!-- #region id="W7lE7pzqRJX7" -->
Test set performance is much worse than training performance. This is due to two factors:

- Our model is likely to perform better on the data that it has seen, simply because it can memorize it. This overfitting phenomenon is especially strong when models have many parameters. It can be mediated by model regularization and use of user and movie features that help the model generalize better to unseen data.
- The model is re-recommending some of users' already watched movies. These known-positive watches can crowd out test movies out of top K recommendations.

The second phenomenon can be tackled by excluding previously seen movies from test recommendations. This approach is relatively common in the recommender systems literature, but we don't follow it in these tutorials. If not recommending past watches is important, we should expect appropriately specified models to learn this behaviour automatically from past user history and contextual information. Additionally, it is often appropriate to recommend the same item multiple times (say, an evergreen TV series or a regularly purchased item).
<!-- #endregion -->

<!-- #region id="qBJQD5E2ROYF" -->
## Making predictions

Now that we have a model, we would like to be able to make predictions. We can use the tfrs.layers.factorized_top_k.BruteForce layer to do this.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="SQnYRG-wfNhT" executionInfo={"status": "ok", "timestamp": 1636021660610, "user_tz": -330, "elapsed": 599, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1f95dd6b-3df3-479a-84d7-ca4bf0dcae45"
# Features of all the available movies.
movies = tfds.load("movielens/1m-movies", split="train", shuffle_files=True)

#The movies dataset contains the movie id, movie title, and data on what genres it belongs to. Note that the genres are encoded with integer labels:
#View the data from movies dataset:
for x in movies.take(1).as_numpy_iterator():
    pprint.pprint(x)

movies = movies.map(lambda x: x["movie_title"])
```

```python id="zSNQGImsRfob"
# Recommend the 5 best movies for user 42:

# Create a model that takes in raw query features, and
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)

# recommends movies out of the entire movies dataset.
index.index(movies.batch(100).map(model.movie_model), movies)

# Get recommendations.
_, titles = index(tf.constant(["42"]))
print(f"Recommendations for user 42: {titles[0, :5]}")
```

<!-- #region id="9LtDRNoDRiTd" -->
## Model serving

After the model is trained, we need a way to deploy it.

In a two-tower retrieval model, serving has two components:

- a serving query model, taking in features of the query and transforming them into a query embedding, and
- a serving candidate model. This most often takes the form of an approximate nearest neighbours (ANN) index which allows fast approximate lookup of candidates in response to a query produced by the query model.

In TFRS, both components can be packaged into a single exportable model, giving us a model that takes the raw user id and returns the titles of top movies for that user. This is done via exporting the model to a SavedModel format, which makes it possible to serve using TensorFlow Serving.

To deploy a model like this, we simply export the BruteForce layer we created above:
<!-- #endregion -->

```python id="kre0DlQXRoh0"
# Export the query model.
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "model")

  # Save the index.
    index.save(path)

  # Load it back; can also be done in TensorFlow Serving.
    loaded = tf.keras.models.load_model(path)

  # Pass a user id in, get top predicted movie titles back.
    scores, titles = loaded(["42"])
    
    print(f"Recommendations: {titles[0][:5]}")
```

```python id="5dRxk087RqnA"
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:75% !important; }</style>"))
```

<!-- #region id="Oj_-StoPSAZ2" -->
## Tuning Retrieval (The Two Towers Model)
<!-- #endregion -->

<!-- #region id="cHwNQJLNSBFD" -->
### Increase embedding_dimension from 32 to 64
<!-- #endregion -->

```python id="kpaQDeDoSFmv"
#Higher values will correspond to models that may be more accurate, but will also be slower to fit and more prone to overfitting:
embedding_dimension = 64
```

```python id="pzhMU-spSGS8"
#We define the embedding on the user side, we must transform the user ids into a vector representation:
#we're going to use Keras preprocessing layers to first convert user ids to integers, and then convert those to user embeddings via an Embedding layer:
user_model = tf.keras.Sequential([
                                  tf.keras.layers.experimental.preprocessing.StringLookup(
                                  #User_ids vocabulary: list of unique integers that represents each user_id
                                  vocabulary=unique_user_ids, mask_token=None),
                                  # We add an additional embedding to account for unknown tokens.
                                  tf.keras.layers.Embedding(len(unique_user_ids) + 1, 
                                  embedding_dimension)#embedding layer with a vector size of 64
                                ])
```

<!-- #region id="7Sm0n0fpSKFB" -->
So, basically we want an embedding layer for however many user ids we have where each one is represented by a vector of 64 floating point values that basically represents in that 64-dimensional space how similar users.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="1V-LYjEbSMhw" executionInfo={"status": "ok", "timestamp": 1636022304061, "user_tz": -330, "elapsed": 603298, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="11e74eef-e09d-47f1-f6b3-613cfdbc9dc4"
# We now define the embedding of the movie portion 
movie_model = tf.keras.Sequential([
                                  tf.keras.layers.experimental.preprocessing.StringLookup(
                                  vocabulary=unique_movie_titles, mask_token=None),
                                  tf.keras.layers.Embedding(len(unique_movie_titles) + 1,
                                  embedding_dimension) #embedding layer with a vector size of 64
                                 ])

#We define the desired metrics : FactorizedTopK
metrics = tfrs.metrics.FactorizedTopK(
                                     candidates=movies.batch(128).map(movie_model)
                                     )

#The Retrieval task is defined according to the FactorizedTopK metrics:
task = tfrs.tasks.Retrieval(
                            metrics=metrics
                           )

#Let's first instantiate the model.
model_1 = MovielensModel(user_model, movie_model)

model_1.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1)) # learning_rate default=0.001

# Then, let's train the model:
history_train_1 = model_1.fit(cached_train, validation_data = cached_test, epochs=2)
```

```python colab={"base_uri": "https://localhost:8080/"} id="biMDmrcpSXIF" executionInfo={"status": "ok", "timestamp": 1636022384672, "user_tz": -330, "elapsed": 80634, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="23c0c31d-55da-4069-bfd1-5f940973780e"
#Evaluate the Base model
model_1.evaluate(cached_test, return_dict=True)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 400} id="29fjszAPSUBR" executionInfo={"status": "ok", "timestamp": 1636022386401, "user_tz": -330, "elapsed": 1746, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="93dc619a-2075-4e6e-96f7-d8a81ad1af85"
plt.subplots(figsize = (16,6))
plt.plot(history_train_1.history['total_loss'] )
plt.title("Total Loss over epochs", fontsize=14)
plt.ylabel('Loss Total')
plt.xlabel('epochs')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 404} id="OxLFe6qwSVfo" executionInfo={"status": "ok", "timestamp": 1636022386402, "user_tz": -330, "elapsed": 29, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8dc1614f-a18d-46c9-9f64-100ca3257777"
plt.subplots(figsize = (16,6))
plt.plot(history_train_1.history['factorized_top_k/top_100_categorical_accuracy'],color='green', alpha=0.8, label='Train' )
plt.plot(history_train_1.history['val_factorized_top_k/top_100_categorical_accuracy'],color='red', alpha=0.8, label='Test' )
plt.title("Accuracy over epochs", fontsize=14)
plt.ylabel('Accuracy')
plt.xlabel('epochs')
plt.legend(loc='upper left')
plt.show()
```

<!-- #region id="vVFj4ie4Sagg" -->
### Reduce Adagrad learning_rate

learning_rate = 0.01
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="nanoYLeXScs1" executionInfo={"status": "ok", "timestamp": 1636022987280, "user_tz": -330, "elapsed": 600904, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="eb56c9ad-3f5d-4f3f-991e-383d8c7ad128"
#Let's first instantiate the model.
model_2 = MovielensModel(user_model, movie_model)

model_2.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.01)) # learning_rate default=0.001

# Then, let's train the model:
history_train_2 = model_2.fit(cached_train, validation_data = cached_test, epochs=2)
```

```python colab={"base_uri": "https://localhost:8080/"} id="5vfZOV7oSjne" executionInfo={"status": "ok", "timestamp": 1636023069178, "user_tz": -330, "elapsed": 81923, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="eede7966-0d05-4b8c-a42a-c92e7c2c3857"
#Evaluate the Base model
model_2.evaluate(cached_test, return_dict=True)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 400} id="l0rdKNWkSgp_" executionInfo={"status": "ok", "timestamp": 1636023069179, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fb3cb3f6-d4ec-42db-8dfe-2a3a2e11741a"
plt.subplots(figsize = (16,6))
plt.plot(history_train_2.history['total_loss'] )
plt.title("Total Loss over epochs", fontsize=14)
plt.ylabel('Loss Total')
plt.xlabel('epochs')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 399} id="U31d62SJSgmR" executionInfo={"status": "ok", "timestamp": 1636023069992, "user_tz": -330, "elapsed": 819, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f6d6c793-5f59-4792-cce0-006b6ce518aa"
plt.subplots(figsize = (16,6))
plt.plot(history_train_2.history['factorized_top_k/top_100_categorical_accuracy'],color='green', alpha=0.8, label='Train' )
plt.plot(history_train_2.history['val_factorized_top_k/top_100_categorical_accuracy'],color='red', alpha=0.8, label='Test' )
plt.title("Accuracy over epochs", fontsize=14)
plt.ylabel('Accuracy')
plt.xlabel('epochs')
plt.legend(loc='upper left')
plt.show()
```

<!-- #region id="9915e501" -->
## Tuning Summary

As we can see below, we managed to improve accuracy and reduce loss by: 

 1. Increase embedding_dimension from 32 -> 64
 2. Decreasing learning_rate 0.1 -> 0.01


| Tuning | top_1_accuracy |top_5_accuracy | top_10_accuracy | top_50_accuracy | top_100_accuracy|loss|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:
|**Baseline Model:** embedding_dimension 32, learning_rate 0.1, epochs=5 | 0.0026 | 0.0169 | 0.0317 | 0.1237 | 0.2056 | 891.81 |
|**Model_1:** embedding_dimension 32 -> 64, learning_rate 0.1 , epochs=2| 0.0027 | 0.0179 | 0.0341 | 0.1297 | 0.2130 | 903.23 |
|**Model_2:** embedding_dimension 64, learning_rate 0.1 -> 0.01, epochs=2| 0.0028 | 0.0189 | 0.0360 | 0.1349 | 0.2205 | 896.75 |
<!-- #endregion -->
