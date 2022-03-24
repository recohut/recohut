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

<!-- #region id="XlJy1ng1kepp" -->
# Features Importance Using Deep & Cross Network on ML-1m in TFRS
<!-- #endregion -->

<!-- #region id="Db4SCHQdcNXQ" -->
## Setup
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="stss4EYmcLuF" executionInfo={"status": "ok", "timestamp": 1636020840802, "user_tz": -330, "elapsed": 4034, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="890d5389-800a-472a-9994-b03be5309148"
!pip install -q tensorflow_recommenders
```

```python id="ZbRt7EjfcJaF"
import os
import pprint
import tempfile
import matplotlib.pyplot as plt
from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Dict, Text
import pandas as pd
import numpy as np

import tensorflow as tf

import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

plt.style.use('ggplot')
```

<!-- #region id="gylBpOf-cNIY" -->
## Data Loading and Processing
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 267, "referenced_widgets": ["b3da79a48a094d8c958e7129cc382d84", "f36e9d18e2ba4585b99f690165f286a9", "b65f4b68cb2d46ef8af1171f56a735cd", "bc3ecb0781f64b8781a58f93dff15c7d", "5d837e6e699a4a7fad4f2b3821e79bef", "c095cd98b9b34f96bd2c28f642817c9e", "c56a52e80c6845d6ae040b23fd913430", "8b8431b9d12040adbfa9d316c479a135", "d30d3a3632ee4d8197a6add33a556116", "10a3904ff91845a39d69f178ab545e3a", "47b8edd09b614ed7b804ef350b72f965", "63f53f8fe7ee4a26809973771a00b943", "235a49347bf1437d8c7ea398e7d057a2", "8122a3abc48747dca3f2abdb16e3e2d3", "46dbf8f5e0da434aa0204e08224bc622", "2aeac4580c8a4aae98d0604eec9effce", "08e66ba5c5384af3881f0da705cf212a", "374f61b8a05f4a3a84574ce3247d4b6e", "0b849873407b47ddbc6501851c0b6b57", "8b875e2b42854ca096973ae740c35432", "af9a2906804f461babea1afd462e3a5e", "02fc038820eb4966acf7d04bfe836ccd", "fcf7aa1b4f764989978f32653d21a9ca", "676e9ba66e4d4775a632306405c734b7", "2f81eccc782d44b5ba4cd4ecf0c1df6a", "4d90cf9969344ed69447ae3734ae6432", "6dc951a1a8ac47c5b5ea1a2403a087c0", "d478847f212c46318c4ff89dc3c74252", "364457da9e1b4b6abcb99f33a8b98b50", "40b8b02b95214b5f8ccee2e7963735d5", "61639a72e8db47fcb8caa067e71a7d70", "e3ad859339504c9fa04fa1c739c6ab51", "8feeb2a22c674c8381850356f6ab49ec", "e3b3d295120b43f6819ca0c94180d82d", "a8d8e6214206448c9560501f65055b9c", "17fab15742594801ba96d8be615de281", "1a86e763e52245af9eaa1de718f14b54", "02ef19d9603c4a8fad07f7156a109838", "fd6d349938e0444699a302c341655f98", "f59320ac3e2c451fba6048c87d8b3a8f", "7015c48ce05442218d5f5f481cc2e60b", "78ff6e33c45e4e959ba6d80af0916cfe", "ca6fab2736644982bed7bc12cd62d393", "ef44aad5a7a3440592782c0014557b9d", "484347b925f043d9b82d7e876770bf93", "00815be440a447b7979daf398f89e080", "cfee3d538bbb4a5f9979be4d6efdf0c3", "31f6b8d0492c49c0bc2e2e62768393e4", "d28c889ef4d247598f2e6942198894d2", "55d276166b3e49c890e06169cba5dd2f", "9d33f08184154e3bb4b654797dd23651", "8582137856184efa8f19ff15c772db4c", "44b59b99a12149ea9525d069e9065f1e", "eaf7abf508374eb6a6dcfd5745994b77", "ae5057290dbc4b8695915241fa547b11"]} id="DbiQMqTqcRxx" executionInfo={"status": "ok", "timestamp": 1636021760410, "user_tz": -330, "elapsed": 891970, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="eb73ab77-9fdf-485a-cd16-99380022cad0"
# Ratings data.
ratings = tfds.load("movielens/1m-ratings", split="train", shuffle_files=True)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Fbf24CuDcVD_" executionInfo={"status": "ok", "timestamp": 1636021760412, "user_tz": -330, "elapsed": 26, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f8c58364-ec12-43ed-ee7e-9b01e7da3227"
#The ratings dataset returns a dictionary of movie id, user id, the assigned rating, timestamp, movie information, and user information:
#View the data from ratings dataset:
for x in ratings.take(1).as_numpy_iterator():
    pprint.pprint(x)
```

<!-- #region id="YLCszCo9cVSN" -->
Next, we're only going to extract the movie title and the user id. So, we're actually not going to extract the rating itself and the reason why is that we're treating these as implicit recommendations in this case because that's easier to do a large scale like we mentioned earlier we want to try to keep things simple in the retrieval stage because it's operating over potentially a massive amount of data so to that end we're just going to assume that any movie that a user rated is one that they were really interested in if they took the time to watch it then it expresses some level of interest:
<!-- #endregion -->

```python id="GCtd3FoMcXWa"
#Let's select the necessary attributes:

ratings = ratings.map(lambda x: {
                                 "movie_id": x["movie_id"],
                                 "user_id": x["user_id"],
                                 "user_rating": x["user_rating"],
                                 "user_gender": int(x["user_gender"]),
                                 "user_zip_code": x["user_zip_code"],
                                 "user_occupation_text": x["user_occupation_text"],
                                 "bucketized_user_age": int(x["bucketized_user_age"]),                                
                                })
```

```python id="VO8OTPcqcYm0"
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

```python id="nOlW6_QHcZ_C"
#Then, let's create vocabulary for each feature:

feature_names = ["movie_id", "user_id", "user_gender", "user_zip_code",
                 "user_occupation_text", "bucketized_user_age"]

vocabularies = {}

for feature_name in feature_names:
    vocab = ratings.batch(l).map(lambda x: x[feature_name])
    vocabularies[feature_name] = np.unique(np.concatenate(list(vocab)))
```

<!-- #region id="GUD2kDHAcbdF" -->
## Model Definition
<!-- #endregion -->

<!-- #region id="mdFwIq3tcdqr" -->
The model architecture we will be building starts with an embedding layer, which is fed into a cross network followed by a deep network. The embedding dimension is set to 32 for all the features. You could also use different embedding sizes for different features.
<!-- #endregion -->

```python id="I9i0bZihce3w"
class DCN(tfrs.Model):
    def __init__(self, use_cross_layer, deep_layer_sizes, projection_dim=None):
        super().__init__()

        self.embedding_dimension = 32

        str_features = ["movie_id", "user_id", "user_zip_code",
                        "user_occupation_text"]
        int_features = ["user_gender", "bucketized_user_age"]

        self._all_features = str_features + int_features
        self._embeddings = {}

        # Compute embeddings for string features.
        for feature_name in str_features:
            vocabulary = vocabularies[feature_name]
            self._embeddings[feature_name] = tf.keras.Sequential(
                                                                [tf.keras.layers.experimental.preprocessing.StringLookup(
                                                                 vocabulary=vocabulary, mask_token=None),
                                                                 tf.keras.layers.Embedding(len(vocabulary) + 1,
                                                                 self.embedding_dimension)
                                           ])
      
        # Compute embeddings for int features.
        for feature_name in int_features:
            vocabulary = vocabularies[feature_name]
            self._embeddings[feature_name] = tf.keras.Sequential(
                                                                 [tf.keras.layers.experimental.preprocessing.IntegerLookup(
                                                                 vocabulary=vocabulary, mask_value=None),
                                                                 tf.keras.layers.Embedding(len(vocabulary) + 1,
                                                                 self.embedding_dimension)
                                           ])

        if use_cross_layer:
            self._cross_layer = tfrs.layers.dcn.Cross(
                                                      projection_dim=projection_dim,
                                                      kernel_initializer="glorot_uniform")
        else:
            self._cross_layer = None

        self._deep_layers = [tf.keras.layers.Dense(layer_size, activation="relu")
            for layer_size in deep_layer_sizes]

        self._logit_layer = tf.keras.layers.Dense(1)

        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError("RMSE")]
            )

    def call(self, features):
        # Concatenate embeddings
        embeddings = []
        for feature_name in self._all_features:
            embedding_fn = self._embeddings[feature_name]
            embeddings.append(embedding_fn(features[feature_name]))

        x = tf.concat(embeddings, axis=1)

        # Build Cross Network
        if self._cross_layer is not None:
            x = self._cross_layer(x)
    
    # Build Deep Network
        for deep_layer in self._deep_layers:
            x = deep_layer(x)

        return self._logit_layer(x)

    def compute_loss(self, features, training=False):
        labels = features.pop("user_rating")
        scores = self(features)
        return self.task(labels=labels,predictions=scores,
        )
```

<!-- #region id="F2_ztDQqchAK" -->
## Model Training
<!-- #endregion -->

```python id="QKQvYD2JcjKS"
# Then shuffle, batch, and cache the training and evaluation data:
# Segment the batches so that the model runs 13 training batches (2^13) and 11 test batches (2^11) per epoch, 
# while having a batch size which is a multiple of 2^n.
cached_train = train.shuffle(l).batch(8192).cache()
cached_test = test.batch(2048).cache()
```

<!-- #region id="u3m3DMeOckp2" -->
Now let's define a function that runs a model multiple times and returns the model's RMSE mean and standard deviation out of multiple runs
<!-- #endregion -->

```python id="aWbpasVRcmVw"
def run_models(use_cross_layer, deep_layer_sizes, projection_dim=None, num_runs=5):
    models = []
    rmses = []

    for i in range(num_runs):
        model = DCN(use_cross_layer=use_cross_layer,
                    deep_layer_sizes=deep_layer_sizes,
                    projection_dim=projection_dim)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))
        models.append(model)

        model.fit(cached_train, epochs=epochs, verbose=False)
        metrics = model.evaluate(cached_test, return_dict=True)
        rmses.append(metrics["RMSE"])

    mean, stdv = np.average(rmses), np.std(rmses)

    return {"model": models, "mean": mean, "stdv": stdv}
```

<!-- #region id="LgWrooXNcno7" -->
We set some hyper-parameters for the models. Note that these hyper-parameters are set globally for all the models for demonstration purpose. If you want to obtain the best performance for each model, or conduct a fair comparison among models, then we'd suggest you to fine-tune the hyper-parameters. Remember that the model architecture and optimization schemes are intertwined.


<!-- #endregion -->

```python id="9eQNfwTMcow_"
epochs = 2
learning_rate = 0.01
```

<!-- #region id="5Bu_zRKYctXu" -->
We first train a DCN model with a stacked structure, that is, the inputs are fed to a cross network followed by a deep network.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="MxYesaCwctw3" executionInfo={"status": "ok", "timestamp": 1636022548846, "user_tz": -330, "elapsed": 379087, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c1703df8-c24f-44b7-82b7-d1b6d5160db3"
dcn_result = run_models(use_cross_layer=True, deep_layer_sizes=[192, 192])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 364} id="y76ONm71cxyY" executionInfo={"status": "ok", "timestamp": 1636022548859, "user_tz": -330, "elapsed": 45, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b878d523-4d6c-4923-b33c-bd1ecae9dbb1"
from mpl_toolkits.axes_grid1 import make_axes_locatable

model = dcn_result["model"][0]
mat = model._cross_layer._dense.kernel
features = model._all_features

block_norm = np.ones([len(features), len(features)])
dim = model.embedding_dimension

# Compute the norms of the blocks.
for i in range(len(features)):
    for j in range(len(features)):
        block = mat[i * dim:(i + 1) * dim,
                j * dim:(j + 1) * dim]
        block_norm[i,j] = np.linalg.norm(block, ord="fro")

plt.figure(figsize=(12,12))
im = plt.matshow(block_norm, cmap=plt.cm.Blues)
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im, cax=cax)
cax.tick_params(labelsize=10) 
_ = ax.set_xticklabels([""] + features, rotation=45, ha="left", fontsize=10)
_ = ax.set_yticklabels([""] + features, fontsize=10)
#plt.title('Visualizing the Weight Matrix Learned by DCN')
print('Visualizing the Weight Matrix Learned by DCN')
```

<!-- #region id="X4o-6uVyczF0" -->
One of the nice things about DCN is that you can visualize the weights from the cross network and see if it has successfully learned the important feature process. As shown above, the stronger the interaction between two features is. In this case, the feature cross of user ID and movie ID is of great importance.


<!-- #endregion -->

<!-- #region id="ajpGin5eczS4" -->
To reduce the training and serving cost, we leverage low-rank techniques to approximate the DCN weight matrices. The rank is passed in through argument projection_dim; a smaller projection_dim results in a lower cost. Note that projection_dim needs to be smaller than (input size)/2 to reduce the cost. In practice, we've observed using low-rank DCN with rank (input size)/4 consistently preserved the accuracy of a full-rank DCN.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="FX9VgWV7c43D" executionInfo={"status": "ok", "timestamp": 1636022703245, "user_tz": -330, "elapsed": 154417, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="748031af-c360-4a2f-cf06-04e859ff545d"
dcn_lr_result = run_models(use_cross_layer=True,
                           projection_dim=20,
                           deep_layer_sizes=[192, 192])
```

<!-- #region id="ptrXkbVVc7Wz" -->
We train a same-sized DNN model as a reference.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="StIN236bc9HN" executionInfo={"status": "ok", "timestamp": 1636022883046, "user_tz": -330, "elapsed": 179827, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5451949e-dbc1-43fc-d151-fa2a0cd6e632"
dnn_result = run_models(use_cross_layer=False,
                        deep_layer_sizes=[192, 192, 192])
```

```python colab={"base_uri": "https://localhost:8080/"} id="O8kGoIX7c-4r" executionInfo={"status": "ok", "timestamp": 1636022883047, "user_tz": -330, "elapsed": 37, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="dca3530a-8c16-4cbd-940e-b25d65e953f3"
print("DCN            RMSE mean: {:.4f}, stdv: {:.4f}".format(dcn_result["mean"], dcn_result["stdv"]))
print("DCN (low-rank) RMSE mean: {:.4f}, stdv: {:.4f}".format(dcn_lr_result["mean"], dcn_lr_result["stdv"]))
print("DNN            RMSE mean: {:.4f}, stdv: {:.4f}".format(dnn_result["mean"], dnn_result["stdv"]))
```
