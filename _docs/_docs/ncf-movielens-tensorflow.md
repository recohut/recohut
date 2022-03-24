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

<!-- #region id="7LrG_dfoZXHz" -->
# NCF from scratch in Tensorflow
> We will build a Neural Collaborative Filtering model from scratch in Tensorflow and train it on movielens data. Then we will compare it side by side with lightfm model.

- toc: true
- badges: true
- comments: true
- categories: [Movie, NCF, Tensorflow, LightFM]
- author: "<a href='https://www.youtube.com/channel/UCKfMWrjJmeSdCBNsjZu3Jrw'>dataroots</a>"
- image:
<!-- #endregion -->

<!-- #region id="1mlny70rYqhK" -->
> youtube: https://youtu.be/SD3irxdKfxk
<!-- #endregion -->

<!-- #region id="Hbh3Iop2Y3eK" -->
## Setup
<!-- #endregion -->

```python id="JqmlfPOIOp_g"
!pip install -q lightfm
```

```python id="Us3S7n9WQHMh"
from scipy import sparse
from typing import List
import datetime
import os

import lightfm
import numpy as np
import pandas as pd
import tensorflow as tf
from lightfm import LightFM
from lightfm.datasets import fetch_movielens

import tensorflow.keras as keras
from tensorflow.keras.layers import (
    Concatenate,
    Dense,
    Embedding,
    Flatten,
    Input,
    Multiply,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


import warnings
warnings.filterwarnings("ignore")

%reload_ext google.colab.data_table
%reload_ext tensorboard
```

```python colab={"base_uri": "https://localhost:8080/"} id="TyE5oqxHQoTV" outputId="5cd6e39e-7358-402d-942f-c9d07d209a33"
!pip install -q watermark
%reload_ext watermark
%watermark -m -iv
```

```python id="SK0cNTzQQ2LF"
TOP_K = 5
N_EPOCHS = 10
```

<!-- #region id="5Fik0pDpY5rI" -->
## Load Data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="SgA6bA85Q_Ov" outputId="af120947-1c79-49c8-c94f-51dce8ef2222"
data = fetch_movielens(min_rating=3.0)

print("Interaction matrix:")
print(data["train"].toarray()[:10, :10])
```

```python colab={"base_uri": "https://localhost:8080/"} id="JPCc11s8RMwe" outputId="afc19592-aaa6-40db-e087-50bdee8d7588"
for dataset in ["test", "train"]:
    data[dataset] = (data[dataset].toarray() > 0).astype("int8")

# Make the ratings binary
print("Interaction matrix:")
print(data["train"][:10, :10])

print("\nRatings:")
unique_ratings = np.unique(data["train"])
print(unique_ratings)
```

<!-- #region id="QqGB6gUpY7y8" -->
## Preprocess
<!-- #endregion -->

```python id="ht_ua_B-RgT9"
def wide_to_long(wide: np.array, possible_ratings: List[int]) -> np.array:
    """Go from wide table to long.
    :param wide: wide array with user-item interactions
    :param possible_ratings: list of possible ratings that we may have."""

    def _get_ratings(arr: np.array, rating: int) -> np.array:
        """Generate long array for the rating provided
        :param arr: wide array with user-item interactions
        :param rating: the rating that we are interested"""
        idx = np.where(arr == rating)
        return np.vstack(
            (idx[0], idx[1], np.ones(idx[0].size, dtype="int8") * rating)
        ).T

    long_arrays = []
    for r in possible_ratings:
        long_arrays.append(_get_ratings(wide, r))

    return np.vstack(long_arrays)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 194} id="m8WEVPFaRstc" outputId="c443e069-77a2-44ed-a623-25a532bfed07"
long_train = wide_to_long(data["train"], unique_ratings)
df_train = pd.DataFrame(long_train, columns=["user_id", "item_id", "interaction"])
df_train.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 211} id="ECg5mEuCR2zn" outputId="1059eb6c-d133-4347-a3f2-69afce2c8dae"
print("Only positive interactions:")
df_train[df_train["interaction"] > 0].head()
```

<!-- #region id="KXh2VlaDSYzu" -->
## NCF Model
<!-- #endregion -->

<!-- #region id="B2S1b8hxSXZT" -->
<!-- #endregion -->

```python id="ooic4sNsR-36"
def create_ncf(
    number_of_users: int,
    number_of_items: int,
    latent_dim_mf: int = 4,
    latent_dim_mlp: int = 32,
    reg_mf: int = 0,
    reg_mlp: int = 0.01,
    dense_layers: List[int] = [8, 4],
    reg_layers: List[int] = [0.01, 0.01],
    activation_dense: str = "relu",
) -> keras.Model:

    # input layer
    user = Input(shape=(), dtype="int32", name="user_id")
    item = Input(shape=(), dtype="int32", name="item_id")

    # embedding layers
    mf_user_embedding = Embedding(
        input_dim=number_of_users,
        output_dim=latent_dim_mf,
        name="mf_user_embedding",
        embeddings_initializer="RandomNormal",
        embeddings_regularizer=l2(reg_mf),
        input_length=1,
    )
    mf_item_embedding = Embedding(
        input_dim=number_of_items,
        output_dim=latent_dim_mf,
        name="mf_item_embedding",
        embeddings_initializer="RandomNormal",
        embeddings_regularizer=l2(reg_mf),
        input_length=1,
    )

    mlp_user_embedding = Embedding(
        input_dim=number_of_users,
        output_dim=latent_dim_mlp,
        name="mlp_user_embedding",
        embeddings_initializer="RandomNormal",
        embeddings_regularizer=l2(reg_mlp),
        input_length=1,
    )
    mlp_item_embedding = Embedding(
        input_dim=number_of_items,
        output_dim=latent_dim_mlp,
        name="mlp_item_embedding",
        embeddings_initializer="RandomNormal",
        embeddings_regularizer=l2(reg_mlp),
        input_length=1,
    )

    # MF vector
    mf_user_latent = Flatten()(mf_user_embedding(user))
    mf_item_latent = Flatten()(mf_item_embedding(item))
    mf_cat_latent = Multiply()([mf_user_latent, mf_item_latent])

    # MLP vector
    mlp_user_latent = Flatten()(mlp_user_embedding(user))
    mlp_item_latent = Flatten()(mlp_item_embedding(item))
    mlp_cat_latent = Concatenate()([mlp_user_latent, mlp_item_latent])

    mlp_vector = mlp_cat_latent

    # build dense layers for model
    for i in range(len(dense_layers)):
        layer = Dense(
            dense_layers[i],
            activity_regularizer=l2(reg_layers[i]),
            activation=activation_dense,
            name="layer%d" % i,
        )
        mlp_vector = layer(mlp_vector)

    predict_layer = Concatenate()([mf_cat_latent, mlp_vector])

    result = Dense(
        1, activation="sigmoid", kernel_initializer="lecun_uniform", name="interaction"
    )

    output = result(predict_layer)

    model = Model(
        inputs=[user, item],
        outputs=[output],
    )

    return model
```

```python colab={"base_uri": "https://localhost:8080/"} id="GQWaSOCrSwSl" outputId="b2d27924-0954-48ed-812e-83b52b132ccd"
n_users, n_items = data["train"].shape
ncf_model = create_ncf(n_users, n_items)

ncf_model.compile(
    optimizer=Adam(),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.TruePositives(name="tp"),
        tf.keras.metrics.FalsePositives(name="fp"),
        tf.keras.metrics.TrueNegatives(name="tn"),
        tf.keras.metrics.FalseNegatives(name="fn"),
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)
ncf_model._name = "neural_collaborative_filtering"
ncf_model.summary()
```

<!-- #region id="KeATHv3FZA7Q" -->
## TF Dataset
<!-- #endregion -->

```python id="aOuwbTKfS6f1"
def make_tf_dataset(
    df: pd.DataFrame,
    targets: List[str],
    val_split: float = 0.1,
    batch_size: int = 512,
    seed=42,
):
    """Make TensorFlow dataset from Pandas DataFrame.
    :param df: input DataFrame - only contains features and target(s)
    :param targets: list of columns names corresponding to targets
    :param val_split: fraction of the data that should be used for validation
    :param batch_size: batch size for training
    :param seed: random seed for shuffling data - `None` won't shuffle the data"""

    n_val = round(df.shape[0] * val_split)
    if seed:
        # shuffle all the rows
        x = df.sample(frac=1, random_state=seed).to_dict("series")
    else:
        x = df.to_dict("series")
    y = dict()
    for t in targets:
        y[t] = x.pop(t)
    ds = tf.data.Dataset.from_tensor_slices((x, y))

    ds_val = ds.take(n_val).batch(batch_size)
    ds_train = ds.skip(n_val).batch(batch_size)
    return ds_train, ds_val
```

```python id="F6QXgUQgTP8d"
# create train and validation datasets
ds_train, ds_val = make_tf_dataset(df_train, ["interaction"])
```

<!-- #region id="S0COxKxkZDh7" -->
## Model Training
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="o46oVj9eTTaq" outputId="5f2f5f96-cc07-45c7-82dc-30ddfb79f7bb"
%%time
# define logs and callbacks
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=2
)

train_hist = ncf_model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=N_EPOCHS,
    callbacks=[tensorboard_callback, early_stopping_callback],
    verbose=1,
)
```

```python id="SHcbr5aYUWSA"
%tensorboard --logdir logs
```

<!-- #region id="8cFvLAbLZHL9" -->
## Inference
<!-- #endregion -->

```python id="2z5WMWIOVFKo"
long_test = wide_to_long(data["train"], unique_ratings)
df_test = pd.DataFrame(long_test, columns=["user_id", "item_id", "interaction"])
ds_test, _ = make_tf_dataset(df_test, ["interaction"], val_split=0, seed=None)
```

```python id="Yo7BxW_rWHO3"
ncf_predictions = ncf_model.predict(ds_test)
df_test["ncf_predictions"] = ncf_predictions
```

```python colab={"base_uri": "https://localhost:8080/", "height": 194} id="MVoy544bWJjH" outputId="3174fd54-6625-488b-aa3b-ad83730fe5a0"
df_test.head()
```

<!-- #region id="03nWlokgX1IX" -->
> Tip: sanity checks. stop execution if low standard deviation (all recommendations are the same)
<!-- #endregion -->

```python id="Gq4B3-uJXxAp"
std = df_test.describe().loc["std", "ncf_predictions"]
if std < 0.01:
    raise ValueError("Model predictions have standard deviation of less than 1e-2.")
```

```python colab={"base_uri": "https://localhost:8080/"} id="OlcxHaCiYCPN" outputId="4e3b66cc-fcc9-486c-9ce7-0d8cdd3a71b7"
data["ncf_predictions"] = df_test.pivot(
    index="user_id", columns="item_id", values="ncf_predictions"
).values
print("Neural collaborative filtering predictions")
print(data["ncf_predictions"][:10, :4])
```

```python colab={"base_uri": "https://localhost:8080/"} id="MEiYaieSYFeh" outputId="fe7cab95-8633-4bd0-fbba-8a6bc399fa07"
precision_ncf = tf.keras.metrics.Precision(top_k=TOP_K)
recall_ncf = tf.keras.metrics.Recall(top_k=TOP_K)

precision_ncf.update_state(data["test"], data["ncf_predictions"])
recall_ncf.update_state(data["test"], data["ncf_predictions"])
print(
    f"At K = {TOP_K}, we have a precision of {precision_ncf.result().numpy():.5f}, and a recall of {recall_ncf.result().numpy():.5f}",
)
```

<!-- #region id="0TRwpTFyZK1O" -->
## Comparison with LightFM (WARP loss) model
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="YxFxJvZmYIBi" outputId="f32bd42c-3da1-49d0-e3e5-aba730a305dd"
# LightFM model
def norm(x: float) -> float:
    """Normalize vector"""
    return (x - np.min(x)) / np.ptp(x)


lightfm_model = LightFM(loss="warp")
lightfm_model.fit(sparse.coo_matrix(data["train"]), epochs=N_EPOCHS)

lightfm_predictions = lightfm_model.predict(
    df_test["user_id"].values, df_test["item_id"].values
)
df_test["lightfm_predictions"] = lightfm_predictions
wide_predictions = df_test.pivot(
    index="user_id", columns="item_id", values="lightfm_predictions"
).values
data["lightfm_predictions"] = norm(wide_predictions)

# compute the metrics
precision_lightfm = tf.keras.metrics.Precision(top_k=TOP_K)
recall_lightfm = tf.keras.metrics.Recall(top_k=TOP_K)
precision_lightfm.update_state(data["test"], data["lightfm_predictions"])
recall_lightfm.update_state(data["test"], data["lightfm_predictions"])
print(
    f"At K = {TOP_K}, we have a precision of {precision_lightfm.result().numpy():.5f}, and a recall of {recall_lightfm.result().numpy():.5f}",
)
```
