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

<!-- #region id="ctgFHhdob5kJ" -->
# BPR on ML-1m in Tensorflow
<!-- #endregion -->

```python id="hcT4DFhGGqkR"
!pip install tensorflow==2.5.0
```

```python colab={"base_uri": "https://localhost:8080/"} id="UiIrgWPHA_sZ" executionInfo={"status": "ok", "timestamp": 1637055802556, "user_tz": -330, "elapsed": 1494, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a512eac2-8721-45c6-d592-6c94f34213b7"
!wget -q --show-progress https://files.grouplens.org/datasets/movielens/ml-1m.zip
!unzip ml-1m.zip
```

```python id="PruyIn0iAyRx"
import os
import pandas as pd
import numpy as np
import random
from time import time
from tqdm.notebook import tqdm
from collections import defaultdict

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Embedding, Input
from tensorflow.keras.regularizers import l2
```

```python id="8LOSVVJUC6au"
!pip install -q watermark
%reload_ext watermark
%watermark -m -iv -u -t -d
```

```python id="9vSaa-LDB4KO"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

file = 'ml-1m/ratings.dat'
trans_score = 1
test_neg_num = 100

embed_dim = 64
mode = 'inner'  # dist
embed_reg = 1e-6  # 1e-6
K = 10 # top-k

learning_rate = 0.001
epochs = 20
batch_size = 512
```

```python id="6YXVIHnhA0_B"
def sparseFeature(feat, feat_num, embed_dim=4):
    """
    create dictionary for sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return:
    """
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}
```

```python id="PDHSNwWoA3Lm"
def create_ml_1m_dataset(file, trans_score=2, embed_dim=8, test_neg_num=100):
    """
    :param file: A string. dataset path.
    :param trans_score: A scalar. Greater than it is 1, and less than it is 0.
    :param embed_dim: A scalar. latent factor.
    :param test_neg_num: A scalar. The number of test negative samples
    :return: user_num, item_num, train_df, test_df
    """
    print('==========Data Preprocess Start=============')
    data_df = pd.read_csv(file, sep="::", engine='python',
                          names=['user_id', 'item_id', 'label', 'Timestamp'])
    # filtering
    data_df['item_count'] = data_df.groupby('item_id')['item_id'].transform('count')
    data_df = data_df[data_df.item_count >= 5]
    # trans score
    data_df = data_df[data_df.label >= trans_score]
    # sort
    data_df = data_df.sort_values(by=['user_id', 'Timestamp'])
    # split dataset and negative sampling
    print('============Negative Sampling===============')
    train_data, val_data, test_data = defaultdict(list), defaultdict(list), defaultdict(list)
    item_id_max = data_df['item_id'].max()
    for user_id, df in tqdm(data_df[['user_id', 'item_id']].groupby('user_id')):
        pos_list = df['item_id'].tolist()

        def gen_neg():
            neg = pos_list[0]
            while neg in set(pos_list):
                neg = random.randint(1, item_id_max)
            return neg

        neg_list = [gen_neg() for i in range(len(pos_list) + test_neg_num)]
        for i in range(1, len(pos_list)):
            hist_i = pos_list[:i]
            if i == len(pos_list) - 1:
                test_data['user_id'].append(user_id)
                test_data['pos_id'].append(pos_list[i])
                test_data['neg_id'].append(neg_list[i:])
            elif i == len(pos_list) - 2:
                val_data['user_id'].append(user_id)
                val_data['pos_id'].append(pos_list[i])
                val_data['neg_id'].append(neg_list[i])
            else:
                train_data['user_id'].append(user_id)
                train_data['pos_id'].append(pos_list[i])
                train_data['neg_id'].append(neg_list[i])
    # feature columns
    user_num, item_num = data_df['user_id'].max() + 1, data_df['item_id'].max() + 1
    feat_col = [sparseFeature('user_id', user_num, embed_dim),
                sparseFeature('item_id', item_num, embed_dim)]
    # shuffle
    random.shuffle(train_data)
    random.shuffle(val_data)
    train = [np.array(train_data['user_id']), np.array(train_data['pos_id']),
               np.array(train_data['neg_id'])]
    val = [np.array(val_data['user_id']), np.array(val_data['pos_id']),
             np.array(val_data['neg_id'])]
    test = [np.array(test_data['user_id']), np.array(test_data['pos_id']),
              np.array(test_data['neg_id'])]
    print('============Data Preprocess End=============')
    return feat_col, train, val, test
```

```python id="5rs0iTy3BhSF"
class BPR(Model):
    def __init__(self, feature_columns, mode='inner', embed_reg=1e-6):
        """
        BPR
        :param feature_columns: A list. user feature columns + item feature columns
        :mode: A string. 'inner' or 'dist'.
        :param embed_reg: A scalar.  The regularizer of embedding.
        """
        super(BPR, self).__init__()
        # feature columns
        self.user_fea_col, self.item_fea_col = feature_columns
        # mode
        self.mode = mode
        # user embedding
        self.user_embedding = Embedding(input_dim=self.user_fea_col['feat_num'],
                                        input_length=1,
                                        output_dim=self.user_fea_col['embed_dim'],
                                        mask_zero=False,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))
        # item embedding
        self.item_embedding = Embedding(input_dim=self.item_fea_col['feat_num'],
                                        input_length=1,
                                        output_dim=self.item_fea_col['embed_dim'],
                                        mask_zero=True,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))

    def call(self, inputs):
        user_inputs, pos_inputs, neg_inputs = inputs  # (None, 1), (None, 1)
        # user info
        user_embed = self.user_embedding(user_inputs)  # (None, 1, dim)
        # item
        pos_embed = self.item_embedding(pos_inputs)  # (None, 1, dim)
        neg_embed = self.item_embedding(neg_inputs)  # (None, 1, dim)
        if self.mode == 'inner':
            # calculate positive item scores and negative item scores
            pos_scores = tf.reduce_sum(tf.multiply(user_embed, pos_embed), axis=-1)  # (None, 1)
            neg_scores = tf.reduce_sum(tf.multiply(user_embed, neg_embed), axis=-1)  # (None, 1)
            # add loss. Computes softplus: log(exp(features) + 1)
            # self.add_loss(tf.reduce_mean(tf.math.softplus(neg_scores - pos_scores)))
            self.add_loss(tf.reduce_mean(-tf.math.log(tf.nn.sigmoid(pos_scores - neg_scores))))
        else:
            # clip by norm
            # user_embed = tf.clip_by_norm(user_embed, 1, -1)
            # pos_embed = tf.clip_by_norm(pos_embed, 1, -1)
            # neg_embed = tf.clip_by_norm(neg_embed, 1, -1)
            pos_scores = tf.reduce_sum(tf.square(user_embed - pos_embed), axis=-1)
            neg_scores = tf.reduce_sum(tf.square(user_embed - neg_embed), axis=-1)
            self.add_loss(tf.reduce_sum(tf.nn.relu(pos_scores - neg_scores + 0.5)))
        logits = tf.concat([pos_scores, neg_scores], axis=-1)
        return logits

    def summary(self):
        user_inputs = Input(shape=(1, ), dtype=tf.int32)
        pos_inputs = Input(shape=(1, ), dtype=tf.int32)
        neg_inputs = Input(shape=(1, ), dtype=tf.int32)
        Model(inputs=[user_inputs, pos_inputs, neg_inputs],
            outputs=self.call([user_inputs, pos_inputs, neg_inputs])).summary()
```

```python id="FiecGHPuBjQT"
def test_model():
    user_features = {'feat': 'user_id', 'feat_num': 100, 'embed_dim': 8}
    item_features = {'feat': 'item_id', 'feat_num': 100, 'embed_dim': 8}
    features = [user_features, item_features]
    model = BPR(features)
    model.summary()
```

```python id="tYj86oraBrKR"
def evaluate_model(model, test, K):
    """
    evaluate model
    :param model: model
    :param test: test set
    :param K: top K
    :return: hit rate, ndcg
    """
    if model.mode == 'inner':
        pred_y = - model.predict(test)
    else:
        pred_y = model.predict(test)
    rank = pred_y.argsort().argsort()[:, 0]
    hr, ndcg = 0.0, 0.0
    for r in rank:
        if r < K:
            hr += 1
            ndcg += 1 / np.log2(r + 2)
    return hr / len(rank), ndcg / len(rank)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["61aaba58d3324661b52a90a48a6946ba", "2ff544deb2e44e0d959271097e9c44af", "db1d007657c6485f9cc738ca12e8ac8a", "9ce1de0d27714039b59ab7b1255abc29", "9df995714bd341059e85b779506fecb5", "9c47c25edd844ad88d695c4da097b565", "33cffb5d3c124b69b77c7ddb9fdb6e7a", "0e8dcaee6ec14dd9a548807282865f64", "bf828936d7a84a0fa093ac6ac4463090", "e12e386149c14075b2d2362a6982298c", "5ded68d3108f4f6c9155677c7db4fd99"]} id="iTbYFKSzCE1I" executionInfo={"status": "ok", "timestamp": 1637056214823, "user_tz": -330, "elapsed": 397148, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7d72f137-9f0d-4c84-9ea7-e253fe03b0d7"
# ========================== Create dataset =======================
feature_columns, train, val, test = create_ml_1m_dataset(file, trans_score, embed_dim, test_neg_num)

# ============================Build Model==========================
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = BPR(feature_columns, mode, embed_reg)
    model.summary()
    # =========================Compile============================
    model.compile(optimizer=Adam(learning_rate=learning_rate))

results = []
for epoch in range(1, epochs + 1):
    # ===========================Fit==============================
    t1 = time()
    model.fit(
        train,
        None,
        validation_data=(val, None),
        epochs=1,
        batch_size=batch_size,
    )
    # ===========================Test==============================
    t2 = time()
    if epoch % 5 == 0:
        hit_rate, ndcg = evaluate_model(model, test, K)
        print('Iteration %d Fit [%.1f s], Evaluate [%.1f s]: HR = %.4f, NDCG = %.4f'
                % (epoch, t2 - t1, time() - t2, hit_rate, ndcg))
        results.append([epoch, t2 - t1, time() - t2, hit_rate, ndcg])
# ========================== Write Log ===========================
pd.DataFrame(results, columns=['Iteration', 'fit_time', 'evaluate_time', 'hit_rate', 'ndcg'])\
    .to_csv('BPR_log_dim_{}_mode_{}_K_{}.csv'.format(embed_dim, mode, K), index=False)
```
