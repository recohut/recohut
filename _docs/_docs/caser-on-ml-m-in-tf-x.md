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

```python id="cafo0FHzZf96"
!pip install tensorflow==2.5.0
```

```python colab={"base_uri": "https://localhost:8080/"} id="dlmK8Cv3ZqRU" executionInfo={"status": "ok", "timestamp": 1637060541417, "user_tz": -330, "elapsed": 1441, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="48365c23-c151-4416-e8d9-2d1efab9e47e"
!wget -q --show-progress https://files.grouplens.org/datasets/movielens/ml-1m.zip
!unzip ml-1m.zip
```

```python id="4pyUJOqTZoA-" executionInfo={"status": "ok", "timestamp": 1637060562489, "user_tz": -330, "elapsed": 2421, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import os
from time import time

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Input, Conv1D, GlobalMaxPooling1D, Dense, Dropout
```

```python id="R-tNVTLpaLQX" executionInfo={"status": "ok", "timestamp": 1637060562492, "user_tz": -330, "elapsed": 18, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

file = 'ml-1m/ratings.dat'
trans_score = 1
maxlen = 200

embed_dim = 50  # 32
hor_n = 8
hor_h = 2
ver_n = 4
dropout = 0.2
activation = 'relu'
embed_reg = 1e-6
K = 10

learning_rate = 0.001
epochs = 10
batch_size = 512
```

```python id="YNaRlOMlZsuP" executionInfo={"status": "ok", "timestamp": 1637060562494, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
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

```python id="R-UK1cSTZu--" executionInfo={"status": "ok", "timestamp": 1637060563295, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def create_implicit_ml_1m_dataset(file, trans_score=2, embed_dim=8, maxlen=40):
    """
    :param file: A string. dataset path.
    :param trans_score: A scalar. Greater than it is 1, and less than it is 0.
    :param embed_dim: A scalar. latent factor.
    :param maxlen: A scalar. maxlen.
    :return: user_num, item_num, train_df, test_df
    """
    print('==========Data Preprocess Start=============')
    data_df = pd.read_csv(file, sep="::", engine='python',
                          names=['user_id', 'item_id', 'label', 'Timestamp'])
    # implicit dataset
    data_df = data_df[data_df.label >= trans_score]

    # sort
    data_df = data_df.sort_values(by=['user_id', 'Timestamp'])

    train_data, val_data, test_data = [], [], []

    item_id_max = data_df['item_id'].max()
    for user_id, df in tqdm(data_df[['user_id', 'item_id']].groupby('user_id')):
        pos_list = df['item_id'].tolist()

        def gen_neg():
            neg = pos_list[0]
            while neg in pos_list:
                neg = random.randint(1, item_id_max)
            return neg

        neg_list = [gen_neg() for i in range(len(pos_list) + 100)]
        for i in range(1, len(pos_list)):
            hist_i = pos_list[:i]
            if i == len(pos_list) - 1:
                test_data.append([user_id, hist_i, pos_list[i], 1])
                for neg in neg_list[i:]:
                    test_data.append([user_id, hist_i, neg, 0])
            elif i == len(pos_list) - 2:
                val_data.append([user_id, hist_i, pos_list[i], 1])
                val_data.append([user_id, hist_i, neg_list[i], 0])
            else:
                train_data.append([user_id, hist_i, pos_list[i], 1])
                train_data.append([user_id, hist_i, neg_list[i], 0])
    # item feature columns
    user_num, item_num = data_df['user_id'].max() + 1, data_df['item_id'].max() + 1
    feature_columns = [sparseFeature('user_id', user_num, embed_dim),
                       sparseFeature('item_id', item_num, embed_dim)]

    # shuffle
    random.shuffle(train_data)
    random.shuffle(val_data)
    # random.shuffle(test_data)

    # create dataframe
    train = pd.DataFrame(train_data, columns=['user_id', 'hist', 'target_item', 'label'])
    val = pd.DataFrame(val_data, columns=['user_id', 'hist', 'target_item', 'label'])
    test = pd.DataFrame(test_data, columns=['user_id', 'hist', 'target_item', 'label'])

    print('==================Padding===================')
    train_X = [train['user_id'].values, pad_sequences(train['hist'], maxlen=maxlen), train['target_item'].values]
    train_y = train['label'].values
    val_X = [val['user_id'].values, pad_sequences(val['hist'], maxlen=maxlen), val['target_item'].values]
    val_y = val['label'].values
    test_X = [test['user_id'].values, pad_sequences(test['hist'], maxlen=maxlen), test['target_item'].values]
    test_y = test['label'].values.tolist()
    print('============Data Preprocess End=============')
    return feature_columns, (train_X, train_y), (val_X, val_y), (test_X, test_y)
```

```python id="xK7G8OWQZx-z" executionInfo={"status": "ok", "timestamp": 1637060564057, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class Caser(Model):
    def __init__(self, feature_columns, maxlen=40, hor_n=2, hor_h=8, ver_n=8, dropout=0.5, activation='relu', embed_reg=1e-6):
        """
        AttRec
        :param feature_columns: A feature columns list. user + seq
        :param maxlen: A scalar. In the paper, maxlen is L, the number of latest items.
        :param hor_n: A scalar. The number of horizontal filters.
        :param hor_h: A scalar. Height of horizontal filters.
        :param ver_n: A scalar. The number of vertical filters.
        :param dropout: A scalar. The number of dropout.
        :param activation: A string. 'relu', 'sigmoid' or 'tanh'.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(Caser, self).__init__()
        # maxlen
        self.maxlen = maxlen
        # feature columns
        self.user_fea_col, self.item_fea_col = feature_columns
        # embed_dim
        self.embed_dim = self.item_fea_col['embed_dim']
        # total number of item set
        self.total_item = self.item_fea_col['feat_num']
        # horizontal filters
        self.hor_n = hor_n
        self.hor_h = hor_h if hor_h <= self.maxlen else self.maxlen
        # vertical filters
        self.ver_n = ver_n
        self.ver_w = 1
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
        # item2 embedding
        self.item2_embedding = Embedding(input_dim=self.item_fea_col['feat_num'],
                                        input_length=1,
                                        output_dim=self.item_fea_col['embed_dim'] * 2,
                                        mask_zero=True,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))
        # horizontal conv
        self.hor_conv = Conv1D(filters=self.hor_n, kernel_size=self.hor_h)
        # vertical conv, should transpose
        self.ver_conv = Conv1D(filters=self.ver_n, kernel_size=self.ver_w)
        # max_pooling
        self.pooling = GlobalMaxPooling1D()
        # dense
        self.dense = Dense(self.embed_dim, activation=activation)
        self.dropout = Dropout(dropout)

    def call(self, inputs):
        # input
        user_inputs, seq_inputs, item_inputs = inputs
        # user info
        user_embed = self.user_embedding(tf.squeeze(user_inputs, axis=-1))  # (None, dim)
        # seq info
        seq_embed = self.item_embedding(seq_inputs)  # (None, maxlen, dim)
        # horizontal conv (None, (maxlen - kernel_size + 2 * pad) / stride +1, hor_n)
        hor_info = self.hor_conv(seq_embed)
        hor_info = self.pooling(hor_info)  # (None, hor_n)
        # vertical conv  (None, (dim - 1 + 2 * pad) / stride + 1, ver_n)
        ver_info = self.ver_conv(tf.transpose(seq_embed, perm=(0, 2, 1)))
        ver_info = tf.reshape(ver_info, shape=(-1, ver_info.shape[1] * ver_info.shape[2]))  # (None, ?)
        # info
        seq_info = self.dense(tf.concat([hor_info, ver_info], axis=-1))  # (None, d)
        seq_info = self.dropout(seq_info)
        # concat
        info = tf.concat([seq_info, user_embed], axis=-1)  # (None, 2 * d)
        # item info
        item_embed = self.item2_embedding(tf.squeeze(item_inputs, axis=-1))  # (None, dim)
        # predict
        outputs = tf.nn.sigmoid(tf.reduce_sum(tf.multiply(info, item_embed), axis=1, keepdims=True))
        return outputs

    def summary(self):
        seq_inputs = Input(shape=(self.maxlen,), dtype=tf.int32)
        user_inputs = Input(shape=(1, ), dtype=tf.int32)
        item_inputs = Input(shape=(1,), dtype=tf.int32)
        Model(inputs=[user_inputs, seq_inputs, item_inputs],
              outputs=self.call([user_inputs, seq_inputs, item_inputs])).summary()
```

```python id="Jkv5hwHmZx73" executionInfo={"status": "ok", "timestamp": 1637060564802, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def test_model():
    user_features = {'feat': 'user_id', 'feat_num': 100, 'embed_dim': 8}
    seq_features = {'feat': 'item_id', 'feat_num': 100, 'embed_dim': 8}

    features = [user_features, seq_features]
    model = Caser(features)
    model.summary()
```

```python id="OlM4xkuzZx42" executionInfo={"status": "ok", "timestamp": 1637060564806, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def getHit(df):
    """
    calculate hit rate
    :return:
    """
    df = df.sort_values('pred_y', ascending=False).reset_index()
    if df[df.true_y == 1].index.tolist()[0] < _K:
        return 1
    else:
        return 0


def getNDCG(df):
    """
    calculate NDCG
    :return:
    """
    df = df.sort_values('pred_y', ascending=False).reset_index()
    i = df[df.true_y == 1].index.tolist()[0]
    if i < _K:
        return np.log(2) / np.log(i+2)
    else:
        return 0.


def evaluate_model(model, test, K):
    """
    evaluate model
    :param model: model
    :param test: test set
    :param K: top K
    :return: hit rate, ndcg
    """
    global _K
    _K = K
    test_X, test_y = test
    pred_y = model.predict(test_X)
    test_df = pd.DataFrame(test_y, columns=['true_y'])
    test_df['user_id'] = test_X[0]
    test_df['pred_y'] = pred_y
    tg = test_df.groupby('user_id')
    hit_rate = tg.apply(getHit).mean()
    ndcg = tg.apply(getNDCG).mean()
    return hit_rate, ndcg
```

```python colab={"base_uri": "https://localhost:8080/"} id="03hx2SFsaSEP" executionInfo={"status": "ok", "timestamp": 1637066167282, "user_tz": -330, "elapsed": 5594061, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="58df5e73-71c2-4728-b22d-e02842b72c44"
# ========================== Create dataset =======================
feature_columns, train, val, test = create_implicit_ml_1m_dataset(file, trans_score, embed_dim, maxlen)
train_X, train_y = train
val_X, val_y = val

# ============================Build Model==========================
model = Caser(feature_columns, maxlen, hor_n, hor_h, ver_n, dropout, activation, embed_reg)
model.summary()
# =========================Compile============================
model.compile(loss=BinaryCrossentropy(), optimizer=Adam(learning_rate=learning_rate))

results = []
for epoch in range(1, epochs + 1):
    # ===========================Fit==============================
    t1 = time()
    model.fit(
        train_X,
        train_y,
        validation_data=(val_X, val_y),
        epochs=1,
        batch_size=batch_size,
    )
    # ===========================Test==============================
    t2 = time()
    if epoch % 5 == 0:
        hit_rate, ndcg = evaluate_model(model, test, K)
        print('Iteration %d Fit [%.1f s], Evaluate [%.1f s]: HR = %.4f, NDCG= %.4f'
                % (epoch, t2 - t1, time() - t2, hit_rate, ndcg))
        results.append([epoch + 1, t2 - t1, time() - t2, hit_rate, ndcg])
# ============================Write============================
pd.DataFrame(results, columns=['Iteration', 'fit_time', 'evaluate_time', 'hit_rate', 'ndcg']).\
    to_csv('Caser_log_maxlen_{}_dim_{}_hor_n_{}_ver_n_{}_K_{}_.csv'.
            format(maxlen, embed_dim, hor_n, ver_n, K), index=False)
```

```python id="eTbvyzKFadMa"

```
