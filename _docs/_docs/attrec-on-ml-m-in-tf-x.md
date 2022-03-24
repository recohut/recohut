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

```python id="gr_exnbyFd9P"
!pip install tensorflow==2.5.0
```

```python colab={"base_uri": "https://localhost:8080/"} id="UiIrgWPHA_sZ" executionInfo={"status": "ok", "timestamp": 1637053914012, "user_tz": -330, "elapsed": 2087, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7ea8e7cd-ed25-4dde-99b3-79000be5aea4"
!wget -q --show-progress https://files.grouplens.org/datasets/movielens/ml-1m.zip
!unzip ml-1m.zip
```

```python id="PruyIn0iAyRx" executionInfo={"status": "ok", "timestamp": 1637055306678, "user_tz": -330, "elapsed": 2559, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
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
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.losses import Loss
```

```python id="8LOSVVJUC6au"
!pip install -q watermark
%reload_ext watermark
%watermark -m -iv -u -t -d
```

```python id="9vSaa-LDB4KO" executionInfo={"status": "ok", "timestamp": 1637055307254, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

file = 'ml-1m/ratings.dat'
trans_score = 1
maxlen = 5

embed_dim = 100
embed_reg = 1e-6  # 1e-6
gamma = 0.5
mode = 'inner'  # 'inner' or 'dist'
w = 0.5
K = 10

learning_rate = 0.001
epochs = 40
batch_size = 1024
```

```python id="6YXVIHnhA0_B" executionInfo={"status": "ok", "timestamp": 1637055309502, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
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

```python id="PDHSNwWoA3Lm" executionInfo={"status": "ok", "timestamp": 1637055310302, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def create_implicit_ml_1m_dataset(file, trans_score=2, embed_dim=8, maxlen=40):
    """
    :param file: A string. dataset path.
    :param trans_score: A scalar. Greater than it is 1, and less than it is 0.
    :param embed_dim: A scalar. latent factor.
    :param maxlen: A scalar. maxlen.
    :return: user_num, item_num, train_df, test_df
    """
    print('==========Data Preprocess Start============')
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
                for neg in neg_list[i:]:
                    test_data.append([user_id, hist_i, pos_list[i], neg])
            elif i == len(pos_list) - 2:
                val_data.append([user_id, hist_i, pos_list[i], neg_list[i]])
            else:
                train_data.append([user_id, hist_i, pos_list[i], neg_list[i]])

    # feature columns
    user_num, item_num = data_df['user_id'].max() + 1, data_df['item_id'].max() + 1
    feature_columns = [sparseFeature('user_id', user_num, embed_dim),
                       sparseFeature('item_id', item_num, embed_dim)]

    # shuffle
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    # create dataframe
    train = pd.DataFrame(train_data, columns=['user_id', 'hist', 'pos_item', 'neg_item'])
    val = pd.DataFrame(val_data, columns=['user_id', 'hist', 'pos_item', 'neg_item'])
    test = pd.DataFrame(test_data, columns=['user_id', 'hist', 'pos_item', 'neg_item'])
    print('==================Padding===================')

    # create dataset
    def df_to_list(data):
        return [data['user_id'].values, pad_sequences(data['hist'], maxlen=maxlen),
                data['pos_item'].values, data['neg_item'].values]

    train_X = df_to_list(train)
    val_X = df_to_list(val)
    test_X = df_to_list(test)
    print('============Data Preprocess End=============')
    return feature_columns, train_X, val_X, test_X
```

```python id="ADXVraJMEp0E" executionInfo={"status": "ok", "timestamp": 1637055311466, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class SelfAttention_Layer(Layer):
    def __init__(self):
        super(SelfAttention_Layer, self).__init__()

    def build(self, input_shape):
        self.dim = input_shape[0][-1]
        self.W = self.add_weight(shape=[self.dim, self.dim], name='weight', 
            initializer='random_uniform')

    def call(self, inputs, **kwargs):
        q, k, v, mask = inputs
        # pos encoding
        k += self.positional_encoding(k)
        q += self.positional_encoding(q)
        # Nonlinear transformation
        q = tf.nn.relu(tf.matmul(q, self.W))  # (None, seq_len, dim)
        k = tf.nn.relu(tf.matmul(k, self.W))  # (None, seq_len, dim)
        mat_qk = tf.matmul(q, k, transpose_b=True)  # (None, seq_len, seq_len)
        dk = tf.cast(self.dim, dtype=tf.float32)
        # Scaled
        scaled_att_logits = mat_qk / tf.sqrt(dk)
        # Mask
        mask = tf.tile(tf.expand_dims(mask, 1), [1, q.shape[1], 1])  # (None, seq_len, seq_len)
        paddings = tf.ones_like(scaled_att_logits) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(mask, 0), paddings, scaled_att_logits)  # (None, seq_len, seq_len)
        # softmax
        outputs = tf.nn.softmax(logits=outputs, axis=-1)  # (None, seq_len, seq_len)
        # output
        outputs = tf.matmul(outputs, v)  # (None, seq_len, dim)
        outputs = tf.reduce_mean(outputs, axis=1)  # (None, dim)
        return outputs

    @staticmethod
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, QK_input):
        angle_rads = self.get_angles(np.arange(QK_input.shape[1])[:, np.newaxis],
                                np.arange(self.dim)[np.newaxis, :], self.dim)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)
```

```python id="5rs0iTy3BhSF" executionInfo={"status": "ok", "timestamp": 1637055312873, "user_tz": -330, "elapsed": 633, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class AttRec(Model):
    def __init__(self, feature_columns, maxlen=40, mode='inner', gamma=0.5, w=0.5, embed_reg=1e-6, **kwargs):
        """
        AttRec
        :param feature_columns: A feature columns list. user + seq
        :param maxlen: A scalar. In the paper, maxlen is L, the number of latest items.
        :param gamma: A scalar. if mode == 'dist', gamma is the margin.
        :param mode: A string. inner or dist.
        :param w: A scalar. The weight of short interest.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(AttRec, self).__init__(**kwargs)
        # maxlen
        self.maxlen = maxlen
        # w
        self.w = w
        self.gamma = gamma
        self.mode = mode
        # feature columns
        self.user_fea_col, self.item_fea_col = feature_columns
        # embed_dim
        self.embed_dim = self.item_fea_col['embed_dim']
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
        # item2 embedding, not share embedding
        self.item2_embedding = Embedding(input_dim=self.item_fea_col['feat_num'],
                                        input_length=1,
                                        output_dim=self.item_fea_col['embed_dim'],
                                        mask_zero=True,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))
        # self-attention
        self.self_attention = SelfAttention_Layer()

    def call(self, inputs, **kwargs):
        # input
        user_inputs, seq_inputs, pos_inputs, neg_inputs = inputs
        # mask
        # mask = self.item_embedding.compute_mask(seq_inputs)
        mask = tf.cast(tf.not_equal(seq_inputs, 0), dtype=tf.float32)  # (None, maxlen)
        # user info
        user_embed = self.user_embedding(tf.squeeze(user_inputs, axis=-1))  # (None, dim)
        # seq info
        seq_embed = self.item_embedding(seq_inputs)  # (None, maxlen, dim)
        # item
        pos_embed = self.item_embedding(tf.squeeze(pos_inputs, axis=-1))  # (None, dim)
        neg_embed = self.item_embedding(tf.squeeze(neg_inputs, axis=-1))  # (None, dim)
        # item2 embed
        pos_embed2 = self.item2_embedding(tf.squeeze(pos_inputs, axis=-1))  # (None, dim)
        neg_embed2 = self.item2_embedding(tf.squeeze(neg_inputs, axis=-1))  # (None, dim)

        # short-term interest
        short_interest = self.self_attention([seq_embed, seq_embed, seq_embed, mask])  # (None, dim)

        # mode
        if self.mode == 'inner':
            # long-term interest, pos and neg
            pos_long_interest = tf.multiply(user_embed, pos_embed2)
            neg_long_interest = tf.multiply(user_embed, neg_embed2)
            # combine
            pos_scores = self.w * tf.reduce_sum(pos_long_interest, axis=-1, keepdims=True) \
                         + (1 - self.w) * tf.reduce_sum(tf.multiply(short_interest, pos_embed), axis=-1, keepdims=True)
            neg_scores = self.w * tf.reduce_sum(neg_long_interest, axis=-1, keepdims=True) \
                         + (1 - self.w) * tf.reduce_sum(tf.multiply(short_interest, neg_embed), axis=-1, keepdims=True)
            self.add_loss(tf.reduce_mean(-tf.math.log(tf.nn.sigmoid(pos_scores - neg_scores))))
        else:
            # clip by norm
            user_embed = tf.clip_by_norm(user_embed, 1, -1)
            pos_embed = tf.clip_by_norm(pos_embed, 1, -1)
            neg_embed = tf.clip_by_norm(neg_embed, 1, -1)
            pos_embed2 = tf.clip_by_norm(pos_embed2, 1, -1)
            neg_embed2 = tf.clip_by_norm(neg_embed2, 1, -1)
            # distance
            # long-term interest, pos and neg
            pos_long_interest = tf.square(user_embed - pos_embed2)  # (None, dim)
            neg_long_interest = tf.square(user_embed - neg_embed2)  # (None, dim)
            # combine. Here is a difference from the original paper.
            pos_scores = self.w * tf.reduce_sum(pos_long_interest, axis=-1, keepdims=True) + \
                         (1 - self.w) * tf.reduce_sum(tf.square(short_interest - pos_embed), axis=-1, keepdims=True)
            neg_scores = self.w * tf.reduce_sum(neg_long_interest, axis=-1, keepdims=True) + \
                         (1 - self.w) * tf.reduce_sum(tf.square(short_interest - neg_embed), axis=-1, keepdims=True)
            # minimize loss
            # self.add_loss(tf.reduce_sum(tf.maximum(pos_scores - neg_scores + self.gamma, 0)))
            self.add_loss(tf.reduce_sum(tf.nn.relu(pos_scores - neg_scores + self.gamma)))
        return pos_scores, neg_scores

    def summary(self):
        seq_inputs = Input(shape=(self.maxlen,), dtype=tf.int32)
        user_inputs = Input(shape=(1, ), dtype=tf.int32)
        pos_inputs = Input(shape=(1, ), dtype=tf.int32)
        neg_inputs = Input(shape=(1, ), dtype=tf.int32)
        Model(inputs=[user_inputs, seq_inputs, pos_inputs, neg_inputs], 
            outputs=self.call([user_inputs, seq_inputs, pos_inputs, neg_inputs])).summary()
```

```python id="FiecGHPuBjQT" executionInfo={"status": "ok", "timestamp": 1637055318637, "user_tz": -330, "elapsed": 621, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def test_model():
    user_features = {'feat': 'user_id', 'feat_num': 100, 'embed_dim': 8}
    seq_features = {'feat': 'item_id', 'feat_num': 100, 'embed_dim': 8}
    features = [user_features, seq_features]
    model = AttRec(features, mode='dist')
    model.summary()
```

```python id="tYj86oraBrKR" executionInfo={"status": "ok", "timestamp": 1637055319499, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def getHit(df):
    """
    calculate hit rate
    :return:
    """
    if sum(df['pred']) < _K:
        return 1
    else:
        return 0


def getNDCG(df):
    """
    calculate NDCG
    :return:
    """
    if sum(df['pred']) < _K:
        return 1 / np.log(sum(df['pred']) + 2)
    else:
        return 0.


def getMRR(df):
    """
    calculate MRR
    :return:
    """
    return 1 / (sum(df['pred']) + 1)


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
    test_X = test
    # predict
    pos_score, neg_score = model.predict(test_X)
    # create dataframe
    test_df = pd.DataFrame(test_X[0], columns=['user_id'])
    # if mode == 'inner', pos score < neg score, pred = 1
    if model.mode == 'inner':
        test_df['pred'] = (pos_score <= neg_score).astype(np.int32)
    else:
        test_df['pred'] = (pos_score >= neg_score).astype(np.int32)
    # groupby
    tg = test_df.groupby('user_id')
    # calculate hit
    hit_rate = tg.apply(getHit).mean()
    # calculate ndcg
    ndcg = tg.apply(getNDCG).mean()
    # calculate mrr
    mrr = tg.apply(getMRR).mean()
    return hit_rate, ndcg, mrr
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["2e75547bbc7c4fffa0295e9eff296fda", "4ce22228654446e3804adccba9b581e0", "42f32229d0284ba5b3c79220472f89f9", "89079249d29a4188ba2353eecdbcb0bb", "34820c17224e46cfb1f3c420229e250e", "37660610c37f4652980211bdb9196c2b", "88e6dab0adf34266ad96b78178852802", "68f7a61b4ce84099b0bec68c3602e70e", "38333b66d8f74363a0cf45b10a4ff3fe", "1415bc028c6a476aad215c062bcc5a72", "71565e08392045f9aee9457de810096e"]} id="iTbYFKSzCE1I" executionInfo={"status": "ok", "timestamp": 1637060785608, "user_tz": -330, "elapsed": 5466124, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fbc0c787-2b05-4351-e422-b3f2bbfc8432"
# ========================== Create dataset =======================
feature_columns, train, val, test = create_implicit_ml_1m_dataset(file, trans_score, embed_dim, maxlen)
train_X = train
val_X = val

# ============================Build Model==========================
model = AttRec(feature_columns, maxlen, mode, gamma, w, embed_reg)
model.summary()
# =========================Compile============================
model.compile(optimizer=Adam(learning_rate=learning_rate))

results = []
for epoch in range(1, epochs + 1):
    # ===========================Fit==============================
    t1 = time()
    model.fit(
        train_X,
        None,
        validation_data=(val_X, None),
        epochs=1,
        # callbacks=[tensorboard, checkpoint],
        batch_size=batch_size,
        )
    # ===========================Test==============================
    t2 = time()
    if epoch % 5 == 0:
        hit_rate, ndcg, mrr = evaluate_model(model, test, K)
        print('Iteration %d Fit [%.1f s], Evaluate [%.1f s]: HR = %.4f, NDCG = %.4f, MRR = %.4f'
                % (epoch, t2 - t1, time() - t2, hit_rate, ndcg, mrr))
        results.append([epoch, t2 - t1, time() - t2, hit_rate, ndcg, mrr])
# ========================== Write Log ===========================
pd.DataFrame(results, columns=['Iteration', 'fit_time', 'evaluate_time',
                                'hit_rate', 'ndcg', 'mrr']).to_csv(
    'AttRec_log_maxlen_{}_dim_{}_K_{}_w_{}.csv'.format(maxlen, embed_dim, K, w), index=False)
```
