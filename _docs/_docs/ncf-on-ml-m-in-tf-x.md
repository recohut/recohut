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

```python id="Zb6n1vfEJq7H"
!pip install tensorflow==2.5.0
```

```python colab={"base_uri": "https://localhost:8080/"} id="thzbCqzfKV75" executionInfo={"status": "ok", "timestamp": 1637056355012, "user_tz": -330, "elapsed": 3517, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="21d2e932-2b22-456b-cbf5-8f70b2f2befc"
!wget -q --show-progress https://files.grouplens.org/datasets/movielens/ml-1m.zip
!unzip ml-1m.zip
```

```python id="oRYCBuX9KV4O" executionInfo={"status": "ok", "timestamp": 1637056667294, "user_tz": -330, "elapsed": 3014, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
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
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer, Dense, Dropout, Embedding, Input
```

```python id="bqhuTZxZLVlH" executionInfo={"status": "ok", "timestamp": 1637056667298, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

file = 'ml-1m/ratings.dat'
trans_score = 1
test_neg_num = 100

embed_dim = 32
hidden_units = [256, 128, 64]
embed_reg = 1e-6  # 1e-6
activation = 'relu'
dropout = 0.2
K = 10

learning_rate = 0.001
epochs = 20
batch_size = 512
```

```python id="XoIea7QCKVzb" executionInfo={"status": "ok", "timestamp": 1637056671922, "user_tz": -330, "elapsed": 20, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
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

```python id="dlHOlqgKKk6S" executionInfo={"status": "ok", "timestamp": 1637056671926, "user_tz": -330, "elapsed": 19, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
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
    item_feat_col = [sparseFeature('user_id', user_num, embed_dim),
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
    return item_feat_col, train, val, test
```

```python id="Y-neX9Y0KuWv" executionInfo={"status": "ok", "timestamp": 1637056671929, "user_tz": -330, "elapsed": 19, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class DNN(Layer):
	"""
	Deep part
	"""
	def __init__(self, hidden_units, activation='relu', dnn_dropout=0., **kwargs):
		"""
		DNN part
		:param hidden_units: A list. List of hidden layer units's numbers
		:param activation: A string. Activation function
		:param dnn_dropout: A scalar. dropout number
		"""
		super(DNN, self).__init__(**kwargs)
		self.dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
		self.dropout = Dropout(dnn_dropout)

	def call(self, inputs, **kwargs):
		x = inputs
		for dnn in self.dnn_network:
			x = dnn(x)
		x = self.dropout(x)
		return x
```

```python id="HK9Sq1d8KuTg" executionInfo={"status": "ok", "timestamp": 1637056673980, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class NCF(Model):
    def __init__(self, feature_columns, hidden_units=None, dropout=0.2, activation='relu', embed_reg=1e-6, **kwargs):
        """
        NCF model
        :param feature_columns: A list. user feature columns + item feature columns
        :param hidden_units: A list.
        :param dropout: A scalar.
        :param activation: A string.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(NCF, self).__init__(**kwargs)
        if hidden_units is None:
            hidden_units = [64, 32, 16, 8]
        # feature columns
        self.user_fea_col, self.item_fea_col = feature_columns
        # MF user embedding
        self.mf_user_embedding = Embedding(input_dim=self.user_fea_col['feat_num'],
                                           input_length=1,
                                           output_dim=self.user_fea_col['embed_dim'],
                                           embeddings_initializer='random_normal',
                                           embeddings_regularizer=l2(embed_reg))
        # MF item embedding
        self.mf_item_embedding = Embedding(input_dim=self.item_fea_col['feat_num'],
                                           input_length=1,
                                           output_dim=self.item_fea_col['embed_dim'],
                                           embeddings_initializer='random_normal',
                                           embeddings_regularizer=l2(embed_reg))
        # MLP user embedding
        self.mlp_user_embedding = Embedding(input_dim=self.user_fea_col['feat_num'],
                                            input_length=1,
                                            output_dim=self.user_fea_col['embed_dim'],
                                            embeddings_initializer='random_normal',
                                            embeddings_regularizer=l2(embed_reg))
        # MLP item embedding
        self.mlp_item_embedding = Embedding(input_dim=self.item_fea_col['feat_num'],
                                            input_length=1,
                                            output_dim=self.item_fea_col['embed_dim'],
                                            embeddings_initializer='random_normal',
                                            embeddings_regularizer=l2(embed_reg))
        # dnn
        self.dnn = DNN(hidden_units, activation=activation, dnn_dropout=dropout)
        self.dense = Dense(1, activation=None)

    def call(self, inputs):
        user_inputs, pos_inputs, neg_inputs = inputs  # (None, 1), (None, 1), (None, 1/101)
        # user info
        mf_user_embed = self.mf_user_embedding(user_inputs)  # (None, 1, dim)
        mlp_user_embed = self.mlp_user_embedding(user_inputs)  # (None, 1, dim)
        # item
        mf_pos_embed = self.mf_item_embedding(pos_inputs)  # (None, 1, dim)
        mf_neg_embed = self.mf_item_embedding(neg_inputs)  # (None, 1/101, dim)
        mlp_pos_embed = self.mlp_item_embedding(pos_inputs)  # (None, 1, dim)
        mlp_neg_embed = self.mlp_item_embedding(neg_inputs)  # (None, 1/101, dim)
        # MF
        mf_pos_vector = tf.nn.sigmoid(tf.multiply(mf_user_embed, mf_pos_embed))  # (None, 1, dim)
        mf_neg_vector = tf.nn.sigmoid(tf.multiply(mf_user_embed, mf_neg_embed))  # (None, 1, dim)
        # MLP
        mlp_pos_vector = tf.concat([mlp_user_embed, mlp_pos_embed], axis=-1)  # (None, 1, 2 * dim)
        mlp_neg_vector = tf.concat([tf.tile(mlp_user_embed, multiples=[1, mlp_neg_embed.shape[1], 1]),
                                    mlp_neg_embed], axis=-1)  # (None, 1/101, 2 * dim)
        mlp_pos_vector = self.dnn(mlp_pos_vector)  # (None, 1, dim)
        mlp_neg_vector = self.dnn(mlp_neg_vector)  # (None, 1/101, dim)
        # concat
        pos_vector = tf.concat([mf_pos_vector, mlp_pos_vector], axis=-1)  # (None, 1, 2 * dim)
        neg_vector = tf.concat([mf_neg_vector, mlp_neg_vector], axis=-1)  # (None, 1/101, 2 * dim)
        # pos_vector = mlp_pos_vector
        # neg_vector = mlp_neg_vector
        # result
        pos_logits = tf.squeeze(self.dense(pos_vector), axis=-1)  # (None, 1)
        neg_logits = tf.squeeze(self.dense(neg_vector), axis=-1)  # (None, 1/101)
        # loss
        losses = tf.reduce_mean(- tf.math.log(tf.nn.sigmoid(pos_logits)) -
                                tf.math.log(1 - tf.nn.sigmoid(neg_logits))) / 2
        self.add_loss(losses)
        logits = tf.concat([pos_logits, neg_logits], axis=-1)
        return logits

    def summary(self):
        user_inputs = Input(shape=(1,), dtype=tf.int32)
        pos_inputs = Input(shape=(1,), dtype=tf.int32)
        neg_inputs = Input(shape=(1,), dtype=tf.int32)
        Model(inputs=[user_inputs, pos_inputs, neg_inputs],
              outputs=self.call([user_inputs, pos_inputs, neg_inputs])).summary()
```

```python id="oqnt9DihKuQG" executionInfo={"status": "ok", "timestamp": 1637056673983, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def test_model():
    user_features = {'feat': 'user_id', 'feat_num': 100, 'embed_dim': 8}
    item_features = {'feat': 'item_id', 'feat_num': 100, 'embed_dim': 8}
    features = [user_features, item_features]
    model = NCF(features)
    model.summary()
```

```python id="Qdc9Il1pKuMl" executionInfo={"status": "ok", "timestamp": 1637056673985, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def evaluate_model(model, test, K):
    """
    evaluate model
    :param model: model
    :param test: test set
    :param K: top K
    :return: hit rate, ndcg
    """
    pred_y = - model.predict(test)
    rank = pred_y.argsort().argsort()[:, 0]
    hr, ndcg = 0.0, 0.0
    for r in rank:
        if r < K:
            hr += 1
            ndcg += 1 / np.log2(r + 2)
    return hr / len(rank), ndcg / len(rank)
```

```python colab={"base_uri": "https://localhost:8080/"} id="gNgKr7GoLK2W" executionInfo={"status": "ok", "timestamp": 1637057570102, "user_tz": -330, "elapsed": 851948, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c96b1407-5bbf-4a10-8f1e-99e6c3291967"
# ========================== Create dataset =======================
feature_columns, train, val, test = create_ml_1m_dataset(file, trans_score, embed_dim, test_neg_num)

# ============================Build Model==========================
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = NCF(feature_columns, hidden_units, dropout, activation, embed_reg)
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
    if epoch % 2 == 0:
        hit_rate, ndcg = evaluate_model(model, test, K)
        print('Iteration %d Fit [%.1f s], Evaluate [%.1f s]: HR = %.4f, NDCG = %.4f'
                % (epoch, t2 - t1, time() - t2, hit_rate, ndcg))
        results.append([epoch, t2 - t1, time() - t2, hit_rate, ndcg])
# ========================== Write Log ===========================
pd.DataFrame(results, columns=['Iteration', 'fit_time', 'evaluate_time', 'hit_rate', 'ndcg'])\
    .to_csv('NCF_log_dim_{}__K_{}.csv'.format(embed_dim, K), index=False)
```
