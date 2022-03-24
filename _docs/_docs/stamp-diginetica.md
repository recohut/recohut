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

<!-- #region id="oMU14qbn6tWx" -->
# STAMP on Diginetica in TF 2.x
<!-- #endregion -->

```python id="N-C8WVmWyt1c"
!pip install tensorflow==2.5.0
!pip install tensorflow-gpu==2.5.0
```

```python id="R3QiS0AHzPNr"
!git clone https://github.com/RecoHut-Datasets/diginetica.git
```

```python id="tOgRfWKuzGx3"
import os
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from time import time

import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import Loss
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, \
    Dropout, Embedding, Flatten, Input
```

```python id="5csalgjaBO6a"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

file = 'diginetica/train-item-views.csv'
maxlen = 8

embed_dim = 20
K = 20

learning_rate = 0.005
batch_size = 1024
epochs = 10
```

```python id="qTaga32Y39Kl"
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

```python id="0tA7dlQVA727"
def denseFeature(feat):
    """
    create dictionary for dense feature
    :param feat: dense feature name
    :return:
    """
    return {'feat': feat}
```

```python id="8xyT05c-A6lF"
def convert_sequence(data_df):
    """
    :param data_df: train, val or test
    """
    data_sequence = []
    for sessionId, df in tqdm(data_df[['sessionId', 'itemId']].groupby(['sessionId'])):
        item_list = df['itemId'].tolist()

        for i in range(1, len(item_list)):
            hist_i = item_list[:i]
            # hist_item, next_click_item(label)
            data_sequence.append([hist_i, item_list[i]])

    return data_sequence
```

```python id="ZQD91xJPA5TV"
def create_diginetica_dataset(file, embed_dim=8, maxlen=40):
    """
    :param file: A string. dataset path
    :param embed_dim: A scalar. latent factor
    :param maxlen: A scalar. 
    :return: feature_columns, behavior_list, train, val, test
    """
    print('==========Data Preprocess Start============')
    # load dataset
    data_df = pd.read_csv(file, sep=";") # (1235380, 5)
    
    # filter out sessions of length of 1
    data_df['session_count'] = data_df.groupby('sessionId')['sessionId'].transform('count')
    data_df = data_df[data_df.session_count > 1]  # (1144686, 6)

    # filter out items that appear less than 5 times
    data_df['item_count'] = data_df.groupby('itemId')['itemId'].transform('count')
    data_df = data_df[data_df.item_count >= 5]  # (1004834, 7)

    # label encoder itemId, {0, 1, ..., }
    le = LabelEncoder()
    data_df['itemId'] = le.fit_transform(data_df['itemId'])
    
     # sorted by eventdate, sessionId
    data_df = data_df.sort_values(by=['eventdate', 'sessionId'])

    # split dataset, 1 day for valdation, 7 days for test
    train = data_df[data_df.eventdate < '2016-05-25']  # (916485, 7)
    val = data_df[data_df.eventdate == '2016-05-25']  # (10400, 7)
    test = data_df[data_df.eventdate > '2016-05-25']  # (77949, 7)

    # convert sequence
    train = pd.DataFrame(convert_sequence(train), columns=['hist', 'label'])
    val = pd.DataFrame(convert_sequence(val), columns=['hist', 'label'])
    test = pd.DataFrame(convert_sequence(test), columns=['hist', 'label'])
    
    # Padding
    # not have dense inputs and other sparse inputs
    print('==================Padding===================')
    train_X = [np.array([0.] * len(train)), np.array([0] * len(train)),
               np.expand_dims(pad_sequences(train['hist'], maxlen=maxlen), axis=1)]
    train_y = train['label'].values
    val_X = [np.array([0] * len(val)), np.array([0] * len(val)),
               np.expand_dims(pad_sequences(val['hist'], maxlen=maxlen), axis=1)]
    val_y = val['label'].values
    test_X = [np.array([0] * len(test)), np.array([0] * len(test)),
               np.expand_dims(pad_sequences(test['hist'], maxlen=maxlen), axis=1)]
    test_y = test['label'].values

    # item pooling
    item_pooling = np.sort(data_df['itemId'].unique().reshape(-1, 1), axis=0)

    # feature columns, dense feature columns + sparse feature columns
    item_num = data_df['itemId'].max() + 1
    feature_columns = [[],
                       [sparseFeature('item_id', item_num, embed_dim)]]

    # behavior list
    behavior_list = ['item_id']

    print('===========Data Preprocess End=============')
    
    return feature_columns, behavior_list, item_pooling, (train_X, train_y), (val_X, val_y), (test_X, test_y)
```

```python id="tcmDAMAn_6My"
class Attention_Layer(Layer):
    """
    Attention Layer
    """
    def __init__(self, d, reg=1e-4):
        """
        :param d: A scalar. The dimension of embedding.
        :param reg: A scalar. The regularizer of parameters
        """
        self.d = d
        self.reg = reg
        super(Attention_Layer, self).__init__()

    def build(self, input_shape):
        self.W0 = self.add_weight(name='W0',
                                  shape=(self.d, 1),
                                  initializer=tf.random_normal_initializer,
                                  regularizer=l2(self.reg),
                                  trainable=True)
        self.W1 = self.add_weight(name='W1',
                                  shape=(self.d, self.d),
                                  initializer=tf.random_normal_initializer,
                                  regularizer=l2(self.reg),
                                  trainable=True)
        self.W2 = self.add_weight(name='W2',
                                  shape=(self.d, self.d),
                                  initializer=tf.random_normal_initializer,
                                  regularizer=l2(self.reg),
                                  trainable=True)
        self.W3 = self.add_weight(name='W3',
                                  shape=(self.d, self.d),
                                  initializer=tf.random_normal_initializer,
                                  regularizer=l2(self.reg),
                                  trainable=True)
        self.b = self.add_weight(name='b',
                                  shape=(self.d,),
                                  initializer=tf.random_normal_initializer,
                                  regularizer=l2(self.reg),
                                  trainable=True)

    def call(self, inputs):
        seq_embed, m_s, x_t = inputs
        """
        seq_embed: (None, seq_len, d)
        W1: (d, d)
        x_t: (None, d)
        W2: (d, d)
        m_s: (None, d)
        W3: (d, d)
        W0: (d, 1)
        """
        alpha = tf.matmul(tf.nn.sigmoid(
            tf.tensordot(seq_embed, self.W1, axes=[2, 0]) + tf.expand_dims(tf.matmul(x_t, self.W2), axis=1) +
            tf.expand_dims(tf.matmul(m_s, self.W3), axis=1) + self.b), self.W0)
        m_a = tf.reduce_sum(tf.multiply(alpha, seq_embed), axis=1)  # (None, d)
        return m_a
```

```python id="CRnDzYLGByvm"
class STAMP(tf.keras.Model):
    def __init__(self, feature_columns, behavior_feature_list, item_pooling, maxlen=40, activation='tanh', embed_reg=1e-4):
        """
        STAMP
        :param feature_columns: A list. dense_feature_columns + sparse_feature_columns
        :param behavior_feature_list: A list. the list of behavior feature names
        :param item_pooling: A Ndarray or Tensor, shape=(m, n),
        m is the number of items, and n is the number of behavior feature. The item pooling.
        :param activation: A String. The activation of FFN.
        :param maxlen: A scalar. Maximum sequence length.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(STAMP, self).__init__()
        # maximum sequence length
        self.maxlen = maxlen

        # item pooling
        self.item_pooling = item_pooling
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns

        # len
        self.other_sparse_len = len(self.sparse_feature_columns) - len(behavior_feature_list)
        self.dense_len = len(self.dense_feature_columns)
        # if behavior feature list contains itemId and item category id, seq_len = 2
        self.seq_len = len(behavior_feature_list)

        # embedding dim, each sparse feature embedding dimension is the same
        self.embed_dim = self.sparse_feature_columns[0]['embed_dim']

        # other embedding layers
        self.embed_sparse_layers = [Embedding(input_dim=feat['feat_num'],
                                              input_length=1,
                                              output_dim=feat['embed_dim'],
                                              embeddings_initializer='random_uniform',
                                              embeddings_regularizer=l2(embed_reg))
                                    for feat in self.sparse_feature_columns
                                    if feat['feat'] not in behavior_feature_list]
        # behavior embedding layers
        self.embed_seq_layers = [Embedding(input_dim=feat['feat_num'],
                                           input_length=1,
                                           output_dim=feat['embed_dim'],
                                           embeddings_initializer='random_uniform',
                                           embeddings_regularizer=l2(embed_reg))
                                 for feat in self.sparse_feature_columns
                                 if feat['feat'] in behavior_feature_list]

        # Attention
        self.attention_layer = Attention_Layer(d=self.embed_dim)

        # FNN, hidden unit must be equal to embedding dimension
        self.ffn1 = Dense(self.embed_dim, activation=activation)
        self.ffn2 = Dense(self.embed_dim, activation=activation)

    def call(self, inputs):
        # dense_inputs and sparse_inputs is empty
        dense_inputs, sparse_inputs, seq_inputs = inputs
        
        x = dense_inputs
        # other
        for i in range(self.other_sparse_len):
            x = tf.concat([x, self.embed_sparse_layers[i](sparse_inputs[:, i])], axis=-1)

        # seq
        seq_embed, m_t, item_pooling_embed = None, None, None
        for i in range(self.seq_len):
            # item sequence embedding
            seq_embed = self.embed_seq_layers[i](seq_inputs[:, i]) if seq_embed is None \
                else seq_embed + self.embed_seq_layers[i](seq_inputs[:, i])
            # last click item embedding
            m_t = self.embed_seq_layers[i](seq_inputs[:, i, -1]) if m_t is None \
                else m_t + self.embed_seq_layers[i](seq_inputs[-1, i, -1])  # (None, d)
            # item pooling embedding 
            item_pooling_embed = self.embed_seq_layers[i](self.item_pooling[:, i]) \
                if item_pooling_embed is None \
                else item_pooling_embed + self.embed_seq_layers[i](self.item_pooling[:, i])  # (m, d)

        # calculate m_s        
        m_s = tf.reduce_mean(seq_embed, axis=1)  # (None, d)

        # attention
        m_a = self.attention_layer([seq_embed, m_s, m_t])  # (None, d)
        # if model is STMP, m_a = m_s
        # m_a = m_s

        # try to add other embedding vector
        if self.other_sparse_len != 0 or self.dense_len != 0:
            m_a = tf.concat([m_a, x], axis=-1)
            m_t = tf.concat([m_t, x], axis=-1)

        # FFN
        h_s = self.ffn1(m_a)  # (None, d)
        h_t = self.ffn2(m_t)  # (None, d)

        # Calculate
        # h_t * item_pooling_embed, (None, 1, d) * (m, d) = (None, m, d)
        # () mat h_s, (None, m, d) matmul (None, d, 1) = (None, m, 1)
        z = tf.matmul(tf.multiply(tf.expand_dims(h_t, axis=1), item_pooling_embed), tf.expand_dims(h_s, axis=-1))
        z = tf.squeeze(z, axis=-1)  # (None, m)

        # Outputs
        outputs = tf.nn.softmax(z)
        return outputs

    def summary(self):
        dense_inputs = Input(shape=(self.dense_len,), dtype=tf.float32)
        sparse_inputs = Input(shape=(self.other_sparse_len,), dtype=tf.int32)
        seq_inputs = Input(shape=(self.seq_len, self.maxlen), dtype=tf.int32)
        tf.keras.Model(inputs=[dense_inputs, sparse_inputs, seq_inputs],
                       outputs=self.call([dense_inputs, sparse_inputs, seq_inputs])).summary()
```

```python id="UiXY7XvoB0LQ"
def test_model():
    dense_features = []  # [{'feat': 'a'}, {'feat': 'b'}]
    sparse_features = [{'feat': 'item_id', 'feat_num': 100, 'embed_dim': 8},
                       {'feat': 'cate_id', 'feat_num': 100, 'embed_dim': 8},
                       {'feat': 'adv_id', 'feat_num': 100, 'embed_dim': 8}]
    behavior_list = ['item_id', 'cate_id']
    item_pooling = tf.constant([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    features = [dense_features, sparse_features]
    model = STAMP(features, behavior_list, item_pooling)
    model.summary()
```

```python id="bPFJ28sdBF9V"
def getHit(pred_y, true_y):
    """
    calculate hit rate
    :return:
    """
    # reversed
    pred_index = np.argsort(-pred_y)[:, :_K]
    return sum([true_y[i] in pred_index[i] for i in range(len(pred_index))]) / len(pred_index)


def getMRR(pred_y, true_y):
    """
    """
    pred_index = np.argsort(-pred_y)[:, :_K]
    return sum([1 / (np.where(true_y[i] == pred_index[i])[0][0] + 1) \
        for i in range(len(pred_index)) if len(np.where(true_y[i] == pred_index[i])[0]) != 0]) / len(pred_index)


def evaluate_model(model, test, K):
    """
    evaluate model
    :param model: model
    :param test: test set
    :param K: top K
    :return: hit rate, mrr
    """
    global _K
    _K = K
    test_X, test_y = test
    pred_y = model.predict(test_X)
    hit_rate = getHit(pred_y, test_y)
    mrr = getMRR(pred_y, test_y)
    
    
    return hit_rate, mrr
```

```python id="cVb72bBnBZWh"
# ========================== Create dataset =======================
feature_columns, behavior_list, item_pooling, train, val, test = create_diginetica_dataset(file, embed_dim, maxlen)
train_X, train_y = train
val_X, val_y = val
# ============================Build Model==========================
model = STAMP(feature_columns, behavior_list, item_pooling, maxlen)
model.summary()
# ============================model checkpoint======================
# check_path = 'save/sas_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
# checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
#                                                 verbose=1, period=5)
# =========================Compile============================
# CrossEntropy()
# tf.losses.SparseCategoricalCrossentropy()
model.compile(loss=tf.losses.SparseCategoricalCrossentropy(), optimizer=Adam(learning_rate=learning_rate))

for epoch in range(epochs):
    # ===========================Fit==============================
    t1 = time()
    model.fit(
        train_X,
        train_y,
        validation_data=(val_X, val_y),
        epochs=1,
        # callbacks=[tensorboard, checkpoint],
        batch_size=batch_size,
        )
    # ===========================Test==============================
    t2 = time()
    hit_rate, mrr = evaluate_model(model, test, K)
    print('Iteration %d Fit [%.1f s], Evaluate [%.1f s]: HR = %.4f, MRR = %.4f, '
            % (epoch, t2 - t1, time() - t2, hit_rate, mrr))
```
