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

```python id="0pxSU24FbSAy"
!pip install tensorflow==2.5.0
```

```python colab={"base_uri": "https://localhost:8080/"} id="QnFg_wXiclo4" executionInfo={"status": "ok", "timestamp": 1637061269255, "user_tz": -330, "elapsed": 135691, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6cc6fdd1-38da-459c-ea28-576e6c5492b1"
!pip install -q -U kaggle
!pip install --upgrade --force-reinstall --no-deps kaggle
!mkdir ~/.kaggle
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d mrkmakr/criteo-dataset
```

```python colab={"base_uri": "https://localhost:8080/"} id="PZYBD38Ad5j2" executionInfo={"status": "ok", "timestamp": 1637061804413, "user_tz": -330, "elapsed": 325474, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="efff747a-58f1-49d5-bb09-51875616aaa0"
!unzip criteo-dataset.zip
```

```python id="BZwknC3Gd8Qg"
import os
import itertools
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Input, ReLU
from tensorflow.keras.layers import Dense, Embedding, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
```

```python id="GDRfYvu4e4mO"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

file = 'dac/train.txt'
read_part = True
sample_num = 100000
test_size = 0.2

embed_dim = 8
att_vector = 8
mode = 'att'  # 'max', 'avg'
dropout = 0.5
activation = 'relu'
embed_reg = 1e-5

learning_rate = 0.001
batch_size = 4096
epochs = 10
```

```python id="9zzb1WXIet8A"
def sparseFeature(feat, feat_num, embed_dim=4):
    """
    create dictionary for sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return:
    """
    return {'feat_name': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def denseFeature(feat):
    """
    create dictionary for dense feature
    :param feat: dense feature name
    :return:
    """
    return {'feat_name': feat}
```

```python id="5NnOfRIQerQh"
def create_criteo_dataset(file, embed_dim=8, read_part=True, sample_num=100000, test_size=0.2):
    """
    a example about creating criteo dataset
    :param file: dataset's path
    :param embed_dim: the embedding dimension of sparse features
    :param read_part: whether to read part of it
    :param sample_num: the number of instances if read_part is True
    :param test_size: ratio of test dataset
    :return: feature columns, train, test
    """
    names = ['label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11',
             'I12', 'I13', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
             'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22',
             'C23', 'C24', 'C25', 'C26']

    if read_part:
        data_df = pd.read_csv(file, sep='\t', iterator=True, header=None,
                          names=names)
        data_df = data_df.get_chunk(sample_num)

    else:
        data_df = pd.read_csv(file, sep='\t', header=None, names=names)

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    features = sparse_features + dense_features

    data_df[sparse_features] = data_df[sparse_features].fillna('-1')
    data_df[dense_features] = data_df[dense_features].fillna(0)

    # Bin continuous data into intervals.
    est = KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='uniform')
    data_df[dense_features] = est.fit_transform(data_df[dense_features])

    for feat in sparse_features:
        le = LabelEncoder()
        data_df[feat] = le.fit_transform(data_df[feat])

    # ==============Feature Engineering===================

    # ====================================================
    feature_columns = [sparseFeature(feat, int(data_df[feat].max()) + 1, embed_dim=embed_dim)
                        for feat in features]
    train, test = train_test_split(data_df, test_size=test_size)

    train_X = train[features].values.astype('int32')
    train_y = train['label'].values.astype('int32')
    test_X = test[features].values.astype('int32')
    test_y = test['label'].values.astype('int32')

    return feature_columns, (train_X, train_y), (test_X, test_y)
```

```python id="K09g18Cmgxrf"
class AFM(Model):
    def __init__(self, feature_columns, mode, att_vector=8, activation='relu', dropout=0.5, embed_reg=1e-6):
        """
        AFM 
        :param feature_columns: A list. sparse column feature information.
        :param mode: A string. 'max'(MAX Pooling) or 'avg'(Average Pooling) or 'att'(Attention)
        :param att_vector: A scalar. attention vector.
        :param activation: A string. Activation function of attention.
        :param dropout: A scalar. Dropout.
        :param embed_reg: A scalar. the regularizer of embedding
        """
        super(AFM, self).__init__()
        self.sparse_feature_columns = feature_columns
        self.mode = mode
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        if self.mode == 'att':
            self.attention_W = Dense(units=att_vector, activation=activation, use_bias=True)
            self.attention_dense = Dense(units=1, activation=None)
        self.dropout = Dropout(dropout)
        self.dense = Dense(units=1, activation=None)

    def call(self, inputs):
        # Input Layer
        sparse_inputs = inputs
        # Embedding Layer 
        embed = [self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]
        embed = tf.transpose(tf.convert_to_tensor(embed), perm=[1, 0, 2])  # (None, len(sparse_inputs), embed_dim)
        # Pair-wise Interaction Layer
        row = []
        col = []
        for r, c in itertools.combinations(range(len(self.sparse_feature_columns)), 2):
            row.append(r)
            col.append(c)
        p = tf.gather(embed, row, axis=1)  # (None, (len(sparse) * len(sparse) - 1) / 2, k)
        q = tf.gather(embed, col, axis=1)  # (None, (len(sparse) * len(sparse) - 1) / 2, k)
        bi_interaction = p * q  # (None, (len(sparse) * len(sparse) - 1) / 2, k)
        # mode
        if self.mode == 'max':
            # MaxPooling Layer
            x = tf.reduce_sum(bi_interaction, axis=1)   # (None, k)
        elif self.mode == 'avg':
            # AvgPooling Layer
            x = tf.reduce_mean(bi_interaction, axis=1)  # (None, k)
        else:
            # Attention Layer
            x = self.attention(bi_interaction)  # (None, k)
        # Output Layer
        outputs = tf.nn.sigmoid(self.dense(x))

        return outputs

    def summary(self):
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()

    def attention(self, bi_interaction):
        a = self.attention_W(bi_interaction)  # (None, (len(sparse) * len(sparse) - 1) / 2, t)
        a = self.attention_dense(a)  # (None, (len(sparse) * len(sparse) - 1) / 2, 1)
        a_score = tf.nn.softmax(a, axis=1)  # (None, (len(sparse) * len(sparse) - 1) / 2, 1)
        outputs = tf.reduce_sum(bi_interaction * a_score, axis=1)  # (None, embed_dim)
        return outputs
```

```python colab={"base_uri": "https://localhost:8080/"} id="dHIfDXsXePmP" executionInfo={"status": "ok", "timestamp": 1637065942187, "user_tz": -330, "elapsed": 173579, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1ee2589a-7fb0-4bf8-a493-55abb07df76a"
# ========================== Create dataset =======================
feature_columns, train, test = create_criteo_dataset(file=file,
                                                        embed_dim=embed_dim,
                                                        read_part=read_part,
                                                        sample_num=sample_num,
                                                        test_size=test_size)
train_X, train_y = train
test_X, test_y = test
# ============================Build Model==========================
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = AFM(feature_columns, mode, att_vector, activation, dropout, embed_reg)
    model.summary()
    # =========================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                    metrics=[AUC()])
# ============================model checkpoint======================
# check_path = 'save/afm_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
# checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
#                                                 verbose=1, period=5)
# ===========================Fit==============================
model.fit(
    train_X,
    train_y,
    epochs=epochs,
    callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)], # checkpoint,
    batch_size=batch_size,
    validation_split=0.1
)
# ===========================Test==============================
print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=batch_size)[1])
```

```python id="UKnzXHU2k0Bt"

```
