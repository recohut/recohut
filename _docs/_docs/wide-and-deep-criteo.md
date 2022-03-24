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

<!-- #region id="BU7SonF1LSiN" -->
# Wide and Deep Model to Predict Ad-click Probability
> Training a Tensorflow based Wide and Deep model on criteo sample dataset to predict the probability of ad-click

- toc: true
- badges: true
- comments: true
- categories: [CTR, Tensorflow, WideAndDeep]
- author: "<a href='https://github.com/liangxiaotian/Recommender-Systems'>liangxiaotian</a>"
- image:
<!-- #endregion -->

<!-- #region id="cD5fspXYL4tG" -->
## Setup
<!-- #endregion -->

```python id="2FdncIZeIX2v"
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Layer, Dropout
from tensorflow.keras.experimental import LinearModel, WideDeepModel
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
```

```python id="0Ab8oZRiI_GG"
read_part = True
sample_num = 5000000
test_size = 0.2
embed_dim = 8
filename = "criteo_sample.txt"
epochs = 100
```

<!-- #region id="ew0o6Y7xIhg2" -->
## Dataset
<!-- #endregion -->

<!-- #region id="XS4p5eO-In-f" -->
dataset：criteo dataset sample

created on July 13, 2020

features：
- Label - Target variable that indicates if an ad was clicked (1) or not (0).
- I1-I13 - A total of 13 columns of integer features (mostly count features).
- C1-C26 - A total of 26 columns of categorical features.
The values of these features have been hashed onto 32 bits for anonymization purposes.

@author: Ziyao Geng(zggzy1996@163.com)
<!-- #endregion -->

```python id="bw60AdksHx4h"
!wget https://github.com/sparsh-ai/reco-data/raw/master/criteo_sample.txt
```

```python id="NJloFqZXIZbb"
def create_criteo_dataset(filename, embed_dim=8, read_part=True, sample_num=100000, test_size=0.2):
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
        data_df = pd.read_csv(filename, sep='\t', iterator=True, header=None,
                          names=names)
        data_df = data_df.get_chunk(sample_num)

    else:
        data_df = pd.read_csv(filename, sep='\t', header=None, names=names)

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

    # feature_columns = [sparseFeature(feat, int(data_df[feat].max()) + 1, embed_dim=embed_dim)
    #                     for feat in features] # 
    train, test = train_test_split(data_df, test_size=test_size)

    train_X = train[features]# .values.astype('int32')
    train_y = train['label']#.values.astype('int32')
    test_X = test[features]#.values.astype('int32')
    test_y = test['label']#.values.astype('int32')

    return sparse_features, dense_features, (train_X, train_y), (test_X, test_y)
```

```python id="3aXB7_0BI89l"
sparse_features, dense_features, train, test = create_criteo_dataset(filename=filename,
                                                                     embed_dim=embed_dim,
                                                                     read_part=read_part,
                                                                     sample_num=sample_num,
                                                                     test_size=test_size)
train_X, train_y = train
test_X, test_y = test
```

```python colab={"base_uri": "https://localhost:8080/", "height": 224} id="uOTudPleJmhE" outputId="301f78ab-39c0-4e73-94bd-a16ad512c6a2"
train_X.head()
```

```python id="RNyC87nCJm--"
linear_inputs = train_X[sparse_features].values.astype('int32')
dnn_inputs = train_X[dense_features].values.astype('int32')
y = train_y.values.astype('int32')
```

```python id="s76dajzpKclB"
test_linear_inputs = test_X[sparse_features].values.astype('int32')
test_dnn_inputs = test_X[dense_features].values.astype('int32')
test_y = test_y.values.astype('int32')
```

```python colab={"base_uri": "https://localhost:8080/"} id="zKxOHatoKd2l" outputId="09619459-4fab-4ccc-8fb3-7cb9b401f0d6"
linear_inputs.shape, dnn_inputs.shape, y.shape
```

<!-- #region id="g4IOicUwKhtM" -->
## Model
<!-- #endregion -->

<!-- #region id="4bkDtmcPKr6v" -->
<!-- #endregion -->

```python id="5PJ1XFx-KeOS"
# wide model
linear_model = LinearModel()
```

```python id="6RSUJtHqKuPA"
# deep model
dnn_model = tf.keras.Sequential()
dnn_model.add(tf.keras.layers.Embedding(input_dim=100, output_dim=8, input_length=len(dense_features)))
dnn_model.add(tf.keras.layers.Flatten()) # or concat embedding layers
```

```python id="2uEXtUc-Kxto"
dnn_model.add(keras.layers.Dense(units = 256))
dnn_model.add(keras.layers.Dense(units = 128))
dnn_model.add(keras.layers.Dense(units = 64))
dnn_model.add(keras.layers.Dense(units = 1, activation = None))
```

```python id="fskWfJBuKzfp"
combined_model = WideDeepModel(linear_model, dnn_model, activation = 'sigmoid')
```

```python id="ZEkxlUhfK128"
combined_model.compile(optimizer=[Adam(learning_rate=0.0001), Adam(learning_rate=0.0001)], 
                       loss= binary_crossentropy, metrics=[AUC()])
```

<!-- #region id="bfyTcZlYL-gf" -->
## Training and Validation
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="x3npKwjlK7tx" outputId="df26e341-d853-410c-ae91-76013657acfa"
history = combined_model.fit([linear_inputs, dnn_inputs], y, epochs = epochs, batch_size = 4096, validation_split = 0.1)
```

```python colab={"base_uri": "https://localhost:8080/"} id="k9M2MrvuLEpL" outputId="7e40cc6f-ba68-4dfb-a964-7d7d7a8e628c"
combined_model.summary()
```

```python colab={"base_uri": "https://localhost:8080/"} id="JHk2FnEQLF8E" outputId="bdcd90cc-1e8d-4e0b-b28e-caad28b8aa88"
print('test mse: %f' % combined_model.evaluate([test_linear_inputs, test_dnn_inputs], test_y, batch_size=32)[1])
```

<!-- #region id="xTekPIauMBbd" -->
## Accuracy and Loss Plot
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 281} id="g9H2YcOYLHp6" outputId="7aed8b49-092b-4fb9-817d-0f4ad5e94bfa"
acc = history.history['auc']
val_acc = history.history['val_auc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
```
