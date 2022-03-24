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

<!-- #region id="SRcy_kIPDo6U" -->
# CTR (Item ranking) models in Tensorflow on ML-1m
<!-- #endregion -->

<!-- #region id="0brc2-VLDo6W" -->
## **Step 1 - Setup the environment**
<!-- #endregion -->

<!-- #region id="rS5pc1KVDo6Y" -->
### **1.1 Install libraries**
<!-- #endregion -->

```python id="iKc5Z6OOGqVD"
!pip install tensorflow==2.5.0
```

```python id="zh3IrkWHGywI"
!pip install -q -U git+https://github.com/RecoHut-Projects/recohut.git -b v0.0.5
```

<!-- #region id="th2G0p2IDo6f" -->
### **1.2 Download datasets**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="lxCe3QHbGrkm" executionInfo={"status": "ok", "timestamp": 1640008563294, "user_tz": -330, "elapsed": 363332, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a5c1b915-5c10-4a3d-bbce-8687be6dca08"
!pip install -q -U kaggle
!pip install --upgrade --force-reinstall --no-deps kaggle
!mkdir ~/.kaggle
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d mrkmakr/criteo-dataset
!unzip criteo-dataset.zip
```

<!-- #region id="fIs1ygR3Do6i" -->
### **1.3 Import libraries**
<!-- #endregion -->

```python id="VjtUOFBEG-Tp" executionInfo={"status": "ok", "timestamp": 1640009732310, "user_tz": -330, "elapsed": 4181, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
```

```python id="wj-hW3UXHZLv" executionInfo={"status": "ok", "timestamp": 1640014506365, "user_tz": -330, "elapsed": 607, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
from recohut.transforms.datasets.criteo import create_criteo_dataset

from recohut.models.tf.fm import FM, FFM, NFM, AFM, DeepFM, xDeepFM
from recohut.models.tf.widedeep import WideDeep
from recohut.models.tf.deepcross import DeepCross
from recohut.models.tf.pnn import PNN
from recohut.models.tf.dcn import DCN
```

<!-- #region id="6nhAeMCWDo6n" -->
### **1.4 Set params**
<!-- #endregion -->

```python id="OJBMn-x6HLah" executionInfo={"status": "ok", "timestamp": 1640010334317, "user_tz": -330, "elapsed": 720, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```

```python id="dWWoB0bnHFLD" executionInfo={"status": "ok", "timestamp": 1640014905006, "user_tz": -330, "elapsed": 704, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class Args:
    def __init__(self, model='fm'):  
        self.file = 'dac/train.txt'
        self.read_part = True
        self.sample_num = 10000
        self.test_size = 0.2
        self.k = self.embed_dim = 8
        self.learning_rate = 0.001
        self.batch_size = 4096
        self.dnn_dropout = 0.5
        self.dropout = 0.5
        self.epochs = 2
        self.att_vector = 8
        self.mode = 'att'  # 'max', 'avg'
        self.activation = 'relu'
        self.embed_reg = 1e-5
        self.hidden_units = [256, 128, 64]
        self.cin_size = [128, 128]
        if model=='ffm':
            self.k = 10
            self.batch_size = 1024
```

<!-- #region id="pn39JAHGDo6t" -->
## **Step 2 - Training & Evaluation**
<!-- #endregion -->

<!-- #region id="Y1VgAkXTHitA" -->
### **2.1 FM**
<!-- #endregion -->

```python id="zVQBZwHKHoD9" executionInfo={"status": "ok", "timestamp": 1640010337793, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
args = Args(model='fm')
```

```python colab={"base_uri": "https://localhost:8080/"} id="CpnvJ3_UHkAi" executionInfo={"status": "ok", "timestamp": 1640010353121, "user_tz": -330, "elapsed": 13440, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="bd530a40-0984-44bc-b18c-7a2b189105e1"
# ========================== Create dataset =======================
feature_columns, train, test = create_criteo_dataset(file=args.file,
                                        read_part=args.read_part,
                                        sample_num=args.sample_num,
                                        test_size=args.test_size)
train_X, train_y = train
test_X, test_y = test
# ============================Build Model==========================
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = FM(feature_columns=feature_columns, k=args.k)
    model.summary()
    # ============================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=args.learning_rate),
                    metrics=[AUC()])
# ============================model checkpoint======================
# check_path = '../save/fm_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
# checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
#                                                 verbose=1, period=5)
# ==============================Fit==============================
model.fit(
    train_X,
    train_y,
    epochs=args.epochs,
    callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],  # checkpoint
    batch_size=args.batch_size,
    validation_split=0.1
)
# ===========================Test==============================
print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=args.batch_size)[1])
```

<!-- #region id="XoJhQzq6O9Gl" -->
### **2.2 FFM**
<!-- #endregion -->

```python id="0093ztfHa3Rh" executionInfo={"status": "ok", "timestamp": 1640013578579, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
args = Args(model='ffm')
```

```python colab={"base_uri": "https://localhost:8080/"} id="b8Ohm6ZzbT1C" executionInfo={"status": "ok", "timestamp": 1640013791161, "user_tz": -330, "elapsed": 156848, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="39f642f5-c3ff-4a85-f2a6-cee9a487e5fc"
# ========================== Create dataset =======================
feature_columns, train, test = create_criteo_dataset(file=args.file,
                                        read_part=args.read_part,
                                        sample_num=args.sample_num,
                                        test_size=args.test_size)
train_X, train_y = train
test_X, test_y = test
# ============================Build Model==========================
model = FFM(feature_columns=feature_columns, k=args.k)
model.summary()
# ============================model checkpoint======================
# check_path = '../save/fm_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
# checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
#                                                 verbose=1, period=5)
# ============================Compile============================
model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=args.learning_rate),
                metrics=[AUC()])
# ==============================Fit==============================
model.fit(
    train_X,
    train_y,
    epochs=args.epochs,
    callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],  # checkpoint
    batch_size=args.batch_size,
    validation_split=0.1
)
# ===========================Test==============================
print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=args.batch_size)[1])
```

<!-- #region id="ovN5AJw6bhgk" -->
### **2.3 Wide & Deep**
<!-- #endregion -->

```python id="JW_Ws-vlbrXt" executionInfo={"status": "ok", "timestamp": 1640013799473, "user_tz": -330, "elapsed": 610, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
args = Args(model='widedeep')
```

```python colab={"base_uri": "https://localhost:8080/"} id="xy-c1JRIcJmY" executionInfo={"status": "ok", "timestamp": 1640013884019, "user_tz": -330, "elapsed": 10308, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6bcfddbd-ee86-42d2-8a22-136b1f85d3ad"
# ========================== Create dataset =======================
feature_columns, train, test = create_criteo_dataset(file=args.file,
                                                        embed_dim=args.embed_dim,
                                                        read_part=args.read_part,
                                                        sample_num=args.sample_num,
                                                        test_size=args.test_size)
train_X, train_y = train
test_X, test_y = test
# ============================Build Model==========================
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = WideDeep(feature_columns, hidden_units=args.hidden_units, dnn_dropout=args.dnn_dropout)
    model.summary()
    # ============================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=args.learning_rate),
                    metrics=[AUC()])
# ============================model checkpoint======================
# check_path = '../save/wide_deep_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
# checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
#                                                 verbose=1, period=5)
# ==============================Fit==============================
model.fit(
    train_X,
    train_y,
    epochs=args.epochs,
    callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],  # checkpoint
    batch_size=args.batch_size,
    validation_split=0.1
)
# ===========================Test==============================
print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=args.batch_size)[1])
```

<!-- #region id="OgBpmMgJcb80" -->
### **2.4 Deep Crossing**
<!-- #endregion -->

```python id="z7cNC2C2ckS1" executionInfo={"status": "ok", "timestamp": 1640013945511, "user_tz": -330, "elapsed": 607, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
args = Args(model='deepcross')
```

```python colab={"base_uri": "https://localhost:8080/"} id="z7Zk6wNhcs-A" executionInfo={"status": "ok", "timestamp": 1640014056055, "user_tz": -330, "elapsed": 11232, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ed5f5085-cae7-4e84-bdaa-46c9e65897c2"
# ========================== Create dataset =======================
feature_columns, train, test = create_criteo_dataset(file=args.file,
                                                        embed_dim=args.embed_dim,
                                                        read_part=args.read_part,
                                                        sample_num=args.sample_num,
                                                        test_size=args.test_size)
train_X, train_y = train
test_X, test_y = test
# ============================Build Model==========================
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = DeepCross(feature_columns, args.hidden_units)
    model.summary()
    # =========================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=args.learning_rate),
                    metrics=[AUC()])
# ============================model checkpoint======================
# check_path = 'save/deep_crossing_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
# checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
#                                                 verbose=1, period=5)
# ===========================Fit==============================
model.fit(
    train_X,
    train_y,
    epochs=args.epochs,
    callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],  # checkpoint
    batch_size=args.batch_size,
    validation_split=0.1
)
# ===========================Test==============================
print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=args.batch_size)[1])
```

<!-- #region id="YcPlkxq0dFo3" -->
### **2.5 PNN**
<!-- #endregion -->

```python id="kNNxWtrWdP2b" executionInfo={"status": "ok", "timestamp": 1640014096024, "user_tz": -330, "elapsed": 793, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
args = Args(model='pnn')
```

```python colab={"base_uri": "https://localhost:8080/"} id="_pSQuGUsdSBo" executionInfo={"status": "ok", "timestamp": 1640014283370, "user_tz": -330, "elapsed": 53689, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2dfad7b5-4ff7-42ee-dea4-7521e132f4b7"
# ========================== Create dataset =======================
feature_columns, train, test = create_criteo_dataset(file=args.file,
                                                        embed_dim=args.embed_dim,
                                                        read_part=args.read_part,
                                                        sample_num=args.sample_num,
                                                        test_size=args.test_size)
train_X, train_y = train
test_X, test_y = test
# ============================Build Model==========================
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = PNN(feature_columns, args.hidden_units, args.dnn_dropout)
    model.summary()
    # =========================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=args.learning_rate),
                    metrics=[AUC()])
# ============================model checkpoint======================
# check_path = 'save/pnn_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
# checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
#                                                 verbose=1, period=5)
# ===========================Fit==============================
model.fit(
    train_X,
    train_y,
    epochs=args.epochs,
    callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],  # checkpoint
    batch_size=args.batch_size,
    validation_split=0.1
)
# ===========================Test==============================
print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=args.batch_size)[1])
```

<!-- #region id="5QGMmWeZdy20" -->
### **2.6 DCN**
<!-- #endregion -->

```python id="sWHzDu7Qd7Eo" executionInfo={"status": "ok", "timestamp": 1640014285372, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
args = Args(model='dcn')
```

```python colab={"base_uri": "https://localhost:8080/"} id="udqJU5Ahd-jq" executionInfo={"status": "ok", "timestamp": 1640014358793, "user_tz": -330, "elapsed": 10320, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ef38e55b-1a5e-400f-8935-425586e3cecb"
# ========================== Create dataset =======================
feature_columns, train, test = create_criteo_dataset(file=args.file,
                                                        embed_dim=args.embed_dim,
                                                        read_part=args.read_part,
                                                        sample_num=args.sample_num,
                                                        test_size=args.test_size)
train_X, train_y = train
test_X, test_y = test
# ============================Build Model==========================
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = DCN(feature_columns, args.hidden_units, dnn_dropout=args.dnn_dropout)
    model.summary()
    # =========================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=args.learning_rate),
                    metrics=[AUC()])
# ============================model checkpoint======================
# check_path = 'save/dcn_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
# checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
#                                                 verbose=1, period=5)
# ===========================Fit==============================
model.fit(
    train_X,
    train_y,
    epochs=args.epochs,
    callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],  # checkpoint
    batch_size=args.batch_size,
    validation_split=0.1
)
# ===========================Test==============================
print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=args.batch_size)[1])
```

<!-- #region id="0sIomQLKeP0E" -->
### **2.7 NFM**
<!-- #endregion -->

```python id="r8mXoX2BejRx" executionInfo={"status": "ok", "timestamp": 1640014513790, "user_tz": -330, "elapsed": 538, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
args = Args(model='nfm')
```

```python colab={"base_uri": "https://localhost:8080/"} id="DFAQtguRemF8" executionInfo={"status": "ok", "timestamp": 1640014528582, "user_tz": -330, "elapsed": 14045, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3cc29459-0a17-47fa-b188-bb2fb0b490ec"
# ========================== Create dataset =======================
feature_columns, train, test = create_criteo_dataset(file=args.file,
                                                        embed_dim=args.embed_dim,
                                                        read_part=args.read_part,
                                                        sample_num=args.sample_num,
                                                        test_size=args.test_size)
train_X, train_y = train
test_X, test_y = test
# ============================Build Model==========================
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = NFM(feature_columns, args.hidden_units, dnn_dropout=args.dnn_dropout)
    model.summary()
    # =========================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=args.learning_rate),
                    metrics=[AUC()])
# ============================model checkpoint======================
# check_path = 'save/nfm_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
# checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
#                                                 verbose=1, period=5)
# ===========================Fit==============================
model.fit(
    train_X,
    train_y,
    epochs=args.epochs,
    callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],  # checkpoint
    batch_size=args.batch_size,
    validation_split=0.1
)
# ===========================Test==============================
print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=args.batch_size)[1])
```

<!-- #region id="EfR2ObfXeZWX" -->
### **2.8 AFM**
<!-- #endregion -->

```python id="BIEsGwIxe6LA" executionInfo={"status": "ok", "timestamp": 1640014721216, "user_tz": -330, "elapsed": 448, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
args = Args(model='afm')
```

```python colab={"base_uri": "https://localhost:8080/"} id="oDrJt6mce78a" executionInfo={"status": "ok", "timestamp": 1640014736598, "user_tz": -330, "elapsed": 14600, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fb7478f2-5526-4068-fc14-58a8c91d704d"
# ========================== Create dataset =======================
feature_columns, train, test = create_criteo_dataset(file=args.file,
                                                        embed_dim=args.embed_dim,
                                                        read_part=args.read_part,
                                                        sample_num=args.sample_num,
                                                        test_size=args.test_size)
train_X, train_y = train
test_X, test_y = test
# ============================Build Model==========================
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = AFM(feature_columns, args.mode, args.att_vector, args.activation, args.dropout, args.embed_reg)
    model.summary()
    # =========================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=args.learning_rate),
                    metrics=[AUC()])
# ============================model checkpoint======================
# check_path = 'save/afm_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
# checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
#                                                 verbose=1, period=5)
# ===========================Fit==============================
model.fit(
    train_X,
    train_y,
    epochs=args.epochs,
    callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)], # checkpoint,
    batch_size=args.batch_size,
    validation_split=0.1
)
# ===========================Test==============================
print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=args.batch_size)[1])
```

<!-- #region id="8fu7UVaZecN2" -->
### **2.9 DeepFM**
<!-- #endregion -->

```python id="nAKsDMERf1jX"
args = Args(model='deepfm')
```

```python colab={"base_uri": "https://localhost:8080/"} id="so06_YSrf4I_" executionInfo={"status": "ok", "timestamp": 1640014858211, "user_tz": -330, "elapsed": 12392, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0bcf7f70-1ae8-4971-b5a5-0d838423398a"
# ========================== Create dataset =======================
feature_columns, train, test = create_criteo_dataset(file=args.file,
                                                        embed_dim=args.embed_dim,
                                                        read_part=args.read_part,
                                                        sample_num=args.sample_num,
                                                        test_size=args.test_size)
train_X, train_y = train
test_X, test_y = test
# ============================Build Model==========================
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = DeepFM(feature_columns, hidden_units=args.hidden_units, dnn_dropout=args.dnn_dropout)
    model.summary()
    # ============================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=args.learning_rate),
                    metrics=[AUC()])
# ============================model checkpoint======================
# check_path = '../save/deepfm_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
# checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
#                                                 verbose=1, period=5)
# ==============================Fit==============================
model.fit(
    train_X,
    train_y,
    epochs=args.epochs,
    callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],  # checkpoint,
    batch_size=args.batch_size,
    validation_split=0.1
)
# ===========================Test==============================
print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=args.batch_size)[1])
```

<!-- #region id="5FW58X-UecQd" -->
### **2.10 xDeepFM**
<!-- #endregion -->

```python id="ldF-rahyefh1" executionInfo={"status": "ok", "timestamp": 1640014915603, "user_tz": -330, "elapsed": 529, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
args = Args(model='xdeepfm')
```

```python colab={"base_uri": "https://localhost:8080/"} id="ZHFgD3rlgaJ6" executionInfo={"status": "ok", "timestamp": 1640015004067, "user_tz": -330, "elapsed": 44685, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b9269666-743e-4c34-dfd5-9f75d5f85cb6"
# ========================== Create dataset =======================
feature_columns, train, test = create_criteo_dataset(file=args.file,
                                                        embed_dim=args.embed_dim,
                                                        read_part=args.read_part,
                                                        sample_num=args.sample_num,
                                                        test_size=args.test_size)
train_X, train_y = train
test_X, test_y = test
# ============================Build Model==========================
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = xDeepFM(feature_columns, args.hidden_units, args.cin_size)
    model.summary()
    # =========================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=args.learning_rate),
                    metrics=[AUC()])
# ============================model checkpoint======================
# check_path = 'save/xdeepfm_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
# checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
#                                                 verbose=1, period=5)
# ===========================Fit==============================
model.fit(
    train_X,
    train_y,
    epochs=args.epochs,
    callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],  # checkpoint
    batch_size=args.batch_size,
    validation_split=0.1
)
# ===========================Test==============================
print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=args.batch_size)[1])
```

<!-- #region id="NCFKoIs5Do6z" -->
## **Closure**
<!-- #endregion -->

<!-- #region id="sSKqb6t5Do60" -->
For more details, you can refer to https://github.com/RecoHut-Stanzas/S021355.
<!-- #endregion -->

<!-- #region id="0R4uWucpDo61" -->
<a href="https://github.com/RecoHut-Stanzas/S021355/blob/main/reports/S021355.ipynb" alt="S021355_Report"> <img src="https://img.shields.io/static/v1?label=report&message=active&color=green" /></a> <a href="https://github.com/RecoHut-Stanzas/S021355" alt="S021355"> <img src="https://img.shields.io/static/v1?label=code&message=github&color=blue" /></a>
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="zuVA3NzxglAX" executionInfo={"status": "ok", "timestamp": 1640015143767, "user_tz": -330, "elapsed": 3846, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="427dd15e-62bb-4c39-f000-b94c68110ce2"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="JOsqyvAkDo62" -->
---
<!-- #endregion -->

<!-- #region id="LLugelemDo63" -->
**END**
<!-- #endregion -->
