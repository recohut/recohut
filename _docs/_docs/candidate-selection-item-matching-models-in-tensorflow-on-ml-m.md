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
# Candidate selection (Item matching) models in Tensorflow on ML-1m
<!-- #endregion -->

<!-- #region id="0brc2-VLDo6W" -->
## **Step 1 - Setup the environment**
<!-- #endregion -->

<!-- #region id="rS5pc1KVDo6Y" -->
### **1.1 Install libraries**
<!-- #endregion -->

```python id="ILFtDCDoDo6Z"
!pip install tensorflow==2.5.0
```

```python id="0pPmqoIRDo6d" outputId="484f33e2-f097-4300-d5c2-14e39f42e50c"
!pip install -q -U git+https://github.com/RecoHut-Projects/recohut.git -b v0.0.5
```

<!-- #region id="th2G0p2IDo6f" -->
### **1.2 Download datasets**
<!-- #endregion -->

```python id="b3gLXIDgDo6h"
!wget -q --show-progress https://files.grouplens.org/datasets/movielens/ml-1m.zip
!unzip ml-1m.zip
```

<!-- #region id="fIs1ygR3Do6i" -->
### **1.3 Import libraries**
<!-- #endregion -->

```python id="IACdit5EDo6k"
import os
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
```

```python id="WO51cxgTDo6m"
# transforms
from recohut.transforms.datasets.movielens import create_ml_1m_dataset
from recohut.transforms.datasets.movielens import create_implicit_ml_1m_dataset

# models
from recohut.models.tf.bpr import BPR
from recohut.models.tf.ncf import NCF
from recohut.models.tf.caser import Caser
from recohut.models.tf.sasrec import SASRec
from recohut.models.tf.attrec import AttRec
```

<!-- #region id="6nhAeMCWDo6n" -->
### **1.4 Set params**
<!-- #endregion -->

```python id="oeDvCJJ6EXYL"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```

```python id="EjLwQ-l_Do6n"
class Args:
    def __init__(self, model='bpr'):
        self.file = '/content/ml-1m/ratings.dat'
        self.epochs = 2
        self.trans_score = 1
        self.test_neg_num = 100
        self.embed_dim = 64
        self.mode = 'inner'  # dist
        self.embed_reg = 1e-6
        self.K = 10
        self.learning_rate = 0.001
        self.batch_size = 512
        self.hidden_units = [256, 128, 64]
        self.activation = 'relu'
        self.dropout = 0.2
        self.mode = 'inner'
        self.maxlen = 200
        self.hor_n = 8
        self.hor_h = 2
        self.ver_n = 4
        self.blocks = 2
        self.num_heads = 1
        self.ffn_hidden_unit = 64
        self.norm_training = True
        self.causality = False
        self.gamma = 0.5
        self.w = 0.5
        if model == 'ncf':
            self.embed_dim = 32
        elif model == 'caser':
            self.embed_dim = 50
        elif model == 'sasrec':
            self.embed_dim = 50
            self.embed_reg = 0
        elif model == 'attrec':
            self.maxlen = 5
            self.embed_dim = 100
            self.batch_size = 1024
```

<!-- #region id="pn39JAHGDo6t" -->
## **Step 2 - Training & Evaluation**
<!-- #endregion -->

```python id="kEriWJafDo6u"
def getHit(df, ver=1):
    """
    calculate hit rate
    :return:
    """
    if ver==1:
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


def evaluate_model(model, test, K, ver=1):
    """
    evaluate model
    :param model: model
    :param test: test set
    :param K: top K
    :return: hit rate, ndcg
    """
    if ver == 1:
        if args.mode == 'inner':
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

    elif ver == 2:
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

<!-- #region id="2BFZdOeHphps" -->
### **2.1 BPR**
<!-- #endregion -->

```python id="OHzU0MoWpk0e"
args = Args(model='bpr')
```

```python colab={"base_uri": "https://localhost:8080/"} id="IttULymDlTSx" executionInfo={"status": "ok", "timestamp": 1640000796316, "user_tz": -330, "elapsed": 70971, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4e4250bc-1d2b-45c6-9a3b-6da163a1bca1"
# ========================== Create dataset =======================
feature_columns, train, val, test = create_ml_1m_dataset(args.file, args.trans_score, args.embed_dim, args.test_neg_num)

# ============================Build Model==========================
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = BPR(feature_columns, args.mode, args.embed_reg)
    model.summary()
    # =========================Compile============================
    model.compile(optimizer=Adam(learning_rate=args.learning_rate))

results = []
for epoch in range(1, args.epochs + 1):
    # ===========================Fit==============================
    t1 = time()
    model.fit(
        train,
        None,
        validation_data=(val, None),
        epochs=1,
        batch_size=args.batch_size,
    )
    # ===========================Test==============================
    t2 = time()
    if epoch % 2 == 0:
        hit_rate, ndcg = evaluate_model(model, test, args.K)
        print('Iteration %d Fit [%.1f s], Evaluate [%.1f s]: HR = %.4f, NDCG = %.4f'
                % (epoch, t2 - t1, time() - t2, hit_rate, ndcg))
        results.append([epoch, t2 - t1, time() - t2, hit_rate, ndcg])

# ========================== Write Log ===========================
pd.DataFrame(results, columns=['Iteration', 'fit_time', 'evaluate_time', 'hit_rate', 'ndcg'])\
    .to_csv('BPR_log_dim_{}_mode_{}_K_{}.csv'.format(args.embed_dim, args.mode, args.K), index=False)
```

<!-- #region id="2hCjizhClWUB" -->
### **2.2 NCF**
<!-- #endregion -->

```python id="a6yJm03_qrjc"
args = Args(model='ncf')
```

```python colab={"base_uri": "https://localhost:8080/"} id="uZLxzb1vqm_u" executionInfo={"status": "ok", "timestamp": 1640000998168, "user_tz": -330, "elapsed": 116819, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c727653d-3839-4abe-b66a-03b09b1f4813"
# ========================== Create dataset =======================
feature_columns, train, val, test = create_ml_1m_dataset(args.file, args.trans_score, args.embed_dim, args.test_neg_num)

# ============================Build Model==========================
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = NCF(feature_columns, args.hidden_units, args.dropout, args.activation, args.embed_reg)
    model.summary()
    # =========================Compile============================
    model.compile(optimizer=Adam(learning_rate=args.learning_rate))

results = []
for epoch in range(1, args.epochs + 1):
    # ===========================Fit==============================
    t1 = time()
    model.fit(
        train,
        None,
        validation_data=(val, None),
        epochs=1,
        batch_size=args.batch_size,
    )
    # ===========================Test==============================
    t2 = time()
    if epoch % 2 == 0:
        hit_rate, ndcg = evaluate_model(model, test, args.K)
        print('Iteration %d Fit [%.1f s], Evaluate [%.1f s]: HR = %.4f, NDCG = %.4f'
                % (epoch, t2 - t1, time() - t2, hit_rate, ndcg))
        results.append([epoch, t2 - t1, time() - t2, hit_rate, ndcg])
# ========================== Write Log ===========================
pd.DataFrame(results, columns=['Iteration', 'fit_time', 'evaluate_time', 'hit_rate', 'ndcg'])\
    .to_csv('NCF_log_dim_{}__K_{}.csv'.format(args.embed_dim, args.K), index=False)
```

<!-- #region id="VGxxoo_rq3_u" -->
### **2.3 Caser**
<!-- #endregion -->

```python id="VZo2vk_otxl6"
args = Args(model='caser')
```

```python colab={"base_uri": "https://localhost:8080/"} id="T8aXXVZhtzo8" executionInfo={"status": "ok", "timestamp": 1640005168571, "user_tz": -330, "elapsed": 1165080, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6864130b-f921-4168-9e1f-0683f1f0d8c1"
# ========================== Create dataset =======================
feature_columns, train, val, test = create_implicit_ml_1m_dataset(args.file, args.trans_score, args.embed_dim, args.maxlen)
train_X, train_y = train
val_X, val_y = val

# ============================Build Model==========================
model = Caser(feature_columns, args.maxlen, args.hor_n, args.hor_h, args.ver_n, args.dropout, args.activation, args.embed_reg)
model.summary()
# =========================Compile============================
model.compile(loss=BinaryCrossentropy(), optimizer=Adam(learning_rate=args.learning_rate))

results = []
for epoch in range(1, args.epochs + 1):
    # ===========================Fit==============================
    t1 = time()
    model.fit(
        train_X,
        train_y,
        validation_data=(val_X, val_y),
        epochs=1,
        batch_size=args.batch_size,
    )
    # ===========================Test==============================
    t2 = time()
    if epoch % 2 == 0:
        hit_rate, ndcg = evaluate_model(model, test, args.K, ver=2)
        print('Iteration %d Fit [%.1f s], Evaluate [%.1f s]: HR = %.4f, NDCG= %.4f'
                % (epoch, t2 - t1, time() - t2, hit_rate, ndcg))
        results.append([epoch + 1, t2 - t1, time() - t2, hit_rate, ndcg])

# ============================Write============================
pd.DataFrame(results, columns=['Iteration', 'fit_time', 'evaluate_time', 'hit_rate', 'ndcg']).\
    to_csv('Caser_log_maxlen_{}_dim_{}_hor_n_{}_ver_n_{}_K_{}_.csv'.
            format(args.maxlen, args.embed_dim, args.hor_n, args.ver_n, args.K), index=False)
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
