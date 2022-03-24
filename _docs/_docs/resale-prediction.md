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

<!-- #region id="hTZC_MeXXLMF" -->
# Resale Price Prediction
<!-- #endregion -->

```python id="A9ltXWmZVa8T"
# import the libraries
import re
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix, hstack
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from keras import backend as K
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Input, Dropout
from keras.models import Model

from utils import *

import warnings
warnings.filterwarnings("ignore")

plt.style.use('fivethirtyeight')
plt.style.use('seaborn-notebook')

%config InlineBackend.figure_format = 'retina'
%reload_ext autoreload
%autoreload 2
```

```python id="2p_NBJkunGRw"
df = pd.read_pickle('./data/df_cleaned.p')
```

```python id="wef35u8BtjoM"
colname_map = {'PRC':'BRAND', 'PARTNO':'PARTNO','UNIT RESALE':'UNITRESALE',
               'ORIG ORDER QTY':'ORDERQTY', 'NEW UNIT COST':'UNITCOST'}
df = prepare_data(df, colname_map)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="oypErAtr0SVX" executionInfo={"status": "ok", "timestamp": 1606979285202, "user_tz": -330, "elapsed": 970, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6d2d2332-b644-4761-b3ce-21ab656868d6"
df.head()
```

```python id="VaNm416UPaj0"
df, fitted_lambda = scale_price(df)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="gVAoBBhTP69k" executionInfo={"status": "ok", "timestamp": 1606976275872, "user_tz": -330, "elapsed": 2353, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6addba67-7a0c-4b0a-c87b-ecc6ad889f73"
df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="UBKM4NEk0WNi" executionInfo={"status": "ok", "timestamp": 1606976289343, "user_tz": -330, "elapsed": 5766, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5db6abc1-0350-460d-a73a-87ae0fbb7c02"
CV1 = CountVectorizer(stop_words=None, 
                      max_df=1.0, 
                      min_df=100, 
                      ngram_range=(1,1),
                      binary=True, 
                      analyzer='char')

CV1.fit(list(set(df['PARTNO'].tolist())))
X1 = CV1.transform(df['PARTNO'].tolist())
X1
```

```python id="_aF67DCu8R0o"
# CV1.vocabulary_
```

```python colab={"base_uri": "https://localhost:8080/"} id="y54nrS_O9Edd" executionInfo={"status": "ok", "timestamp": 1606976303764, "user_tz": -330, "elapsed": 19169, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a2677a87-52c9-494a-fc68-f552b17a0110"
CV2 = CountVectorizer(stop_words=None, 
                      max_df=0.8, 
                      min_df=100, 
                      ngram_range=(2,6), 
                      binary=True,
                      analyzer='char')
CV2.fit(list(set(df['PARTNO'].tolist())))
X2 = CV2.transform(df['PARTNO'].tolist())
X2
```

```python id="09IH88Vq-Xj1"
def tokenizer(text):
  text = text.lower()
  rx1 = r"(?i)(?:(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d))"
  text = re.sub(rx1,' ', text)
  text = re.sub(r'[^a-z0-9]',' ', text)
  text = ' '.join(text.split())
  text = text.split()
  return text
```

```python colab={"base_uri": "https://localhost:8080/"} id="MVY_I573CzXt" executionInfo={"status": "ok", "timestamp": 1606976312708, "user_tz": -330, "elapsed": 26230, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4fc13226-a19f-4f62-9e48-7c1afd03fb5c"
CV3 = TfidfVectorizer(stop_words=None, 
                      max_df=0.5, 
                      min_df=100, 
                      ngram_range=(1,5), 
                      binary=False,
                      analyzer='word',
                      tokenizer=tokenizer)
CV3.fit(list(set(df['PARTNO'].tolist())))
X3 = CV3.transform(df['PARTNO'].tolist())
X3
```

```python colab={"base_uri": "https://localhost:8080/"} id="FqwrqmN_C2de" executionInfo={"status": "ok", "timestamp": 1606976312710, "user_tz": -330, "elapsed": 25629, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="dbedad8f-dce3-480b-9cfd-013b43430fa3"
enc = OneHotEncoder()
ohecols = ['BRAND','QUANTITY']
enc.fit(df[ohecols])
X4 = enc.transform(df[ohecols])
X4
```

```python colab={"base_uri": "https://localhost:8080/"} id="wbSW1c3XENQm" executionInfo={"status": "ok", "timestamp": 1606976313792, "user_tz": -330, "elapsed": 25556, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="037e27a4-cb1d-4e67-8611-01e3c151e8db"
X = hstack([X1, X2, X3, X4])
X
```

```python id="Qva5scOXEN9P"
Y = df['RESALE'].values
Y = Y.reshape(-1,1)
```

```python colab={"base_uri": "https://localhost:8080/"} id="F1c-YWSgK1Yc" executionInfo={"status": "ok", "timestamp": 1606976677537, "user_tz": -330, "elapsed": 2106, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bab87383-7546-425d-879e-e24d01690f60"
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=40)
print("Training Records {}, Testing Records: {}".format(X_train.shape[0],
                                                        X_test.shape[0]))
```

```python colab={"base_uri": "https://localhost:8080/"} id="2nT15mqgMJeJ" executionInfo={"status": "ok", "timestamp": 1606976680729, "user_tz": -330, "elapsed": 2040, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1bf23e40-93cf-4b28-9bb2-f73623859f12"
batch_size = 2048
epochs = 75

inputs = Input(shape=(X_train.shape[1],), sparse=True)
L = Dense(512, activation='relu')(inputs)
L = Dropout(0.5)(L)
L = Dense(10, activation='relu')(L)
outputs = Dense(y_train.shape[1])(L)
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.summary()
```

```python colab={"base_uri": "https://localhost:8080/"} id="qaWtUDzeM_hG" executionInfo={"status": "ok", "timestamp": 1606978751520, "user_tz": -330, "elapsed": 2071911, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f3013f06-2979-46c2-b6bd-c2c819993888"
history = model.fit(nn_batch_generator(X_train, y_train, batch_size),
          steps_per_epoch=len(y_train)//batch_size, 
          validation_data=nn_batch_generator(X_test, y_test, batch_size),
          validation_steps=len(y_test)//batch_size, 
          epochs=100,
          workers=-1, 
          use_multiprocessing=True)
```

```python id="Q8E8k7q0OlXb"
model.save('./models/model_201203.h5')
```

```python id="LBqD7XYjWWEb"
hist_df = pd.DataFrame(history.history) 
hist_csv_file = './outputs/history.csv'
with open(hist_csv_file, mode='w') as f:
  hist_df.to_csv(f)
```

```python id="sr5xfoNKNa3O" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1606978910113, "user_tz": -330, "elapsed": 2418, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fff88fa9-4d82-4cef-f7d4-9d24963e4e52"
from scipy.special import inv_boxcox
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error

y_pred = model.predict(X_test).flatten()
a = inv_boxcox(y_test.flatten(), fitted_lambda)
b = inv_boxcox(y_pred.flatten(), fitted_lambda)
print('r2_score: ', r2_score(a, b))
print('median_absolute_error: ', median_absolute_error(a, b))
print('mean_absolute_error', mean_absolute_error(a, b))
out2 = pd.DataFrame({'y_true':inv_boxcox(y_test.flatten(), fitted_lambda), 'y_pred':inv_boxcox(y_pred.flatten(), fitted_lambda)})
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="ueew4O4GabBJ" executionInfo={"status": "ok", "timestamp": 1606979350479, "user_tz": -330, "elapsed": 2796, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b426a3a9-e89e-45e5-d92e-2865afeedabc"
out2.head()
```

```python id="j6Dv4_PhR93T"
_, out1 = train_test_split(df, test_size=0.1, random_state=40)
out1['RESALE'] = out2.y_true.values
out1['PRED'] = out2.y_pred.values
out1.to_csv('./outputs/result.csv', index=False)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 359} id="3zxuMd7ZZ6jM" executionInfo={"status": "ok", "timestamp": 1606979617741, "user_tz": -330, "elapsed": 2078, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ed72cd65-e1bc-4457-aad1-d704446412aa"
out1.sample(10)
```
