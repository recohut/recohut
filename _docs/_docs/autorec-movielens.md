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

<!-- #region id="VzqbTJF5jKnJ" -->
# Learning latent representations with Autoencoders
> Training AutoRec and DeepAutoRec based simple and hybrid models on movielens-1m

- toc: true
- badges: true
- comments: true
- categories: [Tensorflow, Hybrid, Movie, AutoEncoder]
- image:
<!-- #endregion -->

<!-- #region id="Z1BPPcAGjKnP" -->
| |  |
| :-: | -:|
| Vision | Learn better embeddings using deep learning techniques |
| Mission | Use Autorec and its deep version to train simple and hybrid models  |
| Scope | Movie dataset, Model training and validation, Autorec and DeepAutoRec models |
| Task | Rating prediction |
| Data | Movielens-1m |
| Tool | Tensorflow 1.x, Colab, Python |
| Technique | AutoEncoder, DeepAutoEncoder, Collaborative Filtering, Hybrid Modeling |
| Process | 1) Data ETL, 2) Data Preprocessing, 3) AutoRec model training, 4) DeepAutoRec model training, 5) Denoising experiment, 6) Hybrid modeling experiment |
| Takeaway | Autoencoders are effective to learn embeddings, Hybrid models might not improve model performance in case of rating imbalance |
| Credit | [Zheda (Marco) Mai](https://github.com/RaptorMai) |
| Link | [link1](https://github.com/RaptorMai/Deep-AutoEncoder-Recommendation), [link2](http://cseweb.ucsd.edu/~gary/pubs/mehrab-cikm-2020.pdf) |
<!-- #endregion -->

<!-- #region id="CKzNbzxBYiZX" -->
Autoencoder has been widely adopted into Collaborative Filtering (CF) for recommendation system. A classic CF problem is inferring the missing rating in an MxN matrix R where R(i, j) is the ratings given by the i<sup>th</sup> user to the j<sup>th</sup> item. This project is a Keras implementation of  AutoRec and Deep AutoRec and additional experiments will be run. 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Jb5fOfM6G7kE" outputId="572d9771-ec6b-4116-f95d-7e7d2e97f73c"
%tensorflow_version 1.x
```

```python id="DbcISBgWqNIQ"
import os
import math
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras.utils import plot_model
from scipy.sparse import csr_matrix

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.layers import add, concatenate
from tensorflow.python.keras.layers import Input, Dense, Embedding, Flatten
from tensorflow.python.keras.layers import  Dropout, merge, Activation, BatchNormalization, LeakyReLU

%matplotlib inline
warnings.filterwarnings('ignore')
```

<!-- #region id="GLaeJ3XgFQhQ" -->
## ETL
<!-- #endregion -->

<!-- #region id="CDqSJ_lcZUcd" -->
The raw data file is separated by ```::``` without headers. This part is transforming the raw data file into a CSV with headers, which can be easily imported using Pandas in the following parts. All the user and movie id will be subtracted by 1 for zero-based index. The snippet shows the preprocessing for rating data and similar preprocessing is applied to users data and movies data.
<!-- #endregion -->

<!-- #region id="JRFlquJ2FS4k" -->
### Extract
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="d-jQKdetFC51" outputId="acd788a7-5ac7-4e3b-82b8-b66dff3e8682"
!wget http://files.grouplens.org/datasets/movielens/ml-1m.zip
```

```python colab={"base_uri": "https://localhost:8080/"} id="aFX_vXI2Fb-5" outputId="009eabcb-2272-4d0d-8e74-7e4f32042ca5"
!unzip ml-1m.zip
```

<!-- #region id="IEkmBPaiFjhZ" -->
### Transform
<!-- #endregion -->

```python id="6Hs_zsMgFgjf"
# variables
BASE_DIR = '/content' 
MOVIELENS_DIR = BASE_DIR + '/ml-1m/'
USER_DATA_FILE = 'users.dat'
MOVIE_DATA_FILE = 'movies.dat'
RATING_DATA_FILE = 'ratings.dat'

# ref http://files.grouplens.org/datasets/movielens/ml-1m-README.txt
AGES = { 1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+" }
OCCUPATIONS = { 0: "other or not specified", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
                4: "college/grad student", 5: "customer service", 6: "doctor/health care",
                7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student", 11: "lawyer",
                12: "programmer", 13: "retired", 14: "sales/marketing", 15: "scientist", 16: "self-employed",
                17: "technician/engineer", 18: "tradesman/craftsman", 19: "unemployed", 20: "writer" }
                
RATINGS_CSV_FILE = 'ml1m_ratings.csv'
USERS_CSV_FILE = 'ml1m_users.csv'
MOVIES_CSV_FILE = 'ml1m_movies.csv'
```

```python colab={"base_uri": "https://localhost:8080/"} id="b7LPorTnF5Y0" outputId="92e1dab2-50fd-4c17-a68f-8112b1c5323e"
ratings = pd.read_csv(os.path.join(MOVIELENS_DIR, RATING_DATA_FILE), 
                    sep='::', 
                    engine='python', 
                    encoding='latin-1',
                    names=['userid', 'movieid', 'rating', 'timestamp'])

max_userid = ratings['userid'].drop_duplicates().max()
max_movieid = ratings['movieid'].drop_duplicates().max()

ratings['user_emb_id'] = ratings['userid'] - 1
ratings['movie_emb_id'] = ratings['movieid'] - 1

print(len(ratings), 'ratings loaded')

ratings.to_csv(RATINGS_CSV_FILE, 
               sep='\t', 
               header=True, 
               encoding='latin-1', 
               columns=['user_emb_id', 'movie_emb_id', 'rating', 'timestamp'])

print('Saved to', RATINGS_CSV_FILE)
```

```python colab={"base_uri": "https://localhost:8080/"} id="2qcJJ9UJGC4v" outputId="8f3537e9-fb38-40dc-c8fe-db4be4e9e385"
users = pd.read_csv(os.path.join(MOVIELENS_DIR, USER_DATA_FILE), 
                    sep='::', 
                    engine='python', 
                    encoding='latin-1',
                    names=['userid', 'gender', 'age', 'occupation', 'zipcode'])

users['age_desc'] = users['age'].apply(lambda x: AGES[x])
users['occ_desc'] = users['occupation'].apply(lambda x: OCCUPATIONS[x])

print(len(users), 'descriptions of', max_userid, 'users loaded.')

users['user_emb_id'] = users['userid'] - 1

users.to_csv(USERS_CSV_FILE, 
             sep='\t', 
             header=True, 
             encoding='latin-1',
             columns=['user_emb_id', 'gender', 'age', 'occupation', 'zipcode', 'age_desc', 'occ_desc'])

print('Saved to', USERS_CSV_FILE)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Iab8CFowGJre" outputId="7ebff6c5-7245-486b-e353-f2c1f5614e4c"
movies = pd.read_csv(os.path.join(MOVIELENS_DIR, MOVIE_DATA_FILE), 
                    sep='::', 
                    engine='python', 
                    encoding='latin-1',
                    names=['movieid', 'title', 'genre'])

print(len(movies), 'descriptions of', max_movieid, 'movies loaded.')

movies['movie_emb_id'] = movies['movieid'] - 1

movies.to_csv(MOVIES_CSV_FILE, 
              sep='\t', 
              header=True, 
              columns=['movie_emb_id', 'title', 'genre'])

print('Saved to', MOVIES_CSV_FILE)
```

<!-- #region id="4as0coF9GVzF" -->
### Load
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="p9-HverLGM_S" outputId="beee8817-5494-4096-8426-3b2e5a729612"
df = pd.read_csv(RATINGS_CSV_FILE, sep='\t', encoding='latin-1', 
                      usecols=['user_emb_id', 'movie_emb_id', 'rating', 'timestamp'])

df.head(5)
```

<!-- #region id="Pfe6O-R3HJf4" -->
## Preprocessing
<!-- #endregion -->

<!-- #region id="ub30lPffIp29" -->
### Label encode
<!-- #endregion -->

```python id="TpU8kJldIrwg"
le_user = preprocessing.LabelEncoder()
df.loc[:, 'user_emb_id'] = le_user.fit_transform(df['user_emb_id'].values)

le_item = preprocessing.LabelEncoder()
df.loc[:, 'movie_emb_id'] = le_item.fit_transform(df['movie_emb_id'].values)
```

```python colab={"base_uri": "https://localhost:8080/"} id="gUM6bjSFKR8D" outputId="728c840c-af92-44f1-8e03-b1124b2dc769"
num_users = df['user_emb_id'].nunique()
num_movies = df['movie_emb_id'].nunique()

num_users, num_movies
```

<!-- #region id="QV9J3Hh9Hu0x" -->
### Train/val/test split
<!-- #endregion -->

<!-- #region id="eDQIDmjJZZbi" -->
Split the data into random 90%–10% train-test sets, and hold out 10% of the training set for validation.

Using **Stratify with user_id** . This setting is critical, without this setting, it's possible that reviews of one user are all split into one the training or test set and cause bias. For example if all the reviews of user A are put into the training set, then during test time, there is no test data for this user. The test RMSE will be 0 for this user. On the other hand, if all reviews are put into test set, then there is no review for this user during training time and cause the RMSE higher for this user.  
<!-- #endregion -->

```python id="CDvnECoEGoB0"
train_df, test_df = train_test_split(df,
                                     stratify=df['user_emb_id'],
                                     test_size=0.1,
                                     random_state=42)  
```

```python id="DMwU52E5HPq5"
train_df, validate_df = train_test_split(train_df,
                                 stratify=train_df['user_emb_id'],
                                 test_size=0.1,
                                 random_state=42)  
```

<!-- #region id="LNxQxfidHsvb" -->
### Creating a sparse pivot table with users in rows and items in columns
<!-- #endregion -->

<!-- #region id="43u8ei3YZ47X" -->
In order to apply AutoRec on the dataset, the dataset should be transformed to a MxN matrix where R(i, j) is the ratings given by the i<sup>th</sup> user to the j<sup>th</sup> item. The function ```dataPreprocessor``` is used for this transformation. The init_value is the default rating for unobserved ratings. If ```average ``` is set to ```True```, the unobvserved rating will be set as the average rating of the user.
<!-- #endregion -->

```python id="Jqh-qzW7HRlc"
def dataPreprocessor(rating_df, num_users, num_items, init_value=0, average=False):
    """
        INPUT: 
            data: pandas DataFrame. columns=['userID', 'itemID', 'rating' ...]
            num_row: int. number of users
            num_col: int. number of items
            
        OUTPUT:
            matrix: 2D numpy array. 
    """
    if average:
      matrix = np.full((num_users, num_items), 0.0)
      for (_, userID, itemID, rating, timestamp) in rating_df.itertuples():
        matrix[userID, itemID] = rating
      avergae = np.true_divide(matrix.sum(1), np.maximum((matrix!=0).sum(1), 1))
      inds = np.where(matrix == 0)
      matrix[inds] = np.take(avergae, inds[0])
      
    else:
      matrix = np.full((num_users, num_items), init_value)
      for (_, userID, itemID, rating, timestamp) in rating_df.itertuples():
        matrix[userID, itemID] = rating

    return matrix
```

<!-- #region id="qC9RbBvyb9lS" -->
In the Deep_AE_CF paper, the default rating is 0 while in the AutoRec paper, the default rating is 3. So, we decided to tried different default ratings.
<!-- #endregion -->

```python id="p4SSae5VHZ8H"
users_items_matrix_train_zero = dataPreprocessor(train_df, num_users, num_movies, 0)
users_items_matrix_train_one = dataPreprocessor(train_df, num_users, num_movies, 1)
users_items_matrix_train_two = dataPreprocessor(train_df, num_users, num_movies, 2)
users_items_matrix_train_three = dataPreprocessor(train_df, num_users, num_movies, 3)
users_items_matrix_train_four = dataPreprocessor(train_df, num_users, num_movies, 4)
users_items_matrix_train_five = dataPreprocessor(train_df, num_users, num_movies, 5)

users_items_matrix_train_average = dataPreprocessor(train_df, num_users, num_movies, average=True)

users_items_matrix_validate = dataPreprocessor(validate_df, num_users, num_movies, 0)

users_items_matrix_test = dataPreprocessor(test_df, num_users, num_movies, 0)
```

<!-- #region id="2ZQaZg44KuUY" -->
## Util functions
<!-- #endregion -->

<!-- #region id="4HJawlSJXyQ5" -->
### Plots
<!-- #endregion -->

```python id="q76b4LtmH81Z"
def show_error(history, skip):
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  plt.plot(np.arange(skip, len(loss), 1), loss[skip:])
  plt.plot(np.arange(skip, len(loss), 1), val_loss[skip:])
  plt.title('model train vs validation loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='best')
  plt.show()
```

```python id="BkrVfAwpKxAY"
def show_rmse(history, skip):
  rmse = history.history['masked_rmse_clip']
  val_rmse = history.history['val_masked_rmse_clip']
  plt.plot(np.arange(skip, len(rmse), 1), rmse[skip:])
  plt.plot(np.arange(skip, len(val_rmse), 1), val_rmse[skip:])
  plt.title('model train vs validation masked_rmse')
  plt.ylabel('rmse')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='best')
  plt.show()
```

<!-- #region id="OzajzsoaX4Co" -->
### Save and load model
<!-- #endregion -->

```python id="ZUMtbtRoKyOE"
def load_model(name):
  # load json and create model
  model_file = open('{}.json'.format(name), 'r')
  loaded_model_json = model_file.read()
  model_file.close()
  loaded_model = model_from_json(loaded_model_json)
  # load weights into new model
  loaded_model.load_weights("{}.h5".format(name))
  print("Loaded model from disk")
  return loaded_model
```

```python id="re-vSI5MKz27"
def save_model(name, model):
  # # serialize model to JSON
  model_json = model.to_json()
  with open("{}.json".format(name), "w") as json_file:
      json_file.write(model_json)
  # serialize weights to HDF5
  model.save_weights("{}.h5".format(name))
  print("Saved model to disk")
```

<!-- #region id="zYWowF8HLIJs" -->
### Custom loss functions
<!-- #endregion -->

<!-- #region id="rSD5uUlGaR3x" -->
Since it does not make sense to predict zero in the user rating vector, we follow the AutoRec paper to minimize the Masked Mean Squared Error(MMSE).
<!-- #endregion -->

<!-- #region id="hi9k6_89anJm" -->
<!-- #endregion -->

<!-- #region id="a5VHenLrac_U" -->
where r<sub>i</sub> is the actual rating and y<sub>i</sub> is the reconstructed rating. m<sub>i</sub> is a mask function where m<sub>i</sub> =1 where  r<sub>i</sub> is non-zero else m<sub>i</sub>=0. 

Since Masked Mean Squared Error is not provided in Keras, so we customized the error function.
<!-- #endregion -->

```python id="1Wy3oMFQK12P"
def masked_se(y_true, y_pred):
  # masked function
  mask_true = K.cast(K.not_equal(y_true, 0), K.floatx())
  # masked squared error
  masked_squared_error = K.square(mask_true * (y_true - y_pred))
  masked_mse = K.sum(masked_squared_error, axis=-1)
  return masked_mse
```

```python id="YuMloJs-LAVy"
def masked_mse(y_true, y_pred):
  # masked function
  mask_true = K.cast(K.not_equal(y_true, 0), K.floatx())
  # masked squared error
  masked_squared_error = K.square(mask_true * (y_true - y_pred))
  masked_mse = K.sum(masked_squared_error, axis=-1) / K.maximum(K.sum(mask_true, axis=-1), 1)
  return masked_mse
```

```python id="TF4BXd9RLC7s"
def masked_rmse(y_true, y_pred):
  # masked function
  mask_true = K.cast(K.not_equal(y_true, 0), K.floatx())
  # masked squared error
  masked_squared_error = K.square(mask_true * (y_true - y_pred))
  masked_mse = K.sqrt(K.sum(masked_squared_error, axis=-1) / K.maximum(K.sum(mask_true, axis=-1), 1))
  return masked_mse
```

<!-- #region id="DujJYSRTau6f" -->
The performance of the model is measured by the Masked Root Mean Squared Error (MRMSE). Similar to MMSE, we only take into consideration the error where the rating is not zero in the test set. Also, clipping the predicted rating with 1 as minimum and 5 as maximum.
<!-- #endregion -->

```python id="NrtwNle-LExr"
def masked_rmse_clip(y_true, y_pred):
  # masked function
  mask_true = K.cast(K.not_equal(y_true, 0), K.floatx())
  y_pred = K.clip(y_pred, 1, 5)
  # masked squared error
  masked_squared_error = K.square(mask_true * (y_true - y_pred))
  masked_mse = K.sqrt(K.sum(masked_squared_error, axis=-1) / K.maximum(K.sum(mask_true, axis=-1), 1))
  return masked_mse
```

<!-- #region id="5yeFBZ-wLche" -->
### Test custom loss function
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="GPhF4HS7LPrb" outputId="15689a02-d963-411b-f1a4-43b3a0fdecfc"
y_pred = K.constant([[ 1, 1, 1, 3]])

y_true = K.constant([[ 0, 0, 1, 1]])
true = K.eval(y_true)
pred = K.eval(y_pred)
loss = K.eval(masked_se(y_true, y_pred))
rmse = K.eval(masked_rmse(y_true, y_pred))

for i in range(true.shape[0]):
    print(true[i], pred[i], loss[i], rmse[i], sep='\t')
```

```python colab={"base_uri": "https://localhost:8080/"} id="CohdBG5oLekm" outputId="a20193c6-9271-4550-9a52-f40ee0a25f27"
y_pred = K.constant([[ 1, 1, 1, 1], 
                     [ 1, 1, 1, 8],
                     [ 1, 1, 1, 3],
                     [ 1, 1, 1, 3],
                     [ 1, 1, 1, 3],
                     [ 1, 1, 1, 3]])

y_true = K.constant([[ 1, 1, 1, 1],
                     [ 1, 1, 1, 1],
                     [ 0, 1, 1, 1],
                     [ 0, 0, 1, 1],
                     [ 0, 0, 0, 1],
                     [ 0, 0, 0, 0]])

true = K.eval(y_true)
pred = K.eval(y_pred)
loss = K.eval(masked_se(y_true, y_pred))
rmse = K.eval(masked_rmse(y_true, y_pred))

for i in range(true.shape[0]):
    print(true[i], pred[i], loss[i], rmse[i], sep='\t')
```

<!-- #region id="hfOxanWlMJkG" -->
## AutoRec
<!-- #endregion -->

<!-- #region id="193CyDC8aGas" -->
The model we are going to implement is a user-based AutoRec, which take the partially observed ratings vector of a user, project it into a low dimensional latent space and then reconstruct back to the output space to predict the missing rating. 
<!-- #endregion -->

<!-- #region id="O-8BAKaTaBa2" -->
<!-- #endregion -->

```python id="158lIUNQLqAW"
def AutoRec(X, reg, first_activation, last_activation):
    '''
    AutoRec
        INPUT: 
          X: #_user X #_item matrix
          reg: L2 regularization parameter
          first_activation: activation function for first dense layer
          last_activation: activation function for second dense layer
        
        OUTPUT:
          Keras model
    
    '''
    input_layer = x = Input(shape=(X.shape[1],), name='UserRating')
    x = Dense(500, activation=first_activation, name='LatentSpace', kernel_regularizer=regularizers.l2(reg))(x)
    output_layer = Dense(X.shape[1], activation=last_activation, name='UserScorePred', kernel_regularizer=regularizers.l2(reg))(x)
    model = Model(input_layer, output_layer)

    return model
```

```python id="lAYWXra7MH4H"
def AutoRec_LReLU(X, reg):
    '''
    AutoRec
    LReLu is not implemented in to Dense 
    Need to have seperate LeakyRelu layer 
    '''
    input_layer = x = Input(shape=(X.shape[1],), name='UserRating')
    x = Dense(500, name='LatentSpace', kernel_regularizer=regularizers.l2(reg))(x)
    x = LeakyReLU()(x)
    output_layer = Dense(X.shape[1], activation='linear', name='UserScorePred',kernel_regularizer=regularizers.l2(reg))(x)
    model = Model(input_layer, output_layer)

    return model
```

```python colab={"base_uri": "https://localhost:8080/"} id="9GF8hqRBMMJ2" outputId="13027676-b85a-49e4-e6b4-2cc9ce4e748a"
# Build model
model = AutoRec(users_items_matrix_train_zero, 0.0005, 'elu', 'elu')
model.compile(optimizer = Adam(lr=0.0001), loss=masked_mse, metrics=[masked_rmse_clip])
model.summary()
```

<!-- #region id="e4sURnRra-WC" -->
**Baseline settings of AutoRec**

| L2 Regularization | Optimizer | Learning Rate | Epochs | Batch Size | Activations      | Default Rating |
| :---------------: | :-------: | :-----------: | ------ | :--------: | ---------------- | :------------: |
|       0.001       |   Adam    |    0.0001     | 500    |    256     | Sigmoid + Linear |       0        |
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="hgj-bHiGMRNI" outputId="8a3e14b5-ecf5-4724-9d2f-12c009d4e3b4"
#hide-output
hist_Autorec = model.fit(x=users_items_matrix_train_average, y=users_items_matrix_train_zero,
                  epochs=500,
                  batch_size=256,
                  verbose = 2, 
                  validation_data=[users_items_matrix_train_average, users_items_matrix_validate])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 272} id="50GnieGSPSJZ" outputId="708c46d6-5367-48bd-cd32-ebca4b663869"
tf.keras.utils.plot_model(model)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 295} id="lTtDz8xoMo6H" outputId="947b1104-b202-4fb9-e40e-d5be4c586a64"
show_rmse(hist_Autorec, 30)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 295} id="AGE9pkcbMrb_" outputId="d0b76d9e-06c7-45c8-977d-3bea6a2922a7"
show_error(hist_Autorec, 50)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Rs9DK3-BMtWs" outputId="598cb6ac-2a4a-4507-8ab8-18347d936247"
test_result = model.evaluate(users_items_matrix_train_average, users_items_matrix_test)
```

<!-- #region id="BMCOdRGZbVEC" -->
All the hyper-parameters including L2, lambda, learning rate and epochs are not fine-tuned, so the result is not as good as the AutoRec paper. But we can found 2 things:
1. Activations that perform well in Deep Autoencoder CF do not outperform the Sigmoid+Linear baseline.
2. When changing the activation from Sigmod to other activations with unbounded positive part, the model is easier to overfit. 
<!-- #endregion -->

<!-- #region id="-djm0g_cNfzM" -->
## Deep AutoRec
<!-- #endregion -->

<!-- #region id="hYBuNaH-cTVF" -->
<!-- #endregion -->

```python id="9RJxPZYBNg3t"
def Deep_AE_model(X, layers, activation, last_activation, dropout, regularizer_encode, regularizer_decode, side_infor_size=0):
    '''
    Build Deep AE for CF
        INPUT: 
            X: #_user X #_item matrix
            layers: List, each element is the number of neuron for a layer
            reg: L2 regularization parameter
            activation: activation function for all dense layer except the last
            last_activation: activation function for the last dense layer
            dropout: dropout rate
            regularizer_encode: regularizer for encoder
            regularizer_decode: regularizer for decoder
            side_infor_size: size of the one hot encoding side information
        OUTPUT:
            Keras model
    '''

    # Input
    input_layer = x = Input(shape=(X.shape[1],), name='UserRating')
    
    # Encoder
    # -----------------------------
    k = int(len(layers)/2)
    i = 0
    for l in layers[:k]:
      x = Dense(l, activation=activation,
                      name='EncLayer{}'.format(i), kernel_regularizer=regularizers.l2(regularizer_encode))(x)
      i = i+1
      
      
    # Latent Space
    # -----------------------------
    x = Dense(layers[k], activation=activation, 
                                name='LatentSpace', kernel_regularizer=regularizers.l2(regularizer_encode))(x)
    
    # Dropout
    x = Dropout(rate = dropout)(x)
    
    # Decoder
    # -----------------------------
    for l in layers[k+1:]:
      i = i-1
      x = Dense(l, activation=activation, 
                      name='DecLayer{}'.format(i), kernel_regularizer=regularizers.l2(regularizer_decode))(x)
      
    # Output

    output_layer = Dense(X.shape[1]-side_infor_size, activation=last_activation, name='UserScorePred', kernel_regularizer=regularizers.l2(regularizer_decode))(x)

    # this model maps an input to its reconstruction
    model = Model(input_layer, output_layer)

    return model
```

```python id="vZVIKUWaNmYr"
layers = [256, 512, 256]
#layers = [512, 256, 128, 256, 512]
#layers = [512, 256, 512]
#layers = [128, 256, 512, 256, 128]
#layers = [512, 512, 512]
dropout = 0.8
# activation = 'sigmoid'
# last_activation = 'linear'
activation = 'selu'
last_activation = 'selu'
regularizer_encode = 0.001
regularizer_decode = 0.001
```

<!-- #region id="rZ1ECx4Scdsh" -->
| L2 Regularization | Optimizer | Learning Rate | Epochs | Batch Size | Activations | Default Rating | Dropout |
| :---------------: | :-------: | :-----------: | ------ | :--------: | ----------- | :------------: | :-----: |
|       0.001       |   Adam    |    0.0001     | 500    |    256     | SELU+SELU   |       0        |   0.8   |
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="yPHfw_e6NmUY" outputId="41a32348-f371-4e9a-e078-57119ff6deef"
# Build model
model = Deep_AE_model(users_items_matrix_train_zero, layers, activation, last_activation, dropout, regularizer_encode, regularizer_decode)
model.compile(optimizer = Adam(lr=0.0001), loss=masked_mse, metrics=[masked_rmse_clip]) 
model.summary()
```

```python colab={"base_uri": "https://localhost:8080/"} id="tDixZRk0NmQ8" outputId="8ee90a0b-8a54-4c5d-aca4-baa510f346e4"
#hide-output
hist_Deep_AE = model.fit(x=users_items_matrix_train_zero, y=users_items_matrix_train_zero,
                  epochs=500,
                  batch_size=256,
                  validation_data=[users_items_matrix_train_zero, users_items_matrix_validate], verbose=2)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 564} id="7rAo8af8Ukzi" outputId="a7c0af91-8cb3-469b-ab25-8c1f2bc8c484"
tf.keras.utils.plot_model(Deep_AE)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 295} id="gojRI6m2Nt15" outputId="07a5627d-f297-4acc-f280-50e8257da10e"
show_error(hist_Deep_AE, 100)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 295} id="AgWKNC-bNvRw" outputId="5856a03f-1a2b-4fea-a6e7-dcb3a200b25b"
show_rmse(hist_Deep_AE, 100)
```

```python colab={"base_uri": "https://localhost:8080/"} id="e7QcKz7CN0Kd" outputId="77e53d74-ed13-4b8e-ff2b-71dfde18601b"
test_result_deep = model.evaluate(users_items_matrix_train_zero, users_items_matrix_test)
```

<!-- #region id="6SPXy1_acsRI" -->
### Findings
1. When we compared the average and zero as default rating in AutoRec, we found that average converged faster but with noise. But when the model goes deeper, **the zero default rating converged faster and with less noise.** However, when we take a look at the loss, the gap between training and validation is larger in zero default setting. This means when we use zero as default rating, the model is easier to overfit. 
2. Adding more layers does not help for both BSB and SBS shape. As we go deeper, it’s easier to get overfitted and increasing the regularization parameters will bring the test performance down. So, our case, using three hidden layers is the best option.
3. `[512, 256, 512]` and `[256, 512, 256]` have similar performance but `[256, 512, 256]` has half the number of parameters. So we used `[256, 512, 256]`, as fewer parameters not only allows us to train model with less data but also can mitigate overfitting. 
<!-- #endregion -->

```python id="n8vTD6NQY-Hu" colab={"base_uri": "https://localhost:8080/"} outputId="f5c66cb2-15c6-4e51-e432-62fdd8d732d0"
for layer in model.layers: 
  print(layer.get_config())
```

```python id="jYrEB-sxVqtU" colab={"base_uri": "https://localhost:8080/", "height": 265} outputId="d0ae4ff0-8c85-470b-c90b-940eb4089647"
first_layer_weights = model.layers[1].get_weights()[0]
plt.plot(np.arange(0, 256, 1), first_layer_weights[0])
plt.plot(np.arange(0, 256, 1), first_layer_weights[1])
plt.plot(np.arange(0, 256, 1), first_layer_weights[2])
plt.show()
```

<!-- #region id="kwKGR78KR3X9" -->
> Note: Lot of noise.
<!-- #endregion -->

<!-- #region id="Tko0sfZKndDD" -->
## Denoising
<!-- #endregion -->

<!-- #region id="SB2CBjZNAfZ_" -->
Common corruption choices are the additive Gaussian noise and multiplicative dropout noise. In the Denoising paper, authors used multiplicative dropout noise.
<!-- #endregion -->

<!-- #region id="LM4rPgpJApXN" -->
#### Gussian AutoRec
<!-- #endregion -->

```python id="WNjpJ_gUngjW"
## Adding Gaussin noise to input
noise_factor = 0.4
users_items_matrix_train_average_noisy = users_items_matrix_train_average + noise_factor * np.random.normal(size=users_items_matrix_train_zero.shape) 
users_items_matrix_train_zero_noisy = users_items_matrix_train_zero + noise_factor * np.random.normal(size=users_items_matrix_train_zero.shape) 
```

```python id="dovJDJEAVIjO" colab={"base_uri": "https://localhost:8080/"} outputId="93013f74-ffa6-4e7f-a995-4bb67b376310"
#hide-output
model = AutoRec(users_items_matrix_train_average_noisy, 0.001, 'elu', 'elu')

model.compile(optimizer = Adam(lr=0.0001), loss=masked_mse, metrics=[masked_rmse_clip])

hist_Autorec = model.fit(x=users_items_matrix_train_average_noisy, y=users_items_matrix_train_zero,
                  epochs=500,
                  batch_size=256,
                  verbose = 2, 
                  validation_data=[users_items_matrix_train_average_noisy, users_items_matrix_validate])
```

```python id="dhEUlnQeVIsL" colab={"base_uri": "https://localhost:8080/", "height": 295} outputId="a2e57c66-4a6a-478f-9d0a-7809b88a1d99"
show_error(hist_Autorec, 20)
```

```python id="6cMHBgQFVIwK" colab={"base_uri": "https://localhost:8080/", "height": 295} outputId="87550990-b256-42fc-f387-53e682d88dc9"
show_rmse(hist_Autorec, 300)
```

```python id="OdKZjB6mVIpV" colab={"base_uri": "https://localhost:8080/"} outputId="b0908b76-9ce9-4b85-f8a2-dd9280e0a28a"
test_result_deep = model.evaluate(users_items_matrix_train_average_noisy, users_items_matrix_test)
```

<!-- #region id="srsVRT8bAu-c" -->
#### Gaussian Deep AE CF
<!-- #endregion -->

```python id="8ZaMxFLuplMB"
layers = [256, 512, 256]
dropout = 0.8

activation = 'selu'
last_activation = 'selu'
regularizer_encode = 0.001
regularizer_decode = 0.001
```

```python id="Y2PYRVbrngp8" colab={"base_uri": "https://localhost:8080/"} outputId="647f95b6-0054-46c6-b696-60aaf81a5975"
# Build model
model = Deep_AE_model(users_items_matrix_train_zero_noisy, layers, activation, last_activation, dropout, regularizer_encode, regularizer_decode)
model.compile(optimizer = Adam(lr=0.0001), loss=masked_mse, metrics=[masked_rmse_clip]) 
model.summary()
```

```python id="2IrVVyYBpnN3" colab={"base_uri": "https://localhost:8080/"} outputId="9df52e2d-3773-4435-b309-9b66601bed41"
#hide-output
hist_Deep_AE_denoise = model.fit(x=users_items_matrix_train_zero_noisy, y=users_items_matrix_train_zero,
                  epochs=500,
                  batch_size=256,
                  validation_data=[users_items_matrix_train_zero_noisy, users_items_matrix_validate], verbose=2)
```

```python id="EXTsR-m73LxO" colab={"base_uri": "https://localhost:8080/", "height": 295} outputId="05a574ff-f5a1-4501-c7e5-935fe85e64aa"
show_error(hist_Deep_AE_denoise, 20)
```

```python id="4b3oHDEq3Lu9" colab={"base_uri": "https://localhost:8080/", "height": 295} outputId="4688f1bb-12d5-4b37-d29f-d0b239925d39"
show_rmse(hist_Deep_AE_denoise, 20)
```

```python id="YerhiUcPgMwI" colab={"base_uri": "https://localhost:8080/"} outputId="a3d1b875-0937-4fc1-ae80-2128b76a773f"
test_result_deep = model.evaluate(users_items_matrix_train_zero_noisy, users_items_matrix_test)
```

```python id="uRZ20B1kgMzr" colab={"base_uri": "https://localhost:8080/"} outputId="d331b211-a382-432c-89eb-050038ee993e"
predict_deep = model.predict(users_items_matrix_train_zero_noisy)
loss = K.eval(masked_rmse_clip( 
    K.constant((users_items_matrix_train_zero)), 
    K.constant(predict_deep)))
np.mean(loss)
```

<!-- #region id="uDIsvnprdxxB" -->
Adding Gaussian Noise did not improve the model. As default rating has an impact on the performance, adding noise is changing the default rating and this may be one potential reason. Deep AutoRec has the similar graph as AutoRec.
<!-- #endregion -->

<!-- #region id="rbFBt2fLA2El" -->
### Dropout Noise
<!-- #endregion -->

<!-- #region id="tCj6IRnUeCTc" -->
In the denoising paper, it masked out non-zero elements randomly in each batch and use the masked input. However, using Keras to implement this feature will be the same as using pure TensorFlow. Due to the time limit of this case, we will leave this as future work and we made a compromise by adding a dropout layer between input and first dense layer. This dropout will mask out all elements randomly with a dropout rate.
<!-- #endregion -->

```python id="Lv1MjvDhgNE3"
  def Deep_AE_DropNoise_model(X, layers, activation, last_activation, dropout, regularizer_encode, regularizer_decode, noise):
    '''
    Build Deep AE for CF
    '''

    # Input
    input_layer = x = Input(shape=(X.shape[1],), name='UserRating')
    
    # Dropout Noise
    x = Dropout(rate = noise)(x)
    
    # Encoder
    # -----------------------------
    k = int(len(layers)/2)
    i = 0
    for l in layers[:k]:
      x = Dense(l, activation=activation,
                      name='EncLayer{}'.format(i), kernel_regularizer=regularizers.l2(regularizer_encode))(x)
      i = i+1
      
      
    # Latent Space
    # -----------------------------
    x = Dense(layers[k], activation=activation, 
                                name='LatentSpace', kernel_regularizer=regularizers.l2(regularizer_encode))(x)
    
    # Dropout
    x = Dropout(rate = dropout)(x)
    
    # Decoder
    # -----------------------------
    for l in layers[k+1:]:
      i = i-1
      x = Dense(l, activation=activation, 
                      name='DecLayer{}'.format(i), kernel_regularizer=regularizers.l2(regularizer_decode))(x)
    # Output

    output_layer = Dense(X.shape[1], activation=last_activation, name='UserScorePred', kernel_regularizer=regularizers.l2(regularizer_decode))(x)

    # this model maps an input to its reconstruction
    model = Model(input_layer, output_layer)

    return model
```

```python id="hCNs1U-mxazK" colab={"base_uri": "https://localhost:8080/"} outputId="d0f8c1f3-b40a-4dae-eb45-d544e0d314ed"
#hide-output
layers = [256, 512, 256]
dropout = 0.8
activation = 'selu'
last_activation = 'selu'
regularizer_encode = 0.001
regularizer_decode = 0.001
dropN = 0.1
# Build model
Deep_AE_denoise_dropN = Deep_AE_DropNoise_model(users_items_matrix_train_zero, layers, activation, last_activation, dropout, regularizer_encode, regularizer_decode, dropN)
Deep_AE_denoise_dropN.compile(optimizer = Adam(lr=0.0001), loss=masked_mse, metrics=[masked_rmse_clip]) 
Deep_AE_denoise_dropN.summary()
hist_Deep_AE_denoise_dropN = Deep_AE_denoise_dropN.fit(x=users_items_matrix_train_zero, y=users_items_matrix_train_zero,
                  epochs=500,
                  batch_size=256,
                  validation_data=[users_items_matrix_train_zero, users_items_matrix_validate], verbose=2)
```

```python id="-EtziWmsxaw9" colab={"base_uri": "https://localhost:8080/", "height": 295} outputId="2d649a21-3c12-4680-8a26-5f123c22af61"
show_rmse(hist_Deep_AE_denoise_dropN, 30)
```

```python id="ZzTcRLP8Eqnn" colab={"base_uri": "https://localhost:8080/", "height": 295} outputId="9376cb41-1bdc-4c39-fde6-072fee67a955"
show_error(hist_Deep_AE_denoise_dropN, 30)
```

```python id="wW3I_5cOxate" colab={"base_uri": "https://localhost:8080/"} outputId="49cd13fe-cef2-49d3-f750-b7644982cbbe"
test_result_deep = Deep_AE_denoise_dropN.evaluate(users_items_matrix_train_zero, users_items_matrix_test)
```

<!-- #region id="0ZRSmQ58eaGM" -->
As we can see in figures, when the dropout rate increase for the noise, the RMSE started increasing. When the rate was 0.1, the performance actually was better than the baseline but since it’s only 0.002 difference, it may still be in the range of error. It needs cross-validation for further verification.
<!-- #endregion -->

<!-- #region id="hP0RmB6dkgTP" -->
## Hybrid
<!-- #endregion -->

<!-- #region id="73kcGNd0ed5y" -->
Since we have the information about each user, we will try adding the side-information in this model.
<!-- #endregion -->

<!-- #region id="QqY3NYh4MbBX" -->
### Preprocessing the side information
<!-- #endregion -->

```python id="K0JIjYmBkmIM" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="6ef03ada-8355-49d0-ab87-940abf8370f7"
user_df = pd.read_csv('ml1m_users.csv',sep='\t', encoding='latin-1', 
                      usecols=['user_emb_id', 'gender', 'age', 'occupation'])

user_df.head(5)
```

```python id="t4DaP281kmPo"
# Transform side information to onehot encoding
user_df['age'] = preprocessing.LabelEncoder().fit(user_df['age']).transform(user_df['age'])
user_df['gender'] = preprocessing.LabelEncoder().fit(user_df['gender']).transform(user_df['gender'])
onehot_df = preprocessing.OneHotEncoder(handle_unknown='ignore', sparse=False).fit(user_df[['gender', 'age', 'occupation']]).transform(user_df[['gender', 'age', 'occupation']])
```

```python id="67Aya8pmkmNR" colab={"base_uri": "https://localhost:8080/"} outputId="4499dc59-ca07-4f54-aec0-ed43a9f0c53f"
onehot_df.shape
```

<!-- #region id="UcsJUr6UeoCi" -->
For each user, we have gender, age and occupation and after transforming to one hot encoding format, each user has totally 30 features.
<!-- #endregion -->

<!-- #region id="2-7oFaVXBXY4" -->
### Concatenate content to input keep output the same
<!-- #endregion -->

```python id="bZ1DPBFko_4_"
#Concatenate the one hot encoding side information to the rating matrix
user_items_user_info = np.concatenate((users_items_matrix_train_zero, onehot_df), axis=1)
```

```python id="7XRSIrkLyHUK" colab={"base_uri": "https://localhost:8080/"} outputId="cc01471e-d18e-4fe8-ea54-c9c4f4370d5c"
user_items_user_info.shape
```

<!-- #region id="E7csMVI_eqdT" -->
For this method, we concatenated the side information to the rating matrix, so the shape of the matrix will be changed from 6040x3706 to 6040x3736.
<!-- #endregion -->

```python id="bOPPi_0ppAJP" colab={"base_uri": "https://localhost:8080/"} outputId="affba402-294b-4a34-a4f8-202b2f8e082b"
layers = [256, 512, 256]
dropout = 0.8
activation = 'selu'
last_activation = 'selu'
regularizer_encode = 0.001
regularizer_decode = 0.001
# Build model
Deep_AE_concate = Deep_AE_model(user_items_user_info, layers, activation, last_activation, dropout, regularizer_encode, regularizer_decode, 30)
Deep_AE_concate.compile(optimizer = Adam(lr=0.0001), loss=masked_mse, metrics=[masked_rmse_clip]) 
Deep_AE_concate.summary()
```

```python id="K9-Lzc5bpAQL" colab={"base_uri": "https://localhost:8080/"} outputId="bdc17c42-2f68-4e5d-d871-aaacb27242b4"
#hide-output
hist_Deep_AE_concate = Deep_AE_concate.fit(x=user_items_user_info, y=users_items_matrix_train_zero,
                  epochs=500,
                  batch_size=256,
                  validation_data=[user_items_user_info, users_items_matrix_validate], verbose=2)
```

```python id="dkVz73izpAUV" colab={"base_uri": "https://localhost:8080/", "height": 295} outputId="bb6f81ef-0f5a-402a-800d-76f957806fd3"
show_error(hist_Deep_AE_concate, 20)
```

```python id="cQFeiVUWugqK" colab={"base_uri": "https://localhost:8080/", "height": 295} outputId="a0e35f0d-eda2-4cca-d367-ad5a24171bbe"
show_rmse(hist_Deep_AE_concate, 20)
```

<!-- #region id="cljjDRa3fJMa" -->
We tested this model on the setting of [256, 512, 256]+zero. Adding the side information does not have a limited impact on the result. The error graph, Val RMSE graph and test RMSE are similar to the model without side information. As the repartition of known entries in the dataset is not uniform, the estimates are biased towards users with a lot of rating. For these users, the dataset already has a lot of information and comparing with existing rating features, 30 side information feature will have limited effect. But according to [this](https://arxiv.org/abs/1603.00806) paper, when the users have fewer ratings, the side information will have more effect. 
<!-- #endregion -->
