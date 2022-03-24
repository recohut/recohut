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

<!-- #region id="q9Epz_ny0X9M" -->
# Training and indexing movie recommender
> Building tensorflow model on movielens latest small variant and indexing using NMSlib for efficient retrieval

- toc: true
- badges: true
- comments: true
- categories: [Tensorflow, Movie, NMSLib]
- author: "<a href='https://github.com/BastienVialla/MovieLens'>Bastien Vialla</a>"
- image:
<!-- #endregion -->

<!-- #region id="i452-P3ouLle" -->
## Setup
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ku66w6Lsx51X" outputId="fbf62aa9-d09a-4a15-a793-b08eb7e8c050"
!pip install -q nmslib
```

```python id="q4dUwWK9sH9K"
import re
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import seaborn as sns
from sklearn.model_selection import train_test_split

from keras.models import load_model, model_from_json
from keras.models import Model as KerasModel
from keras.layers import Input, Dense, Activation, Reshape, Dropout
from keras.layers import Concatenate
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras import optimizers
from pathlib import Path
from sklearn.metrics import mean_squared_error

import nmslib

import warnings
warnings.filterwarnings('ignore')
```

<!-- #region id="MOVLvLQ5uNPU" -->
## Loading data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="i2cGX0v-sx8s" outputId="bb1fa7af-95f2-4987-ecd3-0623047b615d"
!wget http://files.grouplens.org/datasets/movielens/ml-latest-small.zip
!unzip ml-latest-small.zip
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="Ryuil9zms7Fz" outputId="1392fbfe-b3c2-457a-af34-a2aa4cd14db4"
PATH = "ml-latest-small"
ratings_raw = pd.read_csv(PATH+"/ratings.csv")
ratings_raw.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="cDytty0HtEkU" outputId="5284aaad-f31f-4d95-c066-a1af9ebbdfd3"
movies_raw = pd.read_csv(PATH+"/movies.csv")
movies_raw.head()
```

<!-- #region id="Br4bz4qVtUbZ" -->
## Create dictionnaries to convert ids and indexes
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="8rnJVVxYtPss" outputId="b2280f7c-9033-4227-b38d-2cee1aaff178"
ratings_train = ratings_raw.copy()

users_uniq = ratings_train.userId.unique()
user2idx = {o:i for i,o in enumerate(users_uniq)}
idx2user = {i:o for i,o in enumerate(users_uniq)}
ratings_train.userId = ratings_train.userId.apply(lambda x: user2idx[x])

movies_uniq = ratings_train.movieId.unique()
movie2idx = {o:i for i,o in enumerate(movies_uniq)}
idx2movie = {i:o for i,o in enumerate(movies_uniq)}
ratings_train.movieId = ratings_train.movieId.apply(lambda x: movie2idx[x])

n_users = int(ratings_train.userId.nunique())
n_movies = int(ratings_train.movieId.nunique())

n_users, n_movies
```

```python id="rSYD5kRVtfhh"
def save_obj(obj, name):  
    with open(Path(f"{name}.pkl"), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

save_obj(user2idx, "user2idx")
save_obj(idx2user, "idx2user")
save_obj(movie2idx, "movie2idx")
save_obj(idx2movie, "idx2movie")
```

<!-- #region id="b3FXwa4HuDmD" -->
## Keras Model
The model works as follows:
1. Embedds the user and movie id.
2. Concanate the user embedding, movie embedding and the weighted rating into one vector.
3. Passes to linear layers with dropout.

The architecture takes as parameters the embedding size, the size of hidden layers, and the dropout probability associate to them.
<!-- #endregion -->

```python id="3SdU9sr6tn1v"
class MovieNet: 
    def rmse(self, y, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y)))

    def custom_activation(self, x):
        return K.sigmoid(x) * (self.max_rating+1)

    def __init__(self, n_users, n_movies, min_rating=0.5, max_rating=5):
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.n_users = n_users
        self.n_movies = n_movies
        
    def build_model(self, emb_size=[50, 50], hl=[10], drop=[0.25], emb_trainable=True):
        inputs = [Input(shape=(1,)), Input(shape=(1,))] #, Input(shape=(1,))]
        users_emb = Embedding(self.n_users, emb_size[0], name='users', trainable=emb_trainable)(inputs[0])
        movies_emb = Embedding(self.n_movies, emb_size[1], name='movies', trainable=emb_trainable)(inputs[1])
        outputs_emb = [Reshape(target_shape=(emb_size[0],))(users_emb), Reshape(target_shape=(emb_size[1],))(movies_emb)]
        
        output_model = Concatenate()(outputs_emb)
        for i in range(0, len(hl)):
            output_model = Dense(hl[i], kernel_initializer='uniform')(output_model)
            output_model = Activation('relu')(output_model)
            output_model = Dropout(drop[i])(output_model)

        output_model = Dense(1)(output_model)

        output_model = Activation(self.custom_activation)(output_model)
        
        self.model = KerasModel(inputs=inputs, outputs=output_model)
        
        opt = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        
        self.model.compile(loss='mse', optimizer=opt, metrics=[self.rmse])
        
          
    def prepare_input(self, _X):
        X = [_X.userId.values, _X.movieId.values]#, _X.ratingWeight]
        return X            
            
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return mean_squared_error(y, y_pred)
    
    def fit(self, X_train, y_train, X_valid, y_valid, epochs=50, batch_size=32, verbose=1):
        self.model.fit(self.prepare_input(X_train), y_train,
                       validation_data=(self.prepare_input(X_valid), y_valid),
                      epochs=epochs, batch_size=batch_size, verbose=verbose)
        # print("Result on validation data: ", self.evaluate(X_valid, y_valid))
        
    def predict(self, X):
        y_pred = self.model.predict(self.prepare_input(X))
        return y_pred.flatten()

    def save_model(self, path=Path(""), name="MovieModel"):
        self.model.save_weights(f"{path}/{name}_weights.h5")
        with open(f"{path}/{name}_arch.json", 'w') as f:
            f.write(self.model.to_json())
    
    def load_model(self, path=Path(""), name="MovieModel"):
        with open(f"{path}/{name}_arch.json", 'r') as f:
            self.model = model_from_json(f.read(), custom_objects={"custom_activation": self.custom_activation})
        self.model.load_weights(f"{path}/{name}_weights.h5") 
```

```python id="Y4phJVhkubkg"
movie_model = MovieNet(n_users, n_movies)
movie_model.build_model(emb_size=[50, 50], hl=[70, 10], drop=[0.4, 0.3])
```

```python colab={"base_uri": "https://localhost:8080/"} id="Y1FNGNvduhwD" outputId="d94071b9-3d9d-4373-f28b-1ede459e986f"
X = ratings_train.drop(['timestamp', 'rating'], axis=1)
y = ratings_train['rating']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
len(X_train), len(X_valid), len(y_train), len(y_valid)
```

<!-- #region id="iaL4HFnwvf66" -->
It's important that every movie are in the training set to have trained embedding of each of them.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Hx_kRu0tvdrz" outputId="6eb78da3-497d-4bb5-8299-8f2b1ef4663a"
len(X_train["movieId"].unique()), n_movies, n_movies - len(X_train["movieId"].unique())
```

```python id="THWmFnBbvjzE"
miss_movies = ratings_train[~ratings_train.movieId.isin(X_train["movieId"].unique())]["movieId"].unique()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="mnrgIsHxvlma" outputId="a1c62716-4e2c-4b79-d15e-60f846b7e404"
concat = pd.DataFrame()
for i in miss_movies:
    concat = concat.append(ratings_train[ratings_train.movieId == i].sample(1))
    
concat.head()
```

<!-- #region id="lVsjsjJ_we8z" -->
### Train and save model
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="VeBXycPmvpWU" outputId="91186dbd-ed93-431f-9f12-fe1d16c2d4c6"
X_valid.drop(concat.index, axis=0, inplace=True)
y_valid.drop(concat.index, axis=0, inplace=True)

X_train = pd.concat([X_train, concat.drop(["rating", "timestamp"], axis=1)])
y_train = pd.concat([y_train, concat["rating"]])

len(X_train["movieId"].unique()), n_movies
```

```python colab={"base_uri": "https://localhost:8080/"} id="P9pspL8Bvs3x" outputId="784b3268-2628-4f61-9e40-b226e66f3019"
movie_model.fit(X_train, y_train, X_valid, y_valid, epochs=5, batch_size=512)
```

```python id="36IRGfdtv-uB"
movie_model.save_model(name="movie_model")
```

<!-- #region id="TiVrv7fLwaw3" -->
## Load objects
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="S2LS9TH4xEV6" outputId="41666fa8-c6c1-4442-a55b-9bc39d7740ba"
PATH = 'ml-latest-small'
ratings_raw = pd.read_csv(PATH+"/ratings.csv")
movies_raw = pd.read_csv(PATH+"/movies.csv")

def load_obj(name):  
    with open(Path(f"{name}.pkl"), 'rb') as f:
        return pickle.load(f)

user2idx = load_obj("user2idx")
idx2user = load_obj("idx2user")
movie2idx = load_obj("movie2idx")
idx2movie = load_obj("idx2movie")

ratings = ratings_raw.copy()
ratings["userId"] = ratings["userId"].apply(lambda x: user2idx[x])
ratings["movieId"] = ratings["movieId"].apply(lambda x: movie2idx[x])
ratings.head()
```

```python id="5bWxU-Dov_El"
movie_model = MovieNet(n_users, n_movies)
movie_model.load_model(name="movie_model")
```

<!-- #region id="wfMACIA5yIA6" -->
## Prediction
<!-- #endregion -->

```python id="OwOua4l6woBI"
X_pred = pd.DataFrame({"userId": [0 for _ in range(n_movies)], "movieId": [i for i in range(n_movies)]})
```

```python id="t1MvHN1Jw2tA"
def predict_user(user_id):
    X_pred["userId"] = X_pred.userId.apply(lambda x: user_id)
    preds = movie_model.predict(X_pred)
    df_preds = pd.DataFrame({"pred": preds, "movieId": [i for i in range(n_movies)],
                             "title": [movies_raw.loc[movies_raw.movieId == idx2movie[i]]["title"].values[0] for i in range(n_movies)]})
    return df_preds

def suggest_user(user_id, m=10):
    preds = predict_user(user_id)
    preds.sort_values("pred", ascending=False, inplace=True)
    r = ratings[ratings.userId == 0]["movieId"].values
    preds.drop(r, axis=0, inplace=True)
    return preds.drop("movieId", axis=1)[:m]

def user_rating(user_id):
    preds = predict_user(user_id)
    return pd.merge(ratings[ratings.userId == user_id][["rating", "movieId"]], preds, on="movieId")
```

```python colab={"base_uri": "https://localhost:8080/"} id="0kyMLGNRw3FH" outputId="d29a5821-e956-41f7-9b15-f56b24d16283"
user_id = np.random.randint(0, n_users)
user_id
```

```python colab={"base_uri": "https://localhost:8080/", "height": 359} id="vJGmB9row5et" outputId="8f8079f0-3628-454d-e958-0d426cea9088"
preds = user_rating(user_id).sort_values("rating", ascending=False)[:]
preds.head(10)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 300} id="G13Nfe-3w52c" outputId="bb669dfb-4202-492e-e817-29dad837ee8f"
sns.boxplot(preds["rating"], preds["pred"])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 359} id="SB52Z46yxoSP" outputId="5cb96746-4d37-461b-d60e-45282fdbc22d"
suggest_user(user_id)
```

<!-- #region id="3wQvVQZDyEwZ" -->
## Recommanding movies using KNN on embedding spaces
<!-- #endregion -->

<!-- #region id="cCwJRqIFyeyd" -->
### Looking at movies embedding space
<!-- #endregion -->

```python id="r5FOT5L4xx0X"
movies_index = nmslib.init(space='angulardist', method='hnsw')
movies_index.addDataPointBatch(movie_model.model.get_layer("movies").get_weights()[0])

M = 100
efC = 1000
efS = 1000
num_threads = 6
index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC, 'post' : 0}
query_time_params = {'efSearch': efS}

movies_index.createIndex(index_time_params)
movies_index.setQueryTimeParams(query_time_params)
```

```python id="bDrXWhsQx0jQ"
def get_knns(index, vecs, n_neighbour):
     return zip(*index.knnQueryBatch(vecs, k=n_neighbour, num_threads=6))

def get_knn(index, vec, n_neighbour):
    return index.knnQuery(vec, k=n_neighbour)

def suggest_movies_knn(movieId, n_suggest = 5):
    res = get_knn(movies_index, movie_model.model.get_layer("movies").get_weights()[0][movieId], n_suggest)[0]
    return movies_raw[movies_raw.movieId.isin([idx2movie[i] for i in res])]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 359} id="AiYbTuP4yQDn" outputId="a32457c9-9c0b-4332-f1d0-be819819f56a"
movie_id = 763
suggest_movies_knn(movie_id, 10)
```

<!-- #region id="GlO8IBy40DFP" -->
### Looking at users embedding space
<!-- #endregion -->

```python id="JA4uaKN7yYn7"
users_index = nmslib.init(space='angulardist', method='hnsw')
users_index.addDataPointBatch(movie_model.model.get_layer("users").get_weights()[0])

M = 100
efC = 1000
efS = 1000
num_threads = 6
index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC, 'post' : 0}
query_time_params = {'efSearch': efS}

users_index.createIndex(index_time_params)
users_index.setQueryTimeParams(query_time_params)
```

```python id="phqnOAUl0Okk"
def suggest_users_knn(user_id, n_suggest = 5):
    res = get_knn(users_index, movie_model.model.get_layer("users").get_weights()[0][user_id], n_suggest)[0]
    for uid in res[1:]:
        moviesId = ratings[ratings.userId == uid].sort_values("rating", ascending=False)[:10]["movieId"].values
        print("From user", uid, ": ")
        display(movies_raw[movies_raw.movieId.isin([idx2movie[i] for i in moviesId])])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="4xflCpdQz3w0" outputId="1f55cb44-a2a6-4395-b58c-0f673c02f9a7"
suggest_users_knn(user_id)
```
