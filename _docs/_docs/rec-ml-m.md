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

<!-- #region id="QBLgWidk3oiR" -->
# Recommenders on ML-1m
<!-- #endregion -->

<!-- #region id="VokPZEtG1Sil" -->
## Setup
<!-- #endregion -->

```python id="Qzecw902yzTt"
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import math

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

import matplotlib.pyplot as plt
```

<!-- #region id="jyg_YwQv1VHA" -->
## Data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="SM8pU9WtyvVN" executionInfo={"status": "ok", "timestamp": 1639029882296, "user_tz": -330, "elapsed": 1458, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4e376858-3639-43e9-dca0-871dcc759328"
!wget -q --show-progress http://files.grouplens.org/datasets/movielens/ml-1m.zip
!unzip ml-1m.zip
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="C7a4g9bKy399" executionInfo={"status": "ok", "timestamp": 1639030735099, "user_tz": -330, "elapsed": 5966, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a5b23854-9921-4589-dc90-b14538f7031c"
df_ratings = pd.read_csv("./ml-1m/ratings.dat",
                         sep="::",
                         header=None,
                         engine='python',
                         names=["UserID", "MovieID", "Rating", "Timestamp"])
df_ratings.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="QeDG1njUy6sx" executionInfo={"status": "ok", "timestamp": 1639030735102, "user_tz": -330, "elapsed": 45, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="24a9ba9c-e2a8-4070-d3ed-6cf7860f56cd"
df_movies = pd.read_csv("./ml-1m/movies.dat",
                         sep="::",
                         header=None,
                         engine='python',
                         names=["MovieID", "Title", "Genres"])
df_movies.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="TMgBHg7Ry-gk" executionInfo={"status": "ok", "timestamp": 1639030735105, "user_tz": -330, "elapsed": 44, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d6fc221e-98a2-44ab-91e4-a4677154a8c9"
df_users = pd.read_csv("./ml-1m/users.dat",
                         sep="::",
                         header=None,
                         engine='python',
                         names=["UserID", "Gender", "Age", "Occupation", "Zip-code"])
df_users.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="3VUpnZnazAsm" executionInfo={"status": "ok", "timestamp": 1639030735108, "user_tz": -330, "elapsed": 43, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="71e9d857-ac35-4382-891c-87cf2cbea1c6"
df_ratings.shape, df_movies.shape, df_users.shape
```

```python id="8Z5Mu6QPzB3-"
# merge
df_ratings_movies = pd.merge(df_ratings, df_movies, on='MovieID')
df = pd.merge(df_ratings_movies, df_users, on="UserID")
df_raw = df.copy()

# drop columns
df = df.drop(['Timestamp','Genres','Gender','Age','Occupation','Zip-code'], axis=1)
```

<!-- #region id="sbEoj-MKxyla" -->
## Recommend Popular (most-watched) Movies
<!-- #endregion -->

```python id="5rpgzYkHx2YA"
# Recommend 5 most seen movies
def recommend_movie(n):
    movie_rank = df['MovieID'].value_counts()[:n]
    recommend_movies = df.loc[movie_rank.index]
    recommend = recommend_movies['Title']
    return recommend
```

```python colab={"base_uri": "https://localhost:8080/"} id="WpYPRsWtx6Tq" executionInfo={"status": "ok", "timestamp": 1639030735978, "user_tz": -330, "elapsed": 18, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2e4efcd7-64b2-4165-aae6-6eff4c513bbf"
recommend_movie(5)
```

<!-- #region id="YUg356Eqx7QW" -->
Evaluate
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="TOUp5_Z7yBN_" executionInfo={"status": "ok", "timestamp": 1639030739176, "user_tz": -330, "elapsed": 606, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d4e10476-63f3-463a-89ab-378090f41609"
# Split train, test set
x_train, x_test = train_test_split(df, test_size=0.05)

# Recommend n most popular movies on a dataset
def popular_movie(dataset, n):
    movie_rank = dataset['MovieID'].value_counts()[:n]
    popular_movies = dataset.iloc[movie_rank.index]
    return popular_movies

# Calculate hitrate@K
def hitrate(K):
    raw_ranking = popular_movie(df, K)
    pred_ranking = popular_movie(x_test, K)
    return raw_ranking['MovieID'].isin(pred_ranking['MovieID']).value_counts(normalize=True)[True]

hitrate(100)
```

<!-- #region id="2kWNgnmHyipn" -->
## Recommend Popular (high-rated) Movies
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="hONKTtwyy6bN" executionInfo={"status": "ok", "timestamp": 1639030054352, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d64742be-963b-4b1f-a761-0e081fa6555c"
# Recommend 5 movies with high ratings
def recommend_movie2(n):
    movie_sort = movie_mean.sort_values(ascending=False)[:n]
    recommend_movies = df.loc[movie_sort.index]
    recommendation = recommend_movies['Title']
    return recommendation

movie_mean = df.groupby(['MovieID'])['Rating'].mean()
recommend_movie2(5)
```

<!-- #region id="SV33kdAGzAzC" -->
Evaluate
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="yBKgYdWCzDSO" executionInfo={"status": "ok", "timestamp": 1639030055490, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9d96f209-11bf-4b4c-c096-00ecf7b37039"
# Split train, test set
x_train, x_test = train_test_split(df, test_size=0.05)

# dRecommend n most popular movies on a dataset
def popular_movie(dataset, n):
    movie_rank = dataset['MovieID'].value_counts()[:n]
    popular_movies = dataset.iloc[movie_rank.index]
    return popular_movies

# Calculate hitrate@K
def hitrate(K):
    raw_ranking = popular_movie(df, K)
    pred_ranking = popular_movie(x_test, K)
    return raw_ranking['MovieID'].isin(pred_ranking['MovieID']).value_counts(normalize=True)[True]

hitrate(100)
```

```python id="i7LBzn1ozbTi"
# Accuracy calculation
def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

rmse = []
for user in set(df.index):
    y_true = df.loc[user]['Rating']
    y_pred = movie_mean[df.loc[user]['MovieID']]
    accuracy = RMSE(y_true, y_pred)
    rmse.append(accuracy)
print(np.mean(rmse))
```

```python id="Cscb1IJ1zhn3"
# Split train, test set
x = df.copy()
y = df['UserID']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.05,stratify=y)

# Accuracy calculation
def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

# Calculate RMSE by model
def score(model):
    id_pairs = zip(x_test['UserID'], x_test['MovieID'])
    y_pred = np.array([model(user,movie) for (user,movie) in id_pairs])
    y_true = np.array(x_test['Rating'])
    return RMSE(y_true, y_pred)

# Get full matrix with training df
rating_matrix = x_train.pivot(index='UserID', columns='MovieID', values='Rating')

# The default model for calculating forecasts by the overall mean
def best_seller(user_id, movie_id):
    try:
        rating = train_mean[movie_id]
    except:
        rating = 3.0
    return rating

train_mean = x_train.groupby(['MovieID'])['Rating'].mean()
score(best_seller)
```

<!-- #region id="emC267Aw0-ZK" -->
## Recommend Genre-wise Popular Movies
<!-- #endregion -->

```python id="cP9S3NS-1mp8"
df = df_raw.copy()
```

<!-- #region id="94ac4HRM1GHW" -->
Data preprocessing (extracting data only from users who have watched more than 50 movies)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 272} id="8iv3B9bs1CdO" executionInfo={"status": "ok", "timestamp": 1630155524770, "user_tz": -330, "elapsed": 758, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0a0c9584-fd5f-476a-d7bc-07d354f1bb0a"
count = df['UserID'].value_counts()
count_index = count[count > 50]
data_pre = df[df['UserID'].isin(count_index)]
data_pre.head()
```

```python id="pavGGPUb1MYR"
x_train, x_test = train_test_split(data_pre, test_size=0.05)
```

<!-- #region id="vZ1ZezYU13yN" -->
Top-10 most-watched movies for the given genre
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 665} id="E7QZyPwN1VmK" executionInfo={"status": "ok", "timestamp": 1630155532129, "user_tz": -330, "elapsed": 1985, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7a5d18e5-51be-4056-8189-2442a25b45f3"
def genre_pop(dataset, genre, n):
    dataset_genres = dataset['Genres'].str.get_dummies("|")
    select_genre = dataset_genres[dataset_genres[genre]==1]
    genre_popmovie = dataset.loc[select_genre.index]
    genre_popmovie = genre_popmovie.reset_index()
    genre_popmovie_rank = genre_popmovie['MovieID'].value_counts()[:n]
    recomm_movie = genre_popmovie.loc[genre_popmovie_rank.index]
    return recomm_movie

genre_pop(x_train, 'Comedy', 10)
```

<!-- #region id="7beDLdyr1Wgy" -->
User's genre ranking in the training set
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="gvPKFhkS2GFo" executionInfo={"status": "ok", "timestamp": 1630155631792, "user_tz": -330, "elapsed": 520, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="34c386ee-5dd5-45bb-f78a-ae483dbd7ba8"
def user_genre(dataset, userid):
    user_data = dataset[dataset['UserID']==userid]
    user_data_genres = user_data['Genres'].str.get_dummies("|")
    user_genre_ranking = user_data_genres.sum().sort_values(ascending=False)
    return user_genre_ranking

user_genre(x_train, 54)
```

<!-- #region id="yRSmOnzY2OZ6" -->
Recommend by genre
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 255} id="9jHEptFJ2RAW" executionInfo={"status": "ok", "timestamp": 1630155677945, "user_tz": -330, "elapsed": 2682, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fcbf1901-5a8f-4ebb-9bf0-044d3cd3680c"
def user_genre_recommend(dataset, userid, n):
    genre_pref = user_genre(dataset, userid)
    recomm = genre_pop(dataset, genre_pref.index[0], n)
    return recomm

user_genre_recommend(x_train, 54, 5)
```

<!-- #region id="SuzCJPL42ZO_" -->
Evaluation
<!-- #endregion -->

<!-- #region id="WtlW0OJ62aas" -->
Ignoring genre
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="I_VLJr462dMi" executionInfo={"status": "ok", "timestamp": 1630157569004, "user_tz": -330, "elapsed": 1149, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7d6244f7-db91-452e-9e4d-77755d854659"
def popular_movie(dataset, n):
    movie_rank = dataset['MovieID'].value_counts()[:n]
    popular_movies = dataset.iloc[movie_rank.index]
    return popular_movies

def hitrate1(K):
    raw_ranking = popular_movie(df, K)
    pred_ranking = popular_movie(x_test, K)
    return raw_ranking['MovieID'].isin(pred_ranking['MovieID']).value_counts(normalize=True)[True]

s = 0
for i in range (100):
    s += hitrate1(100)
s /= 100
s
```

<!-- #region id="nMpUYkdQ2evo" -->
Considering genre
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="B2xUHjRg2xWQ" executionInfo={"status": "ok", "timestamp": 1630158240486, "user_tz": -330, "elapsed": 377175, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2ceeb35c-627e-4633-abc9-58e47ecc7f4e"
def user_genre_recommend(dataset, userid, n):
    genre_pref = user_genre(dataset, userid)
    recomm = genre_pop(dataset, genre_pref.index[0], n)
    return recomm

def hitrate2(K):
    user = x_train.sample(n=1)['UserID'].values[0]
    raw_recomm = user_genre_recommend(data_pre, user, K)
    pred_recomm = user_genre_recommend(x_train, user, K)
    return raw_recomm['MovieID'].isin(pred_recomm['MovieID']).value_counts(normalize=True)[True]

s = 0
count = 0
while count!=100:
    try:
        _s = hitrate2(100)
        count+=1
    except:
        pass
    if count%10==0:
        print(count)
    s += _s
s /= 100
s
```

<!-- #region id="--pOiaTD_GOv" -->
## Recommend Popular movies by user occupation and age
<!-- #endregion -->

```python id="jM4Z9wPE05Km"
df = df_raw.copy()

# drop columns
df = df.drop(['Timestamp','Gender','Zip-code'], axis=1)

# leave only data with a rating of 3 or higher
data_pre = df[df['Rating'] > 2]
```

```python id="X0rucNWx08KC"
x_train, x_test = train_test_split(data_pre, test_size=0.05)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="KCuNZY26_bAt" executionInfo={"status": "ok", "timestamp": 1630158058729, "user_tz": -330, "elapsed": 967, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ca169ded-3f82-4d49-c9af-0af2b59cb012"
# Recommend n popular movies by occupation
def occu_pop(dataset, occu, n):
    data_occu = dataset[dataset['Occupation'] == occu]
    data_occu = data_occu.reset_index()
    occu_pop_rank = data_occu['MovieID'].value_counts()[:n]
    recommend_movies = data_occu.loc[occu_pop_rank.index]
    return recommend_movies

# Recommend n movies depending on user's occupation
def user_occu_recommend(dataset, userid, n):
    user_occu = dataset[dataset['UserID'] == userid]['Occupation'].values[0]
    recomm = occu_pop(dataset, user_occu, n)
    return recomm
    
user_occu_recommend(x_train, 46, 5)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="wXpC7oQG_eyl" executionInfo={"status": "ok", "timestamp": 1630158089448, "user_tz": -330, "elapsed": 883, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="884637c3-af6d-4e0c-8c90-9e74b9e63654"
# Recommend n popular movies by age group
def age_pop(dataset, age, n):
    data_age = dataset[dataset['Age'] == age]
    data_age = data_age.reset_index()
    age_pop_rank = data_age['MovieID'].value_counts()[:n]
    recommend_movies = data_age.loc[age_pop_rank.index]
    return recommend_movies

# Recommend n movies based on user's age
def user_age_recommend(dataset, userid, n):
    user_age = dataset[dataset['UserID'] == userid]['Age'].values[0]
    recomm = age_pop(dataset, user_age, n)
    return recomm

user_age_recommend(x_train, 46, 5)
```

<!-- #region id="KV94gx1P_mYO" -->
Evaluation

Against Popularity baseline
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 83, "referenced_widgets": ["2cd0a08aa8874ccba9ed7a620ff126f7", "e47575e47f2d48beb18a526a9a29fd86", "109bf2c4a72d45fd8b32c9c7054cd689", "9291b2228d874ddbbef72052a322d3aa", "8b28e746b5ca4cdbb3a804f4a450111f", "48606b2370024685bf118fba84ca3a75", "ebad54c8bbba40299fdd784588876c64", "dbb95cb5ea5e45f08a81c6d81e703b94", "c2c8be8f08e344959383284b48834b89", "8fcb107f4f7e44c69493b2fb7b5fd0c1", "36a37abe4b2145f5966a03f0d0c323b5"]} id="MkCy5CVT_n6w" executionInfo={"status": "ok", "timestamp": 1630162834524, "user_tz": -330, "elapsed": 848485, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4eec2bc0-99f8-4711-93f8-e5e96fb1ccab"
def popular_movie(dataset, n):
    data_pop = dataset.copy()
    movie_rank = data_pop['MovieID'].value_counts()[:n]
    return movie_rank

def hitrate1(K):
    raw_ranking = popular_movie(data_pre, K)
    pred_ranking = popular_movie(x_test, K)
    return pd.DataFrame(raw_ranking.index.isin(pred_ranking.index)).value_counts(normalize=True)[True]

s = 0
for i in tqdm(range(len(x_test.index))):
    s += hitrate1(100)
s /= len(x_test.index)
s
```

```python colab={"base_uri": "https://localhost:8080/", "height": 83, "referenced_widgets": ["1d93d22c0997400d9a131eaeddda3fcc", "cd55996c6ccd4b44b6c9086588aadc4b", "94a40cbd77054fe2b16a5165784c871f", "13bea89fdf9c4ac4b919680bc449a3a4", "5a709d16124d4b148c7d7b53f176a377", "6e3d67a851b14b73aa9cdb6b37f397d8", "6b6f013e0a5741a7a8e80be2062015a0", "fbd8372847fd4813a3511c73509acedc", "8dadfeb102124fb1bb6edf8eb19ceb01", "439eda0a39bc4698a620b13d3420955d", "198285cc07e54d2188a50856e639c8f5"]} id="QebIBzON_y4r" executionInfo={"status": "ok", "timestamp": 1630159053287, "user_tz": -330, "elapsed": 660481, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5d72dabb-ea2f-4dd5-bcf4-449b258b3128"
def occu_pop(dataset, occu, n):
    data_occu = dataset[dataset['Occupation'] == occu]
    occu_pop_rank = data_occu['MovieID'].value_counts()[:n]
    return occu_pop_rank

def user_occu_recommend(dataset, userid, n):
    user_occu = dataset[dataset['UserID'] == userid]['Occupation'].values[0]
    recomm = occu_pop(dataset, user_occu, n)
    return recomm

def hitrate2(user, K):
    raw_recomm = user_occu_recommend(data_pre, user, K)
    pred_recomm = user_occu_recommend(x_test, user, K)
    return pd.DataFrame(raw_recomm.index.isin(pred_recomm.index)).value_counts(normalize=True)[True]

s = 0
for i in tqdm(x_test['UserID'].index):
    s += hitrate2(x_test['UserID'][i], 100)
s /= len(x_test.index)
s
```

```python colab={"base_uri": "https://localhost:8080/", "height": 83, "referenced_widgets": ["49882314eac44be7b131a68e86e92927", "0bb9d1b66013481aa1d61c2c73b836f0", "6848963ca083432097cc177512408147", "535d60ae572f41b0a95b04b633336ce7", "13796c34226d40e2b6108837126b891d", "f432e220b999418e971f4cbd4950657a", "c8ee7cdb6cb3423bb67e104f1060c35a", "adb2e8da8b89406e96bd4ce520c2adcd", "ade45857baf64618a1a853a6addb7a0b", "149455b7acd54506aeecf5ceba4c3f74", "bcf482338b70489db9594a864e6b6e92"]} id="zjEHWwz1_30K" executionInfo={"status": "ok", "timestamp": 1630161414008, "user_tz": -330, "elapsed": 2352388, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0d9b3a8b-9760-45dd-9407-8efcc11c270f"
def age_pop(dataset, age, n):
    data_age = dataset[dataset['Age'] == age]
    data_age = data_age.reset_index()
    age_pop_rank = data_age['MovieID'].value_counts()[:n]
    return age_pop_rank

def user_age_recommend(dataset, userid, n):
    user_age = dataset[dataset['UserID'] == userid]['Age'].values[0]
    recomm = age_pop(dataset, user_age, n)
    return recomm

def hitrate3(user, K):
    raw_recomm = user_age_recommend(data_pre, user, K)
    pred_recomm = user_age_recommend(x_train, user, K)
    return pd.DataFrame(raw_recomm.index.isin(pred_recomm.index)).value_counts(normalize=True)[True]

s = 0
for i in tqdm(x_test['UserID'].index):
    s += hitrate3(x_test['UserID'][i], 100)
s /= len(x_test.index)
s
```

<!-- #region id="Ewa4TIIf1LSe" -->
## Autoencoder-based Model
<!-- #endregion -->

```python id="SqSIjz1T1NpK"
df = df_ratings.copy()
df.columns = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
```

```python id="ugBiqIpiB5st"
df = df.drop('unix_timestamp', 1)

input_df = pd.DataFrame(index=range(1,max(df['user_id'])+1), columns=range(1,max(df['movie_id'])+1))

for index,row in df.iterrows():
    input_df[row['movie_id']][row['user_id']]=row['rating']

print(input_df.shape)
input_df
```

```python colab={"base_uri": "https://localhost:8080/"} id="V5GrFhjmCV7w" executionInfo={"status": "ok", "timestamp": 1630159085828, "user_tz": -330, "elapsed": 526, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c9f1f48f-0e85-4e33-ca17-c601d5a6fa24"
input_df = input_df.truncate(after=64, axis=1)
input_df = input_df.dropna(axis=0, how='all')
mean_col = input_df.mean(axis=1)
input_df.shape
```

<!-- #region id="MwULNSpWDs96" -->
### User Similarity
<!-- #endregion -->

```python id="ZS0xlKUeDCzG"
def user_similarity(a,b):
    if (not a in input_df.index or not b in input_df.index):
        return np.nan
    cov = 0.0
    var_a = 0.0
    var_b = 0.0
    for column in input_df:
        avg_rating_a = mean_col[a]
        avg_rating_b = mean_col[b]
        j_rating_a = input_df[column][a]
        j_rating_b = input_df[column][b]
        
        if (not np.isnan(j_rating_a) and not np.isnan(j_rating_b)):
            cov = cov + (j_rating_a - avg_rating_a) * (j_rating_b - avg_rating_b)
            var_a = var_a + (j_rating_a - avg_rating_a) * (j_rating_a - avg_rating_a)
            var_b = var_b + (j_rating_b - avg_rating_b) * (j_rating_b - avg_rating_b)
        if (var_a == 0 or var_b == 0):
            return 0
        return (cov/(math.sqrt(var_a*var_b)))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 168, "referenced_widgets": ["55e1acdf6b4442bda6e21f4816edacaf", "ddaf385d826c43c89207c2c4b9ab5c0f", "a236d5fe10944da491cee35dadf6ce34", "38189709ad334d1399edc9dbb3315975", "e850d6a06aa745c5906a2f3deb2739dc", "59423e1d2c4f486c9a3d0eed4d90f02e", "19d45bc6519a4ae6b9b23620fd9e3b6e", "f6c3d94e650244979f974e593c6a907d", "a06659608ffc4e98bd29a337067eedd4", "4f90565fc8974adb8de9d49beefcd966", "3884e7b460b248e48620b983fd7a4873"]} id="-xHy5LnPDa22" executionInfo={"status": "ok", "timestamp": 1630159584754, "user_tz": -330, "elapsed": 467130, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9c3bb88e-e988-45e8-87c6-ba07d5c1912a"
sim = np.zeros(shape=(max(df['user_id']), max(df['user_id'])))
num_of_users = max(df['user_id'])
it = 0

for i in tqdm(range(num_of_users)):
    for j in range(i+1):
        sim[i][j] = user_similarity(i+1, j+1)
        sim[j][i] = sim[i][j]
sim
```

<!-- #region id="mcl_NZNjDbtQ" -->
### Column Rating
<!-- #endregion -->

```python id="bpkNHhFoDxTU"
def round_off_rating(val):
    new_val = int(val)
    frac = val - int(val)
    if (frac >= 0.75):
        new_val = new_val + 1
    elif (frac >= 0.25):
        new_val = new_val + 0.5
    return max(min(new_val, 5.0), 1)

def predict_column_rating(column_no):
    temp = input_df[input_df[column_no].notnull()][column_no]
    for index, null_rating in input_df[column_no].iteritems():
        num_sum = 0
        den_sum = 0
        if (np.isnan(null_rating)):
            for i,rating in temp.iteritems():
                num_sum = num_sum + sim[index-1][i-1] * (rating - mean_col[i])
                den_sum = den_sum + sim[index-1][i-1]
            if (den_sum == 0):
                input_df[column_no][index] = round_off_rating(mean_col[index])
            else:
                input_df[column_no][index] = round_off_rating(mean_col[index] + num_sum/den_sum)
```

```python id="LmGagn0SDzYj"
for column_no in input_df:
    predict_column_rating(column_no)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 439} id="-e2PhfNzMtgX" executionInfo={"status": "ok", "timestamp": 1630161527521, "user_tz": -330, "elapsed": 653, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6234f81e-fe15-4c7e-8c65-a44376984be0"
input_df
```

<!-- #region id="MY28CE-BEDK1" -->
Dimensionality Reduction using Autoencoders
<!-- #endregion -->

```python id="TW-nb2BPFJ7M"
encoding_dim1 = 16
encoding_dim2 = 5

input_rating = Input(shape=(64,))
encoded = Dense(16, activation='relu')(input_rating)  # 64->16
encoded = Dense(5, activation='relu')(encoded)        # 16->05
decoded = Dense(16, activation='relu')(encoded)       # 05->16
decoded = Dense(64, activation='sigmoid')(decoded)    # 16->64

autoencoder = Model(input_rating, decoded)
encoder1 = Model(input_rating, autoencoder.layers[1](input_rating))

input_encoding = Input(shape=(encoding_dim1,))
encoder2 = Model(input_encoding, autoencoder.layers[2](input_encoding))
encoded_input1 = Input(shape=(encoding_dim2,))
encoded_input2 = Input(shape=(encoding_dim1,))

decoder_layer1 = autoencoder.layers[-2]
decoder_layer2 = autoencoder.layers[-1]
decoder1 = Model(encoded_input1, decoder_layer1(encoded_input1))
decoder2 = Model(encoded_input2, decoder_layer2(encoded_input2))
```

```python id="E_bNp_9pGBW9"
autoencoder.compile(optimizer='adam', loss='mse')
```

```python id="W4H2pXrFGFFA" colab={"base_uri": "https://localhost:8080/", "height": 439} executionInfo={"status": "ok", "timestamp": 1630161541306, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="21865b66-bae9-46ba-e8f4-657c1d40c125"
input_df = input_df/5
input_df
```

```python id="8jhND0V3GGuB" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1630161541308, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="006d8cc8-ceab-478e-885d-c8094448f23e"
x_train = input_df.sample(frac=0.8, random_state=200).astype(float)
x_test = input_df.drop(x_train.index).astype(float)
x_train.shape
```

```python id="eDbRBgp8GINs" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1630161563771, "user_tz": -330, "elapsed": 21802, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="680d580c-8ce4-4e47-9090-de997beccd31"
#collapse-hide
autoencoder.fit(x_train, x_train, epochs=100, batch_size=100, shuffle=True, validation_data=(x_test,x_test))
```

```python id="uznUG75dGKpo" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1630161564735, "user_tz": -330, "elapsed": 973, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f1c32560-609a-405b-bdc5-433ebfa94a95"
encoded_output1 = encoder1.predict(input_df.astype(float))
encoded_output2 = encoder2.predict(encoded_output1)
decoded_output1 = decoder1.predict(encoded_output2)
decoded_output2 = decoder2.predict(decoded_output1)
encoded_output2
```

```python id="ZaBZlZdOGMh-" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1630161564736, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b46e88fd-4dc0-46b7-b2bf-7d68b77c70f6"
decoded_output2
```

```python id="1nUwDmGpGOMb" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1630161567286, "user_tz": -330, "elapsed": 2558, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="423d28c7-e7c4-4487-c8fe-a40db6018157"
ans = decoded_output2 * 5
for (x,y), value in np.ndenumerate(ans):
    ans[x][y] = round_off_rating(ans[x][y])
ans
```

```python id="N7W-vwpHGPnQ" colab={"base_uri": "https://localhost:8080/", "height": 439} executionInfo={"status": "ok", "timestamp": 1630161567289, "user_tz": -330, "elapsed": 37, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c3da9253-cbee-41fd-ff4a-e19949fc3a1e"
ans_df = pd.DataFrame(ans)
df = input_df.copy()
df = df * 5
ans_df
```

```python id="CYngQcO9GR5I" colab={"base_uri": "https://localhost:8080/", "height": 326} executionInfo={"status": "ok", "timestamp": 1630161567291, "user_tz": -330, "elapsed": 34, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9e483d95-74b8-4b63-f715-4920dcc5efb3"
plt.figure(figsize=(10,10))
plt.subplot(211)
line1, = plt.plot(range(1,65), df.iloc[6], 'b')
line2, = plt.plot(range(1,65), ans_df.iloc[3], 'k')
plt.ylabel('ratings')
plt.xlabel('movie ids')
plt.legend([line1, line2], ['Initial Filtered', 'Predicted AE'])
```

```python id="C_L2Z7-PGT6-" colab={"base_uri": "https://localhost:8080/", "height": 594} executionInfo={"status": "ok", "timestamp": 1630161568740, "user_tz": -330, "elapsed": 1470, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="44784d2a-ed59-432e-b45a-266781733c30"
plt.figure(figsize=(50,50))
plt.subplot(212)
line1, = plt.plot(range(1,4812), df[1].tolist(), 'bo')
line2, = plt.plot(range(1,4812), ans_df[0].tolist(), 'ko')
plt.ylabel('ratings')
plt.xlabel('movie ids')
plt.legend([line1, line2], ['Initial Filtered', 'Predicted AE'])
```

<!-- #region id="okSRnAAa3C1h" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="e0yZVSfA3C1i" executionInfo={"status": "ok", "timestamp": 1639031012561, "user_tz": -330, "elapsed": 3342, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e4bd5bc3-8eb5-40f2-933e-649be381b6a2"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d -p tensorflow
```

<!-- #region id="4fXtJXo_3C1j" -->
---
<!-- #endregion -->

<!-- #region id="N_GGXd7u3C1j" -->
**END**
<!-- #endregion -->
