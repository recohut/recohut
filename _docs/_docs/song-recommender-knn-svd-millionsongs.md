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

<!-- #region id="v7BNK0OSzqmy" -->
# Recommending songs using collaborative filtering method
> KNN and SVD models with hyperparameter tuning with the help of Surprise library. Using small subset of Million Songs Dataset.

- toc: true
- badges: true
- comments: true
- categories: [Surprise, KNN, SVD, Songs, MillionSongsDataset, Visualization, EDA]
- author: "<a href='https://github.com/ugis22/music_recommender'>Eugenia Inzaugarat</a>"
- image:
<!-- #endregion -->

<!-- #region id="Q2SPYzUO0SCl" -->
## Setup
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="sAw7uQm3vRZF" outputId="586c5cb4-5aea-4233-fa74-80344abeeedf"
!pip install -q fuzzywuzzy
!pip install scikit-surprise
```

```python id="G7CHlsq8ZlqI"
import numpy as np
import pandas as pd

from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz

from surprise import SVD
from surprise import Dataset, Reader
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV
from surprise.model_selection import cross_validate

import warnings
warnings.filterwarnings('ignore')
```

<!-- #region id="Orws1WfvZ2ON" -->
## Load dataset
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="_Y5v8o-9Z3gF" outputId="26c1d3bc-02b5-47a3-f79c-d5081ba2db05"
songs_meta = pd.read_parquet('https://github.com/recohut/reco-data/raw/master/millionsongs/v1/items.parquet.gzip')
songs_meta.drop_duplicates(subset='song_id', inplace=True)
songs_meta.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="hWOQlDY8aLFM" outputId="de214f49-393a-4182-eca1-31c764ff0dad"
interactions = pd.read_parquet('https://github.com/recohut/reco-data/raw/master/millionsongs/v1/interactions.parquet.gzip')
interactions.drop_duplicates(subset=['song_id','user_id'], inplace=True, keep='last')
interactions.head()
```

<!-- #region id="PMwAgHPHauap" -->
## EDA and cleaning
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="J2CQXF1BckDj" outputId="e890c16f-b76d-4b51-a0b1-2a09442c9d54"
# add the listen count from interactions to the songs_meta
listen_counts = interactions.groupby('song_id', as_index=False)['listen_count'].sum()
songs_meta = pd.merge(songs_meta, listen_counts, on="song_id", how="left")
songs_meta.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="PZr_uY-waxyI" outputId="06da3ae3-b148-4b20-a105-db568a952b61"
songs_meta.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="4FtmUC4EdP-6" outputId="4d3c44be-0f12-438f-965e-bb4f548e5c07"
songs_meta.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 297} id="9ionMHEoawN8" outputId="a5374821-afd7-43af-a8db-2f4a8bdf588e"
songs_meta.describe()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 173} id="pGqG4CW5gBr9" outputId="1cb0f4d5-e5f4-4f5b-9401-6f9deaac2f37"
songs_meta.describe(include=['O'])
```

```python colab={"base_uri": "https://localhost:8080/"} id="7hZt2aPSdOzi" outputId="8860827f-eeb2-40b9-b74b-642167dc35f9"
songs_meta.isnull().sum()
```

```python colab={"base_uri": "https://localhost:8080/"} id="JnQr9cyQd7mx" outputId="78985605-16f2-4aea-b903-434f874040c4"
# replacing NA titles with 'Unknown' string
songs_meta['title'].fillna("Unknown", inplace=True)

# replacing NA release with 'Unknown' string
songs_meta['release'].fillna("Unknown", inplace=True)

# replacing NA listen counts with 0 integer
songs_meta['listen_count'].fillna(0, inplace=True)

# rechecking the status of missing values
songs_meta.isnull().sum()
```

<!-- #region id="C0g6HqfViJCc" -->
### Most popular songs
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 444} id="WarJtN70hF5P" outputId="2ca4e117-0224-4798-eff5-3cdef0c5db33"
ten_pop_songs = songs_meta.sort_values(['listen_count', 'title'], ascending = [0,1])
ten_pop_songs['percentage']  = round(ten_pop_songs['listen_count'].div(ten_pop_songs['listen_count'].sum())*100, 2)
ten_pop_songs = ten_pop_songs[:10]
ten_pop_songs
```

```python colab={"base_uri": "https://localhost:8080/", "height": 265} id="qHA9lb1dh-gS" outputId="0006add4-d089-44b9-ba59-1dcd43c1ddf0"
labels = ten_pop_songs['title'].tolist()
counts = ten_pop_songs['listen_count'].tolist()

plt.figure()
sns.barplot(x=counts, y=labels, palette='Set3')
sns.despine(left=True, bottom=True)
```

<!-- #region id="GZB43l21iM4k" -->
### Most popular artist
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 359} id="6oF7xQrliHuL" outputId="614fea19-c540-4120-c563-fb731decc9c1"
ten_pop_artists  = songs_meta.groupby(['artist_name'])['listen_count'].sum().reset_index().sort_values(['listen_count', 'artist_name'], ascending = [0,1])
ten_pop_artists = ten_pop_artists[:10]
ten_pop_artists
```

```python colab={"base_uri": "https://localhost:8080/", "height": 265} id="rkEUrHJDifUt" outputId="b9dc16ac-b91e-4964-b206-819a83f48add"
plt.figure()
labels = ten_pop_artists['artist_name'].tolist()
counts = ten_pop_artists['listen_count'].tolist()
sns.barplot(x=counts, y=labels, palette='Set2')
sns.despine(left=True, bottom=True)
```

<!-- #region id="L628NfwamWX6" -->
### Listen count by user
<!-- #endregion -->

<!-- #region id="O4gF0Y-emblM" -->
What was the maximum time the same user listen to a same song?
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 80} id="xzeZrsW9iaH2" outputId="77b4f095-a1a4-480c-c601-2451c26b6a04"
interactions[interactions.listen_count==interactions.listen_count.max()]
```

<!-- #region id="D_YAXXp7nUcQ" -->
How many times on average the same user listen to a same song?
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="6feK_yIgmwCt" outputId="48527ed7-8d1c-4553-ac2a-5d142049ab5b"
interactions.listen_count.mean()
```

<!-- #region id="MiPWLNaCngkV" -->
We can also check the distribution of listen_count:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 282} id="3yfvlpyknc1x" outputId="4ff72e1b-503c-41a9-8084-5e44a236fcf2"
plt.figure(figsize=(20, 5))
sns.boxplot(x='listen_count', data=interactions)
sns.despine()
```

<!-- #region id="2OieQ34cnyJF" -->
How many songs does a user listen in average?
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 477} id="IDGGgm3dnnW_" outputId="4e0f286e-c466-4fe0-e6e6-a0469090b41a"
song_user = interactions.groupby('user_id')['song_id'].count()

plt.figure(figsize=(16, 8))
sns.distplot(song_user.values, color='orange')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show();
```

<!-- #region id="-2W9SVUxoOmB" -->
### Minimum interaction filter
Get users which have listened to at least 16 songs, and filter the dataset to keep only those users with more than 16 listened
<!-- #endregion -->

```python id="tI5kticBn30h"
filter_song_ids = song_user[song_user > 16].index.to_list()
filter_song_df = interactions[interactions['user_id'].isin(filter_song_ids)].reset_index(drop=True)

# convert the dataframe into a pivot table
df_songs_features = filter_song_df.pivot(index='song_id', columns='user_id', values='listen_count').fillna(0)

# obtain a sparse matrix
mat_songs_features = csr_matrix(df_songs_features.values)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 410} id="60VfAM-KpMQP" outputId="69022d7d-cf42-4727-dd2e-60149debcb98"
df_songs_features.iloc[:10, :10].T
```

```python colab={"base_uri": "https://localhost:8080/", "height": 289} id="vWfI0OOgpQMm" outputId="81f331eb-2a7e-4685-e3ad-49b55ccb5546"
plt.figure(figsize=(25, 25))
plt.spy(mat_songs_features, markersize=0.005);
```

<!-- #region id="SbD9PTWJu1eP" -->
Because the system will output the id of the song, instead of the title, we'll make a function that maps those indices with the song title.
<!-- #endregion -->

```python id="35ZXG9cDrpBF"
df_unique_songs = songs_meta.drop_duplicates(subset=['song_id']).reset_index(drop=True)[['song_id', 'title']]
decode_id_song = {
    song: i for i, song in 
    enumerate(list(df_unique_songs.set_index('song_id').loc[df_songs_features.index].title))
}
```

<!-- #region id="PQnq289evDw2" -->
## KNN Recommender
<!-- #endregion -->

```python id="o3T5z3hqvE2z"
class KNNRecommender:
    def __init__(self, metric, algorithm, k, data, decode_id_song):
        self.metric = metric
        self.algorithm = algorithm
        self.k = k
        self.data = data
        self.decode_id_song = decode_id_song
        self.data = data
        self.model = self._recommender().fit(data)
    
    def make_recommendation(self, new_song, n_recommendations):
        recommended = self._recommend(new_song=new_song, n_recommendations=n_recommendations)
        return recommended 
    
    def _recommender(self):
        return NearestNeighbors(metric=self.metric, algorithm=self.algorithm, n_neighbors=self.k, n_jobs=-1)
    
    def _recommend(self, new_song, n_recommendations):
        # Get the id of the recommended songs
        recommendations = []
        recommendation_ids = self._get_recommendations(new_song=new_song, n_recommendations=n_recommendations)
        # return the name of the song using a mapping dictionary
        recommendations_map = self._map_indeces_to_song_title(recommendation_ids)
        # Translate this recommendations into the ranking of song titles recommended
        for i, (idx, dist) in enumerate(recommendation_ids):
            recommendations.append(recommendations_map[idx])
        return recommendations
                 
    def _get_recommendations(self, new_song, n_recommendations):
        # Get the id of the song according to the text
        recom_song_id = self._fuzzy_matching(song=new_song)
        # Start the recommendation process
        # Return the n neighbors for the song id
        distances, indices = self.model.kneighbors(self.data[recom_song_id], n_neighbors=n_recommendations+1)
        return sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
    
    def _map_indeces_to_song_title(self, recommendation_ids):
        # get reverse mapper
        return {song_id: song_title for song_title, song_id in self.decode_id_song.items()}
    
    def _fuzzy_matching(self, song):
        match_tuple = []
        # get match
        for title, idx in self.decode_id_song.items():
            ratio = fuzz.ratio(title.lower(), song.lower())
            if ratio >= 60:
                match_tuple.append((title, idx, ratio))
        # sort
        match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
        if not match_tuple:
            print(f"The recommendation system could not find a match for {song}")
            return
        return match_tuple[0][1]
```

```python id="oFPCjWppvWI_"
model = KNNRecommender(metric='cosine', algorithm='brute', k=20, data=mat_songs_features, decode_id_song=decode_id_song)
```

```python colab={"base_uri": "https://localhost:8080/"} id="SfKrUeKFvbW-" outputId="944c2fc1-7834-4250-cab0-ccb1c9a5a591"
song = 'I believe in miracles'

new_recommendations = model.make_recommendation(new_song=song, n_recommendations=10)
print(f"The recommendations for {song} are:\n")
new_recommendations
```

<!-- #region id="V0192dq_wLzS" -->
## Matrix factorization recommender
<!-- #endregion -->

<!-- #region id="Bn36gELcw23B" -->
### Filtering
<!-- #endregion -->

<!-- #region id="oI2BPTUFw5Tb" -->
So we already know that it is a very sparse matrix. Dealing with such a sparse matrix, we'll take a lot of memory and resources. To make our life easier, we aalready selected users that have listened to at least 16 songs. Moreover, we are going to select only those songs which have been listened to by at least 200 users.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="FQVpYxiUw4cx" outputId="dad43036-bf06-4a3a-d75c-a2da81e82781"
song_counts = interactions.groupby('song_id')['user_id'].count()
filter_song_count_ids = song_counts[song_counts > 200].index.to_list()
filter_song_df = interactions[interactions['song_id'].isin(filter_song_count_ids)].reset_index(drop=True)
filter_song_df.shape
```

<!-- #region id="NJhgNkmZyfuI" -->
### Binning to convert implicit signals into target prediction
<!-- #endregion -->

<!-- #region id="qhI9gym5wnq8" -->
Instead of working with the implicit rating as it is, we'll apply the binning technique. We'll define 10 categories. The original data values which fall into the interval from 0 to 1, will be replaced by the representative rating of 1; if they fall into the interval 1 to 2, they will be replaced by 2; and so on and so forth. The last category will be assigned to original values ranging from 9 to 2213.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 487} id="v7bpom4Tvk3M" outputId="3314db2e-eadb-43e4-8777-e041056652cd"
bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2214]
filter_song_df['listen_count'] = pd.cut(filter_song_df['listen_count'], bins=bins, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
listen_counts = pd.DataFrame(filter_song_df.groupby('listen_count').size(), columns=['count']).reset_index(drop=False)

plt.figure(figsize=(16, 8))
sns.barplot(x='listen_count', y='count', palette='Set3', data=listen_counts)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show();
```

<!-- #region id="iwxKpqw0y0Oo" -->
### Load data into Surprise format
<!-- #endregion -->

<!-- #region id="Y1tLNA30y3OY" -->
To load a dataset from our DataFrame, we will use the load_from_df() method.

We will need to pass the following parameters:

- df: The dataframe containing the ratings. It must have three columns, corresponding to the user ids, the song ids, and the ratings.
- reader (Reader): A reader to read the file. Only the rating_scale field needs to be specified.
<!-- #endregion -->

```python id="gHX2xjpeywgW"
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(filter_song_df[['user_id', 'song_id', 'listen_count']], reader)
trainset, testset = train_test_split(data, test_size=.25)
```

<!-- #region id="RLGXHUGZzEfb" -->
### Training and hypertuning
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="IBa14DKHzHrb" outputId="9f68c37c-b23e-450b-d6c7-140793cb44b9"
param_grid = {'n_factors': [160], 
              'n_epochs': [100], 
              'lr_all': [0.001, 0.005],
              'reg_all': [0.08]}

grid_search_svd = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=2, joblib_verbose=4, n_jobs=-2)

# find the best parameters for the model
grid_search_svd.fit(data)
find_algo = grid_search_svd.best_estimator['rmse']
print(grid_search_svd.best_score['rmse'])
print(grid_search_svd.best_params['rmse'])
```

<!-- #region id="mxHhw141zVwO" -->
### Cross validation
<!-- #endregion -->

```python id="w1UTP8SkzVI9"
cross_validate(find_algo, data, measures=['RMSE'], cv=5, verbose=True)
```

<!-- #region id="KuPGl4DXzen_" -->
### Evaluation
<!-- #endregion -->

<!-- #region id="m_7AMyHDzgg5" -->
After finding the best parameters for the model, we create our final model, train it and find the error for the test set.
<!-- #endregion -->

```python id="IahISu_dzirr"
final_algorithm = SVD(n_factors=160, n_epochs=100, lr_all=0.005, reg_all=0.1)
final_algorithm.fit(trainset)
test_predictions = final_algorithm.test(testset)
print(f"The RMSE is {accuracy.rmse(test_predictions, verbose=True)}")
```
