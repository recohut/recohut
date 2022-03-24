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

<!-- #region id="naUxW5XyMDBk" -->
# Spotify Music Recommender using KNN Model
<!-- #endregion -->

<!-- #region id="YbwDdN-RLmR7" -->
## Setup
<!-- #endregion -->

```python id="BN4TC-U9MZly"
import os
import numpy as np
import random
import pandas as pd
import collections

from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from sklearn import set_config
import seaborn as sns
import matplotlib.cm as cm

sns.set()
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)
```

<!-- #region id="FII1ij82LtOk" -->
## Data
<!-- #endregion -->

<!-- #region id="txuGyFOaLwSq" -->
### Ingestion
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="lFZlCkSEMJUF" executionInfo={"status": "ok", "timestamp": 1639472445285, "user_tz": -330, "elapsed": 4236, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3d3ca6b7-3d05-4c8b-df36-576f3e504f28"
!wget -q --show-progress https://github.com/RecoHut-Datasets/spotify/raw/v1/20210824_212829_artists.tsv
!wget -q --show-progress https://github.com/RecoHut-Datasets/spotify/raw/v1/20210824_212829_audios.tsv
!wget -q --show-progress https://github.com/RecoHut-Datasets/spotify/raw/v1/20210824_212829_playlists.tsv
!wget -q --show-progress https://github.com/RecoHut-Datasets/spotify/raw/v1/20210824_212829_tracks.tsv
```

```python colab={"base_uri": "https://localhost:8080/"} id="ELJl2kK_NG8y" executionInfo={"status": "ok", "timestamp": 1639472459836, "user_tz": -330, "elapsed": 1787, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="37037517-cadf-47c1-f26b-dccbedfe743f"
!wget -q --show-progress https://github.com/RecoHut-Datasets/spotify/raw/v1/des_artist.csv
!wget -q --show-progress https://github.com/RecoHut-Datasets/spotify/raw/v1/des_audio.csv
!wget -q --show-progress https://github.com/RecoHut-Datasets/spotify/raw/v1/des_playlist.csv
!wget -q --show-progress https://github.com/RecoHut-Datasets/spotify/raw/v1/des_tracks.csv
```

<!-- #region id="420dNiicMdYr" -->
### Loading
<!-- #endregion -->

```python id="oKOWv-XEMi0Y"
pd_artist = pd.read_csv('20210824_212829_artists.tsv', sep='\t')
pd_audio = pd.read_csv('20210824_212829_audios.tsv', sep='\t')
pd_playlist = pd.read_csv('20210824_212829_playlists.tsv', sep='\t')
pd_track = pd.read_csv('20210824_212829_tracks.tsv', sep='\t')
```

<!-- #region id="Tlsri1zkLxeQ" -->
### EDA
<!-- #endregion -->

<!-- #region id="6smeO8dbiX30" -->
#### Description
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="UgYrAtEKMehZ" executionInfo={"status": "ok", "timestamp": 1631537537418, "user_tz": -330, "elapsed": 28, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="289cff00-cbc2-43ee-fde9-510f86ffb8a8"
pd_track.columns.to_list()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 643} id="WkwvbblSOngf" executionInfo={"status": "ok", "timestamp": 1631537761915, "user_tz": -330, "elapsed": 427, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f4380954-8507-4855-bde0-9d8e2db0e474"
pd_track
```

```python colab={"base_uri": "https://localhost:8080/"} id="7Uny2zTQPB3G" executionInfo={"status": "ok", "timestamp": 1631537879048, "user_tz": -330, "elapsed": 725, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5b259cf1-ae14-4ee2-87e3-a93f3f00f998"
pd_track.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 498} id="qDtgCqoiNgXu" executionInfo={"status": "ok", "timestamp": 1631537582286, "user_tz": -330, "elapsed": 576, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="29195b5b-b232-49f9-fccc-40bd3b784a0a"
pd.read_csv('des_tracks.csv')
```

```python colab={"base_uri": "https://localhost:8080/"} id="wN5SAIYlNgVd" executionInfo={"status": "ok", "timestamp": 1631537640754, "user_tz": -330, "elapsed": 509, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4de7a349-fe9b-46ab-fa6c-8bc7f425437b"
pd_audio.columns.to_list()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 439} id="WdeR_r_IOsxs" executionInfo={"status": "ok", "timestamp": 1631537782415, "user_tz": -330, "elapsed": 476, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5383e02b-3426-4068-df3b-4504acf119b6"
pd_audio
```

```python colab={"base_uri": "https://localhost:8080/"} id="d1jQO94mPMyK" executionInfo={"status": "ok", "timestamp": 1631537913383, "user_tz": -330, "elapsed": 443, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="acac20a7-e4b2-40ac-9450-5114a95a218a"
pd_audio.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 945} id="RUlOQXHTNgSo" executionInfo={"status": "ok", "timestamp": 1631537642113, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6ecd7dbd-9980-428d-b871-65edb73fc1a9"
pd.read_csv('des_audio.csv')
```

```python colab={"base_uri": "https://localhost:8080/"} id="lKnhZ-ywNgPi" executionInfo={"status": "ok", "timestamp": 1631537666559, "user_tz": -330, "elapsed": 446, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e7635a7b-ddde-4530-89e8-0188cbb0ab67"
pd_playlist.columns.to_list()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 606} id="F12czSS-OxL2" executionInfo={"status": "ok", "timestamp": 1631537801086, "user_tz": -330, "elapsed": 467, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="39985713-9de6-4a22-e016-c18e0dc489de"
pd_playlist
```

```python colab={"base_uri": "https://localhost:8080/"} id="mgi7isUCPRjl" executionInfo={"status": "ok", "timestamp": 1631537933080, "user_tz": -330, "elapsed": 433, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b15c6e40-8983-4b5b-ee2c-76933dbe192f"
pd_playlist.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="A49Lf5uSNgNU" executionInfo={"status": "ok", "timestamp": 1631537673157, "user_tz": -330, "elapsed": 422, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4e71d9a9-c552-49c1-84c2-f4799d1b34bc"
pd.read_csv('des_playlist.csv')
```

```python colab={"base_uri": "https://localhost:8080/"} id="rUMYvwVsNgLV" executionInfo={"status": "ok", "timestamp": 1631537686019, "user_tz": -330, "elapsed": 513, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="63de6af5-a8d8-4beb-96f5-4274cfe5c497"
pd_artist.columns.to_list()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 470} id="SGeSe666O1fs" executionInfo={"status": "ok", "timestamp": 1631537816180, "user_tz": -330, "elapsed": 463, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="290368cf-f06c-4e04-fa2c-7ef0dd86ec63"
pd_artist
```

```python colab={"base_uri": "https://localhost:8080/"} id="LHNuInLKPVeG" executionInfo={"status": "ok", "timestamp": 1631541298904, "user_tz": -330, "elapsed": 425, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="65cd336b-7167-4a9b-8839-fafa9262730e"
pd_artist.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 272} id="-tUw1ISZNgIa" executionInfo={"status": "ok", "timestamp": 1631537690354, "user_tz": -330, "elapsed": 442, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b7c0a946-4d02-4c22-8aba-ec289baaf88b"
pd.read_csv('des_artist.csv')
```

<!-- #region id="T0Q3gv-8iheA" -->
#### Bi-variate
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 937} id="tzSKm5lwNgFv" executionInfo={"status": "ok", "timestamp": 1631543051684, "user_tz": -330, "elapsed": 61236, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b0622534-75c5-452c-f357-47e78624e44f"
pd.plotting.scatter_matrix(pd_audio, diagonal="kde", figsize=(20, 20))
plt.show()
```

<!-- #region id="PMwnRUi9LywZ" -->
#### Entity-Relationship Diagram
<!-- #endregion -->

<!-- #region id="QPMR7V-2i2_g" -->
<!-- #endregion -->

<!-- #region id="J2ERGdt9jB1u" -->
#### Merge
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 847} id="vDqDqBEUNgDB" executionInfo={"status": "ok", "timestamp": 1631543096852, "user_tz": -330, "elapsed": 481, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d002c359-44f9-47fa-ccf3-2a7040768749"
pd_track = pd.merge(pd_track, pd_audio, left_on = 'track_id', right_on = 'id')
pd_track
```

```python colab={"base_uri": "https://localhost:8080/"} id="fsNQQXhqNgA7" executionInfo={"status": "ok", "timestamp": 1631543223609, "user_tz": -330, "elapsed": 1086, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="dd3b772e-4722-46d6-983a-2ab9fa91d3de"
pd_full = pd.merge(pd_track, pd_playlist, left_on = 'playlist_id', right_on = 'playlist_id')
pd_full = pd_full.drop('playlist_name', axis = 1)
pd_full = pd_full.drop('playlist_description', axis = 1)
pd_full = pd_full.drop('id', axis = 1)
pd_full = pd_full.drop_duplicates()
pd_full = pd_full.fillna('None')
pd_full.info()
```

<!-- #region id="QUN31G9-Nf8q" -->
#### Correlation
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 862} id="TbnxZEoUjgW-" executionInfo={"status": "ok", "timestamp": 1631543237969, "user_tz": -330, "elapsed": 2930, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="32e3cc48-7466-43fe-f93d-09cedba90407"
names = list(pd_track.select_dtypes(np.number).columns)

data = pd_track[names]

corr = data.corr()
# plot correlation matrix
plt.figure(figsize=(15, 15))
ax = sns.heatmap(
    corr, annot=True,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
```

<!-- #region id="zSDCtmAnLt5h" -->
## Model
<!-- #endregion -->

<!-- #region id="UVQImHsJkEA_" -->
### Utils
<!-- #endregion -->

```python id="Lo-MujsDLu34"
def generate_playlist(pd_track, pd_playlist, num_playlist_to_test=100, threshold=30):
    playlist_sampled = pd_playlist[pd_playlist["playlist_num_tracks"] >= threshold]
    playlist_selected = playlist_sampled["playlist_id"].sample(n=num_playlist_to_test, random_state=0)
    track_ids = {}
    for list_id in playlist_selected:
        track_ids[list_id] = list(pd_track[pd_track['playlist_id'] == list_id]['track_id'])
    return track_ids


def generate_test_playlist(track_ids, missing_rate=0.2):
    track_id_for_test = {}
    for key in track_ids:
        all_tracks = track_ids[key]
        n = len(all_tracks)
        nums_songs_to_test = int(n * (1 - missing_rate))
        track_id_for_test[key] = all_tracks[:nums_songs_to_test]
    return track_id_for_test


def r_precision(prediction, label):
    """
    Calculate r-precision: union(p,l)/size
    :rtype: r-precision score
    """
    prediction = list(set(prediction))
    label = list(set(label))
    try:
        score = len(list(set(prediction) & set(label))) / len(label)
    except Exception:
        print(f"division by zero prediction: {prediction}, label: {label}, len(label): {len(label)}")
    return score
```

<!-- #region id="FOuWWpj4kZ44" -->
### Data Preparation
<!-- #endregion -->

```python id="T-Ntmqpyko91"
pd_artist = pd.read_csv('20210824_212829_artists.tsv', sep='\t')
pd_audio = pd.read_csv('20210824_212829_audios.tsv', sep='\t')
pd_playlist = pd.read_csv('20210824_212829_playlists.tsv', sep='\t')
pd_track = pd.read_csv('20210824_212829_tracks.tsv', sep='\t')
```

```python id="t7-8rgvpmA66"
scaler = MinMaxScaler()

playlist_test_size=100
missing_rate=0.2
```

```python id="h4YUMkxQkt1x"
pd_playlist = pd_playlist.copy()
pd_track = pd_track.copy()
pd_audio = pd_audio.copy()

pd_track = pd.merge(pd_track, pd_audio, left_on='track_id', right_on='id')
pd_full = pd.merge(pd_track, pd_playlist, left_on='playlist_id', right_on='playlist_id')
pd_full = pd_full.drop('playlist_name', axis=1)
pd_full = pd_full.drop('playlist_description', axis=1)
pd_full = pd_full.drop('id', axis=1)
pd_full = pd_full.drop_duplicates()
pd_full = pd_full.fillna('None')

playlist_test = generate_playlist(pd_track, pd_playlist)
playlist_test_all_track = generate_playlist(pd_full, pd_playlist, num_playlist_to_test=playlist_test_size)
playlist_to_test = generate_test_playlist(playlist_test_all_track, missing_rate)
playlist_all_id_to_test = list(playlist_to_test.keys())

pd_full_test = pd_full.copy()
pd_full_test = pd_full_test[pd_full_test['playlist_id'].isin(playlist_all_id_to_test)]
pd_full_test = pd_full_test.reset_index(drop=True)
pd_full_test_copy = pd_full_test.copy()

columns_to_drop = ['playlist_id', 'track_id', 'track_name', 'artist_ids',
                    'artist_names', 'album_id', 'album_name',
                    'playlist_num_tracks', 'playlist_num_followers']
pd_full_test_copy = pd_full_test_copy.drop(columns_to_drop, axis=1)

sum = 0
for key in playlist_test_all_track.keys():
    sum += len(playlist_test_all_track[key])
assert sum == pd_full_test_copy.shape[0]

scaler.fit(pd_full_test_copy)
scaledpd_full_test = scaler.transform(pd_full_test_copy)
```

```python id="ouRJnvtWnrqe"
class ColDropper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns_to_drop = ['playlist_id', 'track_id',
                                'track_name', 'artist_ids',
                                'artist_names', 'album_id',
                                'album_name', 'playlist_num_tracks',
                                'playlist_num_followers']
        self.result_ = None

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        result = x.copy()
        result = result.drop(columns=self.columns_to_drop)
        return result
```

<!-- #region id="ee1DKoSykFNv" -->
### DBSCAN
<!-- #endregion -->

<!-- #region id="5YspQYO5p6M-" -->
#### Fitting
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 322} id="WL5KxOdMmV5m" executionInfo={"status": "ok", "timestamp": 1631544662901, "user_tz": -330, "elapsed": 1470, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="779e67bf-e63e-4570-ed5e-7415ced142c8"
plt.figure(figsize=(10, 5))
nn = NearestNeighbors(n_neighbors=5).fit(scaled_pd_full_test)
distances, idx = nn.kneighbors(scaled_pd_full_test)
distances = np.sort(distances, axis=0)

distances = distances[:, 1]
plt.plot(distances)
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="7G-tuOgbnGrW" executionInfo={"status": "ok", "timestamp": 1631544663979, "user_tz": -330, "elapsed": 1087, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5a2662f1-3f72-448d-ed64-ebe2c051e075"
dbscan = DBSCAN(eps=0.38, algorithm='ball_tree')
dbscan.fit(scaled_pd_full_test)
```

```python id="6rZrnYAXnKR2"
label = dbscan.labels_
pd_full_test_copy['cluster_label'] = label
```

```python colab={"base_uri": "https://localhost:8080/"} id="o78kr2a2nvEg" executionInfo={"status": "ok", "timestamp": 1631544666222, "user_tz": -330, "elapsed": 20, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2b7b0159-bff6-4b57-d4dc-f98bf5e062b9"
full_pipeline = Pipeline(
    [('ColDropper', ColDropper()), ('Scaler', MinMaxScaler()), ('DBSCAN', DBSCAN(eps=0.38, algorithm='ball_tree'))])
full_pipeline
```

```python colab={"base_uri": "https://localhost:8080/"} id="0oDr4ZuanzL2" executionInfo={"status": "ok", "timestamp": 1631544666227, "user_tz": -330, "elapsed": 18, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f9fa5d4c-dcdd-4f16-b85a-497176f1521b"
print(full_pipeline.get_params().keys())
```

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="KxRzTrjEnzJY" executionInfo={"status": "ok", "timestamp": 1631544667899, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="797cb1df-30d4-40e0-af4f-0b696e399353"
playlist_id_to_test = playlist_all_id_to_test[0]
playlist_id_to_test
```

```python colab={"base_uri": "https://localhost:8080/"} id="MnO62HBinzGN" executionInfo={"status": "ok", "timestamp": 1631544668452, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3bd84fd7-f669-469a-b7ad-0fcdf3298460"
all_name_songs = list(
    pd_full[pd_full['track_id'].isin(playlist_test_all_track[playlist_id_to_test])]['track_name'].values)
print('All song in playlist 0:', all_name_songs)

given_name_songs = list(pd_full[pd_full['track_id'].isin(playlist_to_test[playlist_id_to_test])]['track_name'].values)
print('Given in playlist 0 : ', given_name_songs)
```

<!-- #region id="pmc9e_IJpzoD" -->
#### Recommendations
<!-- #endregion -->

```python id="sfXKwjrUnzDH"
def recommend_songs(given_playlist_track, pd_full_test, info_df, n_pred, eps=0.38, random_state=42, algo='ball_tree'):
    pid = list(info_df[info_df['track_id'].isin(given_playlist_track)]['playlist_id'].value_counts().index)[0]

    full_pipeline.set_params(DBSCAN__eps=eps, DBSCAN__algorithm=algo)
    full_pipeline.fit(X=pd_full_test)
    label = full_pipeline['DBSCAN'].labels_

    info_df_copy = info_df.copy()
    info_df_copy['cluster_label'] = label

    most_cluster = list(info_df_copy[info_df['playlist_id'] == pid]['cluster_label'].value_counts().index)[0]

    same_cluster_track_df = info_df[
        ~(info_df['track_id'].isin(given_playlist_track)) & (info_df_copy['cluster_label'] == most_cluster)]
    same_cluster_track_df = same_cluster_track_df.sort_values(by='popularity', ascending=False)

    result = []
    recs_names = list(np.unique(same_cluster_track_df['track_name'].values))
    preds = collections.Counter(recs_names).most_common(n_pred)
    for pred in preds:
        result.append(pred[0])

    return result
```

```python colab={"base_uri": "https://localhost:8080/"} id="nfMsKT13pEfW" executionInfo={"status": "ok", "timestamp": 1631544742894, "user_tz": -330, "elapsed": 1367, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5834d257-a75d-4780-d1a5-0d5683875824"
recommended_songs = recommend_songs(given_playlist_track=playlist_to_test[playlist_id_to_test],
                                    pd_full_test=pd_full_test,
                                    info_df=pd_full_test,
                                    n_pred=len(playlist_test_all_track[playlist_id_to_test]) - len(
                                        playlist_to_test[playlist_id_to_test]))

print(recommended_songs)
```

<!-- #region id="RN_Iz9xkpxfX" -->
#### Testing
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="PxRsSQUOpIBN" executionInfo={"status": "ok", "timestamp": 1631545548726, "user_tz": -330, "elapsed": 781399, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="aa6b45d7-9461-4745-f63a-253c2905e205"
def test_multi(pd_playlist, pd_full, missing_rate=0.2, nums_playlists_test=100):
    r_score = []

    playlist_test_all_track = generate_playlist(pd_full, pd_playlist, num_playlist_to_test=nums_playlists_test)
    playlist_to_test = generate_test_playlist(playlist_test_all_track, missing_rate=missing_rate)

    all_playlistID_for_tests = list(playlist_to_test.keys())

    pd_full_tests = pd_full.copy()
    pd_full_tests = pd_full_tests[pd_full_tests['playlist_id'].isin(all_playlistID_for_tests)]
    pd_full_tests = pd_full_tests.reset_index(drop=True)

    pd_full_tests_copy = pd_full_tests.copy()

    recommended_songs_all_playlist = {}

    for each_playlist in all_playlistID_for_tests:
        recommended_songs = recommend_songs(playlist_to_test[each_playlist],
                                            pd_full_tests_copy,
                                            pd_full_tests,
                                            len(playlist_test_all_track[each_playlist]) - len(
                                                playlist_to_test[each_playlist]))
        recommended_songs_all_playlist[each_playlist] = recommended_songs
        all_name_songs = list(
            pd_full[pd_full['track_id'].isin(playlist_test_all_track[each_playlist])]['track_name'].values)
        given_name_songs = list(pd_full[pd_full['track_id'].isin(playlist_to_test[each_playlist])]['track_name'].values)
        songs_need_to_recommend = [song for song in all_name_songs if song not in given_name_songs]
        r_score.append(r_precision(recommended_songs, songs_need_to_recommend))
    return r_score
    
r_score = test_multi(pd_playlist, pd_full, missing_rate=0.2, nums_playlists_test=100)
missing_rates = np.arange(0.1, 1, 0.1)

for missing_rate in missing_rates:
    r_score = test_multi(pd_playlist, pd_full, missing_rate=missing_rate, nums_playlists_test=100)
    print('*' * 50)
    print('Mising rate = ', missing_rate)
    print('Average R-Precision', np.array(r_score).mean())
    print('Max R-Precision', np.array(r_score).max())
    print('*' * 50)
```

<!-- #region id="Tm7R1Scvpto1" -->
#### Visualization
<!-- #endregion -->

```python id="2OKo67pMpaMv"
def visualized_cluster_result(dataframe, model_label,suptitle="TSNE2D vs. PCA2D",path_to_save="../images/fig.png"):
    sns.set();
    X = dataframe.copy()
    TSNE2 = TSNE(n_components=2,random_state=42).fit_transform(X)
    dftsne = pd.DataFrame(TSNE2)
    dftsne['cluster'] = model_label
    dftsne.columns = ['x1','x2','cluster']

    X = dataframe.copy()
    PCA2 = PCA(n_components=2,random_state=42).fit_transform(X)
    dfpca2 = pd.DataFrame(PCA2)
    dfpca2['cluster'] = model_label
    dfpca2.columns = ['x1','x2','cluster']

    fig, ax = plt.subplots(1, 2, figsize=(18,9))
    sns.scatterplot(data=dftsne,x='x1',y='x2',hue='cluster',legend="full",alpha=0.5,ax=ax[0],palette=plt.get_cmap('tab20'))
    ax[0].set_title('Visualized on TSNE 2D')
    sns.scatterplot(data=dfpca2,x='x1',y='x2',hue='cluster',legend="full",alpha=0.5,ax=ax[1],palette=plt.get_cmap('tab20'))
    ax[1].set_title('Visualized on PCA 2D')
    fig.suptitle(f'{suptitle}');

    fig.savefig(path_to_save)
    return dftsne, dfpca2
```

```python colab={"base_uri": "https://localhost:8080/", "height": 550} id="Tdhnv8Mlpbuw" executionInfo={"status": "ok", "timestamp": 1631545612579, "user_tz": -330, "elapsed": 63866, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="be7d4dda-426a-4b18-873b-7936342166d7"
db_scan_dftsne, db_scan_dfpca2 = visualized_cluster_result(scaled_pd_full_test,label,"Visualized with 7000 tracks","dbscan_7000tracks_13cluster.png")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 883} id="tTX0jn8xpg_I" executionInfo={"status": "ok", "timestamp": 1631545628178, "user_tz": -330, "elapsed": 9493, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fa10208e-c864-4959-d7c9-816c4eb00bd3"
sns.set();
sns.set(rc={'figure.figsize':(15,15)})
sns.scatterplot(data=db_scan_dftsne,x='x1',y='x2',hue='cluster',legend="full",alpha=0.5,palette=plt.get_cmap('tab20'))
plt.savefig("db_scan_dftsne.png")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 883} id="DOU7G7cZpkKe" executionInfo={"status": "ok", "timestamp": 1631545632754, "user_tz": -330, "elapsed": 4584, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="cc80269d-0504-4da0-f144-7b63fd589b60"
sns.set();
sns.set(rc={'figure.figsize':(15,15)})
sns.color_palette("Spectral", as_cmap=True)
sns.scatterplot(data=db_scan_dfpca2,x='x1',y='x2',hue='cluster',legend="full",alpha=0.5,palette=plt.get_cmap('tab20'))
plt.savefig("db_scan_dfpca2.png")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 883} id="mcxgW7xrpnLe" executionInfo={"status": "ok", "timestamp": 1631545632755, "user_tz": -330, "elapsed": 67, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a43c2df7-3e36-40bf-ddca-718404a72fb2"
sns.set();
sns.set(rc={'figure.figsize':(15,15)})
gt_dftsne = db_scan_dftsne.copy()
gt_label = LabelEncoder().fit_transform(pd_full_test["playlist_id"])
gt_dftsne['cluster'] = gt_label
sns.scatterplot(data=gt_dftsne,x='x1',y='x2',hue='cluster',legend="full",alpha=0.5, palette=plt.get_cmap('tab20'))
plt.legend([],[], frameon=False)
plt.savefig("ground_truth.png")
```

<!-- #region id="a3RBWw7NkIN0" -->
### KNN
<!-- #endregion -->

<!-- #region id="gjFPzYvOqV9P" -->
#### Fitting
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="oddLoypRqbgg" executionInfo={"status": "ok", "timestamp": 1631545632763, "user_tz": -330, "elapsed": 56, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c009fb39-a020-4f69-dfe1-bf0a6ffb9b4c"
nn = NearestNeighbors(n_neighbors=10).fit(scaled_pd_full_test)
distances, idx = nn.kneighbors(scaled_pd_full_test)
distances
```

```python colab={"base_uri": "https://localhost:8080/", "height": 72} id="vZKzNCL0qwSo" executionInfo={"status": "ok", "timestamp": 1631545659059, "user_tz": -330, "elapsed": 1626, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="56b0259c-c4dc-4120-94d7-4daed4fb1d99"
full_pipeline = Pipeline([('ColDropper', ColDropper()), ('Scaler', MinMaxScaler()), ('KNN', NearestNeighbors())])
print(full_pipeline.get_params().keys())

playlist_id_to_test = playlist_all_id_to_test[0]
playlist_id_to_test
```

```python colab={"base_uri": "https://localhost:8080/"} id="7Kwj9yCaqr3r" executionInfo={"status": "ok", "timestamp": 1631545661012, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d1ab5872-6b1c-4f5b-9ca3-c321c25147c2"
all_name_songs = list(
    pd_full[pd_full['track_id'].isin(playlist_test_all_track[playlist_id_to_test])]['track_name'].values)
print('All song in playlist 0:', all_name_songs)

given_name_songs = list(pd_full[pd_full['track_id'].isin(playlist_to_test[playlist_id_to_test])]['track_name'].values)
print('Given in playlist 0 : ', given_name_songs)
```

<!-- #region id="2ARG9T1BqURk" -->
#### Recommendations
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="_n3gUbSzqJ6a" executionInfo={"status": "ok", "timestamp": 1631545662896, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0041865c-9be5-43bc-c0bb-83e9329f34fa"
def recommend_songs(given_playlist_track, pd_full_test, info_df, n_pred):
    pid = list(info_df[info_df['track_id'].isin(given_playlist_track)]['playlist_id'].value_counts().index)[0]

    full_pipeline.set_params(KNN__n_neighbors=n_pred)
    full_pipeline.fit(X=pd_full_test)

    pd_given_feature = info_df[info_df['track_id'].isin(given_playlist_track)]
    pd_given_feature = full_pipeline['ColDropper'].fit_transform(pd_given_feature)

    pd_given_feature = full_pipeline['Scaler'].fit_transform(pd_given_feature)

    distances, idx = full_pipeline['KNN'].kneighbors(pd_given_feature)
    idx = idx[:, 1:]
    distances = distances[:, 1:]
    idx = idx.flatten()

    counter = collections.Counter(idx)
    a = counter.most_common(len(counter))

    result = []
    track_duplicates = []
    i = 0
    k = 0

    while k != n_pred:
        track_id = info_df.loc[a[i][0], 'track_id']
        if (track_id not in given_playlist_track) and (track_id not in track_duplicates):
            track_duplicates.append(track_id)
            track_name = info_df.loc[a[i][0], 'track_name']
            result.append(track_name)
            i += 1
            k += 1
        else:
            i += 1

    return result

recommended_songs = recommend_songs(given_playlist_track=playlist_to_test[playlist_id_to_test],
                                    pd_full_test=pd_full_test,
                                    info_df=pd_full_test,
                                    n_pred=len(playlist_test_all_track[playlist_id_to_test]) - len(
                                        playlist_to_test[playlist_id_to_test]))

recommended_songs = recommend_songs(
    given_playlist_track=['5HCyWlXZPP0y6Gqq8TgA20', '4XvcHTUfIlWfyJTRG0aqlo', '27NovPIUIRrOZoCHxABJwK'],
    pd_full_test=pd_full_test,
    info_df=pd_full_test,
    n_pred=len(playlist_test_all_track[playlist_id_to_test]) - len(playlist_to_test[playlist_id_to_test]))

print(recommended_songs)
```

<!-- #region id="MLBuofWXq9Je" -->
#### Testing
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="qNtVfUuDq-jh" executionInfo={"status": "ok", "timestamp": 1631545716458, "user_tz": -330, "elapsed": 48656, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6834ffc0-66ab-4348-9e1f-72ab694ac6ed"
def test_multi(pd_playlist, pd_full, missing_rate=0.2, nums_playlists_test=100):
    r_score = []

    playlist_test_all_track = generate_playlist(pd_full, pd_playlist, num_playlist_to_test=nums_playlists_test)
    playlist_to_test = generate_test_playlist(playlist_test_all_track, missing_rate=missing_rate)

    all_playlistID_for_tests = list(playlist_to_test.keys())

    pd_full_tests = pd_full.copy()
    pd_full_tests = pd_full_tests[pd_full_tests['playlist_id'].isin(all_playlistID_for_tests)]
    pd_full_tests = pd_full_tests.reset_index(drop=True)

    pd_full_tests_copy = pd_full_tests.copy()

    recommended_songs_all_playlist = {}

    for each_playlist in all_playlistID_for_tests:
        recommended_songs = recommend_songs(playlist_to_test[each_playlist],
                                            pd_full_tests_copy,
                                            pd_full_tests,
                                            len(playlist_test_all_track[each_playlist]) - len(
                                                playlist_to_test[each_playlist]))
        recommended_songs_all_playlist[each_playlist] = recommended_songs
        all_name_songs = list(
            pd_full[pd_full['track_id'].isin(playlist_test_all_track[each_playlist])]['track_name'].values)
        given_name_songs = list(pd_full[pd_full['track_id'].isin(playlist_to_test[each_playlist])]['track_name'].values)
        songs_need_to_recommend = [song for song in all_name_songs if song not in given_name_songs]
        r_score.append(r_precision(recommended_songs, songs_need_to_recommend))
    return r_score
    
r_score = test_multi(pd_playlist, pd_full, missing_rate=0.2, nums_playlists_test=100)
missing_rates = np.arange(0.1, 1, 0.1)

for missing_rate in missing_rates:
    r_score = test_multi(pd_playlist, pd_full, missing_rate=missing_rate, nums_playlists_test=100)
    print('*' * 50)
    print('Mising rate = ', missing_rate)
    print('Average R-Precision', np.array(r_score).mean())
    print('Max R-Precision', np.array(r_score).max())
    print('*' * 50)
```

<!-- #region id="4ryGr6h2kIIm" -->
### K-Means
<!-- #endregion -->

<!-- #region id="ddx-gwWcr426" -->
#### Fitting
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 899} id="NaeZ2ccyj-tR" executionInfo={"status": "ok", "timestamp": 1631545763484, "user_tz": -330, "elapsed": 47071, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ff2fec90-506a-45e8-bf73-6c15b9e135cf"
Sum_of_squared_distances = []
K = range(3, 40)
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans = kmeans.fit(scaled_pd_full_test)
    Sum_of_squared_distances.append(kmeans.inertia_)
    
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 439} id="iJoD7sdgj-q_" executionInfo={"status": "ok", "timestamp": 1631545764434, "user_tz": -330, "elapsed": 976, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7253a07f-bf25-4cfb-cfc1-5add145a5a72"
kmeans = KMeans(n_clusters=13)
kmeans.fit(scaled_pd_full_test)

KMeans(n_clusters=13)
label = kmeans.labels_
pd_full_test_copy['cluster_label'] = label
pd_full_test_copy
```

```python colab={"base_uri": "https://localhost:8080/", "height": 72} id="UHvHmDMRj-oJ" executionInfo={"status": "ok", "timestamp": 1631545764436, "user_tz": -330, "elapsed": 45, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="63f7300a-6226-4d6f-cd78-bd4b8627a299"
full_pipeline = Pipeline([('ColDropper', ColDropper()), ('Scaler', MinMaxScaler()), ('KMeans', KMeans(n_clusters=13))])
print(full_pipeline.get_params().keys())

playlist_id_to_test = playlist_all_id_to_test[0]
playlist_id_to_test
```

```python colab={"base_uri": "https://localhost:8080/"} id="cPipaxsOj-kr" executionInfo={"status": "ok", "timestamp": 1631545764437, "user_tz": -330, "elapsed": 24, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e3557d9f-2428-4623-b438-72817780728e"
all_name_songs = list(
    pd_full[pd_full['track_id'].isin(playlist_test_all_track[playlist_id_to_test])]['track_name'].values)
print('All song in playlist 0:', all_name_songs)

given_name_songs = list(pd_full[pd_full['track_id'].isin(playlist_to_test[playlist_id_to_test])]['track_name'].values)
print('Given in playlist 0 : ', given_name_songs)
```

<!-- #region id="qF8Zyz3fr27s" -->
#### Recommendations
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="nWH1T4e8r1vg" executionInfo={"status": "ok", "timestamp": 1631545765103, "user_tz": -330, "elapsed": 680, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="25ca55d4-f7aa-469e-e292-4ecbd2938f30"
def recommend_songs(given_playlist_track, pd_full_test, info_df, n_pred, num_clusters=13, random_state=42):
    pid = list(info_df[info_df['track_id'].isin(given_playlist_track)]['playlist_id'].value_counts().index)[0]
    #     df_features = pd.DataFrame(df_scaled_features.copy())

    full_pipeline.set_params(KMeans__n_clusters=num_clusters, KMeans__random_state=random_state)
    full_pipeline.fit(X=pd_full_test)
    label = full_pipeline['KMeans'].labels_

    info_df_copy = info_df.copy()
    info_df_copy['cluster_label'] = label

    most_cluster = list(info_df_copy[info_df['playlist_id'] == pid]['cluster_label'].value_counts().index)[0]

    same_cluster_track_df = info_df[
        ~(info_df['track_id'].isin(given_playlist_track)) & (info_df_copy['cluster_label'] == most_cluster)]
    same_cluster_track_df = same_cluster_track_df.sort_values(by='popularity', ascending=False)

    result = []
    recs_names = list(np.unique(same_cluster_track_df['track_name'].values))
    preds = collections.Counter(recs_names).most_common(n_pred)
    for pred in preds:
        result.append(pred[0])

    return result

recommended_songs = recommend_songs(given_playlist_track=playlist_to_test[playlist_id_to_test],
                                    pd_full_test=pd_full_test,
                                    info_df=pd_full_test,
                                    n_pred=len(playlist_test_all_track[playlist_id_to_test]) - len(
                                        playlist_to_test[playlist_id_to_test]))

print(recommended_songs)
```

<!-- #region id="GhzJdvs2r_a2" -->
#### Testing
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="rdoxISdZsAg6" executionInfo={"status": "ok", "timestamp": 1631546529356, "user_tz": -330, "elapsed": 764258, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fb21a08b-6aa6-441e-ca91-0f47fcf77c83"
def test_multi(pd_playlist, pd_full, missing_rate=0.2, nums_playlists_test=100):
    r_score = []

    playlist_test_all_track = generate_playlist(pd_full, pd_playlist, num_playlist_to_test=nums_playlists_test)
    playlist_to_test = generate_test_playlist(playlist_test_all_track, missing_rate=missing_rate)

    all_playlistID_for_tests = list(playlist_to_test.keys())

    pd_full_tests = pd_full.copy()
    pd_full_tests = pd_full_tests[pd_full_tests['playlist_id'].isin(all_playlistID_for_tests)]
    pd_full_tests = pd_full_tests.reset_index(drop=True)

    pd_full_tests_copy = pd_full_tests.copy()

    recommended_songs_all_playlist = {}

    for each_playlist in all_playlistID_for_tests:
        recommended_songs = recommend_songs(playlist_to_test[each_playlist],
                                            pd_full_tests_copy,
                                            pd_full_tests,
                                            len(playlist_test_all_track[each_playlist]) - len(
                                                playlist_to_test[each_playlist]))
        recommended_songs_all_playlist[each_playlist] = recommended_songs
        all_name_songs = list(
            pd_full[pd_full['track_id'].isin(playlist_test_all_track[each_playlist])]['track_name'].values)
        given_name_songs = list(pd_full[pd_full['track_id'].isin(playlist_to_test[each_playlist])]['track_name'].values)
        songs_need_to_recommend = [song for song in all_name_songs if song not in given_name_songs]
        r_score.append(r_precision(recommended_songs, songs_need_to_recommend))
    return r_score
    
r_score = test_multi(pd_playlist, pd_full, missing_rate=0.2, nums_playlists_test=100)
missing_rates = np.arange(0.1, 1, 0.1)

for missing_rate in missing_rates:
    r_score = test_multi(pd_playlist, pd_full, missing_rate=missing_rate, nums_playlists_test=100)
    print('*' * 50)
    print('Mising rate = ', missing_rate)
    print('Average R-Precision', np.array(r_score).mean())
    print('Max R-Precision', np.array(r_score).max())
    print('*' * 50)
```

<!-- #region id="Fwt40DQTsGIg" -->
#### Visualization
<!-- #endregion -->

```python id="96uPoBYcsIDf"
def visualized_cluster_result(dataframe, model_label,suptitle="TSNE2D vs. PCA2D",path_to_save="../images/fig.png"):
    sns.set();
    X = dataframe.copy()
    TSNE2 = TSNE(n_components=2,random_state=42).fit_transform(X)
    dftsne = pd.DataFrame(TSNE2)
    dftsne['cluster'] = model_label
    dftsne.columns = ['x1','x2','cluster']

    X = dataframe.copy()
    PCA2 = PCA(n_components=2,random_state=42).fit_transform(X)
    dfpca2 = pd.DataFrame(PCA2)
    dfpca2['cluster'] = model_label
    dfpca2.columns = ['x1','x2','cluster']

    fig, ax = plt.subplots(1, 2, figsize=(18,9))
    sns.scatterplot(data=dftsne,x='x1',y='x2',hue='cluster',legend="full",alpha=0.5,ax=ax[0],palette=plt.get_cmap('tab20'))
    ax[0].set_title('Visualized on TSNE 2D')
    sns.scatterplot(data=dfpca2,x='x1',y='x2',hue='cluster',legend="full",alpha=0.5,ax=ax[1],palette=plt.get_cmap('tab20'))
    ax[1].set_title('Visualized on PCA 2D')
    fig.suptitle(f'{suptitle}');

    fig.savefig(path_to_save)
    return dftsne, dfpca2
```

```python colab={"base_uri": "https://localhost:8080/", "height": 550} id="nAxbGLIBsJMS" executionInfo={"status": "ok", "timestamp": 1631546592486, "user_tz": -330, "elapsed": 63158, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2bfb1ad2-929f-4f41-a271-689c352e2d17"
kmeans_dftsne, kmeans_dfpca2 = visualized_cluster_result(scaled_pd_full_test,label,"Visualized with 7000 tracks","kmean_7000tracks_13cluster.png")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 883} id="Rp12SCjgsJI1" executionInfo={"status": "ok", "timestamp": 1631546594690, "user_tz": -330, "elapsed": 2230, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="346a4990-d4d1-479a-af9b-23bcc4162cd3"
sns.set();
sns.set(rc={'figure.figsize':(15,15)})
sns.scatterplot(data=kmeans_dftsne,x='x1',y='x2',hue='cluster',legend="full",alpha=0.5,palette=plt.get_cmap('tab20'))
plt.savefig("kmeans_dftsne.png")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 883} id="eSykb-HDsJFL" executionInfo={"status": "ok", "timestamp": 1631546596779, "user_tz": -330, "elapsed": 2110, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="23a573bb-1ad0-4fea-e205-bc2182d065eb"
sns.set();
sns.set(rc={'figure.figsize':(15,15)})
sns.scatterplot(data=kmeans_dfpca2,x='x1',y='x2',hue='cluster',legend="full",alpha=0.5,palette=plt.get_cmap('tab20'))
plt.savefig("kmeans_dfpca2.png")
```

<!-- #region id="OKxB3C-qLtTQ" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="sYly5mn2LtTR" executionInfo={"status": "ok", "timestamp": 1639472633512, "user_tz": -330, "elapsed": 3775, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5569da1c-c6c7-4a9c-cc2e-3b36d8e00d28"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="J-5ZxweDLtTS" -->
---
<!-- #endregion -->

<!-- #region id="gJZXV7JRLtTS" -->
**END**
<!-- #endregion -->
