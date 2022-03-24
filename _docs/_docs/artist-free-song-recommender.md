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

<!-- #region id="qWetk6u2Nr9x" -->
# Artist-free Song Recommender
> In this tutorial, we will first fetch the data from Spotify API, then preprocess it and perform EDA. After that we will train a genre predictor model and replace the Spotify's predicted genre with our predicted values. We are doing this replacement because Spotify uses artist's information also in predicting genres but we are building a song recommender that find similar songs purely on the basis of song features because using artist's info adds a fairness bias to the recommendations. Spotify gets higher accuracy but at the cost of fairness to the artists. After building genre classification model, we will train a KNN model, a non-personalized memory-based model, to find top-k similar songs. 

- toc: true
- badges: true
- comments: true
- categories: [Fairness, Music, API, KNN, SHAP, XGBoost]
- image:
<!-- #endregion -->

<!-- #region id="obrCXps3QMW8" -->
## Introduction
<!-- #endregion -->

<!-- #region id="PcfbyJRLQt9x" -->
<!-- #endregion -->

<!-- #region id="TjLbn6I7QN8P" -->
Musicians, often times, have varying styles from album to album and from track to track. They tend to dislike being put in a “genre” box which many organizations tend to do. For example, following his first Grammy win for his album “Igor” in 2020, Tyler the Creator "admitted that while he was "very grateful" for his win, the categorizing of his album as rap was a "backhanded compliment."

Spotify tends to label the genre of songs based off what genre the artist falls under which is quite unfair to an artist if said artist wants to branch out. Not only that, if a user likes just a particular song from an artist, but not every song from said artist, how can one recommend songs based off that particular song without influence from the artists genre? Can we use the predicted genre of a song, as well as other features, to make better song recommendations based off an inputted song?
<!-- #endregion -->

<!-- #region id="2oPSPE4Z1Im6" -->
## Setup
<!-- #endregion -->

```python id="YIy2jS5l-cDB"
!pip install shap
!pip install spotipy
```

```python id="shytFOCclyKj"
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# import imbalanced pipeline since you can't use SMOTE with the regular python pipeline
from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV

import xgboost
from xgboost import XGBClassifier

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

import pickle

import shap

import warnings
warnings.filterwarnings('ignore')

%config InlineBackend.figure_format = 'svg'

plt.style.use('fivethirtyeight')
plt.style.use('seaborn-notebook')
plt.style.use('default')
```

```python colab={"base_uri": "https://localhost:8080/"} id="umdpW6OBNv1s" outputId="4ecb052a-2533-4374-f501-f7ca19273f8b"
!pip install -q watermark
%reload_ext watermark
%watermark -m -iv -u -t -d
```

<!-- #region id="Lz46GX5kglBs" -->
## Fetch data from Spotify API
<!-- #endregion -->

```python id="GH1BcYTqkNgv"
!cp /content/drive/MyDrive/mykeys.py /content
import mykeys
!rm /content/mykeys.py

client_id = mykeys.spotify_client_id
client_secret = mykeys.spotify_client_secret
```

```python id="rO88iswWgxu1"
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))
```

```python id="AsuNBLWFlVKp"
# to convert keys to values
key_octave_values = ['C', 'C#/Db','D','D#/Eb', 'E',
                    'F', 'F#/Gb', 'G', 'G#/Ab', 'A',
                    'A#/Bb', 'B']

mode_mm_values = ['Minor', 'Major']
explicit_values = ['Clean', 'Explicit']
```

```python id="RKfolqlQmDM3"
# static column names to use to build dataframe
column_names = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 
                'instrumentalness', 'liveness', 'valence', 'tempo', 'type', 'id', 'uri', 
                'track_href', 'analysis_url', 'duration_ms', 'time_signature']
```

```python id="nEoa6Za0mGF1"
# convert playlists (From a playlist id) to dataframe 
def user_playlist_tracks_full(spotify_connection=sp, user=None, 
                              playlist_id=None, genre=None):

    # first run through also retrieves total no of songs in library
    response = spotify_connection.user_playlist_tracks(user, playlist_id, limit=100)
    results = response["items"]

    # subsequently runs until it hits the user-defined limit or has read all songs in the library
    # spotify limits 100 songs per request so used a while loop to read all songs
    while len(results) < response["total"]:
        response = spotify_connection.user_playlist_tracks(
            user, playlist_id, limit=100, offset=len(results)
        )
        results.extend(response["items"])
    
    #Retrieve song ids
    playlist_song_ids = []
    for each in range(len(results)):
        playlist_song_ids.append((results[each]['track']['id']))
    #Remove possible nones
    playlist_song_ids = [i for i in playlist_song_ids if i]
    
    #Create dataframe
    theDataFrame = pd.DataFrame(columns = column_names)
    #Add features
    while(len(playlist_song_ids)>0):
        theDataFrame = theDataFrame.append(sp.audio_features(playlist_song_ids[:100]),ignore_index=True)
        playlist_song_ids = playlist_song_ids[100:]
    
    #Pass in genre dependent on name of playlist
    theDataFrame['genre'] = genre

    return theDataFrame
```

```python colab={"base_uri": "https://localhost:8080/"} id="AeBv8m6Aml9t" outputId="4712900f-ddb4-4eef-c08b-701b2f1fe3c4"
# retrieving data for 4 genres from their respective playlists
alt_metal_songs = user_playlist_tracks_full(playlist_id = '40DeXsA9tEIwNwBmrZ4rkt', genre = 'alt-metal')
hiphop_songs = user_playlist_tracks_full(playlist_id = '13u9Bn677jEHePtS7XKmih', genre = 'hip-hop')
rock_songs = user_playlist_tracks_full(playlist_id = '1SY54UtMrIadoVThZsJShG', genre = 'rock')
pop_songs = user_playlist_tracks_full(playlist_id = '1szFiylNjSI99tpQgVZ3ki', genre = 'pop')

# combine the dataframes
all_songs = hiphop_songs.append([pop_songs, rock_songs, alt_metal_songs], ignore_index=True)

# to make sure the number of records are same
assert len(all_songs) == len(alt_metal_songs) + len(hiphop_songs) + len(rock_songs) + len(pop_songs)

# printing number of records of each genre
len(alt_metal_songs), len(hiphop_songs), len(rock_songs), len(pop_songs)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 309} id="KS0dLQjzn8sn" outputId="7972c822-c260-407e-a081-e104d7fbd4bb"
all_songs.head()
```

<!-- #region id="Q69GJGJa1OFy" -->
## EDA
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Fn93tL95oB5U" outputId="0b4a8a26-c8f4-4104-ddb0-1209d2be3686"
all_songs.info()
```

```python id="rca2FA3nm8y-"
# drop unneccessary attributes
all_songs.drop(['type', 'uri', 'track_href', 'analysis_url'], axis = 1, inplace = True)

# convert from object to float
all_songs['duration_ms'] = all_songs['duration_ms'].astype(float)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 568} id="yY90rAbponmh" outputId="63656b23-3c87-4964-e5a2-2236cbce4476"
all_songs.describe(include='all').T
```

```python colab={"base_uri": "https://localhost:8080/", "height": 436} id="fRKa66wbolk9" outputId="36ca7b4a-8114-4c62-bb0d-fc62f19ee6a6"
all_songs.query("danceability == 0")
```

```python id="M339WUEhtFnS"
def song_artist_from_key(sp,key):
    theTrack = sp.track(key)
    if(theTrack is not None):
        song_title = theTrack['name']
        artist_title = theTrack['artists'][0]['name']
        song_link = theTrack['external_urls']['spotify']
        return (song_title, artist_title, song_link)
    else:
        return None
```

```python colab={"base_uri": "https://localhost:8080/"} id="VFX-SxX9tMYU" outputId="f548acda-8f30-4b72-9cb3-6d37af5866b8"
zero_dance_list = all_songs.query("danceability == 0")['id']
for each in zero_dance_list:
    print(song_artist_from_key(sp,each))
```

```python colab={"base_uri": "https://localhost:8080/"} id="FqJRTgpktN-z" outputId="fe0c9fee-477c-4dee-efcc-06f7516463f0"
zero_tempo_list = all_songs.query("tempo == 0")['id']
for each in zero_tempo_list:
    print(song_artist_from_key(sp,each))
```

```python id="oHfEqbRjtRgf"
all_songs = all_songs.query("danceability != 0")
all_songs = all_songs.query(f"duration_ms < {1e6}")
```

```python id="8H_zec6ltxQ5"
def mean_by_genre_plot(ax, groupby, feature):
    sns.barplot(x = all_songs.groupby('genre')[feature].mean().index,
                y = all_songs.groupby('genre')[feature].mean().values,
                ax = ax,)
    ax.set_title(f'Average {feature.title()} by Genre')
    ax.set_ylabel(f'{feature}')
    return ax
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="lXrunfuStyI6" outputId="b52117cd-fa2a-4a2d-a44b-9bb50e5199f8"
genre_group = all_songs.groupby('genre')
continuous_features = list(all_songs.select_dtypes(include = [float, int]).columns)

# for i, each in enumerate(continuous_features):
#     mean_by_genre_plot(genre_group, each)

fig, ax = plt.subplots(2, 5, figsize=(30,12), constrained_layout=True)

# for i in range(3):
#     for j in range(3):
#         col_name = n_cols[i*3+j]
#         ax[i,j] = plot(data[col_name])

for i, each in enumerate(continuous_features):
    n = 5
    try:
        _ax = ax[i//n,i%n]
        ax[i//n,i%n] = mean_by_genre_plot(_ax, genre_group, each)
    except:
        pass

plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 594} id="K-tnN9K5ubrA" outputId="ea8f709a-0ff9-4e98-fc83-2f0e54140700"
fig, ax = plt.subplots(figsize=(12,5))
sns.heatmap(all_songs.corr(), annot=True, cmap='YlGnBu')
plt.title('Correlation Matrix')
plt.show()
```

```python id="j4ov7WeZ1k8W"
all_songs.to_parquet('songs_data.parquet.gzip', compression='gzip')
```

<!-- #region id="p4yXRQGZ1QR0" -->
## Genre Classification Model
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 340} id="rPMmxlbO1SLV" outputId="1c8c964c-3d5f-42c5-93ba-8ae476722354"
df = pd.read_parquet('songs_data.parquet.gzip')
df.set_index('id', inplace=True)
df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="hSFf6W041hi8" outputId="ae15d7c1-7604-4d28-a610-66830c81f627"
df.info()
```

```python id="QjkonylcHfBL"
cat_cols = ['key','mode','time_signature']
df[cat_cols] = df[cat_cols].astype('str')
```

```python colab={"base_uri": "https://localhost:8080/"} id="FDI15qME1yyt" outputId="2739ac76-6fdd-44c0-9922-ec2e1e0343cc"
# drop duplicates
print(df.duplicated().sum())
df.drop_duplicates(inplace=True)
print(df.duplicated().sum())
```

```python colab={"base_uri": "https://localhost:8080/"} id="zBrlzR4K2ALU" outputId="b7c124a1-4c1a-4bc6-a130-eae81decb534"
# split the columns to easily use later
categorical_columns = list(df.drop('genre', axis = 1).select_dtypes('object').columns)
numerical_columns = list(df.drop('genre', axis = 1).select_dtypes(exclude = 'object').columns)

# train test split 
X_train, X_test, y_train, y_test = train_test_split(df.drop(['genre'], axis=1),
                                                    df.genre,
                                                    random_state=42)

# see the percentage of each genre in the whole set
print(y_train.value_counts(normalize=True))
```

```python id="IRz81gWg3FsQ"
# function to easily view results
def evaluation_report(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"F1 Score: {f1_score(y_test, model.predict(X_test), average = 'macro')}")
    plot_confusion_matrix(model, X_test, y_test, cmap='GnBu',xticks_rotation='vertical',
                          values_format = '')
```

<!-- #region id="jRm3LS5W2YF2" -->
### First Simple Model
Give us a base model to compare future models to.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="vyQCSkC92Umc" outputId="91ed8c3f-c74f-4090-ebbd-ec908d12cdae"
lr = LogisticRegression(random_state=42)
ss = StandardScaler()
lr.fit(ss.fit_transform(X_train), y_train)
print(lr.classes_)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 522} id="Vt0Tlplo2peF" outputId="76015a7d-8901-4084-da5a-e058b2aa3e68"
print('Training')
evaluation_report(lr, ss.fit_transform(X_train), y_train)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 517} id="YuhnMaLS2pZg" outputId="25aebb4d-2633-4ae5-9a4c-bc62641104de"
print('Testing')
evaluation_report(lr, ss.transform(X_test), y_test)
```

<!-- #region id="gYRtIh4t4-3D" -->
### Second Model

- Column transform categorical and numerical
- Address the class imbalance through a balanced class weight
- And lets grid search for best params
<!-- #endregion -->

```python id="e5QdaSYY498N"
# create pipelines for preprocessing. One for numerical data, one for categorical data and a column transformer to do both.
numerical_pipeline = imbpipeline(verbose=False,steps=[
                                ('ss', StandardScaler())
                            ])

categorical_pipeline = imbpipeline(verbose=False,steps=[
                            ('ohe', OneHotEncoder(drop='first',
                                                 sparse=False,))
                            ])

col_trans = ColumnTransformer(verbose=False,transformers=[
    ('numerical', numerical_pipeline, numerical_columns),
    ('categorical', categorical_pipeline, categorical_columns)
])

col_ohe_solo = ColumnTransformer(verbose=False,transformers=[
    ('categorical', categorical_pipeline, categorical_columns)
])
```

```python colab={"base_uri": "https://localhost:8080/"} id="JQ2JsqjB5MXp" outputId="f630570d-bacc-4be3-ecb4-12955650be36"
pipe_log = imbpipeline(verbose=False, steps=[
    ('col_trans', col_trans),
    ('lr', LogisticRegression(max_iter = 10000, random_state=9, class_weight='balanced'))
])

param_grid = [
    {
     'lr__penalty' : ['l1','l2'],
     'lr__C' : [.75,.5,.25],
     'lr__solver' : ['saga','sag', 'lbfgs']
    }, 
]

gs_lr = GridSearchCV(pipe_log, param_grid = param_grid, 
                        scoring = 'f1_macro', n_jobs=-1, verbose=True)
gs_lr.fit(X_train, y_train)

gs_lr.best_params_
```

```python colab={"base_uri": "https://localhost:8080/", "height": 519} id="uYET5cTT5R43" outputId="a0e68d73-c0dc-4935-9fe2-caab179f2ba9"
print('Training')
evaluation_report(gs_lr, X_train, y_train)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 517} id="6Rnd00TM5Xew" outputId="150bfe21-92a1-4249-c174-fe22420837e5"
print('Testing')
evaluation_report(gs_lr, X_test, y_test)
```

<!-- #region id="qmxxOrRb6Ohv" -->
### Model 3

Since we have a mixture of categorical and numerical features and a large amount of training samples, lets try XGBoost. Combine that with one hot encoder, smote, and gridsearch and let us see what happens.
<!-- #endregion -->

```python id="UfvXlPIX5XvM"
ohe = OneHotEncoder(drop='first', sparse=False)
sm = SMOTE(random_state = 42)

train_ohe = ohe.fit_transform(X_train[categorical_columns])
test_ohe = ohe.transform(X_test[categorical_columns])

train_ohe = X_train[numerical_columns].reset_index().join(pd.DataFrame(train_ohe)).set_index('id')

train_sm = sm.fit_resample(train_ohe, y_train)
test_ohe = X_test[numerical_columns].reset_index().join(pd.DataFrame(test_ohe)).set_index('id')
```

```python colab={"base_uri": "https://localhost:8080/"} id="C36vHuVr6Tau" outputId="8e36d87a-cd4e-4995-81ff-0a488660ca70"
# make sure not to over fit
xgbc = XGBClassifier(random_state = 9, n_jobs = -1)
params = {
    'learning_rate': [.1,.01,.005],
    'n_estimators': range(180,200,20),
    'booster': ['gbtree', 'dart']
}

# run gridsearch
gs_xgbc = GridSearchCV(xgbc, param_grid=params, scoring='f1_macro', n_jobs =-1)

gs_xgbc.fit(train_sm[0], train_sm[1])

gs_xgbc.best_params_
```

```python colab={"base_uri": "https://localhost:8080/", "height": 517} id="ujDsLR6G6ahr" outputId="dc05b92e-136e-4035-bff3-b6ed01f8f364"
print('Training')
evaluation_report(gs_xgbc, train_sm[0], train_sm[1])
```

<!-- #region id="B6OJ0l3Q6qi6" -->
### Model 4

Let's try Random forest classifier
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="HAa0yuSm7FXs" outputId="f186f876-4c5f-4edc-9241-60bb15faa3c2"
rf = RandomForestClassifier(n_jobs = -1, random_state = 9, class_weight = 'balanced')

params = {
    'max_depth': [6,7],
    'n_estimators': [197,198,199,300],
    'criterion': ['gini', 'entropy'],
}

#Run gridsearch
gs_rf = GridSearchCV(rf, param_grid=params, scoring='f1_macro', n_jobs =-1 )
gs_rf.fit(train_ohe, y_train)

gs_rf.best_params_
```

```python colab={"base_uri": "https://localhost:8080/", "height": 517} id="-oIvp__b7QVR" outputId="9bff395a-b3f3-48db-c5d3-1c07156a5595"
print('Training')
evaluation_report(gs_rf, train_ohe, y_train)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 517} id="jjnDrxJl7QTA" outputId="cc740521-eed9-4b82-eab8-e8fed7205204"
print('Testing')
evaluation_report(gs_rf, test_ohe, y_test)
```

<!-- #region id="8CgZyGRe7WP3" -->
### Model 5

Voting classifier
<!-- #endregion -->

```python id="2-bc1TaB7QQn"
voting_clf = VotingClassifier(
                estimators=[('xgb', gs_xgbc.best_estimator_),
                             ('rf', gs_rf.best_estimator_)], 
                voting='hard')

voting_clf.fit(train_ohe, y_train);
```

```python colab={"base_uri": "https://localhost:8080/", "height": 517} id="a5jeTlYr7QM_" outputId="8d1806f2-4a13-4a4e-e055-1b6f7ec5576e"
print("Training")
evaluation_report(voting_clf, train_ohe, y_train)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 517} id="axAxt2wz7iAv" outputId="e3218e05-22ce-4ffe-9fcc-747421d26c8a"
print("Testing")
evaluation_report(voting_clf, test_ohe, y_test)
```

<!-- #region id="qLBkXeTf778v" -->
### Best Model
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 502} id="Rtrn0nP17oMf" outputId="a17c995d-487d-461b-bb6e-167fb10e5d61"
best_model = voting_clf

X_final = df.drop('genre', axis = 1)
y_final = df.genre

# follow the same transformation on the initial model and do the same to whole dataframe
final_ohe = ohe.transform(X_final[categorical_columns])
final_ohe = X_final[numerical_columns].reset_index().join(pd.DataFrame(final_ohe)).set_index('id')

# fit best model on whole dataset
best_model.fit(final_ohe, y_final)

# See how well it does
evaluation_report(best_model, final_ohe, y_final)
```

<!-- #region id="erseTJp_7oI4" -->
### Model Analysis
See which features are most important in predicting genre
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 437} id="-fCMazTG7n5J" outputId="72852fd2-ebc1-4fce-ed6f-73476a0b0816"
vclf_xgbc = best_model.estimators_[0]

# plot feature importance from gradient boosting classifier
pd.DataFrame([final_ohe.columns,vclf_xgbc.feature_importances_]).T.set_index(0).sort_values(by=1, ascending = False)[:10].sort_values(by=1, ascending = True)\
.plot(kind="barh", width=.2, grid=True, title = "XGB Feature Importance");
```

```python colab={"base_uri": "https://localhost:8080/", "height": 437} id="VblIYGnq8WwA" outputId="3e0d9be5-a880-42c4-f66a-c4fca7e072df"
vclf_rf = best_model.estimators_[1]

# plot feature importance from random forest classifier
pd.DataFrame([final_ohe.columns,vclf_rf.feature_importances_]).T.set_index(0).sort_values(by=1, ascending = False)[:10].sort_values(by=1, ascending = True)\
.plot(kind="barh", width=.2, grid=True, title = "RF Feature Importance");
```

```python colab={"base_uri": "https://localhost:8080/", "height": 789} id="N0QMt1Hy8ws-" outputId="7e6c43cd-52e4-464b-da47-abae0e114b4b"
explainer = shap.TreeExplainer(vclf_xgbc)
shap_values = explainer.shap_values(final_ohe)
shap.summary_plot(shap_values, features=final_ohe, 
                  feature_names=final_ohe.columns, 
                  class_names = voting_clf.classes_,
                  title = "XGBC Feature Importance")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 789} id="aELySqpo87f4" outputId="c8db0a86-a085-49a6-f44c-67c1ba194ba4"
explainer = shap.TreeExplainer(vclf_rf)
shap_values = explainer.shap_values(final_ohe)
shap.summary_plot(shap_values, features=final_ohe, 
                  feature_names=final_ohe.columns, 
                  class_names = voting_clf.classes_)
```

<!-- #region id="eDjxDlRv9b5r" -->
### Save the results
<!-- #endregion -->

```python id="wjYKyZ989dTT"
# modify the dataset with the results
all_songs = df.copy()
all_songs['predicted_genre'] = best_model.predict(final_ohe)
all_songs.drop(['genre'], axis = 1, inplace = True)

# export the data
all_songs.to_parquet('songs_data_processed.parquet.gzip', compression='gzip')

# export the model
with open('genre_predictor.pkl', 'wb') as model_file:
  pickle.dump(best_model, model_file)

# export the OHE encoder
with open('ohe_encoder.pkl', 'wb') as encoder_file:
  pickle.dump(ohe, encoder_file)
```

<!-- #region id="EHxaf1wFAKzJ" -->
## Recommendation Model

A KNN-based non-personalized recommender
<!-- #endregion -->

<!-- #region id="qUiRFuxONhjA" -->
### Util functions
<!-- #endregion -->

```python id="oH-MaVe1APcB"
# Convert a song_id to a dataframe row
def song_to_df (sp, key):
    cat_cols = ['key', 'mode', 'time_signature']
    num_cols = ['danceability','energy','loudness','speechiness','acousticness',
                'instrumentalness','liveness','valence','tempo','duration_ms']
    
    row = pd.DataFrame(sp.audio_features(key)).drop(['type','uri',
                                               'track_href','analysis_url'], axis=1).set_index('id')
    row[cat_cols] = row[cat_cols].astype('str')
    return row

# Do preprocessing and make a genre prediction for a song 
def make_genre_prediction(sp, key, ohe, model):
    cat_cols = ['key', 'mode', 'time_signature']
    num_cols = ['danceability','energy','loudness','speechiness','acousticness',
                'instrumentalness','liveness','valence','tempo','duration_ms']
    row = song_to_df(sp,key)
    temp_ohe = ohe.transform(row[cat_cols])
    returning_obj = row[num_cols].reset_index().join(pd.DataFrame(temp_ohe)).set_index('id')
    return model.predict(returning_obj)

# Get the song info from song_id
def song_artist_from_key(sp,key):
    theTrack = sp.track(key)
    song_title = theTrack['name']
    artist_title = theTrack['artists'][0]['name']
    song_link = theTrack['external_urls']['spotify']
    return (song_title, artist_title, song_link)

# Get the song id from a query
def song_id_from_query(sp, query):
    q = query
    if(sp.search(q, limit=1, offset=0, type='track')['tracks']['total']>0):
        return sp.search( q, limit=1, offset=0, type='track')['tracks']['items'][0]['id']
    else:
        return None
```

<!-- #region id="vbYMnIiiNY-7" -->
### Load the artifacts
<!-- #endregion -->

```python id="VLFBgC7wBLXU"
# import the data
all_songs = pd.read_parquet('songs_data_processed.parquet.gzip')

# import the encoder
with open('ohe_encoder.pkl', 'rb') as encoder_file:
  ohe_make_genre_pred = pickle.load(encoder_file)

# import the model
with open('genre_predictor.pkl', 'rb') as model_file:
  best_model = pickle.load(model_file)
```

```python colab={"base_uri": "https://localhost:8080/"} id="qCXJrFgRILkA" outputId="de973ce7-0c4b-4e79-e620-91d3cb82b02c"
all_songs.info()
```

```python colab={"base_uri": "https://localhost:8080/"} id="POvyIrQ0D2c_" outputId="9b32514f-369f-4d9d-f55f-fe0635eef16a"
# create variables to easily access categorical and numerical columns
categorical_columns = list(all_songs.select_dtypes('object').columns)
numerical_columns = list(all_songs.select_dtypes(exclude = 'object').columns)
categorical_columns
```

```python colab={"base_uri": "https://localhost:8080/", "height": 255} id="lAJcX2CVCGC8" outputId="e3e16d69-c614-4c50-8994-31e23cdb47ff"
all_songs.head()
```

<!-- #region id="XwwaqFBCNc7d" -->
### KNN Model
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Ink78sp_CQVl" outputId="3b678129-3c1e-4a10-9e26-8ef498533cca"
# create a nearest neighbors object using cosine similarity metric.
neigh = NearestNeighbors(n_neighbors=15, radius=0.45, metric='cosine')

X_knn = all_songs

# total dataframe normalizing for nearest neighbors
MMScaler = preprocessing.MinMaxScaler()
MinMaxScaler = preprocessing.MinMaxScaler()
X_knn[numerical_columns] = MinMaxScaler.fit_transform(X_knn[numerical_columns])

# total dataframe dummying
ohe_knn = OneHotEncoder(drop='first', sparse=False)
X_knn_ohe = ohe_knn.fit_transform(X_knn[categorical_columns])
X_knn_transformed = X_knn[numerical_columns].reset_index().join(pd.DataFrame(X_knn_ohe, columns = ohe_knn.get_feature_names(categorical_columns))).set_index('id')

# fit the model
neigh.fit(X_knn_transformed)
```

```python id="BGHvnG3_ClWi"
# preprocessing for a single song
def knn_preprocessing(sp, key, num_col = numerical_columns, 
                      cat_col = categorical_columns,
                      mmScaler = MinMaxScaler, bm = best_model,
                      ohe_knn = ohe_knn, ohe_make_genre_pred = ohe_make_genre_pred):
    # Convert song to the dataframe
    row = song_to_df(sp, key)
    # Make genre prediction for inputted song
    genre = make_genre_prediction(sp,key, ohe_make_genre_pred, bm)
    # Append the predicted genre
    row['predicted_genre'] = genre[0]
    # Dummy the categorical
    row_dummied = ohe_knn.transform(row[cat_col])
    # Normalize the numerical
    row[num_col] = mmScaler.transform(row[num_col])
    # Combine the preprocessed rows and return it
    row = row[num_col].reset_index().join(pd.DataFrame(row_dummied, columns = ohe_knn.get_feature_names(cat_col))).set_index('id')
    return row
```

```python id="Ulv5yvkfCoMk"
def make_song_recommendations(sp, kneighs, query):
    #If the query is aspace or not filled, return no results
    if(query.isspace() or not query):
        return "No results found"
    song_id = song_id_from_query(sp, query)
    # If the query doesn't return an id, return no results
    if(song_id == None):
        return "No results found"
    # Get the song info
    song_plus_artist = song_artist_from_key(sp, song_id)
    # Preprocess the tracks
    song_to_rec = knn_preprocessing(sp, song_id)
    # Get the 15 nearest neighbors to inputted song
    nbrs = neigh.kneighbors(
       song_to_rec, 15, return_distance=False
    )
    # Properly retrieve the song info of each neighbor and return it
    playlist = []
    for each in nbrs[0]:
        the_rec_song = song_artist_from_key(sp, X_knn_transformed.iloc[each].name)
        if (((the_rec_song[0:2]) != song_plus_artist[0:2]) and
           ((the_rec_song[0:2]) not in playlist)):
            playlist.append(song_artist_from_key(sp, X_knn_transformed.iloc[each].name))
    return (playlist)
```

<!-- #region id="AitXAODYNfEn" -->
### Recommendations
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="nhxZcWWeDTEg" outputId="662441e4-1b51-407e-d617-f798a2c5a335"
# knowledge check to see if it matches
song_artist_from_key(sp, '6XGddj522FQRHXEhBtjiJu')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="70QoGc-8C0bd" outputId="eac893db-8597-4c96-cc08-cfbbd9baeaf3"
# get the ID from query
song_id_from_query(sp, "strobelite")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="lTygC4WJDrfo" outputId="b54c8333-5a30-428f-8eaa-d646858d4d2e"
# make the genre prediction
make_genre_prediction(sp, '6XGddj522FQRHXEhBtjiJu', ohe_make_genre_pred, best_model)[0]
```

```python colab={"base_uri": "https://localhost:8080/"} id="UsKoKqJ6DNyk" outputId="78678ae8-f289-4adf-cbca-480ca59a67d4"
# make the song recommendations
make_song_recommendations(sp, neigh, "strobelite")
```

<!-- #region id="3eBI9ba_P5_i" -->
## References

1. [https://github.com/jasonarikupurathu/Music-Recommendation-System](https://github.com/jasonarikupurathu/Music-Recommendation-System) `code`
2. [Getting Started with Spotify’s API & Spotipy](https://medium.com/@maxtingle/getting-started-with-spotifys-api-spotipy-197c3dc6353b) `blog`
<!-- #endregion -->
