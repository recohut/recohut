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

<!-- #region id="JVsnp7g_vxyH" -->
# Spotify Music Recommender
> Spotify recommendation system that recommends new songs for your playlist

- toc: true
- badges: true
- comments: true
- categories: [Music, TFIDF, Reference]
- author: "<a href='https://github.com/madhavthaker/spotify-recommendation-system'>Madhav Thaker</a>"
- image:
<!-- #endregion -->

<!-- #region id="4UWBRfnisalL" -->
## Setup
<!-- #endregion -->

```python id="Kdu7rjUzsg1u"
!pip install -q spotipy
```

```python id="pedQ04GmrffN"
import pandas as pd
import numpy as np
import json
import re 
import sys
import itertools

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import spotipy.util as util

import warnings
warnings.filterwarnings("ignore")
```

```python id="P3LNO8w2rffQ"
%matplotlib inline
```

<!-- #region id="Utat179XveSz" -->
## Data Loading
<!-- #endregion -->

```python id="V57Q9CxDtAyR"
!pip install -q -U kaggle
!pip install --upgrade --force-reinstall --no-deps kaggle
!mkdir ~/.kaggle
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d yamaerenay/spotify-dataset-19212020-160k-tracks
```

```python id="lHFXl1rotKZY"
!unzip /content/spotify-dataset-19212020-160k-tracks.zip
```

```python id="m7VEIbpPrffT"
spotify_df = pd.read_csv('tracks.csv')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 326} id="WRT371S7tU-A" outputId="1feb5f23-a64a-4c69-dedf-1c8aa39b8885"
spotify_df.head()
```

<!-- #region id="MUzocE1krffU" -->
Observations:
1. This data is at a **song level**
2. Many numerical values that I'll be able to use to compare movies (liveness, tempo, valence, etc)
2. Release date will useful but I'll need to create a OHE variable for release date in 5 year increments
3. Similar to 2, I'll need to create OHE variables for the popularity. I'll also use 5 year increments here
4. There is nothing here related to the genre of the song which will be useful. This data alone won't help us find relavent content since this is a content based recommendation system.
<!-- #endregion -->

```python id="Im5fyznNrffV" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="fb908c65-f88d-4f3d-d048-7a229d2befb1"
data_w_genre = pd.read_csv('artists.csv')
data_w_genre.head()
```

<!-- #region id="ZqC4hxdyrffV" -->
Observations:
1. This data is at an **artist level**
2. There are similar continuous variables as our initial dataset but I won't use this. I'll just use the values int he previous dataset. 
3. The genres are going to be really useful here and I'll need to use it moving forward. Now, the genre column appears to be in a list format but my past experience tells me that it's likely not. Let's investigate this further.
<!-- #endregion -->

```python id="DD4Z1dCurffW" colab={"base_uri": "https://localhost:8080/"} outputId="21877ca2-477a-4e9e-9aad-4a86bfadef20"
data_w_genre.dtypes
```

<!-- #region id="zMR-RnT5rffY" -->
Voila, now we have the genre column in a format we can actually use. If you go down, you'll see how we use it. 

Now, if you recall, this data is at a artist level and the previous dataset is at a song level. So what here's what we need to do:
1. Explode artists column in the previous so each artist within a song will have their own row
2. Merge `data_w_genre` to the exploded dataset in Step 1 so that the previous dataset no is enriched with genre dataset

Before I go further, let's complete these two steps.

Step 1. 
Similar to before, we will need to extract the artists from the string list. 
<!-- #endregion -->

```python id="x0EUCzaprffY"
spotify_df['artists_upd_v1'] = spotify_df['artists'].apply(lambda x: re.findall(r"'([^']*)'", x))
```

```python id="LNtDtHDarffY" outputId="79415ae1-6d99-4337-903b-af98590700b2"
spotify_df['artists'].values[0]
```

```python id="pS3w0S2-rffY" outputId="7e492db2-6e07-4eb7-99b6-fd299c8b3ac7"
spotify_df['artists_upd_v1'].values[0][0]
```

<!-- #region id="oJBfMZDGrffZ" -->
This looks good but did this work for every artist string format. Let's double check
<!-- #endregion -->

```python id="bzIaf0AVrffZ" outputId="253d89a3-e021-4381-f30e-550061c00361"
spotify_df[spotify_df['artists_upd_v1'].apply(lambda x: not x)].head(5)
```

<!-- #region id="Ch8MzF4yrffZ" -->
So, it looks like it didn't catch all of them and you can quickly see that it's because artists with an apostrophe in their title and the fact that they are enclosed in a full quotes. I'll write another regex to handle this and then combine the two
<!-- #endregion -->

```python id="VSftJREbrffa"
spotify_df['artists_upd_v2'] = spotify_df['artists'].apply(lambda x: re.findall('\"(.*?)\"',x))
spotify_df['artists_upd'] = np.where(spotify_df['artists_upd_v1'].apply(lambda x: not x), spotify_df['artists_upd_v2'], spotify_df['artists_upd_v1'] )
```

```python id="Ot0AfBRmrffa"
#need to create my own song identifier because there are duplicates of the same song with different ids. I see different
spotify_df['artists_song'] = spotify_df.apply(lambda row: row['artists_upd'][0]+row['name'],axis = 1)
```

```python id="5Fd_82urrffa"
spotify_df.sort_values(['artists_song','release_date'], ascending = False, inplace = True)
```

```python id="9DEjaHgOrffb" outputId="bd532072-ac8d-4339-a219-454a5045dce3"
spotify_df[spotify_df['name']=='Adore You']
```

```python id="mmQXnzzzrffb"
spotify_df.drop_duplicates('artists_song',inplace = True)
```

```python id="B8z7Rp3orffb" outputId="14f29e1a-fac2-4d3a-d80a-960884eca511"
spotify_df[spotify_df['name']=='Adore You']
```

<!-- #region id="2OVRvlVCrffb" -->
Now I can explode this column and merge as I planned to in `Step 2`
<!-- #endregion -->

```python id="UNTTxFKJrffc"
artists_exploded = spotify_df[['artists_upd','id']].explode('artists_upd')
```

```python id="PsXgYTHYrffc"
artists_exploded_enriched = artists_exploded.merge(data_w_genre, how = 'left', left_on = 'artists_upd',right_on = 'artists')
artists_exploded_enriched_nonnull = artists_exploded_enriched[~artists_exploded_enriched.genres_upd.isnull()]
```

```python id="Z7IhSBIzrffc" outputId="ff931e24-6c3f-4419-ff9d-06dbaf79c2fd"
artists_exploded_enriched_nonnull[artists_exploded_enriched_nonnull['id'] =='6KuQTIu1KoTTkLXKrwlLPV']
```

<!-- #region id="e7QInbyZrffc" -->
Alright we're almost their, now we need to:
1. Group by on the song `id` and essentially create lists lists
2. Consilidate these lists and output the unique values
<!-- #endregion -->

```python id="aAjc8IHQrffd"
artists_genres_consolidated = artists_exploded_enriched_nonnull.groupby('id')['genres_upd'].apply(list).reset_index()
```

```python id="6ebu54etrffd"
artists_genres_consolidated['consolidates_genre_lists'] = artists_genres_consolidated['genres_upd'].apply(lambda x: list(set(list(itertools.chain.from_iterable(x)))))
```

```python id="r7R5_o1Jrffd" outputId="4b9f4d2e-c9a5-4086-af00-a16a03038676"
artists_genres_consolidated.head()
```

```python id="u0z6Pxnjrffd"
spotify_df = spotify_df.merge(artists_genres_consolidated[['id','consolidates_genre_lists']], on = 'id',how = 'left')
```

<!-- #region id="h4uX3a7Xrffe" -->
## Feature Engineering

- Normalize float variables
- OHE Year and Popularity Variables
- Create TF-IDF features off of artist genres
<!-- #endregion -->

```python id="Wu2WxzEKrffe" outputId="cb7dae59-7057-41e2-a321-5f9f1a33f0b5"
spotify_df.tail()
```

```python id="wwB29FvBrffe"
spotify_df['year'] = spotify_df['release_date'].apply(lambda x: x.split('-')[0])
```

```python id="nU7L6eqmrffe"
float_cols = spotify_df.dtypes[spotify_df.dtypes == 'float64'].index.values
```

```python id="hJHJlPRdrffe"
ohe_cols = 'popularity'
```

```python id="RW4qp_9Srfff" outputId="77ca26d9-5d43-4eb7-95b6-ff608ef9902d"
spotify_df['popularity'].describe()
```

```python id="5a32WY0brfff"
# create 5 point buckets for popularity 
spotify_df['popularity_red'] = spotify_df['popularity'].apply(lambda x: int(x/5))
```

```python id="AbjdzXVorfff"
# tfidf can't handle nulls so fill any null values with an empty list
spotify_df['consolidates_genre_lists'] = spotify_df['consolidates_genre_lists'].apply(lambda d: d if isinstance(d, list) else [])
```

```python id="KYkCiE8brfff" outputId="31a53d24-9f62-45c1-8424-f418134ddbbe"
spotify_df.head()
```

```python id="86ryLOi-rffg"
#simple function to create OHE features
#this gets passed later on
def ohe_prep(df, column, new_name): 
    """ 
    Create One Hot Encoded features of a specific column

    Parameters: 
        df (pandas dataframe): Spotify Dataframe
        column (str): Column to be processed
        new_name (str): new column name to be used
        
    Returns: 
        tf_df: One hot encoded features 
    """
    
    tf_df = pd.get_dummies(df[column])
    feature_names = tf_df.columns
    tf_df.columns = [new_name + "|" + str(i) for i in feature_names]
    tf_df.reset_index(drop = True, inplace = True)    
    return tf_df

```

```python id="xjsXsqkHrffg" outputId="7f9e3200-dc01-4690-9160-89c84ed0d10d"
from IPython.display import Image
Image("/Users/thakm004/Documents/Spotify/tfidf_4.png")
```

```python id="NMEzZI7xrffg"
#function to build entire feature set
def create_feature_set(df, float_cols):
    """ 
    Process spotify df to create a final set of features that will be used to generate recommendations

    Parameters: 
        df (pandas dataframe): Spotify Dataframe
        float_cols (list(str)): List of float columns that will be scaled 
        
    Returns: 
        final: final set of features 
    """
    
    #tfidf genre lists
    tfidf = TfidfVectorizer()
    tfidf_matrix =  tfidf.fit_transform(df['consolidates_genre_lists'].apply(lambda x: " ".join(x)))
    genre_df = pd.DataFrame(tfidf_matrix.toarray())
    genre_df.columns = ['genre' + "|" + i for i in tfidf.get_feature_names()]
    genre_df.reset_index(drop = True, inplace=True)

    #explicity_ohe = ohe_prep(df, 'explicit','exp')    
    year_ohe = ohe_prep(df, 'year','year') * 0.5
    popularity_ohe = ohe_prep(df, 'popularity_red','pop') * 0.15

    #scale float columns
    floats = df[float_cols].reset_index(drop = True)
    scaler = MinMaxScaler()
    floats_scaled = pd.DataFrame(scaler.fit_transform(floats), columns = floats.columns) * 0.2

    #concanenate all features
    final = pd.concat([genre_df, floats_scaled, popularity_ohe, year_ohe], axis = 1)
     
    #add song id
    final['id']=df['id'].values
    
    return final
```

```python id="2jAZb4ARrffh"
complete_feature_set = create_feature_set(spotify_df, float_cols=float_cols)#.mean(axis = 0)
```

```python id="Sy8rtIxBrffj" outputId="a248c9b3-4b73-4251-aa2c-abd110bd47c2"
complete_feature_set.head()
```

<!-- #region id="Sq_ECDKErffk" -->
## Connect to Spotify API

Useful links:
1. https://developer.spotify.com/dashboard/
2. https://spotipy.readthedocs.io/en/2.16.1/
<!-- #endregion -->

```python id="boGlGN9erffl"
#client id and secret for my application
client_id = 'id'
client_secret= 'secret'
```

```python id="VVAs4dHHrffm"
scope = 'user-library-read'

if len(sys.argv) > 1:
    username = sys.argv[1]
else:
    print("Usage: %s username" % (sys.argv[0],))
    sys.exit()
```

```python id="nlGLB_56rffm"
auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)
```

```python id="NNtcfNgLrffm"
token = util.prompt_for_user_token(scope, client_id= client_id, client_secret=client_secret, redirect_uri='http://localhost:8881/')
```

```python id="JKhBxmzxrffn"
sp = spotipy.Spotify(auth=token)
```

```python id="woMnlbPvrffn"
#gather playlist names and images. 
#images aren't going to be used until I start building a UI
id_name = {}
list_photo = {}
for i in sp.current_user_playlists()['items']:

    id_name[i['name']] = i['uri'].split(':')[2]
    list_photo[i['uri'].split(':')[2]] = i['images'][0]['url']
```

```python id="xjCWF9Ogrffn" outputId="354d1d86-65a8-4741-b614-13bbac0e99f8"
id_name
```

```python id="PGiDdvlXrffo"
def create_necessary_outputs(playlist_name,id_dic, df):
    """ 
    Pull songs from a specific playlist.

    Parameters: 
        playlist_name (str): name of the playlist you'd like to pull from the spotify API
        id_dic (dic): dictionary that maps playlist_name to playlist_id
        df (pandas dataframe): spotify datafram
        
    Returns: 
        playlist: all songs in the playlist THAT ARE AVAILABLE IN THE KAGGLE DATASET
    """
    
    #generate playlist dataframe
    playlist = pd.DataFrame()
    playlist_name = playlist_name

    for ix, i in enumerate(sp.playlist(id_dic[playlist_name])['tracks']['items']):
        #print(i['track']['artists'][0]['name'])
        playlist.loc[ix, 'artist'] = i['track']['artists'][0]['name']
        playlist.loc[ix, 'name'] = i['track']['name']
        playlist.loc[ix, 'id'] = i['track']['id'] # ['uri'].split(':')[2]
        playlist.loc[ix, 'url'] = i['track']['album']['images'][1]['url']
        playlist.loc[ix, 'date_added'] = i['added_at']

    playlist['date_added'] = pd.to_datetime(playlist['date_added'])  
    
    playlist = playlist[playlist['id'].isin(df['id'].values)].sort_values('date_added',ascending = False)
    
    return playlist
```

```python id="0bqogpQLrffo" outputId="e3b16e2f-db57-43b2-d749-7842d4a0b0cd"
id_name
```

```python id="0JWBoFYJrffp"
playlist_EDM = create_necessary_outputs('EDM', id_name,spotify_df)
#playlist_chill = create_necessary_outputs('chill',id_name, spotify_df)
#playlist_classical = create_necessary_outputs('Epic Classical',id_name, spotify_df)
```

```python id="5tcSB8z_rffp"
from skimage import io
import matplotlib.pyplot as plt

def visualize_songs(df):
    """ 
    Visualize cover art of the songs in the inputted dataframe

    Parameters: 
        df (pandas dataframe): Playlist Dataframe
    """
    
    temp = df['url'].values
    plt.figure(figsize=(15,int(0.625 * len(temp))))
    columns = 5
    
    for i, url in enumerate(temp):
        plt.subplot(len(temp) / columns + 1, columns, i + 1)

        image = io.imread(url)
        plt.imshow(image)
        plt.xticks(color = 'w', fontsize = 0.1)
        plt.yticks(color = 'w', fontsize = 0.1)
        plt.xlabel(df['name'].values[i], fontsize = 12)
        plt.tight_layout(h_pad=0.4, w_pad=0)
        plt.subplots_adjust(wspace=None, hspace=None)

    plt.show()
```

```python id="CIvj6C2lrffp" outputId="407cf5f2-b57c-48e1-a566-a1e42dd49d25"
playlist_EDM
```

```python id="5hQAaadKrffq" outputId="a03cc253-8ddd-4ab2-cb71-a56637ff8f7a"
visualize_songs(playlist_EDM)
```

<!-- #region id="qES3r5Vbrffq" -->
## Create Playlist Vector
<!-- #endregion -->

```python id="793T4eEtrffq" outputId="ea5d3811-3613-4d9e-c3a5-0804dd469db3"
from IPython.display import Image
Image("/Users/thakm004/Documents/Spotify/summarization_2.png")
```

```python id="bzxGqeQSrffr"
def generate_playlist_feature(complete_feature_set, playlist_df, weight_factor):
    """ 
    Summarize a user's playlist into a single vector

    Parameters: 
        complete_feature_set (pandas dataframe): Dataframe which includes all of the features for the spotify songs
        playlist_df (pandas dataframe): playlist dataframe
        weight_factor (float): float value that represents the recency bias. The larger the recency bias, the most priority recent songs get. Value should be close to 1. 
        
    Returns: 
        playlist_feature_set_weighted_final (pandas series): single feature that summarizes the playlist
        complete_feature_set_nonplaylist (pandas dataframe): 
    """
    
    complete_feature_set_playlist = complete_feature_set[complete_feature_set['id'].isin(playlist_df['id'].values)]#.drop('id', axis = 1).mean(axis =0)
    complete_feature_set_playlist = complete_feature_set_playlist.merge(playlist_df[['id','date_added']], on = 'id', how = 'inner')
    complete_feature_set_nonplaylist = complete_feature_set[~complete_feature_set['id'].isin(playlist_df['id'].values)]#.drop('id', axis = 1)
    
    playlist_feature_set = complete_feature_set_playlist.sort_values('date_added',ascending=False)

    most_recent_date = playlist_feature_set.iloc[0,-1]
    
    for ix, row in playlist_feature_set.iterrows():
        playlist_feature_set.loc[ix,'months_from_recent'] = int((most_recent_date.to_pydatetime() - row.iloc[-1].to_pydatetime()).days / 30)
        
    playlist_feature_set['weight'] = playlist_feature_set['months_from_recent'].apply(lambda x: weight_factor ** (-x))
    
    playlist_feature_set_weighted = playlist_feature_set.copy()
    #print(playlist_feature_set_weighted.iloc[:,:-4].columns)
    playlist_feature_set_weighted.update(playlist_feature_set_weighted.iloc[:,:-4].mul(playlist_feature_set_weighted.weight,0))
    playlist_feature_set_weighted_final = playlist_feature_set_weighted.iloc[:, :-4]
    #playlist_feature_set_weighted_final['id'] = playlist_feature_set['id']
    
    return playlist_feature_set_weighted_final.sum(axis = 0), complete_feature_set_nonplaylist
```

```python id="dOPMghjlrffr"
complete_feature_set_playlist_vector_EDM, complete_feature_set_nonplaylist_EDM = generate_playlist_feature(complete_feature_set, playlist_EDM, 1.09)
#complete_feature_set_playlist_vector_chill, complete_feature_set_nonplaylist_chill = generate_playlist_feature(complete_feature_set, playlist_chill, 1.09)
```

```python id="4C13MdOzrffr" outputId="d32cf8cc-f9c3-4b01-8e7c-c0a15e812d36"
complete_feature_set_playlist_vector_EDM.shape
```

<!-- #region id="2yMDx0lNrffs" -->
## Generate Recommendations
<!-- #endregion -->

```python id="4NdhN8aqrffs" outputId="14321911-4356-40df-8026-2165b0cbb27f"
from IPython.display import Image
Image("/Users/thakm004/Documents/Spotify/cosine_sim_2.png")
```

```python id="V1kG2EOirffs"
def generate_playlist_recos(df, features, nonplaylist_features):
    """ 
    Pull songs from a specific playlist.

    Parameters: 
        df (pandas dataframe): spotify dataframe
        features (pandas series): summarized playlist feature
        nonplaylist_features (pandas dataframe): feature set of songs that are not in the selected playlist
        
    Returns: 
        non_playlist_df_top_40: Top 40 recommendations for that playlist
    """
    
    non_playlist_df = df[df['id'].isin(nonplaylist_features['id'].values)]
    non_playlist_df['sim'] = cosine_similarity(nonplaylist_features.drop('id', axis = 1).values, features.values.reshape(1, -1))[:,0]
    non_playlist_df_top_40 = non_playlist_df.sort_values('sim',ascending = False).head(40)
    non_playlist_df_top_40['url'] = non_playlist_df_top_40['id'].apply(lambda x: sp.track(x)['album']['images'][1]['url'])
    
    return non_playlist_df_top_40
```

```python id="z8nKGes5rfft"
edm_top40 = generate_playlist_recos(spotify_df, complete_feature_set_playlist_vector_EDM, complete_feature_set_nonplaylist_EDM)
```

```python id="UFjNeaSwrfft" outputId="29a83c6e-d222-4ccf-83b1-ef5c38f7e29f"
from IPython.display import Image
Image("/Users/thakm004/Documents/Spotify/spotify_results.png")
```

```python id="4dcSn8lrrffv" outputId="dab21e09-f58e-4527-d785-ff1ae77a9ea8"
edm_top40
```

```python id="ml2AskS3rffv" outputId="f7d40c6c-0fef-4c12-91df-6e98febe55ce"
visualize_songs(edm_top40)
```

```python id="CiECcFCHrffv"
chill_top40 = generate_playlist_recos(spotify_df, complete_feature_set_playlist_vector_chill, complete_feature_set_nonplaylist_chill)
```
