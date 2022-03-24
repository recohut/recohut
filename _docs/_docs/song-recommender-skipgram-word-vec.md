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

<!-- #region id="xD8tM1pd2NAu" -->
# Song Embeddings - Skipgram Recommender
> In this notebook, we'll use human-made music playlists to learn song embeddings. We'll treat a playlist as if it's a sentence and the songs it contains as words. We feed that to the word2vec algorithm which then learns embeddings for every song we have. These embeddings can then be used to recommend similar songs.

- toc: true
- badges: true
- comments: true
- categories: [Word2Vec, Embedding, Music, Sequence]
- author: "<a href='https://github.com/jalammar/jalammar.github.io'>Jay Alammar</a>"
- image:
<!-- #endregion -->

<!-- #region id="yucM-N8R2VaN" -->
This technique is used by Spotify, AirBnB, Alibaba, and others. It accounts for a vast portion of their user activity, user media consumption, and/or sales (in the case of Alibaba). The dataset we'll use was collected by Shuo Chen from Cornell University. The [dataset](https://www.cs.cornell.edu/~shuochen/lme/data_page.html) contains playlists from hundreds of radio stations from around the US.
<!-- #endregion -->

<!-- #region id="sHZo-F6427A5" -->
## Downloading data
<!-- #endregion -->

```python id="wu70B8qy-zj1"
!wget -q https://www.cs.cornell.edu/~shuochen/lme/dataset.tar.gz
!tar -xf dataset.tar.gz
```

<!-- #region id="otMF4HUD3bg2" -->
## Setup
<!-- #endregion -->

```python id="pV4KRkN4gRec"
import numpy as np
import pandas as pd
import gensim 
from gensim.models import Word2Vec
from urllib import request
```

```python id="duIxrTrfgNPt"
import warnings
warnings.filterwarnings('ignore')
```

<!-- #region id="XhjskIqm3X4e" -->
## Training dataset
<!-- #endregion -->

```python id="AWj3LOqmgmGh"
with open("/content/dataset/yes_complete/train.txt", 'r') as f:
  # skipping first 2 lines as they contain only metadata
  lines = f.read().split('\n')[2:]
  # select playlists with at least 2 songs, a minimum threshold for sequence learning 
  playlists = [s.rstrip().split() for s in lines if len(s.split()) > 1]
```

```python colab={"base_uri": "https://localhost:8080/"} id="JbNTSuUHhQar" executionInfo={"status": "ok", "timestamp": 1625905556323, "user_tz": -330, "elapsed": 652, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fb7a7f90-898e-40ed-8fc2-ef7d3e0ed6a4"
print( 'Playlist #1:\n ', playlists[0], '\n')
print( 'Playlist #2:\n ', playlists[1])
```

<!-- #region id="7ulE0Sei3A5P" -->
## Training Word2vec
<!-- #endregion -->

<!-- #region id="iLphAx7n3guD" -->
Our dataset is now in the shape the the Word2Vec model expects as input. We pass the dataset to the model, and set the following key parameters:

- size: Embedding size for the songs.
- window: word2vec algorithm parameter -- maximum distance between the current and predicted word (song) within a sentence
- negative: word2vec algorithm parameter -- Number of negative examples to use at each training step that the model needs to identify as noise
<!-- #endregion -->

```python id="c3ETOmmrhm9c"
model = Word2Vec(playlists, size=32, window=20, negative=50, min_count=1, workers=-1)
```

<!-- #region id="FcxLNCTU3lH6" -->
The model is now trained. Every song has an embedding. We only have song IDs, though, no titles or other info. Let's grab the song information file.
<!-- #endregion -->

<!-- #region id="HVJexXVa3DmX" -->
## Prepare songs metadata
<!-- #endregion -->

<!-- #region id="Z2NaoFA23PUS" -->
### Title and artist
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="WIz9OQcHfDzd" executionInfo={"status": "ok", "timestamp": 1625905203911, "user_tz": -330, "elapsed": 523, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="09f74766-35a0-4d5d-8f85-b2ffa3c9861e"
!head /content/dataset/yes_complete/song_hash.txt
```

```python id="NlIxHUCiiFnP"
with open("/content/dataset/yes_complete/song_hash.txt", 'r') as f:
  songs_file = f.read().split('\n')
  songs = [s.rstrip().split('\t') for s in songs_file]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 235} id="OyN_asrNitmC" executionInfo={"status": "ok", "timestamp": 1625905648538, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fdf09d8e-08ed-4e22-d979-20b44fe5f4fc"
songs_df = pd.DataFrame(data=songs, columns = ['id', 'title', 'artist'])
songs_df = songs_df.set_index('id')
songs_df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 173} id="vhW9oqJAn6wn" executionInfo={"status": "ok", "timestamp": 1625907246433, "user_tz": -330, "elapsed": 421, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0963aa4a-7c8e-44ad-c3b1-dc80f9b2aeda"
songs_df.iloc[[1,10,100]]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 235} id="4vEME6TgqIcl" executionInfo={"status": "ok", "timestamp": 1625907827879, "user_tz": -330, "elapsed": 439, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="493e5e4e-37bc-4894-eba1-e16857d7af2b"
songs_df[songs_df.artist == 'Rush'].head()
```

<!-- #region id="QnE1OWrx3TIU" -->
### Tags
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="jfLq2JosfHfp" executionInfo={"status": "ok", "timestamp": 1625905206542, "user_tz": -330, "elapsed": 529, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="14b6c27c-8239-4e42-b0fd-3546ba0ac379"
!head /content/dataset/yes_complete/tag_hash.txt
```

```python id="DMBHRM6mh_5N"
with open("/content/dataset/yes_complete/tag_hash.txt", 'r') as f:
  tags_file = f.read().split('\n')
  tags = [s.rstrip().split(',') for s in tags_file]
  tag_name = {a:b.strip() for a,b in tags}
  tag_name['#'] = 'no tag'
```

```python colab={"base_uri": "https://localhost:8080/"} id="WyKDo9aql0KJ" executionInfo={"status": "ok", "timestamp": 1625908586442, "user_tz": -330, "elapsed": 459, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f4af9da6-6237-4ef9-b3d2-77a2b25541ae"
print('Tag name for tag id {} is "{}"\n'.format('10', tag_name['10']))
print('Tag name for tag id {} is "{}"\n'.format('80', tag_name['80']))
print('There are total {} tags'.format(len(tag_name.items())))
```

```python colab={"base_uri": "https://localhost:8080/"} id="s9fh6c7L_Dbv" executionInfo={"status": "ok", "timestamp": 1625906362834, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8c8e29cf-43d9-4332-9de2-b21d03fb2c4a"
!head /content/dataset/yes_complete/tags.txt
```

```python id="7nR06ogPjidw"
with open("/content/dataset/yes_complete/tags.txt", 'r') as f:
  song_tags = f.read().split('\n')
  song_tags = [s.split(' ') for s in song_tags]
  song_tags = {a:b for a,b in enumerate(song_tags)}
```

```python id="_XGMxG7el4ib"
def tags_for_song(song_id=0):
  tag_ids = song_tags[int(song_id)]
  return [tag_name[tag_id] for tag_id in tag_ids]
```

```python colab={"base_uri": "https://localhost:8080/"} id="gm6i0HlMk6jZ" executionInfo={"status": "ok", "timestamp": 1625907197524, "user_tz": -330, "elapsed": 405, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="65da0c9a-3225-4969-d8a8-25c4b936a374"
print('Tags for song "{}" : {}\n'.format(songs_df.iloc[0].title, tags_for_song(0)))
```

<!-- #region id="iyqhTWtn3Jil" -->
## Recommend
<!-- #endregion -->

```python id="xlGQ4iygoL2i"
def recommend(song_id=0, topn=5):
  # song info
  song_info = songs_df.iloc[song_id]
  song_tags = [', '.join(tags_for_song(song_id))]
  query_song = pd.DataFrame({'title':song_info.title,
                             'artist':song_info.artist,
                             'tags':song_tags})

  # similar songs
  similar_songs = np.array(model.wv.most_similar(positive=str(song_id), topn=topn))[:,0]
  recommendations = songs_df.iloc[similar_songs]
  recommendations['tags'] = [tags_for_song(i) for i in similar_songs]

  recommendations = pd.concat([query_song, recommendations])

  axis_name = ['Query'] + ['Recommendation '+str((i+1)) for i in range(topn)]
  # recommendations.index = axis_name
  recommendations = recommendations.style.set_table_styles([{'selector': 'th', 'props': [('background-color', 'gray')]}])
  
  return recommendations
```

```python colab={"base_uri": "https://localhost:8080/", "height": 387} id="F0qBm76OtzBV" executionInfo={"status": "ok", "timestamp": 1625910353643, "user_tz": -330, "elapsed": 423, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="06120e62-fecb-4b27-d821-cc9f9169092e"
recs = recommend(10)
recs
```

<!-- #region id="qYFrMdrZzlFD" -->
### Paranoid Android - Radiohead
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 370} id="bctpVSM3znHy" executionInfo={"status": "ok", "timestamp": 1625910888360, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3c5d212d-eac5-4172-a79f-1c32004d807f"
recommend(song_id=19563)
```

<!-- #region id="4oYa1QAT14DD" -->
### California Love - 2Pac
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 268} id="4DjOvrZb16LE" executionInfo={"status": "ok", "timestamp": 1625910943099, "user_tz": -330, "elapsed": 466, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="026fa5ed-53d0-4c80-8af3-1d29d8c50ca5"
recommend(song_id=842)
```

<!-- #region id="Uw6x-wFV17m5" -->
### Billie Jean - Michael Jackson
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 370} id="X7nA4NWdz22W" executionInfo={"status": "ok", "timestamp": 1625910953058, "user_tz": -330, "elapsed": 896, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="095ba4ad-56b9-4c63-e7ea-b6e6d4087ad1"
recommend(song_id=3822)
```
