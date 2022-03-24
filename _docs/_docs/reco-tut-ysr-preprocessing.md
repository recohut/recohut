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

```python id="86MgMsi_GD70" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628266429855, "user_tz": -330, "elapsed": 537, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1b2e3851-20e4-48bd-d10b-781fccc9323e"
import os
project_name = "reco-tut-ysr"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)

if not os.path.exists(project_path):
    !cp /content/drive/MyDrive/mykeys.py /content
    import mykeys
    !rm /content/mykeys.py
    path = "/content/" + project_name; 
    !mkdir "{path}"
    %cd "{path}"
    import sys; sys.path.append(path)
    !git config --global user.email "recotut@recohut.com"
    !git config --global user.name  "reco-tut"
    !git init
    !git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout main
else:
    %cd "{project_path}"
```

```python id="zzFVExkIFzDe" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628268143932, "user_tz": -330, "elapsed": 545, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="49f5df8c-0ab1-4b4c-90f8-6dd6b47df4ed"
!git status
```

```python id="pXWJ6RWXjvEx" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628268153299, "user_tz": -330, "elapsed": 5746, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9358a41b-f188-4a51-b0db-6e62c22848e2"
!git add . && git commit -m 'commit' && git push origin main
```

<!-- #region id="DqVtQ4T7Fz_l" -->
---
<!-- #endregion -->

```python id="XJhLU7p1l0rL" executionInfo={"status": "ok", "timestamp": 1628266963932, "user_tz": -330, "elapsed": 396, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import pandas as pd
import numpy as np
import itertools
import pickle

import warnings
warnings.filterwarnings("ignore")
```

```python id="Yz9ob0RIRjiJ" executionInfo={"status": "ok", "timestamp": 1628267024808, "user_tz": -330, "elapsed": 613, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def readTXT(filename, start_line=0, sep=None):
    with open(filename) as file:
        return [line.rstrip().split(sep) for line in file.readlines()[start_line:]]
```

<!-- #region id="MH6TgBfpTJWc" -->
## Song
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 235} id="F6r-I7RHQC65" executionInfo={"status": "ok", "timestamp": 1628266881661, "user_tz": -330, "elapsed": 449, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2a95ef78-fcfc-4da6-bf6a-7de170d6e283"
songs = pd.read_csv('./data/bronze/song_hash.txt', sep = '\t', header = None,
                    names = ['song_id', 'title', 'artist'], index_col = 0)
songs['artist - title'] = songs['artist'] + " - " + songs['title']
songs.head()
```

<!-- #region id="iq90Hqr2TN1C" -->
## Tag
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="J3H_5GRuRNyS" executionInfo={"status": "ok", "timestamp": 1628267030164, "user_tz": -330, "elapsed": 692, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f444db86-b053-46a1-82fd-f2be814c0cca"
tags = readTXT('./data/bronze/tags.txt')
tags[7:12]
```

<!-- #region id="wxxyL8hkRunw" -->
> Note: # means the song doesn't have any tag. we can replace it with unknown
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 235} id="MGSBBTOUR8eS" executionInfo={"status": "ok", "timestamp": 1628267202404, "user_tz": -330, "elapsed": 1221, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="151a1991-419f-4746-a6de-ce99b432b638"
mapping_tags = dict(readTXT('./data/bronze/tag_hash.txt', sep = ', '))
mapping_tags['#'] = "unknown"
song_tags = pd.DataFrame({'tag_names': [list(map(lambda x: mapping_tags.get(x), t)) for t in tags]})
song_tags.index.name = 'song_id'
song_tags.head()
```

<!-- #region id="SqAnbR-4SGI1" -->
We will consider song tags as a feature of song, so will merge it in songs dataset
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 303} id="OKl1G9luSOkM" executionInfo={"status": "ok", "timestamp": 1628267234905, "user_tz": -330, "elapsed": 419, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c3019821-3069-41db-c702-639391fcc3ea"
songs = pd.merge(left = songs, right = song_tags, how = 'left',
                 left_index = True, right_index = True)
songs.index = songs.index.astype('str')
songs.head()
```

<!-- #region id="glTqs1F9SkIY" -->
We will remove the unknown songs, which doesn't have title and artist. 
<!-- #endregion -->

```python id="8yMpiBl8S0kl" executionInfo={"status": "ok", "timestamp": 1628267360053, "user_tz": -330, "elapsed": 417, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
unknown_songs = songs[(songs['artist'] == '-') | (songs['title'] == '-')]
songs.drop(unknown_songs.index, inplace = True)
```

<!-- #region id="xptAxlqHTFKU" -->
## Playlist
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="is7VAi__S9mL" executionInfo={"status": "ok", "timestamp": 1628267493957, "user_tz": -330, "elapsed": 644, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d0df14f3-d0f9-4be3-cea0-e32d60d0ceef"
playlist = readTXT('./data/bronze/train.txt', start_line = 2) + readTXT('./data/bronze/test.txt', start_line = 2)
print(f'Playlist Count: {len(playlist)}')
```

```python colab={"base_uri": "https://localhost:8080/"} id="DQyIMJsCS9W1" executionInfo={"status": "ok", "timestamp": 1628267494698, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="64189ca1-e285-4595-df42-29b941e154f4"
for i in range(0, 3):
    print("-------------------------")
    print(f"Playlist Idx. {i}: {len(playlist[i])} Songs")
    print("-------------------------")
    print(playlist[i])
```

```python colab={"base_uri": "https://localhost:8080/"} id="N_NHEeESTmlA" executionInfo={"status": "ok", "timestamp": 1628267642291, "user_tz": -330, "elapsed": 508, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d57a60c2-b352-4e50-be02-d62031a88f2a"
# Remove unknown songs from the playlist.
playlist_wo_unknown = [[song_id for song_id in p if song_id not in unknown_songs.index]
                       for p in playlist]

# Remove playlist with zero or one song, since the model wouldn't capture any sequence in that list.
clean_playlist = [p for p in playlist_wo_unknown if len(p) > 1]
print(f"Playlist Count After Cleansing: {len(clean_playlist)}")

# Remove song that doesn't exist in any playlist.
unique_songs = set(itertools.chain.from_iterable(clean_playlist))
song_id_not_exist = set(songs.index) - unique_songs
songs.drop(song_id_not_exist, inplace = True)
print(f"Unique Songs After Cleansing: {songs.shape[0]}")
```

<!-- #region id="nT9BOpmrT-jf" -->
Before there were 75262 unique songs and 15910 playlists. Now we are ready with 73448 unique songs and 15842 playlists.

<!-- #endregion -->

<!-- #region id="3hRGftseUCwP" -->
## Save the artifacts
<!-- #endregion -->

```python id="8D78skgyU6et" executionInfo={"status": "ok", "timestamp": 1628268127001, "user_tz": -330, "elapsed": 1924, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
!mkdir ./data/silver

with open('./data/silver/songs.pickle', 'wb') as handle:
    pickle.dump(songs, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./data/silver/clean_playlist.pickle', 'wb') as handle:
    pickle.dump(clean_playlist, handle, protocol=pickle.HIGHEST_PROTOCOL)
```
