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

```python
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

```python
!git status
```

```python
!git pull --rebase origin main
```

```python
!git add . && git commit -m 'commit' && git push origin main
```

---


## Setup

```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
sns.set_style("whitegrid")

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from scipy import stats

import math
import random
import itertools
import multiprocessing
from tqdm import tqdm
from time import time
import logging
import pickle

import warnings
warnings.filterwarnings("ignore")
```

```python
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

class Callback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 1
        self.training_loss = []

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 1:
            current_loss = loss
        else:
            current_loss = loss - self.loss_previous_step
        print(f"Loss after epoch {self.epoch}: {current_loss}")
        self.training_loss.append(current_loss)
        self.epoch += 1
        self.loss_previous_step = loss
```

```python
with open('./data/silver/songs.pickle', 'rb') as handle:
    songs = pickle.load(handle)

with open('./data/gold/playlist_train.pickle', 'rb') as handle:
    playlist_train = pickle.load(handle)

with open('./data/gold/playlist_test.pickle', 'rb') as handle:
    playlist_test = pickle.load(handle)

callback = Callback()
model = Word2Vec.load('./model/song2vec.model')
```

We are finally ready with the embeddings for every song that exists in playlist_train. How these song vectors are then used to suggest similar songs based on a certain playlist? One way is to calculate a playlist vector for each playlist by averaging together all the song vectors in that playlist. These vectors then become the query to find similar songs based on cosine similarity.


![](https://github.com/sparsh-ai/reco-tut-ysr/raw/main/images/anim_song2vec_average.gif)


For each playlist in playlist_test, calculate the average vectors using meanVectors() function. If the song hasn't been embedded before, neglect the song instead.

```python
def meanVectors(playlist):
    vec = []
    for song_id in playlist:
        try:
            vec.append(model.wv[song_id])
        except KeyError:
            continue
    return np.mean(vec, axis=0)
    
playlist_vec = list(map(meanVectors, playlist_test))
```

For each playlist vector, recommend top n similar songs based on the cosine similarity.

```python
def similarSongsByVector(vec, n = 10, by_name = True):
    # extract most similar songs for the input vector
    similar_songs = model.wv.similar_by_vector(vec, topn = n)
    
    # extract name and similarity score of the similar products
    if by_name:
        similar_songs = [(songs.loc[song_id, "artist - title"], sim)
                              for song_id, sim in similar_songs]
    
    return similar_songs
```

Let's test the song embeddings to recommend top 10 songs for playlist_test in index 305.

```python
def print_recommended_songs(idx, n):
    print("============================")
    print("SONGS PLAYLIST")
    print("============================")
    for song_id in playlist_test[idx]:
        print(songs.loc[song_id, "artist - title"])
    print()
    print("============================")
    print(f"TOP {n} RECOMMENDED SONGS")
    print("============================")
    for song, sim in similarSongsByVector(playlist_vec[idx], n):
        print(f"[Similarity: {sim:.3f}] {song}")
    print("============================")
```

```python
print_recommended_songs(idx = 305, n = 10)
```

Interestingly, the model is able to capture and recommend new songs based on the "Spanish" genre from playlist_test indexed at 305 without being explicitly stated. Great! The final step is to evaluate how this recommender performs.


## Evaluation


### Evaluation strategy

One way to evaluate the performance of a recommender system is by computing hit rate as follows:
1. For each song in a playlist, intentionally Leave-One-Out (LOO) a song.
2. Ask the recommender for top n recommended songs.
3. If the LOO song appears in the top n recommendation, consider it as a HIT. Otherwise not.
4. Repeat the LOO process until the end of the playlist. Then, the hit rate of a playlist is calculated by dividing the number of HIT with the length of a playlist.
5. Repeat step 1-4 for all playlist in playlist_test and calculate the Average Hit Rate at n (AHR@n).

```python
top_n_songs = 25
```

### Random Recommender
As a baseline, let's try to guess the LOO song randomly without any system.

```python
def hitRateRandom(playlist, n_songs, data):
    hit = 0
    for i, target in enumerate(playlist):
        random.seed(i)
        recommended_songs = random.sample(list(data.index), n_songs)
        hit += int(target in recommended_songs)
    return hit/len(playlist)
```

```python
eval_random = pd.Series([hitRateRandom(p, n_songs = top_n_songs, data=songs)
                         for p in tqdm(playlist_test, position=0, leave=True)])
eval_random.mean()
```

### Song Tags Recommender
It is possible to recommend based on song tags provided on the data as follows:

1. Create a list of song tag_names that surrounds the LOO song. The maximum distance between the LOO and context songs is defined by window.
2. List all possible songs from the list.
3. Take n songs randomly from the possible songs list.

```python
mapping_tag2song = songs.explode('tag_names').reset_index().groupby('tag_names')['song_id'].apply(list)
mapping_tag2song
```

```python
def hitRateContextSongTag(playlist, window, n_songs, data, mapping):
    hit = 0
    context_target_list = [([playlist[w] for w in range(idx-window, idx+window+1)
                             if not(w < 0 or w == idx or w >= len(playlist))], target)
                           for idx, target in enumerate(playlist)]
    for i, (context, target) in enumerate(context_target_list):
        context_song_tags = set(data.loc[context, 'tag_names'].explode().values)
        possible_songs_id = set(mapping[context_song_tags].explode().values)
        
        random.seed(i)
        recommended_songs = random.sample(possible_songs_id, n_songs)
        hit += int(target in recommended_songs)
    return hit/len(playlist)
```

```python
eval_song_tag = pd.Series([hitRateContextSongTag(p, model.window, n_songs = top_n_songs, data=songs, mapping=mapping_tag2song)
                           for p in tqdm(playlist_test, position=0, leave=True)])
eval_song_tag.mean()
```

> Warning: The cluster-based method is not working due to Spherical K-means package dependency issue in earlier notebooks.


### Cluster-based Recommender
To improve further, let's utilize the result of clustering in the modeling section:
1. Identify which cluster number is the most frequent (by majority voting) in surrounding songs. The maximum distance between the LOO and context songs is defined by window.
2. List out possible songs from that majority cluster.
3. Take n songs randomly from the possible songs list.
from logic import hitRateClustering

```python
# def hitRateClustering(playlist, window, n_songs,objectmod, model, cluster):
#     hit = 0
#     context_target_list = [([playlist[w] for w in range(idx-window, idx+window+1)
#                              if not(w < 0 or w == idx or w >= len(playlist))], target)
#                            for idx, target in enumerate(playlist)]
#     for context, target in context_target_list:
#         cluster_numbers = objectmod.predict([model.wv[c] for c in context if c in model.wv.vocab.keys()])
#         majority_voting = stats.mode(cluster_numbers).mode[0]
#         possible_songs_id = list(cluster[cluster['cluster'] == majority_voting].index)
#         recommended_songs = random.sample(possible_songs_id, n_songs)
#         songs_id = list(zip(*recommended_songs))[0]
#         hit += int(target in songs_id)
#     return hit/len(playlist)
```

```python
# pd.Series([hitRateClustering(p, model.window, n_songs = top_n_songs, objectmod=skm_opt, model=model,cluster=songs_cluster)
#                            for p in tqdm(playlist_test, position=0, leave=True)])
```

### Song2Vec Recommender
Lastly, evaluate the CBOW Song2Vec model as follows:
1. Take the average vectors of surrounding context songs using previously defined meanVectors() function. The maximum distance is defined by window.
2. Find top n similar songs based on cosine similarity using similarSongsByVector() function.

```python
def hitRateSong2Vec(playlist, window, n_songs):
    hit = 0
    context_target_list = [([playlist[w] for w in range(idx-window, idx+window+1)
                             if not(w < 0 or w == idx or w >= len(playlist))], target)
                           for idx, target in enumerate(playlist)]
    for context, target in context_target_list:
        context_vector = meanVectors(context)
        recommended_songs = similarSongsByVector(context_vector, n = n_songs, by_name = False)
        songs_id = list(zip(*recommended_songs))[0]
        hit += int(target in songs_id)
    return hit/len(playlist)
```

```python
eval_song2vec = pd.Series([hitRateSong2Vec(p, model.window, n_songs = top_n_songs)
                           for p in tqdm(playlist_test, position=0, leave=True)])
eval_song2vec.mean()
```

### Comparison
Finally, we compare the calculated Average Hit Rate at 25 (AHR@25) of the four recommender systems. The higher the AHR, the better is the system. From the bar plot below, Song2Vec outperforms other methods in terms of hit rate, which means that it can recommend a song well based on surrounding context songs. In a real-life scenario, this system may likely to be low quality since the AHR is only around 10%, but still, it is much better than no recommender system at all.

```python
eval_df = pd.concat([eval_random.rename("Random"),
           eval_song_tag.rename("Song Tag"),
        #    eval_clust.rename("Clustering"),
           eval_song2vec.rename("Song2Vec")], axis = 1)
```

```python
g = eval_df.mean().sort_values().plot(kind = 'barh')
g.set_xlabel("Average Hit Rate")
g.set_title("Recommender Evaluation", fontweight = "bold")
plt.show()
```
