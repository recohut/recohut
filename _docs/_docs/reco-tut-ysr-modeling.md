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

```python id="86MgMsi_GD70" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628268219768, "user_tz": -330, "elapsed": 7152, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2f02a811-c2d9-4f02-8f1d-9121c3088f04"
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

```python id="zzFVExkIFzDe"
!git status
```

```python colab={"base_uri": "https://localhost:8080/"} id="yC2oepPVmsVy" executionInfo={"status": "ok", "timestamp": 1628272521702, "user_tz": -330, "elapsed": 2475, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f325a18b-db8d-465f-d349-744c20e3706e"
!git pull --rebase origin main
```

```python id="pXWJ6RWXjvEx" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628272752720, "user_tz": -330, "elapsed": 6142, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c0732db5-b9c7-4073-8e31-f9f63997a5a0"
!git add . && git commit -m 'commit' && git push origin main
```

<!-- #region id="DqVtQ4T7Fz_l" -->
---
<!-- #endregion -->

<!-- #region id="B21ATxcBYQ_7" -->
## Setup
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="iZbtl7g6WjM7" executionInfo={"status": "ok", "timestamp": 1628268306713, "user_tz": -330, "elapsed": 4764, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cd54e633-ebcd-4de9-8a38-f5833ebef12f"
!pip install -q spherecluster
```

```python id="46SQYm77Wb4M"
# Data Analysis
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
plt.style.use('seaborn')
sns.set_style("whitegrid")

# Modeling
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from spherecluster import SphericalKMeans
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from scipy import stats

# Additional
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

<!-- #region id="hhHKmUpfWyUy" -->
If you get this kind of error:
<!-- #endregion -->

<!-- #region id="kMFYh_0-W1Bl" -->
<!-- #endregion -->

<!-- #region id="DGDiZfsoW3DW" -->
Then to handle this, go to spherical_kmeans.py file of this package and replace _k_means with k_means
<!-- #endregion -->

```python id="8D78skgyU6et" executionInfo={"status": "ok", "timestamp": 1628268539950, "user_tz": -330, "elapsed": 795, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
with open('./data/silver/songs.pickle', 'rb') as handle:
    songs = pickle.load(handle)

with open('./data/silver/clean_playlist.pickle', 'rb') as handle:
    clean_playlist = pickle.load(handle)
```

<!-- #region id="TV5DIqAwYPXU" -->
Don't worry about the code below. We are setting up the logging settings to monitor the training process.
<!-- #endregion -->

```python id="BrfmcVftXil4" executionInfo={"status": "ok", "timestamp": 1628268737519, "user_tz": -330, "elapsed": 837, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
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

<!-- #region id="8vv6XgqBZIkX" -->
## Data split
<!-- #endregion -->

<!-- #region id="HJRbjztXZNw1" -->
The playlist is splitted into playlist_train and playlist_test with test size of 1000 playlist for further evaluation.

<!-- #endregion -->

```python id="klVibDQ5ZMQU" executionInfo={"status": "ok", "timestamp": 1628269037827, "user_tz": -330, "elapsed": 740, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
playlist_train, playlist_test = train_test_split(clean_playlist, test_size = 1000,
                                                 shuffle = True, random_state = 123)
```

```python id="fLUtdE3GnWQ9" executionInfo={"status": "ok", "timestamp": 1628272736518, "user_tz": -330, "elapsed": 1701, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
with open('./data/gold/playlist_train.pickle', 'wb') as handle:
    pickle.dump(playlist_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./data/gold/playlist_test.pickle', 'wb') as handle:
    pickle.dump(playlist_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

<!-- #region id="ZvRErFBgYwj7" -->
## Training
<!-- #endregion -->

<!-- #region id="t1tZDW6KYS1d" -->
By using gensim, the training process can be separated into three distinctive steps:

First, the instance of Word2Vec() is created to set up the parameters of the model and leave the model uninitialized.

- size: dimensionality of the song vectors
- window: maximum distance between context and target
- min_count: frequency cut-off for a song to be considered in the model
- sg = 0: using CBOW architecture
- negative: negative sampling data
- workers: number of CPU used to train the model
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="6IrP_kx4YeJ6" executionInfo={"status": "ok", "timestamp": 1628268902154, "user_tz": -330, "elapsed": 753, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="73505c3b-f58b-4367-cabc-c85eca8a10c0"
model = Word2Vec(
    size = 256,
    window = 10,
    min_count = 1,
    sg = 0,
    negative = 20,
    workers = multiprocessing.cpu_count()-1)

print(model)
```

<!-- #region id="_Hk7AOZHY6_a" -->
Secondly, the method .build_vocab() is called to build the vocabulary from a sequence of playlists and thus initialized the model.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="C0NLioFfY_ug" executionInfo={"status": "ok", "timestamp": 1628269063771, "user_tz": -330, "elapsed": 17143, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="83b5d4ab-9b11-4528-e6c8-241c0a80727e"
logging.disable(logging.NOTSET) # enable logging
t = time()

model.build_vocab(playlist_train)

print(f"Time to build vocab: {round((time() - t), 2)} seconds")
```

<!-- #region id="IIjTOfX1ZwI5" -->
Finally, .train() trains the model. The loggings here are mainly useful for monitoring the loss after each epoch.

- total_examples: count of unique vocabulary (songs)
- epochs: number of iterations over the dataset (whole playlist)
- compute_loss: track model loss
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="sKVK-tU0ZyRy" executionInfo={"status": "ok", "timestamp": 1628271411681, "user_tz": -330, "elapsed": 1526369, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a2dd3b8d-e8bc-496a-ddef-2495a49089a8"
logging.disable(logging.INFO) # disable logging
callback = Callback() # instead, print out loss for each epoch
t = time()

model.train(playlist_train,
            total_examples = model.corpus_count,
            epochs = 100,
            compute_loss = True,
            callbacks = [callback]) 

print(f"Time to train the model: {round((time() - t), 2)} seconds")
```

```python colab={"base_uri": "https://localhost:8080/"} id="kXogrwPmZ7Xj" executionInfo={"status": "ok", "timestamp": 1628272293062, "user_tz": -330, "elapsed": 640, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0b2e7110-0e49-4280-e93f-ade901226332"
print(model)
```

```python id="LI5-m8leaD6H" executionInfo={"status": "ok", "timestamp": 1628272296748, "user_tz": -330, "elapsed": 846, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
model.save('./model/song2vec.model')
```

<!-- #region id="2aWAXVHkaLlc" -->
## Evaluation
<!-- #endregion -->

<!-- #region id="GJAsJNTaaf1r" -->
### Loss evaluation
<!-- #endregion -->

<!-- #region id="pR1vCFjsaVJ0" -->
Plot the training loss, making sure it decreases after each epoch. The closer the loss to a zero value, the better the model is in predicting a target song given surrounding context songs. Thus, the produced song vectors are more meaningful.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 377} id="F35Z_i66aMwj" executionInfo={"status": "ok", "timestamp": 1628272302288, "user_tz": -330, "elapsed": 921, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2ef704d0-eaec-429a-c1c9-5d357936ec9a"
plt.plot(range(1, model.epochs+1), model.callbacks[0].training_loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss", fontweight = "bold")
plt.show()
```

<!-- #region id="-vNugQ-van5I" -->
### Vector visualization
<!-- #endregion -->

<!-- #region id="LPDPsozban23" -->
The song vectors can be visualized using a gradient of colors. The model is trained using 256 dimensions, therefore there will be 256 color bars for each song, representing element values in the vector. The similarity between songs is calculated using cosine similarity:
<!-- #endregion -->

<!-- #region id="GclAGWGvanzs" -->
Mathematically it measures the cosine of the angle between two vectors A and B which projected in a multi-dimensional space. Song vectors with similar context occupy close spatial positions; the cosine between such vectors should be close to 1, i.e. angle is closer to 0. The smaller the angle, the cosine similarity will be higher.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 536} id="JK3tw_GpanwL" executionInfo={"status": "ok", "timestamp": 1628272311542, "user_tz": -330, "elapsed": 3663, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e3a5925c-d758-4337-b5c8-3fb925c6551b"
fig, axes = plt.subplots(6, 1, figsize = (50, 30))

slug = '4162'
song_id_list = [(slug, "Main Song"), *[t for t in model.wv.most_similar(slug)[:5]]] 

for ax, (song_id, sim) in zip(axes.flat, song_id_list):
    ax.imshow([model.wv[song_id]], cmap = "binary", aspect = "auto")
    ax.set_title(songs.loc[song_id, "artist - title"], fontsize = 50)
    ax.yaxis.set_label_position("right")
    ax.set_ylabel(f"Similarity:\n{sim:.3f}" if sim != song_id_list[0][1] else sim,
                  rotation = "horizontal", ha = "left", va = "center", fontsize = 50)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
```

<!-- #region id="BjVS5u32bqdW" -->
The plot above shows five most similar songs to song_id = '4162' (Maroon 5 - She Will Be Loved). Up until now, the model can be used for recommending new songs using cosine similarity, but only based on one main song.
<!-- #endregion -->

<!-- #region id="i2baEF2du5HI" -->
> Warning: The below section is not working due to Spherical K-means package dependency issue. If you have any alternative or idea to resolve, please let me know.
<!-- #endregion -->

<!-- #region id="fQIedrm3epJO" -->
### Clustering

What can we do with the song vectors? One thing is to group them into several clusters using K-Means clustering, but keep in mind that the similarity between vectors is calculated using cosine distance instead of regular (Euclidean) distance. Therefore K-Means with cosine distance should be considered, which often called Spherical K-Means Clustering. The idea is to identify the centroid such that it uniforms and minimizes the angle between each vector in a cluster. The intuition is just like looking at a cluster of stars where each point should have consistent spacing between each other. This spacing is referred to as the cosine similarity.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="lPafZCzXer4n" executionInfo={"status": "ok", "timestamp": 1628272320761, "user_tz": -330, "elapsed": 1550, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f30bfd9c-b33e-4cf4-b595-89d994f035ad"
embedding_matrix = model.wv[model.wv.vocab.keys()]
embedding_matrix.shape
```

```python id="JQFKPbmcew8A"
range_k_clusters = (10, 500)
skm_list = []
for k in tqdm(range(*range_k_clusters, 10)):
    skm = SphericalKMeans(n_clusters = k,
                          n_init = 5, n_jobs = -1,
                          random_state = 123).fit(embedding_matrix)
    
    result_dict = {
        "k": k,
        "WCSS": skm.inertia_,
        "skm_object": skm
    }
    
    skm_list.append(result_dict)
    
skm_df = pd.DataFrame(skm_list).set_index('k')
skm_df.head()
```

```python id="NbAfRAGre795"
# function to find k 
def locateOptimalElbow(x, y):
    # START AND FINAL POINTS
    p1 = (x[0], y[0])
    p2 = (x[-1], y[-1])
    
    # EQUATION OF LINE: y = mx + c
    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    c = (p2[1] - (m * p2[0]))
    
    # DISTANCE FROM EACH POINTS TO LINE mx - y + c = 0
    a, b = m, -1
    dist = np.array([abs(a*x0+b*y0+c)/math.sqrt(a**2+b**2) for x0, y0 in zip(x,y)])
    return x[np.argmax(dist)]
```

```python id="o6sgDdxke_es"
skm_df.WCSS.plot()
plt.xlabel("No. of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method", fontweight = "bold")
plt.show()
```

<!-- #region id="Sk4lQxYJfMdA" -->
How to locate the optimal number of clusters objectively? Here is the idea:
<!-- #endregion -->

<!-- #region id="pcOSAEsKfODG" -->
<!-- #endregion -->

```python id="hpQoa7Z4fKe0"
k_opt = locateOptimalElbow(skm_df.index, skm_df['WCSS'].values)
skm_opt = skm_df.loc[k_opt, "skm_object"]
skm_opt
```

```python id="VS5T2gHcfRd-"
songs_cluster = songs.copy()
songs_cluster.loc[model.wv.vocab.keys(), 'cluster'] = skm_opt.labels_
songs_cluster['cluster'] = songs_cluster['cluster'].fillna(-1).astype('int').astype('category')
```

<!-- #region id="ZIu3NpadfU7C" -->
In the end, the optimal number of clusters is set to be 110. There is a possibility that some songs don't have the embedded vectors since the playlist is split to train and test. For this case, assign the cluster as -1 instead.
<!-- #endregion -->

<!-- #region id="bP8MHpWaffIr" -->
### Visualize Clusters
<!-- #endregion -->

<!-- #region id="KlgAobGDfh3R" -->
It is always quite helpful to visualize the embeddings that have been created. Over here, we have song vectors with 256 dimensions. These high-dimensional vectors can't be visualized in our 3D world, so using dimensionality reduction algorithms such as t-Distributed Stochastic Neighbor Embedding (t-SNE) helps us map the vectors to a lower dimension. The mathematical detail of t-SNE will not be presented here, but in practice, it tends to produce a visualization with distinctly isolated clusters.
<!-- #endregion -->

```python id="v9RGpnaKfVZk"
embedding_tsne = TSNE(n_components = 2, metric = 'cosine',
                      random_state = 123).fit_transform(embedding_matrix)
songs_cluster.loc[model.wv.vocab.keys(), 'x'] = embedding_tsne[:,0]
songs_cluster.loc[model.wv.vocab.keys(), 'y'] = embedding_tsne[:,1]
sns.scatterplot(data = songs_cluster[songs_cluster['cluster'] != -1],
                x = 'x', y = 'y', palette = "viridis",
                hue = 'cluster', legend = False).set_title(f"{k_opt} Clusters of Song2Vec",
                                                           fontweight = "bold")
plt.show()
```

<!-- #region id="Exb8MWA4ft1a" -->
The cluster might look cluttered since all 110 clusters are being plotted at once. Instead, let's just perform t-SNE on randomly selected 10 clusters and visualize the result.
<!-- #endregion -->

```python id="2jovo62BfsFO"
random.seed(100)
random_cluster2plot = random.sample(range(k_opt), 10)
random_songs = songs_cluster[songs_cluster.cluster.isin(random_cluster2plot)].copy()
random_tsne = TSNE(n_components = 2, metric = 'cosine',
                   random_state = 100).fit_transform(model.wv[random_songs.index])
random_songs.loc[random_songs.index, 'x'] = random_tsne[:,0]
random_songs.loc[random_songs.index, 'y'] = random_tsne[:,1]

g = sns.scatterplot(data = random_songs,
                x = 'x', y = 'y', palette = "viridis",
                hue = 'cluster')
g.legend(loc = "upper left", bbox_to_anchor = (1, 1))
g.set_title(f"Randomly selected {len(random_cluster2plot)} clusters of Song2Vec", fontweight = "bold")
plt.show()
```

<!-- #region id="aL44fDjrf18h" -->
Songs that have similar context (by cosine similarity) tend to be plotted next to each other. Thus, creating distinct song clusters.
 
> Note: clusters might look overlap to each other due to the dimensionality reduction, but in the actual dimension, they do not.
<!-- #endregion -->
