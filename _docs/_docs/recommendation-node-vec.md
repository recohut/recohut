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
    language: python
    name: python3
---

<!-- #region id="QB8Kjuy1ApsF" -->
# Recommender System with Node2vec Graph Embeddings
> A tutorial on building a movie recommender system that will learn user-item representation using graph embedding and comparing performance with other methods like matrix factorization

- toc: true
- badges: true
- comments: true
- categories: [graph, embedding]
- image: 
<!-- #endregion -->

<!-- #region id="duOegrbCH_Uj" -->
## Data gathering and exploration
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1904, "status": "ok", "timestamp": 1619255013350, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="M10NbHleDF96" outputId="eb611fd5-dbc7-4e63-aef8-fd8332ed1157"
#hide
!wget https://raw.githubusercontent.com/sparsh-ai/rec-data-public/master/ml-other/ml100k_ratings.csv
!wget https://raw.githubusercontent.com/sparsh-ai/rec-data-public/master/ml-other/ml100k_movies.csv
```

```python executionInfo={"elapsed": 1611, "status": "ok", "timestamp": 1619264673140, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="dW3-0phsDrpc"
#hide
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import svds, norm
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import operator
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
```

```python executionInfo={"elapsed": 1817, "status": "ok", "timestamp": 1619264673561, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="FUzXL9-PpnYs"
#hide
def print_stats(df, uid=1):
  print(df.shape)
  print(df.movieId.nunique())
  print(max(df.movieId))
  if uid:
    print(df.userId.nunique())
    print(max(df.userId))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"elapsed": 1701, "status": "ok", "timestamp": 1619264673562, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="QsIUlasGEG1w" outputId="34d75e94-4073-44c0-8197-d6c34290169c"
rating_df = pd.read_csv('ml100k_ratings.csv', sep=',', header=0)
rating_df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1562, "status": "ok", "timestamp": 1619264673564, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="ZODEGVn1F08H" outputId="b145c97a-94cc-4215-8ca8-39fb47644364"
#hide
print_stats(rating_df)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"elapsed": 1466, "status": "ok", "timestamp": 1619264673565, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="KKxem0FVF_SG" outputId="8485b66d-e917-4ae0-89ac-e7f397f1df60"
movie_df = pd.read_csv('ml100k_movies.csv', sep=',', header=0)
movie_df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1276, "status": "ok", "timestamp": 1619264673568, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="ZmTFFculp52P" outputId="240789c7-6045-419a-ac89-e038938087f1"
#hide
print_stats(movie_df, uid=0)
```

<!-- #region id="XWnDNTepIDmT" -->
## Neighborhood method
<!-- #endregion -->

<!-- #region id="pr1CHpAT-0wZ" -->
### Jaccard Similarity
<!-- #endregion -->

<!-- #region id="LPqcRWFCIrrf" -->
If we ignore the ratings that the users have given to the movies, and consider the movies that the users have watched, we get a set of movies/users for every user/movie. Think of this formulation as a bipartite graph of users and movies where there is an edge between a user and a movie if a user has watched the movie, the edges have all same weights.

Create a dictionary of movies as keys and values as users that have rated them
<!-- #endregion -->

```python executionInfo={"elapsed": 985, "status": "ok", "timestamp": 1619264675613, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="B3NvTMTsGgIW"
#hide
movie_sets = dict((movie, set(users)) for movie, users in rating_df.groupby('movieId')['userId'])
```

<!-- #region id="dyUS2QlFJSc_" -->
Since we have a set of users to characterize each movie, to compute the similarity of two movies, we use Jaccard Index which, for two sets, is the ratio of number of elements in the intersection and number of elements in the union.


<!-- #endregion -->

```python executionInfo={"elapsed": 1049, "status": "ok", "timestamp": 1619264677003, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="mtga_4oHJSAI"
#collapse
def jaccard(movie1, movie2, movie_sets):
    a = movie_sets[movie1]
    b = movie_sets[movie2]
    intersection = float(len(a.intersection(b)))
    return intersection / (len(a) + len(b) - intersection)
```

<!-- #region id="_rXFwdZLJXWN" -->
Let's explore similarity between some movies, qualitatively. We use the movies dataframe to get the names of the movies via their Ids.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"elapsed": 1353, "status": "ok", "timestamp": 1619264679358, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="E_1TCARUK2Ci" outputId="53cd76cf-80bb-4375-d7b2-fbabf26b7325"
movie_df[movie_df.movieId == 260].title.values[0]
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 874, "status": "ok", "timestamp": 1619264683028, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="q-9x2b28JZkT" outputId="6349417d-5c7c-4492-b19c-2ba56d12b7b6"
#hide-input
title = movie_df[movie_df.movieId == 260].title.values[0]
title = ''.join(title.split())

print("Jaccard distance between '%s' and '%s' is %.2f"%(
    title, 
     ''.join(movie_df[movie_df.movieId == 1196].title.values[0].split()), 
    jaccard(260, 1196, movie_sets)))

print("Jaccard distance between '%s' and '%s' is %.2f"%(
    title, 
    ''.join(movie_df[movie_df.movieId == 1210].title.values[0].split()),
    jaccard(260, 1210, movie_sets)))

print("Jaccard distance between '%s' and '%s' is %.2f"%(
    title, 
    ''.join(movie_df[movie_df.movieId == 1].title.values[0].split()),
    jaccard(260, 1, movie_sets)))
```

<!-- #region id="igqVmGzvNLoc" -->
The movie Star Wars IV has higher similarity score with other Star Wars as compared to Toy Story.

Using the Jaccard Index, we can retrieve top-k similar movies to a given movie. This provides a way to recommend movies of a user which are similar to the movies that the user has watched.
<!-- #endregion -->

```python executionInfo={"elapsed": 3296, "status": "ok", "timestamp": 1619264688560, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="wRx6FxiuJf7G"
#hide
import operator 

def get_similar_movies_jaccard(movieid, movie_sets, top_n=5):
    movie = movie_df[movie_df.movieId == movieid].title.values[0]
    jaccard_dict = {x: jaccard(x, movieid, movie_sets) for x in movie_sets}
    ranked_movies = sorted(jaccard_dict.items(), key=operator.itemgetter(1), reverse=True)[:top_n]
    sim_movies = [movie_df[movie_df.movieId == id[0]].title.values[0] for id in ranked_movies]
    return {'movie': movie, 'sim_movies': sim_movies}
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3041, "status": "ok", "timestamp": 1619264688562, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="Ogn6T_5aNVbo" outputId="d61c71f8-2b81-4194-f088-782a2356dc5b"
get_similar_movies_jaccard(260, movie_sets)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2967, "status": "ok", "timestamp": 1619264688563, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="fPNAPxqlNWx-" outputId="09919429-3669-45db-f02f-f2620ae3c776"
get_similar_movies_jaccard(1, movie_sets)
```

<!-- #region id="PDCLeFgJNnxN" -->
### Cosine similarity
<!-- #endregion -->

<!-- #region id="jBZTwurfNmWq" -->
Rather than the set based similarity like Jaccard, we can define every movie as a sparse vector of dimension equal to the number of users and the vector entry corresponding to each user is given by the rating that the user has for the movie or zero if no rating exists (i.e. the user hasn't seen/rated the movie).
<!-- #endregion -->

```python executionInfo={"elapsed": 1501, "status": "ok", "timestamp": 1619264689164, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="TjNjq4YeNarR"
#hide
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1377, "status": "ok", "timestamp": 1619264689166, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="kGcfDDr9O2OF" outputId="5a80fffc-f761-44fc-96c1-4bb43a9adae4"
#hide
num_users = rating_df.userId.nunique()
num_users
```

```python executionInfo={"elapsed": 10293, "status": "ok", "timestamp": 1619264698190, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="lzyItagFO5PI"
#hide
movie_sparse_vecs = []
movies = []
for movie, group in rating_df.groupby('movieId'):
    vec = [0] * num_users
    for x in group[['userId', 'rating']].values:
        vec[int(x[0]) - 1] = x[1]
    movie_sparse_vecs.append(vec)
    movies.append(movie)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 8922, "status": "ok", "timestamp": 1619264698908, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="C7NgHmrZPDxU" outputId="7a9d61cb-7456-4cad-83e9-66d0a3073171"
#hide
movie_sparse_vecs = np.array(movie_sparse_vecs)
print(movie_sparse_vecs.shape)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 8720, "status": "ok", "timestamp": 1619264698910, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="Hd9m0ZIpPdmV" outputId="2049cd8c-5ed0-45a6-bbef-7e0d589469f1"
print(1.0 - cosine(movie_sparse_vecs[224], movie_sparse_vecs[897]))
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 8358, "status": "ok", "timestamp": 1619264698911, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="ZP4K4AIxPD8T" outputId="61630162-b1e6-4956-c632-10abe957225a"
#hide
movie2id = {x:i for i,x in enumerate(movies)}
movie2id[260]
```

```python executionInfo={"elapsed": 4120, "status": "ok", "timestamp": 1619264698912, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="zIMlL-AUPGol"
#collapse
def get_similar_movies_nbd_cosine(movieid, movie_vecs, top_n=5):
    movie = movie_df[movie_df.movieId == movieid].title.values[0]
    movie_idx = movie2id[movieid]
    query = movie_vecs[movie_idx].reshape(1,-1)
    ranking = cosine_similarity(movie_vecs,query)
    top_ids = np.argsort(ranking, axis=0)
    top_ids = top_ids[::-1][:top_n]
    top_movie_ids = [movies[j[0]] for j in top_ids]
    sim_movies = [movie_df[movie_df.movieId == id].title.values[0] for id in top_movie_ids]
    return {'movie': movie, 'sim_movies': sim_movies}
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3060, "status": "ok", "timestamp": 1619264701563, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="oat5HmtoPJdW" outputId="b43fd8e2-a494-4c7c-deca-9311310a6b0e"
movieid = 1
movie_data = movie_sparse_vecs
get_similar_movies_nbd_cosine(movieid, movie_data, top_n=5)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2980, "status": "ok", "timestamp": 1619264701564, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="eMz9kFZrQKUv" outputId="9e9838ab-60df-4bfc-b578-59dd9fa56a75"
movieid = 260
movie_data = movie_sparse_vecs
get_similar_movies_nbd_cosine(movieid, movie_data, top_n=5)
```

<!-- #region id="pw86BbsJ_gF7" -->
## Factorization method
<!-- #endregion -->

<!-- #region id="NeaTEXE4Q6Yz" -->
### Singular Value Decomposition
<!-- #endregion -->

<!-- #region id="7EqGRvLcQ_5Z" -->
A very popular technique for recommendation systems is via matrix factorization. The idea is to reduce the dimensionality of the data before calculating similar movies/users. We factorize the user-item matrix to obtain the user factors and item factors which are the low-dimensional embeddings such that 'similar' user/items are mapped to 'nearby' points.

This kind of analysis can generate matches that are impossible to find with the techniques discussed above as the latent factors can capture attributes which are hard for raw data to deciper e.g. a latent factor can correspond to the degree to which a movie is female oriented or degree to which there is a slow development of the charcters.

Moreover, the user and the movies are embedded to the same space, which provides a direct way to compute user-movie similarity.

We will use Singular Value Decomposition (SVD) for factorizing the matrix.
<!-- #endregion -->

```python executionInfo={"elapsed": 1630, "status": "ok", "timestamp": 1619264758446, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="wqmWohqGsqss"
#hide
le_movie = LabelEncoder()
movie_df = movie_df[movie_df.movieId.isin(rating_df.movieId.unique())]
rating_df.loc[:, 'movieId'] = le_movie.fit_transform(rating_df.loc[:, 'movieId'])
rating_df.loc[:, 'movieId']+=1
movie_df.loc[:, 'movieId'] = le_movie.transform(movie_df.loc[:, 'movieId'])
movie_df.loc[:, 'movieId']+=1
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1448, "status": "ok", "timestamp": 1619264758448, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="yBA0Ha2MQKf2" outputId="e6cd2dae-1723-449f-c01b-7717373cc548"
#hide
ratings_mat = np.ndarray(
    shape=(np.max(rating_df.movieId.values), np.max(rating_df.userId.values)),
    dtype=np.uint8)
ratings_mat[rating_df.movieId.values-1, rating_df.userId.values-1] = rating_df.rating.values
ratings_mat.shape
```

<!-- #region id="sFDqtsgQRFkD" -->
Normalize the rating matrix


<!-- #endregion -->

```python executionInfo={"elapsed": 709, "status": "ok", "timestamp": 1619264761028, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="pdCoya1BRDvC"
normalised_mat = ratings_mat - np.asarray([(np.mean(ratings_mat, 1))]).T
```

<!-- #region id="_z-9rYIKRP8W" -->
The number of the latent-factors is chosen to be 50 i.e. top-50 singular values of the SVD are considered. The choice of the number of latent factors is a hyperparameter of the model, and requires a more sophisticated analysis to tune. We provide no reason for the choice of 50.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3083, "status": "ok", "timestamp": 1619264763734, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="01TPkUiTROKT" outputId="3600670d-7269-435b-bf86-a3eafa269ddd"
n_factors = 50

A = normalised_mat.T / np.sqrt(ratings_mat.shape[0] - 1)
U, S, V = svds(A, n_factors)

print(U.shape, V.shape)
```

```python executionInfo={"elapsed": 1519, "status": "ok", "timestamp": 1619264763736, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="ICeVcjCARW-U"
movie_factors = V.T
user_factors = U
```

<!-- #region id="gbBy8vcOSBsR" -->
Instead of representing each movie as a sparse vector of the ratings of all 360,000 possible users for it, after factorizing the matrix each movie will be represented by a 50 dimensional dense vector.

Define a routine to get top-n movies similar to a given movie.

<!-- #endregion -->

```python executionInfo={"elapsed": 1012, "status": "ok", "timestamp": 1619264767611, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="pE51HtPzRZ--"
#collapse
def get_similar_movies_matrix_factorization(data, movieid, top_n=10):
    index = movieid - 1 # Movie id starts from 1
    movie = movie_df[movie_df.movieId == movieid].title.values[0]
    movie_row = data[index].reshape(1,-1)
    similarity = cosine_similarity(movie_row, data)
    sort_indexes = np.argsort(-similarity)[0]
    return {'movie': movie, 'sim_movies': [movie_df[movie_df.movieId == id].title.values[0] for id in sort_indexes[:top_n] + 1]}
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1353, "status": "ok", "timestamp": 1619264768173, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="E8nbbpG0SJ50" outputId="c3c5ccb6-9e20-41ba-864e-f7e3972e0beb"
movie_id = 260
get_similar_movies_matrix_factorization(movie_factors, movie_id)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1025, "status": "ok", "timestamp": 1619264770791, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="6PHHeIzhSMlr" outputId="d568a692-e449-4497-8cb0-458d1cb68579"
movie_id = 1
get_similar_movies_matrix_factorization(movie_factors, movie_id)
```

<!-- #region id="2qepoNrhYqyH" -->
Since the user and movies are in the same space, we can also compute movies similar to a user. A recommendation model can be defined as showing movies similar to the given user.
<!-- #endregion -->

```python executionInfo={"elapsed": 893, "status": "ok", "timestamp": 1619264772853, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="fX8r-mcHSM0a"
#collapse
def get_recommendations_matrix_factorization(userid, user_factors, movie_factors, top_n=10):
    user_vec = user_factors[userid - 1].reshape(1,-1)
    similarity = cosine_similarity(user_vec, movie_factors)
    sort_indexes = np.argsort(-similarity)[0]
    return [movie_df[movie_df.movieId == id].title.values[0] for id in sort_indexes[:top_n] + 1]  
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 675, "status": "ok", "timestamp": 1619264773378, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="swDNKl96Ytnq" outputId="d2d78898-19b0-4f71-ec36-f09789d46163"
top_recos = get_recommendations_matrix_factorization(1, user_factors, movie_factors)
top_recos
```

<!-- #region id="ZlkzVgEKdcIO" -->
## Graph Embedding method
<!-- #endregion -->

<!-- #region id="qnS4CWLgdeEn" -->
Create a user-movie graph with edge weights as the ratings. We will use DeepWalk to embed every node of the graph to a low-dimensional space.
<!-- #endregion -->

```python executionInfo={"elapsed": 1618, "status": "ok", "timestamp": 1619264777859, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="4ZAhnm2DYt0r"
#hide
import networkx as nx
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"elapsed": 1387, "status": "ok", "timestamp": 1619264777864, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="j0NI-HZUdg4g" outputId="eabcd9c1-8097-4ce2-effa-98447f2f6e7b"
user_item_edgelist = rating_df[['userId', 'movieId', 'rating']]
user_item_edgelist.head()
```

```python executionInfo={"elapsed": 1325, "status": "ok", "timestamp": 1619264777865, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="SdNE_KWgdjI5"
#hide
user2dict = dict()
movie2dict = dict()
cnt = 0
for x in user_item_edgelist.values:
    usr = (x[0], 'user')
    movie = (x[1], 'movie')
    if usr in user2dict:
        pass
    else:
        user2dict[usr] = cnt
        cnt += 1
    if movie in movie2dict:
        pass
    else:
        movie2dict[movie] = cnt
        cnt += 1
```

<!-- #region id="HfRpowdsdqmc" -->
Create a user-movie weighted graph using python library networkx.
<!-- #endregion -->

```python executionInfo={"elapsed": 913, "status": "ok", "timestamp": 1619264778358, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="1GTDXkn3dmXj"
user_movie_graph = nx.Graph()
```

```python executionInfo={"elapsed": 1692, "status": "ok", "timestamp": 1619264779286, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="K_qtHynPdsDw"
for x in user_item_edgelist.values:
    usr = (x[0], 'user')
    movie = (x[1], 'movie')
    user_movie_graph.add_node(user2dict[usr])
    user_movie_graph.add_node(movie2dict[movie])
    user_movie_graph.add_edge(user2dict[usr], movie2dict[movie], weight=float(x[2]))
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1539, "status": "ok", "timestamp": 1619264779287, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="jIpTzjObduIZ" outputId="95dca3ba-c153-46d1-cd4b-461508df8a3a"
user_movie_graph.number_of_edges()
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1192, "status": "ok", "timestamp": 1619264779288, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="c-SQK75OdxfF" outputId="2a87c2a3-8aa0-48cf-ba4f-1969990510a8"
user_movie_graph.number_of_nodes()
```

<!-- #region id="L_LANA0gd1n7" -->
### DeepWalk
<!-- #endregion -->

<!-- #region id="edZM0sM7d3Ta" -->
We will use the implementation of DeepWalk provided in node2vec which is a bit different from original DeepWalk e.g. it uses negative sampling whereas the original DeepWalk paper used hierarchical sampling for the skip-gram model.

To create embeddings from the context and non-context pairs, we are using Gensim python library. One can easily use Google word2vec or Facebook fasttext for this task.
<!-- #endregion -->

```python executionInfo={"elapsed": 1313, "status": "ok", "timestamp": 1619264781829, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="8OQN6lhyd-2o"
#collapse
import numpy as np
import networkx as nx
import random


class Graph():
	def __init__(self, nx_G, is_directed, p, q):
		self.G = nx_G
		self.is_directed = is_directed
		self.p = p
		self.q = q

	def node2vec_walk(self, walk_length, start_node):
		'''
		Simulate a random walk starting from start node.
		'''
		G = self.G
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges

		walk = [start_node]

		while len(walk) < walk_length:
			cur = walk[-1]
			cur_nbrs = sorted(G.neighbors(cur))
			if len(cur_nbrs) > 0:
				if len(walk) == 1:
					walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
				else:
					prev = walk[-2]
					next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], 
						alias_edges[(prev, cur)][1])]
					walk.append(next)
			else:
				break

		return walk

	def simulate_walks(self, num_walks, walk_length):
		'''
		Repeatedly simulate random walks from each node.
		'''
		G = self.G
		walks = []
		nodes = list(G.nodes())
		print('Walk iteration:')
		for walk_iter in range(num_walks):
			print(str(walk_iter+1), '/', str(num_walks))
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

		return walks

	def get_alias_edge(self, src, dst):
		'''
		Get the alias edge setup lists for a given edge.
		'''
		G = self.G
		p = self.p
		q = self.q

		unnormalized_probs = []
		for dst_nbr in sorted(G.neighbors(dst)):
			if dst_nbr == src:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
			elif G.has_edge(dst_nbr, src):
				unnormalized_probs.append(G[dst][dst_nbr]['weight'])
			else:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
		norm_const = sum(unnormalized_probs)
		try:
			normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
		except:
			normalized_probs =  [0.0 for u_prob in unnormalized_probs]

		return alias_setup(normalized_probs)

	def preprocess_transition_probs(self):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''
		G = self.G
		is_directed = self.is_directed

		alias_nodes = {}
		for node in G.nodes():
			unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
			norm_const = sum(unnormalized_probs)
			try:
				normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
			except:
				print(node)
				normalized_probs =  [0.0 for u_prob in unnormalized_probs]
			alias_nodes[node] = alias_setup(normalized_probs)

		alias_edges = {}
		triads = {}

		if is_directed:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
		else:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
				alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

		self.alias_nodes = alias_nodes
		self.alias_edges = alias_edges

		return


def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int)

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
	    q[kk] = K*prob
	    if q[kk] < 1.0:
	        smaller.append(kk)
	    else:
	        larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
	    small = smaller.pop()
	    large = larger.pop()

	    J[small] = large
	    q[large] = q[large] + q[small] - 1.0
	    if q[large] < 1.0:
	        smaller.append(large)
	    else:
	        larger.append(large)

	return J, q

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:
	    return kk
	else:
	    return J[kk]
```

```python executionInfo={"elapsed": 1070, "status": "ok", "timestamp": 1619264781830, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="ra1o_1uqdzEg"
#hide
from gensim.models import Word2Vec
```

```python executionInfo={"elapsed": 1326, "status": "ok", "timestamp": 1619264784214, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="3CDjicONeANx"
G = Graph(user_movie_graph, is_directed=False, p=1, q=1)
```

<!-- #region id="p991cjb0eErl" -->
p,q = 1 for DeeWalk as the random walks are completely unbiased. 
<!-- #endregion -->

```python executionInfo={"elapsed": 328390, "status": "ok", "timestamp": 1619265111579, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="dmLtb2DoeF6o"
# Compute the transition probabilities based on the edge weights. 
G.preprocess_transition_probs()
```

<!-- #region id="B0hqZsFPeVAO" -->
Compute the random walks.
- 10 walks for every node.
- Each walk of length 80.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 637917, "status": "ok", "timestamp": 1619265439282, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="vzo8PeLxeJMa" outputId="f0d0ce4d-179f-48df-f8ab-b68e7ffd14aa"
walks = G.simulate_walks(num_walks=10, walk_length=80)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 637555, "status": "ok", "timestamp": 1619265439283, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="F0khP0gceZKV" outputId="a1fbd0d2-dcbe-4c7e-d9ee-debc72b6bc16"
len(walks)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 637502, "status": "ok", "timestamp": 1619265439285, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="dbQKm1Ffec7G" outputId="8e11c3de-52a5-4b81-e5cc-880813d1c907"
#hide
walks[0]
```

<!-- #region id="P5Si6PgmegPx" -->
Learn Embeddings via Gensim, which creates context/non-context pairs and then Skip-gram.
<!-- #endregion -->

```python executionInfo={"elapsed": 636524, "status": "ok", "timestamp": 1619265439286, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="A5-9EBUreeFl"
#collapse
def learn_embeddings(walks):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    Uses Gensim Word2Vec.
    '''
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=50, window=10, min_count=0, sg=1, workers=8, iter=1)
    return model.wv
```

```python executionInfo={"elapsed": 736449, "status": "ok", "timestamp": 1619265539338, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="gEIgJI3MeiKT"
node_embeddings = learn_embeddings(walks)
```

<!-- #region id="bbWgRE-kekUr" -->
The output of gensim is a specific type of key-value pair with keys as the string-ed node ids and the values are numpy array of embeddings, each of shape (50,)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 732813, "status": "ok", "timestamp": 1619265539357, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="uAWzWP-RejW_" outputId="ff560108-c491-405b-b05b-8b3a3a3cc5df"
node_embeddings['0']
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 732691, "status": "ok", "timestamp": 1619265539360, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="KN4lwEGuem7m" outputId="e0fe51aa-3963-4327-ff29-5eb471330317"
movie1 = str(movie2dict[(260, 'movie')])
movie2 = str(movie2dict[(1196, 'movie')])
1.0 - cosine(node_embeddings[movie1], node_embeddings[movie2])
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 732497, "status": "ok", "timestamp": 1619265539362, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="vDkdWW0detgT" outputId="24946763-00cc-4367-f87c-ee9ba2b03ab0"
movie3 = str(movie2dict[(1210, 'movie')])
1.0 - cosine(node_embeddings[movie1], node_embeddings[movie3])
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 730901, "status": "ok", "timestamp": 1619265539364, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="tME-F_Lhevgj" outputId="1f6f7a96-5b69-43b3-b7bf-cf5d9d276a08"
movie4 = str(movie2dict[(1, 'movie')])
1.0 - cosine(node_embeddings[movie1], node_embeddings[movie4])
```

<!-- #region id="_ZHQE0POeyK1" -->
Since we worked with integer ids for nodes, let's create reverse mapping dictionaries that map integer user/movie to their actual ids.
<!-- #endregion -->

```python executionInfo={"elapsed": 728570, "status": "ok", "timestamp": 1619265539366, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="G7vm86hqewvT"
reverse_movie2dict = {k:v for v,k in movie2dict.items()}
reverse_user2dict = {k:v for v,k in user2dict.items()}
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 728440, "status": "ok", "timestamp": 1619265539368, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="airO9UdDez0A" outputId="d6eaee5b-76f5-475f-e6e9-c47e22aeaccd"
node_vecs = [node_embeddings[str(i)] for i in range(cnt)]
node_vecs = np.array(node_vecs)
node_vecs.shape
```

<!-- #region id="fGx5MRh4e27c" -->
Movies similar to a given movie as an evaluation of the system.
<!-- #endregion -->

```python executionInfo={"elapsed": 726765, "status": "ok", "timestamp": 1619265539370, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="sdGDZOlfe1hy"
#collapse
def get_similar_movies_graph_embeddings(movieid, movie_embed, top_n=10):
    movie_idx = movie2dict[movieid]
    query = movie_embed[movie_idx].reshape(1,-1)
    ranking = cosine_similarity(query, movie_embed)
    top_ids = np.argsort(-ranking)[0]
    top_movie_ids = [reverse_movie2dict[j] for j in top_ids if j in reverse_movie2dict][:top_n]
    sim_movies = [movie_df[movie_df.movieId == id[0]].title.values[0] for id in top_movie_ids]
    return sim_movies
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 726581, "status": "ok", "timestamp": 1619265539371, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="Xe735WY7e4g9" outputId="e0c2b2b4-5e85-4362-d5ba-032ceaa24765"
get_similar_movies_graph_embeddings((260, 'movie'), node_vecs)[:10]
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 990, "status": "ok", "timestamp": 1619267816240, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="bwSSBvVre6uL" outputId="11bc1654-4d6b-4e91-ad18-a37e22ce0e09"
get_similar_movies_graph_embeddings((122, 'movie'), node_vecs)[:10]
```

<!-- #region id="8-IOyAdifFDD" -->
We can also define the recommendation model based on the cosine similarity i.e the movies are ranked for a given user in terms of the cosine similarities of their corresponding embeddings with the embedding of the user.
<!-- #endregion -->

```python executionInfo={"elapsed": 1453, "status": "ok", "timestamp": 1619267821329, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="HaVE4TJ-e8Af"
#collapse
def get_recommended_movies_graph_embeddings(userid, node_embed, top_n=10):
    user_idx = user2dict[userid]
    query = node_embed[user_idx].reshape(1,-1)
    ranking = cosine_similarity(query, node_embed)
    top_ids = np.argsort(-ranking)[0]
    top_movie_ids = [reverse_movie2dict[j] for j in top_ids if j in reverse_movie2dict][:top_n]
    reco_movies = [movie_df[movie_df.movieId == id[0]].title.values[0] for id in top_movie_ids]
    return reco_movies
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1221, "status": "ok", "timestamp": 1619267821735, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="A4yH0XtwfH87" outputId="9f5dfa3f-f636-4f2c-9512-7b0053b3d210"
get_recommended_movies_graph_embeddings((1, 'user'), node_vecs, top_n=10)
```

<!-- #region id="DE84gK9lfd3q" -->
### Evaluation
<!-- #endregion -->

<!-- #region id="Qec-Xatvffbp" -->
As another evalution, let's compare the generated recommendation for a user to the movies tnat the user has actually rated highly. We will get top 10 recommendations for a user, ranked by the cosine similarity, and compute how many of these movies comes from the set of the movies that the user has rated >= 4.5. This tantamounts to Precision@10 metric. For comparison, we will also compute the Precision for the recommendations produced by the matrix factorization model.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1018, "status": "ok", "timestamp": 1619267878052, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="hc_g0J9HfJKW" outputId="80f78ebb-b57e-4616-ef97-01561ff462b9"
idx = 1
recos = set(get_recommended_movies_graph_embeddings((idx, 'user'), node_vecs, top_n=10))
true_pos = set([movie_df[movie_df.movieId == id].title.values[0] for id in rating_df[(rating_df['userId'] == idx) & (rating_df['rating'] >= 4.5)].movieId.values])
recos.intersection(true_pos)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1006, "status": "ok", "timestamp": 1619267844641, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="fNqGo7yTfms_" outputId="6704fbe1-7893-4b10-95f3-93840947338a"
mf_recos = set(get_recommendations_matrix_factorization(idx, user_factors, movie_factors))
mf_recos.intersection(true_pos)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 952, "status": "ok", "timestamp": 1619267900846, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="8IZqMiUi4uJH" outputId="28d11f3f-4cf4-4fa6-f752-741f8e221ffb"
idx = 2
recos = set(get_recommended_movies_graph_embeddings((idx, 'user'), node_vecs, top_n=10))
true_pos = set([movie_df[movie_df.movieId == id].title.values[0] for id in rating_df[(rating_df['userId'] == idx) & (rating_df['rating'] >= 4.5)].movieId.values])
recos.intersection(true_pos)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1284, "status": "ok", "timestamp": 1619267901345, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="bUdrrf394uD4" outputId="3b0b90aa-a1fc-4b8c-fc0f-4eeed919293b"
mf_recos = set(get_recommendations_matrix_factorization(idx, user_factors, movie_factors))
mf_recos.intersection(true_pos)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 921, "status": "ok", "timestamp": 1619267903540, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="4D_XSEcG4t_o" outputId="8abd844c-6001-4fcb-ff31-fd9741ce9aa7"
idx = 3
recos = set(get_recommended_movies_graph_embeddings((idx, 'user'), node_vecs, top_n=10))
true_pos = set([movie_df[movie_df.movieId == id].title.values[0] for id in rating_df[(rating_df['userId'] == idx) & (rating_df['rating'] >= 4.5)].movieId.values])
recos.intersection(true_pos)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 800, "status": "ok", "timestamp": 1619267903542, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="58DgvlWs4t7j" outputId="c4283d6b-1d78-4c1c-9220-684e00ad623b"
mf_recos = set(get_recommendations_matrix_factorization(idx, user_factors, movie_factors))
mf_recos.intersection(true_pos)
```

<!-- #region id="YOZmA9_YfxWt" -->
## Enriched network with additional information : Genres
<!-- #endregion -->

<!-- #region id="DBYwBf61f0jA" -->
Genres of the movies can be used as additional signal for better recommendations
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"elapsed": 1477, "status": "ok", "timestamp": 1619267970778, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="rfnzO7XJfrV4" outputId="f95f4a24-654f-4a74-c101-e9393818bcf8"
movie_genre_edgelist = movie_df[['movieId', 'genres']]
movie_genre_edgelist.head()
```

```python executionInfo={"elapsed": 1186, "status": "ok", "timestamp": 1619267970779, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="PWv1JOfXf3GP"
#hide
genre2int = dict()
for x in movie_genre_edgelist.values:
    genres = x[1].split('|')
    for genre in genres:
        if genre in genre2int:
            pass
        else:
            genre2int[genre] = cnt
            cnt += 1
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 768, "status": "ok", "timestamp": 1619267970781, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="C0K12ahgf4sE" outputId="9c955402-f82d-419d-b4ca-c8ba13e2b39c"
genre2int
```

```python executionInfo={"elapsed": 976, "status": "ok", "timestamp": 1619267972847, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="-VN3ns7Lf6wG"
#hide
movie_genre_graph = nx.Graph()
for x in movie_genre_edgelist.values:
    movie = (x[0], 'movie')
    genres = x[1].split('|')
    if movie in movie2dict:
        for genre in genres:
            if genre in genre2int:
                movie_genre_graph.add_node(movie2dict[movie])
                movie_genre_graph.add_node(genre2int[genre])
                movie_genre_graph.add_edge(movie2dict[movie], genre2int[genre], weight=1.0)
            else:
                pass
```

<!-- #region id="ZJfxRpg_gOw7" -->
Combine the user-movie and movie-genre graph
<!-- #endregion -->

```python executionInfo={"elapsed": 1288, "status": "ok", "timestamp": 1619267973912, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="UzNYrmFbgPLx"
user_movie_genre_graph =  nx.Graph()
user_movie_genre_graph.add_weighted_edges_from([(x,y,user_movie_graph[x][y]['weight']) for x,y in user_movie_graph.edges()])
user_movie_genre_graph.add_weighted_edges_from([(x,y,movie_genre_graph[x][y]['weight']) for x,y in movie_genre_graph.edges()])
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1159, "status": "ok", "timestamp": 1619267975320, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="JQ_8398FgTKV" outputId="8f1000a2-5663-4673-fe30-cedcc82b71cd"
user_movie_genre_graph.number_of_edges()
```

```python executionInfo={"elapsed": 499563, "status": "ok", "timestamp": 1619268473879, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="4SmKuNBFgVG2"
G_enriched = Graph(user_movie_genre_graph, is_directed=False, p=1, q=1)
G_enriched.preprocess_transition_probs()
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 687403, "status": "ok", "timestamp": 1619268661860, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="j8S99KDqgmYZ" outputId="2e406553-a0c5-452c-fef3-b13079b9b053"
walks_enriched = G_enriched.simulate_walks(num_walks=10, walk_length=80)
```

```python executionInfo={"elapsed": 794989, "status": "ok", "timestamp": 1619268769920, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="BwL6dSROgqBt"
node_embeddings_enriched = learn_embeddings(walks_enriched)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 794759, "status": "ok", "timestamp": 1619268769934, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="OTI0WB7pgr8o" outputId="59f4ed2b-e125-4018-8dd4-d69cd226a6bb"
node_vecs_enriched = [node_embeddings_enriched[str(i)] for i in range(cnt)]
node_vecs_enriched = np.array(node_vecs_enriched)
node_vecs_enriched.shape
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 792891, "status": "ok", "timestamp": 1619268769937, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="x88M_qXGgtdJ" outputId="08d23834-a186-4bfa-8645-f1ce9ba139c6"
get_similar_movies_graph_embeddings((260, 'movie'), node_vecs_enriched)[:10]
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 792741, "status": "ok", "timestamp": 1619268769938, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="kp6b2ab-gukp" outputId="8af9684f-d6fd-47c2-c865-8951883e6460"
get_similar_movies_graph_embeddings((260, 'movie'), node_vecs)[:10]
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 792549, "status": "ok", "timestamp": 1619268769940, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="JyM9nEvGgu2Y" outputId="5650a1a8-338c-4c97-b384-bdb9004f2333"
idx = 1
true_pos = set([movie_df[movie_df.movieId == id].title.values[0] for id in rating_df[(rating_df['userId'] == idx) & (rating_df['rating'] >= 4.5)].movieId.values])

mf_recos = set(get_recommendations_matrix_factorization(idx, user_factors, movie_factors))
print(len(mf_recos.intersection(true_pos)))

ge_recos = set(get_recommended_movies_graph_embeddings((idx, 'user'), node_vecs, top_n=10))
print(len(ge_recos.intersection(true_pos)))

ge_enriched_reso = set(get_recommended_movies_graph_embeddings((idx, 'user'), node_vecs_enriched, top_n=10))
print(len(ge_enriched_reso.intersection(true_pos)))
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 789799, "status": "ok", "timestamp": 1619268770837, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="7uHvvdK2g3zq" outputId="2c5f8971-2149-41e8-f11b-9b97288cd506"
idx = 8
true_pos = set([movie_df[movie_df.movieId == id].title.values[0] for id in rating_df[(rating_df['userId'] == idx) & (rating_df['rating'] >= 4.5)].movieId.values])

mf_recos = set(get_recommendations_matrix_factorization(idx, user_factors, movie_factors))
print(len(mf_recos.intersection(true_pos)))

ge_recos = set(get_recommended_movies_graph_embeddings((idx, 'user'), node_vecs, top_n=10))
print(len(ge_recos.intersection(true_pos)))

ge_enriched_reso = set(get_recommended_movies_graph_embeddings((idx, 'user'), node_vecs_enriched, top_n=10))
print(len(ge_enriched_reso.intersection(true_pos)))
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1931, "status": "ok", "timestamp": 1619269384034, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="gmqv_SAl-dcf" outputId="65529849-c907-4317-c6fe-526ccd87340b"
idx = 20
true_pos = set([movie_df[movie_df.movieId == id].title.values[0] for id in rating_df[(rating_df['userId'] == idx) & (rating_df['rating'] >= 4.5)].movieId.values])

mf_recos = set(get_recommendations_matrix_factorization(idx, user_factors, movie_factors))
print(len(mf_recos.intersection(true_pos)))

ge_recos = set(get_recommended_movies_graph_embeddings((idx, 'user'), node_vecs, top_n=10))
print(len(ge_recos.intersection(true_pos)))

ge_enriched_reso = set(get_recommended_movies_graph_embeddings((idx, 'user'), node_vecs_enriched, top_n=10))
print(len(ge_enriched_reso.intersection(true_pos)))
```

```python id="HDFtU73O-e8i"

```
