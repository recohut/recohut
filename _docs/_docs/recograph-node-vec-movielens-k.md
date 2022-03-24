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

```python colab={"base_uri": "https://localhost:8080/"} id="M10NbHleDF96" executionInfo={"status": "ok", "timestamp": 1621252513783, "user_tz": -330, "elapsed": 2538, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="587d6cf3-aa81-4a42-bf03-3485cf2e1c0f"
!wget https://raw.githubusercontent.com/sparsh-ai/rec-data-public/master/ml-other/ml100k_ratings.csv
!wget https://raw.githubusercontent.com/sparsh-ai/rec-data-public/master/ml-other/ml100k_movies.csv
```

```python id="dW3-0phsDrpc"
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

```python id="FUzXL9-PpnYs"
def print_stats(df, uid=1):
  print(df.shape)
  print(df.movieId.nunique())
  print(max(df.movieId))
  if uid:
    print(df.userId.nunique())
    print(max(df.userId))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="QsIUlasGEG1w" executionInfo={"status": "ok", "timestamp": 1621255559438, "user_tz": -330, "elapsed": 1581, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c2be632d-1a12-4aaa-ca32-f3d250eb0e7c"
rating_df = pd.read_csv('ml100k_ratings.csv', sep=',', header=0)
rating_df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="ZODEGVn1F08H" executionInfo={"status": "ok", "timestamp": 1621255562178, "user_tz": -330, "elapsed": 1592, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="539e82a3-407f-4416-e7cc-5348e9cde294"
print_stats(rating_df)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="KKxem0FVF_SG" executionInfo={"status": "ok", "timestamp": 1621255562904, "user_tz": -330, "elapsed": 2160, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b19a0ba4-db32-48dd-9493-f32802bf5ffc"
movie_df = pd.read_csv('ml100k_movies.csv', sep=',', header=0)
movie_df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="ZmTFFculp52P" executionInfo={"status": "ok", "timestamp": 1621255562905, "user_tz": -330, "elapsed": 2075, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6a279ff6-bf8c-40d8-f1a9-815c1ef737fd"
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

```python id="B3NvTMTsGgIW"
movie_sets = dict((movie, set(users)) for movie, users in rating_df.groupby('movieId')['userId'])
```

<!-- #region id="dyUS2QlFJSc_" -->
Since we have a set of users to characterize each movie, to compute the similarity of two movies, we use Jaccard Index which, for two sets, is the ratio of number of elements in the intersection and number of elements in the union.


<!-- #endregion -->

```python id="mtga_4oHJSAI"
def jaccard(movie1, movie2, movie_sets):
    a = movie_sets[movie1]
    b = movie_sets[movie2]
    intersection = float(len(a.intersection(b)))
    return intersection / (len(a) + len(b) - intersection)
```

<!-- #region id="_rXFwdZLJXWN" -->
Let's explore similarity between some movies, qualitatively. We use the movies dataframe to get the names of the movies via their Ids.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="E_1TCARUK2Ci" executionInfo={"status": "ok", "timestamp": 1621255564888, "user_tz": -330, "elapsed": 1137, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6a58c224-b547-49b3-8673-c08697035246"
movie_df[movie_df.movieId == 260].title.values[0]
```

```python colab={"base_uri": "https://localhost:8080/"} id="q-9x2b28JZkT" executionInfo={"status": "ok", "timestamp": 1621255566144, "user_tz": -330, "elapsed": 1891, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="caf8d110-e48f-4b42-e5ed-7ab5d4e54223"
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

```python id="wRx6FxiuJf7G"
import operator 

def get_similar_movies_jaccard(movieid, movie_sets, top_n=5):
    movie = movie_df[movie_df.movieId == movieid].title.values[0]
    jaccard_dict = {x: jaccard(x, movieid, movie_sets) for x in movie_sets}
    ranked_movies = sorted(jaccard_dict.items(), key=operator.itemgetter(1), reverse=True)[:top_n]
    sim_movies = [movie_df[movie_df.movieId == id[0]].title.values[0] for id in ranked_movies]
    return {'movie': movie, 'sim_movies': sim_movies}
```

```python colab={"base_uri": "https://localhost:8080/"} id="Ogn6T_5aNVbo" executionInfo={"status": "ok", "timestamp": 1621255567113, "user_tz": -330, "elapsed": 1995, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8736b258-861a-4e83-d4bd-8ed0468845f1"
get_similar_movies_jaccard(260, movie_sets)
```

```python colab={"base_uri": "https://localhost:8080/"} id="fPNAPxqlNWx-" executionInfo={"status": "ok", "timestamp": 1621255567115, "user_tz": -330, "elapsed": 1926, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8682707d-7e51-4dc2-ef4b-e2af285284a4"
get_similar_movies_jaccard(1, movie_sets)
```

<!-- #region id="PDCLeFgJNnxN" -->
### Cosine similarity
<!-- #endregion -->

<!-- #region id="jBZTwurfNmWq" -->
Rather than the set based similarity like Jaccard, we can define every movie as a sparse vector of dimension equal to the number of users and the vector entry corresponding to each user is given by the rating that the user has for the movie or zero if no rating exists (i.e. the user hasn't seen/rated the movie).
<!-- #endregion -->

```python id="TjNjq4YeNarR"
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
```

```python colab={"base_uri": "https://localhost:8080/"} id="kGcfDDr9O2OF" executionInfo={"status": "ok", "timestamp": 1621255569714, "user_tz": -330, "elapsed": 1878, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="565ffe8a-49ed-448f-b74e-12959db8e710"
num_users = rating_df.userId.nunique()
num_users
```

```python id="lzyItagFO5PI"
movie_sparse_vecs = []
movies = []
for movie, group in rating_df.groupby('movieId'):
    vec = [0] * num_users
    for x in group[['userId', 'rating']].values:
        vec[int(x[0]) - 1] = x[1]
    movie_sparse_vecs.append(vec)
    movies.append(movie)
```

```python colab={"base_uri": "https://localhost:8080/"} id="C7NgHmrZPDxU" executionInfo={"status": "ok", "timestamp": 1621255580141, "user_tz": -330, "elapsed": 11827, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7071ab38-79d0-4c66-db85-6b658ca88f25"
movie_sparse_vecs = np.array(movie_sparse_vecs)
print(movie_sparse_vecs.shape)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Hd9m0ZIpPdmV" executionInfo={"status": "ok", "timestamp": 1621255580142, "user_tz": -330, "elapsed": 11613, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b2865ed8-880a-4136-f51d-148a81a7fd65"
print(1.0 - cosine(movie_sparse_vecs[224], movie_sparse_vecs[897]))
```

```python colab={"base_uri": "https://localhost:8080/"} id="ZP4K4AIxPD8T" executionInfo={"status": "ok", "timestamp": 1621255580143, "user_tz": -330, "elapsed": 11471, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7cfa44fe-72a9-4ee5-d6b4-c4ed9deac28e"
movie2id = {x:i for i,x in enumerate(movies)}
movie2id[260]
```

```python id="zIMlL-AUPGol"
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

```python colab={"base_uri": "https://localhost:8080/"} id="oat5HmtoPJdW" executionInfo={"status": "ok", "timestamp": 1621255581052, "user_tz": -330, "elapsed": 11589, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4d2f0172-0783-4fc7-c59b-c94232cf3072"
movieid = 1
movie_data = movie_sparse_vecs
get_similar_movies_nbd_cosine(movieid, movie_data, top_n=5)
```

```python colab={"base_uri": "https://localhost:8080/"} id="eMz9kFZrQKUv" executionInfo={"status": "ok", "timestamp": 1621255581054, "user_tz": -330, "elapsed": 11513, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6805b9be-ae47-4ab1-a0ca-7d1260a148bd"
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

```python id="wqmWohqGsqss"
le_movie = LabelEncoder()
movie_df = movie_df[movie_df.movieId.isin(rating_df.movieId.unique())]
rating_df.loc[:, 'movieId'] = le_movie.fit_transform(rating_df.loc[:, 'movieId'])
rating_df.loc[:, 'movieId']+=1
movie_df.loc[:, 'movieId'] = le_movie.transform(movie_df.loc[:, 'movieId'])
movie_df.loc[:, 'movieId']+=1
```

```python colab={"base_uri": "https://localhost:8080/"} id="yBA0Ha2MQKf2" executionInfo={"status": "ok", "timestamp": 1621255583158, "user_tz": -330, "elapsed": 1162, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ed85bf9a-20d2-4a08-ed74-3e377ac3bbd8"
ratings_mat = np.ndarray(
    shape=(np.max(rating_df.movieId.values), np.max(rating_df.userId.values)),
    dtype=np.uint8)
ratings_mat[rating_df.movieId.values-1, rating_df.userId.values-1] = rating_df.rating.values
ratings_mat.shape
```

<!-- #region id="sFDqtsgQRFkD" -->
Normalize the rating matrix


<!-- #endregion -->

```python id="pdCoya1BRDvC"
normalised_mat = ratings_mat - np.asarray([(np.mean(ratings_mat, 1))]).T
```

<!-- #region id="_z-9rYIKRP8W" -->
The number of the latent-factors is chosen to be 50 i.e. top-50 singular values of the SVD are considered. The choice of the number of latent factors is a hyperparameter of the model, and requires a more sophisticated analysis to tune. We provide no reason for the choice of 50.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="01TPkUiTROKT" executionInfo={"status": "ok", "timestamp": 1621255588221, "user_tz": -330, "elapsed": 4886, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="46a1e967-3ef6-4a7e-c597-e3fad497aa74"
n_factors = 50

A = normalised_mat.T / np.sqrt(ratings_mat.shape[0] - 1)
U, S, V = svds(A, n_factors)

print(U.shape, V.shape)
```

```python id="ICeVcjCARW-U"
movie_factors = V.T
user_factors = U
```

<!-- #region id="gbBy8vcOSBsR" -->
Instead of representing each movie as a sparse vector of the ratings of all 360,000 possible users for it, after factorizing the matrix each movie will be represented by a 50 dimensional dense vector.

Define a routine to get top-n movies similar to a given movie.

<!-- #endregion -->

```python id="pE51HtPzRZ--"
def get_similar_movies_matrix_factorization(data, movieid, top_n=10):
    index = movieid - 1 # Movie id starts from 1
    movie = movie_df[movie_df.movieId == movieid].title.values[0]
    movie_row = data[index].reshape(1,-1)
    similarity = cosine_similarity(movie_row, data)
    sort_indexes = np.argsort(-similarity)[0]
    return {'movie': movie, 'sim_movies': [movie_df[movie_df.movieId == id].title.values[0] for id in sort_indexes[:top_n] + 1]}
```

```python colab={"base_uri": "https://localhost:8080/"} id="E8nbbpG0SJ50" executionInfo={"status": "ok", "timestamp": 1621255588228, "user_tz": -330, "elapsed": 2630, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6444756b-191b-47aa-9b11-b18c2c14eadd"
movie_id = 260
get_similar_movies_matrix_factorization(movie_factors, movie_id)
```

```python colab={"base_uri": "https://localhost:8080/"} id="6PHHeIzhSMlr" executionInfo={"status": "ok", "timestamp": 1621255588230, "user_tz": -330, "elapsed": 2450, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d46b20ed-b239-452a-fc0b-2de254cca932"
movie_id = 1
get_similar_movies_matrix_factorization(movie_factors, movie_id)
```

<!-- #region id="2qepoNrhYqyH" -->
Since the user and movies are in the same space, we can also compute movies similar to a user. A recommendation model can be defined as showing movies similar to the given user.
<!-- #endregion -->

```python id="fX8r-mcHSM0a"
def get_recommendations_matrix_factorization(userid, user_factors, movie_factors, top_n=10):
    user_vec = user_factors[userid - 1].reshape(1,-1)
    similarity = cosine_similarity(user_vec, movie_factors)
    sort_indexes = np.argsort(-similarity)[0]
    return [movie_df[movie_df.movieId == id].title.values[0] for id in sort_indexes[:top_n] + 1]  
```

```python colab={"base_uri": "https://localhost:8080/"} id="swDNKl96Ytnq" executionInfo={"status": "ok", "timestamp": 1621255589607, "user_tz": -330, "elapsed": 1560, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6d74dab1-066d-43c2-9923-4439a859ad00"
top_recos = get_recommendations_matrix_factorization(1, user_factors, movie_factors)
top_recos
```

<!-- #region id="ZlkzVgEKdcIO" -->
## Graph Embedding method
<!-- #endregion -->

<!-- #region id="qnS4CWLgdeEn" -->
Create a user-movie graph with edge weights as the ratings. We will use DeepWalk to embed every node of the graph to a low-dimensional space.
<!-- #endregion -->

```python id="4ZAhnm2DYt0r"
import networkx as nx
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="j0NI-HZUdg4g" executionInfo={"status": "ok", "timestamp": 1621255606233, "user_tz": -330, "elapsed": 1093, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e3104052-743f-4f2c-d1a2-a793db76c797"
user_item_edgelist = rating_df[['userId', 'movieId', 'rating']]
user_item_edgelist.head()
```

```python id="SdNE_KWgdjI5"
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

```python id="1GTDXkn3dmXj"
user_movie_graph = nx.Graph()
```

```python id="K_qtHynPdsDw"
for x in user_item_edgelist.values:
    usr = (x[0], 'user')
    movie = (x[1], 'movie')
    user_movie_graph.add_node(user2dict[usr])
    user_movie_graph.add_node(movie2dict[movie])
    user_movie_graph.add_edge(user2dict[usr], movie2dict[movie], weight=float(x[2]))
```

```python colab={"base_uri": "https://localhost:8080/"} id="jIpTzjObduIZ" executionInfo={"status": "ok", "timestamp": 1621255613004, "user_tz": -330, "elapsed": 1864, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="50135679-53a2-452c-9574-b07d368c00e4"
user_movie_graph.number_of_edges()
```

```python colab={"base_uri": "https://localhost:8080/"} id="c-SQK75OdxfF" executionInfo={"status": "ok", "timestamp": 1621255613005, "user_tz": -330, "elapsed": 858, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="94c7ab4c-0612-4b0a-c35f-d016dae7142a"
user_movie_graph.number_of_nodes()
```

<!-- #region id="L_LANA0gd1n7" -->
### DeepWalk
<!-- #endregion -->

<!-- #region id="edZM0sM7d3Ta" -->
We will use the implementation of DeepWalk provided in node2vec which is a bit different from original DeepWalk e.g. it uses negative sampling whereas the original DeepWalk paper used hierarchical sampling for the skip-gram model.

To create embeddings from the context and non-context pairs, we are using Gensim python library. One can easily use Google word2vec or Facebook fasttext for this task.
<!-- #endregion -->

```python id="8OQN6lhyd-2o"
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

```python id="ra1o_1uqdzEg"
from gensim.models import Word2Vec
```

```python id="3CDjicONeANx"
G = Graph(user_movie_graph, is_directed=False, p=1, q=1)
```

<!-- #region id="p991cjb0eErl" -->
p,q = 1 for DeeWalk as the random walks are completely unbiased. 
<!-- #endregion -->

```python id="dmLtb2DoeF6o"
# Compute the transition probabilities based on the edge weights. 
G.preprocess_transition_probs()
```

<!-- #region id="B0hqZsFPeVAO" -->
Compute the random walks.
- 10 walks for every node.
- Each walk of length 80.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="vzo8PeLxeJMa" executionInfo={"status": "ok", "timestamp": 1621256339148, "user_tz": -330, "elapsed": 639052, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="22e366b0-4e86-4580-f31b-e959cd9de3b8"
walks = G.simulate_walks(num_walks=10, walk_length=80)
```

```python colab={"base_uri": "https://localhost:8080/"} id="F0khP0gceZKV" executionInfo={"status": "ok", "timestamp": 1621256339155, "user_tz": -330, "elapsed": 638537, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bddb4932-c45b-4eee-f5f4-fd8398971a50"
len(walks)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 86} id="dbQKm1Ffec7G" executionInfo={"status": "ok", "timestamp": 1621258616228, "user_tz": -330, "elapsed": 1320, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6db92e29-0568-44e7-ca8e-bf35673229e5"
' -> '.join([str(x) for x in walks[0]])
```

<!-- #region id="P5Si6PgmegPx" -->
Learn Embeddings via Gensim, which creates context/non-context pairs and then Skip-gram.
<!-- #endregion -->

```python id="A5-9EBUreeFl"
def learn_embeddings(walks):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    Uses Gensim Word2Vec.
    '''
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=50, window=10, min_count=0, sg=1, workers=8, iter=1)
    return model.wv
```

```python id="gEIgJI3MeiKT"
node_embeddings = learn_embeddings(walks)
```

<!-- #region id="bbWgRE-kekUr" -->
The output of gensim is a specific type of key-value pair with keys as the string-ed node ids and the values are numpy array of embeddings, each of shape (50,)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="uAWzWP-RejW_" executionInfo={"status": "ok", "timestamp": 1621259276182, "user_tz": -330, "elapsed": 98422, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5674eb2a-f00f-48ba-b687-4e6302199225"
node_embeddings['0']
```

```python colab={"base_uri": "https://localhost:8080/"} id="KN4lwEGuem7m" executionInfo={"status": "ok", "timestamp": 1621259276183, "user_tz": -330, "elapsed": 76605, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f6ccb6f5-2da3-49bd-ca84-b2ccf39be003"
movie1 = str(movie2dict[(260, 'movie')])
movie2 = str(movie2dict[(1196, 'movie')])
1.0 - cosine(node_embeddings[movie1], node_embeddings[movie2])
```

```python colab={"base_uri": "https://localhost:8080/"} id="vDkdWW0detgT" executionInfo={"status": "ok", "timestamp": 1621259276185, "user_tz": -330, "elapsed": 72405, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="abe77fbd-165f-4d82-de2e-0682a176a3f3"
movie3 = str(movie2dict[(1210, 'movie')])
1.0 - cosine(node_embeddings[movie1], node_embeddings[movie3])
```

```python colab={"base_uri": "https://localhost:8080/"} id="tME-F_Lhevgj" executionInfo={"status": "ok", "timestamp": 1621259276187, "user_tz": -330, "elapsed": 70637, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="51dd190a-100a-48f9-8c7a-22eb0a005d02"
movie4 = str(movie2dict[(1, 'movie')])
1.0 - cosine(node_embeddings[movie1], node_embeddings[movie4])
```

<!-- #region id="_ZHQE0POeyK1" -->
Since we worked with integer ids for nodes, let's create reverse mapping dictionaries that map integer user/movie to their actual ids.
<!-- #endregion -->

```python id="G7vm86hqewvT"
reverse_movie2dict = {k:v for v,k in movie2dict.items()}
reverse_user2dict = {k:v for v,k in user2dict.items()}
```

```python colab={"base_uri": "https://localhost:8080/"} id="airO9UdDez0A" executionInfo={"status": "ok", "timestamp": 1621259276189, "user_tz": -330, "elapsed": 24760, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="86711eb2-bcc5-45f3-965c-53289d3d2326"
node_vecs = [node_embeddings[str(i)] for i in range(cnt)]
node_vecs = np.array(node_vecs)
node_vecs.shape
```

<!-- #region id="fGx5MRh4e27c" -->
Movies similar to a given movie as an evaluation of the system.
<!-- #endregion -->

```python id="sdGDZOlfe1hy"
def get_similar_movies_graph_embeddings(movieid, movie_embed, top_n=10):
    movie_idx = movie2dict[movieid]
    query = movie_embed[movie_idx].reshape(1,-1)
    ranking = cosine_similarity(query, movie_embed)
    top_ids = np.argsort(-ranking)[0]
    top_movie_ids = [reverse_movie2dict[j] for j in top_ids if j in reverse_movie2dict][:top_n]
    sim_movies = [movie_df[movie_df.movieId == id[0]].title.values[0] for id in top_movie_ids]
    return sim_movies
```

```python colab={"base_uri": "https://localhost:8080/"} id="Xe735WY7e4g9" executionInfo={"status": "ok", "timestamp": 1621259276193, "user_tz": -330, "elapsed": 6855, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0a5f33bd-b9ac-4de4-c743-f0e044c6a95c"
get_similar_movies_graph_embeddings((260, 'movie'), node_vecs)[:10]
```

```python colab={"base_uri": "https://localhost:8080/"} id="bwSSBvVre6uL" executionInfo={"status": "ok", "timestamp": 1621259276194, "user_tz": -330, "elapsed": 6075, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="170440e9-999d-4fbd-d091-21f2972eaa77"
get_similar_movies_graph_embeddings((122, 'movie'), node_vecs)[:10]
```

<!-- #region id="8-IOyAdifFDD" -->
We can also define the recommendation model based on the cosine similarity i.e the movies are ranked for a given user in terms of the cosine similarities of their corresponding embeddings with the embedding of the user.
<!-- #endregion -->

```python id="HaVE4TJ-e8Af"
def get_recommended_movies_graph_embeddings(userid, node_embed, top_n=10):
    user_idx = user2dict[userid]
    query = node_embed[user_idx].reshape(1,-1)
    ranking = cosine_similarity(query, node_embed)
    top_ids = np.argsort(-ranking)[0]
    top_movie_ids = [reverse_movie2dict[j] for j in top_ids if j in reverse_movie2dict][:top_n]
    reco_movies = [movie_df[movie_df.movieId == id[0]].title.values[0] for id in top_movie_ids]
    return reco_movies
```

```python colab={"base_uri": "https://localhost:8080/"} id="A4yH0XtwfH87" executionInfo={"status": "ok", "timestamp": 1621259276197, "user_tz": -330, "elapsed": 1106, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7512bbe8-e1fc-429f-ee9f-9ff3e60262a5"
get_recommended_movies_graph_embeddings((1, 'user'), node_vecs, top_n=10)
```

<!-- #region id="DE84gK9lfd3q" -->
### Evaluation
<!-- #endregion -->

<!-- #region id="Qec-Xatvffbp" -->
As another evalution, let's compare the generated recommendation for a user to the movies tnat the user has actually rated highly. We will get top 10 recommendations for a user, ranked by the cosine similarity, and compute how many of these movies comes from the set of the movies that the user has rated >= 4.5. This tantamounts to Precision@10 metric. For comparison, we will also compute the Precision for the recommendations produced by the matrix factorization model.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="hc_g0J9HfJKW" executionInfo={"status": "ok", "timestamp": 1621259304453, "user_tz": -330, "elapsed": 1092, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a306cd0a-d56e-42ef-8aff-6455480a1b9f"
idx = 1
recos = set(get_recommended_movies_graph_embeddings((idx, 'user'), node_vecs, top_n=10))
true_pos = set([movie_df[movie_df.movieId == id].title.values[0] for id in rating_df[(rating_df['userId'] == idx) & (rating_df['rating'] >= 4.5)].movieId.values])
recos.intersection(true_pos)
```

```python colab={"base_uri": "https://localhost:8080/"} id="fNqGo7yTfms_" executionInfo={"status": "ok", "timestamp": 1621259305841, "user_tz": -330, "elapsed": 1190, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c4ae7b1e-c3bd-44e0-c4db-98595e51e787"
mf_recos = set(get_recommendations_matrix_factorization(idx, user_factors, movie_factors))
mf_recos.intersection(true_pos)
```

```python colab={"base_uri": "https://localhost:8080/"} id="8IZqMiUi4uJH" executionInfo={"status": "ok", "timestamp": 1621259310741, "user_tz": -330, "elapsed": 1191, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6f8917db-00c7-4c80-b5a1-a6c4c279ccc2"
idx = 2
recos = set(get_recommended_movies_graph_embeddings((idx, 'user'), node_vecs, top_n=10))
true_pos = set([movie_df[movie_df.movieId == id].title.values[0] for id in rating_df[(rating_df['userId'] == idx) & (rating_df['rating'] >= 4.5)].movieId.values])
recos.intersection(true_pos)
```

```python colab={"base_uri": "https://localhost:8080/"} id="bUdrrf394uD4" executionInfo={"status": "ok", "timestamp": 1621259311650, "user_tz": -330, "elapsed": 995, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8b2f501b-6f2a-4982-b275-f3b77fb74793"
mf_recos = set(get_recommendations_matrix_factorization(idx, user_factors, movie_factors))
mf_recos.intersection(true_pos)
```

```python colab={"base_uri": "https://localhost:8080/"} id="4D_XSEcG4t_o" executionInfo={"status": "ok", "timestamp": 1621259314326, "user_tz": -330, "elapsed": 2335, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="318237b9-0fb7-48a1-b697-69cf3c033331"
idx = 3
recos = set(get_recommended_movies_graph_embeddings((idx, 'user'), node_vecs, top_n=10))
true_pos = set([movie_df[movie_df.movieId == id].title.values[0] for id in rating_df[(rating_df['userId'] == idx) & (rating_df['rating'] >= 4.5)].movieId.values])
recos.intersection(true_pos)
```

```python colab={"base_uri": "https://localhost:8080/"} id="58DgvlWs4t7j" executionInfo={"status": "ok", "timestamp": 1621259314328, "user_tz": -330, "elapsed": 2023, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="37f94d18-23e4-441f-c4c9-678c180cc034"
mf_recos = set(get_recommendations_matrix_factorization(idx, user_factors, movie_factors))
mf_recos.intersection(true_pos)
```

<!-- #region id="YOZmA9_YfxWt" -->
## Enriched network with additional information : Genres
<!-- #endregion -->

<!-- #region id="DBYwBf61f0jA" -->
Genres of the movies can be used as additional signal for better recommendations
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="rfnzO7XJfrV4" executionInfo={"status": "ok", "timestamp": 1621259318155, "user_tz": -330, "elapsed": 1652, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="99ba6f7b-9dd2-436d-e036-3cdea6fe3adc"
movie_genre_edgelist = movie_df[['movieId', 'genres']]
movie_genre_edgelist.head()
```

```python id="PWv1JOfXf3GP"
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

```python colab={"base_uri": "https://localhost:8080/"} id="C0K12ahgf4sE" executionInfo={"status": "ok", "timestamp": 1621259320139, "user_tz": -330, "elapsed": 2131, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="df9dc659-e4ee-42cb-e630-139d17cb3aa6"
genre2int
```

```python id="-VN3ns7Lf6wG"
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

```python id="UzNYrmFbgPLx"
user_movie_genre_graph =  nx.Graph()
user_movie_genre_graph.add_weighted_edges_from([(x,y,user_movie_graph[x][y]['weight']) for x,y in user_movie_graph.edges()])
user_movie_genre_graph.add_weighted_edges_from([(x,y,movie_genre_graph[x][y]['weight']) for x,y in movie_genre_graph.edges()])
```

```python colab={"base_uri": "https://localhost:8080/"} id="JQ_8398FgTKV" executionInfo={"status": "ok", "timestamp": 1621259333349, "user_tz": -330, "elapsed": 1493, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cd683de2-c582-402c-a3d1-541db0eef259"
user_movie_genre_graph.number_of_edges()
```

```python id="4SmKuNBFgVG2"
G_enriched = Graph(user_movie_genre_graph, is_directed=False, p=1, q=1)
G_enriched.preprocess_transition_probs()
```

```python colab={"base_uri": "https://localhost:8080/"} id="j8S99KDqgmYZ" executionInfo={"status": "ok", "timestamp": 1621259965987, "user_tz": -330, "elapsed": 633169, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="38b8ceec-db76-4e3f-e327-1bde043490fb"
walks_enriched = G_enriched.simulate_walks(num_walks=10, walk_length=80)
```

```python id="BwL6dSROgqBt"
node_embeddings_enriched = learn_embeddings(walks_enriched)
```

```python colab={"base_uri": "https://localhost:8080/"} id="OTI0WB7pgr8o" executionInfo={"status": "ok", "timestamp": 1621260062841, "user_tz": -330, "elapsed": 728815, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ebf3f13a-7543-4ed1-ecf3-fcdbad5b90c4"
node_vecs_enriched = [node_embeddings_enriched[str(i)] for i in range(cnt)]
node_vecs_enriched = np.array(node_vecs_enriched)
node_vecs_enriched.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="x88M_qXGgtdJ" executionInfo={"status": "ok", "timestamp": 1621260062843, "user_tz": -330, "elapsed": 728204, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a3b76f8c-7051-4775-ed33-ead574daf341"
get_similar_movies_graph_embeddings((260, 'movie'), node_vecs_enriched)[:10]
```

```python colab={"base_uri": "https://localhost:8080/"} id="kp6b2ab-gukp" executionInfo={"status": "ok", "timestamp": 1621260062845, "user_tz": -330, "elapsed": 726922, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1a55dede-9508-4385-c22f-4d5fbbc16117"
get_similar_movies_graph_embeddings((260, 'movie'), node_vecs)[:10]
```

```python colab={"base_uri": "https://localhost:8080/"} id="JyM9nEvGgu2Y" executionInfo={"status": "ok", "timestamp": 1621260062847, "user_tz": -330, "elapsed": 725654, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c74fbe5c-1349-485d-cbf1-0ebda024c0ca"
idx = 1
true_pos = set([movie_df[movie_df.movieId == id].title.values[0] for id in rating_df[(rating_df['userId'] == idx) & (rating_df['rating'] >= 4.5)].movieId.values])

mf_recos = set(get_recommendations_matrix_factorization(idx, user_factors, movie_factors))
print(len(mf_recos.intersection(true_pos)))

ge_recos = set(get_recommended_movies_graph_embeddings((idx, 'user'), node_vecs, top_n=10))
print(len(ge_recos.intersection(true_pos)))

ge_enriched_reso = set(get_recommended_movies_graph_embeddings((idx, 'user'), node_vecs_enriched, top_n=10))
print(len(ge_enriched_reso.intersection(true_pos)))
```

```python colab={"base_uri": "https://localhost:8080/"} id="7uHvvdK2g3zq" executionInfo={"status": "ok", "timestamp": 1621260062848, "user_tz": -330, "elapsed": 724042, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="83924d91-9629-4556-9de1-c1ee2c99d17b"
idx = 8
true_pos = set([movie_df[movie_df.movieId == id].title.values[0] for id in rating_df[(rating_df['userId'] == idx) & (rating_df['rating'] >= 4.5)].movieId.values])

mf_recos = set(get_recommendations_matrix_factorization(idx, user_factors, movie_factors))
print(len(mf_recos.intersection(true_pos)))

ge_recos = set(get_recommended_movies_graph_embeddings((idx, 'user'), node_vecs, top_n=10))
print(len(ge_recos.intersection(true_pos)))

ge_enriched_reso = set(get_recommended_movies_graph_embeddings((idx, 'user'), node_vecs_enriched, top_n=10))
print(len(ge_enriched_reso.intersection(true_pos)))
```

```python colab={"base_uri": "https://localhost:8080/"} id="gmqv_SAl-dcf" executionInfo={"status": "ok", "timestamp": 1621260062850, "user_tz": -330, "elapsed": 723109, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5388b187-07cf-441e-b793-f5febb61fecb"
idx = 20
true_pos = set([movie_df[movie_df.movieId == id].title.values[0] for id in rating_df[(rating_df['userId'] == idx) & (rating_df['rating'] >= 4.5)].movieId.values])

mf_recos = set(get_recommendations_matrix_factorization(idx, user_factors, movie_factors))
print(len(mf_recos.intersection(true_pos)))

ge_recos = set(get_recommended_movies_graph_embeddings((idx, 'user'), node_vecs, top_n=10))
print(len(ge_recos.intersection(true_pos)))

ge_enriched_reso = set(get_recommended_movies_graph_embeddings((idx, 'user'), node_vecs_enriched, top_n=10))
print(len(ge_enriched_reso.intersection(true_pos)))
```
