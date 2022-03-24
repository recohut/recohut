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

<!-- #region id="-T83BVkZAR_t" -->
# Node2vec on ML-latest in Keras

> Implementing the node2vec model to generate embeddings for movies from the MovieLens dataset.
<!-- #endregion -->

<!-- #region id="Te6L78rxASAE" -->
## Introduction

Learning useful representations from objects structured as graphs is useful for
a variety of machine learning (ML) applications—such as social and communication networks analysis,
biomedicine studies, and recommendation systems.
[Graph representation Learning](https://www.cs.mcgill.ca/~wlh/grl_book/) aims to
learn embeddings for the graph nodes, which can be used for a variety of ML tasks
such as node label prediction (e.g. categorizing an article based on its citations)
and link prediction (e.g. recommending an interest group to a user in a social network).

[node2vec](https://arxiv.org/abs/1607.00653) is a simple, yet scalable and effective
technique for learning low-dimensional embeddings for nodes in a graph by optimizing
a neighborhood-preserving objective. The aim is to learn similar embeddings for
neighboring nodes, with respect to the graph structure.

Given your data items structured as a graph (where the items are represented as
nodes and the relationship between items are represented as edges),
node2vec works as follows:

1. Generate item sequences using (biased) random walk.
2. Create positive and negative training examples from these sequences.
3. Train a [word2vec](https://www.tensorflow.org/tutorials/text/word2vec) model
(skip-gram) to learn embeddings for the items.

In this example, we demonstrate the node2vec technique on the
[small version of the Movielens dataset](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html)
to learn movie embeddings. Such a dataset can be represented as a graph by treating
the movies as nodes, and creating edges between movies that have similar ratings
by the users. The learnt movie embeddings can be used for tasks such as movie recommendation,
or movie genres prediction.

This example requires `networkx` package, which can be installed using the following command:

```shell
pip install networkx
```
<!-- #endregion -->

<!-- #region id="dzCIyOACASAJ" -->
## Setup
<!-- #endregion -->

```python id="MXx4sXh6ASAN"
import os
from collections import defaultdict
import math
import networkx as nx
import random
from tqdm import tqdm
from zipfile import ZipFile
from urllib.request import urlretrieve
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
```

<!-- #region id="gU7M2Ls7ASAT" -->
## Download the MovieLens dataset and prepare the data

The small version of the MovieLens dataset includes around 100k ratings
from 610 users on 9,742 movies.

First, let's download the dataset. The downloaded folder will contain
three data files: `users.csv`, `movies.csv`, and `ratings.csv`. In this example,
we will only need the `movies.dat`, and `ratings.dat` data files.
<!-- #endregion -->

```python id="wxK0i1XSASAV"
urlretrieve(
    "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip", "movielens.zip"
)
ZipFile("movielens.zip", "r").extractall()
```

<!-- #region id="lYtjW9ssASAY" -->
Then, we load the data into a Pandas DataFrame and perform some basic preprocessing.
<!-- #endregion -->

```python id="dVAHpFEoASAa" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1632674927466, "user_tz": -330, "elapsed": 19, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="131ceb68-8d58-4f8d-beb2-890eea619bef"
# Load movies to a DataFrame.
movies = pd.read_csv("ml-latest-small/movies.csv")
# Create a `movieId` string.
movies["movieId"] = movies["movieId"].apply(lambda x: f"movie_{x}")

# Load ratings to a DataFrame.
ratings = pd.read_csv("ml-latest-small/ratings.csv")
# Convert the `ratings` to floating point
ratings["rating"] = ratings["rating"].apply(lambda x: float(x))
# Create the `movie_id` string.
ratings["movieId"] = ratings["movieId"].apply(lambda x: f"movie_{x}")

print("Movies data shape:", movies.shape)
print("Ratings data shape:", ratings.shape)
```

<!-- #region id="lz39ix09ASAd" -->
Let's inspect a sample instance of the `ratings` DataFrame.
<!-- #endregion -->

```python id="DJwYiYElASAh" colab={"base_uri": "https://localhost:8080/", "height": 206} executionInfo={"status": "ok", "timestamp": 1632674930917, "user_tz": -330, "elapsed": 636, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f4a4b664-0393-49a2-90bf-6e1307588a78"
ratings.head()
```

<!-- #region id="vPBTQSfQASAj" -->
Next, let's check a sample instance of the `movies` DataFrame.
<!-- #endregion -->

```python id="f6je_bruASAk" colab={"base_uri": "https://localhost:8080/", "height": 206} executionInfo={"status": "ok", "timestamp": 1632674933809, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1a8405f8-84bd-4aa9-c47e-83637cbbbae6"
movies.head()
```

<!-- #region id="lf6RuNclASAm" -->
Implement two utility functions for the `movies` DataFrame.
<!-- #endregion -->

```python id="eG6k_5ZmASAn"

def get_movie_title_by_id(movieId):
    return list(movies[movies.movieId == movieId].title)[0]


def get_movie_id_by_title(title):
    return list(movies[movies.title == title].movieId)[0]

```

<!-- #region id="cvLfNbdQASAp" -->
## Construct the Movies graph

We create an edge between two movie nodes in the graph if both movies are rated
by the same user >= `min_rating`. The weight of the edge will be based on the
[pointwise mutual information](https://en.wikipedia.org/wiki/Pointwise_mutual_information)
between the two movies, which is computed as: `log(xy) - log(x) - log(y) + log(D)`, where:

* `xy` is how many users rated both movie `x` and movie `y` with >= `min_rating`.
* `x` is how many users rated movie `x` >= `min_rating`.
* `y` is how many users rated movie `y` >= `min_rating`.
* `D` total number of movie ratings >= `min_rating`.
<!-- #endregion -->

<!-- #region id="txbnvsEAASAs" -->
### Step 1: create the weighted edges between movies.
<!-- #endregion -->

```python id="2CGIc-taASAu" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1632674937087, "user_tz": -330, "elapsed": 906, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="cd714e2f-8050-4368-a89f-ea140b8b90e6"
min_rating = 5
pair_frequency = defaultdict(int)
item_frequency = defaultdict(int)

# Filter instances where rating is greater than or equal to min_rating.
rated_movies = ratings[ratings.rating >= min_rating]
# Group instances by user.
movies_grouped_by_users = list(rated_movies.groupby("userId"))
for group in tqdm(
    movies_grouped_by_users,
    position=0,
    leave=True,
    desc="Compute movie rating frequencies",
):
    # Get a list of movies rated by the user.
    current_movies = list(group[1]["movieId"])

    for i in range(len(current_movies)):
        item_frequency[current_movies[i]] += 1
        for j in range(i + 1, len(current_movies)):
            x = min(current_movies[i], current_movies[j])
            y = max(current_movies[i], current_movies[j])
            pair_frequency[(x, y)] += 1
```

<!-- #region id="gUT67RLDASAw" -->
### Step 2: create the graph with the nodes and the edges

To reduce the number of edges between nodes, we only add an edge between movies
if the weight of the edge is greater than `min_weight`.
<!-- #endregion -->

```python id="HMH7s8ohASAy" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1632674938078, "user_tz": -330, "elapsed": 1004, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="978d4908-b6e2-48ca-bb07-51e4e7bf29b1"
min_weight = 10
D = math.log(sum(item_frequency.values()))

# Create the movies undirected graph.
movies_graph = nx.Graph()
# Add weighted edges between movies.
# This automatically adds the movie nodes to the graph.
for pair in tqdm(
    pair_frequency, position=0, leave=True, desc="Creating the movie graph"
):
    x, y = pair
    xy_frequency = pair_frequency[pair]
    x_frequency = item_frequency[x]
    y_frequency = item_frequency[y]
    pmi = math.log(xy_frequency) - math.log(x_frequency) - math.log(y_frequency) + D
    weight = pmi * xy_frequency
    # Only include edges with weight >= min_weight.
    if weight >= min_weight:
        movies_graph.add_edge(x, y, weight=weight)
```

<!-- #region id="5MoixT6kASAz" -->
Let's display the total number of nodes and edges in the graph.
Note that the number of nodes is less than the total number of movies,
since only the movies that have edges to other movies are added.
<!-- #endregion -->

```python id="Xk4XQkXrASA0" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1632674940942, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4399a044-c281-4318-92e3-895eaa0583b4"
print("Total number of graph nodes:", movies_graph.number_of_nodes())
print("Total number of graph edges:", movies_graph.number_of_edges())
```

<!-- #region id="HN8lkNKkASA1" -->
Let's display the average node degree (number of neighbours) in the graph.
<!-- #endregion -->

```python id="rvOhJ429ASA1" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1632674941332, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6ac43860-d216-4632-81e1-472cb6cc7235"
degrees = []
for node in movies_graph.nodes:
    degrees.append(movies_graph.degree[node])

print("Average node degree:", round(sum(degrees) / len(degrees), 2))
```

<!-- #region id="IENzc-f1ASA2" -->
### Step 3: Create vocabulary and a mapping from tokens to integer indices

The vocabulary is the nodes (movie IDs) in the graph.
<!-- #endregion -->

```python id="TQBpdDNaASA3"
vocabulary = ["NA"] + list(movies_graph.nodes)
vocabulary_lookup = {token: idx for idx, token in enumerate(vocabulary)}
```

<!-- #region id="Bhi2QC_GASA3" -->
## Implement the biased random walk

A random walk starts from a given node, and randomly picks a neighbour node to move to.
If the edges are weighted, the neighbour is selected *probabilistically* with
respect to weights of the edges between the current node and its neighbours.
This procedure is repeated for `num_steps` to generate a sequence of *related* nodes.

The [*biased* random walk](https://en.wikipedia.org/wiki/Biased_random_walk_on_a_graph) balances between **breadth-first sampling**
(where only local neighbours are visited) and **depth-first sampling**
(where  distant neighbours are visited) by introducing the following two parameters:

1. **Return parameter** (`p`): Controls the likelihood of immediately revisiting
a node in the walk. Setting it to a high value encourages moderate exploration,
while setting it to a low value would keep the walk local.
2. **In-out parameter** (`q`): Allows the search to differentiate
between *inward* and *outward* nodes. Setting it to a high value biases the
random walk towards local nodes, while setting it to a low value biases the walk
to visit nodes which are further away.
<!-- #endregion -->

```python id="-Sr_nA_DASA4"

def next_step(graph, previous, current, p, q):
    neighbors = list(graph.neighbors(current))

    weights = []
    # Adjust the weights of the edges to the neighbors with respect to p and q.
    for neighbor in neighbors:
        if neighbor == previous:
            # Control the probability to return to the previous node.
            weights.append(graph[current][neighbor]["weight"] / p)
        elif graph.has_edge(neighbor, previous):
            # The probability of visiting a local node.
            weights.append(graph[current][neighbor]["weight"])
        else:
            # Control the probability to move forward.
            weights.append(graph[current][neighbor]["weight"] / q)

    # Compute the probabilities of visiting each neighbor.
    weight_sum = sum(weights)
    probabilities = [weight / weight_sum for weight in weights]
    # Probabilistically select a neighbor to visit.
    next = np.random.choice(neighbors, size=1, p=probabilities)[0]
    return next


def random_walk(graph, num_walks, num_steps, p, q):
    walks = []
    nodes = list(graph.nodes())
    # Perform multiple iterations of the random walk.
    for walk_iteration in range(num_walks):
        random.shuffle(nodes)

        for node in tqdm(
            nodes,
            position=0,
            leave=True,
            desc=f"Random walks iteration {walk_iteration + 1} of {num_walks}",
        ):
            # Start the walk with a random node from the graph.
            walk = [node]
            # Randomly walk for num_steps.
            while len(walk) < num_steps:
                current = walk[-1]
                previous = walk[-2] if len(walk) > 1 else None
                # Compute the next node to visit.
                next = next_step(graph, previous, current, p, q)
                walk.append(next)
            # Replace node ids (movie ids) in the walk with token ids.
            walk = [vocabulary_lookup[token] for token in walk]
            # Add the walk to the generated sequence.
            walks.append(walk)

    return walks

```

<!-- #region id="hx4DEvq4ASA5" -->
## Generate training data using the biased random walk

You can explore different configurations of `p` and `q` to different results of
related movies.
<!-- #endregion -->

```python id="AlTVVRgpASA6" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1632674985523, "user_tz": -330, "elapsed": 40457, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0ff0f920-b0f0-4002-85dc-ccb7d66043a6"
# Random walk return parameter.
p = 1
# Random walk in-out parameter.
q = 1
# Number of iterations of random walks.
num_walks = 5
# Number of steps of each random walk.
num_steps = 10
walks = random_walk(movies_graph, num_walks, num_steps, p, q)

print("Number of walks generated:", len(walks))
```

<!-- #region id="hEaRU0FeASA8" -->
## Generate positive and negative examples

To train a skip-gram model, we use the generated walks to create positive and
negative training examples. Each example includes the following features:

1. `target`: A movie in a walk sequence.
2. `context`: Another movie in a walk sequence.
3. `weight`: How many times these two movies occured in walk sequences.
4. `label`: The label is 1 if these two movies are samples from the walk sequences,
otherwise (i.e., if randomly sampled) the label is 0.
<!-- #endregion -->

<!-- #region id="_dbZhfLXASA9" -->
### Generate examples
<!-- #endregion -->

```python id="BlYw5qOJASA_" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1632675002516, "user_tz": -330, "elapsed": 17029, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3ec5510c-b76d-4fa4-8149-e498fe0fd5ee"

def generate_examples(sequences, window_size, num_negative_samples, vocabulary_size):
    example_weights = defaultdict(int)
    # Iterate over all sequences (walks).
    for sequence in tqdm(
        sequences,
        position=0,
        leave=True,
        desc=f"Generating postive and negative examples",
    ):
        # Generate positive and negative skip-gram pairs for a sequence (walk).
        pairs, labels = keras.preprocessing.sequence.skipgrams(
            sequence,
            vocabulary_size=vocabulary_size,
            window_size=window_size,
            negative_samples=num_negative_samples,
        )
        for idx in range(len(pairs)):
            pair = pairs[idx]
            label = labels[idx]
            target, context = min(pair[0], pair[1]), max(pair[0], pair[1])
            if target == context:
                continue
            entry = (target, context, label)
            example_weights[entry] += 1

    targets, contexts, labels, weights = [], [], [], []
    for entry in example_weights:
        weight = example_weights[entry]
        target, context, label = entry
        targets.append(target)
        contexts.append(context)
        labels.append(label)
        weights.append(weight)

    return np.array(targets), np.array(contexts), np.array(labels), np.array(weights)


num_negative_samples = 4
targets, contexts, labels, weights = generate_examples(
    sequences=walks,
    window_size=num_steps,
    num_negative_samples=num_negative_samples,
    vocabulary_size=len(vocabulary),
)
```

<!-- #region id="o8IE6-sXASBA" -->
Let's display the shapes of the outputs
<!-- #endregion -->

```python id="lWTdap4PASBB" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1632675002517, "user_tz": -330, "elapsed": 30, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="04021d44-250c-45f2-8741-4ea9cf92b1c9"
print(f"Targets shape: {targets.shape}")
print(f"Contexts shape: {contexts.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Weights shape: {weights.shape}")
```

<!-- #region id="nJnAso-BASBC" -->
### Convert the data into `tf.data.Dataset` objects
<!-- #endregion -->

```python id="OQt0G2TYASBD"
batch_size = 1024


def create_dataset(targets, contexts, labels, weights, batch_size):
    inputs = {
        "target": targets,
        "context": contexts,
    }
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels, weights))
    dataset = dataset.shuffle(buffer_size=batch_size * 2)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


dataset = create_dataset(
    targets=targets,
    contexts=contexts,
    labels=labels,
    weights=weights,
    batch_size=batch_size,
)
```

<!-- #region id="9zAf2zxPASBE" -->
## Train the skip-gram model

Our skip-gram is a simple binary classification model that works as follows:

1. An embedding is looked up for the `target` movie.
2. An embedding is looked up for the `context` movie.
3. The dot product is computed between these two embeddings.
4. The result (after a sigmoid activation) is compared to the label.
5. A binary crossentropy loss is used.
<!-- #endregion -->

```python id="H9y4WII4ASBE"
learning_rate = 0.001
embedding_dim = 50
num_epochs = 10
```

<!-- #region id="tGxbuOX8ASBF" -->
### Implement the model
<!-- #endregion -->

```python id="5fqvCesPASBG"

def create_model(vocabulary_size, embedding_dim):

    inputs = {
        "target": layers.Input(name="target", shape=(), dtype="int32"),
        "context": layers.Input(name="context", shape=(), dtype="int32"),
    }
    # Initialize item embeddings.
    embed_item = layers.Embedding(
        input_dim=vocabulary_size,
        output_dim=embedding_dim,
        embeddings_initializer="he_normal",
        embeddings_regularizer=keras.regularizers.l2(1e-6),
        name="item_embeddings",
    )
    # Lookup embeddings for target.
    target_embeddings = embed_item(inputs["target"])
    # Lookup embeddings for context.
    context_embeddings = embed_item(inputs["context"])
    # Compute dot similarity between target and context embeddings.
    logits = layers.Dot(axes=1, normalize=False, name="dot_similarity")(
        [target_embeddings, context_embeddings]
    )
    # Create the model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

```

<!-- #region id="H9bFiYZRASBH" -->
### Train the model
<!-- #endregion -->

<!-- #region id="9jCmIqMjASBH" -->
We instantiate the model and compile it.
<!-- #endregion -->

```python id="0IGjmFDpASBH"
model = create_model(len(vocabulary), embedding_dim)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
)
```

<!-- #region id="cIE7WW5eASBI" -->
Let's plot the model.
<!-- #endregion -->

```python id="ReS9WBU7ASBI" colab={"base_uri": "https://localhost:8080/", "height": 312} executionInfo={"status": "ok", "timestamp": 1632675004582, "user_tz": -330, "elapsed": 1376, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="01c7fd83-144f-47b9-ae3f-b577ea9582a3"
keras.utils.plot_model(
    model, show_shapes=True, show_dtype=True, show_layer_names=True,
)
```

<!-- #region id="3sKUOOMfASBI" -->
Now we train the model on the `dataset`.
<!-- #endregion -->

```python id="2u5QM26OASBJ" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1632675059964, "user_tz": -330, "elapsed": 55400, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0abf3af4-fd52-423b-f997-dc86ee2d5278"
history = model.fit(dataset, epochs=num_epochs)
```

<!-- #region id="Y5rWBMq-ASBJ" -->
Finally we plot the learning history.
<!-- #endregion -->

```python id="vo8rIuhhASBK" colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"status": "ok", "timestamp": 1632675060680, "user_tz": -330, "elapsed": 754, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e56a9075-06ef-4da1-ef25-dc345038d7b5"
plt.plot(history.history["loss"])
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()
```

<!-- #region id="f9x5qeZWASBK" -->
## Analyze the learnt embeddings.
<!-- #endregion -->

```python id="vHjLPXhxASBK" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1632675060682, "user_tz": -330, "elapsed": 30, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2eb2f71c-6ab3-4c9c-a83d-aa5b314c7c1e"
movie_embeddings = model.get_layer("item_embeddings").get_weights()[0]
print("Embeddings shape:", movie_embeddings.shape)
```

<!-- #region id="BV0VxFKWASBL" -->
### Find related movies

Define a list with some movies called `query_movies`.
<!-- #endregion -->

```python id="c7WpjT8lASBL"
query_movies = [
    "Matrix, The (1999)",
    "Star Wars: Episode IV - A New Hope (1977)",
    "Lion King, The (1994)",
    "Terminator 2: Judgment Day (1991)",
    "Godfather, The (1972)",
]
```

<!-- #region id="mIbZHTcSASBM" -->
Get the embeddings of the movies in `query_movies`.
<!-- #endregion -->

```python id="JxzENVjmASBM"
query_embeddings = []

for movie_title in query_movies:
    movieId = get_movie_id_by_title(movie_title)
    token_id = vocabulary_lookup[movieId]
    movie_embedding = movie_embeddings[token_id]
    query_embeddings.append(movie_embedding)

query_embeddings = np.array(query_embeddings)
```

<!-- #region id="RrRNBTxaASBN" -->
Compute the [consine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) between the embeddings of `query_movies`
and all the other movies, then pick the top k for each.
<!-- #endregion -->

```python id="GlfcTvC5ASBO"
similarities = tf.linalg.matmul(
    tf.math.l2_normalize(query_embeddings),
    tf.math.l2_normalize(movie_embeddings),
    transpose_b=True,
)

_, indices = tf.math.top_k(similarities, k=5)
indices = indices.numpy().tolist()
```

<!-- #region id="MViqDx34ASBP" -->
Display the top related movies in `query_movies`.
<!-- #endregion -->

```python id="gPhK7ZeVASBP" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1632675060691, "user_tz": -330, "elapsed": 32, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5ca899b1-5000-417c-b564-a5087a39b2cd"
for idx, title in enumerate(query_movies):
    print(title)
    print("".rjust(len(title), "-"))
    similar_tokens = indices[idx]
    for token in similar_tokens:
        similar_movieId = vocabulary[token]
        similar_title = get_movie_title_by_id(similar_movieId)
        print(f"- {similar_title}")
    print()
```

<!-- #region id="PxFtA08eASBQ" -->
### Visualize the embeddings using the Embedding Projector
<!-- #endregion -->

```python id="snumVN70ASBQ"
import io

out_v = io.open("embeddings.tsv", "w", encoding="utf-8")
out_m = io.open("metadata.tsv", "w", encoding="utf-8")

for idx, movie_id in enumerate(vocabulary[1:]):
    movie_title = list(movies[movies.movieId == movie_id].title)[0]
    vector = movie_embeddings[idx]
    out_v.write("\t".join([str(x) for x in vector]) + "\n")
    out_m.write(movie_title + "\n")

out_v.close()
out_m.close()
```

<!-- #region id="1AlttAcrASBQ" -->
Download the `embeddings.tsv` and `metadata.tsv` to analyze the obtained embeddings
in the [Embedding Projector](https://projector.tensorflow.org/).
<!-- #endregion -->
