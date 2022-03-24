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

<!-- #region id="cV7sn6S27MxX" -->
# DeepWalk on Karateclub
<!-- #endregion -->

<!-- #region id="tlZsvlad7QZz" -->
## Codebase
<!-- #endregion -->

```python id="mrdu9eT1xSi3"
import numpy as np
import networkx as nx
from gensim.models.word2vec import Word2Vec

import random
from functools import partial
from typing import List, Callable

import random
```

```python id="OEEWYVkBXCok"
class RandomWalker:
    """
    Class to do fast first-order random walks.
    Args:
        walk_length (int): Number of random walks.
        walk_number (int): Number of nodes in truncated walk.
    """

    def __init__(self, walk_length: int, walk_number: int):
        self.walk_length = walk_length
        self.walk_number = walk_number

    def do_walk(self, node):
        """
        Doing a single truncated random walk from a source node.
        Arg types:
            * **node** *(int)* - The source node of the random walk.
        Return types:
            * **walk** *(list of strings)* - A single truncated random walk.
        """
        walk = [node]
        for _ in range(self.walk_length - 1):
            nebs = [node for node in self.graph.neighbors(walk[-1])]
            if len(nebs) > 0:
                walk = walk + random.sample(nebs, 1)
        walk = [str(w) for w in walk]
        return walk

    def do_walks(self, graph):
        """
        Doing a fixed number of truncated random walk from every node in the graph.
        Arg types:
            * **graph** *(NetworkX graph)* - The graph to run the random walks on.
        """
        self.walks = []
        self.graph = graph
        for node in self.graph.nodes():
            for _ in range(self.walk_number):
                walk_from_node = self.do_walk(node)
                self.walks.append(walk_from_node)
```

```python id="q0HzTaFsXbYV"
class Estimator(object):
    """Estimator base class with constructor and public methods."""

    seed: int

    def __init__(self):
        """Creating an estimator."""
        pass

    def fit(self):
        """Fitting a model."""
        pass

    def get_embedding(self):
        """Getting the embeddings (graph or node level)."""
        pass

    def get_memberships(self):
        """Getting the membership dictionary."""
        pass

    def get_cluster_centers(self):
        """Getting the cluster centers."""
        pass

    def _set_seed(self):
        """Creating the initial random seed."""
        random.seed(self.seed)
        np.random.seed(self.seed)

    @staticmethod
    def _ensure_integrity(graph: nx.classes.graph.Graph) -> nx.classes.graph.Graph:
        """Ensure walk traversal conditions."""
        edge_list = [(index, index) for index in range(graph.number_of_nodes())]
        graph.add_edges_from(edge_list)

        return graph

    @staticmethod
    def _check_indexing(graph: nx.classes.graph.Graph):
        """Checking the consecutive numeric indexing."""
        numeric_indices = [index for index in range(graph.number_of_nodes())]
        node_indices = sorted([node for node in graph.nodes()])

        assert numeric_indices == node_indices, "The node indexing is wrong."

    def _check_graph(self, graph: nx.classes.graph.Graph) -> nx.classes.graph.Graph:
        """Check the Karate Club assumptions about the graph."""
        self._check_indexing(graph)
        graph = self._ensure_integrity(graph)

        return graph

    def _check_graphs(self, graphs: List[nx.classes.graph.Graph]):
        """Check the Karate Club assumptions for a list of graphs."""
        graphs = [self._check_graph(graph) for graph in graphs]

        return graphs
```

```python id="0L38D0rBXcx0"
class DeepWalk(Estimator):
    r"""An implementation of `"DeepWalk" <https://arxiv.org/abs/1403.6652>`_
    from the KDD '14 paper "DeepWalk: Online Learning of Social Representations".
    The procedure uses random walks to approximate the pointwise mutual information
    matrix obtained by pooling normalized adjacency matrix powers. This matrix
    is decomposed by an approximate factorization technique.
    Args:
        walk_number (int): Number of random walks. Default is 10.
        walk_length (int): Length of random walks. Default is 80.
        dimensions (int): Dimensionality of embedding. Default is 128.
        workers (int): Number of cores. Default is 4.
        window_size (int): Matrix power order. Default is 5.
        epochs (int): Number of epochs. Default is 1.
        learning_rate (float): HogWild! learning rate. Default is 0.05.
        min_count (int): Minimal count of node occurrences. Default is 1.
        seed (int): Random seed value. Default is 42.
    """

    def __init__(
        self,
        walk_number: int = 10,
        walk_length: int = 80,
        dimensions: int = 128,
        workers: int = 4,
        window_size: int = 5,
        epochs: int = 1,
        learning_rate: float = 0.05,
        min_count: int = 1,
        seed: int = 42,
    ):

        self.walk_number = walk_number
        self.walk_length = walk_length
        self.dimensions = dimensions
        self.workers = workers
        self.window_size = window_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.seed = seed

    def fit(self, graph: nx.classes.graph.Graph):
        """
        Fitting a DeepWalk model.
        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        self._set_seed()
        graph = self._check_graph(graph)
        walker = RandomWalker(self.walk_length, self.walk_number)
        walker.do_walks(graph)

        model = Word2Vec(
            walker.walks,
            hs=1,
            alpha=self.learning_rate,
            iter=self.epochs,
            size=self.dimensions,
            window=self.window_size,
            min_count=self.min_count,
            workers=self.workers,
            seed=self.seed,
        )

        num_of_nodes = graph.number_of_nodes()
        self._embedding = [model.wv[str(n)] for n in range(num_of_nodes)]

    def get_embedding(self) -> np.array:
        r"""Getting the node embedding.
        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        return np.array(self._embedding)
```

<!-- #region id="1o-5Qfx7YlCd" -->
## Run 1
<!-- #endregion -->

```python id="-Wc-fN97Xf4k"
g = nx.newman_watts_strogatz_graph(100, 20, 0.05)

model = DeepWalk()

model.fit(g)
```

<!-- #region id="EoNhhW12Ymol" -->
## Run 2
<!-- #endregion -->

```python id="4tz6dV3uXtV6"
def test_deepwalk():
    """
    Testing the DeepWalk class.
    """
    model = DeepWalk()

    graph = nx.watts_strogatz_graph(100, 10, 0.5)

    model.fit(graph)

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == model.dimensions
    assert type(embedding) == np.ndarray

    model = DeepWalk(dimensions=32)

    graph = nx.watts_strogatz_graph(150, 10, 0.5)

    model.fit(graph)

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == model.dimensions
    assert type(embedding) == np.ndarray
```

```python id="a6wF7h7OYRNd"
test_deepwalk()
```

<!-- #region id="r2YmkQOXYn4F" -->
## Run 3
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="LjvtbL89Yi9U" executionInfo={"status": "ok", "timestamp": 1633184529325, "user_tz": -330, "elapsed": 651, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="be54738f-0c50-4f60-f213-07f204f0d525"
graph = nx.gnm_random_graph(100, 1000)

model = DeepWalk()
print(model.dimensions)

model = DeepWalk(dimensions=64)
print(model.dimensions)
```

```python colab={"base_uri": "https://localhost:8080/"} id="sSMXr962Ysjg" executionInfo={"status": "ok", "timestamp": 1633184601920, "user_tz": -330, "elapsed": 1098, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f58a6b10-0b1d-469d-8fb8-b019d95ba45e"
model.fit(graph)
embedding = model.get_embedding()
embedding
```

<!-- #region id="GLQMr5rWYjZV" -->
## Run 4
<!-- #endregion -->

```python id="qqvX1zQ57VZQ"

```
