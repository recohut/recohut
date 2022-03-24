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

<!-- #region id="PGlcxTyE6N04" -->
# Node2vec on Karateclub
<!-- #endregion -->

<!-- #region id="2okzjhg13XoO" -->
## Imports
<!-- #endregion -->

```python id="0FNMggCd3TD6"
from typing import List, Callable

import networkx as nx
import numpy as np
from gensim.models.word2vec import Word2Vec
import random
from functools import partial
```

<!-- #region id="s7FMG6EASwF7" -->
## Utils
<!-- #endregion -->

```python id="KEEJeP4H4Grb"
def _check_value(value, name):
    try:
        _ = 1 / value

    except ZeroDivisionError:
        raise ValueError(
            f"The value of {name} is too small " f"or zero to be used in 1/{name}."
        )


def _undirected(node, graph) -> List[tuple]:
    edges = graph.edges(node)

    return edges


def _directed(node, graph) -> List[tuple]:
    edges = graph.out_edges(node, data=True)

    return edges


def _get_edge_fn(graph) -> Callable:
    fn = _directed if nx.classes.function.is_directed(graph) else _undirected

    fn = partial(fn, graph=graph)
    return fn


def _unweighted(edges: List[tuple]) -> np.ndarray:
    return np.ones(len(edges))


def _weighted(edges: List[tuple]) -> np.ndarray:
    weights = map(lambda edge: edge[-1]["weight"], edges)

    return np.array([*weights])


def _get_weight_fn(graph) -> Callable:
    fn = _weighted if nx.classes.function.is_weighted(graph) else _unweighted

    return fn
```

<!-- #region id="Om8uuk9kSxfb" -->
## Biased random walker
<!-- #endregion -->

```python id="JDcnYEJ638C4"
class BiasedRandomWalker:
    """
    Class to do biased second order random walks.
    Args:
        walk_length (int): Number of random walks.
        walk_number (int): Number of nodes in truncated walk.
        p (float): Return parameter (1/p transition probability) to move towards previous node.
        q (float): In-out parameter (1/q transition probability) to move away from previous node.
    """

    walks: list
    graph: nx.classes.graph.Graph
    edge_fn: Callable
    weight_fn: Callable

    def __init__(self, walk_length: int, walk_number: int, p: float, q: float):
        self.walk_length = walk_length
        self.walk_number = walk_number

        _check_value(p, "p")
        self.p = p

        _check_value(q, "q")
        self.q = q

    def do_walk(self, node: int) -> List[str]:
        """
        Doing a single truncated second order random walk from a source node.
        Arg types:
            * **node** *(int)* - The source node of the random walk.
        Return types:
            * **walk** *(list of strings)* - A single truncated random walk.
        """
        walk = [node]
        previous_node = None
        previous_node_neighbors = []
        for _ in range(self.walk_length - 1):
            current_node = walk[-1]
            edges = self.edge_fn(current_node)
            current_node_neighbors = np.array([edge[1] for edge in edges])

            weights = self.weight_fn(edges)
            probability = np.piecewise(
                weights,
                [
                    current_node_neighbors == previous_node,
                    np.isin(current_node_neighbors, previous_node_neighbors),
                ],
                [lambda w: w / self.p, lambda w: w / 1, lambda w: w / self.q],
            )

            norm_probability = probability / sum(probability)
            selected = np.random.choice(current_node_neighbors, 1, p=norm_probability)[
                0
            ]
            walk.append(selected)

            previous_node_neighbors = current_node_neighbors
            previous_node = current_node

        walk = [str(w) for w in walk]
        return walk

    def do_walks(self, graph) -> None:
        """
        Doing a fixed number of truncated random walk from every node in the graph.
        Arg types:
            * **graph** *(NetworkX graph)* - The graph to run the random walks on.
        """
        self.walks = []
        self.graph = graph

        self.edge_fn = _get_edge_fn(graph)
        self.weight_fn = _get_weight_fn(graph)

        for node in self.graph.nodes():
            for _ in range(self.walk_number):
                walk_from_node = self.do_walk(node)
                self.walks.append(walk_from_node)
```

```python id="LFhpi11a3S_J"
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

```python id="xBwNceLa4ppw"
model = Word2Vec(
            walker.walks,
            hs=1,
            alpha=self.learning_rate,
            epochs=self.epochs,
            size=self.dimensions,
            window=self.window_size,
            min_count=self.min_count,
            workers=self.workers,
            seed=self.seed,
        )


Word2Vec()
```

<!-- #region id="JMPVYv4tS2fQ" -->
## Node2vec
<!-- #endregion -->

```python id="68KDT2alydtR"
class Node2Vec(Estimator):
    r"""An implementation of `"Node2Vec" <https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf>`_
    from the KDD '16 paper "node2vec: Scalable Feature Learning for Networks".
    The procedure uses biased second order random walks to approximate the pointwise mutual information
    matrix obtained by pooling normalized adjacency matrix powers. This matrix
    is decomposed by an approximate factorization technique.

    Args:
        walk_number (int): Number of random walks. Default is 10.
        walk_length (int): Length of random walks. Default is 80.
        p (float): Return parameter (1/p transition probability) to move towards from previous node.
        q (float): In-out parameter (1/q transition probability) to move away from previous node.
        dimensions (int): Dimensionality of embedding. Default is 128.
        workers (int): Number of cores. Default is 4.
        window_size (int): Matrix power order. Default is 5.
        epochs (int): Number of epochs. Default is 1.
        learning_rate (float): HogWild! learning rate. Default is 0.05.
        min_count (int): Minimal count of node occurrences. Default is 1.
        seed (int): Random seed value. Default is 42.
    """
    _embedding: List[np.ndarray]

    def __init__(
        self,
        walk_number: int = 10,
        walk_length: int = 80,
        p: float = 1.0,
        q: float = 1.0,
        dimensions: int = 128,
        workers: int = 4,
        window_size: int = 5,
        epochs: int = 1,
        learning_rate: float = 0.05,
        min_count: int = 1,
        seed: int = 42,
    ):
        super(Node2Vec, self).__init__()

        self.walk_number = walk_number
        self.walk_length = walk_length
        self.p = p
        self.q = q
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
        walker = BiasedRandomWalker(self.walk_length, self.walk_number, self.p, self.q)
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

        n_nodes = graph.number_of_nodes()
        self._embedding = [model.wv[str(n)] for n in range(n_nodes)]


    def get_embedding(self) -> np.array:
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        return np.array(self._embedding)
```

<!-- #region id="9WnOtGZi4LxJ" -->
## Scenario
<!-- #endregion -->

```python id="DXqT-f1b4NMT"
g = nx.newman_watts_strogatz_graph(100, 20, 0.05)

model = Node2Vec()

model.fit(g)
```

```python colab={"base_uri": "https://localhost:8080/"} id="s3IUf3ui4SEG" executionInfo={"status": "ok", "timestamp": 1632673032060, "user_tz": -330, "elapsed": 621, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="750f8513-ac32-4ce9-e5fe-8648504abd40"
model.walk_length
```

```python id="zZNgkLMJ5V22"
def test_node2vec():
    """
    Testing the Node2Vec class.
    """
    model = Node2Vec()

    graph = nx.watts_strogatz_graph(100, 10, 0.5)

    model.fit(graph)

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == model.dimensions
    assert type(embedding) == np.ndarray

    model = Node2Vec(dimensions=32)

    graph = nx.watts_strogatz_graph(150, 10, 0.5)

    model.fit(graph)

    embedding = model.get_embedding()

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == model.dimensions
    assert type(embedding) == np.ndarray
```

```python id="1Od8I_4b5ecE"
test_node2vec()
```
