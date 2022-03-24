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

<!-- #region id="JLumGR08ADBo" -->
# Introduction to Networkx
<!-- #endregion -->

```python id="KDulk7kfUu38"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import networkx as nx
from networkx.drawing.layout import bipartite_layout
```

```python id="qi6Z8gbFG0O8"
def draw_graph(G, pos_nodes, node_names={}, node_size=50, plot_weight=False):
    nx.draw(G, pos_nodes, with_labels=False, node_size=node_size, edge_color='gray', arrowsize=30)
    
    pos_attrs = {}
    for node, coords in pos_nodes.items():
        pos_attrs[node] = (coords[0], coords[1] + 0.08)
        
    nx.draw_networkx_labels(G, pos_attrs, font_family='serif', font_size=20)
    
    
    if plot_weight:
        pos_attrs = {}
        for node, coords in pos_nodes.items():
            pos_attrs[node] = (coords[0], coords[1] + 0.08)
        
        nx.draw_networkx_labels(G, pos_attrs, font_family='serif', font_size=20)
        edge_labels=dict([((a,b,),d["weight"]) for a,b,d in G.edges(data=True)])
        nx.draw_networkx_edge_labels(G, pos_nodes, edge_labels=edge_labels)
    
    plt.axis('off')
    axis = plt.gca()
    axis.set_xlim([1.2*x for x in axis.get_xlim()])
    axis.set_ylim([1.2*y for y in axis.get_ylim()])
```

<!-- #region id="VCtLsW2eJJT8" -->
## Undirected Graph
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 319} id="UmOw746DHCoK" executionInfo={"status": "ok", "timestamp": 1627811674213, "user_tz": -330, "elapsed": 522, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0eab1990-60fb-4419-c4ba-343f6155509d"
G = nx.Graph()
V = {'Dublin', 'Paris', 'Milan', 'Rome'}
E = [('Milan','Dublin'), ('Milan','Paris'), ('Paris','Dublin'), ('Milan','Rome')]
G.add_nodes_from(V)
G.add_edges_from(E)
draw_graph(G, pos_nodes=nx.shell_layout(G), node_size=500)
```

```python colab={"base_uri": "https://localhost:8080/"} id="jDbM13vkHEl0" executionInfo={"status": "ok", "timestamp": 1627811302425, "user_tz": -330, "elapsed": 483, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8258cbb8-1520-43cc-b987-889594d8fedf"
print(f"V = {G.nodes}")
print(f"E = {G.edges}")
```

```python colab={"base_uri": "https://localhost:8080/"} id="mCIhACXzHmmz" executionInfo={"status": "ok", "timestamp": 1627811393880, "user_tz": -330, "elapsed": 466, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="595c271d-a83d-4cb8-f705-f06d8fe2dd31"
print(f"Graph Order: {G.number_of_nodes()}")
print(f"Graph Size: {G.number_of_edges()}")
print(f"Degree for nodes: { {v: G.degree(v) for v in G.nodes} }")
print(f"Neighbors for nodes: { {v: list(G.neighbors(v)) for v in G.nodes} }")
```

<!-- #region id="aINNkFGfII4A" -->
> Note: The neighborhood graph (also known as an ego graph) of a vertex v in a graph G is a subgraph of G, composed of the vertices adjacent to v and all edges connecting vertices adjacent to v.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="9nE2XofrHqyZ" executionInfo={"status": "ok", "timestamp": 1627811543884, "user_tz": -330, "elapsed": 529, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5abcfcf5-ecfb-4d0d-e936-7a53cb3c1a90"
ego_graph_milan = nx.ego_graph(G, "Rome")
print(f"Nodes: {ego_graph_milan.nodes}")
print(f"Edges: {ego_graph_milan.edges}")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 319} id="GJcfH73vIPJn" executionInfo={"status": "ok", "timestamp": 1627811678937, "user_tz": -330, "elapsed": 678, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a0334ed0-3343-4fcb-c536-b1841bce4a13"
new_nodes = {'London', 'Madrid'}
new_edges = [('London','Rome'), ('Madrid','Paris')]
G.add_nodes_from(new_nodes)
G.add_edges_from(new_edges)
draw_graph(G, pos_nodes=nx.shell_layout(G), node_size=500)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 319} id="pWVTcwdxIcNc" executionInfo={"status": "ok", "timestamp": 1627811682636, "user_tz": -330, "elapsed": 910, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5896276a-35af-47cb-ad29-50a31072ebb7"
node_remove = {'London', 'Rome'}
G.remove_nodes_from(node_remove)
draw_graph(G, pos_nodes=nx.shell_layout(G), node_size=500)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 319} id="qSDoOeWXIinh" executionInfo={"status": "ok", "timestamp": 1627811728936, "user_tz": -330, "elapsed": 663, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5e4c9f00-21b0-432b-e7c1-d76a184b9975"
node_edges = [('Milan','Dublin'), ('Milan','Paris')]
G.remove_edges_from(node_edges)
draw_graph(G, pos_nodes=nx.shell_layout(G), node_size=500)
```

```python colab={"base_uri": "https://localhost:8080/"} id="X2jXd1XeJCnT" executionInfo={"status": "ok", "timestamp": 1627811755753, "user_tz": -330, "elapsed": 451, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c81b1963-d3e8-4614-ba14-23061034080a"
print(nx.to_edgelist(G))
```

```python colab={"base_uri": "https://localhost:8080/"} id="UESfmIsLI8iD" executionInfo={"status": "ok", "timestamp": 1627811742289, "user_tz": -330, "elapsed": 435, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="105521a9-0c38-4a16-f7d7-0d2ce721377c"
print(nx.to_pandas_adjacency(G))
```

<!-- #region id="6fqL9VPcJHPi" -->
## Directed Graph
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 319} id="VPc7rBluJNo9" executionInfo={"status": "ok", "timestamp": 1627811823307, "user_tz": -330, "elapsed": 598, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2f09bd7b-c85d-403e-8888-16ab6129c0d4"
G = nx.DiGraph()
V = {'Dublin', 'Paris', 'Milan', 'Rome'}
E = [('Milan','Dublin'), ('Paris','Milan'), ('Paris','Dublin'), ('Milan','Rome')]
G.add_nodes_from(V)
G.add_edges_from(E)
draw_graph(G, pos_nodes=nx.shell_layout(G), node_size=500)
```

```python colab={"base_uri": "https://localhost:8080/"} id="zxzndKnxJZNS" executionInfo={"status": "ok", "timestamp": 1627811856496, "user_tz": -330, "elapsed": 432, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="421b30d3-52e1-4c67-de82-3ef42137ce93"
print(nx.to_pandas_edgelist(G))
```

```python colab={"base_uri": "https://localhost:8080/"} id="RBxjMYStJZK1" executionInfo={"status": "ok", "timestamp": 1627811857797, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7e4cdaf4-0b4d-4059-89a7-93331a4f0eb1"
print(nx.to_pandas_adjacency(G))
```

```python colab={"base_uri": "https://localhost:8080/"} id="IZOQ7AGNJZIL" executionInfo={"status": "ok", "timestamp": 1627811885540, "user_tz": -330, "elapsed": 457, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="034fcb92-06e8-4e12-fef7-78b29e5896d7"
print(f"Indegree for nodes: { {v: G.in_degree(v) for v in G.nodes} }")
print(f"Outdegree for nodes: { {v: G.out_degree(v) for v in G.nodes} }")
```

<!-- #region id="4RSSi597JqZZ" -->
## Weighted Directed Graph
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 319} id="BvjZsc7KJks4" executionInfo={"status": "ok", "timestamp": 1627811925996, "user_tz": -330, "elapsed": 631, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="50b07a3b-42cb-44e4-ee17-9426dcae0c30"
G = nx.MultiDiGraph()
V = {'Paris', 'Dublin','Milan', 'Rome'}
E = [ ('Paris','Dublin', 11), ('Paris','Milan', 8),
     ('Milan','Rome', 5),('Milan','Dublin', 19)]
G.add_nodes_from(V)
G.add_weighted_edges_from(E)
draw_graph(G, pos_nodes=nx.shell_layout(G), node_size=500, plot_weight=True)
```

```python colab={"base_uri": "https://localhost:8080/"} id="UPh6RCdeJsoK" executionInfo={"status": "ok", "timestamp": 1627811964316, "user_tz": -330, "elapsed": 457, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="be674ffa-96fd-4943-c0a3-60a97c897960"
print(nx.to_pandas_edgelist(G))
```

```python colab={"base_uri": "https://localhost:8080/"} id="sRpeHKFmJ2EP" executionInfo={"status": "ok", "timestamp": 1627811967205, "user_tz": -330, "elapsed": 433, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b9dab6e5-dc6e-4694-9757-32b2e2d9dc86"
print(nx.to_pandas_adjacency(G))
```

<!-- #region id="9RedLfIcJ2yA" -->
## Bipartite Graph
<!-- #endregion -->

```python id="0MZ7zNSzJ7Dj"
n_nodes = 10
n_edges = 12
bottom_nodes = [ith for ith in range(n_nodes) if ith % 2 ==0]
top_nodes = [ith for ith in range(n_nodes) if ith % 2 ==1]
iter_edges = zip(
    np.random.choice(bottom_nodes, n_edges),  
    np.random.choice(top_nodes, n_edges))
edges = pd.DataFrame([
    {"source": a, "target": b} for a, b in iter_edges])
B = nx.Graph()
B.add_nodes_from(bottom_nodes, bipartite=0)
B.add_nodes_from(top_nodes, bipartite=1)
B.add_edges_from([tuple(x) for x in edges.values])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 248} id="0hOH5oIDKH2Z" executionInfo={"status": "ok", "timestamp": 1627812071211, "user_tz": -330, "elapsed": 673, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8890f634-4854-4eae-9e93-951c3d31cdd1"
pos = bipartite_layout(B, bottom_nodes)
nx.draw_networkx(B, pos=pos)
```

<!-- #region id="1QZVO1lvKK_D" -->
## Multi Graph
<!-- #endregion -->

<!-- #region id="Yv9zx6UlKmUz" -->
A multigraph G is defined as G=(V, E), where V is a set of nodes and E is a multi-set (a set allowing multiple instances for each of its elements) of edges.

A multigraph is called a directed multigraph if E is a multi-set of ordered couples; otherwise, if E is a multi-set of two-sets, then it is called an undirected multigraph.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 319} id="zOO44q1rKS-a" executionInfo={"status": "ok", "timestamp": 1627812298822, "user_tz": -330, "elapsed": 29, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0143bccd-9c92-4844-aa10-cbe6aa79af0a"
G = nx.MultiDiGraph()
V = {'Dublin', 'Paris', 'Milan', 'Rome'}
E = [('Milan','Dublin'), ('Milan','Dublin'), ('Paris','Milan'), ('Paris','Dublin'), ('Milan','Rome'), ('Milan','Rome')]
G.add_nodes_from(V)
G.add_edges_from(E)

draw_graph(G, pos_nodes=nx.shell_layout(G), node_size=500)
```
