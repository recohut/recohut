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

<!-- #region id="HxK_GJNRWLZ1" -->
# Graph ML Embeddings
<!-- #endregion -->

<!-- #region id="re8Mo8jMcO6t" -->
## Installations
<!-- #endregion -->

```python id="hMoL8rQrs-vX"
!pip install node2vec
!pip install karateclub
!pip install python-Levenshtein
!pip install gensim==3.8.0
!pip install git+https://github.com/palash1992/GEM.git
!pip install stellargraph[demos]==1.2.1
```

<!-- #region id="zz4bIt0ictXI" -->
## Imports
<!-- #endregion -->

```python id="RHBnBBEod9rC"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import networkx as nx
from networkx.drawing.layout import bipartite_layout
import networkx.algorithms.community as nx_comm

import os
from scipy.io import mmread
from collections import Counter

import random
from node2vec import Node2Vec
from karateclub import Graph2Vec
from node2vec.edges import HadamardEmbedder

import os
import numpy as np
import pandas as pd
import networkx as nx

import stellargraph as sg
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GCN

import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, model_selection
from IPython.display import display, HTML
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

%matplotlib inline
```

```python id="3caHHsRhhHM9"
default_edge_color = 'gray'
default_node_color = '#407cc9'
enhanced_node_color = '#f5b042'
enhanced_edge_color = '#cc2f04'
```

<!-- #region id="_TGq1Xhocupl" -->
## Plot utils
<!-- #endregion -->

<!-- #region id="mahxU9csd_dt" -->
### Draw graph
<!-- #endregion -->

```python id="_HBWkZCYeBPX"
def draw_graph(G, pos_nodes=None, node_names={}, node_size=50, plot_weight=False):
    pos_nodes = pos_nodes if pos_nodes else nx.spring_layout(G)
    nx.draw(G, pos_nodes, with_labels=False, node_size=node_size, edge_color='gray', arrowsize=30)
    
    pos_attrs = {}
    for node, coords in pos_nodes.items():
        pos_attrs[node] = (coords[0], coords[1] + 0.08)
        
    nx.draw_networkx_labels(G, pos_attrs, labels=node_names, font_family='serif', font_size=20)
    
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

<!-- #region id="teDs4mIxhinf" -->
### Draw enhanced path on the graph
<!-- #endregion -->

```python id="mJsMLPvUhfSI"
def draw_enhanced_path(G, path_to_enhance, node_names={}, filename=None, layout=None):
    path_edges = list(zip(path,path[1:]))
    pos_nodes = nx.spring_layout(G) if layout is None else layout(G)

    pos_nodes = nx.spring_layout(G)
    nx.draw(G, pos_nodes, with_labels=False, node_size=50, edge_color='gray')
    
    pos_attrs = {}
    for node, coords in pos_nodes.items():
        pos_attrs[node] = (coords[0], coords[1] + 0.08)
        
    nx.draw_networkx_labels(G, pos_attrs, labels=node_names, font_family='serif')
    nx.draw_networkx_edges(G,pos_nodes,edgelist=path_edges, edge_color='#cc2f04', style='dashed', width=2.0)
    
    plt.axis('off')
    axis = plt.gca()
    axis.set_xlim([1.2*x for x in axis.get_xlim()])
    axis.set_ylim([1.2*y for y in axis.get_ylim()])
    
    if filename:
        plt.savefig(filename, format="png")
```

<!-- #region id="enZBIzG-tUKx" -->
## Embedding module
<!-- #endregion -->

```python id="_EV2QuzKtVTI"
class Embeddings:
    """
    Notes
    -----
    Shallow embedding methods
        These methods are able to learn and return only the embedding values 
        for the learned input data. Generally speaking, all the unsupervised 
        embedding algorithms based on matrix factorization use the same principle. 
        They all factorize an input graph expressed as a matrix in different 
        components (commonly knows as matrix factorization). The main difference 
        between each method lies in the loss function used during the optimization 
        process. Indeed, different loss functions allow creating an embedding space 
        that emphasizes specific properties of the input graph.
    """
    @staticmethod
    def generate_random_graphs(n_graphs=20, nx_class=None):
        def generate_random():
            n = random.randint(6, 20)
            k = random.randint(5, n)
            p = random.uniform(0, 1)
            if nx_class:
                return nx_class(n,k,p), [n,k,p]
            else:
                return nx.watts_strogatz_graph(n,k,p), [n,k,p]
        return [generate_random() for x in range(n_graphs)]

    def graph_embedding(self, graph_list=None, dim=2, wl_iterations=10):
        """
        Given a dataset with m different graphs, the task is to build a machine 
        learning algorithm capable of classifying a graph into the right class. 
        We can then see this problem as a classification problem, where the 
        dataset is defined by a list of pairs,  <Gi,yi> , where  Gi  is a graph 
        and  yi  is the class the graph belongs to.
        Representation learning (network embedding) is the task that aims to 
        learn a mapping function  f:G→Rn , from a discrete graph to a continuous 
        domain. Function  f  will be capable of performing a low-dimensional 
        vector representation such that the properties (local and global) of 
        graph G are preserved.
        """
        if graph_list is None:
            graph_list = self.generate_random_graphs()
        model = Graph2Vec(dimensions=dim, wl_iterations=wl_iterations)
        model.fit([x[0] for x in graph_list])
        graph_embeddings = model.get_embedding()
        return graph_list, model, graph_embeddings

    def node_embedding(self, G=None, dim=2, window=10):
        """
        Given a (possibly large) graph G=(V,E), the goal is to classify each 
        vertex v∈V into the right class. In this setting, the dataset includes  
        G and a list of pairs, <vi,yi>, where vi is a node of graph G and yi is 
        the class to which the node belongs. In this case, the mapping function 
        would be f:V→Rn.
        The Node2Vec algorithm can be seen as an extension of DeepWalk. Indeed, 
        as with DeepWalk, Node2Vec also generates a set of random walks used as 
        input to a skip-gram model. Once trained, the hidden layers of the skip-gram
        model are used to generate the embedding of the node in the graph. The main 
        difference between the two algorithms lies in the way the random walks are 
        generated.
        Indeed, if DeepWalk generates random walks without using any bias, in Node2Vec 
        a new technique to generate biased random walks on the graph is introduced. 
        The algorithm to generate the random walks combines graph exploration by merging 
        Breadth-First Search (BFS) and Depth-First Search (DFS). The way those two 
        algorithms are combined in the random walk's generation is regularized by 
        two parameters p, and q. p defines the probability of a random walk getting 
        back to the previous node, while q defines the probability that a random 
        walk can pass through a previously unseen part of the graph.
        Due to this combination, Node2Vec can preserve high-order proximities by 
        preserving local structures in the graph as well as global community structures. 
        This new method of random walk generation allows solving the limitation of 
        DeepWalk preserving the local neighborhood properties of the node.
        """
        if G is None:
            G = nx.barbell_graph(m1=7, m2=4)
        node2vec = Node2Vec(G, dimensions=dim)
        model = node2vec.fit(window=window)
        node_embeddings = [model.wv.get_vector(str(x)) for x in G.nodes()]
        return G, model, node_embeddings

    def edge_embedding(self, G=None, dim=2, window=10):
        """
        Given a (possibly large) graph G=(V,E), the goal is to classify each 
        edge e∈E , into the right class. In this setting, the dataset includes  
        G  and a list of pairs,  <ei,yi>, where ei is an edge of graph G  
        and yi is the class to which the edge belongs. Another typical task for 
        this level of granularity is link prediction, the problem of predicting 
        the existence of a link between two existing nodes in a graph. In this 
        case, the mapping function would be  f:E→Rn.
        Contrary to the other embedding function, the Edge to Vector (Edge2Vec) 
        algorithm generates the embedding space on edges, instead of nodes. 
        This algorithm is a simple side effect of the embedding generated by 
        using Node2Vec. The main idea is to use the node embedding of two adjacent 
        nodes to perform some basic mathematical operations in order to extract 
        the embedding of the edge connecting them.
        """
        G, model, _ = self.node_embedding(G=G, dim=dim, window=window)
        edges_embs = HadamardEmbedder(keyed_vectors=model.wv)
        edge_embeddings = [edges_embs[(str(x[0]), str(x[1]))] for x in G.edges()]
        return G, model, edge_embeddings

    def graph_factorization(self, G=None, data_set=None, max_iter=10000, eta=1*10**-4, regu=1.0):
        """
        The GF algorithm was one of the first models to reach good computational 
        performance in order to perform the node embedding of a given graph. The 
        loss function used in this method was mainly designed to improve GF 
        performances and scalability. Indeed, the solution generated by this 
        method could be noisy. Moreover, it should be noted, by looking at its 
        matrix factorization formulation, that GF performs a strong symmetric 
        factorization. This property is particularly suitable for undirected 
        graphs, where the adjacency matrix is symmetric, but could be a potential 
        limitation for undirected graphs.
        """
        if G is None:
            G = nx.barbell_graph(m1=7, m2=4)
        from gem.embedding.gf import GraphFactorization
        Path("gem/intermediate").mkdir(parents=True, exist_ok=True)
        model = GraphFactorization(d=2, data_set=data_set, max_iter=max_iter, eta=eta, regu=regu)
        model.learn_embedding(G)
        return G, model

    def graph_representation(self, G=None, dimensions=2, order=3):
        """
        Graph representation with global structure information (GraphRep), such 
        as HOPE, allows us to preserve higher-order proximity without forcing 
        its embeddings to have symmetric properties.
        We initialize the GraRep class from the karateclub library. In this 
        implementation, the dimension parameter represents the dimension of the 
        embedding space, while the order parameter defines the maximum number of 
        orders of proximity between nodes. The number of columns of the final 
        embedding matrix (stored, in the example, in the embeddings variable) 
        is dimension x order, since, as we said, for each proximity order an 
        embedding is computed and concatenated in the final embedding matrix.
        """
        if G is None:
            G = nx.barbell_graph(m1=7, m2=4)
        from karateclub.node_embedding.neighbourhood.grarep import GraRep
        model = GraRep(dimensions=dimensions, order=order)
        model.fit(G)
        embeddings = model.get_embedding()
        return G, model, embeddings

    def hope(self, G=None, d=4, beta=0.01):
        if G is None:
            G = nx.barbell_graph(m1=7, m2=4)
        from gem.embedding.hope import HOPE
        model = HOPE(d=d, beta=beta)
        model.learn_embedding(G)
        return G, model

    def gcn_embedding(self, G=None):
        """
        Unsupervised graph representation learning using Graph ConvNet as encoder
        The model embeds a graph by using stacked Graph ConvNet layers
        """
        if G is None:
            G = nx.barbell_graph(m1=7, m2=4)
        order = np.arange(G.number_of_nodes())
        A = nx.to_numpy_matrix(G, nodelist=order)
        I = np.eye(G.number_of_nodes())
        A_hat = A + np.eye(G.number_of_nodes()) # add self-connections
        D_hat = np.array(np.sum(A_hat, axis=0))[0]
        D_hat = np.array(np.diag(D_hat))
        D_hat = np.linalg.inv(sqrtm(D_hat))
        A_hat = D_hat @ A_hat @ D_hat
        gcn1 = GCNLayer(G.number_of_nodes(), 8)
        gcn2 = GCNLayer(8, 4)
        gcn3 = GCNLayer(4, 2)
        H1 = gcn1.forward(A_hat, I)
        H2 = gcn2.forward(A_hat, H1)
        H3 = gcn3.forward(A_hat, H2)
        embeddings = H3
        return G, embeddings
```

```python id="tgsv-QAQUd4r"
class GCNLayer():
    def __init__(self, n_inputs, n_outputs):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.W = self._glorot_init(self.n_outputs, self.n_inputs)
        self.activation = np.tanh

    @staticmethod
    def _glorot_init(nin, nout):
        sd = np.sqrt(6.0 / (nin + nout))
        return np.random.uniform(-sd, sd, size=(nin, nout))
      
    def forward(self, A, X):
        self._X = (A @ X).T # (N,N)*(N,n_outputs) ==> (n_outputs,N)
        H = self.W @ self._X # (N, D)*(D, n_outputs) => (N, n_outputs)
        H = self.activation(H)
        return H.T # (n_outputs, N)
```

<!-- #region id="rMalDeEJcvz4" -->
## Runs
<!-- #endregion -->

```python id="vr_O-a7GeHQ-"
###############################################################################
# Drawing Undirected Graphs
###############################################################################

G = nx.Graph()
V = {'Dublin', 'Paris', 'Milan', 'Rome'}
E = [('Milan','Dublin'), ('Milan','Paris'), ('Paris','Dublin'), ('Milan','Rome')]
G.add_nodes_from(V)
G.add_edges_from(E)
draw_graph(G, pos_nodes=nx.shell_layout(G), node_size=500)

new_nodes = {'London', 'Madrid'}
new_edges = [('London','Rome'), ('Madrid','Paris')]
G.add_nodes_from(new_nodes)
G.add_edges_from(new_edges)
draw_graph(G, pos_nodes=nx.shell_layout(G), node_size=500)

node_remove = {'London', 'Rome'}
G.remove_nodes_from(node_remove)
draw_graph(G, pos_nodes=nx.shell_layout(G), node_size=500)

node_edges = [('Milan','Dublin'), ('Milan','Paris')]
G.remove_edges_from(node_edges)
draw_graph(G, pos_nodes=nx.shell_layout(G), node_size=500)

G = nx.Graph()
nodes = {1:'Dublin',2:'Paris',3:'Milan',4:'Rome',5:'Naples',6:'Moscow',7:'Tokyo'}
G.add_nodes_from(nodes.keys())
G.add_edges_from([(1,2),(1,3),(2,3),(3,4),(4,5),(5,6),(6,7),(7,5)])
draw_graph(G, node_names=nodes, pos_nodes=nx.spring_layout(G), node_size=50)


###############################################################################
# Drawing Directed Graphs
###############################################################################

G = nx.DiGraph()
V = {'Dublin', 'Paris', 'Milan', 'Rome'}
E = [('Milan','Dublin'), ('Paris','Milan'), ('Paris','Dublin'), ('Milan','Rome')]
G.add_nodes_from(V)
G.add_edges_from(E)
draw_graph(G, pos_nodes=nx.shell_layout(G), node_size=500)


###############################################################################
# Drawing Weighted Directed Graphs
###############################################################################

G = nx.MultiDiGraph()
V = {'Paris', 'Dublin','Milan', 'Rome'}
E = [ ('Paris','Dublin', 11), ('Paris','Milan', 8),
     ('Milan','Rome', 5),('Milan','Dublin', 19)]
G.add_nodes_from(V)
G.add_weighted_edges_from(E)
draw_graph(G, pos_nodes=nx.shell_layout(G), node_size=500, plot_weight=True)


###############################################################################
# Drawing Shortest Path
###############################################################################

G = nx.Graph()
nodes = {1:'Dublin',2:'Paris',3:'Milan',4:'Rome',5:'Naples',6:'Moscow',7:'Tokyo'}
G.add_nodes_from(nodes.keys())
G.add_edges_from([(1,2),(1,3),(2,3),(3,4),(4,5),(5,6),(6,7),(7,5)])
path = nx.shortest_path(G, source=1, target=7)
draw_enhanced_path(G, path, node_names=nodes)


###############################################################################
# Network Efficieny Graphs
###############################################################################

G = nx.complete_graph(n=7)
nodes = {0:'Dublin',1:'Paris',2:'Milan',3:'Rome',4:'Naples',5:'Moscow',6:'Tokyo'}
ge = round(nx.global_efficiency(G),2)
ax = plt.gca()
ax.text(-.4, -1.3, "Global Efficiency:{}".format(ge), fontsize=14, ha='left', va='bottom');
draw_graph(G, node_names=nodes, pos_nodes=nx.spring_layout(G))

G = nx.cycle_graph(n=7)
nodes = {0:'Dublin',1:'Paris',2:'Milan',3:'Rome',4:'Naples',5:'Moscow',6:'Tokyo'}
le = round(nx.global_efficiency(G),2)
ax = plt.gca()
ax.text(-.4, -1.3, "Global Efficiency:{}".format(le), fontsize=14, ha='left', va='bottom');
draw_graph(G, node_names=nodes, pos_nodes=nx.spring_layout(G))


###############################################################################
# Clustering Coefficient
###############################################################################

G = nx.Graph()
nodes = {1:'Dublin',2:'Paris',3:'Milan',4:'Rome',5:'Naples',6:'Moscow',7:'Tokyo'}
G.add_nodes_from(nodes.keys())
G.add_edges_from([(1,2),(1,3),(2,3),(3,4),(4,5),(5,6),(6,7),(7,5)])
cc = nx.clustering(G)
node_size=[(v + 0.1) * 200 for v in cc.values()]
draw_graph(G, node_names=nodes, node_size=node_size, pos_nodes=nx.spring_layout(G))


###############################################################################
# Drawing Benchmark Graphs
###############################################################################

complete = nx.complete_graph(n=7)
lollipop = nx.lollipop_graph(m=7, n=3)
barbell = nx.barbell_graph(m1=7, m2=4)

plt.figure(figsize=(15,6))
plt.subplot(1,3,1)
draw_graph(complete)
plt.title("Complete")
plt.subplot(1,3,2)
plt.title("Lollipop")
draw_graph(lollipop)
plt.subplot(1,3,3)
plt.title("Barbell")
draw_graph(barbell)
```

```python id="T4kv7XNxlSwx"
###############################################################################
# Graph2vec embedding
###############################################################################

embedder = Embeddings()
_, _, graph_embeddings = embedder.graph_embedding()

fig, ax = plt.subplots(figsize=(10,10))
for i, vec in enumerate(graph_embeddings):
    ax.scatter(vec[0],vec[1], s=1000)
    ax.annotate(str(i), (vec[0],vec[1]), fontsize=40)


###############################################################################
# Node2vec embedding
###############################################################################

embedder = Embeddings()
G, model, node_embeddings = embedder.node_embedding()

fig, ax = plt.subplots(figsize=(10,10))
for x in G.nodes():
    v = model.wv.get_vector(str(x))
    ax.scatter(v[0],v[1], s=1000)
    ax.annotate(str(x), (v[0],v[1]), fontsize=12)


###############################################################################
# Edge2vec embedding
###############################################################################

embedder = Embeddings()
G, model, edge_embeddings = embedder.edge_embedding()

fig, ax = plt.subplots(figsize=(10,10))
for x,v in zip(G.edges(),edge_embeddings):
    ax.scatter(v[0],v[1], s=1000)
    ax.annotate(str(x), (v[0],v[1]), fontsize=16)


###############################################################################
# Graph Factorization embedding
###############################################################################

embedder = Embeddings()
G, model = embedder.graph_factorization()

fig, ax = plt.subplots(figsize=(10,10))
for x in G.nodes():
    v = model.get_embedding()[x]
    ax.scatter(v[0],v[1], s=1000)
    ax.annotate(str(x), (v[0],v[1]), fontsize=12)


###############################################################################
# Graph Representation embedding
###############################################################################

embedder = Embeddings()
G, model, embeddings = embedder.graph_representation()

fig, ax = plt.subplots(1, 3, figsize=(16,5))
for x in G.nodes():
    v = model.get_embedding()[x]
    ax[0].scatter(v[0],v[1], s=1000)
    ax[0].annotate(str(x), (v[0],v[1]), fontsize=12)
    ax[0].set_title('k=1')
    ax[1].scatter(v[2],v[3], s=1000)
    ax[1].annotate(str(x), (v[2],v[3]), fontsize=12)
    ax[1].set_title('k=2')
    ax[2].scatter(v[4],v[5], s=1000)
    ax[2].annotate(str(x), (v[4],v[5]), fontsize=12)
    ax[2].set_title('k=3')


###############################################################################
# HOPE embedding
###############################################################################

embedder = Embeddings()
G, model = embedder.hope()

fig, ax = plt.subplots(figsize=(10,10))
for x in G.nodes():
    v = model.get_embedding()[x,2:]
    ax.scatter(v[0],v[1], s=1000)
    ax.annotate(str(x), (v[0],v[1]), fontsize=20)
```
