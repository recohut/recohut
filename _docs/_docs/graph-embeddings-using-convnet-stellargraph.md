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

<!-- #region id="YpmJoZkdZLce" -->
# Graph embeddings using Convnet Stellargraph
<!-- #endregion -->

<!-- #region id="lb6FvAQ3eUNs" -->
In this notebook we will be performing unsupervised graph representation learning using Graph ConvNet as encoder.

The model embeds a graph by using stacked Graph ConvNet layers
<!-- #endregion -->

```python id="-JuYVEx4WNLh"
!pip install -q stellargraph[demos]==1.2.1
```

```python id="iafwVXyrL6q6"
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

```python id="RyweACZPHYQA"
G = nx.barbell_graph(m1=10, m2=4)

order = np.arange(G.number_of_nodes())
A = nx.to_numpy_matrix(G, nodelist=order)
I = np.eye(G.number_of_nodes())
```

```python id="JgSsTLzr9a4y"
np.random.seed(7)

A_hat = A + np.eye(G.number_of_nodes()) # add self-connections

D_hat = np.array(np.sum(A_hat, axis=0))[0]
D_hat = np.array(np.diag(D_hat))
D_hat = np.linalg.inv(sqrtm(D_hat))

A_hat = D_hat @ A_hat @ D_hat

def glorot_init(nin, nout):
  sd = np.sqrt(6.0 / (nin + nout))
  return np.random.uniform(-sd, sd, size=(nin, nout))

class GCNLayer():
  def __init__(self, n_inputs, n_outputs):
      self.n_inputs = n_inputs
      self.n_outputs = n_outputs
      self.W = glorot_init(self.n_outputs, self.n_inputs)
      self.activation = np.tanh
      
  def forward(self, A, X):
      self._X = (A @ X).T # (N,N)*(N,n_outputs) ==> (n_outputs,N)
      H = self.W @ self._X # (N, D)*(D, n_outputs) => (N, n_outputs)
      H = self.activation(H)
      return H.T # (n_outputs, N)

gcn1 = GCNLayer(G.number_of_nodes(), 8)
gcn2 = GCNLayer(8, 4)
gcn3 = GCNLayer(4, 2)

H1 = gcn1.forward(A_hat, I)
H2 = gcn2.forward(A_hat, H1)
H3 = gcn3.forward(A_hat, H2)

embeddings = H3
```

```python colab={"base_uri": "https://localhost:8080/", "height": 319} id="OhVzlenz1x97" executionInfo={"status": "ok", "timestamp": 1627983060829, "user_tz": -330, "elapsed": 603, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1c63b23e-db4d-4b64-d4db-a3763ad87976"
def draw_graph(G, filename=None, node_size=50):
  pos_nodes = nx.spring_layout(G)
  nx.draw(G, pos_nodes, with_labels=False, node_size=node_size, edge_color='gray')
  
  pos_attrs = {}
  for node, coords in pos_nodes.items():
    pos_attrs[node] = (coords[0], coords[1] + 0.08)

  plt.axis('off')
  axis = plt.gca()
  axis.set_xlim([1.2*x for x in axis.get_xlim()])
  axis.set_ylim([1.2*y for y in axis.get_ylim()])

embeddings = np.array(embeddings)
draw_graph(G)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 265} id="XLgjmzRLLLcs" executionInfo={"status": "ok", "timestamp": 1627983067632, "user_tz": -330, "elapsed": 476, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4cbd95d2-f079-4d11-cfa0-cc2b6ce95622"
plt.scatter(embeddings[:, 0], embeddings[:, 1])
plt.savefig('embedding_gcn.png',dpi=300)
```

<!-- #region id="C83YCCDLG-Cv" -->
## Unsupervised GCN training using similarity graph distance
<!-- #endregion -->

<!-- #region id="VHU1UGiHfw1e" -->
In this demo, we will be using the PROTEINS dataset, already integrated in StellarGraph
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 51} id="zhttMYjFMu5f" executionInfo={"status": "ok", "timestamp": 1627983148502, "user_tz": -330, "elapsed": 7792, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d62c294d-0e2a-4e9f-851a-e0d364af01f7"
dataset = sg.datasets.PROTEINS()
display(HTML(dataset.description))
graphs, graph_labels = dataset.load()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 315} id="n1A345-rMx8V" executionInfo={"status": "ok", "timestamp": 1627983155494, "user_tz": -330, "elapsed": 541, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="745df1e4-ebc2-4e25-979a-548e565bb573"
# let's print some info to better understand the dataset
print(graphs[0].info())
graph_labels.value_counts().to_frame()
```

<!-- #region id="tVx9OQoSgViY" -->
### Model definition
It's now time to build-up the model. StellarGraph offers several utility function to load and process the dataset, as well as define the GNN model and train.
<!-- #endregion -->

```python id="gn1egwLSgUd3"
generator = sg.mapper.PaddedGraphGenerator(graphs)
```

```python id="vBJo0MkBNCLE"
# define a GCN model containing 2 layers of size 64 and 32, respectively. 
# ReLU activation function is used to add non-linearity between layers
gc_model = sg.layer.GCNSupervisedGraphClassification(
    [64, 32], ["relu", "relu"], generator, pool_all_layers=True
)
```

```python id="6WYIXEO1NHdW"
inp1, out1 = gc_model.in_out_tensors()
inp2, out2 = gc_model.in_out_tensors()

vec_distance = tf.norm(out1 - out2, axis=1)
```

```python id="dG5WFf7LNWTL"
pair_model = Model(inp1 + inp2, vec_distance)
embedding_model = Model(inp1, out1)
```

```python id="liCd_C-JKebp"
def graph_distance(graph1, graph2):
    spec1 = nx.laplacian_spectrum(graph1.to_networkx(feature_attr=None))
    spec2 = nx.laplacian_spectrum(graph2.to_networkx(feature_attr=None))
    k = min(len(spec1), len(spec2))
    return np.linalg.norm(spec1[:k] - spec2[:k])
```

```python id="wN0RSDgSKtVM"
graph_idx = np.random.RandomState(0).randint(len(graphs), size=(100, 2))
targets = [graph_distance(graphs[left], graphs[right]) for left, right in graph_idx]
train_gen = generator.flow(graph_idx, batch_size=10, targets=targets)
```

```python id="HQpoEAdvKzWL"
pair_model.compile(optimizers.Adam(1e-2), loss="mse")
```

```python id="YYVXcTkbXQLY"
history = pair_model.fit(train_gen, epochs=500, verbose=1)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 297} id="aYL3qZXYLGrX" executionInfo={"status": "ok", "timestamp": 1627983354146, "user_tz": -330, "elapsed": 139588, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4f6a6c59-bf88-4e9f-b2cc-ecedd0e23a53"
sg.utils.plot_history(history)
```

```python id="oArvDvO3LOXc"
embeddings = embedding_model.predict(generator.flow(graphs))
```

```python id="jDEfCnALMFm2"
tsne = TSNE(2)
two_d = tsne.fit_transform(embeddings)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 54} id="6XUWp7ZzMMtC" executionInfo={"status": "ok", "timestamp": 1627983377378, "user_tz": -330, "elapsed": 1653, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="484a70e9-0ba6-4a20-b866-1b2953b20225"
plt.scatter(two_d[:, 0], two_d[:, 1], c=graph_labels.cat.codes, cmap="jet", alpha=0.4)
plt.savefig('embedding_TSNE.png',dpi=300)
```
