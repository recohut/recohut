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

# Graph Embeddings using Autoencoder

```python
# !pip install git+https://github.com/palash1992/GEM.git
# !pip install -U Ipython
%tensorflow_version 1.x
```

```python
from gem.embedding.sdne import SDNE
from matplotlib import pyplot as plt
from IPython.display import Code
import networkx as nx
import inspect
```

```python
Code(inspect.getsource(SDNE), language='python')
```

```python
graph = nx.karate_club_graph()

m1 = SDNE(d=2, beta=5, alpha=1e-5, nu1=1e-6, nu2=1e-6, K=3,n_units=[50, 15,], rho=0.3, n_iter=50, 
          xeta=0.01,n_batch=100,
          modelfile=['enc_model.json', 'dec_model.json'],
          weightfile=['enc_weights.hdf5', 'dec_weights.hdf5'])
```

```python
m1.learn_embedding(graph)

x, y = list(zip(*m1.get_embedding()))
```

```python
plt.plot(x, y, 'o',linewidth=None)
```
