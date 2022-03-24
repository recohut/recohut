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

<!-- #region id="GDRzELU-ZruG" -->
# Graph embeddings using SDNE
<!-- #endregion -->

```python id="DIZ33_S_Bxrz"
!pip install git+https://github.com/palash1992/GEM.git
!pip install -U Ipython
```

```python colab={"base_uri": "https://localhost:8080/"} id="hpvRCtZPQR6M" executionInfo={"status": "ok", "timestamp": 1627981550057, "user_tz": -330, "elapsed": 771, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="64318234-214a-4e88-a0bb-2e353520c4cc"
%tensorflow_version 1.x
```

```python id="q_TCzoMJQSbG"
from gem.embedding.sdne import SDNE
from matplotlib import pyplot as plt
from IPython.display import Code
import networkx as nx
import inspect
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="CgRJ2RVuQ1Xw" executionInfo={"status": "ok", "timestamp": 1627981590252, "user_tz": -330, "elapsed": 21, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="98811583-f2c3-4af8-d602-04d6d891ba84"
Code(inspect.getsource(SDNE), language='python')
```

```python id="Ho9iK2TEQWYi"
graph = nx.karate_club_graph()

m1 = SDNE(d=2, beta=5, alpha=1e-5, nu1=1e-6, nu2=1e-6, K=3,n_units=[50, 15,], rho=0.3, n_iter=50, 
          xeta=0.01,n_batch=100,
          modelfile=['enc_model.json', 'dec_model.json'],
          weightfile=['enc_weights.hdf5', 'dec_weights.hdf5'])
```

```python id="A5Mq5ReCRkTG"
m1.learn_embedding(graph)

x, y = list(zip(*m1.get_embedding()))
```

```python id="3CNTNS2qRarA"
plt.plot(x, y, 'o',linewidth=None)
```
