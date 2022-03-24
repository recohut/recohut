---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: py3modeling
    language: python
    name: py3modeling
---

```python id="cpgfYiuw_wHM"
import numpy as np
from scipy.linalg import sqrtm 
from scipy.special import softmax
import networkx as nx
from networkx.algorithms.community.modularity_max import greedy_modularity_communities
import matplotlib.pyplot as plt
from matplotlib import animation
%matplotlib inline
from IPython.display import HTML
```

<!-- #region id="X-A-TQwF_wHV" -->
# Message Passing as Matrix Multiplication
<!-- #endregion -->

```python id="9L6GZDEs_wHY" outputId="6eda630d-5d28-4c8c-a951-6a7298027f43"
A = np.array(
    [[0, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 1, 0, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]]
)
A
```

```python id="xV_iHm9y_wHc" outputId="51bde64d-ada8-410e-d5ed-024d9e5791de"
feats = np.arange(A.shape[0]).reshape((-1,1))+1
feats
```

```python id="S3FkjqXs_wHe" outputId="4a3ee0bc-2c61-495e-fd55-6621bf8fd1ff"
H = A @ feats
H
```

<!-- #region id="1CSpquIi_wHg" -->
## Scale neighborhood sum by neighborhood size (i.e. average values)
<!-- #endregion -->

```python id="NGXMZBNR_wHh" outputId="f97c8466-f968-47f1-c06d-f0fc92f24ac9"
D = np.zeros(A.shape)
np.fill_diagonal(D, A.sum(axis=0))
D
```

```python id="9DH5B7Ps_wHj" outputId="4df1ce9b-904f-451a-daef-296e4efaae6a"
D_inv = np.linalg.inv(D)
D_inv
```

```python id="oGNtuMhl_wHk" outputId="f627a792-8beb-4345-a957-e76639012f55"
D_inv @ A 
```

```python id="9XNzREKp_wHl" outputId="afa9774c-2c87-4ff3-b1dc-8fad8f7513e3"
H_avg = D_inv @ A @ feats
H_avg
```

<!-- #region id="uAX9AksV_wHm" -->
## Normalized Adjacency Matrix
Ultimately want to define and build:

$$ \hat{A} = \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} $$

<!-- #endregion -->

<!-- #region id="kj4DGyUu_wHn" -->
First, create $\tilde{A}$:
$$ \tilde{A} = A + I $$
<!-- #endregion -->

```python id="6JSu_-0N_wHn"
g = nx.from_numpy_array(A)
A_mod = A + np.eye(g.number_of_nodes())
```

<!-- #region id="DcknOZ1s_wHo" -->
Then create $ \tilde{D}^{-\frac{1}{2}} $, where $D$ is the diagonal degree matrix:

$$ (D)_{ij} = \delta_{i,j} \sum_k A_{i,k} $$
<!-- #endregion -->

```python id="U0erlE1f_wHp"
# D for A_mod:
D_mod = np.zeros_like(A_mod)
np.fill_diagonal(D_mod, A_mod.sum(axis=1).flatten())

# Inverse square root of D:
D_mod_invroot = np.linalg.inv(sqrtm(D_mod))
```

```python id="zNkbY-df_wHp" outputId="de792dc9-9c3c-4d5f-e8b0-44d38c62b72e"
D_mod
```

```python id="CpeCM0EE_wHq" outputId="91ecf3fd-d6b0-4002-9b30-37f63175bb87"
D_mod_invroot
```

<!-- #region id="rzfI6rMW_wHr" -->
I.e.: $\frac{1}{\sqrt{2}}$, $\frac{1}{\sqrt{3}}$, $\frac{1}{\sqrt{4}}$, ...etc
<!-- #endregion -->

```python id="C2le3Vc4_wHs"
node_labels = {i: i+1 for i in range(g.number_of_nodes())}
pos = nx.planar_layout(g)
```

```python id="GRYymwcM_wHs" outputId="a413e7d7-6b02-4c37-e672-ecebc7a94741"
fig, ax = plt.subplots(figsize=(10,10))
nx.draw(
    g, pos, with_labels=True, 
    labels=node_labels, 
    node_color='#83C167', 
    ax=ax, edge_color='gray', node_size=1500, font_size=30, font_family='serif'
)
plt.savefig('simple_graph.png', bbox_inches='tight', transparent=True)
```

```python id="CYDhUiyg_wHu" outputId="65234fff-0166-4029-830f-c4ec1b1a9576"
pos
```

<!-- #region id="Ir3SbSyr_wHu" -->
Create $\hat{A}$:

$$ \hat{A} = \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} $$

$$ (\hat{A})_{i,j} = \frac{\tilde{A}_{i,j}}{\sqrt{\tilde{d_i} \tilde{d_j}}} $$
<!-- #endregion -->

```python id="CDBaEX0j_wHv"
A_hat = D_mod_invroot @ A_mod @ D_mod_invroot
```

<!-- #region id="RB1cKpes_wHv" -->
# Water drop 
<!-- #endregion -->

```python id="Zdu5N-u-_wHv"
H = np.zeros((g.number_of_nodes(), 1))
H[0,0] = 1 # the "water drop"
iters = 10
results = [H.flatten()]
for i in range(iters):
    H = A_hat @ H
    results.append(H.flatten())
```

```python id="jZ4_eZ5L_wHw" outputId="e1272bcf-d308-47b8-89d9-2531e4fb5662"
print(f"Initial signal input: {results[0]}")
print(f"Final signal output after running {iters} steps of message-passing:  {results[-1]}")
```

```python id="Zfgi0ids_wHw" outputId="cf0bd6a6-b2d3-4b3c-9112-a9cc77b97578"
fig, ax = plt.subplots(figsize=(10, 10))

kwargs = {'cmap': 'hot', 'node_size': 1500, 'edge_color': 'gray', 
          'vmin': np.array(results).min(), 'vmax': np.array(results).max()*1.1}

def update(idx):
    ax.clear()
    colors = results[idx]
    nx.draw(g, pos, node_color=colors, ax=ax, **kwargs)
    ax.set_title(f"Iter={idx}", fontsize=20)

anim = animation.FuncAnimation(fig, update, frames=len(results), interval=1000, repeat=True)
```

```python id="ZSJ4Ri3n_wHx"
anim.save(
    'water_drop.mp4', 
    dpi=600, bitrate=-1,
    savefig_kwargs={'transparent': True, 'facecolor': 'none'},
)
```

```python id="6VZZbf0q_wHy" outputId="78accd67-3db1-4579-a9d3-7f0414e46c71"
HTML(anim.to_html5_video())
```

```python id="Xrbw6OJL_wHz"

```
