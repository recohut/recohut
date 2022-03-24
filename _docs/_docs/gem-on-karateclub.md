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

<!-- #region id="N53290Za_a0g" -->
# GEM on Karateclub
<!-- #endregion -->

<!-- #region id="fBxWFBr94zRD" -->
## Setup
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="dnTkJjdv4O63" executionInfo={"status": "ok", "timestamp": 1634050233892, "user_tz": -330, "elapsed": 1155, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6a7146dd-5618-49e6-8025-0b50e6f09393"
!git clone https://github.com/palash1992/GEM.git
%cd GEM
!mkdir gem/intermediate
!pip install .
```

```python id="wv3CICQl8Vd8"
%matplotlib inline
```

```python id="mSTMBrVw5Cot"
!pip install keras==2.0.2
```

```python colab={"base_uri": "https://localhost:8080/"} id="ZMVUmdOO37fs" executionInfo={"status": "ok", "timestamp": 1634050235211, "user_tz": -330, "elapsed": 682, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5413f63c-47d9-462c-85fc-31a1d2032db5"
%tensorflow_version 1.x
```

<!-- #region id="L5X-65dp41Ah" -->
## Karateclub Dataset
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="O40bDOFl3A6u" executionInfo={"status": "ok", "timestamp": 1634048571577, "user_tz": -330, "elapsed": 1096, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f33f2c68-0be7-4b0a-d3b4-2823f74fc424"
!wget -q --show-progress https://github.com/palash1992/GEM/raw/master/examples/data/karate.edgelist
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="JK5JobnA8E3t" executionInfo={"status": "ok", "timestamp": 1634049624183, "user_tz": -330, "elapsed": 51226, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8cd6c962-c2dd-479c-83cc-e0d2b3bc65e0"
'''
Run the graph embedding methods on Karate graph and evaluate them on 
graph reconstruction and visualization. Please copy the 
gem/data/karate.edgelist to the working directory
'''
import matplotlib.pyplot as plt
from time import time

from gem.utils      import graph_util, plot_util
from gem.evaluation import visualize_embedding as viz
from gem.evaluation import evaluate_graph_reconstruction as gr

from gem.embedding.gf       import GraphFactorization
from gem.embedding.hope     import HOPE
from gem.embedding.lap      import LaplacianEigenmaps
from gem.embedding.lle      import LocallyLinearEmbedding
from gem.embedding.node2vec import node2vec
from gem.embedding.sdne     import SDNE
from argparse import ArgumentParser


if __name__ == '__main__':
    ''' Sample usage
    python run_karate.py -node2vec 1
    '''
    parser = ArgumentParser(description='Graph Embedding Experiments on Karate Graph')
    parser.add_argument('-node2vec', '--node2vec',
                        help='whether to run node2vec (default: False)')
    args = vars(parser.parse_args(args={}))
    try:
        run_n2v = bool(int(args["node2vec"]))
    except:
        run_n2v = False

    # File that contains the edges. Format: source target
    # Optionally, you can add weights as third column: source target weight
    edge_f = 'karate.edgelist'
    # Specify whether the edges are directed
    isDirected = True

    # Load graph
    G = graph_util.loadGraphFromEdgeListTxt(edge_f, directed=isDirected)
    G = G.to_directed()

    models = []
    # Load the models you want to run
    models.append(GraphFactorization(d=2, max_iter=50000, eta=1 * 10**-4, regu=1.0, data_set='karate'))
    models.append(HOPE(d=4, beta=0.01))
    models.append(LaplacianEigenmaps(d=2))
    models.append(LocallyLinearEmbedding(d=2))
    if run_n2v:
        models.append(
            node2vec(d=2, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=1, inout_p=1)
        )
    models.append(SDNE(d=2, beta=5, alpha=1e-5, nu1=1e-6, nu2=1e-6, K=3,n_units=[50, 15,], rho=0.3, n_iter=50, xeta=0.01,n_batch=100,
                    modelfile=['enc_model.json', 'dec_model.json'],
                    weightfile=['enc_weights.hdf5', 'dec_weights.hdf5']))

    # For each model, learn the embedding and evaluate on graph reconstruction and visualization
    for embedding in models:
        print ('Num nodes: %d, num edges: %d' % (G.number_of_nodes(), G.number_of_edges()))
        t1 = time()
        # Learn embedding - accepts a networkx graph or file with edge list
        Y, t = embedding.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
        print (embedding._method_name+':\n\tTraining time: %f' % (time() - t1))
        # Evaluate on graph reconstruction
        MAP, prec_curv, err, err_baseline = gr.evaluateStaticGraphReconstruction(G, embedding, Y, None)
        #---------------------------------------------------------------------------------
        print(("\tMAP: {} \t preccision curve: {}\n\n\n\n"+'-'*100).format(MAP,prec_curv[:5]))
        #---------------------------------------------------------------------------------
        # Visualize
        viz.plot_embedding2D(embedding.get_embedding(), di_graph=G, node_colors=None)
        plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="4iMYcjNW85Ov" executionInfo={"status": "ok", "timestamp": 1634049725972, "user_tz": -330, "elapsed": 2187, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="62f326b0-2407-49a0-839d-81baddcb11d1"
!wget -q --show-progress https://github.com/palash1992/GEM/raw/master/examples/data/sbm_node_labels.pickle
!wget -q --show-progress https://github.com/palash1992/GEM/raw/master/examples/data/sbm.gpickle
```
