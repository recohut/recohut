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

<!-- #region id="l5uXqP2G7ov1" -->
# DeepWalk in python
<!-- #endregion -->

<!-- #region id="_Ue1g8397t0o" -->
## Imports
<!-- #endregion -->

```python id="V_CAT4tG7om_"
import random
import networkx as nx 
from gensim.models import Word2Vec

import numpy as np
from abc import ABC
import pandas as pd
```

<!-- #region id="_GAkUKgmTkZR" -->
## Deepwalk
<!-- #endregion -->

```python id="6DW32rssTing"
class DeepWalk:
  """
  Implement DeepWalk algorithm.
  reference paper : DeepWalk: Online Learning of Social Representations
  link : https://arxiv.org/abs/1403.6652
  Using the algorithm can get graph embedding model with your network data.
  """
  def __init__(self, G=None, adjlist_path=None, edgelist_path=None):
    """
    Parameters
    G : networkx : networkx graph.
    
    adjlist_path : network file path. 
    """
    if G == adjlist_path == edgelist_path == None:
      raise ValueError('all parameter is None, please check your input.')
      
    try:
      
      if G != None:
        self.G = G
      elif adjlist_path != None:
        self.G = nx.read_adjlist(adjlist_path)
      elif edgelist_path != None:
        self.G = nx.read_edgelist(edgelist_path)

    except Exception as e:
      print(e)


  def random_walk(self, iterations, start_node=None, random_walk_times=5):
    """
    : Implement of random walk algorithm :
    Parameters
    ----------------------------------------
    iterations : int : random walk number of iteration 
    start_node : str : choose start node (random choose a node, if start_node is None)
    random_walk_times : int : random walk times.
    ----------------------------------------
    Returns
    walk_records : list of walks record
    """
    walk_records = []

    for i in range(iterations):
      
      if start_node is None:
        s_node = random.choice(list(self.G.nodes()))
        walk_path = [s_node]
      else:
        walk_path = [start_node]
        
      current_node = s_node
      while(len(walk_path) < random_walk_times):
        neighbors = list(self.G.neighbors(current_node))
        
        
        current_node = random.choice(neighbors)
        walk_path.append(current_node)
          
      walk_records.append(walk_path)
    
    return walk_records

  def buildWord2Vec(self, **kwargs):
    """
    
    Using gensim to build word2vec model
    Parameters
    ----------------------------------------
    **kwargs
    
    walk_path : list : random walk results
    size : int : specific embedding dimension, default : 100 dim
    window : int : specific learn context window size, default : 5
    workers : int : specific workers. default : 2
    ----------------------------------------
    Returns
    walk_records : list of walks record
    """
    
    walk_path = kwargs.get('walk_path', None)
    if walk_path is None:
      return 
    
    size = kwargs.get('size', 100)
    window = kwargs.get('window', 5)
    workers = kwargs.get('workers', 2)

    embedding_model = Word2Vec(walk_path, size=size, window=window, min_count=0, workers=workers, sg=1, hs=1)

    return embedding_model
```

<!-- #region id="B7Kv5bfhaIsh" -->
## Hierarchical Softmax
First, we'll build the components required to use hierarchical softmax. From the paper:

Computing the partition function (normalization factor) is expensive. If we assign the vertices to the leaves of a binary tree, the prediction problem turns into maximizing the probability of a specific path in the tree

Thus, instead of having a classifier that predicts probabilities for each word from our vocabulary (besides the one we're currently iterating on), we can structure the loss function as a binary tree where every internal node contains its own binary classifier. Computing the loss (and gradient) can therefore be done in $O(logv)$ predictions rather than $O(v)$ (as is the case with $v$ labels), where $v$ is the number of vertices in our graph.
<!-- #endregion -->

```python id="OPTTtTT9aIpg"
class Tree(ABC): 
    @staticmethod
    def merge(dims, lr, batch_size, left=None, right=None):
        if left is not None: left.set_left()
        if right is not None: right.set_right()
        return InternalNode(dims, lr, batch_size, left, right)
    
    @staticmethod
    def build_tree(nodes, dims, lr, batch_size):
        if len(nodes) % 2 != 0: nodes.append(None)
        while len(nodes) > 1:
            nodes = [Tree.merge(dims, lr, batch_size, nodes[i], nodes[i+1]) for i in range(0, len(nodes) - 1, 2)]
        return nodes[0]
        
    def set_parent(self, t):
        self.parent = t
        
    def set_left(self): self.is_right = False
        
    def set_right(self): self.is_right = True
```

```python id="Ivhz8tvMaWZg"
class InternalNode(Tree):
    def __init__(self, dims, lr, batch_size, left=None, right=None, parent=None, is_right=None):
        self.dims = dims
        self.set_left_child(left)
        self.set_right_child(right)
        self.set_parent(parent)
        self.is_right = is_right
        self.params = np.random.uniform(size=self.dims) 
        self.gradients = []
        self.lr = lr
        self.batch_size= batch_size
        
    def set_left_child(self, child: Tree):
        self.left = child
        if self.left is not None:
            self.left.set_parent(self)
            self.left.set_left()
            
    def set_right_child(self, child: Tree):
        self.right = child
        if self.right is not None:
            self.right.set_parent(self)
            self.right.set_right()
            
    def set_parent(self, parent: Tree):
        self.parent = parent    
        
    def predict(self, embedding, right=True):
        d = self.params.dot(embedding) if right else -self.params.dot(embedding)
        return 1/(1+np.exp(-d))
    
    def update_gradients(self, gradient: np.array):
        self.gradients.append(gradient)
        if len(self.gradients) >= self.batch_size:
            avg_gradient = np.stack(self.gradients, axis=0).mean(axis=0)
            self.params = self.params - self.lr * avg_gradient
            self.gradients = []
        
    def __eq__(self, other):
        return (
            self.dims == other.dims and
            self.left == other.left and
            self.right == other.right and
            self.lr == other.lr and
            self.batch_size == other.batch_size
        )
```

```python id="s4cUO_97aZgW"
class Leaf(Tree):
    def __init__(self, vertex, parent: InternalNode = None, is_right = False):
        self.parent = parent
        self.is_right = is_right 
        self.vertex = vertex
        
    def update(self, anchor_vertex):
        node = self
        gradients = []
        total_cost = 0.
        emb_grads = []
        while node.parent is not None:
            is_right = node.is_right
            node = node.parent        
            prob = node.predict(anchor_vertex.embedding, is_right)
            log_prob = np.log(prob)
            total_cost -= log_prob
            u = 1 - prob
            node.update_gradients(u*anchor_vertex.embedding)
            emb_grads.append(u*node.params)
        anchor_vertex.update_embedding(sum(emb_grads))
        return total_cost
```

```python id="mE_0ks6PabTD"
class Vertex(object):
    def __init__(self, dim, lr, batch_size):
        self.dim = dim
        self.embedding = np.random.uniform(size=dim)
        self.lr = lr
        self.gradients = []
        self.batch_size = batch_size
        
    def update_embedding(self, gradient: np.array): 
        self.gradients.append(gradient)
        if len(self.gradients) >= self.batch_size:
            avg_gradient = np.stack(self.gradients, axis=0).mean(axis=0)
            self.embedding = self.embedding - self.lr * avg_gradient
            self.gradients = []
```

```python id="Ux9slFZ-adiw"
v = Vertex(8, 1e-1, 1)
v2 = Vertex(8, 1e-1, 1)
leaf = Leaf(v)
leaf2 = Leaf(v2)
i = InternalNode(8, 1e-1, 1, leaf, leaf2)
```

```python colab={"base_uri": "https://localhost:8080/"} id="3dp2DCMgai2j" executionInfo={"status": "ok", "timestamp": 1633185052067, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="41d60917-dfc5-4dc2-a955-d6a34fdadf7a"
before = leaf2.vertex.embedding
before_parent = leaf.parent.params
print(before)
```

```python colab={"base_uri": "https://localhost:8080/"} id="6nTMuDVBajHJ" executionInfo={"status": "ok", "timestamp": 1633185058468, "user_tz": -330, "elapsed": 703, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9b3e5a23-41f5-4d17-eb78-946212109211"
leaf.update(leaf2.vertex)
after = leaf2.vertex.embedding
after_parent = leaf.parent.params
print(after)
```

<!-- #region id="fSmBZypRakhm" -->
Leaves 1 and 2 should share parent i. Also, each should have its own vertex (v and v2 respectively).
<!-- #endregion -->

```python id="_pY0JOE_amtM"
assert leaf.vertex == v
assert leaf.vertex != v2
assert leaf2.vertex == v2
assert leaf2.vertex != v
assert leaf.parent == i
assert leaf2.parent == i
```

<!-- #region id="NRvDC5rnanwm" -->
As a convenience method, we have Tree.merge which should do the same thing as the manual passing to the InternalNode constructor above.
<!-- #endregion -->

```python id="fb6NsUPlapka"
i2 = Tree.merge(8, 1e-1, 1, leaf, leaf2)
```

```python id="2w1yuw3Waqpr"
assert i2 == i
```

<!-- #region id="dYGVHNvYar0D" -->
We should be able to create an internal node with a single child.
<!-- #endregion -->

```python id="yiveInG5atPm"
i3 = InternalNode(8, 0.01, 1, leaf)
assert i3.left == leaf
assert i3.right is None
```

<!-- #region id="6lz38Mizaumb" -->
We should be able to combine two internal nodes under a third internal node.
<!-- #endregion -->

```python id="tN4e7CGDawIn"
two_internal_nodes = Tree.merge(8, 0.01, 1, i, i2)
```

```python id="iJpxRbHlaxsA"
assert two_internal_nodes.left == i
assert two_internal_nodes.right == i2
assert i.parent == two_internal_nodes
assert i2.parent == two_internal_nodes
```

```python id="fxkGgObGazPL"
p = Tree.merge(8, 1e-1, 1, leaf, leaf2)
```

```python colab={"base_uri": "https://localhost:8080/"} id="C6A0p1Mra0jv" executionInfo={"status": "ok", "timestamp": 1633185128668, "user_tz": -330, "elapsed": 532, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4d798539-0996-47cd-ee4a-5170709eb09b"
leaf.parent == leaf2.parent
```

```python colab={"base_uri": "https://localhost:8080/"} id="hlHh21ooa1u-" executionInfo={"status": "ok", "timestamp": 1633185133919, "user_tz": -330, "elapsed": 716, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2a7ca9bc-41dd-4291-8ac0-6e8e7424c848"
leaf.vertex.embedding
```

```python colab={"base_uri": "https://localhost:8080/"} id="Y948sZOYa27u" executionInfo={"status": "ok", "timestamp": 1633185139193, "user_tz": -330, "elapsed": 424, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9a5e0306-365f-43f2-b244-8636e2448fc8"
before = leaf2.vertex.embedding.copy()
before_parent = leaf.parent.params.copy()
leaf.update(leaf2.vertex)
after = leaf2.vertex.embedding
after_parent = leaf.parent.params
(before, after)
```

```python colab={"base_uri": "https://localhost:8080/"} id="V7o7Ugtwa4VA" executionInfo={"status": "ok", "timestamp": 1633185146746, "user_tz": -330, "elapsed": 515, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="48cc8c72-7984-431b-a5a1-b9b080f5c973"
(before_parent, after_parent)
```

```python id="kaZ9yTnba6I1"
assert leaf.parent.predict(leaf2.vertex.embedding, right=False) + leaf.parent.predict(leaf2.vertex.embedding)
```

```python colab={"base_uri": "https://localhost:8080/"} id="rZQVRYJNa7Qg" executionInfo={"status": "ok", "timestamp": 1633185158126, "user_tz": -330, "elapsed": 425, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="27781dd6-6951-4189-a1f2-e9acfcca76a5"
leaf.parent.predict(leaf2.vertex.embedding)
```

```python colab={"base_uri": "https://localhost:8080/"} id="zwNqR-Cpa88M" executionInfo={"status": "ok", "timestamp": 1633185166246, "user_tz": -330, "elapsed": 619, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a4519140-c8c3-465a-c6c5-7ef3db21c18c"
new_leaf = Leaf(Vertex(8, 0.01, 1))
new_leaf2 = Leaf(Vertex(8, 0.01, 1))
merged = Tree.merge(8, 0.01, 1, new_leaf, new_leaf2)
before1 = new_leaf2.vertex.embedding.copy()
new_leaf.update(new_leaf2.vertex)
after1 = new_leaf2.vertex.embedding
(before1, after1)
```

```python colab={"base_uri": "https://localhost:8080/"} id="aQNA3TD1a-2D" executionInfo={"status": "ok", "timestamp": 1633185172323, "user_tz": -330, "elapsed": 645, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9a767b11-b153-49ba-8ede-c2cba83097a5"
before2 = new_leaf.vertex.embedding.copy()
new_leaf2.update(new_leaf.vertex)
after2 = new_leaf.vertex.embedding
(before2, after2)
```

```python id="Jmyp5cHvbAYP"
emb_length = 10
lr = 1e-3
bs = 100
v1 = Vertex(emb_length, lr, bs)
v2 = Vertex(emb_length, lr, bs)
v3 = Vertex(emb_length, lr, bs)
random_walk = [v1, v2, v3]
leaves = list(map(lambda x: Leaf(x), random_walk))
tree = Tree.build_tree(leaves, emb_length, lr, bs)
```

```python id="0fQmteX6bFb_" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633185240163, "user_tz": -330, "elapsed": 662, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e3d0d2a7-05e8-4ab8-8c76-705d4a89b127"
leaves
```

```python id="8ZjzHNozbFcA" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633185241985, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="15d2626b-42e2-4e7d-9c0c-2b197b5f5478"
tree.__class__
```

```python id="f36Fs_C1bFcC" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633185242731, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fafb1c70-5d61-469d-f9bd-38d87564e6d7"
v1.embedding.shape, v2.embedding.shape, v3.embedding.shape
```

```python id="MRKD0Yf3bFcD"
leaf1, leaf2, leaf3, empty_leaf = leaves
```

```python id="jKiUuANrbFcD" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633185244387, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b44b881b-9cf7-4b8a-ff19-e633950f81b4"
leaf3.vertex.embedding
```

```python id="OHqy3C6TbFcD" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633185244842, "user_tz": -330, "elapsed": 30, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a531b6ba-2394-43f4-cb71-6be33db1b548"
leaf1.parent, leaf2.parent, leaf3.parent
```

<!-- #region id="zXoHf1r47yvC" -->
## Plots
<!-- #endregion -->

```python id="JpFNd3dSbFcE" colab={"base_uri": "https://localhost:8080/", "height": 282} executionInfo={"status": "ok", "timestamp": 1633185246043, "user_tz": -330, "elapsed": 1223, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b39addd4-15e2-42ca-9aaf-99e1b4799ee2"
costs1 = []
costs3 = []
combined_cost = []
for i in range(10000):
    cost1 = leaf1.update(leaf2.vertex)
    cost3 = leaf3.update(leaf2.vertex)
    if i % bs == 0:
        costs1.append(cost1) 
        costs3.append(cost3)
        combined_cost.append(cost1+cost3) 
    
pd.Series(costs1).plot(kind='line')
```

```python id="Nyl5xXrybFcE" colab={"base_uri": "https://localhost:8080/", "height": 282} executionInfo={"status": "ok", "timestamp": 1633185246914, "user_tz": -330, "elapsed": 894, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e58ab367-c65b-44d0-fb5f-b72d48c857c6"
pd.Series(costs3).plot(kind='line')
```

```python id="_j2khUZ1bFcE" colab={"base_uri": "https://localhost:8080/", "height": 282} executionInfo={"status": "ok", "timestamp": 1633185248847, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3f3e35d4-c6fe-4762-f26a-39e318594dc2"
pd.Series(combined_cost).plot(kind='line')
```

```python id="EnNn5u0xbFcF"
emb_length, lr, bs = 10, 1e-4, 100
leaves = [Vertex(emb_length, lr, bs) for i in range(100)]
```

```python id="6EKibUn-bFcF"
leaves = [Leaf(v) for v in leaves]
```

```python id="flNh5_jRbFcF"
tree = Tree.build_tree(leaves, emb_length, lr, bs)
```

```python id="gma5jbSzbFcF"
chosen_leaf = leaves[20]
```

```python id="Pam4uPUBbFcG" colab={"base_uri": "https://localhost:8080/", "height": 72} executionInfo={"status": "ok", "timestamp": 1633185274998, "user_tz": -330, "elapsed": 23762, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b97035cc-c4a5-4416-9915-911b8e5741ea"
#slow
costs = []
num_iter = 3000
epoch_costs = [] 
for it in range(num_iter):
    for i in range(100):
        if i == 20:
            continue
        costs.append(leaves[i].update(chosen_leaf.vertex)) 
    epoch_costs.append(np.mean(costs))
    costs = []
s = pd.Series(epoch_costs)
s.plot(kind='line')
```

<!-- #region id="AdYUmWzbbFcG" -->
This is an interesting result -- it seems a little unusual that we would see training loss going up, but some things to consider:
* In the "real" version, the leaf embeddings are (hopefully) going to have some relationship with the internal node model parameters. In this toy version, we've uniformly initialized all parameters and then trained the model on every single leaf for many iterations. It's basically learning how to optimize random noise.
* We're using plain vanilla batch GD here, with no learning rate annealing (or any of the wide number of GD enhancements that exist). It's very possible that we're getting gradient explosions / divergence towards the end here. 
<!-- #endregion -->

<!-- #region id="4Zu0lhlHbFcG" -->
The goal of hierarchical softmax is to make the scoring function run in $O(logv)$ rather than $O(v)$ by organizing the nodes as a binary tree with a binary classifier at each internal node. At a high level, we follow these steps:
1. We identify a leaf that is contained within the window of our vertex within the current random walk
2. We take that leaf's parent and compute the probability of having followed the correct path (left or right) to the leaf we identified in step 1 by using the model parameters for this internal node combined with the features for the current vertex (which is a row in $\Phi$).
3. We repeat step 2 for all internal nodes until we get to the root
4. The product of all of the internal probabilities gives us the probability of seeing a co-occurrence of the neighbor node given what we know about the node we're exploring
5. $-logPr(u_k|\Phi(v_j))$ is our loss function, where $Pr(u_k|\Phi(v_j))$ is the probability we calculated in step 4
6. We use the loss in step 5 to perform a gradient descent step updating both the parameters of our model and $\Phi(v_j)$:

$$\theta \leftarrow \theta - \alpha_\theta * \frac{\partial J}{\partial \theta}$$
<br>
$$\Phi \leftarrow \Phi - \alpha_\Phi * \frac{\partial J}{\partial \Phi}$$

Where $\theta$ represents all of the parameters of all of the models in the internal nodes of the tree, and $\Phi$ represents the latent representation of the current vertex.
<!-- #endregion -->
