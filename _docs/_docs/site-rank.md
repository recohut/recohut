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

<!-- #region id="tRtTnlRZ_ejE" -->
# Using Basic Graph Theory to Rank Websites by Popularity
<!-- #endregion -->

<!-- #region id="TPyxgmcQBKnM" -->
There are many data science websites on the internet. Some sites are more popular than others. Suppose you wish to estimate the most popular data science website using data that is publicly available. This precludes privately tracked traffic data. What should you do? Network theory offers us a simple way of ranking websites based on their public links. To see how, let’s build a simple network composed of two data science websites: a NumPy tutorial and a SciPy tutorial. In graph theory, these websites are referred to as the nodes in the graph. Nodes are network points that can form connections with each other; these connections are called edges. Our two website nodes will form an edge if one site links to the other or vice versa.

**Listing 18. 1. Defining a node list**
<!-- #endregion -->

```python id="43hz_afuAb08"
nodes = ['NumPy', 'SciPy']
```

<!-- #region id="6j21ze4bAb0-" -->
Suppose the _SciPy_ website is discussing NumPy dependencies. This discussion includes a web-link to the _NumPy_ page. We'll treat this connection as an edge that goes from index 1 to index 0. The edge can be expressed as the tuple `(1, 0)`. 

**Listing 18. 2. Defining an edge list**
<!-- #endregion -->

```python id="gzDMPI20Ab0_"
edges = [(1, 0)]
```

<!-- #region id="hb2-hU2AAb1A" -->
Given our directed `edges` list, we can easily check if a webpage at index `i` links a webpage at index `j`. That connection exists if `(i, j) in edges` equals `True`.

**Listing 18. 3. Checking for the existence of an edge**
<!-- #endregion -->

```python id="E3QcoSHPAb1A"
def edge_exists(i, j): return (i, j) in edges

assert edge_exists(1, 0)
assert not edge_exists(0, 1)
```

<!-- #region id="hvH-iElIAb1B" -->
Our `edge_exists` function works, but it's not efficient. The function must traverse a list to check the presence of an edge. One alternative approach is to store the presence or absence of each edge `(i, j)` within the ith row and jth column of a matrix. This matrix representation of a network is known as an **adjacency matrix**. 


**Listing 18. 4. Tracking nodes and edges using a matrix**
<!-- #endregion -->

```python id="tHy0kBHbAb1D" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637507053254, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="32fdcdb9-625a-4808-fc5d-9d28fc8b5daf"
import numpy as np
adjacency_matrix = np.zeros((len(nodes), len(nodes)))
for i, j in edges:
    adjacency_matrix[i][j] = 1
    
assert adjacency_matrix[1][0]
assert not adjacency_matrix[0][1]

print(adjacency_matrix)
```

<!-- #region id="4R5gEKryAb1F" -->
Our matrix print-out permits us to view those edges that are present in the network. Additionally, we can observe potential edges that are missing from the network. Lets turn our attention to the missing edge going from _Node 0_ to _Node 1_. We'll add that edge to our adjacency matrix. This will imply that the _NumPy_ page now links to the _SciPy_ page.

**Listing 18. 5. Adding an edge to the adjacency matrix**
<!-- #endregion -->

```python id="XHK_RPm0Ab1G" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637507055471, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3fd0ec66-b51b-460e-cd41-85618a8f4368"
adjacency_matrix[0][1] = 1
print(adjacency_matrix)
```

<!-- #region id="jzGNOr-tAb1H" -->
Suppose we wish to expand our website network by adding two more data science sites. We'll need to expand the adjacency matrix dimensions from 2-by-2 to 4-by-4. Unfortunately, in NumPy, it's quite hard to resize a matrix while maintaining all existing matrix values.  We need to switch to a different Python library; NextworkX.

### Analyzing Web Networks Using NetworkX

We'll begin by installing NetworkX. Afterwords, we'll import `networkx` as `nx`, per the common NetworkX usage convention.

**Listing 18. 6. Importing the NetworkX library**
<!-- #endregion -->

```python id="pHYencVKAb1I"
import networkx as nx
```

<!-- #region id="x9DsODcuAb1J" -->
Now, we will utilize `nx` to generate a directed graph. In NetworkX, directed graphs are tracked using the `nx.DiGraph` class.

**Listing 18. 7. Initializing a directed graph object**
<!-- #endregion -->

```python id="9dhAfNJOAb1K"
G = nx.DiGraph()
```

<!-- #region id="-hBVP2vXAb1K" -->
Lets slowly expand the directed graph. To start, we'll add a single node.

**Listing 18. 8. Adding a single node to a graph object**
<!-- #endregion -->

```python id="-4p7CgwcAb1L" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637507075592, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8b17dacf-3ce3-43c6-f5a8-c243d9ac2df7"
G.add_node(0)
print(nx.to_numpy_array(G))
```

<!-- #region id="E6oOXRP7Ab1N" -->
Our single node is associated with a _NumPy_ webpage. We can explicitly record this association by executing `G.nodes[0]['webpage'] = 'NumPy'`. 


**Listing 18. 9. Adding an attribute to an existing node**
<!-- #endregion -->

```python id="5kcdmousAb1N" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637507076369, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fd65d884-27b1-4910-ec4b-683dbb6af7b4"
def print_node_attributes():
    for i in G.nodes:
        print(f"The attribute dictionary at node {i} is {G.nodes[i]}")

print_node_attributes()
G.nodes[0]['webpage'] = 'NumPy'
print("\nWe've added a webpage to node 0")
print_node_attributes()
```

<!-- #region id="ptz1T6LlAb1O" -->
We've added a node attribute after first inserting the node into the graph. However, we can also assign attributes directly while inserting a node into the graph.

**Listing 18. 10. Adding a node with an attribute**
<!-- #endregion -->

```python id="zD9nmhSlAb1P" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637507077030, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ba2cf30b-50b8-4d12-bca5-b902b0d4a2c7"
G.add_node(1, webpage='SciPy')
print_node_attributes()
```

<!-- #region id="qgwgqHR0Ab1Q" -->

Please note that we can output all the nodes and together with their attributes simply by running `G.nodes(data=True)`.

**Listing 18. 11. Outputting nodes together with their attributes**
<!-- #endregion -->

```python id="kiEl8fPzAb1Q" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637507077790, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6d92fc6b-8484-44dd-b601-9998fc78a7f6"
print(G.nodes(data=True))
```

<!-- #region id="J7HB9nMvAb1R" -->
Now, lets add a web-link from _Node 1_ (SciPy) to _Node 0_ (NumPy). 

**Listing 18. 12. Adding a single edge to a graph object**
<!-- #endregion -->

```python id="eWtisfLCAb1R" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637507079905, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0ad7d554-91fb-434e-c1f8-7a7f6a143702"
G.add_edge(1, 0)
print(nx.to_numpy_array(G))
```

<!-- #region id="kyDenZOMAb1S" -->
Printing the adjacency matrix has given us a visual representation of the network. Unfortunately, our matrix visualization will grow cumbersome as other nodes are added.  What if instead, we plotted the network directly? Our two nodes could be plotted as two points in 2D space. Meanwhile, our single edge could be plotted as a line segment that connects these points. 

**Listing 18. 13. Plotting a graph object**
<!-- #endregion -->

```python id="MJHMMxlqAb1S" colab={"base_uri": "https://localhost:8080/", "height": 319} executionInfo={"status": "ok", "timestamp": 1637507081410, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2142d5db-a20c-4312-fda7-367ce0345968"
import matplotlib.pyplot as plt 
np.random.seed(0)
nx.draw(G)
plt.show()
```

<!-- #region id="ZJVikYlwAb1T" -->
Our plotted graph could clearly use some improvement. First of all, we need to make our arrow bigger. Also, we will benefit by adding labels to the nodes. 

**Listing 18. 14. Tweaking the graph visulization**
<!-- #endregion -->

```python id="0sDu3FPdAb1T" colab={"base_uri": "https://localhost:8080/", "height": 319} executionInfo={"status": "ok", "timestamp": 1637507082214, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="36a92840-7932-49bc-b44a-7f24766dd18b"
np.random.seed(0)
labels = {i: G.nodes[i]['webpage'] for i in G.nodes}
nx.draw(G, labels=labels, arrowsize=20)
plt.show()
```

<!-- #region id="Nc2h5xA3Ab1U" -->
The arrow is now bigger, and the node labels now partially visible. Unfortunately, these labels are obscured by the dark node color. However, we can make the labels more visible by changing the node color to something lighter, like cyan.

**Listing 18. 15.  Altering the node color**
<!-- #endregion -->

```python id="KzPpmnLzAb1U" colab={"base_uri": "https://localhost:8080/", "height": 319} executionInfo={"status": "ok", "timestamp": 1637507083678, "user_tz": -330, "elapsed": 832, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4d8bd2ed-feaa-4531-8112-5407ed3d1197"
np.random.seed(0)
nx.draw(G, labels=labels, node_color="cyan", arrowsize=20)
plt.show()
```

<!-- #region id="d_VyLGlbAb1V" -->
Within our latest plot, the labels are much more clearly visible. We see the directed link from _SciPy_ to _NumPy_. Now, lets add a reverse web-link from _NumPy_ to _SciPy_ in order to stay consistent with our earlier discussion.

**Listing 18. 16. Adding a back-link between webpages**
<!-- #endregion -->

```python id="bK8coJ4jAb1V" colab={"base_uri": "https://localhost:8080/", "height": 319} executionInfo={"status": "ok", "timestamp": 1637507083679, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5771341c-7b16-4897-d10f-ad425df007fa"
np.random.seed(0)
G.add_edge(0, 1)
nx.draw(G, labels=labels, node_color="cyan", arrowsize=20)
plt.show()
```

<!-- #region id="pXpbVlInAb1W" -->
We are now ready to expand our network by adding two more webpages; _Pandas_ and _Matplotlib_. These webpages will correspond to nodes containing ids 2 and 3, respectively.

**Listing 18. 17. Adding multiple nodes to a graph object** 
<!-- #endregion -->

```python id="mfCRs5nnAb1W" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637507084316, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="328dc485-37c7-4b80-b6d8-cb100465a63e"
webpages = ['Pandas', 'Matplotlib']
new_nodes = [(i, {'webpage': webpage})
             for i, webpage in enumerate(webpages, 2)]
G.add_nodes_from(new_nodes)

print(f"We've added these nodes to our graph:\n{new_nodes}")
print('\nOur updated list of nodes is:')
print(G.nodes(data=True))
```

<!-- #region id="4StXQ_v8Ab1W" -->
We've added the two more nodes. Lets visualize the updated graph.

**Listing 18. 18. Plotting the updated 4-node graph**
<!-- #endregion -->

```python id="xZwhu3rIAb1X" colab={"base_uri": "https://localhost:8080/", "height": 319} executionInfo={"status": "ok", "timestamp": 1637507085621, "user_tz": -330, "elapsed": 754, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="54e822a5-33d9-4e89-ba9c-b1536b36bcd6"
np.random.seed(0)
labels = {i: G.nodes[i]['webpage'] for i in G.nodes}
nx.draw(G, labels=labels, node_color="cyan", arrowsize=20)
plt.show()
```

<!-- #region id="eTeq16ifAb1X" -->
Our current web-link network is disconnected. We'll proceed to add two more web-links.

**Listing 18. 19. Adding multiple edges to a graph object** 
<!-- #endregion -->

```python id="sq7KH7viAb1Y" colab={"base_uri": "https://localhost:8080/", "height": 319} executionInfo={"status": "ok", "timestamp": 1637507087015, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="45e6d029-4177-4cba-c9b9-fed3f5139a0f"
np.random.seed(1)
G.add_edges_from([(0, 2), (3, 0)])
nx.draw(G, labels=labels, node_color="cyan", arrowsize=20)
plt.show()
```

<!-- #region id="XKVBOttbAb1Y" -->
We can infer that _NumPy_ is our most popular site, since it has more inbound links than any other page. We've basically developed a simple metric for ranking websites on the internet. That metric equals the number of inbound edges pointing towards the site, also known as the **in-degree**. We can also compute the in-degree directly from the graph's adjacency matrix. In order to demonstrate how, we'll first print-out our updated adjacency matrix.

**Listing 18. 20. Printing the updated adjacency matrix**
<!-- #endregion -->

```python id="cpbLDJRTAb1Y" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637507087581, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9768b925-abe9-4170-f3d4-b780a5f33750"
adjacency_matrix = nx.to_numpy_array(G)
print(adjacency_matrix)
```

<!-- #region id="B906o3xBAb1a" -->
The ith column in the matrix tracks the inbound edges of node `i`. The total number of inbound edges equals the number of non-zero ones within that column. Therefore, the sum of values in the column is equal to the node's in-degree. In general, executing `adjacency_matrix.sum(axis=0)` will return a vector of in-degrees. That vector's largest element will correspond to the most popular page in our Internet graph.

**Listing 18. 21. Computing in-degrees using the adjacency matrix** 
<!-- #endregion -->

```python id="cfZJKTDrAb1b" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637507089324, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="74a296f8-61fc-468f-e009-97c1938f4126"
in_degrees = adjacency_matrix.sum(axis=0)
for i, in_degree in enumerate(in_degrees):
    page = G.nodes[i]['webpage']
    print(f"{page} has an in-degree of {in_degree}")

top_page = G.nodes[in_degrees.argmax()]['webpage']
print(f"\n{top_page} is the most popular page.")
```

<!-- #region id="o5dVYXocAb1b" -->
Alternatively, we can compute all in-degrees using the NetworkX `in_degree` method.

**Listing 18. 22. Computing in-degrees using NetworkX**
<!-- #endregion -->

```python id="MgBQdzE0Ab1c"
assert G.in_degree(0) == 2
```

<!-- #region id="M_jFGiROAb1c" -->
Tracking the mapping between node ids and page-names can be slightly inconvenient. However, we can bypass that inconvenience by assigning string ids to individual nodes.

**Listing 18. 23. Using strings as node-ids within a graph**
<!-- #endregion -->

```python id="9sRm_pVOAb1c"
G2 = nx.DiGraph()
G2.add_nodes_from(['NumPy', 'SciPy', 'Matplotlib', 'Pandas'])
G2.add_edges_from([('SciPy', 'NumPy'), ('SciPy', 'NumPy'),
                   ('NumPy', 'Pandas'), ('Matplotlib', 'NumPy')])
assert G2.in_degree('NumPy') == 2
```

<!-- #region id="d1wlKMi0CKMX" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="lHkwc2S2B_kh" executionInfo={"status": "ok", "timestamp": 1637507153786, "user_tz": -330, "elapsed": 3188, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="35c5c7f9-b263-417b-9c86-fd956f898791"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="8imjytiuCLfp" -->
---
<!-- #endregion -->
