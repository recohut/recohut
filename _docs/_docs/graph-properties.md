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

<!-- #region id="5rhK8jExAw4c" -->
# Graph properties
<!-- #endregion -->

<!-- #region id="oqvDu21gW2ZB" -->
## Setup
<!-- #endregion -->

```python id="YUUzh2K7M7T2"
!pip install networkx==2.5 
!pip install matplotlib==3.2.2 
!pip install pandas==1.1.3 
!pip install scipy==1.6.2
```

```python id="9Qo5QzNOM9o8"
import pandas as pd
import matplotlib.pyplot as plt

import networkx as nx
import networkx.algorithms.community as nx_comm

%matplotlib inline

default_edge_color = 'gray'
default_node_color = '#407cc9'
enhanced_node_color = '#f5b042'
enhanced_edge_color = '#cc2f04'
```

<!-- #region id="Dd8lpsTnW4dU" -->
## Plot utils
<!-- #endregion -->

```python id="7Nc58ZbqNFa7"
# draw a simple graph
def draw_graph(G, node_names={}, filename=None, node_size=50):
    pos_nodes = nx.spring_layout(G)
    nx.draw(G, pos_nodes, with_labels=False, node_size=node_size, edge_color='gray')
    
    pos_attrs = {}
    for node, coords in pos_nodes.items():
        pos_attrs[node] = (coords[0], coords[1] + 0.08)
        
    nx.draw_networkx_labels(G, pos_attrs, labels=node_names, font_family='serif')
    
    plt.axis('off')
    axis = plt.gca()
    axis.set_xlim([1.2*x for x in axis.get_xlim()])
    axis.set_ylim([1.2*y for y in axis.get_ylim()])
    
    if filename:
        plt.savefig(filename, format="png")
```

```python id="kXYydQ4zNGl-"
# draw enhanced path on the graph
def draw_enhanced_path(G, path_to_enhance, node_names={}, filename=None):
    path_edges = list(zip(path,path[1:]))
    pos_nodes = nx.spring_layout(G)

    plt.figure(figsize=(5,5),dpi=300)
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

<!-- #region id="qmgfMBcMNKis" -->
## Shortest Path
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 319} id="mYhB13NpNN-5" executionInfo={"status": "ok", "timestamp": 1627813023717, "user_tz": -330, "elapsed": 1270, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3060cde8-b91b-4df7-9963-561132026eb2"
G = nx.Graph()
nodes = {1:'Dublin',2:'Paris',3:'Milan',4:'Rome',5:'Naples',6:'Moscow',7:'Tokyo'}
G.add_nodes_from(nodes.keys())
G.add_edges_from([(1,2),(1,3),(2,3),(3,4),(4,5),(5,6),(6,7),(7,5)])
draw_graph(G, node_names=nodes)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="ZOiaUGTfOir2" executionInfo={"status": "ok", "timestamp": 1627813197162, "user_tz": -330, "elapsed": 491, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="92476e29-d58e-457d-e6ff-9408325454ab"
path = nx.shortest_path(G,source=1,target=7)
' -> '.join([nodes[p] for p in path])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="aO_zVJtXNuVk" executionInfo={"status": "ok", "timestamp": 1627813219768, "user_tz": -330, "elapsed": 3447, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cc0d8455-5cba-4adb-8087-10bd150963ed"
draw_enhanced_path(G, path, node_names=nodes, filename='shortest_path.png')
```

<!-- #region id="PYF8dNSaOpxL" -->
## Characteristic path length

The characteristic path length is defined as the average of all the shortest path lengths between all possible pair of nodes. This is one of the most commonly used measures of how efficiently information is spread across a network. Networks having shorter characteristic path lengths promote the quick transfer of information and reduce costs.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="pfxLEfhEPIaO" executionInfo={"status": "ok", "timestamp": 1627813352831, "user_tz": -330, "elapsed": 431, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a7420e64-81cf-437f-bfc0-e9746fec04bd"
nx.average_shortest_path_length(G)
```

<!-- #region id="8x6cKg-5PJDA" -->
However, this metric cannot be always defined since it is not possible to compute a path among all the nodes in disconnected graphs. For this reason, network efficiency is also widely used.
<!-- #endregion -->

<!-- #region id="5XPPRcURPt7w" -->
## Efficiency

Global efficiency is the average of the inverse shortest path length for all pairs of nodes. Such a metric can be seen as a measure of how efficiently information is exchanged across a network. Efficiency is at a maximum when a graph is fully connected, while it is minimal for completely disconnected graphs. Intuitively, the shorter the path, the lower the measure.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="rqAm5T6zPXsU" executionInfo={"status": "ok", "timestamp": 1627813437127, "user_tz": -330, "elapsed": 740, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b323fa8e-c45c-4b1e-cf01-5fb14c0e3dfd"
print(nx.global_efficiency(G))
```

<!-- #region id="4FaaJ-7VP4Vm" -->
The local efficiency of a node can be computed by considering only the neighborhood of the node in the calculation, without the node itself.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="5giL3veBPdke" executionInfo={"status": "ok", "timestamp": 1627813439010, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0923c3c0-13a3-430d-82c1-c24e83a9c874"
print(nx.local_efficiency(G))
```

<!-- #region id="Eke2aVJEQACR" -->
In a fully connected graph, each node can be reached from any other node in the graph, and information is exchanged rapidly across the network. However, in a circular graph, several nodes should instead be traversed to reach the target node, making it less efficient.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 248} id="mTaBUvGrPd7-" executionInfo={"status": "ok", "timestamp": 1627813465393, "user_tz": -330, "elapsed": 1249, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="511187e3-11e0-4bec-9322-23e2db61fcbc"
# higher efficiency
G = nx.complete_graph(n=7)
nodes = {0:'Dublin',1:'Paris',2:'Milan',3:'Rome',4:'Naples',5:'Moscow',6:'Tokyo'}

ge = round(nx.global_efficiency(G),2)

# place the text box in axes coords
ax = plt.gca()
ax.text(-.4, -1.3, "Global Efficiency:{}".format(ge), fontsize=14, ha='left', va='bottom');

draw_graph(G,node_names=nodes,filename='efficiency.png')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 257} id="VAwA8GSyPkV7" executionInfo={"status": "ok", "timestamp": 1627813466593, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="64c23a8b-3486-4443-a4d7-13a6ea39a18c"
# lower efficiency
G = nx.cycle_graph(n=7)
nodes = {0:'Dublin',1:'Paris',2:'Milan',3:'Rome',4:'Naples',5:'Moscow',6:'Tokyo'}

le = round(nx.global_efficiency(G),2)

# place the text box in axes coords
ax = plt.gca()
ax.text(-.4, -1.3, "Global Efficiency:{}".format(le), fontsize=14, ha='left', va='bottom');

draw_graph(G, node_names=nodes,filename='less_efficiency.png')
```

<!-- #region id="oMyhygMiPki4" -->
Integration metrics well describe the connection among nodes. However, more information about the presence of groups can be extracted by considering segregation metrics.
<!-- #endregion -->

<!-- #region id="0wHT-ERfQI8T" -->
## Segregation
<!-- #endregion -->

<!-- #region id="K2xlwGTQypcZ" -->
### Clustering coefficient
<!-- #endregion -->

<!-- #region id="QxqdIttDQ_Qx" -->
The clustering coefficient is a measure of how much nodes cluster together. It is defined as the fraction of triangles (complete subgraph of three nodes and three edges) around a node and is equivalent to the fraction of the node's neighbors that are neighbors of each other.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 319} id="u4-VsVP5QrUp" executionInfo={"status": "ok", "timestamp": 1627813777745, "user_tz": -330, "elapsed": 1696, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="93d31bed-bd01-4580-b96d-896314fdbe5c"
G = nx.Graph()
nodes = {1:'Dublin',2:'Paris',3:'Milan',4:'Rome',5:'Naples',6:'Moscow',7:'Tokyo'}
G.add_nodes_from(nodes.keys())
G.add_edges_from([(1,2),(1,3),(2,3),(3,4),(4,5),(5,6),(6,7),(7,5)])
draw_graph(G, node_names=nodes)
```

<!-- #region id="laGtM7DURFSJ" -->
A global clustering coefficient is computed in networkx using the following command:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="IKa1VRNnVmkw" executionInfo={"status": "ok", "timestamp": 1627813782913, "user_tz": -330, "elapsed": 447, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a5e4e045-76df-4f46-aa4b-6186e546d5a9"
nx.average_clustering(G)
```

<!-- #region id="sC5qVrJ3RHEb" -->
The local clustering coefficient is computed in networkx using the following command:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="JZ1LQZ6Frq0a" executionInfo={"status": "ok", "timestamp": 1627813787410, "user_tz": -330, "elapsed": 484, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="09785aa2-ddec-4c6c-d416-6c8e8fc49a21"
nx.clustering(G)
```

<!-- #region id="NejKuHHJRkGc" -->
In the following graph, two clusters of nodes can be easily identified. By computing the clustering coefficient for each single node, it can be observed that Rome has the lowest value. Tokyo and Moscow, as well as Paris and Dublin, are instead very well connected within their respective groups (notice the size of each node is drawn proportionally to each node's clustering coefficient).
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 319} id="lWdp_yD-y0sx" outputId="eaff0220-a4b1-497e-9137-0e351597f5ac"
cc = nx.clustering(G)
node_size=[(v + 0.1) * 200 for v in cc.values()]
draw_graph(G, node_names=nodes, node_size=node_size,filename='clustering.png')
```

<!-- #region id="DHVDxlT42_TO" -->
### Transitivity
<!-- #endregion -->

<!-- #region id="MTf7k6DiRwUt" -->
A common variant of the clustering coefficient is known as transitivity. This can simply be defined as the ratio between the observed number of closed triplets (complete subgraph with three nodes and two edges) and the maximum possible number of closed triplets in the graph. Transitivity can be computed using networkx, as follows:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 34} id="qjT3Zp40y32X" outputId="44a7badd-b088-48b1-b735-9a7dd0977f52"
nx.transitivity(G)
```

<!-- #region id="senCO6bGR9IY" -->
### Modularity

Modularity was designed to quantify the division of a network in aggregated sets of highly interconnected nodes, commonly known as modules, communities, groups, or clusters. The main idea is that networks having high modularity will show dense connections within the module and sparse connections between modules.

Consider a social network such as Reddit: members of communities related to video games tend to interact much more with other users in the same community, talking about recent news, favorite consoles, and so on. However, they will probably interact less with users talking about fashion. Differently from many other graph metrics, modularity is often computed by means of optimization algorithms.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 319} id="dBZnV1goR-j6" executionInfo={"status": "ok", "timestamp": 1627814215124, "user_tz": -330, "elapsed": 738, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bf78cd50-9c3a-4fdd-f634-a31e09dd45ee"
G = nx.Graph()
nodes = {1:'Dublin',2:'Paris',3:'Milan',4:'Rome',5:'Naples',6:'Moscow',7:'Tokyo'}
G.add_nodes_from(nodes.keys())
G.add_edges_from([(1,2),(1,3),(2,3),(3,4),(4,5),(5,6),(6,7),(7,5)])
draw_graph(G, node_names=nodes)
```

<!-- #region id="uFQ3EUrxSeF-" -->
Modularity in networkx is computed using the modularity function of the networkx.algorithms.community module, as follows:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="W21anpWcSNfn" executionInfo={"status": "ok", "timestamp": 1627814190889, "user_tz": -330, "elapsed": 670, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6ec04ece-04c0-488b-be1a-e7535bd34d6f"
# partitions can be provided manually
print(nx_comm.modularity(G, communities=[{1,2,3,4},{5,6,7}]))
```

```python colab={"base_uri": "https://localhost:8080/"} id="vmN5pJQfSVQ3" executionInfo={"status": "ok", "timestamp": 1627814193694, "user_tz": -330, "elapsed": 469, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e2d0b4d3-158b-4f15-9731-b54de319c05b"
# or automatically computed using networkx
print(nx_comm.modularity(G, nx_comm.label_propagation_communities(G)))
```

<!-- #region id="Ga57kYOA1h47" -->
## Centrality
<!-- #endregion -->

```python id="VV2e-FNe1kWf" colab={"base_uri": "https://localhost:8080/", "height": 319} executionInfo={"status": "ok", "timestamp": 1627814382521, "user_tz": -330, "elapsed": 873, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="32417041-3ab2-4dd4-f575-2cb250857439"
G = nx.Graph()
nodes = {1:'Dublin',2:'Paris',3:'Milan',4:'Rome',5:'Naples',6:'Moscow',7:'Tokyo'}
G.add_nodes_from(nodes.keys())
G.add_edges_from([(1,2),(1,3),(2,3),(3,4),(4,5),(5,6),(6,7),(7,5)])
draw_graph(G, node_names=nodes)
```

<!-- #region id="fV-DSFyoS-BE" -->
One of the most common and simple centrality metrics is the degree centrality metric. This is directly connected with the degree of a node, measuring the number of incident edges on a certain node. Intuitively, the more a node is connected to an other node, the more its degree centrality will assume high values. Note that, if a graph is directed, the in-degree centrality and out-degree centrality will be considered for each node, related to the number of incoming and outcoming edges, respectively. Degree centrality is computed in networkx by using the following command:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="xj38Q2pq0unD" executionInfo={"status": "ok", "timestamp": 1627814388914, "user_tz": -330, "elapsed": 494, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ac9253fd-5505-4e25-d36e-e55fe1b16ced"
nx.degree_centrality(G)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 382} id="dCuRzRQc1zty" executionInfo={"status": "ok", "timestamp": 1627814417348, "user_tz": -330, "elapsed": 1078, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4acd8eb3-b559-42a0-e263-34116b203b8f"
dc = nx.degree_centrality(G)
node_size=[(v + 0.01) * 400 for v in dc.values()]
draw_graph(G, node_names=nodes, node_size=node_size,filename='deg_centr.png')

df = pd.DataFrame(dc,index=['Degree centrality'])
df.columns = nodes.values()
df
```

<!-- #region id="xsZUxO9dTQ1V" -->
The closeness centrality metric attempts to quantify how much a node is close (well connected) to other nodes.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="deUN9xP900v2" executionInfo={"status": "ok", "timestamp": 1627814444450, "user_tz": -330, "elapsed": 629, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="12dfaa8c-1081-45d0-901c-027d672b54af"
nx.closeness_centrality(G)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 382} id="QYqWGlda13Qy" executionInfo={"status": "ok", "timestamp": 1627814461707, "user_tz": -330, "elapsed": 1003, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2f6be7e0-465a-4361-82a0-c1e7e3338b7b"
dc = nx.closeness_centrality(G)
node_size=[(v + 0.1) * 400 for v in dc.values()]
draw_graph(G, node_names=nodes, node_size=node_size,filename='clos_centr.png')

df = pd.DataFrame(dc,index=['Closeness centrality'])
df.columns = nodes.values()
df
```

<!-- #region id="9c3VUQ8kTYIp" -->
The betweenness centrality metric evaluates how much a node acts as a bridge between other nodes. Even if poorly connected, a node can be strategically connected, helping to keep the whole network connected.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ts3sMr0H06-d" executionInfo={"status": "ok", "timestamp": 1627814467223, "user_tz": -330, "elapsed": 1321, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="77b4f768-96f9-408d-f931-cf25470eeb3d"
nx.betweenness_centrality(G)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 382} id="T29a81GV2PFk" executionInfo={"status": "ok", "timestamp": 1627814468908, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="02624c1f-6a3d-4182-ae6a-84575c6d86f0"
dc = nx.betweenness_centrality(G)
node_size=[(v + 0.1) * 400 for v in dc.values()]
draw_graph(G, node_names=nodes, node_size=node_size,filename='bet_centrality.png')

df = pd.DataFrame(dc,index=['Betweenness centrality'])
df.columns = nodes.values()
df
```

<!-- #region id="j31BgsvNSWVL" -->
## Resiliency

Resilience metrics enable us to measure the vulnerability of a graph.
<!-- #endregion -->

<!-- #region id="sO7NejTjTy12" -->
### Assortativity coefficient
Assortativity is used to quantify the tendency of nodes being connected to similar nodes. There are several ways to measure such correlations. One of the most commonly used methods is the Pearson correlation coefficient between the degrees of directly connected nodes (nodes on two opposite ends of a link). The coefficient assumes positive values when there is a correlation between nodes of a similar degree, while it assumes negative values when there is a correlation between nodes of a different degree. Assortativity using the Pearson correlation coefficient is computed in networkx by using the following command:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 319} id="6VHUtob2UGoa" executionInfo={"status": "ok", "timestamp": 1627814666852, "user_tz": -330, "elapsed": 962, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="020b4dfd-e720-483d-8cd1-884e4fc1351f"
G = nx.Graph()
nodes = {1:'Dublin',2:'Paris',3:'Milan',4:'Rome',5:'Naples',6:'Moscow',7:'Tokyo'}
G.add_nodes_from(nodes.keys())
G.add_edges_from([(1,2),(1,3),(2,3),(3,4),(4,5),(5,6),(6,7),(7,5)])

draw_graph(G, node_names=nodes)
```

```python colab={"base_uri": "https://localhost:8080/"} id="MVC82LJ1T0La" executionInfo={"status": "ok", "timestamp": 1627814666854, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3ca84aca-d025-46b9-cf1e-82067766014d"
nx.degree_pearson_correlation_coefficient(G)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 319} id="rnWW2nr3T42x" executionInfo={"status": "ok", "timestamp": 1627814601128, "user_tz": -330, "elapsed": 1042, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1e3122ef-616d-4a91-b3ff-fea8535da49e"
G = nx.Graph()
nodes = {1:'user1', 2:'user2', 3:'Football player', 4:'Fahsion blogger', 5:'user3', 6:'user4',
         7:'user5', 8:'user6'}
G.add_nodes_from(nodes.keys())
G.add_edges_from([(1,3),(2,3),(7,3),(3,4),(5,4),(6,4),(8,4)])

draw_graph(G, node_names=nodes,filename='assortativity.png')
```

```python colab={"base_uri": "https://localhost:8080/"} id="Qx_380DxT9IJ" executionInfo={"status": "ok", "timestamp": 1627814626564, "user_tz": -330, "elapsed": 472, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9e232a48-05df-433c-a0b5-22a257c69d6e"
nx.degree_pearson_correlation_coefficient(G)
```

<!-- #region id="o684wH0eUQkT" -->
Social networks are mostly assortative. However, the so-called influencers (famous singers, football players, fashion bloggers) tend to be followed (incoming edges) by several standard users, while tending to be connected with each other and showing a disassortative behavior.
<!-- #endregion -->

<!-- #region id="JwpSKlARUQ2w" -->
It is important to remark that these properties are a subset of all the possible metrics used to describe graphs. A wider set of metrics and algorithms can be found at https://networkx.org/documentation/stable/reference/algorithms/.
<!-- #endregion -->
