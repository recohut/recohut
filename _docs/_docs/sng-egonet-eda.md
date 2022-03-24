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

# Social Network Graph EgoNet EDA


The growth of social networking sites has been one of the most active trends in digital media over the years. Since the late 1990s, when the first social applications were published, they have attracted billions of active users worldwide, many of whom have integrated digital social interactions into their daily lives. New ways of communication are being driven by social networks such as Facebook, Twitter, and Instagram, among others. Users can share ideas, post updates and feedback, or engage in activities and events while sharing their broader interests on social networking sites.

Besides, social networks constitute a huge source of information for studying user behaviors, interpreting interaction among people, and predicting their interests. Structuring them as graphs, where a vertex corresponds to a person and an edge represents the connection between them, enables a powerful tool to extract useful knowledge.

However, understanding the dynamics that drive the evolution of a social network is a complex problem due to a large number of variable parameters.


In this series, we will talk about how we can analyze the Facebook social network using graph theory and how we can solve useful problems such as link prediction and community detection using machine learning.


We will be using the Social circles [SNAP Facebook public dataset](https://snap.stanford.edu/data/ego-Facebook.html), from Stanford University.

The dataset was created by collecting Facebook user information from survey participants. Ego networks were created from 10 users. Each user was asked to identify all the circles (list of friends) to which their friends belong. On average, each user identified 19 circles in their ego networks, where each circle has on average 22 friends.

For each user, the following information was collected:
- Edges: An edge exists if two users are friends on Facebook.
- Node features: Features were labeled 1 if the user has this property in their profile and 0 otherwise. Features have been anonymized since the names of the features would reveal private data.

The 10 ego networks were then unified in a single graph that we are going to study.

```python
!wget http://snap.stanford.edu/data/facebook_combined.txt.gz
!wget http://snap.stanford.edu/data/facebook.tar.gz
!gzip -d facebook_combined.txt.gz
!tar -xf facebook.tar.gz
```

```python
!head facebook_combined.txt
```

```python
!ls ./facebook
```

```python
!cat ./facebook/0.circles
```

```python
!head -5 ./facebook/0.edges
```

```python
!head -5 ./facebook/0.egofeat
```

```python
!head -20 ./facebook/0.feat
```

```python
!head -10 ./facebook/0.featnames
```

| File | Description |
| ---- | ----------- |
| nodeId.edges | The edges in the ego network for the node 'nodeId'. Edges are undirected for facebook, and directed (a follows b) for twitter and gplus. The 'ego' node does not appear, but it is assumed that they follow every node id that appears in this file. |
| nodeId.circles | The set of circles for the ego node. Each line contains one circle, consisting of a series of node ids. The first entry in each line is the name of the circle. |
| nodeId.feat | The features for each of the nodes that appears in the edge file. |
| nodeId.egofeat | The features for the ego user. |
| nodeId.featnames | The names of each of the feature dimensions. Features are '1' if the user has this property in their profile, and '0' otherwise. This file has been anonymized for facebook users, since the names of the features would reveal private data. |
| facebook_combined.txt | a list of edges from all the ego networks combined. |


Environment setup

```python
!pip install community
```

```python
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import community
from community import community_louvain
import networkx as nx
import networkx.algorithms.community as nx_comm

%matplotlib inline

default_edge_color = 'gray'
default_node_color = '#407cc9'
enhanced_node_color = '#f5b042'
enhanced_edge_color = '#cc2f04'
```

Building graph layout

```python
G = nx.read_edgelist("facebook_combined.txt", create_using=nx.Graph(), nodetype=int)
print(nx.info(G))
```

> Note: Our combined network has 4,039 nodes and more than 80,000 edges.

```python
# let's first create a list of participant ids - we call it ego nodes in literature it seems
ego_nodes = set([int(name.split('.')[0]) for name in os.listdir("./facebook/")])
ego_nodes
```

```python
# let's try to visualize the network
spring_pos = nx.spring_layout(G)
plt.axis("off")
nx.draw_networkx(G, pos=spring_pos, node_color=default_node_color,
                 edge_color=default_edge_color, with_labels=False,
                 node_size=35)
plt.show()
```

Network analysis

```python
def draw_metric(G, dct, spring_pos):
  """ draw the graph G using the layout spring_pos.
      The top 10 nodes w.r.t. values in the dictionary dct
      are enhanced in the visualization """
  top = 10
  max_nodes =  sorted(dct.items(), key = lambda v: -v[1])[:top]
  
  max_keys = [key for key,_ in max_nodes]
  max_vals = [val*300 for _, val in max_nodes]

  plt.axis("off")
  
  nx.draw_networkx(G, 
                   pos=spring_pos, 
                   cmap='Blues', 
                   edge_color=default_edge_color,
                   node_color=default_node_color, 
                   node_size=3,
                   alpha=0.4, 
                   with_labels=False)
  
  nx.draw_networkx_nodes(G, 
                         pos=spring_pos, 
                         nodelist=max_keys, 
                         node_color=enhanced_edge_color,
                         node_size=max_vals)
```

### Topology analysis


> Note: Assortativity reveals information about the tendency of users to be connected with users with a similar degree.

```python
# assortativity
assortativity = nx.degree_pearson_correlation_coefficient(G)
assortativity
```

Here we can observe a positive assortativity, likely showing that well-connected individuals associate with other well-connected individuals. This is expected since inside each circle users might tend to be highly connected to each other.

Transitivity could also help at better understanding how individuals are connected. Recall transitivity indicates the mean probability that two people with a common friend are themselves friends.

```python
t = nx.transitivity(G)
t
```

### Node centrality


> Note: betweenness centrality metric measures how many shortest paths pass through a given node, giving an idea of how central that node is for the spreading of information inside the network.

```python
# betweenness centrality
bC = nx.betweenness_centrality(G)
np.mean(list(bC.values()))
```

The average betweenness centrality is pretty low, which is understandable given the large amount of non-bridging nodes inside the network. However, we could collect better insight by visual inspection of the graph. In particular, we will draw the combined ego network by enhancing nodes with the highest betweenness centrality.

```python
draw_metric(G,bC,spring_pos)
```

```python
# global efficiency
gE = nx.global_efficiency(G)
print(gE)
```

```python
# average clustering
aC = nx.average_clustering(G)
print(aC)
```

```python
# degree centrality
deg_C = nx.degree_centrality(G)
np.mean(list(deg_C.values()))
```

```python
draw_metric(G,deg_C,spring_pos)
```

```python
# closeness centrality
clos_C = nx.closeness_centrality(G)
np.mean(list(clos_C.values()))
```

```python
draw_metric(G,clos_C,spring_pos)
```

From the centrality analysis, it is interesting to observe that each central node seems to be part of a sort of community (this is reasonable, since the central nodes might correspond to the ego nodes of the network). It is also interesting to notice the presence of a bunch of highly interconnected nodes (especially from the closeness centrality analysis).


### Community detection


Since we are performing social network analysis, it is worth exploring one of the most interesting graph structures for social networks: communities. If you use Facebook, it is very likely that your friends reflect different aspects of your life: friends from an educational environment (high school, college, and so on), friends from your weekly football match, friends you have met at parties, and so on.

An interesting aspect of social network analysis is to automatically identify such groups. This can be done automatically, inferring them from topological properties, or semi-automatically, exploiting some prior insight.

One good criterion is to try to minimize intra-community edges (edges connecting members of different communities) while maximizing inter-community edges (connecting members within the same community).


In the following cells we will automatically detect communities using infromation from the network topology

```python
parts = community_louvain.best_partition(G)
values = [parts.get(node) for node in G.nodes()]

for node in ego_nodes:
  print(node, "is in community number", parts.get(node))
  
n_sizes = [5]*len(G.nodes())
for node in ego_nodes:
  n_sizes[node] = 250

plt.axis("off")
nx.draw_networkx(G, pos=spring_pos, cmap=plt.get_cmap("Blues"), edge_color=default_edge_color, node_color=values, node_size=n_sizes, with_labels=False)

# enhance color and size of the ego-nodes
nodes = nx.draw_networkx_nodes(G,spring_pos,ego_nodes,node_color=[parts.get(node) for node in ego_nodes])
nodes.set_edgecolor(enhanced_node_color)
```

It is interesting to notice that some ego users belong to the same community. It is possible that ego users are actual friends on Facebook, and therefore their ego networks are partially shared.

We have now completed our basic understanding of the graph structure. We now know that some important nodes can be identified inside the network. We have also seen the presence of well-defined communities to which those nodes belong. Keep in mind these observations while performing the next part of the analysis, which is applying machine learning methods for supervised and unsupervised tasks.


### Ego-net analysis


Since the combined network we are analyzing is actually composed by 10 sub-networks (ego-networks), it's interesting to inspect all those subnetwork. In the following cells we will analyze the subnetwork of the ego-user "0".

```python
G0 = nx.read_edgelist("./facebook/0.edges", create_using=nx.Graph(), nodetype=int)
for node in G0.copy():
  G0.add_edge(0,node)

plt.axis("off")
pos_G0 = nx.spring_layout(G0)
nx.draw_networkx(G0, pos=pos_G0, with_labels=False, node_size=35, edge_color=default_edge_color)
```

Nodes belonging to each subnetwork are stored in the "facebook" folder under the name nodeId.circles

```python
circles = {}

with open("./facebook/0.circles") as f_in:
  line = f_in.readline().rstrip().split("\t")
  while line and not '' in line:
    circles[line[0]] = [int(v) for v in line[1:]]
    line = f_in.readline().rstrip().split("\t")
```

```python
node_colors = [0] * G0.number_of_nodes()
count = 0
for key in circles:
  circle = circles[key]
  for node in circle:
    if node < G0.number_of_nodes():
      node_colors[node] = count
  count += 1

nx.draw_networkx(G0, pos=pos_G0, with_labels=False, node_size=35, node_color=node_colors, edge_color=default_edge_color)
```

```python
parts = community_louvain.best_partition(G0)
values = [parts.get(node) for node in G0.nodes()]

plt.axis("off")
nx.draw_networkx(G0, pos=pos_G0, cmap=plt.get_cmap("Blues"), edge_color=default_edge_color, node_color=values, node_size=35, with_labels=False)
```

```python
# community found does not reflect the circles
set(parts.values())
len(circles)
```

```python
# a node can be present in more than one list??
for i in circles:
  for j in circles:
    if i != j:
      for n1 in circles[i]:
        for n2 in circles[j]:
          if n1 == n2:
            print(n1, 'present in ',i,'found in', j)
```

```python
vals = {}
vals['Shortest path'] = nx.average_shortest_path_length(G0)
vals['Global efficiency'] = nx.global_efficiency(G0)
vals['Average clustering'] = nx.average_clustering(G0)
vals['Betweenness centrality'] = np.mean(list(nx.betweenness_centrality(G0).values()))
vals['Closeness centrality'] = np.mean(list(nx.closeness_centrality(G0).values()))
vals['Degree centrality'] = np.mean(list(nx.degree_centrality(G0).values()))
vals['Pearson correlation'] = nx.degree_pearson_correlation_coefficient(G)
vals['Transitivity'] = nx.transitivity(G)
vals['Label propagation'] = nx_comm.modularity(G, nx_comm.label_propagation_communities(G))
```

```python
pd.DataFrame(vals, index=['values']).T
```
