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

<!-- #region id="Zsuac_gEdty8" -->
# Learn Embeddings using Graph Networks
> Various short tutorials on using graph network models to learn embeddings, classification and visualization
<!-- #endregion -->

<!-- #region id="3_KS7BCiFrrO" -->
## Setup
<!-- #endregion -->

<!-- #region id="rVVOvFmKFv8W" -->
### Installation
<!-- #endregion -->

```python id="Jmz5qg14nFiC"
!git clone https://github.com/shenweichen/GraphEmbedding.git
!cd GraphEmbedding && python setup.py install
%cd /content/GraphEmbedding/examples
```

```python id="t4Y4SGUmGspR"
!pip install umap-learn
!pip install -q karateclub
```

<!-- #region id="jgqvdubWFueY" -->
### Imports
<!-- #endregion -->

```python id="aOJfY8eWBUQF" outputId="47a3f018-3234-493b-942a-1535ec3b53fb"
%tensorflow_version 1.x
```

```python id="PVz1mbY1_OqP"
from ge.classify import read_node_label, Classifier
from ge import Node2Vec, DeepWalk, LINE, SDNE

import networkx as nx
import json

import umap
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from scipy.linalg import sqrtm

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

from karateclub.utils.walker import RandomWalker, BiasedRandomWalker
from karateclub import DeepWalk, Node2Vec
from gensim.models.word2vec import Word2Vec

import torch 
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import operator

import warnings
warnings.filterwarnings("ignore")
```

```python id="-hkfIqqMotpF"
%matplotlib inline
mpl.rcParams['figure.figsize'] = 18, 7
pd.set_option('display.float_format', lambda x: '%.5f' % x)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
```

<!-- #region id="zelCXsmP_lVt" -->
### Utils
<!-- #endregion -->

```python id="7ZI7u1J1_db-"
def evaluate_embeddings(embeddings):
    X, Y = read_node_label('../data/wiki/wiki_labels.txt')
    tr_frac = 0.8
    print("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)
```

```python id="KyrJadx__hLA"
def plot_embeddings(embeddings):
    X, Y = read_node_label('../data/wiki/wiki_labels.txt')

    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()
```

<!-- #region id="IA1tVuLuJExf" -->
## Concepts
<!-- #endregion -->

<!-- #region id="nJkP1Ey_JGIU" -->
### Random Walk
<!-- #endregion -->

<!-- #region id="szxNmkzAJ8Xu" -->
<img src='https://cdn.analyticsvidhya.com/wp-content/uploads/2019/11/walk.gif'>
<!-- #endregion -->

```python id="BV4rijttJujv" outputId="84539c48-acad-42f6-d00e-d170c843e23c"
G = nx.karate_club_graph()
clubs = []
for n in G.nodes:
    c = G.nodes[n]['club']
    clubs.append(1 if c == 'Officer' else 0)
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx(G, pos=pos, node_color = clubs, cmap='coolwarm')
```

<!-- #region id="XXye5kd4JiHX" -->
Random walk is a sequence of nodes, where next node is chosen randomly from the adjacent nodes. For example, let's start our random walk from node 25. From the graph above we can see that the node 25 (right-most) is connected to the nodes 24 and 31. Hence, using a coin-flip we'll determine where we go next. If we've arrived at the node 24, we can see that it's connected to the members 23, 27, and 31. Again, we need to choose randomly where to go next. This "walk" continues until we've reached the desired walk length. Let's now write a simple function to implement this in code.
<!-- #endregion -->

```python id="QxXXJn08KCt3"
def random_walk(start_node, walk_length):
    walk = [start_node]  # starting node
    
    for i in range(walk_length):
        all_neighbours = [n for n in G.neighbors(start_node)]  # get all neighbours of the node
        next_node = np.random.choice(all_neighbours, 1)[0]  # randomly pick 1 neighbour
        walk.append(next_node)  # append this node to the walk
        start_node = next_node  # this random node is now your current state
    
    return walk
```

```python id="hNFHEpv0KJ3_" outputId="aa644071-8fcd-43d9-eb72-4b92e7858602"
walk = random_walk(6, 20)  # random walk from node 6
print('Steps in random walk:', walk)

walk_graph = G.subgraph(walk)
pos = nx.spring_layout(walk_graph, seed=42)
nx.draw_networkx(walk_graph, pos=pos, cmap='coolwarm')
```

<!-- #region id="0IrxLHn9TguA" -->
So we've generated a random walk with length of 20 starting at node 6. You can follow the steps of this walk on the graph above and see that every step is between connected nodes. By doing this walk we've got useful information about the context of the node 6. By that I mean that we now know some of the neighbours (and neighbours' neighbours) of node 6 which could be useful in classification problem for example. By repeating this random walk multiple times for all the nodes in the graph, we can get a bunch of "walk" sequences that contain useful information. The paper suggests doing around 32 walks per node with the walk length of 40. We could implement this with 2 for-loops but luckily for us, karateclub package has already implemented this for us (and it's much faster)
<!-- #endregion -->

```python id="BClD_3CyTili"
walker = RandomWalker(walk_length = 80, walk_number = 10)
walker.do_walks(G)  # you can access the walks in walker.walks 
```

<!-- #region id="ZyIEsjkgTnjP" -->
### Skip Gram
<!-- #endregion -->

<!-- #region id="kEC-XN3ST4BL" -->
Now the question is - how can we get meaningful embeddings using the generated random walks? Well, you've ever worked with NLP you already know the answer - use the Word2Vec algorithm. In particular, we're going to use the skip-gram model with hierarchical softmax layer. There are a lot of detailed resources about the inner workings of these algorithms, but here are my favourites - Word2Vec explained by [Rasa](https://www.youtube.com/watch?v=BWaHLmG1lak) and hierarchical softmax explained by [Chris McCormick](https://www.youtube.com/watch?v=pzyIWCelt_E).

The main idea of the skip-gram model is to predict the context of a sequence from a particular node (or word). For example, if we want to train embeddings for node 6 (example above), we'll train our model (usually a simple dense neural network) with the goal to predict the nodes that appear in it's random walks. So, the model's input will be the node 6 (one-hot-encoded), middle layer will be the actual embedding, and output will be prediction of the node's context. This is a very high-level explanation and I encourage you to watch the videos above if you feel confused.
<!-- #endregion -->

<!-- #region id="HkJ_yaoDqCro" -->
<img src='https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F337b4217-0e9d-40b2-9a4d-143afdc607a1%2FUntitled.png?table=block&id=dd314356-d9d9-4d93-9f80-3ce4acdeb0b6&spaceId=63b72b1f-0e90-4ab8-a6df-a060a6545a56&width=2000&userId=21ec183f-f0be-4b6b-9b3e-6f0d4e5c5469&cache=v2'>
<!-- #endregion -->

<!-- #region id="cQocoHjvUMzw" -->
we can use the gensim implementation of the algorithm to get the embeddings.
<!-- #endregion -->

```python id="NofFZ2vtTszV"
model = Word2Vec(walker.walks,  # previously generated walks
                 hs=1,  # tells the model to use hierarchical softmax
                 sg = 1,  # tells the model to use skip-gram
                 vector_size=128,  # size of the embedding
                 window=5,
                 min_count=1,
                 workers=4,
                 seed=42)
```

```python id="EVWss0u7USjP" outputId="64a3e556-67c1-4d88-be1a-2e480df4e600"
embeddings = model.wv.vectors
print('Shape of embedding matrix:', embeddings.shape)
```

<!-- #region id="eeBX8kq6Upsw" -->
And that's it! The embeddings are trained, so you can use them e.g. as features for your supervised model or to find clusters in your dataset. Let's now see how we can use DeepWalk on real classification taks.
<!-- #endregion -->

<!-- #region id="JIcoyoVuF2ZX" -->
## DeepWalk
<!-- #endregion -->

<!-- #region id="yDx151blqIV2" -->
<img src='https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F48cf4be5-e56b-409a-8518-75113e01ab2c%2FUntitled.png?table=block&id=0ed64433-5b07-47aa-b9e5-7971f7fbe308&spaceId=63b72b1f-0e90-4ab8-a6df-a060a6545a56&width=2000&userId=21ec183f-f0be-4b6b-9b3e-6f0d4e5c5469&cache=v2'>
<!-- #endregion -->

<!-- #region id="nO_OzvRLDmBv" -->
[DeepWalk](http://www.perozzi.net/publications/14_kdd_deepwalk.pdf) uses short random walks to learn representations for vertices in graphs. It is a type of graph neural network — a type of neural network that operates directly on the target graph structure. It uses a randomized path traversing technique to provide insights into localized structures within networks. It does so by utilizing these random paths as sequences, that are then used to train a Skip-Gram Language Model.

The DeepWalk process operates in 2 steps:
1. For each node, perform N “random steps” starting from that node
2. Treat each walk as a sequence of node-id strings
3. Given a list of these sequences, train a word2vec model using the Skip-Gram algorithm on these string sequences
<!-- #endregion -->

<!-- #region id="ifC_uy3JqMto" -->
<img src='https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Fb90a11f1-0793-475f-8391-f33a36532e5c%2FUntitled.png?table=block&id=2cabccb5-6261-4fbe-828e-85b5e2410c48&spaceId=63b72b1f-0e90-4ab8-a6df-a060a6545a56&width=2000&userId=21ec183f-f0be-4b6b-9b3e-6f0d4e5c5469&cache=v2'>
<!-- #endregion -->

<!-- #region id="DrkrmllV8sHn" -->
### Wiki
<!-- #endregion -->

```python id="RkxvHUDb6ng-" outputId="2d9c4236-4f5c-4bd0-c1cb-1891a188a80b"
G = nx.read_edgelist('../data/wiki/Wiki_edgelist.txt',
                      create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])

model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)
model.train(window_size=5, iter=3)
embeddings = model.get_embeddings()
```

```python id="iBzvrT_664d2" outputId="22780e70-525e-4d57-aaa1-0bb549f7bda9"
evaluate_embeddings(embeddings)
```

```python id="tnRmYbxgAtHF" outputId="ce600846-f551-4641-f673-62d3f2b4b448"
plot_embeddings(embeddings)
```

<!-- #region id="fqV-CCWqGVZq" -->
### Karate club
<!-- #endregion -->

<!-- #region id="ArLP9lvNqRxZ" -->
<img src='https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Fe3791aa4-fb5a-4ce2-915c-1e4390b7b8f3%2FUntitled.png?table=block&id=1d431d4e-1c04-4212-b9d4-7a8d02161e2b&spaceId=63b72b1f-0e90-4ab8-a6df-a060a6545a56&width=2000&userId=21ec183f-f0be-4b6b-9b3e-6f0d4e5c5469&cache=v2'>
<!-- #endregion -->

<!-- #region id="M30v9OL2Ga9t" -->
We are going to use famous Zachary's karate club dataset which comes with NetworkX package and karateclub's implementation of the DeepWalk algorithm. Each student in the graph belongs to 1 of the 2 karate clubs - Officer or Mr. Hi.
<!-- #endregion -->

```python id="jf8a69LXGbsB"
G = nx.karate_club_graph()  # load data

clubs = []  # list to populate with labels
for n in G.nodes:
    c = G.nodes[n]['club']  # karate club name, can be either 'Officer' or 'Mr. Hi'
    clubs.append(1 if c == 'Officer' else 0)
```

```python id="JkfKuCPDGbos" outputId="519c721a-a4ac-4069-ed98-508df7afcf8a"
pos = nx.spring_layout(G, seed=42) # To be able to recreate the graph layout
nx.draw_networkx(G, pos=pos, node_color = clubs, cmap='coolwarm') # Plot the graph
```

<!-- #region id="S1kxTy7BHQ5x" -->
As you can see, members of the karate clubs talk mainly to their club members. This information could be very valuable for e.g. classification or community detection tasks and we can represent it using the node embeddings.
<!-- #endregion -->

```python id="0P67b6qVGbl6" outputId="4c462034-3a77-49f1-b5d0-264c1c5c3dc3"
model = DeepWalk(dimensions=124, epochs=1)  # node embedding algorithm
model.fit(G)  # fit it on the graph
embedding = model.get_embedding()  # extract embeddings

print('Number of karate club members:', len(G.nodes))
print('Embedding array shape:', embedding.shape)
```

<!-- #region id="Q-PNSXAyHdmx" -->
Using DeepWalk (which is a black box algorithm for now) each karate club member is now represented by a vector of size 124. These vectors should reflect the graph structure, i.e. the different clubs should be far away from each other. We can check it by reducing the 124 dimensional data into 2 dimensional data using umap-learn package and making a scatter plot.
<!-- #endregion -->

```python id="sQTU0JmKHch7" outputId="2480f284-1002-4f94-fe85-31b7a906d01f"
u = umap.UMAP(random_state=42)
umap_embs = u.fit_transform(embedding)

ax = sns.scatterplot(x = umap_embs[:, 0], y = umap_embs[:, 1], hue = clubs)
a = pd.DataFrame({'x': umap_embs[:, 0], 'y': umap_embs[:, 1], 'val': G.nodes})
for i, point in a.iterrows():
    ax.text(point['x']+.02, point['y'], str(point['val']))
```

<!-- #region id="hxBaXfb5I9do" -->
As you can see, the embeddings did very well at representing the structure of the graph. Not only the two karate clubs are clearly separated but the members which are connected to the other clubs (e.g. nodes 28, 30, 8, and 2) are sort of more in the middle. In addition, the algorithm seems to have found a sub-community in the "Officer" karate club, which just shows how useful these embeddings can be. To summarise, DeepWalk (and any other node embedding algorithm) tries to represent the nodes as vectors which capture some structural information from the graph.
<!-- #endregion -->

<!-- #region id="3ahp_ltBVv_v" -->
### Facebook
<!-- #endregion -->

```python id="RNTo58v8VxP8"
!wget http://nrvis.com/download/data/soc/fb-pages-politician.zip
!unzip fb-pages-politician
```

```python id="sYlr8TwSV2XB"
edges_path = 'https://github.com/benedekrozemberczki/MUSAE/blob/master/input/edges/facebook_edges.csv?raw=true'
targets_path = 'https://github.com/benedekrozemberczki/MUSAE/blob/master/input/target/facebook_target.csv?raw=true'
features_path = 'https://github.com/benedekrozemberczki/MUSAE/blob/master/input/features/facebook.json?raw=true'
```

```python id="ZmYvBXylnp0c"
features_filename = 'facebook.json'
!wget -O $features_filename $features_path
```

```python id="bY3acw-lV2R_" outputId="6f3280ad-303b-41ee-abd7-2ffc287070fc"
edges = pd.read_csv(edges_path)
display(edges.head())
```

```python id="f35-bGvdV2Mu" outputId="658264d8-5eb7-42d5-9da8-c54df58583df"
targets = pd.read_csv(targets_path)
targets.index = targets.id
targets.head()
```

```python id="qC2VGQV5V2JD"
# Reading the json as a dict
with open(features_filename) as json_data:
    features = json.load(json_data)
```

<!-- #region id="jBk-zwyUoCZN" -->
With data read in, we can build a graph now and generate the embeddings
<!-- #endregion -->

```python id="Hq1PcBVdnjYS"
graph = nx.convert_matrix.from_pandas_edgelist(edges, "id_1", "id_2")
```

```python id="KGE9ll9enjTT" outputId="930b9bd9-f72a-4fd5-ac4b-e10d336b326f"
# Visualise some subgraph of 150 nodes
subgraph_nodes = list(nx.dfs_preorder_nodes(graph, 7))[:150] #Gets all the nodes in the graph that node 7 belongs to. 
G = graph.subgraph(subgraph_nodes)
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx(G, pos=pos, cmap='coolwarm')
```

```python id="hy685VpRnjOq"
# Do random walks
walker = RandomWalker(walk_length = 80, walk_number = 10)
walker.do_walks(graph)
```

```python id="LVu78wCynjKM"
model = Word2Vec(walker.walks,  # previously generated walks
                 hs=1,  # tells the model to use hierarchical softmax
                 sg = 1,  # tells the model to use skip-gram
                 vector_size=128,  # size of the embedding
                 window=10,
                 min_count=1,
                 workers=4,
                 seed=42)
```

<!-- #region id="FgIAG2CippfG" -->
DeepWalk model is now trained, so we can use the embeddings for classification. We can do a quick sense check of the model by looking at the nearest neighbours in the embeddings space of some of the facebook pages. For example, let's check the most similar nodes to the Facebook page of American Express (ID 22196) and the BBC's show Apprentice (ID 451)
<!-- #endregion -->

```python id="B-yr5RngnjGd" outputId="f96b6817-6760-4bf1-e972-50c9053a2162"
similar_to = '22196'
targets.loc[[int(similar_to)] + [int(v[0]) for v in model.wv.most_similar(similar_to)], :].head()
```

```python id="VKaQmoSTnjCy" outputId="889e041b-df9f-43aa-fcda-437a93d87437"
similar_to = '451'
targets.loc[[int(similar_to)] + [int(v[0]) for v in model.wv.most_similar(similar_to)], :].head()
```

<!-- #region id="DdnxgRdLqDY2" -->
As you can see, the nearest neighbours are incredibly similar to the original pages and all of this is achieved without even knowing what the original pages are about! Hence, the embeddings that the DeepWalk has learned are meaningful and we can use them in the classifier. We can build a simple Random Forest model to see what performance we can achieve using purely the embeddings.
<!-- #endregion -->

```python id="B5bkLbTpni_G"
# Get targets 
y = targets.loc[[int(i) for i in list(features.keys())], 'page_type']

# Get corresponding embeddings
X_dw = []
for i in y.index:
    X_dw.append(model.wv.__getitem__(str(i)))
```

```python id="-yZDVcxwni6J" outputId="886e19f9-a5ea-4fa4-df14-16b0c2f599c0"
X_train, X_test, y_train, y_test = train_test_split(X_dw, y, test_size=0.2) # train/test split

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(f1_score(y_test, y_pred, average='micro'))
print(confusion_matrix(y_test, y_pred, normalize='true'))
```

```python id="OaaACyUGx1mH"
sns.heatmap(confusion_matrix(y_test, y_pred, normalize='true'), annot=True)
```

```python id="fvGcW5h3yutU" outputId="aa5ab773-fc97-4bda-bd9f-fb56abd659f9"
dw_micro_f1_scores = []
dw_macro_f1_scores = []
for train_size in tqdm(np.arange(0.05, 1, 0.05)):
    X_train, X_test, y_train, y_test = train_test_split(X_dw, y, 
                                                        train_size=train_size,
                                                        random_state=42)

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    mic = f1_score(y_test, y_pred, average='micro')
    mac = f1_score(y_test, y_pred, average='macro')
    dw_micro_f1_scores.append(mic)
    dw_macro_f1_scores.append(mac)
```

```python id="hFbsd_KByxJU"
sns.lineplot(x = np.arange(0.1, 2, 0.1), y = dw_micro_f1_scores, label='DeepWalk')
plt.xlabel('Labelled Proportion')
plt.plot()
```

```python id="CL4Fuiy7zAcI"
u = umap.UMAP(random_state=42)
dw_umap_embs = u.fit_transform(X_dw)

ax = sns.scatterplot(x = dw_umap_embs[:, 0], y = dw_umap_embs[:, 1], hue = y)
```

<!-- #region id="JjRlXSyYKfKF" -->
### Custom wiki
<!-- #endregion -->

<!-- #region id="5ODhmlRHKZ_M" -->
You can get the dataset from https://densitydesign.github.io/strumentalia-seealsology/

__Steps to download:__

a) Enter some wiki links.

b) Download the TSV file.
<!-- #endregion -->

<!-- #region id="pfwSKFtQqfGW" -->
I used the following setting:
<!-- #endregion -->

<!-- #region id="L1Tn0wNGqZkL" -->
<img src='https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Fd345633f-d320-4e61-8287-1e452c7ee319%2FUntitled.png?table=block&id=e3afab42-7f22-4eb2-a6b7-2b2e753dbd4a&spaceId=63b72b1f-0e90-4ab8-a6df-a060a6545a56&width=2000&userId=21ec183f-f0be-4b6b-9b3e-6f0d4e5c5469&cache=v2'>
<!-- #endregion -->

```python id="S2gNHcfeKZ_O"
df = pd.read_csv("wikidata.tsv", sep = "\t")
```

```python id="jrSvOEjIKZ_O" outputId="716860d7-4777-4687-8f13-0daef205e923"
df.head()
```

```python id="GXq0R65NKZ_R" outputId="5d6ba70e-a5f6-4b60-d09d-679e6a41c9dd"
df.shape
```

```python id="KME5E0uxKZ_S"
# construct an undirected graph
G=nx.from_pandas_edgelist(df, "source", "target", edge_attr=True, create_using=nx.Graph())
```

```python id="8lYrqbUsKZ_S" outputId="c9f22334-2018-48f5-dc90-3cdb44fbc8fc"
len(G) # number of nodes
```

```python id="ei79NkmmKZ_T"
# function to generate random walk sequences of nodes
def get_randomwalk(node, path_length):
    
    random_walk = [node]
    
    for i in range(path_length-1):
        temp = list(G.neighbors(node))
        temp = list(set(temp) - set(random_walk))    
        if len(temp) == 0:
            break

        random_node = random.choice(temp)
        random_walk.append(random_node)
        node = random_node
        
    return random_walk
```

```python id="Try4OG5CKZ_U" outputId="0f2fed37-1471-4012-ab9d-316cbd5965d3"
get_randomwalk('amitabh bachchan', 10)
```

```python id="k214l2bdKZ_V"
all_nodes = list(G.nodes())

random_walks = []

for n in tqdm(all_nodes):
    for i in range(5):
        random_walks.append(get_randomwalk(n,10))
```

```python id="zG8ThINzKZ_W" outputId="a7cfc88a-4dce-4104-eda8-77696d8144cf"
# count of sequences
len(random_walks)
```

```python id="fqPqVISuKZ_Z"
# train word2vec model
model = Word2Vec(window = 4, sg = 1, hs = 0,
                 negative = 10, # for negative sampling
                 alpha=0.03, min_alpha=0.0007,
                 seed = 14)

model.build_vocab(random_walks, progress_per=2)
```

```python id="F7g75PMoKZ_a" outputId="c913d369-730b-46c6-a017-57bc1dd8c6d5"
model.train(random_walks, total_examples = model.corpus_count, epochs=20, report_delay=1)
```

```python id="jL23JQ-gKZ_c" outputId="e38d074d-2d0d-4a25-c125-ab787ef2266e"
print(model)
```

```python id="SH0JrspKN8as" outputId="0cd01d91-953e-4368-e869-1fc1ca2e50a9"
word_list = list(np.random.choice(df.source.unique(),10))
word_list
```

```python id="iAIfcnsRKZ_f"
def plot_nodes(word_list):
    # X = model[word_list]
    X = []
    for w in word_list:
      X.append(model.wv.get_vector(w))
    
    # reduce dimensions to 2
    pca = PCA(n_components=2)
    result = pca.fit_transform(np.array(X))
    
    
    plt.figure(figsize=(12,9))
    # create a scatter plot of the projection
    plt.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(word_list):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))
        
    plt.show()
```

```python id="ab9qCW4EKZ_f" outputId="1e6a28cf-2882-4617-8f7f-c5e7a19685b6"
plot_nodes(word_list)
```

<!-- #region id="C70jHB7wUFD5" -->
### Skills
<!-- #endregion -->

```python id="M28GnWuJUGfk"
!wget -O skills.xlsx https://github.com/jdmoore7/ONET_analysis/blob/master/Technology%20Skills.xlsx?raw=true
```

```python id="Yn9lVPP7UWiW" outputId="cddcbdf2-56b2-49cf-f6db-4e9ce6942b5a"
skill_data = pd.read_excel('skills.xlsx')
skill_data.head()
```

<!-- #region id="uEJUbxFGV_HV" -->
**Matrix factorization**
<!-- #endregion -->

```python id="mACGuB4DUpeB"
x = pd.get_dummies(skill_data.set_index('Title')['Example'])

x = x.groupby(lambda var:var, axis=0).sum()

cols = x.columns.to_list()
rows = x.transpose().columns.to_list()

y = x.to_numpy()

job_skill_tensor = torch.FloatTensor(y)
```

```python id="3_j_5mHfUvA_" outputId="dcbcfaaa-8443-462b-cbbd-f00c6d2892f9"
class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_jobs=len(rows), n_skills=len(cols), n_factors=10):
        super().__init__()
        
        self.job_latent = nn.Parameter(torch.rand(n_jobs,n_factors))
        self.skill_latent = nn.Parameter(torch.rand(n_factors, n_skills))
        
        
    def forward(self):
        return torch.mm(self.job_latent,self.skill_latent)


model = MatrixFactorization()
loss_fn = nn.MSELoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

losses = []
epochs = 1000
for epoch in range(epochs):
    loss = 0
    
    prediction = model.forward()
    loss += loss_fn(prediction, job_skill_tensor)
    losses.append(loss)

    # Reset the gradients to 0
    optimizer.zero_grad()

    # backpropagate
    loss.backward()

    # update weights
    optimizer.step()
    if epoch % 50 == 0:
        print(loss)
```

```python id="aU0VsD-aU776" outputId="5b12b24a-1aba-4232-e5ad-16bcb87feebc"
job_features = np.array(model.job_latent.detach())
skill_features = np.array(model.skill_latent.detach())
job_skill_stacked = np.concatenate((job_features,skill_features.transpose()))
job_skill_sim = cosine_similarity(job_skill_stacked)

entities = []
entities.extend(rows + cols)

def get_similar(node,sim_threshold=None,count_threshold=None,category=None):
  idx = entities.index(node)
  sim_scores = job_skill_sim[idx]
  retrieved = [(elem,score) for elem,score in zip(entities,sim_scores)]

  if category == 'jobs':
    retrieved = [tup for idx,tup in enumerate(retrieved) if idx < len(rows)]
  elif category == 'skills':
    retrieved = [tup for idx,tup in enumerate(retrieved) if idx > len(rows)]
  else:
    pass
  
  
  if sim_threshold:
    retrieved = [(elem,score) for elem,score in retrieved if score > sim_threshold]
  
  retrieved = sorted(retrieved,key=operator.itemgetter(1),reverse=True)

  if count_threshold:
    retrieved = [tup for idx,tup in enumerate(retrieved) if idx < count_threshold]  
  
  return retrieved

get_similar('Python',category='jobs',sim_threshold=0.8,count_threshold=25)
```

```python id="RMz_-wq1V2Kb"
# Save latent feature similarity values in a pickled file!

import pickle
with open('cos_sim_pickle.pkl', 'wb') as f:
  pickle.dump(job_skill_sim, f)

with open('model.pkl', 'wb') as f:
  pickle.dump(model, f)  

with open('latent_features.pkl', 'wb') as f:
  pickle.dump(job_skill_stacked,f)   
```

```python id="5hauFi8OUK1O"
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
p_comps = pca.fit_transform(job_skill_stacked)

from matplotlib.pyplot import xlim,ylim

plt.scatter(
    x=p_comps[:,0],y=p_comps[:,1],color=['r' if idx < len(rows) else 'b' for idx in range(job_skill_stacked.shape[0])],
    marker='+',
    alpha = 0.25,
)
```

<!-- #region id="o8uOH1_gWFS3" -->
**Deep walk**
<!-- #endregion -->

```python id="pNb-3WoSWLYc"
edges = skill_data[['Title','Example']].values.tolist()
net = nx.from_edgelist(edges)
```

```python id="bC0jnGkFWNkO" outputId="f4c97528-2d8c-4c4e-c720-e0a93ec86339"
def random_walk(graph,seed,rounds=20):
  import random
  movements = [seed]
  for round in range(rounds):
    node_choices = [node for node in graph.neighbors(seed)]
    seed = random.choice(node_choices)
    movements.append(seed)
  return movements

random_walk(net,'Python')
```

```python id="bmtMCxmeWNbR"
walks = []
vertices = [n for n in net.nodes]
for v in vertices:
  walks.append(random_walk(graph=net,seed=v))
```

```python id="4SzNTwZyWUk5" outputId="93897245-6776-43d5-a44c-ba45d3aa0280"
embeddings = Word2Vec(walks,vector_size=10,window=5)
embeddings.save("graph2vec2.model")
embeddings.wv.most_similar('C++') ## verify results are sensible
```

```python id="TNknRCCyXefn"
net.nodes.items()
```

```python id="HcwVSdcyYDEy" outputId="905dc3f6-c6d5-4385-f1f6-0fcd9b6e6d29"
    # X = []
    # for w in word_list:
    #   X.append()

embeddings.wv.get_vector('AdSense Tracker')
```

```python id="5sTuBlXAcL7M"
from collections import defaultdict

array_dict = defaultdict()

for node in net.nodes:
  try:
    array_dict[node] = embeddings.wv.get_vector(node)
  except:
    pass
```

```python id="g4EA3iDpXYXO"
embedded_nodes = [node for node in net.nodes if node in array_dict]
arrays = np.array([array_dict[node] for node in embedded_nodes])

skills = [skill for skill in skill_data['Example'].unique()]
jobs = [job for job in skill_data['Title'].unique()]
skill_idx = [idx for idx,elem in enumerate(embedded_nodes) if elem in skills]
job_idx = [idx for idx,elem in enumerate(embedded_nodes) if elem in jobs]
```

```python id="A56Pspk8V4Tc"
pca = PCA(n_components=2)
p_comps = pca.fit_transform(arrays)

from matplotlib.pyplot import xlim,ylim

# xlim(-13,13)
# ylim(-13,13)
plt.scatter(
    # Jobs are red, skills are blue
    x=p_comps[:,0],y=p_comps[:,1],color=['b' if idx in skill_idx else 'r' for idx in range(len(arrays))],
    marker='+',
    alpha = 0.35,
    )
```

<!-- #region id="lpPgQna1Fzbb" -->
## Node2vec
<!-- #endregion -->

<!-- #region id="BGTR8IYWrQhf" -->
### Basics
<!-- #endregion -->

<!-- #region id="SuyOacGmsH5W" -->
Node2Vec is very similar to DeepWalk, but the random walks are generated a bit differently. Recall that in the pure random walk, neighbourhood nodes have an equal propability to be chosen as next step. Here instead, we have 2 hyperparameters to tune - `p` and `q`. `p` and `q` control how fast the walk explores and leaves the neighborhood of starting node u.

* p - high values means that we're less likely to return to the previous node
* q - high values approximate the Breadth-First-Search meaning that the neighbourhood around the node is explored. Low values give higher chance to go outside the neighbourhood and hence approxiamtes the Depth-First-Search 

Let's compare 2 extreme scenarios:
1. p = 10, q = 0.1 - here we expect the random walk to go outwards and explore the adjacent clusters as well 
2. p = 0.1, q = 10 - here we expect the random walk to stay very local and explore the neighbourhood around the starting node

Here's the code block from `karate-club` package that does the Biased Random Walk. I'm showing it here so that you have a better understanding of what's happening under the hood.
<!-- #endregion -->

```python id="zT48X_nMqels"
def biased_walk(start_node, walk_length, p, q):
    walk = [start_node]
    previous_node = None
    previous_node_neighbors = []
    for _ in range(walk_length-1):
        current_node = walk[-1]  # currnet node ID
        current_node_neighbors = np.array(list(graph.neighbors(current_node)))  # negihbours of this node
        probability = np.array([1/q] * len(current_node_neighbors), dtype=float)  # outwards probability weight determined by q
        probability[current_node_neighbors==previous_node] = 1/p  # probability of return determined by p
        probability[(np.isin(current_node_neighbors, previous_node_neighbors))] = 1  # weight of 1 to all the neighbours which are connected to the previous node as well
        norm_probability = probability/sum(probability)  # normalize the probablity
        selected = np.random.choice(current_node_neighbors, 1, p=norm_probability)[0]  # select the node from neighbours according to the probabilities from above
        walk.append(selected)  # append to the walk and continue
        previous_node_neighbors = current_node_neighbors
        previous_node = current_node
    
    return walk
```

```python id="oqc9cgInqeil"
p = 10
q = 0.1
walk = biased_walk(6, 80, p, q)
# Visualise the subgraph
subgraph_nodes = list(nx.dfs_preorder_nodes(graph, 7))
G = graph.subgraph(walk)
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx(G, pos=pos, cmap='coolwarm')
```

```python id="gdZHzuRjqefD"
p = 0.1
q = 10
walk = biased_walk(6, 80, p, q)
# Visualise the subgraph
subgraph_nodes = list(nx.dfs_preorder_nodes(graph, 7)) 
G = graph.subgraph(walk)
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx(G, pos=pos, cmap='coolwarm')
```

<!-- #region id="Gyxcoa3iswa_" -->
<img src='https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Fafc3df5f-7d19-4710-92a0-67c5da83e51f%2FUntitled.png?table=block&id=74d4a809-249b-443c-acb8-2cc9af44d273&spaceId=63b72b1f-0e90-4ab8-a6df-a060a6545a56&width=2000&userId=21ec183f-f0be-4b6b-9b3e-6f0d4e5c5469&cache=v2'>
<!-- #endregion -->

<!-- #region id="qk1ChfYBsMEd" -->
From the images we can see the differences between the resulting random walks. Each problem will have its own perfect `p` and `q` parameters so we can treat them as hyperparameters to tune. For now, let's just set the parameters to `p=0.5` and `q=0.25` but feel free to experiment with other parameters as well. Also, we're going to use the `karate-club` implementation of `BiasedRandomWalker` for the simplicity sake. Pleasd note that biased sampling takes longer to calculate, so grid searching the optimal hyperparameters is a long procedure.
<!-- #endregion -->

<!-- #region id="qpcJ2Tko8phT" -->
### Node2vec Wiki
<!-- #endregion -->

```python id="EMiRQkP4618e" outputId="60f08dda-13d3-4caa-ce1f-a898c380636e"
G=nx.read_edgelist('../data/wiki/Wiki_edgelist.txt',
                      create_using = nx.DiGraph(), nodetype = None, data = [('weight', int)])
model = Node2Vec(G, walk_length=10, num_walks=80,
                  p=0.25, q=4, workers=1, use_rejection_sampling=0)
model.train(window_size = 5, iter = 3)
embeddings=model.get_embeddings()
```

```python id="dFBgwMfn_Gvg" outputId="198b407c-e3e1-4e65-825f-6b2bd57a6692"
evaluate_embeddings(embeddings)
```

```python id="5VgNMb9q_yrb" outputId="8870f07e-b376-407b-fd3a-4565375aea93"
plot_embeddings(embeddings)
```

<!-- #region id="cEeWKtAuqdDQ" -->
### Facebook
<!-- #endregion -->

```python id="_JYbNzEOst3u"
b_walker = BiasedRandomWalker(20, 10, 0.5, 0.25)
b_walker.do_walks(graph)
```

```python id="-Pr4N0T5stwt"
node_vec = Word2Vec(b_walker.walks,  # previously generated walks
                 hs=1,  # tells the model to use hierarchical softmax
                 sg = 1,  # tells the model to use skip-gram
                 vector_size=128,  # size of the embedding
                 window=10,
                 min_count=1,
                 workers=4,
                 seed=42)
```

```python id="KXFgygyqstsY"
# Get corresponding Node2Vec embeddings
X_node_vec = []
for i in y.index:
    X_node_vec.append(node_vec.wv.__getitem__(str(i)))
```

```python id="_u8EGpZfyM1l" outputId="30bfd8d3-99e9-42df-e549-7c7ea4cb06b1"
X_train, X_test, y_train, y_test = train_test_split(X_node_vec, y, test_size=0.2) # train/test split

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(f1_score(y_test, y_pred, average='micro'))
print(confusion_matrix(y_test, y_pred, normalize='true'))
```

```python id="xjCXxp2EyRm5"
sns.heatmap(confusion_matrix(y_test, y_pred, normalize='true'), annot=True)
```

```python id="H6INSqrly1gU" outputId="14bcf99c-154f-43f4-a160-a96765665d53"
nv_micro_f1_scores = []
nv_macro_f1_scores = []
for train_size in tqdm(np.arange(0.05, 1, 0.05)):
    X_train, X_test, y_train, y_test = train_test_split(X_node_vec, y, 
                                                        train_size=train_size,
                                                        random_state=42)

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    mic = f1_score(y_test, y_pred, average='micro')
    mac = f1_score(y_test, y_pred, average='macro')
    nv_micro_f1_scores.append(mic)
    nv_macro_f1_scores.append(mac)
```

```python id="JDD5Fp7Ey23D"
sns.lineplot(x = np.arange(0.1, 2, 0.1), y = nv_micro_f1_scores, label='Node2Vec')
plt.xlabel('Labelled Proportion')
plt.plot()
```

```python id="LrERXw6yzDWy"
u = umap.UMAP(random_state=42)
nv_umap_embs = u.fit_transform(X_node_vec)

ax = sns.scatterplot(x = nv_umap_embs[:, 0], y = nv_umap_embs[:, 1], hue = y)
```

<!-- #region id="aCfJjppRzFta" -->
As can be seen from the embeddings, the company, government, and tvshows are represented by clear clusters whereas politician clusters is kind of scattered around. Plus, there are pages which are not clustered meaning that they are probably much harder to classify.
<!-- #endregion -->

<!-- #region id="RByRio0oB_K-" -->
## LINE Wiki
<!-- #endregion -->

```python id="SxK7SbDrAt1u" outputId="d711c89b-1238-4f41-ac78-d73b78eae99e"
G = nx.read_edgelist('../data/wiki/Wiki_edgelist.txt',
                      create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])

model = LINE(G, embedding_size=128, order='second')
model.train(batch_size=1024, epochs=50, verbose=2)
embeddings = model.get_embeddings()
```

```python id="pL2CEk0ZA-nh" outputId="262c45b0-bdc0-43fb-993f-585c1fe7747a"
evaluate_embeddings(embeddings)
```

```python id="sVDmFcSiB0Jk" outputId="9698bb11-d9f9-4a5e-cec4-d78a60eb48d6"
plot_embeddings(embeddings)
```
