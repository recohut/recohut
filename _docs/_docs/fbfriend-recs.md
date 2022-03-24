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

# Facebook Friend Recommender

<!-- #region id="jDec9it1V4Hn" -->
First we will load our dataset from Kaggle and perform exploratory data analysis on our given data set such as number of followers and followees of each person. Then we will generate some datapoints which were not present in our given data-set, since we have only class label 1 data. Then we will do some feature engineering on dataset like finding shortest path, kartz centrality, jaccard distances, page rank, preferential attachements etc. After performing exploratory data analysis and feature engineering, we will split whole dataset into train and test and perform random forest and xgboost taking f1-score as our metric. At the end we will plot confusion matrix and pretty-table for both algorithm and finf best hyperparameters.
<!-- #endregion -->

<!-- #region id="F3UNi-NOWjb5" -->
## Setup
<!-- #endregion -->

```python id="j2QozaSUWg5b"
import math
import random
import pickle
import os
import csv
import pandas as pd
import datetime
import time
import numpy as np

import matplotlib
import matplotlib.pylab as plt
import seaborn as sns
from matplotlib import rcParams
from sklearn.cluster import MiniBatchKMeans, KMeans
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split

import xgboost as xgb
import networkx as nx
import pdb
import pickle

import warnings
warnings.filterwarnings("ignore")
```

<!-- #region id="dfMsBJhGW6ld" -->
## Load dataset
<!-- #endregion -->

<!-- #region id="6ZCD5IvKXEwH" -->
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 12911, "status": "ok", "timestamp": 1627616227494, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="vWLoLTTQXIG0" outputId="2803f5b6-ec3d-4b8d-a36c-0c998c3bd705"
!pip install -q -U kaggle
!pip install --upgrade --force-reinstall --no-deps kaggle
!mkdir ~/.kaggle
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c FacebookRecruiting
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2716, "status": "ok", "timestamp": 1627616230204, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="fxu9z1aJXJ5Y" outputId="7b430be0-9ead-46cc-b510-19ff71d30fdd"
!unzip FacebookRecruiting.zip
```

<!-- #region id="9ZMu-p1OXmCA" -->
## Reading graph
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"elapsed": 1312, "status": "ok", "timestamp": 1627616502547, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="R1hSHYa2gDQd" outputId="f157932a-13b2-4db7-d977-781a33fe11a0"
traincsv = pd.read_csv('train.csv')
traincsv.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 297} executionInfo={"elapsed": 515, "status": "ok", "timestamp": 1627616608277, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="CfJy1nzZgjuv" outputId="e7505a40-018d-4186-daa5-ca434d03dc33"
traincsv.describe()
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3437, "status": "ok", "timestamp": 1627616633907, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="t5lLT6BQX198" outputId="22f3ac98-ce67-4acb-b396-8a8514f0defd"
print(traincsv[traincsv.isna().any(1)])
print(traincsv.info())
print("Number of diplicate entries: ",sum(traincsv.duplicated()))
traincsv.to_csv('train_woheader.csv',header=False,index=False)
print("saved the graph into file")
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 63716, "status": "ok", "timestamp": 1627616500002, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="cBvsnoX8XgRY" outputId="45c0cb7b-17ac-4adb-ec09-15dd0ed6da7b"
g = nx.read_edgelist('train_woheader.csv',delimiter=',',create_using=nx.DiGraph(),nodetype=int)
print(nx.info(g))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 421} executionInfo={"elapsed": 1927, "status": "ok", "timestamp": 1627616653158, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="ipmUsmU_YJ3c" outputId="ecff532a-c664-480e-ba95-52b6e30bec6c"
traincsv.head(20).to_csv('train_woheader_sample.csv',header=False,index=False)
    
subgraph=nx.read_edgelist('train_woheader_sample.csv',delimiter=',',create_using=nx.DiGraph(),nodetype=int)
# https://stackoverflow.com/questions/9402255/drawing-a-huge-graph-with-networkx-and-matplotlib

pos=nx.spring_layout(subgraph)
nx.draw(subgraph,pos,node_color='#A0CBE2',edge_color='#00bb5e',width=1,edge_cmap=plt.cm.Blues,with_labels=True)
plt.savefig("graph_sample.pdf")
print(nx.info(subgraph))
```

<!-- #region id="K8N2vC3MZfSS" -->
## Exploratory data analysis
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1166, "status": "ok", "timestamp": 1627463762173, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="QirVa7A4Y5Qu" outputId="322e4f1b-0dd3-4b87-c4f7-3efbd523000c"
# No of Unique persons 
print("The number of unique persons",len(g.nodes()))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 388} executionInfo={"elapsed": 2336, "status": "ok", "timestamp": 1627463789524, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="M77RkwjMZmVr" outputId="718ce697-383b-460d-b0ad-12faae326b79"
# No of followers of each person
indegree_dist = list(dict(g.in_degree()).values())
indegree_dist.sort()
plt.figure(figsize=(10,6))
plt.plot(indegree_dist)
plt.xlabel('Index No')
plt.ylabel('No Of Followers')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 388} executionInfo={"elapsed": 3253, "status": "ok", "timestamp": 1627463899564, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="O_ohVDNlZp9K" outputId="016d4a29-8df9-4254-8e0b-f8fa5c887f4d"
indegree_dist = list(dict(g.in_degree()).values())
indegree_dist.sort()
plt.figure(figsize=(10,6))
plt.plot(indegree_dist[0:1500000])
plt.xlabel('Index No')
plt.ylabel('No Of Followers')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 388} executionInfo={"elapsed": 2561, "status": "ok", "timestamp": 1627463912779, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="eg7CBHaoZ_UL" outputId="8cb1deb6-12e2-4e04-a708-01a3587b1fbb"
# No Of people each person is following
outdegree_dist = list(dict(g.out_degree()).values())
outdegree_dist.sort()
plt.figure(figsize=(10,6))
plt.plot(outdegree_dist)
plt.xlabel('Index No')
plt.ylabel('No Of people each person is following')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 9148, "status": "ok", "timestamp": 1627464104647, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="lwHxDmr3aIB7" outputId="c2ffe1b0-d3ed-404a-9881-2170c90eeac3"
print('No of persons who are not following anyone are {} ({:.2%})'.format(sum(np.array(outdegree_dist)==0),
                                                                        sum(np.array(outdegree_dist)==0)/len(outdegree_dist)))
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 9004, "status": "ok", "timestamp": 1627464172658, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="UDh4Y0rnaabu" outputId="95c8c049-6677-4202-9a42-a0e6f4f9a335"
print('No of persons having zero followers are {} ({:.2%})'.format(sum(np.array(indegree_dist)==0),
                                                                        sum(np.array(indegree_dist)==0)/len(indegree_dist)))
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3174, "status": "ok", "timestamp": 1627464207983, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="GWieGOF8bF2s" outputId="48679a10-7704-414d-c3e6-94a92a15aa50"
count=0
for i in g.nodes():
    if len(list(g.predecessors(i)))==0 :
        if len(list(g.successors(i)))==0:
            count+=1
print('No of persons those are not following anyone and also not having any followers are',count)
```

<!-- #region id="ieCAEYAVcXNM" -->
## Negative sampling
<!-- #endregion -->

<!-- #region id="m2pnbqRnhWkA" -->
Generating some edges which are not present in graph for supervised learning. In other words, we are generating bad links from graph which are not in graph and whose shortest path is greater than 2.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 66, "referenced_widgets": ["65b7914b52f04174ae774ba54a9d1599", "2ff239b666964f0284572fd40ef24795", "3d7ffa4d7cd44e22b8ab5bfacb9229fe", "4916c564f3d5428d8858363e51f4d696", "74eeaf34ffd945ef9f424037ef32f1af", "a3dc4e4778f24197ab782d84ded7d189", "dc70e7f348cb4e7ab107f6f4d79bb817", "3ec2255273d844a18a0ebd5c6e7d9368"]} executionInfo={"elapsed": 78359, "status": "ok", "timestamp": 1627617426296, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="U7nXVgfchj-9" outputId="456d9c5d-0783-4543-89af-f83983c88d19"
r = csv.reader(open('train_woheader.csv','r'))
edges = dict()
for edge in r:
    edges[(edge[0], edge[1])] = 1
missing_edges = set([])

with tqdm(total=9437519) as pbar:
    while (len(missing_edges)<9437519):
        a=random.randint(1, 1862220)    
        b=random.randint(1, 1862220)
        tmp = edges.get((a,b),-1)
        if tmp == -1 and a!=b:
            try:
                if nx.shortest_path_length(g,source=a,target=b) > 2: 
                    missing_edges.add((a,b))
                else:
                    continue  
            except:  
                    missing_edges.add((a,b))              
        else:
            continue
        pbar.update(1)
pickle.dump(missing_edges,open('missing_edges_final.p','wb'))
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 479, "status": "ok", "timestamp": 1627617581690, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="RANp2S96jzVJ" outputId="b6a1bf31-53d2-4170-cbac-c96a6956ddec"
list(missing_edges)[:10]
```

<!-- #region id="vcIK4mX1bP8A" -->
## Train/test split
<!-- #endregion -->

<!-- #region id="Xm8UdKXzlpUx" -->
> Tip: We will split positive links and negative links seperatly because we need only positive training data for creating graph and for feature generation.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 45445, "status": "ok", "timestamp": 1627618042831, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="JXaSZnZLkX65" outputId="6f37fdb2-e7ca-4156-b257-2471432fa0fe"
#reading total data df
df_pos = pd.read_csv('train.csv')
df_neg = pd.DataFrame(list(missing_edges), columns=['source_node', 'destination_node'])

print("Number of nodes in the graph with edges", df_pos.shape[0])
print("Number of nodes in the graph without edges", df_neg.shape[0])

#Trian test split 
#Spiltted data into 80-20
X_train_pos, X_test_pos, y_train_pos, y_test_pos  = train_test_split(df_pos,np.ones(len(df_pos)),test_size=0.2, random_state=9)
X_train_neg, X_test_neg, y_train_neg, y_test_neg  = train_test_split(df_neg,np.zeros(len(df_neg)),test_size=0.2, random_state=9)

print('='*60)
print("Number of nodes in the train data graph with edges", X_train_pos.shape[0],"=",y_train_pos.shape[0])
print("Number of nodes in the train data graph without edges", X_train_neg.shape[0],"=", y_train_neg.shape[0])
print('='*60)
print("Number of nodes in the test data graph with edges", X_test_pos.shape[0],"=",y_test_pos.shape[0])
print("Number of nodes in the test data graph without edges", X_test_neg.shape[0],"=",y_test_neg.shape[0])

#removing header and saving
X_train_pos.to_csv('train_pos_after_eda.csv',header=False, index=False)
X_test_pos.to_csv('test_pos_after_eda.csv',header=False, index=False)
X_train_neg.to_csv('train_neg_after_eda.csv',header=False, index=False)
X_test_neg.to_csv('test_neg_after_eda.csv',header=False, index=False)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 108628, "status": "ok", "timestamp": 1627618327578, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="2cltGHkzl7Uf" outputId="f2633d6b-f40a-4bb5-c484-f83d70c6a98c"
train_graph=nx.read_edgelist('train_pos_after_eda.csv',delimiter=',',create_using=nx.DiGraph(),nodetype=int)
test_graph=nx.read_edgelist('test_pos_after_eda.csv',delimiter=',',create_using=nx.DiGraph(),nodetype=int)
print(nx.info(train_graph))
print(nx.info(test_graph))

# finding the unique nodes in both train and test graphs
train_nodes_pos = set(train_graph.nodes())
test_nodes_pos = set(test_graph.nodes())

trY_teY = len(train_nodes_pos.intersection(test_nodes_pos))
trY_teN = len(train_nodes_pos - test_nodes_pos)
teY_trN = len(test_nodes_pos - train_nodes_pos)

print('no of people common in train and test -- ',trY_teY)
print('no of people present in train but not present in test -- ',trY_teN)
print('no of people present in test but not present in train -- ',teY_trN)
print(' % of people not there in Train but exist in Test in total Test data are {} %'.format(teY_trN/len(test_nodes_pos)*100))
```

```python id="Vtp_ZULhnJy6"
X_train_pos = pd.read_csv('train_pos_after_eda.csv', names=['source_node', 'destination_node'])
X_test_pos = pd.read_csv('test_pos_after_eda.csv', names=['source_node', 'destination_node'])
X_train_neg = pd.read_csv('train_neg_after_eda.csv', names=['source_node', 'destination_node'])
X_test_neg = pd.read_csv('test_neg_after_eda.csv', names=['source_node', 'destination_node'])

print('='*60)
print("Number of nodes in the train data graph with edges", X_train_pos.shape[0])
print("Number of nodes in the train data graph without edges", X_train_neg.shape[0])
print('='*60)
print("Number of nodes in the test data graph with edges", X_test_pos.shape[0])
print("Number of nodes in the test data graph without edges", X_test_neg.shape[0])

X_train = X_train_pos.append(X_train_neg,ignore_index=True)
y_train = np.concatenate((y_train_pos,y_train_neg))
X_test = X_test_pos.append(X_test_neg,ignore_index=True)
y_test = np.concatenate((y_test_pos,y_test_neg)) 

X_train.to_csv('train_after_eda.csv',header=False,index=False)
X_test.to_csv('test_after_eda.csv',header=False,index=False)
pd.DataFrame(y_train.astype(int)).to_csv('train_y.csv',header=False,index=False)
pd.DataFrame(y_test.astype(int)).to_csv('test_y.csv',header=False,index=False)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 496, "status": "ok", "timestamp": 1627618515142, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="6Jfsu_BMnlx5" outputId="a178a4db-8fe9-405f-e898-19ff21d713d6"
print("Data points in train data",X_train.shape)
print("Data points in test data",X_test.shape)
print("Shape of traget variable in train",y_train.shape)
print("Shape of traget variable in test", y_test.shape)
```

<!-- #region id="89vYHL5jc708" -->
## Feature engineering
<!-- #endregion -->

<!-- #region id="PiGpdEuwcfAA" -->
### Similarity measures
<!-- #endregion -->

<!-- #region id="bWYSZUYBcmk0" -->
#### Jaccard distance
<!-- #endregion -->

<!-- #region id="OpzpbwbsoYR1" -->
\begin{equation}
j = \frac{|X\cap Y|}{|X \cup Y|} 
\end{equation}
<!-- #endregion -->

```python id="LXpgbkwGoZdG"
def jaccard_for_followees(a,b):
    try:
        if len(set(train_graph.successors(a))) == 0  | len(set(train_graph.successors(b))) == 0:
            return 0
        sim = (len(set(train_graph.successors(a)).intersection(set(train_graph.successors(b)))))/\
                                    (len(set(train_graph.successors(a)).union(set(train_graph.successors(b)))))
    except:
        return 0
    return sim
```

```python id="6x1xsEP0o48W"
def jaccard_for_followers(a,b):
    try:
        if len(set(train_graph.predecessors(a))) == 0  | len(set(g.predecessors(b))) == 0:
            return 0
        sim = (len(set(train_graph.predecessors(a)).intersection(set(train_graph.predecessors(b)))))/\
                                 (len(set(train_graph.predecessors(a)).union(set(train_graph.predecessors(b)))))
        return sim
    except:
        return 0
```

<!-- #region id="Sfdkxae-coU2" -->
#### Cosine distance
<!-- #endregion -->

<!-- #region id="YKbzTnrto-Pv" -->
\begin{equation}
CosineDistance = \frac{|X\cap Y|}{|X|\cdot|Y|} 
\end{equation}
<!-- #endregion -->

```python id="5RN3c0SKo_gn"
def cosine_for_followees(a,b):
    try:
        if len(set(train_graph.successors(a))) == 0  | len(set(train_graph.successors(b))) == 0:
            return 0
        sim = (len(set(train_graph.successors(a)).intersection(set(train_graph.successors(b)))))/\
                                    (math.sqrt(len(set(train_graph.successors(a)))*len((set(train_graph.successors(b))))))
        return sim
    except:
        return 0
```

```python id="-uN89BjupCPI"
def cosine_for_followers(a,b):
    try:
        
        if len(set(train_graph.predecessors(a))) == 0  | len(set(train_graph.predecessors(b))) == 0:
            return 0
        sim = (len(set(train_graph.predecessors(a)).intersection(set(train_graph.predecessors(b)))))/\
                                     (math.sqrt(len(set(train_graph.predecessors(a))))*(len(set(train_graph.predecessors(b)))))
        return sim
    except:
        return 0
```

<!-- #region id="ciMlEw55cqqR" -->
### Ranking measures
<!-- #endregion -->

<!-- #region id="NOzL-Bxcc-sX" -->
#### Pagerank
<!-- #endregion -->

```python id="lKGZ-7W-phGv"
train_graph=nx.read_edgelist('train_pos_after_eda.csv',delimiter=',',create_using=nx.DiGraph(),nodetype=int)
pr = nx.pagerank(train_graph, alpha=0.85)
pickle.dump(pr,open('page_rank.p','wb'))
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1237, "status": "ok", "timestamp": 1627620165703, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="ZCUWzRCeqm3V" outputId="277d91f4-9a8a-4824-9967-a6a875d9d56d"
print('min',pr[min(pr, key=pr.get)])
print('max',pr[max(pr, key=pr.get)])
#for imputing to nodes which are not there in Train data
print('mean_pr',float(sum(pr.values())) / len(pr))
```

<!-- #region id="XedCHTmBdBcc" -->
### Other graph features
<!-- #endregion -->

<!-- #region id="Jp2H0uA3dHv_" -->
#### Shortest path
<!-- #endregion -->

<!-- #region id="o5-0iQX6q4P8" -->
Getting Shortest path between two nodes, and if any 2 given nodes have a direct path i.e directly connected then we are removing that edge and calculating path.
<!-- #endregion -->

```python id="y3iWLy5Bqu2W"
def compute_shortest_path_length(a,b):
    p=-1
    try:
        if train_graph.has_edge(a,b):
            train_graph.remove_edge(a,b)
            p= nx.shortest_path_length(train_graph,source=a,target=b)
            train_graph.add_edge(a,b)
        else:
            p= nx.shortest_path_length(train_graph,source=a,target=b)
        return p
    except:
        return -1
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1627619545396, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="BzslYnz-rKW7" outputId="fd6dac23-5858-481d-e886-668ed50e37b5"
# unit test 1
compute_shortest_path_length(77697, 826021)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1627619546006, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="9dgfHKwGrO27" outputId="d96fadfb-65d9-4e70-b4c2-03525c0d9864"
# unit test 2
compute_shortest_path_length(669354, 1635354)
```

<!-- #region id="iv27JsDLdMQb" -->
#### Same community
<!-- #endregion -->

```python id="_yU7WeJFsFyO"
wcc = list(nx.weakly_connected_components(train_graph))
```

```python id="UlYshkkxsDPr"
def belongs_to_same_wcc(a,b):
    index = []
    if train_graph.has_edge(b,a):
        return 1
    if train_graph.has_edge(a,b):
            for i in wcc:
                if a in i:
                    index= i
                    break
            if (b in index):
                train_graph.remove_edge(a,b)
                if compute_shortest_path_length(a,b)==-1:
                    train_graph.add_edge(a,b)
                    return 0
                else:
                    train_graph.add_edge(a,b)
                    return 1
            else:
                return 0
    else:
            for i in wcc:
                if a in i:
                    index= i
                    break
            if(b in index):
                return 1
            else:
                return 0
```

<!-- #region id="sciSYrVZdNgy" -->
#### Admaic/Adar index
<!-- #endregion -->

<!-- #region id="m_rArhaKsNPd" -->
Adamic/Adar measures is defined as inverted sum of degrees of common neighbours for given two vertices: $A(x,y)=\sum_{u \in N(x) \cap N(y)}\frac{1}{log(|N(u)|)}$
<!-- #endregion -->

```python id="hLsKWMNisi_l"
def calc_adar_in(a,b):
    sum=0
    try:
        n=list(set(train_graph.successors(a)).intersection(set(train_graph.successors(b))))
        if len(n)!=0:
            for i in n:
                sum=sum+(1/np.log10(len(list(train_graph.predecessors(i)))))
            return sum
        else:
            return 0
    except:
        return 0
```

<!-- #region id="9Seitdl9dVdP" -->
### Is person following back?
<!-- #endregion -->

```python id="S7g-HZxHsnI8"
def follows_back(a,b):
    if train_graph.has_edge(b,a):
        return 1
    else:
        return 0
```

<!-- #region id="2XcRYb-idYfe" -->
#### Katz centrality
<!-- #endregion -->

<!-- #region id="1xEzVB5mssgc" -->
Katz centrality computes the centrality for a node based on the centrality of its neighbors. It is a generalization of the eigenvector centrality. The Katz centrality for node i is: $x_i = \alpha \sum_{j} A_{ij} x_j + \beta$
<!-- #endregion -->

```python id="3QWUUsvjs4nJ"
katz = nx.katz.katz_centrality(train_graph,alpha=0.005,beta=1)
pickle.dump(katz,open('katz.p','wb'))
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 9, "status": "ok", "timestamp": 1627620166647, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="4_73m3mds8A8" outputId="555fa6c0-277d-40f9-eb6b-9339ccc21b87"
print('min',katz[min(katz, key=katz.get)])
print('max',katz[max(katz, key=katz.get)])
print('mean',float(sum(katz.values())) / len(katz))
```

<!-- #region id="UfIBDbxDuEch" -->
## Checkpointing
<!-- #endregion -->

```python id="_Aofopy4uIA_"
# !mkdir fbfndrec
# %cd fbfndrec

# !mv ../train.csv .
# !mv ../test.csv .

# !mv ../train_pos_after_eda.csv .
# !mv ../test_pos_after_eda.csv .
# !mv ../train_neg_after_eda.csv .
# !mv ../test_neg_after_eda.csv .

# !mv ../train_after_eda.csv .
# !mv ../test_after_eda.csv .
# !mv ../train_y.csv .
# !mv ../test_y.csv .

# !mv ../page_rank.p .
# !mv ../katz.p .

# !zip fbfndrec.zip ./*

# !mv fbfndrec.zip /content/drive/MyDrive/TempData
```
