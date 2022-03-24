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

```python id="-p45xus_9Lxv"
project_name = "reco-tut-ffr"; branch = "main"; account = "sparsh-ai"
```

```python id="D03Mx8Df9Lx1"
!cp /content/drive/MyDrive/mykeys.py /content
import mykeys
!rm /content/mykeys.py
path = "/content/" + project_name; 
!mkdir "{path}"
%cd "{path}"
import sys; sys.path.append(path)
!git config --global user.email "nb@recohut.com"
!git config --global user.name  "colab-sparsh"
!git init
!git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git
!git pull origin "{branch}"
!git checkout main
```

```python id="KpOfQQp7_-i9"
import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import networkx as nx
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="R1hSHYa2gDQd" executionInfo={"status": "ok", "timestamp": 1627742399531, "user_tz": -330, "elapsed": 1801, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1e31701b-a53a-43be-e308-a2345730a472"
traindf = pd.read_parquet('./data/bronze/train.parquet.gzip')
traindf.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 297} id="CfJy1nzZgjuv" executionInfo={"status": "ok", "timestamp": 1627742407322, "user_tz": -330, "elapsed": 1417, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f501cc92-d4e8-4adb-c197-8494428f344c"
traindf.describe()
```

```python colab={"base_uri": "https://localhost:8080/"} id="txewY4SiA5Di" executionInfo={"status": "ok", "timestamp": 1627742532665, "user_tz": -330, "elapsed": 3546, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ac248fee-f66a-4f51-8328-affa52019d8b"
print(traindf[traindf.isna().any(1)])
print(traindf.info())
print("Number of diplicate entries: ",sum(traindf.duplicated()))
```

```python colab={"base_uri": "https://localhost:8080/"} id="e1uJV_b-B5Fq" executionInfo={"status": "ok", "timestamp": 1627743076480, "user_tz": -330, "elapsed": 62279, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e9184a27-0602-4e22-8296-e0015b68c4db"
g = nx.from_pandas_edgelist(traindf,
                            source='source_node',
                            target='destination_node',
                            create_using=nx.DiGraph())
print(nx.info(g))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 421} id="ipmUsmU_YJ3c" executionInfo={"status": "ok", "timestamp": 1627743174301, "user_tz": -330, "elapsed": 3772, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2925a4dd-687f-4f22-b5b6-0d49c26aa633"
subgraph = nx.from_pandas_edgelist(traindf.head(20),
                            source='source_node',
                            target='destination_node',
                            create_using=nx.DiGraph())
# https://stackoverflow.com/questions/9402255/drawing-a-huge-graph-with-networkx-and-matplotlib

pos = nx.spring_layout(subgraph)
nx.draw(subgraph,pos,node_color='#A0CBE2',edge_color='#00bb5e',width=1,edge_cmap=plt.cm.Blues,with_labels=True)
plt.savefig("graph_sample.pdf")
print(nx.info(subgraph))
```

```python colab={"base_uri": "https://localhost:8080/"} id="QirVa7A4Y5Qu" executionInfo={"status": "ok", "timestamp": 1627743192401, "user_tz": -330, "elapsed": 743, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5249ee0d-f0b0-4979-acd8-6694e524855b"
# No of Unique persons 
print("The number of unique persons",len(g.nodes()))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 388} id="M77RkwjMZmVr" executionInfo={"status": "ok", "timestamp": 1627743240248, "user_tz": -330, "elapsed": 2869, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="28f8a200-b791-496b-a19e-fa55eac1b87c"
# No of followers of each person
indegree_dist = list(dict(g.in_degree()).values())
indegree_dist.sort()
plt.figure(figsize=(10,6))
plt.plot(indegree_dist)
plt.xlabel('Index No')
plt.ylabel('No Of Followers')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 388} id="O_ohVDNlZp9K" executionInfo={"status": "ok", "timestamp": 1627743245103, "user_tz": -330, "elapsed": 1596, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="aa4a5041-ccba-4a16-9651-0eea0805fb7f"
indegree_dist = list(dict(g.in_degree()).values())
indegree_dist.sort()
plt.figure(figsize=(10,6))
plt.plot(indegree_dist[0:1500000])
plt.xlabel('Index No')
plt.ylabel('No Of Followers')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 388} id="eg7CBHaoZ_UL" executionInfo={"status": "ok", "timestamp": 1627743250730, "user_tz": -330, "elapsed": 2656, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="17f67edf-96fd-42ec-a449-ed639c7052f0"
# No Of people each person is following
outdegree_dist = list(dict(g.out_degree()).values())
outdegree_dist.sort()
plt.figure(figsize=(10,6))
plt.plot(outdegree_dist)
plt.xlabel('Index No')
plt.ylabel('No Of people each person is following')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="lwHxDmr3aIB7" executionInfo={"status": "ok", "timestamp": 1627743262927, "user_tz": -330, "elapsed": 9366, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fc19348a-7709-450e-a280-149a4af25616"
print('No of persons who are not following anyone are {} ({:.2%})'.format(sum(np.array(outdegree_dist)==0),
                                                                        sum(np.array(outdegree_dist)==0)/len(outdegree_dist)))
```

```python colab={"base_uri": "https://localhost:8080/"} id="UDh4Y0rnaabu" executionInfo={"status": "ok", "timestamp": 1627743274830, "user_tz": -330, "elapsed": 9220, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="24479d6e-2e2a-44fa-88c9-5236f4a5dc21"
print('No of persons having zero followers are {} ({:.2%})'.format(sum(np.array(indegree_dist)==0),
                                                                        sum(np.array(indegree_dist)==0)/len(indegree_dist)))
```

```python colab={"base_uri": "https://localhost:8080/"} id="GWieGOF8bF2s" executionInfo={"status": "ok", "timestamp": 1627743280273, "user_tz": -330, "elapsed": 3213, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="29819d58-abf6-4f3a-b3cb-582410d4eecc"
count=0
for i in g.nodes():
    if len(list(g.predecessors(i)))==0 :
        if len(list(g.successors(i)))==0:
            count+=1
print('No of persons those are not following anyone and also not having any followers are',count)
```
