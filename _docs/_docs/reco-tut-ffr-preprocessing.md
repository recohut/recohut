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

```python colab={"base_uri": "https://localhost:8080/"} id="Yky_7jNfJb70" executionInfo={"status": "ok", "timestamp": 1627746452144, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bd7c60e1-4d10-4a3b-f798-ebb20518e110"
%cd "/content/reco-tut-ffr"
```

```python id="KpOfQQp7_-i9"
import os
import csv
import pickle
import random
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import networkx as nx
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="R1hSHYa2gDQd" executionInfo={"status": "ok", "timestamp": 1627746559694, "user_tz": -330, "elapsed": 3345, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c457e714-65f7-4da6-88d4-8be1f75964dc"
traindf = pd.read_parquet('./data/bronze/train.parquet.gzip')
traindf.head()
```

```python id="e1uJV_b-B5Fq"
# g = nx.from_pandas_edgelist(traindf,
#                             source='source_node',
#                             target='destination_node',
#                             create_using=nx.DiGraph())
```

<!-- #region id="ieCAEYAVcXNM" -->
## Negative sampling
<!-- #endregion -->

<!-- #region id="m2pnbqRnhWkA" -->
Generating some edges which are not present in graph for supervised learning. In other words, we are generating bad links from graph which are not in graph and whose shortest path is greater than 2.
<!-- #endregion -->

```python id="wUHegz-DKajY"
## This pandas method is super slow, not sure why, compare to csv reader method
# traindf['weight'] = 1
# edges = traindf.set_index(['source_node','destination_node']).T.to_dict('records')[0]

traindf.to_csv('/content/train_woheader.csv', header=False, index=False)
r = csv.reader(open('/content/train_woheader.csv','r'))
edges = dict()
for edge in r:
    edges[(edge[0], edge[1])] = 1
```

```python colab={"base_uri": "https://localhost:8080/", "height": 66, "referenced_widgets": ["1f19e5db128845538e461b252bb0f41a", "edf0536c75eb468eb8ce127eddba4eef", "996cb19e772e4d899d5cd6c730380678", "4dbc176c0dd643aab2e0a7c424968727", "c524412c214b4d1fa8a93921e872ecd8", "b1a707ed44f84873908d37012373618d", "0b02f599f4444309b046801d279889a2", "36ac171d4fd8493596f9f2af056ef454"]} id="LUOab04oEinm" executionInfo={"status": "ok", "timestamp": 1627746652870, "user_tz": -330, "elapsed": 64047, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="55391178-0ca8-486a-b04b-6eb79378126d"
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
```

```python colab={"base_uri": "https://localhost:8080/"} id="PsgRA60dLpcr" executionInfo={"status": "ok", "timestamp": 1627745383641, "user_tz": -330, "elapsed": 1493, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a05df7dc-c078-45f9-fc4c-ff4641b20298"
list(missing_edges)[:10]
```

```python id="La0hfDv3IJ1G"
# pickle.dump(missing_edges,open('/content/missing_edges_final.p','wb'))
```

<!-- #region id="vcIK4mX1bP8A" -->
## Train/test split
<!-- #endregion -->

<!-- #region id="Xm8UdKXzlpUx" -->
> Tip: We will split positive links and negative links seperatly because we need only positive training data for creating graph and for feature generation.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="e_V7--QpL393" executionInfo={"status": "ok", "timestamp": 1627746671634, "user_tz": -330, "elapsed": 18784, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e846c7cc-0276-4fb6-87af-a24158ef6461"
#reading total data df
df_pos = traindf.copy()
df_neg = pd.DataFrame(list(missing_edges), columns=['source_node', 'destination_node'])

print("Number of nodes in the graph with edges", df_pos.shape[0])
print("Number of nodes in the graph without edges", df_neg.shape[0])
```

```python colab={"base_uri": "https://localhost:8080/"} id="fM1fiUw7MWiO" executionInfo={"status": "ok", "timestamp": 1627746674533, "user_tz": -330, "elapsed": 2903, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="24ac8a60-d518-432d-a867-2cb3a9d2d8d7"
#Train test split 
#Spiltted data into 80-20
X_train_pos, X_test_pos, y_train_pos, y_test_pos  = train_test_split(df_pos,np.ones(len(df_pos)),test_size=0.2, random_state=9)
X_train_neg, X_test_neg, y_train_neg, y_test_neg  = train_test_split(df_neg,np.zeros(len(df_neg)),test_size=0.2, random_state=9)

print('='*60)
print("Number of nodes in the train data graph with edges", X_train_pos.shape[0],"=",y_train_pos.shape[0])
print("Number of nodes in the train data graph without edges", X_train_neg.shape[0],"=", y_train_neg.shape[0])
print('='*60)
print("Number of nodes in the test data graph with edges", X_test_pos.shape[0],"=",y_test_pos.shape[0])
print("Number of nodes in the test data graph without edges", X_test_neg.shape[0],"=",y_test_neg.shape[0])
```

```python id="_kyuwJfQOogb"
# X_train = X_train_pos.append(X_train_neg,ignore_index=True)
# y_train = np.concatenate((y_train_pos,y_train_neg))
# X_test = X_test_pos.append(X_test_neg,ignore_index=True)
# y_test = np.concatenate((y_test_pos,y_test_neg)) 

# X_train.to_csv('train_after_eda.csv',header=False,index=False)
# X_test.to_csv('test_after_eda.csv',header=False,index=False)
# pd.DataFrame(y_train.astype(int)).to_csv('train_y.csv',header=False,index=False)
# pd.DataFrame(y_test.astype(int)).to_csv('test_y.csv',header=False,index=False)
```

```python id="ykWX_MMYMcgc"
# #removing header and saving
# X_train_pos.to_csv('train_pos_after_eda.csv',header=False, index=False)
# X_test_pos.to_csv('test_pos_after_eda.csv',header=False, index=False)
# X_train_neg.to_csv('train_neg_after_eda.csv',header=False, index=False)
# X_test_neg.to_csv('test_neg_after_eda.csv',header=False, index=False)

data_path_silver = './data/silver'
if not os.path.exists(data_path_silver):
    os.makedirs(data_path_silver)

def store_df(df, name):
    df.to_parquet(os.path.join(data_path_silver,name+'.parquet.gzip'), compression='gzip')

# store_df(X_train_pos, 'X_train_pos')
# store_df(X_test_pos, 'X_test_pos')
# store_df(X_train_neg, 'X_train_neg')
# store_df(X_test_neg, 'X_test_neg')
# store_df(X_train, 'X_train')
# store_df(X_test, 'X_test')
store_df(pd.DataFrame(y_train.astype(int), columns=['weight']), 'y_train')
store_df(pd.DataFrame(y_test.astype(int), columns=['weight']), 'y_test')
```

```python id="GOR0YVOMM-D8"
train_graph = nx.from_pandas_edgelist(X_train_pos,
                            source='source_node',
                            target='destination_node',
                            create_using=nx.DiGraph())

test_graph = nx.from_pandas_edgelist(X_test_pos,
                            source='source_node',
                            target='destination_node',
                            create_using=nx.DiGraph())
```

```python colab={"base_uri": "https://localhost:8080/"} id="PctZigJ5M8Pm" executionInfo={"status": "ok", "timestamp": 1627745984614, "user_tz": -330, "elapsed": 5991, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8b20a429-c8d8-4774-80f2-8c8c11af1431"
print(nx.info(train_graph))
print(nx.info(test_graph))
```

```python colab={"base_uri": "https://localhost:8080/"} id="b0ueT9XxM2C0" executionInfo={"status": "ok", "timestamp": 1627746018619, "user_tz": -330, "elapsed": 1422, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e9c70a87-ccfd-4214-8a8a-73e7d684fcb8"
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

```python colab={"base_uri": "https://localhost:8080/"} id="cLlrdpuASQnX" executionInfo={"status": "ok", "timestamp": 1627747088540, "user_tz": -330, "elapsed": 1317, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a63887ed-b3ad-4921-87d5-c0c3592e5faf"
!git status
```

```python colab={"base_uri": "https://localhost:8080/"} id="pYVQ2ZTOSS0P" executionInfo={"status": "ok", "timestamp": 1627747220397, "user_tz": -330, "elapsed": 60283, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5d941009-851c-4cbb-a55e-ac4e207e344d"
!git add .
!git commit -m 'added silver data layer'
!git push origin main
```
