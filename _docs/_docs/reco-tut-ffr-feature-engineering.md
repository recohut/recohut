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

```python colab={"base_uri": "https://localhost:8080/"} id="Yky_7jNfJb70" executionInfo={"status": "ok", "timestamp": 1627748440560, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5f452ba5-a5ec-4d63-b13c-62fc4cecedde"
%cd "/content/reco-tut-ffr"
```

```python id="KpOfQQp7_-i9" executionInfo={"status": "ok", "timestamp": 1627752394193, "user_tz": -330, "elapsed": 618, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
import csv
import pickle
import random
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from scipy.sparse.linalg import svds, eigs
import networkx as nx
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
```

```python id="ykWX_MMYMcgc" executionInfo={"status": "ok", "timestamp": 1627749542147, "user_tz": -330, "elapsed": 18641, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
data_path_silver = './data/silver'

def read_df(name):
    return pd.read_parquet(os.path.join(data_path_silver,name+'.parquet.gzip'))

X_train_pos = read_df('X_train_pos')
X_train_neg = read_df('X_train_neg')
X_test_pos = read_df('X_test_pos')
X_test_neg = read_df('X_test_neg')

X_train = X_train_pos.append(X_train_neg, ignore_index=True)
X_test = X_test_pos.append(X_test_neg, ignore_index=True)
X_train.to_csv('/content/train_joined.csv', header=False, index=False)
X_test.to_csv('/content/test_joined.csv', header=False, index=False)

read_df('y_train').to_csv('/content/y_train.csv', header=False, index=False)
read_df('y_test').to_csv('/content/y_test.csv', header=False, index=False)
```

```python id="JCX1Wp7eZmcW" executionInfo={"status": "ok", "timestamp": 1627749304765, "user_tz": -330, "elapsed": 39453, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
filename = "/content/train_joined.csv"
n_train = sum(1 for line in open(filename)) #number of records in file (excludes header)
s = 100000 #desired sample size
skip_train = sorted(random.sample(range(1,n_train+1),n_train-s))
#https://stackoverflow.com/a/22259008/4084039
```

```python id="8DLLpyNdaqQY" executionInfo={"status": "ok", "timestamp": 1627749324163, "user_tz": -330, "elapsed": 8882, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
filename = "/content/test_joined.csv"
n_test = sum(1 for line in open(filename)) #number of records in file (excludes header)
s = 50000 #desired sample size
skip_test = sorted(random.sample(range(1,n_test+1),n_test-s))
```

```python colab={"base_uri": "https://localhost:8080/"} id="FcUl5-Caa3hs" executionInfo={"status": "ok", "timestamp": 1627749336128, "user_tz": -330, "elapsed": 566, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7baccb69-0454-4ac2-c1f7-213a84241bbc"
print("Number of rows in the train data file:", n_train)
print("Number of rows we are going to elimiate in train data are",len(skip_train))
print("Number of rows in the test data file:", n_test)
print("Number of rows we are going to elimiate in test data are",len(skip_test))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 128} id="9vTsUXd6a-4x" executionInfo={"status": "ok", "timestamp": 1627749639872, "user_tz": -330, "elapsed": 6625, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6d75f073-17c5-4709-c396-e63256fdf6dc"
df_final_train = pd.read_csv('/content/train_joined.csv', skiprows=skip_train, names=['source_node', 'destination_node'])
df_final_train['indicator_link'] = pd.read_csv('/content/y_train.csv', skiprows=skip_train, names=['indicator_link'])
print("Our train matrix size ",df_final_train.shape)
df_final_train.head(2)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 128} id="kGbHsjZMcIbL" executionInfo={"status": "ok", "timestamp": 1627749686151, "user_tz": -330, "elapsed": 1007, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8b4ccab3-4d64-43a5-cfb0-401a8a334adf"
df_final_test = pd.read_csv('/content/test_joined.csv', skiprows=skip_test, names=['source_node', 'destination_node'])
df_final_test['indicator_link'] = pd.read_csv('/content/y_test.csv', skiprows=skip_test, names=['indicator_link'])
print("Our test matrix size ",df_final_test.shape)
df_final_test.head(2)
```

```python id="GOR0YVOMM-D8" executionInfo={"status": "ok", "timestamp": 1627749778301, "user_tz": -330, "elapsed": 50525, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
X_train_pos = read_df('X_train_pos')
train_graph = nx.from_pandas_edgelist(X_train_pos,
                            source='source_node',
                            target='destination_node',
                            create_using=nx.DiGraph())
```

```python id="-nWaxxZJXFfT" executionInfo={"status": "ok", "timestamp": 1627748554885, "user_tz": -330, "elapsed": 595, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
data_path_gold = './data/gold'
if not os.path.exists(data_path_gold):
    os.makedirs(data_path_gold)
```

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

```python id="LXpgbkwGoZdG" executionInfo={"status": "ok", "timestamp": 1627748858340, "user_tz": -330, "elapsed": 573, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
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

```python id="6x1xsEP0o48W" executionInfo={"status": "ok", "timestamp": 1627748858927, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
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

```python id="5RN3c0SKo_gn" executionInfo={"status": "ok", "timestamp": 1627748858929, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
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

```python id="-uN89BjupCPI" executionInfo={"status": "ok", "timestamp": 1627748858932, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
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
pr = nx.pagerank(train_graph, alpha=0.85)
pickle.dump(pr,open(os.path.join(data_path_gold,'page_rank.p'),'wb'))
```

```python colab={"base_uri": "https://localhost:8080/"} id="dLfGxokmYJGf" executionInfo={"status": "ok", "timestamp": 1627751958730, "user_tz": -330, "elapsed": 908, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="97d9dad3-562c-4065-fbab-6d7d171a45d0"
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

```python id="y3iWLy5Bqu2W" executionInfo={"status": "ok", "timestamp": 1627748879146, "user_tz": -330, "elapsed": 708, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
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

```python colab={"base_uri": "https://localhost:8080/"} id="BzslYnz-rKW7" executionInfo={"status": "ok", "timestamp": 1627619545396, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fd6dac23-5858-481d-e886-668ed50e37b5"
# unit test 1
compute_shortest_path_length(77697, 826021)
```

```python colab={"base_uri": "https://localhost:8080/"} id="9dgfHKwGrO27" executionInfo={"status": "ok", "timestamp": 1627619546006, "user_tz": -330, "elapsed": 4, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d96fadfb-65d9-4e70-b4c2-03525c0d9864"
# unit test 2
compute_shortest_path_length(669354, 1635354)
```

<!-- #region id="iv27JsDLdMQb" -->
#### Same community
<!-- #endregion -->

```python id="_yU7WeJFsFyO" executionInfo={"status": "ok", "timestamp": 1627749945622, "user_tz": -330, "elapsed": 15807, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
wcc = list(nx.weakly_connected_components(train_graph))
```

```python id="UlYshkkxsDPr" executionInfo={"status": "ok", "timestamp": 1627748895436, "user_tz": -330, "elapsed": 787, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
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

```python id="hLsKWMNisi_l" executionInfo={"status": "ok", "timestamp": 1627748898163, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
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

```python id="S7g-HZxHsnI8" executionInfo={"status": "ok", "timestamp": 1627748899031, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
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
pickle.dump(katz,open(os.path.join(data_path_gold,'katz.p'),'wb'))
```

```python colab={"base_uri": "https://localhost:8080/"} id="4_73m3mds8A8" executionInfo={"status": "ok", "timestamp": 1627751967673, "user_tz": -330, "elapsed": 1678, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="76b36dac-b632-4380-e572-72c112d1dee9"
print('min',katz[min(katz, key=katz.get)])
print('max',katz[max(katz, key=katz.get)])
print('mean',float(sum(katz.values())) / len(katz))
```

<!-- #region id="KsB7rNwYYzBU" -->
## Adding a set of features
we will create these each of these features for both train and test data points:
- jaccard_followers
- jaccard_followees
- cosine_followers
- cosine_followees
- num_followers_s
- num_followees_s
- num_followers_d
- num_followees_d
- inter_followers
- inter_followees
<!-- #endregion -->

```python id="bfhBF46VdDv8" executionInfo={"status": "ok", "timestamp": 1627750019140, "user_tz": -330, "elapsed": 22153, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
#mapping jaccrd followers to train and test data
df_final_train['jaccard_followers'] = df_final_train.apply(lambda row:
                                        jaccard_for_followers(row['source_node'],row['destination_node']),axis=1)
df_final_test['jaccard_followers'] = df_final_test.apply(lambda row:
                                        jaccard_for_followers(row['source_node'],row['destination_node']),axis=1)

#mapping jaccrd followees to train and test data
df_final_train['jaccard_followees'] = df_final_train.apply(lambda row:
                                        jaccard_for_followees(row['source_node'],row['destination_node']),axis=1)
df_final_test['jaccard_followees'] = df_final_test.apply(lambda row:
                                        jaccard_for_followees(row['source_node'],row['destination_node']),axis=1)


#mapping jaccrd followers to train and test data
df_final_train['cosine_followers'] = df_final_train.apply(lambda row:
                                        cosine_for_followers(row['source_node'],row['destination_node']),axis=1)
df_final_test['cosine_followers'] = df_final_test.apply(lambda row:
                                        cosine_for_followers(row['source_node'],row['destination_node']),axis=1)

#mapping jaccrd followees to train and test data
df_final_train['cosine_followees'] = df_final_train.apply(lambda row:
                                        cosine_for_followees(row['source_node'],row['destination_node']),axis=1)
df_final_test['cosine_followees'] = df_final_test.apply(lambda row:
                                        cosine_for_followees(row['source_node'],row['destination_node']),axis=1)
```

```python id="8dPs5yegdgvA" executionInfo={"status": "ok", "timestamp": 1627750029140, "user_tz": -330, "elapsed": 553, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def compute_features_stage1(df_final):
    #calculating no of followers followees for source and destination
    #calculating intersection of followers and followees for source and destination
    num_followers_s=[]
    num_followees_s=[]
    num_followers_d=[]
    num_followees_d=[]
    inter_followers=[]
    inter_followees=[]
    for i,row in df_final.iterrows():
        try:
            s1=set(train_graph.predecessors(row['source_node']))
            s2=set(train_graph.successors(row['source_node']))
        except:
            s1 = set()
            s2 = set()
        try:
            d1=set(train_graph.predecessors(row['destination_node']))
            d2=set(train_graph.successors(row['destination_node']))
        except:
            d1 = set()
            d2 = set()
        num_followers_s.append(len(s1))
        num_followees_s.append(len(s2))

        num_followers_d.append(len(d1))
        num_followees_d.append(len(d2))

        inter_followers.append(len(s1.intersection(d1)))
        inter_followees.append(len(s2.intersection(d2)))
    
    return num_followers_s, num_followers_d, num_followees_s, num_followees_d, inter_followers, inter_followees
```

```python id="AHtRupJXdlDZ" executionInfo={"status": "ok", "timestamp": 1627750155506, "user_tz": -330, "elapsed": 15362, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
if not os.path.isfile(os.path.join(data_path_gold, 'storage_sample_stage1.h5')):
    df_final_train['num_followers_s'], df_final_train['num_followers_d'], \
    df_final_train['num_followees_s'], df_final_train['num_followees_d'], \
    df_final_train['inter_followers'], df_final_train['inter_followees']= compute_features_stage1(df_final_train)
    
    df_final_test['num_followers_s'], df_final_test['num_followers_d'], \
    df_final_test['num_followees_s'], df_final_test['num_followees_d'], \
    df_final_test['inter_followers'], df_final_test['inter_followees']= compute_features_stage1(df_final_test)
    
    hdf = pd.HDFStore(os.path.join(data_path_gold, 'storage_sample_stage1.h5'))
    hdf.put('train_df',df_final_train, format='table', data_columns=True)
    hdf.put('test_df',df_final_test, format='table', data_columns=True)
    hdf.close()
else:
    df_final_train = pd.read_hdf(os.path.join(data_path_gold, 'storage_sample_stage1.h5'), 'train_df',mode='r')
    df_final_test = pd.read_hdf(os.path.join(data_path_gold, 'storage_sample_stage1.h5'), 'test_df',mode='r')
```

<!-- #region id="UsZxvgqaeWFo" -->
## Adding new set of features
we will create these each of these features for both train and test data points:
- adar index
- is following back
- belongs to same weakly connect components
- shortest path between source and destination
<!-- #endregion -->

```python id="hoXX6jkJeZNz" executionInfo={"status": "ok", "timestamp": 1627751369369, "user_tz": -330, "elapsed": 192280, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
if not os.path.isfile(os.path.join(data_path_gold, 'storage_sample_stage2.h5')):
    #mapping adar index on train
    df_final_train['adar_index'] = df_final_train.apply(lambda row: calc_adar_in(row['source_node'],row['destination_node']),axis=1)
    #mapping adar index on test
    df_final_test['adar_index'] = df_final_test.apply(lambda row: calc_adar_in(row['source_node'],row['destination_node']),axis=1)

    #--------------------------------------------------------------------------------------------------------
    #mapping followback or not on train
    df_final_train['follows_back'] = df_final_train.apply(lambda row: follows_back(row['source_node'],row['destination_node']),axis=1)

    #mapping followback or not on test
    df_final_test['follows_back'] = df_final_test.apply(lambda row: follows_back(row['source_node'],row['destination_node']),axis=1)

    #--------------------------------------------------------------------------------------------------------
    #mapping same component of wcc or not on train
    df_final_train['same_comp'] = df_final_train.apply(lambda row: belongs_to_same_wcc(row['source_node'],row['destination_node']),axis=1)

    ##mapping same component of wcc or not on train
    df_final_test['same_comp'] = df_final_test.apply(lambda row: belongs_to_same_wcc(row['source_node'],row['destination_node']),axis=1)
    
    #--------------------------------------------------------------------------------------------------------
    #mapping shortest path on train 
    df_final_train['shortest_path'] = df_final_train.apply(lambda row: compute_shortest_path_length(row['source_node'],row['destination_node']),axis=1)
    #mapping shortest path on test
    df_final_test['shortest_path'] = df_final_test.apply(lambda row: compute_shortest_path_length(row['source_node'],row['destination_node']),axis=1)

    hdf = pd.HDFStore(os.path.join(data_path_gold, 'storage_sample_stage2.h5'))
    hdf.put('train_df',df_final_train, format='table', data_columns=True)
    hdf.put('test_df',df_final_test, format='table', data_columns=True)
    hdf.close()
else:
    df_final_train = pd.read_hdf(os.path.join(data_path_gold, 'storage_sample_stage2.h5'), 'train_df',mode='r')
    df_final_test = pd.read_hdf(os.path.join(data_path_gold, 'storage_sample_stage2.h5'), 'test_df',mode='r')
```

<!-- #region id="AXnsp9Cxewsd" -->
## Adding new set of features
we will create these each of these features for both train and test data points:
- Weight Features
    - weight of incoming edges
    - weight of outgoing edges
    - weight of incoming edges + weight of outgoing edges
    - weight of incoming edges * weight of outgoing edges
    - 2*weight of incoming edges + weight of outgoing edges
    - weight of incoming edges + 2*weight of outgoing edges
- Page Ranking of source
- Page Ranking of dest
- katz of source
- katz of dest
- hubs of source
- hubs of dest
- authorities_s of source
- authorities_s of dest
<!-- #endregion -->

<!-- #region id="RhNp6N_ijgH5" -->
### Weight Features
In order to determine the similarity of nodes, an edge weight value was calculated between nodes. Edge weight decreases as the neighbor count goes up. Intuitively, consider one million people following a celebrity on a social network then chances are most of them never met each other or the celebrity. On the other hand, if a user has 30 contacts in his/her social network, the chances are higher that many of them know each other. credit - Graph-based Features for Supervised Link Prediction William Cukierski, Benjamin Hamner, Bo Yang

$W = \frac{1}{\sqrt{1+|X|}}$

it is directed graph so calculated Weighted in and Weighted out differently.


<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 66, "referenced_widgets": ["66c60af8499148a6baca78999f1ef5b1", "4939499490c344ff8bd7b8766cf66611", "f316972ba0c14fd69909465cc43d275c", "845f0b9dbc6d4d38bb0462ac8fd48df7", "a72c371a43784024b3cfc4d386a5d701", "a7893b46de7f4251b2e88ef24ae0b0ce", "974a30fc3b46449db1d6610573e0a25e", "a6ed65bc39cc4055af2a68f0a82ce7a6"]} id="mL70Wjn7j9tu" executionInfo={"status": "ok", "timestamp": 1627751734904, "user_tz": -330, "elapsed": 19490, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="79f37374-86d2-481e-f10b-18d1819f3a31"
#weight for source and destination of each link
Weight_in = {}
Weight_out = {}
for i in  tqdm(train_graph.nodes()):
    s1=set(train_graph.predecessors(i))
    w_in = 1.0/(np.sqrt(1+len(s1)))
    Weight_in[i]=w_in
    
    s2=set(train_graph.successors(i))
    w_out = 1.0/(np.sqrt(1+len(s2)))
    Weight_out[i]=w_out
    
#for imputing with mean
mean_weight_in = np.mean(list(Weight_in.values()))
mean_weight_out = np.mean(list(Weight_out.values()))
```

```python id="h5KiDZT3kDhq" executionInfo={"status": "ok", "timestamp": 1627751750851, "user_tz": -330, "elapsed": 604, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
#mapping to pandas train
df_final_train['weight_in'] = df_final_train.destination_node.apply(lambda x: Weight_in.get(x,mean_weight_in))
df_final_train['weight_out'] = df_final_train.source_node.apply(lambda x: Weight_out.get(x,mean_weight_out))

#mapping to pandas test
df_final_test['weight_in'] = df_final_test.destination_node.apply(lambda x: Weight_in.get(x,mean_weight_in))
df_final_test['weight_out'] = df_final_test.source_node.apply(lambda x: Weight_out.get(x,mean_weight_out))

#some features engineerings on the in and out weights
df_final_train['weight_f1'] = df_final_train.weight_in + df_final_train.weight_out
df_final_train['weight_f2'] = df_final_train.weight_in * df_final_train.weight_out
df_final_train['weight_f3'] = (2*df_final_train.weight_in + 1*df_final_train.weight_out)
df_final_train['weight_f4'] = (1*df_final_train.weight_in + 2*df_final_train.weight_out)

#some features engineerings on the in and out weights
df_final_test['weight_f1'] = df_final_test.weight_in + df_final_test.weight_out
df_final_test['weight_f2'] = df_final_test.weight_in * df_final_test.weight_out
df_final_test['weight_f3'] = (2*df_final_test.weight_in + 1*df_final_test.weight_out)
df_final_test['weight_f4'] = (1*df_final_test.weight_in + 2*df_final_test.weight_out)
```

```python id="LhTORAblkk3x" executionInfo={"status": "ok", "timestamp": 1627752118556, "user_tz": -330, "elapsed": 3066, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
pr = pickle.load(open(os.path.join(data_path_gold,'page_rank.p'),'rb'))
mean_pr = float(sum(pr.values())) / len(pr)

katz = pickle.load(open(os.path.join(data_path_gold,'katz.p'),'rb'))
mean_katz = float(sum(katz.values())) / len(katz)
```

```python id="uvqNV4l6kI41" executionInfo={"status": "ok", "timestamp": 1627752119721, "user_tz": -330, "elapsed": 1173, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
if not os.path.isfile(os.path.join(data_path_gold, 'storage_sample_stage3.h5')):
    
    #page rank for source and destination in Train and Test
    #if anything not there in train graph then adding mean page rank 
    df_final_train['page_rank_s'] = df_final_train.source_node.apply(lambda x:pr.get(x,mean_pr))
    df_final_train['page_rank_d'] = df_final_train.destination_node.apply(lambda x:pr.get(x,mean_pr))

    df_final_test['page_rank_s'] = df_final_test.source_node.apply(lambda x:pr.get(x,mean_pr))
    df_final_test['page_rank_d'] = df_final_test.destination_node.apply(lambda x:pr.get(x,mean_pr))
    #================================================================================

    #Katz centrality score for source and destination in Train and test
    #if anything not there in train graph then adding mean katz score
    df_final_train['katz_s'] = df_final_train.source_node.apply(lambda x: katz.get(x,mean_katz))
    df_final_train['katz_d'] = df_final_train.destination_node.apply(lambda x: katz.get(x,mean_katz))

    df_final_test['katz_s'] = df_final_test.source_node.apply(lambda x: katz.get(x,mean_katz))
    df_final_test['katz_d'] = df_final_test.destination_node.apply(lambda x: katz.get(x,mean_katz))
    #================================================================================

    hdf = pd.HDFStore(os.path.join(data_path_gold, 'storage_sample_stage3.h5'))
    hdf.put('train_df',df_final_train, format='table', data_columns=True)
    hdf.put('test_df',df_final_test, format='table', data_columns=True)
    hdf.close()
else:
    df_final_train = pd.read_hdf(os.path.join(data_path_gold, 'storage_sample_stage3.h5'), 'train_df',mode='r')
    df_final_test = pd.read_hdf(os.path.join(data_path_gold, 'storage_sample_stage3.h5'), 'test_df',mode='r')
```

<!-- #region id="9372vIYHkW0o" -->
### Adding new feature Preferential Attachement
One well-known concept in social networks is that users with many friends tend to create more connections in the future. This is due to the fact that in some social networks, like in finance, the rich get richer. We estimate how ”rich” our two vertices are by calculating the multiplication between the number of friends (|Γ(x)|) or followers each vertex has.
<!-- #endregion -->

```python id="GtteWonflnTn" executionInfo={"status": "ok", "timestamp": 1627752214843, "user_tz": -330, "elapsed": 626, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Preferential Attachement for followers
#for train dataset
nfs=np.array(df_final_train['num_followers_s'])
nfd=np.array(df_final_train['num_followers_d'])
preferential_followers=[]
for i in range(len(nfs)):
    preferential_followers.append(nfd[i]*nfs[i])
df_final_train['prefer_Attach_followers']= preferential_followers

#for test dataset
nfs=np.array(df_final_test['num_followers_s'])
nfd=np.array(df_final_test['num_followers_d'])
preferential_followers=[]
for i in range(len(nfs)):
    preferential_followers.append(nfd[i]*nfs[i])
df_final_test['prefer_Attach_followers']= preferential_followers

# Preferential Attachement for followers
#for train dataset
nfs=np.array(df_final_train['num_followees_s'])
nfd=np.array(df_final_train['num_followees_d'])
preferential_followees=[]
for i in range(len(nfs)):
    preferential_followees.append(nfd[i]*nfs[i])
df_final_train['prefer_Attach_followees']= preferential_followees

#for test dataset
nfs=np.array(df_final_test['num_followees_s'])
nfd=np.array(df_final_test['num_followees_d'])
preferential_followees=[]
for i in range(len(nfs)):
    preferential_followees.append(nfd[i]*nfs[i])
df_final_test['prefer_Attach_followees']= preferential_followees
```

<!-- #region id="qoYxcRcdl6lz" -->
### SVD features for both source and destination
<!-- #endregion -->

```python id="vMWkHCWYl9uU" executionInfo={"status": "ok", "timestamp": 1627752231740, "user_tz": -330, "elapsed": 820, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def svd(x, S):
    try:
        z = sadj_dict[x]
        return S[z]
    except:
        return [0,0,0,0,0,0]
```

```python colab={"base_uri": "https://localhost:8080/"} id="vWbyNxyfl-va" executionInfo={"status": "ok", "timestamp": 1627752436366, "user_tz": -330, "elapsed": 28654, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e888bc62-6f5c-4085-ae7f-bcf2bf40a7b2"
#for svd features to get feature vector creating a dict node val and index in svd vector
sadj_col = sorted(train_graph.nodes())
sadj_dict = { val:idx for idx,val in enumerate(sadj_col)}

Adj = nx.adjacency_matrix(train_graph,nodelist=sorted(train_graph.nodes())).asfptype()

U, s, V = svds(Adj, k = 6)
print('Adjacency matrix Shape',Adj.shape)
print('U Shape',U.shape)
print('V Shape',V.shape)
print('s Shape',s.shape)
```

```python id="HiTVpbaBmRl1" executionInfo={"status": "ok", "timestamp": 1627752603924, "user_tz": -330, "elapsed": 167577, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
df_final_train[['svd_u_s_1', 'svd_u_s_2','svd_u_s_3', 'svd_u_s_4', 'svd_u_s_5', 'svd_u_s_6']] = \
df_final_train.source_node.apply(lambda x: svd(x, U)).apply(pd.Series)

df_final_train[['svd_u_d_1', 'svd_u_d_2', 'svd_u_d_3', 'svd_u_d_4', 'svd_u_d_5','svd_u_d_6']] = \
df_final_train.destination_node.apply(lambda x: svd(x, U)).apply(pd.Series)
#===================================================================================================

df_final_train[['svd_v_s_1','svd_v_s_2', 'svd_v_s_3', 'svd_v_s_4', 'svd_v_s_5', 'svd_v_s_6',]] = \
df_final_train.source_node.apply(lambda x: svd(x, V.T)).apply(pd.Series)

df_final_train[['svd_v_d_1', 'svd_v_d_2', 'svd_v_d_3', 'svd_v_d_4', 'svd_v_d_5','svd_v_d_6']] = \
df_final_train.destination_node.apply(lambda x: svd(x, V.T)).apply(pd.Series)
#===================================================================================================

df_final_test[['svd_u_s_1', 'svd_u_s_2','svd_u_s_3', 'svd_u_s_4', 'svd_u_s_5', 'svd_u_s_6']] = \
df_final_test.source_node.apply(lambda x: svd(x, U)).apply(pd.Series)

df_final_test[['svd_u_d_1', 'svd_u_d_2', 'svd_u_d_3', 'svd_u_d_4', 'svd_u_d_5','svd_u_d_6']] = \
df_final_test.destination_node.apply(lambda x: svd(x, U)).apply(pd.Series)

#===================================================================================================

df_final_test[['svd_v_s_1','svd_v_s_2', 'svd_v_s_3', 'svd_v_s_4', 'svd_v_s_5', 'svd_v_s_6',]] = \
df_final_test.source_node.apply(lambda x: svd(x, V.T)).apply(pd.Series)

df_final_test[['svd_v_d_1', 'svd_v_d_2', 'svd_v_d_3', 'svd_v_d_4', 'svd_v_d_5','svd_v_d_6']] = \
df_final_test.destination_node.apply(lambda x: svd(x, V.T)).apply(pd.Series)
```

```python colab={"base_uri": "https://localhost:8080/"} id="2JQTraw5mWGK" executionInfo={"status": "ok", "timestamp": 1627752603927, "user_tz": -330, "elapsed": 38, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5b3d95ff-1365-4da2-ae55-22eda9db0814"
df_final_train.columns
```

<!-- #region id="h6C8-gUymYcb" -->
### Adding feature svd_dot
svd_dot is Dot product between sourse node svd and destination node svd features
<!-- #endregion -->

```python id="qloRTneMnEh8" executionInfo={"status": "ok", "timestamp": 1627752617369, "user_tz": -330, "elapsed": 573, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
#for train datasets
s1,s2,s3,s4,s5,s6=df_final_train['svd_u_s_1'],df_final_train['svd_u_s_2'],df_final_train['svd_u_s_3'],df_final_train['svd_u_s_4'],df_final_train['svd_u_s_5'],df_final_train['svd_u_s_6']
s7,s8,s9,s10,s11,s12=df_final_train['svd_v_s_1'],df_final_train['svd_v_s_2'],df_final_train['svd_v_s_3'],df_final_train['svd_v_s_4'],df_final_train['svd_v_s_5'],df_final_train['svd_v_s_6']

d1,d2,d3,d4,d5,d6=df_final_train['svd_u_d_1'],df_final_train['svd_u_d_2'],df_final_train['svd_u_d_3'],df_final_train['svd_u_d_4'],df_final_train['svd_u_d_5'],df_final_train['svd_u_d_6']
d7,d8,d9,d10,d11,d12=df_final_train['svd_v_d_1'],df_final_train['svd_v_d_2'],df_final_train['svd_v_d_3'],df_final_train['svd_v_d_4'],df_final_train['svd_v_d_5'],df_final_train['svd_v_d_6']
```

```python id="_XimNwDZnEeC" executionInfo={"status": "ok", "timestamp": 1627752633899, "user_tz": -330, "elapsed": 15633, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
svd_dot=[]
for i in range(len(np.array(s1))):
    a=[]
    b=[]
    a.append(np.array(s1[i]))
    a.append(np.array(s2[i]))
    a.append(np.array(s3[i]))
    a.append(np.array(s4[i]))
    a.append(np.array(s5[i]))
    a.append(np.array(s6[i]))
    a.append(np.array(s7[i]))
    a.append(np.array(s8[i]))
    a.append(np.array(s9[i]))
    a.append(np.array(s10[i]))
    a.append(np.array(s11[i]))
    a.append(np.array(s12[i]))
    b.append(np.array(d1[i]))
    b.append(np.array(d2[i]))
    b.append(np.array(d3[i]))
    b.append(np.array(d4[i]))
    b.append(np.array(d5[i]))
    b.append(np.array(d6[i]))
    b.append(np.array(d7[i]))
    b.append(np.array(d8[i]))
    b.append(np.array(d9[i]))
    b.append(np.array(d10[i]))
    b.append(np.array(d11[i]))
    b.append(np.array(d12[i]))
    svd_dot.append(np.dot(a,b))
    
df_final_train['svd_dot']=svd_dot   
```

```python colab={"base_uri": "https://localhost:8080/", "height": 309} id="p56Z3wi1nEZK" executionInfo={"status": "ok", "timestamp": 1627752633922, "user_tz": -330, "elapsed": 48, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="628e1f5a-5d48-47e4-f65b-df35aabb17af"
df_final_train.head()
```

```python id="_TGEOUmGnEVN" executionInfo={"status": "ok", "timestamp": 1627752633924, "user_tz": -330, "elapsed": 35, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
#for test dataset
s1,s2,s3,s4,s5,s6=df_final_test['svd_u_s_1'],df_final_test['svd_u_s_2'],df_final_test['svd_u_s_3'],df_final_test['svd_u_s_4'],df_final_test['svd_u_s_5'],df_final_test['svd_u_s_6']
s7,s8,s9,s10,s11,s12=df_final_test['svd_v_s_1'],df_final_test['svd_v_s_2'],df_final_test['svd_v_s_3'],df_final_test['svd_v_s_4'],df_final_test['svd_v_s_5'],df_final_test['svd_v_s_6']

d1,d2,d3,d4,d5,d6=df_final_test['svd_u_d_1'],df_final_test['svd_u_d_2'],df_final_test['svd_u_d_3'],df_final_test['svd_u_d_4'],df_final_test['svd_u_d_5'],df_final_test['svd_u_d_6']
d7,d8,d9,d10,d11,d12=df_final_test['svd_v_d_1'],df_final_test['svd_v_d_2'],df_final_test['svd_v_d_3'],df_final_test['svd_v_d_4'],df_final_test['svd_v_d_5'],df_final_test['svd_v_d_6']
```

```python id="u8cUSvVinMIz" executionInfo={"status": "ok", "timestamp": 1627752641178, "user_tz": -330, "elapsed": 7286, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
svd_dot=[]
for i in range(len(np.array(s1))):
    a=[]
    b=[]
    a.append(np.array(s1[i]))
    a.append(np.array(s2[i]))
    a.append(np.array(s3[i]))
    a.append(np.array(s4[i]))
    a.append(np.array(s5[i]))
    a.append(np.array(s6[i]))
    a.append(np.array(s7[i]))
    a.append(np.array(s8[i]))
    a.append(np.array(s9[i]))
    a.append(np.array(s10[i]))
    a.append(np.array(s11[i]))
    a.append(np.array(s12[i]))
    b.append(np.array(d1[i]))
    b.append(np.array(d2[i]))
    b.append(np.array(d3[i]))
    b.append(np.array(d4[i]))
    b.append(np.array(d5[i]))
    b.append(np.array(d6[i]))
    b.append(np.array(d7[i]))
    b.append(np.array(d8[i]))
    b.append(np.array(d9[i]))
    b.append(np.array(d10[i]))
    b.append(np.array(d11[i]))
    b.append(np.array(d12[i]))
    svd_dot.append(np.dot(a,b))
    
df_final_test['svd_dot']=svd_dot 
```

```python colab={"base_uri": "https://localhost:8080/", "height": 309} id="zf3BfSPnnPM5" executionInfo={"status": "ok", "timestamp": 1627752646000, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f79c2727-3edd-4e70-e376-bdddce3d1990"
df_final_test.head()
```

```python id="g5j7X_fynQ3d" executionInfo={"status": "ok", "timestamp": 1627752654498, "user_tz": -330, "elapsed": 5368, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
hdf = pd.HDFStore(os.path.join(data_path_gold,'storage_sample_stage4.h5'))
hdf.put('train_df',df_final_train, format='table', data_columns=True)
hdf.put('test_df',df_final_test, format='table', data_columns=True)
hdf.close()
```

```python colab={"base_uri": "https://localhost:8080/"} id="Wo9ZBmEInk6F" executionInfo={"status": "ok", "timestamp": 1627752688817, "user_tz": -330, "elapsed": 657, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ee43912e-60af-423f-f752-07065765897a"
!git status
```

```python colab={"base_uri": "https://localhost:8080/"} id="tnTCuyounuNV" executionInfo={"status": "ok", "timestamp": 1627752712979, "user_tz": -330, "elapsed": 9345, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6559d515-fa05-4589-8eee-81bab55932e7"
!git add .
!git commit -m 'added gold data layer'
```

```python colab={"base_uri": "https://localhost:8080/"} id="bz-SpKUonyKi" executionInfo={"status": "ok", "timestamp": 1627752750796, "user_tz": -330, "elapsed": 32229, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="12ee7722-0076-4cab-bf4b-ee48ab43582e"
!git push origin main
```
