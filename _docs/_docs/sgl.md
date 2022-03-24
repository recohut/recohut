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

# Supervised Graph Learning


Feature-based method is a very naive (yet powerful) approach for solving graph-based supervised machine learning. The idea rely on the classic machine learning approach of handcrafted feature extraction. In this demo, we will be using the PROTEINS dataset, already integrated in StellarGraph.

```python
!pip install -q stellargraph
```

```python
from stellargraph import datasets
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
```

```python
dataset = datasets.PROTEINS()
graphs, graph_labels = dataset.load()
dataset.description
```

To compute the graph metrics, one way is to retrieve the adjacency matrix representation of each graph.

```python
pd.Series(graph_labels).value_counts(dropna=False)
```

```python
# convert graphs from StellarGraph format to numpy adj matrices
adjs = [graph.to_adjacency_matrix().A for graph in graphs]

# convert labes fom Pandas.Series to numpy array
labels = graph_labels.to_numpy(dtype=int)

metrics = []

for adj in adjs:
  G = nx.from_numpy_matrix(adj)
  # basic properties
  num_edges = G.number_of_edges()
  # clustering measures
  cc = nx.average_clustering(G)
  # measure of efficiency
  eff = nx.global_efficiency(G)

  metrics.append([num_edges, cc, eff])
```

```python
X_train, X_test, y_train, y_test = train_test_split(metrics, labels, test_size=0.3, random_state=42)
```

As commonly done in many Machine Learning workflows, we preprocess features to have zero mean and unit standard deviation

```python
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

It's now time for training a proper algorithm. We chose a support vector machine for this task

```python
clf = svm.SVC()
clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)

print('Accuracy', accuracy_score(y_test,y_pred))
print('Precision', precision_score(y_test,y_pred))
print('Recall', recall_score(y_test,y_pred))
print('F1-score', f1_score(y_test,y_pred))
```
