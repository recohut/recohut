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

<!-- #region id="yqJxnsvMzFUb" -->
# Graph encoder
<!-- #endregion -->

```python id="5irOe9bCuNr6"
import numpy as np
import pandas as pd
import random
import time
import os
import requests
from scipy import linalg

from sklearn.model_selection import RepeatedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
```

```python id="QPZAx8FXuVrm"
def EncoderLDA(A):
    begin1 = time.time()
    U, s, VT = np.linalg.svd(A)
    end1 = time.time()
    newx=([])
    dmax=50
    tmp = np.zeros([dmax,1])
    ASEKNN=[]

    #Split Data
    rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=5)

    for train_index, test_index in rkf.split(A):
         #print("TRAIN:", train_index, "TEST:", test_index)
         X_train, X_test = A[train_index], A[test_index]
         y_train, y_test = y[train_index], y[test_index]
         X1 = X_train.transpose()
         X_train, X_test = X1[train_index], X1[test_index]
         for d in range (1,dmax):
            newx=U[:,0:d]*s[0:d]**0.5
            #newx=np.reshape(newx,(n,d))
            clf2 = LinearDiscriminantAnalysis()
            newx_train, newx_test =newx[train_index], newx[test_index]
            clf2.fit(newx_train, y_train.ravel())
            tmp[d,0]=tmp[d,0]+clf2.score(newx_test, y_test.ravel())/50
         
         neigh2 = KNeighborsClassifier(weights='distance',metric='euclidean')
         neigh2.fit(newx_train, y_train.ravel())
         ASEKNN.append(neigh2.score(newx_test, y_test))

    db=tmp.argmax()
    ASELDAACC=tmp.max()
    ASELDATIME=end1-begin1
    ASEKNNACC=sum(ASEKNN)/50
    #nk,w,Z
    begin2 = time.time()
    AEEKNNACC=[]
    AEELDAACC=[]
    AEEKNN2ACC=[]

    rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=5)

    for train_index, test_index in rkf.split(A):
         #print("TRAIN:", train_index, "TEST:", test_index)
         X_train, X_test = A[train_index], A[test_index]
         y_train, y_test = y[train_index], y[test_index]
         X1 = X_train.transpose()
         X_train, X_test = X1[train_index], X1[test_index]
            
         begin4 = time.time()
         nk = np.zeros((1,K))

         for i in range(0,len(y_train)-1):
                nk[0,int(y_train[i,0]-1)]=nk[0,int(y_train[i,0]-1)]+1
                w = np.zeros((int(np.size(y_train)),K))

         for i in range(0,int(np.size(y_train))):
                k=int(y_train[i])
                w[i][k-1]=1/nk[0,k-1]

         z_train=np.matmul(X_train,w)
         z_test = np.matmul(X_test,w)
         end4 = time.time()
         zTIME = end4-begin4

         begin2 = time.time()
         neigh = KNeighborsClassifier(weights='uniform',metric='euclidean')
         neigh.fit(z_train, y_train.ravel())
         AEEKNNACC.append(neigh.score(z_test, y_test))
         end2 = time.time()
         AEEKNNTIME=end2-begin2
            
         neigh3 = KNeighborsClassifier(weights='distance',metric='euclidean')
         neigh.fit(z_train, y_train.ravel())
         AEEKNN2ACC.append(neigh.score(z_test, y_test))
            
         begin3 = time.time()
         clf4 = LinearDiscriminantAnalysis()
         clf4.fit(z_train,y_train.ravel())
         AEELDAACC.append(clf4.score(z_test, y_test.ravel()))
         end3 = time.time()
         AEELDATIME=end3-begin3

    #Get result
    AeK=sum(AEEKNNACC)/50
    AeL=sum(AEELDAACC)/50
    AeK2=sum(AEEKNN2ACC)/50
    Acc=np.array([ASELDAACC,ASEKNNACC,AeK,AeL,AeK2])
    Acc=np.round(Acc,3)
    Time=np.array([ASELDATIME,AEEKNNTIME+zTIME,AEELDATIME+zTIME])
    Time=np.round(Time,3)
    
    print(f"d = {db}")
    print(f"Accuracy for each method: ASELDA={Acc[0]},ASEKNN={Acc[1]},AEEKNN={Acc[2]},AEELDA={Acc[3]},AEEKNN2={Acc[4]}")
    print(f"Time for each method: {Time}")
```

<!-- #region id="mzBkUnhivBJY" -->
### Datasets
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="KdNYsi2IvD0M" executionInfo={"status": "ok", "timestamp": 1634046133952, "user_tz": -330, "elapsed": 4038, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6e9e16f3-b74a-4c31-9430-050c2715527c"
!git clone https://github.com/sparsh-ai/graph-embeddings.git
```

```python colab={"base_uri": "https://localhost:8080/"} id="jFTJdmWzvehy" executionInfo={"status": "ok", "timestamp": 1634046184779, "user_tz": -330, "elapsed": 1516, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9a20bf50-247d-46d4-af0a-87f82ebedb10"
!apt-get -qq install tree
!tree --du -h ./graph-embeddings/data/bronze
```

```python id="ZFWH8wzDvqLN"
data_basepath = '/content/graph-embeddings/data/bronze'
```

<!-- #region id="i68J5peZv-kA" -->
### CORA
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Idkp1U0Kunhg" executionInfo={"status": "ok", "timestamp": 1634046370951, "user_tz": -330, "elapsed": 641, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ed761e3e-fcb1-4323-86cd-fc752abf8c6e"
Edges = pd.read_csv(os.path.join(data_basepath,"Cora/cora edges.csv"),header=None)
Edges = np.array(Edges)

Labels = pd.read_csv(os.path.join(data_basepath,"Cora/cora node_labels.csv"),header=None)
y = np.array(Labels)

Edges.max(),np.shape(Edges),np.shape(y)
```

```python id="Dh9DQIxlu0RA"
A = np.zeros((2708,2708))

for i in range (0,5429):
    A[Edges[i,0]-1,Edges[i,1]-1]=1

n = 2708
K = int(y.max())
```

```python colab={"base_uri": "https://localhost:8080/"} id="E-80xl93wSvF" executionInfo={"status": "ok", "timestamp": 1634046466320, "user_tz": -330, "elapsed": 74171, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="229fa386-dec7-42a7-8cd1-04e5e9c56a6b"
EncoderLDA(A)
```

<!-- #region id="knAlPeb8wTCA" -->
### Citeseer
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="-5a-WeyHwm0H" executionInfo={"status": "ok", "timestamp": 1634046492987, "user_tz": -330, "elapsed": 429, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1d688776-9139-4982-925e-7f5336a9fc5e"
Edges = pd.read_csv(os.path.join(data_basepath,"Citeseer/citeseer edges.csv"),header=None)
Edges = np.array(Edges)

Labels = pd.read_csv(os.path.join(data_basepath,"Citeseer/citeseer node_labels.csv"),header=None)
y = np.array(Labels)

Edges.max(),np.shape(Edges),np.shape(y)
```

```python id="NyBfCXiGwm0I"
A = np.zeros((3264,3264))

for i in range (0,4536):
    A[Edges[i,0]-1,Edges[i,1]-1]=1

n = 3264
K = int(y.max())
```

```python colab={"base_uri": "https://localhost:8080/"} id="u5TwgCz4wm0I" executionInfo={"status": "ok", "timestamp": 1634046642546, "user_tz": -330, "elapsed": 100705, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="24fa2377-67a6-4a5b-b39d-70c5a719f40b"
EncoderLDA(A)
```

<!-- #region id="BtsqRcLIwkpR" -->
### EU EMail
<!-- #endregion -->

```python id="erFJ9Q6NxSkD"
df = pd.read_excel(os.path.join(data_basepath,"EU-EMAIL-CORE/Labels.xlsx"), header = None)
y = df.to_numpy()

df2 = pd.read_excel(os.path.join(data_basepath,"EU-EMAIL-CORE/Core.xlsx"), header = None)
Edge = df2.to_numpy()
```

```python id="jsqHi_XJxkpM"
A = np.zeros((1005,1005))

for i in range (0,25571):
    A[Edge[i,0],Edge[i,1]]=1

n=1005
K=int(y.max())
```

```python colab={"base_uri": "https://localhost:8080/"} id="J5DsNgU7xjrL" executionInfo={"status": "ok", "timestamp": 1634046776766, "user_tz": -330, "elapsed": 41572, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a117f370-de19-4ede-dd1b-83c3ccffcb56"
EncoderLDA(A)
```

<!-- #region id="mVBqXAxPxm4M" -->
### DC-SBM
<!-- #endregion -->

```python id="LC26a7hRyEkt"
def ADCSBM(n,k):
    A = np.zeros((n,n))
    theta = np.random.beta(1, 4, n)
    theta2 = theta/sum(theta)
    B=np.array([[0.9,0.1,0.1],[0.1,0.6,0.1],[0.1,0.1,0.3]])
    for i in range(0,n):
        for j in range(i+1,n):
            if y[i] == y[j] == 1:
                A[i,j]=np.random.binomial(1,theta[i]*theta[j]*B[0,0])
            elif y[i] == y[j] == 2:
                A[i,j]=np.random.binomial(1,theta[i]*theta[j]*B[1,1])
            elif y[i] == y[j] == 3:
                A[i,j]=np.random.binomial(1,theta[i]*theta[j]*B[2,2])
            else:
                A[i,j]=np.random.binomial(1,theta[i]*theta[j]*0.1)
    for i in range(0,n):
        for j in range(0,i):
            A[i,j]=A[j,i]
    return A
```

```python id="vZ0kDZdfyGCJ"
def gety(n):
    y = np.zeros((n,1))
    #np.random.seed(2)
    for i in range(0,n):
        y[i,0] = np.random.choice(np.arange(1, 4), p=[0.2, 0.3,0.5])
    return(y)
```

```python colab={"base_uri": "https://localhost:8080/"} id="r1KATAX5yUv5" executionInfo={"status": "ok", "timestamp": 1634047108916, "user_tz": -330, "elapsed": 151716, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a6583636-b425-4cdb-85df-d1766dfd8fe7"
for n in range (100,1000,100):
    k=3
    y=gety(n)
    A=ADCSBM(n,k=k)
    EncoderLDA(A)
```
