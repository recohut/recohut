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

```python id="-p45xus_9Lxv" executionInfo={"status": "ok", "timestamp": 1627825400795, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
project_name = "reco-tut-ffr"; branch = "main"; account = "sparsh-ai"
```

```python id="D03Mx8Df9Lx1" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1627825439745, "user_tz": -330, "elapsed": 38963, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="46928da9-f222-454b-cd62-7ddd27e1c640"
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

```python id="Yky_7jNfJb70"
%cd "/content/reco-tut-ffr"
```

```python id="NTMLq6u29RAu" executionInfo={"status": "ok", "timestamp": 1627826046387, "user_tz": -330, "elapsed": 492, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
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
import math
import pickle
import os
import xgboost as xgb

import networkx as nx
import pdb
import pickle
from pandas import HDFStore,DataFrame
from pandas import read_hdf
from scipy.sparse.linalg import svds, eigs
import gc
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import xgboost as xgb
from prettytable import PrettyTable 
```

```python id="c59AzmkM9iD2" executionInfo={"status": "ok", "timestamp": 1627825635496, "user_tz": -330, "elapsed": 1293, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
data_path_gold = './data/gold'

df_final_train = read_hdf(os.path.join(data_path_gold,'storage_sample_stage4.h5'), 'train_df',mode='r')
df_final_test = read_hdf(os.path.join(data_path_gold,'storage_sample_stage4.h5'), 'test_df',mode='r')
```

```python colab={"base_uri": "https://localhost:8080/"} id="5Zx-fn2u9_ZH" executionInfo={"status": "ok", "timestamp": 1627825644792, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f9657b09-3325-4215-9d04-f99e6201282d"
df_final_train.columns
```

```python id="WTgSxgtc-B5T" executionInfo={"status": "ok", "timestamp": 1627825658294, "user_tz": -330, "elapsed": 488, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
y_train = df_final_train.indicator_link
y_test = df_final_test.indicator_link
```

```python id="Edb_Nm_g-GpJ" executionInfo={"status": "ok", "timestamp": 1627825667152, "user_tz": -330, "elapsed": 515, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
df_final_train.drop(['source_node', 'destination_node','indicator_link'],axis=1,inplace=True)
df_final_test.drop(['source_node', 'destination_node','indicator_link'],axis=1,inplace=True)
```

<!-- #region id="PvjqtG9V-_9g" -->
## Random forest
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Lx5elx75-KcK" executionInfo={"status": "ok", "timestamp": 1627825828675, "user_tz": -330, "elapsed": 120368, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="dc5e5e51-2b51-4431-f1d0-02db4b045522"
estimators = [10,50,100,250,450]
train_scores = []
test_scores = []
for i in estimators:
    clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=5, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=52, min_samples_split=120,
            min_weight_fraction_leaf=0.0, n_estimators=i, n_jobs=-1,random_state=25,verbose=0,warm_start=False)
    clf.fit(df_final_train,y_train)
    train_sc = f1_score(y_train,clf.predict(df_final_train))
    test_sc = f1_score(y_test,clf.predict(df_final_test))
    test_scores.append(test_sc)
    train_scores.append(train_sc)
    print('Estimators = ',i,'Train Score',train_sc,'test Score',test_sc)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 312} id="oEUnX0mJ-Rfn" executionInfo={"status": "ok", "timestamp": 1627825828680, "user_tz": -330, "elapsed": 61, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="26f764a7-1585-4441-a138-957dd8435622"
plt.plot(estimators,train_scores,label='Train Score')
plt.plot(estimators,test_scores,label='Test Score')
plt.xlabel('Estimators')
plt.ylabel('Score')
plt.title('Estimators vs score at depth of 5')
```

```python colab={"base_uri": "https://localhost:8080/"} id="bd17vRb2-SM3" executionInfo={"status": "ok", "timestamp": 1627826039612, "user_tz": -330, "elapsed": 210954, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="40ee41a8-a5d8-49f2-af99-1a25c01d29a8"
depths = [3,9,11,15,20,35,50,70,130]
train_scores = []
test_scores = []
for i in depths:
    clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=i, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=52, min_samples_split=120,
            min_weight_fraction_leaf=0.0, n_estimators=115, n_jobs=-1,random_state=25,verbose=0,warm_start=False)
    clf.fit(df_final_train,y_train)
    train_sc = f1_score(y_train,clf.predict(df_final_train))
    test_sc = f1_score(y_test,clf.predict(df_final_test))
    test_scores.append(test_sc)
    train_scores.append(train_sc)
    print('depth = ',i,'Train Score',train_sc,'test Score',test_sc)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 295} id="Jbprw_5R-WRK" executionInfo={"status": "ok", "timestamp": 1627826039616, "user_tz": -330, "elapsed": 43, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2256e4f2-9971-4ee1-d188-456c945272ad"
plt.plot(depths,train_scores,label='Train Score')
plt.plot(depths,test_scores,label='Test Score')
plt.xlabel('Depth')
plt.ylabel('Score')
plt.title('Depth vs score at depth of 5 at estimators = 115')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="8akBnXVU-Xkj" executionInfo={"status": "ok", "timestamp": 1627827274023, "user_tz": -330, "elapsed": 226241, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="34669503-88b4-4530-97b9-d3528f9fde9f"
param_dist = {"n_estimators":sp_randint(105,115),
              "max_depth": sp_randint(10,12),
              "min_samples_split": sp_randint(130,160),
              "min_samples_leaf": sp_randint(45,65)}

clf = RandomForestClassifier(random_state=25,n_jobs=-1)

rf_random = RandomizedSearchCV(clf, param_distributions=param_dist,
                               n_iter=5,cv=3,scoring='f1',random_state=25,
                               verbose=1, n_jobs=-1)

rf_random.fit(df_final_train,y_train)
```

```python colab={"base_uri": "https://localhost:8080/"} id="PRCyhsLFCv58" executionInfo={"status": "ok", "timestamp": 1627827369309, "user_tz": -330, "elapsed": 530, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3b0d4e0b-29d5-4d5f-dcdb-53073094190d"
print('mean test scores',rf_random.cv_results_['mean_test_score'])
```

```python colab={"base_uri": "https://localhost:8080/"} id="AHx0WkAJ-k9X" executionInfo={"status": "ok", "timestamp": 1627827373050, "user_tz": -330, "elapsed": 462, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f5a876e4-6a82-4cdc-8ab5-25910c68c483"
print(rf_random.best_estimator_)
```

```python id="5KLv2BWd-m3g" executionInfo={"status": "ok", "timestamp": 1627827400389, "user_tz": -330, "elapsed": 483, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
clf=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=14, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=28, min_samples_split=111,
            min_weight_fraction_leaf=0.0, n_estimators=121, n_jobs=-1,
            oob_score=False, random_state=25, verbose=0, warm_start=False)
```

```python id="f4qadagW-oqN" executionInfo={"status": "ok", "timestamp": 1627827430826, "user_tz": -330, "elapsed": 28342, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
clf.fit(df_final_train,y_train)
y_train_pred = clf.predict(df_final_train)
y_test_pred = clf.predict(df_final_test)
```

```python colab={"base_uri": "https://localhost:8080/"} id="7NTw_IIf-qtO" executionInfo={"status": "ok", "timestamp": 1627827430827, "user_tz": -330, "elapsed": 38, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="587adf72-67da-4ad9-9f45-371d34695b25"
print('Train f1 score',f1_score(y_train,y_train_pred))
print('Test f1 score',f1_score(y_test,y_test_pred))
```

```python id="D5Updf_D-w18" executionInfo={"status": "ok", "timestamp": 1627827430829, "user_tz": -330, "elapsed": 35, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def plot_confusion_matrix(test_y, predict_y):
    C = confusion_matrix(test_y, predict_y)
    
    A =(((C.T)/(C.sum(axis=1))).T)
    
    B =(C/C.sum(axis=0))
    plt.figure(figsize=(20,4))
    
    labels = [0,1]
    # representing A in heatmap format
    cmap=sns.light_palette("blue")
    plt.subplot(1, 3, 1)
    sns.heatmap(C, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Confusion matrix")
    
    plt.subplot(1, 3, 2)
    sns.heatmap(B, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Precision matrix")
    
    plt.subplot(1, 3, 3)
    # representing B in heatmap format
    sns.heatmap(A, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Recall matrix")
    
    plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 509} id="3M-9b_Vl-yeK" executionInfo={"status": "ok", "timestamp": 1627827433517, "user_tz": -330, "elapsed": 2722, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="27c6d9b6-c2d0-4e94-c1c4-207f764c8495"
print('Train confusion_matrix')
plot_confusion_matrix(y_train,y_train_pred)
print('Test confusion_matrix')
plot_confusion_matrix(y_test,y_test_pred)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 295} id="nQ8VBbF0-5PE" executionInfo={"status": "ok", "timestamp": 1627827433519, "user_tz": -330, "elapsed": 21, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d93392a4-1887-402f-8344-06e05f2e1b72"
fpr,tpr,ths = roc_curve(y_test,y_test_pred)
auc_sc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='navy',label='ROC curve (area = %0.2f)' % auc_sc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic with test data')
plt.legend()
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 730} id="Dos-qlvE-61i" executionInfo={"status": "ok", "timestamp": 1627827433521, "user_tz": -330, "elapsed": 19, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fc3fd895-8a8e-40f1-9340-d04ad9fd2741"
features = df_final_train.columns
importances = clf.feature_importances_
indices = (np.argsort(importances))[-25:]
plt.figure(figsize=(10,12))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='r', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
```

<!-- #region id="GkE-qvCy_C3O" -->
## XGBoost
<!-- #endregion -->

```python id="-24c0cLi_EOv"
clf = xgb.XGBClassifier()
param_dist = {"n_estimators":sp_randint(105,125),
              "max_depth": sp_randint(10,15)
              }
model = RandomizedSearchCV(clf, param_distributions=param_dist,
                           n_iter=5,cv=3,scoring='f1',random_state=25,
                           verbose=1, n_jobs=-1)

model.fit(df_final_train,y_train)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Sg-UsCq3JoZb" executionInfo={"status": "ok", "timestamp": 1627828686682, "user_tz": -330, "elapsed": 460, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="08d96409-fd08-4d66-fa3e-2521d03b356d"
print('mean test scores',model.cv_results_['mean_test_score'])
```

```python colab={"base_uri": "https://localhost:8080/"} id="-2qaeP2B_HWi" executionInfo={"status": "ok", "timestamp": 1627828688762, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="74e15468-3782-4796-e36c-2ab4c7b74610"
print(model.best_estimator_)
```

```python id="1a9HlbQM_HUg" executionInfo={"status": "ok", "timestamp": 1627828710980, "user_tz": -330, "elapsed": 615, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
clf = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=10, min_child_weight=1, missing=None, n_estimators=109,
       n_jobs=-1, nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)
```

```python id="DXxvodrm_HRk" executionInfo={"status": "ok", "timestamp": 1627828816569, "user_tz": -330, "elapsed": 102209, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
clf.fit(df_final_train,y_train)
y_train_pred = clf.predict(df_final_train)
y_test_pred = clf.predict(df_final_test)
```

```python colab={"base_uri": "https://localhost:8080/"} id="jEkZxUZe_MAQ" executionInfo={"status": "ok", "timestamp": 1627828818528, "user_tz": -330, "elapsed": 29, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="40e54b3d-2e5d-4502-ea8c-8e2c4c5f6539"
print('Train f1 score',f1_score(y_train,y_train_pred))
print('Test f1 score',f1_score(y_test,y_test_pred))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 509} id="EVtWdI2w_L-N" executionInfo={"status": "ok", "timestamp": 1627828822076, "user_tz": -330, "elapsed": 3568, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ada6e9dc-0de0-410d-a4b0-601f82150c7c"
print('Train confusion_matrix')
plot_confusion_matrix(y_train,y_train_pred)
print('Test confusion_matrix')
plot_confusion_matrix(y_test,y_test_pred)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 295} id="E5eteCGd_L7k" executionInfo={"status": "ok", "timestamp": 1627828822081, "user_tz": -330, "elapsed": 26, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="607b92fc-0e12-4411-941b-36c6a290b784"
fpr,tpr,ths = roc_curve(y_test,y_test_pred)
auc_sc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='navy',label='ROC curve (area = %0.2f)' % auc_sc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic with test data')
plt.legend()
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 730} id="lxtWi-iJ_Q6f" executionInfo={"status": "ok", "timestamp": 1627828822083, "user_tz": -330, "elapsed": 23, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="89d6142d-9166-498f-c628-8af67221198c"
features = df_final_train.columns
importances = clf.feature_importances_
indices = (np.argsort(importances))[-25:]
plt.figure(figsize=(10,12))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='r', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
```

<!-- #region id="sTyHrRDg_X6q" -->
## Model comparison
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="3D7yIIAx_Z03" executionInfo={"status": "ok", "timestamp": 1627829010236, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c9dea87b-c7b5-4c00-c541-96f6839e88b0"
x = PrettyTable()
x.field_names = ["Model", "n_estimators", "max_depth", "Train f1-Score","Test f1-Score"]
x.add_row(['Random Forest','1','14','0.964','0.924'])
x.add_row(['XGBOOST','109','10','0.992','0.926'])
print(x)
```
