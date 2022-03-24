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

```python id="Eei0NuRj35QP" executionInfo={"status": "ok", "timestamp": 1628782248441, "user_tz": -330, "elapsed": 33, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-sor"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python colab={"base_uri": "https://localhost:8080/"} id="UM90qwuS4K-k" executionInfo={"status": "ok", "timestamp": 1628782267348, "user_tz": -330, "elapsed": 7462, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9b189387-88c2-4f91-dbf9-711cb11a2487"
if not os.path.exists(project_path):
    !cp /content/drive/MyDrive/mykeys.py /content
    import mykeys
    !rm /content/mykeys.py
    path = "/content/" + project_name; 
    !mkdir "{path}"
    %cd "{path}"
    import sys; sys.path.append(path)
    !git config --global user.email "recotut@recohut.com"
    !git config --global user.name  "reco-tut"
    !git init
    !git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout main
else:
    %cd "{project_path}"
```

```python colab={"base_uri": "https://localhost:8080/"} id="TE_AZh_X4K-p" executionInfo={"status": "ok", "timestamp": 1628783661542, "user_tz": -330, "elapsed": 819, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4065f183-e0cb-459e-c103-2a57a151d76e"
!git status
```

```python colab={"base_uri": "https://localhost:8080/"} id="sd_n1hSi4K-q" executionInfo={"status": "ok", "timestamp": 1628783662522, "user_tz": -330, "elapsed": 990, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1fff4494-77dd-4d90-a172-253855e19afb"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

<!-- #region id="p-GuT3uF4O9n" -->
---
<!-- #endregion -->

<!-- #region id="YuolhUPv-xT-" -->
In this notebook we will be building XGB model and check if the reccomendation engine can be improved by using other algorithms
<!-- #endregion -->

```python id="ACJkzQkQ-xUF" executionInfo={"status": "ok", "timestamp": 1628783286753, "user_tz": -330, "elapsed": 757, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import confusion_matrix,  accuracy_score
from sklearn.metrics import f1_score, recall_score, precision_score
import itertools

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
```

<!-- #region id="gwKXlPYP-xUK" -->
## Creating alternative model

In this step dataset preprocessed in previous step is loaded and simple baseline model is tested.

Each line in a dataset contains data about one user and his final action on the offer. 
Either offer has been ignored, viewed or completed (offer proved to be interesting to a customer).
<!-- #endregion -->

```python id="6Zwf6Afz-xUL" executionInfo={"status": "ok", "timestamp": 1628782284989, "user_tz": -330, "elapsed": 1263, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
df = pd.read_csv('./data/silver/userdata.csv')
```

```python id="lBbfFHil-xUM" colab={"base_uri": "https://localhost:8080/", "height": 224} executionInfo={"status": "ok", "timestamp": 1628782284992, "user_tz": -330, "elapsed": 18, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d914bb25-2501-4571-cd94-7c8b8f4796bf"
df.head()
```

```python id="dn6maXYA-xUO" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628782289974, "user_tz": -330, "elapsed": 716, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6fa4ab36-fdc7-47e7-f833-4f221bbff773"
print("Dataset contains %s actions" % len(df))
```

<!-- #region id="OISYQqVH-xUQ" -->
### Let's plot the actions for one user

From the output can be seen that user completed an offer `0b1e...` and viewed `ae26...`. Offer `2906..` had been ignored twice.
<!-- #endregion -->

```python id="Ak2AIa-g-xUT" colab={"base_uri": "https://localhost:8080/", "height": 193} executionInfo={"status": "ok", "timestamp": 1628782293194, "user_tz": -330, "elapsed": 859, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6105ee51-c808-4991-bf35-86b14a726a3d"
df[df.id == 'e12aeaf2d47d42479ea1c4ac3d8286c6' ]
```

<!-- #region id="wkCYkzsu-xUX" -->
### Preparing data for training
Let's create user-offer matrix by encoding each id into categorical value.
<!-- #endregion -->

<!-- #region id="b_I7Rv1H-xUY" -->
Recommendation matrix is very similar to embeddings. So we will leverage this and will train embedding along the model.
<!-- #endregion -->

<!-- #region id="fNR1WBBC-xUY" -->
### Create additional user and offer details tensors
<!-- #endregion -->

```python id="3UzfJk_u-xUZ" executionInfo={"status": "ok", "timestamp": 1628782293198, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
offer_specs = ['difficulty', 'duration', 'reward', 'web',
       'mobile', 'social', 'bogo', 'discount', 'informational']
user_specs = ['age', 'became_member_on', 'gender', 'income', 'memberdays']
```

```python id="MMTI-Fiw-xUd" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628782295948, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6d1dd241-48c2-4f7b-d5da-abd25aa5b5de"
N_train = int(0.8 * len(df['event']))
N_test = 1000

train_df = df[:N_train]
test_df = df[N_train:]
print(len(train_df))
print(len(test_df))
```

```python id="kcBTZH2n-xUe" executionInfo={"status": "ok", "timestamp": 1628782296811, "user_tz": -330, "elapsed": 871, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def random_forest(train_data, train_true, test_data, test_true):
    #hyper-paramater tuning
    values = [25, 50, 100, 200]
    clf = RandomForestClassifier(n_jobs = -1)
    hyper_parameter = {"n_estimators": values}
    best_parameter = GridSearchCV(clf, hyper_parameter, scoring = "neg_mean_absolute_error", cv = 3)
    best_parameter.fit(train_data, train_true)
    estimators = best_parameter.best_params_["n_estimators"]
    print("Best RF parameter is: ", estimators)
    #applying random forest with best hyper-parameter
    clf = RandomForestClassifier(n_estimators = estimators, n_jobs = -1)
    clf.fit(train_data, train_true)
    #train_pred = clf.predict(train_data)
    return clf

def xgboost_model(train_data, train_true, test_data, test_true):
    #hyper-parameter tuning
    hyper_parameter = {"max_depth":[6, 8, 10, 16], "n_estimators":[60, 80, 100, 120]}
    clf = xgb.XGBClassifier()
    best_parameter = GridSearchCV(clf, hyper_parameter, scoring = "neg_mean_absolute_error", cv = 3)
    best_parameter.fit(train_data, train_true)
    estimators = best_parameter.best_params_["n_estimators"]
    depth = best_parameter.best_params_["max_depth"]
    print("Best XGB parameter is %s estimators and depth %s: " % (estimators, depth))
    clf = xgb.XGBClassifier(max_depth = depth, n_estimators = estimators)
    clf.fit(train_data, train_true)
    #train_pred = clf.predict(train_data)
    return clf
```

```python id="OORDUUNb-xUf" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628783261121, "user_tz": -330, "elapsed": 964316, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e6fddc4b-10a5-44c4-96ca-70feef6c7794"
pred_rf = random_forest(train_df[user_specs + offer_specs], 
                       train_df['event'].values.ravel(), 
                       test_df[user_specs + offer_specs],
                       test_df['event'].values.ravel())

pred_xgb = xgboost_model(train_df[user_specs + offer_specs], 
                       train_df['event'].values.ravel(), 
                       test_df[user_specs + offer_specs],
                        test_df['event'].values.ravel())

# error_table_regressions = pd.DataFrame(columns = ["Model", "TrainMAPE(%)", "TrainMSE", "TestMAPE(%)", "TestMSE"])
# error_table_regressions = error_table_regressions.append(pd.DataFrame([["XGBoost Regressor", trainMAPE_xgb*100, trainMSE_xgb, testMAPE_xgb*100, testMSE_xgb]], columns = ["Model", "TrainMAPE(%)", "TrainMSE", "TestMAPE(%)", "TestMSE"]))
# error_table_regressions = error_table_regressions.append(pd.DataFrame([["Random Forest Regression", trainMAPE_rf*100, trainMSE_rf, testMAPE_rf*100, testMSE_rf]], columns = ["Model", "TrainMAPE(%)", "TrainMSE", "TestMAPE(%)", "TestMSE"]))

# error_table_regressions.reset_index(drop = True, inplace = True)
```

```python id="7gFq90qh_sc1" executionInfo={"status": "ok", "timestamp": 1628783265894, "user_tz": -330, "elapsed": 25, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def random_forest(train_data, train_true, test_data, test_true):
    clf = RandomForestClassifier(n_estimators = 60, n_jobs = -1)
    clf.fit(train_data, train_true)
    #train_pred = clf.predict(train_data)
    return clf

def xgboost_model(train_data, train_true, test_data, test_true):
    #hyper-parameter tuning
    clf = xgb.XGBClassifier(max_depth = 16, n_estimators = 6)
    clf.fit(train_data, train_true)
    #train_pred = clf.predict(train_data)
    return clf
```

```python id="5lw2yAYl-xUg" executionInfo={"status": "ok", "timestamp": 1628783275543, "user_tz": -330, "elapsed": 9673, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
pred_rf = random_forest(train_df[user_specs + offer_specs], 
                       train_df['event'].values.ravel(), 
                       test_df[user_specs + offer_specs],
                       test_df['event'].values.ravel())

pred_xgb = xgboost_model(train_df[user_specs + offer_specs], 
                       train_df['event'].values.ravel(), 
                       test_df[user_specs + offer_specs],
                        test_df['event'].values.ravel())

# error_table_regressions = pd.DataFrame(columns = ["Model", "TrainMAPE(%)", "TrainMSE", "TestMAPE(%)", "TestMSE"])
# error_table_regressions = error_table_regressions.append(pd.DataFrame([["XGBoost Regressor", trainMAPE_xgb*100, trainMSE_xgb, testMAPE_xgb*100, testMSE_xgb]], columns = ["Model", "TrainMAPE(%)", "TrainMSE", "TestMAPE(%)", "TestMSE"]))
# error_table_regressions = error_table_regressions.append(pd.DataFrame([["Random Forest Regression", trainMAPE_rf*100, trainMSE_rf, testMAPE_rf*100, testMSE_rf]], columns = ["Model", "TrainMAPE(%)", "TrainMSE", "TestMAPE(%)", "TestMSE"]))

# error_table_regressions.reset_index(drop = True, inplace = True)
```

```python id="c4aQ0L_I-xUl" executionInfo={"status": "ok", "timestamp": 1628783275545, "user_tz": -330, "elapsed": 139, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          save=False,
                          figname='cm.png'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save:
        plt.savefig(figname, dpi=fig.dpi)
    plt.show()
```

```python id="UJYxzAlS-xUn" colab={"base_uri": "https://localhost:8080/", "height": 379} executionInfo={"status": "ok", "timestamp": 1628783275552, "user_tz": -330, "elapsed": 144, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="80537f83-f718-4257-ce20-02b48466c7a2"
pred1 = pred_rf.predict(test_df[0:1000][user_specs + offer_specs])
test_y = test_df[0:1000]['event'].values.ravel()
#print(pred1)
#print(test_y)
cm = confusion_matrix(test_y, pred1)
classes = [0,1,2]
plot_confusion_matrix(cm, classes, save=True, figname='./outputs/Recommendation-cm.png')
```

```python id="FKFzYE8J-xUo" colab={"base_uri": "https://localhost:8080/", "height": 379} executionInfo={"status": "ok", "timestamp": 1628783275554, "user_tz": -330, "elapsed": 112, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b64b9007-0ceb-451f-9bab-7b82f1599c29"
pred2 = pred_xgb.predict(test_df[0:1000][user_specs + offer_specs])
test_y = test_df[0:1000]['event'].values.ravel()
#print(pred2)
#print(test_y)
cm = confusion_matrix(test_y, pred2)
classes = [0,1,2]
plot_confusion_matrix(cm, classes, save=True, figname='./outputs/RecommendationXGB-cm.png')
```

```python id="cvVbz84J-xUp" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628783275557, "user_tz": -330, "elapsed": 103, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b873fff1-700f-4fc3-dca3-ee7d94e63927"
print("Accuracy for RF model: " + str(100*accuracy_score(test_y, pred1))+ "%" )
print("Accuracy for XGB model: " + str(100*accuracy_score(test_y, pred2))+ "%" )
```

```python id="vFHjwplV-xUr" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628783275560, "user_tz": -330, "elapsed": 99, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="59f6f2bc-656b-493c-c174-e3280ff30bb3"
print("F1 score for RF model: " + str(f1_score(test_y, pred1, average='weighted')))
print("Recall score for RF model: " + str(recall_score(test_y, pred1, average='weighted')))
print("Precision score for RF model: " + str(precision_score(test_y, pred1, average='weighted')))

print("")
print("F1 score for XGB model: " + str(f1_score(test_y, pred2, average='weighted')) )
print("Recall score for XGB model: " + str(recall_score(test_y, pred2, average='weighted')) )
print("Precision score for XGB model: " + str(precision_score(test_y, pred2, average='weighted')) )
```

<!-- #region id="Zv8Zs8Gc-xUs" -->
Results seem to be promising.
Let's try to improve them even more, and simplify data as from the correlation matrix it can be noticed that model has difficulties to differentiate if user will view an offer or even respond to it.
This can be due to the fact that responding to an offer implies that user had definitely viewed an offer.
<!-- #endregion -->

<!-- #region id="KlVgDdKg-xUt" -->
## Approach 2. Remove outlier fields
<!-- #endregion -->

```python id="15tr-Yq2-xUt" executionInfo={"status": "ok", "timestamp": 1628783446884, "user_tz": -330, "elapsed": 1121, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
df = pd.read_csv('./data/silver/userdata.csv')
```

```python id="5TBd1sWu-xUt" executionInfo={"status": "ok", "timestamp": 1628783449485, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
df['member_days'] = pd.to_datetime(df['became_member_on'], format="%Y%m%d")
df['member_days'] = df['member_days'] - df['member_days'].min()
df['member_days'] = df['member_days'].apply(lambda x: int(x.days))
```

<!-- #region id="TC_A-9Jp-xUu" -->
Let's check once again the correlation between gender and event response.
We are interested in X and O genders. Where X is the customers with anonymized data.
<!-- #endregion -->

```python id="8bGCm1bd-xUu" colab={"base_uri": "https://localhost:8080/", "height": 282} executionInfo={"status": "ok", "timestamp": 1628783455913, "user_tz": -330, "elapsed": 1922, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fcefbf3d-8496-49ed-cc5e-db06072eb52d"
df[df.gender == 0]['event'].plot.hist()#.count_values()
```

```python id="G93uW4Gp-xUv" colab={"base_uri": "https://localhost:8080/", "height": 282} executionInfo={"status": "ok", "timestamp": 1628783457715, "user_tz": -330, "elapsed": 1824, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="537ad756-bdb7-4d96-daeb-4183a43258f5"
df[df.gender == 1]['event'].plot.hist()#.count_values()
```

```python id="Z82J5UdX-xUv" colab={"base_uri": "https://localhost:8080/", "height": 282} executionInfo={"status": "ok", "timestamp": 1628783457719, "user_tz": -330, "elapsed": 83, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8c754a84-36af-406b-8c41-099926630e34"
df[df.gender == 2]['event'].plot.hist()#.count_values()
```

```python id="mJTsABBg-xUv" colab={"base_uri": "https://localhost:8080/", "height": 283} executionInfo={"status": "ok", "timestamp": 1628783457721, "user_tz": -330, "elapsed": 69, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8b6d441f-91e4-4ea4-beea-6e9794f4b2e0"
df[df.gender == 3]['event'].plot.hist()#.count_values()
```

```python id="BV1SV0_O-xUw" colab={"base_uri": "https://localhost:8080/", "height": 282} executionInfo={"status": "ok", "timestamp": 1628783457730, "user_tz": -330, "elapsed": 67, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3c303d9a-29c2-4801-b80d-6cf1b2623ebe"
df[df.income == 0]['event'].plot.hist()#.count_values()
```

<!-- #region id="id7BAs87-xUw" -->
Now we test the model performance with removing rows where user with age and income as None
They seem to view offer but rarely respond to it.
<!-- #endregion -->

```python id="r6gee6Dj-xUx" executionInfo={"status": "ok", "timestamp": 1628783482968, "user_tz": -330, "elapsed": 1372, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# We remove them by index as it seems to be the easiest way
indexes_to_drop = list(df[df.gender == 0].index) + list(df[df.income == 0].index)
df = df.drop(df.index[indexes_to_drop]).reset_index()
```

```python id="DxHQl0LR-xUx" executionInfo={"status": "ok", "timestamp": 1628783486241, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
df = df.reset_index()
```

```python id="Cp_xIDML-xUx" colab={"base_uri": "https://localhost:8080/", "height": 379} executionInfo={"status": "ok", "timestamp": 1628783533236, "user_tz": -330, "elapsed": 789, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cba576bc-1722-4274-8097-89c127a81ee1"
df['became_member_date'] = pd.to_datetime(df['became_member_on'], format="%Y%m%d")
df[df['member_days'] == 10]
```

<!-- #region id="EjPDHTi6-xUx" -->
Let's encode `event` field to be only binary value, with event ignored as 0, and offer completed - as 1.
<!-- #endregion -->

```python id="tCW3CGcD-xUy" executionInfo={"status": "ok", "timestamp": 1628783540177, "user_tz": -330, "elapsed": 1857, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
df['event'] = df['event'].map({0:0, 1:0, 2:1})
```

```python id="YDgFopZG-xUy" executionInfo={"status": "ok", "timestamp": 1628783540180, "user_tz": -330, "elapsed": 19, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
offer_specs = ['difficulty', 'duration', 'reward', 'web', 'email',
       'mobile', 'social', 'bogo', 'discount', 'informational']
user_specs = ['age', 'member_days', 'gender', 'income']
```

```python id="yNKN1jAE-xUy" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628783540181, "user_tz": -330, "elapsed": 19, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6d8f8fb9-eb81-43c4-c968-2b0da32dab32"
N_train = int(0.8 * len(df['event']))
N_test = 1000

train_df = df[:N_train]
test_df = df[N_train:]
print(len(train_df))
print(len(test_df))
```

```python id="m9HFFC1s-xU0" executionInfo={"status": "ok", "timestamp": 1628783542460, "user_tz": -330, "elapsed": 2294, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
pred_rf = random_forest(train_df[user_specs + offer_specs], 
                       train_df['event'].values.ravel(), 
                       test_df[user_specs + offer_specs],
                       test_df['event'].values.ravel())

pred_xgb = xgboost_model(train_df[user_specs + offer_specs], 
                       train_df['event'].values.ravel(), 
                       test_df[user_specs + offer_specs],
                        test_df['event'].values.ravel())
```

```python id="XT2M2yDr-xU1" colab={"base_uri": "https://localhost:8080/", "height": 413} executionInfo={"status": "ok", "timestamp": 1628783545623, "user_tz": -330, "elapsed": 41, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c1e4c7b1-4054-4594-979d-0492f5edaf58"
pred1 = pred_rf.predict(test_df[user_specs + offer_specs])
test_y = test_df['event'].values.ravel()
print(pred1)
print(test_y)

print("Accuracy for RF model: " + str(100*accuracy_score(test_y, pred1))+ "%" )
cm = confusion_matrix(test_y, pred1)
classes = [0,1,2]
plot_confusion_matrix(cm, classes)
```

```python id="qLUE1dxx-xU2" colab={"base_uri": "https://localhost:8080/", "height": 413} executionInfo={"status": "ok", "timestamp": 1628783545625, "user_tz": -330, "elapsed": 32, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f6793b35-5a25-4458-9f4a-b363d424c089"
pred2 = pred_xgb.predict(test_df[user_specs + offer_specs])
test_y = test_df['event'].values.ravel()
print(pred2)
print(test_y)

print("Accuracy for XGB model: " + str(100*accuracy_score(test_y, pred2))+ "%" )
cm = confusion_matrix(test_y, pred2)
classes = [0,1,2]
plot_confusion_matrix(cm, classes)
```

<!-- #region id="BN2bB0KH-xU4" -->
It seem that results are the same.
Let's try the model with encoding now 
an `event` field to be only binary value, with event ignored as 0, and offer completed - as 1.
<!-- #endregion -->

<!-- #region id="ty_yc2LZ-xU5" -->
## Approach 3. Building Performance optimized model
<!-- #endregion -->

```python id="AcSVz6kd-xU6" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628783626351, "user_tz": -330, "elapsed": 1387, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7888b3d8-72a6-4c1d-b078-ac0351894cfa"
df = pd.read_csv('./data/silver/userdata.csv')

df['member_days'] = pd.to_datetime(df['became_member_on'], format="%Y%m%d")
df['member_days'] = df['member_days'] - df['member_days'].min()
df['member_days'] = df['member_days'].apply(lambda x: int(x.days))

df['event'] = df['event'].map({0:0, 1:1, 2:1})

df = df.reset_index()

offer_specs = ['difficulty', 'duration', 'reward', 'web', 'email',
       'mobile', 'social', 'bogo', 'discount', 'informational']
user_specs = ['age', 'member_days', 'gender', 'income']

N_train = int(0.8 * len(df['event']))
N_test = 1000

train_df = df[:N_train]
test_df = df[N_train:]
print(len(train_df))
print(len(test_df))

def random_forest(train_data, train_true, test_data, test_true):
   
    clf = RandomForestClassifier(n_estimators = 100, n_jobs = -1)
    clf.fit(train_data, train_true)
    #train_pred = clf.predict(train_data)
    
    return clf

def xgboost_model(train_data, train_true, test_data, test_true):
    
    clf = xgb.XGBClassifier(max_depth = 16, n_estimators = 60)
    clf.fit(train_data, train_true)
    #train_pred = clf.predict(train_data)
    
    return clf
```

```python id="w5HEiXZQ-xU7" executionInfo={"status": "ok", "timestamp": 1628783642166, "user_tz": -330, "elapsed": 15825, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
pred_rf = random_forest(train_df[user_specs + offer_specs], 
                       train_df['event'].values.ravel(), 
                       test_df[user_specs + offer_specs],
                       test_df['event'].values.ravel())

pred_xgb = xgboost_model(train_df[user_specs + offer_specs], 
                       train_df['event'].values.ravel(), 
                       test_df[user_specs + offer_specs],
                        test_df['event'].values.ravel())
```

```python id="Yc7N2gnb-xU8" colab={"base_uri": "https://localhost:8080/", "height": 379} executionInfo={"status": "ok", "timestamp": 1628783642168, "user_tz": -330, "elapsed": 72, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5d3d8d51-40bf-40ad-d60a-7ae47982e034"
pred1 = pred_rf.predict(test_df[0:1000][user_specs + offer_specs])
test_y = test_df[0:1000]['event'].values.ravel()
#print(pred1)
#print(test_y)

print("Accuracy for RF model: " + str(100*accuracy_score(test_y, pred1))+ "%" )
cm = confusion_matrix(test_y, pred1)
classes = [0,1,2]
plot_confusion_matrix(cm, classes, save=True, figname='./outputs/RF-model-cm.png')
```

```python id="nmaGuucz-xU8" colab={"base_uri": "https://localhost:8080/", "height": 379} executionInfo={"status": "ok", "timestamp": 1628783642170, "user_tz": -330, "elapsed": 65, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c0274d5b-02ec-4823-8a95-1998e3b70c9e"
pred2 = pred_xgb.predict(test_df[0:1000][user_specs + offer_specs])
test_y = test_df[0:1000]['event'].values.ravel()
#print(pred2)
#print(test_y)

print("Accuracy for XGB model: " + str(100*accuracy_score(test_y, pred2))+ "%" )
cm = confusion_matrix(test_y, pred2)
classes = [0,1,2]
plot_confusion_matrix(cm, classes, save=True, figname='./outputs/XGB-model-cm.png')
```

<!-- #region id="aFYfyyCC-xU9" -->
This looks like a significant improve that can be used in production to save costs and send offers to those users who are going to be interested in companies offers without ignoring them.
<!-- #endregion -->

```python id="pEainqtS-xU9" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628783642172, "user_tz": -330, "elapsed": 49, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ed08bf59-256b-4663-a366-634d96313a3d"
print("F1 score for RF model: " + str(f1_score(test_y, pred1, average='weighted')))
print("Recall score for RF model: " + str(recall_score(test_y, pred1, average='weighted')))
print("Precision score for RF model: " + str(precision_score(test_y, pred1, average='weighted')))

print("")
print("F1 score for XGB model: " + str(f1_score(test_y, pred2, average='weighted')) )
print("Recall score for XGB model: " + str(recall_score(test_y, pred2, average='weighted')) )
print("Precision score for XGB model: " + str(precision_score(test_y, pred2, average='weighted')) )
```

<!-- #region id="RxPep6_A-xU9" -->
This proves to be a very good model for ad hoc predictions and predictions on subsections of customer by regions or cities.
<!-- #endregion -->
