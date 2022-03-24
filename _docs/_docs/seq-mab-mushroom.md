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

<!-- #region id="i_1jvK-oGLjU" -->
# Sequential Batch Learning in Stochastic MAB and Contextual MAB on Mushroom and Synthetic data
<!-- #endregion -->

<!-- #region id="bLatuDpMGLge" -->
## Executive summary

| | |
| --- | --- |
| Problem | Learning user preferences online might have an impact of delay and training recommender system sequentially for every example is computationally heavy. |
| Hypothesis | A learning agent observes responses batched in groups over a certain time period. The impact of batch learning can be measured in terms of online behavior. |
| Prblm Stmt. | Given a finite set of arms ⁍, an environment ⁍ (⁍ is the distribution of rewards for action ⁍), and a time horizon ⁍, at each time step ⁍, the agent chooses an action ⁍ and receives a reward ⁍. The goal of the agent is to maximize the total reward ⁍. |
| Solution | Sequential batch learning is a more generalized way of learning which covers both offline and online settings as special cases bringing together their advantages. Unlike offline learning, sequential batch learning retains the sequential nature of the problem. Unlike online learning, it is often appealing to implement batch learning in large scale bandit problems. In this setting, responses are grouped in batches and observed by the agent only at the end of each batch. |
| Dataset | Mushroom, Synthetic |
| Preprocessing | Train/test split, label encoding |
| Metrics | Conversion rate, regret |
| Credits | [Danil Provodin](https://github.com/danilprov) |
<!-- #endregion -->

<!-- #region id="y-pKQhr9GaAv" -->
### Environments

| Name | Type | Rewards |
| --- | --- | --- |
| env1 | 2-arm environment | [0.7, 0.5] |
| env2 | 2-arm environment | [0.7, 0.4] |
| env3 | 2-arm environment | [0.7, 0.1] |
| env4 | 4-arm environment | [0.35, 0.18, 0.47, 0.61] |
| env5 | 4-arm environment | [0.40, 0.75, 0.57, 0.49] |
| env6 | 4-arm environment | [0.70, 0.50, 0.30, 0.10] |
<!-- #endregion -->

<!-- #region id="QPyNX42KGiaR" -->
### Simulation

| Application | Policy |
| --- | --- |
| Multi-armed bandit (MAB) | Thompson Sampling (TS) |
| Multi-armed bandit (MAB) | Upper Confidence Bound (UCB) |
| Contextual MAB (CMAB) | Linear Thompson Sampling (LinTS) |
| Contextual MAB (CMAB) | Linear UCB (LinUCB) |
<!-- #endregion -->

<!-- #region id="4veQdUjFGCwk" -->
## Process flow
<!-- #endregion -->

<!-- #region id="Pm8qSJsaGEUN" -->
![](https://github.com/RecoHut-Stanzas/S873634/raw/main/images/process_flow.svg)
<!-- #endregion -->

<!-- #region id="VAwEfoPurXGo" -->
## Setup
<!-- #endregion -->

<!-- #region id="ngwffbD4rXE3" -->
### Imports
<!-- #endregion -->

```python id="vSHj786VrXC8"
import pandas as pd
import numpy as np
from numpy.linalg import inv
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate
from sklearn.tree import DecisionTreeClassifier
from scipy.optimize import minimize
from lightgbm import LGBMClassifier
from scipy.stats import beta
import pickle
import os
import shutil

import tqdm
from tqdm.notebook import tqdm
from multiprocessing.dummy import Pool
from IPython.display import clear_output
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from __future__ import print_function
from abc import ABCMeta, abstractmethod
```

<!-- #region id="XXMnN2vioutk" -->
## Data
<!-- #endregion -->

<!-- #region id="W_L8uOd0rfK6" -->
### Download
<!-- #endregion -->

<!-- #region id="-f9daEHdrG5E" -->
This data set includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family (pp. 500-525). Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous one. The Guide clearly states that there is no simple rule for determining the edibility of a mushroom; no rule like ``leaflets three, let it be'' for Poisonous Oak and Ivy. More details [here](https://archive.ics.uci.edu/ml/datasets/mushroom).
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="adGWyfxFrHij" executionInfo={"status": "ok", "timestamp": 1639145418029, "user_tz": -330, "elapsed": 1715, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="eb9c650f-63bc-4fcf-ef24-b1f698d71d4a"
!mkdir -p data
!cd data && wget -q --show-progress https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data
!cd data && wget -q --show-progress https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.names
```

<!-- #region id="Aal7wGRfrhfe" -->
### Preprocessing
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 278} id="TshNYFbQrhdi" executionInfo={"status": "ok", "timestamp": 1639145489020, "user_tz": -330, "elapsed": 516, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="88c714b1-d1c5-4666-8f70-9db8b6fe6584"
mushroom_data = pd.read_csv("data/agaricus-lepiota.data", header=None)

column_names = ["classes", "cap-shape", "cap-surface", "cap-color", "bruises?", "odor", "gill-attachment",
                "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
                "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", 
                "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"]
                
mushroom_data.columns = column_names
mushroom_data.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="sIbAiLjtrhbZ" executionInfo={"status": "ok", "timestamp": 1639145504479, "user_tz": -330, "elapsed": 518, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f0602325-8225-4047-b8a9-503031bba8c4"
mushroom_data.dtypes
```

```python id="I420evWNrhZQ"
# label encoding
for column in column_names:
    mushroom_data[column] = mushroom_data[column].astype('category')
    mushroom_data[column] = mushroom_data[column].cat.codes

# split
idx_trn, idx_tst = train_test_split(mushroom_data.index, test_size=0.2, random_state=42, 
                                    stratify=mushroom_data[['classes']])
```

<!-- #region id="c6AARTV0sSEw" -->
gini by factors
<!-- #endregion -->

```python id="hjgbgx7NrhWv"
def gini(var):
    df = mushroom_data.copy()
    x_trn = df.loc[idx_trn, var]
    y_trn = df.loc[idx_trn, 'classes']
    x_tst = df.loc[idx_tst, var]
    y_tst = df.loc[idx_tst, 'classes']
    
    if x_trn.dtype in ['O','object']:
        cats = pd.DataFrame({'x': x_trn, 'y': y_trn}).fillna('#NAN#').groupby('x').agg('mean').sort_values('y').index.values
        X_trn = pd.Categorical(x_trn.fillna('#NAN#'), categories=cats, ordered=True).codes.reshape(-1, 1)
        X_tst = pd.Categorical(x_tst.fillna('#NAN#'), categories=cats, ordered=True).codes.reshape(-1, 1)
    else:
        repl = min(x_trn.min(), x_tst.min())-1 if np.isfinite(min(x_trn.min(), x_tst.min())-1) else -999999
        #repl = x_trn.min()-1 if np.isfinite(x_trn.min())-1 else -999999
        X_trn = x_trn.fillna(repl).replace(np.inf, repl).replace(-np.inf, repl).values.reshape(-1, 1)
        X_tst = x_tst.fillna(repl).replace(np.inf, repl).replace(-np.inf, repl).values.reshape(-1, 1)
    
    obvious_gini_trn = 2*roc_auc_score(y_trn, X_trn)-1
    obvious_gini_tst = 2*roc_auc_score(y_tst, X_tst)-1

    if obvious_gini_trn < 0:
        obvious_gini_trn = -obvious_gini_trn
        obvious_gini_tst = -obvious_gini_tst

    parameters = {'min_samples_leaf':[0.01, 0.025, 0.05, 0.1]}
    dt = DecisionTreeClassifier(random_state=1)
    clf = GridSearchCV(dt, parameters, cv=4, scoring='roc_auc', n_jobs=10)
    clf.fit(X_trn, y_trn)

    true_gini_trn = 2*clf.best_score_-1
    true_gini_tst = 2*roc_auc_score(y_tst, clf.predict_proba(X_tst)[:, 1])-1

    if true_gini_trn < 0:
        true_gini_trn = -true_gini_trn
        true_gini_tst = -true_gini_tst

    if obvious_gini_trn > true_gini_trn:
        return [var, obvious_gini_trn, obvious_gini_tst]
    else:
        return [var, true_gini_trn, true_gini_tst]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["2b72cbdd08374942859625d2047795f2", "396e4164d325484e9956f597c2e133e0", "46ce32b9cbc74045b96fc2fbc93f0575", "596798eaeac14aaea8c3cc476b7f329f", "c60ea6ae22a2457786ca6b8e3446e3f2", "342e5fcfeeb04720834331a4274b5e28", "9cbda8d73a0a41f8acb9c3bdacdc7a9b", "042addbfa66d4dd4a8543124cc312905", "3e9ddf2ca50f4c24bcf276eeeee63d01", "ce6fa32eee4044df843b45c5693e065e", "5ac6a207b502473fa8bd11d2adcc1201"]} id="pwtreallsHVN" executionInfo={"status": "ok", "timestamp": 1639145595544, "user_tz": -330, "elapsed": 9639, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d6e113bd-e273-4922-f04d-a4d6448d26b2"
with Pool(20) as p:
    vars_gini = list(tqdm(p.imap(gini, column_names), total=len(column_names)))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 184} id="wqBNXxJVsLoh" executionInfo={"status": "ok", "timestamp": 1639145626752, "user_tz": -330, "elapsed": 527, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="98514987-c253-42d6-8c63-a7fb38cf9c5a"
vars_gini = pd.DataFrame(vars_gini)
vars_gini.set_index(0, inplace=True)
vars_gini.columns = ['gini_train', 'gini_test']

vars_gini.T
```

<!-- #region id="2CWLsNxLsVYI" -->
Correlation analysis
<!-- #endregion -->

```python id="U94p4ETPscxi"
vars_corrs = mushroom_data.loc[:, column_names].corr().abs().stack().reset_index().drop_duplicates()
vars_corrs = vars_corrs[vars_corrs.level_0!=vars_corrs.level_1]
vars_corrs.columns = ['var_1', 'var_2', 'correlation']
vars_corrs = vars_corrs.set_index(['var_1', 'var_2'], drop=True).sort_values(by='correlation', ascending=False)

vars_drop = []

for v in vars_corrs[vars_corrs.correlation > 0.7].index.values:
    if v[0] not in vars_drop and v[1] not in vars_drop:
        vars_drop.append(v[1] if vars_gini.loc[v[0], 'gini_train'] > vars_gini.loc[v[1], 'gini_train'] else v[0])
        
del v
```

<!-- #region id="q1MuoCp4sk8r" -->
Feature selection
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 153, "referenced_widgets": ["3d6f8baea53a484faab78b4c50fbf620", "9689c9d0d5354f7782671e9b301c5840", "4550c6895fa244efb6dd5cbc9dd98324", "b70c31ffb8784655ad0d4e59e12343d6", "c14978517f8047718a1c084947d702b0", "60d54986769b4e859de45bbce659b7f1", "0181512ceea0401db61675636ae223b5", "4b34f4a343b242abaef58ebe57e07a4d", "476d47eec56d4ff0bec2900ed3505ba2", "27609bb3e0594b5084e01fae4b04cc9c", "606916b9b63b43969a01e84ec142f341", "b8cc488f50764d7a80c565e120acf3bd", "73139f73ef7d4769807b6dc081a48fb1", "8c0a7a56a8d946bf953390fd61675c4d", "63ead4d72b27495cac3bb6a385895123", "a1d8d9e611534997828fa51cdab2bd0a", "afe8ad0a833b4f399a1f6d8367786fd3", "c55e9f4461c24c9f88ba2098ff957634", "4472bedf87ec41aeb3306d6364386540", "edc2a60460ef48bdb78972f403c14175", "d971324520f34ce5b3b6994ddd0d0102", "6d4282c72b80490cbdb027d3592849ea", "77454d4570db40ad9d90ed0f9d80f019", "efe993ec7ad34cfaac8ca10da8b06b25", "a845d33bdbc048af807e19eb89699a1b", "e16dcd3650f64c61b1c53445f96d13d5", "124e06ade51442c3af9d364455ec278f", "a87b20c55b6547d7bad171519b9c2029", "ce8d826cba9f41f8bb33997ceceab3c0", "3ad0f66ba72c47ce8537870c6af56c0c", "3f68a6b94c614e87aea69a1fecf11d2e", "3b3af273314945ef951076ac38deb74c", "ffd1831574034bc59136368ad3e60108", "d8e841629bac43fbbea7f0a924aa9b94", "dcde233fe4804bd8be4a35f7a781a985", "a31bcdbac3f54376814d4b9cd40d545b", "806b8ca3d7fe4e05b22b038a82eb4545", "a26e8dfe75d540208dae3d35a0444551", "9a8c2b7f24ad47dea0cc84b99639f151", "721da9071ffc493d97a523d2f2ac4f59", "0f5ac0bc528b4fa1b8d5a7aafeb36c79", "a37b2e9543074546bf43c7741f3a5650", "f95e707810f3426eb20134bad69e6a7f", "c065af81128d4149b26e71b7a6cb4c14", "153549b4beb84e54b5fe4bec2eecf075", "d7e2ff7ce49d44588f16e7abfbb9885c", "6a821f0913784d64b2e98282b6ae15a1", "f736e0b03dea46dea371a5abcbcde9d7", "f0f773f30026425fa0dfdcefd1f7d2dc", "914c417613a446c28921a6cb430218dc", "76e6aa6b457347af8ef3146d3d1e7437", "3d87aae69d6149fc98c0b0a642fd9194", "4a588c2572094039983c106ffd558d86", "346efff4305041a5bd4913a00ebae73a", "3a36af3ccfd54942b5e4643c373f60a2", "d1538f6f3b7641ccb48e2bbdef414389", "3810f04feb094355bdc304447a3356b5", "2528ae76b0a7423a8910adf401a5514a", "d8d6807eee1e44f78e10dcd5251c3e58", "21eac8d36aaf4fa2a0565329dedd965b", "411e0eeb49834c04bbca486de6d10b5d", "d3282d958c424f84a1b0e2f65dc6602b", "28e68ecc22cf450d8ac9b8067ea0f801", "310d810b45114471bbc5a6658de789df", "9c7bed0810f34fe5948110f9c4ad1894", "c8dc39d329de455ea39ec34406e6bd59"]} id="WzyBNZ9psmGi" executionInfo={"status": "ok", "timestamp": 1639145844301, "user_tz": -330, "elapsed": 68523, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c12cae13-09ca-4583-fa7d-1cb2fe20519f"
# all variables
vars0 = column_names[1:]

# drop values with gini less than 3%
vars1 = [v for v in vars0 if vars_gini.loc[v, 'gini_train'] >= 0.03]

# drop correlated variables
vars2 = [v for v in vars1 if v not in vars_drop]

i = 0

for var_lst in [vars0, vars1, vars2]:
    i += 1
    lgb = LGBMClassifier(max_depth=1, n_estimators=250, random_state=42, n_jobs=30)
    
    cv = cross_validate(lgb, mushroom_data.loc[:, var_lst], mushroom_data.loc[:, 'classes'], 
                        cv=5, scoring='roc_auc', n_jobs=20, return_train_score=True)
    
    lgb.fit(mushroom_data[var_lst], mushroom_data['classes'])
    
    print({'Variables': len(var_lst), 
           'Train CV': round(cv['train_score'].mean()*2-1, 4), 
           'Test CV': round(cv['test_score'].mean()*2-1, 4)})
    
var_lst_imp = pd.Series(dict(zip(var_lst, lgb.feature_importances_)))
var_lst = [i for i in var_lst_imp.index if var_lst_imp.loc[i]>0]
print({'exclude': [i for i in var_lst_imp.index if var_lst_imp.loc[i]<=0]})
print(len(var_lst))

forw_cols = []
current_ginis = pd.Series({'Train CV':0, 'Test CV':0})

def forw(x):
    lgb = LGBMClassifier(max_depth=1, n_estimators=250, random_state=42, n_jobs=1)
    cv = cross_validate(lgb, mushroom_data.loc[:, forw_cols+[x]], mushroom_data.loc[:, 'classes'],
                        cv=5, scoring='roc_auc', n_jobs=1, return_train_score=True)
    lgb.fit(mushroom_data.loc[:, forw_cols+[x]], mushroom_data.loc[:, 'classes'])
    return x, pd.Series({
        'Train CV': cv['train_score'].mean()*2-1,
        'Test CV': cv['test_score'].mean()*2-1
    })

forwards_log = []
while len(forw_cols)<30:
    with Pool(20) as p:
        res = list(tqdm(p.imap(forw, [i for i in var_lst if i not in forw_cols]), total=len(var_lst)-len(forw_cols), leave=False))
    res = pd.DataFrame({i[0]:i[1] for i in res}).T
    delta = res - current_ginis
    if delta['Test CV'].max()<0:
        break
    best_var = delta['Test CV'].idxmax()
    forw_cols = forw_cols + [best_var]
    current_ginis = res.loc[best_var]
    forwards_log.append(current_ginis)
    clear_output()
    print(pd.DataFrame(forwards_log))

clear_output()
forwards_log = pd.DataFrame(forwards_log)
forwards_log['Uplift Train CV'] = forwards_log['Train CV']-forwards_log['Train CV'].shift(1).fillna(0)
forwards_log['Uplift Test CV'] = forwards_log['Test CV']-forwards_log['Test CV'].shift(1).fillna(0)
print(forwards_log)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 238} id="jjPfCIG-sw2k" executionInfo={"status": "ok", "timestamp": 1639145844304, "user_tz": -330, "elapsed": 47, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9fa96409-314f-4dab-a148-c38601d0e10b"
ids_vars = forwards_log[forwards_log['Uplift Test CV']>0.001].index.values.tolist()
vars_gini.loc[ids_vars,:]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="vEXcrpMis7Pi" executionInfo={"status": "ok", "timestamp": 1639145844306, "user_tz": -330, "elapsed": 39, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="bcb3014b-6da3-4627-e4a8-9e224a253f2a"
mushroom_data_features = mushroom_data[ids_vars + ["classes"]]
mushroom_data = mushroom_data_features.loc[mushroom_data_features.index.repeat(4)].reset_index(drop=True)

mushroom_data["a"] = np.random.choice([0, 1], mushroom_data.shape[0])
mushroom_data["probs"] = 1
mushroom_data["y"] = 0

eat_edible = (1-mushroom_data["classes"]) * mushroom_data["a"] * 1
eat_poisonous = mushroom_data["classes"] * mushroom_data["a"] * np.random.choice([1, -1], mushroom_data.shape[0])
mushroom_data["y"] = eat_edible + eat_poisonous
new_names = ['X_' + str(i+1) for i in range(len(ids_vars))]    
mushroom_data = mushroom_data.rename(columns=dict(zip(ids_vars, new_names)))

mushroom_data_final = mushroom_data[new_names + ['a', 'y', 'probs']]
mushroom_data_final.head()
```

```python id="ZTavtFcks9iR"
with open('data/mushroom_data_final.pickle', 'wb') as handle:
    pickle.dump(mushroom_data_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

<!-- #region id="p4B8_mndtBlr" -->
## Utilities
<!-- #endregion -->

<!-- #region id="dHg46NSstX9e" -->
### Softmax
<!-- #endregion -->

```python id="utuZpkQktdjN"
def softmax(action_values, tau=1.0):
    """
    Args:
        action_values (Numpy array): A 2D array of shape (batch_size, num_actions).
                       The action-values computed by an action-value network.
        tau (float): The temperature parameter scalar.
    Returns:
        A 2D array of shape (batch_size, num_actions). Where each column is a probability distribution over
        the actions representing the policy.
    """

    # Compute the preferences by dividing the action-values by the temperature parameter tau
    preferences = action_values / tau
    # Compute the maximum preference across the actions
    max_preference = np.max(preferences, axis=1)

    # your code here

    # Reshape max_preference array which has shape [Batch,] to [Batch, 1]. This allows NumPy broadcasting
    # when subtracting the maximum preference from the preference of each action.
    reshaped_max_preference = max_preference.reshape((-1, 1))
    # print(reshaped_max_preference)

    # Compute the numerator, i.e., the exponential of the preference - the max preference.
    exp_preferences = np.exp(preferences - reshaped_max_preference)
    # print(exp_preferences)
    # Compute the denominator, i.e., the sum over the numerator along the actions axis.
    sum_of_exp_preferences = np.sum(exp_preferences, axis=1)
    # print(sum_of_exp_preferences)

    # your code here

    # Reshape sum_of_exp_preferences array which has shape [Batch,] to [Batch, 1] to  allow for NumPy broadcasting
    # when dividing the numerator by the denominator.
    reshaped_sum_of_exp_preferences = sum_of_exp_preferences.reshape((-1, 1))
    # print(reshaped_sum_of_exp_preferences)

    # Compute the action probabilities according to the equation in the previous cell.
    action_probs = exp_preferences / reshaped_sum_of_exp_preferences
    # print(action_probs)

    # your code here

    # squeeze() removes any singleton dimensions. It is used here because this function is used in the
    # agent policy when selecting an action (for which the batch dimension is 1.) As np.random.choice is used in
    # the agent policy and it expects 1D arrays, we need to remove this singleton batch dimension.
    action_probs = action_probs.squeeze()
    return action_probs
```

```python colab={"base_uri": "https://localhost:8080/"} id="Mbk-YyyHtfqc" executionInfo={"status": "ok", "timestamp": 1639145939697, "user_tz": -330, "elapsed": 477, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c0665afb-a0c1-460c-e4d7-7d9e20103146"
# if __name__ == '__main__':
#     rand_generator = np.random.RandomState(0)
#     action_values = rand_generator.normal(0, 1, (2, 4))
#     tau = 0.5

#     action_probs = softmax(action_values, tau)
#     print("action_probs", action_probs)

#     assert (np.allclose(action_probs, np.array([
#         [0.25849645, 0.01689625, 0.05374514, 0.67086216],
#         [0.84699852, 0.00286345, 0.13520063, 0.01493741]
#     ])))

#     action_values = np.array([[0.0327, 0.0127, 0.0688]])
#     tau = 1.
#     action_probs = softmax(action_values, tau)
#     print("action_probs", action_probs)

#     assert np.allclose(action_probs, np.array([0.3315, 0.3249, 0.3436]), atol=1e-04)

#     print("Passed the asserts! (Note: These are however limited in scope, additional testing is encouraged.)")
```

<!-- #region id="CsJuwC4nth4A" -->
### Replay buffer
<!-- #endregion -->

```python id="mPZIB5GOtm7D"
class ReplayBuffer:
    def __init__(self, size, seed):
        """
        Args:
            size (integer): The size of the replay buffer.
            minibatch_size (integer): The sample size.
            seed (integer): The seed for the random number generator.
        """
        self.buffer = []
        self.rand_generator = np.random.RandomState(seed)
        self.max_size = size

    def append(self, state, action, reward):
        """
        Args:
            state (Numpy array): The state.
            action (integer): The action.
            reward (float): The reward.
            terminal (integer): 1 if the next state is a terminal state and 0 otherwise.
            next_state (Numpy array): The next state.
        """
        if len(self.buffer) == self.max_size:
            del self.buffer[0]
        self.buffer.append([state, action, reward])

    def sample(self, last_action):
        """
        Returns:
            A list of transition tuples including state, action, reward, terinal, and next_state
        """
        state, action, reward = map(list, zip(*self.buffer))
        idxs = [elem == last_action for elem in action]
        X = [b for a, b in zip(idxs, state) if a]
        y = [b for a, b in zip(idxs, reward) if a]

        return X, y

    def size(self):
        return len(self.buffer)
```

```python colab={"base_uri": "https://localhost:8080/"} id="XCsLBOzDtozO" executionInfo={"status": "ok", "timestamp": 1639145971783, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fbd79fdd-1c19-4e74-ce5d-77d8b66a834c"
# if __name__ == "__main__":

#     buffer = ReplayBuffer(size=100000, seed=1)
#     buffer.append([1, 2, 3], 0, 1)
#     buffer.append([4, 21, 3], 1, 1)
#     buffer.append([0, 1, 1], 0, 0)

#     print(buffer.sample(0))
```

<!-- #region id="MGlV70x3tpu7" -->
### Data Generator
<!-- #endregion -->

```python id="2bxVP6RGtxDa"
def generate_samples(num_samples, num_features, num_arms, return_dataframe=False):
    np.random.seed(1)
    # generate pseudo features X and "true" arms' weights
    X = np.random.randint(0, 4, size=(num_samples, num_features))
    actions_weights = np.random.normal(loc=-1., scale=1, size=(num_arms, num_features))

    # apply data generating policy
    policy_weights = np.random.normal(size=(num_arms, num_features))
    action_scores = np.dot(X, policy_weights.T)
    action_probs = softmax(action_scores, tau=10)
    A = np.zeros((num_samples, 1))
    for i in range(num_samples):
        A[i, 0] = np.random.choice(range(num_arms), 1, p=action_probs[i, :])

    # store probabilities of choosing a particular action
    _rows = np.zeros_like(A, dtype=np.intp)
    _columns = A.astype(int)
    probs = action_probs[_rows, _columns]

    # calculate "true" outcomes Y
    ## broadcasting chosen actions to action weights
    matrix_multiplicator = actions_weights[_columns].squeeze()  # (num_samples x num_features) matrix
    rewards = np.sum(X * matrix_multiplicator, axis=1).reshape(-1, 1)
    Y = (np.sign(rewards) + 1) / 2

    if return_dataframe:
        column_names = ['X_' + str(i+1) for i in range(num_features)]
        X = pd.DataFrame(X, columns=column_names)
        A = pd.DataFrame(A, columns=['a'])
        Y = pd.DataFrame(Y, columns=['y'])
        probs = pd.DataFrame(probs, columns=['probs'])

        return pd.concat([X, A, Y, probs], axis=1)
    else:
        return X, A, Y, probs
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="UH5mKo3htyWH" executionInfo={"status": "ok", "timestamp": 1639146016409, "user_tz": -330, "elapsed": 4932, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fca14966-0540-4ed6-e390-3ad54fb80465"
# dataset = generate_samples(100000, 4, 3, True)
# dataset.head()
```

<!-- #region id="PYr5NIxatzVv" -->
### Data loader
<!-- #endregion -->

```python id="3sW2YJahuCbl"
def data_randomizer(pickle_file, seed=None):
    if isinstance(pickle_file, str):
        with open(pickle_file, 'rb') as f:
            dataset = pickle.load(f)
    else:
        dataset = pickle_file

    actions = sorted(dataset.iloc[:, -3].unique().tolist())
    tst_smpl = pd.DataFrame().reindex_like(dataset).dropna()
    ratio = 0.1

    for action in actions:
        action_subsample = dataset[dataset.iloc[:, -3] == action]
        action_drop, action_use = train_test_split(action_subsample.index, test_size=ratio,
                                                   random_state=seed,
                                                   stratify=action_subsample.iloc[:, -2])
        tst_smpl = pd.concat([tst_smpl,
                              action_subsample.loc[action_use]]).sample(frac=1, random_state=seed)

    tst_smpl = tst_smpl.reset_index(drop=True)

    del action_drop, action_use

    X = tst_smpl.iloc[:, :-3].to_numpy()
    A = tst_smpl.iloc[:, -3].to_numpy()
    Y = tst_smpl.iloc[:, -2].to_numpy()
    probs = tst_smpl.iloc[:, -1].to_numpy()

    return X, A, Y/probs
```

```python id="NcdcUSeruEUD"
class BanditDataset(Dataset):
    def __init__(self, pickle_file, seed=None):
        # load dataset
        X, A, Y = data_randomizer(pickle_file, seed)
        self.features = X
        self.actions = A
        self.rewards = Y

    def __len__(self):
        return len(self.rewards)

    def __getitem__(self, idx):
        feature_vec = self.features[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]

        return feature_vec, action, reward
```

```python colab={"base_uri": "https://localhost:8080/"} id="8v7s0-QIuGPR" executionInfo={"status": "ok", "timestamp": 1639146132864, "user_tz": -330, "elapsed": 5534, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="dada2526-7ef3-476c-e6f7-141d7f62e7c0"
# if __name__ == '__main__':
#     dir = 'data/mushroom_data_final.pickle'
#     data = data_randomizer(dir)

#     dataset = BanditDataset(pickle_file=dir, seed=1)
#     print(len(dataset))
#     print(dataset.__len__())
#     print(dataset[420])
#     print(dataset[421])
#     print(dataset[0])
#     print(dataset[1])

#     dl = DataLoader(dataset, batch_size=2, shuffle=True)

#     print(next(iter(dl)))

#     dataset = generate_samples(100000, 4, 3, True)
#     dataset = BanditDataset(pickle_file=dataset, seed=1)
#     print(len(dataset))
#     print(dataset.__len__())
#     print(dataset[420])
#     print(dataset[421])
#     print(dataset[0])
#     print(dataset[1])
```

<!-- #region id="IsZ9fTsiuPwc" -->
### Plot script
<!-- #endregion -->

```python id="Whuf3fm3uUdS"
def get_leveled_data(arr):
    """
    Args:
        arr: list of lists os different length
    Returns:
        average result over arr, axis=0
    """
    b = np.zeros([len(arr), len(max(arr, key=lambda x: len(x)))])
    b[:, :] = np.nan
    for i, j in enumerate(arr):
        b[i][0:len(j)] = j

    return b
```

```python id="eDcvasCOubnj"
def smooth(data, k):
    num_episodes = data.shape[1]
    num_runs = data.shape[0]

    smoothed_data = np.zeros((num_runs, num_episodes))

    for i in range(num_episodes):
        if i < k:
            smoothed_data[:, i] = np.mean(data[:, :i + 1], axis=1)
        else:
            smoothed_data[:, i] = np.mean(data[:, i - k:i + 1], axis=1)

    return smoothed_data
```

```python id="6DI-dp6fuawY"
def plot_result(result_batch, result_online, batch_size):
    plt_agent_sweeps = []
    num_steps = np.inf

    fig, ax = plt.subplots(figsize=(8, 6))

    for data, label in zip([result_batch, result_online], ['batch', 'online']):
        sum_reward_data = get_leveled_data(data)

        # smooth data
        smoothed_sum_reward = smooth(data=sum_reward_data, k=100)

        mean_smoothed_sum_reward = np.mean(smoothed_sum_reward, axis=0)

        if mean_smoothed_sum_reward.shape[0] < num_steps:
            num_steps = mean_smoothed_sum_reward.shape[0]

        plot_x_range = np.arange(0, mean_smoothed_sum_reward.shape[0])
        graph_current_agent_sum_reward, = ax.plot(plot_x_range, mean_smoothed_sum_reward[:],
                                                  label=label)
        plt_agent_sweeps.append(graph_current_agent_sum_reward)


    update_points = np.ceil(np.arange(num_steps) / batch_size).astype(int)
    ax.plot(plot_x_range, mean_smoothed_sum_reward[update_points], label='upper bound')

    ax.legend(handles=plt_agent_sweeps, fontsize=13)
    ax.set_title("Learning Curve", fontsize=15)
    ax.set_xlabel('Episodes', fontsize=14)
    ax.set_ylabel('reward', rotation=0, labelpad=40, fontsize=14)
    # ax.set_ylim([-300, 300])

    plt.tight_layout()
    plt.show()
```

<!-- #region id="ndaq3Ha6ucgA" -->
## Agents
<!-- #endregion -->

<!-- #region id="nK6ccVZfukKl" -->
### Base Agent
<!-- #endregion -->

<!-- #region id="5vxRMmXluqMk" -->
An abstract class that specifies the Agent API for RL-Glue-py.
<!-- #endregion -->

```python id="tIP4FINouoP1"
class BaseAgent:
    """Implements the agent for an RL-Glue environment.
    Note:
        agent_init, agent_start, agent_step, agent_end, agent_cleanup, and
        agent_message are required methods.
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts."""

    @abstractmethod
    def agent_start(self, observation):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            observation (Numpy array): the state observation from the environment's evn_start function.
        Returns:
            The first action the agent takes.
        """

    @abstractmethod
    def agent_step(self, reward, observation):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            observation (Numpy array): the state observation from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """

    @abstractmethod
    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the terminal state.
        """

    @abstractmethod
    def agent_cleanup(self):
        """Cleanup done after the agent ends."""

    @abstractmethod
    def agent_message(self, message):
        """A function used to pass information from the agent to the experiment.
        Args:
            message: The message passed to the agent.
        Returns:
            The response (or answer) to the message.
        """
```

<!-- #region id="aJdhotjJuyCg" -->
### Random Agent
<!-- #endregion -->

```python id="_nofgAnQu3em"
class RandomAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.num_actions = None

    def agent_init(self, agent_info=None):
        if agent_info is None:
            agent_info = {}
        self.num_actions = agent_info.get('num_actions', 2)

    def agent_start(self, observation):
        pass

    def agent_step(self, reward, observation):
        pass

    def agent_end(self, reward):
        pass

    def agent_cleanup(self):
        pass

    def agent_message(self, message):
        pass

    def agent_policy(self, observation):
        return np.random.choice(self.num_actions)
```

```python colab={"base_uri": "https://localhost:8080/"} id="HwKhrIoVu5M3" executionInfo={"status": "ok", "timestamp": 1639146301441, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="57147a1f-d407-49ad-8e62-4d6c9d90922e"
# if __name__ == '__main__':
#     ag = RandomAgent()
#     print(ag.num_actions)

#     ag.agent_init()
#     print(ag.num_actions)
```

```python id="egiYidMHvP7D"
class Agent(BaseAgent):
    """agent does *no* learning, selects random action always"""

    def __init__(self):
        super().__init__()
        self.arm_count = None
        self.last_action = None
        self.num_actions = None
        self.q_values = None
        self.step_size = None
        self.initial_value = 0.0
        self.batch_size = None
        self.q_values_oracle = None  # used for batch updates

    def agent_init(self, agent_info=None):
        """Setup for the agent called when the experiment first starts."""

        if agent_info is None:
            agent_info = {}

        self.num_actions = agent_info.get("num_actions", 2)
        self.initial_value = agent_info.get("initial_value", 0.0)
        self.q_values = np.ones(agent_info.get("num_actions", 2)) * self.initial_value
        self.step_size = agent_info.get("step_size", 0.1)
        self.batch_size = agent_info.get('batch_size', 1)
        self.q_values_oracle = self.q_values.copy()
        self.arm_count = np.zeros(self.num_actions)  # [0.0 for _ in range(self.num_actions)]
        # self.last_action = np.random.choice(self.num_actions)  # set first action to random

    def agent_start(self, observation):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            observation (Numpy array): the state observation from the
                environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        self.last_action = np.random.choice(self.num_actions)

        return self.last_action

    def agent_step(self, reward, observation):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            observation (Numpy array): the state observation from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """
        # local_action = 0  # choose the action here
        self.last_action = np.random.choice(self.num_actions)

        return self.last_action

    def agent_end(self, reward):
        pass

    def agent_cleanup(self):
        pass

    def agent_message(self, message):
        pass
```

```python id="5w620FFCvTu8"
def argmax(q_values):
    """
    Takes in a list of q_values and returns the index of the item
    with the highest value. Breaks ties randomly.
    returns: int - the index of the highest value in q_values
    """
    top_value = float("-inf")
    ties = []

    for i in range(len(q_values)):
        if q_values[i] > top_value:
            ties = [i]
            top_value = q_values[i]
        elif q_values[i] == top_value:
            ties.append(i)

    return np.random.choice(ties)
```

<!-- #region id="F_7QyQIsvraK" -->
### Greedy Agent
<!-- #endregion -->

```python id="my_xM4bpvVDf"
class GreedyAgent(Agent):
    def __init__(self):
        super().__init__()

    def agent_init(self, agent_info=None):
        if agent_info is None:
            agent_info = {}

        super().agent_init(agent_info)

    def agent_step(self, reward, observation):
        """
        Takes one step for the agent. It takes in a reward and observation and
        returns the action the agent chooses at that time step.
        Arguments:
        reward -- float, the reward the agent received from the environment after taking the last action.
        observation -- float, the observed state the agent is in. Do not worry about this as you will not use it
                              until future lessons
        Returns:
        current_action -- int, the action chosen by the agent at the current time step.
        """

        a = self.last_action
        self.arm_count[a] += 1
        self.q_values_oracle[a] = self.q_values_oracle[a] + 1 / self.arm_count[a] * (reward - self.q_values_oracle[a])

        if sum(self.arm_count) % self.batch_size == 0:
            self.q_values = self.q_values_oracle.copy()

        current_action = argmax(self.q_values)
        self.last_action = current_action

        return current_action
```

<!-- #region id="6SzDe38oviFg" -->
### ϵ-Greedy Agent
<!-- #endregion -->

```python id="bPTYzIzTvWYZ"
class EpsilonGreedyAgent(Agent):
    def __init__(self):
        super().__init__()
        self.epsilon = None

    def agent_init(self, agent_info=None):
        if agent_info is None:
            agent_info = {}

        super().agent_init(agent_info)
        self.epsilon = agent_info.get("epsilon", 0.1)

    def agent_step(self, reward, observation):
        """
        Takes one step for the agent. It takes in a reward and observation and
        returns the action the agent chooses at that time step.
        Arguments:
        reward -- float, the reward the agent received from the environment after taking the last action.
        observation -- float, the observed state the agent is in. Do not worry about this as you will not use it
                              until future lessons
        Returns:
        current_action -- int, the action chosen by the agent at the current time step.
        """

        a = self.last_action

        self.arm_count[a] += 1
        self.q_values_oracle[a] = self.q_values_oracle[a] + 1 / self.arm_count[a] * (reward - self.q_values_oracle[a])

        if np.sum(self.arm_count) % self.batch_size == 0:
            self.q_values = self.q_values_oracle.copy()

        if np.random.random() < self.epsilon:
            current_action = np.random.choice(range(len(self.arm_count)))
        else:
            current_action = argmax(self.q_values)

        self.last_action = current_action

        return current_action
```

<!-- #region id="Y635PDE6vgdI" -->
### UCB Agent
<!-- #endregion -->

```python id="PV_TaTk1vX0T"
class UCBAgent(Agent):
    def __init__(self):
        super().__init__()
        self.upper_bounds = None
        self.alpha = None  # exploration parameter

    def agent_init(self, agent_info=None):
        if agent_info is None:
            agent_info = {}

        super().agent_init(agent_info)
        self.alpha = agent_info.get("alpha", 1.0)
        self.arm_count = np.ones(self.num_actions)
        self.upper_bounds = np.sqrt(np.log(np.sum(self.arm_count)) / self.arm_count)

    def agent_step(self, reward, observation):
        a = self.last_action

        self.arm_count[a] += 1
        self.q_values_oracle[a] = self.q_values_oracle[a] + 1 / self.arm_count[a] * (reward - self.q_values_oracle[a])

        # since we start with arms_count = np.ones(num_actions),
        # we should subtract num_actions to get number of the current round
        if (np.sum(self.arm_count) - self.num_actions) % self.batch_size == 0:
            self.q_values = self.q_values_oracle.copy()
            self.upper_bounds = np.sqrt(np.log(np.sum(self.arm_count)) / self.arm_count)

        # if min(self.q_values + self.alpha * self.upper_bounds) < max(self.q_values):
        #     print(f'Distinguish suboptimal arm at step {sum(self.arm_count)}')
        current_action = argmax(self.q_values + self.alpha * self.upper_bounds)
        # current_action = np.argmax(self.q_values + self.alpha * self.upper_bounds)

        self.last_action = current_action

        return current_action
```

<!-- #region id="UiTTsyhFvdwc" -->
### TS Agent
<!-- #endregion -->

```python id="TT3oZeNavY4e"
class TSAgent(Agent):
    def agent_step(self, reward, observation):
        a = self.last_action
        self.arm_count[a] += 1
        self.q_values_oracle[a] = self.q_values_oracle[a] + 1 / self.arm_count[a] * (reward - self.q_values_oracle[a])

        if (np.sum(self.arm_count) - self.num_actions) % self.batch_size == 0:
            self.q_values = self.q_values_oracle.copy()

        # sample from posteriors
        theta = [beta.rvs(a + 1, b + 1, size=1) for a, b in
                 zip(self.q_values * self.arm_count, self.arm_count - self.q_values * self.arm_count)]
        # choose the max realization
        current_action = argmax(theta)
        self.last_action = current_action

        return current_action
```

<!-- #region id="-IL38-vIzIG1" -->
### LinUCB Agent
<!-- #endregion -->

```python id="gyXQwM2-zKR_"
class LinUCBAgent(BaseAgent):

    def __init__(self):
        super().__init__()
        self.name = "LinUCB"

    def agent_init(self, agent_info=None):

        if agent_info is None:
            agent_info = {}

        self.num_actions = agent_info.get('num_actions', 3)
        self.alpha = agent_info.get('alpha', 1)
        self.batch_size = agent_info.get('batch_size', 1)
        # Set random seed for policy for each run
        self.policy_rand_generator = np.random.RandomState(agent_info.get("seed", None))

        self.last_action = None
        self.last_state = None
        self.num_round = None

    def agent_policy(self, observation):
        p_t = np.zeros(self.num_actions)

        for i in range(self.num_actions):
            # initialize theta hat
            self.theta = inv(self.A[i]).dot(self.b[i])
            # get context of each arm from flattened vector of length 100
            cntx = observation
            # get gain reward of each arm
            p_t[i] = self.theta.T.dot(cntx) + self.alpha * np.sqrt(np.maximum(cntx.dot(inv(self.A[i]).dot(cntx)), 0))
        # action = np.random.choice(np.where(p_t == max(p_t))[0])
        action = self.policy_rand_generator.choice(np.where(p_t == max(p_t))[0])

        return action

    def agent_start(self, observation):
        # Specify feature dimension
        self.ndims = len(observation)

        self.A = np.zeros((self.num_actions, self.ndims, self.ndims))
        # Instantiate b as a 0 vector of length ndims.
        self.b = np.zeros((self.num_actions, self.ndims, 1))
        # set each A per arm as identity matrix of size ndims
        for arm in range(self.num_actions):
            self.A[arm] = np.eye(self.ndims)

        self.A_oracle = self.A.copy()
        self.b_oracle = self.b.copy()

        self.last_state = observation
        self.last_action = self.agent_policy(self.last_state)
        self.num_round = 0

        return self.last_action

    def agent_update(self, reward):
        self.A_oracle[self.last_action] = self.A_oracle[self.last_action] + np.outer(self.last_state, self.last_state)
        self.b_oracle[self.last_action] = np.add(self.b_oracle[self.last_action].T, self.last_state * reward).reshape(self.ndims, 1)

    def agent_step(self, reward, observation):
        if reward is not None:
            self.agent_update(reward)
            # it is a good question whether I should increment num_round outside
            # condition or not (since theoretical result doesn't clarify this
            self.num_round += 1

        if self.num_round % self.batch_size == 0:
            self.A = self.A_oracle.copy()
            self.b = self.b_oracle.copy()

        self.last_state = observation
        self.last_action = self.agent_policy(self.last_state)

        return self.last_action

    def agent_end(self, reward):
        if reward is not None:
            self.agent_update(reward)
            self.num_round += 1

        if self.num_round % self.batch_size == 0:
            self.A = self.A_oracle.copy()
            self.b = self.b_oracle.copy()

    def agent_message(self, message):
        pass

    def agent_cleanup(self):
        pass
```

```python colab={"base_uri": "https://localhost:8080/"} id="Nkdq0ya9zNlE" executionInfo={"status": "ok", "timestamp": 1639148410291, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b7a9b9c5-e3d1-4fc8-bdc5-321172c6ebdb"
# if __name__ == '__main__':
#     agent_info = {'alpha': 2,
#                   'num_actions': 4,
#                   'seed': 1}

#     # check initialization
#     linucb = LinUCBAgent()
#     linucb.agent_init(agent_info)
#     print(linucb.num_actions, linucb.alpha)

#     assert linucb.num_actions == 4
#     assert linucb.alpha == 2

#     # check policy
#     observation = np.array([1, 2, 5, 0])
#     linucb.A = np.zeros((linucb.num_actions, len(observation), len(observation)))
#     # Instantiate b as a 0 vector of length ndims.
#     linucb.b = np.zeros((linucb.num_actions, len(observation), 1))
#     # set each A per arm as identity matrix of size ndims
#     for arm in range(linucb.num_actions):
#         linucb.A[arm] = np.eye(len(observation))

#     action = linucb.agent_policy(observation)
#     print(action)

#     assert action == 1

#     # check start
#     observation = np.array([1, 2, 5, 0])
#     linucb.agent_start(observation)
#     print(linucb.ndims)
#     print(linucb.last_state, linucb.last_action)

#     assert linucb.ndims == len(observation)
#     assert np.allclose(linucb.last_state, observation)
#     assert np.allclose(linucb.b, np.zeros((linucb.num_actions, len(observation), 1)))
#     assert np.allclose(linucb.A, np.array([np.eye(len(observation)), np.eye(len(observation)),
#                                            np.eye(len(observation)), np.eye(len(observation))]))
#     assert linucb.last_action == 3

#     # check step
#     observation = np.array([5, 3, 1, 2])
#     reward = 1

#     action = linucb.agent_step(reward, observation)
#     print(linucb.A)
#     print(linucb.b)
#     print(action)

#     true_A = np.array([[2., 2., 5., 0.],
#                        [2., 5., 10., 0.],
#                        [5., 10., 26., 0.],
#                        [0., 0., 0., 1.]])

#     true_b = np.array([[1.],
#                        [2.],
#                        [5.],
#                        [0.]])

#     for i in range(3):
#         assert np.allclose(linucb.A[i], np.eye(4))
#         assert np.allclose(linucb.b[i], np.zeros((linucb.num_actions, 4, 1)))
#     assert np.allclose(linucb.A[3], true_A)
#     assert np.allclose(linucb.b[3], true_b)
#     assert linucb.last_action == 0

#     observation = np.array([3, 1, 3, 5])
#     reward = None

#     action = linucb.agent_step(reward, observation)
#     print(linucb.A)
#     print(linucb.b)
#     print(action)

#     assert np.allclose(linucb.A[3], true_A)
#     assert np.allclose(linucb.b[3], true_b)
#     assert action == 0

#     # check batch size
#     agent_info = {'alpha': 2,
#                   'num_actions': 4,
#                   'seed': 1,
#                   'batch_size': 2}
#     linucb = LinUCBAgent()
#     linucb.agent_init(agent_info)
#     observation = np.array([1, 2, 5, 0])
#     linucb.agent_start(observation)
#     assert linucb.num_round == 0
#     assert linucb.last_action == 1

#     observation = np.array([5, 3, 1, 2])
#     reward = 1

#     action = linucb.agent_step(reward, observation)
#     assert linucb.num_round == 1
#     assert np.allclose(linucb.b, np.zeros((linucb.num_actions, len(observation), 1)))
#     assert np.allclose(linucb.A, np.array([np.eye(len(observation)), np.eye(len(observation)),
#                                            np.eye(len(observation)), np.eye(len(observation))]))

#     for i in [0, 2, 3]:
#         assert np.allclose(linucb.A_oracle[i], np.eye(4))
#         assert np.allclose(linucb.b_oracle[i], np.zeros((linucb.num_actions, 4, 1)))
#     assert np.allclose(linucb.A_oracle[1], true_A)
#     assert np.allclose(linucb.b_oracle[1], true_b)

#     observation = np.array([3, 1, 3, 5])
#     reward = None
#     action = linucb.agent_step(reward, observation)
#     # sinse reward is None, nothing should happen
#     assert linucb.num_round == 1
#     assert np.allclose(linucb.b, np.zeros((linucb.num_actions, len(observation), 1)))
#     assert np.allclose(linucb.A, np.array([np.eye(len(observation)), np.eye(len(observation)),
#                                            np.eye(len(observation)), np.eye(len(observation))]))

#     for i in [0, 2, 3]:
#         assert np.allclose(linucb.A_oracle[i], np.eye(4))
#         assert np.allclose(linucb.b_oracle[i], np.zeros((linucb.num_actions, 4, 1)))
#     assert np.allclose(linucb.A_oracle[1], true_A)
#     assert np.allclose(linucb.b_oracle[1], true_b)

#     observation = np.array([3, 0, 2, 5])
#     reward = 0
#     action = linucb.agent_step(reward, observation)

#     assert linucb.num_round == 2
#     assert np.allclose(linucb.b, linucb.b_oracle)
#     assert np.allclose(linucb.A, linucb.A_oracle)
```

<!-- #region id="4L_OJQQEvP4k" -->
### LinTS Agent
<!-- #endregion -->

```python id="X7mSGLS4vP20"
class LinTSAgent(BaseAgent):

    def __init__(self):
        super().__init__()
        self.name = "LinTS"

    def agent_init(self, agent_info=None):

        if agent_info is None:
            agent_info = {}

        self.num_actions = agent_info.get('num_actions', 3)
        self.alpha = agent_info.get('alpha', 1)
        self.lambda_ = agent_info.get('lambda', 1)
        self.batch_size = agent_info.get('batch_size', 1)
        # Set random seed for policy for each run
        self.policy_rand_generator = np.random.RandomState(agent_info.get("seed", None))

        self.replay_buffer = ReplayBuffer(agent_info['replay_buffer_size'],
                                          agent_info.get("seed"))


        self.last_action = None
        self.last_state = None
        self.num_round = None

    def agent_policy(self, observation, mode='sample'):
        p_t = np.zeros(self.num_actions)
        cntx = observation

        for i in range(self.num_actions):
            # sampling weights after update
            self.w = self.get_weights(i)

            # using weight depending on mode
            if mode == 'sample':
                w = self.w  # weights are samples of posteriors
            elif mode == 'expected':
                w = self.m[i]  # weights are expected values of posteriors
            else:
                raise Exception('mode not recognized!')

            # calculating probabilities
            p_t[i] = 1 / (1 + np.exp(-1 * cntx.dot(w)))
            action = self.policy_rand_generator.choice(np.where(p_t == max(p_t))[0])
            # probs = softmax(p_t.reshape(1, -1))
            # action = self.policy_rand_generator.choice(a=range(self.num_actions), p=probs)

        return action

    def get_weights(self, arm):
        return np.random.normal(self.m[arm], self.alpha * self.q[arm] ** (-1.0), size=len(self.w))

        # the loss function
    def loss(self, w, *args):
        X, y = args
        return 0.5 * (self.q[self.last_action] * (w - self.m[self.last_action])).dot(w - self.m[self.last_action]) + np.sum(
            [np.log(1 + np.exp(-y[j] * w.dot(X[j]))) for j in range(y.shape[0])])

    # the gradient
    def grad(self, w, *args):
        X, y = args
        return self.q[self.last_action] * (w - self.m[self.last_action]) + (-1) * np.array(
            [y[j] * X[j] / (1. + np.exp(y[j] * w.dot(X[j]))) for j in range(y.shape[0])]).sum(axis=0)

    # fitting method
    def agent_update(self, X, y):
        # step 1, find w
        self.w = minimize(self.loss, self.w, args=(X, y), jac=self.grad, method="L-BFGS-B",
                          options={'maxiter': 20, 'disp': False}).x
        # self.m_oracle[self.last_action] = self.w
        self.m[self.last_action] = self.w

        # step 2, update q
        P = (1 + np.exp(1 - X.dot(self.m[self.last_action]))) ** (-1)
        #self.q_oracle[self.last_action] = self.q[self.last_action] + (P * (1 - P)).dot(X ** 2)
        self.q[self.last_action] = self.q[self.last_action] + (P * (1 - P)).dot(X ** 2)

    def agent_start(self, observation):
        # Specify feature dimension
        self.ndims = len(observation)

        # initializing parameters of the model
        self.m = np.zeros((self.num_actions, self.ndims))
        self.q = np.ones((self.num_actions, self.ndims)) * self.lambda_
        # initializing weights using any arm (e.g. 0) because they all equal
        self.w = np.array([0.]*self.ndims, dtype=np.float64)

        # self.m_oracle = self.m.copy()
        # self.q_oracle = self.q.copy()

        self.last_state = observation
        self.last_action = self.agent_policy(self.last_state)
        self.num_round = 0

        return self.last_action


    def agent_step(self, reward, observation):
        # Append new experience to replay buffer
        if reward is not None:
            self.replay_buffer.append(self.last_state, self.last_action, reward)
            # it is a good question whether I should increment num_round outside
            # condition or not (since theoretical result doesn't clarify this
            self.num_round += 1

            if self.num_round % self.batch_size == 0:
                X, y = self.replay_buffer.sample(self.last_action)
                X = np.array(X)
                y = np.array(y)
                self.agent_update(X, y)
                # self.m = self.m_oracle.copy()
                # self.q = self.q_oracle.copy()

        self.last_state = observation
        self.last_action = self.agent_policy(self.last_state)

        return self.last_action

    def agent_end(self, reward):
        # Append new experience to replay buffer
        if reward is not None:
            self.replay_buffer.append(self.last_state, self.last_action, reward)
            # it is a good question whether I should increment num_round outside
            # condition or not (since theoretical result doesn't clarify this
            self.num_round += 1

            if self.num_round % self.batch_size == 0:
                X, y = self.replay_buffer.sample(self.last_action)
                X = np.array(X)
                y = np.array(y)
                self.agent_update(X, y)
                # self.m = self.m_oracle.copy()
                # self.q = self.q_oracle.copy()

    def agent_message(self, message):
        pass

    def agent_cleanup(self):
        pass
```

```python colab={"base_uri": "https://localhost:8080/"} id="yr9HfGeJy-vR" executionInfo={"status": "ok", "timestamp": 1639148425016, "user_tz": -330, "elapsed": 3, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4037b3ed-73b8-4057-8ea7-c1cb4d7fd177"
# if __name__ == '__main__':
#     agent_info = {'alpha': 2,
#                   'num_actions': 3,
#                   'seed': 1,
#                   'lambda': 2,
#                   'replay_buffer_size': 100000}

#     np.random.seed(1)
#     # check initialization
#     lints = LinTSAgent()
#     lints.agent_init(agent_info)
#     print(lints.num_actions, lints.alpha, lints.lambda_)

#     assert lints.num_actions == 3
#     assert lints.alpha == 2
#     assert lints.lambda_ == 2

#     # check agent policy
#     observation = np.array([1, 2, 5, 0])
#     lints.m = np.zeros((lints.num_actions, len(observation)))
#     lints.q = np.ones((lints.num_actions, len(observation))) * lints.lambda_
#     lints.w = np.random.normal(lints.m[0], lints.alpha * lints.q[0] ** (-1.0), size=len(observation))
#     print(lints.w)
#     action = lints.agent_policy(observation)
#     print(action)

#     # check agent start
#     observation = np.array([1, 2, 5, 0])
#     lints.agent_start(observation)
#     # manually reassign w to np.random.normal, because I np.seed doesn't work inside the class
#     np.random.seed(1)
#     lints.w = np.random.normal(lints.m[0], lints.alpha * lints.q[0] ** (-1.0), size=len(observation))
#     print(lints.ndims)
#     print(lints.last_state, lints.last_action)
#     print(lints.last_action)
#     assert lints.ndims == len(observation)
#     assert np.allclose(lints.last_state, observation)
#     assert np.allclose(lints.m, np.zeros((lints.num_actions, lints.ndims)))
#     assert np.allclose(lints.q, np.ones((lints.num_actions, lints.ndims)) * lints.lambda_)
#     assert np.allclose(lints.w, np.array([ 1.62434536, -0.61175641, -0.52817175, -1.07296862]))
#     # assert lints.last_action == 1

#     # check step
#     observation = np.array([5, 3, 1, 2])
#     reward = 1
#     action = lints.agent_step(reward, observation)
#     print(action)

#     observation = np.array([1, 3, 2, 1])
#     reward = 0
#     action = lints.agent_step(reward, observation)
#     print(action)
```

<!-- #region id="IQr77JFsu6Jq" -->
## Environments
<!-- #endregion -->

<!-- #region id="oAuUEthsu9TG" -->
### Base Environment
<!-- #endregion -->

<!-- #region id="pEz4-17VvAZQ" -->
Abstract environment base class for RL-Glue-py.
<!-- #endregion -->

```python id="qqZDb6A9u-85"
class BaseEnvironment:
    """Implements the environment for an RLGlue environment
    Note:
        env_init, env_start, env_step, env_cleanup, and env_message are required
        methods.
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        reward = None
        observation = None
        termination = None
        self.reward_state_term = (reward, observation, termination)

    @abstractmethod
    def env_init(self, env_info={}):
        """Setup for the environment called when the experiment first starts.
        Note:
            Initialize a tuple with the reward, first state observation, boolean
            indicating if it's terminal.
        """

    @abstractmethod
    def env_start(self):
        """The first method called when the experiment starts, called before the
        agent starts.
        Returns:
            The first state observation from the environment.
        """

    @abstractmethod
    def env_step(self, action):
        """A step taken by the environment.
        Args:
            action: The action taken by the agent
        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """

    @abstractmethod
    def env_cleanup(self):
        """Cleanup done after the environment ends"""

    @abstractmethod
    def env_message(self, message):
        """A message asking the environment for information
        Args:
            message: the message passed to the environment
        Returns:
            the response (or answer) to the message
        """
```

<!-- #region id="_Mm2zTKyv3Fb" -->
### k-arm Environment
<!-- #endregion -->

```python id="YLTkxvLGvC4g"
class Environment(BaseEnvironment):
    """Implements the environment for an RLGlue environment
    Note:
        env_init, env_start, env_step, env_cleanup, and env_message are required
        methods.
    """

    actions = [0]

    def __init__(self):
        super().__init__()
        reward = None
        observation = None
        termination = None
        self.seed = None
        self.k = None
        self.reward_type = None
        self.custom_arms = None
        self.reward_state_term = (reward, observation, termination)
        self.count = 0
        self.arms = []
        self.subopt_gaps = None

    def env_init(self, env_info=None):
        """Setup for the environment called when the experiment first starts.
        Note:
            Initialize a tuple with the reward, first state observation, boolean
            indicating if it's terminal.
        """

        if env_info is None:
            env_info = {}
        self.k = env_info.get("num_actions", 2)
        self.reward_type = env_info.get("reward_type", "subgaussian")
        self.custom_arms = env_info.get("arms_values", None)

        if self.reward_type not in ['Bernoulli', 'subgaussian']:
            raise ValueError('Unknown reward_type: ' + str(self.reward_type))

        if self.custom_arms is None:
            if self.reward_type == 'Bernoulli':
                self.arms = np.random.uniform(0, 1, self.k)
            else:
                self.arms = np.random.randn(self.k)
        else:
            self.arms = self.custom_arms
        self.subopt_gaps = np.max(self.arms) - self.arms

        local_observation = 0  # An empty NumPy array

        self.reward_state_term = (0.0, local_observation, False)

    def env_start(self):
        """The first method called when the experiment starts, called before the
        agent starts.
        Returns:
            The first state observation from the environment.
        """

        return self.reward_state_term[1]

    def env_step(self, action):
        """A step taken by the environment.
        Args:
            action: The action taken by the agent
        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """
        if self.reward_type == 'Bernoulli':
            reward = np.random.binomial(1, self.arms[action], 1)
        else:
            reward = self.arms[action] + np.random.randn()
        obs = self.reward_state_term[1]

        self.reward_state_term = (reward, obs, False)

        return self.reward_state_term

    def env_cleanup(self):
        """Cleanup done after the environment ends"""
        pass

    def env_message(self, message):
        """A message asking the environment for information
        Args:
            message (string): the message passed to the environment
        Returns:
            string: the response (or answer) to the message
        """
        if message == "what is the current reward?":
            return "{}".format(self.reward_state_term[0])

        # else
        return "I don't know how to respond to your message"
```

<!-- #region id="naYFRhirzYfu" -->
### Offline Evaluator
<!-- #endregion -->

```python id="iJLVLYunzbPe"
class OfflineEvaluator:
    def __init__(self, eval_info=None):

        if eval_info is None:
            eval_info = {}

        self.dataset = eval_info['dataset']
        self.agent = eval_info['agent']

        if not isinstance(self.dataset, Dataset):
            raise TypeError('dataset ' + "must be a " + str(Dataset))
        if not isinstance(self.agent, BaseAgent):
            raise TypeError('agent ' + "must be a " + str(BaseAgent))

        self.total_reward = None
        self.average_reward = None
        self.num_matches = None
        self.idxs = range(self.dataset.__len__())
        self.counter = None

    def eval_start(self):
        self.total_reward = 0
        self.average_reward = [0]
        self.num_matches = 0
        self.idxs = range(self.dataset.__len__())
        self.counter = 0

    def _get_observation(self):
        idx = self.idxs[self.counter]
        self.counter += 1

        return self.dataset.__getitem__(idx)

    def eval_step(self):
        observation = self._get_observation()

        state = observation[0]
        true_action = observation[1]
        reward = observation[2]

        pred_action = self.agent.agent_policy(state)

        if true_action != pred_action:
            return

        self.num_matches += 1
        aw_reward = self.average_reward[-1] + (reward - self.average_reward[-1]) / self.num_matches
        self.average_reward.append(aw_reward)
        self.total_reward += reward

    def eval_run(self):
        self.eval_start()

        while self.counter < self.dataset.__len__():
            self.eval_step()

        return self.average_reward
```

```python colab={"base_uri": "https://localhost:8080/", "height": 282} id="BY8sASekzc6S" executionInfo={"status": "ok", "timestamp": 1639148441141, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1ddaab3a-5b2f-41a8-842f-407aaa3b1f62"
# if __name__ == '__main__':

#     dir1 = 'data/mushroom_data_final.pickle'

#     ra = RandomAgent()
#     agent_info = {'num_actions': 2}
#     ra.agent_init(agent_info)

#     result = []
#     result1 = []

#     for seed_ in [1, 5, 10]:  # , 2, 3, 32, 123, 76, 987, 2134]:
#         dataset = BanditDataset(pickle_file=dir1, seed=seed_)

#         eval_info = {'dataset': dataset, 'agent': ra}
#         evaluator = OfflineEvaluator(eval_info)

#         reward = evaluator.eval_run()

#         result.append(reward)
#         result1.append(evaluator.total_reward)

#     for elem in result:
#         plt.plot(elem)
#     plt.legend()
#     plt.show()
```

<!-- #region id="tL1Ur1CtzlGx" -->
### Replay Environment
<!-- #endregion -->

```python id="YK8fE2VyzlEO"
class ReplayEnvironment(BaseEnvironment):
    dataset: BanditDataset

    def __init__(self):
        super().__init__()
        self.counter = None
        self.last_observation = None

    def env_init(self, env_info=None):
        """
        Set parameters needed to setup the replay SavePilot environment.
        Assume env_info dict contains:
        {
            pickle_file: data directory [str]
        }
        Args:
            env_info (dict):
        """
        if env_info is None:
            env_info = {}

        directory = env_info['pickle_file']
        seed = env_info.get('seed', None)
        self.dataset = BanditDataset(directory, seed)
        self.idxs = range(self.dataset.__len__())
        self.counter = 0

    def _get_observation(self):
        idx = self.idxs[self.counter]

        return self.dataset.__getitem__(idx)

    def env_start(self):
        self.last_observation = self._get_observation()

        state = self.last_observation[0]
        reward = None
        is_terminal = False

        self.reward_state_term = (reward, state, is_terminal)
        self.counter += 1

        # return first state from the environment
        return self.reward_state_term[1]

    def env_step(self, action):
        true_action = self.last_observation[1]
        reward = self.last_observation[2]

        if true_action != action:
            reward = None

        observation = self._get_observation()
        state = observation[0]

        if self.counter == self.dataset.__len__() - 1:
            is_terminal = True
        else:
            is_terminal = False

        self.reward_state_term = (reward, state, is_terminal)

        self.last_observation = observation
        self.counter += 1

        return self.reward_state_term

    def env_cleanup(self):
        pass

    def env_message(self, message):
        pass
```

<!-- #region id="s42XsgHev0lv" -->
## Wrappers
<!-- #endregion -->

<!-- #region id="OlbboMWrwRV2" -->
### RL Glue
<!-- #endregion -->

```python id="InGf3vAFwTIF"
class RLGlue:
    """RLGlue class
    args:
        env_name (string): the name of the module where the Environment class can be found
        agent_name (string): the name of the module where the Agent class can be found
    """

    def __init__(self, env_class, agent_class):
        self.environment = env_class()
        self.agent = agent_class()

        self.total_reward = None
        self.average_reward = None
        self.last_action = None
        self.num_steps = None
        self.num_episodes = None
        self.num_matches = None

    def rl_init(self, agent_init_info={}, env_init_info={}):
        """Initial method called when RLGlue experiment is created"""
        self.environment.env_init(env_init_info)
        self.agent.agent_init(agent_init_info)

        self.total_reward = 0.0
        self.average_reward = [0]
        self.num_steps = 0
        self.num_episodes = 0
        self.num_matches = 0

    def rl_start(self):
        """Starts RLGlue experiment
        Returns:
            tuple: (state, action)
        """

        last_state = self.environment.env_start()
        self.last_action = self.agent.agent_start(last_state)

        observation = (last_state, self.last_action)

        return observation

    def rl_agent_start(self, observation):
        """Starts the agent.
        Args:
            observation: The first observation from the environment
        Returns:
            The action taken by the agent.
        """
        return self.agent.agent_start(observation)

    def rl_agent_step(self, reward, observation):
        """Step taken by the agent
        Args:
            reward (float): the last reward the agent received for taking the
                last action.
            observation : the state observation the agent receives from the
                environment.
        Returns:
            The action taken by the agent.
        """
        return self.agent.agent_step(reward, observation)

    def rl_agent_end(self, reward):
        """Run when the agent terminates
        Args:
            reward (float): the reward the agent received when terminating
        """
        self.agent.agent_end(reward)

    def rl_env_start(self):
        """Starts RL-Glue environment.
        Returns:
            (float, state, Boolean): reward, state observation, boolean
                indicating termination
        """
        self.total_reward = 0.0
        self.num_steps = 1

        this_observation = self.environment.env_start()

        return this_observation

    def rl_env_step(self, action):
        """Step taken by the environment based on action from agent
        Args:
            action: Action taken by agent.
        Returns:
            (float, state, Boolean): reward, state observation, boolean
                indicating termination.
        """
        ro = self.environment.env_step(action)
        (this_reward, _, terminal) = ro

        self.total_reward += this_reward

        if terminal:
            self.num_episodes += 1
        else:
            self.num_steps += 1

        return ro

    def rl_step(self):
        """Step taken by RLGlue, takes environment step and either step or
            end by agent.
        Returns:
            (float, state, action, Boolean): reward, last state observation,
                last action, boolean indicating termination
        """

        (reward, last_state, term) = self.environment.env_step(self.last_action)

        if reward is not None:
            self.num_matches += 1
            aw_reward = self.average_reward[-1] + (reward - self.average_reward[-1]) / self.num_matches
            self.average_reward.append(aw_reward)
            self.total_reward += reward

        if term:
            self.num_episodes += 1
            self.agent.agent_end(reward)
            roat = (reward, last_state, None, term)
        else:
            self.num_steps += 1
            self.last_action = self.agent.agent_step(reward, last_state)
            roat = (reward, last_state, self.last_action, term)

        return roat

    def rl_cleanup(self):
        """Cleanup done at end of experiment."""
        self.environment.env_cleanup()
        self.agent.agent_cleanup()

    def rl_agent_message(self, message):
        """Message passed to communicate with agent during experiment
        Args:
            message: the message (or question) to send to the agent
        Returns:
            The message back (or answer) from the agent
        """

        return self.agent.agent_message(message)

    def rl_env_message(self, message):
        """Message passed to communicate with environment during experiment
        Args:
            message: the message (or question) to send to the environment
        Returns:
            The message back (or answer) from the environment
        """
        return self.environment.env_message(message)

    def rl_episode(self, max_steps_this_episode):
        """Runs an RLGlue episode
        Args:
            max_steps_this_episode (Int): the maximum steps for the experiment to run in an episode
        Returns:
            Boolean: if the episode should terminate
        """
        is_terminal = False

        self.rl_start()

        while (not is_terminal) and ((max_steps_this_episode == 0) or
                                     (self.num_steps < max_steps_this_episode)):
            rl_step_result = self.rl_step()
            is_terminal = rl_step_result[3]

        return is_terminal

    def rl_return(self):
        """The total reward
        Returns:
            float: the total reward
        """
        return self.total_reward

    def rl_num_steps(self):
        """The total number of steps taken
        Returns:
            Int: the total number of steps taken
        """
        return self.num_steps

    def rl_num_episodes(self):
        """The number of episodes
        Returns
            Int: the total number of episodes
        """
        return self.num_episodes
```

<!-- #region id="MCXJTVo5xRpq" -->
### Policy
<!-- #endregion -->

```python id="9RsPVJpnwmad"
class Policy:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.rl_glue = None

    @abstractmethod
    def get_average_performance(self, agent_info=None, env_info=None, exper_info=None):
        raise NotImplementedError
```

<!-- #region id="RbiWk0wWxTD-" -->
### Bandit wrapper
<!-- #endregion -->

```python id="zAzjgURdwsl5"
class BanditWrapper(Policy):
    def get_average_performance(self, agent_info=None, env_info=None, exper_info=None):

        if exper_info is None:
            exper_info = {}
        if env_info is None:
            env_info = {}
        if agent_info is None:
            agent_info = {}

        num_runs = exper_info.get("num_runs", 100)
        num_steps = exper_info.get("num_steps", 1000)
        return_type = exper_info.get("return_type", None)
        seed = exper_info.get("seed", None)

        np.random.seed(seed)
        seeds = np.random.randint(0, num_runs * 100, num_runs)

        all_averages = []
        subopt_arm_average = []
        best_arm = []
        worst_arm = []
        all_chosen_arm = []
        average_regret = []

        for run in tqdm(range(num_runs)):
            np.random.seed(seeds[run])

            self.rl_glue = RLGlue(self.env, self.agent)
            self.rl_glue.rl_init(agent_info, env_info)
            (first_state, first_action) = self.rl_glue.rl_start()

            worst_position = np.argmin(self.rl_glue.environment.arms)
            best_value = np.max(self.rl_glue.environment.arms)
            worst_value = np.min(self.rl_glue.environment.arms)
            best_arm.append(best_value)
            worst_arm.append(worst_value)

            scores = [0]
            averages = []
            subopt_arm = []
            chosen_arm_log = []

            cum_regret = [0]
            delta = self.rl_glue.environment.subopt_gaps[first_action]
            cum_regret.append(cum_regret[-1] + delta)

            # first action was made in rl_start, that's why run over num_steps-1
            for i in range(num_steps-1):
                reward, _, action, _ = self.rl_glue.rl_step()
                chosen_arm_log.append(action)
                scores.append(scores[-1] + reward)
                averages.append(scores[-1] / (i + 1))
                subopt_arm.append(self.rl_glue.agent.arm_count[worst_position])

                delta = self.rl_glue.environment.subopt_gaps[action]
                cum_regret.append(cum_regret[-1] + delta)

            all_averages.append(averages)
            subopt_arm_average.append(subopt_arm)
            all_chosen_arm.append(chosen_arm_log)

            average_regret.append(cum_regret)

        if return_type is None:
            returns = (np.mean(all_averages, axis=0),
                       np.mean(best_arm))
        elif return_type == 'regret':
            returns = np.mean(average_regret, axis=0)
        elif return_type == 'regret_reward':
            returns = (np.mean(average_regret, axis=0),
                       np.mean(all_averages, axis=0))
        elif return_type == 'arm_choice_analysis':
            returns = (np.mean(all_averages, axis=0),
                       np.mean(best_arm),
                       np.mean(all_chosen_arm, axis=0))
        elif return_type == 'complex':
            returns = (np.mean(all_averages, axis=0),
                       np.mean(subopt_arm_average, axis=0),
                       np.array(best_arm), np.array(worst_arm),
                       np.mean(average_regret, axis=0))

        return returns
```

<!-- #region id="58Lm6nzi3c7a" -->
### Run experiments
<!-- #endregion -->

```python id="GWc2CLhz3c2E"
def run_experiment(environment, agent, environment_parameters, agent_parameters,
                   experiment_parameters, save_data=True, dir=''):
    rl_glue = RLGlue(environment, agent)

    # save sum of reward at the end of each episode
    agent_sum_reward = []

    env_info = environment_parameters
    agent_info = agent_parameters

    # one agent setting
    for run in tqdm(range(1, experiment_parameters["num_runs"] + 1)):
        env_info["seed"] = run

        rl_glue.rl_init(agent_info, env_info)
        rl_glue.rl_episode(0)
        agent_sum_reward.append(rl_glue.average_reward)

    leveled_result = get_leveled_data(agent_sum_reward)
    if save_data:
        save_name = "{}-{}".format(rl_glue.agent.name, rl_glue.agent.batch_size)
        file_dir = "results/{}".format(dir)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        np.save("{}/sum_reward_{}".format(file_dir, save_name), leveled_result)

    return leveled_result
```

```python colab={"base_uri": "https://localhost:8080/", "height": 297, "referenced_widgets": ["87c240ca0fe8431096f5092d00b6d63f", "6933623da13c468ea1caba8cb1e126f5", "3beaa54b58654d2199153c4229b50f84", "317c46945fd64488ad7b2cc0bf30b7d2", "fa2e58ea24f84364b95fcc3f1587e9ad", "afa78843d54d404ab99cd87d55a36bb9", "0c9761efc1a04534871cb6c05f232f4a", "21b40574fdb44788af67ece3c7b770c6", "0a8e3b93b06740ad8cb9075dbf5d041a", "bcaa037ccf1b47bf8d40956a370f93dc", "2258f153219f48cc882eda53f0db33a2"]} id="oLbyRqGi3miH" executionInfo={"status": "ok", "timestamp": 1639148611213, "user_tz": -330, "elapsed": 8524, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d2341afc-3a7d-4ca0-ff62-a1904cad3dc8"
# if __name__ == '__main__':

#     num_experements = 10
#     batch_size = 100
#     data_dir = 'data/mushroom_data_final.pickle'

#     experiment_parameters = {"num_runs": num_experements}
#     env_info = {'pickle_file': data_dir}
#     agent_info = {'alpha': 2,
#                   'num_actions': 3,
#                   'seed': 1,
#                   'batch_size': 1}

#     agent = LinUCBAgent
#     environment = ReplayEnvironment

#     result = run_experiment(environment, agent, env_info, agent_info, experiment_parameters, save_data=False)

#     smoothed_leveled_result = smooth(result, 100)
#     mean_smoothed_leveled_result = np.mean(smoothed_leveled_result, axis=0)

#     plt.plot(mean_smoothed_leveled_result, lw=3, ls='-.', label='online policy')
#     plt.show()
```

<!-- #region id="kD7w8VVTwmV-" -->
## Experiments
<!-- #endregion -->

<!-- #region id="S9-ruIHEwmTS" -->
### UCB
<!-- #endregion -->

<!-- #region id="iSarqXSj5buA" -->
#### UCB dynamic by timesteps
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 359, "referenced_widgets": ["6574536e49dd498bae5d87ab60144c59", "a787648e15174568a65f7b5e17ee43d2", "9fccdb73039f4b649f5389aa2b4099ad", "701a04ff58fc492f88774aa7139dcbc2", "37296861808045328c2c4eb74625987a", "8a9c221fcb4a46f9af525b5b0373a598", "f155f2ec83cb416d8d375ae77330165b", "af831b75ff7146fc99abb8677c84bdf3", "8fe71971bf8049bca4a5c6fde2bf7471", "9a87fac6283543008f922a57f98b8514", "ea8e8fab825a499c8fe628a021b731ca", "d2a7f4cc89a44501b6423cff8f76f77d", "9231d7a857624fbfa4c6dc1ba3307814", "7dbee5d5ef884477be34e5a9889d13b1", "ed0108c515b94090bc7ee284284b535b", "ef228c5730d943149399ec81f4de4d46", "0dd4ce26eb994bf89f9d429982672f1d", "3c78bcaf65d745928c7cec9b3618f2f4", "3f9147ea50504b5e941cf600259b64c6", "3cb22a9a850b42ce8d72132b34cede77", "3929bb10e2044d26869495a947ecd340", "cf4c98d2628e418b84ef3c798588675a"]} id="Vc7uTIz8xARE" executionInfo={"status": "ok", "timestamp": 1639148297436, "user_tz": -330, "elapsed": 1433361, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="38fb8767-6cf3-41cc-8cd3-1b9abd760ed6"
env = Environment
agent = UCBAgent

alpha = 1

num_runs = 1000
num_steps = 10000
seed = None
if_save = False
exper_info = {"num_runs": num_runs,
              "num_steps": num_steps,
              "seed": seed,
              "return_type": "regret"}

k = 2
arms_values = [0.7, 0.65]
reward_type = 'Bernoulli'
env_info = {"num_actions": k,
            "reward_type": reward_type,
            "arms_values": arms_values}

# batch-online experiment
batch_res = []
online_res = []
batch = 10
agent_info_batch = {"num_actions": k, "batch_size": batch, "alpha": alpha}
agent_info_online = {"num_actions": k, "batch_size": 1, "alpha": alpha}

exp1 = BanditWrapper(env, agent)
batch_res.append(exp1.get_average_performance(agent_info_batch, env_info, exper_info))
online_res.append(exp1.get_average_performance(agent_info_online, env_info, exper_info))

av_online_res = np.mean(online_res, axis=0)
av_batch_res = np.mean(batch_res, axis=0)

plt.plot(av_batch_res, label='batch')
plt.plot(av_online_res, label='online')

M = int(num_steps / batch)
update_points = np.ceil(np.arange(num_steps) / batch).astype(int)
plt.plot(av_online_res[update_points] * batch, ls='--',
         label='upper bound, batch size = 10')
plt.title('Cumulative Regret averaged over ' + str(num_runs) + ' runs')
plt.xlabel('time steps')
plt.ylabel('regret')
plt.grid(b=True, which='major', linestyle='--', alpha=0.5)
plt.minorticks_on()
plt.grid(b=True, which='minor', linestyle=':', alpha=0.2)
plt.legend()
if if_save:
    plt.savefig('results/UCB transform example.png', bbox_inches='tight')
plt.show()

if if_save:
    name = 'batch_result, runs=' + str(num_runs) + ', steps=' + str(num_steps)
    with open('results/' + '/' + name + '.pickle', 'wb') as handle:
        pickle.dump(batch_res, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

<!-- #region id="82qKEl1F5dMB" -->
#### UCB dynamic by batches
<!-- #endregion -->

```python id="L3KoEguL5dIz"
model_dir = 'results/UCB/dynamic_by_batches'
if not os.path.exists(model_dir):
    print(f'Creating a new model directory: {model_dir}')
    os.makedirs(model_dir)

num_runs = 10 # 500
num_steps = 10001
seed = None
exper_info = {"num_runs": num_runs,
              "num_steps": num_steps,
              "seed": seed,
              "return_type": "regret"}

environments = [[0.7, 0.5], [0.7, 0.4], [0.7, 0.1],
                [0.35, 0.18, 0.47, 0.61],
                [0.4, 0.75, 0.57, 0.49],
                [0.70, 0.50, 0.30, 0.10]]

for arms_values in environments:
    k = len(arms_values)
    reward_type = 'Bernoulli'
    env_info = {"num_actions": k,
                "reward_type": reward_type,
                "arms_values": arms_values}
    env = Environment
    agent = UCBAgent
    alpha = 1

    # run online agent
    agent_info_online = {"num_actions": k, "batch_size": 1, "alpha": alpha}
    experiment = BanditWrapper(env, agent)
    online_regret = experiment.get_average_performance(agent_info_online, env_info, exper_info)

    # run batch agent
    batches = np.logspace(1.0, 3.0, num=20).astype(int)
    actual_regret = []
    upper_bound = []

    for batch in batches:
        agent_info_batch = {"num_actions": k, "batch_size": batch, "alpha": alpha}
        experiment = BanditWrapper(env, agent)
        batch_regret = experiment.get_average_performance(agent_info_batch, env_info, exper_info)
        actual_regret.append(batch_regret[-1])
        M = int(num_steps / batch)
        upper_bound.append(online_regret[M] * batch)

    # save data
    name = 'dyn_by_batch_' + str(arms_values)
    name1 = name + ' batch_regret'
    with open(model_dir + '/' + name1 + '.pickle', 'wb') as handle:
        pickle.dump(actual_regret, handle, protocol=pickle.HIGHEST_PROTOCOL)

    name2 = name + ' online_regret'
    with open(model_dir + '/' + name2 + '.pickle', 'wb') as handle:
        pickle.dump(online_regret, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("End!")
```

<!-- #region id="t618mNgBxKKM" -->
### TS
<!-- #endregion -->

<!-- #region id="oNTgxitH5Dy_" -->
#### TS dynamic by timesteps
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 359, "referenced_widgets": ["959b3e666a064b6d8917c8625ecd8fee", "ba3a1e89d28342f683d0013065137cb2", "53079098231c4906af72b59e6773052f", "3415712fbd16477b8eb7bf6844c56d51", "7269023608aa4b9c833466b4634a02cd", "23ca025a590c4caba147cfee82b8fb82", "6892149d580c4521bfbc8ce341445d61", "0767ce00fdc24b709c3ca2169e13e545", "c11a3dcd03f24240826793c233950bea", "ff5a4612a01b47a8a805ca8a643956c3", "f8e2be434a1846c797556912bad038f1", "85a8b0c33c104b8eb22fe9e21dc3d543", "f8a6e8d21951434b84789ff596bf188b", "3825eaf811174f30b38725c44d1536a6", "f0fad1d578be4cb790bcbc91e2088d85", "367d61cced8a4b7a9eb3763f11856cce", "0655f854114a4705ba29eabd0cc99e31", "f06c08cde5e14462a35c9b411e3cb167", "d769f331f73d42a1ae05b69c07023890", "52ec6649d8e94615bf9397cbd9d85b48", "fa0bd788f3a8421d949951e60f790c07", "2464f2977ba04e8a855461a081005305"]} id="9POfXn2TxLWQ" executionInfo={"status": "ok", "timestamp": 1639148363702, "user_tz": -330, "elapsed": 55499, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="15423772-d107-4547-f6fa-59c4a54546e3"
env = Environment
agent = TSAgent

num_runs = 10 # 1000
num_steps = 10000
seed = None
if_save = False
exper_info = {"num_runs": num_runs,
              "num_steps": num_steps,
              "seed": seed,
              "return_type": "regret"}

k = 2
arms_values = [0.7, 0.65]
reward_type = 'Bernoulli'
env_info = {"num_actions": k,
            "reward_type": reward_type,
            "arms_values": arms_values}

# batch-online experiment
batch_res = []
online_res = []
batch = 10
agent_info_batch = {"num_actions": k, "batch_size": batch}
agent_info_online = {"num_actions": k, "batch_size": 1}

exp1 = BanditWrapper(env, agent)
batch_res.append(exp1.get_average_performance(agent_info_batch, env_info, exper_info))
online_res.append(exp1.get_average_performance(agent_info_online, env_info, exper_info))

av_online_res = np.mean(online_res, axis=0)
av_batch_res = np.mean(batch_res, axis=0)

plt.plot(av_batch_res, label='batch')
plt.plot(av_online_res, label='online')

M = int(num_steps / batch)
update_points = np.ceil(np.arange(num_steps) / batch).astype(int)
plt.plot(av_online_res[update_points] * batch, ls='--',
         label='upper bound, batch size = 10')
plt.title('Cumulative Regret averaged over ' + str(num_runs) + ' runs')
plt.xlabel('time steps')
plt.ylabel('regret')
plt.grid(b=True, which='major', linestyle='--', alpha=0.5)
plt.minorticks_on()
plt.grid(b=True, which='minor', linestyle=':', alpha=0.2)
plt.legend()
if if_save:
    plt.savefig('results/TS  example.png', bbox_inches='tight')
plt.show()

if if_save:
    name = 'batch_result, runs=' + str(num_runs) + ', steps=' + str(num_steps)
    with open('results/' + '/' + name + '.pickle', 'wb') as handle:
        pickle.dump(batch_res, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

<!-- #region id="koXASf5O5GR2" -->
#### TS dynamic by batches
<!-- #endregion -->

```python id="T-CjxemW5GNq"
model_dir = 'results/TS/dynamic_by_batches'
if not os.path.exists(model_dir):
    print(f'Creating a new model directory: {model_dir}')
    os.makedirs(model_dir)

num_runs = 10 # 500
num_steps = 10001
seed = None
exper_info = {"num_runs": num_runs,
              "num_steps": num_steps,
              "seed": seed,
              "return_type": "regret"}

environments = [[0.7, 0.5], [0.7, 0.4], [0.7, 0.1],
                [0.35, 0.18, 0.47, 0.61],
                [0.4, 0.75, 0.57, 0.49],
                [0.70, 0.50, 0.30, 0.10]]

for arms_values in environments:
    k = len(arms_values)
    reward_type = 'Bernoulli'
    env_info = {"num_actions": k,
                "reward_type": reward_type,
                "arms_values": arms_values}
    env = Environment
    agent = TSAgent

    # run online agent
    agent_info_online = {"num_actions": k, "batch_size": 1}
    experiment = BanditWrapper(env, agent)
    online_regret = experiment.get_average_performance(agent_info_online, env_info, exper_info)

    # run batch agent
    batches = np.logspace(1.0, 3.0, num=20).astype(int)
    actual_regret = []
    upper_bound = []

    for batch in batches:
        agent_info_batch = {"num_actions": k, "batch_size": batch}
        experiment = BanditWrapper(env, agent)
        batch_regret = experiment.get_average_performance(agent_info_batch, env_info, exper_info)
        actual_regret.append(batch_regret[-1])
        M = int(num_steps / batch)
        upper_bound.append(online_regret[M] * batch)

    # save data
    name = 'dyn_by_batch_' + str(k) + str(arms_values)
    name1 = name + ' batch_regret'
    with open(model_dir + '/' + name1 + '.pickle', 'wb') as handle:
        pickle.dump(actual_regret, handle, protocol=pickle.HIGHEST_PROTOCOL)

    name2 = name + ' online_regret'
    with open(model_dir + '/' + name2 + '.pickle', 'wb') as handle:
        pickle.dump(online_regret, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("End!")
```

<!-- #region id="L-4KgfMU0iE9" -->
### LinUCB
<!-- #endregion -->

<!-- #region id="elLqw6xk4083" -->
#### LinUCB by timesteps
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 359, "referenced_widgets": ["07d74cc8c1034e8e88e0ba68681e7d83", "20755197a75045769c76c25cf9997309", "3aaecc6ab5334dbaaff91f3fd3d2dc92", "ccdfd084e55f4d48aa8d62ff900c54dc", "602801fccd6d4538bd22a5494c8a538e", "81b219b6e4414ff888646a18f038c6d2", "4622f6b8c7914be18ec086e119b14510", "c603ecbfec974667805abf5f8366dde5", "508403ada600443f97467dde6fb0dd3d", "9840772f43134cee8eabacd2509f9fd7", "4e96a7f4873f4360b2118e7a3ddbfc48", "c19256a397ea4e79b3a0cbf5a424a3ea", "4d1f5676bbaa4b3d8b81cc7be3cf6da3", "eae2220f0119401a8759637bcf0e9f4c", "4aa44dd37d5243dfa24fea8f3dc6839b", "2deab0a58cd04226b834a99ee4ff4c90", "16acd859901c4b8aa5f6b4473b74e249", "e4edb80a747a4e9684a556ff96563e7f", "3d74bbdd72dc4e0aadb4bd517358bbe7", "809de5d470244a2b90daffd0477c28a4", "4e13cae2ba7a44daa51794431d7af0c0", "2780d1d40a0644c498e860cf4839ac25"]} id="tXey4Vyi06pm" executionInfo={"status": "ok", "timestamp": 1639148649109, "user_tz": -330, "elapsed": 32385, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="77e181e9-b27e-474c-834d-4db5a099aeb1"
num_experiments = 20
batch_size = 100
data_dir = 'data/mushroom_data_final.pickle'
env_info = {'pickle_file': data_dir}
output_dir = 'LinUCB/dynamic_by_timesteps'

agent_info = {'alpha': 2,
              'num_actions': 3,
              'seed': 1,
              'batch_size': 1}
agent_info_batch = {'alpha': 2,
                    'num_actions': 3,
                    'seed': 1,
                    'batch_size': batch_size}
experiment_parameters = {"num_runs": num_experiments}

agent = LinUCBAgent
environment = ReplayEnvironment

online_result = run_experiment(environment, agent, env_info, agent_info,
                               experiment_parameters, True, output_dir)
batch_result = run_experiment(environment, agent, env_info, agent_info_batch,
                              experiment_parameters, True, output_dir)

smoothed_leveled_result = smooth(online_result, 100)
smoothed_leveled_result1 = smooth(batch_result, 100)

mean_smoothed_leveled_result = np.mean(smoothed_leveled_result, axis=0)
mean_smoothed_leveled_result1 = np.mean(smoothed_leveled_result1, axis=0)

num_steps = np.minimum(len(mean_smoothed_leveled_result), len(mean_smoothed_leveled_result1))
update_points = np.ceil(np.arange(num_steps) / batch_size).astype(int)

pic_filename = "results/{}/UCB_transform_timesteps.png".format(output_dir)
plt.plot(mean_smoothed_leveled_result1, lw=3, label='batch, batch size = ' + str(batch_size))
plt.plot(mean_smoothed_leveled_result, lw=3, ls='-.', label='online policy')
plt.plot(mean_smoothed_leveled_result[update_points], lw=3, ls='-.', label='dumb policy')
plt.legend()
plt.xlabel('time steps')
plt.title("Smooth Cumulative Reward averaged over {} runs".format(num_experiments))
plt.ylabel('smoothed reward')
plt.grid(b=True, which='major', linestyle='--', alpha=0.5)
plt.minorticks_on()
plt.grid(b=True, which='minor', linestyle=':', alpha=0.2)
plt.savefig(pic_filename, bbox_inches='tight')
plt.show()
```

<!-- #region id="q7xl1u_F45d5" -->
#### LinUCB by batches
<!-- #endregion -->

```python id="XxBGmxib45aV"
num_experiments = 20
data_dir = 'data/mushroom_data_final.pickle'
env_info = {'pickle_file': data_dir}
output_dir = 'LinUCB/dynamic_by_batches'

agent_info = {'alpha': 2,
              'num_actions': 3,
              'seed': 1,
              'batch_size': 1}
experiment_parameters = {"num_runs": num_experiments}

agent = LinUCBAgent
environment = ReplayEnvironment

# run online agent
online_result = run_experiment(environment, agent, env_info, agent_info,
                               experiment_parameters, True, output_dir)
# smooth and average the result
smoothed_leveled_result = smooth(online_result, 100)
mean_smoothed_leveled_result = np.mean(smoothed_leveled_result, axis=0)
mean_smoothed_leveled_result = mean_smoothed_leveled_result[~np.isnan(mean_smoothed_leveled_result)]

# run batch agent
batch_sizes = np.logspace(1.0, 2.7, num=20).astype(int)
actual_regret = []
upper_bound = []
for batch in batch_sizes:
    agent_info_batch = {'alpha': 2,
                        'num_actions': 3,
                        'seed': 1,
                        'batch_size': batch}
    batch_result = run_experiment(environment, agent, env_info, agent_info_batch,
                                  experiment_parameters, True, output_dir)
    # smooth and average the result
    smoothed_leveled_result1 = smooth(batch_result, 100)
    mean_smoothed_leveled_result1 = np.mean(smoothed_leveled_result1, axis=0)
    mean_smoothed_leveled_result1 = mean_smoothed_leveled_result1[~np.isnan(mean_smoothed_leveled_result1)]

    actual_regret.append(mean_smoothed_leveled_result1[-1])

    # fetch dumb result
    M = int(len(mean_smoothed_leveled_result1) / batch)
    upper_bound.append(mean_smoothed_leveled_result[M])

pic_filename = "results/{}/UCB_transform_batchsize.png".format(output_dir)
plt.plot(batch_sizes, actual_regret, label='actual regret')
plt.plot(batch_sizes, [mean_smoothed_leveled_result[-1]]*len(batch_sizes), label='online policy')
plt.plot(batch_sizes, upper_bound, label='dumb policy')
plt.legend()
plt.title("Reward as a f-n of batch size (each point is averaged over {} runs)".format(num_experiments))
plt.xlabel('batch size (log scale)')
plt.ylabel('reward')
plt.grid(b=True, which='major', linestyle='--', alpha=0.5)
plt.minorticks_on()
plt.grid(b=True, which='minor', linestyle=':', alpha=0.2)
plt.savefig(pic_filename, bbox_inches='tight')
plt.show()
```

<!-- #region id="zaZs-8O31BHz" -->
### LinTS
<!-- #endregion -->

<!-- #region id="apXIaCEY4nih" -->
#### LinTS by timesteps
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 359, "referenced_widgets": ["a2c68a56898548f8a7e190dcf2d034f4", "e009631910e042a2a07afe8cf0e6d118", "a850623290b14d07838e4203a29f9ba2", "9f140b090cb44361b39a6e09205c087c", "36fc4560cb99439c945c620138005a1b", "638ee1bca5d64cc7966556e3fb64c06e", "a454e1e1460046e3a98a4b28c48ecda0", "d0a15e2054af430bad52b6fc28fd0994", "f9295dd431fe4587a816d6c59f464399", "41ca699547f14bda8a83777d78e6bc86", "c29bc574f0cd46309d8eb6ec40378025", "858e460d943744248a4109a5c328d77c", "700c0953693b48bea968c6c67964684d", "c70158af000c40e59b588e745be4a2dd", "21b1b038a25f469295ea8e2b75c470ae", "c20bc14a583e452ba263c4e4f4f3147e", "e1199401ceee4d0ca1aaa72020618241", "57b794458691470ba34efd35f0cd8a9c", "4bd9a0cc11664bf4870c7fc46d151bda", "301a50e882864065b9108e0a0cf0edde", "93f8c74e75504aadbf1cdec1c5b6a43f", "a4d3c862e74242e4beeaaa880f4e3fc8"]} id="6uyZk7PT1Cjy" executionInfo={"status": "ok", "timestamp": 1639149620282, "user_tz": -330, "elapsed": 952413, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="72dd4d89-25ac-408a-c033-d05d6bee7628"
num_experiments = 10
batch_size = 100
data_dir = 'data/mushroom_data_final.pickle'
env_info = {'pickle_file': data_dir}
output_dir = 'LinTS/dynamic_by_timesteps'

agent_info = {'alpha': 1,
              'num_actions': 3,
              'seed': 1,
              'batch_size': 1,
              'replay_buffer_size': 100000}
agent_info_batch = {'alpha': 1,
                    'num_actions': 3,
                    'seed': 1,
                    'batch_size': batch_size,
                    'replay_buffer_size': 100000}
experiment_parameters = {"num_runs": num_experiments}

agent = LinTSAgent
environment = ReplayEnvironment

online_result = run_experiment(environment, agent, env_info, agent_info,
                               experiment_parameters, True, output_dir)
batch_result = run_experiment(environment, agent, env_info, agent_info_batch,
                              experiment_parameters, True, output_dir)

smoothed_leveled_result = smooth(online_result, 100)
smoothed_leveled_result1 = smooth(batch_result, 100)

mean_smoothed_leveled_result = np.mean(smoothed_leveled_result, axis=0)
mean_smoothed_leveled_result1 = np.mean(smoothed_leveled_result1, axis=0)

num_steps = np.minimum(len(mean_smoothed_leveled_result), len(mean_smoothed_leveled_result1))
update_points = np.ceil(np.arange(num_steps) / batch_size).astype(int)

pic_filename = "results/{}/TS_transform_timesteps.png".format(output_dir)
plt.plot(mean_smoothed_leveled_result1, lw=3, label='batch, batch size = ' + str(batch_size))
plt.plot(mean_smoothed_leveled_result, lw=3, ls='-.', label='online policy')
plt.plot(mean_smoothed_leveled_result[update_points], lw=3, ls='-.', label='dumb policy')
plt.legend()
plt.xlabel('time steps')
plt.title("Smooth Cumulative Reward averaged over {} runs".format(num_experiments))
plt.ylabel('smoothed reward')
plt.grid(b=True, which='major', linestyle='--', alpha=0.5)
plt.minorticks_on()
plt.grid(b=True, which='minor', linestyle=':', alpha=0.2)
plt.savefig(pic_filename, bbox_inches='tight')
plt.show()
```

<!-- #region id="pVbE43Lm4p5H" -->
#### LinTS by batches
<!-- #endregion -->

```python id="K63E-61k4p03"
num_experiments = 20
data_dir = 'data/mushroom_data_final.pickle'
env_info = {'pickle_file': data_dir}
output_dir = 'LinTS/dynamic_by_batches'

agent_info = {'alpha': 1,
              'num_actions': 3,
              'seed': 1,
              'batch_size': 1,
              'replay_buffer_size': 100000}
experiment_parameters = {"num_runs": num_experiments}

agent = LinTSAgent
environment = ReplayEnvironment

# run online agent
online_result = run_experiment(environment, agent, env_info, agent_info,
                               experiment_parameters, True, output_dir)
# smooth and average the result
smoothed_leveled_result = smooth(online_result, 100)
mean_smoothed_leveled_result = np.mean(smoothed_leveled_result, axis=0)
mean_smoothed_leveled_result = mean_smoothed_leveled_result[~np.isnan(mean_smoothed_leveled_result)]

# run batch agent
batch_sizes = np.logspace(1.0, 2.7, num=20).astype(int)
actual_regret = []
upper_bound = []
for batch in batch_sizes:
    agent_info_batch = {'alpha': 1,
                        'num_actions': 3,
                        'seed': 1,
                        'batch_size': batch,
                        'replay_buffer_size': 100000}
    batch_result = run_experiment(environment, agent, env_info, agent_info_batch,
                                  experiment_parameters, True, output_dir)
    # smooth and average the result
    smoothed_leveled_result1 = smooth(batch_result, 100)
    mean_smoothed_leveled_result1 = np.mean(smoothed_leveled_result1, axis=0)
    mean_smoothed_leveled_result1 = mean_smoothed_leveled_result1[~np.isnan(mean_smoothed_leveled_result1)]

    actual_regret.append(mean_smoothed_leveled_result1[-1])

    # fetch dumb result
    M = int(len(mean_smoothed_leveled_result1) / batch)
    upper_bound.append(mean_smoothed_leveled_result[M])

pic_filename = "results/{}/TS_transform_batchsize.png".format(output_dir)
plt.plot(batch_sizes, actual_regret, label='actual regret')
plt.plot(batch_sizes, [mean_smoothed_leveled_result[-1]]*len(batch_sizes), label='online policy')
plt.plot(batch_sizes, upper_bound, label='dumb policy')
plt.legend()
plt.title("Reward as a f-n of batch size (each point is averaged over {} runs)".format(num_experiments))
plt.xlabel('batch size (log scale)')
plt.ylabel('reward')
plt.grid(b=True, which='major', linestyle='--', alpha=0.5)
plt.minorticks_on()
plt.grid(b=True, which='minor', linestyle=':', alpha=0.2)
plt.savefig(pic_filename, bbox_inches='tight')
plt.show()
```

<!-- #region id="w5S8ABzm1JMM" -->
### CMAB demo on Mushroom dataset
<!-- #endregion -->

```python id="pHKrIm_Q1R7S"
data_dir = 'data/mushroom_data_final.pickle'
env_info = {'pickle_file': data_dir,
            'seed': 1}
# init env
environment = ReplayEnvironment

# init random agent
random_agent_info = {'num_actions': 2}
ra = RandomAgent()
ra.agent_init(random_agent_info)

# learn LinUCB agent
agent_info = {'alpha': 2,
              'num_actions': 2,
              'seed': 1,
              'batch_size': 1}

agent = LinUCBAgent
rl_glue = RLGlue(environment, agent)

for i in range(4):    
    rl_glue.rl_init(agent_info, env_info)
    rl_glue.rl_episode(0)
UCB_agent = rl_glue.agent

# learn LinTS agent
agent_info = {'num_actions': 2,
              'replay_buffer_size': 200,
              'seed': 1,
              'batch_size': 1}
agent = LinTSAgent
rl_glue = RLGlue(environment, agent)

for i in range(4):    
    rl_glue.rl_init(agent_info, env_info)
    rl_glue.rl_episode(0)

TS_agent = rl_glue.agent
result = []
result1 = []
result2 = []

exper_seeds = [2, 5, 10, 12, 54, 32, 15, 76, 45, 56]
for seed_ in exper_seeds:
    dataset = BanditDataset(pickle_file=data_dir, seed=seed_)

    eval_info = {'dataset': dataset, 'agent': UCB_agent}
    eval_info1 = {'dataset': dataset, 'agent': TS_agent}
    eval_info2 = {'dataset': dataset, 'agent': ra}

    evaluator = OfflineEvaluator(eval_info)
    evaluator1 = OfflineEvaluator(eval_info1)
    evaluator2 = OfflineEvaluator(eval_info2)

    reward = evaluator.eval_run()
    reward1 = evaluator1.eval_run()
    reward2 = evaluator2.eval_run()

    result.append(reward)
    result1.append(reward1)
    result2.append(reward2)

labels = ['UCB agent', 'TS agent', 'Random agent']
for i, res in enumerate([result, result1, result2]):
    for elem in res:
        plt.plot(elem, linewidth=0.1)
    avg = [float(sum(col))/len(col) for col in zip(*res)]
    plt.plot(avg, label=labels[i])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 269} id="ZbxPyHo_1Zgo" executionInfo={"status": "ok", "timestamp": 1639152031963, "user_tz": -330, "elapsed": 1780, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0b963035-788e-4b43-b73d-9ee1df8e7cc7"
labels = ['UCB agent', 'TS agent', 'Random agent']
for i, res in enumerate([result, result1, result2]):
    for elem in res:
        plt.plot(elem, linewidth=0.1)
    avg = [float(sum(col))/len(col) for col in zip(*res)]
    plt.plot(avg, label=labels[i])
plt.legend()
plt.ylim([0.1, 0.7])
plt.grid(b=True, which='major', linestyle='--', alpha=0.5)
plt.minorticks_on()
plt.grid(b=True, which='minor', linestyle=':', alpha=0.2)
plt.show()
```

<!-- #region id="o_S529BK1qop" -->
### CMAB demo on Simulated dataset
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="8OOtls5G1qlT" executionInfo={"status": "ok", "timestamp": 1639149837227, "user_tz": -330, "elapsed": 4833, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="597535e2-9ebb-4bce-a443-6c90344656dc"
# generate 100 000 samples with 4 features and 3 actions
dataset = generate_samples(100000, 4, 3, True)
dataset.head()
```

<!-- #region id="h9x6zOMx1qdr" -->
#### LinTS dynamic by steps
<!-- #endregion -->

```python id="SDmmh_Js1qag"
num_experiments = 10
batch_size1 = 30
batch_size2 = 100
env_info = {'pickle_file': dataset}

agent1_info = {'alpha': 1,
              'num_actions': 3,
              'seed': 1,
              'batch_size': batch_size1,
              'replay_buffer_size': 100000}
agent2_info = {'alpha': 1,
                    'num_actions': 3,
                    'seed': 1,
                    'batch_size': batch_size2,
                    'replay_buffer_size': 100000}
experiment_parameters = {"num_runs": num_experiments}
```

```python colab={"base_uri": "https://localhost:8080/", "height": 81, "referenced_widgets": ["79e9b193d9284e4b898cb4344db92ef3", "8ea51857d7cf48c1beddb0a9bd3b764b", "7b0e4960186b4307ae594b269043c16d", "aa39361015a84c158e922f8e0b8dca9a", "63acbf6a68e6455c91cbea40169d1ee9", "e2c2139f96b345768e075ef3593fac06", "3dcbffa6176e4824b9a48cccfaf024cc", "740c9d9905b542d4a0edef78f1d8ca30", "cd2dc9d9c7504cd4b118a33b7c24f683", "33707c49e4fd42ad932dd74e59f4a060", "60745768aedc4942bad8883d708b0e25", "48077097754d4b96b1345fb1c06560b8", "760943cea30f4236a0715b61a047fd99", "3bf9f4be47424b13b1db937f67894360", "28c1b766f56c41179840f2140da35b44", "487beca1cfc847c0a41a414968097b6b", "c28b5aecd2eb4353b99534e8eef2380c", "f167c63d8ac54b07b31b261c77a98391", "4a8e550188ac481fa2838df0996d3b0b", "50dfc1413ebd4298af9fc979de4ebb63", "f36eb7ce107741159cfc86169724580f", "5b4edf526246433d8cca04eba8be3229"]} id="Sp9wGAPS1-V-" executionInfo={"status": "ok", "timestamp": 1639150020202, "user_tz": -330, "elapsed": 182996, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b1fc5a37-fb85-44dc-cb62-57e3a9813b63"
agent = LinTSAgent
environment = ReplayEnvironment

result1 = run_experiment(environment, agent, env_info, agent1_info, experiment_parameters, False)
result2 = run_experiment(environment, agent, env_info, agent2_info, experiment_parameters, False)

smoothed_leveled_result1 = smooth(result1, 100)
smoothed_leveled_result2 = smooth(result2, 100)

mean_smoothed_leveled_result1 = np.mean(smoothed_leveled_result1, axis=0)
mean_smoothed_leveled_result2 = np.mean(smoothed_leveled_result2, axis=0)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 295} id="t2CezR4D2AeW" executionInfo={"status": "ok", "timestamp": 1639150021342, "user_tz": -330, "elapsed": 1192, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="dedc8712-71e2-4c3e-f805-6c5902a2de89"
plt.plot(mean_smoothed_leveled_result1, label='batch size = ' + str(batch_size1))
plt.plot(mean_smoothed_leveled_result2, label='batch size = ' + str(batch_size2))
plt.legend()
plt.xlabel('time steps')
plt.title("Smooth Cumulative Reward averaged over {} runs".format(num_experiments))
plt.ylabel('smoothed conversion rate')
plt.grid(b=True, which='major', linestyle='--', alpha=0.5)
plt.minorticks_on()
plt.grid(b=True, which='minor', linestyle=':', alpha=0.2)
plt.show()
```

<!-- #region id="B3W58Zqv2CnH" -->
#### LinTS dynamic by batches
<!-- #endregion -->

```python id="Vs_O6HE92Ckv"
num_experiments = 20
env_info = {'pickle_file': dataset}
experiment_parameters = {"num_runs": num_experiments}

agent = LinTSAgent
environment = ReplayEnvironment

# run batch agent
batch_sizes = np.logspace(1.0, 2.7, num=20).astype(int)
actual_regret = []
for batch in batch_sizes:
    agent_info_batch = {'alpha': 1,
                        'num_actions': 3,
                        'seed': 1,
                        'batch_size': batch,
                        'replay_buffer_size': 100000}
    batch_result = run_experiment(environment, agent, env_info, agent_info_batch,
                                  experiment_parameters, False)
    # smooth and average the result
    smoothed_leveled_result1 = smooth(batch_result, 100)
    mean_smoothed_leveled_result1 = np.mean(smoothed_leveled_result1, axis=0)
    mean_smoothed_leveled_result1 = mean_smoothed_leveled_result1[~np.isnan(mean_smoothed_leveled_result1)]

    actual_regret.append(mean_smoothed_leveled_result1[-1])
```

```python id="4iErUrNt2CiB"
plt.plot(batch_sizes, actual_regret, label='actual regret')
plt.legend()
plt.title("Reward as a f-n of batch size (each point is averaged over {} runs)".format(num_experiments))
plt.xlabel('batch size (log scale)')
plt.ylabel('reward')
plt.grid(b=True, which='major', linestyle='--', alpha=0.5)
plt.minorticks_on()
plt.grid(b=True, which='minor', linestyle=':', alpha=0.2)
plt.show()
```

```python id="ajBoPT1U2CfL"

```
