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

```python id="UV_mis-jdwLd" executionInfo={"status": "ok", "timestamp": 1629280163592, "user_tz": -330, "elapsed": 718, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-mlh"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python id="KRGLEjqMd3dV" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629280170044, "user_tz": -330, "elapsed": 5766, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="15420db8-787b-4b5d-dfc9-e67a16def521"
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

```python colab={"base_uri": "https://localhost:8080/"} id="Aa6AQmftAovn" executionInfo={"status": "ok", "timestamp": 1629281660973, "user_tz": -330, "elapsed": 612, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="35a506ea-c9be-47a9-d10d-cd6c6478efb2"
!git status
```

```python colab={"base_uri": "https://localhost:8080/"} id="aG5PN_2EAovn" executionInfo={"status": "ok", "timestamp": 1629281665362, "user_tz": -330, "elapsed": 1475, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="29c81915-94e5-416c-8850-a47260dff894"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python id="HDaN0hpStJVT" executionInfo={"status": "ok", "timestamp": 1629281003527, "user_tz": -330, "elapsed": 605, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import sys
sys.path.insert(0,'./code')
```

<!-- #region id="xH7kkcmQqmIu" -->
---
<!-- #endregion -->

```python id="Rnu09KFktDWl" executionInfo={"status": "ok", "timestamp": 1629280823675, "user_tz": -330, "elapsed": 716, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
%reload_ext autoreload
%autoreload 2
```

```python id="AdGxtDldrF0K" executionInfo={"status": "ok", "timestamp": 1629280304821, "user_tz": -330, "elapsed": 1430, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
from datetime import datetime
from functools import reduce
from os.path import exists
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import gc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys

from models.ALS import ALSRecommender
from models.baselines import DampedUserMovieBaselineModel
```

```python colab={"base_uri": "https://localhost:8080/", "height": 442} id="Yyn6I68UrSTZ" executionInfo={"status": "ok", "timestamp": 1629280448858, "user_tz": -330, "elapsed": 761, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cd8d6bc6-88f9-49da-fa2d-5f1c75b33ae1"
ratings_df = pd.read_csv('./data/bronze/u.data', sep='\t', header=None, 
                         names=['userId', 'movieId', 'rating', 'timestamp'])
ratings_df['timestamp'] = ratings_df['timestamp'].apply(datetime.fromtimestamp)
ratings_df = ratings_df.sort_values('timestamp')

print('First 5:')
display(ratings_df.head())
print()
print('Last 5:')
display(ratings_df.tail())
```

<!-- #region id="g8wB3jHfq_w0" -->
### Determine how many epochs are necessary
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ROYMhgPGsNib" executionInfo={"status": "ok", "timestamp": 1629281342773, "user_tz": -330, "elapsed": 165394, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3926a3ca-dd4d-4f7d-f2fb-2cfad42373e3"
%%time
n_splits = 5
max_epochs = 50
kf = KFold(n_splits=n_splits, random_state=0, shuffle=True)
train_errs = np.zeros((n_splits, max_epochs))
test_errs = np.zeros((n_splits, max_epochs))

for i_fold, (train_inds, test_inds) in enumerate(kf.split(ratings_df)):
    print("i_fold={}".format(i_fold))
    train_df, test_df = ratings_df.iloc[train_inds], ratings_df.iloc[test_inds]
    baseline_algo = DampedUserMovieBaselineModel(damping_factor=10)
    rec = ALSRecommender(k=20, baseline_algo=baseline_algo, verbose=False, max_epochs=max_epochs)
    for i_epoch in range(max_epochs):
        rec.fit(train_df, n_epochs=1)
        preds = rec.predict(test_df[['userId', 'movieId']])
        test_err = mean_absolute_error(preds, test_df['rating'])
        # print("[Epoch {}/{}] test MAE: {}".format(i_epoch, max_epochs, test_err))
        test_errs[i_fold, i_epoch] = test_err
    train_errs[i_fold, :] = np.array(rec.train_errors)

print(f"There are {len(rec.user_map)} users and {len(rec.item_map)} items")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 297} id="ZbL97R36ucsT" executionInfo={"status": "ok", "timestamp": 1629281349831, "user_tz": -330, "elapsed": 1182, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4ad3c6ad-c7bb-478e-876b-3c639d1e2836"
train_avg, train_std = train_errs.mean(axis=0), train_errs.std(axis=0)
test_avg, test_std = test_errs.mean(axis=0), test_errs.std(axis=0)
l, = plt.plot(np.arange(max_epochs), train_avg, label='Train Error')
plt.fill_between(np.arange(max_epochs), train_avg-train_std, train_avg+train_std,
                 color=l.get_color(), alpha=0.3)
l, = plt.plot(np.arange(max_epochs), test_avg, label='Test Error')
plt.fill_between(np.arange(max_epochs), test_avg-test_std, test_avg+test_std,
                 color=l.get_color(), alpha=0.3)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title(r'Errors vs Epoch for $k=20$, $n_{ratings}=100K$')
plt.show()
```

<!-- #region id="aBUQ2cMCu7W2" -->
> Note: 15 or 20 epochs seems like enough for Test Error to start plateauing.
<!-- #endregion -->

<!-- #region id="xmpgnWsvu8j_" -->
### Find optimal K
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="nNF_6CoKvBm3" executionInfo={"status": "ok", "timestamp": 1629281641107, "user_tz": -330, "elapsed": 261895, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="813fd0c7-bf8d-4765-dffb-1e4e8a4b0b1a"
n_splits = 5
max_epochs = 15
kf = KFold(n_splits=n_splits, random_state=0, shuffle=True)
k_list = [1, 2, 5, 10, 20, 50, 100]
small_df = ratings_df.iloc[:100000]
train_errs = np.zeros((n_splits, len(k_list)))
test_errs = np.zeros((n_splits, len(k_list)))

for i_fold, (train_inds, test_inds) in enumerate(kf.split(small_df)):
    print("i_fold={}: ".format(i_fold), end='')
    train_df, test_df = small_df.iloc[train_inds], small_df.iloc[test_inds]
    baseline_algo = DampedUserMovieBaselineModel(damping_factor=10)
    for i_k, k in enumerate(k_list):
        print("k={}, ".format(k), end='')
        rec = ALSRecommender(k=k, baseline_algo=baseline_algo, verbose=False, max_epochs=max_epochs)
        rec.fit(train_df)
        preds = rec.predict(test_df[['userId', 'movieId']])
        test_err = mean_absolute_error(preds, test_df['rating'])
        test_errs[i_fold, i_k] = test_err
        train_errs[i_fold, i_k] = rec.train_errors[-1]
    print()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 299} id="lnsy9PBCvEzw" executionInfo={"status": "ok", "timestamp": 1629281642616, "user_tz": -330, "elapsed": 1521, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7d035715-aecc-4464-ebf7-15a1a3d0a285"
train_avg, train_std = train_errs.mean(axis=0), train_errs.std(axis=0)
test_avg, test_std = test_errs.mean(axis=0), test_errs.std(axis=0)
l, = plt.semilogx(k_list, train_avg, label='Train Error')
plt.fill_between(k_list, train_avg-train_std, train_avg+train_std,
                 color=l.get_color(), alpha=0.3)
l, = plt.semilogx(k_list, test_avg, label='Test Error')
plt.fill_between(k_list, test_avg-test_std, test_avg+test_std,
                 color=l.get_color(), alpha=0.3)
plt.xticks(k_list, k_list)
plt.legend()
plt.xlabel(r'$k$')
plt.ylabel('MAE')
plt.title(r'Errors vs $k$ after {} epochs'.format(max_epochs))
plt.show()
```

<!-- #region id="8G1ENCz9vPbS" -->
> Note: It looks like we have a Test Error minimum around k=10, although since k=5 is so close, we can go with that since it is likely to generalize better.
<!-- #endregion -->
