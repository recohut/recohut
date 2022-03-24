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

<!-- #region id="b2V_6GjQrQxL" -->
# Comparing Implicit Models on Synthetic Data
> Generating a small synthetic implicit data with Gamma distributed interaction and fitting ALS, BPR, PoissonMF, and HPFRec models

- toc: true
- badges: true
- comments: true
- categories: [Implicit, SyntheticDataset]
- author: "<a href='https://github.com/david-cortes'>David Cortes</a>"
- image:
<!-- #endregion -->

<!-- #region id="jf8Ogz1Ug3hF" -->
### Installation
<!-- #endregion -->

```python id="cesa4F19ftIC"
# !pip install --no-use-pep517 poismf implicit hpfrec
```

```python id="QeRhQj-LecfP"
import numpy as np
import pandas as pd

from scipy.sparse import coo_matrix, csr_matrix, csc_matrix

from poismf import PoisMF
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from hpfrec import HPF ### <- Bayesian version
```

<!-- #region id="SC8h0jkqpBuv" -->
### Synthetic Data Generation
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="NkdkettffLTh" outputId="a7cf99f4-37da-4145-ffbb-2ecfb22f84bc"
## Generating random sparse data
nusers = 10 ** 2
nitems = 10 ** 3
nnz    = 10 ** 4

np.random.seed(1)
df = pd.DataFrame({
    'UserId' : np.random.randint(nusers, size = nnz),
    'ItemId' : np.random.randint(nitems, size = nnz),
    'Count'  : 1 + np.random.gamma(1, 1, size = nnz).astype(int)
})

df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 142} id="5Iaq0tkLff7b" outputId="22c92f66-a4d5-4410-ef35-a395bb7cf144"
df.describe().T
```

<!-- #region id="S_IzUa6Bo8yO" -->
### Train/Test Split
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="x5ugJgBJi4ri" outputId="cf265e97-0a63-4b84-b093-bcb5d6385805"
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=.2)
df_train = df_train.copy()
users_train = np.unique(df_train.UserId.to_numpy())
items_train = np.unique(df_train.ItemId.to_numpy())
df_test = df_test.loc[df_test.UserId.isin(users_train) &
                      df_test.ItemId.isin(items_train)]
df_train["UserId"] = pd.Categorical(df_train.UserId, users_train).codes
df_train["ItemId"] = pd.Categorical(df_train.ItemId, items_train).codes
df_test["UserId"] = pd.Categorical(df_test.UserId, users_train).codes
df_test["ItemId"] = pd.Categorical(df_test.ItemId, items_train).codes
users_test = np.unique(df_test.UserId.to_numpy())

print("Number of entries in training data: {:,}".format(df_train.shape[0]))
print("Number of entries in test data: {:,}".format(df_test.shape[0]))
print("Number of users in training data: {:,}".format(users_train.shape[0]))
print("Number of users in test data: {:,}".format(users_test.shape[0]))
print("Number of items in training and test data: {:,}".format(items_train.shape[0]))
```

<!-- #region id="RJPvtKIzo4cI" -->
### Util function to print ranking metrics
<!-- #endregion -->

```python id="9_iUr3-xkfPr"
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

## Note: this is a computationally inefficient implementation of the
## test metrics, not recommended to use outside of this notebook
def print_ranking_metrics(A, B, df_train, df_test, users_test,
                          nusers=1000, top_n=5, seed=1,
                          njobs=-1):
    """
    Parameters
    ----------
    A : array(m, k)
        The user-factor matrix.
    B : array(n, k)
        The item-factor matrix
    df_train : DataFrame(n_train, [user, item, value])
        The training triplets.
    df_test : DataFrame(n_test, [user, item, value])
        The hold-out triplets.
    n_user : int
        Number of users to sample.
    top_n : int
        Number of top-ranked items to calculate precision.
    seed : int
        Random seed used to select the users.
    njobs : int
        Number of jobs to run in parallel.
    """
    n_users = A.shape[0]
    n_items = B.shape[0]
    rng = np.random.default_rng(seed=seed)
    chosen_users = rng.choice(users_test, size=nusers, replace=False)
    all_train = df_train.loc[df_train.UserId.isin(chosen_users)]
    all_test = df_test.loc[df_test.UserId.isin(chosen_users)]
    
    def metrics_single_user(user):
        ypos = all_test.ItemId.loc[all_test.UserId == user].to_numpy()
        ytrain = all_train.ItemId.loc[all_train.UserId == user].to_numpy()
        yneg = np.setdiff1d(np.arange(n_items), np.r_[ypos, ytrain])
        ytest = np.r_[yneg, ypos]
        yhat = B[ytest].dot(A[user])
        auc = roc_auc_score(np.r_[np.zeros(yneg.shape[0]),
                                  np.ones(ypos.shape[0])],
                            yhat)
        topN = np.argsort(-yhat)[:top_n]
        p_at_k = np.mean(topN >= yneg.shape[0])
        p_at_k_rnd = ypos.shape[0] / ytest.shape[0] ## <- baseline
        return auc, p_at_k, p_at_k_rnd

    res_triplets = Parallel(n_jobs = njobs)\
                    (delayed(metrics_single_user)(u) \
                        for u in chosen_users)

    res_triplets = np.array(res_triplets)
    auc = np.mean(res_triplets[:,0])
    p_at_k = np.mean(res_triplets[:,1])
    p_at_k_rnd = np.mean(res_triplets[:,2])
    print("AUC: %.4f [random: %.2f]" % (auc, 0.5))
    print("P@%d: %.4f [random: %.4f]" % (top_n,
                                         p_at_k,
                                         p_at_k_rnd))
```

<!-- #region id="T2dQooW0o1rX" -->
### PoisMF
<!-- #endregion -->

<!-- #region id="5TChLomHpgAk" -->
**Poisson factorization**

The model is described in more detail in [Fast Non-Bayesian Poisson Factorization for Implicit-Feedback Recommendations](https://arxiv.org/abs/1811.01908).

The basic idea is to take a sparse input matrix of counts $\mathbf{X}_{m,n}$, which in this case is given by the number of times each user (row in the matrix) played each song (column in the matrix), and find an approximation as the product of two non-negative lower-dimensional latent factor matrices $\mathbf{A}_{m,k}$ and $\mathbf{B}_{n,k}$ by maximizing Poisson likelihood, i.e. fit a model:
$$
\mathbf{X} \sim \text{Poisson}(\mathbf{A} \mathbf{B}^T)
$$

Which is then used to make predictions on the missing (zero-valued) entries, with the highest-predicted items for each user being the best candidates to recommend.

The poisemf package offers different optimization methods which have different advantages in terms of speed and quality, and depending on the settings, is usually able to find good solutions in which the latent factors matrices $\mathbf{A}$ and $\mathbf{B}$ are sparse (i.e. most entries are exactly zero).
** *
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Vjtv-4wWkhOG" outputId="6946c13b-d7c9-43ba-ce12-ad4763ee98f6"
model = PoisMF(k=5, method="pg")
model.fit(df_train)

print_ranking_metrics(model.A, model.B,
                      df_train, df_test, users_test,
                      nusers=20)
```

```python colab={"base_uri": "https://localhost:8080/"} id="4jxLC1dOmo8I" outputId="91f53b66-6b0e-4ba7-afca-fbacfd047f3c"
model = PoisMF(k=5, method="tncg")
model.fit(df_train)

print_ranking_metrics(model.A, model.B,
                      df_train, df_test, users_test,
                      nusers=20)
```

```python colab={"base_uri": "https://localhost:8080/"} id="qlWsR-qRkhKB" outputId="544c9c04-4855-4303-ebc9-30f8e8a0a1cc"
model = PoisMF(k=5, method="cg")
model.fit(df_train)

print_ranking_metrics(model.A, model.B,
                      df_train, df_test, users_test,
                      nusers=20)
```

```python colab={"base_uri": "https://localhost:8080/"} id="tnNJBZPYlZ4y" outputId="82e240f2-b133-4eec-8fdf-aa3414610906"
model.A[0]
```

<!-- #region id="lF3brYajovB2" -->
### Ranking and Prediction
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Yul771tglZ22" outputId="bac440ee-8ef1-40db-fbe7-cb61a7b4d03a"
model.topN(user = 2, n = 5, exclude = df_train.ItemId.loc[df_train.UserId==2])
```

```python colab={"base_uri": "https://localhost:8080/"} id="-xhfAdGDlZ0m" outputId="fc90b95e-c367-430f-c9a1-a455b1c10ee3"
model.topN_new(df_train.loc[df_train.UserId==2], n = 5, exclude = df_train.ItemId.loc[df_train.UserId==2])
```

```python colab={"base_uri": "https://localhost:8080/"} id="m_rqylmQlZwc" outputId="5cccda5b-f8a0-461c-d3ab-abd108e49a38"
model.predict(user=[3,3,3], item=[3,4,11])
```

<!-- #region id="qrx6Ke1eoqlE" -->
### Sparse Matrix
<!-- #endregion -->

```python id="Lba_G0vAjyfd"
## Note: package implicit takes a matrix of shape [items, users]
## Other packages take a matrix of shape [users, items]
Xcoo = coo_matrix((df_train.Count, (df_train.UserId, df_train.ItemId)))
Xcoo_T = Xcoo.T
Xcsr_T = csr_matrix(Xcoo_T)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 141} id="dSD3_O5AfhVR" outputId="a6554d84-c63d-4c62-98f3-602ce579fc23"
import matplotlib.pyplot as plt
plt.figure(figsize=(40,80))
plt.spy(Xcoo, markersize=5)
```

<!-- #region id="T8vUbXJRopTj" -->
### ALS
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 100, "referenced_widgets": ["a2b4a4e18a5e4dc191adc49934da1d53", "2259bc6cadf744e3ab2db78d0d0c8260", "f9c58d79db2845aea87089ad760003c3", "d5f40f6a5e18401089241449f4086bc7", "01aa850c78774850812ab6b981716665", "7134adc5ae8945a1a411d7fe05b2943f", "8dcd231173b74f83b0ee1b3fe2f2d4bb", "926a1b7ced5e41a7bd25e1acf6aa16b1"]} id="jA_D8jbgjTW9" outputId="d3f5b225-32b5-4a29-9878-cf5cfbbb8aac"
ials = AlternatingLeastSquares(factors=5, regularization=0.01,
                               dtype=np.float64, iterations=5,
                               use_gpu=False)
ials.fit(Xcsr_T)

print_ranking_metrics(ials.user_factors, ials.item_factors,
                      df_train, df_test, users_test, nusers=20)
```

<!-- #region id="iZ4CKICIon1u" -->
### BPR
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 100, "referenced_widgets": ["412716fd20be443fb46c28c8e220970d", "c5f8db1d490d44558ebc64108f83fb44", "80038a9e69294bc299de5091cb628085", "9a0e2d29b6cc4c74b446a8201ecebdc1", "f4c0c23badf746008a533d005c35dfb3", "b56bf1a50c994d7ba0985c082f7b0c36", "89bef3d0469d430b9de01e0a14bde1f2", "c6eead5d942e43a2955355fa8721b1aa"]} id="1W-q3dkHn1om" outputId="0e4d4492-8242-4e12-a621-e08b7e92768e"
bpr = BayesianPersonalizedRanking(factors=5, regularization=0.01,
                               dtype=np.float64, iterations=5,
                               use_gpu=False)
bpr.fit(Xcsr_T)

print_ranking_metrics(bpr.user_factors, bpr.item_factors,
                      df_train, df_test, users_test, nusers=20)
```

<!-- #region id="Zx2iLLu0oldb" -->
### HPF
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="WOmJ_M3soVwu" outputId="2d9e0960-cf04-4108-9aac-e99f60c6468d"
hpf = HPF(k=5, verbose=False, use_float=False).fit(Xcoo)

print_ranking_metrics(hpf.Theta, hpf.Beta,
                      df_train, df_test, users_test, nusers=20)
```
