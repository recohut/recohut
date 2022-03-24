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
# Comparing Implicit Models on LastFM Music Data
> Fitting ALS, BPR, PoissonMF, and HPFRec models on LastFM-250K music dataset

- toc: true
- badges: true
- comments: true
- categories: [Implicit, Music]
- author: "<a href='https://github.com/david-cortes'>David Cortes</a>"
- image:
<!-- #endregion -->

<!-- #region id="jf8Ogz1Ug3hF" -->
## Installation
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

<!-- #region id="rHJQt6lBfSzD" -->
## Data Load
<!-- #endregion -->

<!-- #region id="KrFMw9_mfV-4" -->
### Download Alternative 1
<!-- #endregion -->

```python id="qBTX60B8ePBm"
!pip install -q -U kaggle
!pip install --upgrade --force-reinstall --no-deps kaggle
!mkdir ~/.kaggle
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d neferfufi/lastfm
```

```python id="0PpzrjUce7-z"
!unzip lastfm.zip
```

<!-- #region id="zml3SUS9fYYN" -->
### Download Alternative 2
<!-- #endregion -->

```python id="0A1W4V1YeSPr"
!wget http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-360K.tar.gz
```

```python colab={"base_uri": "https://localhost:8080/", "height": 142} id="OvM5sjd5wa_5" outputId="b77a1e63-db0b-4912-87ee-e338f7f694a9"
lfm = pd.read_table('usersha1-artmbid-artname-plays.tsv',
                           sep='\t', header=None, names=['UserId','ItemId', 'Artist','Count'])
lfm.columns = ['UserId', 'ItemId', 'Artist', 'Count']
lfm.head(3)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 142} id="6Z2_sNunxaMx" outputId="11b8af9b-744f-4c9c-d7d8-27afe02bae3e"
lfm = lfm.drop('Artist', axis=1)
lfm = lfm.loc[lfm.Count > 0]
lfm['UserId'] = pd.Categorical(lfm.UserId).codes
lfm['ItemId'] = pd.Categorical(lfm.ItemId).codes
lfm.head(3)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 142} id="5Iaq0tkLff7b" outputId="b2e42e09-8308-4df1-c6e0-c814a6a72e93"
lfm.describe(include='all').T
```

<!-- #region id="S_IzUa6Bo8yO" -->
### Train/Test Split
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="x5ugJgBJi4ri" outputId="a2861c1b-e120-4f19-8f1c-5b69c88b9a89"
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(lfm, test_size=.3)
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

<!-- #region id="eps-jLDxqY7g" -->
The models fit here will be evaluated by AUC and P@5, calculated for individual users and then averaged across a random sample of 1,000 users. These metrics are calculated for each user separately, by taking the entries in the hold-out test set as a positive class, entries which are neither in the training or test sets as a negative class, and producing predictions for all the entries that were not in the training set - the idea being that models which tend to rank highest the items that the users ended up consuming are better.
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

```python colab={"base_uri": "https://localhost:8080/"} id="Vjtv-4wWkhOG" outputId="8d5bea4d-4286-4f32-e209-62ecd4c32ca4"
%%time
model_fast = PoisMF(reindex=False, method="pg", use_float=False,
                    k=10, niter=10, maxupd=1, l2_reg=1e9)\
                .fit(df_train)
```

```python colab={"base_uri": "https://localhost:8080/"} id="3i-ImEeuw0Mm" outputId="2525db5f-a7a6-4b65-cc35-8bc0f8c09ed0"
print_ranking_metrics(model_fast.A, model_fast.B,
                      df_train, df_test, users_test)
```

```python colab={"base_uri": "https://localhost:8080/"} id="4jxLC1dOmo8I" outputId="c3318400-8ab6-4c2a-c23e-c032459ecb59"
%%time
model_balanced = PoisMF(reindex=False, method="cg", use_float=False,
                        k=50, niter=30, maxupd=5, l2_reg=1e4)\
                    .fit(df_train)
```

```python colab={"base_uri": "https://localhost:8080/"} id="qlWsR-qRkhKB" outputId="8e48a2bb-bda3-4571-e1c7-832f4e5105b2"
print_ranking_metrics(model_balanced.A, model_balanced.B,
                      df_train, df_test, users_test)
```

```python colab={"base_uri": "https://localhost:8080/"} id="tnNJBZPYlZ4y" outputId="769ee3b1-f981-494c-ec1c-82e230755854"
model.A[0]
```

<!-- #region id="lF3brYajovB2" -->
### Ranking and Prediction
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Yul771tglZ22" outputId="07200077-c840-4364-f2a0-4dcedfd0a7a7"
model.topN(user = 2, n = 5, exclude = df_train.ItemId.loc[df_train.UserId==2])
```

```python colab={"base_uri": "https://localhost:8080/"} id="-xhfAdGDlZ0m" outputId="c9defc32-d31d-4138-8647-82d08a77af75"
model.topN_new(df_train.loc[df_train.UserId==2], n = 5, exclude = df_train.ItemId.loc[df_train.UserId==2])
```

```python colab={"base_uri": "https://localhost:8080/"} id="m_rqylmQlZwc" outputId="d84303b8-1892-434d-f4a6-82dfa6e6f39e"
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

<!-- #region id="T8vUbXJRopTj" -->
### ALS
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 100, "referenced_widgets": ["baf57328a893499d8cc7840cd3c98d99", "75af5a769cfd4b9fbf6c95d534ba8f3b", "7650d168da03480097eeb51e6f26eb96", "68a6f0bebee345278ea394fe2aa5161f", "a9eb17088c214829a1cb3a64944a4787", "e1a0835c1332417ea8e89b5cd4b84b83", "42afcd20588e49babe13b77456a37dc6", "bb788dac0a6741989be196de2ef2ebcc"]} id="jA_D8jbgjTW9" outputId="30e90f5d-1815-461c-8354-6cd724eb96b4"
ials = AlternatingLeastSquares(factors=50, regularization=0.01,
                               dtype=np.float64, iterations=50,
                               use_gpu=False)
ials.fit(Xcsr_T)

print_ranking_metrics(ials.user_factors, ials.item_factors,
                      df_train, df_test, users_test)
```

<!-- #region id="iZ4CKICIon1u" -->
### BPR
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 100, "referenced_widgets": ["e9f1d11e0fce4eefa69eafbf3b9c4542", "31463f1b3f3e41108831826ce471704b", "e2368b70b06e45e593e0246b690b980c", "9554c0299a784792b14a887ae4b46f97", "4a1d932d03094804b919fd5806d1aa9c", "e4af58b5aea9451ca55030e2040a8447", "95c2195cce1b4fcd888ff0c86cfb8d5b", "1be995ae33e446ec91299e01225cd1c5"]} id="1W-q3dkHn1om" outputId="1de41f0b-c73e-46d3-f8d6-c138c49b94d9"
bpr = BayesianPersonalizedRanking(factors=50, regularization=0.01,
                               dtype=np.float64, iterations=50,
                               use_gpu=False)
bpr.fit(Xcsr_T)

print_ranking_metrics(bpr.user_factors, bpr.item_factors,
                      df_train, df_test, users_test)
```

<!-- #region id="Zx2iLLu0oldb" -->
### HPF
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="WOmJ_M3soVwu" outputId="67b5de3b-5aa9-4230-a17f-46e261263b2b"
hpf = HPF(k=5, verbose=False, use_float=False).fit(Xcoo)

print_ranking_metrics(hpf.Theta, hpf.Beta,
                      df_train, df_test, users_test)
```
