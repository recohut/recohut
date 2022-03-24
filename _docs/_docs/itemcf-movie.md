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

<!-- #region id="bm3HWGGdxysN" -->
# Item-item CF and SVD on ML-1m
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Wz4-0LEWzXqi" executionInfo={"status": "ok", "timestamp": 1638107336767, "user_tz": -330, "elapsed": 1165, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="aae89f16-abc2-4a07-cbde-9614c57e7bdc"
!wget -q --show-progress -O movies.dat https://github.com/RecoHut-Datasets/movielens_1m/raw/main/ml1m_items.dat
!wget -q --show-progress -O ratings.dat https://github.com/RecoHut-Datasets/movielens_1m/raw/main/ml1m_ratings.dat
```

```python id="dr4Z5Thk_OcR"
import numpy as np
import pandas as pd
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="vsFUbv9y_UaT" executionInfo={"status": "ok", "timestamp": 1638107356351, "user_tz": -330, "elapsed": 471, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ae8a8a74-bd86-4fc2-d577-b1675a0612d4"
movie = pd.read_csv('movies.dat',
                     sep="\t",
                     header=None,
                     engine='python',
                     names=['movieId', 'title', 'genre'])

movie.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="dxxvXVWyAESn" executionInfo={"status": "ok", "timestamp": 1638107372609, "user_tz": -330, "elapsed": 5100, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3a684ede-dcca-4ea2-b296-a32dc03cce08"
rating = pd.read_csv('ratings.dat',
                     sep="\t",
                     header=None,
                     engine='python',
                     names=['userId', 'movieId', 'rating', 'time'])

rating.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="nmAH3FIiArj_" executionInfo={"status": "ok", "timestamp": 1638107374389, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e4ff2908-e631-4590-9d4b-2e2817037264"
df = pd.merge(rating, movie, on='movieId')
df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 614} id="3x-30wNsAuv6" executionInfo={"status": "ok", "timestamp": 1638107377558, "user_tz": -330, "elapsed": 1378, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="547e042d-4d34-44b2-a785-21ab2d5abe67"
u_m = pd.pivot_table(df, index='userId', values='rating', columns='title')
u_m
```

```python colab={"base_uri": "https://localhost:8080/"} id="ipD8atJoBUyf" executionInfo={"status": "ok", "timestamp": 1638107378338, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="790e4afc-05f3-499d-9773-d169f1ec3c49"
print('{} users x {} movies'.format(u_m.shape[0], u_m.shape[1]))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 708} id="dX-BhlKaAzGp" executionInfo={"status": "ok", "timestamp": 1630259084335, "user_tz": -330, "elapsed": 19, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="52adee6c-f41e-4981-98f3-6bff86aecf00"
m_u = u_m.T
m_u
```

```python colab={"base_uri": "https://localhost:8080/"} id="FEI5JS3jBhIe" executionInfo={"status": "ok", "timestamp": 1630259283556, "user_tz": -330, "elapsed": 583, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cdcfe20a-cc79-412e-a200-df0c8cf2e4c9"
print('{} movies x {} users'.format(m_u.shape[0], m_u.shape[1]))
```

```python colab={"base_uri": "https://localhost:8080/"} id="2-4kSqFyBMCd" executionInfo={"status": "ok", "timestamp": 1630259465322, "user_tz": -330, "elapsed": 704, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="075f7cde-2245-4f40-ed0f-63d41debef20"
min_watch = 20
rtd = u_m.columns.values[u_m.count() < min_watch]
print('{} movies had been watched less than {} times, some of them are: \n{}'.format(len(rtd), min_watch, rtd[:5]))
```

```python colab={"base_uri": "https://localhost:8080/"} id="_GbIeQ9sAbfp" executionInfo={"status": "ok", "timestamp": 1630259533738, "user_tz": -330, "elapsed": 626, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d0dcdfca-73b4-4e15-811f-2c758ee3f005"
u_m = u_m.drop(rtd, axis=1).fillna(0)
m_u = m_u.drop(rtd).fillna(0)
print('{} users x {} movies'.format(u_m.shape[0], u_m.shape[1]))
```

<!-- #region id="gnMlNKAqClI9" -->
## Item-based Collaborative Filtering - Cosine Similarity Method
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 590} id="aQK6_cGdCwg3" executionInfo={"status": "ok", "timestamp": 1630259678129, "user_tz": -330, "elapsed": 3207, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9dbdd732-7b80-4e08-ebca-22b0c6994827"
from sklearn.metrics.pairwise import cosine_similarity

sim_df = pd.DataFrame(cosine_similarity(m_u), columns=m_u.index, index=m_u.index)
sim_df.head()
```

```python id="zhN9DYA1DHwi"
def get_item_based_cf(movieId, top_n=10):
    return sim_df[movieId].sort_values(ascending=False).iloc[1:top_n+1]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 941} id="T0q_HryiDbkh" executionInfo={"status": "ok", "timestamp": 1630260216137, "user_tz": -330, "elapsed": 801, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="14adb881-19fb-45e7-d3d7-61c3ada37dd8"
n_random_movies = np.random.choice(m_u.index, 5)
item_based_df = pd.DataFrame({i:get_item_based_cf(i).index for i in n_random_movies})
item_based_df
```

<!-- #region id="AagdkegdERU8" -->
## Item-based Collaborative Filtering - SVD Latent-factor Method
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="_-i7J3YPFVhK" executionInfo={"status": "ok", "timestamp": 1630260461404, "user_tz": -330, "elapsed": 2394, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0b6b770a-49f1-4cb3-ec43-4eae63b3f598"
from sklearn.decomposition import TruncatedSVD

SVD = TruncatedSVD(n_components=12)
mat = SVD.fit_transform(m_u)
mat
```

```python colab={"base_uri": "https://localhost:8080/"} id="DqTbncjNGHI1" executionInfo={"status": "ok", "timestamp": 1630260464280, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c91dea44-0758-489a-a230-8d96d101f407"
corr = np.corrcoef(mat)
corr
```

```python id="Guol-JhQGIaw"
def get_lf_cf(title, top_n=10):
    titles = list(m_u.index)
    idx = list(m_u.index).index(title)
    cor = corr[idx]
    return np.array(titles)[np.argsort(-cor)[1:top_n+1]]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 529} id="d7XGAwaAGiC3" executionInfo={"status": "ok", "timestamp": 1630260974125, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c9b5e8a2-cbd2-4bf2-8fa3-47f72d785a63"
item_based_df = pd.DataFrame({i:get_lf_cf(i) for i in n_random_movies})
item_based_df
```

<!-- #region id="ZDTNWheRIA-X" -->
## Full User-Item Matrix Rating Prediction using SVD MF
<!-- #endregion -->

```python id="8Gr3U_XcJvT3"
def non_zero_mean(x):
    return np.sum(x, axis=1) / np.count_nonzero(x, axis=1)
```

```python id="T34QEq7lJxft"
v = u_m.values
diff_v = np.where(v==0, 0, v-non_zero_mean(v).reshape(-1, 1))
u_m_ = pd.DataFrame(diff_v, columns=u_m.columns, index=u_m.index)
```

```python colab={"base_uri": "https://localhost:8080/"} id="__tFzwJIJh1U" executionInfo={"status": "ok", "timestamp": 1630261450553, "user_tz": -330, "elapsed": 2559, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="67a51596-b61e-4a73-a32d-1549c3038e39"
from scipy.sparse.linalg import svds

u,sigma, vt = svds(u_m_, k=12)
sigma = np.diag(sigma)
u.shape, sigma.shape, vt.shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 606} id="ehuNidLtJrv6" executionInfo={"status": "ok", "timestamp": 1630261471986, "user_tz": -330, "elapsed": 1052, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7c8e0c12-aecd-4fa2-dd71-30754ed6d800"
pred_ratings = np.dot(np.dot(u, sigma), vt) + non_zero_mean(v).reshape(-1, 1)
pred_ratings = pd.DataFrame(pred_ratings, index=u_m.index, columns=u_m.columns)
pred_ratings
```

```python colab={"base_uri": "https://localhost:8080/", "height": 279} id="kdXjPp7MKPF4" executionInfo={"status": "ok", "timestamp": 1630261588917, "user_tz": -330, "elapsed": 740, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="11b6b738-1cb1-4936-db71-cf71144df1b4"
import seaborn as sns

sns.kdeplot(pred_ratings.iloc[6]);
```
