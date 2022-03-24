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

```python id="UV_mis-jdwLd" executionInfo={"status": "ok", "timestamp": 1629283390937, "user_tz": -330, "elapsed": 22, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-mlh"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python id="KRGLEjqMd3dV"
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

```python id="wpdAcaW80D3o"
!git pull --rebase origin "{branch}"
```

```python id="Aa6AQmftAovn"
!git status
```

```python id="aG5PN_2EAovn"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python id="7J9mKTyRyFwQ" executionInfo={"status": "ok", "timestamp": 1629283395334, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import sys
sys.path.insert(0,'./code')
```

<!-- #region id="y1Fne9fbwMl5" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Dq_dZSZuwXN9" executionInfo={"status": "ok", "timestamp": 1629283402510, "user_tz": -330, "elapsed": 4343, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="97ea250e-3d20-41c8-f8c8-a5cb500b498d"
!pip install -q pyreclab
```

```python id="laKkuV4RwM4R" executionInfo={"status": "ok", "timestamp": 1629283510094, "user_tz": -330, "elapsed": 1061, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import pandas as pd
import pyreclab
import seaborn as sns
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
```

```python id="BROZFpzNwVw4" executionInfo={"status": "ok", "timestamp": 1629283510095, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
columns = ['movieid', 'title', 'release_date', 'video_release_date', \
           'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', \
           'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', \
           'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', \
           'Thriller', 'War', 'Western']
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="4e9BSTWxw92C" executionInfo={"status": "ok", "timestamp": 1629283510096, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b76f3f72-acbe-4afc-cf8c-2b08b7d860d0"
df_train = pd.read_csv('./data/bronze/u2.base',
                         sep='\t',
                         names=['userid', 'itemid', 'rating', 'timestamp'],
                         header=None)

# rating> = 3 is relevant (1) and rating less than 3 is not relevant (0)
df_train.rating = [1 if x >=3 else 0 for x in df_train.rating ]

df_train.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 391} id="6YKLBSeCxNdZ" executionInfo={"status": "ok", "timestamp": 1629283510096, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="061f9a6e-f6fa-49e1-db77-fdbda324c0f0"
df_items = pd.read_csv('./data/bronze/u.item',
                        sep='|',
                        index_col=0,
                        names = columns,
                        header=None, 
                        encoding='latin-1')

df_items.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="Rb9Jswpv3V1l" executionInfo={"status": "ok", "timestamp": 1629283634908, "user_tz": -330, "elapsed": 1392, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3fea4b41-0c60-48cb-d502-08dd8534fb95"
user_id = 2
user_df = df_train[df_train['userid'] == user_id]
df_items.loc[user_df['itemid'].values,['title','release_date']]
```

<!-- #region id="cBZ2dZK44tFI" -->
## MostPop
<!-- #endregion -->

```python id="2w2glF3V4uZo" executionInfo={"status": "ok", "timestamp": 1629283912916, "user_tz": -330, "elapsed": 919, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
most_popular = pyreclab.MostPopular(dataset='./data/bronze/u2.base',
                   dlmchar=b'\t',
                   header=False,
                   usercol=0,
                   itemcol=1,
                   ratingcol=2)

most_popular.train()
```

```python colab={"base_uri": "https://localhost:8080/"} id="kheaB67d46IA" executionInfo={"status": "ok", "timestamp": 1629283933616, "user_tz": -330, "elapsed": 2044, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="64d82763-948f-4795-ad40-772f10cec3da"
top_n = 10

recommendList, maprec, ndcg = most_popular.testrec(input_file='./data/bronze/u2.test',
                                          dlmchar=b'\t',
                                          header=False,
                                          usercol=0,
                                          itemcol=1,
                                          ratingcol=2,
                                          topn=top_n,
                                          relevance_threshold=2,
                                          includeRated=False)

print('MAP: {}\nNDCG@{}: {}'.format(maprec, top_n, ndcg))
```

```python colab={"base_uri": "https://localhost:8080/"} id="NpS2XZdY5SqD" executionInfo={"status": "ok", "timestamp": 1629284049926, "user_tz": -330, "elapsed": 1042, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="924dc7b1-68ba-4d16-fa0c-a728ccc92246"
ranking = [int(r) for r in most_popular.recommend(str(user_id), top_n, includeRated=False)]
print('Recommendation for user {}: {}'.format(user_id, ranking))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 390} id="-AFUbVkx5SqF" executionInfo={"status": "ok", "timestamp": 1629284065160, "user_tz": -330, "elapsed": 747, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9821ccd5-c2fa-44ad-b008-b21b0ed1b3ab"
df_items.loc[ranking,['title','release_date']]
```

<!-- #region id="TZBzRJYJ5DEA" -->
## ItemAvg
<!-- #endregion -->

```python id="xqymBKjH5Eqd" executionInfo={"status": "ok", "timestamp": 1629283978144, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
item_avg = pyreclab.ItemAvg(dataset='./data/bronze/u2.base',
                   dlmchar=b'\t',
                   header=False,
                   usercol=0,
                   itemcol=1,
                   ratingcol=2)

item_avg.train()
```

```python colab={"base_uri": "https://localhost:8080/"} id="7YnM_maU5IxY" executionInfo={"status": "ok", "timestamp": 1629283989270, "user_tz": -330, "elapsed": 725, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5e5a41da-1e22-47f6-ca07-9b602d4389c5"
top_n = 10

recommendList, maprec, ndcg = item_avg.testrec(input_file='./data/bronze/u2.test',
                                          dlmchar=b'\t',
                                          header=False,
                                          usercol=0,
                                          itemcol=1,
                                          ratingcol=2,
                                          topn=top_n,
                                          relevance_threshold=2,
                                          includeRated=False)

print('MAP: {}\nNDCG@{}: {}'.format(maprec, top_n, ndcg))
```

```python colab={"base_uri": "https://localhost:8080/"} id="hNRX45Y_5he0" executionInfo={"status": "ok", "timestamp": 1629284103078, "user_tz": -330, "elapsed": 9961, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0bccc41a-66e9-41c3-9a5e-da13167bece1"
ranking = [int(r) for r in item_avg.recommend(str(user_id), top_n, includeRated=False)]
print('Recommendation for user {}: {}'.format(user_id, ranking))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 390} id="Cvqyt0xI5he5" executionInfo={"status": "ok", "timestamp": 1629284103080, "user_tz": -330, "elapsed": 33, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ca17bc46-0700-4d19-8d24-95d7ab93b648"
df_items.loc[ranking,['title','release_date']]
```

<!-- #region id="XaKXPw0n55cR" -->
## UserKNN
<!-- #endregion -->

```python id="amrkrB_s6FaZ" executionInfo={"status": "ok", "timestamp": 1629284303210, "user_tz": -330, "elapsed": 873, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
myUserKnn = pyreclab.UserKnn(dataset='./data/bronze/u1.base',
                             dlmchar=b'\t',
                             header=False,
                             usercol=0,
                             itemcol=1,
                             ratingcol=2)

myUserKnn.train(k=7, similarity='pearson')
```

```python colab={"base_uri": "https://localhost:8080/"} id="AZhEe1HR6W2s" executionInfo={"status": "ok", "timestamp": 1629284592490, "user_tz": -330, "elapsed": 3691, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5b19f92e-eb27-492b-bc20-817ca0d7e448"
predlist, mae, rmse = myUserKnn.test(input_file='./data/bronze/u2.test',
                               dlmchar=b'\t',
                               header=False,
                               usercol=0,
                               itemcol=1,
                               ratingcol=2)

print('MAE: {}\nRMSE: {}'.format(mae, rmse))
```

```python colab={"base_uri": "https://localhost:8080/"} id="8HKIe4tS7i5S" executionInfo={"status": "ok", "timestamp": 1629284637747, "user_tz": -330, "elapsed": 403, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e2849e11-674e-4655-d6a5-67ee7d3f8b9d"
ranking = [int(r) for r in myUserKnn.recommend(str(user_id), top_n, includeRated=False)]
print('Recommendation for user {}: {}'.format(user_id, ranking))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 390} id="G0X80PZ47i5k" executionInfo={"status": "ok", "timestamp": 1629284639807, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="09b9aba0-ffaf-4293-8fea-63bd402f4fff"
df_items.loc[ranking,['title','release_date']]
```

<!-- #region id="y7J3Y2fs7sG-" -->
## ItemKNN
<!-- #endregion -->

```python id="z1r7EXMzSGM7" executionInfo={"status": "ok", "timestamp": 1629290557297, "user_tz": -330, "elapsed": 10571, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
myItemKnn = pyreclab.ItemKnn(dataset='./data/bronze/u1.base',
                             dlmchar=b'\t',
                             header=False,
                             usercol=0,
                             itemcol=1,
                             ratingcol=2)

myItemKnn.train(k=7, similarity='cosine')
```

```python colab={"base_uri": "https://localhost:8080/"} id="eIPAbWrVSGNL" executionInfo={"status": "ok", "timestamp": 1629290565121, "user_tz": -330, "elapsed": 5167, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="35a93086-e797-43b5-c61b-63e755b8ad39"
predlist, mae, rmse = myItemKnn.test(input_file='./data/bronze/u2.test',
                               dlmchar=b'\t',
                               header=False,
                               usercol=0,
                               itemcol=1,
                               ratingcol=2)

print('MAE: {}\nRMSE: {}'.format(mae, rmse))
```

```python colab={"base_uri": "https://localhost:8080/"} id="iurexL9xSGNL" executionInfo={"status": "ok", "timestamp": 1629290572256, "user_tz": -330, "elapsed": 470, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="17f34f20-177f-4f64-bdc0-59ad0deacfa6"
ranking = [int(r) for r in myItemKnn.recommend(str(user_id), top_n, includeRated=False)]
print('Recommendation for user {}: {}'.format(user_id, ranking))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 390} id="p1DJVHFOSGNM" executionInfo={"status": "ok", "timestamp": 1629290573736, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="baef20e7-ddb7-4659-b381-72ed214cfd06"
df_items.loc[ranking,['title','release_date']]
```

<!-- #region id="9QtK5OQE337n" -->
## FunkSVD
<!-- #endregion -->

```python id="WJTeUCdA3a21" executionInfo={"status": "ok", "timestamp": 1629283688638, "user_tz": -330, "elapsed": 543, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
svd = pyreclab.SVD(dataset='./data/bronze/u2.base',
                   dlmchar=b'\t',
                   header=False,
                   usercol=0,
                   itemcol=1,
                   ratingcol=2)

svd.train(factors=100, maxiter=100, lr=0.01, lamb=0.1)
```

```python colab={"base_uri": "https://localhost:8080/"} id="8gm8sUem3_-0" executionInfo={"status": "ok", "timestamp": 1629283696849, "user_tz": -330, "elapsed": 742, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f338dbd8-7614-4a22-f500-d21f2fade009"
predlist, mae, rmse = svd.test(input_file='./data/bronze/u2.test',
                               dlmchar=b'\t',
                               header=False,
                               usercol=0,
                               itemcol=1,
                               ratingcol=2)

print('MAE: {}\nRMSE: {}'.format(mae, rmse))
```

```python colab={"base_uri": "https://localhost:8080/"} id="d9duKmfl4DSF" executionInfo={"status": "ok", "timestamp": 1629283708286, "user_tz": -330, "elapsed": 594, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c33d7ae3-d03f-4960-fb8e-b67102a2edc0"
top_n = 20

recommendList, maprec, ndcg = svd.testrec(input_file='./data/bronze/u2.test',
                                          dlmchar=b'\t',
                                          header=False,
                                          usercol=0,
                                          itemcol=1,
                                          ratingcol=2,
                                          topn=top_n,
                                          relevance_threshold=2,
                                          includeRated=False)

print('MAP: {}\nNDCG@{}: {}'.format(maprec, top_n, ndcg))
```

```python colab={"base_uri": "https://localhost:8080/"} id="lkA-hTFh4Gae" executionInfo={"status": "ok", "timestamp": 1629283733692, "user_tz": -330, "elapsed": 729, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7357335f-f711-4501-92b9-600fdb5c4e40"
ranking = [int(r) for r in svd.recommend(str(user_id), top_n, includeRated=False)]
print('Recommendation for user {}: {}'.format(user_id, ranking))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 700} id="IbKTKsa04Mo8" executionInfo={"status": "ok", "timestamp": 1629283751585, "user_tz": -330, "elapsed": 501, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a8593309-a647-4ec1-ac18-997eff9d5807"
df_items.loc[ranking,['title','release_date']]
```

<!-- #region id="_ydd1juB4RGD" -->
## SlopeOne
<!-- #endregion -->

```python id="09TWRnKISiRk" executionInfo={"status": "ok", "timestamp": 1629290673410, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
slopeone = pyreclab.SlopeOne(dataset='./data/bronze/u2.base',
                   dlmchar=b'\t',
                   header=False,
                   usercol=0,
                   itemcol=1,
                   ratingcol=2)

slopeone.train()
```

```python colab={"base_uri": "https://localhost:8080/"} id="UqtCV9-ISiRl" executionInfo={"status": "ok", "timestamp": 1629290682958, "user_tz": -330, "elapsed": 459, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="75efc440-5642-4213-bc32-8367ab9624bf"
predlist, mae, rmse = slopeone.test(input_file='./data/bronze/u2.test',
                               dlmchar=b'\t',
                               header=False,
                               usercol=0,
                               itemcol=1,
                               ratingcol=2)

print('MAE: {}\nRMSE: {}'.format(mae, rmse))
```

```python colab={"base_uri": "https://localhost:8080/"} id="q_PUFD60SiRm" executionInfo={"status": "ok", "timestamp": 1629290712276, "user_tz": -330, "elapsed": 433, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="26b2dfb8-d1bd-4bd7-f7b3-9b36d8d3e872"
top_n = 20

recommendList, maprec, ndcg = slopeone.testrec(input_file='./data/bronze/u2.test',
                                          dlmchar=b'\t',
                                          header=False,
                                          usercol=0,
                                          itemcol=1,
                                          ratingcol=2,
                                          topn=top_n,
                                          relevance_threshold=2,
                                          includeRated=False)

print('MAP: {}\nNDCG@{}: {}'.format(maprec, top_n, ndcg))
```

```python colab={"base_uri": "https://localhost:8080/"} id="s9wG-ca5SiRm" executionInfo={"status": "ok", "timestamp": 1629290722641, "user_tz": -330, "elapsed": 912, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a5e20e79-6349-4166-935a-8180ee987812"
ranking = [int(r) for r in slopeone.recommend(str(user_id), top_n, includeRated=False)]
print('Recommendation for user {}: {}'.format(user_id, ranking))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 700} id="Sl9XT0MlSiRn" executionInfo={"status": "ok", "timestamp": 1629290725841, "user_tz": -330, "elapsed": 405, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b051f7f8-bc3c-4fe3-80ad-e9230681feb6"
df_items.loc[ranking,['title','release_date']]
```
