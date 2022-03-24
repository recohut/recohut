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

<!-- #region id="1-TOAriPHRIu" -->
> Note: KNN is a memory-based model, that means it will memorize the patterns and not generalize. It is simple yet powerful technique and compete with SOTA models like BERT4Rec.
<!-- #endregion -->

```python id="xWTTsFsu3idp" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628096233726, "user_tz": -330, "elapsed": 2459, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="31f3b304-3368-405c-9aba-0a9309c20037"
import os
project_name = "reco-tut-itr"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)

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

```python id="fZtYfVlgGURe" executionInfo={"status": "ok", "timestamp": 1628096259909, "user_tz": -330, "elapsed": 397, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
import numpy as np
import pandas as pd
import scipy.sparse
from scipy.spatial.distance import correlation
```

```python colab={"base_uri": "https://localhost:8080/"} id="f6uglOI8Gb-V" executionInfo={"status": "ok", "timestamp": 1628096800010, "user_tz": -330, "elapsed": 454, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7be89cd3-b007-4049-9c15-a2cd274059ea"
df = pd.read_parquet('./data/silver/rating.parquet.gz')
df.info()
```

```python colab={"base_uri": "https://localhost:8080/"} id="e0oKCHseIkHG" executionInfo={"status": "ok", "timestamp": 1628096861155, "user_tz": -330, "elapsed": 417, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b540c2af-c662-4729-b8c7-5209e400b042"
df2 = pd.read_parquet('./data/silver/items.parquet.gz')
df2.info()
```

```python colab={"base_uri": "https://localhost:8080/"} id="FBkKJvjPIp2q" executionInfo={"status": "ok", "timestamp": 1628096888907, "user_tz": -330, "elapsed": 444, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8a2c4297-707c-4ac0-e202-edf7a8e630fb"
df = pd.merge(df, df2, on='itemId')
df.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 439} id="e4XoIzO8GWIE" executionInfo={"status": "ok", "timestamp": 1628096367203, "user_tz": -330, "elapsed": 675, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="90ade036-d416-4ba9-d6c9-1ab3dbfb7777"
rating_matrix = pd.pivot_table(df, values='rating',
                               index=['userId'], columns=['itemId'])
rating_matrix
```

```python id="QWfR5ZHwGwVI" executionInfo={"status": "ok", "timestamp": 1628096380422, "user_tz": -330, "elapsed": 396, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def similarity(user1, user2):
    try:
        user1=np.array(user1)-np.nanmean(user1)
        user2=np.array(user2)-np.nanmean(user2)
        commonItemIds=[i for i in range(len(user1)) if user1[i]>0 and user2[i]>0]
        if len(commonItemIds)==0:
           return 0
        else:
           user1=np.array([user1[i] for i in commonItemIds])
           user2=np.array([user2[i] for i in commonItemIds])
           return correlation(user1,user2)
    except ZeroDivisionError:
        print("You can't divide by zero!")
```

```python id="24XPWqs7G0I_" executionInfo={"status": "ok", "timestamp": 1628097264660, "user_tz": -330, "elapsed": 470, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def nearestNeighbourRatings(activeUser, K):
    try:
        similarityMatrix=pd.DataFrame(index=rating_matrix.index,columns=['Similarity'])
        for i in rating_matrix.index:
            similarityMatrix.loc[i]=similarity(rating_matrix.loc[activeUser],rating_matrix.loc[i])
        similarityMatrix=pd.DataFrame.sort_values(similarityMatrix,['Similarity'],ascending=[0])
        nearestNeighbours=similarityMatrix[:K]
        neighbourItemRatings=rating_matrix.loc[nearestNeighbours.index]
        predictItemRating=pd.DataFrame(index=rating_matrix.columns, columns=['Rating'])
        for i in rating_matrix.columns:
            predictedRating=np.nanmean(rating_matrix.loc[activeUser])
            for j in neighbourItemRatings.index:
                if rating_matrix.loc[j,i]>0:
                   predictedRating += (rating_matrix.loc[j,i]-np.nanmean(rating_matrix.loc[j]))*nearestNeighbours.loc[j,'Similarity']
                predictItemRating.loc[i,'Rating']=predictedRating
    except ZeroDivisionError:
        print("You can't divide by zero!")            
    return predictItemRating
```

```python id="HA09XRRfHOZ5" executionInfo={"status": "ok", "timestamp": 1628097419122, "user_tz": -330, "elapsed": 384, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def topNRecommendations(activeUser, N):
    try:
        predictItemRating = nearestNeighbourRatings(activeUser,N)
        placeAlreadyWatched = list(rating_matrix.loc[activeUser].loc[rating_matrix.loc[activeUser]>0].index)
        predictItemRating = predictItemRating.drop(placeAlreadyWatched)
        topRecommendations = pd.DataFrame.sort_values(predictItemRating,['Rating'],ascending = [0])[:N]
        topRecommendationTitles = (df.loc[df.itemId.isin(topRecommendations.index)])
    except ZeroDivisionError:
        print("You can't divide by zero!")
    return list([topRecommendationTitles.location,
                 topRecommendationTitles.place,
                 topRecommendationTitles.state,
                 topRecommendationTitles.location_rating])
```

```python id="xp-VaE0pIHIl" executionInfo={"status": "ok", "timestamp": 1628097633307, "user_tz": -330, "elapsed": 1651, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def favoritePlace(activeUser,N):
    topPlace=pd.DataFrame.sort_values(df[df.userId==activeUser],['rating'],ascending=[0])[:N]
    return list([topPlace.location,
                 topPlace.place,
                 topPlace.state,
                 topPlace.location_rating])
```

```python id="w_klxgDlH7o4" executionInfo={"status": "ok", "timestamp": 1628097422831, "user_tz": -330, "elapsed": 4, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
activeUser = 4
```

```python colab={"base_uri": "https://localhost:8080/", "height": 190} id="KMj2yGvtLk7n" executionInfo={"status": "ok", "timestamp": 1628097662575, "user_tz": -330, "elapsed": 458, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5195a7f3-4b8d-4fd0-e59a-b48a788dd649"
print("Your favorite places are: ")
fav_place=pd.DataFrame(favoritePlace(str(activeUser),4))
fav_place=fav_place.T
fav_place=fav_place.sort_values(by='location_rating', ascending=False)
fav_place
```

```python colab={"base_uri": "https://localhost:8080/", "height": 190} id="ZyMlJdYIH9dB" executionInfo={"status": "ok", "timestamp": 1628097678985, "user_tz": -330, "elapsed": 488, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="98c5b7ff-b122-4402-dc5f-3ca3c4ecd19d"
print("The recommended places for you are: ")
topN = pd.DataFrame(topNRecommendations(str(activeUser), 4))
topN = topN.T
topN = topN.sort_values(by = 'location_rating', ascending=False).drop_duplicates().reset_index(drop=True)
topN
```
