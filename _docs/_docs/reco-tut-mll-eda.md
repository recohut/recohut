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

```python id="1uICbB4nDexm" executionInfo={"status": "ok", "timestamp": 1628499805389, "user_tz": -330, "elapsed": 515, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-mll"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python colab={"base_uri": "https://localhost:8080/"} id="EFnuEM16DqQd" executionInfo={"status": "ok", "timestamp": 1628499808795, "user_tz": -330, "elapsed": 2696, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="35bad2fa-a9c5-4202-aea5-d5572f8e8f57"
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

```python id="7bkm0Tb0DqQq"
!git status
```

```python id="gGB7bN_TTnGo"
!git pull --rebase origin main
```

```python id="9nEA2fSADqQr"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

<!-- #region id="Omgrutt9Jx2t" -->
---
<!-- #endregion -->

```python id="646XDJ96T3oG" executionInfo={"status": "ok", "timestamp": 1628503718424, "user_tz": -330, "elapsed": 2163, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
%matplotlib inline
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="XmtYiQXMT_pO" executionInfo={"status": "ok", "timestamp": 1628502513615, "user_tz": -330, "elapsed": 496, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4c840990-6468-4d9d-a820-4cfc98e39f7b"
ratings = pd.read_parquet('./data/silver/ratings.parquet.gzip')
ratings.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 326} id="vOMu7TycUF-z" executionInfo={"status": "ok", "timestamp": 1628502527336, "user_tz": -330, "elapsed": 422, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="13f16ebd-a98f-4118-bf25-970932acbe17"
movies = pd.read_parquet('./data/silver/movies.parquet.gzip')
movies.head()
```

<!-- #region id="0YAYWNGPUJam" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 485} id="8rvqo-TQT2AW" executionInfo={"status": "ok", "timestamp": 1628502557642, "user_tz": -330, "elapsed": 926, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cc17a9f6-0594-4d3b-931c-4715c47a4c8d"
# Number of Ratings Per Year
ratings_per_year = ratings[['rating', 'timestamp']].groupby('timestamp').count()
ratings_per_year.columns = ['Rating Count']
ax1 = ratings_per_year.plot(kind='line',figsize=(12,8))
ax1.set_xlabel('Year')
ax1.set_ylabel('Number of ratings given')
plt.title('Number of Movies Rated Per Year')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 235} id="AcE_Ya28UXPu" executionInfo={"status": "ok", "timestamp": 1628502590958, "user_tz": -330, "elapsed": 561, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c700c4f4-020f-4f07-f768-5daf4baaa489"
ratings_df = ratings[['rating', 'timestamp']].groupby('timestamp').count().sort_values(by="rating", ascending=False)
ratings_df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 354} id="pWTDBh93UmWl" executionInfo={"status": "ok", "timestamp": 1628502673830, "user_tz": -330, "elapsed": 714, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8aae5861-236f-4711-8ef5-d42ae03172f9"
# Movies Released per Year
dftmp = movies[['movieId', 'year']].groupby('year')

fig, ax1 = plt.subplots(figsize=(10,5))
ax1.plot(dftmp.year.first(), dftmp.movieId.nunique(), "g-o")
ax1.grid(None)
ax1.set_ylim(0,)
ax1.set_xlabel('Year')
ax1.set_ylabel('Number of movies released')
plt.title('Movies per year')
plt.show()
```

```python id="Cb0b_6O0U-87" executionInfo={"status": "ok", "timestamp": 1628502807149, "user_tz": -330, "elapsed": 444, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
insights = []
insights.append('Most Movies are released in between 1980 and 2020')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 354} id="YQYsUDOqVRO-" executionInfo={"status": "ok", "timestamp": 1628502827203, "user_tz": -330, "elapsed": 1455, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="03bbd1c3-edc6-47d5-ca5e-ffaae3a3addf"
# Ratings per Year
dftmp = ratings[['rating', 'timestamp']].groupby('timestamp')

fig, ax1 = plt.subplots(figsize=(10,5))
ax1.plot(dftmp.timestamp.first(), dftmp.rating.count(), "r-o")
ax1.grid(None)
ax1.set_ylim(0,)
ax1.set_xlabel('Year')
ax1.set_ylabel('Number of ratings given')
plt.title('Ratings per year')
plt.show()
```

```python id="B55rzOt9Vcns" executionInfo={"status": "ok", "timestamp": 1628502925172, "user_tz": -330, "elapsed": 421, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
insights.append('Ratings given vary in different years. Most Ratings are given around year 2000')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 354} id="OK9D8CGhVueC" executionInfo={"status": "ok", "timestamp": 1628502950075, "user_tz": -330, "elapsed": 1131, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a50a11c3-544f-427a-8406-f54906617215"
# Average Movie Rating
dftmp = movies[['movieId', 'year']].set_index('movieId').join(
    ratings[['movieId','rating']].groupby('movieId').mean())
dftmp.rating.hist(bins=25, grid=False, edgecolor='b',figsize=(10,5))
plt.xlim(0,5)
plt.xlabel('Average Movie rating')
plt.ylabel('Number of Movies')
plt.title('Number of Movies Vs Average Rating')
plt.show()
```

```python id="MxaWM0r7Vw5m" executionInfo={"status": "ok", "timestamp": 1628502989402, "user_tz": -330, "elapsed": 582, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
insights.append('Average movie ratings creates normal distrubition peaked at about 3.5')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 354} id="5O5cTbNWWAyE" executionInfo={"status": "ok", "timestamp": 1628503022073, "user_tz": -330, "elapsed": 1735, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="70ff1559-a282-4b0d-b5e1-d10d73fb686a"
# Average Movie Ratings Per Year
dftmp = movies[['movieId', 'year']].set_index('movieId')
dftmp = dftmp.join(ratings[ratings.set_index('movieId').index.isin(dftmp.index)][['movieId', 'rating']]
                   .groupby('movieId').mean())
dftmp = dftmp.groupby('year').mean()

plt.figure(figsize=(10,5))
plt.plot(dftmp, "r-o", label='All genres', color='black')
plt.xlabel('Release Year')
plt.ylabel('Average Rating')
plt.title('Average Movie Ratings Per Release Year')
plt.ylim(0,)
plt.show()
```

```python id="F2NNg2IdWEem" executionInfo={"status": "ok", "timestamp": 1628503038141, "user_tz": -330, "elapsed": 625, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
insights.append('While movies released after 1960s have more stable average rating, old movies have huge variation in consecutive years')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 354} id="0oouUvukWQbG" executionInfo={"status": "ok", "timestamp": 1628503086357, "user_tz": -330, "elapsed": 1502, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="796bfb1a-82b0-4b6a-de9c-4e4f512e8562"
# Average Movie Ratings Per Year In Detail
first_rating_timestamp = ratings['timestamp'].min()
dftmp = movies[['movieId', 'year']].set_index('movieId')
dftmp = dftmp[ (dftmp['year'] >= first_rating_timestamp.year) ]
dftmp = dftmp.join(ratings[ratings.set_index('movieId').index.isin(dftmp.index)][['movieId', 'rating']]
                   .groupby('movieId').mean())
dftmp = dftmp.groupby('year').mean()

plt.figure(figsize=(10,5))
plt.plot(dftmp, "r-o", label='All genres', color='black')
plt.xlabel('Release Year')
plt.ylabel('Average Rating')
plt.title('Average Movie Ratings Per Year For Movies Released After First Rating Given')
plt.show()
```

```python id="wgCPw6GQWYJk" executionInfo={"status": "ok", "timestamp": 1628503119168, "user_tz": -330, "elapsed": 445, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
insights.append('Users bias changes in different years, as we can see in 2010 users gave average of 3.35 to movies and about 2.5 in 2015')
insights.append('When we take a closer look at the average rating of the movies that has been released after first rating given in the dataset, average ratings seem to change a lot as the years pass by. And, average ratings tend to go down. This raises questions like, does the movies released in adjacent years changes a lot, or the users having a different trend after the first trend and new movies that has been released on the adjacent years tend to be similar with old trend, which results in lower averages.')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 354} id="hXiZypLuWcfw" executionInfo={"status": "ok", "timestamp": 1628503135167, "user_tz": -330, "elapsed": 719, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="62bdc8c1-4d62-475c-d240-4db75274c9dd"
# Average Rating Per User
dftmp = ratings[['userId','rating']].groupby('userId').mean()
dftmp.rating.hist(bins=100, grid=False, edgecolor='b',figsize=(10,5))

plt.xlim(1,5)
plt.xlabel ('Average movie rating')
plt.ylabel ('Number of users')
plt.title ('Average ratings per user')
plt.show()
```

```python id="Vbr2-36mWgRO" executionInfo={"status": "ok", "timestamp": 1628503150381, "user_tz": -330, "elapsed": 622, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
insights.append('Users on average gives 3.7 to movies but different users have different average which shows us some of the users are inclined to give low rating and some of them inclined to give high ratings.')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 354} id="JCTSEF2IWkQo" executionInfo={"status": "ok", "timestamp": 1628503167774, "user_tz": -330, "elapsed": 1390, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="119fb1e4-7b13-43ef-adbf-09807528f034"
# Ratings Per User
dftmp = ratings[['userId', 'movieId']].groupby('userId').count()
dftmp.columns=['num_ratings']
dftmp.sort_values(by='num_ratings', inplace=True, ascending=False)

plt.figure(figsize=(15,5))
plt.scatter(dftmp.index, dftmp.num_ratings, edgecolor='black')
plt.xlim(0,len(dftmp.index))
plt.ylim(0,)
plt.title('Number of Ratings per user')
plt.xlabel('userId')
plt.ylabel('Number of ratings given')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 354} id="4PiLvuDaWuLR" executionInfo={"status": "ok", "timestamp": 1628503208803, "user_tz": -330, "elapsed": 1766, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e76c4182-a13c-423f-b7ca-e27ba8653b46"
# Histogram of ratings counts.
plt.figure(figsize=(10,5))
plt.hist(dftmp.num_ratings, bins=100, edgecolor='black', log=True)
plt.title('Number of Ratings per user')
plt.xlabel('Number of ratings given')
plt.ylabel('Number of users')
plt.xlim(0,)
plt.show()
```

```python id="4iKl7R0cWyG7" executionInfo={"status": "ok", "timestamp": 1628503224352, "user_tz": -330, "elapsed": 894, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
insights.append('while high active users are tend to rate 200-500 movies, most of the users gave only few ratings almost 0. Dataset is quite sparse.')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 354} id="_VvKsbooW1Jj" executionInfo={"status": "ok", "timestamp": 1628503237120, "user_tz": -330, "elapsed": 1494, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="21f0bdd9-52ca-4a54-8034-a76535ae229f"
# Rating Per Movie
dftmp = ratings[['userId', 'movieId']].groupby('movieId').count()
dftmp.columns=['num_ratings']

plt.figure(figsize=(15,5))
plt.scatter(dftmp.index, dftmp.num_ratings, edgecolor='black')
plt.xlim(0,dftmp.index.max())
plt.ylim(0,)
plt.title('Ratings per movie')
plt.xlabel('movieId')
plt.ylabel('Number of ratings received')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 354} id="WgV_SPMTW4rV" executionInfo={"status": "ok", "timestamp": 1628503251583, "user_tz": -330, "elapsed": 1797, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2cd35e1c-0dfc-48ff-f1ee-9bba960e8069"
# Histogram of ratings counts.
plt.figure(figsize=(15,5))
plt.hist(dftmp.num_ratings, bins=100, edgecolor='black', log=True)
plt.title('Ratings per movie')
plt.xlabel('Number of ratings received')
plt.ylabel('Number of movieIds')
plt.xlim(0,)
plt.show()
```

```python id="wXW8wjkRW8Mi" executionInfo={"status": "ok", "timestamp": 1628503264631, "user_tz": -330, "elapsed": 564, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
insights.append('Almost %99 percent of the movies taken less than 150 ratings.')
```

```python colab={"base_uri": "https://localhost:8080/"} id="--CPXmg3XBsY" executionInfo={"status": "ok", "timestamp": 1628503294233, "user_tz": -330, "elapsed": 427, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ff95ce5b-09a3-4c17-e567-44b13dd5bbf2"
# Let's check those movies with +150 reviews, those should be pretty popular movies!
movies.set_index('movieId').loc[dftmp.index[dftmp.num_ratings>150]]['title'][:10]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 299} id="rYwBVCMVXF6G" executionInfo={"status": "ok", "timestamp": 1628503305347, "user_tz": -330, "elapsed": 1705, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1fc0d52b-021a-49ce-ab0e-a35554bfde34"
# Let's check the average rating too, those should be pretty good movies!
ratings.set_index('movieId').loc[dftmp.index[dftmp.num_ratings>150]].groupby('movieId').mean().rating.plot(style='o')
plt.ylabel('Average rating')
plt.title('Most rated movies')
plt.show()
```

```python id="m-s0rPhSXJNa" executionInfo={"status": "ok", "timestamp": 1628503317710, "user_tz": -330, "elapsed": 421, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
insights.append('Most rated movies also tend to be most liked movies.')
```

```python colab={"base_uri": "https://localhost:8080/"} id="jEe19EPyJyWg" executionInfo={"status": "ok", "timestamp": 1628503322788, "user_tz": -330, "elapsed": 466, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="110caf53-89e9-4155-eb8e-a2f8ead2758e"
# Which is the best most popular movie ever??
tmp = ratings.set_index('movieId').loc[dftmp.index[dftmp.num_ratings>100]].groupby('movieId').mean()
best = movies.set_index('movieId').loc[tmp.rating.idxmax].title
print ('Best most popular movie ever is...%s' %best)
```
