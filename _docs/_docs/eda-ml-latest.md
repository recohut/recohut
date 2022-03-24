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

<!-- #region id="o6wusT5VNzF4" -->
# EDA of ML-Latest-small Dataset
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="LJ2hyoJilWk6" executionInfo={"status": "ok", "timestamp": 1638019880545, "user_tz": -330, "elapsed": 934, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7e8ebeac-a088-4b26-948c-e270e4870fad"
!wget -q --show-progress http://files.grouplens.org/datasets/movielens/ml-latest-small.zip
!unzip ml-latest-small.zip
```

```python id="AczdI7z5mPwk"
import matplotlib.pyplot as plt
import pandas as pd
import zipfile
import urllib.request
import sys
import os
```

```python id="T15LdRgNmMHs"
DOWNLOAD_DESTINATION_DIR = '/content/ml-latest-small'
```

```python id="qMxJlAhwmJAR"
ratings_path = os.path.join(DOWNLOAD_DESTINATION_DIR, 'ratings.csv')
ratings = pd.read_csv(
    ratings_path,
    sep=',',
    names=["userid", "itemid", "rating", "timestamp"],
    skiprows=1
)

movies_path = os.path.join(DOWNLOAD_DESTINATION_DIR, 'movies.csv')
movies = pd.read_csv(
    movies_path,
    sep=',',
    names=["itemid", "title", "genres"],
    encoding='latin-1',
    skiprows=1
)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="WeswVuzPmVU9" executionInfo={"status": "ok", "timestamp": 1638020028701, "user_tz": -330, "elapsed": 756, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b4266e8c-7510-401e-8232-b8cd73737714"
ratings.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="XnAz-wnlmhHM" executionInfo={"status": "ok", "timestamp": 1638020032893, "user_tz": -330, "elapsed": 454, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b3998c6c-4fe2-44f9-cac4-58c4368eb3bc"
movies.head()
```

<!-- #region id="Fg-adKwlmk7E" -->
### Histogram of ratings
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 302} id="4rkLuGshmh4f" executionInfo={"status": "ok", "timestamp": 1638020164411, "user_tz": -330, "elapsed": 459, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7a47e222-ae5f-4e80-c0d8-f6a63e1201d5"
ratings.groupby('rating').size().plot(kind='bar')
```

<!-- #region id="UagZLtohnCZk" -->
Ratings range from 0.5 to 5.0, with a step of 0.5. The above histogram presents the repartition of ratings in the dataset. the two most commun ratings are 4.0 and 3.0 and the less commun ratings are 0.5 and  1.5
<!-- #endregion -->

<!-- #region id="mETzLloMnILq" -->
### Average ratings of movies
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 709} id="ErMA0yq4nKm-" executionInfo={"status": "ok", "timestamp": 1638020205305, "user_tz": -330, "elapsed": 2069, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2eba456c-afad-4648-fd7f-5e4eb3b5654a"
movie_means = ratings.join(movies['title'], on='itemid').groupby('title').rating.mean()
movie_means[:50].plot(kind='bar', grid=True, figsize=(16,6), title="mean ratings of 50 movies")
```

<!-- #region id="9dqn7DONnL9I" -->
### 30 most rated movies vs. 30 less rated movies
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 716} id="FuvgqaQonOj8" executionInfo={"status": "ok", "timestamp": 1638020221784, "user_tz": -330, "elapsed": 2257, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d276f277-30df-46fd-ece5-e409a38c397e"
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16,4), sharey=True)
movie_means.nlargest(30).plot(kind='bar', ax=ax1, title="Top 30 movies in data set")
movie_means.nsmallest(30).plot(kind='bar', ax=ax2, title="Bottom 30 movies in data set")
```

<!-- #region id="iiUICv1NnVg3" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="G09AvueCnVg6" executionInfo={"status": "ok", "timestamp": 1638020261393, "user_tz": -330, "elapsed": 3500, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d1d362f3-3aae-4f91-8ce5-43ba0260c944"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="fJf0Yn5YnVg7" -->
---
<!-- #endregion -->

<!-- #region id="zwp2fxcunVg7" -->
**END**
<!-- #endregion -->
