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

```python id="UV_mis-jdwLd" executionInfo={"status": "ok", "timestamp": 1628672913121, "user_tz": -330, "elapsed": 576, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-mlh"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python id="KRGLEjqMd3dV" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628672916875, "user_tz": -330, "elapsed": 3217, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8351b547-2c2f-4c86-f975-fbfb9ab0dc58"
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

<!-- #region id="oYyiMQC8eIQH" -->
---
<!-- #endregion -->

<!-- #region id="9vXKzTEeqvfZ" -->
# Exploratory Data Analysis

In this notebook we explore the MovieLens 100k dataset.


*   Find missing/null values
*   Examine the distribution of ratings
*   Examine movies and users with most reviews
*   Examine correlation between time and reviews


<!-- #endregion -->

<!-- #region id="j79HVjHKtQGo" -->
# Imports
<!-- #endregion -->

```python id="6QFb137tY185" executionInfo={"status": "ok", "timestamp": 1628673441861, "user_tz": -330, "elapsed": 397, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import requests
import seaborn as sns
from scipy.stats.stats import pearsonr
from tqdm import tqdm
```

<!-- #region id="5UAnfzLKq8Oh" -->
# Prepare data
<!-- #endregion -->

```python id="YAHS9ItRqmbE" colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"status": "ok", "timestamp": 1628673616305, "user_tz": -330, "elapsed": 689, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e6768f4c-f4ce-4a0e-bc1d-7d2b714f6dc4"
# Load reviews.
fp = os.path.join('./data/bronze', 'u.data')
raw_data = pd.read_csv(fp, sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'])
raw_data.head()
```

```python id="VAWCFj4ux2sA" colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"status": "ok", "timestamp": 1628673624557, "user_tz": -330, "elapsed": 518, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8bc8a8ee-709b-4b81-a393-559cb346b6ee"
# Load movie titles.
fp = os.path.join('./data/bronze', 'u.item')
movie_titles = pd.read_csv(fp, sep='|', names=['movieId', 'title'], usecols = range(2), encoding='iso-8859-1')
movie_titles.head()
```

```python id="oeRGj4z8yeCK" colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"status": "ok", "timestamp": 1628673636182, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="19f686b5-66c4-4568-af63-a133a6ca5e9a"
# Merge dataframes.
raw_data = raw_data.merge(movie_titles, how='left', on='movieId')
raw_data.head()
```

```python id="G_zAF0hYAHEZ" colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"status": "ok", "timestamp": 1628673648227, "user_tz": -330, "elapsed": 457, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ebce283c-3c3b-4ccd-a9cb-13237a521192"
# Change timestamp to datatime.
raw_data.timestamp = pd.to_datetime(raw_data.timestamp, unit='s')
raw_data.head()
```

<!-- #region id="bRYN5MjTrJah" -->
# Exploration
<!-- #endregion -->

<!-- #region id="74k4_H20sxdu" -->
## Unique and null values
<!-- #endregion -->

<!-- #region id="EIxA4LenrGzR" -->
We first see that there are 100k observations in our dataset. There are 943 unique users and 1682 unique movies, and the rating system is out of 5. We then check to see if there are any missing data points in the set, which we find there are none.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 221} id="drkZDk91rQFA" executionInfo={"status": "ok", "timestamp": 1628673672533, "user_tz": -330, "elapsed": 823, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f12696ed-bd2b-4ac5-a182-154bb389defe"
print(f'Shape: {raw_data.shape}')
raw_data.sample(5, random_state=123)
```

```python colab={"base_uri": "https://localhost:8080/"} id="xeAirhNAs4tT" executionInfo={"status": "ok", "timestamp": 1628673676394, "user_tz": -330, "elapsed": 474, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="06c52cd6-ad2f-4c7c-f284-389285d206d2"
raw_data.nunique()
```

```python colab={"base_uri": "https://localhost:8080/"} id="ao3Ca1L-q4uy" executionInfo={"status": "ok", "timestamp": 1628673678441, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d2e2c3e7-cbcb-4850-d38c-2d5f5f0ac3fa"
raw_data.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 221} id="Zir1ffvGyWGK" executionInfo={"status": "ok", "timestamp": 1628673684678, "user_tz": -330, "elapsed": 440, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="52f87347-3aea-4d00-b588-f270a02fc585"
print(f'Shape: {movie_titles.shape}')
movie_titles.sample(5, random_state=123)
```

<!-- #region id="Il1s2lAktUtX" -->
## Summary Stats
<!-- #endregion -->

<!-- #region id="qoObzQE9rhfL" -->
### Ratings

Next, we look at the summary statistics of each feature in the dataset. We notice that the mean rating of the movies is 3.5 and that the minimum and maximum rating is 1 and 5 respectivle, and that the ratings are discrete (no in-between values). The most common rating is 4, with the second most common being 3. There are very few reviews with a 1 rating (about 6000/100,000). In fact looking at our boxplots, reviews where the movie is rated 1 might even be considered an outlier.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 297} id="I9SKl7VAq5vI" executionInfo={"status": "ok", "timestamp": 1628673689935, "user_tz": -330, "elapsed": 419, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4c4715dc-5505-4ef8-9b51-2b6685e631e3"
raw_data.describe()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 334} id="iSwGFQxtsLok" executionInfo={"status": "ok", "timestamp": 1628673692204, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cfe96d27-95f9-4694-d06d-2162518c2c28"
plt.figure(figsize=(7,5))
sns.histplot(raw_data.rating)
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 388} id="oLpmVgyHsQzh" executionInfo={"status": "ok", "timestamp": 1628673730716, "user_tz": -330, "elapsed": 485, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f7a672d9-997d-4534-f970-d00febd949d6"
plt.figure(figsize=(10,6))
sns.boxplot(x = raw_data.rating)
plt.show()
```

<!-- #region id="FTVWQTZH28_7" -->
### Time

Actual reviews were made starting from September 20, 1997 to April 22, 1998, about 7 months of data.

Actual movies reviewed were released from 1922 to 1998, with 4 years missing in that timespan. There are also a couple of movies with no year given. We assigned these movies to year 0.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="VBgDF0CMBkQ6" executionInfo={"status": "ok", "timestamp": 1628673758689, "user_tz": -330, "elapsed": 418, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="72cc6741-4d10-44f1-a1e3-e6a665a9fe84"
raw_data.timestamp.describe(datetime_is_numeric=True)
```

```python id="yEB87ebKC8fe" executionInfo={"status": "ok", "timestamp": 1628673773253, "user_tz": -330, "elapsed": 572, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def get_year(title):
    year=re.search(r'\(\d{4}\)', title)
    if year:
        year=year.group()
        return int(year[1:5])
    else:
        return 0
```

```python colab={"base_uri": "https://localhost:8080/"} id="lQbarLQgCtRt" executionInfo={"status": "ok", "timestamp": 1628673774651, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4f29197f-5480-4f92-c676-5686c791b40f"
raw_data['year'] = raw_data.title.apply(get_year)
raw_data.year.sort_values().unique()
```

```python colab={"base_uri": "https://localhost:8080/"} id="5ZdTXGSfEEeH" executionInfo={"status": "ok", "timestamp": 1628673775126, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="798a1e0e-64b0-4c9b-bcff-a72f6b554e81"
raw_data[['year']].nunique()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 279} id="pFqjNddrG2VA" executionInfo={"status": "ok", "timestamp": 1628673777559, "user_tz": -330, "elapsed": 1010, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9621bd5d-ca45-4668-903f-0bcd8194170a"
sns.histplot(raw_data['year'][raw_data['year'] != 0])
plt.show()
```

<!-- #region id="93Yb1nzL3UpS" -->
## Users with most reviews

The most movies single user has reviewed is 737 reviews. The minimum number of reviews a user has reviewed in the dataset is 20. This is good since when creating recommendation systems, you want users with lots or reviews, allowing for us to test our recomendations. We also notice that most users reviewed less than 65 movies.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="iEbXXr343k2_" executionInfo={"status": "ok", "timestamp": 1628673785373, "user_tz": -330, "elapsed": 977, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="937c185f-bc32-4fcc-da11-6b4070e37dcc"
users_count = raw_data.groupby('userId')['rating'].count().sort_values(ascending=False).reset_index()
users_count
```

```python colab={"base_uri": "https://localhost:8080/", "height": 388} id="hgokbiG44RyH" executionInfo={"status": "ok", "timestamp": 1628673788036, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0691db45-e754-4131-c3ba-d653ca3262ff"
# Plot how many movies a user reviewed
plt.figure(figsize=(10, 6))
fig = sns.histplot(users_count['rating'])
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="8iSXp_OM68LH" executionInfo={"status": "ok", "timestamp": 1628673789116, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d988d164-405a-4176-e3ce-4fae73d749ff"
users_count['rating'].median()
```

<!-- #region id="v6fTG72kxjnC" -->
## Movies with most reviews

As we can expect, popular movies such as 'Star Wars' and 'Toy Story' have the most reviews. The highest number of reviews is 583 while the lowest number of reviews is 1.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="XCVoLpIQ0Dex" executionInfo={"status": "ok", "timestamp": 1628673789572, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="eff8cc79-49a3-451d-94f8-be77932fc955"
movies_count = raw_data.groupby('title')['rating'].count().sort_values(ascending=False).reset_index()
movies_count
```

```python colab={"base_uri": "https://localhost:8080/", "height": 644} id="zLMPmg2-0NOs" executionInfo={"status": "ok", "timestamp": 1628673795399, "user_tz": -330, "elapsed": 2584, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="70189f6c-55b5-4bf2-ca90-b44e07ae519a"
# Plot 50 most reviewed movies.
plt.figure(figsize=(15,10))
fig = sns.barplot(x=movies_count.head(50)['title'], y=movies_count.head(50)['rating'])
fig.set_xticklabels(fig.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.tight_layout()
plt.show()
```

<!-- #region id="1DrTjMKDuUhQ" -->
## Time correlation

Lastly we will examine if there is a correlation between then the movie was made and the rating given.
<!-- #endregion -->

<!-- #region id="7X9A299HFEqJ" -->
## Year movie released vs rating
<!-- #endregion -->

<!-- #region id="R-v_OBcoGZU3" -->
With a correlation coefficient of -0.1050, there is a tiny inverse relationship between when a movie was released and the rating given to it. The p-value is also much lower than 0.05 meaning that we can conclude that the correlation is statistically significant. Older movies were rating more generously than newer movies.

This could be because older movies do not have as many ratings as the newer movies. People who would actually watch and rate old movies from the 20s and 30s would typically be film enthusiasts and thus have a bias towards older movies.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 388} id="9P0VXqaxJRbt" executionInfo={"status": "ok", "timestamp": 1628673803421, "user_tz": -330, "elapsed": 777, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f73308b9-75c8-4293-c187-11f8bcbe3279"
plt.figure(figsize=(10, 6))
mean_rating = raw_data.groupby('year')['rating'].mean().reset_index()
mean_rating = mean_rating[mean_rating.year != 0]
sns.lineplot(x=mean_rating.year, y=mean_rating.rating)
plt.ylabel('avg_rating')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="mHasBND6FrYL" executionInfo={"status": "ok", "timestamp": 1628673807475, "user_tz": -330, "elapsed": 546, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="86dbb21f-1a4b-4e39-ffdb-bac1eb2537b2"
pearsonr(raw_data.year, raw_data.rating)
```
