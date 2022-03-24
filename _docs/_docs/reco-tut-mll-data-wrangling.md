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

```python id="1uICbB4nDexm" executionInfo={"status": "ok", "timestamp": 1628518838553, "user_tz": -330, "elapsed": 825, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-mll"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python colab={"base_uri": "https://localhost:8080/"} id="EFnuEM16DqQd" executionInfo={"status": "ok", "timestamp": 1628518840886, "user_tz": -330, "elapsed": 2345, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bce6fa09-dadd-4c86-dabf-5d378c65c625"
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

```python id="7bkm0Tb0DqQq" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628519156617, "user_tz": -330, "elapsed": 424, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="76cc9769-befb-48e3-f76b-874c7a1b36ff"
!git status
```

```python id="9nEA2fSADqQr" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628519163769, "user_tz": -330, "elapsed": 3508, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cfa68bfe-dd93-49de-b761-5ec8d100253d"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

<!-- #region id="J85QGBmjGys7" -->
---
<!-- #endregion -->

<!-- #region id="Q-ljt1LLHbeI" -->
## Setup
<!-- #endregion -->

```python id="8fFuc9JbGzHk" executionInfo={"status": "ok", "timestamp": 1628518851950, "user_tz": -330, "elapsed": 444, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

<!-- #region id="jZNhNwy6HgVI" -->
---
<!-- #endregion -->

<!-- #region id="f5XvsquMHd6i" -->
## Data Loading
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="tD8_pVZ_HF7F" executionInfo={"status": "ok", "timestamp": 1628518853389, "user_tz": -330, "elapsed": 997, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="83ff05de-7967-4e93-a8a6-606b16d0f8cf"
movies = pd.read_parquet('./data/bronze/movies.parquet.gzip')
movies.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="NqgrCFajHkS1" executionInfo={"status": "ok", "timestamp": 1628518856924, "user_tz": -330, "elapsed": 736, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ce6b8cb0-97e7-445b-a8b0-43913122e1aa"
movies.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="Xyn8MgPcHSDV" executionInfo={"status": "ok", "timestamp": 1628518856926, "user_tz": -330, "elapsed": 20, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3eafa71b-bfb1-4d8d-a2a4-8f4ed2a62b20"
ratings = pd.read_parquet('./data/bronze/ratings.parquet.gzip')
ratings.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="UvjEvvxsHoN2" executionInfo={"status": "ok", "timestamp": 1628518856928, "user_tz": -330, "elapsed": 18, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="65c1c791-64ad-4c6f-e13f-daa423253812"
ratings.info()
```

<!-- #region id="_9xp5zuuHpW4" -->
---
<!-- #endregion -->

<!-- #region id="qblTc6AEHo82" -->
## Wrangling
<!-- #endregion -->

<!-- #region id="hXaWZRFGHs7y" -->
Organize ratings
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="YWKIv87lHv9Z" executionInfo={"status": "ok", "timestamp": 1628518861175, "user_tz": -330, "elapsed": 750, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ed865949-289c-4eba-9eb0-7304ca0cd422"
ratings.sort_values(by='movieId', inplace=True)
ratings.reset_index(inplace=True, drop=True)
ratings.head()
```

<!-- #region id="NqtjkcaoIuP8" -->
Modify rating timestamp format (from seconds to datetime year)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="88MlhqbGIn6B" executionInfo={"status": "ok", "timestamp": 1628518862369, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c639370c-b4a8-40f9-ec9a-30374273442c"
ratings.timestamp = pd.to_datetime(ratings.timestamp, unit='s', origin='unix')
ratings.head()
```

<!-- #region id="53xDrY5FH5Uy" -->
Split title and release year in separate columns in movies dataframe. Convert year to timestamp.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="8BC7m-sgH1os" executionInfo={"status": "ok", "timestamp": 1628518862804, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="11021e7c-1c5e-47c9-d5c2-9af50da789f7"
movies['year'] = movies.title.str.extract("\((\d{4})\)", expand=True)
movies.year = pd.to_datetime(movies.year, format='%Y')
movies.year = movies.year.dt.year # As there are some NaN years, resulting type will be float (decimals)
movies.title = movies.title.str[:-7]
movies.head()
```

<!-- #region id="4tJ9jsrAIPHK" -->
Categorize movies genres properly
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="8pQmPEOWIKo1" executionInfo={"status": "ok", "timestamp": 1628518864389, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3533a318-ea47-4d77-a1d0-be1b163bfbac"
genres_unique = pd.DataFrame(movies.genres.str.split('|').tolist()).stack().unique()
genres_unique = pd.DataFrame(genres_unique, columns=['genres']) # Format into DataFrame to store later
genres_unique.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 326} id="su9jZfC3IRvt" executionInfo={"status": "ok", "timestamp": 1628518864828, "user_tz": -330, "elapsed": 448, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2c53f1c5-ec3a-4281-e494-98e90bdc5538"
movies = movies.join(movies.genres.str.get_dummies().astype(bool))
movies.drop('genres', inplace=True, axis=1)
movies.head()
```

<!-- #region id="vDZDcX0UJI8q" -->
Check and clean NaN values
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="LR8QB57DJIKG" executionInfo={"status": "ok", "timestamp": 1628518866310, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2da16216-5dca-4634-d61d-c50f570f9f78"
print ("Number of movies Null values: ", max(movies.isnull().sum()))
print ("Number of ratings Null values: ", max(ratings.isnull().sum()))
movies.dropna(inplace=True)
ratings.dropna(inplace=True)
```

```python id="pAjRZrpHJCVG"
movies.sort_values(by='movieId', inplace=True)
movies.reset_index(inplace=True, drop=True)

ratings.sort_values(by='movieId', inplace=True)
ratings.reset_index(inplace=True, drop=True)
```

```python id="XAAhICslTVZ8"
!mkdir ./data/silver
```

```python id="CFrCIXRPTX-t"
movies.to_parquet('./data/silver/movies.parquet.gzip', compression='gzip')
ratings.to_parquet('./data/silver/ratings.parquet.gzip', compression='gzip')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="ggjeW2nTSgS-" executionInfo={"status": "ok", "timestamp": 1628519132873, "user_tz": -330, "elapsed": 431, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="db97184c-14c0-497f-e4fb-fbc141196120"
### Creating joined dataset
movies = pd.read_parquet('./data/bronze/movies.parquet.gzip')
ratings = pd.read_parquet('./data/bronze/ratings.parquet.gzip')
ratings.columns = ['user_id', 'item_id', 'rating', 'timestamp']
ratings.sort_values(by='item_id', inplace=True)
ratings.reset_index(inplace=True, drop=True)
ratings.timestamp = pd.to_datetime(ratings.timestamp, unit='s', origin='unix')
movies.columns = ['item_id', 'title', 'genres']
movies['year'] = movies.title.str.extract("\((\d{4})\)", expand=True)
movies.year = pd.to_datetime(movies.year, format='%Y')
movies.year = movies.year.dt.year
movies.title = movies.title.str[:-7]
movie_ratings = pd.merge(ratings, movies, on='item_id')
movie_ratings.head()
```

```python id="23H3e_wETdMG" executionInfo={"status": "ok", "timestamp": 1628519149170, "user_tz": -330, "elapsed": 413, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
movie_ratings.to_parquet('./data/silver/movie_ratings.parquet.gzip', compression='gzip')
```
