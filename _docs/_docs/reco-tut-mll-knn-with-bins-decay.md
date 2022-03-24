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

```python id="1uICbB4nDexm" executionInfo={"status": "ok", "timestamp": 1628519273254, "user_tz": -330, "elapsed": 787, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-mll"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python colab={"base_uri": "https://localhost:8080/"} id="EFnuEM16DqQd" executionInfo={"status": "ok", "timestamp": 1628519275734, "user_tz": -330, "elapsed": 2019, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a2ae111b-32f6-44f2-953e-c243d27994eb"
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

```python id="9nEA2fSADqQr"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

<!-- #region id="BxUxe2OtbV3Y" -->
---
<!-- #endregion -->

```python id="wQATOAHpdmAi" executionInfo={"status": "ok", "timestamp": 1628520869833, "user_tz": -330, "elapsed": 423, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt
from math import ceil
from collections import defaultdict
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="vOMu7TycUF-z" executionInfo={"status": "ok", "timestamp": 1628519322789, "user_tz": -330, "elapsed": 1348, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="16a3315a-8b14-4d85-bfeb-e7554a7308c7"
movie_ratings = pd.read_parquet('./data/silver/movie_ratings.parquet.gzip')
movie_ratings.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 235} id="myi66wQxdHxy" executionInfo={"status": "ok", "timestamp": 1628519361492, "user_tz": -330, "elapsed": 526, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cb347c47-41dd-4f04-bdbc-8728f7d0cf92"
# Most rated movies
ratings = pd.DataFrame(movie_ratings.groupby('title')['rating'].mean())
ratings['No_of_ratings'] = pd.DataFrame(movie_ratings.groupby('title')['rating'].count())
ratings.sort_values(by=['No_of_ratings'], ascending=False).head()
```

<!-- #region id="3cWQlX1MUyWC" -->
Discovering K Nearest Neighbours For Movie 'Shawshank Redemption, The'
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 235} id="b8yYWLcXfHpq" executionInfo={"status": "ok", "timestamp": 1628519393377, "user_tz": -330, "elapsed": 438, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d8550f9c-d03e-45ca-d8cd-74f6d693327a"
user_movie_matrix = movie_ratings.pivot_table(index='user_id', columns='title', values='rating')
user_movie_matrix.iloc[:5,:5]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 235} id="4GmResmkUmxu" executionInfo={"status": "ok", "timestamp": 1628519531808, "user_tz": -330, "elapsed": 461, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9d11e4bc-96da-442b-f0f9-c040cc2de9e6"
# Get Movie Correlations
corrs = user_movie_matrix.corr(method='pearson', min_periods=20)
redemption_corrs = corrs['Shawshank Redemption, The']
redemption_corrs.dropna(inplace=True)
movies_like_redemption = pd.DataFrame(redemption_corrs)
movies_like_redemption.columns= ['correlation']
movies_like_redemption.sort_values(by='correlation', ascending=False, inplace=True)
movies_like_redemption.head()
```

<!-- #region id="-b1Fdv50U1LV" -->
Designing basic KNN
<!-- #endregion -->

```python id="tLhbACkHVghE"
corrs = user_movie_matrix.corr(method='pearson',min_periods=min_common_elements)
```

```python id="QGtOStPpVD_U" executionInfo={"status": "ok", "timestamp": 1628519720813, "user_tz": -330, "elapsed": 446, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def print_knn(movie_name, k=10, min_common_elements=20):
    movie_corrs = corrs[movie_name]
    movie_corrs.dropna(inplace=True)
    movies_alike = pd.DataFrame(movie_corrs)
    movies_alike.columns= ['correlation']
    movies_alike.sort_values(by='correlation', ascending=False, inplace=True)
    return movies_alike.head(k)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 390} id="H2vHMgUPVZji" executionInfo={"status": "ok", "timestamp": 1628519725505, "user_tz": -330, "elapsed": 487, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="329002da-f815-4129-9e88-db8328473eff"
print_knn("Shawshank Redemption, The", k=10)
```

<!-- #region id="LkDj-jNIVxpK" -->
Designing KNN Function With Basic Filters
<!-- #endregion -->

```python id="kBGucoWfWZzU" executionInfo={"status": "ok", "timestamp": 1628519930360, "user_tz": -330, "elapsed": 451, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def print_filtered_knn(rating_matrix, movie_name, k=10, min_common_elements=20, filter_date=dt.strptime('01/01/18 13:55:26', '%m/%d/%y %H:%M:%S')):
    user_movie_matrix = rating_matrix[rating_matrix.timestamp < filter_date].pivot_table(index='user_id', columns='title', values='rating')
    corrs = user_movie_matrix.corr(method='pearson',min_periods=min_common_elements)
    movie_corrs = corrs[movie_name]
    movie_corrs.dropna(inplace=True)
    movies_alike = pd.DataFrame(movie_corrs)
    movies_alike.columns= ['correlation']
    movies_alike.sort_values(by='correlation', ascending=False, inplace=True)
    return movies_alike.head(k)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 390} id="sxU_VwveVvUV" executionInfo={"status": "ok", "timestamp": 1628519930834, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="66f5b88d-648c-4dfe-98df-112fa4489615"
print_filtered_knn(movie_ratings,"Shawshank Redemption, The", k=10)
```

<!-- #region id="soTttyidWTkp" -->
Designing KNN Func With Ranges
<!-- #endregion -->

```python id="Xa2Xs7BkWIVq" executionInfo={"status": "ok", "timestamp": 1628519954960, "user_tz": -330, "elapsed": 630, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def print_ranged_knn(rating_matrix, movie_name, k=10, min_common_elements=20, 
                       range_start_dt=dt.strptime('01/01/01 13:55:26', '%m/%d/%y %H:%M:%S'),
                       range_end_dt=dt.strptime('01/01/06 13:55:26', '%m/%d/%y %H:%M:%S')):
                    # With default range of 2001 and 2006
    user_movie_matrix = rating_matrix[(rating_matrix.timestamp >= range_start_dt) & (rating_matrix.timestamp < range_end_dt)].pivot_table(index='user_id', columns='title', values='rating')
    corrs = user_movie_matrix.corr(method='pearson',min_periods=min_common_elements)
    movie_corrs = corrs[movie_name]
    movie_corrs.dropna(inplace=True)
    movies_alike = pd.DataFrame(movie_corrs)
    movies_alike.columns= ['correlation']
    movies_alike.sort_values(by='correlation', ascending=False, inplace=True)
    return movies_alike.head(k)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 266} id="TmqWT1BwWMvn" executionInfo={"status": "ok", "timestamp": 1628519956563, "user_tz": -330, "elapsed": 1168, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c5566cfa-e1c1-41f2-f8b3-873ddb5ea81d"
print_ranged_knn(movie_ratings,"Shawshank Redemption, The", k=10)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 390} id="Whi8BOLRWMtA" executionInfo={"status": "ok", "timestamp": 1628519958253, "user_tz": -330, "elapsed": 1697, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f83c9624-6f1f-4f26-8dc3-a814e2cdb2f6"
print_ranged_knn(movie_ratings,"Shawshank Redemption, The", k=10, 
                   range_start_dt=dt.strptime('01/01/06 13:55:26', '%m/%d/%y %H:%M:%S'),
                   range_end_dt=dt.strptime('01/01/11 13:55:26', '%m/%d/%y %H:%M:%S'))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 390} id="rOQXCaBmWMj_" executionInfo={"status": "ok", "timestamp": 1628519959636, "user_tz": -330, "elapsed": 1390, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5748fbc4-b2ea-48bd-bfd0-8ef886c28793"
print_ranged_knn(movie_ratings,"Shawshank Redemption, The", k=10, 
                   range_start_dt=dt.strptime('01/01/11 13:55:26', '%m/%d/%y %H:%M:%S'),
                   range_end_dt=dt.strptime('01/01/16 13:55:26', '%m/%d/%y %H:%M:%S'))
```

<!-- #region id="faXFcNwZXQP9" -->
## KNN with bins
<!-- #endregion -->

```python id="h4-I908KXZhm" executionInfo={"status": "ok", "timestamp": 1628520246928, "user_tz": -330, "elapsed": 421, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def get_knn_with_bin(rating_matrix, movie_name, k=10, min_common_elements=20, 
                         start_year=95,
                         end_year=20):
    user_movie_matrix = rating_matrix[(rating_matrix.timestamp >= dt(start_year, 1, 15)) & (rating_matrix.timestamp < dt(end_year, 1, 15))].pivot_table(index='user_id', columns='title', values='rating')
    corrs = user_movie_matrix.corr(method='pearson',min_periods=min_common_elements)
    movie_corrs = corrs[movie_name]
    movie_corrs.dropna(inplace=True)
    movies_alike = pd.DataFrame(movie_corrs)
    movies_alike.columns= ['correlation']
    movies_alike.sort_values(by='correlation', ascending=False, inplace=True)
    print(movies_alike.head(k))
```

```python id="oyQYdGgVXR6n" executionInfo={"status": "ok", "timestamp": 1628520251231, "user_tz": -330, "elapsed": 406, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def print_knn_with_bins(rating_matrix, movie_name, k=10, min_common_elements=20, 
                         start_year=2001,
                         end_year=2021,
                         bin_size_in_years=5):
    while(start_year < end_year):
        bin_end_year = start_year+bin_size_in_years
        bin_start_year = start_year
        print(f"\nRange:{bin_start_year}:{bin_end_year}-->\n")
        get_knn_with_bin(rating_matrix, movie_name, k=15, start_year=start_year, end_year=bin_end_year)
        start_year += bin_size_in_years
```

```python colab={"base_uri": "https://localhost:8080/"} id="rPc7phY3Xjtt" executionInfo={"status": "ok", "timestamp": 1628520258122, "user_tz": -330, "elapsed": 6450, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="743889eb-946e-474d-fa07-843ca3385876"
print_knn_with_bins(movie_ratings,"Shawshank Redemption, The", k=15)
```

<!-- #region id="xb6cR-icZevh" -->
Designing KNN Func With Basic Decay
<!-- #endregion -->

```python id="3DHky4GUZglH" executionInfo={"status": "ok", "timestamp": 1628520743526, "user_tz": -330, "elapsed": 740, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def get_bin(movie_ratings_data, movie_name, min_common_elements=20, 
                         start_year=1995,
                         end_year=2000):
    user_movie_matrix = movie_ratings_data[(movie_ratings_data.timestamp >= dt(start_year, 1, 15)) & (movie_ratings_data.timestamp < dt(end_year, 1, 15))].pivot_table(index='user_id', columns='title', values='rating')
    corrs = user_movie_matrix.corr(method='pearson', min_periods=min_common_elements)
    movie_corrs = corrs[movie_name]
    movie_corrs.dropna(inplace=True)
    movies_alike = pd.DataFrame(movie_corrs)
    movies_alike.columns= ['correlation']
    movies_alike.sort_values(by='correlation', ascending=False, inplace=True)
    return movies_alike
```

```python colab={"base_uri": "https://localhost:8080/", "height": 390} id="8J4SJ5yUZjN0" executionInfo={"status": "ok", "timestamp": 1628520768262, "user_tz": -330, "elapsed": 24743, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="74b71b5b-4df9-4e3e-a4b9-f2eb8832e42f"
x = get_bin(movie_ratings,"Shawshank Redemption, The", start_year=2000, end_year=2020)
x.head(10)
```

```python colab={"base_uri": "https://localhost:8080/"} id="1PrDmjbxZrlh" executionInfo={"status": "ok", "timestamp": 1628520791773, "user_tz": -330, "elapsed": 453, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5974e1ec-3a18-414d-84ca-e9236399c79b"
print("First Movie Rating: ", movie_ratings['timestamp'].min() )
print("Today: ", dt.now())
```

```python id="974ypTkZZyXR" executionInfo={"status": "ok", "timestamp": 1628520820142, "user_tz": -330, "elapsed": 727, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def print_knn_decay(rating_data, movie_name, k=10, min_common_elements=20,decay_rate=1, decay_interval_in_years=5):
    """
    decay_rate = importance of newer bins, 1 means all equal, 2 means new one 2x imporant than the one before itself
    """
    start_dt = movie_ratings['timestamp'].min()
    end_dt = dt.now()
    n_bins = ceil( (end_dt.year - start_dt.year) / decay_interval_in_years )
    correlation_dict = defaultdict(float)

    for i in range(n_bins):
        bin_start_dt = start_dt
        start_dt = dt(start_dt.year+decay_interval_in_years,1,1) 
        bin_end_dt = start_dt
        #print(f"\nBin {i}: {bin_start_dt}-{bin_end_dt}")
        
        curr_bin = get_bin(movie_ratings,movie_name, start_year=bin_start_dt.year, end_year=bin_end_dt.year)
        
        for index,row in x.iterrows():
            correlation_dict[index] += row.correlation * ((decay_rate ** (i+1)) / ((decay_rate ** (n_bins+1) - 1)/(decay_rate-1)))
            
        #print(curr_bin.query(f"correlation > 0.2 & title != '{movie_name}'").head(k))
    dictlist = list()
    for key, value in correlation_dict.items():
        temp = [key,value]
        dictlist.append(temp)
    sum_correlations = pd.DataFrame(dictlist, columns = ['title','correlation'])

    print(sum_correlations.head(k))
```

```python colab={"base_uri": "https://localhost:8080/"} id="HIj17KpqZ7zv" executionInfo={"status": "ok", "timestamp": 1628520904114, "user_tz": -330, "elapsed": 7790, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5f904878-fb8a-481f-ac53-856aff25d0a7"
print_knn_decay(movie_ratings,"Shawshank Redemption, The", k=10, decay_rate=2, decay_interval_in_years=5)
```

```python colab={"base_uri": "https://localhost:8080/"} id="nwAq0sjEZV8L" executionInfo={"status": "ok", "timestamp": 1628520932762, "user_tz": -330, "elapsed": 12038, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="04a957ea-b87c-4dd0-b1c5-589ae919c287"
print_knn_decay(movie_ratings,"Shawshank Redemption, The", k=10, decay_rate=2, decay_interval_in_years=10)
```
