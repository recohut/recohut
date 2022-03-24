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
    language: python
    name: python3
---

```python id="AQXHs6j3KUn8" executionInfo={"status": "ok", "timestamp": 1621839506840, "user_tz": -330, "elapsed": 1168, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import pandas as pd 
import numpy as np 
import random
import matplotlib.pyplot as plt 
import json
import folium
import networkx as nx
```

<!-- #region id="QDi5x2nvKUoC" -->
# 1. load data
The data file introduction is below:
1. **u.data**:   The full u data set, `100000 ratings` by `943 users` on `1682 items`. Each user has rated at least 20 movies.  Users and items are numbered consecutively from 1.  The data is randomly ordered. The time stamps are unix seconds since 1/1/1970 UTC.   

2. **u.item**: Information about the items (movies); The last `19 fields` are the genres, a 1 indicates the movie is of that genre, a 0 indicates it is not; movies can be in several genres at once. 

3. **u.genre**: A list of the genres.

4. **u.user**: Demographic information about the users

5. **u.occupation**: A list of the occupations.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="23_v5KZaKttk" executionInfo={"status": "ok", "timestamp": 1621839557479, "user_tz": -330, "elapsed": 1153, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="21e68a56-62b1-4b19-c496-1d5644af77ab"
!wget http://files.grouplens.org/datasets/movielens/ml-100k.zip
!unzip ml-100k.zip
```

<!-- #region id="IiUtii1qKUoD" -->
# 2. Data analysis
<!-- #endregion -->

<!-- #region id="UL_4OPgHKUoD" -->
## 2.1 Movie data
<!-- #endregion -->

```python id="1SmYeLZCKUoE" colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"status": "ok", "timestamp": 1621839583070, "user_tz": -330, "elapsed": 391, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9e386fec-994f-468e-ff46-bd0d7ccecd00"
############################### Genre of the movies ############################### 
genre_data= pd.read_csv('ml-100k/u.genre',sep='|',names=["movie_type", "type_id"])
genre_data.head()
```

```python id="rivq_pRLKUoP" executionInfo={"status": "ok", "timestamp": 1621839585141, "user_tz": -330, "elapsed": 4, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
genre_cls = ["unknown", "Action", "Adventure", "Animation", \
              "Childrens", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", \
              "Sci-Fi", "Thriller", "War", "Western"]
```

```python id="Mhehh9ewKUoQ" colab={"base_uri": "https://localhost:8080/", "height": 728} executionInfo={"status": "ok", "timestamp": 1621839586737, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b343d120-12aa-4e95-973f-0306cf0005f4"
############################### Information about the items (movies) ###############################
column_names = ["movie_id", "movie_title", "release_date", "video_release_date", "IMDb_URL", "unknown", "Action", "Adventure", "Animation", \
              "Childrens", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", \
              "Sci-Fi", "Thriller", "War", "Western"]
movies_data = pd.read_csv('ml-100k/u.item',sep='|', names=column_names,encoding = "ISO-8859-1")
movies_data['release_date'] = pd.to_datetime(movies_data['release_date'])
movies_data.rename(columns = {'movie_id':'item_id'}, inplace = True) 
movies_data
```

<!-- #region id="FdhjuSC6KUoR" -->
### Duplicated records
We found that there are movies that share exactly the same information but with different item_id (primary key). And there are exactly 18 movies that has such a duplication so in total 36 records that are not unique. We deal with these duplications later on, after checking if the original and the duplicate are both rated by users.
<!-- #endregion -->

```python id="X_6_dth2KUoR" colab={"base_uri": "https://localhost:8080/", "height": 377} executionInfo={"status": "ok", "timestamp": 1621839590955, "user_tz": -330, "elapsed": 694, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f74ae1d7-2337-48fe-e93f-d11f4ff6b3c7"
duplicated = movies_data[movies_data.duplicated('movie_title', False)].sort_values(by = 'movie_title')
duplicated.head()
```

<!-- #region id="NS_E34xmKUoS" -->
### NaN record
We find that there is a record contains NaN for most of its attributes. However, we later on also found that there are users who did rate this item. So we decide not to drop this record.
<!-- #endregion -->

```python id="xneubLZIKUoT" colab={"base_uri": "https://localhost:8080/", "height": 117} executionInfo={"status": "ok", "timestamp": 1621839603569, "user_tz": -330, "elapsed": 538, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="87d55088-d467-43e4-fdfa-820fcabde95a"
movies_data[movies_data.release_date.isnull()]
```

```python id="tzEuDgyYKUoT" colab={"base_uri": "https://localhost:8080/", "height": 117} executionInfo={"status": "ok", "timestamp": 1621839605941, "user_tz": -330, "elapsed": 21, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="eedb8a12-2d5e-473b-e9d3-34d9982625dd"
movies_data[movies_data.index == 266]
```

<!-- #region id="24YZ7OajKUoU" -->
### Year Cleaning
The movies lie in a span of 77 years.
<!-- #endregion -->

```python id="No1kgvIoKUoU" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1621839607674, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c7877d1c-214f-4f55-88d3-813c3b7f967f"
l = sorted(movies_data.release_date.dt.year.unique().tolist())
max(l)  - min(l) + 1
```

<!-- #region id="ceygC3f-KUoV" -->
#### Histogram of movies w.r.t. release year
<!-- #endregion -->

```python id="VcYCpnzYKUoV" colab={"base_uri": "https://localhost:8080/", "height": 609} executionInfo={"status": "ok", "timestamp": 1621839610783, "user_tz": -330, "elapsed": 925, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a9a70fcd-9882-4985-8269-980fc42fafea"
movies_data.release_date.hist(bins = 77, figsize = (10, 10))
```

<!-- #region id="Ara3w1DEKUoV" -->
Observing from the histogram showing the number of movies for each each, we notice that the movies mainly are released during 1990 and 1998. In order to facilitate the computation of similarity, we wish to aggregate years in which too few moives are released.
<!-- #endregion -->

```python id="DlwEvLtRKUoW" executionInfo={"status": "ok", "timestamp": 1621839614037, "user_tz": -330, "elapsed": 3, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def compute_year_label(row):
    year = row['release_date'].year
    
    if year <= 1990 or np.isnan(year):
        return 1990
    else:
        return year
```

```python id="gfT43SMgKUoW" executionInfo={"status": "ok", "timestamp": 1621839615121, "user_tz": -330, "elapsed": 565, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
movies_data['year_label'] = movies_data.apply(lambda row: compute_year_label(row), axis = 1)
```

```python id="D8z8GFDiKUoW" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1621839615123, "user_tz": -330, "elapsed": 566, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="741146fc-ee36-4f5b-9984-9b70bdbb2b50"
movies_data.year_label.unique()
```

<!-- #region id="q5HZSwt8KUoY" -->
### 2.1.1 statistical infos
<!-- #endregion -->

<!-- #region id="NIlrXnIGKUoY" -->
#### Histogram of movies under each year label.
<!-- #endregion -->

```python id="4nJ9IzpNKUoZ" colab={"base_uri": "https://localhost:8080/", "height": 422} executionInfo={"status": "ok", "timestamp": 1621839619621, "user_tz": -330, "elapsed": 1017, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5e2e700a-0446-4a58-e65b-559de9863453"
movies_data.year_label.value_counts().sort_index()\
.plot(kind = 'bar', rot = 45, figsize = (10, 6), title = 'Distribution of Movies Across Years')
```

<!-- #region id="wuvVRwJUKUoa" -->
#### The number of movies that falls in each genre.
<!-- #endregion -->

```python id="zUQ1xpiXKUob" colab={"base_uri": "https://localhost:8080/", "height": 382} executionInfo={"status": "ok", "timestamp": 1621839622118, "user_tz": -330, "elapsed": 968, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f95c83da-1380-449c-b65c-e512b6445f95"
movies_data.sum()[movies_data.columns[5:-1]].plot(kind = 'bar', figsize = (15, 5), rot = 45)
```

<!-- #region id="VdfpjZiKKUob" -->
Below we take a glimpse of the percentage of each genere's movies in each year.
<!-- #endregion -->

```python id="Ee_hFcwrKUoc" colab={"base_uri": "https://localhost:8080/", "height": 561} executionInfo={"status": "ok", "timestamp": 1621839625245, "user_tz": -330, "elapsed": 1288, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="75c69b70-adf9-4843-f429-9d05fdd652b7"
movies_data[movies_data.movie_title != 'unknown'].groupby('year_label').sum()[genre_cls].T\
        .plot(kind = 'bar', rot=45, figsize=(15,8), title = 'Distribution of Movie Genre')
```

<!-- #region id="fo-lSA_wKUoc" -->
Below we take a look at year 1995 - 1998 as an exmaple.
<!-- #endregion -->

```python id="q5Wt9SnIKUod" executionInfo={"status": "ok", "timestamp": 1621839625893, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
movie_1995 = movies_data[movies_data['release_date'].dt.year == 1995]
movie_1996 = movies_data[movies_data['release_date'].dt.year == 1996]
movie_1997 = movies_data[movies_data['release_date'].dt.year == 1997]
movie_1998 = movies_data[movies_data['release_date'].dt.year == 1998]
```

```python id="i3WecmqaKUod" colab={"base_uri": "https://localhost:8080/", "height": 506} executionInfo={"status": "ok", "timestamp": 1621839626651, "user_tz": -330, "elapsed": 762, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b1088b02-4ab4-4f47-932b-b8700e2da5eb"
Year = {}
Year[1995] = movie_1995[genre_cls].sum()/len(movie_1995)
Year[1996] = movie_1996[genre_cls].sum()/len(movie_1996)
Year[1997] = movie_1997[genre_cls].sum()/len(movie_1997)
Year[1998] = movie_1998[genre_cls].sum()/len(movie_1998)
movie_year = pd.DataFrame(Year) 
axes = movie_year.plot.bar(rot=45,figsize=(15,7))
axes.set_title('genres percentage over year')
```

<!-- #region id="TlGwqnyXKUoe" -->
We conclude that the distribution of movie genres are generally balanced and hence are not biased.
<!-- #endregion -->

<!-- #region id="dRhOOl0zKUoe" -->
## 2.2 user data
<!-- #endregion -->

```python id="LMOjhJHfKUoe" colab={"base_uri": "https://localhost:8080/", "height": 419} executionInfo={"status": "ok", "timestamp": 1621839628107, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4665e0fe-2dd6-46b5-b7e8-125ba47f1e5e"
############################### Demographic information about the users ###############################
column_names = ["user_id", "age", "gender", "occupation", "zip_code"]
user_data = pd.read_csv('ml-100k/u.user',sep='|', names=column_names)
user_data
```

```python id="VpXFUwjJKUof" colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"status": "ok", "timestamp": 1621839628783, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fed68842-70b6-4501-fe7c-9705169b7a67"
############################### A list of the occupations(the jobs types of users). ############################### 
occupation_data = pd.read_csv('ml-100k/u.occupation',sep='|',names=["occupation"])
occupation_data = occupation_data.reset_index().rename(columns={'index':'occupation_id'})

occupation_data.head()
```

<!-- #region id="ILfNo45OKUog" -->
### 2.2.1 statistical info

#### Gender
<!-- #endregion -->

```python id="kCDDdT73KUog" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1621839632206, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="59c78e88-b359-4770-8693-20bf10abd8bf"
user_data['gender'].value_counts().plot(kind='pie', autopct='%1.1f%%')
```

<!-- #region id="v1ONrH5bKUoh" -->
#### Age
<!-- #endregion -->

```python id="IwGcjtuHKUoh" colab={"base_uri": "https://localhost:8080/", "height": 483} executionInfo={"status": "ok", "timestamp": 1621839635053, "user_tz": -330, "elapsed": 2125, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f14205c1-1158-42e8-a046-21bdb5fa250e"
user_data['age'].value_counts().plot(kind='pie', autopct='%1.1f%%',figsize=(8,8))
```

<!-- #region id="mAWyXTfrKUoh" -->
#### Occupation
<!-- #endregion -->

```python id="oh_6xlNOKUoh" colab={"base_uri": "https://localhost:8080/", "height": 483} executionInfo={"status": "ok", "timestamp": 1621839635055, "user_tz": -330, "elapsed": 22, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6036fb52-dd75-4725-e511-2596f206bf69"
user_data['occupation'].value_counts().plot(kind='pie', autopct='%1.1f%%',figsize=(8,8))
```

<!-- #region id="JOYloX--KUoi" -->
### 2.2.2 location
<!-- #endregion -->

<!-- #region id="sazYlYQ1KUoi" -->
We wish to find the geographical distribution of the users and to show them on a map. The only information we are given about this is the zip code and we find out there are only two countries involving these zip codes: America and Canada. And below we find out the number of users in Canada.
<!-- #endregion -->

```python id="gUS41442KUoj" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1621839638701, "user_tz": -330, "elapsed": 646, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d267e3a6-f762-4dbb-8175-41fc6cd755c7"
canada = 0
for i in range(len(user_data)):
    if user_data.loc[i].zip_code.isdigit() == False:
        canada += 1

canada
```

<!-- #region id="hEhOhQt4KUoj" -->
Geopy is used to find the exact coordinate corresponding to a zip code. And this information is stored in a dictionary.
<!-- #endregion -->

```python id="M-Ut9goRLfJc" executionInfo={"status": "ok", "timestamp": 1621839717390, "user_tz": -330, "elapsed": 727, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
geolocator.geocode?
```

```python id="A8wqRkIGKUok" executionInfo={"status": "ok", "timestamp": 1621840189240, "user_tz": -330, "elapsed": 389601, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
code_table = {x: (0, 0) for x in user_data.zip_code.unique().tolist()}

from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="liu")

for zip_code in code_table:
    if code_table[zip_code] != (0, 0):
        continue
    
    query_code = zip_code
    
    if query_code.isdigit() == False:
        continue

    # location = location = geolocator.geocode(query_code, country_codes = ['US'], timeout = 10)
    location = location = geolocator.geocode(query_code, timeout = 10)
    if not location:
        continue
    code_table[zip_code] = (location.latitude, location.longitude)
```

```python id="JEUbsrqkKUok" executionInfo={"status": "ok", "timestamp": 1621840199016, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
user_data['coordinate'] = user_data.apply(lambda row: code_table[row['zip_code']], axis = 1)
```

```python id="at7DBYb-KUok" colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"status": "ok", "timestamp": 1621840200903, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2c070781-6358-4324-dc60-38eeefe0d38f"
user_data.head()
```

```python id="KNrQ-8GCKUol" colab={"base_uri": "https://localhost:8080/", "height": 763, "output_embedded_package_id": "1APz_GAH2XuaEsKyAcfYYY9MFfe2PNGhg"} executionInfo={"status": "ok", "timestamp": 1621840818056, "user_tz": -330, "elapsed": 5180, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="084203f4-3bef-403a-fb01-8a00e3cbd022"
map2 = folium.Map(location=[38.9, -77.05], zoom_start=11)

from folium.plugins import MarkerCluster
marker_cluster = MarkerCluster().add_to(map2)


for i in range(0, len(user_data)):
    if user_data.loc[i].coordinate == (0, 0) or user_data.loc[i].zip_code.isdigit() == False:
        continue
    folium.Marker(user_data.iloc[i].coordinate, popup=str(user_data.loc[i].user_id), icon=folium.Icon(color='darkblue', icon_color='white', icon='male', angle=0, prefix='fa')).add_to(marker_cluster)

map2
```

<!-- #region id="q_X4gb65KUom" -->
## 2.3 Relational Info between Users and Movies
<!-- #endregion -->

```python id="aITMY-yfKUom" colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"status": "ok", "timestamp": 1621840834161, "user_tz": -330, "elapsed": 573, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1124d0c2-9816-4869-c378-db63b990b204"
############################### Create user_item_matrix ############################### 
data= pd.read_csv('ml-100k/u.data',sep='\t', names=["user_id", "item_id", "rating", "timestamp"])
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
data.head()
```

<!-- #region id="9LeR1xKeKUom" -->
Here we check for the duplicated items to see whether both of the two duplicated items are rated by users in this relational table.
<!-- #endregion -->

```python id="LUm8FV45KUon" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1621840836534, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="23168242-f1f7-4dfa-a476-916946619335"
data.merge(duplicated.item_id, on = 'item_id').item_id.nunique()
```

<!-- #region id="KyaB1VJkKUon" -->
Because all the 36 items are reviewed by some users, so we conclude that the duplicated items have been both rated by users. Therefore, to remove the duplicates, we need to select one of the duplicated items as the main movie and direct all the ratings towards the other movie to this main one.
<!-- #endregion -->

```python id="7Mbzxe3yKUoo" colab={"base_uri": "https://localhost:8080/", "height": 419} executionInfo={"status": "ok", "timestamp": 1621840838439, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cd039b2f-ee0f-4f65-972d-d9501b96e60a"
data.merge(duplicated.item_id, on = 'item_id')
```

<!-- #region id="m8xTYsYlKUop" -->
Below we find the item id to be replaced and the item id that is going to be used.
<!-- #endregion -->

```python id="08o-yxZTKUop" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1621840839351, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="591af0ee-827d-4785-a271-6f0aa5108c67"
duplicated_items = duplicated[['item_id', 'movie_title']].groupby('movie_title').apply(lambda x: list(x.item_id))
# remove_pattern = pd.DataFrame(duplicated_items.tolist(), index = duplicated_items.index, columns = ['item_id1', 'item_id2'])
to_replace = {}
for i, j in duplicated_items.values.tolist():
    to_replace[i] = j
to_replace
```

<!-- #region id="QdEx-e-KKUop" -->
Replace the duplicated item id with the replacement pattern shown above
<!-- #endregion -->

```python id="fPoP99FXKUoq" executionInfo={"status": "ok", "timestamp": 1621840842634, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
data.replace({'item_id': to_replace}, inplace = True)
```

```python id="AqDjgK6xKUoq" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1621840842635, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="db4f47be-6965-456e-a7b1-9fbca8972d94"
data.item_id.nunique()
```

```python id="plESw0ihKUoq" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1621840843029, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a2c548b3-a340-4bd8-a9c6-6e5ee1ea1860"
movies_data.item_id.nunique()
```

<!-- #region id="J6AF2XfiKUos" -->
We see that the 18 duplicated items are correctly replaced by its counterpart.
<!-- #endregion -->

```python id="HmrXlufSKUot" colab={"base_uri": "https://localhost:8080/", "height": 796} executionInfo={"status": "ok", "timestamp": 1621840844893, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="17ec3f21-ede9-4883-d53a-f54b4fe2624a"
data_merged = pd.merge(data,user_data,on='user_id',how='left')
data_merged = pd.merge(data_merged,movies_data,on='item_id',how='left')
data_merged
```

<!-- #region id="tMDP5MZbKUou" -->
We find that users only rated all these movies in 1997 and 1998.
<!-- #endregion -->

```python id="eARArspUKUou"
data_merged.timestamp.dt.year.unique()
```

<!-- #region id="wQSWuHOCKUou" -->
#### top5 most rated movies for each year
Below we find the top 5 movies that are rated the most in each year and in total respectively.
<!-- #endregion -->

```python id="1fd_wkGbKUou" executionInfo={"status": "ok", "timestamp": 1621840932144, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
data_1997 = data_merged[data_merged['timestamp'].dt.year == 1997]
data_1998 = data_merged[data_merged['timestamp'].dt.year == 1998]
```

```python id="Tv9xRIyuKUov" colab={"base_uri": "https://localhost:8080/", "height": 235} executionInfo={"status": "ok", "timestamp": 1621840932938, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ec3bf86e-ee3d-4125-b8be-c0c49c6603d5"
data_1997.groupby('movie_title').count()[['item_id']].nlargest(5, columns = 'item_id').rename(columns={"item_id": 'count'})
```

```python id="VPSkceN1KUov" colab={"base_uri": "https://localhost:8080/", "height": 235} executionInfo={"status": "ok", "timestamp": 1621840933495, "user_tz": -330, "elapsed": 26, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7602ecac-1921-4957-e5a1-1c77d340b326"
data_1998.groupby('movie_title').count()[['item_id']].nlargest(5, columns = 'item_id').rename(columns={"item_id": 'count'})
```

```python id="2mPCT6_wKUov" colab={"base_uri": "https://localhost:8080/", "height": 235} executionInfo={"status": "ok", "timestamp": 1621840933497, "user_tz": -330, "elapsed": 22, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9dff38fa-730f-4148-8551-893e7cbb4ecf"
data_merged.groupby('movie_title').count()[['item_id']].nlargest(5, columns = 'item_id').rename(columns={"item_id": 'count'})
```

```python id="bXlDsBKrKUow" colab={"base_uri": "https://localhost:8080/", "height": 506} executionInfo={"status": "ok", "timestamp": 1621840937450, "user_tz": -330, "elapsed": 1131, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="02b070b9-fd6c-488a-c575-2586cd28fdd7"
## the most rated movie genre every year
Popular = {}
Popular[1997] = data_1997[genre_cls].sum()/len(data_1997)
Popular[1998] = data_1998[genre_cls].sum()/len(data_1998)
Popular_year = pd.DataFrame(Popular) 
axes = Popular_year.plot.bar(rot=45,figsize=(15,7))
axes.set_title('genres percentage over year')
```

<!-- #region id="FstELIdWKUow" -->
# 3. Graph analysis
<!-- #endregion -->

<!-- #region id="B29lCIgeKUow" -->
## 3.1 user graph
<!-- #endregion -->

```python id="6HdnIx6bKUox" executionInfo={"status": "ok", "timestamp": 1621840937454, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def plt_graph(adjacency,data,title):
    graph_user = nx.from_numpy_matrix(adjacency)
    print('The number of connected components is {}'.format(nx.number_connected_components(graph_user)))
    coords = nx.spring_layout(graph_user,k=0.03)  # Force-directed layout.
    fig=plt.figure(figsize=(15, 10))
    labels = data.iloc[np.sort(nx.nodes(graph_user))]
    im=nx.draw_networkx_nodes(graph_user, coords, node_size=40,node_color=labels, cmap='tab20b',vmin=min(data), vmax=max(data))
    nx.draw_networkx_edges(graph_user, coords, alpha=0.1, width=0.7)
    plt.title(title)
    plt.colorbar(im)
    return graph_user
```

```python id="NAo1KHlsKUox" executionInfo={"status": "ok", "timestamp": 1621840938514, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Initialize the adjacency matrix
n_users = len(user_data)
adjacency_user = np.zeros((n_users, n_users), dtype=float)
user_features1 =user_data.copy()[['user_id', 'age', 'gender', 'occupation', 'zip_code']]
user_features1['age_normal'] = user_features1['age']/max(user_features1['age'])
user_features1= pd.merge(user_features1,occupation_data,on='occupation',how='left')
user_features1['gender_id'] = user_features1['gender'].replace(['M','F'],[1,0])
```

```python id="1SWXIL51KUoy" executionInfo={"status": "ok", "timestamp": 1621840939160, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
user_features2=user_features1[['user_id','age','gender','occupation_id']].copy()
user_features2['avg_rating'] = data_merged[['user_id','item_id','rating']].groupby('user_id').mean()['rating'].values
user_features2['movie'] = data_merged[['user_id','item_id','rating']].groupby('user_id')['item_id'].apply(set).values
```

```python id="Xm58WR_LKUoy" colab={"base_uri": "https://localhost:8080/", "height": 419} executionInfo={"status": "ok", "timestamp": 1621840939611, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1906cb44-0cc4-4e4e-a415-53eea2d721ab"
user_features1
```

```python id="F3gDvAm7KUoz" colab={"base_uri": "https://localhost:8080/", "height": 419} executionInfo={"status": "ok", "timestamp": 1621840940445, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e5b44efb-0e4c-46d3-e96e-06715662f079"
user_features2
```

<!-- #region id="W-AH1JlpKUo0" -->
### Metric 1: measure similarity between users by their age, gender, occupation and residence.
<!-- #endregion -->

```python id="JP-PidJUKUo0" executionInfo={"status": "ok", "timestamp": 1621840943590, "user_tz": -330, "elapsed": 809, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def similarity(row,data):
    sim = pd.DataFrame(np.cos(row['age_normal']-data['age_normal']))
    sim['gender'] = (row['gender']==data['gender'])
    sim['occupation'] = (row['occupation']==data['occupation'])
    sim['zip_code'] = (row['zip_code'] == data['zip_code'])
    return sim
```

```python id="M7E0kTF3KUo0" executionInfo={"status": "ok", "timestamp": 1621840949052, "user_tz": -330, "elapsed": 5475, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
for i in range(n_users):
    adjacency_user[i,:] = similarity(user_features1.loc[i,:],user_features1).mean(axis=1)
```

<!-- #region id="FZWdXXIlKUo1" -->
#### Adjacency Matrix of users
<!-- #endregion -->

```python id="RhKpWDLQKUo1" colab={"base_uri": "https://localhost:8080/", "height": 524} executionInfo={"status": "ok", "timestamp": 1621840949053, "user_tz": -330, "elapsed": 34, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f8d9342f-3e9e-4f88-e353-bcd54ef80b48"
mask = adjacency_user<=0.5
adjacency = adjacency_user.copy()
adjacency[mask]=0

plt.figure(figsize=(8,8))
plt.spy(adjacency,markersize=0.1)
plt.title('Adjacency matrix')
```

<!-- #region id="0KXSZqeOKUo1" -->
#### User Graph 1 with colors representing gender

No specific pattern is identified from the distribution of the colours of the nodes.
<!-- #endregion -->

```python id="h44DK7c1KUo2" colab={"base_uri": "https://localhost:8080/", "height": 630} executionInfo={"status": "ok", "timestamp": 1621840955699, "user_tz": -330, "elapsed": 6665, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8723552f-0e0f-470b-a21c-bbead7fe103d"
graph_user = nx.from_numpy_matrix(adjacency)
G = graph_user
print('The number of connected components is {}'.format(nx.number_connected_components(G)))
coords = nx.spring_layout(G,k=0.03)  # Force-directed layout.
fig=plt.figure(figsize=(15, 10))
labels = user_features1['gender_id'].iloc[np.sort(nx.nodes(G))]
im=nx.draw_networkx_nodes(G, coords, node_size=40,node_color=labels, cmap='plasma',vmin=0, vmax=1)
nx.draw_networkx_edges(graph_user, coords, alpha=0.1, width=0.7)
plt.title('User graph from personal information with threshold 0.5')
plt.colorbar(im)
```

<!-- #region id="15WUy2u7KUo3" -->
#### User Graph 1 with colors representing Occupation

We do observe that there is nice and clear pattern here, as the nodes of the same colour falls in one cluster, meaning that people of the same occupation do share lots of similarities with repsect to movies of interest.
<!-- #endregion -->

```python id="STBAdhbOKUo3" colab={"base_uri": "https://localhost:8080/", "height": 630} executionInfo={"status": "ok", "timestamp": 1621840964234, "user_tz": -330, "elapsed": 8544, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ff52d110-df86-4177-8cb4-1c707adc182d"
graph_user = nx.from_numpy_matrix(adjacency)
G = graph_user
print('The number of connected components is {}'.format(nx.number_connected_components(G)))
coords = nx.spring_layout(G,k=0.03)  # Force-directed layout.
fig=plt.figure(figsize=(15, 10))
labels = user_features1['occupation_id'].iloc[np.sort(nx.nodes(G))]
im=nx.draw_networkx_nodes(G, coords, node_size=40,node_color=labels, cmap='tab20',vmin=0, vmax=20)
nx.draw_networkx_edges(graph_user, coords, alpha=0.1, width=0.7)
plt.title('User graph from personal information with threshold 0.5')
plt.colorbar(im)
```

<!-- #region id="I8af3f1bKUo3" -->
#### User Graph 1 with colors representing Age
No specific pattern is identified from the distribution of the colours of the nodes.
<!-- #endregion -->

```python id="zcNcIpD4KUo4" colab={"base_uri": "https://localhost:8080/", "height": 625} executionInfo={"status": "ok", "timestamp": 1621840972403, "user_tz": -330, "elapsed": 8203, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c1a29b52-0d9c-4ba3-c90f-d5a724a8c3bd"
graph_user = nx.from_numpy_matrix(adjacency)
G = graph_user
print('The number of connected components is {}'.format(nx.number_connected_components(G)))
coords = nx.spring_layout(G,k=0.03)  # Force-directed layout.
fig=plt.figure(figsize=(15, 10))
labels = user_features1['age'].iloc[np.sort(nx.nodes(G))]
im=nx.draw_networkx_nodes(G, coords, node_size=40,node_color=labels, cmap='tab20',vmin=min(labels), vmax=max(labels))
nx.draw_networkx_edges(graph_user, coords, alpha=0.1, width=0.7)
plt.title('User graph from personal information with threshold 0.5')
plt.colorbar(im)
```

<!-- #region id="fzopruBYKUo5" -->
#### Giant Components in User Graph 1
<!-- #endregion -->

```python id="9eIpQwU0KUo5" colab={"base_uri": "https://localhost:8080/", "height": 613} executionInfo={"status": "ok", "timestamp": 1621841013812, "user_tz": -330, "elapsed": 9619, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6ca99b7d-2271-4302-a2ab-b58129ad796a"
G = graph_user
Gc = max([G.subgraph(c) for c in nx.connected_components(G)], key=len)
coords_Gc = nx.spring_layout(Gc,k=0.03)  # Force-directed layout.

print('The number of nodes is is {}'.format(Gc.number_of_nodes()))
labels = user_features1['occupation_id'].iloc[np.sort(nx.nodes(Gc))]
fig=plt.figure(figsize=(15, 10))
im=nx.draw_networkx_nodes(Gc, coords_Gc, node_size=10,node_color=labels, cmap='tab20b',vmin=0, vmax=20)
nx.draw_networkx_edges(Gc, coords_Gc, alpha=0.1, width=0.7)
plt.title('Giant component of the users connected by at least 0.5 similarity')
plt.colorbar(im);
```

<!-- #region id="ukLWqB08KUo6" -->
#### Below we analyse the graph from the view of spectal theory
<!-- #endregion -->

```python id="BI3imvmPKUo9" executionInfo={"status": "ok", "timestamp": 1621841017040, "user_tz": -330, "elapsed": 448, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def compute_laplacian(adjacency: np.ndarray, normalize: bool):
    """ Return:
        L (n x n ndarray): combinatorial or symmetric normalized Laplacian.
    """
    d = np.sum(adjacency, axis = 1)
    d_sqrt = np.sqrt(d)
    D = np.diag(1 / d_sqrt)
    if normalize:
        L = np.eye(adjacency.shape[0]) - (adjacency.T / d_sqrt).T / d_sqrt
    else:
        L = np.diag(d) - adjacency
    return L

def spectral_decomposition(laplacian: np.ndarray):
    """ Return:
        lamb (np.array): eigenvalues of the Laplacian
        U (np.ndarray): corresponding eigenvectors.
    """
    lamb, U = np.linalg.eigh(laplacian)
    
    return lamb, U
```

```python id="Zo-lHM41KUo9" executionInfo={"status": "ok", "timestamp": 1621841090934, "user_tz": -330, "elapsed": 414, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
laplacian_norm = compute_laplacian(adjacency, normalize=True)
lamb_norm, U_norm = spectral_decomposition(laplacian_norm)
```

```python id="YEA_yPR2KUo9" colab={"base_uri": "https://localhost:8080/", "height": 352} executionInfo={"status": "ok", "timestamp": 1621841094149, "user_tz": -330, "elapsed": 1311, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3140c232-5111-40c5-b748-a06b10a6712b"
plt.figure(figsize=(15,5))
plt.subplot(121)
plt.plot(lamb_norm)
plt.xlabel('Index')
plt.ylabel('Eigenvalue')
plt.title('Eigenvalues $L_{norm}$')

plt.subplot(122)
first_k = 70
plt.scatter(range(first_k), lamb_norm[:first_k])
plt.xlabel('Index')
plt.ylabel('Eigenvalue')
plt.title('First 70 Eigenvalues $L_{norm}$')

plt.show()
```

<!-- #region id="1FRW8hJOKUo-" -->
In order to observe the properties of the eigenvalues calculated, we zoom in the eigenvalues and we observe that there is a significant gap of eighvalues as shown in the plot, which correspond to the fact that there are about 40 clear clusters in the graph. As we know that if the data has exactly k clear clusters, there will be a gap in the Laplacian spectrum after the k-th eigenvalue. Here the clusters are generally defined by the occupation of the users. 
<!-- #endregion -->

<!-- #region id="Lgm3gOkAKUo-" -->
### Metric 2: meaure similarities between users by how many common movies they have rated

The more common movies two users have rated, the more similar they are.
<!-- #endregion -->

```python id="T55jnRHiKUo-" executionInfo={"status": "ok", "timestamp": 1621841099447, "user_tz": -330, "elapsed": 888, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Calulate the number of common movies they have rated between two users
def common_movie(i,j,data):
    left = data[data['user_id']==i+1]['movie'].values.tolist()[0]
    right = data[data['user_id']==j+1]['movie'].values.tolist()[0]
    common = left.intersection(right)
    return len(common)
```

```python id="UZhbQHvgKUo_" executionInfo={"status": "ok", "timestamp": 1621841875729, "user_tz": -330, "elapsed": 730, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
adjacency_user2 = np.zeros((n_users, n_users), dtype=float)
for i in range(n_users):
    for j in range(n_users):
        if j<i:
            adjacency_user2[i,j] = adjacency_user2[j,i]
        else:
            adjacency_user2[i,j] = common_movie(i,j,user_features2)
            
np.save('adjacency_user2.npy', adjacency_user2)
```

<!-- #region id="tMfsr9FGKUo_" -->
Histogram of the Median of common movies
<!-- #endregion -->

```python id="m1w3lr9OKUo_" colab={"base_uri": "https://localhost:8080/", "height": 312} executionInfo={"status": "ok", "timestamp": 1621841887923, "user_tz": -330, "elapsed": 949, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="158bce20-52e9-43f0-f94f-9539db429744"
adjacency_user2 = np.load('adjacency_user2.npy')
median = []
for i in range(n_users):
    median.append(np.median(adjacency_user2[i,:]))

plt.hist(median, density=True)
plt.xlabel('number of common movies')
plt.ylabel('Frequency')
plt.title('histogram of number of common movies')
```

<!-- #region id="Fm8YhkK7KUpA" -->
The adjacency matrix of **Metric 2**
<!-- #endregion -->

```python id="sK9Q7NGgKUpA" colab={"base_uri": "https://localhost:8080/", "height": 524} executionInfo={"status": "ok", "timestamp": 1621841895261, "user_tz": -330, "elapsed": 2936, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3aedcd8e-cce8-42b5-b729-46f35d0e0632"
mask2 = adjacency_user2<20
adjacency = adjacency_user2.copy()
adjacency[mask2]=0

# Normalize 
adjacency_normalized = np.divide(adjacency,adjacency.max());
adjacency_normalized = adjacency
plt.figure(figsize=(8,8))
plt.spy(adjacency_normalized,markersize=0.1)
plt.title('Adjacency matrix')
```

<!-- #region id="iR_yFaaoKUpA" -->
#### User Graph 2 with colors representing Occupation
<!-- #endregion -->

```python id="RPHIv0m6KUpB" colab={"base_uri": "https://localhost:8080/", "height": 630} executionInfo={"status": "ok", "timestamp": 1621841914149, "user_tz": -330, "elapsed": 15799, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b61595bc-1919-415d-a3fd-7d1944759ec8"
graph_user2 = nx.from_numpy_matrix(adjacency_normalized)
G = graph_user2
print('The number of connected components is {}'.format(nx.number_connected_components(G)))
coords = nx.spring_layout(G,k=0.03)  # Force-directed layout.
fig=plt.figure(figsize=(15, 10))
labels = user_features1['occupation_id'].iloc[np.sort(nx.nodes(G))]
im=nx.draw_networkx_nodes(G, coords, node_size=40,node_color=labels, cmap='tab20',vmin=0, vmax=20)
nx.draw_networkx_edges(graph_user, coords, alpha=0.1, width=0.7)
plt.title('User graph from common movie with threshold 20')
plt.colorbar(im)
```

<!-- #region id="yfY26yfMKUpB" -->
#### User Graph 2 with colors representing Average Rating
<!-- #endregion -->

```python id="iyOFfctrKUpB" colab={"base_uri": "https://localhost:8080/", "height": 630} executionInfo={"status": "ok", "timestamp": 1621841926280, "user_tz": -330, "elapsed": 12146, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f19ae133-6cf4-4af0-b69b-0f1bc8940586"
graph_user2 = nx.from_numpy_matrix(adjacency_normalized)
G = graph_user2
print('The number of connected components is {}'.format(nx.number_connected_components(G)))
coords = nx.spring_layout(G,k=0.03)  # Force-directed layout.
fig=plt.figure(figsize=(15, 10))
labels = user_features2['avg_rating'].iloc[np.sort(nx.nodes(G))]
im=nx.draw_networkx_nodes(G, coords, node_size=40,node_color=labels, cmap='tab20c',vmin=0, vmax=5)
nx.draw_networkx_edges(graph_user, coords, alpha=0.1, width=0.7)
plt.title('User graph from common movie with threshold 20')
plt.colorbar(im)
```

<!-- #region id="pHgsMXRoKUpC" -->
#### User Graph 2 with colors representing Age of user
<!-- #endregion -->

```python id="JX-MyH0rKUpC" colab={"base_uri": "https://localhost:8080/", "height": 625} executionInfo={"status": "ok", "timestamp": 1621841939389, "user_tz": -330, "elapsed": 13139, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8d20bfba-bac1-43fa-e5c3-da3243e6e160"
graph_user2 = nx.from_numpy_matrix(adjacency_normalized)
G = graph_user2
print('The number of connected components is {}'.format(nx.number_connected_components(G)))
coords = nx.spring_layout(G,k=0.03)  # Force-directed layout.
fig=plt.figure(figsize=(15, 10))
labels = user_features2['age'].iloc[np.sort(nx.nodes(G))]
im=nx.draw_networkx_nodes(G, coords, node_size=40,node_color=labels, cmap='Blues',vmin=min(labels), vmax=max(labels))
nx.draw_networkx_edges(graph_user, coords, alpha=0.1, width=0.7)
plt.title('User graph from common movie with threshold 20')
plt.colorbar(im)
```

<!-- #region id="DvyAcLmWKUpD" -->
#### Giant component of the User Graph 2 connected by at least 20 common movies
<!-- #endregion -->

```python id="y_RaOGMGKUpD" colab={"base_uri": "https://localhost:8080/", "height": 613} executionInfo={"status": "ok", "timestamp": 1621841954612, "user_tz": -330, "elapsed": 15251, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4019b45b-30eb-4d73-e3e6-647c25fcd70e"
G = graph_user2
Gc = max([G.subgraph(c) for c in nx.connected_components(G)], key=len)
coords_Gc = nx.spring_layout(Gc,k=0.03)  # Force-directed layout.
print('The number of nodes is is {}'.format(Gc.number_of_nodes()))
labels = user_features2['avg_rating'].iloc[np.sort(nx.nodes(Gc))]
fig=plt.figure(figsize=(15, 10))
im=nx.draw_networkx_nodes(Gc, coords_Gc, node_size=10,node_color=labels, cmap='tab20',vmin=0, vmax=20)
nx.draw_networkx_edges(Gc, coords_Gc, alpha=0.1, width=0.7)
plt.title('Giant component of the users connected by at least 20 common movies')
plt.colorbar(im);
```

```python id="spfj6VIMKUpE" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1621841954615, "user_tz": -330, "elapsed": 45, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="23b175d6-b944-4aab-db0b-32f759a82e76"
laplacian_norm = compute_laplacian(adjacency_normalized, normalize=True)
lamb_norm, U_norm = spectral_decomposition(laplacian_norm)
```

```python id="kL7cvZmFKUpE" colab={"base_uri": "https://localhost:8080/", "height": 352} executionInfo={"status": "ok", "timestamp": 1621841955691, "user_tz": -330, "elapsed": 1107, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="779bded5-d92c-4e4b-916f-5b40342c78a3"
plt.figure(figsize=(15,5))
plt.subplot(121)
plt.plot(lamb_norm)
plt.xlabel('Index')
plt.ylabel('Eigenvalue')
plt.title('Eigenvalues $L_{norm}$')

plt.subplot(122)
first_k = 70
plt.scatter(range(first_k), lamb_norm[:first_k])
plt.xlabel('Index')
plt.ylabel('Eigenvalue')
plt.title('First 70 Eigenvalues $L_{norm}$')

plt.show()
```

<!-- #region id="0PtZ6xqdKUpF" -->
In order to observe the properties of the eigenvalues calculated, we zoom in the eigenvalues and we observe that there is a significant gap of eighvalues as shown in the plot, which correspond to the fact that there are about 35 clear clusters in the graph. As we know that if the data has exactly k clear clusters, there will be a gap in the Laplacian spectrum after the k-th eigenvalue. However, the pattern here in this user graph is not clear as most of them are not really clusters by outliers. There is only one giant component that dominates the graph.
<!-- #endregion -->

<!-- #region id="E7_hd_pEKUpK" -->
## 3.2 movie graph
<!-- #endregion -->

<!-- #region id="uPtgu8v4KUpK" -->
### Metric 1: similarity between moives measured by genres
<!-- #endregion -->

```python id="kql2Nlh2KUpL" colab={"base_uri": "https://localhost:8080/", "height": 660} executionInfo={"status": "ok", "timestamp": 1621842063901, "user_tz": -330, "elapsed": 997, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="61c6adcd-2c59-428c-c999-5b9e736c7a2b"
movie_features1 = movies_data[['item_id']+genre_cls]
movie_features1.loc[:, 'year_label'] = movies_data['year_label']
movie_features1.reset_index(drop = True, inplace = True)
movie_features1
```

```python id="wa2RNAnNKUpL" executionInfo={"status": "ok", "timestamp": 1621842067247, "user_tz": -330, "elapsed": 564, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Initialize the adjacency matrix
n_movies = len(movie_features1)
adjacency_movie = np.zeros((n_movies, n_movies), dtype=float)
```

```python id="uZqteCW1KUpM" executionInfo={"status": "ok", "timestamp": 1621842073824, "user_tz": -330, "elapsed": 5936, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
for i in range(n_movies):
    adjacency_movie[i,:] = np.logical_and(movie_features1.loc[i,:][genre_cls], movie_features1[genre_cls])\
    .sum(axis=1)
```

```python id="YmPAX-KwKUpM" colab={"base_uri": "https://localhost:8080/", "height": 524} executionInfo={"status": "ok", "timestamp": 1621842073826, "user_tz": -330, "elapsed": 38, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="db9d774c-ba07-4f76-df92-c489d6e4a27a"
# Normalize 
mask_movie = adjacency_movie<2
adjacency = adjacency_movie.copy()
adjacency[mask_movie] = 0 
adjacency_movie_nor =np.divide(adjacency,adjacency.max())
plt.figure(figsize=(8,8))
plt.spy(adjacency_movie_nor,markersize=0.1)
plt.title('Adjacency matrix')
```
