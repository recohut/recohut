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

<!-- #region id="4sQi0ThqJwGV" -->
# Simple Movie Recommender App
> Simple end-to-end content-based movie recommender app built using streamlit and TMDB data

- toc: true
- badges: true
- comments: true
- categories: [TMDB, Movie, Streamlit, App, NLP]
- author: "<a href='https://github.com/campusx-official/movie-recommender-system-tmdb-dataset'>CampusX</a>"
- image:
<!-- #endregion -->

<!-- #region id="nRXwrM6AKPwL" -->
### Tutorial video
<!-- #endregion -->

<!-- #region id="SYkc7RSPKVl9" -->
> youtube: https://youtu.be/1xtrIEwY_zY
<!-- #endregion -->

<!-- #region id="e6YjSLVBEg6O" -->
### Connect gdrive
<!-- #endregion -->

```python id="TdO_I2rnEMK1"
from google.colab import drive
drive.mount('/content/drive')
```

<!-- #region id="cpEcPONFEkCF" -->
### Download data from kaggle
<!-- #endregion -->

```python id="L53bsLwmEXuP"
!pip install -q -U kaggle
!pip install --upgrade --force-reinstall --no-deps kaggle
!mkdir ~/.kaggle
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d tmdb/tmdb-movie-metadata
```

<!-- #region id="R8dBzMtWEqHo" -->
### Unzip the data
<!-- #endregion -->

```python id="9D9aGbokEAg1"
!unzip /content/tmdb-movie-metadata.zip
```

<!-- #region id="FaAaMqcTE310" -->
### Import libraries
<!-- #endregion -->

```python id="NucNlL28HDmp"
!pip install -q streamlit
```

```python _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" id="uX5lX0DuD56s"
import os
import ast
import pickle
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```

<!-- #region id="m_iBK6ZGE5fh" -->
### Load data
<!-- #endregion -->

```python id="2V4rkSIkD563"
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
```

```python id="BpJznMdjD565"
movies.head(2)
```

```python id="WRDG8q2VD567" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1625730301753, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="95bc43f4-a0f7-4531-ce84-4c7da904282f"
movies.shape
```

```python id="Zij5tOzkD567"
credits.head()
```

<!-- #region id="g05m3J73E9Z4" -->
### Merge credits data with movies on title column
<!-- #endregion -->

```python id="EuBkAGbaD567"
movies = movies.merge(credits,on='title')
```

```python id="NubDhuh0D568"
movies.head()
```

<!-- #region id="Lo2v04gnFD8W" -->
### Select important features
<!-- #endregion -->

```python id="uIyFZXeiD569"
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
```

```python id="NJnmjvqgD56-"
movies.head()
```

<!-- #region id="ovkTdHp3FSJx" -->
### Extract name values from dictionaries of genre and keywords column
<!-- #endregion -->

```python id="7NRrEvs0D56_"
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 
```

```python id="fsMUvgALD57A"
movies.dropna(inplace=True)
```

```python id="vNe8fXhKD57B"
movies['genres'] = movies['genres'].apply(convert)
movies.head()
```

```python id="UVpVggG7D57F"
movies['keywords'] = movies['keywords'].apply(convert)
movies.head()
```

<!-- #region id="cbCBP7_kFhJO" -->
### Extract name values from dictionaries of cast column
<!-- #endregion -->

```python id="iHixXop6D57G"
def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L 
```

```python id="qrSotfyED57H"
movies['cast'] = movies['cast'].apply(convert)
movies.head()
```

```python id="GezHWBB-D57I"
movies['cast'] = movies['cast'].apply(lambda x:x[0:3])
```

<!-- #region id="Wb_PL7iEFoes" -->
### Extract name values from dictionaries of director column
<!-- #endregion -->

```python id="HcwaLSneD57J"
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 
```

```python id="i2BAlKCQD57K"
movies['crew'] = movies['crew'].apply(fetch_director)
```

<!-- #region id="wbFjYIW6FuBu" -->
### Convert overview column's string to list
<!-- #endregion -->

```python id="s7AK4-x3D57K"
movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies.sample(5)
```

<!-- #region id="YIG1OBkWF87t" -->
### Remove extra spaces
<!-- #endregion -->

```python id="pNXZ3yHaD57L"
def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1
```

```python id="6jPW1XpJD57M"
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)
```

```python id="4i0VVni3D57M"
movies.head()
```

<!-- #region id="kPlnunMGGJHl" -->
### Combine all list into single columns named ```tag```
<!-- #endregion -->

```python id="d6nE9nItD57N"
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
```

```python id="D_A7prOYD57N" colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"status": "ok", "timestamp": 1625730643277, "user_tz": -330, "elapsed": 18, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="42848ec3-71fe-4b9c-e9cd-0c8e8b72ceea"
new = movies.drop(columns=['overview','genres','keywords','cast','crew'])
new.head()
```

```python id="23olYIF3D57N" colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"status": "ok", "timestamp": 1625730643279, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f9fab8c3-e0e8-4ba7-e86d-e3ce1d213b21"
new['tags'] = new['tags'].apply(lambda x: " ".join(x))
new.head()
```

<!-- #region id="erJDaEkEGRFe" -->
### Build count vectorizer model
<!-- #endregion -->

```python id="p3mUXSB3D57O"
cv = CountVectorizer(max_features=5000,stop_words='english')    
```

```python id="xDpexf64D57O"
vector = cv.fit_transform(new['tags']).toarray()
```

```python id="ehqIIi6iD57P" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1625730675926, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="722aff80-362d-41ca-925d-af213457b4a2"
vector.shape
```

<!-- #region id="pF8NzL3hGcP1" -->
### Calculate cosine similarity
<!-- #endregion -->

```python id="BCiBCwqxD57P"
similarity = cosine_similarity(vector)
```

```python id="H9xWCrwRD57Q" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1625730707802, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4d2ec67b-43ff-4e2e-fcf5-2907449c0519"
similarity
```

```python id="a8hZW7_AD57Q" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1625730771280, "user_tz": -330, "elapsed": 977, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="18fc5fde-a7d6-4de7-baaf-32d8e97feff5"
new[new['title'] == 'The Lego Movie'].index[0]
```

<!-- #region id="ylpu28ULGvez" -->
### Create recommender
<!-- #endregion -->

```python id="LVi8IXUeD57Q"
def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)
```

```python id="iuZQzecDD57R" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1625730791314, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7ef1c779-2f9f-4a00-84ba-f73889868ff7"
recommend('Gandhi')
```

<!-- #region id="u40tmi3xD57R" -->
### Save the movie list and similarity matrix as pickle
<!-- #endregion -->

```python id="P1plVCU1D57R"
pickle.dump(new,open('movie_list.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))
```

<!-- #region id="k5zRfd5yHsDW" -->
### Streamlit app
<!-- #endregion -->

```python id="DmT_oxQCD57S" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1625731175123, "user_tz": -330, "elapsed": 489, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="299167a9-8e57-4a76-cfa0-4b126f057fe9"
%%writefile app.py
import pickle
import streamlit as st
import requests

def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        # fetch the movie poster
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names,recommended_movie_posters


st.header('Movie Recommender System')
movies = pickle.load(open('model/movie_list.pkl','rb'))
similarity = pickle.load(open('model/similarity.pkl','rb'))

movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    recommended_movie_names,recommended_movie_posters = recommend(selected_movie)
    col1, col2, col3, col4, col5 = st.beta_columns(5)
    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0])
    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1])

    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2])
    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3])
    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4])
```

<!-- #region id="Srt4GxyPIY7l" -->
### Run the app
<!-- #endregion -->

```python id="_g_1PBk3Il0u" colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"status": "ok", "timestamp": 1625731440602, "user_tz": -330, "elapsed": 4481, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="300b3ca7-8738-4e24-97bf-2b0a73529814"
from google.colab.output import eval_js
print(eval_js("google.colab.kernel.proxyPort(9999)"))
```

```python id="7_mfH8pSHxDW" colab={"base_uri": "https://localhost:8080/"} outputId="0ddd9aea-120e-47c0-ef57-45890537267d"
!nohup streamlit run app.py &
!pip install -q colab-everything
from colab_everything import ColabStreamlit
ColabStreamlit('app.py')
```

<!-- #region id="sz_JudJkJQ72" -->
<!-- #endregion -->
