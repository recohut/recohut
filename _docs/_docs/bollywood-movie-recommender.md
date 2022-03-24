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

<!-- #region id="_2m8gGTI-Sct" -->
# Bollywood Movie Recommender
> Applying TFIDF and Cosine similarity techniques on bollywood IMDB movie dataset to build a simple content-based recommender and wrapping into an app

- toc: true
- badges: true
- comments: true
- categories: [TFIDF, Movie, Flask, App, Heroku]
- author: "<a href='https://github.com/DipabaliHalder/Bollywood-Movie-Recommender'>Dipabali_Halder</a>"
- image:
<!-- #endregion -->

<!-- #region id="a-Kjb8NC9lRo" -->
## Setup
<!-- #endregion -->

```python id="5Gd90Zw_4nJS"
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pickle

import warnings
warnings.filterwarnings('ignore')
```

<!-- #region id="WM5ApMZ59XW-" -->
## Data Loading
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="_HturH-44nDt" outputId="2fd9930a-9bc9-4048-b2be-51b228d33932"
df = pd.read_csv('https://github.com/sparsh-ai/Bollywood-Movie-Recommender/raw/main/model/imdb_2000bollywood_movies.csv')
df.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 649} id="kjbCIvN040Am" outputId="34491804-e204-4a9c-f399-b4853b1395dc"
df.head()
```

<!-- #region id="3tPr101a9UxE" -->
## Data Preprocessing
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="qjDm8Fo3448J" outputId="99cc83d1-857f-4b4d-886f-c4dfa38395fd"
# Combine action names to make it a string
df['Actor'] = df['Actor1']+","+df['Actor2']+","+df['Actor3']+","+df['Actor4']

# select columns of interest
movies = df[['img','movie_name','year','genre','Overview','Director','Actor']]

# remove "-" sign from movie year column
movies['year'] = movies['year'].apply(lambda x:x.replace('-',''))

# convert overview, actor and director column string values into a list
movies['Overview'] = movies['Overview'].apply(lambda x:str(x).replace(" ","  ").split())
movies['Actor'] = movies['Actor'].apply(lambda x:str(x).replace(" ","").split())
movies['Director'] = movies['Director'].apply(lambda x:x.replace(" ","").split())

# remove "," in genres
movies['genre'] = movies['genre'].apply(lambda x:x.replace(",","").split())

# combine genres, overview, actor and director columns into a single list
movies['tags'] = movies['genre']+movies['Overview']+movies['Actor']+movies['Director']

# dropping the extra columns
new = movies.drop(columns=['Overview','genre','Actor','Director','img'])

# convert list into a text string
new['tags']=new['tags'].apply(lambda x:" ".join(x))

# checking the data
new.head()
```

<!-- #region id="4CThtGmb9R-J" -->
## Memory-based Model
<!-- #endregion -->

<!-- #region id="WPpR3ssd9KoM" -->
### Count Vectorizer
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="EwEd1uFk8Q3i" outputId="be3bac91-6681-453b-a685-32c55aaeb321"
cv=CountVectorizer(max_features=10000,stop_words='english')
vector = cv.fit_transform(new['tags']).toarray()
vector[:10,:10]
```

<!-- #region id="8WaXx9989M2U" -->
### Cosine Similarity Matrix
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="OStOWVYS8ZuG" outputId="0899909a-93ca-46e6-c1b1-13302ebcd489"
similarity=cosine_similarity(vector)
similarity[0]
```

<!-- #region id="bLQoUhK18_Ou" -->
## Inference
<!-- #endregion -->

```python id="giivTxoi8fNX"
def recommend(movie):
  index = new[new['movie_name']== movie].index[0]        
  distances = sorted(list(enumerate(similarity[index])),reverse=True,key= lambda x: x[1])
  for i in distances[1:6]:
    movie=new.iloc[i[0]].movie_name
    print(movie)
```

```python colab={"base_uri": "https://localhost:8080/"} id="qjsGsGER8r_U" outputId="b7e39020-dd58-474b-87ae-f089b8c45cba"
recommend('3 Idiots')
```

```python colab={"base_uri": "https://localhost:8080/"} id="oRKYqAnH8r7X" outputId="79f857c2-b388-44e7-d815-764ed5c331ac"
recommend('Commando')
```

<!-- #region id="8PYHQC7Z80vw" -->
## Save the Artifacts
<!-- #endregion -->

```python id="wvTT6GSX8r43"
pickle.dump(new,open('finalmovie_list.pkl','wb'))
pickle.dump(similarity,open('finalsimilarity.pkl','wb'))
```

<!-- #region id="sr1eCHdI8r2V" -->
## API
<!-- #endregion -->

```python id="MZZ5vaQm4LvI"
import streamlit as st
import pickle
import numpy as np
import requests
import pandas as pd

def fetch_poster(movie):
    url = "https://www.omdbapi.com/?apikey=21dcff44&t={}".format(movie)
    data = requests.get(url)
    data = data.json()
    try:
        return data['Poster']
    except:
        return ('')

def recommend(movie):
    index = movies[movies['movie_name'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names=[]
    recommended_movie_posters=[]
    for i in distances[1:6]:
        m=movies.iloc[i[0]].movie_name
        recommended_movie_names.append(m)
        recommended_movie_posters.append(fetch_poster(m))
    return recommended_movie_names,recommended_movie_posters

st.header("**Movie Recommender System**")

movies=pd.read_pickle('finalmovie_list.pkl')
similarity= pickle.load(open('finalsimilarity.pkl','rb'))

movie_list = np.sort(movies['movie_name'].unique())
movie_list=np.insert(movie_list,0,'')
selected_movie = st.selectbox("Choose a Movie: ",movie_list)

if st.button("Recommend >>"):
    if (selected_movie ==''):
        st.subheader("*No movie selected!!! Please select a movie.*")
    else:
        recommended_movie_names,recommended_movie_posters = recommend(selected_movie)
        col1, col2, col3, col4, col5 = st.beta_columns(5)
        with col1:
            st.text(recommended_movie_names[0])
            if(recommended_movie_posters[0]!=''):
                st.image(recommended_movie_posters[0])
            else:
                st.image("https://m.media-amazon.com/images/S/sash/4FyxwxECzL-U1J8.png")
        with col2:
            st.text(recommended_movie_names[1])
            if (recommended_movie_posters[1]!= ''):
                st.image(recommended_movie_posters[1])
            else:
                st.image("https://m.media-amazon.com/images/S/sash/4FyxwxECzL-U1J8.png")
        with col3:
            st.text(recommended_movie_names[2])
            if (recommended_movie_posters[2] != ''):
                st.image(recommended_movie_posters[2])
            else:
                st.image("https://m.media-amazon.com/images/S/sash/4FyxwxECzL-U1J8.png")
        with col4:
            st.text(recommended_movie_names[3])
            if (recommended_movie_posters[3] != ''):
                st.image(recommended_movie_posters[3])
            else:
                st.image("https://m.media-amazon.com/images/S/sash/4FyxwxECzL-U1J8.png")
        with col5:
            st.text(recommended_movie_names[4])
            if (recommended_movie_posters[4] != ''):
                st.image(recommended_movie_posters[4])
            else:
                st.image("https://m.media-amazon.com/images/S/sash/4FyxwxECzL-U1J8.png")
```

<!-- #region id="FhDfP8Vi4LrP" -->
> Note: App link - https://bollywood-movie-recommender.herokuapp.com/
<!-- #endregion -->

<!-- #region id="afIA2gX44Tly" -->
<!-- #endregion -->
