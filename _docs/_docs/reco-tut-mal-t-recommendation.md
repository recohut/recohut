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

```python id="jVkAVV4pixpb" executionInfo={"status": "ok", "timestamp": 1629180828841, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-mal"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python colab={"base_uri": "https://localhost:8080/"} id="RDSfrKdHi4C8" executionInfo={"status": "ok", "timestamp": 1629180833565, "user_tz": -330, "elapsed": 4098, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="09f2bc56-3a1a-4d0b-be57-dee773e9accc"
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

```python id="22P-ZOjbi4C_"
!git status
```

```python id="9LDKaBYRi4DA"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

<!-- #region id="93HdcwgQvq1B" -->
---
<!-- #endregion -->

```python id="KNyqxDwIvrKi" executionInfo={"status": "ok", "timestamp": 1629181185769, "user_tz": -330, "elapsed": 911, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```

```python colab={"base_uri": "https://localhost:8080/", "height": 581} id="66jwtjlZxACo" executionInfo={"status": "ok", "timestamp": 1629182733811, "user_tz": -330, "elapsed": 642, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c8b5eb9d-63a6-4824-cf06-f6d5c9c9757f"
df = pd.read_csv("./data/silver/anime_data.csv", compression='gzip', index_col=[0])
df.reset_index(drop=True, inplace=True)
df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="BljPfaMWxGju" executionInfo={"status": "ok", "timestamp": 1629182736066, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="573ca32b-8c36-4bb6-8ad9-241f84c0586d"
df.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 581} id="UOWPFPm7xZ89" executionInfo={"status": "ok", "timestamp": 1629182736067, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cc2018f1-b9fb-415f-a7c2-521e15b9ab32"
# drop unnecessary columns
df.drop(['Aired','Premiered','Broadcast','Duration','Favorites','link'],axis = 1,inplace=True)

# dealing with null values
features = ['Licensors', 'Genres', 'Studios', 'Rating','Score']
for feature in features:
    df[feature] = df[feature].fillna('')

# converting the datatype so combine all the required columns to prepare cosine simarity matrix
df.Rating = df.Rating.astype(str)
df.Score = df.Score.astype(str)
df.Genres = df.Genres.astype(str)
df.Studios = df.Studios.astype(str)
df.Type = df.Type.astype(str)

def combined_features(row):
    return row['Licensors']+" "+row['Genres']+" "+row['Studios']+" "+row['Rating']+" "+row['Score']+" "+row['Type']
df["combined_features"] = df.apply(combined_features, axis=1)

df.head()
```

```python id="MKKElEsYoiO3" executionInfo={"status": "ok", "timestamp": 1629182747311, "user_tz": -330, "elapsed": 10347, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])
cosine_sim = cosine_similarity(count_matrix)
```

```python id="GGlZ1t2coiO7" executionInfo={"status": "ok", "timestamp": 1629182749120, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def get_index_from_title(a): #Getting indx of title that user has given
    return df.index[df.Title == x].values[0]

def get_title_from_index(index): # Getting the recommended titles with index
    return df[df.index == index]["Title"].values[0]
```

```python id="XznVAleJoiO8" executionInfo={"status": "ok", "timestamp": 1629182749121, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def print_anime_recommendation(Anime_user_likes, k=10):
    Anime_index = get_index_from_title(Anime_user_likes)
    similar_anime = list(enumerate(cosine_sim[Anime_index]))
    sorted_similar_anime = sorted(similar_anime, key=lambda x:x[1], reverse=True)
    i=0
    for anime in sorted_similar_anime:
        print(get_title_from_index(anime[0]))
        i+=1
        if i>k:
            break
```

```python id="A2uaXKauoiO9" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629182749123, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e6598b94-b413-4ddc-a40a-ea6b5636dc52"
anme = np.random.choice(df.Title.to_list(),1)[0]
topk = 5
print('{} similar Animes for "{}" are:'.format(topk, anme))
print_anime_recommendation(anme, 5)
```
