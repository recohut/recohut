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

```python id="1uICbB4nDexm" executionInfo={"status": "ok", "timestamp": 1628519194994, "user_tz": -330, "elapsed": 736, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-mll"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python colab={"base_uri": "https://localhost:8080/"} id="EFnuEM16DqQd" executionInfo={"status": "ok", "timestamp": 1628519200205, "user_tz": -330, "elapsed": 4607, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="73e82b0f-dab9-4119-becd-46cf3f356289"
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

```python id="wQATOAHpdmAi" executionInfo={"status": "ok", "timestamp": 1628519200207, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="XmtYiQXMT_pO" executionInfo={"status": "ok", "timestamp": 1628505024740, "user_tz": -330, "elapsed": 888, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="99993573-4ec6-46c7-b61c-339d21fbb10a"
ratings = pd.read_parquet('./data/silver/ratings.parquet.gzip')
ratings.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 326} id="vOMu7TycUF-z" executionInfo={"status": "ok", "timestamp": 1628505024742, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4a494f17-debb-4d80-b518-e132a263005d"
movies = pd.read_parquet('./data/silver/movies.parquet.gzip')
movies.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 235} id="myi66wQxdHxy" executionInfo={"status": "ok", "timestamp": 1628505440937, "user_tz": -330, "elapsed": 531, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2ff0e921-ab89-4f8d-ed0f-11e286dd9662"
movie_ratings = pd.merge(ratings, movies, on='movieId')
ratings = pd.DataFrame(movie_ratings.groupby('title')['rating'].mean())
ratings['No_of_ratings'] = pd.DataFrame(movie_ratings.groupby('title')['rating'].count())
ratings.sort_values(by=['No_of_ratings'], ascending=False).head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 235} id="b8yYWLcXfHpq" executionInfo={"status": "ok", "timestamp": 1628505452834, "user_tz": -330, "elapsed": 931, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e94b6d9f-739b-458b-a635-61de7875a221"
user_movie_matrix = movie_ratings.pivot_table(index='userId', columns='title', values='rating')
user_movie_matrix.iloc[:5,:5]
```

```python colab={"base_uri": "https://localhost:8080/"} id="5TlpVWE6dkdL" executionInfo={"status": "ok", "timestamp": 1628505461509, "user_tz": -330, "elapsed": 901, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="deae2f8a-b7ae-44bb-9cf5-d71e8f62dfcd"
matrix_movie_ratings = user_movie_matrix['Matrix, The']
matrix_movie_ratings.dropna(inplace=True)   # Remove users that have not given any rating
matrix_movie_ratings.head()
```

```python id="gW-qzCY8dt9S"
# Retrieve all movies related to matrix
movies_like_the_matrix = user_movie_matrix.corrwith(matrix_movie_ratings)

# Create a DataFrame that contains movies and correlation
corr_the_matrix = pd.DataFrame(movies_like_the_matrix, columns=['Correlation'])

# Drop all NA values
corr_the_matrix.dropna(inplace=True)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 235} id="teo1QHiPebEA" executionInfo={"status": "ok", "timestamp": 1628505246505, "user_tz": -330, "elapsed": 19, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8c3b03a0-1e34-46b6-e4e7-2b1a978cb44a"
corr_the_matrix.head()
```

```python id="H7VvRjWbegIA"
corr_the_matrix = corr_the_matrix.join(ratings['No_of_ratings'])

# In order to increase confidence interval, only take movies with more than 20 ratings
corr_the_matrix = corr_the_matrix[corr_the_matrix['No_of_ratings'] > 20]

# R2 is the percentage of variation explained by the relationship between two variables
# In this case R2 shows the percentage of the relationship between two movies.
# Take the square of R which is the correlation
corr_the_matrix['R_Square'] = np.square(corr_the_matrix['Correlation']) 

# Show as Percantage
corr_the_matrix['Similarity'] = np.multiply(corr_the_matrix['R_Square'], 100)
corr_the_matrix['Similarity'] = np.multiply(corr_the_matrix['Similarity'], np.sign(corr_the_matrix['Correlation'])) 
similarity_df = pd.DataFrame(corr_the_matrix, columns=['Correlation', 'No_of_ratings', 'R_Square', 'Similarity'])
similarity_df['Similarity']= similarity_df['Similarity'].map('{:,.2f}%'.format)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 700} id="qgezVAh4ezC-" executionInfo={"status": "ok", "timestamp": 1628505481394, "user_tz": -330, "elapsed": 775, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8d9481c6-b846-4d8c-a607-8fa16d0c64af"
similarity_df.sort_values('Correlation', ascending=False).head(20)
```
