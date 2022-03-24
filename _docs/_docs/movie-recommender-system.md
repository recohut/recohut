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

<!-- #region id="lK5S69YCS3Uh" -->
# Movie Recommender System
> Content-based and collaborative recommendation methods on MovieLens

- toc: true
- badges: true
- comments: true
- categories: [movie]
- image:
<!-- #endregion -->

<!-- #region id="2VxsglaJ6GST" -->
## Load data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="-KT8-sN7wg-F" outputId="fdcb38a6-8c68-4550-f1a4-ddffa0f7d566"
!mkdir '/content/data'

from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(file_id='1Of9rK8ds1a1iyl1jFnf_7oRgPB-8bfdK',
                                    dest_path='/content/data/data.zip',
                                    unzip=True)
```

<!-- #region id="pW_zIWNPhHHE" -->
## Clean data
<!-- #endregion -->

```python id="lAtXbV6NtiuZ"
import os
import numpy as np
import pandas as pd
from ast import literal_eval
```

```python colab={"base_uri": "https://localhost:8080/"} id="aVBVNlyyfIre" outputId="d87618ea-4ef7-40d0-f76b-9c7ac0342f3c"
#hide-output
md = pd.read_csv("/content/data/imdb/movies_metadata.csv")
credits = pd.read_csv('/content/data/imdb/credits.csv')
keywords = pd.read_csv('/content/data/imdb/keywords.csv')
links_small = pd.read_csv('/content/data/imdb/links_small.csv')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 666} id="_3fb0KsuB-kC" outputId="90285303-87cd-4fa1-9254-9d4e8c508ec3"
md.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="SuHC-XzGDUWu" outputId="3d1bb83e-45dd-496b-cfea-fb655ce4ea13"
#hide-output
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
md.loc[:, 'genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
md = md.drop([19730, 29503, 35587])
keywords.loc[:, 'id'] = keywords['id'].astype('int')
credits.loc[:, 'id'] = credits['id'].astype('int')
md.loc[:, 'id'] = md['id'].astype('int')

md = md.merge(credits, on='id')
md = md.merge(keywords, on='id')

smd = md[md['id'].isin(links_small)]

smd.loc[:, 'tagline'] = smd['tagline'].fillna('')

smd.loc[:,'cast'] = smd['cast'].apply(literal_eval)
smd.loc[:,'crew'] = smd['crew'].apply(literal_eval)
smd.loc[:,'keywords'] = smd['keywords'].apply(literal_eval)
smd.loc[:,'cast_size'] = smd['cast'].apply(lambda x: len(x))
smd.loc[:,'crew_size'] = smd['crew'].apply(lambda x: len(x))

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

smd.loc[:,'director'] = smd['crew'].apply(get_director)
smd.loc[:,'cast'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd.loc[:,'cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)
smd.loc[:,'keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

s = smd.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'
s = s.value_counts()
s = s[s > 1]

def filter_keywords(x):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words
smd.loc[:,'keywords'] = smd['keywords'].apply(filter_keywords)
smd.drop_duplicates(subset ="title",
                     keep = 'first', inplace = True)
```

```python id="WnOJnf9sYmqN"
out_df = smd[['id', 'title', 'year', 'director', 'cast',  'genres', 'vote_count', 'vote_average',  'overview', 'keywords']]
out_df.head()
out_df.to_csv('super_clean_data.csv', index=False)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 527} id="SbfF_9NteBs5" outputId="eb9e4776-61f2-494d-f266-49b7d977f39f"
out_df.head()
```

<!-- #region id="ARKpHfZahP0S" -->
## Content-based Recommender
<!-- #endregion -->

```python id="xNLtLlWmhZk1"
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
```

```python colab={"base_uri": "https://localhost:8080/", "height": 323} id="sHu4JOUphdlB" outputId="635d414a-e138-46bd-bda6-303d198b6c6f"
ori_df = pd.read_csv('/content/super_clean_data.csv')
df = ori_df.copy()
df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="ju_S1ThQMvXv" outputId="0a413ff0-906f-4730-ad29-556e94edb02a"
print(f"No of records: {len(df)}")
```

<!-- #region id="qhXLpEfqiJLE" -->
### Preprocess data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 289} id="w7NI-Us4iG0m" outputId="def9d626-0849-436b-9a5e-d1316f58c565"
df.loc[:,'cast'] = df['cast'].apply(literal_eval)
df.loc[:,'genres'] = df['genres'].apply(literal_eval)
df.loc[:,'keywords'] = df['keywords'].apply(literal_eval)

stemmer = SnowballStemmer('english')

def preprocess(x, remove_spaces=False, stemming=False):
    if isinstance(x, list):
        y = []
        for i in x:
            token = preprocess(i, remove_spaces, stemming)
            if token is not None:
                y.append(token)
    else:
        
        y = str(x)

        # Lower all words
        y = str.lower(y)

        # Remove spaces (for person's name)
        if remove_spaces:
            y = y.replace(" ", "")

        # Remove digits
        y = ''.join([i for i in y if not i.isdigit()])

        # Stemming words
        if stemming:
            y = stemmer.stem(y)

        if len(y) <=1:
            return None

    return y


df.loc[:,'cast'] = df['cast'].apply(lambda x: preprocess(x, remove_spaces=True))
df.loc[:,'director'] = df['director'].astype('str').apply(lambda x: preprocess(x, remove_spaces=True))
df.loc[:, 'title'] = df['title'].apply(lambda x: preprocess(x, stemming=True))
df.loc[:, 'overview'] = df['overview'].apply(lambda x: preprocess(str.split(str(x)), stemming=True))
df.loc[:, 'genres'] = df['genres'].apply(lambda x: preprocess(x, stemming=True))
df.loc[:,'keywords'] = df['keywords'].apply(lambda x: preprocess(x, stemming=True))
df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="eIuBkqxzpp62" outputId="84b714b6-53a7-4298-b890-1dd87be8f915"
df.shape
```

<!-- #region id="q0KlmRDAm1FN" -->
### Vectorize using TF-IDF
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="cuOOz3ZVmrkv" outputId="99ce7e00-29fc-40eb-f565-0e10a527fa7f"
dictionary = []
for i, row in df.iterrows():
    item = [row.title, row.director] + row.cast + row.genres + row.keywords
    string = ' '.join([j for j in item if j is not None])
    dictionary.append(string)


tf = TfidfVectorizer(analyzer='word',min_df=2, stop_words='english')
tfidf_matrix = tf.fit_transform(dictionary)
print(tfidf_matrix.shape)
print(tf.get_feature_names()[:10])
```

<!-- #region id="cNyDbFIY5erU" -->
### Cosine similarity matrix
<!-- #endregion -->

```python id="z40xraE55jdI"
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
```

<!-- #region id="MJnPDtU-5oLW" -->
### Get recommendations
<!-- #endregion -->

```python id="6gHbeZS8Fg5P"
def get_recommendations(query_title, cosine_sim, df, top_k=10):
    df = df.reset_index()
    titles = df['title']
    indices = pd.Series(df.index, index=df['title'])

    # query_title = preprocess(query_title)
    query_idx = indices[query_title]

    # Get similarity score of current movie with others
    sim_scores = list(enumerate(cosine_sim[query_idx]))

    # Sort scores and get top k
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_k+1]

    movie_indices = [i[0] for i in sim_scores]
    movie_scores = [i[1] for i in sim_scores]
    result = titles.iloc[movie_indices].to_frame()
    result['matching_score'] = movie_scores
    return result
```

```python colab={"base_uri": "https://localhost:8080/", "height": 359} id="PGz8zsDg2a3L" outputId="358bc2ce-214e-4225-e9ba-9093e0596953"
get_recommendations("The Dark Knight", cosine_sim, ori_df)
```

<!-- #region id="yYhJk-wB6Ni5" -->
## Collaborative Filtering
<!-- #endregion -->

<!-- #region id="tzuoZXi27gQN" -->
### Item-based Recommender

<!-- #endregion -->

```python id="yYOpjJLF6gn4"
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.metrics.pairwise import cosine_similarity
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="CPJwxGn9eypG" outputId="ce695f1c-7056-4ed4-f558-e78b7b70a082"
ratings = pd.read_csv("/content/data/imdb/ratings_small.csv")
ratings.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="E-81FBpjPFBq" outputId="f7f02a45-7fd6-47cd-d16c-3b44aa296abf"
movie_data = pd.read_csv("/content/super_clean_data.csv")
movie_id_title = movie_data[['id', 'title']]
movie_id_title.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 359} id="VV2pX2VkTddL" outputId="8c916341-df33-48ef-b73a-86c678ae4b59"
top_ratings = movie_data[['title', 'vote_count']]
top_ratings.sort_values('vote_count', ascending=False).head(10)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="OuLXwcWKPeIP" outputId="ac473808-a5aa-4f51-aa81-ac337af5e1cd"
new_ratings = ratings.merge(movie_id_title, left_on='movieId', right_on='id')
new_ratings.head()
```

<!-- #region id="qPUKY2sPDOK3" -->
#### User-Item matrix
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 369} id="OC1sQl55BNnw" outputId="ef7ae194-c6c3-4843-d4d6-b0448a04309c"
ui_matrix = new_ratings.pivot(index = 'userId', columns ='title', values = 'rating').fillna(0)
ui_matrix.head()
```

```python id="ukmx-ZnXXlQ_"
movie_title = ui_matrix.columns
index_movies = pd.Series(movie_title, index=(range(len(movie_title))))
movie_indices = pd.Series(range(len(movie_title)), index=movie_title)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Cl7hMul-X-O0" outputId="6a54a5f5-7039-4fa7-d79d-7a2f45bba04a"
movie_indices
```

<!-- #region id="lbBYQ4y-Dz7d" -->
#### Mean rating of each movie
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="yO0Swlj5DgiO" outputId="e3047ae8-bf0a-491b-cae4-2724ae8d6f30"
sum_ratings = ui_matrix.sum(axis=0)
num_ratings = ui_matrix[ui_matrix>0].count()
mean_ratings = sum_ratings/num_ratings
mean_ratings.head()
```

<!-- #region id="z9OaU9cR_kIo" -->
#### Use k nearest neighbors to predict score
<!-- #endregion -->

```python id="f_EM6e3M_sgi"
def predict_score(ui_matrix, user_name, movie_name, mean_ratings, k =2):
    
    movie_id = movie_indices[movie_name]
    ui_matrix_ = ui_matrix.dropna()
    cosine_sim = cosine_similarity(ui_matrix_.T, ui_matrix_.T)

    # nearest neighbors
    sim_scores = list(enumerate(cosine_sim[movie_id]))
    
    # Sort scores and get top k
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:k+1]

    # print(f"Nearest movies of {movie_name}:", end='')
    # nearest_neighor_movies = [index_movies[i[0]] for i in sim_scores]
    # print(nearest_neighor_movies)

    r_ui = mean_ratings[movie_name]

    total_scores = sum([i[1] for i in sim_scores])
    for movie_j, score_ij in sim_scores:
        r_uj = ui_matrix.loc[user_name, index_movies[movie_j]]
        rmean_j = mean_ratings.iloc[movie_j]
        r_ui += ((score_ij*(r_uj - rmean_j))/total_scores)

    return r_ui
```

```python colab={"base_uri": "https://localhost:8080/"} id="ziY5NZRXFpjk" outputId="0a854ff7-0ce8-4f60-e559-a1bc046a848c"
user_id = 4
movie_name = "Young Frankenstein"
num_neighbors = 10

score_4yf = ui_matrix.loc[user_id, movie_name]
print(f"True real rating of user {user_id} for movie {movie_name} is {score_4yf}")

pred_4yf = predict_score(ui_matrix, user_id, movie_name, mean_ratings, k=num_neighbors)
print(f"True predicted rating of {user_id} for movie {movie_name} is {pred_4yf}")
```

<!-- #region id="NWqFXNsQ7Xg1" -->
### Model-based Recommender
<!-- #endregion -->

```python id="jxm4yP9v7ju8"
import pandas as pd
import numpy as np
from ast import literal_eval
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="x4J5HS-FVEiD" outputId="1bdbfb76-81e1-4344-f2c3-ff8bc959154c"
ratings = pd.read_csv("/content/data/imdb/ratings_small.csv")
ratings.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="yCvOSVhcVM6C" outputId="52652162-592e-4383-ec78-c3b27d86a5e5"
movie_data = pd.read_csv("/content/super_clean_data.csv")
movie_id_title = movie_data[['id', 'title']]
movie_id_title.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="LiPk6CFtVQdO" outputId="886fba1b-4e3a-49ca-f79d-bc0011e1452a"
new_ratings = ratings.merge(movie_id_title, left_on='movieId', right_on='id')
new_ratings.head()
```

<!-- #region id="vqOnotxNVXjN" -->
#### User-Item matrix
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 369} id="KSQlnCj5VXjP" outputId="0c4c3e30-db3e-486d-ffd6-db4079617eb6"
ui_matrix = new_ratings.pivot(index = 'userId', columns ='title', values = 'rating').fillna(0)
ui_matrix.head()
```

<!-- #region id="S7WgcSjHY77J" -->
#### SVD Decomposition: Low rank factorization
<!-- #endregion -->

```python id="dUxzD_WtVXjS"
# Singular Value Decomposition
U, sigma, Vt = svds(ui_matrix, k = 600)

# Construct diagonal array in SVD
sigma = np.diag(sigma)
```

```python colab={"base_uri": "https://localhost:8080/"} id="xInU-FY6V0KU" outputId="eff46f80-8cd2-44b3-aaac-de2c906865c7"
print("X = U * sigma * Vt")
print(f"{ui_matrix.shape} = {U.shape} * {sigma.shape} * {Vt.shape}")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 338} id="5TfQkyr3XJDm" outputId="41d747ef-b68a-4be4-bed1-9ae13eee4a73"
# Low-rank matrix
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 

# Convert predicted ratings to dataframe
pred_ui_matrix = pd.DataFrame(all_user_predicted_ratings, columns = ui_matrix.columns)
pred_ui_matrix.head()
```

<!-- #region id="0wH_B5XfZJyZ" -->
#### Predict score 
<!-- #endregion -->

```python id="pRfA7YM1ZMhO"
def predict_score(pred_ui_matrix, user_id, movie_name):
    return pred_ui_matrix.loc[user_id-1, movie_name]
```

```python colab={"base_uri": "https://localhost:8080/"} id="7RCuq5_PXYdi" outputId="3299a200-4fca-4867-c095-90334b721883"
user_id = 4
movie_name = "Young Frankenstein"

score_4yf = ui_matrix.loc[user_id, movie_name]
print(f"True real rating of user {user_id} for movie {movie_name} is {score_4yf}")

pred_4yf = predict_score(pred_ui_matrix, user_id, movie_name)
print(f"True predicted rating of {user_id} for movie {movie_name} is {pred_4yf}")
```

<!-- #region id="MPbKpny4aKmL" -->
#### Evaluate model
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 235} id="9QlBcse2aNZc" outputId="35cf3f60-ea7b-48bc-94d7-af96033a71e2"
rmse_df = pd.concat([ui_matrix.mean(), pred_ui_matrix.mean()], axis=1)
rmse_df.columns = ['Avg_actual_ratings', 'Avg_predicted_ratings']
rmse_df['item_index'] = np.arange(0, rmse_df.shape[0], 1)
rmse_df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="bWRw1EhzapAq" outputId="87a6dbbc-a044-4a10-f9a9-17b6f1e926a5"
RMSE = round((((rmse_df.Avg_actual_ratings - rmse_df.Avg_predicted_ratings) ** 2).mean() ** 0.5), 5)
print(f'RMSE SVD Model = {RMSE}')
```

<!-- #region id="bm9JLC3Mbiwa" -->
##### Evaluate with different value k
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="39D8sUD7bhvd" outputId="50bddce0-c523-4d73-9fb2-e89ffc9ae793"
for i in [10, 100, 300, 500, 600]:

    # Singular Value Decomposition
    U, sigma, Vt = svds(ui_matrix, k = i)

    # Construct diagonal array in SVD
    sigma = np.diag(sigma)

    # Low-rank matrix
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 

    # Convert predicted ratings to dataframe
    pred_ui_matrix = pd.DataFrame(all_user_predicted_ratings, columns = ui_matrix.columns)

    rmse_df = pd.concat([ui_matrix.mean(), pred_ui_matrix.mean()], axis=1)
    rmse_df.columns = ['Avg_actual_ratings', 'Avg_predicted_ratings']
    rmse_df['item_index'] = np.arange(0, rmse_df.shape[0], 1)

    RMSE = round((((rmse_df.Avg_actual_ratings - rmse_df.Avg_predicted_ratings) ** 2).mean() ** 0.5), 5)
    print(f'RMSE with value k = {i} : {RMSE}')
```

<!-- #region id="_LCh2Or-dJ4K" -->
#### Recommend movies
<!-- #endregion -->

```python id="CQMAWqXZdOwC"
# Recommend the items with the highest predicted ratings

def recommend_items(user_id, ui_matrix, pred_ui_matrix, num_recommendations=5):

    # Get and sort the user's ratings
    sorted_user_ratings = ui_matrix.loc[user_id].sort_values(ascending=False)
    #sorted_user_ratings
    sorted_user_predictions = pred_ui_matrix.loc[user_id-1].sort_values(ascending=False)
    #sorted_user_predictions
    temp = pd.concat([sorted_user_ratings, sorted_user_predictions], axis=1)
    temp.index.name = 'Recommended Items'
    temp.columns = ['user_ratings', 'user_predictions']
    temp = temp.loc[temp.user_ratings == 0]   
    temp = temp.sort_values('user_predictions', ascending=False)
    print('\nBelow are the recommended items for user(user_id = {}):\n'.format(user_id))
    print(temp.head(num_recommendations))
```

```python colab={"base_uri": "https://localhost:8080/"} id="hgClAT4MdxhM" outputId="d8d26e70-a72c-42c3-df8d-0eef1672ad5d"
recommend_items(4, ui_matrix, pred_ui_matrix, num_recommendations=5)
```
