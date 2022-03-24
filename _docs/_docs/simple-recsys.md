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

# Basic Movie Recommenders
> Content-based, Collaborative filtering, model-based and neural MF models on movielens data.

- toc: true
- badges: true
- comments: true
- categories: [recsys, movie]

```python _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" colab={} colab_type="code" id="VhfCjBdRFW9A"
import numpy as np
import pandas as pd

movies = pd.read_csv('movies.csv', sep=',')
ratings = pd.read_csv('ratings.csv', sep=',')
```

<!-- #region colab_type="text" id="-EQCCB4HFW9F" -->
### Content-based and Collaborative filtering algorithms
Two most ubiquitous types of personalized recommendation systems are Content-Based and Collaborative Filtering. Collaborative filtering produces recommendations based on the knowledge of users’ attitude to items, that is it uses the “wisdom of the crowd” to recommend items. In contrast, content-based recommendation systems focus on the attributes of the items and give you recommendations based on the similarity between them.

Ref.- https://github.com/khanhnamle1994/movielens/blob/master/Content_Based_and_Collaborative_Filtering_Models.ipynb
<!-- #endregion -->

```python colab={} colab_type="code" id="NgFSTADaFW9G" outputId="854155da-9698-47bb-f356-c116143d2751"
### content-based recommendation engine - TF-IDF vectorizer with movie-similarity based recommendation ###

# Break up the big genre string into a string array
movies['genres'] = movies['genres'].str.split('|')
# Convert genres to string value
movies['genres'] = movies['genres'].fillna("").astype('str')

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(movies['genres'])

#cosine distance metric to measure the similarity between movies
from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function that get movie recommendations based on the cosine similarity score of movie genres
titles = movies['title']
indices = pd.Series(movies.index, index=movies['title'])
def genre_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]

genre_recommendations('Toy Story (1995)').head(5)
```

```python colab={} colab_type="code" id="6EtVJEcyFW9K" outputId="347f241e-a37f-4dca-9458-a901859e53fc"
### collaborative filtering recommendation engine - (item-item similarity) and (user-user similarity), memory-based recommendation ###

#preprocessing
ratings['userId'] = ratings['userId'].fillna(0)
ratings['movieId'] = ratings['movieId'].fillna(0)
ratings['rating'] = ratings['rating'].fillna(ratings['rating'].mean())
#sampling
sample_ratings = ratings.sample(frac=0.2)

from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(sample_ratings, test_size=0.2)

# Create two user-item matrices, one for training and another for testing
train_data_matrix = train_data.as_matrix(columns = ['userId', 'movieId', 'rating'])
test_data_matrix = test_data.as_matrix(columns = ['userId', 'movieId', 'rating'])

from sklearn.metrics.pairwise import pairwise_distances
# User Similarity Matrix
user_correlation = 1 - pairwise_distances(train_data, metric='correlation')
user_correlation[np.isnan(user_correlation)] = 0
# Item Similarity Matrix
item_correlation = 1 - pairwise_distances(train_data_matrix.T, metric='correlation')
item_correlation[np.isnan(item_correlation)] = 0

# Function to predict ratings
def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        # Use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

# Function to calculate RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return sqrt(mean_squared_error(pred, actual))

# Predict ratings on the training data with both similarity score
user_prediction = predict(train_data_matrix, user_correlation, type='user')
item_prediction = predict(train_data_matrix, item_correlation, type='item')

# RMSE on the train data
print('User-based CF RMSE (Train): ' + str(rmse(user_prediction, train_data_matrix)))
print('User-based CF RMSE (Test): ' + str(rmse(user_prediction, test_data_matrix)))

# RMSE on the test data
print('Item-based CF RMSE (Train): ' + str(rmse(item_prediction, train_data_matrix)))
print('Item-based CF RMSE (Test): ' + str(rmse(item_prediction, test_data_matrix)))
```

<!-- #region colab_type="text" id="k4lpjOGZFW9O" -->
### Model-Based Collaborative Filtering
Model-based Collaborative Filtering is based on matrix factorization (MF) which has received greater exposure, mainly as an unsupervised learning method for latent variable decomposition and dimensionality reduction. Matrix factorization is widely used for recommender systems where it can deal better with scalability and sparsity than Memory-based CF:
1. The goal of MF is to learn the latent preferences of users and the latent attributes of items from known ratings (learn features that describe the characteristics of ratings) to then predict the unknown ratings through the dot product of the latent features of users and items.
2. When you have a very sparse matrix, with a lot of dimensions, by doing matrix factorization, you can restructure the user-item matrix into low-rank structure, and you can represent the matrix by the multiplication of two low-rank matrices, where the rows contain the latent vector.
3. You fit this matrix to approximate your original matrix, as closely as possible, by multiplying the low-rank matrices together, which fills in the entries missing in the original matrix.

https://nbviewer.jupyter.org/github/khanhnamle1994/movielens/blob/master/SVD_Model.ipynb
<!-- #endregion -->

```python colab={} colab_type="code" id="hFoZpJywFW9P" outputId="b3bd63c4-1717-4d23-fbeb-d0fde3cff639"
n_users = ratings.userId.unique().shape[0]
n_movies = ratings.movieId.unique().shape[0]
print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_movies))

Ratings = ratings.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)

user_ratings_mean = np.mean(Ratings.values, axis = 1)
Ratings_demeaned = Ratings.values - user_ratings_mean.reshape(-1, 1)
Ratings.head()
```

```python colab={} colab_type="code" id="E4fOE5LPFW9S" outputId="2f488b62-15d9-43a4-e451-e582384b9094"
#check sparsity
sparsity = round(1.0 - len(ratings) / float(n_users * n_movies), 3)
print('The sparsity level of dataset is ' +  str(sparsity * 100) + '%')

#singular value decomposition
from scipy.sparse.linalg import svds
U, sigma, Vt = svds(Ratings_demeaned, k = 50)
sigma = np.diag(sigma)

#add the user means back to get the actual star ratings prediction
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

#predictions
preds = pd.DataFrame(all_user_predicted_ratings, columns = Ratings.columns)

#function to return the movies with the highest predicted rating that the specified user hasn't already rated
def recommend_movies(predictions, userID, movies, original_ratings, num_recommendations):
    # Get and sort the user's predictions
    user_row_number = userID - 1 # User ID starts at 1, not 0
    sorted_user_predictions = preds.iloc[user_row_number].sort_values(ascending=False) # User ID starts at 1
    # Get the user's data and merge in the movie information.
    user_data = original_ratings[original_ratings.userId == (userID)]
    user_full = (user_data.merge(movies, how = 'left', left_on = 'movieId', right_on = 'movieId').
                     sort_values(['rating'], ascending=False))
    print('User {0} has already rated {1} movies.'.format(userID, user_full.shape[0]))
    print('Recommending highest {0} predicted ratings movies not already rated.'.format(num_recommendations))
    
    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (movies[~movies['movieId'].isin(user_full['movieId'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'movieId',
               right_on = 'movieId').
         rename(columns = {user_row_number: 'Predictions'}).
         sort_values('Predictions', ascending = False).
                       iloc[:num_recommendations, :-1]
                      )
    return user_full, recommendations

#checking predicitons for user ID '600'
already_rated, predictions = recommend_movies(preds, 1, movies, ratings, 5)
# Top 5 movies that User has rated 
print(already_rated.head())
predictions
```

<!-- #region colab_type="text" id="41PR5KbBFW9V" -->
### Neural collaborative filtering with FastAI 
(https://jvn.io/aakashns/5bc23520933b4cc187cfe18e5dd7e2ed)
<!-- #endregion -->

```python colab={} colab_type="code" id="TPBdZy9HFW9W" outputId="294e7a02-f347-4718-fb2b-a18cb9f5552f"
from fastai.collab import CollabDataBunch, collab_learner
data = CollabDataBunch.from_df(ratings, valid_pct=0.1)
data.show_batch()
```

<!-- #region colab_type="text" id="L5POHgVxFW9b" -->
The model itself is quite simple. We represent each user u and each movie m by vector of a predefined length n. The rating for the movie m by the user u, as predicted by the model is simply the dot product of the two vectors.
<img src="https://cdn-images-1.medium.com/max/800/1*RuAjbXDwvTAv74NtPS-FbQ.png" width=400>
<!-- #endregion -->

```python colab={} colab_type="code" id="94zKrdSxFW9b" outputId="166735e6-56fb-48ae-d579-1ec342c329c3"
learn = collab_learner(data, n_factors=40, y_range=[0,5.5], wd=.1)
learn.fit_one_cycle(5, 0.01)
```

<!-- #region colab_type="text" id="5Z0eRKH-FW9e" -->
### Collaborative Filtering using Weighted BiPartite Graph Projection
https://pdfs.semanticscholar.org/fd9a/bc146ef857b55eebfe38977cd65e976f36db.pdf
<!-- #endregion -->
