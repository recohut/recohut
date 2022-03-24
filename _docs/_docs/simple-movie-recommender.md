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

<!-- #region id="m_NQm5KcOMLz" -->
# Simple movie recommender in implicit, explicit, and cold-start settings
> Applying data cleaning, exploration, explicit KNN model, implicit ALS model, and cold-start scenario on movielens dataset

- toc: true
- badges: true
- comments: true
- categories: [Movie, Implicit, KNN]
- author: "<a href='https://github.com/topspinj/recommender-tutorial'>Jill Cates</a>"
- image:
<!-- #endregion -->

<!-- #region id="yzwCHncOE8uf" -->
## Import the Dependencies
<!-- #endregion -->

```python id="ChD__RSUNauT"
!pip install -q fuzzywuzzy python-Levenshtein implicit
```

```python id="9Js8DW0HE8ug"
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
import re 
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
from scipy.sparse import csr_matrix
import implicit

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
```

<!-- #region id="sXE4BARrE8uh" -->
## Load the Data

Let's download a small version of the [MovieLens](https://www.wikiwand.com/en/MovieLens) dataset. You can access it via the zip file url [here](https://grouplens.org/datasets/movielens/), or directly download [here](http://files.grouplens.org/datasets/movielens/ml-latest-small.zip). We're working with data in `ml-latest-small.zip` and will need to add the following files to our local directory: 
- ratings.csv
- movies.csv

These are also located in the data folder inside this GitHub repository. 

Alternatively, you can access the data here: 
- https://s3-us-west-2.amazonaws.com/recommender-tutorial/movies.csv
- https://s3-us-west-2.amazonaws.com/recommender-tutorial/ratings.csv

Let's load in our data and take a peek at the structure.
<!-- #endregion -->

```python id="TPH1aXhhE8ui" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="6f628bcf-f4d5-4d37-dc2c-ce93840df69c"
ratings = pd.read_csv("https://s3-us-west-2.amazonaws.com/recommender-tutorial/ratings.csv")
ratings.head()
```

```python id="FSIDTQsfE8uk" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="c0dbdf10-fe5c-4d2f-a4d4-1ea14d8228be"
movies = pd.read_csv("https://s3-us-west-2.amazonaws.com/recommender-tutorial/movies.csv")
movies.head()
```

<!-- #region id="1AB0Y3FaF1dX" -->
## Data Cleaning and Exploration
<!-- #endregion -->

<!-- #region id="gAnqa_yWF1dY" -->
### Converting Genres from String Format to List 

The genres column is currently a string separated with pipes. Let's convert this into a list using the "split" function.

We want 
`"Adventure|Children|Fantasy"`
to convert to this:
`[Adventure, Children, Fantasy]`.
<!-- #endregion -->

```python id="b3FbtK2dF1da" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="ad93034f-6c56-4505-84f7-c0eb45e3e0da"
movies['genres'] = movies['genres'].apply(lambda x: x.split("|"))
movies.head()
```

<!-- #region id="ZiX7_ztSF1db" -->
### How many movie genres are there?

We can use Python's Counter to create a dictionary containing frequency counts of each genre in our dataset.
<!-- #endregion -->

```python id="kXnM7XXHF1db" colab={"base_uri": "https://localhost:8080/"} outputId="e5247427-3f6f-41b5-ebd7-c4e05f7c2369"
genres_counts = Counter(g for genres in movies['genres'] for g in genres)
print(f"There are {len(genres_counts)} genre labels.")
genres_counts
```

<!-- #region id="ZJjes2VqF1dc" -->
There are 20 genre labels and 19 genres that are used to describe movies in this dataset. Some movies don't have any genres, hence the label `(no genres listed)`. 

Let's remove all movies having `(no genres listed)` as its genre label. We'll also remove this from our `genre_counts` dictionary. 
<!-- #endregion -->

```python id="wtaP--SRF1dd"
movies = movies[movies['genres']!='(no genres listed)']

del genres_counts['(no genres listed)']
```

<!-- #region id="QoqCUjZ6F1dd" -->
### What are the most popular genres?

We can use `Counter`'s [most_common()](https://docs.python.org/2/library/collections.html#collections.Counter.most_common) method to get the genres with the highest movie counts.
<!-- #endregion -->

```python id="n56wR-7XF1dd" colab={"base_uri": "https://localhost:8080/"} outputId="cb2c0fe1-5962-4336-f0dd-0a1e92389395"
print("The 5 most common genres: \n", genres_counts.most_common(5))
```

<!-- #region id="DkB5GkZIF1de" -->
The top 5 genres are: `Drama`, `Comedy`, `Thriller`, `Action` and `Romance`. 

Let's also visualize genres popularity with a barplot.
<!-- #endregion -->

```python id="U6_IrqtKF1de" colab={"base_uri": "https://localhost:8080/", "height": 391} outputId="634f2cc4-363b-4da8-f189-645d7ec75401"
genres_counts_df = pd.DataFrame([genres_counts]).T.reset_index()
genres_counts_df.columns = ['genres', 'count']
genres_counts_df = genres_counts_df.sort_values(by='count', ascending=False)

plt.figure(figsize=(10,5))
sns.barplot(x='genres', y='count', data=genres_counts_df, palette='viridis')
plt.xticks(rotation=90)
plt.show()
```

<!-- #region id="exnlL95vF1df" -->
The plot above shows that `Drama` and `Comedy` are the two most popular movie genres. The least popular movie genres are `Westerns`, `IMAX`, and `Film-Noir`.
<!-- #endregion -->

<!-- #region id="2pNlk9T6F1df" -->
### Parsing out year from movie title

In our dataset, movie titles currently the year of release appended to it in brackets, e.g., `"Toy Story (1995)"`. We want to use the year of a movie's release as a feature, so let's parse it out from the title string and create a new `year` column for it.

We can start with writing a function that parses out year from the title string. In the code below, `extract_year_from_title()` takes in the title and does the following:

- generates a list by splitting out each word by spaces (e.g., `["Toy", "Story", "(1995)"]`)
- gets the last element of the list (e.g., `"(1995)"`)
- if the last element has brackets surrounding it, these `()` brackets get stripped (e.g., `"1995"`)
- converts the year into an integer 
<!-- #endregion -->

```python id="3gNjFuIhF1dg"
def extract_year_from_title(title):
    t = title.split(' ')
    year = None
    if re.search(r'\(\d+\)', t[-1]):
        year = t[-1].strip('()')
        year = int(year)
    return year
```

<!-- #region id="2CEO6Ka0F1dg" -->
We can test out this function with our example of `"Toy Story (1995)"`:
<!-- #endregion -->

```python id="SdsQplVCF1dh" colab={"base_uri": "https://localhost:8080/"} outputId="ff78137f-4b87-4868-a8b0-bef5c6b6bebb"
title = "Toy Story (1995)"
year = extract_year_from_title(title)
print(f"Year of release: {year}")
print(type(year))
```

<!-- #region id="hM2tjQCBF1di" -->
Our function `extract_year_from_title()` works! It's able to successfully parse out year from the title string as shown above. We can now apply this to all titles in our `movies` dataframe using Pandas' [apply()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html) method.
<!-- #endregion -->

```python id="-JGlXtQfF1di" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="22904da9-4d14-45e7-8f77-00cb321ee2e3"
movies['year'] = movies['title'].apply(extract_year_from_title)
movies.head()
```

<!-- #region id="0D715fW-F1di" -->
### How many different years of release are covered in our dataset?
<!-- #endregion -->

```python id="yw8qQ5G6F1dj" colab={"base_uri": "https://localhost:8080/"} outputId="a7913242-3ddc-4259-d83a-a882c7dbcedb"
movies['year'].nunique()
```

<!-- #region id="tsC-849xF1dj" -->
There are over 100 years of release in our dataset. Let's collapse this down into decades to get a general sense of when movies were released in our dataset. 
<!-- #endregion -->

<!-- #region id="QoVnzmIhF1dj" -->
### What was the most popular decade of movie release?

Before we begin, we'll remove all movies with null year.
<!-- #endregion -->

```python id="yLNEzHugF1dk" colab={"base_uri": "https://localhost:8080/"} outputId="cc1c6014-cfec-4862-c5dc-1c4b92a8eeaf"
print(f"Original number of movies: {movies['movieId'].nunique()}")
```

```python id="Dc7pSzBmF1dk" colab={"base_uri": "https://localhost:8080/"} outputId="59a964ce-7e15-4ace-f703-5a55edc6cf95"
movies = movies[~movies['year'].isnull()]
print(f"Number of movies after removing null years: {movies['movieId'].nunique()}")
```

<!-- #region id="sWIHktG6F1dk" -->
We filtered out 24 movies that don't have a year of release. 

Now, there are two ways to get the decade of a year:

1. converting year to string, replacing the fourth (last) number with a 0
2. rounding year down to the nearest 10 

We'll show both implementations in the code below:
<!-- #endregion -->

```python id="3KNsd3QrF1dl" colab={"base_uri": "https://localhost:8080/"} outputId="6b4206a2-d6de-48bf-cb26-b8b7b7f53e5d"
x = 1995

def get_decade(year):
    year = str(year)
    decade_prefix = year[0:3] # get first 3 digits of year
    decade = f'{decade_prefix}0' # append 0 at the end
    return int(decade)

get_decade(x)
```

```python id="EYoQONVAF1dl" colab={"base_uri": "https://localhost:8080/"} outputId="93fa3006-0f10-4b32-f54f-63ef63ed1382"
def round_down(year):
    return year - (year%10)

round_down(x)
```

<!-- #region id="u6sa9iK1F1dl" -->
The two functions `get_decade()` and `round_down()` both accomplish the same thing: they both get the decade of a year.

We can apply either of these functions to all years in our `movies` dataset. We'll use `round_down()` in this example to a create a new column called `'decade'`:
<!-- #endregion -->

```python id="nVXsZjkXF1dm"
movies['decade'] = movies['year'].apply(round_down)
```

```python id="QLlkbHF1F1dm" colab={"base_uri": "https://localhost:8080/", "height": 447} outputId="c1a6fe07-3735-4bca-8167-a6c767d7f62b"
plt.figure(figsize=(10,6))
sns.countplot(movies['decade'], palette='Blues')
plt.xticks(rotation=90)
```

<!-- #region id="WYr5NlqiF1dm" -->
As we can see from the plot above, the most common decade is the 2000s followed by the 1990s for movies in our dataset.
<!-- #endregion -->

```python id="Cmx7ux-gE8um" colab={"base_uri": "https://localhost:8080/"} outputId="58139af5-8b62-4532-ddce-d7d3e5526225"
n_ratings = len(ratings)
n_movies = ratings['movieId'].nunique()
n_users = ratings['userId'].nunique()

print(f"Number of ratings: {n_ratings}")
print(f"Number of unique movieId's: {n_movies}")
print(f"Number of unique users: {n_users}")
print(f"Average number of ratings per user: {round(n_ratings/n_users, 2)}")
print(f"Average number of ratings per movie: {round(n_ratings/n_movies, 2)}")
```

<!-- #region id="qcXuHxW3E8um" -->
Now, let's take a look at users' rating counts. We can do this using pandas' `groupby()` and `count()` which groups the data by `userId`'s and counts the number of ratings for each userId. 
<!-- #endregion -->

```python id="UtEkdmKbE8un" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="81a400e5-01b9-4c93-bf1b-6056a1c74a2a"
user_freq = ratings[['userId', 'movieId']].groupby('userId').count().reset_index()
user_freq.columns = ['userId', 'n_ratings']
user_freq.head()
```

```python id="Z4eY4A43E8un" colab={"base_uri": "https://localhost:8080/"} outputId="f19a1f44-0245-445e-af48-33c3dda2e706"
print(f"Mean number of ratings for a given user: {user_freq['n_ratings'].mean():.2f}.")
```

<!-- #region id="dDe7HDYtE8uo" -->
On average, a user will have rated ~165 movies. Looks like we have some avid movie watchers in our dataset.
<!-- #endregion -->

```python id="N_CA6X8dE8uo" colab={"base_uri": "https://localhost:8080/", "height": 350} outputId="46f9dc7f-5797-4d52-80fc-95c437d862f7"
sns.set_style("whitegrid")
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
ax = sns.countplot(x="rating", data=ratings, palette="viridis")
plt.title("Distribution of movie ratings")

plt.subplot(1,2,2)
ax = sns.kdeplot(user_freq['n_ratings'], shade=True, legend=False)
plt.axvline(user_freq['n_ratings'].mean(), color="k", linestyle="--")
plt.xlabel("# ratings per user")
plt.ylabel("density")
plt.title("Number of movies rated per user")
plt.show()
```

<!-- #region id="CJxvIxOkE8up" -->
The most common rating is 4.0, while lower ratings such as 0.5 or 1.0 are much more rare. 
<!-- #endregion -->

<!-- #region id="IgqwQQUzE8up" -->
### Which movie has the lowest and highest average rating?
<!-- #endregion -->

```python id="Q1N70i7wE8uq" colab={"base_uri": "https://localhost:8080/", "height": 80} outputId="3171e256-8af4-439a-c9f1-7750e20b6df6"
mean_rating = ratings.groupby('movieId')[['rating']].mean()

lowest_rated = mean_rating['rating'].idxmin()
movies.loc[movies['movieId'] == lowest_rated]
```

<!-- #region id="wCzgGjDfE8uq" -->
Santa with Muscles is the worst rated movie!
<!-- #endregion -->

```python id="GkSZ2o0pE8uq" colab={"base_uri": "https://localhost:8080/", "height": 80} outputId="59a04fb6-0784-4a88-ba6e-d13330acd9b3"
highest_rated = mean_rating['rating'].idxmax()
movies.loc[movies['movieId'] == highest_rated]
```

<!-- #region id="nHQzK0hKE8ur" -->
Lamerica may be the "highest" rated movie, but how many ratings does it have?
<!-- #endregion -->

```python id="vXNyFd9mE8ur" colab={"base_uri": "https://localhost:8080/", "height": 111} outputId="e53a3a8d-83dd-4e29-8b31-6c76788eac6a"
ratings[ratings['movieId']==highest_rated]
```

<!-- #region id="9yht1-MSE8us" -->
Lamerica has only 2 ratings. A better approach for evaluating movie popularity is to look at the [Bayesian average](https://en.wikipedia.org/wiki/Bayesian_average).
<!-- #endregion -->

<!-- #region id="CJ4dp2fyE8ut" -->
### Bayesian Average

Bayesian Average is defined as:

$r_{i} = \frac{C \times m + \Sigma{\text{reviews}}}{C+N}$

where $C$ represents our confidence, $m$ represents our prior, and $N$ is the total number of reviews for movie $i$. In this case, our prior will be the average rating across all movies. By defintion, C represents "the typical dataset size". Let's make $C$ be the average number of ratings for a given movie.
<!-- #endregion -->

```python id="ytMWg-HbE8uu"
movie_stats = ratings.groupby('movieId')[['rating']].agg(['count', 'mean'])
movie_stats.columns = movie_stats.columns.droplevel()
```

```python id="e1Z5hBcCE8uv"
C = movie_stats['count'].mean()
m = movie_stats['mean'].mean()

def bayesian_avg(ratings):
    bayesian_avg = (C*m+ratings.sum())/(C+ratings.count())
    return bayesian_avg

bayesian_avg_ratings = ratings.groupby('movieId')['rating'].agg(bayesian_avg).reset_index()
bayesian_avg_ratings.columns = ['movieId', 'bayesian_avg']
movie_stats = movie_stats.merge(bayesian_avg_ratings, on='movieId')
```

```python id="r2hkOiW-E8uv" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="080412c1-3ebb-44cf-936e-a2c023c945ea"
movie_stats = movie_stats.merge(movies[['movieId', 'title']])
movie_stats.sort_values('bayesian_avg', ascending=False).head()
```

<!-- #region id="ioewti2RE8uw" -->
Using the Bayesian average, we see that `Shawshank Redemption`, `The Godfather`, and `Fight Club` are the most highly rated movies. This result makes much more sense since these movies are critically acclaimed films.

Now which movies are the worst rated, according to the Bayesian average?
<!-- #endregion -->

```python id="ZEN4WtQ_E8ux" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="8c193e0a-05ea-4918-c8ea-02108493c76c"
movie_stats.sort_values('bayesian_avg', ascending=True).head()
```

<!-- #region id="29CoIE31E8ux" -->
With Bayesian averaging, it looks like `Speed 2: Cruise Control`, `Battlefield Earth`, and `Godzilla` are the worst rated movies. `Gypsy` isn't so bad after all!
<!-- #endregion -->

<!-- #region id="ucmiRmpaGph8" -->
## Building an Item-Item Recommender

If you use Netflix, you will notice that there is a section titled "Because you watched Movie X", which provides recommendations for movies based on a recent movie that you've watched. This is a classic example of an item-item recommendation. We will generate item-item recommendations using a technique called [collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering). Let's get started! 
<!-- #endregion -->

<!-- #region id="sQ5W_oCTE8uy" -->
### Transforming the data

We will be using a technique called [collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering) to generate user recommendations. This technique is based on the assumption of "homophily" - similar users like similar things. Collaborative filtering is a type of unsupervised learning that makes predictions about the interests of a user by learning from the interests of a larger population.

The first step of collaborative filtering is to transform our data into a `user-item matrix` - also known as a "utility" matrix. In this matrix, rows represent users and columns represent items. The beauty of collaborative filtering is that it doesn't require any information about the users or items to generate recommendations. 
<!-- #endregion -->

<!-- #region id="zCZsygWQMOGp" -->
<!-- #endregion -->

<!-- #region id="gjLiI7JbE8uy" -->
The `create_X()` function outputs a sparse matrix X with four mapper dictionaries:
- **user_mapper:** maps user id to user index
- **movie_mapper:** maps movie id to movie index
- **user_inv_mapper:** maps user index to user id
- **movie_inv_mapper:** maps movie index to movie id

We need these dictionaries because they map which row and column of the utility matrix corresponds to which user ID and movie ID, respectively.

The **X** (user-item) matrix is a [scipy.sparse.csr_matrix](scipylinkhere) which stores the data sparsely.
<!-- #endregion -->

```python id="A8JQWUGtE8uz"
def create_X(df):
    """
    Generates a sparse matrix from ratings dataframe.
    
    Args:
        df: pandas dataframe
    
    Returns:
        X: sparse matrix
        user_mapper: dict that maps user id's to user indices
        user_inv_mapper: dict that maps user indices to user id's
        movie_mapper: dict that maps movie id's to movie indices
        movie_inv_mapper: dict that maps movie indices to movie id's
    """
    N = df['userId'].nunique()
    M = df['movieId'].nunique()

    user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(M))))
    
    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["movieId"])))
    
    user_index = [user_mapper[i] for i in df['userId']]
    movie_index = [movie_mapper[i] for i in df['movieId']]

    X = csr_matrix((df["rating"], (movie_index, user_index)), shape=(M, N))
    
    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper
```

```python id="nnAsL8MdE8uz"
X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_X(ratings)
```

<!-- #region id="Bo9HhKO0E8u2" -->
Let's check out the sparsity of our X matrix.

Here, we calculate sparsity by dividing the number of non-zero elements by total number of elements as described in the equation below: 

$$S=\frac{\text{# non-zero elements}}{\text{total elements}}$$
<!-- #endregion -->

```python id="Abdey5CwE8u3" colab={"base_uri": "https://localhost:8080/"} outputId="cc7ccab2-c940-4141-b90a-ef19a1f9cd3e"
sparsity = X.count_nonzero()/(X.shape[0]*X.shape[1])

print(f"Matrix sparsity: {round(sparsity*100,2)}%")
```

<!-- #region id="9p2q4RhKE8u4" -->
Only 1.7% of cells in our user-item matrix are populated with ratings. But don't be discouraged by this sparsity! User-item matrices are typically very sparse. A general rule of thumb is that your matrix sparsity should be no lower than 0.5% to generate decent results.
<!-- #endregion -->

<!-- #region id="RHkZvrsKE8u5" -->
**Writing your matrix to a file**

We're going to save our user-item matrix for the next part of this tutorial series. Since our matrix is represented as a scipy sparse matrix, we can use the [scipy.sparse.save_npz](https://docs.scipy.org/doc/scipy-1.1.0/reference/generated/scipy.sparse.load_npz.html) method to write the matrix to a file. 
<!-- #endregion -->

```python id="DHBCXQPAE8u6"
save_npz('user_item_matrix.npz', X)
```

<!-- #region id="kPuRkNQUE8u6" -->
### Finding similar movies using k-Nearest Neighbours

This approach looks for the $k$ nearest neighbours of a given movie by identifying $k$ points in the dataset that are closest to movie $m$. kNN makes use of distance metrics such as:

1. Cosine similarity
2. Euclidean distance
3. Manhattan distance
4. Pearson correlation 

Although difficult to visualize, we are working in a M-dimensional space where M represents the number of movies in our X matrix. 
<!-- #endregion -->

<!-- #region id="BTgAlqHLOAIM" -->
<!-- #endregion -->

```python id="nKVcgXYKE8u7"
from sklearn.neighbors import NearestNeighbors

def find_similar_movies(movie_id, X, k, metric='cosine', show_distance=False):
    """
    Finds k-nearest neighbours for a given movie id.
    
    Args:
        movie_id: id of the movie of interest
        X: user-item utility matrix
        k: number of similar movies to retrieve
        metric: distance metric for kNN calculations
    
    Returns:
        list of k similar movie ID's
    """
    neighbour_ids = []
    
    movie_ind = movie_mapper[movie_id]
    movie_vec = X[movie_ind]
    k+=1
    kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    kNN.fit(X)
    if isinstance(movie_vec, (np.ndarray)):
        movie_vec = movie_vec.reshape(1,-1)
    neighbour = kNN.kneighbors(movie_vec, return_distance=show_distance)
    for i in range(0,k):
        n = neighbour.item(i)
        neighbour_ids.append(movie_inv_mapper[n])
    neighbour_ids.pop(0)
    return neighbour_ids
```

<!-- #region id="kK_iTvHCE8u8" -->
`find_similar_movies()` takes in a movieId and user-item X matrix, and outputs a list of $k$ movies that are similar to the movieId of interest. 

Let's see how it works in action. We will first create another mapper that maps `movieId` to `title` so that our results are interpretable. 
<!-- #endregion -->

```python id="n53VmB8mE8u8" colab={"base_uri": "https://localhost:8080/"} outputId="839c481b-6123-4965-810b-0271e4fbc684"
movie_titles = dict(zip(movies['movieId'], movies['title']))

movie_id = 1

similar_ids = find_similar_movies(movie_id, X, k=10)
movie_title = movie_titles[movie_id]

print(f"Because you watched {movie_title}")
for i in similar_ids:
    print(movie_titles[i])
```

<!-- #region id="uHi3vuR5E8u9" -->
The results above show the 10 most similar movies to Toy Story. Most movies in this list are family movies from the 1990s, which seems pretty reasonable. Note that these recommendations are based solely on user-item ratings. Movie features such as genres are not taken into consideration in this approach.  

You can also play around with the kNN distance metric and see what results you would get if you use "manhattan" or "euclidean" instead of "cosine".
<!-- #endregion -->

```python id="guTmgTw1E8u-" colab={"base_uri": "https://localhost:8080/"} outputId="d43996f4-fb55-4480-9516-e0ebbcaee276"
movie_titles = dict(zip(movies['movieId'], movies['title']))

movie_id = 1
similar_ids = find_similar_movies(movie_id, X, k=10, metric="euclidean")

movie_title = movie_titles[movie_id]
print(f"Because you watched {movie_title}:")
for i in similar_ids:
    print(movie_titles[i])
```

<!-- #region id="Kjtx3RhTG2d1" -->
## Handling the Cold Start Problem with Content-Based Filtering

Collaborative filtering relies solely on user-item interactions within the utility matrix. The issue with this approach is that brand new users or items with no interactions get excluded from the recommendation system. This is called the "cold start" problem. Content-based filtering is a way to handle this problem by generating recommendations based on user and item features. We will generate item-item recommendations using content-based filtering.
<!-- #endregion -->

<!-- #region id="T2-n45P7F1dn" -->
### Transforming the Data

In order to build a content-based filtering recommender, we need to set up our dataset so that rows represent movies and columns represent features (i.e., genres and decades).

First, we need to manipulate the `genres` column so that each genre is represented as a separate binary feature. "1" indicates that the movie falls under a given genre, while "0" does not. 
<!-- #endregion -->

```python id="rzfF9M0FF1dn"
genres = list(genres_counts.keys())

for g in genres:
    movies[g] = movies['genres'].transform(lambda x: int(g in x))
```

<!-- #region id="JPhLcZhDF1do" -->
Let's take a look at what the movie genres columns look like:
<!-- #endregion -->

```python id="TVXd1_82F1do" colab={"base_uri": "https://localhost:8080/", "height": 241} outputId="5537bc12-57ff-4da4-d543-0eab203be7ec"
movies[genres].head()
```

<!-- #region id="sZrFABRXF1dp" -->
Great! Our genres columns are represented as binary feautres. The next step is to wrangle our `decade` column so that each decade has its own column. We can do this using pandas' [get_dummies()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html) function, which works by creating a categorical variable into binary variables.
<!-- #endregion -->

```python id="BIBbW2KjF1dp" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="573fd243-a089-4b6b-cc26-03419f463bcb"
movie_decades = pd.get_dummies(movies['decade'])
movie_decades.head()
```

<!-- #region id="aaxkVEKYF1dq" -->
Now, let's create a new `movie_features` dataframe by combining our genres features and decade features. We can do this using pandas' [concat](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html) function which concatenates (appends) genres and decades into a single dataframe.
<!-- #endregion -->

```python id="Ns-cKboRF1dq" colab={"base_uri": "https://localhost:8080/", "height": 241} outputId="296fd3b2-69cb-4bc0-ca08-c1c4ff48609c"
movie_features = pd.concat([movies[genres], movie_decades], axis=1)
movie_features.head()
```

<!-- #region id="neDXHuCgF1dr" -->
Our `movie_features` dataframe is ready. The next step is to start building our recommender. 
<!-- #endregion -->

<!-- #region id="lN753p5RF1dt" -->
### Building a "Similar Movies" Recommender Using Cosine Similarity

We're going to build our item-item recommender using a similarity metric called [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity). 

Cosine similarity looks at the cosine angle between two vectors (e.g., $A$ and $B$). The smaller the cosine angle, the higher the degree of similarity between $A$ and $B$. You can calculate the similarity between $A$ and $B$ with this equation:

$$\cos(\theta) = \frac{A\cdot B}{||A|| ||B||}$$

In this tutorial, we're going to use scikit-learn's cosine similarity [function](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html) to generate a cosine similarity matrix of shape $(n_{\text{movies}}, n_{\text{movies}})$. With this cosine similarity matrix, we'll be able to extract movies that are most similar to the movie of interest.
<!-- #endregion -->

```python id="kJ585XD-F1dt" colab={"base_uri": "https://localhost:8080/"} outputId="cb7b42d7-f9f2-4a2f-f321-cd4d4e3c1c1a"
cosine_sim = cosine_similarity(movie_features, movie_features)
print(f"Dimensions of our movie features cosine similarity matrix: {cosine_sim.shape}")
```

<!-- #region id="dkPhlDO8F1du" -->
As expected, after passing the `movie_features` dataframe into the `cosine_similarity()` function, we get a cosine similarity matrix of shape $(n_{\text{movies}}, n_{\text{movies}})$.

This matrix is populated with values between 0 and 1 which represent the degree of similarity between movies along the x and y axes.
<!-- #endregion -->

<!-- #region id="YISwPL0FF1dv" -->
### Let's create a movie finder function

Let's say we want to get recommendations for movies that are similar to Jumanji. To get results from our recommender, we need to know the exact title of a movie in our dataset. 

In our dataset, Jumanji is actually listed as `'Jumanji (1995)'`. If we misspell Jumanji or forget to include its year of release, our recommender won't be able to identify which movie we're interested in.  

To make our recommender more user-friendly, we can use a Python package called [fuzzywuzzy](https://pypi.org/project/fuzzywuzzy/) which will find the most similar title to a string that you pass in. Let's create a function called `movie_finder()` which take advantage of `fuzzywuzzy`'s string matching algorithm to get the most similar title to a user-inputted string. 
<!-- #endregion -->

```python id="h2HqtO7qF1dv"
def movie_finder(title):
    all_titles = movies['title'].tolist()
    closest_match = process.extractOne(title,all_titles)
    return closest_match[0]
```

<!-- #region id="308_MlFaF1dw" -->
Let's test this out with our Jumanji example. 
<!-- #endregion -->

```python id="SIL6w8bQF1dw" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="b09f211c-c8b7-404f-9b27-d43df321227e"
title = movie_finder('juminji')
title
```

<!-- #region id="Q-eZJqeFF1dx" -->
To get relevant recommendations for Jumanji, we need to find its index in the cosine simialrity matrix. To identify which row we should be looking at, we can create a movie index mapper which maps a movie title to the index that it represents in our matrix. 

Let's create a movie index dictionary called `movie_idx` where the keys are movie titles and values are movie indices:
<!-- #endregion -->

```python id="_ePelWunF1dx" colab={"base_uri": "https://localhost:8080/"} outputId="4bf52805-c3d8-4fec-f0a3-e801e86c83d2"
movie_idx = dict(zip(movies['title'], list(movies.index)))
idx = movie_idx[title]
idx
```

<!-- #region id="j30w1MgLF1dy" -->
Using this handy `movie_idx` dictionary, we know that Jumanji is represented by index 1 in our matrix. Let's get the top 10 most similar movies to Jumanji.
<!-- #endregion -->

```python id="Ucv5-AawF1dy"
n_recommendations=10
sim_scores = list(enumerate(cosine_sim[idx]))
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
sim_scores = sim_scores[1:(n_recommendations+1)]
similar_movies = [i[0] for i in sim_scores]
```

<!-- #region id="TuFlPwvxF1dy" -->
`similar_movies` is an array of indices that represents Jumanji's top 10 recommendations. We can get the corresponding movie titles by either creating an inverse `movie_idx` mapper or using `iloc` on the title column of the `movies` dataframe.
<!-- #endregion -->

```python id="WFZgduiiF1dy" colab={"base_uri": "https://localhost:8080/"} outputId="8e051a05-e71b-408a-88b9-4c35eb95b3b3"
print(f"Because you watched {title}:")
movies['title'].iloc[similar_movies]
```

<!-- #region id="wgtE76QzF1dz" -->
Cool! These recommendations seem pretty relevant and similar to Jumanji. The first 5 movies are family-friendly films from the 90s. 

We can test our recommender further with other movie titles. For your convenience, I've packaged the steps into a single function which takes in the movie title of interest and number of recommendations. 
<!-- #endregion -->

```python id="q4FquYfpF1dz"
def get_content_based_recommendations(title_string, n_recommendations=10):
    title = movie_finder(title_string)
    idx = movie_idx[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:(n_recommendations+1)]
    similar_movies = [i[0] for i in sim_scores]
    print(f"Recommendations for {title}:")
    print(movies['title'].iloc[similar_movies])
```

```python id="idxwY6wRF1d0" colab={"base_uri": "https://localhost:8080/"} outputId="ed8acfcd-1cf4-4de6-fc02-96fd868d6e75"
get_content_based_recommendations('aladin', 5)
```

<!-- #region id="zxlGP6fJIyYE" -->
## Building a Recommender System with Implicit Feedback

In this section, we will build an implicit feedback recommender system using the [implicit](https://github.com/benfred/implicit) package.

What is implicit feedback, exactly? Let's revisit collaborative filtering. In [Part 1](https://github.com/topspinj/recommender-tutorial/blob/master/part-1-item-item-recommender.ipynb), we learned that [collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering) is based on the assumption that `similar users like similar things`. The user-item matrix, or "utility matrix", is the foundation of collaborative filtering. In the utility matrix, rows represent users and columns represent items. 

The cells of the matrix are populated by a given user's degree of preference towards an item, which can come in the form of:

1. **explicit feedback:** direct feedback towards an item (e.g., movie ratings which we explored in [Part 1](https://github.com/topspinj/recommender-tutorial/blob/master/part-1-item-item-recommender.ipynb))
2. **implicit feedback:** indirect behaviour towards an item (e.g., purchase history, browsing history, search behaviour)

Implicit feedback makes assumptions about a user's preference based on their actions towards items. Let's take Netflix for example. If you binge-watch a show and blaze through all seasons in a week, there's a high chance that you like that show. However, if you start watching a series and stop halfway through the first episode, there's suspicion to believe that you probably don't like that show. 
<!-- #endregion -->

```python id="eAnEN09xIyYN"
ratings = pd.read_csv("https://s3-us-west-2.amazonaws.com/recommender-tutorial/ratings.csv")
movies = pd.read_csv("https://s3-us-west-2.amazonaws.com/recommender-tutorial/movies.csv")
```

<!-- #region id="OZTKDZyVIyYP" -->
For this implicit feedback tutorial, we'll treat movie ratings as the number of times that a user watched a movie. For example, if Jane (a user in our database) gave `Batman` a rating of 1 and `Legally Blonde` a rating of 5, we'll assume that Jane watched Batman one time and Legally Blonde five times. 
<!-- #endregion -->

<!-- #region id="38uKPEcjIyYQ" -->
### Transforming the Data

We need to transform the `ratings` dataframe into a user-item matrix where rows represent users and columns represent movies. The cells of this matrix will be populated with implicit feedback: in this case, the number of times a user watched a movie. 

The `create_X()` function outputs a sparse matrix **X** with four mapper dictionaries:
- **user_mapper:** maps user id to user index
- **movie_mapper:** maps movie id to movie index
- **user_inv_mapper:** maps user index to user id
- **movie_inv_mapper:** maps movie index to movie id

We need these dictionaries because they map which row and column of the utility matrix corresponds to which user ID and movie ID, respectively.

The **X** (user-item) matrix is a [scipy.sparse.csr_matrix](scipylinkhere) which stores the data sparsely.
<!-- #endregion -->

```python id="ufs5319IIyYS"
def create_X(df):
    """
    Generates a sparse matrix from ratings dataframe.
    
    Args:
        df: pandas dataframe
    
    Returns:
        X: sparse matrix
        user_mapper: dict that maps user id's to user indices
        user_inv_mapper: dict that maps user indices to user id's
        movie_mapper: dict that maps movie id's to movie indices
        movie_inv_mapper: dict that maps movie indices to movie id's
    """
    N = df['userId'].nunique()
    M = df['movieId'].nunique()

    user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(M))))
    
    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["movieId"])))
    
    user_index = [user_mapper[i] for i in df['userId']]
    movie_index = [movie_mapper[i] for i in df['movieId']]

    X = csr_matrix((df["rating"], (movie_index, user_index)), shape=(M, N))
    
    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper
```

```python id="igP7vyw-IyYU"
X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_X(ratings)
```

<!-- #region id="t3_giHPnIyYU" -->
### Creating Movie Title Mappers

We need to interpret a movie title from its index in the user-item matrix and vice versa. Let's create 2 helper functions that make this interpretation easy:

- `get_movie_index()` - converts a movie title to movie index
    - Note that this function uses [fuzzywuzzy](https://github.com/seatgeek/fuzzywuzzy)'s string matching to get the approximate movie title match based on the string that gets passed in. This means that you don't need to know the exact spelling and formatting of the title to get the corresponding movie index.
- `get_movie_title()` - converts a movie index to movie title
<!-- #endregion -->

```python id="grgTTgwCIyYV"
def movie_finder(title):
    all_titles = movies['title'].tolist()
    closest_match = process.extractOne(title,all_titles)
    return closest_match[0]

movie_title_mapper = dict(zip(movies['title'], movies['movieId']))
movie_title_inv_mapper = dict(zip(movies['movieId'], movies['title']))

def get_movie_index(title):
    fuzzy_title = movie_finder(title)
    movie_id = movie_title_mapper[fuzzy_title]
    movie_idx = movie_mapper[movie_id]
    return movie_idx

def get_movie_title(movie_idx): 
    movie_id = movie_inv_mapper[movie_idx]
    title = movie_title_inv_mapper[movie_id]
    return title 
```

<!-- #region id="n5mmbIuOIyYV" -->
It's time to test it out! Let's get the movie index of `Legally Blonde`. 
<!-- #endregion -->

```python id="j-n5nHwDIyYW" colab={"base_uri": "https://localhost:8080/"} outputId="1e1cdb58-df76-438d-c597-103d12dfc42a"
get_movie_index('Legally Blonde')
```

<!-- #region id="c4Gliw9LIyYW" -->
Let's pass this index value into `get_movie_title()`. We're expecting Legally Blonde to get returned.
<!-- #endregion -->

```python id="ahUOVknvIyYX" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="9f569f04-b724-4b7a-bfb0-bf9f0a662923"
get_movie_title(3282)
```

<!-- #region id="h_fO8nlxIyYX" -->
Great! These helper functions will be useful when we want to interpret our recommender results.
<!-- #endregion -->

<!-- #region id="T-15xfBBIyYY" -->
### Building Our Implicit Feedback Recommender Model


We've transformed and prepared our data so that we can start creating our recommender model.

The [implicit](https://github.com/benfred/implicit) package is built around a linear algebra technique called [matrix factorization](https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems)), which can help us discover latent features underlying the interactions between users and movies. These latent features give a more compact representation of user tastes and item descriptions. Matrix factorization is particularly useful for very sparse data and can enhance the quality of recommendations. The algorithm works by factorizing the original user-item matrix into two factor matrices:

- user-factor matrix (n_users, k)
- item-factor matrix (k, n_items)

We are reducing the dimensions of our original matrix into "taste" dimensions. We cannot interpret what each latent feature $k$ represents. However, we could imagine that one latent feature may represent users who like romantic comedies from the 1990s, while another latent feature may represent movies which are independent foreign language films.

$$X_{mn} \approx P_{mk} \times Q_{nk}^T = \hat{X}$$

<img src="images/matrix-factorization.png" width="60%"/>

In traditional matrix factorization, such as SVD, we would attempt to solve the factorization at once which can be very computationally expensive. As a more practical alternative, we can use a technique called `Alternating Least Squares (ALS)` instead. With ALS, we solve for one factor matrix at a time:

- Step 1: hold user-factor matrix fixed and solve for the item-factor matrix
- Step 2: hold item-factor matrix fixed and solve for the user-item matrix

We alternate between Step 1 and 2 above, until the dot product of the item-factor matrix and user-item matrix is approximately equal to the original X (user-item) matrix. This approach is less computationally expensive and can be run in parallel.

The [implicit](https://github.com/benfred/implicit) package implements matrix factorization using Alternating Least Squares (see docs [here](https://implicit.readthedocs.io/en/latest/als.html)). Let's initiate the model using the `AlternatingLeastSquares` class.
<!-- #endregion -->

```python id="TWZbWWmmIyYZ"
model = implicit.als.AlternatingLeastSquares(factors=50, use_gpu=False)
```

<!-- #region id="Teg2v2iEIyYa" -->
This model comes with a couple of hyperparameters that can be tuned to generate optimal results:

- factors ($k$): number of latent factors,
- regularization ($\lambda$): prevents the model from overfitting during training

In this tutorial, we'll set $k = 50$ and $\lambda = 0.01$ (the default). In a real-world scenario, I highly recommend tuning these hyperparameters before generating recommendations to generate optimal results.

The next step is to fit our model with our user-item matrix. 
<!-- #endregion -->

```python id="A4okNfkXIyYb" colab={"base_uri": "https://localhost:8080/", "height": 66, "referenced_widgets": ["dc4a760deb274503a96719e55b57671b", "a089930bb0764fd0bc48b913cfa317f2", "da03231059ee43c090e03a59aaeddd61", "b45f3a45776c433fa85ce21838a22d7d", "bf973860201a4f1d841e9e08cc70f448", "0ecc80f43e5e41fd8a79383a61de3e35", "250fd11bb37a4daab20d489ef22fca0b", "935fa43fcffc454d9437141ddf219a4c"]} outputId="66b65257-6874-49fe-8110-c046ce844d0e"
model.fit(X)
```

<!-- #region id="EuOcJLGIIyYb" -->
Now, let's test out the model's recommendations. We can use the model's `similar_items()` method which returns the most relevant movies of a given movie. We can use our helpful `get_movie_index()` function to get the movie index of the movie that we're interested in.
<!-- #endregion -->

```python id="d7BnI91TIyYb" colab={"base_uri": "https://localhost:8080/"} outputId="6fa0532a-8f8f-4ae1-b237-42d54b2041c2"
movie_of_interest = 'forrest gump'

movie_index = get_movie_index(movie_of_interest)
related = model.similar_items(movie_index)
related
```

<!-- #region id="xxdtZ6ZwIyYc" -->
The output of `similar_items()` is not user-friendly. We'll need to use our `get_movie_title()` function to interpret what our results are. 
<!-- #endregion -->

```python id="QLcgZ0IgIyYc" colab={"base_uri": "https://localhost:8080/"} outputId="2c347fcd-507c-48fe-9fa2-5fa03abe6d3d"
print(f"Because you watched {movie_finder(movie_of_interest)}...")
for r in related:
    recommended_title = get_movie_title(r[0])
    if recommended_title != movie_finder(movie_of_interest):
        print(recommended_title)
```

<!-- #region id="QO0T9u_HIyYd" -->
When we treat user ratings as implicit feedback, the results look pretty good! You can test out other movies by changing the `movie_of_interest` variable.
<!-- #endregion -->

<!-- #region id="hINJBrH9IyYd" -->
### Generating User-Item Recommendations

A cool feature of [implicit](https://github.com/benfred/implicit) is that you can pull personalized recommendations for a given user. Let's test it out on a user in our dataset.
<!-- #endregion -->

```python id="7EaolVg7IyYd"
user_id = 95
```

```python id="NrSaS7hZIyYe" colab={"base_uri": "https://localhost:8080/"} outputId="9009e838-d7f8-4f96-a37b-7f249a06a6ef"
user_ratings = ratings[ratings['userId']==user_id].merge(movies[['movieId', 'title']])
user_ratings = user_ratings.sort_values('rating', ascending=False)
print(f"Number of movies rated by user {user_id}: {user_ratings['movieId'].nunique()}")
```

<!-- #region id="xi3RLzBAIyYe" -->
User 95 watched 168 movies. Their highest rated movies are below:
<!-- #endregion -->

```python id="dvN7zjeRIyYf" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="1f5bed0e-d8df-44c3-b0fc-7273a47e6ec2"
user_ratings = ratings[ratings['userId']==user_id].merge(movies[['movieId', 'title']])
user_ratings = user_ratings.sort_values('rating', ascending=False)
top_5 = user_ratings.head()
top_5
```

<!-- #region id="qK7RYCp5IyYf" -->
Their lowest rated movies:
<!-- #endregion -->

```python id="ygiaO33-IyYg" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="34149ee4-fe1c-41bd-a6dc-075db1eaf8aa"
bottom_5 = user_ratings[user_ratings['rating']<3].tail()
bottom_5
```

<!-- #region id="xxwj5F2jIyYg" -->
Based on their preferences above, we can get a sense that user 95 likes action and crime movies from the early 1990's over light-hearted American comedies from the early 2000's. Let's see what recommendations our model will generate for user 95.

We'll use the `recommend()` method, which takes in the user index of interest and transposed user-item matrix. 
<!-- #endregion -->

```python id="Id54l7ooIyYg" colab={"base_uri": "https://localhost:8080/"} outputId="19401cd3-f9b9-400a-f9f8-cdb0eb22384b"
X_t = X.T.tocsr()

user_idx = user_mapper[user_id]
recommendations = model.recommend(user_idx, X_t)
recommendations
```

<!-- #region id="DwlSw7xPIyYh" -->
We can't interpret the results as is since movies are represented by their index. We'll have to loop over the list of recommendations and get the movie title for each movie index. 
<!-- #endregion -->

```python id="zk__BZVMIyYh" colab={"base_uri": "https://localhost:8080/"} outputId="39577b39-27f0-41ca-8a8b-3428e7dd26ad"
for r in recommendations:
    recommended_title = get_movie_title(r[0])
    print(recommended_title)
```

<!-- #region id="AS62m5RrIyYj" -->
User 95's recommendations consist of action, crime, and thrillers. None of their recommendations are comedies. 
<!-- #endregion -->
