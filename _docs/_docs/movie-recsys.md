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

<!-- #region id="GLNC2HYiklWl" -->
# Movie Recommender
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="txUrd30jP6jX" executionInfo={"status": "ok", "timestamp": 1615180365604, "user_tz": -330, "elapsed": 13261, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1641bc32-370a-4d62-ab91-1d84b3e59802"
!pip install -q -U kaggle
!pip install --upgrade --force-reinstall --no-deps kaggle
!mkdir ~/.kaggle
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d rounakbanik/the-movies-dataset
```

```python colab={"base_uri": "https://localhost:8080/"} id="VxSAqBRmQErv" executionInfo={"status": "ok", "timestamp": 1615180391940, "user_tz": -330, "elapsed": 10309, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="47f339e4-5ebd-47d5-a3ad-134f53f2abf5"
!unzip the-movies-dataset.zip
```

```python colab={"base_uri": "https://localhost:8080/", "height": 700} id="OpWMotILQLah" executionInfo={"status": "ok", "timestamp": 1615184027598, "user_tz": -330, "elapsed": 3459, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1e98fb57-02ad-42c6-9813-42aa0926ecad"
import pandas as pd
import numpy as np

df = pd.read_csv('movies_metadata.csv')
df.head()
```

<!-- #region id="OXoVndvgQwa9" -->
## Simple Recommender

The choice of a metric is arbitrary. One of the simplest metrics that can be used is the movie rating. However, this suffers from a variety of disadvantages. In the first place, the movie rating does not take the popularity of a movie into consideration. Therefore, a movie rated 9 by 100,000 users will be placed below a movie rated 9.5 by 100 users.
This is not desirable as it is highly likely that a movie watched and rated only by 100 people caters to a very specific niche and may not appeal as much to the average person as the former.

It is also a well-known fact that as the number of voters increase, the rating of a movie normalizes and it approaches a value that is reflective of the movie's quality and popularity with the general populace. To put it another way, movies with very few ratings are not very reliable. A movie rated 10/10 by five users doesn't necessarily mean that it's a good movie.

Therefore, what we need is a metric that can, to an extent, take into account the movie rating and the number of votes it has garnered (a proxy for popularity). This would give a greater preference to a blockbuster movie rated 8 by 100,000 users over an art house movie rated 9 by 100 users.

Fortunately, we do not have to brainstorm a mathematical formula for the metric. As the title of this chapter states, we are building an IMDB top 250 clone. Therefore, we shall use IMDB's weighted rating formula as our metric. Mathematically, it can be represented as follows:
<!-- #endregion -->

<!-- #region id="AS0xSnq_Q-0Z" -->
<!-- #endregion -->

<!-- #region id="ytYw_CELRKa4" -->
For our recommender, we will use the number of votes garnered by the 80th percentile movie as our value for m. In other words, for a movie to be considered in the rankings, it must have garnered more votes than at least 80% of the movies present in our dataset. Additionally, the number of votes garnered by the 80th percentile movie is used in the weighted formula to come up with the value for the scores.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="WrP81j1eQebU" executionInfo={"status": "ok", "timestamp": 1615180690425, "user_tz": -330, "elapsed": 808, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="68c0a1e5-e5c8-406f-a4a8-cf4ae1a7c321"
#Calculate the number of votes garnered by the 80th percentile movie
m = df['vote_count'].quantile(0.80)
m
```

<!-- #region id="hyD5cpfNRgSj" -->
We can see that only 20% of the movies have gained more than 50 votes. Therefore, our value of m is 50.

Another prerequisite that we want in place is the runtime. We will only consider movies that are greater than 45 minutes and less than 300 minutes in length. Let us define a new DataFrame, q_movies, which will hold all the movies that qualify to appear in the chart:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="FMWkY7frRXBG" executionInfo={"status": "ok", "timestamp": 1615180752650, "user_tz": -330, "elapsed": 1096, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3be0fb96-0bda-4c5d-dcda-d06d8fb20170"
#Only consider movies longer than 45 minutes and shorter than 300 minutes
q_movies = df[(df['runtime'] >= 45) & (df['runtime'] <= 300)]

#Only consider movies that have garnered more than m votes
q_movies = q_movies[q_movies['vote_count'] >= m]

#Inspect the number of movies that made the cut
q_movies.shape
```

<!-- #region id="qN7-uUb1RoXx" -->
We see that from our dataset of 45,000 movies approximately 9,000 movies (or 20%) made the cut. 

Let's calculate C, the mean rating for all the movies in the dataset:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ViSSRxCTRmJJ" executionInfo={"status": "ok", "timestamp": 1615180827830, "user_tz": -330, "elapsed": 1393, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bdccda19-3e3f-4d9a-8f26-b119fdc96ec8"
# Calculate C
C = df['vote_average'].mean()
C
```

<!-- #region id="41D8-wYRSDsK" -->
We can see that the average rating of a movie is approximately 5.6/10. It seems that IMDB happens to be particularly strict with their ratings. Now that we have the value of C, we can go about calculating our score for each movie.

First, let us define a function that computes the rating for a movie, given its features and the values of m and C:
<!-- #endregion -->

```python id="gABUnu0kR4a-"
# Function to compute the IMDB weighted rating for each movie
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Compute the weighted score
    return (v/(v+m) * R) + (m/(m+v) * C)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 824} id="w3N9aNoaSJVm" executionInfo={"status": "ok", "timestamp": 1615180946859, "user_tz": -330, "elapsed": 1437, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="988c9b04-25f2-480a-8de4-e98012b5bbb8"
# Compute the score using the weighted_rating function defined above
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

#Sort movies in descending order of their scores
q_movies = q_movies.sort_values('score', ascending=False)

#Print the top 25 movies
q_movies[['title', 'vote_count', 'vote_average', 'score', 'runtime']].head(25)
```

<!-- #region id="UCTFtpQRScMh" -->
We can see that the Bollywood film Dilwale Dulhania Le Jayenge figures at the top of the list. We can also see that it has a noticeably smaller number of votes than the other Top 25 movies. This strongly suggests that we should probably explore a higher value of m.
<!-- #endregion -->

<!-- #region id="VyFD02cwSrRD" -->
## Knowledge-based Recommender

This will be a simple function that will perform the following tasks:
- Ask the user for the genres of movies he/she is looking for
- Ask the user for the duration
- Ask the user for the timeline of the movies recommended
- Using the information collected, recommend movies to the user that have a high weighted rating (according to the IMDB formula) and that satisfy the preceding conditions

The data that we have has information on the duration, genres, and timelines, but it isn't currently in a form that is directly usable. In other words, our data needs to be wrangled before it can be put to use to build this recommender.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="Hr2iLUwnSVeW" executionInfo={"status": "ok", "timestamp": 1615184034200, "user_tz": -330, "elapsed": 2162, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6007342d-25e8-443f-e32b-b36c7d164cae"
#Only keep those features that we require 
df = df[['title','genres', 'release_date', 'runtime', 'vote_average', 'vote_count']]

df.head()
```

<!-- #region id="xbYQ9hTjTJv1" -->
Next, let us extract the year of release from our release_date feature:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="9zpoADQWTI0X" executionInfo={"status": "ok", "timestamp": 1615184037421, "user_tz": -330, "elapsed": 3496, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b50e074e-0534-4416-bb9f-40a231941b5e"
#Convert release_date into pandas datetime format
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

#Extract year from the datetime
df['year'] = df['release_date'].apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

#Helper function to convert NaT to 0 and all other years to integers.
def convert_int(x):
    try:
        return int(x)
    except:
        return 0

#Apply convert_int to the year feature
df['year'] = df['year'].apply(convert_int)

#Drop the release_date column
df = df.drop('release_date', axis=1)

#Display the dataframe
df.head()
```

<!-- #region id="9RrttzUBT5YC" -->
Upon preliminary inspection, we can observe that the genres are in a format that looks like a JSON object (or a Python dictionary). Let us take a look at the genres object of one of our movies:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="iW6RLaGzTcLQ" executionInfo={"status": "ok", "timestamp": 1615181364989, "user_tz": -330, "elapsed": 1524, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9f2c7b0a-edef-46ef-e7e5-b7028ff4e75b"
#Print genres of the first movie
df.iloc[0]['genres']
```

<!-- #region id="Sw0oTk1qUIzm" -->
We can observe that the output is a stringified dictionary. In order for this feature to be usable, it is important that we convert this string into a native Python dictionary. Fortunately, Python gives us access to a function called literal_eval (available in the ast library) which does exactly that. literal_eval parses any string passed into it and converts it into its corresponding Python object.

Also, each dictionary represents a genre and has two keys: id and name. However, for this exercise (as well as all subsequent exercises), we only require the name. Therefore, we shall convert our list of dictionaries into a list of strings, where each string is a genre name:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="U2SEq-hOT7hs" executionInfo={"status": "ok", "timestamp": 1615184040946, "user_tz": -330, "elapsed": 2717, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="16a3d1a6-cb4e-40e3-aa12-3fe84f10ded9"
#Import the literal_eval function from ast
from ast import literal_eval

#Convert all NaN into stringified empty lists
df['genres'] = df['genres'].fillna('[]')

#Apply literal_eval to convert to the list object
df['genres'] = df['genres'].apply(literal_eval)

#Convert list of dictionaries to a list of strings
df['genres'] = df['genres'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

df.head()
```

<!-- #region id="E8QwjwmXUli-" -->
The last step is to explode the genres column. In other words, if a particular movie has multiple genres, we will create multiple copies of the movie, with each movie having one of the genres.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 258} id="uJI5rFFNUcp5" executionInfo={"status": "ok", "timestamp": 1615181571003, "user_tz": -330, "elapsed": 19230, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f4971372-1ca1-4936-8783-8a3bf57be271"
#Create a new feature by exploding genres
s = df.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)

#Name the new feature as 'genre'
s.name = 'genre'

#Create a new dataframe gen_df which by dropping the old 'genres' feature and adding the new 'genre'.
gen_df = df.drop('genres', axis=1).join(s)

#Print the head of the new gen_df
gen_df.head()
```

<!-- #region id="M5HuOPiiU6_z" -->
build_chart function
1. Get user input on their preferences
2. Extract all movies that match the conditions set by the user
3. Calculate the values of m and C for only these movies and proceed to build the chart
<!-- #endregion -->

```python id="Yxr73mN7UpgU"
def build_chart(gen_df, percentile=0.8):
    #Ask for preferred genres
    print("Input preferred genre")
    genre = input()
    
    #Ask for lower limit of duration
    print("Input shortest duration")
    low_time = int(input())
    
    #Ask for upper limit of duration
    print("Input longest duration")
    high_time = int(input())
    
    #Ask for lower limit of timeline
    print("Input earliest year")
    low_year = int(input())
    
    #Ask for upper limit of timeline
    print("Input latest year")
    high_year = int(input())
    
    #Define a new movies variable to store the preferred movies. Copy the contents of gen_df to movies
    movies = gen_df.copy()
    
    #Filter based on the condition
    movies = movies[(movies['genre'] == genre) & 
                    (movies['runtime'] >= low_time) & 
                    (movies['runtime'] <= high_time) & 
                    (movies['year'] >= low_year) & 
                    (movies['year'] <= high_year)]
    
    #Compute the values of C and m for the filtered movies
    C = movies['vote_average'].mean()
    m = movies['vote_count'].quantile(percentile)
    
    #Only consider movies that have higher than m votes. Save this in a new dataframe q_movies
    q_movies = movies.copy().loc[movies['vote_count'] >= m]
    
    #Calculate score using the IMDB formula
    q_movies['score'] = q_movies.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) 
                                       + (m/(m+x['vote_count']) * C)
                                       ,axis=1)

    #Sort movies in descending order of their scores
    q_movies = q_movies.sort_values('score', ascending=False)
    
    return q_movies
```

```python colab={"base_uri": "https://localhost:8080/", "height": 374} id="BlXFb-esVdWW" executionInfo={"status": "ok", "timestamp": 1615182062884, "user_tz": -330, "elapsed": 25098, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="83fde3a7-4420-4f14-a8fd-126c04d839fe"
#Generate the chart for top animation movies and display top 5.
build_chart(gen_df).head()
```

<!-- #region id="hSbFtc-UWuLz" -->
We can see that the movies that it outputs satisfy all the conditions we passed in as input. Since we applied IMDB's metric, we can also observe that our movies are very highly rated and popular at the same time.
<!-- #endregion -->

<!-- #region id="nOgpWczTYyP0" -->
## Content-based Recommender
<!-- #endregion -->

<!-- #region id="XO5_1g_ZYZ4C" -->
The simple recommender did not take into consideration an individual user's preferences. The knowledge-based recommender did take account of the user's preference for genres, timelines, and duration, but the model and its recommendations still remained very generic. Imagine that Alice likes the movies The Dark Knight, Iron Man, and Man of Steel. It is pretty evident that Alice has a taste for superhero movies. However, our models would not be able to capture this detail. The best it could do is suggest action movies (by making Alice input action as the preferred genre), which is a superset of superhero movies.

It is also possible that two movies have the same genre, timeline, and duration characteristics, but differ hugely in their audience. Consider The Hangover and Forgetting Sarah Marshall, for example. Both these movies were released in the first decade of the 21st century, both lasted around two hours, and both were comedies. However, the kind of audience that enjoyed these movies was very different.

We are going to build two types of content-based recommender:
1. Plot description-based recommender: This model compares the descriptions and taglines of different movies, and provides recommendations that have the most similar plot descriptions.
2. Metadata-based recommender: This model takes a host of features, such as genres, keywords, cast, and crew, into consideration and provides recommendations that are the most similar with respect to the aforementioned features.
<!-- #endregion -->

<!-- #region id="Ad_RgpyebIE5" -->
### Plot description-based recommender
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="ETEGVHJxVgye" executionInfo={"status": "ok", "timestamp": 1615185474482, "user_tz": -330, "elapsed": 2285, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="01bf97a3-e8d7-4b1d-b22b-5b04a3642c09"
#Import the original file
orig_df = pd.read_csv('movies_metadata.csv', low_memory=False)

#Add the useful features into the cleaned dataframe
df['overview'], df['id'] = orig_df['overview'], orig_df['id']

df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="6X_DTcQ9bUk6" executionInfo={"status": "ok", "timestamp": 1615185492775, "user_tz": -330, "elapsed": 3780, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0529e893-5390-4035-aa2f-8d28d378e93f"
#Import TfIdfVectorizer from the scikit-learn library
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stopwords
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
df['overview'] = df['overview'].fillna('')

#Construct the required TF-IDF matrix by applying the fit_transform method on the overview feature
tfidf_matrix = tfidf.fit_transform(df['overview'])

#Output the shape of tfidf_matrix
tfidf_matrix
```

<!-- #region id="POsQgYaqb9jp" -->
We see that the vectorizer has created a 75,827-dimensional vector for the overview of every movie.

The next step is to calculate the pairwise cosine similarity score of every movie. In other words, we are going to create a 45,466 × 45,466 matrix, where the cell in the ith row and jth column represents the similarity score between movies i and j. We can easily see that this matrix is symmetric in nature and every element in the diagonal is 1, since it is the similarity score of the movie with itself.
<!-- #endregion -->

```python id="AxfMSGT9b6oh"
# Import linear_kernel to compute the dot product
from sklearn.metrics.pairwise import linear_kernel
```

```python id="s9Win0W6cQMV"
#Construct a reverse mapping of indices and movie titles, and drop duplicate titles, if any
indices = pd.Series(df.index, index=df['title']).drop_duplicates()
```

```python id="qPVKFJree0ng"
# Function that takes in movie title as input and gives recommendations 
def content_recommender(title, tfidf_matrix=tfidf_matrix, df=df, indices=indices):
    # Obtain the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    # And convert it into a list of tuples as described above
    sim_scores = list(enumerate(linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()))

    # Sort the movies based on the cosine similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies. Ignore the first movie.
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df['title'].iloc[movie_indices]
```

```python colab={"base_uri": "https://localhost:8080/"} id="mTwVEbwZfvFM" executionInfo={"status": "ok", "timestamp": 1615184498614, "user_tz": -330, "elapsed": 1204, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ea31d037-ca6d-40d2-d265-9728c794fa44"
#Get recommendations for The Lion King
content_recommender('The Lion King')
```

<!-- #region id="Fp004B1vgHlp" -->
We see that our recommender has suggested all of The Lion King's sequels in its top-10 list. We also notice that most of the movies in the list have to do with lions.

It goes without saying that a person who loves The Lion King is very likely to have a thing for Disney movies. They may also prefer to watch animated movies. Unfortunately, our plot description recommender isn't able to capture all this information.

Therefore, in the next section, we will build a recommender that uses more advanced metadata, such as genres, cast, crew, and keywords (or sub-genres). This recommender will be able to do a much better job of identifying an individual's taste for a particular director, actor, sub-genre, and so on.
<!-- #endregion -->

<!-- #region id="qqxbOwNygMwg" -->
### Metadata-based recommender

To build this model, we will be using the following metdata:
- The genre of the movie. 
- The director of the movie. This person is part of the crew.
- The movie's three major stars. They are part of the cast.
- Sub-genres or keywords.
<!-- #endregion -->

```python id="NPxTvHHIf4pu"
# Load the keywords and credits files
cred_df = pd.read_csv('credits.csv')
key_df = pd.read_csv('keywords.csv')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="xworY12agg2I" executionInfo={"status": "ok", "timestamp": 1615185502919, "user_tz": -330, "elapsed": 2475, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a6f79693-b74c-486f-f9f5-3ff8a13ba0a9"
#Print the head of the credit dataframe
cred_df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="iPNMzp0egk7F" executionInfo={"status": "ok", "timestamp": 1615185503745, "user_tz": -330, "elapsed": 1701, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bbf4ab20-d1d0-4da5-8de1-2b79025cbd81"
#Print the head of the keywords dataframe
key_df.head()
```

<!-- #region id="37IKPlNrg1Fr" -->
We can see that the cast, crew, and the keywords are in the familiar list of dictionaries form. Just like genres, we have to reduce them to a string or a list of strings.

Before we do this, however, we will join the three DataFrames so that all our features are in a single DataFrame. Joining pandas DataFrames is identical to joining tables in SQL. The key we're going to use to join the DataFrames is the id feature. However, in order to use this, we first need to explicitly convert is listed as an ID. This is clearly bad data. 
<!-- #endregion -->

```python id="qW-AL0ZSgmfk"
#Convert the IDs of df into int
df['id'] = df['id'].astype('float').astype('int')
```

<!-- #region id="Gk1nwgfniK6C" -->
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 612} id="L0BOPHwug5qo" executionInfo={"status": "ok", "timestamp": 1615185515654, "user_tz": -330, "elapsed": 1435, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a21d3e94-073e-4980-ef9f-dfb616b1c6f6"
# Function to convert all non-integer IDs to NaN
def clean_ids(x):
    try:
        return int(x)
    except:
        return np.nan

#Clean the ids of df
df['id'] = df['id'].apply(clean_ids)

#Filter all rows that have a null ID
df = df[df['id'].notnull()]

# Convert IDs into integer
df['id'] = df['id'].astype('int')
key_df['id'] = key_df['id'].astype('int')
cred_df['id'] = cred_df['id'].astype('int')

# Merge keywords and credits into your main metadata dataframe
df = df.merge(cred_df, on='id')
df = df.merge(key_df, on='id')

#Display the head of df
df.head()
```

```python id="mJoIFJWLicsb"
# Convert the stringified objects into the native python objects
from ast import literal_eval

features = ['cast', 'crew', 'keywords']
for feature in features:
    df[feature] = df[feature].apply(literal_eval)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Ay_LiRQsirxO" executionInfo={"status": "ok", "timestamp": 1615185568670, "user_tz": -330, "elapsed": 38675, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e2f7786a-79cf-4308-8f17-988a612ba0f7"
#Print the first cast member of the first movie in df
df.iloc[0]['crew'][0]
```

```python colab={"base_uri": "https://localhost:8080/"} id="xI_DMHyLiuqG" executionInfo={"status": "ok", "timestamp": 1615185568672, "user_tz": -330, "elapsed": 35568, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8d4873d7-d526-485a-a82c-ca2ab577b0e6"
# Extract the director's name. If director is not listed, return NaN
def get_director(x):
    for crew_member in x:
        if crew_member['job'] == 'Director':
            return crew_member['name']
    return np.nan

#Define the new director feature
df['director'] = df['crew'].apply(get_director)

#Print the directors of the first five movies
df['director'].head()
```

<!-- #region id="qgFDCE43k1Ew" -->
Both keywords and cast are dictionary lists as well. And, in both cases, we need to extract the top three name attributes of each list. Therefore, we can write a single function to wrangle both these features. Also, just like keywords and cast, we will only consider the top three genres for every movie:
<!-- #endregion -->

```python id="5G_e9y06k16a"
# Returns the list top 3 elements or entire list; whichever is more.
def generate_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="yUdWZ6AMk4Nh" executionInfo={"status": "ok", "timestamp": 1615185831214, "user_tz": -330, "elapsed": 2104, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0499478b-64b5-4d94-c9d5-8cf93da0bd9f"
#Apply the generate_list function to cast and keywords
df['cast'] = df['cast'].apply(generate_list)
df['keywords'] = df['keywords'].apply(generate_list)

#Only consider a maximum of 3 genres
df['genres'] = df['genres'].apply(lambda x: x[:3])

# Print the new features of the first 5 movies along with title
df[['title', 'cast', 'director', 'keywords', 'genres']].head()
```

<!-- #region id="XayhoiKKlG-G" -->
In the subsequent steps, we are going to use a vectorizer to build document vectors. If two actors had the same first name (say, Ryan Reynolds and Ryan Gosling), the vectorizer will treat both Ryans as the same, although they are clearly different entities. This will impact the quality of the recommendations we receive. If a person likes Ryan Reynolds' movies, it doesn't imply that they like movies by all Ryans. 

Therefore, the last step is to strip the spaces between keywords, and actor and director names, and convert them all into lowercase. Therefore, the two Ryans in the preceding example will become ryangosling and ryanreynolds, and our vectorizer will now be able to distinguish between them:
<!-- #endregion -->

```python id="HU01ebJok9xP"
# Function to sanitize data to prevent ambiguity. It removes spaces and converts to lowercase
def sanitize(x):
    if isinstance(x, list):
        #Strip spaces and convert to lowercase
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
```

```python id="Abx4KawjlLo2"
#Apply the generate_list function to cast, keywords, director and genres
for feature in ['cast', 'director', 'genres', 'keywords']:
    df[feature] = df[feature].apply(sanitize)
```

<!-- #region id="6hqBSF-UlaPV" -->
In the plot description-based recommender, we worked with a single overview feature, which was a body of text. Therefore, we were able to apply our vectorizer directly.

However, this is not the case with our metadata-based recommender. We have four features to work with, of which three are lists and one is a string. What we need to do is create a soup that contains the actors, director, keywords, and genres. This way, we can feed this soup into our vectorizer and perform similar follow-up steps to before:
<!-- #endregion -->

```python id="JQtVIRQ3lNcY"
#Function that creates a soup out of the desired metadata
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="0JznVqnvlReA" executionInfo={"status": "ok", "timestamp": 1615185922097, "user_tz": -330, "elapsed": 2403, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bd52c2e8-bd37-4f06-d7c7-2295d4e9fac3"
# Create the new soup feature
df['soup'] = df.apply(create_soup, axis=1)

#Display the soup of the first movie
df.iloc[0]['soup']
```

<!-- #region id="-HgY8GKeltfz" -->
The next steps are almost identical to the corresponding steps from the previous section.

Instead of using TF-IDFVectorizer, we will be using CountVectorizer. This is because using TF-IDFVectorizer will accord less weight to actors and directors who have acted and directed in a relatively larger number of movies.

This is not desirable, as we do not want to penalize artists for directing or appearing in more movies:
<!-- #endregion -->

```python id="avhMnI9slT4F"
# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#Define a new CountVectorizer object and create vectors for the soup
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df['soup'])

#Import cosine_similarity function
from sklearn.metrics.pairwise import cosine_similarity

# Reset index of your df and construct reverse mapping again
df = df.reset_index()
indices2 = pd.Series(df.index, index=df['title'])
```

```python id="3_rl43IzmNpg"
# Function that takes in movie title as input and gives recommendations 
def content_recommender(title, matrix=tfidf_matrix, df=df, indices=indices):
    # Obtain the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    # And convert it into a list of tuples as described above
    sim_scores = list(enumerate(linear_kernel(matrix[idx], matrix).flatten()))

    # Sort the movies based on the cosine similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies. Ignore the first movie.
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df['title'].iloc[movie_indices]
```

```python colab={"base_uri": "https://localhost:8080/"} id="oPB3tyWtnAO9" executionInfo={"status": "ok", "timestamp": 1615186380693, "user_tz": -330, "elapsed": 1439, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1d150cd3-284f-4b94-9345-af31b395bbaa"
content_recommender('The Lion King', count_matrix, df, indices2)
```

<!-- #region id="AKZxqFjvqYvW" -->
## EDA

https://www.kaggle.com/rounakbanik/the-story-of-film
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 847} id="N9GbK0smshAd" executionInfo={"status": "ok", "timestamp": 1615187824827, "user_tz": -330, "elapsed": 1718, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="15f46021-23b4-44f0-c46a-055351b3ba4a"
df = pd.read_csv('movies_metadata.csv')
df.head(2).transpose()
```

<!-- #region id="850W_dIosUK6" -->
### **Features**

- **adult:** Indicates if the movie is X-Rated or Adult.
- **belongs_to_collection:** A stringified dictionary that gives information on the movie series the particular film belongs to.
- **budget:** The budget of the movie in dollars.
- **genres:** A stringified list of dictionaries that list out all the genres associated with the movie.
- **homepage:** The Official Homepage of the move.
- **id:** The ID of the move.
- **imdb_id:** The IMDB ID of the movie.
- **original_language:** The language in which the movie was originally shot in.
- **original_title:** The original title of the movie.
- **overview:** A brief blurb of the movie.
- **popularity:** The Popularity Score assigned by TMDB.
- **poster_path:** The URL of the poster image.
- **production_companies:** A stringified list of production companies involved with the making of the movie.
- **production_countries:** A stringified list of countries where the movie was shot/produced in.
- **release_date:** Theatrical Release Date of the movie.
- **revenue:** The total revenue of the movie in dollars.
- **runtime:** The runtime of the movie in minutes.
- **spoken_languages:** A stringified list of spoken languages in the film.
- **status:** The status of the movie (Released, To Be Released, Announced, etc.)
- **tagline:** The tagline of the movie.
- **title:** The Official Title of the movie.
- **video:** Indicates if there is a video present of the movie with TMDB.
- **vote_average:** The average rating of the movie.
- **vote_count:** The number of votes by users, as counted by TMDB.
<!-- #endregion -->

<!-- #region id="lq9zGh1Go-O0" -->
References
- [Kaggle](https://www.kaggle.com/rounakbanik/the-movies-dataset)
- [eBook](https://learning.oreilly.com/library/view/hands-on-recommendation-systems/9781788993753/)
<!-- #endregion -->

```python id="2GOtKWhXnEFQ"

```
