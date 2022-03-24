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

<!-- #region id="o29AE6V8YBde" colab_type="text" -->
## Knowledge-based Simple IMDB Recommendation System
<!-- #endregion -->

```python id="IkM433KiIv6H" colab_type="code" colab={}
FILE_PATH = '/content/drive/My Drive/Recommendation/movies_metadata.csv'
```

```python id="AdL8Ok2taVDa" colab_type="code" colab={}
import numpy as np
import pandas as pd
from ast import literal_eval
```

```python id="cJ3KBDFQbjJO" colab_type="code" outputId="47c5eb47-1349-47e6-8c85-d3a8db6fe850" executionInfo={"status": "ok", "timestamp": 1586632813124, "user_tz": -330, "elapsed": 2437, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} colab={"base_uri": "https://localhost:8080/", "height": 793}
df = pd.read_csv(FILE_PATH, low_memory=False)

df.head(3).T
```

```python id="HJq8qhjIbmn2" colab_type="code" outputId="7bd689fd-c970-4d83-fc75-fc21b4b92de1" executionInfo={"status": "ok", "timestamp": 1586632814758, "user_tz": -330, "elapsed": 3963, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} colab={"base_uri": "https://localhost:8080/", "height": 204}
#Only keep those features that we require 
df = df[['title','genres', 'release_date', 'runtime', 'vote_average', 'vote_count']]

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

#Convert all NaN into stringified empty lists
df['genres'] = df['genres'].fillna('[]')

#Apply literal_eval to convert stringified empty lists to the list object
df['genres'] = df['genres'].apply(literal_eval)

#Convert list of dictionaries to a list of strings
df['genres'] = df['genres'].apply(lambda x: [i['name'].lower() for i in x] if isinstance(x, list) else [])

df.head()
```

```python id="k06ItDL1fOQy" colab_type="code" outputId="99f8f88c-7f91-4502-813f-598de951028a" executionInfo={"status": "ok", "timestamp": 1586632836807, "user_tz": -330, "elapsed": 25885, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} colab={"base_uri": "https://localhost:8080/", "height": 258}
#Create a new feature by exploding genres
s = df.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)

#Name the new feature as 'genre'
s.name = 'genre'

#Create a new dataframe gen_df which by dropping the old 'genres' feature and adding the new 'genre'.
gen_df = df.drop('genres', axis=1).join(s)

#Print the head of the new gen_df
gen_df.head()
```

```python id="Uh7Fc2SOftjS" colab_type="code" outputId="50c4fb21-7964-418b-f393-9cf4c80e5ae0" executionInfo={"status": "ok", "timestamp": 1586632389113, "user_tz": -330, "elapsed": 22680, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} colab={"base_uri": "https://localhost:8080/", "height": 136}
gen_df.dtypes
```

```python id="cNHH8E0WfYva" colab_type="code" colab={}
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

```python id="i-ytAQCvfZER" colab_type="code" outputId="dfd95478-5eaa-4a8f-eab9-1b932f9722db" executionInfo={"status": "ok", "timestamp": 1586631289953, "user_tz": -330, "elapsed": 25414, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} colab={"base_uri": "https://localhost:8080/", "height": 374}
#Generate the chart for top animation movies and display top 5.
build_chart(gen_df).head()
```

<!-- #region id="kBJJSNmshHCd" colab_type="text" -->
## Content-based Recommendation System
<!-- #endregion -->

<!-- #region id="a3BwssPthKWX" colab_type="text" -->
We built an IMDB Top 250 clone (a type of simple recommender) and a knowledge-based recommender that suggested movies based on timeline, genre, and duration. However, these systems were extremely primitive. The simple recommender did not take into consideration an individual user's preferences. The knowledge-based recommender did take account of the user's preference for genres, timelines, and duration, but the model and its recommendations still remained very generic.
<!-- #endregion -->

<!-- #region id="nSA2mJpaiD1u" colab_type="text" -->
#### Now, we are going to build two types of content-based recommender:

- Plot description-based recommender: This model compares the descriptions and taglines of different movies, and provides recommendations that have the most similar plot descriptions.
- Metadata-based recommender: This model takes a host of features, such as genres, keywords, cast, and crew, into consideration and provides recommendations that are the most similar with respect to the aforementioned features

<!-- #endregion -->

<!-- #region id="cJgwDassikNu" colab_type="text" -->
### Plot description based recommender

Our plot description-based recommender will take in a movie title as an argument and output a list of movies that are most similar based on their plots. These are the steps we are going to perform in building this model:

* Obtain the data required to build the model
* Create TF-IDF vectors for the plot description (or overview) of every movie
* Compute the pairwise cosine similarity score of every movie
* Write the recommender function that takes in a movie title as an argument and outputs movies most similar to it based on the plot


<!-- #endregion -->

```python id="DYljaloUfao3" colab_type="code" outputId="b98e48b5-ca87-46c5-e2d8-4fb5f95358fe" executionInfo={"status": "ok", "timestamp": 1586632837241, "user_tz": -330, "elapsed": 16931, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} colab={"base_uri": "https://localhost:8080/", "height": 289}
#Import the original file
orig_df = pd.read_csv(FILE_PATH, low_memory=False)

#Add the useful features into the cleaned dataframe
df['overview'], df['id'] = orig_df['overview'], orig_df['id']

df.head()
```

```python id="78rmAUVUjTCk" colab_type="code" colab={}
#Import TfIdfVectorizer from the scikit-learn library
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stopwords
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
df['overview'] = df['overview'].fillna('')

#Construct the required TF-IDF matrix by applying the fit_transform method on the overview feature
tfidf_matrix = tfidf.fit_transform(df['overview'])
```

```python id="7HwOXFORrLt-" colab_type="code" colab={}
# # Import linear_kernel to compute the dot product
# from sklearn.metrics.pairwise import linear_kernel

# # Compute the cosine similarity matrix
# cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# #Construct a reverse mapping of indices and movie titles, and drop duplicate titles, if any
# indices = pd.Series(df.index, index=df['title']).drop_duplicates()
```

```python id="rJx3chjZlgaA" colab_type="code" colab={}
# Function that takes in movie title as input and gives recommendations 
def content_recommender(title, cosine_sim=cosine_sim, df=df, indices=indices):
    # Obtain the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    # And convert it into a list of tuples as described above
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the cosine similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies. Ignore the first movie.
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df['title'].iloc[movie_indices]
```

```python id="mT1RLJ8dlsMO" colab_type="code" outputId="3ffa2eab-fd9c-4879-c110-fd99876d6e63" executionInfo={"status": "ok", "timestamp": 1586631326039, "user_tz": -330, "elapsed": 27273, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} colab={"base_uri": "https://localhost:8080/", "height": 204}
#Get recommendations for The Lion King
content_recommender('The Lion King')
```

<!-- #region id="17C4wWQwlxOM" colab_type="text" -->
### Metadata Based Recommender
To build this model, we will be using the following metdata:

* The genre of the movie.
* The director of the movie. This person is part of the crew.
* The movie's three major stars. They are part of the cast.
* Sub-genres or keywords.


With the exception of genres, our DataFrames (both original and cleaned) do not contain the data that we require. Therefore, for this exercise, we will need to download two additional files: credits.csv, which contains information on the cast and crew of the movies, and keywords.csv, which contains information on the sub-genres

<!-- #endregion -->

```python id="qxh9fW1gluq1" colab_type="code" colab={}
# !wget -x --load-cookies cookies.txt "https://www.kaggle.com/rounakbanik/the-movies-dataset/download/zPeWVcFHd4FVILkEkFTP%2Fversions%2FUr5p8EUb593KaWc1xRXm%2Ffiles%2Fcredits.csv?datasetVersionNumber=7" -O credits.zip
# !unzip credits.zip

# !wget -x --load-cookies cookies.txt "https://www.kaggle.com/rounakbanik/the-movies-dataset/download/zPeWVcFHd4FVILkEkFTP%2Fversions%2FUr5p8EUb593KaWc1xRXm%2Ffiles%2Fkeywords.csv?datasetVersionNumber=7" -O keywords.zip
# !unzip keywords.zip
```

```python id="xMbOmvUuoUxp" colab_type="code" colab={}
# Load the keywords and credits files
cred_df = pd.read_csv('credits.csv')
key_df = pd.read_csv('keywords.csv')
```

```python id="Nakyr8E8pBC8" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="2856bf80-021e-4e72-9138-4bed8a131312" executionInfo={"status": "ok", "timestamp": 1586632845993, "user_tz": -330, "elapsed": 5818, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
#Print the head of the credit dataframe
cred_df.head()
```

```python id="XijMiBHtpEil" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="c44fe9ee-8bf0-46cc-aaea-65978337e818" executionInfo={"status": "ok", "timestamp": 1586632845993, "user_tz": -330, "elapsed": 5808, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
#Print the head of the keywords dataframe
key_df.head()
```

```python id="8GjdEtHSpFjm" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 666} outputId="9bbc526a-472a-4fa4-b030-388199db0988" executionInfo={"status": "ok", "timestamp": 1586632845994, "user_tz": -330, "elapsed": 5799, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
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

```python id="Q5-J1by3pKp2" colab_type="code" colab={}
# Convert the stringified objects into the native python objects
from ast import literal_eval

features = ['cast', 'crew', 'keywords']
for feature in features:
    df[feature] = df[feature].apply(literal_eval)
```

```python id="P2JyookcpXY_" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 289} outputId="10bf7f3a-a2c8-41b2-a48a-15b66d6748d0" executionInfo={"status": "ok", "timestamp": 1586632880248, "user_tz": -330, "elapsed": 40023, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Extract the director's name. If director is not listed, return NaN
def get_director(x):
    for crew_member in x:
        if crew_member['job'] == 'Director':
            return crew_member['name']
    return np.nan

#Define the new director feature
df['director'] = df['crew'].apply(get_director)

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

#Apply the generate_list function to cast and keywords
df['cast'] = df['cast'].apply(generate_list)
df['keywords'] = df['keywords'].apply(generate_list)

#Only consider a maximum of 3 genres
df['genres'] = df['genres'].apply(lambda x: x[:3])

# Print the new features of the first 5 movies along with title
df[['title', 'cast', 'director', 'keywords', 'genres']].head()
```

```python id="pbWDYx_KqidI" colab_type="code" colab={}
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

#Apply the generate_list function to cast, keywords, director and genres
for feature in ['cast', 'director', 'genres', 'keywords']:
    df[feature] = df[feature].apply(sanitize)
```

```python id="NxktAdfWqxNy" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 34} outputId="17701dbb-3752-4681-a030-7fda7f0ddf33" executionInfo={"status": "ok", "timestamp": 1586632883974, "user_tz": -330, "elapsed": 43714, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
#Function that creates a soup out of the desired metadata
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
  
# Create the new soup feature
df['soup'] = df.apply(create_soup, axis=1)

#Display the soup of the first movie
df.iloc[0]['soup']
```

```python id="bQz78lxKq1DA" colab_type="code" colab={}
# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#Define a new CountVectorizer object and create vectors for the soup
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df['soup'])
```

```python id="fnn0Y-VarAmy" colab_type="code" colab={}
#Import cosine_similarity function
from sklearn.metrics.pairwise import cosine_similarity

#Compute the cosine similarity score (equivalent to dot product for tf-idf vectors)
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
```

```python id="lsqSR_kZrGJp" colab_type="code" colab={}
# Reset index of your df and construct reverse mapping again
df = df.reset_index()
indices2 = pd.Series(df.index, index=df['title'])
```

```python id="nicWB5LjreAY" colab_type="code" colab={}
content_recommender('The Lion King', cosine_sim2, df, indices2)
```

<!-- #region id="MJmJoMFVt1Ix" colab_type="text" -->
## Collaborative Filtering Recommendation System
<!-- #endregion -->

```python id="xbXqyA6Ev2tf" colab_type="code" colab={}
BASE_PATH = '/content/drive/My Drive/Recommendation'
import os
import pandas as pd
import numpy as np
```

```python id="aoDD-SaaxvS8" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="500df6cf-25c9-4894-9448-f597d5a64302" executionInfo={"status": "ok", "timestamp": 1586634516369, "user_tz": -330, "elapsed": 615, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
#Load the u.user file into a dataframe
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']

users = pd.read_csv(os.path.join(BASE_PATH,'u.user'), sep='|', 
                    names=u_cols, encoding='latin-1')

users.head()
```

```python id="xGe-zXvRyLK_" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="31e4a934-e6de-4ed7-a75e-f121a3c78f2d" executionInfo={"status": "ok", "timestamp": 1586634573592, "user_tz": -330, "elapsed": 2337, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
#Load the u.item file into a dataframe
i_cols = ['movie_id', 'title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

movies = pd.read_csv(os.path.join(BASE_PATH,'u.item'), sep='|', names=i_cols, encoding='latin-1')

#Remove all information except Movie ID and title
movies = movies[['movie_id', 'title']]

movies.head()
```

```python id="iTatgPcnyc8V" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="06e45510-a23a-4ddf-8253-9e85906d21f1" executionInfo={"status": "ok", "timestamp": 1586634593664, "user_tz": -330, "elapsed": 1329, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
#Load the u.data file into a dataframe
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']

ratings = pd.read_csv(os.path.join(BASE_PATH,'u.data'), sep='\t', 
                      names=r_cols, encoding='latin-1')

ratings.head()
```

```python id="popxI3T-yo_8" colab_type="code" colab={}
#Drop the timestamp column
ratings = ratings.drop('timestamp', axis=1)
```

```python id="fZjd5Em3yt0J" colab_type="code" colab={}
#Import the train_test_split function
from sklearn.model_selection import train_test_split

#Assign X as the original ratings dataframe and y as the user_id column of ratings.
X = ratings.copy()
y = ratings['user_id']

#Split into training and test datasets, stratified along user_id
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, stratify=y, random_state=42)
```

```python id="T9rbIlACyvSC" colab_type="code" colab={}
#Import the mean_squared_error function
from sklearn.metrics import mean_squared_error

#Function that computes the root mean squared error (or RMSE)
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
```

```python id="-pfOLAgdywTP" colab_type="code" colab={}
#Define the baseline model to always return 3.
def baseline(user_id, movie_id):
    return 3.0
```

```python id="9j343DVLyxnL" colab_type="code" colab={}
#Function to compute the RMSE score obtained on the testing set by a model
def score(cf_model):
    
    #Construct a list of user-movie tuples from the testing dataset
    id_pairs = zip(X_test['user_id'], X_test['movie_id'])
    
    #Predict the rating for every user-movie tuple
    y_pred = np.array([cf_model(user, movie) for (user, movie) in id_pairs])
    
    #Extract the actual ratings given by the users in the test data
    y_true = np.array(X_test['rating'])
    
    #Return the final RMSE score
    return rmse(y_true, y_pred)
```

```python id="Je_uu75SyzRC" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 34} outputId="52ba7066-f6ff-4f6c-eaa1-277f48e4cab6" executionInfo={"status": "ok", "timestamp": 1586634641585, "user_tz": -330, "elapsed": 2202, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
score(baseline)
```

```python id="0xsIxcD8zHKY" colab_type="code" colab={}
# !pip install surprise
```

```python id="45XBkdOCy0e7" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 510} outputId="92ed11d3-0c62-4833-9f7b-e32fe8a4d3f5" executionInfo={"status": "ok", "timestamp": 1586634973168, "user_tz": -330, "elapsed": 20342, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
#Import the required classes and methods from the surprise library
from surprise import Reader, Dataset, KNNBasic
from surprise.model_selection import cross_validate

#Define a Reader object
#The Reader object helps in parsing the file or dataframe containing ratings
reader = Reader()

#Create the dataset to be used for building the filter
data = Dataset.load_from_df(ratings, reader)

#Define the algorithm object; in this case kNN
knn = KNNBasic()

#Evaluate the performance in terms of RMSE
cross_validate(knn, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
```

```python id="N0eTD24IzF1f" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 340} outputId="f4e3dcbf-5947-467c-e930-81a6b4b7ec8b" executionInfo={"status": "ok", "timestamp": 1586635012177, "user_tz": -330, "elapsed": 27406, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
#Import SVD
from surprise import SVD

#Define the SVD algorithm object
svd = SVD()

#Evaluate the performance in terms of RMSE
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
```

<!-- #region id="hFuyL-Ll2c_h" colab_type="text" -->
## Hybrid Recommendation System
<!-- #endregion -->

<!-- #region id="BPBmavP82feW" colab_type="text" -->
In other words, the workflow of our hybrid model will be as follows:

* Take in a movie title and user as input
* Use a content-based model to compute the 25 most similar movies
* Compute the predicted ratings that the user might give these 25 movies using a collaborative filter
* Return the top 10 movies with the highest predicted rating


<!-- #endregion -->

```python id="fjLM3V8X3iqq" colab_type="code" colab={}
#Import or compute the cosine sim mapping matrix
cosine_sim_map = pd.read_csv('../data/cosine_sim_map.csv', header=None)

#Convert cosine_sim_map into a Pandas Series
cosine_sim_map = cosine_sim_map.set_index(0)
cosine_sim_map = cosine_sim_map[1]
```

```python id="NPS6LwOj3zbw" colab_type="code" colab={}
#Build the SVD based Collaborative filter
from surprise import SVD, Reader, Dataset

reader = Reader()
ratings = pd.read_csv('../data/ratings_small.csv')
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
data.split(n_folds=5)
svd = SVD()
trainset = data.build_full_trainset()
svd.train(trainset)
```

```python id="WJLLuGaQ30Dh" colab_type="code" colab={}
#Build title to ID and ID to title mappings
id_map = pd.read_csv('../data/movie_ids.csv')
id_to_title = id_map.set_index('id')
title_to_id = id_map.set_index('title')
```

```python id="ZVdzC4j730-A" colab_type="code" colab={}
#Import or compute relevant metadata of the movies
smd = pd.read_csv('../data/metadata_small.csv')
```

```python id="olhzEnMN30rN" colab_type="code" colab={}
def hybrid(userId, title):
    #Extract the cosine_sim index of the movie
    idx = cosine_sim_map[title]
    
    #Extract the TMDB ID of the movie
    tmdbId = title_to_id.loc[title]['id']
    
    #Extract the movie ID internally assigned by the dataset
    movie_id = title_to_id.loc[title]['movieId']
    
    #Extract the similarity scores and their corresponding index for every movie from the cosine_sim matrix
    sim_scores = list(enumerate(cosine_sim[str(int(idx))]))
    
    #Sort the (index, score) tuples in decreasing order of similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    #Select the top 25 tuples, excluding the first 
    #(as it is the similarity score of the movie with itself)
    sim_scores = sim_scores[1:26]
    
    #Store the cosine_sim indices of the top 25 movies in a list
    movie_indices = [i[0] for i in sim_scores]

    #Extract the metadata of the aforementioned movies
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
    
    #Compute the predicted ratings using the SVD filter
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, id_to_title.loc[x]['movieId']).est)
    
    #Sort the movies in decreasing order of predicted rating
    movies = movies.sort_values('est', ascending=False)
    
    #Return the top 10 movies as recommendations
    return movies.head(10)
```

```python id="Qc9A82Xp30qC" colab_type="code" colab={}
hybrid(1, 'Avatar')
```
