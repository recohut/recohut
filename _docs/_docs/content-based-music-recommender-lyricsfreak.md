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

<!-- #region id="zzkmBup61MsP" -->
# Content-based method for song recommendation
> Applying TF-IDF on song lyrics to find the similar songs for users using LyricsFreak dataset

- toc: true
- badges: true
- comments: true
- categories: [Songs, ContentBased, TFIDF]
- author: "<a href='https://github.com/ugis22/music_recommender'>Eugenia Inzaugarat</a>"
- image:
<!-- #endregion -->

<!-- #region id="Wfo_aABo1KfQ" -->
One of the most used machine learning algorithms is recommendation systems. A **recommender** (or recommendation) **system** (or engine) is a filtering system which aim is to predict a rating or preference a user would give to an item, eg. a film, a product, a song, etc.

Which type of recommender can we have?   

There are two main types of recommender systems: 
- Content-based filters
- Collaborative filters
  
> Content-based filters predicts what a user likes based on what that particular user has liked in the past. On the other hand, collaborative-based filters predict what a user like based on what other users, that are similar to that particular user, have liked.

### Content-based filters

Recommendations done using content-based recommenders can be seen as a user-specific classification problem. This classifier learns the user's likes and dislikes from the features of the song.

The most straightforward approach is **keyword matching**.

In a few words, the idea behind is to extract meaningful keywords present in a song description a user likes, search for the keywords in other song descriptions to estimate similarities among them, and based on that, recommend those songs to the user.

*How is this performed?*

In our case, because we are working with text and words, **Term Frequency-Inverse Document Frequency (TF-IDF)** can be used for this matching process.
  
We'll go through the steps for generating a **content-based** music recommender system.
<!-- #endregion -->

<!-- #region id="WhK3oBjx1KfS" -->
### Importing required libraries

First, we'll import all the required libraries.
<!-- #endregion -->

```python id="8aasRkrD1KfT"
import numpy as np
import pandas as pd
```

```python id="xSwQYacy1KfU"
from typing import List, Dict
```

<!-- #region id="JnG7R4ry1KfV" -->
We are going to use TfidfVectorizer from the Scikit-learn package again.
<!-- #endregion -->

```python id="tQ7Pvp6E1KfW"
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```

<!-- #region id="1AChOTis1KfW" -->
### Dataset
<!-- #endregion -->

<!-- #region id="9TdYc2It1KfX" -->
This dataset contains name, artist, and lyrics for *57650 songs in English*. The data has been acquired from LyricsFreak through scraping.
<!-- #endregion -->

```python id="b7HzUVCD1KfY"
songs = pd.read_parquet('https://github.com/recohut/reco-data/raw/master/lyricsfreak/v1/items.parquet.gzip')
```

```python id="KNKl8Anh1KfY" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="8fb346e9-f127-4177-f32c-a2d147ff52e4"
songs.head()
```

<!-- #region id="5nPuwy6u1KfZ" -->
Because of the dataset being so big, we are going to resample only 5000 random songs.
<!-- #endregion -->

```python id="iJr37an61Kfa"
songs = songs.sample(n=5000).drop('link', axis=1).reset_index(drop=True)
```

<!-- #region id="URqgskOa1Kfa" -->
We can notice also the presence of `\n` in the text, so we are going to remove it.
<!-- #endregion -->

```python id="gSjdBiQ81Kfb"
songs['text'] = songs['text'].str.replace(r'\n', '')
```

<!-- #region id="Gm5Krbr22MQ5" -->
## TF-IDF
<!-- #endregion -->

<!-- #region id="_-zKDYQg1Kfb" -->
After that, we use TF-IDF vectorizerthat calculates the TF-IDF score for each song lyric, word-by-word. 

Here, we pay particular attention to the arguments we can specify.
<!-- #endregion -->

```python id="rMllj_Hs1Kfb"
tfidf = TfidfVectorizer(analyzer='word', stop_words='english')
```

```python id="Sq18HIqg1Kfc"
lyrics_matrix = tfidf.fit_transform(songs['text'])
```

<!-- #region id="2nRSButX2Pc8" -->
## Cosine similarity
<!-- #endregion -->

<!-- #region id="NPpLyZCr1Kfc" -->
*How do we use this matrix for a recommendation?* 

We now need to calculate the similarity of one lyric to another. We are going to use **cosine similarity**.

We want to calculate the cosine similarity of each item with every other item in the dataset. So we just pass the lyrics_matrix as argument.
<!-- #endregion -->

```python id="a8qq0sO91Kfd"
cosine_similarities = cosine_similarity(lyrics_matrix) 
```

<!-- #region id="ltc6cp6B1Kfd" -->
Once we get the similarities, we'll store in a dictionary the names of the 50  most similar songs for each song in our dataset.
<!-- #endregion -->

```python id="4iNcSgYV1Kfd"
similarities = {}
```

```python id="TiBFLfiV1Kfe"
for i in range(len(cosine_similarities)):
    # Now we'll sort each element in cosine_similarities and get the indexes of the songs. 
    similar_indices = cosine_similarities[i].argsort()[:-50:-1] 
    # After that, we'll store in similarities each name of the 50 most similar songs.
    # Except the first one that is the same song.
    similarities[songs['song'].iloc[i]] = [(cosine_similarities[i][x], songs['song'][x], songs['artist'][x]) for x in similar_indices][1:]
```

<!-- #region id="eNG2TV_k2SAw" -->
## Model
<!-- #endregion -->

<!-- #region id="t20PLqAG1Kfe" -->
After that, all the magic happens. We can use that similarity scores to access the most similar items and give a recommendation.

For that, we'll define our Content based recommender class.
<!-- #endregion -->

```python id="4h-qtqQA1Kff"
class ContentBasedRecommender:
    def __init__(self, matrix):
        self.matrix_similar = matrix

    def _print_message(self, song, recom_song):
        rec_items = len(recom_song)
        
        print(f'The {rec_items} recommended songs for {song} are:')
        for i in range(rec_items):
            print(f"Number {i+1}:")
            print(f"{recom_song[i][1]} by {recom_song[i][2]} with {round(recom_song[i][0], 3)} similarity score") 
            print("--------------------")
        
    def recommend(self, recommendation):
        # Get song to find recommendations for
        song = recommendation['song']
        # Get number of songs to recommend
        number_songs = recommendation['number_songs']
        # Get the number of songs most similars from matrix similarities
        recom_song = self.matrix_similar[song][:number_songs]
        # print each item
        self._print_message(song=song, recom_song=recom_song)
```

<!-- #region id="Xe0etWnS1Kfg" -->
Now, instantiate class
<!-- #endregion -->

```python id="PL-Y-yNJ1Kfh"
recommedations = ContentBasedRecommender(similarities)
```

<!-- #region id="9SYt7rep2UcE" -->
## Recommendations
<!-- #endregion -->

<!-- #region id="sStjSUZU1Kfh" -->
Then, we are ready to pick a song from the dataset and make a recommendation.
<!-- #endregion -->

```python id="-P7S5lMq1Kfh"
recommendation = {
    "song": songs['song'].iloc[10],
    "number_songs": 4 
}
```

```python id="52w0WqHj1Kfi" colab={"base_uri": "https://localhost:8080/"} outputId="ddca0d19-5c0c-4c7b-cb74-64beff9dadd7"
recommedations.recommend(recommendation)
```

<!-- #region id="dbaQl3Af1Kfi" -->
And we can pick another random song and recommend again:
<!-- #endregion -->

```python id="Parq1nXP1Kfi"
recommendation2 = {
    "song": songs['song'].iloc[120],
    "number_songs": 4 
}
```

```python id="qdz2vzhH1Kfj" colab={"base_uri": "https://localhost:8080/"} outputId="92cbdc71-b592-4a26-f4ad-abe0ee36e683"
recommedations.recommend(recommendation2)
```
