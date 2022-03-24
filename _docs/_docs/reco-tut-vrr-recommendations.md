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

```python id="QHfJAkdQk3jv"
import os
project_name = "reco-tut-vrr"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python colab={"base_uri": "https://localhost:8080/"} id="3APW7y3clBlH" outputId="17900773-0b16-41fd-c1d3-57d20c5a748b"
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

```python id="yGo_LJd6lBlN"
!git status
```

```python colab={"base_uri": "https://localhost:8080/"} id="SWzR_TDJlBlO" outputId="1f0aa5a6-f8e0-4bd1-8b83-cea3865fa6dc"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

<!-- #region id="ra084G6QmDcL" -->
---
<!-- #endregion -->

```python id="Nrb7EyBKmEmk"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import linear_kernel
```

```python colab={"base_uri": "https://localhost:8080/"} id="ce393Lpposh_" outputId="2db10727-181b-4a94-bc29-e6d5422a74cf"
nltk.download('wordnet')
nltk.download('stopwords')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 289} id="HF4J7q62mJ7u" outputId="deeca935-07f5-4b92-9835-da83411264d5"
df = pd.read_parquet('./data/silver/reviews.parquet.gzip')
df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="knyLYsCEmT0d" outputId="d600bbe4-a1f2-4401-8f92-ede0b17210db"
df.info()
```

<!-- #region id="n2-h8eFMyf1d" -->
## Similar Rental Recommender based on ratings
<!-- #endregion -->

<!-- #region id="PNJ3ReFlt54U" -->
Data Vectorization and Cosine Similarity Construction
<!-- #endregion -->

```python id="7z_RJguQt48z"
tfidf = TfidfVectorizer(stop_words='english')   

#transforming Rentals Reviews
tfidf_matrix = tfidf.fit_transform(df['Reviews']) 

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#Construct a reverse map of indices and Rentals Titles
indices = pd.Series(df.index, index=df['Title']).drop_duplicates()
```

<!-- #region id="PlkJYDYJt9B4" -->
Function that takes in Rental title as input and outputs most similar Rentals
<!-- #endregion -->

```python id="R1C3sljlvL5d"
outstr = '{} in {} city of {} province with {} reviews and an average rating of {}\n{}'
def outformat(_x):
    print(outstr.format(_x.Title, _x.City, _x.Province, _x.Rating, _x.Review_counts, '='*100))
```

```python id="NNZn-IG6trGf"
def get_recommendations(title, cosine_sim=cosine_sim, k=10):
    idx = indices[title]                            # Get the index of the Review that matches the title
    sim_scores = list(enumerate(cosine_sim[idx]))       # Get the pairwsie similarity scores of all Reviews with that dataframe
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)       # Sort the Reviews based on the similarity scores
    sim_scores = sim_scores[1:k+1]                               # Get the scores of the 3 most similar blog
    df_indices = [i[0] for i in sim_scores]  
    print('Top {} recommendations:\n'.format(k))
    [outformat(df.iloc[_x]) for _x in df_indices]
```

```python colab={"base_uri": "https://localhost:8080/"} id="4dYA0qByuC1-" outputId="dd8900a0-e978-45b3-fb12-1317768abfb1"
get_recommendations("Mistiso's Place Vacation Rentals- Purcell Suite", k=5)
```

<!-- #region id="lUvzFG_-zFmz" -->
## Similar Rental Recommender based on reviews
<!-- #endregion -->

<!-- #region id="DDHefTLbykNK" -->
Text Preprocessing
<!-- #endregion -->

```python id="_UVZlNTCyme5"
df['Reviews']= df['Reviews'].astype(str)
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
```

```python id="eOlu5_exyo-w"
def clean_text(text):
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    return text
```

```python id="Ay2_n2tayq09"
df['reviews_clean'] = df['Reviews'].apply(clean_text)
```

<!-- #region id="WPaQblNyyr-m" -->
Modeling
<!-- #endregion -->

```python id="RgSEEByHyuR2"
df.set_index('Title', inplace = True)
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(df['reviews_clean'])
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index)
```

```python id="gunRT-Veyyck"
def recommendations(Title, cosine_similarities = cosine_similarities):  
    recommended_rentals = []
    
    # getting the index of the hotel that matches the name
    idx = indices[indices == Title].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending = False)

    # getting the indexes of the 10 most similar hotels except itself
    top_10_indexes = list(score_series.iloc[1:11].index)
    
    # populating the list with the names of the top 10 matching hotels
    for i in top_10_indexes:
        recommended_rentals.append(list(df.index)[i])
        
    return recommended_rentals
```

<!-- #region id="aUQV5eISy1Rm" -->
Recommendations
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="M2BDpjOYvrrG" outputId="b9ec66db-23cf-4282-c8c5-1ded5a396b97"
recommendations("Mistiso's Place Vacation Rentals- Purcell Suite")
```
