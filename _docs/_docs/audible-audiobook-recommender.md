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

<!-- #region id="9kwmww3PqNY7" -->
# Audible Book Recommender
> Finding similar books using simple text countvectorizer model on audible dataset

- toc: false
- badges: true
- comments: true
- categories: [Books, CountVectorizer]
- image:
<!-- #endregion -->

```python id="3wQ4xg6-mJqQ"
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```

```python colab={"base_uri": "https://localhost:8080/", "height": 661} id="YU2R9RnOmLc8" outputId="15d4c7ba-57b8-446a-8d07-1a5f4c69b0cd"
audible_data = pd.read_csv("https://github.com/sparsh-ai/reco-data/raw/audible/audible/audible.csv",
                           encoding='latin1')
audible_data.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="dh_0Gvc9megx" outputId="7a6c2b8e-0213-46e6-983b-ff40bc284493"
audible_data.info()
```

```python id="MlU2jIzlmnTs"
# Selecting 4 columns: Title, Author, Narrator,Categories(Genre)
audible_data = audible_data[['Book Title', 'Book Author', 'Book Narrator', 'Categories']]

# Remove all 'Categories', and 'Book Narrator' NaN records
audible_data = audible_data[audible_data['Categories'].notna()]
audible_data = audible_data[audible_data['Book Narrator'].notna()]

# lower case and split on commas or &-sign 'Categories'
audible_data['Categories'] = audible_data['Categories'].map(
    lambda x: x.lower().replace(' &', ',').replace('genre', '').split(','))
# Book Author
audible_data['Book Author'] = audible_data['Book Author'].map(lambda x: x.lower().replace(' ', '').split(' '))
# Book Narrator
audible_data['Book Narrator'] = audible_data['Book Narrator'].map(lambda x: x.lower().replace(' ', '').split(' '))

for index, row in audible_data.iterrows():
    # row['Book Narrator'] = [x.replace(' ','') for x in row['Book Narrator']]
    row['Book Author'] = ''.join(row['Book Author'])
```

```python id="H1U617hkmnRm"
# make 'Book Title' as an index
audible_data.set_index('Book Title', inplace=True)

audible_data['bag_of_words'] = ''
for index, row in audible_data.iterrows():
    words = ''
    for col in audible_data.columns:
        if col != 'Book Author':
            words = words + ' '.join(row[col]) + ' '
        else:
            words = words + row[col] + ' '
    row['bag_of_words'] = words

audible_data.drop(columns=[x for x in audible_data.columns if x != 'bag_of_words'], inplace=True)
```

```python id="iEFzDIAumnKp"
recommendation_movies = []

# Vectorizing the entire matrix as described above!
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(audible_data['bag_of_words'])

# running pairwise cosine similarity
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)  # getting a similarity matrix
```

```python id="GWWiWb8Wpj8D"
def recommend(k=5):
    # gettin the index of the book that matches the title
    indices = pd.Series(audible_data.index)
    idx = indices.sample(1)

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim2[idx.index[0]]).sort_values(ascending=False)

    # getting the indexes of the k most similar audiobooks
    top_k_indexes = list(score_series.iloc[1:k+1].index)

    topk = indices[top_k_indexes].tolist()

    print("For '{}', Top {} similar audiobooks are {}".format(idx.values[0], k, topk))
```

```python colab={"base_uri": "https://localhost:8080/"} id="65NE6oyence8" outputId="fb3f5ed6-0f68-4146-caff-608c70a90fd5"
recommend()
```

```python colab={"base_uri": "https://localhost:8080/"} id="z9xBOtlznaZn" outputId="69182b9a-6b8d-4ac4-e0c3-e89f407c686e"
recommend()
```

```python colab={"base_uri": "https://localhost:8080/"} id="W-PNtqJJp7Y8" outputId="46e74752-8696-49b4-aeb1-88386c8b9d80"
recommend()
```

```python colab={"base_uri": "https://localhost:8080/"} id="yTNWmARoqA2j" outputId="5248fd90-eeae-4fd2-ae18-1a2730f9bf76"
recommend()
```

```python colab={"base_uri": "https://localhost:8080/"} id="a01LJFT5qst2" outputId="de33b1ec-ce3c-444d-ae08-62de500a2a95"
recommend(10)
```
