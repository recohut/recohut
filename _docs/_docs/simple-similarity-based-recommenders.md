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

<!-- #region id="c8To3DkIoRnq" -->
# Simple Similarity based Recommmendations
> A beginner guide to understand the similarity based recommendations from the ground-up

- toc: true
- badges: true
- comments: true
- categories: [similarity]
- image: 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="A76fRIL0HlS2" outputId="cb956978-e8e6-419c-d28f-ad55cd5b9a49"
!pip install names
!pip install faker_music
!pip install faker
```

```python id="evDsuEV0Da6G"
import numpy as np
import names
from faker import Faker
from faker_music import MusicProvider
from itertools import product
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

fake = Faker()
fake.add_provider(MusicProvider)
```

```python id="jWAdPyPVPAvd"
def generate_data():
  
  NUM_USER = 8
  NUM_ITEM = 8

  RATING_OPTIONS = np.arange(0,5.5,0.5)

  while True:
    
    users = [names.get_full_name() for i in range(NUM_USER)]
    items = [fake.music_genre() for i in range(NUM_ITEM)]

    data = pd.DataFrame(list(product(users, items)), columns=['USERS', 'ITEMS'])

    PROBS = [0.6]
    PROBS.extend([0.05]*(len(RATING_OPTIONS)-1))
    PROBS = [i/sum(PROBS) for i in PROBS]

    data['RATINGS'] = np.random.choice(RATING_OPTIONS, NUM_USER*NUM_ITEM, p=PROBS)

    data['RATINGS'] = data['RATINGS'].astype('float32')

    data = pd.pivot_table(data, index='ITEMS', columns='USERS', values='RATINGS')

    data = data.replace(0, np.nan)

    if len(np.unique(items))==NUM_ITEM:
      return data, users
```

```python colab={"base_uri": "https://localhost:8080/", "height": 328} id="aJVkrUVrDPi1" outputId="d5e9844f-0952-4b25-af11-ffac420de3c9"
data, users = generate_data()
data
```

```python colab={"base_uri": "https://localhost:8080/", "height": 297} id="KVX-mbV_cmMm" outputId="a505b93c-c49b-432f-a8f4-ef5581c8e115"
# with open('data.json', 'w') as f:
#   f.write(json.dumps([row.dropna().to_dict() for index, row in data.iterrows()]))
data.to_json('data.json')
pd.read_json('data.json')
```

```python colab={"base_uri": "https://localhost:8080/"} id="tFEfIJEJTHvq" outputId="f9a195f8-4342-4149-9fb8-a3bf2c004ef1"
import json

with open('data.json') as file:
    data = json.load(file)

data
```

```python id="QrrSHG-ndYHt"
def del_none(d):
    """
    Delete keys with the value ``None`` in a dictionary, recursively.

    This alters the input so you may wish to ``copy`` the dict first.
    """
    # For Python 3, write `list(d.items())`; `d.items()` won’t work
    # For Python 2, write `d.items()`; `d.iteritems()` won’t work
    for key, value in list(d.items()):
        if value is None:
            del d[key]
        elif isinstance(value, dict):
            del_none(value)
    return d  # For convenience
```

```python colab={"base_uri": "https://localhost:8080/"} id="qOO1zgN4dbCH" outputId="81af4f55-8cbf-47a0-888e-18ac49221e93"
data = del_none(data)
data
```

<!-- #region id="kdx77iFIScCI" -->
### Minkowski Distance
<!-- #endregion -->

```python id="kFmKHtWWMPlj"
# defining distance function
def minkowski(ratings1, ratings2, r=2):
    """
    Compute the Minkowski Distance between two users.
    If `r` is not given, it defaults to Euclidian.
    """

    mink_distance = 0

    for key in ratings1:
        if key in ratings2:
            mink_distance += abs(ratings1[key] - ratings2[key])**r

    mink_distance = mink_distance ** (1/r)

    return mink_distance
```

```python id="O-7b5rX_SpAw"
# finding nearest neighbors
def nearest_neighbors(username, users, r=2):
    """Create a sorted list of users
    based on their Minkowski Distance Metric
    (Euclidian by default) to username"""

    distances = []

    for user in users:
        if user != username:
            mnht_distance = minkowski(users[username], users[user], r)
            distances.append((mnht_distance, user))

    distances.sort()

    return distances
```

```python id="FdgsKSfVStny"
# the recommender system
def recommend_bands(username, users):
    """Recommend bands based on other users' ratings"""
    
    neighbor = nearest_neighbors(username, users)[0][1]

    print("{}'s neighbor is {}".format(username, neighbor))

    recom_bands = []

    for band, rating in users[neighbor].items():
        if not band in users[username]:
            print("{} gave {} stars to {} and {} not listened it before!"\
                  .format(neighbor, rating, band, username))
            recom_bands.append((rating, band))
        else:
            print("{} gave {} stars to {} but {} already listened it!"\
                  .format(neighbor, rating, band, username))
    
    recom_bands.sort(reverse=True)

    return [band[1] for band in recom_bands]
```

```python colab={"base_uri": "https://localhost:8080/"} id="ILhhGx8TS1_g" outputId="36203f97-918e-4d0b-fef4-9a7b71debb08"
# testing our recommender
for user in users:
  print('Recommendations for {}: {}'.format(user, recommend_bands(user, data)))
  print("\n{}\n".format('='*50))
```

<!-- #region id="DnLejZ71ifDw" -->
### Pearson Correlation Coefficient
<!-- #endregion -->

```python id="mBxo2sTojJ7O"
def pearson_corrcoef(x, y):
    
    x_mean = x.mean()
    y_mean = y.mean()

    numer = np.sum( (x - x_mean) * (y - y_mean) )
    denom = ( np.sum( (x - x_mean)**2 ) )**0.5 * ( np.sum( (y - y_mean)**2 ) )**0.5

    return numer / denom
```

```python id="Obo3_1Syg37T"
# defining a function to use with our users
def pearson_users(user1, user2):
    
    global data
    ratings1 = []
    ratings2 = []

    for key in data[user1]:
        if key in data[user2]:
            ratings1.append(data[user1][key])
            ratings2.append(data[user2][key])

    ratings1 = np.array(ratings1)
    ratings2 = np.array(ratings2)

    return pearson_corrcoef(ratings1, ratings2)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Gm-nrSisj2DM" outputId="23906edd-bf06-42ec-869b-b96fe7b9197e"
np.isnan(10)
```

```python colab={"base_uri": "https://localhost:8080/"} id="r9lFqy3Hg338" outputId="1322fe8f-c7d5-4b56-dcd1-b281a397992f"
for user1 in users:
  for user2 in users:
    if user1!=user2:
      pearson_corr = pearson_users(user1, user2)
      if not np.isnan(pearson_corr):
        print("Pearson correlation between {} and {} is {:.2f}"\
              .format(user1, user2, pearson_corr))
```

<!-- #region id="_EOE0pwmkSSo" -->
### Cosine Similarity
<!-- #endregion -->

```python id="uLO13U_2nG5l"
johnson = pd.read_json('data.json').fillna(0)['Johnson Butera'].values
halina = pd.read_json('data.json').fillna(0)['Halina Manganaro'].values
```

```python colab={"base_uri": "https://localhost:8080/"} id="zv_hlnQWg30G" outputId="74411f69-86c0-4e26-ade3-99aeb3ddcac8"
# comparing Johnson and Halina (perfect similarity) using cosine similarity
x_size = np.sqrt( np.sum(johnson**2) )
y_size = np.sqrt( np.sum(halina**2) )
dot_prod = np.dot(johnson, halina)

dot_prod / (x_size * y_size)
```
