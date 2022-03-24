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

<!-- #region id="jX4a2Mj0LNef" -->
# Kafka MongoDB Real-time Streaming MongoDB Listener
> Listening mongoDB data events in real-time.

- toc: true
- badges: true
- comments: true
- categories: [mongodb, real time]
- image: 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="fIk1Q3gTuAlw" outputId="799cf775-ec13-4c20-ca9f-018b6ece7ca3"
!pip uninstall pymongo
!pip install pymongo[srv]
```

```python id="PEMdGJpYuEd4"
import os
import pymongo
from bson.json_util import dumps

MONGODB_USER = 'kafka-demo'
MONGODB_PASSWORD = '<your-pass>'
MONGODB_CLUSTER = 'cluster0.ca4wh.mongodb.net'
MONGODB_DATABASE = 'movielens'

mongo_uri = f"mongodb+srv://{MONGODB_USER}:{MONGODB_PASSWORD}@{MONGODB_CLUSTER}/{MONGODB_DATABASE}?retryWrites=true&w=majority"
client = pymongo.MongoClient(mongo_uri)
```

```python id="3qT-btVF3Q-F"
import pandas as pd
movies = pd.DataFrame(columns=['_id','movieId','title','genres'])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 80} id="qCvUib_euNGA" outputId="f45485da-8f84-4bc2-a8af-14a830a14e84"
change_stream = client.movielens.movies.watch()
for change in change_stream:
  _temp = change['fullDocument']
  movies = movies.append(pd.DataFrame(_temp, columns=_temp.keys(), index=[0]), ignore_index=True)
  display(movies)
```
