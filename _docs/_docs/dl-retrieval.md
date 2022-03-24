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

<!-- #region id="cNfAFrCcPjbS" -->
# Large-scale Document Retrieval with ElasticSearch
> A tutorial to understand the process of retrieving documents/items using elastic search and vector indexing methods.

- toc: true
- badges: true
- comments: true
- categories: [elasticsearch, jupyter]
- image: 
<!-- #endregion -->

<!-- #region id="yy3ey7nmL2Pk" -->
## Retrieval Flow Overview

<!-- #endregion -->

<!-- #region id="CuqDb8yOPrLR" -->
<!-- #endregion -->

<!-- #region id="Mz_q-We5Ms_c" -->
## Part 1 - Setting up Elasticsearch
- Download the elasticsearch archive (linux), setup a local server
- Create a client connection to the local elasticsearch instance
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1418, "status": "ok", "timestamp": 1615029015710, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="Hq97N16ZN-qh" outputId="75399aa2-1c93-4f2a-b03b-8d19cc53232b"
# download the latest elasticsearch version
!wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.11.1-linux-x86_64.tar.gz
!tar -xzvf elasticsearch-7.11.1-linux-x86_64.tar.gz
!chown -R daemon:daemon elasticsearch-7.11.1

# prep the elasticsearch server
import os
from subprocess import Popen, PIPE, STDOUT
es_subprocess = Popen(['elasticsearch-7.11.1/bin/elasticsearch'], stdout=PIPE, stderr=STDOUT, preexec_fn=lambda : os.setuid(1))

# wait for a few minutes for the local host to start
!curl -X GET "localhost:9200/"

# install elasticsearch python api
!pip install -q elasticsearch
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 6330, "status": "ok", "timestamp": 1615029165349, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="fbiXCkTKPAuL" outputId="a72eb45a-395a-4811-ddbe-1c702f015350"
# check if elasticsearch server is properly running in the background
from elasticsearch import Elasticsearch, helpers
es_client = Elasticsearch(['localhost'])
es_client.info()
```

<!-- #region id="bYunh06DPbGC" -->
## Part 2 - Walking through an embedding-based retrieval system

<!-- #endregion -->

<!-- #region id="Gn9I5VRcQGpF" -->
### Download MovieLens dataset
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 12390, "status": "ok", "timestamp": 1615029677488, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="lxKakOKSPLZa" outputId="aa7123d4-25ce-4c50-c468-72812d8786a2"
!wget https://files.grouplens.org/datasets/movielens/ml-25m.zip --no-check-certificate
!unzip ml-25m.zip
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"elapsed": 9955, "status": "ok", "timestamp": 1615029693713, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="Q9_kY-H3QohB" outputId="c5d24801-27ef-4c62-c79b-73ba62493882"
import pandas as pd
data = pd.read_csv('ml-25m/movies.csv').drop_duplicates()
data.head()
```

<!-- #region id="3UluFk94RVvq" -->
### Build index with document vectors
<!-- #endregion -->

```python id="ir84MOryRUVs"
import tensorflow_hub as hub
from timeit import default_timer as timer
import json
```

```python id="NBct0WgySCNC"
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
```

```python id="Po31_fncSbo0"
# constants
INDEX_NAME = "movie_title"
BATCH_SIZE = 200
SEARCH_SIZE = 10
MAPPINGS = {
    'mappings': {'_source': {'enabled': 'true'},
                 'dynamic': 'true',
                 'properties': {'title_vector':
                                {'dims': 512, 'type': 'dense_vector'},
                                'movie_id': {'type': 'keyword'},
                                'genres': {'type': 'keyword'}
                                }
                 },
            'settings': {'number_of_replicas': 1, 'number_of_shards':2}
}
```

```python id="x-aKQfPwT1o9"
def index_movie_lens(df, num_doc=500):
  print('creating the {} index.'.format(INDEX_NAME))
  es_client.indices.delete(index=INDEX_NAME, ignore=[404])
  es_client.indices.create(index=INDEX_NAME, body=json.dumps(MAPPINGS))

  requests = []
  count = 0
  start = timer()

  for row_index, doc in df.iterrows():

    # specify the index size to avoid long waiting time
    if count >= num_doc:
      break
    
    # construct requests
    if len(requests) < BATCH_SIZE:

      title_text = doc.title
      genres_text = doc.genres
      title_vector = embed([title_text]).numpy().tolist()[0]

      request = {
          "op_type": "index",
          "_index": INDEX_NAME,
          "_id": row_index,
          "title": title_text,
          "genres": genres_text,
          "title_vector": title_vector,
          "movie_id": doc.movieId
      }

      requests.append(request)
    
    else:
      helpers.bulk(es_client, requests)
      count += len(requests)
      requests.clear()
      if count % (BATCH_SIZE * 2) == 0:
        print("Indexed {} documents in {:.2f} seconds.".format(count, timer()-start))
    
  # Index the remaining
  helpers.bulk(es_client, requests)
  end = timer()

  print("Done indexing {} documents in {:.2f} seconds".format(count, end-start))
```

<!-- #region id="y4cN-nnnWmIe" -->
Ref - https://youtu.be/F4D08uU3mPA
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 113963, "status": "ok", "timestamp": 1615031305076, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="hQgtzMszWnsN" outputId="e29325d4-4399-4d47-d0c5-7e34630a2c1b"
index_movie_lens(data, num_doc=2000)
```

<!-- #region id="nG4Yhjk7YB1X" -->
### Search with query vector
<!-- #endregion -->

```python id="rqo27lJmW8ML"
def return_top_movies(query):

  embedding_start = timer()
  query_vector = embed([query]).numpy().tolist()[0]
  embedding_time = timer() - embedding_start
  formula = "cosineSimilarity(params.query_vector, 'title_vector') + 1.0"

  script_query = {
      "script_score": {
          "query": {"match_all": {}},
          "script": {
              "source": formula,
              "params": {"query_vector": query_vector}
          }
      }
  }

  search_start = timer()
  response = es_client.search(
      index=INDEX_NAME,
      body={
          "size":SEARCH_SIZE,
          "query": script_query,
          "_source": {"includes": ["title", "genres"]}
      }
  )
  search_time = timer() - search_start

  print()
  print("{} total hits.".format(response["hits"]["total"]["value"]))
  
  for hit in response["hits"]["hits"]:

    print("id: {}, score: {}".format(hit["_id"], hit["_score"] - 1))
    print(hit["_source"])
    print()
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1820, "status": "ok", "timestamp": 1615032247857, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="PlUfXg8ZagyB" outputId="ee332b7a-20c0-4c7c-f5fd-0c6593c581db"
return_top_movies("war")
```

<!-- #region id="VT0xw-Ykbh6d" -->
## Part 3 - Approximate Nearest Neighbor (ANN) Algorithms
<!-- #endregion -->

<!-- #region id="fW6Iwum1bxEX" -->
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 30432, "status": "ok", "timestamp": 1615032911597, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="G4PKIA4majzy" outputId="35130bdd-4b13-42f5-f452-7e111152a8f2"
#hide_output
!pip install faiss
!pip install nmslib
!apt-get install libomp-dev

import faiss
import nmslib
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3266, "status": "ok", "timestamp": 1615033157297, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="29Ui3p1Qdg7U" outputId="f21e2eec-8b42-41cf-f2ca-b646eebb12ed"
documents = data['title'].to_list()[:2000]
# # OOM for large document size
embeddings = embed(documents).numpy()
embeddings.shape
```

```python id="xdb1-u8DefgR"
class DemoIndexLSH():
  def __init__(self, dimension, documents, embeddings):
    self.dimension = dimension
    self.documents = documents
    self.embeddings = embeddings

  def build(self, num_bits=8):
    self.index = faiss.IndexLSH(self.dimension, num_bits)
    self.index.add(self.embeddings)

  def query(self, input_embedding, k=5):
    distances, indices = self.index.search(input_embedding, k)

    return [(distance, self.documents[index]) for distance, index in zip(distances[0], indices[0])]

index_lsh = DemoIndexLSH(512, documents, embeddings)
index_lsh.build(num_bits=16)
```

```python id="wEubyvLzgKF_"
class DemoIndexIVFPQ():
  def __init__(self, dimension, documents, embeddings):
    self.dimension = dimension
    self.documents = documents
    self.embeddings = embeddings

  def build(self,
            number_of_partition=2,
            number_of_subquantizers=2,
            subvector_bits=4):
    quantizer = faiss.IndexFlatL2(self.dimension)
    self.index = faiss.IndexIVFPQ(quantizer, 
                                  self.dimension,
                                  number_of_partition,
                                  number_of_subquantizers,
                                  subvector_bits)
    self.index.train(self.embeddings)
    self.index.add(self.embeddings)

  def query(self, input_embedding, k=5):
    distances, indices = self.index.search(input_embedding, k)

    return [(distance, self.documents[index]) for distance, index in zip(distances[0], indices[0])]

index_pq = DemoIndexIVFPQ(512, documents, embeddings)
index_pq.build()
```

```python id="lk7zw9GCiLbg"
class DemoHNSW():
  def __init__(self, dimension, documents, embeddings):
    self.dimension = dimension
    self.documents = documents
    self.embeddings = embeddings

  def build(self, num_bits=8):
    self.index = nmslib.init(method='hnsw', space='cosinesimil')
    self.index.addDataPointBatch(self.embeddings)
    self.index.createIndex({'post': 2}, print_progress=True)

  def query(self, input_embedding, k=5):
    indices, distances = self.index.knnQuery(input_embedding, k)

    return [(distance, self.documents[index]) for distance, index in zip(distances, indices)]

index_hnsw = DemoHNSW(512, documents, embeddings)
index_hnsw.build()
```

```python id="H98OFMKwi7LX"
class DemoIndexFlatL2():
  def __init__(self, dimension, documents, embeddings):
    self.dimension = dimension
    self.documents = documents
    self.embeddings = embeddings

  def build(self, num_bits=8):
    self.index = faiss.IndexFlatL2(self.dimension)
    self.index.add(self.embeddings)

  def query(self, input_embedding, k=5):
    distances, indices = self.index.search(input_embedding, k)

    return [(distance, self.documents[index]) for distance, index in zip(distances[0], indices[0])]

index_flat = DemoIndexFlatL2(512, documents, embeddings)
index_flat.build()
```

```python id="hD98Z5DmjhmT"
def return_ann_top_movies(ann_index, query, k=SEARCH_SIZE):
  query_vector = embed([query]).numpy()
  search_start = timer()
  top_docs = ann_index.query(query_vector, k)
  search_time = timer() - search_start
  print("search time: {:.2f} ms".format(search_time * 1000))
  return top_docs
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1734, "status": "ok", "timestamp": 1615034644346, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="iRdd27NykJRH" outputId="2deed406-bfc1-4e19-f6ff-10fcc097b70b"
return_ann_top_movies(index_flat, "romance")
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1396, "status": "ok", "timestamp": 1615034666880, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="70cRnxLQkO-k" outputId="b4dffd45-358d-472d-a750-2b99d14dd9f9"
return_ann_top_movies(index_lsh, "romance")
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 8106, "status": "ok", "timestamp": 1615034689193, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="I5p67iuXkUaL" outputId="5f08deab-3779-4358-a446-4424bb7def02"
return_ann_top_movies(index_pq, "romance")
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1914, "status": "ok", "timestamp": 1615034943538, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="zJaAlR4ukX5H" outputId="6cbb5b1a-77cf-4c1b-c4d0-12b04f15d6d4"
return_ann_top_movies(index_hnsw, "romance")
```
