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

<!-- #region id="VTHU63hYgwzT" -->
summary: A tutorial to understand the process of retrieving documents/items using elastic search and vector indexing methods.
id: how-to-create-large-scale-retrieval-system-using-elasticsearch
categories: Pytorch
tags: codelabs
status: Published 
authors: Sparsh A.
Feedback Link: https://github.com/sparsh-ai/reco-tutorials/issues
<!-- #endregion -->

<!-- #region id="uC6AkZdChapT" -->
# Large-scale Document Retrieval using ElasticSearch
<!-- #endregion -->

<!-- #region id="DnhxC8mMSj1n" -->
<!-- ------------------------ -->
## Introduction
Duration: 5

### What you'll learn?
- Lorem ipsum

### Why is this important?
- Lorem ipsum

### How it will work?
- Lorem ipsum

### Who is this for?
- Lorem ipsum

### Important resources
- Lorem ipsum
<!-- #endregion -->

<!-- #region id="j-qiA_MTVV24" -->
<!-- ------------------------ -->
## Understand the Process
Duration: 5
<!-- #endregion -->

<!-- #region id="-ANVtHZGXV7C" -->
![elasticsearch_process](img/elasticsearch_process.png)

As shown in the chart above, there are two main steps in the embedding-based retrieval system using Elasticsearch:
1. Indexing: documents are first converted to vectors using deep learning models (aka embedding models). They are then indexed and stored on disk by Elasticsearch.
2. Retrieving: a user query is first converted to its vector representation. Elasticsearch then uses this query vector to evaluate the similarity against indexed documents and returns top-scored ones.

In this tutorial, we will use the Universal Sentence Encoder (USE) model which has been trained to learn the representation of a sentence semantic meaning from large public corpus. Such models usually provide a decent baseline for NLP tasks. In practice, it is necessary to train a deep model to learn the embedding for the target applications for performance boosting.
<!-- #endregion -->

<!-- #region id="BRtkjIHtYpNj" -->
<!-- ------------------------ -->
## Setting up Elasticsearch
Duration: 5
<!-- #endregion -->

<!-- #region id="8B0NQU64Yu2a" -->
### Download the latest elasticsearch version
<!-- #endregion -->

```python id="akmN1OISYyPD"
!wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.11.1-linux-x86_64.tar.gz
!tar -xzvf elasticsearch-7.11.1-linux-x86_64.tar.gz
!chown -R daemon:daemon elasticsearch-7.11.1
```

<!-- #region id="7_b7rFsoZhqQ" -->
### Prep the elasticsearch server
<!-- #endregion -->

```python id="acCIKFS7Y6Jo" executionInfo={"status": "ok", "timestamp": 1622363336437, "user_tz": -330, "elapsed": 406, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
from subprocess import Popen, PIPE, STDOUT
es_subprocess = Popen(['elasticsearch-7.11.1/bin/elasticsearch'], stdout=PIPE, stderr=STDOUT, preexec_fn=lambda : os.setuid(1))
```

<!-- #region id="SivnyteIYxXz" -->
### Create a client connection to the local elasticsearch instance
<!-- #endregion -->

<!-- #region id="2v1zaI_AaMHg" -->
<aside class="positive">
wait for a few minutes for the local host to start
</aside>
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="RWvhJ6sUZAJ3" executionInfo={"status": "ok", "timestamp": 1622363369542, "user_tz": -330, "elapsed": 721, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d6552746-42cd-48da-94e8-f9e29202c87c"
!curl -X GET "localhost:9200/"
```

<!-- #region id="gQNczHHlaS8c" -->
### Install elasticsearch python api
<!-- #endregion -->

```python id="2fH4wOKXZB9q"
!pip install -q elasticsearch
```

<!-- #region id="v29gZ6dZaWAa" -->
### Check if elasticsearch server is properly running in the background
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ZLiVXpGEVVhZ" executionInfo={"status": "ok", "timestamp": 1622363705187, "user_tz": -330, "elapsed": 396, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="46632405-5a91-4e31-c913-f3ddfd724038"
from elasticsearch import Elasticsearch, helpers
es_client = Elasticsearch(['localhost'])
es_client.info()
```

<!-- #region id="fXoi9w45aqn8" -->
<!-- ------------------------ -->
## Embed Movielens Dataset
Duration: 10
<!-- #endregion -->

<!-- #region id="AhZ1jObla-_9" -->
### Download MovieLens dataset
<!-- #endregion -->

```python id="q4GWhXvnVVff"
!wget https://files.grouplens.org/datasets/movielens/ml-25m.zip --no-check-certificate
!unzip ml-25m.zip
```

<!-- #region id="vtsI69FdbEV6" -->
### Read the dataset
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="zMgPeK42VVdE" executionInfo={"status": "ok", "timestamp": 1622363893421, "user_tz": -330, "elapsed": 392, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="30d03b22-11db-42a6-80db-b056120c5698"
import pandas as pd
data = pd.read_csv('ml-25m/movies.csv').drop_duplicates()
data.head()
```

<!-- #region id="3yxA4eHzbijE" -->
### Download USE - a pre-trained text embedding model
<!-- #endregion -->

```python id="uM_Ut5eDbOgb" executionInfo={"status": "ok", "timestamp": 1622363990742, "user_tz": -330, "elapsed": 29230, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import tensorflow_hub as hub
from timeit import default_timer as timer
import json

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
```

<!-- #region id="IQBS0C-DbznN" -->
### Define variables
<!-- #endregion -->

```python id="u8Y4Y1o8bxpC"
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

<!-- #region id="qK-hpmVhbJUe" -->
### Build index with document vectors
<!-- #endregion -->

```python id="mzy4fw4_bOeK" executionInfo={"status": "ok", "timestamp": 1622364099222, "user_tz": -330, "elapsed": 661, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
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

```python colab={"base_uri": "https://localhost:8080/"} id="QL9r5SBpbObf" executionInfo={"status": "ok", "timestamp": 1622364218729, "user_tz": -330, "elapsed": 119521, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8b6d2923-f7d6-4f18-e899-5c1c9a706989"
index_movie_lens(data, num_doc=2000)
```

<!-- #region id="QSYZYgerb7Qs" -->
### Search with query vector
<!-- #endregion -->

```python id="uJi96NLnb_Rr" executionInfo={"status": "ok", "timestamp": 1622364246384, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
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

```python colab={"base_uri": "https://localhost:8080/"} id="p6D3cBO5b_Pc" executionInfo={"status": "ok", "timestamp": 1622364248012, "user_tz": -330, "elapsed": 994, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4ccf96d7-76af-4a51-b7e7-d46dbb9a8696"
return_top_movies("war")
```

<!-- #region id="SakNJPTEcLZg" -->
<!-- ------------------------ -->
## Approximate Retrival
Duration: 2
<!-- #endregion -->

<!-- #region id="9oRyAAp4dB4O" -->
In the last step, we used brute-force method (match the given vector to all other vectors in the database) to find similar movies. This gives us accurate results but it is slow and memory-intensive. This will not work for industrial-scale retrieval demand where we have to retrieve thousands of matching vectors per user, for millions of users in a near-realtime settings. To overcome this challenge, researchers found a technique called **Approximate Nearest Neighbour (ANN)**. In this technique, instead of exhaustively searching the full vector space, we only retrieve top-k nearest neighbour vectors. The accuracy slightly gets reduced but the gain in retrieval speed is worth the tradeoff. Read [this](https://towardsdatascience.com/comprehensive-guide-to-approximate-nearest-neighbors-algorithms-8b94f057d6b6) article to know more about ANN algorithms.
<!-- #endregion -->

<!-- #region id="ggrVVfuLgnno" -->
We'll go through a few common ANN algorithms with open-sourced library nmslib and faiss
• Locality-sensitive hashing
• Product quantization with inverted file
• Hierarchical Navigable Small World Graphs

#### Locality-sensitive hashing (LSH)
LSH is a very classical binary hash. Its core is to create multiple hash functions to map vectors into binary codes. Vectors closely related are
expected to hashed into the same codes.

#### Product quantization with inverted file (IVFPQ)
Product quantization adopts k-means as its core quantizer and drastically increases the number of centroids by dividing each vector into many
subvectors and runs the quantizer on all of these subvectors. The IVFPQ index relies on two levels of quantization.

#### Hierarchical Navigable Small World Graphs (HNSW)
This method relies on exploring the graph based on the closeness relation between a node and its neighbors and neighbors' neighbors and.
HNSW stores the full length vectors and the full graph structure in memory (RAM).
<!-- #endregion -->

<!-- #region id="rui6Z0Nih0qC" -->
### Install libraries
<!-- #endregion -->

```python id="sE2Zfswzb_L-"
!pip install faiss
!pip install nmslib
!apt-get install libomp-dev
```

```python id="78L62P2Fh4Xl" executionInfo={"status": "ok", "timestamp": 1622365704687, "user_tz": -330, "elapsed": 447, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import faiss
import nmslib
```

<!-- #region id="RTJMAJ4FiCPr" -->
### Embed the documents
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="4CxQSAzxh5Zf" executionInfo={"status": "ok", "timestamp": 1622365733965, "user_tz": -330, "elapsed": 10323, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ee4fcd63-a58a-4170-c785-7d437aef08b0"
documents = data['title'].to_list()[:2000]
# # OOM for large document size
embeddings = embed(documents).numpy()
embeddings.shape
```

<!-- #region id="BjX0hdPfiLhQ" -->
<!-- ------------------------ -->
## Compare ANNs
Duration: 2
<!-- #endregion -->

<!-- #region id="dI14yKuLiRYD" -->
### Defining base classes
<!-- #endregion -->

```python id="cGOAIxYuh5Wz" executionInfo={"status": "ok", "timestamp": 1622365781128, "user_tz": -330, "elapsed": 612, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
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

```python id="zj6QGOhyh5Tq" executionInfo={"status": "ok", "timestamp": 1622365790175, "user_tz": -330, "elapsed": 453, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
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

```python id="je3bCuLmidHt" executionInfo={"status": "ok", "timestamp": 1622365824122, "user_tz": -330, "elapsed": 2669, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
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

```python id="pBEA9YB4idFI" executionInfo={"status": "ok", "timestamp": 1622365827541, "user_tz": -330, "elapsed": 392, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
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

<!-- #region id="F9ZebCb1ihhF" -->
### Define retrieval function
<!-- #endregion -->

```python id="t6FfPSLMidCX" executionInfo={"status": "ok", "timestamp": 1622365862250, "user_tz": -330, "elapsed": 574, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def return_ann_top_movies(ann_index, query, k=SEARCH_SIZE):
  query_vector = embed([query]).numpy()
  search_start = timer()
  top_docs = ann_index.query(query_vector, k)
  search_time = timer() - search_start
  print("search time: {:.2f} ms".format(search_time * 1000))
  return top_docs
```

<!-- #region id="9CKl1hcoio6M" -->
### Retrieve the documents using different methods
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="UzWnvHkSit6j" executionInfo={"status": "ok", "timestamp": 1622365891893, "user_tz": -330, "elapsed": 437, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5e6b6c48-f520-41eb-e5bc-19153cf0ef9a"
return_ann_top_movies(index_flat, "romance")
```

```python colab={"base_uri": "https://localhost:8080/"} id="82RjQTKRit4C" executionInfo={"status": "ok", "timestamp": 1622365897482, "user_tz": -330, "elapsed": 435, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5a21bb93-6603-43cd-b16d-8f6bd44cc030"
return_ann_top_movies(index_lsh, "romance")
```

```python colab={"base_uri": "https://localhost:8080/"} id="bhfndwW7it1D" executionInfo={"status": "ok", "timestamp": 1622365903387, "user_tz": -330, "elapsed": 405, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6635c074-bca7-4f77-9b8f-ce9d6a248439"
return_ann_top_movies(index_pq, "romance")
```

```python colab={"base_uri": "https://localhost:8080/"} id="_TLKHYr4iysH" executionInfo={"status": "ok", "timestamp": 1622365910345, "user_tz": -330, "elapsed": 493, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0bd3d047-af4e-42a8-9434-adf1a9182522"
return_ann_top_movies(index_hnsw, "romance")
```

<!-- #region id="Hq-Tn1PsMXXq" -->
<!-- ------------------------ -->
## Conclusion
Duration: 2

Congratulations!

### What we've covered
- Lorem ipsum

### Next steps
- Lorem ipsum

### Links and References
- Lorem ipsum

### Have a Question?
- [Fill out this form](https://form.jotform.com/211377288388469)
- [Raise issue on Github](https://github.com/sparsh-ai/reco-tutorials/issues)
<!-- #endregion -->

```python id="eyZYWimNcKUo"

```
