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

<!-- #region id="JuP8_LTZpq6W" -->
# Real-time In-session Personalization
> Personalization with deep learning in 100 lines of code using Elasticsearch, Redis and Word2vec

- toc: true
- badges: true
- comments: true
- categories: [Elasticsearch, Redis, Word2vec, Visualization, Session, Sequential, Embedding, RealTime, Retail]
- image:
<!-- #endregion -->

<!-- #region id="PfGYKsKLto_Y" -->
### Introduction
<!-- #endregion -->

<!-- #region id="Etc2uAVCqzoW" -->
| |  |
| :-: | -:|
| Vision | Personalizing Customer's In-session experience in real-time |
| Mission | Learn embeddings for a small sequetial dataset and use efficient indexing and database techniques to retrieve results in real-time |
| Scope | Synthetic dataset, Model training, indexing and retrieval, eCommerce space |
| Task | Next-item Prediction |
| Data | Synthetic Sequential |
| Tool | Word2vec, Elasticsearch, Redis, Colab |
| Technique | Word2vec for embedding, Elasticsearch for indexing, Redis for in-memory fast retrieval |
| Process | 1) Setup environment by installing and starting elaticsearch, redis backend server, 2) Generate synthetic data, 3) Train the embedding model, 4) Visualize the results, 5) Index the embedding vectors, 6) Retrieve the top-k results in real-time |
| Takeaway | Elasticsearch and Redis - a good combination for fast and efficient retrieval at scale, Word2vec is simple yet effective entry-level technique for learning sequential embeddings |
| Credit | [Jacopo Tagliabue](https://github.com/jacopotagliabue/clothes-in-space) |
| Link | [link1](https://github.com/jacopotagliabue), [link2](https://blog.coveo.com/clothes-in-space-real-time-personalization-in-less-than-100-lines-of-code/) |
<!-- #endregion -->

<!-- #region id="CGj9t27ro2uY" -->
### Install Elasticsearch
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="8XIwkKAEgo__" outputId="bda87def-02e4-45d1-a1c4-bec3562084eb"
!wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-oss-7.0.5-linux-x86_64.tar.gz
```

```bash id="YUj0878jPyz7" colab={"base_uri": "https://localhost:8080/"} outputId="d57caf6f-4902-4278-a2b2-6eba7b17ada9"

wget -q https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-oss-7.0.1-linux-x86_64.tar.gz
wget -q https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-oss-7.0.1-linux-x86_64.tar.gz.sha512
tar -xzf elasticsearch-oss-7.0.1-linux-x86_64.tar.gz
sudo chown -R daemon:daemon elasticsearch-7.0.1/
shasum -a 512 -c elasticsearch-oss-7.0.1-linux-x86_64.tar.gz.sha512 
```

<!-- #region id="vAzfu_WiEs4F" -->
Run the instance as a daemon process
<!-- #endregion -->

```bash id="n9ujlunrWgRx" colab={"base_uri": "https://localhost:8080/"} outputId="5584f918-2492-471f-9895-1c0866460228" magic_args="--bg"

sudo -H -u daemon elasticsearch-7.0.1/bin/elasticsearch
```

```python id="XyUa9r6MgWtW"
# Sleep for few seconds to let the instance start.
import time
time.sleep(20)
```

<!-- #region id="f6qxCdypE1DD" -->
Once the instance has been started, grep for `elasticsearch` in the processes list to confirm the availability.
<!-- #endregion -->

```bash id="48LqMJ1BEHm5" colab={"base_uri": "https://localhost:8080/"} outputId="b64af411-85b9-468e-a343-2ff90b916d28"

ps -ef | grep elasticsearch
```

<!-- #region id="wBuRpiyf_kNS" -->
query the base endpoint to retrieve information about the cluster.
<!-- #endregion -->

```bash id="ILyohKWQ_XQS" colab={"base_uri": "https://localhost:8080/"} outputId="41c4d0e8-1888-4b9b-9975-5163756d7dc5"

curl -sX GET "localhost:9201/"
```

```bash colab={"base_uri": "https://localhost:8080/"} id="BpdcqLnYh_ks" outputId="080872ed-e2e5-4909-f897-9dc5109b7b9b"

curl -sX GET "localhost:9201/"
```

<!-- #region id="jloHOErjLNwe" -->
### Install Redis
<!-- #endregion -->

```python id="RnbYNpCpHrwM"
!wget http://download.redis.io/releases/redis-stable.tar.gz --no-check-certificate
!tar -xf redis-stable.tar.gz && cd redis-stable/src && make
```

```python colab={"base_uri": "https://localhost:8080/"} id="9Ze3TwogHy4I" outputId="25b9f210-e9bc-48eb-f9ec-1243d8d68182"
! nohup ./redis-stable/src/redis-server > redis_nohup.out &
! cat redis_nohup.out
```

```python colab={"base_uri": "https://localhost:8080/"} id="EmkCMB9gH0yi" outputId="eedda604-f247-425a-ec4b-e08643eb451c"
!pip install redis

import redis
r = redis.StrictRedis(host="127.0.0.1", port=6379)
```

<!-- #region id="XZTmvkfgLSQU" -->
### Generate Synthetic Data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="TooXXcTNIC1T" outputId="5a2f67cc-0f47-4114-c09b-de154ec242c3"
%%writefile catalog.csv
sku,name,target,image
01111,shoesA,men,"https://static.nike.com/a/images/c_limit,w_592,f_auto/t_product_v1/e706a692-be9e-4d5a-9537-3d64bd2d0e34/blazer-mid-77-shoe-9SvVxB.png"
01112,shoesB,women,"https://static.nike.com/a/images/c_limit,w_592,f_auto/t_product_v1/b98cdc08-0691-48a8-99f7-bd4856ab5628/air-force-1-07-shoe-AKTdww3y.png"
01113,shoesC,men,"https://static.nike.com/a/images/c_limit,w_592,f_auto/t_product_v1/fe07ff2a-e259-4664-b83e-003f68dfaf62/air-max-96-2-shoe-6l7J5T.png"
01114,shoesD,men,"https://static.nike.com/a/images/c_limit,w_592,f_auto/t_product_v1/9313fb42-b9fb-4a13-bd94-46137fd19c12/air-jordan-1-mid-se-shoe-zgPD6z.png"
01115,shoesE,women,"https://static.nike.com/a/images/c_limit,w_592,f_auto/t_product_v1/e1eaf5d3-dfca-4bba-b0f3-2f356520b655/blazer-mid-77-se-shoe-97H9PZ.png"
01116,tshirtA,women,"https://static.nike.com/a/images/c_limit,w_592,f_auto/t_product_v1/a7f6f9ff-8fd2-4acc-b547-8ea5497e9def/nikelab-t-shirt-Q89CVt.png"
01117,tshirtB,women,"https://static.nike.com/a/images/c_limit,w_592,f_auto/t_product_v1/a7b40e60-9626-4538-bdbb-51a0e85da6d3/dri-fit-race-short-sleeve-running-top-h2qbZD.png"
01118,tshirtC,women,"https://static.nike.com/a/images/c_limit,w_592,f_auto/t_product_v1/46752967-b4eb-4d9b-8150-75901f3e87a8/dri-fit-run-division-running-tank-K7wCpp.png"
01119,tshirtD,men,"https://static.nike.com/a/images/c_limit,w_592,f_auto/t_product_v1/f20341c3-0620-4430-9c04-783f9b507789/sb-skate-t-shirt-LHlzwG.png"
01120,tshirtE,men,"https://static.nike.com/a/images/c_limit,w_592,f_auto/t_product_v1/259734bd-93b6-420e-adc6-31cb187e4442/dri-fit-superset-short-sleeve-training-top-6LKnxr.png"
```

```python colab={"base_uri": "https://localhost:8080/"} id="pBXL-Q7CJydq" outputId="e5434b3f-4668-40cc-f278-774c674cafbd"
%%writefile sessions.txt
0	01112 01115 01115 01112
1	01118 01116 01120 01118 01117 01116 01118
2	01113 01114 01114 01111 01111 01113
3 01116 01117
4 01111 01113 01114 01118 01113 01114 01111
5 01117 01117 01118 01116
6	01112 01112 01115 01112
7	01120 01120 01119 01120 01119 01120
8	01113 01114 01114 01111 01113
9 01116 01117 01116 01117 01118
10 01111 01113 01114 01119 01120 01119
11 01117 01117 01118 01116
```

<!-- #region id="RchMHrYbmxS2" -->
### Import and global variables
<!-- #endregion -->

```python id="YattbDl9J2Ey"
import os
import csv
from datetime import datetime
import string
import requests
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import redis  # to communicate with redis
import gensim # to talk to gensim
from IPython.display import Image  # to display URL in noteboook for visual debug
from IPython.core.display import display # to display URL in noteboook for visual debug
from elasticsearch import Elasticsearch, helpers # remember to !pip install elasticsearch
```

```python id="3-FB2-l4J66f"
DATA_FOLDER = ''
CATALOGUE_FILE = os.path.join(DATA_FOLDER, 'catalog.csv')
SESSION_FILE = os.path.join(DATA_FOLDER, 'sessions.txt')
```

```python id="QaPInbJdKGPa"
EMBEDDING_DIMS = 5 # specify embedding size
```

<!-- #region id="0EDt629Qncz0" -->
### Python clients for Redis and ElasticSearch
<!-- #endregion -->

```python id="VRHMIUj4KM_n"
# redis credentials here!
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_PWD = None
# redis data structure
REDIS_HASH_FORMAT = 'product_h'
# start redis client
redis_client = redis.StrictRedis(host=REDIS_HOST, 
                                 port=REDIS_PORT, 
                                 db=REDIS_DB, 
                                 password=REDIS_PWD)
```

```python id="9ViWcJFwKSu_"
INDEX_NAME = 'catalog'
ES_HOST = {"host": "localhost", "port": 9200}
# if you're running with the docker-compose option and not the manuel docker instance, use:
# ES_HOST = {"host": "elasticsearch", "port": 9200}
es_client = Elasticsearch(hosts=[ES_HOST])
```

<!-- #region id="50p-Clm1njtG" -->
### Train Embeddings
<!-- #endregion -->

<!-- #region id="AyFHo7ADmxTL" -->
_First of all, get products from the catalogue dump into a usable form_
<!-- #endregion -->

```python id="-D8h1wzvKXpd"
# First of all, get products from the catalogue dump into a usable form
def get_products_from_catalogue(catalog_file):
    """
    parse catalogue file into a map SKU -> properties (sku, name, target, image url)
    """
    products = {}
    with open(catalog_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['sku'] and row['image'].endswith('.png'):
                products[row['sku']] = row
    
    return products
```

```python colab={"base_uri": "https://localhost:8080/"} id="IfdN790_Kebj" outputId="83342647-da5a-403b-a765-4ed5f4697d6f"
products = get_products_from_catalogue(CATALOGUE_FILE)
print('{} products in catalog!'.format(len(products)))
dict(list(products.items())[0:3])
```

```python id="yhpq7QZnKjNm"
def train_embedding_model(training_data):
    """
    training_data is a list of lists (list of words, products, etc.)
    
    """
    # train model with standard params
    model = gensim.models.Word2Vec(training_data,
                                   min_count=1,
                                   sample=1,
                                   size=EMBEDDING_DIMS,
                                   workers=-1,
                                   window=3,
                                   iter=20)
    vectors = model.wv
    # remove model from memory
    del model
    
    # return vectors as TOKEN -> VECTOR map
    return vectors

def solve_vector_analogy(vectors, man, king, women):
    # MAN : KING = WOMAN : ? -> QUEEN
    return vectors.most_similar_cosmul(positive=[king, women], negative=[man])
```

```python id="2tnJ5TyYKpU9"
def get_products_from_sessions(session_file):
    """
        Our file from the analytics service conveniently dumps, line by line,
        user sessions. We just read the file and return a list of lists!
        
        Every line is:
        
        LINE_ID (as INT) TAB PRODUCT 1 TAB PRODUCT 2 ...
        
        P.s.: our file has been pre-processed to include only session with length >= 3 and < 200
    """
    sessions = []
    with open(session_file) as session_f:
        for line in session_f:
            products = line.strip().split(' ')[1:]
            sessions.append(products)
        
    return sessions
```

```python colab={"base_uri": "https://localhost:8080/"} id="igHQ_LXBMKds" outputId="8ae8cf2b-e033-42a3-9763-0aded9fa441b"
training_session_data = get_products_from_sessions(SESSION_FILE)
print('Total sessions: {}, first is: {}'.format(len(training_session_data), training_session_data[0]))
```

```python id="QszMVo-KKrtI"
product_embeddings = train_embedding_model(training_session_data)
```

<!-- #region id="9yrviXSInoOE" -->
### Visualize Results
<!-- #endregion -->

<!-- #region id="lRiw5i-8mxTU" -->
_Check item-item similarity by looking at product vectors close together in the space_
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 552} id="U1iolrrhKv-7" outputId="0a6c9d12-8074-4f01-fcba-88f59033ce91"
TEST_PRODUCT = '01111'

matches = product_embeddings.most_similar_cosmul(positive=[TEST_PRODUCT])
# display top N
print("For this:")
display(Image(products[TEST_PRODUCT]['image'], width=150, unconfined=True))
for i,m in enumerate(matches[:2]):
    print("\nRecommendation {}:".format(i+1))
    display(Image(products[m[0]]['image'], width=150, unconfined=True))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 552} id="Bb4cfcMWVX4W" outputId="412147da-7d81-4021-ecb0-6c6ece0f0d02"
TEST_PRODUCT = '01112'

matches = product_embeddings.most_similar_cosmul(positive=[TEST_PRODUCT])
# display top N
print("For this:")
display(Image(products[TEST_PRODUCT]['image'], width=150, unconfined=True))
for i,m in enumerate(matches[:2]):
    print("\nRecommendation {}:".format(i+1))
    display(Image(products[m[0]]['image'], width=150, unconfined=True))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 552} id="7usD5zN9YX-1" outputId="80cac680-b125-40d4-9ec6-bbced7a6b72e"
TEST_PRODUCT = '01114'

matches = product_embeddings.most_similar_cosmul(positive=[TEST_PRODUCT])
# display top N
print("For this:")
display(Image(products[TEST_PRODUCT]['image'], width=150, unconfined=True))
for i,m in enumerate(matches[:2]):
    print("\nRecommendation {}:".format(i+1))
    display(Image(products[m[0]]['image'], width=150, unconfined=True))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 552} id="OosIQItKYMyK" outputId="3b53c612-f8b0-4d9a-ebbe-776ea86a1072"
TEST_PRODUCT = '01116'

matches = product_embeddings.most_similar_cosmul(positive=[TEST_PRODUCT])
# display top N
print("For this:")
display(Image(products[TEST_PRODUCT]['image'], width=150, unconfined=True))
for i,m in enumerate(matches[:2]):
    print("\nRecommendation {}:".format(i+1))
    display(Image(products[m[0]]['image'], width=150, unconfined=True))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 552} id="sB9_68_TYU0Q" outputId="de4939e3-da83-48df-8b68-d4339d0af56d"
TEST_PRODUCT = '01118' 

matches = product_embeddings.most_similar_cosmul(positive=[TEST_PRODUCT])
# display top N
print("For this:")
display(Image(products[TEST_PRODUCT]['image'], width=150, unconfined=True))
for i,m in enumerate(matches[:2]):
    print("\nRecommendation {}:".format(i+1))
    display(Image(products[m[0]]['image'], width=150, unconfined=True))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 552} id="xnHKYorjYr52" outputId="c05932ee-1626-4b36-ada6-f446da06331e"
TEST_PRODUCT = '01120' 

matches = product_embeddings.most_similar_cosmul(positive=[TEST_PRODUCT])
# display top N
print("For this:")
display(Image(products[TEST_PRODUCT]['image'], width=150, unconfined=True))
for i,m in enumerate(matches[:2]):
    print("\nRecommendation {}:".format(i+1))
    display(Image(products[m[0]]['image'], width=150, unconfined=True))
```

<!-- #region id="vWQRYc34ntn8" -->
### Vector Analogy
<!-- #endregion -->

<!-- #region id="kNnaLfxXmxTV" -->
_Playing with some analogies here_
<!-- #endregion -->

<!-- #region id="G246yUs_pN6D" -->
> Note: if women shoes goes with women tshirt, then men shoes goes with what?
<!-- #endregion -->

```python id="arkAwb24bZkg"
# fill here with your product IDs to test for analogies
PRODUCT1 = '01112'
PRODUCT1_MATCH = '01117'
PRODUCT2 = '01113'
```

```python id="9vcm7i98YtuS"
assert all(_ in product_embeddings.vocab for _ in [PRODUCT1, PRODUCT1_MATCH, PRODUCT2])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 417} id="TgyOLfqib0qm" outputId="a0671b5b-ca64-4921-c47b-9fea1646fa83"
matches = solve_vector_analogy(product_embeddings, PRODUCT1, PRODUCT1_MATCH, PRODUCT2)
# first show products
for _ in [PRODUCT1, PRODUCT1_MATCH, PRODUCT2]:
    display(Image(products[_]['image'], width=100, unconfined=True))
# then display matches
for m in matches[:1]:
    if m[0] in products:
        display(Image(products[m[0]]['image'], width=100, unconfined=True))
```

<!-- #region id="cd06UqOkpYMz" -->
> Warning: Analogy test failed.
<!-- #endregion -->

<!-- #region id="LuoewrktmxTX" -->
_Finally, we add the vectors to our product dictionary_
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ubvBCRYub5EY" outputId="b7775c32-5055-478c-95b7-33ae09da8f39"
# add vector to products
for sku, p in products.items():
    p['vector'] = product_embeddings[p['sku']].tolist() if p['sku'] in product_embeddings else None
    p['popularity'] = random.randint(0, 1000)  # add a popularity field to fake popularity data for later retrieval
# remove products without vectors for simplicity
products = {k: v for k,v in products.items() if v['vector'] is not None}
len(products)
```

<!-- #region id="igNBEsAPmxTZ" -->
### Personalizing search
<!-- #endregion -->

```python id="b0mXwT4vfHrj"
def re_rank_results(session_vector, skus):
    results_vectors = retrieve_vectors_from_redis(skus)
    distance_matrix = cosine_similarity(session_vector.reshape(1, -1), results_vectors)[0]
    so = np.argsort(distance_matrix)
    return list(reversed(list(np.array(skus)[so])))
```

<!-- #region id="-aCOJjyLqbm9" -->
> Tip: dense_vector data type comes with premium version of elasticsearch (line #23 in the below cell). So we used text field instead.
<!-- #endregion -->

```python id="W3oLmO5WfL2D"
def upload_docs_to_es(index_name, docs):
    """
    index_name is a string 
    docs is a map doc id -> doc as a Python dictionary (in our case SKU -> product)
    """
    # first we delete an index with the same name if any 
    # ATTENTION: IF YOU USE THIS CODE IN THE REAL WORLD THIS LINE WILL DELETE THE INDEX
    if es_client.indices.exists(index_name):
        print("Deleting {}".format(index_name))
        es_client.indices.delete(index=index_name)    
    # next we define our index
    body = {
        'settings': {
            "number_of_shards" : 1,
            "number_of_replicas" : 0
        },
        "mappings": {
          "properties": {
                "name": { "type": "text"},
                "target": { "type": "text" },
                "image": { "type": "text" } ,
                "vector": {
                      # "type": "dense_vector",
                      "type": "text",
                      "dims": EMBEDDING_DIMS
                    }
            }
        }
    }
    # create index
    res = es_client.indices.create(index=index_name, body=body)
    # finally, we bulk upload the documents
    actions = [{
                   "_index": index_name,
                   "_id": sku,
                   "_source": doc
               } for sku, doc in docs.items()
            ]
    # bulk upload
    res = helpers.bulk(es_client, actions)
    
    return res

def query_with_es(index_name, search_query, n=5):
    search_query = {
        "from": 0,
        "size": n,
        "query" : {
            "script_score" : {
                "query": {
                        "match" : {
                            "name" : {
                                "query" : search_query
                            }
                        }
                    },
                "script": {
                  "source" : "doc['popularity'].value / 10"
                }
            }
         }
    }
    res = es_client.search(index=index_name, body=search_query)
    print("Total hits: {}, returned {}\n".format(res['hits']['total']['value'], len(res['hits']['hits'])))
    return [(hit["_source"]['sku'], hit["_source"]['image']) for hit in res['hits']['hits']]

def query_and_display_results_with_es(index_name, search_query, n=5):
    res = query_with_es(index_name, search_query, n=n)
    return display_image(res)

def display_image(skus, n=5):
    for (s, image) in skus[:n]:
        print('{} - {}\n'.format(s, image))
        display(Image(image, width=150, unconfined=True))
            
def query_and_rerank_and_display_results_with_es(index_name, search_query, n, session_vector):
    res = query_with_es(index_name, search_query, n=n)
    skus = [r[0] for r in res]
    re_ranked_sku = re_rank_results(session_vector, skus)

    return display_image([(sku, res[skus.index(sku)][1]) for sku in re_ranked_sku])
```

```python colab={"base_uri": "https://localhost:8080/"} id="FYaklITifT9t" outputId="3837c5c7-f3ed-4dd5-f918-ccfbe81a491e"
upload_result = upload_docs_to_es(INDEX_NAME, products)
upload_result
```

```python colab={"base_uri": "https://localhost:8080/"} id="mscKXiQYfWjm" outputId="69089427-3f5a-4fdf-a67a-a89ebf9fcc35"
es_client.indices.refresh(INDEX_NAME)
resp = es_client.get(index=INDEX_NAME, id=PRODUCT1)
print(resp)
```

<!-- #region id="wRAlr04amxTk" -->
### Load data into Redis

> Note: To simulate a real-time use case
<!-- #endregion -->

```python id="OmxQeSuelVVV"
def redis_upload(redis_client, rows):
    with redis_client.pipeline() as pipe:
        for r in rows:
            pipe.hset(REDIS_HASH_FORMAT, r['sku'], json.dumps(r))
        res = pipe.execute()
    
    return

def load_vectors_to_cache(products, batch_size):
    # first we flush the cache
    # ATTENTION: IF YOU USE THIS CODE IN THE REAL WORLD THIS LINE WILL DELETE ALL DATA
    redis_client.flushall()
    # upload data in bulk with pipeline
    rows = list(products.values())
    for i in range(0, len(rows), batch_size):
        print("Uploading {} rows {} at {}...".format(len(rows), i, datetime.utcnow()))
        redis_upload(redis_client, rows[i: i + batch_size])
    # do some test
    print(redis_client.hmget(REDIS_HASH_FORMAT, [r['sku'] for r in rows[:1]]))
    #return total number of rows uploaded
    return len(rows)
```

```python colab={"base_uri": "https://localhost:8080/"} id="zDgw57CBljAv" outputId="72e4d549-1c8a-411e-830a-0b430b3ce25c"
load_vectors_to_cache(products, batch_size=5)
```

```python id="D9w9dkpgmOGW"
QUERY1 = '"match_all":{}' # put here the first query to test
QUERY2 = '' # put here the second query to test
TOP_N = 5 # top N results to re-rank
```

```python id="Czd1DyAKmxTM"
def get_products_from_catalogue(catalog_file):
    """
    parse catalogue file into a map SKU -> properties (sku, name, target, image url)
    """
    products = {}
    with open(catalog_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['sku'] and row['image'].endswith('.jpg'):
                products[row['sku']] = row
    
    return products
```

```python id="EBb4mDdrmxTN"
products = get_products_from_catalogue(CATALOGUE_FILE)
print('{} products in catalog!'.format(len(products)))
```

<!-- #region id="FphkKUqvmxTm" -->
### Query ES
<!-- #endregion -->

<!-- #region id="5SSQH5JFmxTm" -->
_First, we query ES for a "vanilla" search_
<!-- #endregion -->

```python id="-f0JhV__mxTn"
query_and_display_results_with_es(INDEX_NAME, QUERY1, TOP_N)
```

<!-- #region id="l64aRiasmxTo" -->
_Now, we retrieve from Redis the vectors for products in the session_
<!-- #endregion -->

```python id="CylOQKEdmxTo"
def build_average_vector(vectors, v_shape):
    """
    not exactly fancy, but... 
    see for example https://districtdatalabs.silvrback.com/modern-methods-for-sentiment-analysis
    """
    category_vec = np.zeros(v_shape[0]).reshape(v_shape)
    for v in vectors:
        assert v.shape == category_vec.shape
        category_vec += v
    
    return category_vec / len(vectors)
```

```python id="GjD-XTiimxTp"
def retrieve_vectors_from_redis(skus):
    session_products = redis_client.hmget(REDIS_HASH_FORMAT, skus)
    return [np.array(json.loads(s)["vector"]) for s in session_products if s]

def retrieve_session_vector_from_redis(products_in_session):
    session_vectors = retrieve_vectors_from_redis(products_in_session)
    
    return build_average_vector(session_vectors, session_vectors[0].shape)
```

```python id="81meOQBBmxTq"
session_vector = retrieve_session_vector_from_redis(PRODUCTS_IN_SESSION)
# debug
print(len(session_vector), session_vector[:10])
```

<!-- #region id="753ON5cXmxTq" -->
_Finally use the session vector to query_
<!-- #endregion -->

```python id="8d6LX8NAmxTr"
query_and_rerank_and_display_results_with_es(INDEX_NAME, QUERY1, TOP_N, session_vector)
```

```python id="JiYYrqLLmxTr"
# uncomment here if you like to use Coveo index instead
# query_and_rerank_and_display_results_with_coveo(QUERY1, TOP_N, session_vector)
```

<!-- #region id="ujyjCHSbmxTs" -->
_Try some other query_
<!-- #endregion -->

```python id="mow2FVHJmxTt"
# vanilla query
query_and_display_results_with_es(INDEX_NAME, QUERY2, TOP_N)
```

```python id="pXK_Rio-mxTu"
# now personalized
query_and_rerank_and_display_results_with_es(INDEX_NAME, QUERY2, TOP_N, session_vector)
```

<!-- #region id="1tmClLFcmxTv" -->
### Appendix 1: word embeddings, where it all started
<!-- #endregion -->

```python id="8cpFXxIomxTv"
def get_sentences_from_corpus(corpus_file, max_sentences=None):
    """
        Read the text file and process it as a list of lists, where each list is 
        the tokens in a sentence. Don't care too much about pre-processing,
        just get stuff done.
    """
    sentences = []
    with open(corpus_file) as c_file:
        for line in c_file:
            # remove punctuation, strip lines, lower case it and normalize spaces
            cleaned_line = ' '.join(line.translate(str.maketrans('', '', string.punctuation)).strip().lower().split())
            if not cleaned_line:
                continue
            sentences.append(cleaned_line.split())
            # check if we reached a max number of sentences for training
            if max_sentences and len(sentences) == max_sentences:
                return sentences
    
    return sentences
```

```python id="XMiZs625mxTw"
# texts from 1BN words dataset - if you're using the docker-compose setup, change the path to tmp folder
TEXT_FILE = os.path.join(DATA_FOLDER, 'corpus.txt')  
MAX_SENTENCES = 1000000 # how many sentences to load for the word2vec example: keep it small if you don't like to wait!
```

```python id="SuEfzKsMmxTx"
training_sentences_data = get_sentences_from_corpus(TEXT_FILE, max_sentences=MAX_SENTENCES)
print('Total sentences: {}, first is: {}'.format(len(training_sentences_data), training_sentences_data[0]))
word_embeddings = train_embedding_model(training_sentences_data)
```

<!-- #region id="QDBAFjoymxTx" -->
_Now test similarities and play with analogies_
<!-- #endregion -->

```python id="vhvXuhFfmxTy"
for _ in ['paris', 'france']:
    print('###{}\n{}\n'.format(_, word_embeddings.most_similar_cosmul(positive=[_])))
```

```python id="RNNqcsqrmxTy"
print("BOY : KING = WOMAN : {}\n".format(solve_vector_analogy(word_embeddings, 'boy', 'king', 'girl')[0][0]))
print("PARIS : FRANCE = BERLIN : {}\n".format(solve_vector_analogy(word_embeddings, 'paris', 'france', 'berlin')[0][0]))
```

<!-- #region id="uNab76iymxTz" -->
### Appendix 2: how to visualize vectors and impress friends
<!-- #endregion -->

```python id="hreWrrhWmxTz"
def visualize_word_embeddings_tsne(word_embeddings):
    # colors
    colors = ['red', 'green', 'blue', 'purple', 'yellow', 'black']
    interesting_word_groups = [
        (['he', 'she', 'it', 'they', 'i', 'you', 'we'], 'pronouns'),
        (['london', 'paris', 'berlin', 'budapest', 'amsterdam', 'prague', 'rome'], 'cities'),
        (['italy', 'germany', 'spain', 'romania', 'finland', 'poland', 'norway', 'sweden', 'austria', 'brazil'], 'countries'),
        (['pasta', 'pizza', 'steak', 'pie', 'fries', 'burger', 'salmon'], 'food'),
        (['john', 'mark', 'jane', 'jessica', 'donald', 'simon'], 'names'),
        ([random.choice(list(word_embeddings.vocab)) for _ in range(0, 100)], 'other')
    ]
    all_words = []
    for words, group in interesting_word_groups:
        for w in words:
            all_words.append(w)
    all_keys = [w for w in list(word_embeddings.vocab) if w in all_words]
    all_vectors = [word_embeddings[e] for e in all_keys]
    # get projection
    X_embedded = TSNE(n_components=2).fit_transform(all_vectors)
    word_2_emb = {k: e for k, e in zip(all_keys, X_embedded)}
    # divide groups
    data = []
    groups = []
    for words, group in interesting_word_groups:
        groups.append(group)
        data.append([word_2_emb[w] for w in words])
    print(groups, data[0])
    # create plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    # add groups
    for d, color, group in zip(data, colors, groups):
        x = [_[0] for _ in d]
        y = [_[1] for _ in d]
        ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)
    # show plot
    plt.title('Plot color-coded embeddings')
    plt.legend(loc=2)
    plt.show()
    
    return

visualize_word_embeddings_tsne(word_embeddings)
```

```python id="HJii5BWcmxT0"
def export_vectors_for_projector_visualization(product_2_vectors,
                                               product_2_label,
                                               target_folder):
    # map dictionary to list to preserve order when exporting
    all_p = [p for p in list(product_2_vectors.vocab) if (not product_2_label or p in product_2_label)]
    all_v = [product_2_vectors[p] for p in all_p]
    # write vectors
    with open(os.path.join(target_folder, 'vectors.tsv'), 'w') as v_f:
        for v in all_v:
            v_f.write('{}\n'.format('\t'.join(['{:.5f}'.format(_) for _ in v])))
    # if avalaible, labels can be paired with SKUs for visualization purposes
    # if a mapping is specified, we produce a "meta" file, otherwise we just return
    if not product_2_label:
        return
    # write meta if mapping is available
    with open(os.path.join(target_folder, 'meta.tsv', 'w')) as m_f:
        # header
        m_f.write('sku\tlabel\n')
        for sku in all_p:
            m_f.write('{}\t{}\n'.format(sku, product_2_label[sku]))

    return
```
