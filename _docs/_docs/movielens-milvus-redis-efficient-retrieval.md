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

<!-- #region id="8FgLYOButqII" -->
# Recommender with Redis and Milvus
> Storing the pre-calculated user and items vectors of movielens dataset into redis in-memory database and then indexing into milvus for efficient large-scale retrieval

- toc: true
- badges: true
- comments: true
- categories: [retrieval, redis, milvus, movie]
- image:
<!-- #endregion -->

<!-- #region id="3dfeb422" -->
|  Packages |  Servers |
| --------------- | -------------- |
| pymilvus        | milvus-1.1.0   |
| redis           | redis          |
| paddle_serving_app |
| paddlepaddle |
<!-- #endregion -->

```python id="tA3G1ESOMkrS"
!pip install pymilvus==1.1.0
!pip install paddle_serving_app==0.3.1
!pip install paddlepaddle
!pip install redis
```

<!-- #region id="WNYdYI3numT-" -->
### Install and run Milvus server
<!-- #endregion -->

<!-- #region id="IUtzLMjevQRn" -->
> Warning: It will take ~40 minutes to install!
<!-- #endregion -->

```python id="h8kdCNOvMsdb"
!git clone -b 1.1 https://github.com/milvus-io/milvus.git
% cd /content/milvus/core
! ./ubuntu_build_deps.sh
!./build.sh -t Release
# !./build.sh -t Release -g

% cd /content/milvus/core/milvus
! echo $LD_LIBRARY_PATH
import os
os.environ['LD_LIBRARY_PATH'] +=":/content/milvus/core/milvus/lib"
! echo $LD_LIBRARY_PATH
% cd scripts
! nohup ./start_server.sh &
! cat nohup.out
```

```python colab={"base_uri": "https://localhost:8080/"} id="XHBfpO_DkNEA" outputId="e22b1bb3-ea15-46cf-e203-65f2d05969e5"
!cat nohup.out
```

<!-- #region id="-fQOxxmXONX5" -->
We are using Redis as a metadata storage service. Code can easily be modified to use a python dictionary, but that usually does not work in any use case outside of quick examples. We need a metadata storage service in order to be able to be able to map between embeddings and the corresponding data.
<!-- #endregion -->

<!-- #region id="8l6Uz5UNuqsX" -->
### Install and run Redis server
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Uqac1hT5NUs7" outputId="5a5192ca-bb04-4259-b9d4-4e050db5a0b2"
#hide-output
!wget http://download.redis.io/releases/redis-stable.tar.gz --no-check-certificate
!tar -xf redis-stable.tar.gz && cd redis-stable/src && make
```

```python colab={"base_uri": "https://localhost:8080/"} id="uOpL7Krfm1jk" outputId="5c32eb6b-30bf-4e77-db61-9a0ac6220863"
! nohup ./redis-stable/src/redis-server > redis_nohup.out &
! cat redis_nohup.out
```

```python id="4K0PQx_Qpk-d"
!pip install -U grpcio
```

```python colab={"base_uri": "https://localhost:8080/"} id="ryH_eaIEpcuF" outputId="32f9e0db-636f-4698-e1c8-8dee54961fcd"
%cd /content
```

<!-- #region id="cac14f13" -->
### Downloading Pretrained Models

This PaddlePaddle model is used to transform user information into vectors.
<!-- #endregion -->

```python id="b52f00c3" colab={"base_uri": "https://localhost:8080/"} outputId="c20670c6-1a1d-4ec8-edf5-4981c8499a0f"
!wget https://paddlerec.bj.bcebos.com/aistudio/user_vector.tar.gz --no-check-certificate
!mkdir -p movie_recommender/user_vector_model
!tar xf user_vector.tar.gz -C movie_recommender/user_vector_model/
!rm user_vector.tar.gz
```

<!-- #region id="1ab3a252" -->
Downloading Data
<!-- #endregion -->

```python id="39a7facb" colab={"base_uri": "https://localhost:8080/"} outputId="86a36475-e468-447b-d572-9b5d8eb6c53f"
# Download movie information
!wget -P movie_recommender https://paddlerec.bj.bcebos.com/aistudio/movies.dat --no-check-certificate
# Download movie vecotrs
!wget -P movie_recommender https://paddlerec.bj.bcebos.com/aistudio/movie_vectors.txt --no-check-certificate
```

<!-- #region id="e994eb1e-aa76-446b-98c6-02c74f050ba5" -->
Importing Movies into Milvus
<!-- #endregion -->

<!-- #region id="3a999eeb-bcc6-4800-9039-f9c57ea399f1" -->
#### 1. Connectings to Milvus and Redis
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="fQnASRYRno5g" outputId="59d20c18-64d4-40b8-dc0e-50ee4eac09db"
! lsof -i -P -n | grep -E 'milvus|redis'
```

```python id="d8de5e40"
from milvus import Milvus, IndexType, MetricType, Status
import redis

milv = Milvus(host = '127.0.0.1', port = 19530)
r = redis.StrictRedis(host="127.0.0.1", port=6379) 
```

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="KGwZgi9Hqdhs" outputId="c3e09d0e-d937-4daf-ab4d-6c9d605b684f"
milv.client_version()
```

<!-- #region id="a3c114a7" -->
#### 2. Loading Movies into Redis
We begin by loading all the movie files into redis. 
<!-- #endregion -->

```python id="f56cf19c"
import json
import codecs

#1::Toy Story (1995)::Animation|Children's|Comedy
def process_movie(lines, redis_cli):
    for line in lines:
        if len(line.strip()) == 0:
            continue
        tmp = line.strip().split("::")
        movie_id = tmp[0]
        title = tmp[1]
        genre_group = tmp[2]
        tmp = genre_group.strip().split("|")
        genre = tmp
        movie_info = {"movie_id" : movie_id,
                "title" : title,
                "genre" : genre
                }
        redis_cli.set("{}##movie_info".format(movie_id), json.dumps(movie_info))
        
with codecs.open("movie_recommender/movies.dat", "r",encoding='utf-8',errors='ignore') as f:
        lines = f.readlines()
        process_movie(lines, r)
```

<!-- #region id="a54a6046" -->
#### 3. Creating Partition and Collection in Milvus
<!-- #endregion -->

```python id="ef3ef1f7" colab={"base_uri": "https://localhost:8080/"} outputId="2f9f5133-bfc1-47b8-faf3-66984ea71774"
COLLECTION_NAME = 'demo_films'
PARTITION_NAME = 'Movie'

#Dropping collection for clean slate run
milv.drop_collection(COLLECTION_NAME)


param = {'collection_name':COLLECTION_NAME, 
         'dimension':32, 
         'index_file_size':2048, 
         'metric_type':MetricType.L2
        }

milv.create_collection(param)
# milv.create_partition(COLLECTION_NAME, PARTITION_NAME)
```

```python colab={"base_uri": "https://localhost:8080/"} id="399Cxz4cqhZ4" outputId="5ac70704-2877-4f59-fa0a-3df4db2ded6b"
milv.get_collection_info(COLLECTION_NAME)
```

<!-- #region id="d298372e" -->
#### 4. Getting Embeddings and IDs
The vectors in `movie_vectors.txt` are obtained from the `user_vector_model` downloaded above. So we can directly get the vectors and the IDs by reading the file.
<!-- #endregion -->

```python id="1aaee36b"
def get_vectors():
    with codecs.open("movie_recommender/movie_vectors.txt", "r", encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    ids = [int(line.split(":")[0]) for line in lines]
    embeddings = []
    for line in lines:
        line = line.strip().split(":")[1][1:-1]
        str_nums = line.split(",")
        emb = [float(x) for x in str_nums]
        embeddings.append(emb)
    return ids, embeddings

ids, embeddings = get_vectors()
```

<!-- #region id="3a6140b1" -->
#### 4. Importing Vectors into Milvus
Import vectors into the partition **Movie** under the collection **demo_films**.
<!-- #endregion -->

```python id="4ac4cfff" colab={"base_uri": "https://localhost:8080/"} outputId="8aff463d-ec4b-4ecc-e076-2d626d54d536"
# status = milv.insert(collection_name=COLLECTION_NAME, records=embeddings, ids=ids, partition_tag=PARTITION_NAME)
status = milv.insert(collection_name=COLLECTION_NAME, records=embeddings, ids=ids)
status[0]
```

<!-- #region id="e93feb30" -->
### Recalling Vectors in Milvus
#### 1. Genarating User Embeddings
Pass in the gender, age and occupation of the user we want to recommend. **user_vector_model** model will generate the corresponding user vector.
Occupation is chosen from the following choices:
*  0:  "other" or not specified
*  1:  "academic/educator"
*  2:  "artist"
*  3:  "clerical/admin"
*  4:  "college/grad student"
*  5:  "customer service"
*  6:  "doctor/health care"
*  7:  "executive/managerial"
*  8:  "farmer"
*  9:  "homemaker"
*  10:  "K-12 student"
*  11:  "lawyer"
*  12:  "programmer"
*  13:  "retired"
*  14:  "sales/marketing"
*  15:  "scientist"
*  16:  "self-employed"
*  17:  "technician/engineer"
*  18:  "tradesman/craftsman"
*  19:  "unemployed"
*  20:  "writer"
<!-- #endregion -->

```python tags=[] id="1a35a9d4" colab={"base_uri": "https://localhost:8080/"} outputId="6eaf7dbd-95e8-43a6-c43a-9e2c57fb5c81"
import numpy as np
from paddle_serving_app.local_predict import LocalPredictor

class RecallServerServicer(object):
    def __init__(self):
        self.uv_client = LocalPredictor()
        self.uv_client.load_model_config("movie_recommender/user_vector_model/serving_server_dir") 
        
    def hash2(self, a):
        return hash(a) % 1000000

    def get_user_vector(self):
        dic = {"userid": [], "gender": [], "age": [], "occupation": []}
        lod = [0]
        dic["userid"].append(self.hash2('0'))
        dic["gender"].append(self.hash2('M'))
        dic["age"].append(self.hash2('23'))
        dic["occupation"].append(self.hash2('6'))
        lod.append(1)

        dic["userid.lod"] = lod
        dic["gender.lod"] = lod
        dic["age.lod"] = lod
        dic["occupation.lod"] = lod
        for key in dic:
            dic[key] = np.array(dic[key]).astype(np.int64).reshape(len(dic[key]),1)
        fetch_map = self.uv_client.predict(feed=dic, fetch=["save_infer_model/scale_0.tmp_1"], batch=True)
        return fetch_map["save_infer_model/scale_0.tmp_1"].tolist()[0]

recall = RecallServerServicer()
user_vector = recall.get_user_vector()
```

```python colab={"base_uri": "https://localhost:8080/"} id="vfzaCguwtLgL" outputId="78506dc2-aa85-497b-f412-cdc31fb38a31"
user_vector
```

<!-- #region tags=[] id="e15ea6e8" -->
#### 2. Searching
Pass in the user vector, and then recall vectors in the previously imported data collection and partition.
<!-- #endregion -->

```python id="e4d91d02"
TOP_K = 20
SEARCH_PARAM = {'nprobe': 20}
status, results = milv.search(collection_name=COLLECTION_NAME, query_records=[user_vector], top_k=TOP_K, params=SEARCH_PARAM)
```

<!-- #region id="9c847608" -->
#### 3. Returning Information by IDs
<!-- #endregion -->

```python id="90a56325" outputId="bffcbddb-2b25-4320-b8c9-1af5b0b6a75f"
recall_results = []
for x in results[0]:
    recall_results.append(r.get("{}##movie_info".format(x.id)).decode('utf-8'))
recall_results
```

<!-- #region id="d4f7e3c5" -->
### Conclusion
<!-- #endregion -->

<!-- #region id="843120ee" -->
After completing the recall service, the results can be further sorted using the **movie_recommender** model, and then the movies with high similarity scores can be recommended to users. You can try this deployable recommendation system using this [quick start](QUICK_START.md).
<!-- #endregion -->
