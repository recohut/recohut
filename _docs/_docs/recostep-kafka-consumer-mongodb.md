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

<!-- #region id="H6scelNfLkw8" -->
# Kafka MongoDB Real-time Streaming Kafka Consumer and MongoDB
> Listening from kafka topic in real-time and storing in mongodb

- toc: true
- badges: true
- comments: true
- categories: [mongodb, kafka, real time]
- image: 
<!-- #endregion -->

```python id="eqHjK5UbBEqK"
!pip install confluent_kafka -q
```

```python id="ppjOFhRnCTJu"
import json
import sys
import os
import pandas as pd

from confluent_kafka import Producer
from confluent_kafka import Consumer, KafkaException, KafkaError
```

<!-- #region id="9mHX3NBGLTMb" -->
### Consumer Setup [notebook]
<!-- #endregion -->

```python id="RXBtNbCTJZb2"
CLOUDKARAFKA_TOPIC = 'yx03wajr-demo'
CLOUDKARAFKA_BROKERS = 'dory-01.srvs.cloudkafka.com:9094, \
dory-02.srvs.cloudkafka.com:9094, \
dory-03.srvs.cloudkafka.com:9094'
CLOUDKARAFKA_USERNAME = 'yx03wajr'
CLOUDKARAFKA_PASSWORD = 'pHva0afDUXPya6JfKrbM1j549G*****'
```

```python id="lAYgDcRZLf9f"
topics = CLOUDKARAFKA_TOPIC.split(",")

# Consumer configuration
conf = {
    'bootstrap.servers': CLOUDKARAFKA_BROKERS,
    'group.id': "%s-consumer" % CLOUDKARAFKA_USERNAME,
    'session.timeout.ms': 6000,
    'default.topic.config': {'auto.offset.reset': 'smallest'},
    'security.protocol': 'SASL_SSL',
    'sasl.mechanisms': 'SCRAM-SHA-256',
    'sasl.username': CLOUDKARAFKA_USERNAME,
    'sasl.password': CLOUDKARAFKA_PASSWORD
}
```

```python id="kQZv-OBALWR_"
c = Consumer(**conf)
c.subscribe(topics)
```

```python colab={"base_uri": "https://localhost:8080/"} id="QLo8xNhFKYrf" outputId="0ceb91f4-842e-4e25-9df0-da503db8aed5"
# while True:
for i in range(10):
  i+=1
  print(i)
  msg = c.poll(timeout=1.0)
  if msg is None:
      continue
  if msg.error():
      # Error or event
      if msg.error().code() == KafkaError._PARTITION_EOF:
          # End of partition event
          sys.stderr.write('%% %s [%d] reached end at offset %d\n' %
                            (msg.topic(), msg.partition(), msg.offset()))
      elif msg.error():
          # Error
          raise KafkaException(msg.error())
  else:
      # Proper message
      sys.stderr.write('%% %s [%d] at offset %d with key %s:\n' %
                        (msg.topic(), msg.partition(), msg.offset(),
                        str(msg.key())))
      print(msg.value())

c.close()
```

<!-- #region id="DOslXBsRTGP_" -->
### Consumer Setup [terminal]
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="A7C7RJnOL6qK" outputId="9e24943e-be6b-443e-e207-642bcc9e0207"
%%writefile consumer.py

import sys
import os

from confluent_kafka import Consumer, KafkaException, KafkaError


CLOUDKARAFKA_TOPIC = 'yx03wajr-demo'
CLOUDKARAFKA_BROKERS = 'dory-01.srvs.cloudkafka.com:9094, \
dory-02.srvs.cloudkafka.com:9094, \
dory-03.srvs.cloudkafka.com:9094'
CLOUDKARAFKA_USERNAME = 'yx03wajr'
CLOUDKARAFKA_PASSWORD = 'pHva0afDUXPya6JfKrbM1j549G*****'

if __name__ == '__main__':
    topics = CLOUDKARAFKA_TOPIC.split(",")

    # Consumer configuration
    # See https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md
    conf = {
        'bootstrap.servers': CLOUDKARAFKA_BROKERS,
        'group.id': "%s-consumer" % CLOUDKARAFKA_USERNAME,
        'session.timeout.ms': 6000,
        'default.topic.config': {'auto.offset.reset': 'smallest'},
        'security.protocol': 'SASL_SSL',
        'sasl.mechanisms': 'SCRAM-SHA-256',
        'sasl.username': CLOUDKARAFKA_USERNAME,
        'sasl.password': CLOUDKARAFKA_PASSWORD
    }

    c = Consumer(**conf)
    c.subscribe(topics)
    try:
        while True:
            msg = c.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                # Error or event
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # End of partition event
                    sys.stderr.write('%% %s [%d] reached end at offset %d\n' %
                                     (msg.topic(), msg.partition(), msg.offset()))
                elif msg.error():
                    # Error
                    raise KafkaException(msg.error())
            else:
                # Proper message
                sys.stderr.write('%% %s [%d] at offset %d with key %s:\n' %
                                 (msg.topic(), msg.partition(), msg.offset(),
                                  str(msg.key())))
                print(msg.value())

    except KeyboardInterrupt:
        sys.stderr.write('%% Aborted by user\n')

    # Close down consumer to commit final offsets.
    c.close()
```

```python colab={"base_uri": "https://localhost:8080/"} id="kdTMz6JyTQyy" outputId="5c37503a-6f65-44d3-c8ec-48f3cf91b30d"
!python consumer.py
```

<!-- #region id="E9fHshxUdrS2" -->
### MongoDB Setup
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="abIKLu67dq65" outputId="7fed633c-53fb-4bc3-90c5-7a923c165b42"
!pip uninstall pymongo
!pip install pymongo[srv]
```

```python id="K2TRWDWs5IVt"
MONGODB_USER = 'kafka-demo'
MONGODB_PASSWORD = '<your-pass>'
MONGODB_CLUSTER = 'cluster0.ca4wh.mongodb.net'
MONGODB_DATABASE = 'movielens'
```

```python id="fpSqox-OX54F"
import pymongo
import urllib 

mongo_uri = f"mongodb+srv://{MONGODB_USER}:{MONGODB_PASSWORD}@{MONGODB_CLUSTER}/{MONGODB_DATABASE}?retryWrites=true&w=majority"
client = pymongo.MongoClient(mongo_uri)
```

```python colab={"base_uri": "https://localhost:8080/"} id="F-pbkwAukbVx" outputId="e8f63ddb-0cba-4939-92c3-f99ba400c2d3"
mydb = client["movielens"]
mydb.list_collection_names()
```

```python colab={"base_uri": "https://localhost:8080/"} id="FUlch4VvmQFf" outputId="3e25cff4-3a14-4044-c55a-5af3bd98539a"
client.list_database_names()
```

```python id="iNsVjxr8mXXX"
movies = mydb.movies
```

```python colab={"base_uri": "https://localhost:8080/"} id="JIzwhu6qmhb8" outputId="d3071aad-2fc5-4836-fc51-da3767408c63"
result = movies.insert_one({'movieId': 3, 'title': 'Grumpier Old Men (1995)', 'genres': 'Comedy|Romance'})
result
```

```python colab={"base_uri": "https://localhost:8080/"} id="h2VDDAqNhZvF" outputId="80ff7a7a-c562-489e-f9e4-e32a43a06457"
print(f"One movie: {result.inserted_id}")
```

```python colab={"base_uri": "https://localhost:8080/"} id="bFLfB7CGvXFR" outputId="629cd933-85e1-46a5-92cb-6f8fc7711a2b"
# single-line command to insert record
print(client.movielens.movies.insert_one({'movieId':5, 'title':'Bride', 'genres':'Comedy'}).inserted_id)
```

```python colab={"base_uri": "https://localhost:8080/"} id="r2sOG1h_hZqr" outputId="ffb4618d-9917-40aa-ede9-7ef20954042f"
movie2 = {'movieId': 2, 'title': 'Jumanji (1995)', 'genres': 'Adventure|Children|Fantasy'}
movie3 = {'movieId': 3, 'title': 'Grumpier Old Men (1995)', 'genres': 'Comedy|Romance'}

new_result = movies.insert_many([movie2, movie3])
print(f"Multiple movies: {new_result.inserted_ids}")
```

```python colab={"base_uri": "https://localhost:8080/"} id="Ud9mkOYqnTcC" outputId="32960e28-7b2a-409f-8745-6bdd52a4f0e0"
import pprint

for doc in movies.find():
  pprint.pprint(doc)
```

```python colab={"base_uri": "https://localhost:8080/"} id="6o5C_7kxnbGW" outputId="a879b8ca-224d-427a-f97d-7df8375fc453"
%%writefile consumer.py

import sys
import os

from confluent_kafka import Consumer, KafkaException, KafkaError
import pymongo

CLOUDKARAFKA_TOPIC = 'yx03wajr-demo'
CLOUDKARAFKA_BROKERS = 'dory-01.srvs.cloudkafka.com:9094, \
dory-02.srvs.cloudkafka.com:9094, \
dory-03.srvs.cloudkafka.com:9094'
CLOUDKARAFKA_USERNAME = 'yx03wajr'
CLOUDKARAFKA_PASSWORD = 'pHva0afDUXPya6JfKrbM1j549G*****'

MONGODB_USER = 'kafka-demo'
MONGODB_PASSWORD = '<your-pass>'
MONGODB_CLUSTER = 'cluster0.ca4wh.mongodb.net'
MONGODB_DATABASE = 'movielens'

mongo_uri = f"mongodb+srv://{MONGODB_USER}:{MONGODB_PASSWORD}@{MONGODB_CLUSTER}/{MONGODB_DATABASE}?retryWrites=true&w=majority"
client = pymongo.MongoClient(mongo_uri)
mydb = client[MONGODB_DATABASE]
movies = mydb.movies

if __name__ == '__main__':
    topics = CLOUDKARAFKA_TOPIC.split(",")

    # Consumer configuration
    # See https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md
    conf = {
        'bootstrap.servers': CLOUDKARAFKA_BROKERS,
        'group.id': "%s-consumer" % CLOUDKARAFKA_USERNAME,
        'session.timeout.ms': 6000,
        'default.topic.config': {'auto.offset.reset': 'smallest'},
        'security.protocol': 'SASL_SSL',
        'sasl.mechanisms': 'SCRAM-SHA-256',
        'sasl.username': CLOUDKARAFKA_USERNAME,
        'sasl.password': CLOUDKARAFKA_PASSWORD
    }

    c = Consumer(**conf)
    c.subscribe(topics)
    try:
        while True:
            msg = c.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                # Error or event
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # End of partition event
                    sys.stderr.write('%% %s [%d] reached end at offset %d\n' %
                                     (msg.topic(), msg.partition(), msg.offset()))
                elif msg.error():
                    # Error
                    raise KafkaException(msg.error())
            else:
                # Proper message
                sys.stderr.write('%% %s [%d] at offset %d with key %s:\n' %
                                 (msg.topic(), msg.partition(), msg.offset(),
                                  str(msg.key())))
                print(msg.value())
                try:
                  movies.insert_one(eval(msg.value().decode('utf-8')))
                except:
                  movies.insert_one({"err_flag":True, "msg":str(msg.value())})

    except KeyboardInterrupt:
        sys.stderr.write('%% Aborted by user\n')

    # Close down consumer to commit final offsets.
    c.close()
```

```python colab={"base_uri": "https://localhost:8080/"} id="FMAspFyRoVKq" outputId="f08c8f0a-45d4-4404-acf5-9bc17a5311e6"
!python consumer.py
```
