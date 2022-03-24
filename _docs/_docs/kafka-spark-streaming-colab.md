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

<!-- #region id="QryzL_Qkbr5X" -->
# Kafka and Spark Streaming in Colab
> Installing Kafka and Spark streaming in colab and streaming movielens dataset

- toc: true
- badges: true
- comments: true
- categories: [spark, pyspark, kafka, movie]
- image:
<!-- #endregion -->

<!-- #region id="0hSquhjLMfO6" -->
<!-- #endregion -->

<!-- #region id="xwaPmnI-MoPD" -->
There are several benefits of implementing Spark-Kafka integration. You can ensure minimum data loss through Spark Streaming while saving all the received Kafka data synchronously for an easy recovery. Users can read messages from a single topic or multiple Kafka topics. 

Along with this level of flexibility you can also access high scalability, throughput and fault-tolerance and a range of other benefits by using Spark and Kafka in tandem. This integration can be understood with a data pipeline that functions in the methodology shown below:
<!-- #endregion -->

<!-- #region id="NO9ENdLXMp5z" -->
<!-- #endregion -->

```python id="48B9eAMMhAgw" colab={"base_uri": "https://localhost:8080/"} outputId="269db02f-3d9c-441a-e3d1-27025e2f0cc1"
!pip install kafka-python
```

<!-- #region id="gjrZNJQRJP-U" -->
### Import packages
<!-- #endregion -->

```python id="m6KXZuTBWgRm"
import os
from datetime import datetime
import time
import threading
import json
from kafka import KafkaProducer
from kafka.errors import KafkaError
import pandas as pd
from sklearn.model_selection import train_test_split
```

<!-- #region id="yZmI7l_GykcW" -->
## Download and setup Kafka and Zookeeper instances

For demo purposes, the following instances are setup locally:

- Kafka (Brokers: 127.0.0.1:9092)
- Zookeeper (Node: 127.0.0.1:2181)

<!-- #endregion -->

```python id="YUj0878jPyz7"
!curl -sSOL https://downloads.apache.org/kafka/2.7.0/kafka_2.13-2.7.0.tgz
!tar -xzf kafka_2.13-2.7.0.tgz
```

<!-- #region id="vAzfu_WiEs4F" -->
Using the default configurations (provided by Apache Kafka) for spinning up the instances.
<!-- #endregion -->

```python id="n9ujlunrWgRx" colab={"base_uri": "https://localhost:8080/"} outputId="b0d4b179-966a-4b25-b5a6-6a253a5536e4"
!./kafka_2.13-2.7.0/bin/zookeeper-server-start.sh -daemon ./kafka_2.13-2.7.0/config/zookeeper.properties
!./kafka_2.13-2.7.0/bin/kafka-server-start.sh -daemon ./kafka_2.13-2.7.0/config/server.properties
!echo "Waiting for 10 secs until kafka and zookeeper services are up and running"
!sleep 10
```

<!-- #region id="f6qxCdypE1DD" -->
Once the instances are started as daemon processes, grep for `kafka` in the processes list. The two java processes correspond to zookeeper and the kafka instances.
<!-- #endregion -->

```python id="48LqMJ1BEHm5" colab={"base_uri": "https://localhost:8080/"} outputId="609accd7-ae85-4781-c68b-8270313ed94f"
!ps -ef | grep kafka
```

<!-- #region id="Z3TntBqanQnh" -->
Create the kafka topics with the following specs:

- susy-train: partitions=1, replication-factor=1 
- susy-test: partitions=2, replication-factor=1 
<!-- #endregion -->

```python id="lXJWqMmWnPyP" colab={"base_uri": "https://localhost:8080/"} outputId="15258496-47ab-4cf6-f29f-601a40995acd"
!./kafka_2.13-2.7.0/bin/kafka-topics.sh --create --bootstrap-server 127.0.0.1:9092 --replication-factor 1 --partitions 1 --topic reco-train
!./kafka_2.13-2.7.0/bin/kafka-topics.sh --create --bootstrap-server 127.0.0.1:9092 --replication-factor 1 --partitions 2 --topic reco-test

```

<!-- #region id="kNxf_NqjnycC" -->
Describe the topic for details on the configuration
<!-- #endregion -->

```python id="apCf9pfVnwn7" colab={"base_uri": "https://localhost:8080/"} outputId="6fad23dd-0d3f-4d78-f320-bef657a92395"
!./kafka_2.13-2.7.0/bin/kafka-topics.sh --describe --bootstrap-server 127.0.0.1:9092 --topic reco-train
!./kafka_2.13-2.7.0/bin/kafka-topics.sh --describe --bootstrap-server 127.0.0.1:9092 --topic reco-test
```

<!-- #region id="jKVnz3Pjot9t" -->
The replication factor 1 indicates that the data is not being replicated. This is due to the presence of a single broker in our kafka setup.
In production systems, the number of bootstrap servers can be in the range of 100's of nodes. That is where the fault-tolerance using replication comes into picture.

Please refer to the [docs](https://kafka.apache.org/documentation/#replication) for more details.

<!-- #endregion -->

<!-- #region id="bjCy3zaCQJ7-" -->
## Movielens Dataset

Kafka being an event streaming platform, enables  data from various sources to be written into it. For instance:

- Web traffic logs
- Astronomical measurements
- IoT sensor data
- Product reviews and many more.

For the purpose of this tutorial, lets download the [Movielens](https://github.com/sparsh-ai/reco-data/blob/master/MovieLens_100K_ratings.csv?raw=true) dataset and feed the data into kafka manually.

<!-- #endregion -->

```python id="emslB2EGQMCR" colab={"base_uri": "https://localhost:8080/"} outputId="b61abc78-3cc6-44d8-f8c8-1e55cded6569"
!wget -O ml_ratings.csv https://github.com/sparsh-ai/reco-data/blob/master/MovieLens_100K_ratings.csv?raw=true
```

<!-- #region id="4CfKVmCvwcL7" -->
## Explore the dataset
<!-- #endregion -->

```python id="nC-yt_c9u0sH" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="646f1c04-d5e5-49ef-bf74-911a7aa26702"
movielens_df = pd.read_csv('ml_ratings.csv')
movielens_df.head()
```

```python id="AlNuW7xbu6o8" colab={"base_uri": "https://localhost:8080/"} outputId="d9fb4319-5e28-4b17-f43a-777e0aea647f"
# Number of datapoints and columns
len(movielens_df), len(movielens_df.columns)
```

<!-- #region id="tF5K9xtmlT2P" -->
## Split the dataset

<!-- #endregion -->

```python id="n-ku_X0Wld59"
train_df, test_df = train_test_split(movielens_df, test_size=0.4, shuffle=True)
print("Number of training samples: ",len(train_df))
print("Number of testing sample: ",len(test_df))

x_train_df = train_df.drop(["Rating"], axis=1)
y_train_df = train_df["Rating"]

x_test_df = test_df.drop(["Rating"], axis=1)
y_test_df = test_df["Rating"]

# The labels are set as the kafka message keys so as to store data
# in multiple-partitions. Thus, enabling efficient data retrieval
# using the consumer groups.
x_train = list(filter(None, x_train_df.to_csv(index=False).split("\n")[1:]))
y_train = list(filter(None, y_train_df.to_csv(index=False).split("\n")[1:]))

x_test = list(filter(None, x_test_df.to_csv(index=False).split("\n")[1:]))
y_test = list(filter(None, y_test_df.to_csv(index=False).split("\n")[1:]))
```

```python id="YHXk0x2MXVgL"
NUM_COLUMNS = len(x_train_df.columns)
len(x_train), len(y_train), len(x_test), len(y_test)
```

<!-- #region id="wwP5U4GqmhoL" -->
## Store the train and test data in kafka

Storing the data in kafka simulates an environment for continuous remote data retrieval for training and inference purposes.
<!-- #endregion -->

```python id="YhwFImSqncLE"
def error_callback(exc):
    raise Exception('Error while sendig data to kafka: {0}'.format(str(exc)))

def write_to_kafka(topic_name, items):
  count=0
  producer = KafkaProducer(bootstrap_servers=['127.0.0.1:9092'])
  for message, key in items:
    producer.send(topic_name, key=key.encode('utf-8'), value=message.encode('utf-8')).add_errback(error_callback)
    count+=1
  producer.flush()
  print("Wrote {0} messages into topic: {1}".format(count, topic_name))
```

```python id="UP_Hyjy0uyPN"
write_to_kafka("reco-train", zip(x_train, y_train))
write_to_kafka("reco-test", zip(x_test, y_test))
```

```python colab={"base_uri": "https://localhost:8080/"} id="fX8HRyZXSGyJ" outputId="dbee1154-6cdc-4d17-ce9a-ef82950d2754"
# ! /content/kafka_2.13-2.7.0/bin/kafka-console-consumer.sh \
# --bootstrap-server localhost:9092 \
# --topic reco-train \
# --from-beginning
```

<!-- #region id="V8m1_WSMSHL7" -->
## Spark Streaming
<!-- #endregion -->

```python id="7xD4cawyvQun"
!apt-get install openjdk-8-jdk-headless -qq > /dev/null
!wget https://downloads.apache.org/spark/spark-2.4.8/spark-2.4.8-bin-hadoop2.7.tgz
!tar -xvf spark-2.4.8-bin-hadoop2.7.tgz
!pip install findspark
```

```python colab={"base_uri": "https://localhost:8080/"} id="8oJknXAgzMW6" outputId="5ef5eb7f-9037-4fe4-c58e-2a477d707c01"
!wget "https://repo1.maven.org/maven2/org/apache/spark/spark-streaming-kafka-0-8-assembly_2.11/2.4.8/spark-streaming-kafka-0-8-assembly_2.11-2.4.8.jar"
```

```python id="y1qgT9emv57O"
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-2.4.8-bin-hadoop2.7"
os.environ['PYSPARK_SUBMIT_ARGS'] = '--jars /content/spark-streaming-kafka-0-8-assembly_2.11-2.4.8.jar pyspark-shell'
```

```python id="m_jvWtSFCFCF"
import findspark
findspark.init()
```

```python id="C7BBe1jvwms-"
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import Normalizer, StandardScaler
import random
import pyspark
import sys
from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from uuid import uuid1
import time

kafka_topic_name = "reco-train"
kafka_bootstrap_servers = 'localhost:9092'
```

<!-- #region id="NaCvL_IDQF_4" -->
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="vxvuH_ZaKZdJ" outputId="adf3d5ca-83e5-4acf-e7ae-fe853b122a85"
from datetime import datetime

now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)
```

```python colab={"base_uri": "https://localhost:8080/"} id="kx4Ktlvr1Jfc" outputId="564a2344-fd33-4881-8ede-73df74334e11"
sc = pyspark.SparkContext()
ssc = StreamingContext(sc,5)

kafka_topic_name = "reco-train"
kafka_bootstrap_servers = 'localhost:9092'

kvs = KafkaUtils.createStream(ssc, kafka_bootstrap_servers, 'spark-streaming-consumer', {kafka_topic_name:1}) 
kvs = KafkaUtils.createDirectStream(ssc, [kafka_topic_name], {"metadata.broker.list": kafka_bootstrap_servers})
kvs = KafkaUtils.createDirectStream(ssc, [kafka_topic_name], {
                        'bootstrap.servers':kafka_bootstrap_servers,
                        'group.id':'test-group',
                        'auto.offset.reset':'largest'})

lines = kvs.map(lambda x: x[1])
counts = lines.flatMap(lambda line: line.split(' '))
counts = lines.flatMap(lambda line: line.split(' ')).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a+b)
counts.pprint()
ssc.start()
# stream will run for 50 sec
ssc.awaitTerminationOrTimeout(50)
ssc.stop()
sc.stop()
```

<!-- #region id="bolvdIncbjsD" -->
## Further exploration
- https://towardsdatascience.com/enabling-streaming-data-with-spark-structured-streaming-and-kafka-93ce91e5b435

<!-- #endregion -->
