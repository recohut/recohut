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

<!-- #region id="h7LRAwH-L7eP" -->
# Kafka MongoDB Real-time Streaming Kafka Producer
> Sending events to Kafka broker in real-time

- toc: true
- badges: true
- comments: true
- categories: [kafka, real time]
- image: 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="eqHjK5UbBEqK" outputId="6fd57d88-339d-4857-f469-593261489da3"
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

```python colab={"base_uri": "https://localhost:8080/"} id="OBkfvdEJJUjF" outputId="de3441a9-70fb-4d76-846a-dd4680ab8cac"
!wget http://files.grouplens.org/datasets/movielens/ml-latest-small.zip
!unzip ml-latest-small.zip
```

```python colab={"base_uri": "https://localhost:8080/"} id="JcnH7FiTHqfJ" outputId="babb4fdf-9244-4b80-9c69-5b3093618aa8"
df = pd.read_csv('./ml-latest-small/movies.csv')
df.to_json('df.json')
df_json = pd.read_json('df.json')
movie_list= df_json.to_dict(orient="records")
print(movie_list[0])
```

<!-- #region id="t5fqJSfYK5Ac" -->
### Producer Setup [notebook]
<!-- #endregion -->

```python id="RXBtNbCTJZb2"
CLOUDKARAFKA_TOPIC = 'yx03wajr-demo'
CLOUDKARAFKA_BROKERS = 'dory-01.srvs.cloudkafka.com:9094, \
dory-02.srvs.cloudkafka.com:9094, \
dory-03.srvs.cloudkafka.com:9094'
CLOUDKARAFKA_USERNAME = 'yx03wajr'
CLOUDKARAFKA_PASSWORD = 'pHva0afDUXPya6JfKrbM1******'
```

```python id="U0Ill6RU2DK-"
topic = CLOUDKARAFKA_TOPIC.split(",")[0]

conf = {
    'bootstrap.servers': CLOUDKARAFKA_BROKERS,
    'session.timeout.ms': 6000,
    'default.topic.config': {'auto.offset.reset': 'smallest'},
    'security.protocol': 'SASL_SSL',
    'sasl.mechanisms': 'SCRAM-SHA-256',
    'sasl.username': CLOUDKARAFKA_USERNAME,
    'sasl.password': CLOUDKARAFKA_PASSWORD
    }
```

```python id="-rBf0QNBJR8B"
p = Producer(**conf)
```

```python id="sNdPhyNCKBA3"
def delivery_callback(err, msg):
  if err:
      sys.stderr.write('%% Message failed delivery: %s\n' % err)
  else:
      sys.stderr.write('%% Message delivered to %s [%d]\n' %
                        (msg.topic(), msg.partition()))
```

```python colab={"base_uri": "https://localhost:8080/"} id="A-IYHF2pKNze" outputId="0bb2741a-781c-4c5f-b1e7-efc2f72104ac"
for movie in movie_list[0:5]:
    try:
        print("Message to be send : ", movie)
        p.produce(topic, str(movie), callback=delivery_callback)
    except BufferError as e:
        sys.stderr.write('%% Local producer queue is full (%d messages awaiting delivery): try again\n' %
                          len(p))
    p.poll(0)

sys.stderr.write('%% Waiting for %d deliveries\n' % len(p))
p.flush()
```

<!-- #region id="M0_ITYAfN2E7" -->
### Producer Setup [terminal]
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="BHPP91vaM620" outputId="acc686e3-d027-4790-f551-1e4b0d5926c5"
%%writefile producer.py

import sys
import os

from confluent_kafka import Producer

CLOUDKARAFKA_TOPIC = 'yx03wajr-demo'
CLOUDKARAFKA_BROKERS = 'dory-01.srvs.cloudkafka.com:9094, \
dory-02.srvs.cloudkafka.com:9094, \
dory-03.srvs.cloudkafka.com:9094'
CLOUDKARAFKA_USERNAME = 'yx03wajr'
CLOUDKARAFKA_PASSWORD = 'pHva0afDUXPya6JfKrbM1******'

if __name__ == '__main__':
    topic = CLOUDKARAFKA_TOPIC.split(",")[0]

    # Consumer configuration
    # See https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md
    conf = {
      'bootstrap.servers': CLOUDKARAFKA_BROKERS,
      'session.timeout.ms': 6000,
      'default.topic.config': {'auto.offset.reset': 'smallest'},
      'security.protocol': 'SASL_SSL',
      'sasl.mechanisms': 'SCRAM-SHA-256',
      'sasl.username': CLOUDKARAFKA_USERNAME,
      'sasl.password': CLOUDKARAFKA_PASSWORD
      }

    p = Producer(**conf)

    def delivery_callback(err, msg):
        if err:
            sys.stderr.write('%% Message failed delivery: %s\n' % err)
        else:
            sys.stderr.write('%% Message delivered to %s [%d]\n' %
                             (msg.topic(), msg.partition()))

    for line in sys.stdin:
        try:
            p.produce(topic, line.rstrip(), callback=delivery_callback)
        except BufferError as e:
            sys.stderr.write('%% Local producer queue is full (%d messages awaiting delivery): try again\n' %
                             len(p))
        p.poll(0)

    sys.stderr.write('%% Waiting for %d deliveries\n' % len(p))
    p.flush()
```

```python colab={"base_uri": "https://localhost:8080/"} id="tU7pY0SFNMF2" outputId="cff8f390-dd63-42b5-a1e0-f79322b3b3a7"
!python producer.py
```
