# Real-time event capturing with Kafka and MongoDB

| | |
| --- | --- |
| Problem | Real-time event capturing is important in recommender systems because consumer behavior and underlying patterns changes rapidly. But how to capture these events in real-time? |
| Solution | There are many tools and technologies to capture events (clickstream) in real-time and also train/infer real-time recommendations. Kafka + MongoDB is one such set of tools/techs for this objective. We will use Kafka producer at the user's endpoint to send the events to a Kafka consumer/listener via a relevant topic stream (called Kafka topic). Then this received events can directly be fed into recsys training process. We will store these events in MongoDB for long-term persistence. |
| Dataset | ML-100k |
| Preprocessing | We will convert the records into JSON format. |
| Tools | Kafka, MongoDB |
| Tags | EventCapture, RealTime, Kafka, MongoDB, CloudKarafka, Atlas, Clickstream |

## Why is this important?

- Real-time capture of event is very important to serve real-time recommendations
- Everything is on cloud
- Free of cost to start with at least (and to run this experiment obviously)

## How it will work?

![https://github.com/RecoHut-Stanzas/S307784/raw/main/images/S307784%20%7C%20Real-time%20event%20capturing%20with%20Kafka%20and%20MongoDB.drawio.svg](https://github.com/RecoHut-Stanzas/S307784/raw/main/images/S307784%20%7C%20Real-time%20event%20capturing%20with%20Kafka%20and%20MongoDB.drawio.svg)

1. Setup kafka and mongodb clusters
2. Configure the connnections in colab notebooks by saving the right credentials
3. Load and prepare movielens dataset
4. Start Kafka producer, consumer and mongodb listener notebooks
5. Send a record from kafka producer to kafka consumer
6. Save the received record in mongodb
7. Read the record from mongodb stream and convert into pandas dataframe

## Setup Kafka

### Sign-up

Go to [this](https://customer.cloudkarafka.com/signup) link and signup

### Create a team

Enter a team name and create the team. Below is the settings that I used:

![/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled.png](/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled.png)

### Create an instance

Click on *Create New Instance* button and fill-out the details. For instance, I filled the below ones in my case:

![/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-1.png](/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-1.png)

![/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-2.png](/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-2.png)

![/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-3.png](/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-3.png)

![/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-4.png](/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-4.png)

### Create Kafka topic

Click on the kafka instance and you will see something like this:

![/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-5.png](/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-5.png)

We will be needing these details to configure Kafka in our notebook.

For now, let's create a topic first. Click on topic tab, from left-side navigation pane.

![/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-6.png](/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-6.png)

You can use default one also, but for our use case, we are creating a new topic named "<user_name>-<topic_name>".

![/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-7.png](/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-7.png)

Keep this instance detail page open or note down the details safely. We will be needing these details later.

## Setup MongoDB

### Sign up

Go to [this](https://account.mongodb.com/account/register?n=%2Fv2%2F6056079edc5be0371e355b11&nextHash=%23clusters) link and sign-up. I used *Sign up with Google* option.

### Create organization

I got this as my welcome page:

![/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-8.png](/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-8.png)

Click on *Create an Organization.* Give it a name and click on *Next:*

![/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-9.png](/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-9.png)

Click on *Create Organization*:

![/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-10.png](/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-10.png)

### Create a project

Click on *Create New Project*, name it and click on *Next*:

![/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-11.png](/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-11.png)

Then, click on *Create Project*:

![/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-12.png](/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-12.png)

### Create a cluster

Click on *Build a Cluster*:

![/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-13.png](/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-13.png)

Choose the free version:

![/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-14.png](/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-14.png)

Click on *Create Cluster*:

![/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-15.png](/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-15.png)

### Configure user

Click on *CONNECT* button and select *Connect from anywhere.* A default IP 0.0.0.0 will come and click Ok. 

![/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-16.png](/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-16.png)

Click on *Connect* and enter username and password:

![/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-17.png](/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-17.png)

Also get the connection string that we will use in our python notebook code to connnect:

![/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-18.png](/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-18.png)

### Create a database

Click on *COLLECTIONS*, the third button on center-left of this image:

![/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-19.png](/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-19.png)

Click on *Add My Own Data* and Create the *Database*:

![/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-20.png](/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-20.png)

## Kafka producer

Open [this](https://github.com/RecoHut-Stanzas/S307784/blob/main/nbs/Kafka%20Producer.ipynb) jupyter notebook in any of your favorite editor. I used Google Colab. 

### Configure Kafka credentials

![/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-21.png](/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-21.png)

We need to get these 4 parameters from CloudKerafka account that we created. 

Once configured, we are ready to stream events to the topic server. There are 2 ways to run the stream: We can either send a fixed number of events or run a script endlessly that will ask for the message to be sent via terminal. 

In practical scenarios, we inject a javascript into the user's browser session that sends the event to a listening server and that server sends those messages to the kafka producer.

We sent 5 messages to the topic server which would look like this:

![/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-22.png](/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-22.png)

The cloudkarafka manager dashboard would look like this:

![/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-23.png](/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-23.png)

### End-to-end producer script

```python
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

## Kafka consumer and mongodb

Open [this](https://github.com/RecoHut-Stanzas/S307784/blob/main/nbs/Kafka%20Consumer.ipynb) jupyter notebook in any of your favorite editor. I used Google Colab. 

### Configure Kafka credentials

Configuration process is same as previous step.

Once configured, the received messages would look like this:

![/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-24.png](/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-24.png)

We are able to successfully receive the messages in real-time.

### Configure MongoDB

Now, let's store these messages in a persistent database. We are using MongoDB for that.

Enter these 4 parameters that we received during mongoDB configuration:

![/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-25.png](/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-25.png)

### End-to-end consumer + mongodb script

```python
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

## MongoDB listener

Open [this](https://github.com/RecoHut-Stanzas/S307784/blob/main/nbs/MongoDB%20Listener.ipynb) jupyter notebook in any of your favorite editor. I used Google Colab. 

On every message that we are receiving from Kafka, we are storing it in MongoDB. A more efficient process is to store in batch. Here is the side-by-side notebook snapshot of the entire process. Left is the producer who is sending the JSON to Kafka broker, the center is the consumer who is subscribed to the topic and receiving the messages in real-time. And on the right side, we have a MongoDB listener which is connected to MongoDB, listening to the events and converting them into a pandas data frame. 

![/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-26.png](/img/content-tutorials-raw-real-time-event-capturing-with-kafka-and-mongodb-untitled-26.png)

## Conclusion

Congratulations! You have successfully completed the first step in building a real-time recommendation system. 

### What we've covered

- Loaded and preprocessed movielens dataset
- Trained and compared 4 versions of neural net models using tensorflow keras

### Next steps

- Build a recommender model on the data that we received from mongodb stream
- We can create a batch of let's say 10 records to update the model parameters or keep updating these parameters on every single record

## References

1. [https://github.com/RecoHut-Stanzas/S307784](https://github.com/RecoHut-Stanzas/S307784)