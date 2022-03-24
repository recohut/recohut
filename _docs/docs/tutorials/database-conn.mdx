# Database Connections

## Mongodb

<a href="https://nbviewer.org/github/recohut/nbs/blob/main/2021-06-11-recostep-mongodb-listener.ipynb" alt=""> <img src="https://colab.research.google.com/assets/colab-badge.svg" /></a>


```python
!pip uninstall pymongo
!pip install pymongo[srv]

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

## MongoDB to CSV conversion

> Pull a noSQL data from MongoDB and convert into Pandas dataframe
>

<a href="https://nbviewer.org/github/recohut/nbs/blob/main/2020-06-20-mongodb-to-csv-conversion.ipynb" alt=""> <img src="https://colab.research.google.com/assets/colab-badge.svg" /></a>

```python
import pymongo as pm
from pymongo import MongoClient
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize

def _connect_mongo(username, password, host, port, db):
    mongo_uri = 'mongodb://%s:%s@%s:%s/%s' % (username, password, host, port, db)
    conn = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000, ssl=True)
    return conn[db] 

db = _connect_mongo('xxxx', 'xxxx', 'xxxx', xxxx, 'xxxx')
collection = db['xxxx']
test = pd.DataFrame(list(collection.find({}, {"var1":1, "var2":1})))
print(test.info())

# Extracting Non-array data
with open('non_array_features.txt') as f:
    content = f.readlines()
non_array_features = [x.strip() for x in content]

query = ""
for x in non_array_features:
    query+='"'+ x + '":1, '
query = query[:-2]

flat_df = json_normalize(list(collection.find({}, {<paste query string here without ''>})))
flat_df.shape

# Extracting Array data
def extract_array(feature, key, val):
    n1 = pd.DataFrame(list(collection.aggregate([{"$unwind" : "$"+str(feature)}, 
                                        {"$project" : {'key' : "$"+str(feature) + "." + key,
                                                       'value' : "$"+str(feature) + "." + val}}])))
    n2 = n1[~n1['_id'].astype(str).str.contains('timestamp', regex=False)]
    n3 = n2[~n2.set_index(['_id','key'])['value'].index.duplicated(keep='first')]
    n4 = n3.set_index(['_id','key'])['value'].unstack().add_suffix('_').reset_index()
    return n4


'''https://stackoverflow.com/questions/51402430/concatenate-columns-with-same-id-pandas-dataframe'''
def collide_me(x):
    x = x[~x['_id'].astype(str).str.contains('timestamp', regex=False)]
    y = (x.set_index(['_id', x.groupby(['_id']).cumcount()]).unstack().sort_index(axis=1, level=1))
    y.columns = ['{}_{}'.format(i, j) for i, j in y.columns]
    y = y.reset_index()
    return y

def extract_ndarray(df, key, value):
    n1 = df[['_id', key, value]]
    n2 = n1[~n1['_id'].astype(str).str.contains('timestamp', regex=False)]
    n3 = n2[~n2.set_index(['_id',key])[value].index.duplicated(keep='first')]
    n4 = n3.set_index(['_id',key])[value].unstack().add_prefix(key+'_').reset_index()
    return n4

# Key-value feature extraction
af1 = extract_array('array_feature_1', 'key', 'value')
af2 = extract_array('array_feature_2', 'key', 'value')

# Key-multivalue feature extraction
af3 = pd.DataFrame(list(collection.aggregate([{"$unwind" : "$array_feature_3"}, 
                                        {"$project" : {'featurename_31':'$array_feature_3.featurename_31',
                                                       'featurename_32':'$array_feature_3.featurename_32',
                                                       'featurename_33':'$array_feature_3.featurename_33'
                                                      }}])))
af3 = collide_me(af3)

# Key-value multi-dimensional feature extraction
af4 = json_normalize(list(collection.aggregate([{"$unwind": '$array_feature_4'},
                                        {"$project" : {'feature41':'$array_feature_4.feature41'}}
                                        ,{"$unwind": '$responses'}
                                        ,{"$project" : {'feature41_key':'$feature41.key',
                                                        'feature41_value':'$feature41.value'}}
                                       ])))

af4 = extract_ndarray(af4, 'feature41_key', 'feature41_value')

# Joining and exporting data
df = pd.merge(flat_df, af1, on='_id', how='outer')
df = pd.merge(df, af2, on='_id', how='outer')
df = pd.merge(df, af3, on='_id', how='outer')
df = pd.merge(df, af4, on='_id', how='outer')
df.to_csv('mongoDB_to_CSV_converted.csv')
```

## Cassendra

<a href="https://nbviewer.org/github/recohut/nbs/blob/main/2021-07-01-read-data-from-cassandra-into-pandas.ipynb" alt=""> <img src="https://colab.research.google.com/assets/colab-badge.svg" /></a>


```python
import os
from cassandra.cqlengine.models import Model
from cassandra.cqlengine import columns
from datetime import datetime
import pandas as pd
from datetime import datetime

from cassandra.cqlengine.management import sync_table
from cassandra.policies import TokenAwarePolicy
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import (
    Cluster,
    DCAwareRoundRobinPolicy
)
from cassandra.cqlengine.connection import (
    register_connection,
    set_default_connection
)

CASSANDRA_USERNAME='cassandra'
CASSANDRA_PASSWORD='cassandra'
CASSANDRA_HOST='127.0.0.1'
CASSANDRA_PORT=9042
session = None
cluster = None

auth_provider = PlainTextAuthProvider(username=CASSANDRA_USERNAME, password=CASSANDRA_PASSWORD)
cluster = Cluster([CASSANDRA_HOST],
load_balancing_policy=TokenAwarePolicy(DCAwareRoundRobinPolicy()),
port=CASSANDRA_PORT,
auth_provider=auth_provider,
executor_threads=2,
protocol_version=4,
)           

session = cluster.connect()
register_connection(str(session), session=session)
set_default_connection(str(session))
rows = session.execute('select * from demo.click_stream;')
df = pd.DataFrame(list(rows))
df.head()
```

## MS-SQL

<a href="https://nbviewer.org/github/recohut/nbs/blob/main/2022-01-02-email-classification.ipynb" alt=""> <img src="https://colab.research.google.com/assets/colab-badge.svg" /></a>


```python
!apt install unixodbc-dev
!pip install pyodbc

%%sh
curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
curl https://packages.microsoft.com/config/ubuntu/16.04/prod.list > /etc/apt/sources.list.d/mssql-release.list
sudo apt-get update
sudo ACCEPT_EULA=Y apt-get -q -y install msodbcsql17

import os
import pyodbc
import urllib
import pandas as pd
from sqlalchemy import create_engine

driver = [item for item in pyodbc.drivers()][-1]
conn_string = f'Driver={driver};Server=tcp:server.<domain>.com,<port>;Database=<db>;Uid=<userid>;Pwd=<pass>;Encrypt=yes;TrustServerCertificate=yes;Connection Timeout=30;'
conn = pyodbc.connect(conn_string)
cursor = conn.cursor()

# params = urllib.parse.quote_plus(conn_string)
# conn_str = 'mssql+pyodbc:///?odbc_connect={}'.format(params)
# engine_feat = create_engine(conn_str, echo=True)
# print(engine_feat.table_names())

tname = 'tbl_Final_Lable_Data_18_n_19'
query = f'select count(*) from {tname}'

cursor.execute(query)
cursor.fetchall()

query = f'select top 5 * from {tname}'
df = pd.read_sql(query, conn)
df.info()
```