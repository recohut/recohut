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

<!-- #region id="Jf0aHUhtsgPE" -->
# Storing ML features using Feast
> Storing features in feast (a featurestore system), tried out on movielens & ad-click datasets

- toc: true
- badges: true
- comments: true
- categories: [FeatureStore]
- image:
<!-- #endregion -->

<!-- #region id="47p4Kt6jPZbE" -->
## Feast
<!-- #endregion -->

<!-- #region id="NwwcuVvSPgV0" -->
<!-- #endregion -->

<!-- #region id="V7mxBOaBPn1_" -->
Feast (Feature Store) is an operational data system for managing and serving machine learning features to models in production.

[Git](https://github.com/feast-dev/feast)

<!-- #endregion -->

<!-- #region id="0xKlcDKOP2BY" -->
<!-- #endregion -->

```python id="mQCHBlg-PSee" colab={"base_uri": "https://localhost:8080/"} outputId="46b232e2-b5eb-4eec-e8cc-12057210d518"
!pip install -q feast
```

<!-- #region id="bnTSti6YU6ib" -->
A feature repository is a directory that contains the configuration of the feature store and individual features. This configuration is written as code (Python/YAML) and it's highly recommended that teams track it centrally using git. 
<!-- #endregion -->

<!-- #region id="YfYconoLVO36" -->
Edit the example feature definitions in  example.py and run feast apply again to change feature definitions.
<!-- #endregion -->

<!-- #region id="ljhwODVAW9iX" -->
Feast uses a time-series data model to represent data. This data model is used to interpret feature data in data sources in order to build training datasets or when materializing features into an online store.
Below is an example data source with a single entity (driver) and two features (trips_today, and rating).
<!-- #endregion -->

<!-- #region id="8WH6lGaHW_rA" -->
<!-- #endregion -->

<!-- #region id="ArwtR5Wx_J7o" -->
## Movielens
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="1du3HIsPNDVx" outputId="c6958224-77aa-4f70-f05e-cd4d53de07ee"
!pip install -q git+https://github.com/sparsh-ai/recochef.git
from recochef.datasets.movielens import MovieLens
```

<!-- #region id="GGBxfgV1CqYv" -->
### Load
<!-- #endregion -->

```python id="4famy0k3Cpac"
ml = MovieLens()
df = ml.load_interactions()
df.head()
```

<!-- #region id="xRnxWKjzCl9-" -->
### Transform
<!-- #endregion -->

```python id="PELmaKuICkVw"
from recochef.preprocessing import encode, split
train, test = split.chrono_split(df)
train, umap = encode.label_encode(train, col='USERID')
train, imap = encode.label_encode(train, col='ITEMID')
test = encode.label_encode(test, col='USERID', maps=umap)
test = encode.label_encode(test, col='ITEMID', maps=imap)
train.head()
```

<!-- #region id="DOaCtwiKCtRb" -->
### Create a feature repository
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="5cLD9ScIQFUP" outputId="cfbb34c5-6531-41fb-a4b8-906c0b14dee1"
!feast init my_movielens_repo
%cd my_movielens_repo
```

```python id="C4PtfmVlVrdG"
train.to_parquet("./data/movielens_train.parquet")
test.to_parquet("./data/movielens_test.parquet")
```

```python colab={"base_uri": "https://localhost:8080/"} id="61DYdzh1Vh9V" outputId="f75424bf-5615-4993-be90-6589f5d332f2"
%%writefile example.py
from google.protobuf.duration_pb2 import Duration

from feast import Entity, Feature, FeatureView, ValueType
from feast.data_source import FileSource


movielens_train = FileSource(
    path="/content/my_movielens_repo/data/movielens_train.parquet",
    event_timestamp_column="datetime",
    created_timestamp_column="created",
)

movielens_test = FileSource(
    path="/content/my_movielens_repo/data/movielens_test.parquet",
    event_timestamp_column="datetime",
    created_timestamp_column="created",
)


itemid = Entity(name="ITEMID", value_type=ValueType.INT64, description="movie id")
userid = Entity(name="USERID", value_type=ValueType.INT64, description="user id")


movielens_train_view = FeatureView(
    name="movielens_train",
    entities=["itemid","userid"],
    ttl=Duration(seconds=86400 * 1),
    features=[
        Feature(name="RATING", dtype=ValueType.FLOAT),
        Feature(name="TIMESTAMP", dtype=ValueType.FLOAT),
    ],
    online=True,
    input=movielens_train,
    tags={},
)

movielens_test_view = FeatureView(
    name="movielens_test",
    entities=["itemid","userid"],
    ttl=Duration(seconds=86400 * 1),
    features=[
        Feature(name="RATING", dtype=ValueType.FLOAT),
        Feature(name="TIMESTAMP", dtype=ValueType.FLOAT),
    ],
    online=True,
    input=movielens_test,
    tags={},
)
```

<!-- #region id="LHI_B14wVj8a" -->
Register your feature definitions and set up your feature store
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="R2ifKusVQKZH" outputId="2f91b0f9-18b5-4f00-a87f-4490e94a2224"
!feast apply
```

<!-- #region id="ZV56zvD-Qm1x" -->
### Build a training dataset
<!-- #endregion -->

<!-- #region id="CrxmVUTKbiwQ" -->
Feast allows users to build a training dataset from time-series feature data that already exists in an offline store. Users are expected to provide a list of features to retrieve (which may span multiple feature views), and a dataframe to join the resulting features onto. Feast will then execute a point-in-time join of multiple feature views onto the provided dataframe, and return the full resulting dataframe.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 111} id="xgniqdUWa56m" outputId="55bf0004-a386-43c6-8717-0eac6b4c98a9"
train.sample(2)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Ho7ddXxddSay" outputId="c0ab1d17-895e-4d2f-8db7-a95ff956f789"
present_time
```

```python colab={"base_uri": "https://localhost:8080/", "height": 578} id="j9p3IDrGa5I-" outputId="30d0c398-4321-4711-e03f-0225e5721e07"
from feast import FeatureStore
import pandas as pd
from datetime import datetime

present_time = pd.Timestamp(datetime.now(), tz="UTC")

entity_df = pd.DataFrame.from_dict({
    "userid": [212, 390],
    "itemid": [408, 53],
    "datetime": [present_time,
                        present_time]
})

store = FeatureStore(repo_path=".")

training_df = store.get_historical_features(
    entity_df=entity_df, 
    feature_refs = [
        'movielens_train:RATING',
        # 'movielens_train:TIMESTAMP',
    ],
).to_df()

training_df
```

```python id="PMAjXPbvaYnY"
from feast import FeatureStore
import pandas as pd
from datetime import datetime

entity_df = pd.DataFrame.from_dict({
    "driver_id": [1001, 1002, 1003, 1004],
    "event_timestamp": [
        datetime(2021, 4, 12, 10, 59, 42),
        datetime(2021, 4, 12, 8,  12, 10),
        datetime(2021, 4, 12, 16, 40, 26),
        datetime(2021, 4, 12, 15, 1 , 12)
    ]
})

store = FeatureStore(repo_path=".")

training_df = store.get_historical_features(
    entity_df=entity_df, 
    feature_refs = [
        'driver_hourly_stats:conv_rate',
        'driver_hourly_stats:acc_rate',
        'driver_hourly_stats:avg_daily_trips'
    ],
).to_df()
```

```python id="MdPaCw_ZQbia"
from feast import FeatureStore
import pandas as pd
from datetime import datetime

entity_df = pd.DataFrame.from_dict({
    "driver_id": [1001, 1002, 1003, 1004],
    "event_timestamp": [
        datetime(2021, 4, 12, 10, 59, 42),
        datetime(2021, 4, 12, 8,  12, 10),
        datetime(2021, 4, 12, 16, 40, 26),
        datetime(2021, 4, 12, 15, 1 , 12)
    ]
})

store = FeatureStore(repo_path=".")

training_df = store.get_historical_features(
    entity_df=entity_df, 
    feature_refs = [
        'driver_hourly_stats:conv_rate',
        'driver_hourly_stats:acc_rate',
        'driver_hourly_stats:avg_daily_trips'
    ],
).to_df()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 261} id="iqMZM-xQQ6vC" outputId="f784eabd-b3af-4dcb-a363-650efe45eb9b"
training_df.head()
```

<!-- #region id="GQAaubc5RmnR" -->
### Load feature values into your online store
<!-- #endregion -->

```sh colab={"base_uri": "https://localhost:8080/"} id="s8OrKWPvRz-0" outputId="63ff1864-5bf4-4476-8aaa-f6d5bf505819"
CURRENT_TIME=$(date -u +"%Y-%m-%dT%H:%M:%S")
feast materialize-incremental $CURRENT_TIME
```

<!-- #region id="CIJCDiEMSV4m" -->
### Read online features at low latency
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="szaiNCXURJgt" outputId="2b877525-0e8e-4c42-cbbf-1e0ea4372a51"
from pprint import pprint
from feast import FeatureStore

store = FeatureStore(repo_path=".")

feature_vector = store.get_online_features(
    feature_refs=[
        'driver_hourly_stats:conv_rate',
        'driver_hourly_stats:acc_rate',
        'driver_hourly_stats:avg_daily_trips'
    ],
    entity_rows=[{"driver_id": 1001}]
).to_dict()

pprint(feature_vector)  
```

<!-- #region id="eHMPt2cfAAEu" -->
## Ad-click dataset
<!-- #endregion -->

<!-- #region id="r2MjWFcbSF-7" -->
### Download the dataset
<!-- #endregion -->

<!-- #region id="x1n_POn5AG9L" -->
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="moZd0KLNAL9l" outputId="bc89560e-976e-49af-a8b2-9067931ed381"
!pip install -q -U kaggle
!pip install --upgrade --force-reinstall --no-deps kaggle
!mkdir ~/.kaggle
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d arashnic/ctrtest
!unzip ctrtest
```

```python id="p7MKxMvJBYRz"
import os
import pandas as pd
from datetime import datetime
from feast import FeatureStore
from feast import Entity, ValueType, Feature, FeatureView
from feast.data_format import ParquetFormat
from feast.data_source import FileSource
from google.protobuf.duration_pb2 import Duration
```

<!-- #region id="MX6Bv1x6C1DV" -->
### Initializing the feature store
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="k46fEozUC2nP" outputId="2ba334fb-942d-4470-a810-7ff85e8945bb"
!feast init click_data
%cd click_data
```

<!-- #region id="xpoDsN6bCzAc" -->
### ETL
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="WJ_g0usFB7R2" outputId="d6187d44-5f92-4d2d-a84a-7559aac3868c"
data = pd.read_csv("/content/train_adc/train.csv")
#Convert it to datetime before writing to quaaquet.
data['impression_time'] = pd.to_datetime(data['impression_time'])
data.head()
```

```python id="v22kUx4OCDHg"
data.to_parquet("./data/train.parquet")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="Qa1AeoakCGY1" outputId="d55f6cf0-0d4b-4405-bfee-81d86fbce675"
item = pd.read_csv("/content/train_adc/item_data.csv")
item.to_parquet("./data/item_data.parquet")
item.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="tBlU-8KICGVi" outputId="4eecc6f8-9425-4aa2-c089-fb71d3c7d6cc"
view_log = pd.read_csv("/content/train_adc/view_log.csv")
view_log['server_time'] = pd.to_datetime(view_log['server_time'])
view_log.to_parquet("./data/view_log.parquet")
view_log.head()
```

```python id="p4FrToWQBYU2"
os.environ["TRAIN_DATA"] = "./data/view_log.parquet"
os.environ["ITEM_DATA"] = "./data/item_data.parquet"
os.environ["VIEW_LOG_DATA"] = "./data/view_log.parquet"
```

<!-- #region id="u8YXLod8AfcG" -->
### Re-write the featureTable definition as feature store
<!-- #endregion -->

```python id="Idy4E5rGAQ8v"
class ContextAdClickData:

    def __init__(self) -> None:
        self.features = {}

    def train_view_source(self):
        return FileSource(
            event_timestamp_column="impression_time",
                # created_timestamp_column="created",
            file_format=ParquetFormat(),
            path=os.environ.get("TRAIN_DATA"),
        )
    
    def item_data_view_source(self):
        return FileSource(
            file_format=ParquetFormat(),
            path=os.environ.get("ITEM_DATA")
            # path="s3://{bucket_name}/data/item_data.parquet"
        )
    
    def view_log_data_view_source(self):
        return FileSource(
            event_timestamp_column="server_time",
            file_format=ParquetFormat(),
            path=os.environ.get("VIEW_LOG_DATA")
        )

    def trainView(self):
        """Defines the train table for the click data.
        :params:
            - column_type_dict - A dictionary of columns and the data type
        
        """
        name = "train_table"
        return FeatureView(
            name=name,
            entities=[self.train_entity().name],
            ttl=Duration(seconds=86400 * 1),
            features=[
                self.feature_create("user_id", ValueType.STRING),
                self.feature_create("impression_id", ValueType.STRING),
                self.feature_create("app_code", ValueType.INT32),
                self.feature_create("os_version", ValueType.STRING),
                self.feature_create("is_4G", ValueType.INT32),
                self.feature_create("is_click", ValueType.INT32),
            ],
            online=True,
            input=self.train_view_source(),
            tags={}
        )
    
    def viewLogView(self):
        name = "view_log_table"
        return FeatureView(
            name=name,
            entities=[self.view_log_entity().name],
            ttl=Duration(seconds=86400 * 1),
            features=[
                # self.feature_create("server_time", ValueType.UNIX_TIMESTAMP),
                self.feature_create("device_type", ValueType.STRING),
                # self.feature_create("session_id", ValueType.INT32),
                self.feature_create("user_id", ValueType.INT64),
                self.feature_create("item_id", ValueType.INT64)
            ],
            online=True,
            input=self.view_log_data_view_source(),
            tags={}
        )

    def itemDataView(self):
        name = "item_data_table"
        feature_table = FeatureView(
            name=name,
            entities=[self.item_data_entity().name],
            ttl=Duration(seconds=86400 * 1),
            features=[
                self.feature_create("item_id", ValueType.INT32),
                self.feature_create("item_price", ValueType.INT32),
                self.feature_create("category_1", ValueType.INT32),
                self.feature_create("category_2", ValueType.INT32),
                self.feature_create("category_3", ValueType.INT32),
                self.feature_create("product_type", ValueType.INT32)
            ],
            online=True,
            input=self.item_data_view_source(),
            tags={}
        )
        return feature_table
        
    
    def train_entity(self):
        name = "impression_id"
        return Entity(name, value_type=ValueType.INT32, description="Impression logs with click details")

    def view_log_entity(self):
        name = "session_id"
        #TODO: Check how to merge the user_id in this entity and user id in click entity.
        return Entity(name=name, value_type=ValueType.INT64, description="View log containing user_id and item_id being viewed")
    
    def item_data_entity(self):
        name="item_id"
        return Entity(name=name, value_type=ValueType.INT32, description="Item data")

    def feature_create(self, name, value):
        """Add features """
        self.features[name] = Feature(name, dtype=value)
        assert name in self.features
        return self.features[name]
```

```python id="Pa6mY1iuA5g7"
addClick = ContextAdClickData()

en_train = addClick.train_entity()
en_item = addClick.item_data_entity()
en_view_log = addClick.view_log_entity()

x = addClick.trainView()
y = addClick.itemDataView()
z = addClick.viewLogView()
```

<!-- #region id="JerMdLxBIq4Q" -->
### Registering the features to local feature store
<!-- #endregion -->

```python id="T3GekxbrCGSH"
store = FeatureStore(repo_path=".")
```

```python id="0sspvdoXGuQH"
store.apply([x,en_train])
# store.apply([y,en_item])
store.apply([z,en_view_log])
```

<!-- #region id="DCKG3RWeIj8S" -->
### Retrieving some features from local store
<!-- #endregion -->

```python id="H07v33CHCGOP"
entity_df = pd.DataFrame.from_dict({
    "session_id": [218564],
    "event_timestamp" : datetime(2018, 10, 15, 8, 58, 00),
})
```

```python colab={"base_uri": "https://localhost:8080/", "height": 80} id="hWFHO2lEGk5P" outputId="e9113179-84f3-4a4e-9e7d-1e4e2b46abdb"
data_df = store.get_historical_features(feature_refs=["view_log_table:device_type"], entity_df=entity_df)
ex_data = data_df.to_df()
ex_data.head()
```
