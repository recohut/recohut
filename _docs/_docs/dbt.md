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

# dbt Postgres Data Pipeline

<!-- #region id="ieYrSBLcemsE" -->
### Setup the environment
<!-- #endregion -->

<!-- #region id="DXuYcyLZepGJ" -->
Install postgresql
<!-- #endregion -->

```python id="zAWMOlX6dXEB"
# Install postgresql server
!sudo apt-get -y -qq update
!sudo apt-get -y -qq install postgresql
!sudo service postgresql start

# Setup a password `postgres` for username `postgres`
!sudo -u postgres psql -U postgres -c "ALTER USER postgres PASSWORD 'postgres';"
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 397, "status": "ok", "timestamp": 1629965694730, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="qLI7IpcQhpZj" outputId="0c14e3cc-d84b-40db-bb5f-5bc93a1598d0"
! lsof -i -P -n | grep -E 'postgres'
```

<!-- #region id="xHWbsmgkerS3" -->
Install dbt
<!-- #endregion -->

```python id="4gTjWyOnXJCo"
!pip install dbt
```

<!-- #region id="ZrTpTt0_eshL" -->
### Initiate a project
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4233, "status": "ok", "timestamp": 1629964857651, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="9cCwoOXLdGQM" outputId="207e75c5-1bf9-41ed-f079-1113c55d1520"
!dbt init dbt_demo
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 443, "status": "ok", "timestamp": 1629968870259, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="SjcFqfNUed4Y" outputId="2566049a-3d71-4727-ba17-c6b86a682060"
%cd dbt_demo
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 648, "status": "ok", "timestamp": 1629968872611, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="f85UHmxAe1Nq" outputId="f00d460c-1018-4355-d36e-b99844790daf"
!ls -la
```

<!-- #region id="oCMdf_HJneig" -->
### Load sample data into database
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 13455, "status": "ok", "timestamp": 1629967656828, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="6s4J2wrenhZ2" outputId="1e4dea8f-e342-4680-e2f8-730052384a59"
# data source - https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page
!wget https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2019-01.csv
!wget https://s3.amazonaws.com/nyc-tlc/misc/taxi+_zone_lookup.csv
```

```python colab={"base_uri": "https://localhost:8080/", "height": 224} executionInfo={"elapsed": 21394, "status": "ok", "timestamp": 1629968913439, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="PAEDTktgpLFh" outputId="23ff0805-c3dd-4d99-e214-867d436307ae"
import pandas as pd

yellow_tripdata_2019_df = pd.read_csv('yellow_tripdata_2019-01.csv')
yellow_tripdata_2019_df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 8, "status": "ok", "timestamp": 1629969277623, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="thIkWH6KvU17" outputId="140df205-a737-4ed6-ca95-0fd342bc0555"
yellow_tripdata_2019_df.columns
```

```python executionInfo={"elapsed": 718, "status": "ok", "timestamp": 1629969344310, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="RcDoM8UruvKv"
yellow_tripdata_2019_df = yellow_tripdata_2019_df[['VendorID',
                                                   'tpep_pickup_datetime',
                                                   'tpep_dropoff_datetime',
                                                   'passenger_count',
                                                   'PULocationID',
                                                   'DOLocationID',
                                                   'fare_amount']]

yellow_tripdata_2019_df.columns = ['vendor_id',
                                   'pickup_datetime',
                                   'dropoff_datetime',
                                   'passenger_count',
                                   'pickup_location_id',
                                   'dropoff_location_id',
                                   'fare_amount']
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 803, "status": "ok", "timestamp": 1629969345106, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="YhZC2HuOpZiw" outputId="ab5a824c-3151-4b97-b85a-7e81ce3183f3"
yellow_tripdata_2019_df.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"elapsed": 20, "status": "ok", "timestamp": 1629968913442, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="2kg7Y_SVpUy1" outputId="e7ddbfb6-d8dd-4f8c-f3dd-209e9badde43"
taxi_zone_lookup = pd.read_csv('taxi+_zone_lookup.csv')
taxi_zone_lookup.head()
```

```python executionInfo={"elapsed": 545, "status": "ok", "timestamp": 1629969073968, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="bz1BV60XuVuk"
taxi_zone_lookup.columns = ['locationid','borough','zone','service_zone']
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1629969074668, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="shIUzXb4plKQ" outputId="7b93baaf-461d-4738-8ef7-af2c0d2d032c"
taxi_zone_lookup.info()
```

```python executionInfo={"elapsed": 7775, "status": "ok", "timestamp": 1629969373576, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="l9WRwJxtr2qF"
from sqlalchemy import create_engine
import psycopg2

alchemyEngine = create_engine('postgresql+psycopg2://postgres:postgres@127.0.0.1/postgres', pool_recycle=3600);
postgreSQLConnection = alchemyEngine.connect();

yellow_tripdata_2019_df.sample(100000).to_sql('yellow_tripdata_sample_2019_01', postgreSQLConnection, if_exists='replace');
taxi_zone_lookup.to_sql('taxi_zone_lookup', postgreSQLConnection, if_exists='replace');

postgreSQLConnection.close();
```

<!-- #region id="99UMBNYmei1-" -->
### Configure the project
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 442, "status": "ok", "timestamp": 1629965089113, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="p8yo_soFfCLW" outputId="eaa9736d-5e51-4c92-f7ca-359dd8b67f11"
!ls ~ -la
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 440, "status": "ok", "timestamp": 1629965096018, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="dBjIMuu8fPK3" outputId="4a7e0a86-304a-4aa7-f830-975e38e62fa4"
!ls ~/.dbt -la
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 689, "status": "ok", "timestamp": 1629965722736, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="S6TM0X6xfY1R" outputId="681b3f2c-602b-480d-8f0a-116dc7ab4705"
%%writefile ~/.dbt/profiles.yml
default:
  outputs:
    dev:
      type: postgres
      threads: 1
      host: localhost
      port: 5432
      user: postgres
      pass: postgres
      dbname: postgres
      schema: public
    prod:
      type: postgres
      threads: 1
      host: localhost
      port: 5432
      user: postgres
      pass: postgres
      dbname: postgres
      schema: public
  target: dev
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3020, "status": "ok", "timestamp": 1629965725750, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="e5t4mCVTgeUv" outputId="611f755e-0721-4bfc-e28a-24f34c2d7577"
!dbt debug
```

<!-- #region id="2L0EYNWrghvR" -->
### Run dbt sample models
<!-- #endregion -->

```python id="at-m9KasipUV"
!apt-get --quiet install tree
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 535, "status": "ok", "timestamp": 1629965974370, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="f5pKLI7Kh3zU" outputId="45b7cc80-2f7b-46fa-9910-97cc9945fda0"
!tree ./models
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3851, "status": "ok", "timestamp": 1629965997475, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="pxox1H_ZibE5" outputId="417f7887-ee94-4051-f1b2-c87099a64c7f"
!dbt run
```

<!-- #region id="BOHnBgCDiz-P" -->
### Create new models
<!-- #endregion -->

<!-- #region id="SE-QibiDkjDa" -->
Staging

These files are so-called staging models, a pattern commonly used in dbt to prevent access to raw data. Staging models typically simply select from the source data and, if needed, contain some light transformations such as column renamings.
<!-- #endregion -->

```python executionInfo={"elapsed": 394, "status": "ok", "timestamp": 1629966153598, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="wrT8Q1_Ti6rr"
!mkdir -p ./models/taxi/staging
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1629966221261, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="9LlF89CNjbM4" outputId="39c4bc70-2824-4366-f987-a4c2244f719b"
%%writefile ./models/taxi/staging/schema_staging.yml
version: 2

sources:
  - name: source
    schema: public
    tables:
      - name: taxi_zone_lookup
      - name: yellow_tripdata_sample_2019_01

models:
  - name: stg_taxi_zone_lookup
    description: "A list of all taxi zones with codes in NYC"
    columns:
      - name: locationid
        tests:
          - not_null
      - name: borough
        tests:
          - not_null
      - name: zone
        tests:
          - not_null
      - name: service_zone
        tests:
          - not_null
  - name: stg_taxi_trips
    description: "A reduced version of yellow taxi trip data in NYC"
    columns:
      - name: vendor_id
        tests:
          - not_null
          - accepted_values:
              values: ['1', '2', '4']
      - name: pickup_datetime
        tests:
          - not_null
      - name: dropoff_datetime
        tests:
          - not_null
      - name: passenger_count
        tests:
          - not_null
      - name: pickup_location_id
        tests:
          - not_null
      - name: dropoff_location_id
        tests:
          - not_null
      - name: fare_amount
        tests:
          - not_null
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 444, "status": "ok", "timestamp": 1629966966181, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="ZAy8m9y3jxrV" outputId="3ec1fe2c-b23c-45c7-c8fe-28ec23638d2b"
%%writefile ./models/taxi/staging/stg_taxi_trips.sql
select 
    vendor_id,
    pickup_datetime, 
    dropoff_datetime, 
    passenger_count, 
    pickup_location_id, 
    dropoff_location_id, 
    fare_amount
from {{ source('source', 'yellow_tripdata_sample_2019_01') }}
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 426, "status": "ok", "timestamp": 1629966962997, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="sb1m8tGZjruc" outputId="4c53579b-2e98-411e-e7e8-191e2afe24f3"
%%writefile ./models/taxi/staging/stg_taxi_zone_lookup.sql
select 
    locationid,
    borough,
    zone,
    service_zone
from {{ source('source', 'taxi_zone_lookup') }}
```

<!-- #region id="KJSrshQRk2d-" -->
Create a New Model

We will now create our first dbt model, which combines data from the two staging models. Let's assume we want to write a query to join the staging tables on the location ID fields and add the actual location names to the pickup and dropoff locations of the taxi ride data.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 436, "status": "ok", "timestamp": 1629966622895, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="b4fUtbZWkPtC" outputId="9f979478-e2e3-48aa-bb65-121ba94d0585"
%%writefile ./models/taxi/trips_with_borough_name.sql
select
    t.vendor_id,
    t.pickup_datetime,
    t.dropoff_datetime,
    z1.borough as pickup_borough,
    z2.borough as dropoff_borough,
    t.passenger_count,
    t.fare_amount
from {{ ref('stg_taxi_trips') }} t
left join {{ ref('stg_taxi_zone_lookup') }} z1
on t.pickup_location_id = z1.locationid
left join {{ ref('stg_taxi_zone_lookup') }} z2
on t.dropoff_location_id = z2.locationid
```

<!-- #region id="NoyupjXklLU3" -->
Create the schema
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1629966624736, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="Rq_mMKiwkA23" outputId="14ed9c26-5324-4209-b098-a0585d62ca0d"
%%writefile ./models/taxi/schema.yml
version: 2

models:
  - name: trips_with_borough_name
    description: "Combines taxi rides with the borough names for pickup and dropoff locations."
    columns:
      - name: vendor_id
      - name: pickup_datetime
      - name: dropoff_datetime
      - name: pickup_borough
      - name: dropoff_borough
      - name: passenger_count
      - name: fare_amount
```

<!-- #region id="oM8qkKvWlOIf" -->
Configure the dbt_project.yml file
<!-- #endregion -->

<!-- #region id="kp6XdjGZlcaR" -->
This file defines how the data in the models under taxi will be materialized. Modify the section to replace example with our new taxi directory:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 635, "status": "ok", "timestamp": 1629966797805, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="UBcQEMMylYoQ" outputId="50e52821-196e-42ce-c547-02f2bfebe0ba"
!cat dbt_project.yml
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1768, "status": "ok", "timestamp": 1629966874828, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="x-gkTNPZltCs" outputId="ce836e82-5aad-4676-c6bb-79513dafc62b"
%%writefile dbt_project.yml
name: 'my_new_project'
version: '1.0.0'
config-version: 2
profile: 'default'
source-paths: ["models"]
analysis-paths: ["analysis"]
test-paths: ["tests"]
data-paths: ["data"]
macro-paths: ["macros"]
snapshot-paths: ["snapshots"]
target-path: "target"
clean-targets:
    - "target"
    - "dbt_modules"
models:
  my_new_project:
      taxi:
          materialized: view
```

<!-- #region id="UE4StlwSmKsk" -->
Run the new pipeline
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4272, "status": "ok", "timestamp": 1629969385852, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="GLIig3GdmN3Y" outputId="71ac2e2e-b5ea-4e66-cc66-b38534257895"
!dbt run
```

<!-- #region id="CHU-h2wUmOpQ" -->
### Running tests
<!-- #endregion -->

<!-- #region id="eOuPs6yiv9Jv" -->
Run the staging schema tests
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4455, "status": "ok", "timestamp": 1629969487449, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="w4qtE15mwBRm" outputId="c4178908-f686-4e65-d448-07a918628bd7"
!dbt test -m stg_taxi_trips stg_taxi_zone_lookup
```

<!-- #region id="pBoxls4-wIEu" -->
Correct the failure
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 9, "status": "ok", "timestamp": 1629969847968, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="mmIdkeQdxP_a" outputId="f05424f5-7507-4fac-8ca9-fbdce6d5718f"
%%writefile ./models/taxi/staging/stg_taxi_zone_lookup.sql
select 
    locationid,
    borough,
    zone,
    service_zone
from {{ source('source', 'taxi_zone_lookup') }} lk
where lk.zone is not null
and lk.service_zone is not null
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4335, "status": "ok", "timestamp": 1629969857229, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="BDSN2kFGxhF5" outputId="f0c0e3da-a458-42a9-fb75-330fd1bda60d"
!dbt run
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4331, "status": "ok", "timestamp": 1629969875516, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="F_xkJ7S4xidU" outputId="3b862c06-0bbc-4135-90e1-a5ce44410968"
!dbt test -m stg_taxi_trips stg_taxi_zone_lookup
```
