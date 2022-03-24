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

# Data Validation with Great Expectations toolkit

<!-- #region id="6H5J88nGebIY" -->
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

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 23, "status": "ok", "timestamp": 1629998356230, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="qLI7IpcQhpZj" outputId="5c0b854f-cfab-4bd3-89fd-b2e3e78563fd"
! lsof -i -P -n | grep -E 'postgres'
```

<!-- #region id="IsNn601ZeMjP" -->
Install great-expectations
<!-- #endregion -->

```python id="-GVRFADVdKCd"
!pip install great-expectations sqlalchemy psycopg2-binary
```

<!-- #region id="97zpQrEQeRNA" -->
Initialize the project
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 11034, "status": "ok", "timestamp": 1629998272853, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="ejKu-zNEdezr" outputId="04ff48eb-44f4-4ed5-fa37-5d12acc82940"
!mkdir ge_demo
%cd ge_demo
!great_expectations --v3-api init
```

<!-- #region id="oCMdf_HJneig" -->
### Load sample data into database
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 34472, "status": "ok", "timestamp": 1629998467209, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="6s4J2wrenhZ2" outputId="ad8d93b9-c9d1-42cf-b7bf-95a9be38e826"
# data source - https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page
!wget https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2019-01.csv
!wget https://s3.amazonaws.com/nyc-tlc/misc/taxi+_zone_lookup.csv
```

```python colab={"base_uri": "https://localhost:8080/", "height": 224} executionInfo={"elapsed": 22090, "status": "ok", "timestamp": 1630000411663, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="PAEDTktgpLFh" outputId="2890e380-b26e-4d03-dca0-a5d0d772744a"
import pandas as pd

yellow_tripdata_2019_df = pd.read_csv('yellow_tripdata_2019-01.csv')
yellow_tripdata_2019_df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 34, "status": "ok", "timestamp": 1630000411664, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="thIkWH6KvU17" outputId="37a5ed10-767f-41ee-bd00-c6d96561e3f2"
yellow_tripdata_2019_df.columns
```

```python executionInfo={"elapsed": 32, "status": "ok", "timestamp": 1630000411666, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="RcDoM8UruvKv"
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

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 32, "status": "ok", "timestamp": 1630000411667, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="YhZC2HuOpZiw" outputId="cd4a8f3a-109a-4912-d35a-0c177bda6344"
yellow_tripdata_2019_df.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"elapsed": 26, "status": "ok", "timestamp": 1630000411668, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="2kg7Y_SVpUy1" outputId="884a694c-22ba-4558-f18f-51fa08c27ca7"
taxi_zone_lookup = pd.read_csv('taxi+_zone_lookup.csv')
taxi_zone_lookup.head()
```

```python executionInfo={"elapsed": 25, "status": "ok", "timestamp": 1630000411669, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="bz1BV60XuVuk"
taxi_zone_lookup.columns = ['locationid','borough','zone','service_zone']
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 26, "status": "ok", "timestamp": 1630000411671, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="shIUzXb4plKQ" outputId="38ac2b0e-9f24-421c-8912-35389d113d48"
taxi_zone_lookup.info()
```

```python executionInfo={"elapsed": 8062, "status": "ok", "timestamp": 1630000419713, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="l9WRwJxtr2qF"
from sqlalchemy import create_engine
import psycopg2

alchemyEngine = create_engine('postgresql+psycopg2://postgres:postgres@127.0.0.1/postgres', pool_recycle=3600);
postgreSQLConnection = alchemyEngine.connect();

yellow_tripdata_2019_df.sample(100000).to_sql('yellow_tripdata_sample_2019_01', postgreSQLConnection, if_exists='replace');
taxi_zone_lookup.to_sql('taxi_zone_lookup', postgreSQLConnection, if_exists='replace');

postgreSQLConnection.close();
```

<!-- #region id="-kXRz7pWeura" -->
### Create datasource
<!-- #endregion -->

<!-- #region id="dW7Lu8Qyewg4" -->
Create a new data source configuration
<!-- #endregion -->

```python executionInfo={"elapsed": 1824, "status": "ok", "timestamp": 1629998607066, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="7wovS3DVey8Y"
!mkdir -p scripts
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 434, "status": "ok", "timestamp": 1629998700218, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="TWVyVxlxfIAd" outputId="39067003-d58e-467e-bab4-1f61b0369b45"
%%writefile ./scripts/create_datasource.py
import great_expectations as ge
from great_expectations.cli.datasource import sanitize_yaml_and_save_datasource

context = ge.get_context()

config = f"""
name: my_datasource
class_name: Datasource
execution_engine:
  class_name: SqlAlchemyExecutionEngine
  credentials:
    host: localhost
    port: '5432'
    username: postgres
    password: postgres
    database: postgres
    drivername: postgresql
data_connectors:
  default_runtime_data_connector_name:
    class_name: RuntimeDataConnector
    batch_identifiers:
      - default_identifier_name
  default_inferred_data_connector_name:
    class_name: InferredAssetSqlDataConnector
    name: whole_table"""

sanitize_yaml_and_save_datasource(context, config, overwrite_existing=True)
```

```python executionInfo={"elapsed": 4622, "status": "ok", "timestamp": 1629998753570, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="A2HnhlKQflIj"
!python ./scripts/create_datasource.py
```

<!-- #region id="hBYpqjnVfxIE" -->
Confirm that the Datasource was added correctly to the configuration file by running the following command in the Terminal tab:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3407, "status": "ok", "timestamp": 1629998806009, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="cqyudsI5f1Tj" outputId="ff090162-2542-4bdf-b61d-287035fce915"
!great_expectations --v3-api datasource list
```

<!-- #region id="DcDMR9qSf-Hg" -->
The following file has been generated using the built-in profiler that inspected the data in the yellow_tripdata_sample_2019_01 table in the PostgreSQL database and created Expectations based on the types and values that are found in the data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 454, "status": "ok", "timestamp": 1629999002233, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="X2S5wktagUkn" outputId="bd233b91-43fa-4af6-849f-8f706f21b1af"
%%writefile ./great_expectations/expectations/my_suite.json
{
    "data_asset_type": null,
    "expectation_suite_name": "my_suite",
    "expectations": [{
            "expectation_type": "expect_table_columns_to_match_ordered_list",
            "kwargs": {
                "column_list": [
                    "vendor_id",
                    "pickup_datetime",
                    "dropoff_datetime",
                    "passenger_count",
                    "pickup_location_id",
                    "dropoff_location_id",
                    "fare_amount"
                ]
            },
            "meta": {}
        },
        {
            "expectation_type": "expect_table_row_count_to_be_between",
            "kwargs": {
                "max_value": 10000,
                "min_value": 10000
            },
            "meta": {}
        },
        {
            "expectation_type": "expect_column_values_to_be_in_set",
            "kwargs": {
                "column": "vendor_id",
                "value_set": [
                    1,
                    2,
                    4
                ]
            },
            "meta": {}
        },
        {
            "expectation_type": "expect_column_values_to_not_be_null",
            "kwargs": {
                "column": "vendor_id"
            },
            "meta": {}
        },
        {
            "expectation_type": "expect_column_values_to_be_in_set",
            "kwargs": {
                "column": "passenger_count",
                "value_set": [
                    1,
                    2,
                    3,
                    4,
                    5,
                    6
                ]
            },
            "meta": {}
        },
        {
            "expectation_type": "expect_column_values_to_not_be_null",
            "kwargs": {
                "column": "passenger_count"
            },
            "meta": {}
        },
        {
            "expectation_type": "expect_column_mean_to_be_between",
            "kwargs": {
                "column": "passenger_count",
                "max_value": 1.5716,
                "min_value": 1.5716
            },
            "meta": {}
        }
    ],
    "meta": {
        "great_expectations_version": "0.13.19"
    }
}
```

<!-- #region id="F7Uo8bfFgu15" -->
### Generate Data Docs
Data Docs are HTML pages showing your Expectation Suites and validation results. Let's look at my_suite in Data Docs to see which Expectations it contains.

Run the following command to generate Data Docs:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 6760, "status": "ok", "timestamp": 1629999075306, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="fS85iRsDg0lo" outputId="11d46b68-2074-404e-d4ee-f64d820691f5"
!great_expectations --v3-api docs build --no-view
```

```python colab={"base_uri": "https://localhost:8080/", "height": 68} executionInfo={"elapsed": 5399, "status": "ok", "timestamp": 1629999565674, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="nNeV3LOOiVta" outputId="c674be75-0c33-4739-972c-0b4a57cec17a"
import portpicker
from google.colab.output import eval_js
port = portpicker.pick_unused_port()
print(eval_js("google.colab.kernel.proxyPort({})".format(port)))
%cd ./great_expectations/uncommitted/data_docs/local_site
!nohup python3 -m http.server $port &
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 523, "status": "ok", "timestamp": 1629999811381, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="XTvMxrL4jxix" outputId="b04cece6-4d0e-463b-c1cf-2d8ca843e42d"
%cd /content/ge_demo
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 535, "status": "ok", "timestamp": 1629999579349, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="xTU_5JzjhLZt" outputId="8368bf20-903f-4c4b-e80a-2bf45cc91d58"
!cat nohup.out
```

<!-- #region id="Mc1HEHyfjGf0" -->
<!-- #endregion -->

<!-- #region id="JAdN6s2jlltp" -->
## Load new data
<!-- #endregion -->

```python id="auKLcNNGmP2Y"
!wget https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2019-02.csv
```

```python id="340MgN4ulnFl"
yellow_tripdata_2019_df = pd.read_csv('yellow_tripdata_2019-02.csv')
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

```python executionInfo={"elapsed": 8027, "status": "ok", "timestamp": 1630000501885, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="sxe_4GlUmRw4"
postgreSQLConnection = alchemyEngine.connect();
yellow_tripdata_2019_df.sample(100000).to_sql('yellow_tripdata_sample_2019_02', postgreSQLConnection, if_exists='replace');
postgreSQLConnection.close();
```

<!-- #region id="nylWZf7mi7ic" -->
## Set up a Checkpoint to Run Validation

In this step, you will use your newly generated Expectation Suite to validate a new data asset. Recall that the Expectation Suite was created based on the data found in the yellow_tripdata_sample_2019_01 table. You will now create a Checkpoint that uses this suite to validate the yellow_tripdata_sample_2019_02 table and identify any unexpected differences in the data.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1349, "status": "ok", "timestamp": 1629999820314, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="ovmWbntgjQHw" outputId="6a5f2904-d5dd-4c8e-8398-9d4c354fffdd"
%%writefile ./scripts/create_checkpoint.py
from ruamel.yaml import YAML
import great_expectations as ge

yaml = YAML()
context = ge.get_context()

config = f"""
name: my_checkpoint
config_version: 1.0
class_name: SimpleCheckpoint
run_name_template: "%Y%m%d-%H%M%S-validation-run"
validations:
  - batch_request:
      datasource_name: my_datasource
      data_connector_name: default_inferred_data_connector_name
      data_asset_name: yellow_tripdata_sample_2019_02
      data_connector_query:
        index: -1
    expectation_suite_name: my_suite
"""

context.add_checkpoint(**yaml.load(config))
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3709, "status": "ok", "timestamp": 1629999831166, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="ClzbefaBjt43" outputId="eaac35fc-58bc-4e7f-95bc-6b4c0b4d4936"
!python ./scripts/create_checkpoint.py
```

<!-- #region id="BudeOXxyj4RE" -->
This will create a configuration for a new Checkpoint called my_checkpoint and save it to the Data Context of your project. In order to confirm that the Checkpoint was correctly created, run the command to list all Checkpoints in the project:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3775, "status": "ok", "timestamp": 1629999915888, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="-dIjVGkYkBtm" outputId="295a4106-c674-4be8-8c44-1465a2f9c750"
!great_expectations --v3-api checkpoint list
```

<!-- #region id="UHOLQlezkNE5" -->
### Run validation with a Checkpoint
To run the Checkpoint and validate the yellow_tripdata_sample_2019_02 with my_suite, execute:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 6092, "status": "ok", "timestamp": 1630000519843, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="UcChXpVqksyD" outputId="e0c5ad0d-52a9-43b1-ad08-1f199d5c7183"
!great_expectations --v3-api checkpoint run my_checkpoint
```

<!-- #region id="SmatuUAhkxnS" -->
This will correctly show the validation output as "Failed", meaning that Great Expectations has detected some data in this table that does not meet the Expectations in my_suite.
<!-- #endregion -->

<!-- #region id="u6mTMXMZmxBF" -->
### Inspect validation results in Data Docs
In order to see the validation results in Data Docs, you'll need to once again build the docs and start up the web server. Run the following command to build the docs and confirm with ENTER when prompted:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 20174, "status": "ok", "timestamp": 1630000615416, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="1-8Y7ukkmyca" outputId="7b41e9b3-6db8-4754-d63c-d0b118ba68d4"
!great_expectations --v3-api docs build --no-view
```

<!-- #region id="Luxx2dijmjKU" -->
Open the Data Docs site again

- You will now see an additional tab Validation Results on the index page, listing a timestamped
- Click into the first row to go to the validation results detail page.
- On the detail page, you will see that the validation run is marked as "Failed."
- Scroll down to see which Expectations failed and why.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"elapsed": 6165, "status": "ok", "timestamp": 1630000717505, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="WtwKphlanMzi" outputId="a07cd652-f718-4006-a908-97230586f92a"
import portpicker
from google.colab.output import eval_js
port = portpicker.pick_unused_port()
print(eval_js("google.colab.kernel.proxyPort({})".format(port)))
%cd ./great_expectations/uncommitted/data_docs/local_site
!nohup python3 -m http.server $port &
!cat nohup.out
```

<!-- #region id="hQgtDzRNnZk7" -->
<!-- #endregion -->

<!-- #region id="NKAWIRYfn_e4" -->
<!-- #endregion -->

```python id="ADMaFsUunZ_P"

```
