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

# Running Great Expectations Pipeline in Airflow

<!-- #region id="6TXEb1JWzHz3" -->
### Git repo
<!-- #endregion -->

```python executionInfo={"elapsed": 19, "status": "ok", "timestamp": 1630003853977, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="U57Wh7-QwikM"
import os
project_name = "reco-tut-de"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2080, "status": "ok", "timestamp": 1630003856042, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="x8AO_wohwn9G" outputId="f8b53f56-04dd-44a9-f798-ff2a23b389f7"
if not os.path.exists(project_path):
    !cp /content/drive/MyDrive/mykeys.py /content
    import mykeys
    !rm /content/mykeys.py
    path = "/content/" + project_name; 
    !mkdir "{path}"
    %cd "{path}"
    import sys; sys.path.append(path)
    !git config --global user.email "recotut@recohut.com"
    !git config --global user.name  "reco-tut"
    !git init
    !git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout main
else:
    %cd "{project_path}"
```

```python id="OrqXufqUwn9J"
!git status
```

```python id="XhSPdtgR3Wkj"
!git add .
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 739, "status": "ok", "timestamp": 1630005535651, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="TsOUxoxw5n42" outputId="dba12023-28f0-4e1e-ed28-9a3553bb354e"
!git pull --rebase origin main
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1351, "status": "ok", "timestamp": 1630009796241, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="BPVM-5NKwn9K" outputId="9e5b5d27-eb87-4c48-829d-08a486a720d0"
!git commit -m 'commit' && git push origin "{branch}"
```

<!-- #region id="9dH5eFOtuDEw" -->
### Setup the Environment
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

<!-- #region id="Sp3DATHDwdXh" -->
Install the libraries
<!-- #endregion -->

```python id="n0pVMieTub-V"
!pip install -q dbt apache-airflow great-expectations sqlalchemy psycopg2-binary
```

<!-- #region id="cmv-PNsh0QNs" -->
Initialize the airflow database
<!-- #endregion -->

```python id="Ij7Cvpui0TyH"
!airflow db init
```

<!-- #region id="W-WuS_KN0Zsw" -->
Setup the airflow credentials
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 5410, "status": "ok", "timestamp": 1630004176508, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="zRAUB2FI0cW_" outputId="098e4da4-197b-4981-fbed-6f5fe126e5c7"
!airflow users create \
    --username admin \
    --firstname Firstname \
    --lastname Lastname \
    --role Admin \
    --email admin@example.org \
    --password password
```

<!-- #region id="ewgzAO-o70pu" -->
Start Airflow in the background
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 5052, "status": "ok", "timestamp": 1630006140838, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="TFGaRFs49icl" outputId="d392f4f9-4120-4abb-d0f9-b1d70f9a8ff1"
!airflow webserver --port 8080 -D
!nohup airflow scheduler &
!cat nohup.out
```

<!-- #region id="DPd50nZH2JGO" -->
Initialize dbt
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4596, "status": "ok", "timestamp": 1630004626488, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="ZdhDbvaA2KqU" outputId="d9fac670-6132-4e48-93a3-3958ad6f33ea"
!dbt init dbt_demo
```

<!-- #region id="DXpGNgIA30xp" -->
Initialize great expectations
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 27861, "status": "ok", "timestamp": 1630005097686, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="T08OIdu7333M" outputId="a9d69306-818b-4c0a-ed89-5245a16af2a7"
!great_expectations --v3-api init
```

<!-- #region id="oCMdf_HJneig" -->
### Load sample data into database
<!-- #endregion -->

```python id="6s4J2wrenhZ2"
# data source - https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page
!wget https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2019-01.csv
!wget https://s3.amazonaws.com/nyc-tlc/misc/taxi+_zone_lookup.csv
```

```python colab={"base_uri": "https://localhost:8080/", "height": 224} executionInfo={"elapsed": 17617, "status": "ok", "timestamp": 1630002803787, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="PAEDTktgpLFh" outputId="8247e33f-09b8-4461-bd67-66ed71a7cafc"
import pandas as pd

yellow_tripdata_2019_df = pd.read_csv('yellow_tripdata_2019-01.csv')
yellow_tripdata_2019_df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 58, "status": "ok", "timestamp": 1630002803788, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="thIkWH6KvU17" outputId="4c7ca5f2-8955-47dc-869f-24ab6a66b313"
yellow_tripdata_2019_df.columns
```

```python executionInfo={"elapsed": 49, "status": "ok", "timestamp": 1630002803789, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="RcDoM8UruvKv"
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

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 43, "status": "ok", "timestamp": 1630002803790, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="YhZC2HuOpZiw" outputId="ed5f171d-97c8-47b8-9dbb-41067cdadefd"
yellow_tripdata_2019_df.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"elapsed": 36, "status": "ok", "timestamp": 1630002803791, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="2kg7Y_SVpUy1" outputId="953857a8-2712-48ce-92ff-57d78c807184"
taxi_zone_lookup = pd.read_csv('taxi+_zone_lookup.csv')
taxi_zone_lookup.head()
```

```python executionInfo={"elapsed": 32, "status": "ok", "timestamp": 1630002803792, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="bz1BV60XuVuk"
taxi_zone_lookup.columns = ['locationid','borough','zone','service_zone']
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 30, "status": "ok", "timestamp": 1630002803793, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="shIUzXb4plKQ" outputId="95002a99-f8de-4f88-8736-29cd2590da14"
taxi_zone_lookup.info()
```

```python id="l9WRwJxtr2qF"
from sqlalchemy import create_engine
import psycopg2

alchemyEngine = create_engine('postgresql+psycopg2://postgres:postgres@127.0.0.1/postgres', pool_recycle=3600);
postgreSQLConnection = alchemyEngine.connect();

yellow_tripdata_2019_df.sample(100000, random_state=42).to_sql('yellow_tripdata_sample_2019_01', postgreSQLConnection, if_exists='replace');
taxi_zone_lookup.to_sql('taxi_zone_lookup', postgreSQLConnection, if_exists='replace');

postgreSQLConnection.close();
```

<!-- #region id="EtLcS8dGvLPE" -->
### Analyze the environment
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 5110, "status": "ok", "timestamp": 1630003902226, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="x1if1HG0zXDd" outputId="4418304b-0035-4a3a-88c4-664c0bb73529"
!apt-get -q install tree
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 507, "status": "ok", "timestamp": 1630003932447, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="77sdaDAkzZgu" outputId="1b3ef92b-dca6-476c-884c-574a9283aee7"
!tree ./code/dag_stack/
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 511, "status": "ok", "timestamp": 1630003992514, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="mbKPneHLzvun" outputId="d333697f-87f7-4241-ae74-264e67656c95"
%cd ./code/dag_stack
```

<!-- #region id="Gai8llQVzejm" -->
### The airflow_dags Folder
The airflow_dags folder contains a dag.py DAG definition file. The DAG currently only defines a single task using the DummyOperator.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 493, "status": "ok", "timestamp": 1630004020462, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="9mAw_IQXzt61" outputId="927a122b-d35c-4022-87b8-c44b0a1ac850"
!cat airflow_dags/dag.py
```

<!-- #region id="Y_WIEujRzoj9" -->
Change the dag path in airflow configuration and list out the dags
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 728, "status": "ok", "timestamp": 1630004337353, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="WevJ7meT07dC" outputId="7f011d83-7794-4d85-ecdb-0b04cd5b7d4a"
%env AIRFLOW__CORE__DAGS_FOLDER=./airflow_dags
!airflow config get-value core dags_folder
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3241, "status": "ok", "timestamp": 1630004343352, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="maVvb4jT0H4h" outputId="c1341edf-56ed-4ea3-d058-1a896f5fdcdc"
!airflow dags list
```

<!-- #region id="0ryCE8bo0JxG" -->
### The dbt folder
<!-- #endregion -->

```python executionInfo={"elapsed": 655, "status": "ok", "timestamp": 1630004767268, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="MSc5wMnj19Bk"
!cp ./dbt/profiles.yml ~/.dbt/profiles.yml
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3187, "status": "ok", "timestamp": 1630004986032, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="cNCeUx3p2wVx" outputId="ba3b1d28-08ec-4a9a-a812-23a4143377b1"
!dbt debug --project-dir ./dbt
```

<!-- #region id="F7yvi_Ry3jF7" -->
### The great_expectations folder
<!-- #endregion -->

<!-- #region id="bi70PoUF6BtF" -->
Confirm that the Great Expectations component is set up correctly by listing out the individual assets:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 10042, "status": "ok", "timestamp": 1630005660599, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="cqyudsI5f1Tj" outputId="ebd89e2c-f21f-4542-d50e-5f07c33f0b72"
!great_expectations --v3-api datasource list
!great_expectations --v3-api suite list
!great_expectations --v3-api checkpoint list
```

<!-- #region id="eIWjj-Bq6GF5" -->
## Add a Great Expectations Task to the DAG
In this step, you will create a new task in the Airflow DAG to run validation of the source data table using a Great Expectations Checkpoint. While a GreatExpectationsOperator for Airflow does exist, it currently does not support the v3 API in Great Expectations. Hence, we will trigger a Checkpoint run with a BashOperator.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 509, "status": "ok", "timestamp": 1630005841208, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="21C8iyVr6y7I" outputId="d4684a2b-b5a0-4d43-e48b-e380a15a5335"
!cat ./airflow_dags/dag.py
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 493, "status": "ok", "timestamp": 1630006902032, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="URHPM27E6nZz" outputId="994df978-862f-4e0a-eb6d-dd767e01b5a3"
%%writefile ./airflow_dags/dag.py
from airflow.operators.bash import BashOperator
from airflow import DAG

from datetime import datetime

# Default settings applied to all tasks
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'catchup': False,
    'start_date': datetime(2021, 1, 1)
}

with DAG(
    dag_id='dag_stack',
    description='An example Airflow DAG running dbt and Great Expectations tasks',
    schedule_interval=None,
    default_args=default_args
    ) as dag:

    validate_load = BashOperator(
        task_id='validate_load',
        bash_command='great_expectations --v3-api --config /content/reco-tut-de/code/dag_stack/great_expectations checkpoint run my_checkpoint_01; exit 99;'
    )
```

<!-- #region id="K7k8HsSY7QuT" -->
The Checkpoint my_checkpoint_01 is configured to test the yellow_tripdata_sample_2019_01 database table, which will pass validation. You can confirm that the validate_load task runs correctly by executing the following command in the Terminal:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3379, "status": "ok", "timestamp": 1630009674717, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="w626Q9NJ9egU" outputId="a86cd925-33f7-4df8-998d-7e80ae40dbde"
!airflow db init
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 5808, "status": "ok", "timestamp": 1630009680500, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="MhXASido7Yc5" outputId="b3f313cd-35ce-4e90-cb21-5d60664fd0c0"
!airflow tasks test dag_stack validate_load 2021-01-01
```

<!-- #region id="PY2U4om67fDa" -->
## Add a dbt Task to the DAG
In this step, you will create a new task in the Airflow DAG to trigger a dbt run after the source data has been validated. This will use the DbtRunOperator in Airflow.
<!-- #endregion -->

```python id="i8ZsoX7t_CRI"
!pip install --upgrade-strategy only-if-needed airflow-dbt==0.3.0
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 523, "status": "ok", "timestamp": 1630007114490, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="7TyAyKL6_MLw" outputId="198f5ba5-08d8-4885-9906-e8db5afff635"
%%writefile ./airflow_dags/dag.py
from airflow_dbt.operators.dbt_operator import DbtRunOperator
from airflow.operators.bash import BashOperator
from airflow import DAG

from datetime import datetime

# Default settings applied to all tasks
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'catchup': False,
    'start_date': datetime(2021, 1, 1)
}

with DAG(
    dag_id='dag_stack',
    description='An example Airflow DAG running dbt and Great Expectations tasks',
    schedule_interval=None,
    default_args=default_args
    ) as dag:

    validate_load = BashOperator(
        task_id='validate_load',
        bash_command='great_expectations --v3-api --config /content/reco-tut-de/code/dag_stack/great_expectations checkpoint run my_checkpoint_01; exit 99;'
    )

    dbt_run = DbtRunOperator(
        task_id='dbt_run',
        dir='/content/reco-tut-de/code/dag_stack/dbt',
        target='dev'
    )

    validate_load >> dbt_run
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 6029, "status": "ok", "timestamp": 1630007160173, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="fIF2Jg20_rV9" outputId="229077b4-137c-4d41-dec1-76b6a559d9e1"
!airflow tasks test dag_stack dbt_run 2021-01-01
```

<!-- #region id="2lvKRyfQ_1Mb" -->
## Trigger the pipeline
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2687, "status": "ok", "timestamp": 1630009700327, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="2k3pN5zLAUMD" outputId="0d496f77-1010-4efd-e8d6-5b769b892c9d"
!airflow dags trigger dag_stack
```

```python colab={"base_uri": "https://localhost:8080/", "height": 194} executionInfo={"elapsed": 2237, "status": "ok", "timestamp": 1630009702556, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="ytG1ZFl0AiIC" outputId="88cc8afa-0429-4e05-8da1-36d38f2e2df0"
!airflow dags show dag_stack --save example_complex.png
from IPython.display import Image
Image('example_complex.png')
```
