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

# Apache Airflow

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 35753, "status": "ok", "timestamp": 1629996792697, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="vIHctYpDzUnP" outputId="fa8bccad-7f44-4674-8937-4a59764a6f4b"
!pip install -q apache-airflow
```

<!-- #region id="W_QPaeOb84A5" -->
Airflow uses a relational database as a backend to store configuration data. By default, this is a SQLite database, which will be stored in ~/airflow/airflow.db. You initialize the database in your environment by running the following command in the terminal:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 15130, "status": "ok", "timestamp": 1629996807771, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="R7Rm-2J686e0" outputId="082e9832-43f1-4b23-e7f9-5d37b9e9b662"
!airflow db init
```

<!-- #region id="oocqsTwF9Ghy" -->
Next, you need to create a user that can log into the Airflow UI. Enter the following in your terminal to create a user named admin with admin permissions:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 5212, "status": "ok", "timestamp": 1629996812953, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="BpYdgTPL9G-7" outputId="8584b4aa-8ef1-4724-f2ba-2a1e9d6c6595"
!airflow users create \
    --username admin \
    --firstname Firstname \
    --lastname Lastname \
    --role Admin \
    --email admin@example.org \
    --password password
```

<!-- #region id="7mJHMCba9QsO" -->
To confirm that the configuration works correctly, you can run the Airflow web server and scheduler and log into the UI. Execute the following commands in the terminal to start the web server and scheduler:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4929, "status": "ok", "timestamp": 1629996344650, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="TFGaRFs49icl" outputId="58f9c611-9329-4594-f5d4-6218bc054130"
!airflow webserver --port 8080 -D
```

```python executionInfo={"elapsed": 23, "status": "ok", "timestamp": 1629996327694, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="0eZQSULV9mVc"
!nohup airflow scheduler &
!cat nohup.out
```

```python id="FLfepxRcFd0q"
# !airflow config list
# %env AIRFLOW__API__AUTH_BACKEND=airflow.api.auth.backend.basic_auth
# %env AIRFLOW__API__AUTH_BACKEND=airflow.api.auth.backend.default
# !airflow config get-value api auth_backend
# %env AIRFLOW__API__ENABLE_EXPERIMENTAL_API=True
# %env AIRFLOW__API__AUTH_BACKEND=airflow.api.auth.backend.default
# %env AIRFLOW__WEBSERVER__RBAC=False
# %env AIRFLOW__WEBSERVER__AUTHENTICATE=False
```

<!-- #region id="wN0gaN1EZLyr" -->
### Create a new Airflow DAG
<!-- #endregion -->

```python executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1629996890419, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="i-c2PlxDCfp4"
!mkdir -p airflow_demo/dags
```

<!-- #region id="ZDTjF_TRY2Ha" -->
You will now add some Python code to create a minimal definition for a DAG called airflow_demo with two basic Bash tasks:

- t0: Runs echo to print some output.
- t1: Runs sleep for 5 seconds, then prints the date.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2313, "status": "ok", "timestamp": 1629996990561, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="P0BtmReKYeSu" outputId="86b61ac4-e369-4fab-e39a-ce1e42dd3bc1"
%%writefile ./airflow_demo/dags/dag.py
from airflow import DAG
from airflow.operators.bash import BashOperator
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
    dag_id='airflow_demo_dag',
    description='An example Airflow DAG',
    schedule_interval=None,
    default_args=default_args
    ) as dag:

    t0 = BashOperator(
        task_id='bash_task_0',
        bash_command='echo "Hi there, this is the first Airflow task!"'
    )

    t1 = BashOperator(
        task_id='bash_task_1',
        bash_command='echo "Sleeping..." && sleep 5s && date'
    )

    t0 >> t1
```

```python executionInfo={"elapsed": 1429, "status": "ok", "timestamp": 1629997010668, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="Hkw_70VAZFAO"
!python ./airflow_demo/dags/dag.py
```

<!-- #region id="OsQFyzGJZDBh" -->
This will check for Python syntax errors in the file. If successful, there will be no output. Once you have confirmed that the file is error-free, proceed to the next step.
<!-- #endregion -->

<!-- #region id="8wFHPlODZPct" -->
### Run the DAG
<!-- #endregion -->

<!-- #region id="jOJPyHMmZRVO" -->
In order for the Airflow scheduler to find the new DAG file, you need to modify the configuration file.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1535, "status": "ok", "timestamp": 1629997264178, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="YOTbivT5ZYbX" outputId="33e6e345-260e-4a8b-9224-fbf46c3f5f6e"
%env AIRFLOW__CORE__DAGS_FOLDER=/content/airflow_demo/dags
!airflow config get-value core dags_folder
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2058, "status": "ok", "timestamp": 1629997267180, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="vBKb2_uhZp-Y" outputId="a4534f8a-3b07-4814-ae35-2a603709eadd"
!airflow dags list
```

<!-- #region id="SA4EUEXdZeVn" -->
Finally, in order to run the new DAG, you will need to reinitialize the Airflow database in order for it to pick up the new DAG. Execute the following command in the terminal:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2780, "status": "ok", "timestamp": 1629997358030, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="Kr62X6ShaSMa" outputId="4cca3e38-61b3-4710-fe1a-42c2a5d293e1"
!airflow db init
```

<!-- #region id="mq7vBvJkaZcp" -->
In general use, you can skip this step when the scheduler is running and picks up the DAG at a pre-defined interval, but in this example, we manually reinitialize the database.

Then trigger the DAG run:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2073, "status": "ok", "timestamp": 1629997399334, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="zsYrhQzKabK9" outputId="c2fafcce-b9f0-4e80-ed60-480622f554fc"
!airflow dags trigger airflow_demo_dag
```

<!-- #region id="LepoU_GKaYBA" -->
This will run your new DAG and will show the output of the echo statement and the date output in the terminal.
<!-- #endregion -->

<!-- #region id="bb3RykwLaaw4" -->
### UI

> Note: Not worked in colab but image snips from 3rd party run
<!-- #endregion -->

<!-- #region id="InGQP67BbqoE" -->
<!-- #endregion -->

<!-- #region id="2paCYvOEb71s" -->
<!-- #endregion -->

<!-- #region id="UZhOftQ0cYQe" -->
<!-- #endregion -->
