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

<!-- #region id="nkmIxgH_bDqk" -->
# RePlay Library Experiments on ML-1m [Part 1]
<!-- #endregion -->

<!-- #region id="6h0pIiJtWS-z" -->
## Setup
<!-- #endregion -->

<!-- #region id="wk4572ITYLdb" -->
### Spark installation
<!-- #endregion -->

```python id="MhhXDw00Ol_H"
!apt-get install openjdk-8-jdk-headless -qq > /dev/null
!wget -q https://archive.apache.org/dist/spark/spark-3.0.0/spark-3.0.0-bin-hadoop3.2.tgz
!tar xf spark-3.0.0-bin-hadoop3.2.tgz
!pip install -q findspark
!pip install -q pyspark
```

```python id="Ro_aHv8kOhuB"
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.0.0-bin-hadoop3.2"

import findspark
findspark.init()
```

<!-- #region id="o2VJk6PYYP8S" -->
### RePlay library installation
<!-- #endregion -->

```python id="-YJwm-W8PB0i"
!pip install replay-rec #v0.6.1
```

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="P8s7qttvaptZ" executionInfo={"status": "ok", "timestamp": 1636104327665, "user_tz": -330, "elapsed": 688, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9b57918f-5183-4bcb-c7f6-e47e84c97c2f"
import replay
replay.__version__
```

<!-- #region id="nC0PIqGVYSn0" -->
### Environment setup
<!-- #endregion -->

```python id="AARhI1KISamZ"
import warnings
warnings.filterwarnings('ignore')
```

```python id="2Ra-1BCKZtOw"
import logging
logger = logging.getLogger("replay")
```

<!-- #region id="VTZcq9P-YYC6" -->
### Downloading ML-1m datasets
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="gYKEylAPQrfA" executionInfo={"status": "ok", "timestamp": 1636103228955, "user_tz": -330, "elapsed": 4864, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="435baa9a-0fbb-4c3d-aa09-1354aa03ad01"
!wget -q --show-progress https://github.com/RecoHut-Datasets/movielens_1m/raw/main/ml1m_items.dat
!wget -q --show-progress https://github.com/RecoHut-Datasets/movielens_1m/raw/main/ml1m_ratings.dat
!wget -q --show-progress https://github.com/RecoHut-Datasets/movielens_1m/raw/main/ml1m_users.dat
!wget -q --show-progress https://github.com/RecoHut-Datasets/movielens_1m/raw/main/ml_ratings.csv
```

<!-- #region id="JfRscx7oXBdx" -->
### Spark Session State
<!-- #endregion -->

<!-- #region id="rcrcdAwmXDKE" -->
State object allows passing existing Spark session or create a new one, which will be used by the all RePlay modules.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 216} id="CjsWGpU5XEJo" executionInfo={"status": "ok", "timestamp": 1636104101404, "user_tz": -330, "elapsed": 6371, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="57f08fe2-bb4f-4556-b1fb-541a10657adc"
from replay.session_handler import State

spark = State().session
spark
```

<!-- #region id="4YiLkci6YkjL" -->
### Params
<!-- #endregion -->

```python id="fT5uzpghYKYA"
K = 5
SEED=1234
```

<!-- #region id="yKV8__0AYKYC" -->
## Data preprocessing

We will use MovieLens 1m as an example.
<!-- #endregion -->

```python jupyter={"outputs_hidden": false} id="8QcoNKbMYKYD"
import pandas as pd
df = pd.read_csv("ml1m_ratings.dat", sep="\t", names=["user_id", "item_id", "relevance", "timestamp"])
users = pd.read_csv("ml1m_users.dat", sep="\t", names=["user_id", "gender", "age", "occupation", "zip_code"])
```

<!-- #region id="_a-iF8XmYKYG" -->
### DataPreparator
<!-- #endregion -->

<!-- #region id="z1NidmnnYKYI" -->
An inner data format in RePlay is a spark dataframe.
You can pass spark or pandas dataframe as an input. Columns ``item_id`` and ``user_id`` are required for interaction matrix.
Optional columns for interaction matrix are ``relevance`` and interaction ``timestamp``. 

We implemented DataPreparator class to convert dataframes to spark format and preprocess the data, including renaming/creation of required and optional interaction matrix columns, null check and dates parsing.

To convert pandas dataframe to spark as is use function ``convert_to_spark`` from ``replay.utils``.
<!-- #endregion -->

```python jupyter={"outputs_hidden": false} id="8i_VSRrjYKYK"
from replay.data_preparator import DataPreparator

log = DataPreparator().transform(
    data=df,
    columns_names={
        "user_id": "user_id",
        "item_id": "item_id",
        "relevance": "relevance",
        "timestamp": "timestamp"
    }
)
```

```python id="c7fuJUX8YKYL" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1636103832385, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="14e3c2e9-ef22-4fd8-b76b-4b2fc9182add"
log.show(3)
```

```python id="tFxZ7aieYKYM"
from replay.utils import convert2spark
users = convert2spark(users)
```

<!-- #region id="aYMDAwUYYKYN" -->
### Split
<!-- #endregion -->

<!-- #region id="dAsB2P0pYKYO" -->
RePlay provides you with data splitters to reproduce a validation schemas widely-used in recommender systems.

`UserSplitter` takes ``item_test_size`` items for each user to the test dataset.
<!-- #endregion -->

```python jupyter={"outputs_hidden": false} id="vWZAGhhAYKYP" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1636104162747, "user_tz": -330, "elapsed": 25118, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="545310bf-3998-49be-c744-03487298afa5"
from replay.splitters import UserSplitter

splitter = UserSplitter(
    drop_cold_items=True,
    drop_cold_users=True,
    item_test_size=K,
    user_test_size=500,
    seed=SEED,
    shuffle=True
)
train, test = splitter.split(log)
(
    train.count(), 
    test.count()
)
```

<!-- #region id="rae0FFJAYKYQ" -->
## Models training
<!-- #endregion -->

<!-- #region id="cN7XTgCXY5ct" -->
#### SLIM
<!-- #endregion -->

```python id="5zP6N6WHYKYQ"
from replay.models import SLIM

slim = SLIM(lambda_=0.01, beta=0.01, seed=SEED)
```

```python id="SAT4bVq4YKYS" outputId="ca5a5358-ce10-43ab-b228-77f5b26e7a4d"
%%time

slim.fit(log=train)
```

```python id="I796enVrYKYU" outputId="d311dccf-82c1-40ec-9f5e-8fdded1b0a8f"
%%time

recs = slim.predict(
    k=K,
    users=test.select('user_id').distinct(),
    log=train,
    filter_seen_items=True
)
```

<!-- #region id="ovdftDB6YKYW" -->
## Models evaluation
<!-- #endregion -->

<!-- #region id="K76j8hKvYKYY" -->
RePlay implements some popular recommenders' quality metrics. Use pure metrics or calculate a set of chosen metrics and compare models with the ``Experiment`` class.
<!-- #endregion -->

```python jupyter={"outputs_hidden": false} id="FCeje2CRYKYY"
from replay.metrics import HitRate, NDCG, MAP
from replay.experiment import Experiment

metrics = Experiment(test, {NDCG(): K,
                            MAP() : K,
                            HitRate(): [1, K]})

```

```python id="F305wdcsYKYZ" outputId="eb965b0b-2821-4c55-8062-06eed8071e63"
%%time
metrics.add_result("SLIM", recs)
metrics.results
```

<!-- #region id="_h5YjK5lYKYa" -->
## Hyperparameters optimization
<!-- #endregion -->

```python id="WdK-XhcSYKYa"
# data split for hyperparameters optimization
train_opt, val_opt = splitter.split(train)
```

```python id="L_bY-W6VYKYb" outputId="b3f012ad-4ddd-47db-85e6-c3ef28c67673"
best_params = slim.optimize(train_opt, val_opt, criterion=NDCG(), k=K, budget=10)
```

```python id="mUl_bKCcYKYc" outputId="4bd6db8c-adb5-4195-e8ba-11bedffed937"
best_params
```

```python id="abM1CU9BYKYd" outputId="f7d70d20-e7ff-4a25-88d9-4446b8f995e7"
slim = SLIM(**best_params, seed=SEED)

slim.fit(log=train)

recs = slim.predict(
    k=K,
    users=test.select('user_id').distinct(),
    log=train,
    filter_seen_items=True
)

metrics.add_result("SLIM_optimized", recs)
metrics.results
```

<!-- #region id="K8xm3RDFYKYe" -->
### Convert to pandas
<!-- #endregion -->

```python id="WlzSIYdCYKYe" outputId="72e75aa0-47e0-4f5c-ad07-174497be97c1"
recs_pd = recs.toPandas()
recs_pd.head(3)
```

<!-- #region id="IvzDP3d1YKYf" -->
## Other RePlay models
<!-- #endregion -->

<!-- #region id="49xAba3gYKYf" -->
#### ALS
Commonly-used matrix factorization algorithm.
<!-- #endregion -->

```python jupyter={"outputs_hidden": false} id="IYHrutsDYKYf"
from replay.models import ALSWrap

als = ALSWrap(rank=100, seed=SEED)
```

```python jupyter={"outputs_hidden": false} id="txQHSIg5YKYg" outputId="85b27ef5-077d-40c5-a99d-4c313dca2421"
%%time
als.fit(log=train)
```

```python jupyter={"outputs_hidden": false} id="GhsHHEW4YKYg" outputId="8f2a7aca-2c50-43af-aab7-37d2b0df0247"
%%time
recs = als.predict(
    k=K,
    users=test.select('user_id').distinct(),
    log=train,
    filter_seen_items=True
)
```

```python jupyter={"outputs_hidden": false} id="kPAYQdWIYKYh" outputId="7c5c5e77-a2c5-4981-ebac-67b0b52e8b8d"
%%time
metrics.add_result("ALS", recs)
metrics.results
```

<!-- #region id="QZlTmCanYKYi" -->
#### MultVAE 
Variational autoencoder for a recommendation task
<!-- #endregion -->

```python id="Y-IYF7akYKYi"
from replay.models import MultVAE

multvae = MultVAE(epochs=100)
```

```python jupyter={"outputs_hidden": false} id="jaMpFWHsYKYk" outputId="a064206b-c5a3-4778-cac8-8e3b2ac077f5"
%%time
multvae.fit(log=train)
```

```python jupyter={"outputs_hidden": false} id="hHRoF12GYKYl" outputId="b9793e76-215c-40a7-be31-6df57f974a8f"
%%time

recs = multvae.predict(
    k=K,
    users=test.select('user_id').distinct(),
    log=train,
    filter_seen_items=True
)
```

```python id="y0702agjYKYm" outputId="42425d47-ac35-4f36-f520-712b8fcfb36b"
%%time
metrics.add_result("MultVAE", recs)
metrics.results
```

<!-- #region id="6IYn7FF2YKYm" -->
## Compare RePlay models with others
To easily evaluate recommendations obtained from other sources, read and pass these recommendations to ``Experiment``
<!-- #endregion -->

<!-- #region id="0rKJ-FMdYKYn" -->
#### Save your recommendations as dataframe with columns `user_id - item_id - relevance`
<!-- #endregion -->

```python id="Wi1CZZ6kYKYn"
from pyspark.sql.functions import rand
```

```python jupyter={"outputs_hidden": false} id="imPIWDwSYKYo"
recs.withColumn('relevance', rand(seed=123)).toPandas().to_csv("recs.csv", index=False)
```

<!-- #region id="BqJqkicSYKYo" -->
#### Read with DataPreparator
<!-- #endregion -->

```python jupyter={"outputs_hidden": false} id="2gO_Gg2LYKYp"
recs = DataPreparator().transform(
    path="recs.csv",
    columns_names={
        "user_id": "user_id",
        "item_id": "item_id",
        "relevance": "relevance"
    },
    reader_kwargs={"header":True},
    format_type="csv"
)
```

<!-- #region id="v11rkZE6YKYp" -->
#### Compare with Experiment
<!-- #endregion -->

```python jupyter={"outputs_hidden": false} id="N2OLp37LYKYq" outputId="c68ca24f-a5cb-4f97-c55f-d7db40003d03"
metrics.add_result("my_model", recs)
metrics.results.sort_values("NDCG@5", ascending=False)
```
