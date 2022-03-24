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

<!-- #region id="2cxACca4iYYj" -->
# Music Recommender System with PySpark
<!-- #endregion -->

<!-- #region id="D1EnBeugwSRX" -->
## Process flow

- Importing the csv file
- Prepare our dataset by performing an Aggregation
- Converting String columns into columns with unique numerical values
- Creating the ALS model
- Suggest top 10 tracks for each user
<!-- #endregion -->

<!-- #region id="gWlAHMPzwdN3" -->
---

*Learn pySpark and how to work wth a large dataset (1 GB) in this tool. Also I used pySpark's ALS tools to recommend music to the user based on the implicit listening count for that user.*

---
<!-- #endregion -->

<!-- #region id="hMZ64UxxrMvL" -->
Let's install pyspark
<!-- #endregion -->

```python id="Dal3Np2Imn8u"
!pip install pyspark
```

<!-- #region id="eK9D-CNgru_s" -->
Importing the modules
<!-- #endregion -->

```python id="Z_cmzia9nA1W"
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, desc , col, max
from pyspark.ml.feature import  StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
```

<!-- #region id="pjM5P6b0rVpH" -->
Creating the spark session

<!-- #endregion -->

```python id="gvSzI0zMn4Hk"
spark = SparkSession.builder.appName('lastfm').getOrCreate()
```

<!-- #region id="KnyYdMVmnkp5" -->
## Loading the dataset
<!-- #endregion -->

```python id="PywJeVgDngVd" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1609450567455, "user_tz": -330, "elapsed": 20024, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="652fcaff-6cc8-432e-ea75-a55b3d9162b6"
!gdown --id 1q8VWIZFjlOP_91z0GjbCe4RpmtGVDkvz
!gdown --id 14dMLzOTIf1GK-P6bA9rVEI_1WSedOdZU
```

```python colab={"base_uri": "https://localhost:8080/"} id="fMBBbT42uoQ9" executionInfo={"status": "ok", "timestamp": 1609450818522, "user_tz": -330, "elapsed": 1635, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1026d060-1e70-4436-b918-9159ddc31aa5"
file_path = '/content/listenings.csv'
df_listenings = spark.read.format('csv').option('header',True).option('inferSchema',True).load(file_path)
df_listenings.show()
```

<!-- #region id="YUHBP0yjoJvc" -->
## Cleaning tables 
<!-- #endregion -->

```python id="N8luDo3HndTb" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1609451006682, "user_tz": -330, "elapsed": 1134, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b2d581fd-60fc-47a8-adc9-6626254daa51"
df_listenings = df_listenings.drop('date')
df_listenings.show()
```

```python id="5rDadwmpowII" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1609451063722, "user_tz": -330, "elapsed": 1567, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5f751867-a2f6-42f7-ceee-7dc32c0ba8be"
df_listenings = df_listenings.na.drop()
df_listenings.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="IyfBuz1gzgxS" executionInfo={"status": "ok", "timestamp": 1609451913665, "user_tz": -330, "elapsed": 28100, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8b2be111-f511-49eb-a5ac-bead1bcc5ab7"
row_numbers = df_listenings.count()
column_numbers = len(df_listenings.columns)
print(row_numbers, column_numbers)
```

<!-- #region id="cyf1XITGpMWV" -->
## Let's Perform some aggregation
to see how many times each user has listened to specific track

<!-- #endregion -->

```python id="PeH7vYKEoyWa" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1609452249898, "user_tz": -330, "elapsed": 66715, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1264c664-ec2e-4f8b-f766-6501d6bbca8a"
df_listenings_agg = df_listenings.select('user_id','track').groupby('user_id','track').agg(count('*').alias('count')).orderBy('user_id')
df_listenings_agg.show()
```

```python id="RhVz-SvapIyr" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1609452344976, "user_tz": -330, "elapsed": 54517, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a3e076df-1793-46ab-d140-1bc9a6893c07"
row_numbers = df_listenings_agg.count()
column_numbers = len(df_listenings_agg.columns)
print(row_numbers, column_numbers)
```

```python id="KsTXVsO7I7Jw"
df_listenings_agg = df_listenings_agg.limit(20000)
```

<!-- #region id="vuebvg7UqzsO" -->
## Let's convert the user id and track columns into unique integers



<!-- #endregion -->

```python id="F9RSpj3DN6aX" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1609453289850, "user_tz": -330, "elapsed": 161737, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5d5d7271-66a4-4a30-a31e-acc3a2a8f3fb"
indexer = [StringIndexer(inputCol=col, outputCol=col+'_index').fit(df_listenings_agg) for col in list(set(df_listenings_agg.columns) - set(['count']))]

pipeline = Pipeline(stages=indexer)

data = pipeline.fit(df_listenings_agg).transform(df_listenings_agg)
data.show()
```

```python id="efL-hiR-q-AO" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1609453368467, "user_tz": -330, "elapsed": 55488, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="108bbb05-55fd-496c-d68b-e22842cfb660"
data = data.select('user_id_index','track_index','count').orderBy('user_id_index')
data.show()
```

<!-- #region id="IEwviAxXsHwN" -->
## Train and Test data
<!-- #endregion -->

```python id="DRgu2p-PsYUw"
(train, test) = data.randomSplit([0.5, 0.5])
```

<!-- #region id="M46wV6Gusdi5" -->
## Let's Create our Model
<!-- #endregion -->

```python id="bwPk25M3sfRu"
USERID = 'user_id_index'
ITEMID = 'track_index'
COUNT = 'count'

als = ALS(maxIter=5, regParam=0.01, userCol=USERID, itemCol=ITEMID, ratingCol=COUNT)
model = als.fit(train)

predictions = model.transform(test)
```

<!-- #region id="lqBkLowzsoj8" -->
## Generate top 10 Track recommendations for each user
<!-- #endregion -->

```python id="00mtv7XUsZGg"
recs = model.recommendForAllUsers(10)
```

```python id="_EQnSUh3ncar" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1609453764684, "user_tz": -330, "elapsed": 8283, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="174ce7e5-0cd4-43d3-d385-ead72c3b925a"
recs.show()
```

```python id="stUaAoKpTgqT" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1609453770893, "user_tz": -330, "elapsed": 6185, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c2bf1b34-a6c0-41bf-c227-54989bcdb27a"
recs.take(1)
```

```python id="vQDUO8G763F9"

```
