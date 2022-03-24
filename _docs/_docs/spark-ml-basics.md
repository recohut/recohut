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

<!-- #region id="kEn3QO6f35ZP" -->
# SparkML Basics
> Basics of PySpark ML Lib module

- toc: true
- badges: true
- comments: true
- categories: [spark, pyspark]
- image:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Rg_FleseSdTF" outputId="35f0b560-a4af-4146-a8e0-b8e9ad203672"
!git clone https://github.com/PacktPublishing/Mastering-Big-Data-Analytics-with-PySpark
```

```python colab={"base_uri": "https://localhost:8080/"} id="ZPeES7P-S0Be" outputId="fad9f736-6c93-4046-c988-9b33d3dc65b3"
%cd Mastering-Big-Data-Analytics-with-PySpark/
```

```python id="wb1bz2N4cnEK"
!python download_data.py
```

```python id="V0jui5N5Trjt"
# !apt-get install openjdk-8-jdk-headless -qq > /dev/null
# !wget https://downloads.apache.org/spark/spark-3.0.2/spark-3.0.2-bin-hadoop3.2.tgz
# !tar -xvf spark-3.0.2-bin-hadoop3.2.tgz
# !pip install -q findspark
```

```python id="1az9qwMn14l5"
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/Mastering-Big-Data-Analytics-with-PySpark/spark-3.0.2-bin-hadoop3.2"
```

<!-- #region id="mZdQp0NlUV0X" -->
### Hello world Spark!
<!-- #endregion -->

```python id="CKBsHoTRTEDg"
from pyspark.sql import SparkSession
```

```python id="zZH5ExEDTdm9"
spark = SparkSession.builder.appName("HelloWorldApp").getOrCreate()
```

```python colab={"base_uri": "https://localhost:8080/"} id="M1y6cG1xUFBU" outputId="ab705dfa-702b-4877-8549-95de10a583f7"
# Using Spark SQL, we create a dataframe which holds our `hello world` data
df = spark.sql('SELECT "hello world" as c1')

# We can then use the `show()` method to see what the DataFrame we just created looks like
df.show()
```

```python id="ORCesjawUPKc"
spark.stop()
```

<!-- #region id="7TH96Tefgpgd" -->
### Preparing data using SparkSQL
- How to use read.csv() to load CSV files, and how to control the settings of this method
- By default, CSVs are parsed with all columns being cast to StringType
- inferSchema allows Spark to guess what schema should be used
- To ensure proper Type Safety, we can use Hive Schema DDL to set an explicit schema
<!-- #endregion -->

```python id="tMkqQ8LWczaO"
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("31LoadingDataFromCSV").getOrCreate()
```

```python id="8IrygRwtgxFE"
RATINGS_CSV_LOCATION = "/content/Mastering-Big-Data-Analytics-with-PySpark/data-sets/ml-latest-small/ratings.csv"
```

```python colab={"base_uri": "https://localhost:8080/"} id="-fKZMsXag7M_" outputId="cc0502e7-7b74-4b10-a68a-dcdaa63c2967"
df = spark.read.csv(RATINGS_CSV_LOCATION)

df.show()
df.printSchema()
```

<!-- #region id="CkEMHQ6shSZC" -->
What you can see, is that the data is being loaded, but it does not quite appear to be right. Additionally, all the columns appear to be cast as a StringType - which is not ideal. We can fix the aformentioned issues by giving the read.csv() method the correct settings.

To parse the CSV correctly, we are going to need to set the following on our read.csv() method:
1. We leave the same path as before, referring to RATINGS_CSV_LOCATION that we set previously.
2. Since we have comma-seperated-values, we need to set sep to ','.
3. Since we have a single header row, we need to set header to True.
4. Since columns that contain commas (,) are escaped using double-quotes ("), we set quote to '"'.
5. Since the files are encoded as UTF-8, we set encoding to UTF-8.
6. Additionally, since we observed that all values are cast to StringType by default, we set inferSchema to True.


<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="4_MfppJchSno" outputId="4e9f7179-fce9-4591-e870-e93ca625f4ea"
# Loading CSV file with proper parsing and inferSchema
df = spark.read.csv(
    path=RATINGS_CSV_LOCATION,
    sep=",",
    header=True,
    quote='"',
    encoding="UTF-8",
    inferSchema=True,
)

# Displaying results of the load
df.show()
df.printSchema()
```

```python colab={"base_uri": "https://localhost:8080/"} id="WtBGKr8dhw-B" outputId="b4099782-5fa4-4c5b-d7ba-ee655d9dd848"
#  Type safe loading of ratings.csv file
df = spark.read.csv(
    path=RATINGS_CSV_LOCATION,
    sep=",",
    header=True,
    quote='"',
    encoding="UTF-8",
    schema="userId INT, movieId INT, rating DOUBLE, timestamp INT",
)

# Displaying results of the load
df.show()
df.printSchema()
df.describe().show()
df.explain()
```

```python colab={"base_uri": "https://localhost:8080/"} id="fI4UWcJyjav4" outputId="bc5a287b-6937-4322-cbd6-9e7c08af376b"
from pyspark.sql import functions as f

ratings = (
    spark.read.csv(
        path=RATINGS_CSV_LOCATION,
        sep=",",
        header=True,
        quote='"',
        schema="userId INT, movieId INT, rating DOUBLE, timestamp INT",
    )
    .withColumnRenamed("timestamp", "timestamp_unix")
    .withColumn("timestamp", f.to_timestamp(f.from_unixtime("timestamp_unix")))
)

ratings.show(5)
ratings.printSchema()
ratings.drop("timestamp_unix", "foobar").show(5)
```

```python colab={"base_uri": "https://localhost:8080/"} id="DMC8M6CLkoio" outputId="755e82d2-9fd5-4a91-d6d4-2b1df396cc47"
MOVIES_CSV_LOCATION = "/content/Mastering-Big-Data-Analytics-with-PySpark/data-sets/ml-latest-small/movies.csv"

movies = (
    spark.read.csv(
        path=MOVIES_CSV_LOCATION,
        sep=",",
        header=True,
        quote='"',
        schema="movieId INT, title STRING, genres STRING",
    )
)
movies.show(15, truncate=False)
movies.printSchema()
```

```python colab={"base_uri": "https://localhost:8080/"} id="lHzUF7b8k7JA" outputId="c57a7c8e-534a-4632-d696-487e61e66167"
movies.where(f.col("genres") == "Action").show(5, False)
movies.where("genres == 'Action'").show(5, False)
```

```python colab={"base_uri": "https://localhost:8080/"} id="bdr37suXlZYv" outputId="d1581c34-f251-4ab5-84d9-8568af06f4fd"
movie_genre = (
    movies
    .withColumn("genres_array", f.split("genres", "\|"))
    .withColumn("genre", f.explode("genres_array"))
    .select("movieId", "title", "genre")
)

movie_genre.show(10, False)
```

```python colab={"base_uri": "https://localhost:8080/"} id="SB0Ez1Nklg0G" outputId="719e3268-3a54-448d-f12f-20782924eaaf"
available_genres = movie_genre.select("genre").distinct()
available_genres.show()

movies_without_genre = movies.where(f.col("genres") == "(no genres listed)")
print(movies_without_genre.count())
movies_without_genre.show()
```

<!-- #region id="QdSNFz8Jjx6K" -->
### Grouping, Joining and Aggregating
<!-- #endregion -->

```python id="62iqvcMsjgBS"
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, IntegerType, StructType, StructField
```

```python colab={"base_uri": "https://localhost:8080/"} id="XqlO_uC8xevw" outputId="700638d9-50df-4b74-d655-0dd91600a690"
spark = SparkSession.builder.appName("join_tests").getOrCreate()
schema = StructType(
    [StructField("id", IntegerType()), StructField("value", StringType())]
)


A = spark.createDataFrame(
    schema=schema, data=[
        (1, "A"),
        (2, "B"),
        (3, "C"),
        (4, "D"),
        (5, "E"),
        (None, "Z")
    ]
)

B = spark.createDataFrame(
    schema=schema, data=[
        (3, "C"),
        (4, "D"),
        (5, "E"),
        (6, "F"),
        (7, "G")
    ]
)

A.show()
B.show()
```

```python id="0EFstaZ1xhES"
# INNER JOINS
# A.join(B, ["id"], "inner").show()

# CROSS JOINS (CARTESIAN PRODUCT)
# A.crossJoin(B).show()

# FULL JOINS
# A.join(B, ["id"], "outer").show()
# A.join(B, ["id"], "full").show()
# A.join(B, ["id"], "full_outer").show()

# LEFT OUTER
# A.join(B, ["id"], "left").show()
# A.join(B, ["id"], "left_outer").show()

# RIGHT OUTER
# A.join(B, ["id"], "right").show()
# A.join(B, ["id"], "right_outer").show()

# LEFT SPECIAL
# A.join(B, ["id"], "left_semi").show()
# A.join(B, ["id"], "left_anti").show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="SntUyi9PzMZK" outputId="a0e22f6e-85df-44a5-86b5-ed12c61e098d"
links = spark.read.csv(
    path="/content/Mastering-Big-Data-Analytics-with-PySpark/data-sets/ml-latest-small/links.csv",
    sep=",",
    header=True,
    quote='"',
    schema="movieId INT, imdbId STRING, tmdbId INT",
)

tags = spark.read.csv(
    path="/content/Mastering-Big-Data-Analytics-with-PySpark/data-sets/ml-latest-small/tags.csv",
    sep=",",
    header=True,
    quote='"',
    inferSchema=True,
    schema="userId INT, movieId INT, tag STRING, timestamp INT",
).withColumn("timestamp", f.to_timestamp(f.from_unixtime("timestamp")))

links.show(5)
tags.show(5)
```

```python colab={"base_uri": "https://localhost:8080/"} id="wHCJtSnszxoa" outputId="7ea134ee-0a54-4cb0-d572-92edf56babd5"
movie_per_genre = movie_genre.groupBy("genre").count()
movie_per_genre.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="yuiiUX-V1JqR" outputId="051bca2e-b254-4cbf-cc61-33f3fd0bec8c"
# opinions = movies.join(tags, movies['movieId'] == tags['movieId'])
# opinions = movies.join(tags, ["movieId"])
opinions = movies.join(tags, ["movieId"], "inner")
opinions.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="xHP2RES415rE" outputId="999d4c16-b163-4f61-8a7c-24834353a6c1"
opinions = (
    movies
    .join(tags, ["movieId"], "inner")
    .select("userId", "movieId", "title", "tag", "timestamp")
)
opinions.show(5, False)
```

```python colab={"base_uri": "https://localhost:8080/"} id="GMCGePR_2xM-" outputId="63dc58f6-de60-4156-9854-c8572a68ff67"
opinions_ext = opinions.withColumnRenamed("timestamp", "tag_time").join(ratings, ["movieId", "userId"])
opinions_ext.show(5)
```

```python colab={"base_uri": "https://localhost:8080/"} id="RWwUadXn3qhK" outputId="40cc381f-dce0-4b3a-ea81-52e81762c10a"
ratings.groupBy("movieId").agg(
    f.count("*"),
    f.min("rating"),
    f.min("rating"),
    f.avg("rating"),
    f.min("timestamp"),
    f.max("timestamp"),
).show(5)
```

<!-- #region id="0M3rXQVR5eKr" -->
<!-- #endregion -->

<!-- #region id="MvLSdZjj8Et2" -->
### ALS Recommender System
<!-- #endregion -->

```python id="5gF2zclO3qeE"
from pyspark.sql import SparkSession
from pyspark.sql import functions as f

spark = SparkSession.builder.appName("als-recommender").getOrCreate()
```

```python id="ThehCadk3qaY"
ratings = (
    spark.read.csv(
        path="/content/Mastering-Big-Data-Analytics-with-PySpark/data-sets/ml-latest-small/ratings.csv",
        sep=",",
        header=True,
        quote='"',
        schema="userId INT, movieId INT, rating DOUBLE, timestamp INT",
    )
    # .withColumn("timestamp", f.to_timestamp(f.from_unixtime("timestamp")))
    .drop("timestamp")
    .cache()
)
```

<!-- #region id="iQ7YY90f8bn4" -->
<!-- #endregion -->

```python id="-9S-wo4t3qV_"
from pyspark.ml.recommendation import ALS
```

```python id="mBycntJP8dqB"
model = (
    ALS(
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
    ).fit(ratings)
)
```

```python colab={"base_uri": "https://localhost:8080/"} id="DP_VSR9S8fLR" outputId="139a8cdd-2612-4908-bcc0-499ca6d2d239"
predictions = model.transform(ratings)
predictions.show(10, False)
```

```python colab={"base_uri": "https://localhost:8080/"} id="ZFXCGziI8g4O" outputId="439a3b8a-c009-4f31-e607-5564179a0667"
model.userFactors.show(5)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Gmj0um2E8iQL" outputId="4ba48a1f-7ba6-40fe-8f60-c57c19ce0145"
model.itemFactors.show(5)
```

```python id="dC7NDHF18jqZ"
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
```

```python id="oCnQSleL8zu2"
als = ALS(
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
)

(training_data, validation_data) = ratings.randomSplit([8.0, 2.0])

evaluator = RegressionEvaluator(
    metricName="rmse", labelCol="rating", predictionCol="prediction"
)

model = als.fit(training_data)
predictions = model.transform(validation_data)
```

```python colab={"base_uri": "https://localhost:8080/"} id="jec6qWF383v0" outputId="390464d2-ce41-4af1-ea26-bf16bcc0f888"
predictions.show(10, False)
```

```python colab={"base_uri": "https://localhost:8080/"} id="-CCE4WTd85Pm" outputId="3f182fa8-e129-4e54-eb56-ea9988bdfcf0"
rmse = evaluator.evaluate(predictions.na.drop())
print(rmse)
```

```python id="Bejqt5_J9Cv0"
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

parameter_grid = (
    ParamGridBuilder()
    .addGrid(als.rank, [1, 5, 10])
    .addGrid(als.maxIter, [20])
    .addGrid(als.regParam, [0.05, 0.1])
    .build()
)
```

```python colab={"base_uri": "https://localhost:8080/"} id="l-CChHqs9MFW" outputId="52b63536-75c2-4e8c-d398-118a7c6f5e79"
from pprint import pprint

pprint(parameter_grid)
```

```python id="YvFCYHdx9MgF"
crossvalidator = CrossValidator(
    estimator=als,
    estimatorParamMaps=parameter_grid,
    evaluator=evaluator,
    numFolds=2,
)

crossval_model = crossvalidator.fit(training_data)
predictions = crossval_model.transform(validation_data)
```

```python colab={"base_uri": "https://localhost:8080/"} id="zuJ8jdGg9PNa" outputId="82f01809-8cf4-4127-a24e-77fdfd3c42db"
rmse = evaluator.evaluate(predictions.na.drop())
print(rmse)
```

```python id="rWIGzq5M9S7I"
model = crossval_model.bestModel
```

<!-- #region id="Kiy1fo25wmKB" -->
### MLlib data source APIs
<!-- #endregion -->

<!-- #region id="CVQbCK_7ytgO" -->
Vectors and Matrices
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="_22TVFwnwn8s" outputId="56d7d80c-5bab-45b5-99ee-c63aace40c7e"
import numpy as np
import scipy.sparse as sps
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.linalg import Matrix, Matrices

spark = SparkSession.builder.getOrCreate()

# Use a NumPy array as a dense vector.
dv1 = np.array([1.0, 0.0, 3.0])
# Use a Python list as a dense vector.
dv2 = [1.0, 0.0, 3.0]
print("Dense vector 1:", dv1)
print("Dense vector 2:", dv2)

# Create a SparseVector.
sv1 = Vectors.sparse(3, [0, 2], [1.0, 3.0])
# Use a single-column SciPy csc_matrix as a sparse vector.
sv2 = sps.csc_matrix((np.array([1.0, 3.0]), np.array([0, 2]), np.array([0, 2])), shape=(3, 1))
print("Sparse vector 1:", sv1)
print("Sparse vector 2:", sv2)

# Create a dense matrix
dm = Matrices.dense(3, 2, [1, 3, 5, 2, 4, 6])
# Create a sparse matrix
sm = Matrices.sparse(3, 2, [0, 1, 3], [0, 2, 1], [9, 6, 8])
print("Dense matrix:", dm)
print("Sparse matrix:", sm)
```

<!-- #region id="qXDXrp2d0_ZK" -->
Images
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 173} id="YDm2socEzjxD" outputId="be2eb555-22f8-4e95-8a31-16dff6541f39"
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

PATH = "./spark-3.0.2-bin-hadoop3.2/data/mllib/images/origin/kittens"
df = (
    spark.read.format("image")
    .option("dropInvalid", True)
    .load(PATH)
    .select("image.origin", "image.height", "image.width", "image.nChannels", "image.mode", "image.data")
)
df.toPandas()
```

<!-- #region id="6v8WGBph4Ocr" -->
libSVM
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="7VMILKvg32qg" outputId="b30b7b97-a132-4d48-9470-8fca2d19d613"
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

PATH = "./spark-3.0.2-bin-hadoop3.2/data/mllib/sample_libsvm_data.txt"

df = spark.read.format("libsvm").option("numFeatures", "780").load(PATH)
df.show()
```

<!-- #region id="cH3-B5q430E8" -->
### NLP and Hyperparameter Tuning in PySpark
<!-- #endregion -->

```python id="FwtnjDME30rU"
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
```

```python colab={"base_uri": "https://localhost:8080/"} id="pLpIwr8EIUpZ" outputId="b7db4e2a-2261-48ae-aac4-4df8baa50aa6"
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Prepare training documents, which are labeled.
training_data = [
    (0, "a b c d e spark", 1.0),
    (1, "b d", 0.0),
    (2, "spark f g h", 1.0),
    (3, "hadoop mapreduce", 0.0),
    (4, "b spark who", 1.0),
    (5, "g d a y", 0.0),
    (6, "spark fly", 1.0),
    (7, "was mapreduce", 0.0),
    (8, "e spark program", 1.0),
    (9, "a e c l", 0.0),
    (10, "spark compile", 1.0),
    (11, "hadoop software", 0.0),
]
training = spark.createDataFrame(training_data, ["id", "text", "label"])
print("Dataset used for training (labeled):")
training.show()

# Prepare test documents, which are unlabeled.
test_data = [
    (4, "spark i j k"),
    (5, "l m n"),
    (6, "mapreduce spark"),
    (7, "apache hadoop"),
]
test = spark.createDataFrame(test_data, ["id", "text"],)
print("Dataset used for testing (unlabeled):")
test.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="2wEzIxLGIbNl" outputId="099ce818-e1d3-4929-ff61-6fc2035d7985"
# Configure an ML pipeline, which consists of tree stages: tokenizer, hashingTF, and lr.
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression(maxIter=10)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

# We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
# This will allow us to jointly choose parameters for all Pipeline stages.
# A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
# We use a ParamGridBuilder to construct a grid of parameters to search over.
# With 3 values for hashingTF.numFeatures and 2 values for lr.regParam,
# this grid will have 3 x 2 = 6 parameter settings for CrossValidator to choose from.
paramGrid = (
    ParamGridBuilder()
    .addGrid(hashingTF.numFeatures, [10, 100, 1000])
    .addGrid(lr.regParam, [0.1, 0.01])
    .build()
)

crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=BinaryClassificationEvaluator(),
    numFolds=2,
)  # use 3+ folds in practice

# Run cross-validation, and choose the best set of parameters.
cvModel = crossval.fit(training)

# Make predictions on test documents. cvModel uses the best model found (lrModel).
prediction = cvModel.transform(test)
selected = prediction.select("id", "text", "probability", "prediction")
selected.show(100, False)
```

<!-- #region id="p--Z0152S29w" -->
## Twitter Sentiment Analysis
<!-- #endregion -->

<!-- #region id="UGgHlJMwS6np" -->
### Environment Setup
<!-- #endregion -->

```python id="WG54aiBII2pD"
import pandas as pd
from IPython.core.display import display
import seaborn as sns

from pyspark.sql import SparkSession
from pyspark.sql import functions as f

# General settings for display purposes
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.max_colwidth = 144
sns.set(color_codes=True)
```

```python id="YnNJoeIkTFMU"
spark = SparkSession.builder.getOrCreate()

# Source sentiment140: http://help.sentiment140.com/for-students/
schema = "polarity FLOAT, id LONG, date_time STRING, query STRING, user STRING, text STRING"
spark_reader = spark.read.schema(schema)

# file 1: testdata.manual.2009.06.14.csv
TESTDATA_PATH = ("./data-sets/sentiment-140-training-data/testdata.manual.2009.06.14.csv")
raw_test_data = spark_reader.csv(
    TESTDATA_PATH,
    quote='"',
    header=False,
    inferSchema=True,
    columnNameOfCorruptRecord="corrupt_data",
).cache()

# file 2: training.1600000.processed.noemoticon.csv
TRAININGDATA_PATH = "./data-sets/sentiment-140-training-data/training.1600000.processed.noemoticon.csv"
raw_training_data = spark_reader.csv(
    TRAININGDATA_PATH,
    quote='"',
    header=False,
    inferSchema=True,
    columnNameOfCorruptRecord="corrupt_data",
).cache()

# path that we will write our raw data to
OUTPUT_PATH = ("./data-sets/sentiment-140-training-data/RAW")
```

<!-- #region id="54DRgtHXbTkc" -->
### Data Exploration
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="Qo0rcNjlToGq" outputId="db17951d-58b9-45ae-e50e-c471c35ce84e"
# Count of data
print(f"Overall data count: {raw_test_data.count()}")

# Data summary
display(raw_test_data.summary().toPandas())
print("Data schema")
raw_test_data.printSchema()

# Let's look at 50 rows of data
display(raw_test_data.limit(50).toPandas())
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="taEwhrtrTqtG" outputId="1db18c5d-c3df-45d4-cb78-e23836310161"
# Count of data
print(f"Overall data count: {raw_training_data.count()}")

# Data summary
display(raw_training_data.summary().toPandas())
print("Data schema")
raw_training_data.printSchema()

# Let's look at 50 rows of data
display(raw_training_data.limit(50).toPandas())
```

<!-- #region id="AW9d4yr9UE3O" -->
Initial Findings

- We need to apply a proper schema
- The date column needs fixing
- We need to extract twitter user names/handles (we'll extract it and call the output column `users_mentioned`)
- We need to extract hashtags and replace them with the words from the hashtag (we'll extract it and call the output column `hashtags`)
- We need to extract URLs, as our algorithm won't need that or use that (we'll simply remove it from the data)
- The same goes for email-address
- HTML does not appear properly unescaped, we're going to have to fix that (example: `&lt;3` and `s&amp;^t`)
- Encoding seems to be 'broken' (example: `�����ߧ�ǿ�����ж�؜��� &lt;&lt;----I DID NOT KNOW I CUD or HOW TO DO ALL DAT ON MY PHONE TIL NOW. WOW..MY LIFE IS NOW COMPLETE. JK.`)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 480} id="PU9FSDkbUMia" outputId="85f11495-f621-468e-9500-100046b20a96"
df = raw_training_data.select("polarity").na.drop()
print(f"No of rows with Polarity: {df.count()} / {raw_training_data.count()}")

display(df.groupBy("polarity").count().toPandas())
sns.displot(df.toPandas());
```

<!-- #region id="kiH7klg1XND7" -->
Store our raw data
- keep the format CSV
- partition the data by polarity, this will create 2 subfolders inside our output folder
- repartition the data in 20 partitions: This will ensure that we have 20 smaller csv files per partition
- As 498 rows is way too little for us to train a model on, we're going to disregard this dataset and focus on the Training Data
<!-- #endregion -->

```python id="ci_2d6KXWr5E"
raw_training_data.repartition(20).write.partitionBy("polarity").csv(OUTPUT_PATH, mode="overwrite")
```

<!-- #region id="d0MWjyELbWNR" -->
### Data Wrangling
<!-- #endregion -->

<!-- #region id="lHdpTHe0csWJ" -->
[notebook](https://nbviewer.jupyter.org/github/PacktPublishing/Mastering-Big-Data-Analytics-with-PySpark/blob/master/Section%206%20-%20Analyzing%20Big%20Data/6.3/data_wrangling.ipynb)
<!-- #endregion -->

```python id="JygjkDBKaWNm"
import html
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as f

spark = SparkSession.builder.appName("DataCleaning").getOrCreate()

pd.options.display.max_columns = None
pd.options.display.max_rows = 250
pd.options.display.max_colwidth = 150

schema = "polarity FLOAT, id LONG, date_time TIMESTAMP, query STRING, user STRING, text STRING"
timestampformat = "EEE MMM dd HH:mm:ss zzz yyyy"

IN_PATH = "./data-sets/sentiment-140-training-data/RAW"
OUT_PATH = "./data-sets/sentiment-140-training-data/CLEAN"

spark_reader = spark.read.schema(schema)

url_regex = r"((https?|ftp|file):\/{2,3})+([-\w+&@#/%=~|$?!:,.]*)|(www.)+([-\w+&@#/%=~|$?!:,.]*)"
email_regex = r"[\w.-]+@[\w.-]+\.[a-zA-Z]{1,}"
user_regex = r"(@\w{1,15})"
hashtag_regex = "(#\w{1,})"
hashtag_replace_regex = "#(\w{1,})"

@f.udf
def html_unescape(s: str):
    if isinstance(s, str):
        return html.unescape(s)
    return s


def clean_data(df):
    df = (
        df
        .withColumn("original_text", f.col("text"))
        .withColumn("text", f.regexp_replace(f.col("text"), url_regex, ""))
        .withColumn("text", f.regexp_replace(f.col("text"), email_regex, ""))
        .withColumn("text", f.regexp_replace(f.col("text"), user_regex, ""))
        .withColumn("text", f.regexp_replace(f.col("text"), "#", " "))
        .withColumn("text", html_unescape(f.col("text")))
        .filter("text != ''")
    )
    return df

df_raw = spark_reader.csv(IN_PATH, timestampFormat=timestampformat) 
df_clean = clean_data(df_raw)

df_clean.write.partitionBy("polarity").parquet(OUT_PATH, mode="overwrite")
```
