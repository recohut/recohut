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

<!-- #region id="yud2BfhB-CuV" -->
# Apache Spark 3 for Data Engineering and Analytics with Python
<!-- #endregion -->

```python id="3fk43JipXGt3"
!pip install -U -q dvc dvc[gdrive]
!dvc pull
```

```python colab={"base_uri": "https://localhost:8080/"} id="lv6ULevCWJwn" executionInfo={"status": "ok", "timestamp": 1630755185399, "user_tz": -330, "elapsed": 1383, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cc3cbb16-5f5e-4785-837f-aa54d705e89a"
!dvc status
```

```python id="t7HNm_d_hLGE"
!dvc commit && dvc push
```

```python id="1QIhuIPjV2tM"
!dvc add /content/reco-tut-de/data/bronze/pyspark_tutorial/*.
```

```python colab={"base_uri": "https://localhost:8080/"} id="AMRq-xkLYhE6" executionInfo={"status": "ok", "timestamp": 1630737592959, "user_tz": -330, "elapsed": 67613, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="767ef1f2-48cf-4f30-bad9-c551e044e831"
%cd /content
!apt-get install openjdk-8-jdk-headless -qq > /dev/null
!wget -q https://archive.apache.org/dist/spark/spark-3.0.0/spark-3.0.0-bin-hadoop3.2.tgz
!tar xf spark-3.0.0-bin-hadoop3.2.tgz
!pip install -q findspark
!pip install -q pyspark

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.0.0-bin-hadoop3.2"

import findspark
findspark.init()
```

```python colab={"base_uri": "https://localhost:8080/"} id="mLYqwIvqgeuh" executionInfo={"status": "ok", "timestamp": 1630737592961, "user_tz": -330, "elapsed": 35, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9f9dbc1c-45f3-492f-86e3-8fa5655484f8"
%cd "{project_path}"
```

```python colab={"base_uri": "https://localhost:8080/"} id="s-khtpKFVU5a" executionInfo={"status": "ok", "timestamp": 1630751020297, "user_tz": -330, "elapsed": 1721, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8aa93b43-7c90-4375-f89c-703813a18d6e"
!dvc add /content/reco-tut-de/data/bronze/pyspark_tutorial/salesdata/*.csv
```

```python colab={"base_uri": "https://localhost:8080/", "height": 216} id="kB-F2KgWf5vi" executionInfo={"status": "ok", "timestamp": 1630737910080, "user_tz": -330, "elapsed": 6051, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f625477b-eda4-4bb5-f855-9fa283c7fe49"
from pyspark.sql import SparkSession

spark = SparkSession.builder\
        .master("local")\
        .appName("Colab")\
        .config('spark.ui.port', '4050')\
        .getOrCreate()

spark
```

```python colab={"base_uri": "https://localhost:8080/"} id="_aG3BQzyY50W" executionInfo={"status": "ok", "timestamp": 1630736999546, "user_tz": -330, "elapsed": 481, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2af98d01-a4a9-4e8f-f20d-3cb174e7ce5d"
type(spark)
```

```python id="JHdMI_r1clNi"
from pyspark.sql.functions import count
```

```python colab={"base_uri": "https://localhost:8080/"} id="0GlupG_fYvFg" executionInfo={"status": "ok", "timestamp": 1630738796598, "user_tz": -330, "elapsed": 8713, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4dd4f59e-975b-4356-b806-94b7af62f230"
sales_file = './data/bronze/pyspark_tutorial/sales_records.csv'

sales_df = (spark.read.format('csv').option('header','true').option('inferSchema','true').load(sales_file))

sales_df.select('Region', 'Country', 'Order ID').show(n=10, truncate=False)
```

```python colab={"base_uri": "https://localhost:8080/"} id="hJJ1MXWFbcjZ" executionInfo={"status": "ok", "timestamp": 1630746807199, "user_tz": -330, "elapsed": 522, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a38a9f9d-1770-4340-c3d9-0a792b714409"
type(sales_df)
```

```python colab={"base_uri": "https://localhost:8080/"} id="o_gaU5Ffbwe7" executionInfo={"status": "ok", "timestamp": 1630746813841, "user_tz": -330, "elapsed": 4621, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="274f66c9-3a16-4fea-e2c3-320ab55f6b37"
count_sales_df = (sales_df.select('Region', 'Country', 'Order ID')\
                  .groupby('Region', 'Country')\
                  .agg(count('Order ID').alias('Total Orders'))\
                  .orderBy('Total Orders', ascending=False)
                  )

count_sales_df.show(n=10, truncate=False)
print('Total Rows = ', count_sales_df.count())
```

<!-- #region id="q_K73OwupdLB" -->
### Spark UI
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="o3FVXQmZmEuF" executionInfo={"status": "ok", "timestamp": 1630738602333, "user_tz": -330, "elapsed": 523, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="19b769be-5923-4fbe-8d00-b7be80a69ac5"
# !wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
# !unzip ngrok-stable-linux-amd64.zip
# get_ipython().system_raw('./ngrok http 4050 &')
!curl -s http://localhost:4040/api/tunnels
```

<!-- #region id="DwGsZAMRpJJI" -->
<!-- #endregion -->

<!-- #region id="xqQAEzS6pO1n" -->
<!-- #endregion -->

<!-- #region id="QgG7AxjsnG0Y" -->
## RDDs
<!-- #endregion -->

```python id="NqcUMPfXoaZR"
words_list = "Spark makes life a lot easier and put me into good Spirits, Spark is too Awesome!".split(" ")
```

```python colab={"base_uri": "https://localhost:8080/"} id="6EI7flFlrQc7" executionInfo={"status": "ok", "timestamp": 1630739964870, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c7d446ce-5ad1-4679-d9e1-ba6f98f7544e"
type(words_list)
```

```python colab={"base_uri": "https://localhost:8080/"} id="rEL0HDm4rSBA" executionInfo={"status": "ok", "timestamp": 1630739976077, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e3a6c4be-22c7-40cb-89fa-aa3a141415f9"
print(words_list)
```

```python id="Jlg3qIzKrTt1"
words_rdd = spark.sparkContext.parallelize(words_list)
```

```python colab={"base_uri": "https://localhost:8080/"} id="j0auwkxyrazV" executionInfo={"status": "ok", "timestamp": 1630740052678, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="79834a70-046e-46d9-cffa-747c2fb9f88f"
words_data = words_rdd.collect()

for word in words_data:
    print(word)
```

```python colab={"base_uri": "https://localhost:8080/"} id="UE-9Cq4Krnak" executionInfo={"status": "ok", "timestamp": 1630740694528, "user_tz": -330, "elapsed": 2276, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f93911d1-3a7e-4955-801c-adae12807d9f"
words_rdd.count()
```

```python colab={"base_uri": "https://localhost:8080/"} id="ifHZAJisuDTC" executionInfo={"status": "ok", "timestamp": 1630741069017, "user_tz": -330, "elapsed": 602, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="79be3833-7ecd-41c2-9553-b5d7fd57ea3b"
# task - count distinct words
# note - we will use distinct method and the count action on top of that
words_rdd.distinct().count()
```

```python colab={"base_uri": "https://localhost:8080/"} id="FTMTLwoxuF2o" executionInfo={"status": "ok", "timestamp": 1630741007277, "user_tz": -330, "elapsed": 430, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ec2469bd-21be-452e-8d5b-ca1c8f29f83a"
# task - from the above rdd, create an rdd that only contains words starting with S
# note - we will user the filter method with lambda, and then collect action on top of that
words_rdd.filter(lambda word: word.startswith("S")).collect()
```

```python colab={"base_uri": "https://localhost:8080/"} id="6sHA8v2ovQdx" executionInfo={"status": "ok", "timestamp": 1630741587149, "user_tz": -330, "elapsed": 418, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f2df21a3-b9c0-44e9-f38a-a482017ee382"
# task - create a list of tuples for each word, in which first element is the word itself
# second element is the first letter of the word, and third element is a bool whether first letter is S
words_trd_rdd = words_rdd.map(lambda word: (word, word[0], word.startswith("S")))

for element in words_trd_rdd.collect():
    print(element)
```

```python colab={"base_uri": "https://localhost:8080/"} id="q6hwX3iyxb2V" executionInfo={"status": "ok", "timestamp": 1630742185438, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e5478a68-c23d-4ea0-f187-385efd815021"
# task - create an rdd for a list of named tuples
# and first sort by key and by value

cntry_list = [('India',10), ('USA',15), ('Japan',25)]
cntry_rdd = spark.sparkContext.parallelize(cntry_list)

srtd_cntry_rdd = cntry_rdd.sortByKey().collect()

for element in srtd_cntry_rdd:
    print(element)
```

```python colab={"base_uri": "https://localhost:8080/"} id="7PpK7q_xzUe8" executionInfo={"status": "ok", "timestamp": 1630742249934, "user_tz": -330, "elapsed": 525, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9115bde0-8a9f-41da-d072-3c1f14b4424a"
rsrtd_cntry_rdd = cntry_rdd.map(lambda x: (x[1], x[0])).sortByKey().collect()

for element in rsrtd_cntry_rdd:
    print(element)
```

```python colab={"base_uri": "https://localhost:8080/"} id="0j4yn5ntz_xi" executionInfo={"status": "ok", "timestamp": 1630742972317, "user_tz": -330, "elapsed": 453, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="11ba3ea7-c0dc-4585-917b-53c8623b346c"
def wordLengthReducer(leftWord, rightWord):
    if len(leftWord) > len(rightWord):
        return leftWord
    else:
        return rightWord


words_rdd.reduce(wordLengthReducer)
```

<!-- #region id="rizW9-Ah2QLY" -->
### Temperature conversion task
<!-- #endregion -->

<!-- #region id="nVwL6qTv3LBQ" -->
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="WUugmBZV3PGZ" executionInfo={"status": "ok", "timestamp": 1630743526757, "user_tz": -330, "elapsed": 1045, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fa235a00-3542-48f9-f231-7b851f4d0f16"
def f2c(f):
    return (f-32)*(5/9)

temp_list = [59, 57.2, 53.6, 55.4, 51.8, 53.6, 55.4]

temp_rdd = spark.sparkContext.parallelize(temp_list)

temp_rdd.map(f2c).filter(lambda x: x>=13).collect()
```

<!-- #region id="hXXv7LRB4Hx5" -->
### XYZ Research
<!-- #endregion -->

<!-- #region id="uQ40S4w95WhS" -->
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="nMq9zEZs6IIw" executionInfo={"status": "ok", "timestamp": 1630743967900, "user_tz": -330, "elapsed": 497, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="69e47b8e-5823-41f6-9c43-422b6638efa7"
!cat ./extras/XYZ\ Research.txt
```

```python id="5YZHGTIG5BiG"
data2001List = ['RIN1', 'RIN2', 'RIN3', 'RIN4', 'RIN5', 'RIN6', 'RIN7']
data2002List = ['RIN3', 'RIN4', 'RIN7', 'RIN8', 'RIN9']
data2003List = ['RIN4', 'RIN8', 'RIN10', 'RIN11', 'RIN12']
```

```python id="X6ayrkoH56KV"
data2001RDD = spark.sparkContext.parallelize(data2001List)
data2002RDD = spark.sparkContext.parallelize(data2002List)
data2003RDD = spark.sparkContext.parallelize(data2003List)
```

```python colab={"base_uri": "https://localhost:8080/"} id="2M5BKIMp6gef" executionInfo={"status": "ok", "timestamp": 1630744589487, "user_tz": -330, "elapsed": 573, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="29e913e3-a36b-4b68-cdf7-227b1ddb36c9"
# How many research projects were initiated in the three years?
union20012002 = data2001RDD.union(data2002RDD)
unionAll = union20012002.union(data2003RDD)
unionAll.distinct().count()
```

```python colab={"base_uri": "https://localhost:8080/"} id="UKrOllis8xWz" executionInfo={"status": "ok", "timestamp": 1630744697777, "user_tz": -330, "elapsed": 501, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="168e70d7-ca93-453a-e709-747a26fb27d9"
# How many projects were completed in the first year?
firstYearCompletionRDD = data2001RDD.subtract(data2002RDD)
firstYearCompletionRDD.collect()
```

```python colab={"base_uri": "https://localhost:8080/"} id="9Kvt3WY49Sqs" executionInfo={"status": "ok", "timestamp": 1630744797631, "user_tz": -330, "elapsed": 682, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a5b02109-25cd-4b9c-8a83-d5a3e8d21ead"
# How many projects were completed in the first two years?
union20012002.subtract(data2003RDD).distinct().collect()
```

<!-- #region id="QroeWePK9twu" -->
## Structured API - Spark Dataframe
<!-- #endregion -->

```python id="0NtoD5cT93FH"
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, DateType
from pyspark.sql.functions import date_format
```

```python id="XmZy-h_h_Z7w"
schema = StructType([
                     StructField("Order ID", StringType(), True),
                     StructField("Product", StringType(), True),
                     StructField("Quantity Ordered", IntegerType(), True),
                     StructField("Price Each", FloatType(), True),
                     StructField("Order Date", DateType(), True),
                     StructField("Purchase Address", StringType(), True),
])
```

```python colab={"base_uri": "https://localhost:8080/"} id="8-eg00FlEKgQ" executionInfo={"status": "ok", "timestamp": 1630749812411, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6e8875cd-53f2-47dd-b2a1-c25b82cbe67d"
sales_data_fpath = './data/bronze/pyspark_tutorial/salesdata'
sales_raw_df = (spark.read.format('csv')\
                .option('header', True)\
                .schema(schema)\
                .load(sales_data_fpath)
)
sales_raw_df.show(10)
```

```python colab={"base_uri": "https://localhost:8080/"} id="ggkbzZSjNnPQ" executionInfo={"status": "ok", "timestamp": 1630749813940, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0e72034d-5cdf-4d77-c5ce-1ce6c5cc7f5b"
sales_raw_df.show(10, truncate=False)
```

```python colab={"base_uri": "https://localhost:8080/"} id="eFOAGWjSGVWo" executionInfo={"status": "ok", "timestamp": 1630749127792, "user_tz": -330, "elapsed": 19, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c58cc339-0c17-46a1-a585-5a80070dd04f"
sales_raw_df.printSchema()
```

```python colab={"base_uri": "https://localhost:8080/"} id="-OatdjgVG__4" executionInfo={"status": "ok", "timestamp": 1630749135838, "user_tz": -330, "elapsed": 726, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6ac65baa-d1ff-4e72-e871-c2bd666ee306"
sales_raw_df.count()
```

```python colab={"base_uri": "https://localhost:8080/"} id="AsZAkjAEJWaE" executionInfo={"status": "ok", "timestamp": 1630749137932, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6541eb8d-578f-4e50-d176-518ca7f69549"
sales_raw_df.columns
```

<!-- #region id="GTVg7aKHHUGV" -->
<!-- #endregion -->

<!-- #region id="rNn-gri6HEzG" -->
### Adding, renaming and dropping columns
<!-- #endregion -->

```python id="rmBp3HZANuth"
from pyspark.sql.functions import round, expr, year
```

```python colab={"base_uri": "https://localhost:8080/"} id="mB61t1XpQAxD" executionInfo={"status": "ok", "timestamp": 1630749701683, "user_tz": -330, "elapsed": 1308, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="43d0f70c-305c-4819-a519-adfecf31ec4d"
sales_raw_df.withColumn('New Price Each', expr('`Price Each` * 0.10 + `Price Each`')).show(10)
```

```python colab={"base_uri": "https://localhost:8080/"} id="BKx7VjjaNzf-" executionInfo={"status": "ok", "timestamp": 1630750120855, "user_tz": -330, "elapsed": 530, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4f17f208-447f-4505-f5e1-c9bcf8081b60"
sales_df1 = (sales_raw_df\
             .withColumn('New Price Each', expr('`Price Each` * 0.10 + `Price Each`'))\
            #  .withColumn('Order Year', year('`Order Date`'))\
             .withColumnRenamed('Purchase Address','Address')\
             .withColumn('Price x10', round('New Price Each',2))\
             .drop('New Price Each')
             )
sales_df1.show(10)
```

<!-- #region id="nFW9jIdiPx8c" -->
### Handle bad data
<!-- #endregion -->

```python id="Hm0E0HOFSZKn"
from pyspark.sql import Row
from pyspark.sql.functions import col
```

```python colab={"base_uri": "https://localhost:8080/"} id="iToUtnUCSOXd" executionInfo={"status": "ok", "timestamp": 1630750280397, "user_tz": -330, "elapsed": 522, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="dc76fed6-d477-4221-ae22-7ef06f7f694d"
bad_movies_list = [Row(None, None, None),
                   Row(None, None, 2020),
                   Row("John Doe", "Awesome Movie", None),
                   Row(None, "Awesome Movie", 2021),
                   Row("Mary Jane", None, 2019),
                   Row("Vikter Duplaix", "Not another teen movie", 2001)]
bad_movies_list                   
```

```python id="aXVrEH7qSUEu"
bad_movies_columns = ['actor_name', 'movie_title', 'producer_year']
```

```python colab={"base_uri": "https://localhost:8080/"} id="YnBOimTNS-t8" executionInfo={"status": "ok", "timestamp": 1630750373516, "user_tz": -330, "elapsed": 659, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="45d5c238-2499-4d21-c2e8-0c0e97552194"
bad_movies_df = spark.createDataFrame(bad_movies_list, schema=bad_movies_columns)
bad_movies_df.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="N9EfSkTnS-9D" executionInfo={"status": "ok", "timestamp": 1630750414757, "user_tz": -330, "elapsed": 694, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="31a962cf-fe8f-4e96-e6c1-032077c4c3be"
bad_movies_df.na.drop().show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="FczN5NcUTJGv" executionInfo={"status": "ok", "timestamp": 1630750447846, "user_tz": -330, "elapsed": 512, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="803fa9a7-b4e1-4179-d511-4be75a365bff"
bad_movies_df.na.drop('all').show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="Q4lMl7jdTPLw" executionInfo={"status": "ok", "timestamp": 1630750527416, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ee755c37-5cc2-459a-ad62-d02153f041f4"
bad_movies_df.filter(col('actor_name').isNull()!=True).show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="3p_LDP32TY-3" executionInfo={"status": "ok", "timestamp": 1630750591321, "user_tz": -330, "elapsed": 497, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d36132e8-27e3-4239-dc1a-c4e89813b585"
bad_movies_df.describe('producer_year').show()
```

<!-- #region id="BnfIJpIETw35" -->
## Data Preparation and Cleansing
<!-- #endregion -->

```python id="uMKBtjWYW3-i"
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, DateType
from pyspark.sql.functions import date_format, col, split, to_timestamp, month, year
```

```python colab={"base_uri": "https://localhost:8080/", "height": 216} id="4xN6Cm_RWzym" executionInfo={"status": "ok", "timestamp": 1630751479276, "user_tz": -330, "elapsed": 1095, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6d526788-6195-49b6-97fa-44e5e97989ad"
spark.stop()

spark = SparkSession.builder\
        .master("local")\
        .appName("Colab")\
        .config('spark.ui.port', '4050')\
        .getOrCreate()
        
spark
```

```python id="jegJu2DjXE5m"
schema = StructType([
                     StructField("Order ID", StringType(), True),
                     StructField("Product", StringType(), True),
                     StructField("Quantity Ordered", StringType(), True),
                     StructField("Price Each", StringType(), True),
                     StructField("Order Date", StringType(), True),
                     StructField("Purchase Address", StringType(), True),
])
```

```python colab={"base_uri": "https://localhost:8080/"} id="gQ8cNswwXE5m" executionInfo={"status": "ok", "timestamp": 1630751493074, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f18b5769-d1eb-403c-97af-01adae7aebac"
sales_data_fpath = './data/bronze/pyspark_tutorial/salesdata'
sales_raw_df = (spark.read.format('csv')\
                .option('header', True)\
                .schema(schema)\
                .load(sales_data_fpath)
)
sales_raw_df.show(10)
```

```python colab={"base_uri": "https://localhost:8080/"} id="f8OpxDNeXQfH" executionInfo={"status": "ok", "timestamp": 1630751685860, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e7ac4230-ae5d-4cf8-ab53-698b19ef1585"
sales_raw_df.filter(col('Order ID').isNull()==True).show(10)
```

```python id="nF--rQ9bX_c4"
sales_raw_df = sales_raw_df.na.drop('any')
```

```python colab={"base_uri": "https://localhost:8080/"} id="EX_sm7nXYI9q" executionInfo={"status": "ok", "timestamp": 1630751726347, "user_tz": -330, "elapsed": 493, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b977604b-d350-4956-a9b8-6a55748038a5"
sales_raw_df.filter(col('Order ID').isNull()==True).show(10)
```

```python colab={"base_uri": "https://localhost:8080/"} id="IQzbP82xYJQl" executionInfo={"status": "ok", "timestamp": 1630751765951, "user_tz": -330, "elapsed": 2272, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ea67269f-face-450d-cb9d-970145bac9fa"
sales_raw_df.describe(sales_raw_df.columns).show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="wnDFl4U2YSc6" executionInfo={"status": "ok", "timestamp": 1630751809421, "user_tz": -330, "elapsed": 459, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="824e7b7a-e4c2-4371-ae89-dfc5347fc220"
sales_raw_df.filter(col('Order ID')=='Order ID').show(10)
```

```python colab={"base_uri": "https://localhost:8080/"} id="lal2zGgCYdsl" executionInfo={"status": "ok", "timestamp": 1630752293629, "user_tz": -330, "elapsed": 1121, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9ed03d50-11af-40ed-8e84-8080d647cba0"
sales_temp_df = sales_raw_df.distinct() #remove duplicate records
sales_temp_df = sales_temp_df.filter(col('Order ID')!='Order ID')
sales_temp_df.show(10)
```

```python colab={"base_uri": "https://localhost:8080/"} id="FZvEbm5SZ7pB" executionInfo={"status": "ok", "timestamp": 1630752319898, "user_tz": -330, "elapsed": 454, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="694774a9-848d-4610-9499-a536138136d3"
sales_temp_df.filter(col('Order ID')=='Order ID').show(10)
```

```python colab={"base_uri": "https://localhost:8080/"} id="aGGVxvxyaXd1" executionInfo={"status": "ok", "timestamp": 1630752371808, "user_tz": -330, "elapsed": 14026, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="38175a29-a2c6-4ba6-8ab4-54454364474a"
sales_temp_df.describe(sales_temp_df.columns).show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="f2MiNAhEajrG" executionInfo={"status": "ok", "timestamp": 1630752461279, "user_tz": -330, "elapsed": 921, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0b7bbf8b-6572-45f6-91f6-99440070c366"
sales_temp_df.select('Purchase Address').show(10, False)
```

```python colab={"base_uri": "https://localhost:8080/"} id="diUklEu-a8n7" executionInfo={"status": "ok", "timestamp": 1630752556524, "user_tz": -330, "elapsed": 429, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d7819240-b695-4447-c272-72c10e751916"
sales_temp_df.select('Purchase Address', split(col('Purchase Address'), ',')).show(10, False)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Mj-K_mY0bTzd" executionInfo={"status": "ok", "timestamp": 1630752662099, "user_tz": -330, "elapsed": 423, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5c7feb21-6146-4efd-fb6b-68a897865d4f"
sales_temp_df.select('Purchase Address', split(col('Purchase Address'), ',').getItem(1)).show(10, False)
```

```python colab={"base_uri": "https://localhost:8080/"} id="A3j6P4FlboBg" executionInfo={"status": "ok", "timestamp": 1630752688069, "user_tz": -330, "elapsed": 491, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1138dda0-ce11-4b87-976b-0822ade715d9"
sales_temp_df.select('Purchase Address', split(col('Purchase Address'), ',').getItem(2)).show(10, False)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Y8pIrec6b0Li" executionInfo={"status": "ok", "timestamp": 1630753010989, "user_tz": -330, "elapsed": 577, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e6fc1f20-a5d5-4a9b-ed1a-025587987d51"
sales_temp_df = (sales_temp_df\
                 .withColumn('City', split(col('Purchase Address'), ',').getItem(1))\
                 .withColumn('State', split(split(col('Purchase Address'),',').getItem(2), ' ').getItem(1))
                 )

sales_temp_df.show(10, False)
```

```python colab={"base_uri": "https://localhost:8080/"} id="9H6PrJ2ZdC5t" executionInfo={"status": "ok", "timestamp": 1630753540221, "user_tz": -330, "elapsed": 624, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b3d29587-9881-4032-fb67-a611118eb3a3"
sales_temp_df = (sales_temp_df\
                 .withColumn('OrderID', col('Order ID').cast(IntegerType()))\
                 .withColumn('Quantity', col('Quantity Ordered').cast(FloatType()))\
                 .withColumn('Price', col('Price Each').cast(FloatType()))\
                 .withColumn('OrderDate', to_timestamp(col('Order Date'), 'MM/dd/yy HH:mm'))\
                 .withColumnRenamed('Purchase Address', 'StoreAddress')\
                 .drop('Order ID')\
                 .drop('Quantity Ordered')\
                 .drop('Price Each')\
                 .drop('Order Date')\
                 )

sales_temp_df.show(10, False)
```

```python colab={"base_uri": "https://localhost:8080/"} id="945MqwHnerGD" executionInfo={"status": "ok", "timestamp": 1630753709746, "user_tz": -330, "elapsed": 459, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="218c3566-3b5a-46f4-e991-969d1d4b3573"
sales_temp_df = (sales_temp_df\
                 .withColumn('ReportYear', year(col('OrderDate')))\
                 .withColumn('Month', month(col('OrderDate')))
                 )
sales_temp_df.show(10, False)
```

```python colab={"base_uri": "https://localhost:8080/"} id="mKvaM6KZfrBT" executionInfo={"status": "ok", "timestamp": 1630754388034, "user_tz": -330, "elapsed": 560, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a0c3c4b6-16bc-4e9e-dc2a-4811e17eed7c"
ordered_columns = ['OrderID', 'Product', 'Quantity', 'Price', 'OrderDate',
                   'StoreAddress', 'City', 'ReportYear', 'Month']

sales_temp_df = sales_temp_df.select(ordered_columns)

sales_temp_df.show(10, False)
```

```python id="53fhwkXSiSgC"
!mkdir -p /content/temp
output_path = '/content/temp'
sales_temp_df.write.mode('overwrite').partitionBy('ReportYear','Month').parquet(output_path)
```
