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

<!-- #region id="KTmfWUvx4aUt" -->
# Movie recommender on PySpark
> Building a scalable movie recommendation system using PySpark trained on movielens

- toc: true
- badges: true
- comments: true
- categories: [spark, pyspark, movie]
- image:
<!-- #endregion -->

<!-- #region id="V8pr43LABj4a" -->
## Environment Setup
<!-- #endregion -->

```python id="_d937xTX9ycO"
!apt-get install openjdk-8-jdk-headless -qq > /dev/null
!wget https://downloads.apache.org/spark/spark-3.1.1/spark-3.1.1-bin-hadoop3.2.tgz
!tar -xvf spark-3.1.1-bin-hadoop3.2.tgz
!pip install -q findspark
```

```python id="HRwFs7eH97bt"
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.1.1-bin-hadoop3.2"
```

```python id="RjoJVzu2_lFK"
# import findspark
# findspark.init()
```

```python id="4y1d1JrF-ihC"
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()
```

```python id="g_3LmCAS-tGw"
# df = spark.createDataFrame([{"hello": "world"} for x in range(1000)])
# df.show(3)
```

```python id="HQcZl-WxYgDS"
!pip install koalas
```

```python colab={"base_uri": "https://localhost:8080/"} id="KY2_c8UOAEnd" outputId="a4e1fbfd-05c9-4c76-b2a6-63ab8d671ee3"
# Default Packages (available by Default in Google Colab)
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import random
from pprint import pprint
from matplotlib.lines import Line2D

# Downloaded Packages (not available by Default)
import databricks.koalas

# PySpark Utilities
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql import SparkSession, Row
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics

# Random Seed
SEED = 1492

# Set-up
plt.style.use('seaborn')
```

<!-- #region id="evGR8Qv2Bm09" -->
## Data Loading
<!-- #endregion -->

```python id="vv06-6Bm9P1H"
complete_dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-latest.zip'
small_dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
```

<!-- #region id="2EOHjq7YCW7J" -->
We also need to define download locations.
<!-- #endregion -->

```python id="KwQ0hd209PyQ"
import os

datasets_path = os.path.join('.', 'datasets')
os.makedirs(datasets_path, exist_ok=True)
complete_dataset_path = os.path.join(datasets_path, 'ml-latest.zip')
small_dataset_path = os.path.join(datasets_path, 'ml-latest-small.zip')
```

<!-- #region id="9Quh2RCcCUjr" -->
Now we can proceed with both downloads.
<!-- #endregion -->

```python id="7Lb05U_8CLH1"
import urllib.request

small_f = urllib.request.urlretrieve (small_dataset_url, small_dataset_path)
complete_f = urllib.request.urlretrieve (complete_dataset_url, complete_dataset_path)
```

<!-- #region id="uPNswlODCSlS" -->
Both of them are zip files containing a folder with ratings, movies, etc. We need to extract them into its individual folders so we can use each file later on.
<!-- #endregion -->

```python id="TvJxk-4aCLE4"
import zipfile

with zipfile.ZipFile(small_dataset_path, "r") as z:
    z.extractall(datasets_path)

with zipfile.ZipFile(complete_dataset_path, "r") as z:
    z.extractall(datasets_path)
```

<!-- #region id="gdlZMzMMAFY2" -->
## Basic example
<!-- #endregion -->

```python id="eJCM51y3AKmu"
spark = SparkSession\
    .builder\
    .appName("ALSExample")\
    .getOrCreate()

lines = spark.read.text(os.path.join(os.getenv('SPARK_HOME'),"data/mllib/als/sample_movielens_ratings.txt")).rdd
parts = lines.map(lambda row: row.value.split("::"))
ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                      rating=float(p[2]), timestamp=int(p[3])))
ratings = spark.createDataFrame(ratingsRDD)
(training, test) = ratings.randomSplit([0.8, 0.2])

# Build the recommendation model using ALS on the training data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating")
# als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
model = als.fit(training)

# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# Generate top 10 movie recommendations for each user
userRecs = model.recommendForAllUsers(10)
userRecs.show()

# Generate top 10 user recommendations for each movie
movieRecs = model.recommendForAllItems(10)
movieRecs.show()

# Generate top 10 movie recommendations for a specified set of users
users = ratings.select(als.getUserCol()).distinct().limit(3)
userSubsetRecs = model.recommendForUserSubset(users, 10)
userSubsetRecs.show()

# Generate top 10 user recommendations for a specified set of movies
movies = ratings.select(als.getItemCol()).distinct().limit(3)
movieSubSetRecs = model.recommendForItemSubset(movies, 10)
movieSubSetRecs.show()

spark.stop()
```

<!-- #region id="ZAlOk9ppBe2L" -->
## Advanced example
<!-- #endregion -->

<!-- #region id="OA5r5D-_Lpv3" -->
- https://nbviewer.jupyter.org/github/SonalSavaliya/Movie-Recommender-System/blob/master/movie_recommender_using_spark.ipynb
- https://nbviewer.jupyter.org/github/Ansu-John/Movie-Recommender-System/blob/main/Movie%20Recommender%20System.ipynb
- https://nbviewer.jupyter.org/github/assadullah1467/PySpark-Recommendation-Engine/blob/master/Recommender_System_PySpark.ipynb
<!-- #endregion -->

```python id="zybYwD9jGVwO"
spark = SparkSession.builder.appName("Reco-Spark-Example2").getOrCreate()
```

```python id="VKbE1wi4HD73"
data = spark.read.csv(os.path.join(datasets_path,'ml-latest-small','ratings.csv'),
                      inferSchema=True, header=True)
```

```python colab={"base_uri": "https://localhost:8080/"} id="kJ746b0eHWnW" outputId="9e16e10d-9de0-4746-8de6-77abba0ca886"
data.show(5)
data.printSchema()
data.describe().show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="R-C0YGyvSHdg" outputId="491bef48-ca9b-4524-ac50-bdc7dadbc68d"
titles = spark.read.csv(os.path.join(datasets_path,'ml-latest-small','movies.csv'),
                        inferSchema=True, header=True)

titles.show(5)
```

```python colab={"base_uri": "https://localhost:8080/"} id="2J9hnoYZSL2I" outputId="c0f8198d-4b87-48f3-e8c7-38dd4a9bae2d"
data = data.join(titles,data.movieId==titles.movieId,"left").select([data.movieId,
                                                              titles.title,
                                                              data.userId,
                                                              data.rating])
data.show(5)
```

```python id="OWuvHOj9U3eE"
from pyspark.sql.functions import rand, col, lit
```

```python colab={"base_uri": "https://localhost:8080/"} id="qrgN_LslN5yC" outputId="56ef1dd8-dff2-43da-f253-81070ad75f44"
data.orderBy(rand()).show(10,False)
data.groupBy('userId').count().orderBy('count',ascending=False).show(10,False)
data.groupBy('userId').count().orderBy('count',ascending=True).show(10,False)
data.groupBy('title').count().orderBy('count',ascending=False).show(10,False)
data.groupBy('title').count().orderBy('count',ascending=True).show(10,False)
```

```python id="dSnZO47lKeE1"
# Smaller dataset so we will use 0.8 / 0.2
(train_data, test_data) = data.randomSplit([0.8, 0.2], seed=42)
```

```python id="Fmmc35sDKuSD"
# Build the recommendation model using ALS on the training data
als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating")
# als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
model = als.fit(train_data)
```

```python id="PpE6lxY-Kw8p"
# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test_data)
```

```python colab={"base_uri": "https://localhost:8080/"} id="EL0aS_3lKzDg" outputId="065de754-bf4a-48bf-e5c9-4280820cd4f9"
predictions.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="tcZ_8vvWK21W" outputId="9942ac68-adfa-4024-f46e-621a6c5a0139"
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))
```

<!-- #region id="-NAYrHKoMVBd" -->
A NaN result is due to SPARK-14489 and because the model can't predict values for users for which there's no data. 
A temporary workaround is to exclude rows with predicted NaN values or to replace them with a constant, for instance,
the general mean rating. However, to map to a real business problem, the data scientist, in collaboration with the 
business owner, must define what happens if such an event occurs. For example, you can provide no recommendation for 
a user until that user rates a few items. Alternatively, before user rates five items, you can use a user-based recommender
system that's based on the user's profile (that's another recommender system to develop).

Replace predicted NaN values with the average rating and evaluate the model:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="NL57PNOYMehn" outputId="b7ffa688-8289-41d3-d945-710e7f439929"
avgRatings = data.select('rating').groupBy().avg().first()[0]
print('The average rating in the dataset is: {}'.format(avgRatings))

evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')
print('The root mean squared error for our model is: {}'.format(evaluator.evaluate(predictions.na.fill(avgRatings))))
```

<!-- #region id="wCX91jluM5sC" -->
Now exclude predicted NaN values and evaluate the model
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="YoQDazzYM7xM" outputId="a2af389e-e7af-4543-9087-837aa2c24012"
evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')
print ('The root mean squared error for our model is: {}'.format(evaluator.evaluate(predictions.na.drop())))
```

```python colab={"base_uri": "https://localhost:8080/"} id="EBQZPavLLXxb" outputId="8f3426b4-635a-4304-d0a5-502f71745f19"
single_user = test_data.filter(test_data['userId']==12).select(['movieId','userId'])
single_user.show()

recommendations = model.transform(single_user)
recommendations.orderBy('prediction', ascending=False).show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="CLo5h_suLhQB" outputId="4d0ebafb-0016-437d-8c02-924b0ce11ded"
#create dataset of all distinct movies 
unique_movies=data.select('movieId').distinct()
unique_movies.count()

#assigning alias name 'a' to unique movies df
a = unique_movies.alias('a')

#selecting a user
user_id=12

#creating another dataframe which contains already watched movie by active user 
watched_movies=indexed.filter(indexed['userId'] == user_id).select('movieId').distinct()
watched_movies.count()

#assigning alias name 'b' to watched movies df
b=watched_movies.alias('b')

#joining both tables on left join 
total_movies = a.join(b, a.movieId == b.movieId,how='left')

#selecting movies which active user is yet to rate or watch
remaining_movies=total_movies.where(col("b.movieId").isNull()).select(a.movieId).distinct()
remaining_movies=remaining_movies.withColumn("userId",lit(int(user_id)))

#making recommendations using ALS recommender model and selecting only top 'n' movies
recommendations=model.transform(remaining_movies).orderBy('prediction',ascending=False)
recommendations.show(5,False)
```
