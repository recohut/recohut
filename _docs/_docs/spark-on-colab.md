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

<!-- #region id="4NYBaZN83C6A" -->
# Spark on Colab
> Run spark on google colab

- toc: true
- badges: true
- comments: true
- categories: [spark, pyspark]
- image:
<!-- #endregion -->

<!-- #region id="kvD4HBMi0ohY" -->
## Install Java, Spark, and Findspark
This installs Apache Spark 3.0.0, Java 8, and [Findspark](https://github.com/minrk/findspark), a library that makes it easy for Python to find Spark.
<!-- #endregion -->

<!-- #region id="3-pFvzkpvlZs" -->
> Note: If you get ```HTTP request sent, awaiting response... 404 Not Found``` error, Go to this [link](https://downloads.apache.org/spark) to find the latest version.
<!-- #endregion -->

```python id="fUhBhrGmyAvs" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="ee0be783-ca9e-4211-db94-614a0aa74727"
#hide-output
!apt-get install openjdk-8-jdk-headless -qq > /dev/null
!wget https://downloads.apache.org/spark/spark-3.1.2/spark-3.1.2-bin-hadoop3.2.tgz
!tar -xvf spark-3.1.2-bin-hadoop3.2.tgz
!pip install findspark
```

<!-- #region id="b4Kjvk_h1AHl" -->
## Set Environment Variables
Set the locations where Spark and Java are installed.
<!-- #endregion -->

```python id="8Xnb_ePUyQIL"
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.1.2-bin-hadoop3.2"
```

<!-- #region id="NwU28K5f1H3P" -->
## Start a SparkSession
This will start a local Spark session.
<!-- #endregion -->

```python id="zgReRGl0y23D"
import findspark
findspark.init()
from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local[*]").getOrCreate()
```

<!-- #region id="T3ULPx4Y1LiR" -->
## Use Spark!
That's all there is to it - you're ready to use Spark!
<!-- #endregion -->

```python id="XJp8ZI-VzYEz" colab={"base_uri": "https://localhost:8080/", "height": 222} outputId="f8b33305-730e-47b7-e1ae-915d51419166"
df = spark.createDataFrame([{"hello": "world"} for x in range(1000)])
df.show(3)
```
