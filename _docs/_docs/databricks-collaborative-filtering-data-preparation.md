---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
---

<!-- #region id="Ei97LLNo8syp" -->
# Build a product recommender databricks CF 01 - data preparation
> Preparing data for an end to end collaborative filtering based recommender system

- toc: true
- badges: true
- comments: true
- categories: [retail, databricks, ETL]
- image:
<!-- #endregion -->

<!-- #region application/vnd.databricks.v1+cell={"title": "", "showTitle": false, "inputWidgets": {}, "nuid": "a31b69ac-94f3-4d37-9c55-1fadd1049d8c"} id="QzPrQXgf8VLw" -->
The purpose of this notebook is to prepare the dataset we will use to explore collaborative filtering recommenders.  This notebook should be run on a **Databricks 7.1+ cluster**.
<!-- #endregion -->

<!-- #region application/vnd.databricks.v1+cell={"title": "", "showTitle": false, "inputWidgets": {}, "nuid": "cc5c3c39-13a0-46d9-a5d9-699e72f5cc5b"} id="dwOEb6vM8VMC" -->
## Introduction 

Collaborative filters are an important enabler of modern recommendation experiences.  ***Customers like you also bought***-type recommendations provide us an important means of identifying products that are likely to be of interest based on the buying patterns of closely related customers:

<img src="https://brysmiwasb.blob.core.windows.net/demos/images/instacart_collabrecom.png" width="600">
<!-- #endregion -->

```python application/vnd.databricks.v1+cell={"title": "Import Required Libraries", "showTitle": true, "inputWidgets": {}, "nuid": "8a90bb8d-d0e0-40e5-a178-2736cf06b844"} id="b3qtlX8f8VME" outputId="0c7e5b82-5db4-4e67-e3b9-1c1e7d9405f1"
from pyspark.sql.types import *
from pyspark.sql.functions import count, countDistinct, avg, log, lit, expr

import shutil
```

<!-- #region application/vnd.databricks.v1+cell={"title": "", "showTitle": false, "inputWidgets": {}, "nuid": "8e8ca031-4399-4a93-a9cd-4f6183415b92"} id="3P26iEQk8VMH" -->
## Step 1: Load the Data

The basic building block of this kind of recommendation is customer transaction data. To provide us data of this type, we'll be using the popular [Instacart dataset](https://www.kaggle.com/c/instacart-market-basket-analysis). This dataset provides cart-level details on over 3 million grocery orders placed by over 200,000 Instacart users across of portfolio of nearly 50,000 products.

**NOTE** Due to the terms and conditions by which these data are made available, anyone interested in recreating this work will need to download the data files from Kaggle and upload them to a folder structure as described below.

The primary data files available for download are organized as follows under a pre-defined [mount point](https://docs.databricks.com/data/databricks-file-system.html#mount-object-storage-to-dbfs) that we have named */mnt/instacart*:

<img src='https://brysmiwasb.blob.core.windows.net/demos/images/instacart_filedownloads.png' width=250>



Read into dataframes, these files form the following data model which captures the products customers have included in individual transactions:

<img src='https://brysmiwasb.blob.core.windows.net/demos/images/instacart_schema2.png' width=300>

We will apply minimal transformations to this data, persisting it to the Delta Lake format for speedier access:
<!-- #endregion -->

```python application/vnd.databricks.v1+cell={"title": "Create Database", "showTitle": true, "inputWidgets": {}, "nuid": "74260186-6eeb-415e-9d8a-46035f29188b"} id="J01s4JCW8VMJ" outputId="eff9473d-3201-470b-8f81-c01ded1140ed"
_ = spark.sql('CREATE DATABASE IF NOT EXISTS instacart')
```

<!-- #region application/vnd.databricks.v1+cell={"title": "", "showTitle": false, "inputWidgets": {}, "nuid": "6be6b60d-396d-47b2-979f-2ef972ddf2ad"} id="Wq3Ocu8i8VML" -->
**NOTE** The orders data set is pre-split into *prior* and *training* datasets.  Because date information in this dataset is very limited, we'll need to work with these pre-defined splits.  We'll treat the *prior* dataset as our ***calibration*** dataset and we'll treat the *training* dataset as our ***evaluation*** dataset. To minimize confusion, we'll rename these as part of our data preparation steps.
<!-- #endregion -->

```python application/vnd.databricks.v1+cell={"title": "Orders", "showTitle": true, "inputWidgets": {}, "nuid": "b3f752ca-b326-44c2-b7a6-9fb6276517df"} id="ZbvFIWcK8VMM" outputId="4edf4902-d8c8-47f3-9a4a-56ef302ecfc6"
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS instacart.orders')

# drop any old delta lake files that might have been created
shutil.rmtree('/dbfs/mnt/instacart/silver/orders', ignore_errors=True)

# define schema for incoming data
orders_schema = StructType([
  StructField('order_id', IntegerType()),
  StructField('user_id', IntegerType()),
  StructField('eval_set', StringType()),
  StructField('order_number', IntegerType()),
  StructField('order_dow', IntegerType()),
  StructField('order_hour_of_day', IntegerType()),
  StructField('days_since_prior_order', FloatType())
  ])

# read data from csv
orders = (
  spark
    .read
    .csv(
      '/mnt/instacart/bronze/orders',
      header=True,
      schema=orders_schema
      )
  )

# rename eval_set entries
orders_transformed = (
  orders
    .withColumn('split', expr("CASE eval_set WHEN 'prior' THEN 'calibration' WHEN 'train' THEN 'evaluation' ELSE NULL END"))
    .drop('eval_set')
  )

# write data to delta
(
  orders_transformed
    .write
    .format('delta')
    .mode('overwrite')
    .save('/mnt/instacart/silver/orders')
  )

# make accessible as spark sql table
_ = spark.sql('''
  CREATE TABLE instacart.orders
  USING DELTA
  LOCATION '/mnt/instacart/silver/orders'
  ''')

# present the data for review
display(
  spark.table('instacart.orders')
  )
```

```python application/vnd.databricks.v1+cell={"title": "Products", "showTitle": true, "inputWidgets": {}, "nuid": "38ee2b03-caba-48e1-9c0f-46a8b16b9a9b"} id="RdbJu9158VMO" outputId="4456c4f8-a6cf-465b-f016-2fee5a89240a"
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS instacart.products')

# drop any old delta lake files that might have been created
shutil.rmtree('/dbfs/mnt/instacart/silver/products', ignore_errors=True)

# define schema for incoming data
products_schema = StructType([
  StructField('product_id', IntegerType()),
  StructField('product_name', StringType()),
  StructField('aisle_id', IntegerType()),
  StructField('department_id', IntegerType())
  ])

# read data from csv
products = (
  spark
    .read
    .csv(
      '/mnt/instacart/bronze/products',
      header=True,
      schema=products_schema
      )
  )

# write data to delta
(
  products
    .write
    .format('delta')
    .mode('overwrite')
    .save('/mnt/instacart/silver/products')
  )

# make accessible as spark sql table
_ = spark.sql('''
  CREATE TABLE instacart.products
  USING DELTA
  LOCATION '/mnt/instacart/silver/products'
  ''')

# present the data for review
display(
  spark.table('instacart.products')
  )
```

```python application/vnd.databricks.v1+cell={"title": "Order Products", "showTitle": true, "inputWidgets": {}, "nuid": "5322e224-1ff7-4c32-bc35-96477725436a"} id="k0hMAUOX8VMQ" outputId="df7b3f26-795a-4993-f833-82ce5009cecd"
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS instacart.order_products')

# drop any old delta lake files that might have been created
shutil.rmtree('/dbfs/mnt/instacart/silver/order_products', ignore_errors=True)

# define schema for incoming data
order_products_schema = StructType([
  StructField('order_id', IntegerType()),
  StructField('product_id', IntegerType()),
  StructField('add_to_cart_order', IntegerType()),
  StructField('reordered', IntegerType())
  ])

# read data from csv
order_products = (
  spark
    .read
    .csv(
      '/mnt/instacart/bronze/order_products',
      header=True,
      schema=order_products_schema
      )
  )

# write data to delta
(
  order_products
    .write
    .format('delta')
    .mode('overwrite')
    .save('/mnt/instacart/silver/order_products')
  )

# make accessible as spark sql table
_ = spark.sql('''
  CREATE TABLE instacart.order_products
  USING DELTA
  LOCATION '/mnt/instacart/silver/order_products'
  ''')

# present the data for review
display(
  spark.table('instacart.order_products')
  )
```

```python application/vnd.databricks.v1+cell={"title": "Departments", "showTitle": true, "inputWidgets": {}, "nuid": "f74e60aa-c172-477c-a3bd-1da5a378df56"} id="SJhtzOFd8VMS" outputId="e5ceda93-6efc-4d4a-9470-6c81a57c3409"
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS instacart.departments')

# drop any old delta lake files that might have been created
shutil.rmtree('/dbfs/mnt/instacart/silver/departments', ignore_errors=True)

# define schema for incoming data
departments_schema = StructType([
  StructField('department_id', IntegerType()),
  StructField('department', StringType())  
  ])

# read data from csv
departments = (
  spark
    .read
    .csv(
      '/mnt/instacart/bronze/departments',
      header=True,
      schema=departments_schema
      )
  )

# write data to delta
(
  departments
    .write
    .format('delta')
    .mode('overwrite')
    .save('/mnt/instacart/silver/departments')
  )

# make accessible as spark sql table
_ = spark.sql('''
  CREATE TABLE instacart.departments
  USING DELTA
  LOCATION '/mnt/instacart/silver/departments'
  ''')

# present the data for review
display(
  spark.table('instacart.departments')
  )
```

```python application/vnd.databricks.v1+cell={"title": "Aisles", "showTitle": true, "inputWidgets": {}, "nuid": "b9fa5946-979c-414e-9eac-643443682611"} id="YppOnGRC8VMT" outputId="15d5fc34-273a-40ca-c867-3f5e64bf87e1"
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS instacart.aisles')

# drop any old delta lake files that might have been created
shutil.rmtree('/dbfs/mnt/instacart/silver/aisles', ignore_errors=True)

# define schema for incoming data
aisles_schema = StructType([
  StructField('aisle_id', IntegerType()),
  StructField('aisle', StringType())  
  ])

# read data from csv
aisles = (
  spark
    .read
    .csv(
      '/mnt/instacart/bronze/aisles',
      header=True,
      schema=aisles_schema
      )
  )

# write data to delta
(
  aisles
    .write
    .format('delta')
    .mode('overwrite')
    .save('/mnt/instacart/silver/aisles')
  )

# make accessible as spark sql table
_ = spark.sql('''
  CREATE TABLE instacart.aisles
  USING DELTA
  LOCATION '/mnt/instacart/silver/aisles'
  ''')

# present the data for review
display(
  spark.table('instacart.aisles')
  )
```

<!-- #region application/vnd.databricks.v1+cell={"title": "", "showTitle": false, "inputWidgets": {}, "nuid": "385f89f1-0c9d-45d8-a1c3-ae54b9426dc0"} id="fX37TttN8VMW" -->
## Step 2: Derive Product *Ratings*

For our collaborative filter (CF), we need a way to understand user preferences for individual products. In some scenarios, explicit user ratings, such as a 3 out of 5 stars rating, may be provided, but not every interaction receives a rating and in many transactional engagements the idea of asking customers for such ratings just seems out of place. In these scenarios, we might use other user-generated data to indicate product preferences. In the context of the Instacart dataset, the frequency of product purchases by a user may serve as such an indicator:
<!-- #endregion -->

```python application/vnd.databricks.v1+cell={"title": "", "showTitle": false, "inputWidgets": {}, "nuid": "44767615-7c06-43ea-bf7e-8d87e6994556"} id="aOsUcLeO8VMW" outputId="415978c1-d8b9-49c3-f647-3ecf5d77130d"
# drop any old delta lake files that might have been created
shutil.rmtree('/dbfs/mnt/instacart/gold/ratings__user_product_orders', ignore_errors=True)

# identify number of times product purchased by user
user_product_orders = (
  spark
    .table('instacart.orders')
    .join(spark.table('instacart.order_products'), on='order_id')
    .groupBy('user_id', 'product_id', 'split')
    .agg( count(lit(1)).alias('purchases') )
  )

# write data to delta
(
  user_product_orders
    .write
    .format('delta')
    .mode('overwrite')
    .save('/mnt/instacart/gold/ratings__user_product_orders')
  )

# display results
display(
  spark.sql('''
    SELECT * 
    FROM DELTA.`/mnt/instacart/gold/ratings__user_product_orders` 
    ORDER BY split, user_id, product_id
    ''')
)
```

<!-- #region application/vnd.databricks.v1+cell={"title": "", "showTitle": false, "inputWidgets": {}, "nuid": "9ea8adf7-dab9-448b-a1f0-b1679214d5ff"} id="pRJe6VZU8VMX" -->
Using product purchases as *implied ratings* presents us with a scaling problem.  Consider a scenario where a user purchases a given product 10 times while another user purchases a product 20 times.  Does the first user have a stronger preference for the product?  What if we new the first customer has made 10 purchases in total so that this product was included in each checkout event while the second user had made 50 total purchases, only 20 of which included the product of interest?  Does our understanding of the users preferences change in light of this additional information?

Rescaling our data to account for differences in overall purchase frequency will provide us a more reliable basis for the comparison of users. There are several options for doing this, but because of how we intend to measure the similarity between users (to provide the basis of collaborative filtering), our preference will be to use what is referred to as L2-normalization.

To understand L2-normalization, consider two users who have purchased products X and Y. The first user has purchased product X 10 times and product Y 5 times. The second user has purchased products X and Y 20 times each.  We might plot these purchases (with product X on the x-axis and product Y on the y-axis) as follows:

<img src='https://brysmiwasb.blob.core.windows.net/demos/images/lsh_norm01.png' width=380>

To determine similarity, we'll be measuring the (Euclidean) distance between the points formed at the intersection of these two axes, *i.e.* the peak of the two triangles in the graphic.  Without rescaling, the first user resides about 11 units from the origin and the second user resides about 28 units.  Calculating the distance between these two users in this space would provide a measure of both differing product preferences AND purchase frequencies. Rescaling the distance each user resides from the origin of the space eliminates the differences related to purchase frequencies, allowing us to focus on differences in product preferences:

<img src='https://brysmiwasb.blob.core.windows.net/demos/images/lsh_norm02.png' width=400>

The rescaling is achieved by calculating the Euclidean distance between each user and the origin - there's no need to limit ourselves to two-dimensions for this math to work - and then dividing each product-specific value for that user by this distance which is referred to as the L2-norm.  Here, we apply the L2-norm to our implied ratings:
<!-- #endregion -->

```python application/vnd.databricks.v1+cell={"title": "", "showTitle": false, "inputWidgets": {}, "nuid": "5d841933-49ae-45a2-85ad-07ed67c4c0e0"} id="pyyZIkuc8VMZ" outputId="68492835-655a-455d-e41f-3252ee716371"
%sql
DROP VIEW IF EXISTS instacart.user_ratings;

CREATE VIEW instacart.user_ratings 
AS
  WITH ratings AS (
    SELECT
      split,
      user_id,
      product_id,
      SUM(purchases) as purchases
    FROM DELTA.`/mnt/instacart/gold/ratings__user_product_orders`
    GROUP BY split, user_id, product_id
    )
  SELECT
    a.split,
    a.user_id,
    a.product_id,
    a.purchases,
    a.purchases/b.l2_norm as normalized_purchases
  FROM ratings a
  INNER JOIN (
    SELECT
      split,
      user_id,
      POW( 
        SUM(POW(purchases,2)),
        0.5
        ) as l2_norm
    FROM ratings
    GROUP BY user_id, split
    ) b
    ON a.user_id=b.user_id AND a.split=b.split;
  
SELECT * FROM instacart.user_ratings ORDER BY user_id, split, product_id;
```

<!-- #region application/vnd.databricks.v1+cell={"title": "", "showTitle": false, "inputWidgets": {}, "nuid": "64183cf2-f524-4f2a-a54f-2aac566a433d"} id="uClmlMWV8VMa" -->
You may have noted that we elected to implement these calculations through a view.  If we consider the values for a user must be recalculated with each purchase event by that user as that event will impact the value of the L2-norm by which each implied rating is adjusted. Persisting raw purchase counts in our base *ratings* table provides us an easy way to incrementally add new information to this table without having to re-traverse a user's entire purchase history.  Aggregating and normalizing the values in that table on the fly through a view gives us an easy way to extract normalized data with less ETL effort.

It's important to consider which data is included in these calculations. Depending on your scenario, it might be appropriate to limit the transaction history from which these *implied ratings* are derived to a period within which expressed preferences would be consistent with the user's preferences in the period over which the recommender might be used.  In some scenarios, this may mean limiting historical data to a month, quarter, year, etc.  In other scenarios, this may mean limiting historical data to periods with comparable seasonal components as the current or impending period.  For example, a user may have a strong preference for pumpkin spice flavored products in the Fall but may not be really keen on it during the Summer months.  For demonstration purposes, we'll just use the whole transaction history as the basis of our ratings but this is a point you'd want to carefully consider for a real-world implementation.
<!-- #endregion -->

<!-- #region application/vnd.databricks.v1+cell={"title": "", "showTitle": false, "inputWidgets": {}, "nuid": "d826b64b-b3eb-4f1e-baf4-be9aae420f39"} id="j-l7bxAH8VMb" -->
## Step 3: Derive Naive Product *Ratings*

A common practice when evaluating a recommender is to compare it to a prior or alternative recommendation engine to see which better helps the organization achieve its goals. To provide us a starting point for such comparisons, we might consider using overall product popularity as the basis for making *naive* collaborative recommendations. Here, we calculate normalized product ratings based on overall purchase frequencies to enable this work:
<!-- #endregion -->

```python application/vnd.databricks.v1+cell={"title": "", "showTitle": false, "inputWidgets": {}, "nuid": "8843f867-703e-4fca-a210-65fd735d8922"} id="NzkS8_aA8VMb" outputId="db2a10a9-c35f-4913-f121-63c4833f0f87"
%sql
DROP VIEW IF EXISTS instacart.naive_ratings;

CREATE VIEW instacart.naive_ratings 
AS
  WITH ratings AS (
    SELECT
      split,
      product_id,
      SUM(purchases) as purchases
    FROM DELTA.`/mnt/instacart/gold/ratings__user_product_orders`
    GROUP BY split, product_id
    )
  SELECT
    a.split,
    a.product_id,
    a.purchases,
    a.purchases/b.l2_norm as normalized_purchases
  FROM ratings a
  INNER JOIN (
    SELECT
      split,
      POW( 
        SUM(POW(purchases,2)),
        0.5
        ) as l2_norm
    FROM ratings
    GROUP BY split
    ) b
    ON a.split=b.split;
  
SELECT * FROM instacart.naive_ratings ORDER BY split, product_id;
```
