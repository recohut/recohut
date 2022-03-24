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

<!-- #region id="Ogxr7vpABetD" -->
# PedidosYa Product Recommender System using TFRS
<!-- #endregion -->

<!-- #region id="BXJY8c9d4Xi5" -->
## Installations
<!-- #endregion -->

```python id="DctyNOSdx-7h"
!pip install -q tensorflow-recommenders==0.6.0
!pip install -q scann==1.2.3
```

<!-- #region id="T9Lfnzum3VqJ" -->
## Git
<!-- #endregion -->

```python id="_CnY6rQ-3Vnm"
!git clone -b T308050 https://github.com/sparsh-ai/general-recsys.git
```

```python colab={"base_uri": "https://localhost:8080/"} id="wUXzAWlg3d49" executionInfo={"status": "ok", "timestamp": 1636799756937, "user_tz": -330, "elapsed": 20, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2cf3cdc8-5a43-48a2-9fee-aa013149930f"
%cd general-recsys
```

<!-- #region id="GB_yDppW3_Yt" -->
## Imports
<!-- #endregion -->

```python id="vrEmNkAAsQlM"
import numpy as np
from tqdm.notebook import tqdm
import sys
import os
import logging
import pandas as pd
import pandas_gbq
from os import path as osp
from pathlib import Path

from typing import Dict, Text

import matplotlib.pyplot as plt
import seaborn as sns

from google.cloud import bigquery
from google.cloud.bigquery_storage_v1beta1 import BigQueryStorageClient

import tensorflow as tf
import tensorflow_recommenders as tfrs
```

```python colab={"base_uri": "https://localhost:8080/"} id="G9TB4wJucLzv" executionInfo={"status": "ok", "timestamp": 1636725513441, "user_tz": -330, "elapsed": 24026, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d40cce2c-1e76-46d3-e650-8b23611f99a8"
from google.colab import auth
from google.colab import drive

auth.authenticate_user()
print('Authenticated')

IN_COLAB = True
```

<!-- #region id="NyxCtlrJ3_Ta" -->
## Params
<!-- #endregion -->

```python id="MXBwnUCD3_RD"
class Args:
    datapath_bronze = '/content'
    datapath_silver = '/content'

args = Args()
```

<!-- #region id="Q40X4lHf4JHw" -->
## Logger
<!-- #endregion -->

```python id="cibwpV5L4JFb"
logging.basicConfig(stream=sys.stdout,
                    level = logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] : %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

logger = logging.getLogger('Logger')
```

<!-- #region id="qY9Y0q2sz1MS" -->
## Utils
<!-- #endregion -->

<!-- #region id="xt_fzKKr0pMF" -->
## Dataset
<!-- #endregion -->

<!-- #region id="spgnC7lm5wjn" -->
### Define SQL queries
<!-- #endregion -->

<!-- #region id="HV5pjVGo2Ehf" -->
**interactions.sql**
<!-- #endregion -->

<!-- #region id="n5nvCH6F13Nb" -->
```sql
CREATE OR REPLACE TABLE `peya-food-and-groceries.user_fiorella_dirosario.order_sep2020_sep2021` AS
SELECT
    o.user.id as user_id
     , o.is_pre_order
     ,o.pickup
     ,o.discount_type
     , o.order_id
     , o.registered_at as timestamp
FROM `peya-bi-tools-pro.il_core.fact_orders` o
WHERE registered_date BETWEEN "2020-09-01" AND "2021-09-30"
  AND o.business_type.business_type_id = 2
  AND o.order_status = "CONFIRMED"
  AND o.country.country_id = 3
```
<!-- #endregion -->

<!-- #region id="f3GRgJ9F2B1G" -->
**order_details.sql**
<!-- #endregion -->

<!-- #region id="8kGe2oZV19QJ" -->
```sql
CREATE OR REPLACE TABLE `peya-food-and-groceries.user_fiorella_dirosario.order_details_sep2020_sep2021` AS
SELECT
    o.order_id
     , CASE WHEN SPLIT(d.integration_code, "-")[SAFE_OFFSET(0)] IN ("D", "P") THEN SPLIT(d.integration_code, "-")[SAFE_OFFSET(2)]
            WHEN SPLIT(d.integration_code, "-")[SAFE_OFFSET(0)] IN ("S", "G") THEN SPLIT(d.integration_code, "-")[SAFE_OFFSET(3)]
            ELSE dp.gtin
    END as gtin

     , CASE WHEN SPLIT(d.integration_code, "-")[SAFE_OFFSET(0)] IN ("D", "P", "S", "G") THEN 1
            WHEN dp.gtin IS NOT NULL THEN 1
            ELSE 0
    END as has_gtin
     , d.integration_code
     , d.product.product_id
     , d.product_name
     , o.restaurant.id as partner_id
     , o.restaurant.name as partner_name
FROM `peya-bi-tools-pro.il_core.fact_orders` o
   , UNNEST(details) d
         LEFT JOIN `peya-bi-tools-pro.il_core.dim_product` dp ON dp.id = d.product.product_id
WHERE registered_date BETWEEN "2020-09-01" AND "2021-09-30"
  AND o.business_type.business_type_id = 2
  AND o.order_status = "CONFIRMED"
  AND o.country.country_id = 3
```
<!-- #endregion -->

<!-- #region id="JqiByeU82OOv" -->
**product_attributes.sql**
<!-- #endregion -->

<!-- #region id="it6yi4Ji19ZU" -->
```sql
CREATE OR REPLACE TABLE `peya-food-and-groceries.user_fiorella_dirosario.product_attributes_sep2020_sep2021` AS
    WITH info AS (
        SELECT
            od.gtin as gtin
             , MIN(dp.date_created) as date_created
             , MIN(brand_id) as brand_id
             , MIN(cb.name) as brand_name
             , MIN(cb.group_id) as brand_group_id
             , MIN(cbg.name) as brand_group_name
             , MIN(category_id) as category_id
             , MIN(cc.name) as category_name
        FROM `peya-food-and-groceries.user_fiorella_dirosario.order_details_sep2020_sep2021` od
                 INNER JOIN `peya-bi-tools-pro.il_core.dim_product` dp ON dp.id = od.product_id
                 LEFT JOIN `peya-data-origins-pro.cl_catalogue.product` cp ON cp.gtin = od.gtin
                 LEFT JOIN `peya-data-origins-pro.cl_catalogue.brand` cb ON cb.id = cp.brand_id
                 LEFT JOIN `peya-data-origins-pro.cl_catalogue.category` cc ON cc.id = cp.category_id

                 LEFT JOIN `peya-data-origins-pro.cl_catalogue.brand_group` cbg ON cbg.id = cb.group_id
        GROUP BY 1
    )
    SELECT
        gtin
         , category_id
         , category_name
         , brand_id
         , brand_name
         , brand_group_id
         , brand_group_name
         , DATE_DIFF("2021-09-30", DATE(date_created), DAY) as age
    FROM info
```
<!-- #endregion -->

<!-- #region id="03dZmbQB2UqJ" -->
**user_attributes.sql**
<!-- #endregion -->

<!-- #region id="PFEcjGhc19gS" -->
```sql
CREATE OR REPLACE TABLE `peya-food-and-groceries.user_fiorella_dirosario.user_attributes_sep2020_sep2021` AS
    WITH most_freq_attr AS (
        SELECT
            user.id as user_id
             , APPROX_TOP_COUNT(o.city.city_id, 1)[OFFSET(0)].value AS most_frequent_city
             , APPROX_TOP_COUNT(o.application, 1)[OFFSET(0)].value AS most_frequent_platform
        FROM `peya-bi-tools-pro.il_core.fact_orders` o
        WHERE registered_date BETWEEN "2020-09-01" AND "2021-09-30"
          AND o.order_status = "CONFIRMED"
          AND o.country.country_id = 3
        GROUP BY 1
    )
       , users_period AS
        (
            SELECT
                DISTINCT
                user_id
            FROM `peya-food-and-groceries.user_fiorella_dirosario.user_order_sep2020_sep2021`
        )
    SELECT
        up.user_id
         , freq.most_frequent_city as city_id
         , freq.most_frequent_platform as platform
         , us.segment
         ,last_order_date
         ,days_from_first_order
    FROM users_period up
             LEFT JOIN most_freq_attr freq ON freq.user_id = up.user_id
             LEFT JOIN `peya-growth-and-onboarding.automated_tables_reports.user_segments` us ON us.user_id = up.user_id
        AND DATE(us.date) = "2021-09-30"
```
<!-- #endregion -->

<!-- #region id="Zv_8RKzY6Ke8" -->
**Query for 2-tower model**
<!-- #endregion -->

<!-- #region id="tOUROWgy0qYJ" -->
We use the interaction between user ans product for user that has at least 2 orders.
<!-- #endregion -->

```python id="Hc2eVHNwthsI"
MINIMUM_PRODUCTS = 5
MINIMUM_ORDERS = 2
data_project_id = "peya-food-and-groceries"
data_dataset_id = "user_fiorella_dirosario"
data_table_orders = "order_sep2020_sep2021"
data_table_order_details = "order_details_sep2020_sep2021"
data_table_users = "attributes_sep2020_sep2021"
data_table_products = "product_attributes_sep2020_sep2021"

interaction_query_train = f"""
DECLARE minimum_products INT64;
DECLARE minimum_orders INT64;

SET minimum_products = {MINIMUM_PRODUCTS};
SET minimum_orders = {MINIMUM_ORDERS};

WITH products_by_user AS (
    SELECT
        uo.user_id
      , COUNT(DISTINCT gtin) as cant_products
      , COUNT(DISTINCT uo.order_id) as cant_orders
      , MAX(uo.order_id) as last_order_id
    FROM 
      `{data_project_id}.{data_dataset_id}.{data_table_orders}` as uo
    JOIN 
      `{data_project_id}.{data_dataset_id}.{data_table_order_details}` as od
    ON 
      uo.order_id = od.order_id  
    WHERE 
      uo.user_id IS NOT NULL
      AND od.gtin IS NOT NULL
      AND od.has_gtin = 1
    GROUP BY 1
)
SELECT DISTINCT
    CAST(uo.user_id AS STRING) AS user_id
  , CAST(od.gtin AS STRING) AS product_id
  --, uo.order_id
  --, uo.timestamp
FROM 
  `{data_project_id}.{data_dataset_id}.{data_table_orders}` as uo
JOIN 
  `{data_project_id}.{data_dataset_id}.{data_table_order_details}` as od
ON uo.order_id = od.order_id
LEFT JOIN products_by_user pbu ON pbu.user_id = uo.user_id
WHERE uo.user_id IS NOT NULL
  AND od.gtin IS NOT NULL
  AND od.has_gtin = 1
  AND cant_products >= minimum_products
  AND cant_orders >= minimum_orders
  AND uo.order_id != pbu.last_order_id 
"""

interaction_query_test = f"""
DECLARE minimum_products INT64;
DECLARE minimum_orders INT64;

SET minimum_products = {MINIMUM_PRODUCTS};
SET minimum_orders = {MINIMUM_ORDERS};

WITH products_by_user AS (
    SELECT
        uo.user_id
      , COUNT(DISTINCT gtin) as cant_products
      , COUNT(DISTINCT uo.order_id) as cant_orders
      , MAX(uo.order_id) as last_order_id
    FROM 
      `{data_project_id}.{data_dataset_id}.{data_table_orders}` as uo
    JOIN 
      `{data_project_id}.{data_dataset_id}.{data_table_order_details}` as od
    ON 
      uo.order_id = od.order_id  
    WHERE 
      uo.user_id IS NOT NULL
      AND od.gtin IS NOT NULL
      AND od.has_gtin = 1
    GROUP BY 1
)
SELECT DISTINCT
    CAST(uo.user_id AS STRING) AS user_id
  , CAST(od.gtin AS STRING) AS product_id
  --, uo.order_id
  --, uo.timestamp
FROM 
  `{data_project_id}.{data_dataset_id}.{data_table_orders}` as uo
JOIN 
  `{data_project_id}.{data_dataset_id}.{data_table_order_details}` as od
ON
  uo.order_id = od.order_id
LEFT JOIN 
  products_by_user pbu 
ON 
  pbu.user_id = uo.user_id
WHERE 
  uo.user_id IS NOT NULL
  AND od.gtin IS NOT NULL
  AND od.has_gtin = 1
  AND cant_products >= minimum_products
  AND cant_orders >= minimum_orders
  AND uo.order_id = pbu.last_order_id 
"""

product_query = f"""
DECLARE minimum_products INT64;
DECLARE minimum_orders INT64;

SET minimum_products = {MINIMUM_PRODUCTS};
SET minimum_orders = {MINIMUM_ORDERS};


WITH products_by_user AS (
    SELECT
        uo.user_id
      , COUNT(DISTINCT od.gtin) as cant_products
      , COUNT(DISTINCT uo.order_id) as cant_orders
      , MAX(uo.order_id) as last_order_id
    FROM 
      `{data_project_id}.{data_dataset_id}.{data_table_orders}` as uo
    JOIN 
      `{data_project_id}.{data_dataset_id}.{data_table_order_details}` as od
    ON 
      uo.order_id = od.order_id  
    WHERE 
      uo.user_id IS NOT NULL
      AND od.gtin IS NOT NULL
      AND od.has_gtin = 1
    GROUP BY 1
    HAVING 
      cant_products >= minimum_products AND cant_orders >= minimum_orders
)
, products AS (
  SELECT DISTINCT
    od.gtin
  FROM
    `{data_project_id}.{data_dataset_id}.{data_table_order_details}` od
  JOIN 
    `{data_project_id}.{data_dataset_id}.{data_table_orders}` uo ON uo.order_id = od.order_id
  JOIN 
    products_by_user pbu ON pbu.user_id = uo.user_id
  WHERE od.gtin IS NOT NULL AND od.has_gtin = 1
)

SELECT 
    CAST(pa.gtin AS STRING) AS product_id
  , IF(pa.category_id IS NULL, "", CAST(pa.category_id AS STRING))  AS category_id
  , IF(pa.brand_id IS NULL, "", CAST(pa.brand_id AS STRING)) AS brand_id
  , CAST(pa.age AS STRING) AS age
FROM 
  `{data_project_id}.{data_dataset_id}.{data_table_products}` pa
JOIN products p ON p.gtin = pa.gtin
"""

user_query = f"""
DECLARE minimum_products INT64;
DECLARE minimum_orders INT64;

SET minimum_products = {MINIMUM_PRODUCTS};
SET minimum_orders = {MINIMUM_ORDERS};

WITH products_by_user AS (
    SELECT
    uo.user_id
    , COUNT(DISTINCT gtin) as cant_products
    , COUNT(DISTINCT uo.order_id) as cant_orders
    , MAX(uo.order_id) as last_order_id
    FROM 
    `{data_project_id}.{data_dataset_id}.{data_table_orders}` as uo
    JOIN 
    `{data_project_id}.{data_dataset_id}.{data_table_order_details}` as od
    ON uo.order_id = od.order_id  
    WHERE uo.user_id IS NOT NULL
    AND od.gtin IS NOT NULL
    AND od.has_gtin = 1
    GROUP BY 1
)
SELECT 
    CAST(ua.user_id AS STRING) AS user_id
  , CAST(ua.city_id AS STRING) AS city_id
  , ua.platform
  , IF(ua.segment IS NULL, "Not set", ua.segment) AS segment
FROM 
  `{data_project_id}.{data_dataset_id}.{data_table_users}` ua
LEFT JOIN products_by_user pbu ON pbu.user_id = ua.user_id
WHERE cant_products >= minimum_products
      AND cant_orders >= minimum_orders
"""
```

<!-- #region id="Rq8DHT_e6Pmy" -->
**Query for 2-tower model with context**
<!-- #endregion -->

```python id="h9UAZPxl51RI"
MINIMUM_PRODUCTS = 5
MINIMUM_ORDERS = 2
data_project_id = "peya-food-and-groceries"
data_dataset_id = "user_fiorella_dirosario"
data_table_orders = "order_sep2020_sep2021"
data_table_order_details = "order_details_sep2020_sep2021"
data_table_users = "attributes_sep2020_sep2021"
data_table_products = "product_attributes_sep2020_sep2021"

interaction_query_train = f"""
DECLARE minimum_products INT64;
DECLARE minimum_orders INT64;

SET minimum_products = {MINIMUM_PRODUCTS};
SET minimum_orders = {MINIMUM_ORDERS};

WITH products_by_user AS (
    SELECT
        uo.user_id
      , COUNT(DISTINCT gtin) as cant_products
      , COUNT(DISTINCT uo.order_id) as cant_orders
      , MAX(uo.order_id) as last_order_id
    FROM 
      `{data_project_id}.{data_dataset_id}.{data_table_orders}` as uo
    JOIN 
      `{data_project_id}.{data_dataset_id}.{data_table_order_details}` as od
    ON 
      uo.order_id = od.order_id  
    WHERE 
      uo.user_id IS NOT NULL
      AND od.gtin IS NOT NULL
      AND od.has_gtin = 1
    GROUP BY 1
)
SELECT DISTINCT
    uo.user_id
  , EXTRACT(DAYOFWEEK FROM uo.timestamp) as dow
  , EXTRACT(hour FROM uo.timestamp) as hod
  , od.gtin
  , IF(od.partner_id IS NULL, -1, od.partner_id) AS partner_id
  , IF(pa.category_id IS NULL, -1, pa.category_id) AS category_id
  , IF(pa.brand_id IS NULL, -1, pa.brand_id) AS brand_id
  , IF(pa.age IS NULL, 1000, pa.age) as age
FROM 
  `{data_project_id}.{data_dataset_id}.{data_table_orders}` as uo
JOIN 
  `{data_project_id}.{data_dataset_id}.{data_table_order_details}` as od
ON 
  uo.order_id = od.order_id
LEFT JOIN 
  products_by_user pbu 
ON pbu.user_id = uo.user_id
LEFT JOIN 
  `{data_project_id}.{data_dataset_id}.{data_table_products}` as pa
ON 
  pa.gtin = od.gtin
WHERE uo.user_id IS NOT NULL
  AND od.gtin IS NOT NULL
  AND od.has_gtin = 1
  AND cant_products >= minimum_products
  AND cant_orders >= minimum_orders
  AND uo.order_id != pbu.last_order_id 
"""

user_query_query_test = f"""
DECLARE minimum_products INT64;
DECLARE minimum_orders INT64;

SET minimum_products = {MINIMUM_PRODUCTS};
SET minimum_orders = {MINIMUM_ORDERS};

WITH products_by_user AS (
    SELECT
        uo.user_id
      , COUNT(DISTINCT gtin) as cant_products
      , COUNT(DISTINCT uo.order_id) as cant_orders
      , MAX(uo.order_id) as last_order_id
    FROM 
      `{data_project_id}.{data_dataset_id}.{data_table_orders}` as uo
    JOIN 
      `{data_project_id}.{data_dataset_id}.{data_table_order_details}` as od
    ON 
      uo.order_id = od.order_id  
    WHERE 
      uo.user_id IS NOT NULL
      AND od.gtin IS NOT NULL
      AND od.has_gtin = 1
    GROUP BY 1
)
SELECT DISTINCT
    uo.user_id
  , EXTRACT(DAYOFWEEK FROM uo.timestamp) as dow
  , EXTRACT(hour FROM uo.timestamp) as hod
FROM 
  `{data_project_id}.{data_dataset_id}.{data_table_orders}` as uo
JOIN 
  `{data_project_id}.{data_dataset_id}.{data_table_order_details}` as od
ON 
  uo.order_id = od.order_id
LEFT JOIN 
  products_by_user pbu 
ON pbu.user_id = uo.user_id
LEFT JOIN 
  `{data_project_id}.{data_dataset_id}.{data_table_products}` as pa
ON 
  pa.gtin = od.gtin
WHERE uo.user_id IS NOT NULL
  AND od.gtin IS NOT NULL
  AND od.has_gtin = 1
  AND cant_products >= minimum_products
  AND cant_orders >= minimum_orders
  AND uo.order_id = pbu.last_order_id 
"""
```

<!-- #region id="LOLxNOMg5rlh" -->
### Fetching data from Google BigQuery
<!-- #endregion -->

```python pycharm={"name": "#%%\n"} id="xEHVmL-BnR7V"
project_id = "peya-data-analyt-factory-stg"  # ["peya-data-analyt-factory-stg", "peya-food-and-groceries", "peya-growth-and-onboarding"]
client = bigquery.client.Client(project=project_id)
bq_storage_client = BigQueryStorageClient()

interactions_train = (
    client.query(interaction_query_train)
        .result()
        .to_arrow(bqstorage_client=bq_storage_client)
        .to_pandas()
)

interactions_test = (
    client.query(interaction_query_test)
        .result()
        .to_arrow(bqstorage_client=bq_storage_client)
        .to_pandas()
)

users = (
    client.query(user_query)
        .result()
        .to_arrow(bqstorage_client=bq_storage_client)
        .to_pandas()
)
products = (
    client.query(product_query)
        .result()
        .to_arrow(bqstorage_client=bq_storage_client)
        .to_pandas()
)

user_ids = users["user_id"].unique().tolist()
product_ids = products["product_id"].unique().tolist()
```

```python id="KylQcll3Cbmi" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1635442155270, "user_tz": 180, "elapsed": 69957, "user": {"displayName": "Cesar Reyes", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GivCEktqhpVTBc89dTYts8q4--6-tXOlPrzRm22=s64", "userId": "04702276980078964785"}} outputId="f03d4e3e-9e93-46ed-d774-34b21891f977"
interactions_train_ds = tf.data.Dataset.from_tensor_slices(interactions_train.to_dict(orient="list"))
interactions_train_ds = interactions_train_ds.map(lambda x: {"user_id": x["user_id"], "product_id": x["product_id"]})

users_ds = tf.data.Dataset.from_tensor_slices(users.to_dict(orient="list"))
products_ds = tf.data.Dataset.from_tensor_slices(products.to_dict(orient="list"))

tf.random.set_seed(42)
interactions_train_ds = interactions_train_ds.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

for k in interactions_train_ds.take(1).as_numpy_iterator():
    print(k)
for k in products_ds.take(1).as_numpy_iterator():
    print(k)
for k in users_ds.take(1).as_numpy_iterator():
    print(k)
```

```python id="Z6klo3wjgqJ3"
products_ds = products_ds.map(lambda x: x["product_id"])
users_ds = users_ds.map(lambda x: x["user_id"])
```

```python pycharm={"name": "#%%\n"} id="ddcbEvz_Hl4t"
project_id = "peya-growth-and-onboarding"  # ["peya-data-analyt-factory-stg", "peya-food-and-groceries", "peya-growth-and-onboarding"]
client = bigquery.client.Client(project=project_id)
bq_storage_client = BigQueryStorageClient()

interactions_train = (
    client.query(interaction_query_train)
        .result()
        .to_arrow(bqstorage_client=bq_storage_client)
        .to_pandas()
        .fillna(-1)
)

query_test = (
    client.query(user_query_query_test)
        .result()
        .to_arrow(bqstorage_client=bq_storage_client)
        .to_pandas()
        .fillna(-1)
)
```

```python id="ApwLr3xEisib"
if not os.path.exists("data"):
    os.makedirs("data")
```

```python id="kqTTezdKtIhi"
query_fields = [
    'user_id',
    'dow',
    'hod',
]
candidate_fields = [
    'gtin',
    # 'product_name',
    # 'partner_id',
    # 'partner_name',
    'category_id',
    'brand_id',
    # 'age'
]
```

```python id="tQS1gtUvkXPH" pycharm={"is_executing": true}
step = 100_000
total_samples = len(interactions_train)
for k in range(0, total_samples, step):
    interactions_train.loc[k:k + step, query_fields + candidate_fields].to_csv(f"data/interactions_{k}.csv",
                                                                               index=False)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="mkzLG47xvD1y" executionInfo={"status": "ok", "timestamp": 1635445267180, "user_tz": 180, "elapsed": 4210, "user": {"displayName": "Cesar Reyes", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GivCEktqhpVTBc89dTYts8q4--6-tXOlPrzRm22=s64", "userId": "04702276980078964785"}} outputId="2afbdc58-50e8-4dfb-9b93-990f8b2a9c64"
queries = interactions_train[query_fields].drop_duplicates()
candidates = interactions_train[candidate_fields].drop_duplicates()

user_ids = queries['user_id'].unique()
user_ids = user_ids[user_ids >= 0]

gtin_ids = candidates['gtin'].unique()
#product_names = candidates['product_name'].unique()
category_ids = candidates['category_id'].unique()
category_ids = category_ids[category_ids >= 0]
brand_ids = candidates['brand_id'].unique()
brand_ids = brand_ids[brand_ids >= 0]
# candidates['age'].mean()
# candidates['age'].var()
candidates
```

```python colab={"base_uri": "https://localhost:8080/"} id="kPI2Qfx9kkHj" executionInfo={"status": "ok", "timestamp": 1635445286965, "user_tz": 180, "elapsed": 19798, "user": {"displayName": "Cesar Reyes", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GivCEktqhpVTBc89dTYts8q4--6-tXOlPrzRm22=s64", "userId": "04702276980078964785"}} outputId="a2555566-e3cc-4cff-f4a3-a5192432e9c1"
tf.random.set_seed(42)
interactions_train_ds = tf.data.experimental.make_csv_dataset(
    "data/interactions*.csv",
    batch_size=8192,
    num_epochs=1,
    num_parallel_reads=20,
    shuffle_buffer_size=10000
).prefetch(tf.data.AUTOTUNE).cache().shuffle(100000, seed=42)
users_ds = tf.data.Dataset.from_tensor_slices(queries.to_dict(orient="list"))
products_ds = tf.data.Dataset.from_tensor_slices(candidates.to_dict(orient="list"))

for k in products_ds.take(1).as_numpy_iterator():
    print(k)
for k in users_ds.take(1).as_numpy_iterator():
    print(k)
```

```python pycharm={"name": "#%%\n"} id="W3IT78rbgTI9"
project_id = "peya-data-analyt-factory-stg"  # ["peya-data-analyt-factory-stg", "peya-food-and-groceries", "peya-growth-and-onboarding"]
client = bigquery.client.Client(project=project_id)
bq_storage_client = BigQueryStorageClient()

interactions_train = (
    client.query(interaction_query_train)
        .result()
        .to_arrow(bqstorage_client=bq_storage_client)
        .to_pandas()
        .fillna(-1)
)

query_test = (
    client.query(user_query_query_test)
        .result()
        .to_arrow(bqstorage_client=bq_storage_client)
        .to_pandas()
        .fillna(-1)
)
```

```python id="njq0pfys-6oB"
if not os.path.exists("data"):
    os.makedirs("data")
```

```python id="X5Fxx8yl-6oB"
query_fields = [
    'user_id',
    'dow',
    'hod',
]
candidate_fields = [
    'gtin',
]
```

```python id="pPZpswa9-6oC"
step = 100_000
total_samples = len(interactions_train)
for k in range(0, total_samples, step):
    interactions_train.loc[k:k + step, query_fields + candidate_fields].to_csv(f"data/interactions_{k}.csv",
                                                                               index=False)

```

```python id="MWUxwiG5-6oC"
queries = interactions_train[query_fields].drop_duplicates()
candidates = interactions_train[candidate_fields].drop_duplicates()

user_ids = queries['user_id'].unique()
user_ids = user_ids[user_ids >= 0]
gtin_ids = candidates['gtin'].unique()
```

```python colab={"base_uri": "https://localhost:8080/"} id="1JqzRY1D-6oD" executionInfo={"status": "ok", "timestamp": 1635434925134, "user_tz": 180, "elapsed": 17193, "user": {"displayName": "Cesar Reyes", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GivCEktqhpVTBc89dTYts8q4--6-tXOlPrzRm22=s64", "userId": "04702276980078964785"}} outputId="55770cba-a12d-4cd2-9097-5358e5e8cad9"
tf.random.set_seed(42)
interactions_train_ds = tf.data.experimental.make_csv_dataset(
    "data/interactions*.csv",
    batch_size=8192,
    num_epochs=1,
    num_parallel_reads=20,
    shuffle_buffer_size=10000
).prefetch(tf.data.AUTOTUNE).cache().shuffle(100000, seed=42)
users_ds = tf.data.Dataset.from_tensor_slices(queries.to_dict(orient="list"))
products_ds = tf.data.Dataset.from_tensor_slices(candidates.to_dict(orient="list"))

for k in products_ds.take(1).as_numpy_iterator():
    print(k)
for k in users_ds.take(1).as_numpy_iterator():
    print(k)
```

<!-- #region id="COcDkuIR_isf" -->
## EDA
<!-- #endregion -->

<!-- #region id="lbCwnISK_iqN" -->
### Age Distribution

Older products are more likely to be recommended than the younger ones. That's why we need to know how many products were at least 6 months in the app.
<!-- #endregion -->

```python id="d39a87f0" outputId="1da2f275-99e2-49ea-f9b7-0fe3de4e6b17"
QUERY = """
SELECT
gtin
, age
FROM `peya-food-and-groceries.user_fiorella_dirosario.product_attributes_sep2020_sep2021` p 
"""

df = pandas_gbq.read_gbq(QUERY, project_id="peya-growth-and-onboarding")
df.to_csv("gtin_age.csv", index=False)
```

```python id="8938a99a"
df = pd.read_csv("gtin_age.csv")

# Age is in days. I need it in years
df["age"] = df["age"] / 365
```

```python id="87d6ab60" outputId="e742a89e-2532-4b0e-f49b-77109f9b8885"
sns.histplot(data=df, x="age", bins=[x/10 for x in range(25) if (x % 2 == 0)])
plt.xlabel("Age (years)")
plt.ylabel("Number of Products")
plt.xticks([x/10 for x in range(25) if (x % 2 == 0)])
plt.title("Distribution of the Age of the Product")
total = df["gtin"].count()

ax = plt.gca()

if total > 0:
    for p in ax.patches:
        if pd.notna(p.get_height()):
            ax.text(p.get_x() + p.get_width() / 2., p.get_height(),
                       '{:.1%}'.format(float(p.get_height() / total)),
                       fontsize=10, color='black', ha='center', va='bottom')
        else:
            ax.text(p.get_x() + p.get_width() / 2., 0, '0', fontsize=10,
                       color='black', ha='center', va='bottom')
plt.show()
```

<!-- #region id="ylDB7oCT_ioE" -->
### Distribution of Sales for different Categories

We want to check if sales distribution change across categories for different days of the week. Besides of this, the distribution by hour may change for each day. This could show us that the context is importante for the model
<!-- #endregion -->

```python id="2df4bea9"
# Plot day of the week sales distribution for different categories

def plot_categories_by_day(categories):
    _, ax = plt.subplots(2, 2, sharey=False, figsize=(15, 15))

    for i, category in zip([[0, 0], [0, 1], [1, 0], [1, 1]], categories):

        df_2 = df.loc[df.category_name.isin([category]), :]

        df_gr = df_2.groupby(["category_name", "day_name"])[["times"]].sum().reset_index()

        sns.barplot(data=df_gr, x="day_name", y="times", ax=ax[i[0], i[1]], order=["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"])
        ax[i[0], i[1]].set_title(category)
        ax[i[0], i[1]].set_xlabel("")
        ax[i[0], i[1]].set_ylabel("Productos Vendidos")

    plt.suptitle("Ventas por Día de la Semana")
    plt.show()
    
    return None
```

```python id="36b6e76c"
# Plot hourly sales distribution for different days for a category

def plot_category_by_day(category, days):

    _, ax = plt.subplots(2, 2, sharey=False, figsize=(15, 15))

    for i, day in zip([[0, 0], [0, 1], [1, 0], [1, 1]], days):

        df_2 = df.loc[(df.category_name == category) &
                      (df.day_name == day), :]

        df_gr = df_2.groupby(["category_name", "day_name", "HOUR"])[["times"]].sum().reset_index()

        sns.barplot(data=df_gr, x="HOUR", y="times", ax=ax[i[0], i[1]])
        ax[i[0], i[1]].set_title(day)
        ax[i[0], i[1]].set_xlabel("")
        ax[i[0], i[1]].set_ylabel("Productos Vendidos")

    plt.suptitle(f"Ventas por Día de la Semana y Hora para {category}")
    plt.show()
    
    return None
```

```python id="8889bfb9"
# Parameters
DAYS = {1: "Domingo", 2: "Lunes", 3: "Martes", 4: "Miércoles", 5: "Jueves", 6: "Viernes", 7: "Sábado"}
```

```python id="c216c0f7" outputId="34bf7bee-d92a-42ce-81b4-e42865da6623"
QUERY = """
SELECT
category_name
, EXTRACT(DAYOFWEEK FROM timestamp) as DAY
, EXTRACT(HOUR FROM timestamp) as HOUR
, COUNT(*) as times
FROM `peya-food-and-groceries.user_fiorella_dirosario.order_sep2020_sep2021` o
LEFT JOIN `peya-food-and-groceries.user_fiorella_dirosario.order_details_sep2020_sep2021` od ON od.order_id = o.order_id
LEFT JOIN `peya-food-and-groceries.user_fiorella_dirosario.product_attributes_sep2020_sep2021` p ON p.gtin=od.gtin
WHERE od.has_gtin=1
GROUP BY 1,2,3
ORDER BY 1,2,3
"""

df = pandas_gbq.read_gbq(QUERY, project_id="peya-growth-and-onboarding")
df.to_csv("category_day_hour.csv", index=False)

```

```python id="232a1691"
df = pd.read_csv("category_day_hour.csv")

df.loc[df.category_name.isin(['Cervezas Negras',
                              'Cervezas Rojas',
                              'Cervezas Rubias']), "category_name"] = "Cervezas"
                              
df["day_name"] = df.DAY.map(DAYS)

```

```python id="3c743a51" outputId="263e8ed1-d822-428a-b713-5268f67ba4e7"
plot_categories_by_day(['Alimentos Para Mascotas', 'Cervezas', 'Limpieza De Cocina', 'Papas Fritas'])
```

```python id="7543bbec" outputId="7669a136-7b80-4382-ca22-0cc7179fc098"
plot_category_by_day('Gaseosas', ["Martes", "Viernes", "Sábado", "Domingo"])
```

```python id="70ef1225" outputId="99e1bce2-14dd-4d77-ba2d-415732b994f6"
plot_category_by_day('Tintos', ["Martes", "Viernes", "Sábado", "Domingo"])
```

```python id="5d48b289" outputId="25910732-d9ea-49db-8361-e6ba513d6ce9"
plot_category_by_day('Cervezas', ["Martes", "Viernes", "Sábado", "Domingo"])
```

```python id="7e228b6b" outputId="0867c335-2164-462f-8a59-afde20f4e85b"
plot_category_by_day('Papas Fritas', ["Martes", "Viernes", "Sábado", "Domingo"])
```

```python id="292a5b59" outputId="25c4403c-d343-4f0a-d60c-e30b1b6f25af"
plot_category_by_day('Limpieza De Cocina', ["Martes", "Viernes", "Sábado", "Domingo"])
```

<!-- #region id="csWiNrIj_ild" -->
### Orders Distribution by User and Most Popular Markets
<!-- #endregion -->

```python id="40d78bfa"
COUNTRY = "Argentina"
```

```python id="dd3354f1" outputId="c0fed5e5-8f7c-4e12-9a08-5697051c5544"
QUERY = """
WITH most_frequent AS (
SELECT
o.user.id as user_id
, APPROX_TOP_COUNT(restaurant.name, 1)[OFFSET(0)].value as top_market

 FROM `peya-bi-tools-pro.il_core.fact_orders` o
 
 WHERE true
 AND registered_date BETWEEN "2020-09-01" AND "2021-09-30"
 AND business_type.business_type_id = 2
 AND o.order_status = "CONFIRMED"
GROUP BY 1
)

, min_date AS (
SELECT
o.user.id as user_id
, MIN(CASE WHEN business_type.business_type_id = 2 THEN o.registered_date END) as first_market_order_date
, MIN(CASE WHEN o.is_first_confirmed_order_user = 1 THEN o.registered_date END) as first_order_date 
 FROM `peya-bi-tools-pro.il_core.fact_orders` o
 
 WHERE true
 AND registered_date <= "2021-09-30"
 AND o.order_status = "CONFIRMED"
GROUP BY 1
)

SELECT

o.user.id as user_id
, u.country.country_name as country_name
, mf.top_market
, md.first_order_date
, DATE_TRUNC(md.first_order_date, MONTH) as first_order_month

, md.first_market_order_date
, DATE_TRUNC(md.first_market_order_date, MONTH) as first_market_order_month

, COUNT(DISTINCT o.order_id) as total_orders

, COUNT(DISTINCT DATE_TRUNC(o.registered_date, MONTH) ) as months_with_orders
, COUNT(DISTINCT CASE WHEN o.registered_date > md.first_market_order_date THEN o.order_id END) as total_orders_after_first_market
, COUNT(DISTINCT CASE WHEN o.registered_date > md.first_market_order_date THEN DATE_TRUNC(o.registered_date, MONTH) END) as months_with_orders_after_first_market

, COUNT(DISTINCT CASE WHEN business_type.business_type_id = 2 THEN o.order_id END) as total_orders_markets
, COUNT(DISTINCT CASE WHEN business_type.business_type_id = 2 AND restaurant.name = mf.top_market THEN DATE_TRUNC(o.registered_date, MONTH) END) as months_most_freq
, COUNT(DISTINCT CASE WHEN business_type.business_type_id = 2 AND restaurant.name = mf.top_market THEN o.order_id END) as orders_most_freq
, SAFE_DIVIDE(COUNT(DISTINCT CASE WHEN business_type.business_type_id = 2 AND restaurant.name = mf.top_market THEN o.order_id END), COUNT(DISTINCT CASE WHEN business_type.business_type_id = 2 THEN o.order_id END)) AS pct_orders_most_freq
, SAFE_DIVIDE(COUNT(DISTINCT CASE WHEN business_type.business_type_id = 2 AND restaurant.name = mf.top_market THEN DATE_TRUNC(o.registered_date, MONTH) END), COUNT(DISTINCT CASE WHEN business_type.business_type_id = 2 THEN DATE_TRUNC(o.registered_date, MONTH) END)) as pct_months_most_freq


, COUNT(DISTINCT CASE WHEN business_type.business_type_id = 2 AND DATE_TRUNC(o.registered_date, MONTH) = "2021-09-01" THEN o.order_id END) as market_orders_sept
, COUNT(DISTINCT CASE WHEN business_type.business_type_id = 2 AND DATE_TRUNC(o.registered_date, MONTH) = "2021-08-01" THEN o.order_id END) as market_orders_agosto
, COUNT(DISTINCT CASE WHEN business_type.business_type_id = 2 AND DATE_TRUNC(o.registered_date, MONTH) = "2021-07-01" THEN o.order_id END) as market_orders_julio
, COUNT(DISTINCT CASE WHEN business_type.business_type_id = 2 AND DATE_TRUNC(o.registered_date, MONTH) = "2021-06-01" THEN o.order_id END) as market_orders_junio
, COUNT(DISTINCT CASE WHEN business_type.business_type_id = 2 AND DATE_TRUNC(o.registered_date, MONTH) = "2021-05-01" THEN o.order_id END) as market_orders_mayo
, COUNT(DISTINCT CASE WHEN business_type.business_type_id = 2 AND DATE_TRUNC(o.registered_date, MONTH) = "2021-04-01" THEN o.order_id END) as market_orders_abril

, COUNT(DISTINCT CASE WHEN DATE_TRUNC(o.registered_date, MONTH) = "2021-09-01" THEN o.order_id END) as orders_sept
, COUNT(DISTINCT CASE WHEN DATE_TRUNC(o.registered_date, MONTH) = "2021-08-01" THEN o.order_id END) as orders_agosto
, COUNT(DISTINCT CASE WHEN DATE_TRUNC(o.registered_date, MONTH) = "2021-07-01" THEN o.order_id END) as orders_julio
, COUNT(DISTINCT CASE WHEN DATE_TRUNC(o.registered_date, MONTH) = "2021-06-01" THEN o.order_id END) as orders_junio
, COUNT(DISTINCT CASE WHEN DATE_TRUNC(o.registered_date, MONTH) = "2021-05-01" THEN o.order_id END) as orders_mayo
, COUNT(DISTINCT CASE WHEN DATE_TRUNC(o.registered_date, MONTH) = "2021-04-01" THEN o.order_id END) as orders_abril

FROM `peya-bi-tools-pro.il_core.fact_orders` o
INNER JOIN most_frequent mf ON mf.user_id = o.user.id
LEFT JOIN `peya-bi-tools-pro.il_core.dim_user` u ON u.user_id = o.user.id
LEFT JOIN min_date md ON md.user_id = o.user.id
WHERE o.registered_date BETWEEN "2020-09-01" AND "2021-10-04"
AND o.order_status = "CONFIRMED"
GROUP BY 1, 2, 3, 4, 5, 6, 7
"""

df = pandas_gbq.read_gbq(QUERY, project_id="peya-growth-and-onboarding", dialect='standard')

df.to_csv("data/orders_by_user.csv", index=False)
```

```python id="1575f62a"
df = pd.read_csv("data/orders_by_user.csv")
df = df[df.country_name == COUNTRY]
df.loc[(df.top_market.str.contains("PedidosYa Market")) &
       (~pd.isna(df.top_market)), "top_market"] = "PedidosYa Market"
       
df["has_1_order"] = df["total_orders_markets"] == 1
df["has_2_orders"] = df["total_orders_markets"] == 2
df["has_3_orders"] = df["total_orders_markets"] == 3
df["has_4_orders"] = df["total_orders_markets"] == 4
df["has_5_orders"] = df["total_orders_markets"] == 5
df["has_6_or_more_orders"] = df["total_orders_markets"] > 5

df["activo_agosto"] = df["market_orders_agosto"] > 0

df['first_market_order_month'] = pd.to_datetime(df['first_market_order_month'])
df['first_order_month'] = pd.to_datetime(df['first_order_month'])


df["is_acq"] = df['first_market_order_month'] == df['first_order_month']

df.loc[df['first_market_order_month'] < "2020-09-01", "first_market_order_month"] = "Before 2020-09"
df.loc[~(df['first_order_month'] >= "2020-09-01"), "first_order_month"] = "Before 2020-09"

df["quantity_orders"]  = df["total_orders_markets"]
df.loc[df["quantity_orders"] > 15, "quantity_orders"] = "+ 15"
```

```python id="7419545c" outputId="39e58055-76dc-4de1-b09c-7c3ea15824c1"
df.head()
```

<!-- #region id="c6d4c4eb" -->
**Quantity of orders by user group by first order month in Markets**
<!-- #endregion -->

```python id="b4a8b3ca" outputId="d1f99b27-5d9c-4e62-f1fd-e95926a1643f"
df.groupby(['first_market_order_month']).agg( users=("user_id", "nunique"),
                                              is_acq=("is_acq", "sum"),
                                              activo_agosto=("activo_agosto", "sum"),
                                              orders_agosto=("market_orders_agosto", "sum"),
                                              has_1_orders=("has_1_order", "mean"),
                                              has_2_orders=("has_2_orders", "mean"),
                                              has_3_orders=("has_3_orders", "mean"),
                                              has_4_orders=("has_4_orders", "mean"),
                                              has_5_orders=("has_5_orders", "mean"),
                                              has_6_or_more_orders=("has_6_or_more_orders", "mean")
                                              )
```

<!-- #region id="1e59d237" -->
**Active users in August by total orders of user**
<!-- #endregion -->

```python id="f91a84c4" outputId="05f5120e-d319-4f54-b3ba-ece6e9bbcf78"

df.groupby(["quantity_orders"]).agg(users=("user_id", "nunique"),
                                    activo_agosto=("activo_agosto", "sum"),
                                    orders_agosto=("market_orders_agosto", "sum")
                                    )
```

<!-- #region id="e20940ca" -->
**Users with more than 5 Orders**
<!-- #endregion -->

```python id="da7be5d3" outputId="08dfecd0-6c51-4879-9995-6851d83f8f17"
df_more_5 = df[df.total_orders_markets >= 5]
total = df_more_5.shape[0]

ax = plt.gca()

sns.histplot(data=df_more_5, x="pct_orders_most_freq",
             bins=[x / 10 for x in range(0, 11)], ax=ax)
             


for p in ax.patches:
      ax.text(p.get_x() + p.get_width()/2., p.get_height(),
                  '{:.1%}'.format(float(p.get_height()/total)),
              fontsize=10, color='black', ha='center', va='bottom')

ax.set_title("Percentage of Market orders in most frequent partner for users +5 orders")

plt.show()
```

```python id="1d2d394f" outputId="95bf8f64-1019-4fb8-f711-bb42d5ea926a"
df_gr = df_more_5.groupby(["top_market"])[["user_id"]].nunique().sort_values(by="user_id", ascending=False)

print(df_gr)
```

<!-- #region id="ZQbufjtx_ijD" -->
### Plotting a word cloud for most popular products in Markets
<!-- #endregion -->

```python id="9c9b9e0c"
# Function for Word Cloud Colors
def black_color_func(word, font_size, position,orientation,random_state=None, **kwargs):
    return("hsl(0,100%, 1%)")
```

```python id="1f8dc466" outputId="78e6392c-a4c3-48e5-cbc7-e02f80cbafe8"
QUERY = """
WITH product_name AS (
SELECT
gtin
, MAX(product_name) as name
FROM `peya-food-and-groceries.user_fiorella_dirosario.order_details_sep2020_sep2021`
WHERE has_gtin = 1 AND product_name IS NOT NULL
GROUP BY 1
)
SELECT
pn.name
, COUNT(*) as count
FROM `peya-food-and-groceries.user_fiorella_dirosario.order_details_sep2020_sep2021` od
LEFT JOIN product_name pn ON pn.gtin = od.gtin
WHERE has_gtin = 1
GROUP BY 1
"""

file = pandas_gbq.read_gbq(QUERY, project_id="peya-growth-and-onboarding")
file.to_csv("wordcloud.csv", index=False)
file = pd.read_csv(r"C:\Users\patricio.woodley\Documents\GitHub\repo-data-analytics-growth\dataton\wordcloud.csv")

print(file.head())
```

```python id="f50a7023" outputId="7a011f79-c2b9-42e6-a09c-39fce7935c94"
file = pd.read_csv("wordcloud.csv")

# Preprocessing
file = file.dropna()
file["count"] = file["count"].astype("int")
file = file.sort_values(by="count", ascending=False)
file = file.set_index(file["name"], drop=True)
del file["name"]

file_dict = file["count"].to_dict()

print(file_dict)
```

```python id="16e6a9ec" outputId="8d4d3718-944a-445c-9d6f-a5e0f95c6973"
wordcloud = WordCloud(font_path=r"C:\Users\patricio.woodley\Documents\GitHub\repo-data-analytics-growth\dataton\arial-unicode-ms.ttf",
                      background_color="white", width=2000, height=1500, max_words=150).generate_from_frequencies(file_dict)

wordcloud.recolor(color_func=black_color_func)
plt.figure(figsize=[15, 10])
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
```

<!-- #region id="W4jukMaJ0qTo" -->
## Models
<!-- #endregion -->

<!-- #region id="YKdhNt783sKx" -->
### Baseline
<!-- #endregion -->

```python id="8qVl7Nmk3sDg"
input_path = "models/base_line_model/index_model/1"
model = tf.saved_model.load(input_path)
```

<!-- #region id="9-bSMmUl3zs0" -->
Recommend products to out of bag users (cold start problem)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="OD1WkYsg3sA7" executionInfo={"status": "ok", "timestamp": 1636799912813, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b15aa3ff-3402-4a7a-c2f3-da385c02f72e"
score, products = model(tf.constant(["oob"]))
products.numpy()
```

```python colab={"base_uri": "https://localhost:8080/"} id="82CimZvz34X5" executionInfo={"status": "ok", "timestamp": 1636799914300, "user_tz": -330, "elapsed": 606, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="35d54e04-8da3-45b1-b83f-4a0f382d042f"
score, products = model(tf.constant([""]))
products.numpy()
```

<!-- #region id="HYUjZ0J636sf" -->
Recommend products to existent users
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="xfPu8Ltc372i" executionInfo={"status": "ok", "timestamp": 1636799916929, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f6ab443e-f268-4b34-9e22-f2b921399447"
score, products = model(tf.constant(["4655437"]))
products.numpy()
```

```python colab={"base_uri": "https://localhost:8080/"} id="oc6OdTux393f" executionInfo={"status": "ok", "timestamp": 1636799921199, "user_tz": -330, "elapsed": 820, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="784cf33f-a9f7-4e55-c963-904d74f5224d"
score, products = model(tf.constant(["35417324"]))
products.numpy()
```

<!-- #region id="uHSgLYKl0QaQ" -->
### Two-tower model
<!-- #endregion -->

<!-- #region id="z4S_J99QtqEj" -->
The most simple recommendation system model is a matrix factorization, so we build a FM using a Tensorflow to learn the user and product embedding or latent vector. The idea is get a numeric representation of users and product and then get the dot product to get the implicit preferences.
[Reference](https://www.tensorflow.org/recommenders/examples/basic_retrieval)
<!-- #endregion -->

```python id="cwMuldWCtpVB"
class MarketsModel(tfrs.Model):

    def __init__(
            self,
            user_model: tf.keras.Model,
            product_model: tf.keras.Model,
            task: tf.keras.layers.Layer
    ):
        super().__init__()
        self.user_model = user_model
        self.product_model = product_model
        self.task: tf.keras.layers.Layer = task

    def compute_loss(
            self,
            features: Dict[Text, tf.Tensor],
            training=False
    ) -> tf.Tensor:
        user_embeddings = self.user_model(features["user_id"])
        positive_product_embeddings = self.product_model(features["product_id"])
        compute_metrics = False if training else True
        return self.task(
            user_embeddings,
            positive_product_embeddings,
            compute_metrics=compute_metrics
        )
```

```python pycharm={"name": "#%%\n"} id="MN4uOlJYnR7Z"
embedding_dimension = 32

user_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(
        vocabulary=tf.convert_to_tensor(user_ids),
        mask_token=None
    ),
    tf.keras.layers.Embedding(len(user_ids) + 1, embedding_dimension)
])

product_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(
        vocabulary=tf.convert_to_tensor(product_ids)
        , mask_token=None
    ),
    tf.keras.layers.Embedding(len(product_ids) + 1, embedding_dimension)
])

metrics = tfrs.metrics.FactorizedTopK(
    candidates=products_ds.batch(128).map(product_model)
)

task = tfrs.tasks.Retrieval(
    metrics=metrics
)
```

```python id="QpO3m0mEeOhU"
model = MarketsModel(user_model, product_model, task)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
```

```python colab={"base_uri": "https://localhost:8080/"} id="LBLVmvHbuVJ8" executionInfo={"status": "ok", "timestamp": 1635442313798, "user_tz": 180, "elapsed": 157198, "user": {"displayName": "Cesar Reyes", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GivCEktqhpVTBc89dTYts8q4--6-tXOlPrzRm22=s64", "userId": "04702276980078964785"}} outputId="5f6b05da-84a3-440a-87ad-ac3a192b58e9"
cached_train = interactions_train_ds.batch(8192).cache()
history = model.fit(cached_train, epochs=5)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 284} id="fB2thoVeuYrJ" executionInfo={"status": "ok", "timestamp": 1635442314496, "user_tz": 180, "elapsed": 720, "user": {"displayName": "Cesar Reyes", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GivCEktqhpVTBc89dTYts8q4--6-tXOlPrzRm22=s64", "userId": "04702276980078964785"}} outputId="75ebc43b-88ae-4d5d-db1b-5989460d0ea8"
pd.DataFrame(history.history).plot()
```

<!-- #region id="EFNLD3Qt0QSZ" -->
### Two-tower model with context
<!-- #endregion -->

```python id="uH01zzrRbLxS"
class QueryModel(tf.keras.Model):

    def __init__(self, user_ids: list):
        super().__init__()

        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.IntegerLookup(vocabulary=user_ids),
            tf.keras.layers.Embedding(len(user_ids) + 1, 32)
        ])
        self.dow_embedding = tf.keras.Sequential([
            tf.keras.layers.IntegerLookup(vocabulary=[f"{k}" for k in range(7)]),
            tf.keras.layers.Embedding(8, 4)
        ])
        self.hod_embedding = tf.keras.Sequential([
            tf.keras.layers.IntegerLookup(vocabulary=[f"{k}" for k in range(24)]),
            tf.keras.layers.Embedding(25, 4)
        ])

    def call(self, inputs: Dict[Text, tf.Tensor]):
        return tf.concat([
            self.user_embedding(inputs["user_id"]),
            self.dow_embedding(inputs["dow"]),
            self.hod_embedding(inputs["hod"]),
        ], axis=1)
```

```python id="Vs5-XI0_xJJa"
class CandidateModel(tf.keras.Model):

    def __init__(
            self,
            gtin_ids: list,
            category_ids: list,
            brand_ids: list,
            age_mean: float = None,
            age_var: float = None
    ):
        super().__init__()

        #product_name_tokenizer = tf.keras.layers.TextVectorization()
        #product_name_tokenizer.adapt(products_ds.map(lambda x: x["product_name"]))
        #self.title_text_embedding = tf.keras.Sequential([
        #  product_name_tokenizer,
        #  tf.keras.layers.Embedding(input_dim=10_000, output_dim=32, mask_zero=True),
        #  tf.keras.layers.GlobalAveragePooling1D(),
        #])

        self.gtin_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=gtin_ids),
            tf.keras.layers.Embedding(len(gtin_ids) + 1, 32)
        ])

        self.category_id_embedding = tf.keras.Sequential([
            tf.keras.layers.IntegerLookup(vocabulary=category_ids, mask_token=None),
            tf.keras.layers.Embedding(len(category_ids) + 1, 6)
        ])

        self.brand_id_embedding = tf.keras.Sequential([
            tf.keras.layers.IntegerLookup(vocabulary=brand_ids, mask_token=None),
            tf.keras.layers.Embedding(len(brand_ids) + 1, 10)
        ])

        #self.normalized_age = tf.keras.layers.Normalization(mean=age_mean, variance=age_var)

    def call(self, inputs: Dict[Text, tf.Tensor]):
        return tf.concat([
            self.gtin_embedding(inputs["gtin"]),
            self.category_id_embedding(inputs["category_id"]),
            self.brand_id_embedding(inputs["brand_id"]),
            #tf.reshape(self.normalized_age(inputs["age"]), (-1, 1)),
        ], axis=1)
```

```python id="AkMbfYUNKXUL"
class MarketsModel(tfrs.models.Model):

    def __init__(
            self,
            query_model: tf.keras.Model,
            candidate_model: tf.keras.Model,
    ):
        super().__init__()
        self.query_model = tf.keras.Sequential([
            query_model,
            tf.keras.layers.Dense(32)
        ])
        self.candidate_model = tf.keras.Sequential([
            candidate_model,
            tf.keras.layers.Dense(32)
        ])
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=products_ds.batch(128).map(self.candidate_model),
            ),
        )

    def compute_loss(
            self,
            features: Dict[Text, tf.Tensor],
            training=False,
    ):
        query_embeddings = self.query_model({
            "user_id": features["user_id"],
            "dow": features["dow"],
            "hod": features["hod"],
        })
        candidate_embeddings = self.candidate_model({
            "gtin": tf.strings.as_string(features["gtin"]),
            "category_id": features["category_id"],
            "brand_id": features["brand_id"],
            #"age": features["age"],
        })
        compute_metrics = False if training else True

        return self.task(query_embeddings, candidate_embeddings, compute_metrics=compute_metrics)
```

```python id="ZUJKrtYpQDhZ"
query_model = QueryModel(user_ids)
product_model = CandidateModel(gtin_ids, category_ids, brand_ids)
```

```python colab={"base_uri": "https://localhost:8080/"} id="jl0ZQq9KqKjj" executionInfo={"status": "ok", "timestamp": 1635445287439, "user_tz": 180, "elapsed": 32, "user": {"displayName": "Cesar Reyes", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GivCEktqhpVTBc89dTYts8q4--6-tXOlPrzRm22=s64", "userId": "04702276980078964785"}} outputId="7ea99624-9e00-4e1c-ebe9-add73ac671ee"
for row in products_ds.batch(2).take(1):
    print(row)
    print(f"Computed representations: {product_model(row)[0, :].shape}")
```

```python colab={"base_uri": "https://localhost:8080/"} id="hvHhPVX6v5My" executionInfo={"status": "ok", "timestamp": 1635445287440, "user_tz": 180, "elapsed": 22, "user": {"displayName": "Cesar Reyes", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GivCEktqhpVTBc89dTYts8q4--6-tXOlPrzRm22=s64", "userId": "04702276980078964785"}} outputId="edf9c779-d329-4b4c-a500-f80fb82ccb10"
for row in users_ds.batch(2).take(1):
    print(row)
    print(f"Computed representations: {query_model(row)[0, :].shape}")
```

```python colab={"base_uri": "https://localhost:8080/"} id="YApXoYqgkZ3X" executionInfo={"status": "ok", "timestamp": 1635445287812, "user_tz": 180, "elapsed": 385, "user": {"displayName": "Cesar Reyes", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GivCEktqhpVTBc89dTYts8q4--6-tXOlPrzRm22=s64", "userId": "04702276980078964785"}} outputId="14363f65-8fa8-4d53-d6fc-2f0e947df2d8"
model = MarketsModel(query_model, product_model)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
```

```python colab={"base_uri": "https://localhost:8080/"} id="r9QY2LQwkrYy" executionInfo={"status": "ok", "timestamp": 1635445612381, "user_tz": 180, "elapsed": 324598, "user": {"displayName": "Cesar Reyes", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GivCEktqhpVTBc89dTYts8q4--6-tXOlPrzRm22=s64", "userId": "04702276980078964785"}} outputId="6eb52a74-5cf5-4ae5-97c0-ab265995ee4a"
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1)

history = model.fit(interactions_train_ds, epochs=2, callbacks=[callback])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 285} id="rs9uI45iR0PT" executionInfo={"status": "ok", "timestamp": 1635445613117, "user_tz": 180, "elapsed": 740, "user": {"displayName": "Cesar Reyes", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GivCEktqhpVTBc89dTYts8q4--6-tXOlPrzRm22=s64", "userId": "04702276980078964785"}} outputId="a22bada9-4662-4f4f-b03e-e512ebee0629"
pd.DataFrame(history.history)["loss"].plot()
```

<!-- #region id="WKKixLq60QUf" -->
### Two-tower model with context (simple)
<!-- #endregion -->

<!-- #region pycharm={"name": "#%% md\n"} id="aEDDXG50gTJE" -->
In this experiment we build a two tower model with 2 inputs, query and candidates. In this version we only add mode context in query tower using user_id, day of week and hour of the day.


![](https://1.bp.blogspot.com/-ww8cKT3nIb8/X2pdWAWWNmI/AAAAAAAADl8/pkeFRxizkXYbDGbOcaAnZkorjEuqtrabgCLcBGAsYHQ/s0/TF%2BRecommenders%2B06.gif)
<!-- #endregion -->

```python id="AMt7SJcy_Hu6"
class QueryModel(tf.keras.Model):

    def __init__(self, user_ids: list):
        super().__init__()

        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.IntegerLookup(vocabulary=user_ids),
            tf.keras.layers.Embedding(len(user_ids) + 1, 32)
        ])
        self.dow_embedding = tf.keras.Sequential([
            tf.keras.layers.IntegerLookup(vocabulary=[f"{k}" for k in range(7)]),
            tf.keras.layers.Embedding(8, 4)
        ])
        self.hod_embedding = tf.keras.Sequential([
            tf.keras.layers.IntegerLookup(vocabulary=[f"{k}" for k in range(24)]),
            tf.keras.layers.Embedding(25, 4)
        ])

    def call(self, inputs: Dict[Text, tf.Tensor]):
        return tf.concat([
            self.user_embedding(inputs["user_id"]),
            self.dow_embedding(inputs["dow"]),
            self.hod_embedding(inputs["hod"]),
        ], axis=1)
```

```python id="i8y2tZlK_Hu7"
class CandidateModel(tf.keras.Model):

    def __init__(self, gtin_ids):
        super().__init__()

        self.gtin_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=gtin_ids),
            tf.keras.layers.Embedding(len(gtin_ids) + 1, 32)
        ])

    def call(self, inputs: Dict[Text, tf.Tensor]):
        return self.gtin_embedding(inputs["gtin"])
```

```python id="9ye65VcG_Hu8"
class MarketsModel(tfrs.models.Model):

    def __init__(
            self,
            query_model: tf.keras.Model,
            candidate_model: tf.keras.Model,
    ):
        super().__init__()
        self.query_model = tf.keras.Sequential([
            query_model,
            tf.keras.layers.Dense(32)
        ])
        self.candidate_model = tf.keras.Sequential([
            candidate_model,
            tf.keras.layers.Dense(32)
        ])
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=products_ds.batch(128).map(self.candidate_model),
            ),
        )

    def compute_loss(
            self,
            features: Dict[Text, tf.Tensor],
            training=False,
    ):
        query_embeddings = self.query_model({
            "user_id": features["user_id"],
            "dow": features["dow"],
            "hod": features["hod"],
        })
        candidate_embeddings = self.candidate_model({"gtin": tf.strings.as_string(features["gtin"])})
        compute_metrics = False if training else True

        return self.task(query_embeddings, candidate_embeddings, compute_metrics=compute_metrics)
```

```python colab={"base_uri": "https://localhost:8080/"} id="LriPbASn_Hu9" executionInfo={"status": "ok", "timestamp": 1635434925137, "user_tz": 180, "elapsed": 46, "user": {"displayName": "Cesar Reyes", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GivCEktqhpVTBc89dTYts8q4--6-tXOlPrzRm22=s64", "userId": "04702276980078964785"}} outputId="3ec6d933-477d-4aa3-e2a5-687f81354b31"
query_model = QueryModel(user_ids)
product_model = CandidateModel(gtin_ids)
```

```python colab={"base_uri": "https://localhost:8080/"} id="XOucaXFP_Hu-" executionInfo={"status": "ok", "timestamp": 1635434925138, "user_tz": 180, "elapsed": 37, "user": {"displayName": "Cesar Reyes", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GivCEktqhpVTBc89dTYts8q4--6-tXOlPrzRm22=s64", "userId": "04702276980078964785"}} outputId="3e89c6ac-743b-4627-d4c7-be93ea040243"
for row in products_ds.batch(2).take(1):
    print(row)
    print(f"Computed representations: {product_model(row)[0, :].shape}")
```

```python colab={"base_uri": "https://localhost:8080/"} id="5DLqtIel_Hu_" executionInfo={"status": "ok", "timestamp": 1635434925139, "user_tz": 180, "elapsed": 33, "user": {"displayName": "Cesar Reyes", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GivCEktqhpVTBc89dTYts8q4--6-tXOlPrzRm22=s64", "userId": "04702276980078964785"}} outputId="2f849a24-5a76-4435-9847-60c1cefa4518"
for row in users_ds.batch(2).take(1):
    print(row)
    print(f"Computed representations: {query_model(row)[0, :].shape}")
```

```python colab={"base_uri": "https://localhost:8080/"} id="PdIWcSe5_HvA" executionInfo={"status": "ok", "timestamp": 1635434925139, "user_tz": 180, "elapsed": 29, "user": {"displayName": "Cesar Reyes", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GivCEktqhpVTBc89dTYts8q4--6-tXOlPrzRm22=s64", "userId": "04702276980078964785"}} outputId="155467c2-71a1-4337-ceda-d7a93420a323"
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=1000,
    decay_rate=0.9)

model = MarketsModel(query_model, product_model)
model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule))
```

```python colab={"base_uri": "https://localhost:8080/"} id="QOQyfHBi_HvB" executionInfo={"status": "ok", "timestamp": 1635436343764, "user_tz": 180, "elapsed": 1418650, "user": {"displayName": "Cesar Reyes", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GivCEktqhpVTBc89dTYts8q4--6-tXOlPrzRm22=s64", "userId": "04702276980078964785"}} outputId="334bcb02-2b3a-4410-bf55-db349c1174d4"
history = model.fit(
    interactions_train_ds,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)],
    epochs=50)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 282} id="Sn_8VNeK_HvB" executionInfo={"status": "ok", "timestamp": 1635436343765, "user_tz": 180, "elapsed": 29, "user": {"displayName": "Cesar Reyes", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GivCEktqhpVTBc89dTYts8q4--6-tXOlPrzRm22=s64", "userId": "04702276980078964785"}} outputId="323af32a-c158-4163-a4f1-ed2a1626da9a"
pd.DataFrame(history.history)["loss"].plot()
```

<!-- #region id="H2V6HcT3BAeI" -->
## Evaluation
<!-- #endregion -->

<!-- #region id="mGvoQc1oBAbx" -->
In order to calculate the evaluation metrics quickly, we are using Bigquery.
<!-- #endregion -->

```python id="78675334"
def get_metrics(table_id, k, project_id, minimum_orders=2, maximum_orders=1000, is_most_popular=False, is_context_model=False):
    
    if is_most_popular:
        comm1 = "--"
        comm2 = ""
    else:
        comm1 = ""
        comm2 = "--"
        
    if is_context_model:
        comm3 = "--"
        comm4 = ""
    else:
        comm3 = ""
        comm4 = "--"   

    query = f"""
    
    DECLARE minimum_products INT64;
    DECLARE minimum_orders INT64;
    DECLARE maximum_orders INT64;
    DECLARE k INT64;

    SET minimum_products = 5;
    SET minimum_orders = {minimum_orders};
    SET maximum_orders = {maximum_orders};
    SET k = {k};

    WITH products_by_user AS (
        SELECT
            uo.user_id
          , COUNT(DISTINCT gtin) as cant_products
          , COUNT(DISTINCT uo.order_id) as cant_orders
          , MAX(uo.order_id) as last_order_id
        FROM 
          `peya-food-and-groceries.user_fiorella_dirosario.order_sep2020_sep2021` as uo
        JOIN 
          `peya-food-and-groceries.user_fiorella_dirosario.order_details_sep2020_sep2021` as od
        ON 
          uo.order_id = od.order_id  
        WHERE 
          uo.user_id IS NOT NULL
          AND od.gtin IS NOT NULL
          AND od.has_gtin = 1
        GROUP BY 1
    )

    , test AS (
    SELECT DISTINCT
        CAST(uo.user_id AS STRING) AS user_id
      , CAST(od.gtin AS STRING) AS product_id
      --, uo.order_id
      --, uo.timestamp
    FROM 
      `peya-food-and-groceries.user_fiorella_dirosario.order_sep2020_sep2021` as uo
    JOIN 
      `peya-food-and-groceries.user_fiorella_dirosario.order_details_sep2020_sep2021` as od
    ON
      uo.order_id = od.order_id
    LEFT JOIN 
      products_by_user pbu 
    ON 
      pbu.user_id = uo.user_id
    WHERE 
      uo.user_id IS NOT NULL
      AND od.gtin IS NOT NULL
      AND od.has_gtin = 1
      AND cant_products >= minimum_products
      AND cant_orders >= minimum_orders
      AND cant_orders <= maximum_orders
      AND uo.order_id = pbu.last_order_id 
    )

    , predictions AS 

    (
        {comm3}SELECT * FROM `{table_id}` WHERE rank < k
        {comm4}SELECT CAST(user_id AS STRING) as user_id, gtin as product_id, rank FROM `{table_id}` WHERE rank < k
    )


    , cross_join AS (
    SELECT
    t.user_id
    , p.rank
    , p.product_id as product_predicted
    , t.product_id  as product_test
    , CASE WHEN t.product_id = p.product_id THEN 1 ELSE 0 END as hit_rate
    , CASE WHEN t.product_id = p.product_id THEN 1 / (p.rank + 1) ELSE 0 END as map_at_k
    FROM test t 
    {comm1}LEFT JOIN predictions p ON p.user_id = t.user_id
    {comm2}CROSS JOIN predictions p
    WHERE p.rank < k
    ORDER BY 1, 4, 2
    )

    , metric_by_user AS (
    SELECT
    user_id
    , SUM(hit_rate) as hit_rate
    , SUM(map_at_k) as map_at_k
    FROM cross_join 
    GROUP BY 1

    )

    SELECT
    COUNT(DISTINCT user_id) as users
    , AVG(hit_rate) as hit
    , AVG(map_at_k) as precision
    FROM metric_by_user
    
    """
    
    df = pandas_gbq.read_gbq(query, project_id=project_id, dialect='standard')

    return df

```

<!-- #region id="2ac0d8e7" -->
### Random Predictions
<!-- #endregion -->

```python id="6fef6bcb" outputId="ad050bd6-56b5-4fa8-cbb8-b7c0f60fd59f"
get_metrics(table_id='peya-growth-and-onboarding.user_patricio_woodley.user_recommendation_random', k=5, project_id='peya-growth-and-onboarding', is_most_popular=False)
```

```python id="eddb9531" outputId="33d8898b-e5a6-4569-8442-39f649a9b9db"
get_metrics(table_id='peya-growth-and-onboarding.user_patricio_woodley.user_recommendation_random', k=10, project_id='peya-growth-and-onboarding', is_most_popular=False)
```

<!-- #region id="3ffa8585" -->
### k-popular Products
<!-- #endregion -->

```python id="79a2c1b9" outputId="005b9658-3813-4fda-9195-da4ac2c301b7"
get_metrics(table_id='peya-food-and-groceries.user_fiorella_dirosario.popular_products_train', k=5, project_id='peya-growth-and-onboarding', is_most_popular=True)
```

```python id="f36d8151" outputId="9db43bce-3b82-49c4-e63b-27b5f137468c"
get_metrics(table_id='peya-food-and-groceries.user_fiorella_dirosario.popular_products_train', k=10, project_id='peya-growth-and-onboarding', is_most_popular=True)
```

```python id="302281f9" outputId="ded734d1-af55-493b-aeb0-674898d6a530"
get_metrics(table_id='peya-food-and-groceries.user_fiorella_dirosario.popular_products_train', k=50, project_id='peya-growth-and-onboarding', is_most_popular=True)
```

```python id="5dc15da5" outputId="d8de7b78-466b-4df8-f763-9caa09a42d02"
get_metrics(table_id='peya-food-and-groceries.user_fiorella_dirosario.popular_products_train', k=100, project_id='peya-growth-and-onboarding', is_most_popular=True)
```

```python id="8dd77e22" outputId="6544dedc-da81-4619-f4e8-8b48175bf748"
get_metrics(table_id='peya-food-and-groceries.user_fiorella_dirosario.popular_products_train', minimum_orders=2, maximum_orders=5, k=10, project_id='peya-growth-and-onboarding', is_most_popular=True)
```

```python id="2d5b68ff" outputId="e4152457-bec2-4824-a9ac-45c109c1bd4e"
get_metrics(table_id='peya-food-and-groceries.user_fiorella_dirosario.popular_products_train', minimum_orders=6, maximum_orders=10, k=10, project_id='peya-growth-and-onboarding', is_most_popular=True)
```

```python id="0d215d61" outputId="2f5315b6-7939-4b85-f882-aced1ba04029"
get_metrics(table_id='peya-food-and-groceries.user_fiorella_dirosario.popular_products_train', minimum_orders=11, maximum_orders=10000, k=10, project_id='peya-growth-and-onboarding', is_most_popular=True)
```

<!-- #region id="60d33aa8" -->
### Baseline Model
<!-- #endregion -->

```python id="12b5a6c1" outputId="016969d7-933e-448d-e23a-01e7e45fb79b"
get_metrics(table_id='peya-growth-and-onboarding.user_patricio_woodley.user_recommendation_baseline', k=5, project_id='peya-growth-and-onboarding', is_most_popular=False)
```

```python id="78010983" outputId="5f90e59c-4359-4dc3-8595-2bfd1022af45"
get_metrics(table_id='peya-growth-and-onboarding.user_patricio_woodley.user_recommendation_baseline', k=10, project_id='peya-growth-and-onboarding', is_most_popular=False)
```

```python id="96ef8632" outputId="c11038d8-1244-4a4d-f885-d2cef79b7fec"
get_metrics(table_id='peya-growth-and-onboarding.user_patricio_woodley.user_recommendation_baseline', k=10, minimum_orders=2, maximum_orders=5, project_id='peya-growth-and-onboarding', is_most_popular=False)
```

```python id="949e6565" outputId="ed4bf0d4-a029-4747-8eb0-a1b84196cbd7"
get_metrics(table_id='peya-growth-and-onboarding.user_patricio_woodley.user_recommendation_baseline', k=10, minimum_orders=6, maximum_orders=10, project_id='peya-growth-and-onboarding', is_most_popular=False)
```

```python id="e94690e2" outputId="892b3780-e9e5-4e89-9bc7-c2d640224ef2"
get_metrics(table_id='peya-growth-and-onboarding.user_patricio_woodley.user_recommendation_baseline', k=10, minimum_orders=11, maximum_orders=10000, project_id='peya-growth-and-onboarding', is_most_popular=False)
```

```python id="6a885109" outputId="a09a02f6-535d-41cf-c686-8b69963f35da"
get_metrics(table_id='peya-growth-and-onboarding.user_patricio_woodley.user_recommendation_baseline', k=10, minimum_orders=2, maximum_orders=5, project_id='peya-growth-and-onboarding', is_most_popular=False)
```

```python id="5e9928da" outputId="0b77631e-0e72-49c7-ed01-59f53b578aa7"
get_metrics(table_id='peya-growth-and-onboarding.user_patricio_woodley.user_recommendation_baseline', k=10, minimum_orders=6, maximum_orders=10, project_id='peya-growth-and-onboarding', is_most_popular=False)
```

```python id="9d98949d" outputId="4ec1721f-6a35-41be-ab55-7bf10a321928"
get_metrics(table_id='peya-growth-and-onboarding.user_patricio_woodley.user_recommendation_baseline', k=10, minimum_orders=11, maximum_orders=10000, project_id='peya-growth-and-onboarding', is_most_popular=False)
```

<!-- #region id="c7d2e9d2" -->
### Context Model
<!-- #endregion -->

```python id="c8ce9e2d" outputId="7b99e1bf-f051-467a-f9c1-5438ad36883e"
get_metrics(table_id="peya-food-and-groceries.user_fiorella_dirosario.recommendation_context_model", k=5, project_id='peya-growth-and-onboarding', is_context_model=True)
```

```python id="e5416732" outputId="0fc1ea4f-16f0-4490-dac7-5406522840f2"
get_metrics(table_id="peya-food-and-groceries.user_fiorella_dirosario.recommendation_context_model", k=10, project_id='peya-growth-and-onboarding', is_context_model=True)
```

```python id="d289b340" outputId="34b00b81-9742-4404-e41d-d415beb702cf"
get_metrics(table_id="peya-food-and-groceries.user_fiorella_dirosario.recommendation_context_model", minimum_orders=2, maximum_orders=5, k=10, project_id='peya-growth-and-onboarding', is_context_model=True)
```

```python id="3c3f8d27" outputId="3cb1c140-dc4f-4c8c-9ffe-1efff5050cbc"
get_metrics(table_id="peya-food-and-groceries.user_fiorella_dirosario.recommendation_context_model", minimum_orders=6, maximum_orders=10, k=10, project_id='peya-growth-and-onboarding', is_context_model=True)
```

```python id="d6db8ed8" outputId="b510ca24-a8ed-41a9-bfc7-ab4017ef33df"
get_metrics(table_id="peya-food-and-groceries.user_fiorella_dirosario.recommendation_context_model", minimum_orders=11, maximum_orders=10000, k=10, project_id='peya-growth-and-onboarding', is_context_model=True)
```

<!-- #region id="QE4zjjVp0ZQJ" -->
## Inference
<!-- #endregion -->

<!-- #region id="ESlsc9_34Y2a" -->
### Brute force

Brute force compute the product between user and all product.
<!-- #endregion -->

```python id="xAwmbZaffthN" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1635442314827, "user_tz": 180, "elapsed": 336, "user": {"displayName": "Cesar Reyes", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GivCEktqhpVTBc89dTYts8q4--6-tXOlPrzRm22=s64", "userId": "04702276980078964785"}} outputId="2b6734d2-ecd3-4257-a2ee-e042928274d1"
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
index.index_from_dataset(
    tf.data.Dataset.zip((products_ds.batch(100), products_ds.batch(100).map(model.product_model)))
)
```

```python id="wLzwkGE0mwr1" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1635442314829, "user_tz": 180, "elapsed": 13, "user": {"displayName": "Cesar Reyes", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GivCEktqhpVTBc89dTYts8q4--6-tXOlPrzRm22=s64", "userId": "04702276980078964785"}} outputId="6b0377b0-e0d7-4e47-b539-5501ef28c217"
# Get recommendations.
score, titles = index(tf.constant(["42"]))
print(f"Recommendations for user 42: {titles[0, :20]}")
```

```python colab={"base_uri": "https://localhost:8080/"} id="v54AGarmoeW2" executionInfo={"status": "ok", "timestamp": 1635442323520, "user_tz": 180, "elapsed": 8698, "user": {"displayName": "Cesar Reyes", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GivCEktqhpVTBc89dTYts8q4--6-tXOlPrzRm22=s64", "userId": "04702276980078964785"}} outputId="850bf572-fba3-4e0d-80a9-58276872b47e"
% timeit _, titles = index(tf.constant(["42"]))
```

<!-- #region id="tX7gV2LG411z" -->
### ScaNN
<!-- #endregion -->

<!-- #region id="ZZTAhrWR4q3Y" -->
Scann is an efficient Nearest neighbor approximation to retrieve the top k product.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="c3UGZvkSZSC7" executionInfo={"status": "ok", "timestamp": 1635442325017, "user_tz": 180, "elapsed": 1515, "user": {"displayName": "Cesar Reyes", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GivCEktqhpVTBc89dTYts8q4--6-tXOlPrzRm22=s64", "userId": "04702276980078964785"}} outputId="909460e9-8095-4e49-92d1-5cc0aef8cd47"
scann = tfrs.layers.factorized_top_k.ScaNN(model.user_model)
scann.index_from_dataset(
    tf.data.Dataset.zip((products_ds.batch(100), products_ds.batch(100).map(model.product_model)))
)
```

```python colab={"base_uri": "https://localhost:8080/"} id="CT5Mat0FaODX" executionInfo={"status": "ok", "timestamp": 1635442325020, "user_tz": 180, "elapsed": 29, "user": {"displayName": "Cesar Reyes", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GivCEktqhpVTBc89dTYts8q4--6-tXOlPrzRm22=s64", "userId": "04702276980078964785"}} outputId="c4630998-e9e3-4648-cc8b-440381e44394"
score, titles = scann(tf.constant(["42"]))
print(f"Recommendations for user 42: {titles[0, :20]}")
```

```python colab={"base_uri": "https://localhost:8080/"} id="RXcw9iHLLe5R" executionInfo={"status": "ok", "timestamp": 1635442337016, "user_tz": 180, "elapsed": 12008, "user": {"displayName": "Cesar Reyes", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GivCEktqhpVTBc89dTYts8q4--6-tXOlPrzRm22=s64", "userId": "04702276980078964785"}} outputId="febdf660-82f4-4757-9187-00140a8a37cb"
% timeit score, titles = scann(tf.constant(["42"]))
```

```python colab={"base_uri": "https://localhost:8080/"} id="ROE2_nWWlwtn" executionInfo={"status": "ok", "timestamp": 1635445614993, "user_tz": 180, "elapsed": 1888, "user": {"displayName": "Cesar Reyes", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GivCEktqhpVTBc89dTYts8q4--6-tXOlPrzRm22=s64", "userId": "04702276980078964785"}} outputId="8b94c235-7797-48c5-9a19-982e4e540172"
scann = tfrs.layers.factorized_top_k.ScaNN(model.query_model)
scann.index_from_dataset(
    tf.data.Dataset.zip(
        (products_ds.map(lambda x: x["gtin"]).batch(100), products_ds.batch(100).map(model.candidate_model)))
)
```

```python id="5aUNy-Ic_LKd"
scann = tfrs.layers.factorized_top_k.ScaNN(model.query_model)
scann.index_from_dataset(
    tf.data.Dataset.zip(
        (products_ds.map(lambda x: x["gtin"]).batch(100), products_ds.batch(100).map(model.candidate_model)))
)
```

<!-- #region id="EY7Cc-as4ra9" -->
## Serving
<!-- #endregion -->

<!-- #region id="NQXYRbB95Fi1" -->
### Export predictions to BigQuery
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="1eYoNxe7ix6_" executionInfo={"status": "ok", "timestamp": 1635442341438, "user_tz": 180, "elapsed": 4433, "user": {"displayName": "Cesar Reyes", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GivCEktqhpVTBc89dTYts8q4--6-tXOlPrzRm22=s64", "userId": "04702276980078964785"}} outputId="20983314-1ab8-486b-b041-8fdb3fb63ecf"
step = 1000
len_queries = len(user_ids)
partial_result = []

for k in range(0, len_queries, step):
    rec_score, rec_products = index(tf.constant([user_ids[k:k + step]]))
    partial_result.append([rec_score.numpy(), rec_products.numpy()])

recommendation = np.concatenate([k[1] for k in partial_result], axis=1).squeeze()
recommendation = pd.DataFrame(recommendation, index=user_ids).stack().reset_index()
recommendation.columns = ["user_id", "rank", "product_id"]
recommendation['product_id'] = recommendation['product_id'].apply(lambda x: x.decode())

recommendation
```

```python colab={"base_uri": "https://localhost:8080/"} id="60HuZ3xun_ix" executionInfo={"status": "ok", "timestamp": 1635442370115, "user_tz": 180, "elapsed": 28691, "user": {"displayName": "Cesar Reyes", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GivCEktqhpVTBc89dTYts8q4--6-tXOlPrzRm22=s64", "userId": "04702276980078964785"}} outputId="17f6ba63-2160-4597-b79e-af11d998add1"
result_project_name = "peya-food-and-groceries"
result_dataset_name = "user_fiorella_dirosario"
result_table_name = "user_recommendation_baseline"

recommendation.to_gbq(
    destination_table=f"{result_dataset_name}.{result_table_name}",
    project_id=result_project_name,
    if_exists="replace"
)
```

```python id="djd4sP0a7D2v"
step = 1000
len_queries = query_test.shape[0]
partial_result = []

for k in range(0, len_queries, step):
    sample = query_test.iloc[k:k + step, :].to_dict(orient="list")
    rec_score, rec_products = scann({
        "user_id": tf.convert_to_tensor(sample["user_id"]),
        "dow": tf.convert_to_tensor(sample["dow"]),
        "hod": tf.convert_to_tensor(sample["hod"]),
    })
    partial_result.append([rec_score.numpy(), rec_products.numpy()])

recommendation = np.concatenate([k[1] for k in partial_result], axis=0).squeeze()
recommendation = (
    pd.concat([query_test, pd.DataFrame(recommendation)], axis=1)
        .set_index(["user_id", "dow", "hod"])
        .stack()
        .reset_index()
)
recommendation.columns = ["user_id", "dow", "hod", "rank", "gtin"]
recommendation['gtin'] = recommendation['gtin'].apply(lambda x: x.decode())
```

```python id="TSCHZURp7G28"
result_project_name = "peya-food-and-groceries"
result_dataset_name = "user_fiorella_dirosario"
result_table_name = "recommendation_context_model_category_6_brand_10_epoch_2"

recommendation.to_gbq(
    destination_table=f"{result_dataset_name}.{result_table_name}",
    project_id=result_project_name,
    if_exists="replace"
)
```

<!-- #region id="rRay33v05MRQ" -->
### Saving models
<!-- #endregion -->

```python id="AMGBRJkFmtXr"
if IN_COLAB:
    ouput_path = "./models/"
else:
    ouput_path = "../models/"
```

```python id="gdn3vvH4d_5R"
path = os.path.join(ouput_path, "base_line_model", "index_model")
if not os.path.exists(path):
    os.makedirs(path)
tf.saved_model.save(index, path)
```

```python id="s-JhGXcllgPT"
path = os.path.join(ouput_path, "context_model_category_6_brand_10_epoch_2", "index_model")
if not os.path.exists(path):
    os.makedirs(path)
tf.saved_model.save(
    scann,
    path,
    options=tf.saved_model.SaveOptions(namespace_whitelist=["Scann"])
)
```

```python id="35KwsEsbLSop"
query_model_path = os.path.join(ouput_path, "context_model_category_6_brand_10_epoch_2", "query_model")
if not os.path.exists(query_model_path):
    os.makedirs(query_model_path)
model.query_model.save(query_model_path)
```

```python id="GX4vb3RanBgm"
candidate_model_path = os.path.join(ouput_path, "context_model_category_6_brand_10_epoch_2", "candidate_model")

if not os.path.exists(candidate_model_path):
    os.makedirs(candidate_model_path)
model.candidate_model.save(candidate_model_path)
```

```python id="8qSFR4AM_arZ"
path = os.path.join(ouput_path, "context_simple_model", "index_model")
if not os.path.exists(path):
    os.makedirs(path)
tf.saved_model.save(
    scann,
    path,
    options=tf.saved_model.SaveOptions(namespace_whitelist=["Scann"])
)
```

```python id="2INLxSq4_cWn"
query_model_path = os.path.join(ouput_path, "context_simple_model", "query_model")
if not os.path.exists(query_model_path):
    os.makedirs(query_model_path)
model.query_model.save(query_model_path)
```

```python id="15SBY3_S_deQ"
candidate_model_path = os.path.join(ouput_path, "context_simple_model", "candidate_model")
if not os.path.exists(candidate_model_path):
    os.makedirs(candidate_model_path)
model.candidate_model.save(candidate_model_path)
```
