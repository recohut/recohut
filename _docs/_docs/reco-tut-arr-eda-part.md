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

```python id="-UOOzCs9ukul"
project_name = "reco-tut-arr"; branch = "main"; account = "sparsh-ai"
```

```python colab={"base_uri": "https://localhost:8080/"} id="yjoT7OzOxK8t" executionInfo={"status": "ok", "timestamp": 1628007806152, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b7a0d74b-3e84-416f-fbfa-6ad72bb15dea"
import os

if not os.path.exists('/content/reco-tut-arr'):
    !cp /content/drive/MyDrive/mykeys.py /content
    import mykeys
    !rm /content/mykeys.py
    path = "/content/" + project_name; 
    !mkdir "{path}"
    %cd "{path}"
    import sys; sys.path.append(path)
    !git config --global user.email "arr@recohut.com"
    !git config --global user.name  "reco-tut-arr"
    !git init
    !git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout main
else:
    %cd '/content/reco-tut-arr'
```

```python id="ljYLIkBI_ijb"
!git add . && git commit -m 'commit' && git push origin main
```

```python id="h75rrYXdzr6r"
import sys
sys.path.insert(0,f'/content/{project_name}/code')
from utils import *
```

```python id="J6GnSXizHC08"
import os
import numpy as np
from numpy import log, sqrt
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F 

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 1000)
%matplotlib inline
```

```python id="Z12sTqxTDvMB"
vendors = pd.read_parquet('./data/bronze/vendors.parquet.gz')
orders = pd.read_parquet('./data/bronze/orders.parquet.gz')
train_customers = pd.read_parquet('./data/bronze/train_customers.parquet.gz')
train_locations = pd.read_parquet('./data/bronze/train_locations.parquet.gz')
test_customers = pd.read_parquet('./data/bronze/test_customers.parquet.gz')
test_locations = pd.read_parquet('./data/bronze/test_locations.parquet.gz')
```

<!-- #region id="JU1LVY2kxw-3" -->
## Orders
<!-- #endregion -->

<!-- #region id="8Oc55GYj2XCl" -->
---
<!-- #endregion -->

<!-- #region id="gvU9Ydkixw-9" -->
> Notes
- Is **NOT** split into train/test
- 135,233 orders
  - 131,942 made by customers in train_customers.csv
  - Other ~3k orders are ???
- grand_total can be 0
- vendor_discount_amount and promo_discount_percentage are mostly 0
- vendor and driver ratings are mostly either 0 or 5
- deliverydistance can be 0(?) and is at most ~20
- delivery_date can be null but created_at is similar and never null
- promo_code_discount_percentage is unreliable

<!-- #endregion -->

<!-- #region id="B0iJeFRU2X9V" -->
---
<!-- #endregion -->

<!-- #region id="CdNL1CLCxw-_" -->
### Check Some Values
<!-- #endregion -->

```python id="lp6u7fUWxw_C"
# Train / Test split
train_orders = orders[orders['customer_id'].isin(train_customers['akeed_customer_id'])]
test_orders = orders[orders['customer_id'].isin(test_customers['akeed_customer_id'])]
```

```python id="B91O4RHExw_H"
# Remove duplicate customers and their orders
x = train_customers.groupby('akeed_customer_id').size()
duplicate_train_customers = train_customers[train_customers['akeed_customer_id'].isin(x[x>1].index)]['akeed_customer_id'].unique()
train_customers = train_customers[~train_customers['akeed_customer_id'].isin(duplicate_train_customers)]
train_orders = train_orders[~train_orders['customer_id'].isin(duplicate_train_customers)]
```

```python id="HvBRz9L-xw_K" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628008214448, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="aa970576-1cce-4ac2-c49f-1367a9432600"
num_train_orders = orders[orders['customer_id'].isin(train_customers['akeed_customer_id'])].shape[0]
num_test_orders = orders[orders['customer_id'].isin(test_customers['akeed_customer_id'])].shape[0]
print(f'Num Orders: {orders.shape[0]}\nNum Train: {num_train_orders}\nNum Test: {num_test_orders}')
```

```python id="L89_w49gxw_N" colab={"base_uri": "https://localhost:8080/", "height": 343} executionInfo={"status": "ok", "timestamp": 1628008217935, "user_tz": -330, "elapsed": 727, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="741d5ef1-bace-4c86-ee8b-7a8c46ea6886"
train_orders.head(5)
```

```python id="-yZSHDkXxw_O" colab={"base_uri": "https://localhost:8080/", "height": 419} executionInfo={"status": "ok", "timestamp": 1628008227472, "user_tz": -330, "elapsed": 638, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a2dea60b-f766-4e02-e7c6-7a620555021e"
pd.concat([train_orders.dtypes.rename('dtype'), train_orders.isnull().sum().rename('num_null')], axis=1)
```

```python id="iX3Cp0SQxw_P" colab={"base_uri": "https://localhost:8080/", "height": 317} executionInfo={"status": "ok", "timestamp": 1628008233251, "user_tz": -330, "elapsed": 691, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="28cb544e-917e-4ccc-8beb-677834e78e97"
train_orders.describe()
```

```python id="W13Sb39Pxw_Q" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1628008236930, "user_tz": -330, "elapsed": 1014, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1e5e86db-593c-4017-c7ea-11f35f5b1409"
train_orders[train_orders['item_count'] < 20.5]['item_count'].hist(bins=20);
```

```python id="rf6amYX6xw_S" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1628008248835, "user_tz": -330, "elapsed": 729, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d042ddf7-bec1-420b-e743-425ffaf7d91d"
train_orders['vendor_rating'].hist();
```

```python id="rOfl9QO4xw_U" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1628008255097, "user_tz": -330, "elapsed": 1188, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="22e5bf5e-aaf5-4381-a44e-81442be602cc"
train_orders['driver_rating'].hist();
```

```python id="gi6u_t5Axw_V" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628008261673, "user_tz": -330, "elapsed": 582, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="920b4327-290d-47a5-b4b7-98976af64064"
train_orders['is_favorite'].value_counts(dropna=False)
```

```python id="R6gW0ssmxw_X" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628008270260, "user_tz": -330, "elapsed": 722, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="036a5ef7-e3d3-4ce8-8014-f0786ac7516d"
train_orders['is_rated'].value_counts(dropna=False)
```

```python id="A6pMO0w-xw_Z" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1628008272688, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ddc67810-ccdb-404f-f1e8-a605c72f6f0f"
train_orders['deliverydistance'].hist(bins=20);
```

```python id="m5shydknxw_a" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628008280499, "user_tz": -330, "elapsed": 566, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="53adfc60-68ea-426f-f773-7f104a485c14"
train_orders['delivery_date'].isnull().value_counts(dropna=False)
```

```python id="Dhhi0sOexw_b" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628008285309, "user_tz": -330, "elapsed": 586, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5f27a088-d4af-41b0-bd16-c4f113f186fb"
train_orders['created_at'].isnull().value_counts(dropna=False)
```

<!-- #region id="81DHXt5Rxw_c" -->
## Customers
<!-- #endregion -->

<!-- #region id="cTXX1I1x2xDH" -->
---
<!-- #endregion -->

<!-- #region id="nIAKmIzixw_g" -->
> Notes
- 34,467 customers
  - 26,741 have made at least 1 order
- Most customers have only 1 location
- Outliers in num_locations, dob
- Constant columns: language
<!-- #endregion -->

<!-- #region id="CJOXBf1d20Ny" -->
---
<!-- #endregion -->

<!-- #region id="7juy79E3xw_h" -->
### Check Some Values
<!-- #endregion -->

```python id="phmGRM9yxw_i" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628008315952, "user_tz": -330, "elapsed": 622, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3774bd8e-dfe8-4f64-929d-d8354a44ce43"
train_customers.shape[0]
```

```python id="caKArvQxxw_j" colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"status": "ok", "timestamp": 1628008316999, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e60f6aa8-3d75-4524-ad76-b7d1c98748dc"
train_customers.head(5)
```

```python id="W8Lyk-EOxw_j" colab={"base_uri": "https://localhost:8080/", "height": 297} executionInfo={"status": "ok", "timestamp": 1628008317001, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9b614021-0ea7-4cf0-e912-629ccb5f7cb7"
pd.concat([train_customers.dtypes.rename('dtype'), train_customers.isnull().sum().rename('num_null')], axis=1)
```

```python id="y_v21dpKxw_k"
# Add num_locations as new column in customer table
locations_customer_grp = train_locations.groupby(by=['customer_id'])
locations_per_customer = locations_customer_grp['location_number'].count().rename('num_locations')
train_customers = train_customers.merge(locations_per_customer, how='left', left_on='akeed_customer_id', right_index=True)
```

```python id="y5Bby0FIxw_k"
# Add num_orders as new column in customer table
orders_per_customer = train_orders.groupby('customer_id')['akeed_order_id'].count().rename('num_orders')
train_customers = train_customers.merge(orders_per_customer, how='left', left_on='akeed_customer_id', right_index=True)
```

```python id="sJeOdgaqxw_l" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628008331129, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ba9112f7-199d-4f9b-9187-b6574e9b9573"
train_customers[train_customers['num_orders'] < 1].shape[0]
```

```python id="7iQoqHoOxw_l" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628008341964, "user_tz": -330, "elapsed": 732, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="276539ba-8e93-44bb-c64f-f340b2f804aa"
train_customers['num_orders'].value_counts(dropna=False).sort_index()[:5]
```

```python id="LbmZ3FQSxw_m" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628008347319, "user_tz": -330, "elapsed": 572, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2b1992c6-0be9-429d-b6d7-0a11035f4dc1"
train_customers['num_orders'].isna().sum()
```

```python id="R5Bhe_VExw_n"
# Remove customers with no orders
train_customers = train_customers[train_customers['num_orders'] > 0]
```

```python id="HJQmEgKdxw_n" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628008348286, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b15d2965-3150-4e7e-8908-1d1c56b45435"
train_customers.shape[0]
```

```python id="WV2N3djZxw_o" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628008349726, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7fc082de-3721-41f3-e89a-7edf9a4798ef"
train_customers['gender'].value_counts(dropna=False)
```

```python id="fBcFWwaexw_o" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628008368237, "user_tz": -330, "elapsed": 611, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bdbe9bb6-5168-480e-f25a-813ade47bea4"
# Clean gender column and remove outliers
train_customers['gender'] = train_customers['gender'].str.strip()
train_customers['gender'] = train_customers['gender'].str.lower()
gender_filter = (train_customers['gender'] == 'male') | (train_customers['gender'] == 'female')
train_customers = train_customers[gender_filter]
train_customers['gender'].value_counts(dropna=False)
```

```python id="9jgKB53pxw_o" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628008382485, "user_tz": -330, "elapsed": 1378, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fdab15c2-e511-49c4-de07-01464d9b2073"
train_customers['language'].value_counts(dropna=False)
```

```python id="XM0Ks0-yxw_q" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628008385378, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a75c8d22-39a7-477b-e817-cb65f17aea49"
ser = train_customers['created_at'] == train_customers['updated_at']
ser.value_counts(dropna=False)
```

<!-- #region id="ZhzF9ujJxw_q" -->
## Vendors

We should prioritize cleaning this table because it will likely be the most useful data for our model. 
<!-- #endregion -->

<!-- #region id="Yeholxcq3K02" -->
---
<!-- #endregion -->

<!-- #region id="_MJSy8Wtxw_r" -->
> Notes
- 100 resturaunts
- Outliers present in `latitude`, `longitude`, `preparation_time`, `discount_percentage`
- Constant columns: `commission`, `open_close_flags`, `country_id`, `city_id`, `display_orders`, `one_click_vendor`, `is_akeed_delivering`, `language`
- Nearly constant: `discount_percentage`
- Columns (`status`, `verified`), (`vendor_category_en`, `vendor_category_id`) are almost equal 
- Median and Max `serving_distance` is 15
- `rank` is either 1 or 11...?
- `vendor_rating` has small variance
- `vendor_tag` and `vendor_tag_name` are the same: lists of food types
- Columns recording open/close times are confusing... What is the difference between 1 and 2 and `opening_time`?
- `delivery_charge` is actually categorical
<!-- #endregion -->

<!-- #region id="i5nmrc5R3MeF" -->
---
<!-- #endregion -->

<!-- #region id="LG26Kqh0xw_s" -->
### Checking Some Values
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="eU775Wi244Ur" executionInfo={"status": "ok", "timestamp": 1628010768538, "user_tz": -330, "elapsed": 549, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="feaf461e-8135-42a3-ef7a-430b52bd548c"
vendors = pd.read_parquet('./data/bronze/vendors.parquet.gz')
vendors.info()
```

```python id="NWRrlhSlxw_t" colab={"base_uri": "https://localhost:8080/", "height": 309} executionInfo={"status": "ok", "timestamp": 1628009539901, "user_tz": -330, "elapsed": 758, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7bf407ed-17ab-4221-aac7-57337ee3582d"
vendors.head(5)
```

```python id="yp89Yf7pxw_t" colab={"base_uri": "https://localhost:8080/", "height": 340} executionInfo={"status": "ok", "timestamp": 1628010774416, "user_tz": -330, "elapsed": 999, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bb9edb31-5a49-4aba-81c4-9a5edc46db25"
# Set id column to index
vendors.sort_values(by='id')
vendors, v_id_map, v_inv_map = integer_encoding(df=vendors, cols=['id'], drop_old=True, monotone_mapping=True)
vendors.set_index('id', inplace=True)
vendors.head()
```

```python id="nWKcffJ7xw_u" colab={"base_uri": "https://localhost:8080/", "height": 419} executionInfo={"status": "ok", "timestamp": 1628010774418, "user_tz": -330, "elapsed": 20, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0ec14153-f7ad-49e0-d391-27be64f03661"
pd.concat([vendors.dtypes.rename('dtype'), vendors.isnull().sum().rename('num_null')], axis=1)
```

```python id="Sr0sx-7Exw_v" colab={"base_uri": "https://localhost:8080/", "height": 317} executionInfo={"status": "ok", "timestamp": 1628010774420, "user_tz": -330, "elapsed": 19, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ab413a1d-620e-40f0-d6cb-1c35ec6512d2"
vendors.describe()
```

```python id="ebPdclz9xw_w" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628010777755, "user_tz": -330, "elapsed": 35, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4c54899f-be79-447e-8b4b-fdf029d28123"
vendors['is_akeed_delivering'].value_counts(dropna=False)
```

```python id="5g5Q4xDSxw_w" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628010777756, "user_tz": -330, "elapsed": 21, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0b664c5b-c020-4213-8eab-11977845e208"
vendors['language'].value_counts(dropna=False)
```

```python id="GW-_FjH2xw_x" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628010777757, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="96112977-f1e9-4a86-cbd8-7eae1d3897f8"
vendors['one_click_vendor'].value_counts(dropna=False)
```

```python id="NSMYiQmgxw_y" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1628010777760, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bdf2385e-eb2a-4a80-fba5-06d6eaba37a3"
vendors['delivery_charge'].hist();
```

```python id="d7uhLWhLxw_y" colab={"base_uri": "https://localhost:8080/", "height": 266} executionInfo={"status": "ok", "timestamp": 1628010779164, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="77ff2588-8546-4f78-a5a5-f5f604734f26"
vendors['serving_distance'].hist();
```

```python id="B-0D2rUExw_z" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1628010780183, "user_tz": -330, "elapsed": 33, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8ee759ef-06e5-4f8a-a65a-b86d93623fa2"
vendors['prepration_time'].hist();
```

```python id="VuLCcxBLxw_z" colab={"base_uri": "https://localhost:8080/", "height": 268} executionInfo={"status": "ok", "timestamp": 1628010780184, "user_tz": -330, "elapsed": 31, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="facec5f8-e1ac-4f0f-f4d1-2b41daa2fd0e"
vendors['discount_percentage'].hist();
```

```python id="qkLnuKqQxw_0" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628010780186, "user_tz": -330, "elapsed": 25, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fc298188-61fc-4ecf-8db4-fc8fafc33618"
vendors['rank'].value_counts(dropna=False)
```

```python id="fEHJbNOexw_0" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1628010784386, "user_tz": -330, "elapsed": 851, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3b3295a3-08a1-4ff1-9cfb-43ea96515db2"
vendors['vendor_rating'].hist();
```

```python id="ebFehzJ4xw_0" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628010784387, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5264ce41-2dd6-4221-bc26-2c74d043a3cc"
vendors['status'].value_counts(dropna=False)
```

```python id="lKo8wU65xw_1" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628010785219, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fb481187-fba3-414a-9d53-93353187ee9e"
vendors['verified'].value_counts(dropna=False)
```

```python id="2Le1DfgRxw_1" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628010785221, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cbaed13e-ba82-4b95-9d66-7122826513ca"
vendors[vendors['verified'] == 0]['status'].value_counts(dropna=False)
```

```python id="WZ8DfDH4xw_1" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628010786076, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="99f7c4f8-3cd8-4239-b776-96a413ec398b"
vendors['device_type'].value_counts(dropna=False)
```

```python id="g2Laxjehxw_2" colab={"base_uri": "https://localhost:8080/", "height": 148} executionInfo={"status": "ok", "timestamp": 1628010786077, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="85ee3335-7447-40ed-f710-2bd4c3430676"
vendors[vendors.device_type == 1]   # Is a location outlier...?
```

<!-- #region id="B_ixru8Yxw_2" -->
### Cleaning Vendor Categories
<!-- #endregion -->

```python id="PXRyEnjJxw_2" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628010787746, "user_tz": -330, "elapsed": 18, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b3986f0e-45b4-4afd-f3ae-5ef46c87d839"
vendors['vendor_category_en'].value_counts(dropna=False)
```

```python id="469HaU--xw_2" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628010787747, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e28ecec9-65aa-4a36-e5e8-a3b8b3f4f291"
vendors['vendor_category_id'].value_counts(dropna=False)
```

```python id="ntmtl_D7xw_3" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628010787748, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6df8672a-109c-45b3-951f-b76b0d12c6f2"
vendors[(vendors['vendor_category_en'] == "Sweets & Bakes") & (vendors['vendor_category_id'] == 3.0)].shape[0]
```

```python id="cwjKSc6Axw_3" colab={"base_uri": "https://localhost:8080/", "height": 165} executionInfo={"status": "ok", "timestamp": 1628010789928, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c1b997f7-fe97-4e16-f959-2ab1033842a0"
vendors[(vendors['vendor_category_en'] == "Sweets & Bakes") & (vendors['vendor_category_id'] == 2.0)]
```

```python id="QXJFCFWBxw_3" executionInfo={"status": "ok", "timestamp": 1628010789929, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Fix incorrect vendor_category_id
vendors.loc[28, 'vendor_category_id'] = 3.0
```

<!-- #region id="TK99vLG5xw_3" -->
### Cleaning Vendor Tags
<!-- #endregion -->

```python id="SYpExicnxw_4" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628010794528, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="42678b71-61a9-43ac-bb4d-3a5b7c5c2da6"
vendors['primary_tags'].value_counts(dropna=False)
```

```python id="caDUQdoqxw_4" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628010796093, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="82a5007c-c3e7-4fa6-bc91-691c62572671"
# Fill na with -1
# Strip unnecessary characters
vendors['primary_tags'] = vendors['primary_tags'].fillna("{\"primary_tags\":\"-1\"}").apply(lambda x: int(str(x).split("\"")[3]))
vendors['primary_tags'].value_counts(dropna=False).head(5)
```

```python id="LNwA4tlexw_4" colab={"base_uri": "https://localhost:8080/", "height": 580} executionInfo={"status": "ok", "timestamp": 1628010796766, "user_tz": -330, "elapsed": 23, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cf88f49e-fe23-4544-89b0-ac9b0b04f62c"
vendors[vendors['primary_tags'] == 134]
```

```python id="zRKPK4pDxw_7" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628010798702, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="88d80a91-e465-4657-fea9-1bffa2a3a0b4"
# Fill na with -1
# Turn vendor_tag into list-valued
vendors['vendor_tag'] = vendors['vendor_tag'].fillna(str(-1)).apply(lambda x: x.split(",")).apply(lambda x: [int(i) for i in x])
vendors['vendor_tag'].head(10)
```

```python id="NuQFiZsExw_8" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628010799539, "user_tz": -330, "elapsed": 20, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d3efd459-a74d-49f7-a685-9640d47c85ad"
# Get unique vendor tags
# Map values to range(len(vendor_tags))
vendor_tags = [int(i) for i in vendors['vendor_tag'].explode().unique()]
vendor_tags.sort()
vendor_map = dict()
for i, tag in enumerate(vendor_tags):
    vendor_map[tag] = i
vendors['vendor_tag'] = vendors['vendor_tag'].apply(lambda tags: [vendor_map[tag] for tag in tags])
vendors['vendor_tag'].head(10)
```

```python id="29J1YKvAxw_8" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628010799540, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="94810dea-dc0d-4216-a65e-f2b55932d2d6"
# Combine status and verified features
vendors['status_and_verified'] = vendors['status'] * vendors['verified']
vendors['status_and_verified'].value_counts(dropna=False)
```

<!-- #region id="--nZra3lxw_8" -->
### Creating Some Order-Based Features
<!-- #endregion -->

```python id="RgrE-VQhxw_8" executionInfo={"status": "ok", "timestamp": 1628010801621, "user_tz": -330, "elapsed": 840, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Add num_orders, amt_sales, and avg_sale as new columns in vendor table

orders_vendor_grp = train_orders.groupby(by=['vendor_id'])
orders_per_vendor = orders_vendor_grp['akeed_order_id'].count().rename('num_orders')
grand_total_per_vendor = orders_vendor_grp['grand_total'].sum().rename('amt_sales')

vendors = vendors.merge(orders_per_vendor, how='left', left_on='id', right_index=True)
vendors = vendors.merge(grand_total_per_vendor, how='left', left_on='id', right_index=True)
vendors['avg_sale'] = vendors['amt_sales'] / vendors['num_orders']
```

```python id="98Ht3BHBxw_8" colab={"base_uri": "https://localhost:8080/", "height": 268} executionInfo={"status": "ok", "timestamp": 1628010801627, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1562a4c7-f720-4782-9c20-2dc1e4a4ae1d"
vendors['num_orders_log3'] = vendors['num_orders'].apply(log).apply(log).apply(log)
vendors['num_orders_log3'].hist();
```

```python id="sef9t0Ygxw_9" colab={"base_uri": "https://localhost:8080/", "height": 266} executionInfo={"status": "ok", "timestamp": 1628010803700, "user_tz": -330, "elapsed": 816, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="028b2c22-4811-4832-e58a-1c34ab284a5e"
vendors['amt_sales_log3'] = vendors['amt_sales'].apply(log).apply(log).apply(log)
vendors['amt_sales_log3'].hist();
```

```python id="nPQ2vtjTxw_9" colab={"base_uri": "https://localhost:8080/", "height": 266} executionInfo={"status": "ok", "timestamp": 1628010803702, "user_tz": -330, "elapsed": 18, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a824d8ca-ac0d-4f72-8d01-916b7dd7695d"
vendors['avg_sale_log'] = vendors['avg_sale'].apply(log)
vendors['avg_sale_log'].hist();
```

<!-- #region id="hPOYG5UXxw_9" -->
### Transforming Location Outliers
<!-- #endregion -->

```python id="eCdrZFt9xw_-" colab={"base_uri": "https://localhost:8080/", "height": 196} executionInfo={"status": "ok", "timestamp": 1628010806050, "user_tz": -330, "elapsed": 908, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bdc127ed-38e7-45b3-97fa-b85bdbed55e6"
# Examine the location outliers

vendors[vendors['latitude'] > 3]
```

```python id="UyKIIQrdxw_-" colab={"base_uri": "https://localhost:8080/", "height": 148} executionInfo={"status": "ok", "timestamp": 1628010806052, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3871a675-84de-4756-f17e-4e67bf3dee03"
vendors[vendors['longitude'] > 3]
```

```python id="owWOl-P-xw_-" executionInfo={"status": "ok", "timestamp": 1628010806054, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
orders_231 = train_orders[train_orders['vendor_id'] == 231]
orders_907 = train_orders[train_orders['vendor_id'] == 907]

orders_231 = orders_231.merge(train_locations, how='left', left_on=['customer_id', 'LOCATION_NUMBER'], right_on=['customer_id', 'location_number'])
orders_907 = orders_907.merge(train_locations, how='left', left_on=['customer_id', 'LOCATION_NUMBER'], right_on=['customer_id', 'location_number'])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 694} id="WM5iBSGnAgHt" executionInfo={"status": "ok", "timestamp": 1628010871299, "user_tz": -330, "elapsed": 835, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="709cd5a6-177c-4374-9098-42601fe22ed0"
vendors.id
```

```python id="ajAREqjaxw_-" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628011278563, "user_tz": -330, "elapsed": 944, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a45fbcbd-d25e-45f7-ea2e-a35297df057a"
lat231 = vendors[vendors.index == v_id_map['id'][231]].latitude.item()
long231 = vendors[vendors.index == v_id_map['id'][231]].longitude.item()
lat907 = vendors[vendors.index == v_id_map['id'][907]].latitude.item()
long907 = vendors[vendors.index == v_id_map['id'][907]].longitude.item()

print(f'231 actual: \tLat = {lat231:.3f}, Long = {long231:.3f}')
print(f'231 estimate: \tLat = {orders_231.latitude.median():.3f}, Long = {orders_231.longitude.median():.3f}')
print(f'907 actual: \tLat = {lat907:.3f}, Long = {long907:.3f}')
print(f'907 estimate: \tLat = {orders_907.latitude.median():.3f}, Long = {orders_907.longitude.median():.3f}')
```

```python id="cLBjaesWxw_-" colab={"base_uri": "https://localhost:8080/", "height": 317} executionInfo={"status": "ok", "timestamp": 1628010810775, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b4f9d496-2e21-40fd-f9b6-5f4cfaea062d"
orders_907.describe()
```

```python id="F4_qdkPBxw__" colab={"base_uri": "https://localhost:8080/", "height": 317} executionInfo={"status": "ok", "timestamp": 1628009621680, "user_tz": -330, "elapsed": 731, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="edb674e5-211b-4962-d4a9-a4bc12022759"
orders_231.describe()
```

```python id="D1kTU7Aqxw__" colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"status": "ok", "timestamp": 1628009623617, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="691f7a87-d4f4-4353-946e-fdb5fe31ce8e"
train_locations.head(5)
```

```python id="hDikXnA6xw__" colab={"base_uri": "https://localhost:8080/", "height": 297} executionInfo={"status": "ok", "timestamp": 1628009635573, "user_tz": -330, "elapsed": 782, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="695a4643-ebe7-4593-fff2-7d3502136c3b"
pd.concat([train_locations.dtypes.rename('dtype'), train_locations.isnull().sum().rename('num_null')], axis=1)
```

```python id="qfqQQfcpxw__"
# Aggregate # orders, $ sales, and avg spent by customer location
# (customers can have multiple locations registered to themselves)

orders_location_grp = train_orders.groupby(['customer_id', 'LOCATION_NUMBER'])
orders_per_location = orders_location_grp['akeed_order_id'].count().rename('num_orders')    # multi index: [customer_id, LOCATION_NUMBER]
sales_per_location = orders_location_grp['grand_total'].sum().rename('amt_spent')           # multi index: [customer_id, LOCATION_NUMBER]

train_locations = train_locations.merge(sales_per_location, how='left', left_on=['customer_id', 'location_number'], right_index=True)
train_locations = train_locations.merge(orders_per_location, how='left', left_on=['customer_id', 'location_number'], right_index=True)
train_locations['avg_spend'] = train_locations['amt_spent'] / train_locations['num_orders']
```

```python id="e6ICPnbbxw__"
# Filter locations which have not been ordered from
train_locations = train_locations[train_locations['num_orders'] != 0]
```

```python id="I0I-aDQaxxAA" colab={"base_uri": "https://localhost:8080/", "height": 317} executionInfo={"status": "ok", "timestamp": 1628009712973, "user_tz": -330, "elapsed": 784, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9743c964-1291-4e4b-c976-3382e5586fb1"
train_locations.describe()
```

```python id="qv2BRRizxxAA" colab={"base_uri": "https://localhost:8080/", "height": 247} executionInfo={"status": "ok", "timestamp": 1628009715177, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="461d2ed4-6dab-45fd-9c7b-51a21baab852"
train_locations[train_locations['amt_spent'] == 0].merge(train_orders, left_on=['customer_id', 'location_number'], right_on=['customer_id', 'LOCATION_NUMBER']).head(3)     # Free orders
```

<!-- #region id="WDiJNr4PxxAA" -->
**Note from VariableDefinitions.txt:** 

"Not true latitude and longitude - locations have been masked, but nearby locations remain nearby in the new reference frame and can thus be used for clustering. However, not all locations are useful due to GPS errors and missing data - you may want to treat outliers separately."

This will make our life difficult because we have no way of knowing how the location data has been transformed, thus it's not really clear how we should define "outlier".

Almost all vendors are clustered very close to each other, but we will soon see that about 1/3rd of customer locations are "far" from this cluster.

<!-- #endregion -->

```python id="7GYW6qgxxxAA" colab={"base_uri": "https://localhost:8080/", "height": 893} executionInfo={"status": "ok", "timestamp": 1628009726308, "user_tz": -330, "elapsed": 1735, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f5c13489-88eb-4c5b-efb1-28320b33d393"
# Map out customer locations and vendor locations

plt.figure(figsize=(15, 15))
plt.scatter(x=train_locations.longitude, y=train_locations.latitude, label='Customers', marker='s', alpha=0.2)
plt.scatter(x=vendors.longitude, y=vendors.latitude, label='Vendors', marker='*', alpha=0.5, s=vendors['num_orders']/5, c=vendors['avg_sale'], cmap='plasma')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(loc='lower right')
plt.colorbar(label='$ Avg Sale')
plt.title('Customer + Vendor Locations')
plt.show()

# Stars:
#   Size: Unpopular <----------> Popular
#   Heat:     Cheap <----------> Expensive
```

```python id="WXqaK7nPxxAA" colab={"base_uri": "https://localhost:8080/", "height": 893} executionInfo={"status": "ok", "timestamp": 1628009740616, "user_tz": -330, "elapsed": 2606, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7bdbbdc0-30af-40dc-e6bd-4d2d3b4abf3e"
# Outliers in location are probably a mistake (GPS error?)
# Zoom in on area close with most activity
#     Marker Size = # Orders
#           Color = $ Grand Total

lo = -5
hi = 5

filt1 = (lo < train_locations['longitude']) & (train_locations['longitude'] < hi)
filt2 = (lo < vendors['longitude']) & (vendors['longitude'] < hi)
train_locations_cut = train_locations[filt1]
vendors_cut = vendors[filt2]

plt.figure(figsize=(15, 15))
plt.scatter(x=train_locations_cut.longitude, y=train_locations_cut.latitude, label='Customers', marker='s', alpha=0.1)
plt.scatter(x=vendors_cut.longitude, y=vendors_cut.latitude, label='Vendors', marker='*', alpha=0.5, s=vendors_cut['num_orders']/7, c=vendors_cut['avg_sale'], cmap='plasma')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(loc='lower right')
plt.colorbar(label='$ Avg Sale')
plt.title('Customer + Vendor Locations (Zoomed)')
plt.show()

# Stars:
#   Size: Unpopular <----------> Popular
#   Heat:     Cheap <----------> Expensive

```

```python id="7iHRPRKpxxAB" colab={"base_uri": "https://localhost:8080/", "height": 297} executionInfo={"status": "ok", "timestamp": 1628009741516, "user_tz": -330, "elapsed": 35, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0c7cc5bd-4827-4444-a3b0-a93218300bae"
# Define outliers for customer locations
# There are a lot of customers that are outside the "vendor bubble"
#   vendor bubble == customers that aren't outliers

lat_lo, lat_hi = -25, 25
long_lo, long_hi = -5, 5
c_outliers = (train_locations['latitude'] < lat_lo) | (train_locations['latitude'] > lat_hi) | (train_locations['longitude'] < long_lo) | (train_locations['longitude'] > long_hi)
v_outliers = (vendors['latitude'] < lat_lo) | (vendors['latitude'] > lat_hi) | (vendors['longitude'] < long_lo) | (vendors['longitude'] > long_hi)


train_locations[c_outliers].describe()
```

```python id="_l-rFnAExxAB" colab={"base_uri": "https://localhost:8080/", "height": 927} executionInfo={"status": "ok", "timestamp": 1628009765702, "user_tz": -330, "elapsed": 24218, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f9387c6f-6c05-47a9-9f34-880c6a82e2fc"
# Want to transform outliers so that they are closer to vendors, but also stay in their clusters
# Project outliers onto ellipse around bubble

lat_radius = lat_hi
long_radius = long_hi

# Project customer outliers
for i in tqdm(train_locations[c_outliers].index):
        lat = train_locations.loc[i, 'latitude']
        long = train_locations.loc[i, 'longitude']
        mag = sqrt(lat**2 + long**2)
        train_locations.loc[i, 'latitude'] = lat / mag * lat_radius
        train_locations.loc[i, 'longitude'] = long / mag * long_radius

# Project vendor outliers
for i in tqdm(vendors[v_outliers].index):
        lat = vendors.loc[i, 'latitude']
        long = vendors.loc[i, 'longitude']
        mag = sqrt(lat**2 + long**2)
        vendors.loc[i, 'latitude'] = lat / mag * lat_radius
        vendors.loc[i, 'longitude'] = long / mag * long_radius

plt.figure(figsize=(15, 15))
plt.scatter(x=train_locations.longitude, y=train_locations.latitude, label='Customers', marker='s', alpha=0.2)
plt.scatter(x=vendors.longitude, y=vendors.latitude, label='Vendors', marker='*', alpha=0.5, s=vendors['num_orders']/5, c=vendors['avg_sale'], cmap='plasma')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(loc='lower right')
plt.colorbar(label='$ Avg Sale')
plt.title('Customer + Vendor Locations (Outliers Transformed)')
plt.show()
```

<!-- #region id="BOXxbhnqxxAC" -->
### Drop Columns
<!-- #endregion -->

```python id="UUFSJiWYxxAC"
# Throw away some columns
keep_continuous = ['latitude', 'longitude', 'serving_distance', 'prepration_time', 'vendor_rating', 'num_orders_log3', 'amt_sales_log3', 'avg_sale_log']
keep_categorical = ['vendor_category_id', 'delivery_charge', 'status', 'rank', 'primary_tags', 'vendor_tag']
keep_columns = keep_continuous + keep_categorical
vendors = vendors[keep_columns]
```

```python id="t-bbKDqAxxAC" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628009765708, "user_tz": -330, "elapsed": 28, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b2a2a09e-726e-4341-8905-22cf61212605"
vendors.isnull().sum()
```

<!-- #region id="a67V-d7exxAC" -->
### Encode Categorical Features
- `vendor_category_id`  -> single binary variable           -> remap to [0,1]
- `delivery_charge`     -> single binary variable           -> remap to [0,1]
- `status`              -> single binary variable           -> remap to [0,1]
- `rank`                -> single binary variable           -> remap to [0,1]
- `primary_tags`        -> single multi-class variable      -> remap to [0,C]                   ->   one-hot encode in [0,C]
- `vendor_tag`          -> multiple binary variables        -> one-to-many encode in [0,1]^C
<!-- #endregion -->

```python id="7sPUz7MoxxAC" colab={"base_uri": "https://localhost:8080/", "height": 284} executionInfo={"status": "ok", "timestamp": 1628009768723, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="af0dab94-ef34-43ba-ef14-1adca018c65f"
vendors, _, _ = integer_encoding(df=vendors, cols=['vendor_category_id', 'delivery_charge', 'status', 'rank', 'primary_tags'], drop_old=True, monotone_mapping=True)
vendors = multiclass_list_encoding(df=vendors, cols=['primary_tags', 'vendor_tag'], drop_old=True)
vendors.head(5)
```

<!-- #region id="2Istf2YHxxAD" -->
## Represent Customers
<!-- #endregion -->

<!-- #region id="Jbdt3eiG8dNt" -->
---
<!-- #endregion -->

<!-- #region id="Yycg4K9dxxAD" -->
> Notes
- Need to construct training data. 
- First construct sequences of orders from users or user locations 
  - Group orders by `customer_id`
  - Sort each group chronologically, by created_at
  - Collect each group into a list of ints
<!-- #endregion -->

<!-- #region id="j8KFM7E38gCW" -->
---
<!-- #endregion -->

```python id="LEdTVAuAxxAD"
# Sort orders by datetime

train_orders['created_at'] = pd.to_datetime(train_orders['created_at'])
train_orders.sort_values(by=['created_at'], inplace=True)
```

```python id="hvxNCLIXxxAD" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628009797378, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="99d28bc4-d96a-4b25-fa6f-a66a7e7b1710"
orders_grp = train_orders.groupby(by=['customer_id'])
orders_grp['vendor_id'].count().value_counts(normalize=True).head(5)
```

```python id="5vPZNSR2xxAE" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628009799042, "user_tz": -330, "elapsed": 943, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bd519c7f-d624-45bc-e91c-e7f96b4ec79c"
# Map vendor ids to range(0,num_vendors)
train_orders, v_id_map, v_inv_map = integer_encoding(df=train_orders, cols=['vendor_id'], drop_old=True, monotone_mapping=True)

# Group sequences by customer_id
train_sequences = get_sequences(df=train_orders, target='vendor_id', group_by=['customer_id'])
train_sequences.head(10)
```

```python id="l9l0nFq5xxAE"
# Represent customers as averages of the vendors they purchased from

train_customer_encoded = pool_encodings_from_sequences(sequences=train_sequences, pool_from=vendors)
```

```python id="bIC3lbt0xxAE" colab={"base_uri": "https://localhost:8080/", "height": 439} executionInfo={"status": "ok", "timestamp": 1628010128020, "user_tz": -330, "elapsed": 50, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="dfe6ef58-60cd-4476-bf7d-f343fd6ed6b6"
train_customer_encoded.head(10)
```

```python id="M7gcaYnmxxAE" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628010128022, "user_tz": -330, "elapsed": 47, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c32bbc88-4237-436b-dd3b-31dec22e78cb"
vendors[vendors.isna().sum(axis=1) > 0].isna().sum()
```
