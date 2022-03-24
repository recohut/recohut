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

<!-- #region id="a45LiITGi78l" -->
# Recommending Retail Products with Memory-based model
> Fitting KNN on a retail dataset to recommend retail items to customers

- toc: true
- badges: true
- comments: true
- categories: [KNN, Retail, MongoDB, PrivateData]
- image:
<!-- #endregion -->

<!-- #region id="pSYjZLXVfQ0m" -->
## Setup
<!-- #endregion -->

```python id="V2A5eM_reuS4"
!pip install -q dnspython
```

```python id="vjcyS0HCeup7"
import os
import dns
import json
import pickle
import numpy as np
import pandas as pd
from pymongo import MongoClient
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
```

<!-- #region id="86i0SkDKfShL" -->
## Data ingestion
<!-- #endregion -->

```python id="D22qE0yXLeHG"
client = MongoClient("mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
db = client.get_database("OnlineGroceryDB")

for col in db.collection_names():
  cursor = db[col].find()
  pd.DataFrame(list(cursor)).to_csv('{}.csv'.format(col))

!zip retail_data.zip ./*.csv
```

<!-- #region id="nFhAWy_NfuL4" -->
## Data schema

Let's analyze the full data schema
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="JBuAqoCkhSRT" outputId="633307ca-245c-4bff-bcd3-a837eae54e0c"
pd.read_csv("Customer.csv", index_col=0).columns
```

```python colab={"base_uri": "https://localhost:8080/", "height": 452} id="IIyyyfaGf_OW" outputId="45791ccc-9bb4-4b70-c74c-48321e5afe83"
import glob
import pandas as pd
from pathlib import Path

pd.set_option('display.expand_frame_repr', False)

schema = pd.DataFrame()

for i, filepath in enumerate(glob.glob("./*.csv")):
  df = pd.read_csv(filepath, index_col=0)
  schema.loc[i,"filename"] = Path(filepath).stem
  schema.loc[i,"shape"] = str(df.shape)
  schema.loc[i,"columns"] = str(list(df.columns))

schema
```

<!-- #region id="BG2uMFY2f2xC" -->
> Note: We are only using a small part of the full dataset
<!-- #endregion -->

<!-- #region id="kQDeGGLcfVAV" -->
## Preprocessing and Modeling
<!-- #endregion -->

```python id="huOeGirjcNVo"
data_path = '/content'

df_Order = pd.read_csv(os.path.join(data_path,'Order.csv'), usecols=['customer_id', 'order_id'])
df_Order_Item = pd.read_csv(os.path.join(data_path,'Order_Item.csv'), usecols=['product_id', 'order_id'])
df_Product_Review = pd.read_csv(os.path.join(data_path,'Product_Review.csv'), usecols=['product_id', 'ratings'])
df_order_order_item = df_Order.merge(df_Order_Item, on='order_id')
Product_rating_by_Customer = df_order_order_item.merge(df_Product_Review, on='product_id') # Change the join key to customer_id once data available
Product_rating_by_Customer.drop_duplicates(subset=['customer_id', 'product_id'], keep='first', inplace=True)
df_User_interaction_mat = Product_rating_by_Customer.pivot(index='product_id', columns='customer_id', values='ratings').fillna(0)
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(csr_matrix(df_User_interaction_mat.values))

def getPrediction(product_id):
  df_product = pd.read_csv(os.path.join(data_path,'Product.csv'))
  query_index = df_User_interaction_mat.index.get_loc(product_id)
  distances, indices = model_knn.kneighbors(df_User_interaction_mat.iloc[query_index, :].values.reshape(1, -1), n_neighbors=6)
  lst=[]
  for i in range(1, len(distances.flatten())):
      lst.append(df_product[df_product.product_id == df_User_interaction_mat.index[indices.flatten()[i]]].set_index('product_id').to_dict(orient="index"))
  return lst
```

<!-- #region id="xCSxyrpBfZPj" -->
## Inference
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="BNJh_J1CeCDg" outputId="8e04f2ef-59d7-4156-e1d5-00e2a08cfa9e"
getPrediction("PRO012")
```

```python colab={"base_uri": "https://localhost:8080/"} id="XjMUktaveM3Z" outputId="9e3deece-b286-4ff8-cc5a-cf212630334b"
getPrediction("PRO010")
```

```python colab={"base_uri": "https://localhost:8080/"} id="AjyWoMNceOj8" outputId="a3e1a0b6-8b8f-4daa-ffbf-2cd666d722c5"
getPrediction("PRO015")
```
